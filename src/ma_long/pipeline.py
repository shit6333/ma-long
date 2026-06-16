"""ma_long sequential long-sequence pipeline (P2 — no loop closure yet).

Pipeline:
    1. split the sequence into overlapping chunks
    2. run MapAnything per chunk (multi-modal), cache results to disk
    3. align consecutive chunks on their overlap region -> relative SE3/Sim3
    4. accumulate to per-chunk global transforms (chunk 0 = world frame)
    5. apply transforms, write per-frame global c2w poses + a merged point cloud

The geometry / alignment maths is the vendored `align.sim3utils`. MapAnything
already returns chunk-local `world_points` (anchored to view 0) and c2w poses, so —
unlike the DA3 pipeline — there is no depth->pointcloud or w2c inversion step here.

Loop closure (P3) plugs in between steps 3 and 4 via `Sim3LoopOptimizer`.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from align.sim3utils import (
    accumulate_sim3_transforms,
    apply_sim3_direct,
    compute_sim3_ab,
    merge_ply_files,
    process_loop_list,
    save_confident_pointcloud_batch,
    weighted_align_point_maps,
)

# Default config. MapAnything is metric, so align with SE3 (scale ≈ 1) by default;
# 'sim3' is available if you want alignment to absorb residual scale drift.
_DEFAULT_SALAD_CKPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "dino_salad.ckpt")

DEFAULT_CONFIG: Dict = {
    "Model": {
        "chunk_size": 60,
        "overlap": 20,
        "align_lib": "torch",      # 'torch'|'numpy'|'numba'|'triton'
        "align_method": "se3",     # 'se3'|'sim3'|'scale+se3'
        "loop_enable": True,
        "loop_chunk_size": 20,     # loop window = 2 * (loop_chunk_size // 2) frames
        "IRLS": {"delta": 0.1, "max_iters": 5, "tol": "1e-9"},
        "Pointcloud_Save": {"sample_ratio": 0.05, "conf_threshold_coef": 0.75},
    },
    "Weights": {"SALAD": _DEFAULT_SALAD_CKPT},
    "Loop": {
        # appearance-based candidates (SALAD VPR)
        "salad_enable": True,
        "SALAD": {"image_size": [322, 322], "batch_size": 32,
                  "similarity_threshold": 0.5, "top_k": 5,
                  "use_nms": True, "nms_threshold": 25},
        # geometric candidates: temporally-far but spatially-near frames in the
        # provisional (sequential-only) trajectory. Catches large-viewpoint-change
        # revisits that appearance descriptors miss.
        "geometric_enable": True,
        "geometric": {"min_time_gap": 30, "dist_thresh": 1.0,
                      "top_k_per_frame": 1, "nms_time": 20, "max_candidates": 30},
        # geometric verification applied to ALL candidates (salad + geometric):
        # re-infer the loop window jointly and require the two sides to actually
        # co-locate (camera-centre distance / scene scale) and a sane relative scale.
        # coloc_ratio 0.6 = sweet spot on scene0011 (0.5 rejects the true loop, 0.8+ admits
        # weaker ones); see docs/RESULTS.md coloc sweep.
        "verify": {"coloc_ratio": 0.6, "min_scale": 0.5, "max_scale": 2.0},
        "SIM3_Optimizer": {"lang_version": "python", "max_iterations": 30,
                           "lambda_init": "1e-6"},
    },
}


def get_chunk_indices(n: int, chunk_size: int, overlap: int):
    """Overlapping (start, end) windows tiling ``[0, n)`` (same scheme as DA3-Streaming)."""
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")
    if n <= chunk_size:
        return [(0, n)]
    step = chunk_size - overlap
    num = (n - overlap + step - 1) // step
    return [(i * step, min(i * step + chunk_size, n)) for i in range(num)]


class MaLongPipeline:
    def __init__(self, config: Optional[Dict] = None, model=None, device: str = "cuda"):
        self.cfg = config or DEFAULT_CONFIG
        self.device = device
        self._model = model  # lazily built so importing the pipeline needs no GPU

    @property
    def model(self):
        if self._model is None:
            from model.ma_infer import MaChunkModel
            self._model = MaChunkModel(device=self.device)
        return self._model

    def run(
        self,
        image_paths: Sequence[str],
        output_dir: str,
        *,
        mode: str = "rgb",
        depth_paths: Optional[Sequence[str]] = None,
        intrinsics: Optional[np.ndarray] = None,
        depth_scale: float = 1000.0,
        keep_cache: bool = True,
    ) -> Dict:
        m = self.cfg["Model"]
        chunk_size, overlap = m["chunk_size"], m["overlap"]
        n = len(image_paths)
        chunks = get_chunk_indices(n, chunk_size, overlap)
        print(f"[ma_long] {n} frames -> {len(chunks)} chunks "
              f"(size={chunk_size}, overlap={overlap}, mode={mode})")

        cache_dir = os.path.join(output_dir, "_chunks")
        pcd_dir = os.path.join(output_dir, "pcd")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(pcd_dir, exist_ok=True)

        # ---- 1+2+3: per-chunk inference (cached to disk) + sequential alignment ----
        sim3_list: List = []
        prev_wp = prev_conf = None
        for ci, (s, e) in enumerate(chunks):
            print(f"[ma_long] chunk {ci+1}/{len(chunks)}  frames [{s}:{e}]")
            out = self.model.infer_chunk(
                image_paths[s:e], mode=mode,
                depth_paths=None if depth_paths is None else depth_paths[s:e],
                intrinsics=intrinsics, depth_scale=depth_scale,
            )
            chunk = {k: (v.numpy() if v is not None and hasattr(v, "numpy") else v)
                     for k, v in out.items()}
            np.save(os.path.join(cache_dir, f"chunk_{ci}.npy"), chunk, allow_pickle=True)

            if ci > 0:
                pm1, c1 = prev_wp[-overlap:], prev_conf[-overlap:]
                pm2, c2 = chunk["world_points"][:overlap], chunk["world_points_conf"][:overlap]
                conf_threshold = float(min(np.median(c1), np.median(c2)) * 0.1)
                s_, R_, t_ = weighted_align_point_maps(
                    pm1, c1, pm2, c2, conf_threshold=conf_threshold, config=self.cfg,
                )
                sim3_list.append((s_, R_, t_))
                print(f"   aligned chunk {ci}->{ci-1}: s={s_:.4f} |t|={np.linalg.norm(t_):.4f}")

            prev_wp, prev_conf = chunk["world_points"], chunk["world_points_conf"]

        # ---- 3b: loop closure (optional) — dual (geometric + SALAD) + global opt ----
        n_loops = 0
        if m.get("loop_enable") and len(chunks) > 1 and sim3_list:
            cand = self._collect_candidates(sim3_list, chunks, image_paths, output_dir,
                                            cache_dir, n, overlap)
            loop_sim3 = self._loop_constraints(
                cand, chunks, image_paths, mode, depth_paths, intrinsics,
                depth_scale, cache_dir)
            n_loops = len(loop_sim3) if loop_sim3 else 0
            if loop_sim3:
                from align.sim3loop import Sim3LoopOptimizer
                print(f"[ma_long] loop closure: {len(sim3_list)} sequential + "
                      f"{len(loop_sim3)} verified loop constraints -> optimizing")
                sim3_list = Sim3LoopOptimizer(self.cfg, device="cpu").optimize(
                    sim3_list, loop_sim3)
            else:
                print("[ma_long] loop closure: no verified loop constraints")

        # ---- 4+5: accumulate, apply, save point clouds + global poses ----
        cum = accumulate_sim3_transforms(sim3_list) if sim3_list else []
        all_poses, all_K = self._assemble_poses(chunks, cum, cache_dir, overlap, n)
        self._save_pointclouds(chunks, cum, cache_dir, pcd_dir,
                               m["Pointcloud_Save"]["conf_threshold_coef"],
                               m["Pointcloud_Save"]["sample_ratio"], overlap,
                               fuse_overlap=m.get("fuse_overlap", True))

        combined_ply = os.path.join(output_dir, "combined_pcd.ply")
        merge_ply_files(pcd_dir, combined_ply)
        poses_txt = os.path.join(output_dir, "camera_poses.txt")
        self._write_poses(all_poses, poses_txt)
        print(f"[ma_long] done. poses -> {poses_txt}  pcd -> {combined_ply}")

        if not keep_cache:
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)

        return {"poses": all_poses, "intrinsics": all_K, "n_loops": n_loops,
                "poses_txt": poses_txt, "combined_ply": combined_ply}

    # --------------------------------------------------------- transforms -> outputs
    def _chunk_transform(self, cum, ci):
        """4x4 Sim3 (s*R | t) mapping chunk ci into the global frame; None for ci==0."""
        if ci == 0:
            return None, 1.0
        s_, R_, t_ = cum[ci - 1]
        S = np.eye(4, dtype=np.float64)
        S[:3, :3], S[:3, 3] = s_ * R_, t_
        return S, s_

    def _assemble_poses(self, chunks, cum, cache_dir, overlap, n):
        """Per-frame global c2w + K, using overlap_s=0 frame ownership (chunk ci owns
        [start, end-overlap) except the last chunk)."""
        all_poses: List[Optional[np.ndarray]] = [None] * n
        all_K: List[Optional[np.ndarray]] = [None] * n
        for ci, (s, e) in enumerate(chunks):
            chunk = np.load(os.path.join(cache_dir, f"chunk_{ci}.npy"), allow_pickle=True).item()
            poses, K = chunk["poses"], chunk["intrinsics"]
            S, s_ = self._chunk_transform(cum, ci)
            local_end = (e - s) if ci == len(chunks) - 1 else (e - s - overlap)
            for li in range(0, local_end):
                c2w = poses[li].astype(np.float64)
                if S is not None:
                    c2w = S @ c2w
                    c2w[:3, :3] /= s_
                all_poses[s + li] = c2w
                all_K[s + li] = K[li]
        return all_poses, all_K

    def _save_pointclouds(self, chunks, cum, cache_dir, pcd_dir, coef, ratio, overlap,
                          fuse_overlap=True):
        """Write one PLY per chunk over its *owned* frames only (no duplicated overlap),
        confidence-weighting the overlap band with the previous chunk's estimate
        (AMB3R-style fusion) so revisited surfaces don't ghost."""
        prev_tail_wp = prev_tail_conf = None  # previous chunk's last `overlap` frames (global)
        for ci, (s, e) in enumerate(chunks):
            chunk = np.load(os.path.join(cache_dir, f"chunk_{ci}.npy"), allow_pickle=True).item()
            wp, conf, imgs = chunk["world_points"], chunk["world_points_conf"], chunk["images"]
            if self._chunk_transform(cum, ci)[0] is not None:
                s_, R_, t_ = cum[ci - 1]
                wp = apply_sim3_direct(wp, s_, R_, t_)

            last = ci == len(chunks) - 1
            local_end = (e - s) if last else (e - s - overlap)  # owned frames [0, local_end)
            owned_wp, owned_conf = wp[:local_end].copy(), conf[:local_end]
            owned_imgs = imgs[:local_end]

            # fuse the leading overlap band with the previous chunk's trailing band
            if fuse_overlap and prev_tail_wp is not None:
                k = min(overlap, local_end, len(prev_tail_wp))
                ca, cb = owned_conf[:k][..., None], prev_tail_conf[:k][..., None]
                owned_wp[:k] = (ca * owned_wp[:k] + cb * prev_tail_wp[:k]) / (ca + cb + 1e-8)

            confs = owned_conf.reshape(-1)
            save_confident_pointcloud_batch(
                points=owned_wp, colors=owned_imgs, confs=owned_conf,
                output_path=os.path.join(pcd_dir, f"{ci}_pcd.ply"),
                conf_threshold=float(np.mean(confs) * coef), sample_ratio=ratio,
            )
            prev_tail_wp = None if last else wp[-overlap:].copy()
            prev_tail_conf = None if last else conf[-overlap:]

    # ------------------------------------------------------------------ loop closure
    def _collect_candidates(self, sim3_list, chunks, image_paths, output_dir,
                            cache_dir, n, overlap):
        """Union of SALAD (appearance) and geometric (provisional-trajectory) loop
        candidates, as a deduped list of (i, j) frame pairs with i > j."""
        L = self.cfg["Loop"]
        cands = set()
        if L.get("salad_enable", True):
            for i, j in self._detect_loops(image_paths, output_dir):
                cands.add((max(i, j), min(i, j)))
        if L.get("geometric_enable", True):
            cum = accumulate_sim3_transforms(sim3_list)
            prov_poses, _ = self._assemble_poses(chunks, cum, cache_dir, overlap, n)
            geo = self._geometric_candidates(prov_poses)
            print(f"[ma_long] geometric proposal found {len(geo)} candidate pairs:")
            for i, j in geo:
                print(f"    frame {i:3d} <-> {j:3d}  (span {abs(i-j):3d})  [geometric]")
            cands.update((max(i, j), min(i, j)) for i, j in geo)
        return sorted(cands)

    def _geometric_candidates(self, poses):
        """Temporally-far, spatially-near frame pairs in the provisional trajectory."""
        g = self.cfg["Loop"]["geometric"]
        pos = np.array([(p[:3, 3] if p is not None else [np.nan] * 3) for p in poses])
        n = len(pos)
        scored = []
        for i in range(n):
            hi = i - g["min_time_gap"]
            if hi <= 0 or not np.isfinite(pos[i]).all():
                continue
            d = np.linalg.norm(pos[:hi] - pos[i], axis=1)
            d[~np.isfinite(d)] = np.inf
            for j in np.argsort(d)[: g["top_k_per_frame"]]:
                if d[j] < g["dist_thresh"]:
                    scored.append((float(d[j]), i, int(j)))
        # temporal NMS: greedily keep closest pairs, suppress nearby (in time) ones
        scored.sort(key=lambda x: x[0])
        kept, used = [], []
        for dist, i, j in scored:
            if any(abs(i - ki) < g["nms_time"] and abs(j - kj) < g["nms_time"]
                   for ki, kj in used):
                continue
            kept.append((i, j)); used.append((i, j))
            if len(kept) >= g["max_candidates"]:
                break
        return kept

    def _align(self, pm1, c1, pm2, c2):
        """Robust transform mapping pm2 -> pm1 over confident overlap pixels."""
        conf_threshold = float(min(np.median(c1), np.median(c2)) * 0.1)
        return weighted_align_point_maps(pm1, c1, pm2, c2,
                                         conf_threshold=conf_threshold, config=self.cfg)

    def _detect_loops(self, image_paths, output_dir):
        """Run SALAD VPR loop detection; returns deduped [(i, j)] frame pairs (i > j)."""
        from pathlib import Path
        from loop.loop_detector import LoopDetector
        det = LoopDetector(
            image_dir=os.path.dirname(image_paths[0]),
            output=os.path.join(output_dir, "loop_closures.txt"), config=self.cfg)
        det.image_paths = [Path(p) for p in image_paths]  # keep pipeline frame indexing
        det.load_model()
        pairs = det.find_loop_closures()  # [(i, j, sim)], i > j
        det.save_results()
        del det.model
        import torch
        torch.cuda.empty_cache()
        print(f"[ma_long] loop detector found {len(pairs)} candidate pairs:")
        for i, j, s in pairs:
            print(f"    frame {int(i):3d} <-> {int(j):3d}  (span {abs(int(i)-int(j)):3d})  sim={s:.3f}")
        return [(int(i), int(j)) for i, j, _ in pairs]

    def _loop_constraints(self, loop_pairs, chunks, image_paths, mode,
                          depth_paths, intrinsics, depth_scale, cache_dir):
        """Re-infer loop windows and turn each into a chunk_a<->chunk_b Sim3 constraint."""
        if not loop_pairs:
            return []
        V = self.cfg["Loop"]["verify"]
        half = self.cfg["Model"]["loop_chunk_size"] // 2
        items = process_loop_list(chunks, loop_pairs, half_window=half)
        seen, uniq = set(), []
        for it in items:  # dedup on (chunk_a, chunk_b)
            key = (it[0], it[2])
            if key not in seen and it[0] != it[2]:
                seen.add(key); uniq.append(it)

        def slice_paths(rng):
            return list(image_paths[rng[0]:rng[1]])

        loop_sim3 = []
        for cia, ra, cib, rb in uniq:
            la, lb = ra[1] - ra[0], rb[1] - rb[0]
            dp = (None if depth_paths is None
                  else list(depth_paths[ra[0]:ra[1]]) + list(depth_paths[rb[0]:rb[1]]))
            out = self.model.infer_chunk(slice_paths(ra) + slice_paths(rb), mode=mode,
                                         depth_paths=dp, intrinsics=intrinsics,
                                         depth_scale=depth_scale)
            lw = out["world_points"].numpy(); lc = out["world_points_conf"].numpy()
            lpose = out["poses"].numpy()  # (la+lb, 4, 4) in the loop chunk's own frame

            # --- geometric verification: do the two windows actually co-locate? ---
            centers = lpose[:, :3, 3]
            ref = float(np.median(np.linalg.norm(centers - centers.mean(0), axis=1))) + 1e-6
            coloc = float(np.linalg.norm(centers[:la].mean(0) - centers[la:].mean(0)) / ref)
            if coloc > V["coloc_ratio"]:
                print(f"   reject {cia}<->{cib}: not co-located (coloc={coloc:.2f})")
                continue

            ca = np.load(os.path.join(cache_dir, f"chunk_{cia}.npy"), allow_pickle=True).item()
            cb = np.load(os.path.join(cache_dir, f"chunk_{cib}.npy"), allow_pickle=True).item()
            a0, b0 = chunks[cia][0], chunks[cib][0]
            pm_a, conf_a = ca["world_points"][ra[0]-a0:ra[1]-a0], ca["world_points_conf"][ra[0]-a0:ra[1]-a0]
            pm_b, conf_b = cb["world_points"][rb[0]-b0:rb[1]-b0], cb["world_points_conf"][rb[0]-b0:rb[1]-b0]
            try:
                S_a = self._align(pm_a, conf_a, lw[:la], lc[:la])         # loop_a -> chunk_a
                S_b = self._align(pm_b, conf_b, lw[la:la+lb], lc[la:la+lb])  # loop_b -> chunk_b
            except Exception as ex:
                print(f"   loop {cia}<->{cib} align failed: {ex}"); continue
            sim3 = compute_sim3_ab(S_a, S_b)
            if not (V["min_scale"] <= sim3[0] <= V["max_scale"]):
                print(f"   reject {cia}<->{cib}: bad scale s={sim3[0]:.3f}")
                continue
            loop_sim3.append((cia, cib, sim3))
            print(f"   accept loop constraint chunk {cia}<->{cib} (coloc={coloc:.2f})")
        return loop_sim3

    @staticmethod
    def _write_poses(all_poses: List[Optional[np.ndarray]], path: str):
        with open(path, "w") as f:
            for p in all_poses:
                p = np.eye(4) if p is None else p
                f.write(" ".join(str(x) for x in p.flatten()) + "\n")
