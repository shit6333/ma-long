"""ma_slam Solver — VGGT-SLAM-style submap SLAM on a MapAnything backbone (metric SE3).

Pipeline per submap (mirrors ``vggt_slam/solver.py``, adapted to metric geometry):
  1. Chunk: ``submap_size + overlap`` consecutive frames; the last frame is shared with
     the next submap (``overlap=1``).
  2. Infer: one ``MaChunkModel.infer_chunk`` call (mode = rgb / rgb+intr / rgb+depth /
     rgb+depth+intr) → local point maps, conf, depth, cam->world poses, intrinsics.
  3. Place: align the submap to the global map through the shared overlap frame — the
     current submap's first frame == the previous submap's last frame, so the new submap
     is rigidly placed at the previous frame's optimized global pose. In scaleless modes
     (rgb / rgb+intr) a pairwise depth-ratio scale is folded in first so the placement
     stays SE3.
  4. Graph: every frame is a ``Pose3`` node; add intra-submap (consecutive) + inter-submap
     (overlap = identity tie) constraints, plus any verified loop constraints.
  5. Optimize: re-run gtsam LM over the whole graph after every submap (incremental global).

Loop closure: SALAD retrieves a revisited earlier frame; the pair is re-inferred and
co-location-verified (metric analogue of VGGT-SLAM's ``image_match_ratio`` gate) before a
loop ``BetweenFactorPose3`` is added.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from ma_slam.graph import (PoseGraph, Sim3PoseGraph, _sim3_mul, _sim3_inv,
                           _mat_to_sim3, _sim3_to_mat)
from ma_slam.map import GraphMap
from ma_slam.submap import Submap
from ma_slam.loop import ImageRetrieval
from align.sim3utils import weighted_align_point_maps

DEFAULT_CONFIG: Dict = {
    "submap_size": 16,        # new frames per submap (excludes the shared overlap frame)
    "overlap": 1,             # shared frames between consecutive submaps (only 1 supported)
    "conf_percentile": 25.0,  # per-submap conf threshold (drop lowest X% when exporting)
    "scale_correction": True, # auto-gated: only applied in scaleless modes (no depth input)
    # Inter-submap constraint: optionally align the shared overlap frame's dense point maps
    # (robust IRLS) for a geometry-based relative SE3 instead of the single-frame pose tie.
    # OFF by default: at overlap=1 it's neutral-to-slightly-worse (one frame's point-align is
    # no better than MapAnything's already-consistent predicted pose). The real win needs
    # overlap>1 (multi-frame robust averaging), which is not wired yet.
    "InterSubmap": {"align": False, "align_lib": "torch",
                    "IRLS": {"delta": 0.1, "max_iters": 5, "tol": "1e-9"}},
    "Graph": {"manifold": "se3", "inner_sigma": 0.03, "intra_sigma": 0.05,
              "anchor_sigma": 1e-6, "loop_sigma": 0.10,
              "loop_robust": "huber", "loop_robust_k": 1.345},
    "Loop": {"enable": True, "sim_threshold": 0.50, "min_submap_gap": 2,
             "coloc_ratio": 0.70, "max_per_submap": 1, "half_window": 0,
             # debug: dump each loop candidate's two re-inferred windows as a green(query)/
             # red(match) PLY in <out>/coloc/ so co-location accept/reject can be eyeballed.
             "debug_coloc": False,
             # used only by the Sim3 backend (manifold='sim3') via Sim3LoopOptimizer
             "SIM3_Optimizer": {"lang_version": "python", "max_iterations": 30, "lambda_init": "1e-6"}},
    "Pointcloud": {"voxel_size": 0.0, "max_points": 2_000_000, "conf_coef": 0.75},
}


class MaSlam:
    def __init__(self, config: Optional[Dict] = None, model=None, device: str = "cuda"):
        self.cfg = config or DEFAULT_CONFIG
        self.device = device
        self._model = model
        g = self.cfg["Graph"]
        if g["manifold"] == "sim3":
            self.graph = Sim3PoseGraph(self.cfg)        # pypose Sim3LoopOptimizer backend
        elif g["manifold"] == "sl4":
            raise NotImplementedError(
                "manifold='sl4' needs the MIT-SPARK gtsam fork with SL4 "
                "(build thirdparty/gtsam_with_sl4); not wired yet.")
        else:
            self.graph = PoseGraph(manifold="se3", inner_sigma=g["inner_sigma"],
                                   intra_sigma=g["intra_sigma"], anchor_sigma=g["anchor_sigma"],
                                   loop_sigma=g["loop_sigma"],
                                   loop_robust=g.get("loop_robust"), loop_robust_k=g.get("loop_robust_k", 1.345))
        self.map = GraphMap()
        self.retrieval = ImageRetrieval(device=device) if self.cfg["Loop"]["enable"] else None
        self._next_base = 0
        # set in run()
        self._mode = "rgb"; self._input_K = None; self._depth_scale = 1000.0; self._depth_max = None
        self._coloc_dir: Optional[str] = None    # set in run() when Loop.debug_coloc is on
        self.stats = dict(n_loops=0, n_submaps=0)
        self.loop_log: List[str] = []   # every loop candidate decision (saved to loops.txt)

    @property
    def model(self):
        if self._model is None:
            from model.ma_infer import MaChunkModel
            self._model = MaChunkModel(device=self.device)
        return self._model

    # ------------------------------------------------------------------ inference
    def _infer(self, image_paths, depth_paths):
        out = self.model.infer_chunk(
            list(image_paths), mode=self._mode,
            depth_paths=None if depth_paths is None else list(depth_paths),
            intrinsics=self._input_K, depth_scale=self._depth_scale, depth_max=self._depth_max)
        return {k: (v.numpy() if v is not None and hasattr(v, "numpy") else v)
                for k, v in out.items()}

    def _build_submap(self, base, image_paths, depth_paths, out) -> Submap:
        sm = Submap(base)
        sm.set_frames(out["images"], image_paths)   # frames stored for parity; SALAD uses paths
        sm.set_geometry(points=out["world_points"], colors=out["images"],
                        conf=out["world_points_conf"], poses_local=out["poses"],
                        intrinsics=out["intrinsics"], conf_percentile=self.cfg["conf_percentile"])
        sm.depth = out["depth"]
        sm.mask = out.get("mask")
        sm.depth_paths = None if depth_paths is None else list(depth_paths)
        return sm

    # ------------------------------------------------------------------ scale
    def _overlap_scale(self, prev: Submap, prev_i: int, curr: Submap, curr_i: int) -> float:
        """Median depth ratio (prev/curr) over the shared overlap frame -> rescale curr."""
        dp, dc = prev.depth[prev_i], curr.depth[curr_i]
        m = (dp > 0) & (dc > 0) & prev.conf_mask_frame(prev_i) & curr.conf_mask_frame(curr_i)
        if m.sum() < 100:
            m = (dp > 0) & (dc > 0)
        if m.sum() < 100:
            return 1.0
        s = float(np.median(dp[m] / dc[m]))
        return s if 0.2 < s < 5.0 else 1.0   # reject implausible scales

    # ------------------------------------------------------------------ graph wiring
    def _add_first_submap(self, sm: Submap):
        for i in range(sm.n):
            self.graph.add_pose(sm.key(i), sm.poses_local[i])
        self.graph.add_prior(sm.key(0), sm.poses_local[0])      # anchor world gauge
        for i in range(1, sm.n):
            rel = np.linalg.inv(sm.poses_local[i - 1]) @ sm.poses_local[i]
            self.graph.add_between(sm.key(i - 1), sm.key(i), rel, kind="inner")

    def _align_overlap(self, prev: Submap, pl: int, curr: Submap, cl: int, force_sim3: bool = False):
        """Robust IRLS alignment of the overlap frame's dense point maps.

        The overlap frame is the SAME physical image in both submaps (pixel-wise
        correspondence), reconstructed in each submap's local frame. Aligning the two dense
        point maps yields a geometry-based ``(s, R, t)`` mapping curr-local -> prev-local —
        a far stronger inter-submap measurement than tying the single predicted camera pose.
        ``force_sim3`` estimates the inter-submap scale even for metric models (Sim3 backend).
        """
        Qp, cp = prev.points[pl], prev.conf[pl]
        Qc, cc = curr.points[cl], curr.conf[cl]
        ct = float(min(np.median(cp), np.median(cc)) * 0.1)
        method = "sim3" if force_sim3 else ("se3" if self.model.is_metric(self._mode) else "sim3")
        ic = self.cfg["InterSubmap"]
        acfg = {"Model": {"align_lib": ic["align_lib"], "align_method": method, "IRLS": ic["IRLS"]}}
        return weighted_align_point_maps(Qp[None], cp[None], Qc[None], cc[None],
                                         conf_threshold=ct, config=acfg)

    def _add_submap(self, sm: Submap, prev: Submap):
        if self.cfg["Graph"]["manifold"] == "sim3":
            self._add_submap_sim3(sm, prev)
        else:
            self._add_submap_se3(sm, prev)

    def _add_submap_sim3(self, sm: Submap, prev: Submap):
        """Sim3 placement: align the overlap point maps WITH scale and accumulate the
        inter-submap relative as a Sim3 (no per-submap pre-scaling; the graph carries scale)."""
        pl = prev.last_frame_index()
        g_pl = _mat_to_sim3(self.graph.get_pose(prev.key(pl)))   # (s,R,t) world<-cam of overlap frame
        try:
            s, R, t = self._align_overlap(prev, pl, sm, 0, force_sim3=True)   # Sc -> Sp, with scale
            Tsc = (s, R, t)
        except Exception as e:
            print(f"[ma_slam] sim3 overlap align failed ({e}); identity tie")
            Tsc = (1.0, np.eye(3), np.zeros(3))
        Lpl = (1.0, prev.poses_local[pl][:3, :3], prev.poses_local[pl][:3, 3])   # SE3 (s=1)
        M_pl = _sim3_mul(g_pl, _sim3_inv(*Lpl))      # world <- prev-local (Sim3)
        M_sc = _sim3_mul(M_pl, Tsc)                  # world <- curr-local (Sim3)
        for i in range(sm.n):
            Li = (1.0, sm.poses_local[i][:3, :3], sm.poses_local[i][:3, 3])
            self.graph.add_pose(sm.key(i), _sim3_to_mat(*_sim3_mul(M_sc, Li)))
        # intra/inter ties are encoded by the placement; loops added in _loop_closure.

    def _add_submap_se3(self, sm: Submap, prev: Submap):
        pl = prev.last_frame_index()
        G_pl = self.graph.get_pose(prev.key(pl))                 # overlap frame global cam->world
        T_curr = None
        if self.cfg["InterSubmap"].get("align", True):
            try:
                s, R, t = self._align_overlap(prev, pl, sm, 0)   # curr-local -> prev-local
                if self.cfg["scale_correction"] and not self.model.is_metric(self._mode):
                    sm.apply_scale(s)                            # fold inter-submap scale into curr
                T_sp_sc = np.eye(4); T_sp_sc[:3, :3], T_sp_sc[:3, 3] = R, t
                T_curr = (G_pl @ np.linalg.inv(prev.poses_local[pl])) @ T_sp_sc   # world <- curr-local
            except Exception as e:
                print(f"[ma_slam] overlap align failed ({e}); pose-tie fallback")
                T_curr = None
        if T_curr is None:                                       # single-frame pose tie (+ depth-ratio scale)
            if self.cfg["scale_correction"] and not self.model.is_metric(self._mode):
                sm.apply_scale(self._overlap_scale(prev, pl, sm, 0))
            T_curr = G_pl @ np.linalg.inv(sm.poses_local[0])
        for i in range(sm.n):
            self.graph.add_pose(sm.key(i), T_curr @ sm.poses_local[i])
        # inter-submap constraint between the two reconstructions of the overlap frame.
        rel_overlap = np.linalg.inv(self.graph.get_pose(prev.key(pl))) @ self.graph.get_pose(sm.key(0))
        self.graph.add_between(prev.key(pl), sm.key(0), rel_overlap, kind="intra")
        # intra-submap consecutive constraints.
        for i in range(1, sm.n):
            rel = np.linalg.inv(sm.poses_local[i - 1]) @ sm.poses_local[i]
            self.graph.add_between(sm.key(i - 1), sm.key(i), rel, kind="inner")

    # ------------------------------------------------------------------ loop closure
    def _loop_closure(self, sm: Submap):
        lc = self.cfg["Loop"]
        descs = self.retrieval.describe(sm.image_paths)
        recent = {b for b in [self.map.largest_base(), sm.base_id]
                  if b is not None}
        # also exclude the most-recent `min_submap_gap` submaps
        ordered = sorted(self.map.submaps)
        for b in ordered[-lc["min_submap_gap"]:]:
            recent.add(b)
        cands = self.retrieval.query(descs, sm.base_id, lc["sim_threshold"], recent)
        cands = sorted(cands, key=lambda c: -c.similarity)[:lc["max_per_submap"]]
        accepted = 0
        for c in cands:
            if self._add_loop_constraint(sm, c):
                accepted += 1
        self.retrieval.add(descs, sm.base_id)
        return accepted

    def _add_loop_constraint(self, sm: Submap, cand) -> bool:
        """Re-infer a window around each candidate frame jointly, verify co-location, add edge.

        ``half_window`` (hw) controls the verification window: hw=0 re-infers just the two
        candidate frames; hw>0 also pulls each frame's in-submap neighbours so MapAnything
        has more co-visible geometry to register large-viewpoint revisits (which 2 frames
        alone often can't). Co-location compares the two windows' camera centroids,
        normalised by scene depth; the loop measurement is the relative pose between the
        actual query/match frames in the joint reconstruction.
        """
        hw = self.cfg["Loop"].get("half_window", 0)
        match_sm = self.map.get(cand.match_base)
        qa = list(range(max(0, cand.query_frame - hw), min(sm.n, cand.query_frame + hw + 1)))
        mb = list(range(max(0, cand.match_frame - hw), min(match_sm.n, cand.match_frame + hw + 1)))
        q_paths = [sm.image_paths[i] for i in qa]
        m_paths = [match_sm.image_paths[i] for i in mb]
        q_dp = None if sm.depth_paths is None else [sm.depth_paths[i] for i in qa]
        m_dp = None if match_sm.depth_paths is None else [match_sm.depth_paths[i] for i in mb]
        dpaths = None if q_dp is None else q_dp + m_dp
        out = self._infer(q_paths + m_paths, dpaths)
        poses = out["poses"]                       # (len(qa)+len(mb), 4, 4), joint-local frame
        la = len(qa)
        # geometric verification: the two windows' camera centroids must co-locate (scene-scale).
        ctr = poses[:, :3, 3]
        d = out["depth"]; valid = d > 0
        ref = float(np.median(d[valid])) + 1e-6 if valid.any() else 1.0
        coloc = float(np.linalg.norm(ctr[:la].mean(0) - ctr[la:].mean(0)) / ref)
        passed = coloc <= self.cfg["Loop"]["coloc_ratio"]
        # Parseable per-candidate record (image paths included for manual inspection).
        qimg, mimg = sm.image_paths[cand.query_frame], match_sm.image_paths[cand.match_frame]
        line = (f"LOOPCAND {'ACCEPT' if passed else 'REJECT'} "
                f"coloc={coloc:.3f} sim={cand.similarity:.2f} hw={hw} "
                f"q={cand.query_base}+{cand.query_frame} m={cand.match_base}+{cand.match_frame} "
                f"qimg={qimg} mimg={mimg}")
        self.loop_log.append(line)
        print(f"[ma_slam] {line}")
        if self._coloc_dir is not None:
            self._save_coloc_ply(out, la, len(self.loop_log), passed, coloc, cand)
        if not passed:
            return False
        # measurement between the actual query and match frames in the joint reconstruction.
        qi = qa.index(cand.query_frame)
        mi = la + mb.index(cand.match_frame)
        rel = np.linalg.inv(poses[qi]) @ poses[mi]
        self.graph.add_between(sm.key(cand.query_frame), match_sm.key(cand.match_frame),
                               rel, kind="loop")
        self.stats["n_loops"] += 1
        return True

    # ------------------------------------------------------------------ coloc debug
    @staticmethod
    def _conf_points(pts, conf, color, cap=200_000, rng=None):
        """Confident points (conf > median) of a window, painted a solid color."""
        P = pts.reshape(-1, 3)
        c = conf.reshape(-1)
        m = (c > np.median(c)) & np.isfinite(P).all(1)
        P = P[m]
        if cap and len(P) > cap:
            rng = rng if rng is not None else np.random.default_rng(0)
            P = P[rng.choice(len(P), cap, replace=False)]
        col = np.tile(np.asarray(color, np.uint8), (len(P), 1))
        return P, col

    def _save_coloc_ply(self, out, la, idx, passed, coloc, cand):
        """Dump the two re-inferred loop windows (query=green, match=red) to one PLY.

        Points are in the JOINT-local reconstruction frame (the same frame coloc is measured
        in), so if the loop is real the green and red clouds should sit on top of each other;
        a rejected candidate shows them flying apart.
        """
        wp, cf = out["world_points"], out["world_points_conf"]
        rng = np.random.default_rng(0)
        pq, cq = self._conf_points(wp[:la], cf[:la], (0, 220, 0), rng=rng)    # query = green
        pm, cm = self._conf_points(wp[la:], cf[la:], (220, 0, 0), rng=rng)    # match = red
        xyz = np.concatenate([pq, pm], 0).astype("<f4")
        rgb = np.concatenate([cq, cm], 0).astype("u1")
        tag = "ACCEPT" if passed else "REJECT"
        fn = (f"coloc_{idx:02d}_{tag}_c{coloc:.3f}_"
              f"q{cand.query_base}+{cand.query_frame}_m{cand.match_base}+{cand.match_frame}.ply")
        path = os.path.join(self._coloc_dir, fn)
        vert = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                         ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        vert["x"], vert["y"], vert["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        vert["red"], vert["green"], vert["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        header = ("ply\nformat binary_little_endian 1.0\n"
                  f"element vertex {len(vert)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  "end_header\n").encode()
        with open(path, "wb") as fh:
            fh.write(header); fh.write(vert.tobytes())
        print(f"[ma_slam] coloc ply -> {path} ({len(vert):,} pts: green=query red=match)")

    # ------------------------------------------------------------------ run
    def process_submap(self, image_paths, depth_paths):
        out = self._infer(image_paths, depth_paths)
        base = self._next_base
        sm = self._build_submap(base, image_paths, depth_paths, out)
        prev = self.map.latest(ignore_lc=True)
        if prev is None:
            self._add_first_submap(sm)
        else:
            self._add_submap(sm, prev)   # placement + scale handled inside (overlap align)
        self.map.add(sm)
        self._next_base += sm.n
        self.stats["n_submaps"] += 1

        if self.retrieval is not None and prev is not None:
            self._loop_closure(sm)

        self.graph.optimize()

    def run(self, image_paths: Sequence[str], output_dir: str, *, mode: str = "rgb",
            depth_paths: Optional[Sequence[str]] = None,
            intrinsics: Optional[np.ndarray] = None, depth_scale: float = 1000.0,
            depth_max: Optional[float] = None) -> Dict:
        import time
        import torch
        self._mode, self._input_K, self._depth_scale = mode, intrinsics, depth_scale
        self._depth_max = depth_max
        os.makedirs(output_dir, exist_ok=True)
        if self.cfg["Loop"].get("debug_coloc") and self.cfg["Loop"]["enable"]:
            self._coloc_dir = os.path.join(output_dir, "coloc")
            os.makedirs(self._coloc_dir, exist_ok=True)
        n = len(image_paths)
        W = self.cfg["submap_size"] + self.cfg["overlap"]
        step = self.cfg["submap_size"]

        cuda = torch.cuda.is_available()
        _ = self.model                       # force model load before timing (exclude load)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        starts = list(range(0, n, step))
        for s in starts:
            idxs = list(range(s, min(s + W, n)))
            if s > 0 and len(idxs) <= self.cfg["overlap"]:
                break                                   # trailing remnant already covered
            paths = [image_paths[i] for i in idxs]
            dpaths = None if depth_paths is None else [depth_paths[i] for i in idxs]
            self.process_submap(paths, dpaths)
            print(f"[ma_slam] submap {self.stats['n_submaps']} frames[{idxs[0]}:{idxs[-1]}] "
                  f"| nodes={len(self.graph.initialized)} loops={self.stats['n_loops']}")

        elapsed = time.time() - t0
        fps = n / elapsed if elapsed > 0 else 0.0
        vram = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
        print(f"[ma_slam] timing: {elapsed:.1f}s for {n} frames = {fps:.2f} fps "
              f"(streaming, model-load excluded) | peak VRAM {vram:.1f} GB | "
              f"submap_size={self.cfg['submap_size']}")

        poses_txt = os.path.join(output_dir, "camera_poses.txt")
        self.map.write_poses(self.graph, poses_txt)
        combined_ply = os.path.join(output_dir, "combined_pcd.ply")
        pc = self.cfg["Pointcloud"]
        self.map.write_points(self.graph, combined_ply, voxel_size=pc["voxel_size"],
                              max_points=pc["max_points"], conf_coef=pc["conf_coef"])
        print(f"[ma_slam] done. {n} frames | submaps={self.map.num_submaps()} "
              f"loops={self.graph.get_num_loops()} | poses -> {poses_txt}  pcd -> {combined_ply}")

        # always persist run stats + loop events to the output dir (esp. for no-GT in-the-wild runs)
        n_accept = sum(1 for ln in self.loop_log if "ACCEPT" in ln)
        stats = {
            "backend": type(self.model).__name__, "mode": mode,
            "submap_size": self.cfg["submap_size"], "overlap": self.cfg["overlap"],
            "n_frames": n, "n_submaps": self.map.num_submaps(),
            "loops_accepted": n_accept, "loops_candidates": len(self.loop_log),
            "fps": round(fps, 2), "seconds": round(elapsed, 1), "peak_vram_gb": round(vram, 2),
            "coloc_ratio": self.cfg["Loop"]["coloc_ratio"], "loop_enabled": self.cfg["Loop"]["enable"],
            "depth_max": self._depth_max,
        }
        import json
        with open(os.path.join(output_dir, "run_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        with open(os.path.join(output_dir, "run_stats.txt"), "w") as f:
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
        with open(os.path.join(output_dir, "loops.txt"), "w") as f:
            f.write(f"# {n_accept} accepted / {len(self.loop_log)} candidates\n")
            f.write("\n".join(self.loop_log) + ("\n" if self.loop_log else ""))
        print(f"[ma_slam] stats -> {os.path.join(output_dir, 'run_stats.txt')}  "
              f"loops -> {os.path.join(output_dir, 'loops.txt')}")
        return {"n_submaps": self.map.num_submaps(), "n_loops": self.graph.get_num_loops(),
                "poses_txt": poses_txt, "combined_ply": combined_ply,
                "fps": fps, "seconds": elapsed, "peak_vram_gb": vram}
