"""ma_long online SLAM: AMB3R-style fused-map keyframe front-end + online loop closure.

Design (faithful to AMB3R's slam/, on a MapAnything backbone for stable metric scale):
  - Streaming keyframe windows: each step reconstructs [active keyframes] + [new frames]
    together; the active keyframes (whose global geometry is already in the map) align the
    window back to the global frame.
  - **Persistent fused map** (the AMB3R core, `weighted_average_alignment`): every frame's
    points/pose live in a persistent map and are refined by confidence + iteration-weighted
    fusion each time they are re-observed (keyframes get cleaner over time).
  - Keyframe selection (pose-distance + confidence) and bounded resampling (Points 1 & 3).
  - **Online loop closure** (the piece AMB3R lacks): when a new keyframe is spatially near an
    old one but temporally far, re-infer a loop window, verify co-location, add a loop edge,
    and run an incremental Sim3 pose-graph optimization over the keyframes RIGHT THEN —
    correcting the whole loop and propagating to all owned frames. Not a final batch pass.

Outputs match the other pipelines (global camera_poses.txt + merged point cloud).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from align.sim3utils import (
    accumulate_sim3_transforms, apply_sim3_direct, compute_sim3_ab, merge_ply_files,
    save_confident_pointcloud_batch, weighted_align_point_maps,
)
from align.sim3loop import Sim3LoopOptimizer
from ma_long.tools.keyframes import resample_keyframes
from ma_long.tools.pose_dist import extrinsic_distance
from ma_long.tools.pose_interp import interpolate_poses

SLAM_DEFAULT_CONFIG: Dict = {
    "Model": {
        "init_window": 20, "step": 8,
        "align_lib": "torch", "align_method": "se3",
        "IRLS": {"delta": 0.1, "max_iters": 5, "tol": "1e-9"},
        "Pointcloud_Save": {"sample_ratio": 0.05, "conf_threshold_coef": 0.75},
    },
    "Keyframe": {
        "threshold": 0.3, "max_active": 10, "min_active": 7,
        "top_k": 4, "bridge": 1, "lambda_t": 1.0,
    },
    "Loop": {
        "enable": True,
        # Detection must be GENEROUS: a big loop has accumulated drift, so its two ends are
        # FAR apart in the (drifted) live estimate even though they are the same place. A tight
        # `dist` misses exactly the large loops that matter; geometric verification (coloc) then
        # rejects the false positives this admits. ~1/5 of the scene extent is a good start.
        "dist": 1.2,            # metres: spatial proximity for an online loop candidate
        "min_gap": 40,          # frames: temporal separation required
        "half_window": 8,       # loop-window radius for re-inference (bigger = better constraint)
        "coloc_ratio": 0.6,     # geometric verification (two sides must co-locate)
        "SIM3_Optimizer": {"lang_version": "python", "max_iterations": 30, "lambda_init": "1e-6"},
    },
}


def _sim3_inv(s, R, t):
    return 1.0 / s, R.T, -(R.T @ t) / s


def _sim3_mul(a, b):
    sa, Ra, ta = a; sb, Rb, tb = b
    return sa * sb, Ra @ Rb, sa * (Ra @ tb) + ta


def _c2w_to_sim3(M):
    return 1.0, M[:3, :3].copy(), M[:3, 3].copy()


def _sim3_to_c2w(s, R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    return M


class MaLongSLAM:
    def __init__(self, config: Optional[Dict] = None, model=None, device: str = "cuda"):
        self.cfg = config or SLAM_DEFAULT_CONFIG
        self.device = device
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from model.ma_infer import MaChunkModel
            self._model = MaChunkModel(device=self.device)
        return self._model

    def _infer(self, idxs, image_paths, mode, depth_paths, intrinsics, depth_scale):
        out = self.model.infer_chunk(
            [image_paths[i] for i in idxs], mode=mode,
            depth_paths=None if depth_paths is None else [depth_paths[i] for i in idxs],
            intrinsics=intrinsics, depth_scale=depth_scale)
        return {k: (v.numpy() if v is not None and hasattr(v, "numpy") else v)
                for k, v in out.items()}

    def _align(self, g_pts, g_conf, l_pts, l_conf):
        """Robust transform mapping local anchors -> their global geometry (s, R, t)."""
        ct = float(min(np.median(g_conf), np.median(l_conf)) * 0.1)
        return weighted_align_point_maps(g_pts, g_conf, l_pts, l_conf,
                                         conf_threshold=ct, config=self.cfg)

    def run(self, image_paths: Sequence[str], output_dir: str, *, mode: str = "rgb",
            depth_paths: Optional[Sequence[str]] = None,
            intrinsics: Optional[np.ndarray] = None, depth_scale: float = 1000.0,
            keep_cache: bool = True) -> Dict:
        m, kf_c, loop_c = self.cfg["Model"], self.cfg["Keyframe"], self.cfg["Loop"]
        init_w, step = m["init_window"], m["step"]
        thr, lam = kf_c["threshold"], kf_c["lambda_t"]
        n = len(image_paths)
        os.makedirs(output_dir, exist_ok=True)

        # ---- persistent map (lazily sized after first inference) ----
        MAP: Dict[str, np.ndarray] = {}
        valid = np.zeros(n, dtype=bool)
        owner = np.full(n, -1, dtype=int)     # which keyframe each frame is anchored to
        pool: List[int] = []                  # all keyframe indices (global graph nodes)
        active: List[int] = []                # bounded active anchor set
        loop_edges: List[tuple] = []          # (node_i, node_j, sim3) constraints
        stats = dict(n_loops=0, n_optimize=0)

        def alloc(H, W):
            MAP["pts"] = np.zeros((n, H, W, 3), np.float32)
            MAP["conf"] = np.zeros((n, H, W), np.float32)
            MAP["img"] = np.zeros((n, H, W, 3), np.uint8)
            MAP["pose"] = np.tile(np.eye(4), (n, 1, 1)).astype(np.float64)
            MAP["K"] = np.tile(np.eye(3), (n, 1, 1)).astype(np.float64)
            MAP["iter"] = np.zeros(n, np.float32)

        def conf_sig(c):  # DUSt3R-family confidence -> (0,1) weight
            cs = (c - 1.0) / np.maximum(c, 1e-6); return np.maximum(cs, 1e-6)

        def write_new(idx, pts, conf, pose, img, K):
            MAP["pts"][idx], MAP["conf"][idx] = pts, conf
            MAP["pose"][idx], MAP["img"][idx], MAP["K"][idx] = pose, img, K
            MAP["iter"][idx] = 1.0; valid[idx] = True

        def fuse(idx, pts, conf, pose):
            """Confidence + iteration weighted fusion of a re-observation into the map."""
            cg = MAP["conf"][idx] * MAP["iter"][idx]          # accumulated map weight
            cl = conf_sig(conf)
            w = (cg[..., None] + cl[..., None])
            MAP["pts"][idx] = (cg[..., None] * MAP["pts"][idx] + cl[..., None] * pts) / np.maximum(w, 1e-9)
            MAP["conf"][idx] = (cg + cl) / (MAP["iter"][idx] + 1.0)
            MAP["pose"][idx] = interpolate_poses(MAP["pose"][idx][None], pose[None],
                                                 MAP["conf"][idx][None], cl[None])[0]
            MAP["iter"][idx] += 1.0

        # ---- init window: defines the world frame ----
        init_idxs = list(range(0, min(init_w, n)))
        o = self._infer(init_idxs, image_paths, mode, depth_paths, intrinsics, depth_scale)
        H, W = o["world_points"].shape[1:3]
        alloc(H, W)
        for j, fi in enumerate(init_idxs):
            write_new(fi, o["world_points"][j], o["world_points_conf"][j],
                      o["poses"][j].astype(np.float64), o["images"][j], o["intrinsics"][j])
        self._discover_keyframes(init_idxs, MAP, pool, active, thr, lam, owner, seed=True)

        # ---- streaming ----
        cur = len(init_idxs)
        while cur < n:
            new_idxs = list(range(cur, min(cur + step, n)))
            anchors = list(active)
            ka = len(anchors)
            window = anchors + new_idxs
            o = self._infer(window, image_paths, mode, depth_paths, intrinsics, depth_scale)
            lp, lc, lpose = o["world_points"], o["world_points_conf"], o["poses"].astype(np.float64)
            limg, lK = o["images"], o["intrinsics"]

            g_pts = MAP["pts"][anchors]; g_conf = conf_sig(MAP["conf"][anchors])
            s, R, t = self._align(g_pts, g_conf, lp[:ka], conf_sig(lc[:ka]))
            S = np.eye(4); S[:3, :3], S[:3, 3] = s * R, t

            # active keyframes: refine via fusion; new frames: write
            for j, fi in enumerate(anchors):
                gp = apply_sim3_direct(lp[j][None], s, R, t)[0]
                gpose = S @ lpose[j]; gpose[:3, :3] /= s
                fuse(fi, gp, lc[j], gpose)
            new_pts = apply_sim3_direct(lp[ka:], s, R, t)
            for j, fi in enumerate(new_idxs):
                gpose = S @ lpose[ka + j]; gpose[:3, :3] /= s
                write_new(fi, new_pts[j], lc[ka + j], gpose, limg[ka + j], lK[ka + j])

            new_kfs = self._discover_keyframes(new_idxs, MAP, pool, active, thr, lam, owner)
            if len(active) > kf_c["max_active"]:
                keep = set(resample_keyframes(active, MAP["pose"][active], num_keep=kf_c["min_active"],
                                              lambda_t=lam, top_k=kf_c["top_k"], bridge=kf_c["bridge"]))
                active[:] = [i for i in active if i in keep]

            if loop_c.get("enable") and new_kfs:
                self._online_loop_closure(new_kfs, pool, loop_edges, MAP, valid, owner,
                                          image_paths, mode, depth_paths, intrinsics,
                                          depth_scale, stats)
            cur += step

        # ---- emit point cloud + poses from the fused map ----
        pcd_dir = os.path.join(output_dir, "pcd"); os.makedirs(pcd_dir, exist_ok=True)
        coef = m["Pointcloud_Save"]["conf_threshold_coef"]; ratio = m["Pointcloud_Save"]["sample_ratio"]
        for i in range(0, n, 20):
            sl = slice(i, min(i + 20, n))
            vmask = valid[sl]
            if not vmask.any():
                continue
            confs = MAP["conf"][sl][vmask].reshape(-1)
            save_confident_pointcloud_batch(
                points=MAP["pts"][sl][vmask], colors=MAP["img"][sl][vmask], confs=MAP["conf"][sl][vmask],
                output_path=os.path.join(pcd_dir, f"{i}_pcd.ply"),
                conf_threshold=float(np.mean(confs) * coef), sample_ratio=ratio)
        combined_ply = os.path.join(output_dir, "combined_pcd.ply"); merge_ply_files(pcd_dir, combined_ply)
        poses_txt = os.path.join(output_dir, "camera_poses.txt")
        with open(poses_txt, "w") as f:
            for i in range(n):
                p = MAP["pose"][i] if valid[i] else np.eye(4)
                f.write(" ".join(str(x) for x in p.flatten()) + "\n")
        print(f"[ma_long-slam] {n} frames | keyframes={len(pool)} active={len(active)} | "
              f"loops={stats['n_loops']} posegraph_opts={stats['n_optimize']}")
        print(f"[ma_long-slam] done. poses -> {poses_txt}  pcd -> {combined_ply}")
        return {"poses": [MAP["pose"][i] if valid[i] else None for i in range(n)],
                "n_keyframes": len(pool), "n_loops": stats["n_loops"],
                "poses_txt": poses_txt, "combined_ply": combined_ply}

    # ------------------------------------------------------------------ keyframes
    def _discover_keyframes(self, idxs, MAP, pool, active, thr, lam, owner, seed=False):
        """Add frames far (in pose) from all active keyframes; record each frame's owner kf."""
        new_kfs = []
        for fi in idxs:
            act_poses = [MAP["pose"][a] for a in active]
            if seed and not active:
                is_kf = True
            else:
                is_kf = all(extrinsic_distance(MAP["pose"][fi], ap, lam) > thr for ap in act_poses)
            # owner = nearest active keyframe in pose (for loop-correction propagation)
            if active:
                owner[fi] = min(active, key=lambda a: extrinsic_distance(MAP["pose"][fi], MAP["pose"][a], lam))
            if is_kf:
                pool.append(fi); active.append(fi); new_kfs.append(fi); owner[fi] = fi
        return new_kfs

    # ------------------------------------------------------------------ online LC
    def _online_loop_closure(self, new_kfs, pool, loop_edges, MAP, valid, owner,
                             image_paths, mode, depth_paths, intrinsics, depth_scale, stats):
        loop_c = self.cfg["Loop"]
        dist, gap, hw = loop_c["dist"], loop_c["min_gap"], loop_c["half_window"]
        found_new = False
        for kn in new_kfs:
            pc = MAP["pose"][kn][:3, 3]
            cands = [(np.linalg.norm(MAP["pose"][ko][:3, 3] - pc), ko) for ko in pool
                     if ko != kn and abs(ko - kn) > gap]
            cands = sorted([(d, ko) for d, ko in cands if d < dist])
            for _, ko in cands[:1]:                      # closest valid revisit
                con = self._loop_constraint(ko, kn, hw, MAP, valid,
                                            image_paths, mode, depth_paths, intrinsics, depth_scale)
                if con is not None:
                    loop_edges.append((ko, kn, con)); stats["n_loops"] += 1; found_new = True
                    print(f"[ma_long-slam] online loop {ko}<->{kn} accepted")
                break
        if found_new:
            self._optimize_posegraph(pool, loop_edges, MAP, valid, owner, stats)

    def _loop_constraint(self, ko, kn, hw, MAP, valid, image_paths, mode,
                         depth_paths, intrinsics, depth_scale):
        """Re-infer a joint window around both keyframes; verify co-location; return the
        relative c2w transform ko->kn (Sim3) measured in the joint reconstruction.

        This is the loop edge in the SAME convention as the sequential transforms
        (inv(c2w_i) ∘ c2w_j), so the Sim3 pose-graph optimizer stays consistent.
        """
        n = len(image_paths)
        ra = [i for i in range(max(0, ko - hw), min(n, ko + hw + 1)) if valid[i]]
        rb = [i for i in range(max(0, kn - hw), min(n, kn + hw + 1)) if valid[i]]
        if ko not in ra or kn not in rb:
            return None
        o = self._infer(ra + rb, image_paths, mode, depth_paths, intrinsics, depth_scale)
        lpose = o["poses"].astype(np.float64)
        la = len(ra)
        # geometric verification: the two sides must co-locate in the joint reconstruction
        ctr = lpose[:, :3, 3]
        ref = float(np.median(np.linalg.norm(ctr - ctr.mean(0), axis=1))) + 1e-6
        coloc = float(np.linalg.norm(ctr[:la].mean(0) - ctr[la:].mean(0)) / ref)
        if coloc > self.cfg["Loop"]["coloc_ratio"]:
            print(f"[ma_long-slam] loop {ko}<->{kn} rejected (coloc={coloc:.2f})")
            return None
        # Loop edge for node pair (ii=ko, jj=kn). The optimizer's sequential constraint is
        # dS = inv(P_jj) @ P_ii, so the loop measurement must match: inv(c2w_kn) @ c2w_ko.
        Mo = lpose[ra.index(ko)]; Mn = lpose[la + rb.index(kn)]
        Mrel = np.linalg.inv(Mn) @ Mo
        return (1.0, Mrel[:3, :3].copy(), Mrel[:3, 3].copy())

    def _optimize_posegraph(self, pool, loop_edges, MAP, valid, owner, stats):
        """Incremental Sim3 pose-graph optimization over keyframes; propagate to owned frames.

        Works in node-0's frame (the optimizer treats node 0 as the identity origin):
        sequential = K-1 relative transforms inv(abs_k) ∘ abs_{k+1}; loop edges in the same
        convention. After optimizing, propagate each keyframe's correction to its owned frames.
        """
        nodes = sorted(pool)
        if len(nodes) < 2:
            return
        idx_of = {g: k for k, g in enumerate(nodes)}
        absG = [_c2w_to_sim3(MAP["pose"][g]) for g in nodes]
        abs0 = absG[0]
        # sequential = K-1 relatives between consecutive keyframes (inv(abs_k) ∘ abs_{k+1})
        seq = [_sim3_mul(_sim3_inv(*absG[k]), absG[k + 1]) for k in range(len(nodes) - 1)]
        loops = [(idx_of[i], idx_of[j], con) for (i, j, con) in loop_edges
                 if i in idx_of and j in idx_of]
        if not loops:
            return
        opt = Sim3LoopOptimizer(self.cfg, device="cpu").optimize(seq, loops)
        # optimized absolute poses relative to node 0 (node 0 = identity origin)
        opt_rel = [(1.0, np.eye(3), np.zeros(3))] + accumulate_sim3_transforms(opt)
        delta = {}  # keyframe -> Sim3 world correction: (abs0 ∘ opt_rel_k) ∘ inv(abs_k)
        for k, g in enumerate(nodes):
            world_opt = _sim3_mul(abs0, opt_rel[k])
            delta[g] = _sim3_mul(world_opt, _sim3_inv(*absG[k]))
        # propagate each frame's pose + points by its owner keyframe's correction
        for fi in range(len(valid)):
            if not valid[fi]:
                continue
            kf = owner[fi] if owner[fi] in delta else (nodes[0] if nodes else -1)
            if kf not in delta:
                continue
            s, R, t = delta[kf]
            MAP["pts"][fi] = apply_sim3_direct(MAP["pts"][fi][None], s, R, t)[0]
            M = _sim3_to_c2w(s, R, t) @ MAP["pose"][fi]; M[:3, :3] /= s
            MAP["pose"][fi] = M
        stats["n_optimize"] += 1
