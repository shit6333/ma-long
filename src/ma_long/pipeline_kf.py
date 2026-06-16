"""Keyframe-anchored streaming pipeline for ma_long (AMB3R-style; Point 1 + Point 3).

Instead of fixed overlapping chunks, this processes the sequence online:
  - init: reconstruct the first `init_window` frames (defines the world frame),
    select keyframes among them (pose-distance + confidence -> *Point 1*).
  - step: each iteration feeds [active keyframes] + [next `step` new frames] through
    MapAnything together; the keyframes (whose global geometry is known) align the
    window's local reconstruction back to the global frame; new frames are written out,
    new keyframes are added, and the active set is resampled to a bounded, diverse set
    (*Point 3*: global pool -> bounded active subset).

Keyframes are the alignment anchors (replacing fixed overlap). A diverse active set
gives older anchors more reach than a fixed overlap, so revisits seen while an anchor is
still active are corrected implicitly. Outputs match the chunk pipeline (global
`camera_poses.txt` + merged point cloud) for direct ATE comparison.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from align.sim3utils import (
    apply_sim3_direct,
    merge_ply_files,
    save_confident_pointcloud_batch,
    weighted_align_point_maps,
)
from ma_long.tools.pose_dist import extrinsic_distance, pairwise_extrinsic_distance
from ma_long.tools.keyframes import resample_keyframes, select_keyframes

KF_DEFAULT_CONFIG: Dict = {
    "Model": {
        "init_window": 20,
        "step": 8,
        "align_lib": "torch",
        "align_method": "se3",
        "IRLS": {"delta": 0.1, "max_iters": 5, "tol": "1e-9"},
        "Pointcloud_Save": {"sample_ratio": 0.05, "conf_threshold_coef": 0.75},
    },
    "Keyframe": {
        "threshold": 0.3,        # pose-distance for a new keyframe (metric: rot/180 + metres)
        "max_active": 10,        # resample trigger
        "min_active": 7,         # target after resample
        "top_k": 4,
        "bridge": 1,
        "lambda_t": 1.0,
        # online loop closure via keyframe reactivation: pull global-pool keyframes that are
        # spatially near the current position but temporally far back into the window as anchors.
        "reactivate": True,
        "reactivate_dist": 0.5,      # metres: how close (camera centre) to count as a revisit
        "reactivate_min_gap": 30,    # frames: must be this far back in time to be a loop
        "reactivate_max": 3,         # cap reactivated anchors per window
    },
}


class MaLongKeyframePipeline:
    def __init__(self, config: Optional[Dict] = None, model=None, device: str = "cuda"):
        self.cfg = config or KF_DEFAULT_CONFIG
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

    def run(self, image_paths: Sequence[str], output_dir: str, *, mode: str = "rgb",
            depth_paths: Optional[Sequence[str]] = None,
            intrinsics: Optional[np.ndarray] = None, depth_scale: float = 1000.0,
            keep_cache: bool = True) -> Dict:
        m, kf_cfg = self.cfg["Model"], self.cfg["Keyframe"]
        init_w, step = m["init_window"], m["step"]
        thr, lam = kf_cfg["threshold"], kf_cfg["lambda_t"]
        coef = m["Pointcloud_Save"]["conf_threshold_coef"]
        ratio = m["Pointcloud_Save"]["sample_ratio"]
        n = len(image_paths)
        pcd_dir = os.path.join(output_dir, "pcd")
        os.makedirs(pcd_dir, exist_ok=True)

        all_poses: List[Optional[np.ndarray]] = [None] * n
        all_K: List[Optional[np.ndarray]] = [None] * n
        pool: List[Dict] = []     # ALL keyframes ever (idx, pts, conf, pose) — global pool
        active_ids: List[int] = []  # bounded active subset (frame indices)
        n_resample = 0
        n_reactivated = 0
        ply_i = 0
        react = kf_cfg.get("reactivate", False)

        def active():
            ids = set(active_ids)
            return [k for k in pool if k["idx"] in ids]

        def save_cloud(pts, colors, conf):
            nonlocal ply_i
            confs = conf.reshape(-1)
            save_confident_pointcloud_batch(
                points=pts, colors=colors, confs=conf,
                output_path=os.path.join(pcd_dir, f"{ply_i}_pcd.ply"),
                conf_threshold=float(np.mean(confs) * coef), sample_ratio=ratio)
            ply_i += 1

        def add_keyframes(idxs, pts, conf, poses_g):
            """Add frames whose global pose is > thr from ALL active keyframes (-> pool+active)."""
            act = active()
            for j, fi in enumerate(idxs):
                if all(extrinsic_distance(poses_g[j], a["pose"], lam) > thr for a in act):
                    k = dict(idx=fi, pts=pts[j], conf=conf[j], pose=poses_g[j].astype(np.float64))
                    pool.append(k); active_ids.append(fi); act.append(k)

        def reactivate(query_pose):
            """Global-pool keyframes spatially near `query_pose` but temporally far + inactive."""
            if not react or query_pose is None:
                return []
            d = kf_cfg["reactivate_dist"]; gap = kf_cfg["reactivate_min_gap"]
            cur_idx = int(query_pose[1]); qc = query_pose[0][:3, 3]
            cand = [(float(np.linalg.norm(k["pose"][:3, 3] - qc)), k) for k in pool
                    if k["idx"] not in active_ids and abs(k["idx"] - cur_idx) > gap]
            cand = [(dist, k) for dist, k in cand if dist < d]
            cand.sort(key=lambda x: x[0])
            return [k for _, k in cand[: kf_cfg["reactivate_max"]]]

        # ---- init: first window defines the world frame ----
        init_idxs = list(range(0, min(init_w, n)))
        o = self._infer(init_idxs, image_paths, mode, depth_paths, intrinsics, depth_scale)
        wp, conf, poses, imgs, K = (o["world_points"], o["world_points_conf"],
                                    o["poses"].astype(np.float64), o["images"], o["intrinsics"])
        for j, fi in enumerate(init_idxs):
            all_poses[fi], all_K[fi] = poses[j], K[j]
        save_cloud(wp, imgs, conf)
        add_keyframes(init_idxs, wp, conf, poses)

        # ---- streaming steps ----
        cur = len(init_idxs)
        last_pose = (all_poses[init_idxs[-1]], init_idxs[-1])
        while cur < n:
            new_idxs = list(range(cur, min(cur + step, n)))
            # anchors = bounded active set + reactivated loop keyframes (online LC)
            anchors = active()
            extra = reactivate(last_pose)
            if extra:
                n_reactivated += len(extra)
                anchors = anchors + extra
            ka = len(anchors)
            window = [a["idx"] for a in anchors] + new_idxs
            o = self._infer(window, image_paths, mode, depth_paths, intrinsics, depth_scale)
            lp, lc, lpose, limg, lK = (o["world_points"], o["world_points_conf"],
                                       o["poses"].astype(np.float64), o["images"], o["intrinsics"])

            # align window-local anchors -> their known global geometry
            g_pts = np.stack([a["pts"] for a in anchors])
            g_conf = np.stack([a["conf"] for a in anchors])
            ct = float(min(np.median(g_conf), np.median(lc[:ka])) * 0.1)
            s, R, t = weighted_align_point_maps(g_pts, g_conf, lp[:ka], lc[:ka],
                                                conf_threshold=ct, config=self.cfg)
            S = np.eye(4); S[:3, :3], S[:3, 3] = s * R, t

            new_pts = apply_sim3_direct(lp[ka:], s, R, t)
            new_poses_g = np.empty((len(new_idxs), 4, 4))
            for j, fi in enumerate(new_idxs):
                c2w = S @ lpose[ka + j]; c2w[:3, :3] /= s
                all_poses[fi], all_K[fi], new_poses_g[j] = c2w, lK[ka + j], c2w
            save_cloud(new_pts, limg[ka:], lc[ka:])
            last_pose = (new_poses_g[-1], new_idxs[-1])

            add_keyframes(new_idxs, new_pts, lc[ka:], new_poses_g)
            if len(active_ids) > kf_cfg["max_active"]:
                act = active()
                keep = set(resample_keyframes(
                    [a["idx"] for a in act], np.stack([a["pose"] for a in act]),
                    num_keep=kf_cfg["min_active"], lambda_t=lam,
                    top_k=kf_cfg["top_k"], bridge=kf_cfg["bridge"]))
                active_ids[:] = [i for i in active_ids if i in keep]
                n_resample += 1
            cur += step

        combined_ply = os.path.join(output_dir, "combined_pcd.ply")
        merge_ply_files(pcd_dir, combined_ply)
        poses_txt = os.path.join(output_dir, "camera_poses.txt")
        with open(poses_txt, "w") as f:
            for p in all_poses:
                p = np.eye(4) if p is None else p
                f.write(" ".join(str(x) for x in p.flatten()) + "\n")
        print(f"[ma_long-kf] {n} frames | global keyframe pool={len(pool)} | "
              f"final active={len(active_ids)} | resamples={n_resample} | "
              f"loop reactivations={n_reactivated}")
        print(f"[ma_long-kf] done. poses -> {poses_txt}  pcd -> {combined_ply}")
        return {"poses": all_poses, "intrinsics": all_K, "n_pool": len(pool),
                "n_active": len(active_ids), "n_reactivated": n_reactivated,
                "poses_txt": poses_txt, "combined_ply": combined_ply}
