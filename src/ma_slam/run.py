"""CLI runner for ma_slam (VGGT-SLAM-style submap SLAM on MapAnything).

    python src/ma_slam/run.py --scene data/scene0011_00 --mode rgb+depth+intr \
        --out outputs/maslam_s0011 --gt data/scene0011_00/gt_pose.txt
"""

from __future__ import annotations

import argparse
import copy
import glob
import os
import sys
from typing import List, Optional

# Make the src/ sibling packages importable when run directly as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from model.inputs import MODES, load_intrinsics
from ma_slam.solver import MaSlam, DEFAULT_CONFIG


def _frames(d: str) -> List[str]:
    fs: List[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG"):
        fs += glob.glob(os.path.join(d, ext))
    return sorted(fs)


def main():
    ap = argparse.ArgumentParser(description="ma_slam (VGGT-SLAM-style, MapAnything backbone)")
    ap.add_argument("--scene"); ap.add_argument("--image_dir"); ap.add_argument("--depth_dir")
    ap.add_argument("--intrinsic"); ap.add_argument("--mode", default="rgb", choices=list(MODES))
    ap.add_argument("--backend", default="ma", choices=["ma", "da3"],
                    help="reconstruction backbone: 'ma' MapAnything (4 modes) | "
                         "'da3' DA3NESTED-GIANT-LARGE-1.1 (metric rgb / rgb+intr, no depth input)")
    ap.add_argument("--manifold", default="se3", choices=["se3", "sim3", "sl4"],
                    help="pose-graph backend + inter-submap transform group: "
                         "se3 (metric, default) | sim3 (+scale, pypose) | sl4 (projective, needs gtsam fork)")
    ap.add_argument("--out", required=True); ap.add_argument("--gt")
    ap.add_argument("--submap_size", type=int, default=32)
    ap.add_argument("--no_loop", action="store_true")
    ap.add_argument("--sim_threshold", type=float, default=DEFAULT_CONFIG["Loop"]["sim_threshold"])
    ap.add_argument("--coloc_ratio", type=float, default=DEFAULT_CONFIG["Loop"]["coloc_ratio"])
    ap.add_argument("--loop_half_window", type=int, default=DEFAULT_CONFIG["Loop"]["half_window"],
                    help="verification window radius: 0 = the 2 candidate frames only; "
                         ">0 also re-infers each frame's in-submap neighbours")
    ap.add_argument("--debug_coloc", action="store_true",
                    help="dump each loop candidate's two re-inferred windows as a "
                         "green(query)/red(match) PLY in <out>/coloc/ for visual inspection")
    ap.add_argument("--keyframe_disparity", type=float, default=25.0,
                    help="LK disparity keyframe gate (pixels): drop frames whose mean optical-flow "
                         "displacement vs the last keyframe is below this. Default 25 (on); "
                         "0 = off (consecutive frames). submap_size then counts keyframes.")
    ap.add_argument("--voxel_size", type=float, default=DEFAULT_CONFIG["Pointcloud"]["voxel_size"])
    ap.add_argument("--max_points", type=int, default=DEFAULT_CONFIG["Pointcloud"]["max_points"],
                    help="cap on merged point-cloud size (uniform random sample); 0 = no cap")
    ap.add_argument("--conf_coef", type=float, default=DEFAULT_CONFIG["Pointcloud"]["conf_coef"],
                    help="multiplier on the per-submap confidence threshold (>1 = stricter)")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--depth_scale", type=float, default=1000.0)
    ap.add_argument("--depth_max", type=float, default=0.0,
                    help="zero out depth beyond this many metres (drops far sensor noise; 0 = off)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    image_dir = a.image_dir or (os.path.join(a.scene, "rgb") if a.scene else None)
    if not image_dir:
        ap.error("provide --scene or --image_dir")
    depth_dir = a.depth_dir or (os.path.join(a.scene, "depth") if a.scene else None)
    intr_path = a.intrinsic or (os.path.join(a.scene, "intrinsic.txt") if a.scene else None)

    image_paths = _frames(image_dir)
    if a.max_frames:
        image_paths = image_paths[: a.max_frames]
    need_depth = "depth" in a.mode
    need_intr = ("intr" in a.mode) or need_depth
    depth_paths: Optional[List[str]] = None
    intrinsics: Optional[np.ndarray] = None
    if need_depth:
        depth_paths = _frames(depth_dir)
        # color/depth are index-aligned; if counts differ (e.g. a trailing color frame with no
        # depth), truncate BOTH to the common length so depth_paths[i] never goes out of range.
        n = min(len(image_paths), len(depth_paths))
        if len(image_paths) != len(depth_paths):
            print(f"[ma_slam] color/depth count mismatch ({len(image_paths)}/{len(depth_paths)}); "
                  f"truncating both to {n}")
        image_paths, depth_paths = image_paths[:n], depth_paths[:n]
    if need_intr:
        intrinsics = load_intrinsics(intr_path)

    # VGGT-SLAM-style keyframe gate (opt-in): drop low-parallax frames BEFORE the pipeline,
    # so submap_size counts keyframes (not consecutive frames). Causal → stream-compatible.
    if a.keyframe_disparity > 0:
        from ma_slam.keyframe_flow import select_keyframes_by_disparity
        kf = select_keyframes_by_disparity(image_paths, a.keyframe_disparity)
        print(f"[ma_slam] keyframe gate: kept {len(kf)}/{len(image_paths)} frames "
              f"(disparity > {a.keyframe_disparity:g}px)")
        image_paths = [image_paths[i] for i in kf]
        if depth_paths is not None:
            depth_paths = [depth_paths[i] for i in kf]
        # subsample GT to the same keyframes so eval's per-index correspondence still holds
        # (GT row i == original frame i; this is also how VGGT-SLAM scores ATE — on keyframes).
        if a.gt:
            gt_lines = [ln for ln in open(a.gt) if ln.strip() and not ln.startswith("#")]
            os.makedirs(a.out, exist_ok=True)
            kf_gt = os.path.join(a.out, "gt_kf.txt")
            with open(kf_gt, "w") as f:
                f.write("# timestamp tx ty tz qx qy qz qw\n")
                f.write("".join(gt_lines[i] for i in kf))
            a.gt = kf_gt

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["submap_size"] = a.submap_size
    cfg["Graph"]["manifold"] = a.manifold
    cfg["Loop"].update(enable=not a.no_loop, sim_threshold=a.sim_threshold,
                       coloc_ratio=a.coloc_ratio, half_window=a.loop_half_window,
                       debug_coloc=a.debug_coloc)
    cfg["Pointcloud"].update(voxel_size=a.voxel_size, max_points=a.max_points, conf_coef=a.conf_coef)

    model = None
    if a.backend == "da3":
        from model.da3_infer import Da3ChunkModel
        if "depth" in a.mode:
            ap.error("backend da3 supports rgb / rgb+intr only (DA3 does not ingest depth)")
        model = Da3ChunkModel(device=a.device)

    res = MaSlam(config=cfg, model=model, device=a.device).run(
        image_paths, a.out, mode=a.mode, depth_paths=depth_paths,
        intrinsics=intrinsics, depth_scale=a.depth_scale,
        depth_max=(a.depth_max if a.depth_max > 0 else None))

    if a.gt:
        from eval import ate_rmse
        s = ate_rmse(res["poses_txt"], a.gt, "sim3"); se = ate_rmse(res["poses_txt"], a.gt, "se3")
        print(f"[ma_slam] Sim3-ATE={s['ate_rmse']:.4f} | SE3-ATE={se['ate_rmse']:.4f} | "
              f"scale={s['scale']:.3f} | submaps={res['n_submaps']} | loops={res['n_loops']}")


if __name__ == "__main__":
    main()
