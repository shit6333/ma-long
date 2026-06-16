"""Standalone runner for the online keyframe SLAM (ma_long.slam).

    python src/ma_long/run_slam.py --scene data/scene0011_00 --mode rgb+depth+intr \
        --out outputs/slam_s0011 --gt data/scene0011_00/gt_pose.txt
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
from ma_long.slam import MaLongSLAM, SLAM_DEFAULT_CONFIG


def _frames(d):
    fs: List[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG"):
        fs += glob.glob(os.path.join(d, ext))
    return sorted(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene"); ap.add_argument("--image_dir"); ap.add_argument("--depth_dir")
    ap.add_argument("--intrinsic"); ap.add_argument("--mode", default="rgb", choices=list(MODES))
    ap.add_argument("--out", required=True); ap.add_argument("--gt")
    ap.add_argument("--init_window", type=int, default=SLAM_DEFAULT_CONFIG["Model"]["init_window"])
    ap.add_argument("--step", type=int, default=SLAM_DEFAULT_CONFIG["Model"]["step"])
    ap.add_argument("--kf_threshold", type=float, default=SLAM_DEFAULT_CONFIG["Keyframe"]["threshold"])
    ap.add_argument("--no_loop", action="store_true")
    ap.add_argument("--loop_dist", type=float, default=SLAM_DEFAULT_CONFIG["Loop"]["dist"])
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--depth_scale", type=float, default=1000.0)
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
        depth_paths = _frames(depth_dir)[: len(image_paths)]
    if need_intr:
        intrinsics = load_intrinsics(intr_path)

    cfg = copy.deepcopy(SLAM_DEFAULT_CONFIG)
    cfg["Model"].update(init_window=a.init_window, step=a.step)
    cfg["Keyframe"]["threshold"] = a.kf_threshold
    cfg["Loop"].update(enable=not a.no_loop, dist=a.loop_dist)

    os.makedirs(a.out, exist_ok=True)
    res = MaLongSLAM(config=cfg, device=a.device).run(
        image_paths, a.out, mode=a.mode, depth_paths=depth_paths,
        intrinsics=intrinsics, depth_scale=a.depth_scale)

    if a.gt:
        from eval import ate_rmse
        s = ate_rmse(res["poses_txt"], a.gt, "sim3"); se = ate_rmse(res["poses_txt"], a.gt, "se3")
        print(f"[ma_long-slam] Sim3-ATE={s['ate_rmse']:.4f} | SE3-ATE={se['ate_rmse']:.4f} | "
              f"scale={s['scale']:.3f} | keyframes={res['n_keyframes']} | loops={res['n_loops']}")


if __name__ == "__main__":
    main()
