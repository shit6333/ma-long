"""Standalone ma_long runner.

Examples:
    # image-only on a ScanNet-style scene
    python src/ma_long/run.py --scene data/scene0011_00 --mode rgb --out outputs/s0011_rgb

    # multi-modal (uses depth/ + intrinsic.txt in the scene dir)
    python src/ma_long/run.py --scene data/scene0011_00 --mode rgb+depth+intr \
        --out outputs/s0011_rgbdi

A "scene" dir is expected to contain rgb/ (frames), and for multi-modal modes
depth/ (aligned 1:1 with rgb) and intrinsic.txt. Use --image_dir/--depth_dir/
--intrinsic to point at non-standard layouts.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional

# Make the src/ sibling packages importable when run directly as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from model.inputs import MODES, load_intrinsics
from ma_long.pipeline import DEFAULT_CONFIG, MaLongPipeline


def _list_frames(d: str) -> List[str]:
    fs: List[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG"):
        fs += glob.glob(os.path.join(d, ext))
    return sorted(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", help="ScanNet-style scene dir (rgb/, depth/, intrinsic.txt)")
    ap.add_argument("--image_dir")
    ap.add_argument("--depth_dir")
    ap.add_argument("--intrinsic", help="path to a 3x3/4x4 intrinsics matrix file")
    ap.add_argument("--mode", default="rgb", choices=list(MODES))
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--chunk_size", type=int, default=DEFAULT_CONFIG["Model"]["chunk_size"])
    ap.add_argument("--overlap", type=int, default=DEFAULT_CONFIG["Model"]["overlap"])
    ap.add_argument("--align_method", default=DEFAULT_CONFIG["Model"]["align_method"])
    ap.add_argument("--align_lib", default=DEFAULT_CONFIG["Model"]["align_lib"])
    ap.add_argument("--no_loop", action="store_true", help="disable loop closure")
    ap.add_argument("--loop_sim_thresh", type=float, default=None,
                    help="override SALAD loop similarity threshold")
    ap.add_argument("--depth_scale", type=float, default=1000.0)
    ap.add_argument("--max_frames", type=int, default=0, help="cap frames (0 = all)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gt", help="optional GT pose file for ATE eval after the run")
    args = ap.parse_args()

    image_dir = args.image_dir or (os.path.join(args.scene, "rgb") if args.scene else None)
    if not image_dir:
        ap.error("provide --scene or --image_dir")
    depth_dir = args.depth_dir or (os.path.join(args.scene, "depth") if args.scene else None)
    intr_path = args.intrinsic or (os.path.join(args.scene, "intrinsic.txt") if args.scene else None)

    image_paths = _list_frames(image_dir)
    if args.max_frames:
        image_paths = image_paths[: args.max_frames]
    if not image_paths:
        ap.error(f"no frames found in {image_dir}")

    need_depth = "depth" in args.mode
    need_intr = ("intr" in args.mode) or need_depth
    depth_paths: Optional[List[str]] = None
    intrinsics: Optional[np.ndarray] = None
    if need_depth:
        dfs = _list_frames(depth_dir)
        if len(dfs) < len(image_paths):
            ap.error(f"need >= {len(image_paths)} depth frames, found {len(dfs)} in {depth_dir}")
        depth_paths = dfs[: len(image_paths)]
    if need_intr:
        if not intr_path or not os.path.exists(intr_path):
            ap.error(f"mode {args.mode} needs intrinsics; not found at {intr_path}")
        intrinsics = load_intrinsics(intr_path)

    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)  # carry Weights + Loop sections, not just Model
    cfg["Model"].update(chunk_size=args.chunk_size, overlap=args.overlap,
                        align_method=args.align_method, align_lib=args.align_lib,
                        loop_enable=not args.no_loop)
    if args.loop_sim_thresh is not None:
        cfg["Loop"]["SALAD"]["similarity_threshold"] = args.loop_sim_thresh

    os.makedirs(args.out, exist_ok=True)
    pipe = MaLongPipeline(config=cfg, device=args.device)
    res = pipe.run(image_paths, args.out, mode=args.mode,
                   depth_paths=depth_paths, intrinsics=intrinsics, depth_scale=args.depth_scale)

    if args.gt:
        from eval import ate_rmse
        sim3 = ate_rmse(res["poses_txt"], args.gt, align="sim3")
        se3 = ate_rmse(res["poses_txt"], args.gt, align="se3")
        print(f"[ma_long] Sim3-ATE = {sim3['ate_rmse']:.4f} m | "
              f"SE3-ATE(metric) = {se3['ate_rmse']:.4f} m | "
              f"est scale vs gt = {sim3['scale']:.4f} | n={sim3['n']}")


if __name__ == "__main__":
    main()
