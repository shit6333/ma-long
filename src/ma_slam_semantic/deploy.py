"""Robot-deployment export — turn a finished ma_slam_semantic output dir into the
robot-ready artifact set, via the vendored OpenSe3r postprocess scripts:

    robot_deploy_<name>_camera_trajectory.json        per-frame c2w (ROS bridge schema)
    robot_deploy_<name>_instance_semantic_table.json  per-instance class (single level)
    robot_deploy_<name>_instance_semantic_table_{coarse,medium,fine,all_levels}.json
    robot_deploy_<name>_{inst,rgb,sem}_color.ply      per-point clouds (inst_id vertex
                                                      property -> table; --max_points cap)
    robot_deploy_<name>_sem_color_<level>.ply         per-level semantic colours + legends

Standalone — runs on an EXISTING output dir (no SLAM state needed):

    PYTHONPATH=src python -m ma_slam_semantic.deploy \
        --dir outputs/ma_sem/<scene> --name <scene> [--out <dir>/deploy]

Coordinate-frame rule: every artifact must live in ONE frame. A robot run (no
--gt) leaves everything in the ma_slam world frame — consistent by construction.
A benchmark run (--gt) exports the instance npz/table in the GT-mesh frame while
camera_poses.txt / combined_pcd.ply stay in the SLAM frame; this tool detects the
mismatch via <dir>/alignment.json and re-applies the same T_align to poses + pcd.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from ma_slam_semantic.config import OPENSE3R_ROOT, ensure_opense3r


def _load_poses_txt(path: str) -> np.ndarray:
    """camera_poses.txt (one flattened 4x4 per row, frame-id order) -> [T,4,4]."""
    return np.loadtxt(path).reshape(-1, 4, 4)


def _apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def prepare_inputs(run_dir: str, name: str, fps: float = 30.0) -> np.ndarray:
    """Write <name>_results.npz + <name>_geo.ply next to the existing artifacts,
    re-applying the run's T_align (alignment.json) when one was recorded so ALL
    deploy artifacts share the instance-map frame. Returns T_align or identity."""
    from plyfile import PlyData, PlyElement

    T_align = np.eye(4)
    align_json = os.path.join(run_dir, "alignment.json")
    if os.path.isfile(align_json):
        T_align = np.asarray(json.load(open(align_json))["T_align"], dtype=np.float64)
        print(f"[deploy] alignment.json found — re-applying T_align "
              f"(|t|={np.linalg.norm(T_align[:3, 3]):.3f} m) to poses + pcd")

    poses = _load_poses_txt(os.path.join(run_dir, "camera_poses.txt"))
    poses = np.einsum("ij,tjk->tik", T_align, poses)
    np.savez(os.path.join(run_dir, f"{name}_results.npz"),
             poses=poses.astype(np.float32), fps=np.float32(fps))
    print(f"[deploy] wrote {name}_results.npz  ({len(poses)} poses)")

    geo_out = os.path.join(run_dir, f"{name}_geo.ply")
    src_ply = os.path.join(run_dir, "combined_pcd.ply")
    if not os.path.isfile(src_ply):
        raise FileNotFoundError(f"{src_ply} missing (need the ma_slam point cloud)")
    v = PlyData.read(src_ply)["vertex"]
    xyz = _apply_T(T_align, np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64))
    vert = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vert["x"], vert["y"], vert["z"] = xyz[:, 0].astype("<f4"), xyz[:, 1].astype("<f4"), xyz[:, 2].astype("<f4")
    vert["red"], vert["green"], vert["blue"] = v["red"], v["green"], v["blue"]
    PlyData([PlyElement.describe(vert, "vertex")], text=False).write(geo_out)
    print(f"[deploy] wrote {name}_geo.ply  ({len(vert):,} pts)")
    return T_align


def run_deploy(run_dir: str, name: str, out_dir: str = None, *, fps: float = 30.0,
               max_points: int = 2_000_000, levels: str = "coarse,medium,fine",
               backbone: str = "tips-v2-l14", python: str = sys.executable) -> str:
    run_dir = os.path.abspath(run_dir)
    out_dir = os.path.abspath(out_dir or os.path.join(run_dir, "deploy"))
    os.makedirs(out_dir, exist_ok=True)
    prepare_inputs(run_dir, name, fps=fps)

    root = ensure_opense3r()
    common = ["--slam_dir", run_dir, "--scene_name", name,
              "--output_dir", out_dir, "--backbone", backbone]
    print("[deploy] postprocess_slam_for_robot.py ...")
    subprocess.run([python, os.path.join(root, "postprocess_slam_for_robot.py"),
                    *common, "--max_points", str(max_points)], cwd=root, check=True)
    print("[deploy] postprocess_slam_multi_level_tables.py ...")
    subprocess.run([python, os.path.join(root, "postprocess_slam_multi_level_tables.py"),
                    *common, "--levels", levels], cwd=root, check=True)
    print(f"[deploy] robot artifacts -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Robot-deployment export (post-SLAM)")
    ap.add_argument("--dir", required=True, help="finished ma_slam_semantic output dir")
    ap.add_argument("--name", default=None, help="artifact prefix (default: basename of --dir)")
    ap.add_argument("--out", default=None, help="deploy output dir (default: <dir>/deploy)")
    ap.add_argument("--fps", type=float, default=30.0, help="trajectory timestamp rate")
    ap.add_argument("--max_points", type=int, default=2_000_000,
                    help="per-point cloud sample cap")
    ap.add_argument("--levels", default="coarse,medium,fine",
                    help="multi-level label sets (built-in)")
    ap.add_argument("--backbone", default="tips-v2-l14")
    a = ap.parse_args()
    name = a.name or os.path.basename(os.path.normpath(a.dir))
    run_deploy(a.dir, name, a.out, fps=a.fps, max_points=a.max_points,
               levels=a.levels, backbone=a.backbone)


if __name__ == "__main__":
    main()
