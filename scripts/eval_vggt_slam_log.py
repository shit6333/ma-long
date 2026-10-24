"""Evaluate a VGGT-SLAM pose log with ma_slam's own Sim3/SE3 ATE (src/eval.py).

Their `--log_results` file is TUM-style ``frame_id tx ty tz qx qy qz qw`` where
frame_id is parsed from the image filename; for our index-named TUM frames
(``frame_00012.png``) that is the frame index, which is also the row of
``gt_pose.txt`` (one row per frame, 0-based). So: pick the GT rows their keyframes
correspond to, write both as the files ``eval.ate_rmse`` expects (est = 4x4 per
row, gt = TUM rows) and evaluate with the identical alignment code ma_slam uses.

    PYTHONPATH=src python scripts/eval_vggt_slam_log.py --log <poses.txt> \
        --gt data/tum/fr1_xyz_dist/gt_pose.txt --out <dir>
"""
import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from eval import ate_rmse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    est = np.loadtxt(a.log).reshape(-1, 8)
    gt_lines = [ln for ln in open(a.gt) if ln.strip() and not ln.startswith("#")]
    idx = np.rint(est[:, 0]).astype(int)
    keep = (idx >= 0) & (idx < len(gt_lines))
    if not keep.all():
        print(f"[eval_vggt] dropping {(~keep).sum()} poses with frame ids outside GT range")
    est, idx = est[keep], idx[keep]
    os.makedirs(a.out, exist_ok=True)
    mats = []
    for row in est:
        T = np.eye(4)
        T[:3, :3] = R.from_quat(row[4:8]).as_matrix()
        T[:3, 3] = row[1:4]
        mats.append(T.reshape(-1))
    est_path = os.path.join(a.out, "camera_poses.txt")
    np.savetxt(est_path, np.array(mats))
    gt_path = os.path.join(a.out, "gt_kf.txt")
    with open(gt_path, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        f.write("".join(gt_lines[i] for i in idx))
    s = ate_rmse(est_path, gt_path, "sim3")
    se = ate_rmse(est_path, gt_path, "se3")
    print(f"[eval_vggt] n={s['n']} Sim3-ATE={s['ate_rmse']:.4f} | SE3-ATE={se['ate_rmse']:.4f} "
          f"| scale={s['scale']:.3f}")


if __name__ == "__main__":
    main()
