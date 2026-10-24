"""Convert the ScanNet++ eval set (/mnt/HDD4/ricky/data/scannetpp_eval, iPhone stream at
frame stride 20) to the repo's scene format:

  data/scannetpp/<id>/rgb/frame_XXXXX.jpg     (hard link -> iphone/rgb/frame_000NNN.jpg, 1920x1440)
  data/scannetpp/<id>/depth/frame_XXXXX.png   (hard link -> iphone/depth, 256x192 uint16 mm; GT-labelling only)
  data/scannetpp/<id>/gt_pose.txt             TUM rows, timestamp = frame index; from iphone/poses.npy
                                              (verified == inv(colmap images.txt) = COLMAP c2w in the
                                              laser-scan frame -> metric GT)
  data/scannetpp/<id>/intrinsic.txt           4x4, K of the 1920x1440 rgb (iphone/intrinsic.txt)
  data/scannetpp/<id>/depth_intrinsic.txt     4x4, same K scaled to the 256x192 depth grid

    python scripts/prep_scannetpp.py [--list my_arrow_val_50.txt] [--ids ...]
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

SRC = "/mnt/HDD4/ricky/data/scannetpp_eval"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "scannetpp")


def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(os.path.abspath(src), dst)


def prep(sid):
    ip = os.path.join(SRC, "data", sid, "iphone")
    rgbs = sorted(glob.glob(os.path.join(ip, "rgb", "frame_*.jpg")))
    if not rgbs:
        print(f"[{sid}] no rgb"); return
    P = np.load(os.path.join(ip, "poses.npy"))
    assert len(P) == len(rgbs), (sid, len(P), len(rgbs))
    out = os.path.join(OUT, sid)
    os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(out, "depth"), exist_ok=True)
    rows = []
    for i, c in enumerate(rgbs):
        stem = os.path.basename(c)[:-4]
        link(c, os.path.join(out, "rgb", f"frame_{i:05d}.jpg"))
        d = os.path.join(ip, "depth", stem + ".png")
        if os.path.exists(d):
            link(d, os.path.join(out, "depth", f"frame_{i:05d}.png"))
        T = P[i].astype(np.float64)
        q = R.from_matrix(T[:3, :3]).as_quat()
        rows.append(f"{i} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    with open(os.path.join(out, "gt_pose.txt"), "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n" + "\n".join(rows) + "\n")
    K = np.loadtxt(os.path.join(ip, "intrinsic.txt"))[:3, :3]
    K4 = np.eye(4); K4[:3, :3] = K
    np.savetxt(os.path.join(out, "intrinsic.txt"), K4, fmt="%.6f")
    im = cv2.imread(rgbs[0]); dp = cv2.imread(os.path.join(ip, "depth", os.path.basename(rgbs[0])[:-4] + ".png"), -1)
    if dp is not None:
        sx, sy = dp.shape[1] / im.shape[1], dp.shape[0] / im.shape[0]
        Kd = K4.copy(); Kd[0, :] *= sx; Kd[1, :] *= sy
        np.savetxt(os.path.join(out, "depth_intrinsic.txt"), Kd, fmt="%.6f")
    print(f"[{sid}] {len(rgbs)} frames, rgb {im.shape[1]}x{im.shape[0]}, depth {None if dp is None else dp.shape[::-1]} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=os.path.join(SRC, "my_arrow_val_50.txt"))
    ap.add_argument("--ids", nargs="*", default=None)
    a = ap.parse_args()
    ids = a.ids or [l.strip() for l in open(a.list) if l.strip()]
    for sid in ids:
        prep(sid)
