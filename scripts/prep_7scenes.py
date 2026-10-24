"""Convert 7-Scenes seq-01 (raw Microsoft layout) to the repo's ScanNet-like scene format.

  data/7scenes/_raw/<scene>/seq-01/frame-XXXXXX.{color.png,depth.png,pose.txt}
      -> data/7scenes/<scene>/{rgb,depth}/frame_XXXXX.png, gt_pose.txt (TUM rows, timestamp =
         frame index), intrinsic.txt (4x4, Kinect 585/585/320/240 as in the 7-Scenes release).

Depth: uint16 millimetres, 65535 = invalid -> written as 0 (our loaders treat <=0 as invalid;
depth_scale 1000). Poses: *.pose.txt is camera-to-world (4x4), which is what gt_pose.txt stores.
Files are hard-linked (fall back to copy) so the raw folder can still serve VGGT-SLAM's own
main.py (`--image_folder data/7scenes/_raw/<scene>/seq-01`, which globs *.color.png and skips
*depth* / *txt*).

    python scripts/prep_7scenes.py [--scenes chess fire ...]
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "7scenes", "_raw")
OUT = os.path.join(ROOT, "data", "7scenes")
FX = FY = 585.0
CX, CY = 320.0, 240.0


def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def prep(scene):
    src = os.path.join(RAW, scene, "seq-01")
    colors = sorted(glob.glob(os.path.join(src, "frame-*.color.png")))
    if not colors:
        print(f"[{scene}] no frames in {src}"); return
    out = os.path.join(OUT, scene)
    os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(out, "depth"), exist_ok=True)
    rows = []
    n_bad = 0
    for i, c in enumerate(colors):
        stem = c[:-len(".color.png")]
        d = stem + ".depth.png"
        p = stem + ".pose.txt"
        link(c, os.path.join(out, "rgb", f"frame_{i:05d}.png"))
        dep = cv2.imread(d, cv2.IMREAD_UNCHANGED)
        dep = dep.astype(np.uint16)
        dep[dep == 65535] = 0
        cv2.imwrite(os.path.join(out, "depth", f"frame_{i:05d}.png"), dep)
        T = np.loadtxt(p)
        if not np.all(np.isfinite(T)) or abs(np.linalg.det(T[:3, :3]) - 1) > 1e-2:
            n_bad += 1
        q = R.from_matrix(T[:3, :3]).as_quat()   # x y z w
        rows.append(f"{i} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    with open(os.path.join(out, "gt_pose.txt"), "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n" + "\n".join(rows) + "\n")
    K = np.array([[FX, 0, CX, 0], [0, FY, CY, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    np.savetxt(os.path.join(out, "intrinsic.txt"), K, fmt="%.6f")
    print(f"[{scene}] {len(colors)} frames -> {out}  (bad poses: {n_bad})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"])
    a = ap.parse_args()
    for s in a.scenes:
        prep(s)
