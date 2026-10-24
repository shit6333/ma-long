"""Convert a Clio scene (MIT-SPARK, RealSense D455 + COLMAP) to the repo's scene format.

  data/clio/_raw/<scene>_unzip/{images/rgb_N.jpg, depth/depth_N.png, sparse/0 (COLMAP), <scene>.bag}
      -> data/clio/<scene>/rgb/frame_XXXXX.jpg, depth/frame_XXXXX.png, gt_pose.txt, intrinsic.txt,
         name_map.txt (frame index -> original file), scale.txt

GT poses = COLMAP sparse model (registered in the laser/COLMAP frame, NOT metric) rescaled to
metres with the median ratio depth_png[m] / colmap_z over the registered 3D points (apartment:
0.655, IQR 0.64-0.67). Intrinsics: COLMAP estimates a per-image SIMPLE_RADIAL focal; we write
the median f with the fixed principal point (distortion ignored). Frame index = position in the
numerically sorted rgb list (only frames registered by COLMAP are kept).

    python scripts/prep_clio.py --scenes apartment office cubicle building
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "clio", "_raw")
OUT = os.path.join(ROOT, "data", "clio")


def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(os.path.abspath(src), dst)


def cam_from_world(img):
    T = img.cam_from_world
    T = T() if callable(T) else T
    M = np.eye(4)
    if hasattr(T, "matrix"):
        M[:3, :4] = np.asarray(T.matrix())
    else:
        M[:3, :3] = T.rotation.matrix(); M[:3, 3] = T.translation
    return M


def prep(scene, n_scale_imgs=60):
    import pycolmap
    src = os.path.join(RAW, f"{scene}_unzip")
    # several COLMAP models may exist (cubicle: sparse/0 = 37 images, sparse/1 = 640) -> take the largest
    models = sorted(glob.glob(os.path.join(src, "sparse", "*")))
    rec = max((pycolmap.Reconstruction(m) for m in models), key=lambda r: len(r.images))
    by_name = {img.name: img for img in rec.images.values()}
    rgbs = sorted(glob.glob(os.path.join(src, "images", "rgb_*.jpg")),
                  key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    rgbs = [p for p in rgbs if os.path.basename(p) in by_name]
    # metric scale from depth pngs
    ratios = []
    for p in rgbs[:: max(1, len(rgbs) // n_scale_imgs)]:
        img = by_name[os.path.basename(p)]; cam = rec.cameras[img.camera_id]
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
        dp = cv2.imread(os.path.join(src, "depth", os.path.basename(p).replace("rgb_", "depth_").replace(".jpg", ".png")), -1)
        if dp is None:
            continue
        M = cam_from_world(img)
        for p2 in img.points2D:
            if not p2.has_point3D():
                continue
            Pc = M[:3, :3] @ rec.points3D[p2.point3D_id].xyz + M[:3, 3]
            if Pc[2] <= 0:
                continue
            u, v = int(f * Pc[0] / Pc[2] + cx), int(f * Pc[1] / Pc[2] + cy)
            if 0 <= u < dp.shape[1] and 0 <= v < dp.shape[0] and dp[v, u] > 0:
                ratios.append(dp[v, u] / 1000.0 / Pc[2])
    s = float(np.median(ratios))
    out = os.path.join(OUT, scene)
    os.makedirs(os.path.join(out, "rgb"), exist_ok=True); os.makedirs(os.path.join(out, "depth"), exist_ok=True)
    rows, names, fs = [], [], []
    for i, p in enumerate(rgbs):
        base = os.path.basename(p)
        img = by_name[base]; cam = rec.cameras[img.camera_id]; fs.append(cam.params[0])
        link(p, os.path.join(out, "rgb", f"frame_{i:05d}.jpg"))
        d = os.path.join(src, "depth", base.replace("rgb_", "depth_").replace(".jpg", ".png"))
        if os.path.exists(d):
            link(d, os.path.join(out, "depth", f"frame_{i:05d}.png"))
        c2w = np.linalg.inv(cam_from_world(img)); c2w[:3, 3] *= s
        q = R.from_matrix(c2w[:3, :3]).as_quat()
        rows.append(f"{i} {c2w[0,3]:.6f} {c2w[1,3]:.6f} {c2w[2,3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
        names.append(f"{i} {base}")
    with open(os.path.join(out, "gt_pose.txt"), "w") as fh:
        fh.write("# timestamp tx ty tz qx qy qz qw  (COLMAP c2w x metric scale)\n" + "\n".join(rows) + "\n")
    with open(os.path.join(out, "name_map.txt"), "w") as fh:
        fh.write("\n".join(names) + "\n")
    with open(os.path.join(out, "scale.txt"), "w") as fh:
        fh.write(f"{s:.6f}  # metres per COLMAP unit (median depth_png/colmap_z over {len(ratios)} pts)\n")
    cam0 = rec.cameras[by_name[os.path.basename(rgbs[0])].camera_id]
    K = np.eye(4); K[0, 0] = K[1, 1] = float(np.median(fs)); K[0, 2] = cam0.params[1]; K[1, 2] = cam0.params[2]
    np.savetxt(os.path.join(out, "intrinsic.txt"), K, fmt="%.6f")
    ext = np.array([np.linalg.inv(cam_from_world(by_name[os.path.basename(p)]))[:3, 3] for p in rgbs]) * s
    print(f"[{scene}] {len(rgbs)} registered frames (of {len(glob.glob(os.path.join(src,'images','rgb_*.jpg')))} rgb) | "
          f"scale {s:.3f} m/unit ({len(ratios)} pts) | f median {np.median(fs):.1f} (p10-p90 {np.percentile(fs,10):.0f}-{np.percentile(fs,90):.0f}) | "
          f"extent {np.round(ext.max(0)-ext.min(0),1)} m -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=["apartment", "office", "cubicle", "building"])
    a = ap.parse_args()
    for sc in a.scenes:
        if glob.glob(os.path.join(RAW, f"{sc}_unzip", "sparse", "*")):
            prep(sc)
        else:
            print(f"[{sc}] no COLMAP sparse model (skipped)")
