"""Convert TUM RGB-D freiburg1 sequences into ma_slam scene dirs (UNDISTORTED).

Standalone data-prep tool — touches NO pipeline code. For each sequence it:
  1. Associates rgb <-> depth <-> groundtruth by nearest timestamp (<= max_diff).
     Only frames with BOTH a depth and a GT match are kept, so rgb[i]/depth[i]/
     gt-row[i] are index-aligned (ma_slam + eval assume per-index correspondence).
  2. UNDISTORTS rgb (bilinear) and depth (nearest, to avoid blending depths) with
     the fr1 pinhole K + plumb_bob distortion. Output keeps the SAME K (cv2 default),
     so intrinsic.txt is the fr1 K and the images are now true pinhole — the correct
     input for DA3 / MapAnything (which assume pinhole, unlike VGGT-SLAM which feeds
     raw distorted RGB and absorbs the calibration error projectively on SL(4)).
  3. Re-indexes frames to frame_00000.png ... and writes intrinsic.txt (4x4) +
     gt_pose.txt (TUM header, one row per kept frame, in frame order).

Usage:
    python scripts/prep_tum_fr1.py --raw data/tum/_raw --out data/tum [--seqs xyz desk ...]

TUM fr1 calibration (ROS default, 640x480):
    fx=517.306408 fy=516.469215 cx=318.643040 cy=255.313989
    dist (k1,k2,p1,p2,k3) = 0.262383 -0.953104 -0.005358 0.002628 1.163314
Depth: 16-bit PNG, 5000 units / metre  (pass --depth_scale 5000 to ma_slam.run).
"""

from __future__ import annotations

import argparse
import os
import tarfile
from typing import List, Tuple

import cv2
import numpy as np

FR1_K = np.array([[517.306408, 0.0, 318.643040],
                  [0.0, 516.469215, 255.313989],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
FR1_DIST = np.array([0.262383, -0.953104, -0.005358, 0.002628, 1.163314], dtype=np.float64)
ALL_SEQS = ["360", "desk", "desk2", "floor", "plant", "room", "rpy", "teddy", "xyz"]


def _read_tum_list(path: str) -> List[Tuple[float, str]]:
    """Read a TUM index file (`timestamp  relative/path`), skipping comments."""
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            out.append((float(parts[0]), parts[1]))
    return out


def _read_gt(path: str) -> Tuple[np.ndarray, List[str]]:
    """Read groundtruth.txt -> (timestamps array, raw line strings)."""
    ts, lines = [], []
    with open(path) as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            ts.append(float(s.split()[0]))
            lines.append(s)
    return np.asarray(ts), lines


def _nearest(query: float, ts: np.ndarray) -> Tuple[int, float]:
    """Index of the nearest timestamp in (sorted) `ts` and the abs diff."""
    j = int(np.searchsorted(ts, query))
    best_i, best_d = -1, float("inf")
    for k in (j - 1, j, j + 1):
        if 0 <= k < len(ts):
            d = abs(ts[k] - query)
            if d < best_d:
                best_i, best_d = k, d
    return best_i, best_d


def prep_sequence(seq_dir: str, out_dir: str, max_diff: float = 0.02,
                  undistort: bool = True) -> int:
    rgb = _read_tum_list(os.path.join(seq_dir, "rgb.txt"))
    depth = _read_tum_list(os.path.join(seq_dir, "depth.txt"))
    gt_ts, gt_lines = _read_gt(os.path.join(seq_dir, "groundtruth.txt"))
    d_ts = np.asarray([t for t, _ in depth])

    os.makedirs(os.path.join(out_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "depth"), exist_ok=True)

    # undistortion maps (same K in -> K out: result is true pinhole at fr1 K)
    sample = cv2.imread(os.path.join(seq_dir, rgb[0][1]))
    h, w = sample.shape[:2]
    m1, m2 = cv2.initUndistortRectifyMap(FR1_K, FR1_DIST, None, FR1_K, (w, h), cv2.CV_32FC1)

    kept_gt: List[str] = []
    i = 0
    for ts, rpath in rgb:
        di, dd = _nearest(ts, d_ts)
        gi, gd = _nearest(ts, gt_ts)
        if di < 0 or dd > max_diff or gi < 0 or gd > max_diff:
            continue  # need both a depth and a GT match
        crgb = cv2.imread(os.path.join(seq_dir, rpath), cv2.IMREAD_COLOR)
        cdep = cv2.imread(os.path.join(seq_dir, depth[di][1]), cv2.IMREAD_UNCHANGED)  # 16-bit
        if crgb is None or cdep is None:
            continue
        if undistort:
            urgb = cv2.remap(crgb, m1, m2, cv2.INTER_LINEAR)
            udep = cv2.remap(cdep, m1, m2, cv2.INTER_NEAREST)        # nearest: no depth blending
        else:
            urgb, udep = crgb, cdep                                  # keep raw (distorted) pixels
        cv2.imwrite(os.path.join(out_dir, "rgb", f"frame_{i:05d}.png"), urgb)
        cv2.imwrite(os.path.join(out_dir, "depth", f"frame_{i:05d}.png"), udep)
        kept_gt.append(gt_lines[gi])
        i += 1

    K4 = np.eye(4); K4[:3, :3] = FR1_K
    np.savetxt(os.path.join(out_dir, "intrinsic.txt"), K4, fmt="%.6f")
    with open(os.path.join(out_dir, "gt_pose.txt"), "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        f.write("\n".join(kept_gt) + ("\n" if kept_gt else ""))
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/tum/_raw", help="dir with the .tgz archives")
    ap.add_argument("--out", default="data/tum", help="output root; scenes -> <out>/fr1_<seq>")
    ap.add_argument("--seqs", nargs="*", default=ALL_SEQS)
    ap.add_argument("--max_diff", type=float, default=0.02, help="max timestamp diff (s) for association")
    ap.add_argument("--no_undistort", action="store_true",
                    help="keep raw (distorted) pixels; output to fr1_<seq>_dist (intrinsic still fr1 K)")
    a = ap.parse_args()

    undistort = not a.no_undistort
    suffix = "" if undistort else "_dist"
    for seq in a.seqs:
        name = f"rgbd_dataset_freiburg1_{seq}"
        extracted = os.path.join(a.raw, name)
        if not os.path.isdir(extracted):
            tgz = os.path.join(a.raw, name + ".tgz")
            if not os.path.isfile(tgz):
                print(f"[prep] SKIP {seq}: no {tgz} and no extracted dir")
                continue
            print(f"[prep] extracting {tgz} ...")
            with tarfile.open(tgz) as t:
                t.extractall(a.raw)
        out_dir = os.path.join(a.out, f"fr1_{seq}{suffix}")
        n = prep_sequence(extracted, out_dir, a.max_diff, undistort)
        kind = "undistorted" if undistort else "RAW-distorted"
        print(f"[prep] fr1_{seq}{suffix}: {n} aligned+{kind} frames -> {out_dir}")


if __name__ == "__main__":
    main()
