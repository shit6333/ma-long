"""GT-label the loops of a VGGT-SLAM run on a RAW TUM folder (timestamp filenames): true / weak / FALSE by
GT co-visibility (nearest-timestamp GT pose + depth, occlusion-aware reprojection, same rule as loop_precision).

    python scripts/vggt1_lc/label_raw_tum.py --run outputs/vggt1_lc_sl4_d50/360_sl4_relax95_gate --raw data/tum/_raw/rgbd_dataset_freiburg1_360
Prints one line per candidate (verdict from semlc_log.txt when present, else 'ADDED') and a summary.
"""
import argparse, os, re, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loop_precision import parse_run
from scipy.spatial.transform import Rotation as R

K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]], np.float32)   # TUM fr1


def load_gt(path):
    rows = [l.split() for l in open(path) if l.strip() and not l.startswith("#")]
    t = np.array([float(r[0]) for r in rows]); P = np.array([[float(x) for x in r[1:8]] for r in rows])
    return t, P


def pose_at(gt, ts):
    t, P = gt; i = int(np.argmin(np.abs(t - ts)))
    T = np.eye(4); T[:3, :3] = R.from_quat(P[i, 3:7]).as_matrix(); T[:3, 3] = P[i, :3]
    return T


def depth_at(raw, ts, cache={}):
    if "list" not in cache:
        rows = [l.split() for l in open(os.path.join(raw, "depth.txt")) if not l.startswith("#")]
        cache["list"] = (np.array([float(r[0]) for r in rows]), [r[1] for r in rows])
    t, f = cache["list"]; i = int(np.argmin(np.abs(t - ts)))
    d = cv2.imread(os.path.join(raw, f[i]), cv2.IMREAD_UNCHANGED)
    return d.astype(np.float32) / 5000.0


def covis(dq, Tq, dr, Tr, stride=8):
    def one(d_src, T_src, d_dst, T_dst):
        h, w = d_src.shape; v, u = np.mgrid[0:h:stride, 0:w:stride]
        z = d_src[v, u]; ok = z > 0; u, v, z = u[ok].astype(np.float32), v[ok].astype(np.float32), z[ok]
        P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
        Pw = P @ T_src[:3, :3].T + T_src[:3, 3]; Ti = np.linalg.inv(T_dst); Pc = Pw @ Ti[:3, :3].T + Ti[:3, 3]
        zc = Pc[:, 2]; front = zc > 0.05
        uu = K[0, 0] * Pc[:, 0] / np.maximum(zc, 1e-6) + K[0, 2]; vv = K[1, 1] * Pc[:, 1] / np.maximum(zc, 1e-6) + K[1, 2]
        inside = front & (uu >= 0) & (uu < w - 1) & (vv >= 0) & (vv < h - 1)
        if inside.sum() == 0: return 0.0
        zd = d_dst[vv[inside].astype(int), uu[inside].astype(int)]
        return float(((zd > 0) & (np.abs(zd - zc[inside]) < 0.15 * zc[inside])).sum() / max(1, len(z)))
    return max(one(dr, Tr, dq, Tq), one(dq, Tq, dr, Tr))


def ts_of(p):
    return float(re.search(r"(\d+\.\d+)", os.path.basename(p)).group(1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--raw", required=True)
    ap.add_argument("--quiet", action="store_true"); a = ap.parse_args()
    gt = load_gt(os.path.join(a.raw, "groundtruth.txt"))
    verdict = {}
    lp = os.path.join(a.run, "semlc_log.txt")
    if os.path.exists(lp):
        for l in open(lp):
            m = re.match(r"\[semlc\] (ACCEPT|REJECT) sub(\d+):(\d+)\(.*?\) <-> sub(\d+):(\d+)\(", l)
            if m: verdict[(int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))] = m.group(1)
    cnt = {"ADDED": [0, 0, 0], "ACCEPT": [0, 0, 0], "REJECT": [0, 0, 0]}
    for l in parse_run(a.run):
        tq, tr = ts_of(l["q_path"]), ts_of(l["r_path"])
        cv = covis(depth_at(a.raw, tq), pose_at(gt, tq), depth_at(a.raw, tr), pose_at(gt, tr))
        k = 0 if cv >= 0.2 else (1 if cv >= 0.05 else 2); lab = ["true", "weak", "FALSE"][k]
        v = verdict.get((l["qs"], l["qf"], l["ds"], l["df"]), "ADDED"); cnt[v][k] += 1
        Tq, Tr = pose_at(gt, tq), pose_at(gt, tr); dist = np.linalg.norm(Tq[:3, 3] - Tr[:3, 3])
        ang = np.degrees(np.arccos(np.clip((np.trace(Tq[:3, :3].T @ Tr[:3, :3]) - 1) / 2, -1, 1)))
        if not a.quiet:
            print(f"  {v:6s} sub{l['qs']}:{l['qf']}<->sub{l['ds']}:{l['df']} L2={l['score']:.3f} dist={dist:.2f}m ang={ang:.0f} covis={cv:.2f} {lab}")
    added = cnt["ADDED"] if not verdict else cnt["ACCEPT"]
    print(f"  SUMMARY in-map loops: true={added[0]} weak={added[1]} FALSE={added[2]}"
          + (f" | gate rejected: true={cnt['REJECT'][0]} weak={cnt['REJECT'][1]} FALSE={cnt['REJECT'][2]}" if verdict else ""))


if __name__ == "__main__":
    main()
