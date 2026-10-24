"""GT-based loop precision for the VGGT-SLAM 1.0 arms (A/B/C).

Parses a VGGT-SLAM run.log (per-submap image lists + `detected_loops [LoopMatch(...)]`),
maps each loop to its two frame files, and measures the GT relative pose between them
(translation distance, rotation angle). For arm C, only loops ACCEPTed in semlc_log.txt
count as added; rejected ones are reported separately (so we can see whether the gate
removed true or false loops). "False" flags: dist > 1.0 m or angle > 90 deg (loose) and
dist > 0.5 m or angle > 60 deg (tight).

    python scripts/vggt1_lc/loop_precision.py --run outputs/vggt1_lc/360_B --gt data/tum/fr1_360_dist/gt_pose.txt
"""
import argparse
import ast
import os
import re

import numpy as np
from scipy.spatial.transform import Rotation as R

LOOP_RE = re.compile(r"LoopMatch\(similarity_score=([\d.]+), query_submap_id=(\d+), "
                     r"query_submap_frame=(\d+), detected_submap_id=(\d+), detected_submap_frame=(?:np\.int64\()?(\d+)\)?\)")


def parse_run(run_dir, submap_size=32):
    """Reconstruct per-submap frame lists from the keyframe stream (their per-submap
    list print is interleaved with tqdm output, so parse paths, not lists) and the
    LoopMatch prints. Submap k = keyframes[k*w : k*w + w + 1] (overlap 1)."""
    paths, seen, loops = [], set(), []
    for ln in open(os.path.join(run_dir, "run.log"), errors="ignore"):
        for pth in re.findall(r"'(/[^']+?\.(?:png|jpg|jpeg))'", ln):
            if pth not in seen:
                seen.add(pth); paths.append(pth)
        for m in LOOP_RE.finditer(ln):
            sc, qs, qf, ds, df = m.groups()
            loops.append(dict(score=float(sc), qs=int(qs), qf=int(qf), ds=int(ds), df=int(df)))
    uniq, seen_l = [], set()
    for l in loops:
        k = (l["qs"], l["qf"], l["ds"], l["df"])
        if k not in seen_l:
            seen_l.add(k); uniq.append(l)
    for l in uniq:
        l["q_path"] = paths[l["qs"] * submap_size + l["qf"]]
        l["r_path"] = paths[l["ds"] * submap_size + l["df"]]
    return uniq


def frame_num(path):
    return int(re.search(r"(\d+)", os.path.basename(path)).group(1))


def load_gt(path):
    rows = [ln.split() for ln in open(path) if ln.strip() and not ln.startswith("#")]
    T = []
    for r_ in rows:
        v = np.array(r_[1:8], float)
        M = np.eye(4); M[:3, :3] = R.from_quat(v[3:7]).as_matrix(); M[:3, 3] = v[:3]
        T.append(M)
    return np.stack(T)


def semlc_verdicts(run_dir):
    p = os.path.join(run_dir, "semlc_log.txt")
    if not os.path.exists(p):
        return None
    out = {}
    for ln in open(p):
        m = re.match(r"\[semlc\] (ACCEPT|REJECT) sub(\d+):(\d+)\((\S+)\) <-> sub(\d+):(\d+)\((\S+)\)", ln)
        if m:
            out[(m.group(4), m.group(7))] = m.group(1)
    return out


def covis(gt, iq, ir, K, dep_dir, depth_scale, stride=8):
    """Fraction of frame ir's GT-depth points that land inside frame iq's image with
    consistent depth (occlusion-aware), and vice versa; returns max of the two."""
    import cv2
    def load(i):
        d = cv2.imread(os.path.join(dep_dir, f"frame_{i:05d}.png"), cv2.IMREAD_UNCHANGED)
        return None if d is None else d.astype(np.float32) / depth_scale
    dq, dr = load(iq), load(ir)
    if dq is None or dr is None:
        return float("nan")
    def one(d_src, T_src, d_dst, T_dst):
        h, w = d_src.shape
        v, u = np.mgrid[0:h:stride, 0:w:stride]
        z = d_src[v, u]; ok = z > 0
        u, v, z = u[ok].astype(np.float32), v[ok].astype(np.float32), z[ok]
        P = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1)
        Pw = P @ T_src[:3, :3].T + T_src[:3, 3]
        Ti = np.linalg.inv(T_dst)
        Pc = Pw @ Ti[:3, :3].T + Ti[:3, 3]
        zc = Pc[:, 2]; front = zc > 0.05
        uu = K[0, 0] * Pc[:, 0] / np.maximum(zc, 1e-6) + K[0, 2]
        vv = K[1, 1] * Pc[:, 1] / np.maximum(zc, 1e-6) + K[1, 2]
        inside = front & (uu >= 0) & (uu < w - 1) & (vv >= 0) & (vv < h - 1)
        if inside.sum() == 0:
            return 0.0
        zd = d_dst[vv[inside].astype(int), uu[inside].astype(int)]
        vis = (zd > 0) & (np.abs(zd - zc[inside]) < 0.15 * zc[inside])
        return float(vis.sum() / max(1, len(z)))
    return max(one(dr, gt[ir], dq, gt[iq]), one(dq, gt[iq], dr, gt[ir]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--depth_dir", default="", help="GT depth dir (frame_XXXXX.png) for co-visibility")
    ap.add_argument("--intrinsic", default="", help="intrinsic.txt (4x4 or 3x3)")
    ap.add_argument("--depth_scale", type=float, default=5000.0)
    ap.add_argument("--submap_size", type=int, default=32)
    a = ap.parse_args()
    K = np.loadtxt(a.intrinsic)[:3, :3] if a.intrinsic else None
    loops = parse_run(a.run, a.submap_size)
    gt = load_gt(a.gt)
    verd = semlc_verdicts(a.run)
    rows = []
    for l in loops:
        iq, ir = frame_num(l["q_path"]), frame_num(l["r_path"])
        if iq >= len(gt) or ir >= len(gt):
            continue
        rel = np.linalg.inv(gt[ir]) @ gt[iq]
        d = float(np.linalg.norm(rel[:3, 3]))
        ang = float(np.degrees(np.linalg.norm(R.from_matrix(rel[:3, :3]).as_rotvec())))
        v = "ADDED" if verd is None else verd.get((os.path.basename(l["q_path"]), os.path.basename(l["r_path"])), "?")
        cv = covis(gt, iq, ir, K, a.depth_dir, a.depth_scale) if (K is not None and a.depth_dir) else float("nan")
        rows.append((l, iq, ir, d, ang, v, cv))
    name = os.path.basename(a.run.rstrip("/"))
    print(f"[{name}] {len(rows)} SALAD candidates")
    def truth(d, ang, cv):
        if not np.isnan(cv):
            return "true" if cv >= 0.20 else ("weak" if cv >= 0.05 else "FALSE")
        return "FALSE" if (d > 1.0 or ang > 90) else ("weak" if (d > 0.5 or ang > 60) else "true")
    for l, iq, ir, d, ang, v, cv in rows:
        print(f"  {v:6s} f{iq:05d}<->f{ir:05d} sub{l['qs']}<->sub{l['ds']} L2={l['score']:.3f} "
              f"GT dist={d:.2f}m ang={ang:.0f}deg covis={cv:.2f} -> {truth(d, ang, cv)}")
    added = [r for r in rows if r[5] in ("ADDED", "ACCEPT")]
    rej = [r for r in rows if r[5] == "REJECT"]
    nf = sum(1 for r in added if truth(r[3], r[4], r[6]) == "FALSE")
    nt_rej = sum(1 for r in rej if truth(r[3], r[4], r[6]) == "true")
    print(f"  SUMMARY added={len(added)} false={nf} | rejected={len(rej)} of which true={nt_rej}")


if __name__ == "__main__":
    main()
