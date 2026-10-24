"""Offline evaluation of the proposed object-level rigid-consistency loop verifier on the
recorded candidate dumps (scripts/render_semlc_graph.py JSONs), labelled with GT co-visibility.

Verifier (3 parameters, all physical):
  1. match: mutual-NN instance pairs with descriptor cos >= tau     (dumps store the MNN pairs)
  2. rigid check: Kabsch over triplets of matched camera-frame centres -> R,t; a pair is an
     inlier if |R c_r + t - c_q| < eps, eps = rel * d_ref (d_ref = median depth of matched
     history centres; scale-free so it also works on VGGT's arbitrary scale)
  3. verdict: max #inliers over triplets >= m  -> PASS, else REJECT (no third state)

Compares, on the same candidates, against the recorded S1-only and S1+S2 (old rules) verdicts.

    python scripts/replay_rigid_verify.py --dumps outputs/sem_lc_shadow/vis/fr1_*_w32/semlc_dump outputs/vggt1_lc/*_C/semlc_dump ...
"""
import argparse
import glob
import itertools
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vggt1_lc"))
from loop_precision import covis, load_gt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def kabsch(P, Q):
    """R,t minimising |R P + t - Q| (P,Q: (n,3))."""
    mp, mq = P.mean(0), Q.mean(0)
    H = (P - mp).T @ (Q - mq)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def rigid_inliers(Cq, Cr, eps):
    """Max inlier count over all triplet hypotheses (exhaustive; n is small)."""
    n = len(Cq)
    if n < 3:
        return 0, None
    best, best_rt = 0, None
    for tri in itertools.combinations(range(n), 3):
        R, t = kabsch(Cr[list(tri)], Cq[list(tri)])
        res = np.linalg.norm((Cr @ R.T + t) - Cq, axis=1)
        k = int((res < eps).sum())
        if k > best:
            best, best_rt = k, (R, t)
    return best, best_rt


def scene_gt(q_path):
    """Dataset root + depth/K/scale from the dump's image path."""
    m = re.search(r"(data/(?:tum/fr1_\w+?_dist|scene\d+_\d\d|7scenes/\w+|scannetpp/\w+|clio/\w+))/", q_path)
    if not m:
        return None
    d = os.path.join(ROOT, m.group(1))
    if not os.path.exists(os.path.join(d, "gt_pose.txt")):
        return None
    scale = 5000.0 if "/tum/" in d else 1000.0
    kf = "depth_intrinsic.txt" if os.path.exists(os.path.join(d, "depth_intrinsic.txt")) else "intrinsic.txt"
    return d, np.loadtxt(os.path.join(d, kf))[:3, :3], scale


def frame_num(p):
    return int(re.search(r"frame_(\d+)", os.path.basename(p)).group(1))


def load_candidates(dump_dirs):
    cands, gts = [], {}
    for dd in dump_dirs:
        for f in sorted(glob.glob(os.path.join(dd, "*.json"))):
            r = json.load(open(f))
            g = scene_gt(r["q_path"])
            if g is None:
                continue
            d, K, scale = g
            if d not in gts:
                gts[d] = load_gt(os.path.join(d, "gt_pose.txt"))
            gt = gts[d]
            q, m = frame_num(r["q_path"]), frame_num(r["r_path"])
            if q >= len(gt) or m >= len(gt):
                continue
            cv = covis(gt, q, m, K, os.path.join(d, "depth"), scale)
            if np.isnan(cv):
                continue
            r["_truth"] = "true" if cv >= 0.2 else ("weak" if cv >= 0.05 else "FALSE")
            r["_covis"] = cv
            r["_src"] = os.path.basename(os.path.dirname(dd)) + "/" + os.path.basename(dd.rstrip("/"))
            cands.append(r)
    return cands


def old_rules(r):
    """Recorded verdict under the current (old) rules: S1 PASS needed and S2 not a hard-reject."""
    s1, s2 = r["s1"]["status"], (r["s2"] or {}).get("status")
    if s1 != "PASS":
        return "REJECT"
    if s2 in ("LARGE_VIEW_CHANGE", "T_FAR", "M_LOW"):
        return "REJECT"
    return "PASS" if s2 in (None, "PASS") else "SOFT"   # SOFT = FAIL -> raised coloc bar


def rigid_verdict(r, tau, rel, m):
    xq = {n["row"]: n["xyz"] for n in r["q_nodes"] if n.get("xyz")}
    xr = {n["row"]: n["xyz"] for n in r["r_nodes"] if n.get("xyz")}
    pairs = [p for p in r["pairs"] if p["sim"] >= tau and p["q_row"] in xq and p["r_row"] in xr]
    if len(pairs) < m:
        return "REJECT", len(pairs), 0
    Cq = np.array([xq[p["q_row"]] for p in pairs]); Cr = np.array([xr[p["r_row"]] for p in pairs])
    d_ref = float(np.median(np.linalg.norm(Cr, axis=1)))
    k, _ = rigid_inliers(Cq, Cr, rel * max(d_ref, 1e-6))
    return ("PASS" if k >= m else "REJECT"), len(pairs), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True)
    ap.add_argument("--taus", default="0.85,0.90")
    ap.add_argument("--rels", default="0.10,0.15,0.20")
    ap.add_argument("--m", type=int, default=3)
    a = ap.parse_args()
    dirs = [d for pat in a.dumps for d in sorted(glob.glob(pat))]
    cands = load_candidates(dirs)
    T = [c for c in cands if c["_truth"] == "true"]; F = [c for c in cands if c["_truth"] == "FALSE"]
    print(f"{len(cands)} GT-labelled candidates from {len(dirs)} dumps: true={len(T)} false={len(F)} weak={len(cands)-len(T)-len(F)}")

    def table(name, fn):
        tp = sum(fn(c) == "PASS" for c in T); fp = sum(fn(c) == "PASS" for c in F)
        print(f"  {name:38s} true kept {tp:3d}/{len(T)} ({tp/len(T)*100:3.0f}%)   false let through {fp:2d}/{len(F)}")
    print("--- recorded verdicts")
    table("S1 only (PASS)", lambda c: "PASS" if c["s1"]["status"] == "PASS" else "REJECT")
    table("S1+S2 old rules (hard rejects only)", lambda c: "PASS" if old_rules(c) in ("PASS", "SOFT") else "REJECT")
    print("--- rigid verifier (MNN pairs >= tau, Kabsch triplets, inliers >= m)")
    for tau in [float(x) for x in a.taus.split(",")]:
        for rel in [float(x) for x in a.rels.split(",")]:
            table(f"tau={tau:.2f} eps={rel:.2f}*d_ref m={a.m}", lambda c, tau=tau, rel=rel: rigid_verdict(c, tau, rel, a.m)[0])
    # where do the true-loop losses come from: too few matches (tau) vs geometry (inliers)?
    for tau in [float(x) for x in a.taus.split(",")]:
        few = sum(1 for c in T if rigid_verdict(c, tau, 0.15, a.m)[1] < a.m)
        geo = sum(1 for c in T if rigid_verdict(c, tau, 0.15, a.m)[1] >= a.m and rigid_verdict(c, tau, 0.15, a.m)[2] < a.m)
        print(f"  true-loop losses @tau={tau:.2f} eps=0.15 m={a.m}: {few} have <{a.m} matched pairs (matching-limited), {geo} have enough pairs but <{a.m} inliers (geometry-limited)")
    # inlier-count distributions at a reference setting
    tau, rel = 0.90, 0.15
    kt = [rigid_verdict(c, tau, rel, a.m)[2] for c in T]; kf = [rigid_verdict(c, tau, rel, a.m)[2] for c in F]
    print(f"--- inliers @ tau={tau} eps={rel}*d_ref: true  " + " ".join(f"{k}:{kt.count(k)}" for k in sorted(set(kt))))
    print(f"                                     false " + " ".join(f"{k}:{kf.count(k)}" for k in sorted(set(kf))))
    print("--- false loops that PASS the rigid check (tau 0.90, eps 0.15):")
    for c in F:
        v, npair, k = rigid_verdict(c, tau, rel, a.m)
        if v == "PASS":
            print(f"    {c['_src']} {os.path.basename(c['q_path'])}<->{os.path.basename(c['r_path'])} pairs={npair} inliers={k} covis={c['_covis']:.2f} s1={c['s1']['status']} s2={(c['s2'] or {}).get('status')}")
    print("--- true loops REJECTED by the rigid check but kept by old rules:")
    for c in T:
        v, npair, k = rigid_verdict(c, tau, rel, a.m)
        if v == "REJECT" and old_rules(c) in ("PASS", "SOFT"):
            print(f"    {c['_src']} {os.path.basename(c['q_path'])}<->{os.path.basename(c['r_path'])} pairs={npair} inliers={k} nq/nr={c['s1']['n_q']}/{c['s1']['n_r']} covis={c['_covis']:.2f}")


if __name__ == "__main__":
    main()
