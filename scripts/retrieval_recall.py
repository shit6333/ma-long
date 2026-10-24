"""Retrieval-recall audit for one of our ma_slam runs: embed its keyframes with SALAD, label every
keyframe pair (submap gap >= min_gap) with GT co-visibility, and report how many TRUE revisits exist,
their SALAD similarity, and what top-1-per-submap / threshold retrieval would catch.

    python scripts/retrieval_recall.py --run outputs/clio_kf50/apartment_base --data data/clio/apartment --depth_scale 1000
"""
import argparse, os, sys, re
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vggt1_lc"))
from loop_precision import covis, load_gt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--depth_scale", type=float, default=1000.0); ap.add_argument("--submap", type=int, default=32)
    ap.add_argument("--min_gap", type=int, default=2); ap.add_argument("--covis_true", type=float, default=0.3)
    a = ap.parse_args()
    kf = [int(l.split()[0]) for l in open(os.path.join(a.run, "gt_kf.txt")) if l.strip() and not l.startswith("#")]
    ext = "jpg" if os.path.exists(os.path.join(a.data, "rgb", f"frame_{kf[0]:05d}.jpg")) else "png"
    paths = [os.path.join(a.data, "rgb", f"frame_{i:05d}.{ext}") for i in kf]
    from ma_slam.loop import ImageRetrieval
    D = ImageRetrieval().describe(paths)                       # (N, d) L2-normalised
    S = D @ D.T
    K = np.loadtxt(os.path.join(a.data, "intrinsic.txt"))[:3, :3]
    gt = load_gt(os.path.join(a.data, "gt_pose.txt"))
    n = len(kf); sub = np.arange(n) // (a.submap + 1)
    pairs = [(i, j) for i in range(n) for j in range(i) if sub[i] - sub[j] >= a.min_gap]
    cv = np.array([covis(gt, kf[i], kf[j], K, os.path.join(a.data, "depth"), a.depth_scale) for i, j in pairs])
    sim = np.array([S[i, j] for i, j in pairs])
    true = cv >= a.covis_true; false = cv < 0.05
    print(f"{n} keyframes, {sub.max()+1} submaps, {len(pairs)} cross-submap pairs (gap>={a.min_gap}): TRUE revisits (covis>={a.covis_true}) = {true.sum()}, false = {false.sum()}")
    if true.sum():
        print(f"  SALAD sim of TRUE pairs : p10 {np.percentile(sim[true],10):.2f} med {np.median(sim[true]):.2f} p90 {np.percentile(sim[true],90):.2f} max {sim[true].max():.2f}")
    print(f"  SALAD sim of FALSE pairs: p90 {np.percentile(sim[false],90):.2f} p99 {np.percentile(sim[false],99):.2f} max {sim[false].max():.2f}")
    for thr in (0.4, 0.5, 0.6, 0.7, 0.8):
        print(f"  sim>={thr}: true pairs kept {int((true & (sim>=thr)).sum())}/{int(true.sum())}  false pairs admitted {int((false & (sim>=thr)).sum())}")
    # which submaps have any true revisit to an older submap, and what our top-1-per-query-frame retrieval sees
    print("  per submap: #true-revisit pairs / best sim among true / best-sim pair overall (truth)")
    for s in range(a.min_gap, sub.max()+1):
        idx = [k for k,(i,j) in enumerate(pairs) if sub[i]==s]
        if not idx: continue
        t = [k for k in idx if true[k]]
        best = max(idx, key=lambda k: sim[k]); i,j = pairs[best]
        print(f"    submap {s:2d}: true pairs {len(t):4d}  best true sim {max(sim[k] for k in t) if t else float('nan'):.2f}  | best overall sim {sim[best]:.2f} (kf {kf[i]}<->{kf[j]}, covis {cv[best]:.2f})")
if __name__ == "__main__":
    main()
