"""Trajectory figure for the loop-closure story: GT vs without IDLV vs with IDLV.

Each estimate is Sim(3)-aligned to GT (the same alignment the reported Sim3-ATE uses), projected on the
two dominant world axes (top-down), and annotated with its ATE and accepted-loop count. Loop edges can be
drawn as chords between the two ends of each accepted loop, coloured by GT co-visibility (true / FALSE) —
that is what makes the "w/o IDLV" panel show its false loops.

    PYTHONPATH=src python scripts/plot_traj_lc.py --gt <run>/gt_kf.txt \
        --off <run_without>/  --on <run_with>/ --title "TUM fr1_room (relaxed retrieval)" --out fig.png
"""
import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from eval import umeyama_sim3, load_estimated_positions, load_gt_positions

ACC = re.compile(r"LOOPCAND ACCEPT coloc=([\d.]+) sim=([\d.]+).*?frame_(\d+)\.(?:png|jpg) mimg=.*?frame_(\d+)\.(?:png|jpg)")


def align(est, gt):
    n = min(len(est), len(gt)); est, gt = est[:n], gt[:n]
    s, R, _ = umeyama_sim3(est, gt)
    t = gt.mean(0) - s * R @ est.mean(0)
    a = (s * (est @ R.T)) + t
    return a, gt, float(np.sqrt(np.mean(np.sum((a - gt) ** 2, 1))))


def loops_of(run, data_dir, gt_full, K, depth_dir, depth_scale):
    """accepted loops as (idx_q, idx_r, truth) using the run's own loops.txt + GT co-visibility."""
    out, seen = [], set()
    f = os.path.join(run, "loops.txt")
    if not os.path.exists(f):
        return out
    sys.path.insert(0, "scripts/vggt1_lc")
    from loop_precision import covis
    for l in open(f):
        m = ACC.search(l)
        if not m:
            continue
        q, r = int(m.group(3)), int(m.group(4))
        if (q, r) in seen:
            continue
        seen.add((q, r))
        cv = covis(gt_full, q, r, K, depth_dir, depth_scale) if K is not None else float("nan")
        out.append((q, r, "true" if cv >= 0.2 else ("weak" if cv >= 0.05 else "FALSE")))
    return out


def panel(ax, est, gt, title, loops=None, kf_ids=None, axes=(0, 1)):
    i, j = axes
    ax.plot(gt[:, i], gt[:, j], "-", color="0.35", lw=2.4, label="GT", zorder=2)
    ax.plot(est[:, i], est[:, j], "-", color="#d62728", lw=1.8, label="estimate", zorder=3)
    ax.plot(gt[0, i], gt[0, j], "o", color="0.2", ms=7, zorder=5)
    if loops and kf_ids is not None:
        pos = {f: k for k, f in enumerate(kf_ids)}
        for q, r, truth in loops:
            if q not in pos or r not in pos:
                continue
            a, b = est[pos[q]], est[pos[r]]
            c = "#2ca02c" if truth == "true" else ("#ff7f0e" if truth == "weak" else "#e31a1c")
            ax.plot([a[i], b[i]], [a[j], b[j]], "--", color=c, lw=2.0 if truth == "FALSE" else 1.2,
                    alpha=0.95 if truth == "FALSE" else 0.55, zorder=4)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal"); ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", required=True, help="run dir WITHOUT IDLV")
    ap.add_argument("--on", required=True, help="run dir WITH IDLV")
    ap.add_argument("--data", default=None, help="scene dir (for GT depth/K -> loop truth colours)")
    ap.add_argument("--depth_scale", type=float, default=5000.0)
    ap.add_argument("--title", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gtK, depth_dir, gt_full = None, None, None
    if a.data:
        kf = "depth_intrinsic.txt" if os.path.exists(os.path.join(a.data, "depth_intrinsic.txt")) else "intrinsic.txt"
        gtK = np.loadtxt(os.path.join(a.data, kf))[:3, :3]
        depth_dir = os.path.join(a.data, "depth")
        sys.path.insert(0, "scripts/vggt1_lc")
        from loop_precision import load_gt as lp_load_gt
        gt_full = lp_load_gt(os.path.join(a.data, "gt_pose.txt"))

    panels = []
    for run, lab in ((a.off, "w/o IDLV"), (a.on, "w/ IDLV")):
        est = load_estimated_positions(os.path.join(run, "camera_poses.txt"))
        gt = load_gt_positions(os.path.join(run, "gt_kf.txt"))
        # gt_kf.txt carries the ORIGINAL timestamps of the kept keyframes -> map back to frame indices
        # by matching them against the scene's full gt_pose.txt (row order == frame number).
        kf_ids = None
        if a.data:
            full = [l.split()[0] for l in open(os.path.join(a.data, "gt_pose.txt"))
                    if l.strip() and not l.startswith("#")]
            idx_of = {t: i for i, t in enumerate(full)}
            kf_ids = [idx_of.get(l.split()[0], -1) for l in open(os.path.join(run, "gt_kf.txt"))
                      if l.strip() and not l.startswith("#")]
        aligned, gtc, ate = align(est, gt)
        lp = loops_of(run, a.data, gt_full, gtK, depth_dir, a.depth_scale) if a.data else []
        nf = sum(1 for x in lp if x[2] == "FALSE")
        panels.append((aligned, gtc, f"{lab}   ATE {ate:.3f} m   loops {len(lp)}"
                                    + (f" ({nf} FALSE)" if nf else " (0 false)"), lp, kf_ids))

    allp = np.concatenate([p[0] for p in panels] + [p[1] for p in panels])
    ext = allp.max(0) - allp.min(0)
    keep = tuple(int(x) for x in np.argsort(-ext)[:2])            # drop the thinnest axis -> top-down
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (est, gt, title, lp, kf) in zip(axs, panels):
        panel(ax, est, gt, title, lp, kf, keep)
    axs[0].legend(loc="best", fontsize=9, framealpha=0.9)
    lbl = "xyz"
    for ax in axs:
        ax.set_xlabel(f"{lbl[keep[0]]} (m)", fontsize=9); ax.set_ylabel(f"{lbl[keep[1]]} (m)", fontsize=9)
    if a.title:
        fig.suptitle(a.title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if a.title else 1))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=200)
    print(f"-> {a.out}")
    for _, _, t, lp, _ in panels:
        print(f"   {t}" + ("" if not lp else "  | " + ", ".join(f"{q}<->{r}:{s}" for q, r, s in lp[:8])))


if __name__ == "__main__":
    main()
