"""E4: does inst-LC heal the instance MAP? Compares the LC-ON and LC-OFF arms on the quantity the
experiment is about — DUPLICATE instances (one GT object split into several predicted clusters) — not just AP.

Duplicate count (docs/e4_lc_map_healing.md §4, cheapest version): for each GT instance, count the predicted
clusters whose best match is that GT with IoU >= 0.25; duplicates = sum(max(0, k - 1)).  Read from the
evaluator's own `--save_json` dump (`pred_details`: best_iou / matched_gt per predicted cluster), so both arms
are scored by exactly the same matcher.

    PYTHONPATH=src python scripts/e4_dup_count.py --on outputs/ma_sem/scannetpp_final \
        --off outputs/ma_sem/scannetpp_noLC --details eval_details.json [--exclude 3db0a1c8f3 d755b3d9d8]
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter

import numpy as np


def scene_stats(run_dir, sc, details, iou_floor):
    f = os.path.join(run_dir, sc, details)
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    pd = d.get("pred_details", [])
    per_gt = Counter(int(p["matched_gt"]) for p in pd
                     if p.get("matched_gt", -1) >= 0 and p.get("best_iou", 0.0) >= iou_floor)
    dup = sum(max(0, k - 1) for k in per_gt.values())
    ag = d.get("class_agnostic", {})
    return dict(n_pred=len(pd), n_gt=int(ag.get("n_gt", 0)), matched_gt=len(per_gt), dup=dup,
                AP=100 * ag.get("AP", float("nan")), AP50=100 * ag.get("AP50", float("nan")),
                AP25=100 * ag.get("AP25", float("nan")))


def run_stats(run_dir):
    p = os.path.join(run_dir, "SUMMARY.csv")
    return {r["scene"]: r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True); ap.add_argument("--off", required=True)
    ap.add_argument("--details", default="eval_details.json")
    ap.add_argument("--iou_floor", type=float, default=0.25)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--per_scene", action="store_true")
    a = ap.parse_args()
    S_on, S_off = run_stats(a.on), run_stats(a.off)
    scenes = [os.path.basename(d.rstrip("/")) for d in sorted(glob.glob(a.on + "/*/"))]
    scenes = [s for s in scenes if not s.startswith("_") and s not in a.exclude]
    rows = []
    for sc in scenes:
        on, off = scene_stats(a.on, sc, a.details, a.iou_floor), scene_stats(a.off, sc, a.details, a.iou_floor)
        if on is None or off is None:
            continue
        r = dict(scene=sc, on=on, off=off,
                 on_loops=float(S_on.get(sc, {}).get("loops") or 0), off_loops=float(S_off.get(sc, {}).get("loops") or 0),
                 on_se3=float(S_on.get(sc, {}).get("se3_ate") or "nan"), off_se3=float(S_off.get(sc, {}).get("se3_ate") or "nan"),
                 on_sim3=float(S_on.get(sc, {}).get("sim3_ate") or "nan"), off_sim3=float(S_off.get(sc, {}).get("sim3_ate") or "nan"),
                 on_al=float(S_on.get(sc, {}).get("align_rmse") or "nan"), off_al=float(S_off.get(sc, {}).get("align_rmse") or "nan"))
        rows.append(r)
        if a.per_scene:
            print(f"  {sc[:16]:16s} loops {r['on_loops']:5.0f}/{r['off_loops']:3.0f} | AP50 {on['AP50']:5.1f}/{off['AP50']:5.1f} "
                  f"| n_pred {on['n_pred']:4d}/{off['n_pred']:4d} | dup {on['dup']:3d}/{off['dup']:3d} | SE3 {r['on_se3']:.3f}/{r['off_se3']:.3f}")
    if not rows:
        print("no paired scenes with details JSON — run the evals with --save_json first"); return
    m = lambda f: np.nanmean([f(r) for r in rows])
    print(f"\nE4 — inst-LC map healing, n={len(rows)} scenes (IoU floor {a.iou_floor})")
    print(f"{'metric':22s} {'LC ON':>10s} {'LC OFF':>10s} {'delta':>9s}")
    for name, fon, foff in [
            ("loops accepted", lambda r: r["on_loops"], lambda r: r["off_loops"]),
            ("Sim3-ATE (m)", lambda r: r["on_sim3"], lambda r: r["off_sim3"]),
            ("SE3-ATE (m)", lambda r: r["on_se3"], lambda r: r["off_se3"]),
            ("align RMSE (m)", lambda r: r["on_al"], lambda r: r["off_al"]),
            ("AP", lambda r: r["on"]["AP"], lambda r: r["off"]["AP"]),
            ("AP50", lambda r: r["on"]["AP50"], lambda r: r["off"]["AP50"]),
            ("AP25", lambda r: r["on"]["AP25"], lambda r: r["off"]["AP25"]),
            ("n_pred", lambda r: r["on"]["n_pred"], lambda r: r["off"]["n_pred"]),
            ("GT matched (IoU>=t)", lambda r: r["on"]["matched_gt"], lambda r: r["off"]["matched_gt"]),
            ("duplicate instances", lambda r: r["on"]["dup"], lambda r: r["off"]["dup"])]:
        a_, b_ = m(fon), m(foff)
        print(f"{name:22s} {a_:10.3f} {b_:10.3f} {a_-b_:+9.3f}")
    d_on = np.array([r["on"]["dup"] for r in rows], float); d_off = np.array([r["off"]["dup"] for r in rows], float)
    better = int((d_on < d_off).sum()); worse = int((d_on > d_off).sum())
    print(f"\nduplicates: LC-ON lower in {better}/{len(rows)} scenes, higher in {worse}, tied in {len(rows)-better-worse}"
          f" | total {int(d_on.sum())} vs {int(d_off.sum())}")


if __name__ == "__main__":
    main()
