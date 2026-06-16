"""Sweep loop-closure verification parameters (mainly coloc_ratio) on both scenes.

rgb+depth+intr, cs=20, LC on. Reports Sim3/SE3 ATE, #accepted loop constraints, FPS.
"""
from __future__ import annotations
import argparse, copy, glob, json, os, time
import numpy as np, torch
from model.ma_infer import MaChunkModel
from model.inputs import load_intrinsics
from ma_long.pipeline import MaLongPipeline, DEFAULT_CONFIG
from eval import ate_rmse

SCENES = {"s0011": "data/scene0011_00", "s0378": "data/scene0378_00"}


def run(model, sp, out, coloc, dist_thresh=None):
    imgs = sorted(glob.glob(f"{sp}/rgb/*.png"))
    deps = [p.replace("/rgb/", "/depth/") for p in imgs]
    K = load_intrinsics(f"{sp}/intrinsic.txt"); gt = f"{sp}/gt_pose.txt"
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["Model"].update(chunk_size=20, overlap=8, loop_enable=True)
    cfg["Loop"]["verify"]["coloc_ratio"] = coloc
    if dist_thresh is not None:
        cfg["Loop"]["geometric"]["dist_thresh"] = dist_thresh
    os.makedirs(out, exist_ok=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.time()
    res = MaLongPipeline(config=cfg, model=model).run(
        imgs, out, mode="rgb+depth+intr", depth_paths=deps, intrinsics=K, keep_cache=False)
    torch.cuda.synchronize(); dt = time.time() - t0
    a, a3 = ate_rmse(res["poses_txt"], gt, "sim3"), ate_rmse(res["poses_txt"], gt, "se3")
    return dict(coloc=coloc, dist=dist_thresh, n_loops=res["n_loops"],
                sim3=round(a["ate_rmse"], 4), se3=round(a3["ate_rmse"], 4),
                fps=round(len(imgs) / dt, 2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/sweep_lc")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    print("=== loading MapAnything once ===")
    model = MaChunkModel(device="cuda")
    out = {}
    for sk, sp in SCENES.items():
        rows = []
        for coloc in [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
            print(f"\n### {sk} coloc={coloc}")
            rows.append(run(model, sp, f"{args.out}/{sk}_c{int(coloc*100):03d}", coloc))
        out[sk] = rows
    json.dump(out, open(f"{args.out}/summary.json", "w"), indent=2)
    print("\n\n================ COLOC SWEEP (rgb+depth+intr, cs20, LC) ================")
    for sk, rows in out.items():
        print(f"\n### {sk}")
        print(f"{'coloc':>6} | {'n_loops':>7} | {'Sim3-ATE':>9} | {'SE3-ATE':>8} | {'fps':>5}")
        print("-" * 48)
        for r in rows:
            print(f"{r['coloc']:>6} | {r['n_loops']:>7} | {r['sim3']:>9} | {r['se3']:>8} | {r['fps']:>5}")
    print(f"\nsummary -> {args.out}/summary.json")


if __name__ == "__main__":
    main()
