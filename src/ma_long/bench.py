"""ma_long benchmark: input modes, loop-closure ablation, chunk-size sweep.

Measures Sim3-ATE, metric SE3-ATE, scale, FPS (frames / end-to-end run time) and
peak VRAM, on both ScanNet scenes. The MapAnything backbone is loaded once and reused
across all runs; per-run VRAM is isolated via reset_peak_memory_stats. Loop windows /
SALAD are part of each loop-closure run, so reported FPS/VRAM include them.

Run:  CUDA_VISIBLE_DEVICES=0 python -m ma_long.bench --out outputs/bench
"""

from __future__ import annotations
import argparse, copy, glob, json, os, time
import numpy as np
import torch

from model.ma_infer import MaChunkModel
from model.inputs import load_intrinsics
from ma_long.pipeline import MaLongPipeline, DEFAULT_CONFIG
from eval import ate_rmse

SCENES = {"s0011": "data/scene0011_00", "s0378": "data/scene0378_00"}
MODES = ["rgb", "rgb+intr", "rgb+depth", "rgb+depth+intr"]
OVERLAP = {20: 8, 30: 10, 40: 13, 60: 20}


def scene_data(path):
    imgs = sorted(glob.glob(f"{path}/rgb/*.png"))
    deps = [p.replace("/rgb/", "/depth/") for p in imgs]
    return imgs, deps, load_intrinsics(f"{path}/intrinsic.txt"), f"{path}/gt_pose.txt"


def mode_kwargs(mode, deps, K):
    kw = {}
    if "depth" in mode:
        kw["depth_paths"] = deps
    if "depth" in mode or "intr" in mode:
        kw["intrinsics"] = K
    return kw


def run_one(model, scene_path, out, mode, cs, loop):
    imgs, deps, K, gt = scene_data(scene_path)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["Model"].update(chunk_size=cs, overlap=OVERLAP[cs], loop_enable=loop)
    os.makedirs(out, exist_ok=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.time()
    res = MaLongPipeline(config=cfg, model=model).run(
        imgs, out, mode=mode, keep_cache=False, **mode_kwargs(mode, deps, K))
    torch.cuda.synchronize()
    dt = time.time() - t0
    a, a3 = ate_rmse(res["poses_txt"], gt, "sim3"), ate_rmse(res["poses_txt"], gt, "se3")
    return dict(mode=mode, cs=cs, loop=loop, n=len(imgs), sec=round(dt, 1),
                fps=round(len(imgs) / dt, 2), vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1),
                sim3=round(a["ate_rmse"], 4), se3=round(a3["ate_rmse"], 4), scale=round(a["scale"], 3))


def table(rows, cols, title):
    print(f"\n### {title}")
    head = " | ".join(f"{c:>9}" for c in cols)
    print(head); print("-" * len(head))
    for r in rows:
        print(" | ".join(f"{str(r.get(c,'')):>9}" for c in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/bench")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print("=== loading MapAnything once ===")
    model = MaChunkModel(device="cuda")
    results = {"A_modes": [], "B_no_lc": [], "C_chunk": []}

    # ---- Set A: cs=20, LC on, 4 input modes, both scenes ----
    for sk, sp in SCENES.items():
        for mode in MODES:
            print(f"\n### A {sk} {mode} cs20 LC")
            r = run_one(model, sp, f"{args.out}/A_{sk}_{mode.replace('+','_')}", mode, 20, True)
            r["scene"] = sk; results["A_modes"].append(r)

    # best input mode = lowest mean metric SE3-ATE across both scenes
    by_mode = {m: [] for m in MODES}
    for r in results["A_modes"]:
        by_mode[r["mode"]].append(r["se3"])
    best_mode = min(by_mode, key=lambda m: np.mean(by_mode[m]))
    print(f"\n=== best input mode (by mean SE3-ATE): {best_mode} ===")

    # ---- Set B: best mode, LC OFF, cs=20, both scenes ----
    for sk, sp in SCENES.items():
        print(f"\n### B {sk} {best_mode} cs20 NO-LC")
        r = run_one(model, sp, f"{args.out}/B_{sk}_noLC", best_mode, 20, False)
        r["scene"] = sk; results["B_no_lc"].append(r)

    # ---- Set C: best mode, LC on, cs sweep, both scenes ----
    for sk, sp in SCENES.items():
        for cs in [20, 30, 40, 60]:
            print(f"\n### C {sk} {best_mode} cs{cs} LC")
            r = run_one(model, sp, f"{args.out}/C_{sk}_cs{cs}", best_mode, cs, True)
            r["scene"] = sk; results["C_chunk"].append(r)

    results["best_mode"] = best_mode
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(results, f, indent=2)

    cols = ["scene", "mode", "cs", "loop", "sim3", "se3", "scale", "fps", "vram_gb"]
    print("\n\n================ SUMMARY ================")
    table(results["A_modes"], cols, "A) cs=20, LC on — input modes")
    table(results["B_no_lc"], cols, f"B) {best_mode}, cs=20 — LC ablation (compare to A)")
    table(results["C_chunk"], cols, f"C) {best_mode}, LC on — chunk-size sweep")
    print(f"\nsummary json -> {args.out}/summary.json")


if __name__ == "__main__":
    main()
