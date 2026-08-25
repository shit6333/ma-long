"""Profiling harness for the semantic side — per-block time + VRAM.

Wraps the OpenSe3r model's submodule entry points with sync'd timers +
per-block peak-VRAM capture (blocks are sequential inside forward_semantic,
so per-block reset_peak is sound), runs the golden 005 path, and prints:

  - weights VRAM per submodule (params+buffers)
  - per-block time (ms/frame) + share of the forward, activation peak, alloc delta
  - CPU post-processing: make_proposals / table.match_chunk / bridge
  - KV-cache bytes growth per chunk

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 python -m ma_slam_semantic.profile_sem \
        --data /mnt/HDD4/ricky/data/instance_eval_data/scenenn/oni/005 --max_frames 64
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import torch

from ma_slam_semantic.config import SemConfig
from ma_slam_semantic.bridge import gt_chunks, gt_seq_box_scale
from ma_slam_semantic.sem_model import SemanticModel
from ma_slam_semantic.proposals import make_proposals
from ma_slam_semantic.instance_map import InstanceTable, StaticGeoResolver
from ma_slam_semantic.run import _load_gt_batch

STATS = defaultdict(list)          # key -> list of (seconds, peak_bytes, alloc_delta)


def _wrap(obj, attr: str, key: str) -> bool:
    fn = getattr(obj, attr, None)
    if fn is None:
        return False

    def wrapped(*a, **k):
        torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        STATS[key].append((time.perf_counter() - t0,
                           torch.cuda.max_memory_allocated(),
                           torch.cuda.memory_allocated() - m0))
        return out

    setattr(obj, attr, wrapped)
    return True


def _module_bytes(m) -> int:
    if m is None or not hasattr(m, "parameters"):
        return 0
    return (sum(p.numel() * p.element_size() for p in m.parameters())
            + sum(b.numel() * b.element_size() for b in m.buffers()))


def _tensor_bytes(x) -> int:
    if torch.is_tensor(x):
        return x.numel() * x.element_size()
    if isinstance(x, (list, tuple)):
        return sum(_tensor_bytes(e) for e in x)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/HDD4/ricky/data/instance_eval_data/scenenn/oni/005")
    ap.add_argument("--demo_type", default="scenenn_gt")
    ap.add_argument("--max_frames", type=int, default=64)
    ap.add_argument("--chunk_size", type=int, default=8)
    a = ap.parse_args()

    cfg = SemConfig(chunk_size=a.chunk_size, box_scale_mode="gt_seq_median")
    batch = _load_gt_batch(a.data, a.demo_type, cfg.resolution, a.max_frames)
    T = batch["images"].shape[0]

    sem = SemanticModel(cfg)
    sem.set_box_scale(gt_seq_box_scale(batch))
    m = sem.model

    # ---------------- weights VRAM per submodule
    fastsam_inner = getattr(getattr(m, "_fastsam_seg", None), "model", None)
    weights = {
        "vggt (DINOv2 tokens)": _module_bytes(getattr(m, "vggt", None)),
        "semantic_backbone (TIPSv2-L14)": _module_bytes(getattr(m, "semantic_backbone", None)),
        "causal_branch (12 blk)": _module_bytes(getattr(m, "causal_branch", None)),
        "semantic_head (DPT)": _module_bytes(getattr(m, "semantic_head", None)),
        "instance_head (M2F)": _module_bytes(getattr(m, "instance_head", None)),
        "depth_encoder": _module_bytes(getattr(m, "depth_encoder", None)),
        "pos3d_encoder": _module_bytes(getattr(m, "pos3d_encoder", None)),
        "region_pool": _module_bytes(getattr(m, "region_pool", None)),
        "FastSAM-x": _module_bytes(getattr(fastsam_inner, "model", fastsam_inner)),
    }

    # ---------------- wrap sequential blocks inside forward_semantic
    wrapped = []
    for obj, attr, key in [
        (getattr(m, "_fastsam_seg", None), "segment", "FastSAM.segment"),
        (getattr(getattr(m, "vggt", None), "aggregator", None), "encode_patch_tokens", "DINO patch tokens"),
        (getattr(m, "semantic_backbone", None), "forward_prior_native", "TIPS prior (native)"),
        (getattr(m, "causal_branch", None), "forward", "causal_branch"),
        (getattr(m, "semantic_head", None), "forward", "semantic_head (DPT)"),
        (getattr(m, "instance_head", None), "forward", "instance_head (M2F)"),
        (getattr(m, "depth_encoder", None), "forward", "depth_encoder"),
        (getattr(m, "pos3d_encoder", None), "forward", "pos3d_encoder"),
        (getattr(m, "region_pool", None), "forward", "region_pool"),
    ]:
        if obj is not None and _wrap(obj, attr, key):
            wrapped.append(key)
    print(f"[profile] wrapped blocks: {wrapped}")

    resolver = StaticGeoResolver()
    table = InstanceTable(cfg, resolver)
    kv_bytes = []
    baseline = torch.cuda.memory_allocated()

    n_done = 0
    t_start = time.perf_counter()
    for chunk in gt_chunks(batch, cfg.chunk_size):
        n = chunk["images"].shape[1]

        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = sem.forward_chunk(chunk)
        torch.cuda.synchronize()
        STATS["TOTAL forward_chunk"].append((time.perf_counter() - t0, 0, 0))

        t0 = time.perf_counter()
        props, pix_map, pts_hd = make_proposals(out, chunk, cfg,
                                                sem_backbone=sem.semantic_backbone)
        STATS["make_proposals (CPU)"].append((time.perf_counter() - t0, 0, 0))

        for t_i, key in enumerate(chunk["frame_keys"]):
            resolver.register(key, pts_hd[t_i])

        t0 = time.perf_counter()
        table.match_chunk(props, chunk["frame_keys"], pix_map)
        STATS["table.match_chunk (CPU)"].append((time.perf_counter() - t0, 0, 0))

        kv_bytes.append(_tensor_bytes(sem.kv))
        n_done += n
    wall = time.perf_counter() - t_start

    # ---------------- report
    G = 1024 ** 3
    print("\n" + "=" * 88)
    print(f"PROFILE  {a.data}  frames={n_done}  chunk={a.chunk_size}  "
          f"wall={wall:.1f}s = {n_done / wall:.2f} fps  instances={len(table)}")
    print("=" * 88)
    print("\n-- weights VRAM --")
    for k, v in sorted(weights.items(), key=lambda kv: -kv[1]):
        if v:
            print(f"  {k:34s} {v / G:7.2f} GB")
    print(f"  {'SUM':34s} {sum(weights.values()) / G:7.2f} GB   "
          f"(allocated after load+1st scale pass: {baseline / G:.2f} GB)")

    fwd = sum(t for t, _, _ in STATS["TOTAL forward_chunk"])
    print(f"\n-- per-block time (over {n_done} frames) --")
    print(f"  {'block':34s} {'ms/frame':>9s} {'% of fwd':>9s} {'peak GB':>8s} {'dAlloc MB':>10s}")
    order = sorted(STATS.items(), key=lambda kv: -sum(t for t, _, _ in kv[1]))
    for k, rows in order:
        tt = sum(t for t, _, _ in rows)
        pk = max((p for _, p, _ in rows), default=0)
        da = np.mean([d for _, _, d in rows]) if rows else 0
        share = f"{100 * tt / fwd:8.1f}%" if ("CPU" not in k and k != "TOTAL forward_chunk") else "        -"
        print(f"  {k:34s} {1000 * tt / n_done:9.2f} {share} {pk / G:8.2f} {da / 1024**2:10.1f}")
    other = fwd - sum(sum(t for t, _, _ in rows) for k, rows in STATS.items()
                      if k not in ("TOTAL forward_chunk", "make_proposals (CPU)", "table.match_chunk (CPU)"))
    print(f"  {'(forward other/glue)':34s} {1000 * other / n_done:9.2f} {100 * other / fwd:8.1f}%")

    print("\n-- KV cache growth --")
    for i, b in enumerate(kv_bytes):
        print(f"  after chunk {i:2d} ({(i + 1) * a.chunk_size:3d} frames): {b / G:6.3f} GB"
              + (f"   (+{(b - kv_bytes[i - 1]) / 1024**2:.0f} MB/chunk)" if i else ""))
    if len(kv_bytes) > 1:
        per_frame = (kv_bytes[-1] - kv_bytes[0]) / (n_done - a.chunk_size)
        print(f"  growth ≈ {per_frame / 1024**2:.1f} MB/frame "
              f"-> {per_frame * 1000 / G:.1f} GB @ 1000 frames")


if __name__ == "__main__":
    main()
