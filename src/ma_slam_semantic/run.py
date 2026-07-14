"""CLI runner for ma_slam_semantic.

Golden-test mode (S2/S3 correctness harness — GT geometry, bypasses ma_slam; must
reproduce the reference OpenSe3r 005 GT run):

    PYTHONPATH=src python -m ma_slam_semantic.run --geometry gt \
        --data /mnt/HDD4/ricky/data/instance_eval_data/scenenn/oni/005 \
        --demo_type scenenn_gt --out outputs/ma_sem/005_gt --save_2d_masks

ma_slam-geometry mode (S4) plugs SemanticHooks into ``ma_slam.run`` instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ma_slam_semantic.config import SemConfig, ensure_opense3r
from ma_slam_semantic.bridge import gt_chunks, gt_seq_box_scale
from ma_slam_semantic.sem_model import SemanticModel


def _load_gt_batch(data_path: str, demo_type: str, res, max_frames: int = 0) -> dict:
    ensure_opense3r()
    if demo_type == "scenenn_gt":
        from slam_opense3r.datasets.scenenn import SceneNNSLAMGTDataset
        ds = SceneNNSLAMGTDataset(data_path, resolution=list(res))
    elif demo_type == "scannet_gt":
        from slam_opense3r.datasets.scannet import ScannetSLAMGTDataset
        ds = ScannetSLAMGTDataset(data_path, resolution=list(res))
    else:
        raise ValueError(f"unknown demo_type {demo_type!r}")
    batch = ds[0]
    if max_frames:
        batch = {k: (v[:max_frames] if torch.is_tensor(v) and v.dim() >= 1 else v)
                 for k, v in batch.items()}
    return batch


def main():
    ap = argparse.ArgumentParser(description="ma_slam_semantic runner")
    ap.add_argument("--geometry", default="gt", choices=["gt"],
                    help="'gt' = golden-test geometry (OpenSe3r GT dataset); "
                         "ma_slam mode is wired through ma_slam.run hooks (S4)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--demo_type", default="scenenn_gt", choices=["scenenn_gt", "scannet_gt"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None, help="export prefix (default: basename of --data)")
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--save_2d_masks", action="store_true",
                    help="dump per-frame top-K instance-mask grids (S2 eyeball gate)")
    ap.add_argument("--mask_top_k", type=int, default=15)
    a = ap.parse_args()

    cfg = SemConfig(chunk_size=a.chunk_size, box_scale_mode="gt_seq_median",
                    save_2d_masks=a.save_2d_masks, mask_2d_top_k=a.mask_top_k)
    name = a.name or os.path.basename(os.path.normpath(a.data))
    os.makedirs(a.out, exist_ok=True)

    batch = _load_gt_batch(a.data, a.demo_type, cfg.resolution, a.max_frames)
    T = batch["images"].shape[0]
    print(f"[ma_sem] {T} frames at {tuple(batch['images'].shape[-2:])} from {a.data}")

    model = SemanticModel(cfg)
    model.set_box_scale(gt_seq_box_scale(batch))

    t0 = time.time()
    n_done = 0
    for chunk in gt_chunks(batch, cfg.chunk_size):
        out = model.forward_chunk(chunk)
        n = chunk["images"].shape[1]
        if a.save_2d_masks:
            from slam_opense3r.vis_2d_masks import dump_frame_instance_masks
            dump_frame_instance_masks(
                images=chunk["images"].to(model.device), sem_res=out,
                frame_start=chunk["frame_indices"][0],
                out_dir=os.path.join(a.out, "2d_inst_masks"),
                demo_name=name, top_k=cfg.mask_2d_top_k)
        n_done += n
        obj = out["objectness"][0].float()
        print(f"[ma_sem] frames {n_done:4d}/{T}  "
              f"obj>0.5/frame={(obj.sigmoid() > 0.5).sum().item() / n:5.1f}  "
              f"{n_done / (time.time() - t0):.1f} fps")

    print(f"[ma_sem] done ({time.time() - t0:.1f}s). S3 adds proposals+instance table; "
          f"S4 adds export.")


if __name__ == "__main__":
    main()
