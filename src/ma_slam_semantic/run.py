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
from ma_slam_semantic.proposals import make_proposals
from ma_slam_semantic.instance_map import InstanceTable, StaticGeoResolver


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


def run_ma_slam(a, cfg: SemConfig) -> None:
    """ma_slam-geometry mode: full online SLAM + semantic hooks."""
    import copy
    import glob

    import numpy as np

    from model.inputs import load_intrinsics
    from ma_slam.solver import MaSlam, DEFAULT_CONFIG
    from ma_slam_semantic.hooks import SemanticHooks

    def _frames(d):
        fs = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG", "*.tiff", "*.TIFF"):
            fs += glob.glob(os.path.join(d, ext))
        return sorted(fs)

    name = a.name or os.path.basename(os.path.normpath(a.data))
    image_paths = _frames(os.path.join(a.data, "rgb"))
    if a.max_frames:
        image_paths = image_paths[: a.max_frames]
    need_depth = "depth" in a.mode
    depth_paths = None
    intrinsics = None
    depth_scale = 1000.0                     # uint16-mm convention (ScanNet PNG)
    if need_depth:
        depth_paths = _frames(os.path.join(a.data, "depth"))
        if not depth_paths:
            raise FileNotFoundError(f"no depth frames under {a.data}/depth")
        n = min(len(image_paths), len(depth_paths))
        if len(image_paths) != len(depth_paths):
            print(f"[ma_sem] color/depth count mismatch "
                  f"({len(image_paths)}/{len(depth_paths)}); truncating to {n}")
        image_paths, depth_paths = image_paths[:n], depth_paths[:n]
        # float depth files (e.g. .TIFF) are already metres -> scale 1.0
        from PIL import Image
        d0 = np.asarray(Image.open(depth_paths[0]))
        if np.issubdtype(d0.dtype, np.floating):
            depth_scale = 1.0
            print("[ma_sem] float depth detected -> depth_scale=1.0 (metres)")
    if ("intr" in a.mode) or need_depth:
        intrinsics = load_intrinsics(os.path.join(a.data, "intrinsic.txt"))

    slam_cfg = copy.deepcopy(DEFAULT_CONFIG)
    slam_cfg["submap_size"] = a.submap_size
    slam_cfg["Loop"]["enable"] = not a.no_loop

    backend = None
    if a.backend == "da3":
        if "depth" in a.mode:
            raise SystemExit("backend da3 supports rgb / rgb+intr only (no depth input)")
        from model.da3_infer import Da3ChunkModel
        backend = Da3ChunkModel(device="cuda")

    # GT trajectory: exports get SE3-Umeyama-aligned into the GT-mesh frame.
    gt_positions = None
    if a.gt:
        gt_positions = np.loadtxt(a.gt, comments="#").reshape(-1, 8)[:, 1:4]

    hooks = SemanticHooks(cfg, a.out, name=name, gt_positions=gt_positions)
    slam = MaSlam(config=slam_cfg, model=backend, hooks=hooks)
    res = slam.run(image_paths, a.out, mode=a.mode, depth_paths=depth_paths,
                   intrinsics=intrinsics, depth_scale=depth_scale,
                   depth_max=(a.depth_max if a.depth_max > 0 else None))

    if a.gt:
        from eval import ate_rmse
        s = ate_rmse(res["poses_txt"], a.gt, "sim3")
        se = ate_rmse(res["poses_txt"], a.gt, "se3")
        print(f"[ma_sem] Sim3-ATE={s['ate_rmse']:.4f} | SE3-ATE={se['ate_rmse']:.4f} | "
              f"loops={res['n_loops']}")


def main():
    ap = argparse.ArgumentParser(description="ma_slam_semantic runner")
    ap.add_argument("--geometry", default="gt", choices=["gt", "ma"],
                    help="'gt' = golden-test geometry (OpenSe3r GT dataset); "
                         "'ma' = full ma_slam online SLAM + semantic hooks")
    ap.add_argument("--data", required=True)
    ap.add_argument("--demo_type", default="scenenn_gt", choices=["scenenn_gt", "scannet_gt"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None, help="export prefix (default: basename of --data)")
    ap.add_argument("--chunk_size", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--save_2d_masks", action="store_true",
                    help="dump per-frame top-K instance-mask grids (S2 eyeball gate)")
    ap.add_argument("--mask_top_k", type=int, default=15)
    # ma_slam-geometry mode args
    ap.add_argument("--mode", default="rgb+depth+intr",
                    help="[ma] ma_slam input mode (rgb / rgb+intr / rgb+depth / rgb+depth+intr)")
    ap.add_argument("--backend", default="ma", choices=["ma", "da3"],
                    help="[ma] reconstruction backbone: MapAnything | DA3 (rgb-only, metric)")
    ap.add_argument("--submap_size", type=int, default=20, help="[ma] frames per submap")
    ap.add_argument("--no_loop", action="store_true", help="[ma] disable loop closure")
    ap.add_argument("--depth_max", type=float, default=0.0, help="[ma] clamp far depth (m)")
    ap.add_argument("--gt", default=None, help="[ma] GT trajectory for ATE report")
    ap.add_argument("--conf_gate_coef", type=float, default=1.0,
                    help="[ma] scale on the per-submap mean-conf validity gate "
                         "(<1 keeps more points -> more coverage, noisier voxels)")
    ap.add_argument("--kv_policy", default="none", choices=["none", "window", "h2o"],
                    help="CausalBranch KV retention: 'none' (flat, unbounded), "
                         "'window' (flat trim), 'h2o' (STAC temporal compression: "
                         "pinned frame-0 + recent window + heavy-hitter tokens)")
    ap.add_argument("--kv_window_size", type=int, default=None,
                    help="[window] sliding-window frames")
    ap.add_argument("--kv_keyframe_stride", type=int, default=None,
                    help="[window] keep every S-th past frame")
    ap.add_argument("--h2o_recent", type=int, default=8,
                    help="[h2o] recent-window budget (frames)")
    ap.add_argument("--h2o_hh", type=int, default=8,
                    help="[h2o] heavy-hitter budget (frame-equivalents)")
    ap.add_argument("--h2o_decay", type=float, default=0.9,
                    help="[h2o] attention-score decay γ")
    ap.add_argument("--h2o_subsample", type=float, default=1.0,
                    help="[h2o] query subsample ratio for colsum scores")
    a = ap.parse_args()

    kv_kwargs = dict(kv_policy=a.kv_policy,
                     kv_window_size=a.kv_window_size,
                     kv_keyframe_stride=a.kv_keyframe_stride,
                     h2o_recent_frames=a.h2o_recent, h2o_hh_frames=a.h2o_hh,
                     h2o_decay=a.h2o_decay, h2o_subsample=a.h2o_subsample)

    if a.geometry == "ma":
        cfg = SemConfig(box_scale_mode="first_chunk", conf_gate_coef=a.conf_gate_coef,
                        **kv_kwargs)
        run_ma_slam(a, cfg)
        return

    cfg = SemConfig(chunk_size=a.chunk_size, box_scale_mode="gt_seq_median",
                    save_2d_masks=a.save_2d_masks, mask_2d_top_k=a.mask_top_k,
                    **kv_kwargs)
    name = a.name or os.path.basename(os.path.normpath(a.data))
    os.makedirs(a.out, exist_ok=True)

    batch = _load_gt_batch(a.data, a.demo_type, cfg.resolution, a.max_frames)
    T = batch["images"].shape[0]
    print(f"[ma_sem] {T} frames at {tuple(batch['images'].shape[-2:])} from {a.data}")

    model = SemanticModel(cfg)
    model.set_box_scale(gt_seq_box_scale(batch))
    resolver = StaticGeoResolver()
    table = InstanceTable(cfg, resolver)

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
        props, pix_map, pts_hd = make_proposals(out, chunk, cfg)
        for t, key in enumerate(chunk["frame_keys"]):
            resolver.register(key, pts_hd[t])
        table.match_chunk(props, chunk["frame_keys"], pix_map)
        n_done += n
        print(f"[ma_sem] frames {n_done:4d}/{T}  proposals={len(props):3d}  "
              f"instances={len(table)}  {n_done / (time.time() - t0):.1f} fps")
        if hasattr(model.policy, "stats"):
            info = model.policy.stats()
            if info is not None:
                print(f"[ma_sem]   h2o-kv: tokens/layer={info['kvcache_size'][0]}"
                      f"  used={info['kvcache_used_mb']:.0f}MB"
                      f"  prune={model.policy.prune_ms:.1f}ms")

    print(f"[ma_sem] streaming done ({time.time() - t0:.1f}s): {len(table)} instances")
    from ma_slam_semantic.export import export_all
    res = export_all(table, num_frames=T, out_dir=a.out, name=name, cfg=cfg)
    print(f"[ma_sem] exported {res['n_instances']} instances "
          f"({res['n_dropped']} noise-filtered) -> {a.out}")


if __name__ == "__main__":
    main()
