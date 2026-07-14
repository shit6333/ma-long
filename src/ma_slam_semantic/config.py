"""Configuration for ma_slam_semantic.

Two layers, mirroring the reference run
``outputs/slam_opense3r_query/scannet_gt8_fastsam/005_..._conf032_gt/005_run_args.json``:

- ``FROZEN_MODEL_ARGS`` — the architecture + weight args consumed by
  ``slam_opense3r.run.load_model``. These MUST match the checkpoint (it is loaded
  ``strict=False``, so a mismatch fails *silently*). Do not tune.
- ``SemConfig`` — the ~20 tunable pipeline knobs (proposal formation, instance
  matching, noise filter, KV policy), defaults = the reference run's values.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Tuple

OPENSE3R_ROOT = "/mnt/HDD4/ricky/OpenSe3r"


def ensure_opense3r(root: str = OPENSE3R_ROOT) -> None:
    """Make OpenSe3r importable (it appends its own thirdparty paths on import)."""
    if root not in sys.path:
        sys.path.append(root)


# ---------------------------------------------------------------------------
# Frozen model args (arch flags verbatim from 005_run_args.json, paths absolutized).
# backend='da3' + gt_geo_input=True builds OpenSe3rDA3 WITHOUT the DA3 geometry
# backbone — geometry always enters externally via forward_semantic, which is exactly
# the contract ma_slam feeds. dino_backbone='vggt' still loads VGGT for DINOv2 tokens.
# ---------------------------------------------------------------------------
FROZEN_MODEL_ARGS = dict(
    backend="da3",
    gt_geo_input=True,           # skip DA3 geometry backbone (external geometry)
    ckpt_path=os.path.join(OPENSE3R_ROOT, "outputs/opense3r_fastsam_query_semconf/checkpoint-last.pth"),
    vggt_ckpt=os.path.join(OPENSE3R_ROOT, "checkpoints/VGGT.pt"),
    da3_ckpt=os.path.join(OPENSE3R_ROOT, "checkpoints/da3_large"),          # unused (skipped)
    lbotv_ckpt=os.path.join(OPENSE3R_ROOT, "checkpoints/lingbot_vision_vitl"),  # unused (dino=vggt)
    lseg_ckpt=os.path.join(OPENSE3R_ROOT, "checkpoints/demo_e200.ckpt"),    # unused (tips backbone)
    # encoders / causal branch
    dino_backbone="vggt",
    semantic_backbone="tips-v2-l14",
    prior_method="native",
    sem_supervision="textregion",
    depth_norm_mode="chunk",
    gt_anchor_mode="se3",
    depth_cond=False,
    point_cond=False,
    pose_cond=False,
    rope_mode="2d",
    rope_voxel_size=0.1,
    no_head_sem_cond=True,
    use_sem_conf_head=True,
    sem_channels="0,768",
    ins_channels="512,1024",
    sem_dim=1024,
    ins_dim=16,
    n_blocks=12,
    pos3d_cond=True,
    no_depth_cross_attn=True,
    # query instance head (Mask2Former, FastSAM-seeded external queries)
    instance_head_type="query",
    query_arch="mask2former",
    query_init_type="external",
    num_instance_queries=0,
    instance_query_block_indices=None,
    instance_queries_init_type="learnable",
    query_num_queries=100,
    query_d_model=256,
    query_c_mask=256,
    query_descriptor_dim=256,
    query_n_heads=8,
    query_n_layers=9,
    query_num_feature_levels=3,
    query_descriptor_mlp_layers=3,
    query_use_box_head=False,
    query_box_mode="aabb",
    query_box_size_anchor=False,
    query_box_query_only=False,
    query_fastsam_mask_prior=False,
    query_box_size_extent_aware=False,
    use_merge_head=False,
    fastsam_query=True,
    fastsam_ckpt="/mnt/HDD4/ricky/sam_test/FastSAM-x.pt",
    fastsam_conf=0.2,
    fastsam_iou=0.9,
    fastsam_max_regions=100,
)


@dataclass
class SemConfig:
    """Tunable knobs (defaults = the 005 reference run)."""

    opense3r_root: str = OPENSE3R_ROOT
    device: str = "cuda"
    resolution: Tuple[int, int] = (518, 336)      # (W, H) — must match training
    attn_mode: str = "causal"
    chunk_size: int = 8                           # frames per causal batch (gt mode)

    # geometry conditioning scale: 'gt_seq_median' (golden gt mode, computed upfront)
    # or 'first_chunk' (online: capture from the first chunk's output, then freeze —
    # the streaming stand-in for a sequence-global scale).
    box_scale_mode: str = "first_chunk"

    # proposal formation (instance_post.process_chunk_to_proposals)
    obj_threshold: float = 0.5
    obj_top_k: int = 0
    nms_iou: float = 0.5
    score_threshold: float = 0.25
    nms_mode: str = "hard"
    matrix_nms_kernel: str = "linear"
    panoptic: bool = True
    sp_score_thr: float = 0.5
    outlier_iqr_mult: float = 1.0
    min_points: int = 0
    voxel_size: float = 0.05
    # keep_mask gate on geometry conf. GT route: conf is 0/1 validity so any value in
    # (0,1] behaves the same (reference used 0.32 on da3 conf). MA route: MapAnything
    # conf is NOT in (0,1) — the bridge substitutes a per-submap mean-conf gate.
    point_conf_threshold: float = 0.32

    # instance matching (score = w_desc*cos + w_vox*iou + w_geo*exp(-d/sigma), hungarian)
    knn_radius: float = 1.0
    max_candidates: int = 32
    match_threshold: float = 0.32
    w_desc: float = 0.3
    w_sem: float = 0.0
    w_vox: float = 0.3
    w_geo: float = 0.25
    sigma_dist: float = 0.25
    embed_update: str = "mean"
    embed_momentum: float = 0.75
    merge_desc_high: float = 0.75                 # final merge pass (export)

    # noise filter at export (subset of filter_noise_instances)
    filter_min_count: int = 1
    filter_end_grace: int = 10
    filter_max_extent_m: float = 4.0
    filter_max_size_m3: float = 5.0
    filter_oversized_max_count: int = 3
    filter_tiny_voxels: int = 0
    filter_tiny_max_count: int = 2
    filter_elongated_max_count: int = 15
    filter_min_avg_objectness: float = 0.0

    # KV policy: 'none' (frozen-config default: flat KV, no pruning) | 'window'
    kv_policy: str = "none"
    kv_window_size: Optional[int] = None
    kv_keyframe_stride: Optional[int] = None
    # KV reset valve (default OFF; log pose deltas regardless — measure first)
    kv_reset: bool = False
    kv_reset_delta: float = 0.5                   # max frame translation delta (m)

    # debug
    save_2d_masks: bool = False
    mask_2d_top_k: int = 15

    def model_args(self) -> Namespace:
        """The frozen args as the Namespace ``slam_opense3r.run.load_model`` expects."""
        return Namespace(**FROZEN_MODEL_ARGS)

    @property
    def patch_size(self) -> int:
        return 14                                  # DINOv2 ViT-L/14 (dino_backbone=vggt)

    @property
    def n_patches(self) -> int:
        w, h = self.resolution
        return (h // self.patch_size) * (w // self.patch_size)
