"""Configuration for ma_slam_semantic.

Two layers:

- **Checkpoint** (``CKPTS`` / ``SemConfig.ckpt``) — the model ARCHITECTURE is no
  longer configured here. ``sem_model`` builds it from the checkpoint's own stored
  args, corrected against the weights (``semantic_mapping.model``): the
  ``FROZEN_MODEL_ARGS`` table this file used to carry could only describe one
  model and was loaded ``strict=False`` (mismatches failed silently — see
  docs/opense3r_model_handoff.md §0/§5).
- ``SemConfig`` — the ~20 tunable pipeline knobs (proposal formation, instance
  matching, noise filter, KV policy), defaults = the reference run's values.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

# OpenSe3r code root resolution, in priority order:
#   1. $OPENSE3R_ROOT env var (e.g. point at the live repo to compare against it)
#   2. the self-contained vendored copy at <ma_long>/vendor (default — 帶著走)
# vendor/ mirrors the OpenSe3r root layout, so slam_opense3r/run.py's own
# file-relative sys.path appends (thirdparty/, STream3R/) keep working verbatim.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VENDOR_ROOT = os.path.join(_REPO_ROOT, "vendor")
OPENSE3R_ROOT = os.environ.get("OPENSE3R_ROOT", VENDOR_ROOT)

# frozen-backbone weight files per root: the vendored copies are renamed under
# vendor/weights/ (gitignored); the live repo keeps its original locations.
# (The checkpoint's own stored paths are relative to the OpenSe3r repo, so they
# must be overridden per root — see sem_model.build_model_from_ckpt.)
_WEIGHTS = {
    "vendor": dict(
        vggt_ckpt=os.path.join(VENDOR_ROOT, "weights/vggt_1b.pt"),
        fastsam_ckpt=os.path.join(VENDOR_ROOT, "weights/fastsam_x.pt"),
        # TIPS npz is found module-relatively (tips/pytorch/checkpoints/ holds
        # relative symlinks into vendor/weights/) — no path arg needed.
    ),
    "live": dict(
        vggt_ckpt="{root}/checkpoints/VGGT.pt",
        fastsam_ckpt="/mnt/HDD4/ricky/sam_test/FastSAM-x.pt",
    ),
}


def _weight_paths(root: str) -> dict:
    if os.path.abspath(root) == os.path.abspath(VENDOR_ROOT):
        return dict(_WEIGHTS["vendor"])
    return {k: v.format(root=root) for k, v in _WEIGHTS["live"].items()}


def ensure_opense3r(root: str = None) -> str:
    """Make OpenSe3r importable (it appends its own thirdparty paths on import)."""
    root = root or OPENSE3R_ROOT
    if root not in sys.path:
        sys.path.append(root)
    return root


# ---------------------------------------------------------------------------
# Checkpoint registry (docs/opense3r_model_handoff.md). Named variants resolve per
# root; ``SemConfig.ckpt`` also accepts a bare path.
#   semconf     ep44 naive semconf — the golden-gate reference (pre-P/Q)
#   semconf_pq  + P/Q ranking head on the SAME trunk (paper numbers; descfix ckpt:
#               its stored args say query_descriptor_mlp_layers=0 but the file
#               carries 3 layers -> arch must be inferred from the WEIGHTS)
#   a1          textregion seeding, dense semantic head REMOVED (semantic_feat is
#               None -> proposals pool frozen-TIPS textregion instead); training
#               until ~2026-08-28, P/Q head to be added afterwards
# ---------------------------------------------------------------------------
CKPTS = {
    "vendor": dict(
        semconf=os.path.join(VENDOR_ROOT, "weights/opense3r_fastsam_query_semconf.pth"),
        semconf_pq=os.path.join(VENDOR_ROOT, "weights/opense3r_semconf_pq_descfix.pth"),
        a1=os.path.join(VENDOR_ROOT, "weights/opense3r_a1.pth"),
    ),
    "live": dict(
        semconf="{root}/outputs/opense3r_fastsam_query_semconf/checkpoint-last.pth",
        semconf_pq="{root}/outputs/opense3r_pq_only_on_semconf/checkpoint-last-descfix.pth",
        a1="{root}/outputs/opense3r_fastsam_query_a1/checkpoint-last.pth",
    ),
}
DEFAULT_CKPT = "semconf_pq"


def resolve_ckpt(name_or_path: str, root: str = None) -> str:
    """Named variant (``CKPTS``) -> path for the active root; paths pass through."""
    root = root or OPENSE3R_ROOT
    if name_or_path and (os.path.sep in name_or_path or name_or_path.endswith(".pth")):
        return name_or_path
    table = (CKPTS["vendor"] if os.path.abspath(root) == os.path.abspath(VENDOR_ROOT)
             else {k: v.format(root=root) for k, v in CKPTS["live"].items()})
    if name_or_path not in table:
        raise KeyError(f"unknown checkpoint {name_or_path!r}; expected one of "
                       f"{sorted(table)} or a .pth path")
    return table[name_or_path]


@dataclass
class SemConfig:
    """Tunable knobs (defaults = the 005 reference run)."""

    opense3r_root: str = OPENSE3R_ROOT
    device: str = "cuda"
    ckpt: str = DEFAULT_CKPT                      # CKPTS name or a .pth path
    # cluster confidence at export (semantic_mapping.stages.ranking): 'p_top25q' =
    # meanP x mean(top-25% Q) — needs the P/Q head, falls back to objectness
    # automatically (and LOUDLY, see export) on checkpoints without it. 'obj' =
    # mean objectness (the pre-P/Q baseline).
    ranker: str = "p_top25q"
    # per-proposal semantic vector (sem_query): 'auto' = mask-mean of the dense
    # semantic head when the model runs one, else frozen-TIPS textregion pool over
    # the predicted masks (A1 removed the dense head). 'dense' / 'textregion' force.
    sem_agg: str = "auto"
    resolution: Tuple[int, int] = (518, 336)      # (W, H) — must match training
    attn_mode: str = "causal"
    # frames per causal batch. KV-causal => results are batching-invariant; this is
    # purely a VRAM knob (activations ~2.3 GB/frame: 8 -> ~19 GB peak, 4 -> ~10 GB).
    chunk_size: int = 4

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
    # conf is NOT in (0,1) — the bridge substitutes a per-submap mean-conf gate,
    # scaled by conf_gate_coef (1.0 = the ma_slam export default; LOWER keeps more
    # points => more GT coverage at eval, at the cost of noisier voxels).
    point_conf_threshold: float = 0.32
    conf_gate_coef: float = 1.0

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
    # | 'h2o' (STAC-style temporal compression — pinned frame-0 + recent window
    # + per-layer/per-head heavy-hitter tokens; bounded memory; plan S5).
    kv_policy: str = "none"
    kv_window_size: Optional[int] = None
    kv_keyframe_stride: Optional[int] = None
    # h2o knobs (kv_policy='h2o'); budgets are in frames / frame-equivalents.
    h2o_recent_frames: int = 8
    h2o_hh_frames: int = 8
    h2o_decay: float = 0.9                        # score decay γ (STAC 'temperature')
    h2o_subsample: float = 1.0                    # query subsample for colsum scores
    h2o_attn_backend: str = "triton"              # 'triton' | 'cuda' (attn-cuda ext)
    # KV reset valve (default OFF; log pose deltas regardless — measure first)
    kv_reset: bool = False
    kv_reset_delta: float = 0.5                   # max frame translation delta (m)

    # debug
    save_2d_masks: bool = False
    mask_2d_top_k: int = 15

    @property
    def ckpt_path(self) -> str:
        return resolve_ckpt(self.ckpt, self.opense3r_root)

    def backbone_paths(self) -> dict:
        """Frozen-backbone weight files for the active root (override the
        repo-relative paths stored inside the checkpoint)."""
        return _weight_paths(self.opense3r_root)

    @property
    def patch_size(self) -> int:
        return 14                                  # DINOv2 ViT-L/14 (dino_backbone=vggt)

    @property
    def n_patches(self) -> int:
        w, h = self.resolution
        return (h // self.patch_size) * (w // self.patch_size)
