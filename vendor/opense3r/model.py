import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map_torch

from .causal_branch import CausalBranch
from .depth_encoder import DepthEncoder
from .dpt_heads import FeatureDPTHead
from .cond_encoders import (
    PoseTokenEncoder, PluckerEncoder,
    make_relative_pose, pose_to_features, compute_plucker_rays,
)
from .semantic_backbones import build_semantic_backbone


def trim_vggt_modules(vggt, keep_geometry: bool):
    """Free VGGT submodules not used at runtime, AFTER CausalBranch
    init_from_vggt has already consumed the aggregator's attention blocks.

    Call this once at the end of __init__ (before ``.to(device)``) so the
    dropped weights never occupy GPU memory.

    Args:
        keep_geometry:
            - True  (VGGT backend): geometry uses ``aggregator`` (full) +
              ``depth_head`` + ``point_head`` + ``camera_head``.  Only
              ``track_head`` is unused → that's all we can drop.
            - False (DA3 backend): geometry comes from DA3.  Runtime only calls
              ``aggregator.encode_patch_tokens`` (per-frame DINOv2 patch tokens),
              which needs just ``aggregator.patch_embed`` (+ its norm buffers).
              Drop the alternating-attention trunk (frame/global blocks) and ALL
              geometry heads — this is the bulk of VGGT's parameters.

    Returns the list of freed submodule names (for logging).
    """
    freed = []
    # track_head is never used at runtime in either backend.
    if getattr(vggt, 'track_head', None) is not None:
        vggt.track_head = None
        freed.append('track_head')

    if not keep_geometry:
        for head in ('depth_head', 'point_head', 'camera_head'):
            if getattr(vggt, head, None) is not None:
                setattr(vggt, head, None)
                freed.append(head)
        agg = getattr(vggt, 'aggregator', None)
        if agg is not None:
            # encode_patch_tokens only touches patch_embed + _resnet_mean/std.
            if getattr(agg, 'frame_blocks', None) is not None and len(agg.frame_blocks) > 0:
                agg.frame_blocks = nn.ModuleList()
                freed.append('aggregator.frame_blocks')
            if getattr(agg, 'global_blocks', None) is not None and len(agg.global_blocks) > 0:
                agg.global_blocks = nn.ModuleList()
                freed.append('aggregator.global_blocks')
    return freed


def fill_invalid_world_points(world_points, valid):
    """Replace invalid pixels' world points with the per-frame valid mean.

    Shared by GT-geometry training (`build_gt_geometry_from_frames`) and the
    GT-geo-input SLAM pipeline so both preprocess pos3d identically. Filling
    with the frame mean (not 0) keeps the pos3d patch avg-pool from being
    dragged toward a spurious origin by depth holes.

    Args:
        world_points: [..., H, W, 3]  (leading dims are batch/time)
        valid:        [..., H, W]     bool mask
    Returns: [..., H, W, 3] with invalid pixels filled.
    """
    lead = world_points.shape[:-3]
    H, W = world_points.shape[-3], world_points.shape[-2]
    v   = valid.reshape(*lead, H * W, 1).to(world_points.dtype)
    wpf = world_points.reshape(*lead, H * W, 3)
    mean = (wpf * v).sum(dim=-2, keepdim=True) / v.sum(dim=-2, keepdim=True).clamp(min=1.0)
    out = torch.where(v.bool(), wpf, mean)
    return out.reshape(*lead, H, W, 3)


def valid_depth_box_scale(depth, valid):
    """Per-frame valid-depth median, then median over frames → box_scale.

    Holes are excluded via nanmedian. Shared by GT-geometry training and the
    GT-geo-input SLAM pipeline so the normalization scale matches.

    Args:
        depth: [..., T, H, W]   valid: [..., T, H, W] bool
    Returns: [...]  (the per-batch box_scale; clamped ≥ 0.1, NaN→1.0)
    """
    lead = depth.shape[:-3]
    T, H, W = depth.shape[-3], depth.shape[-2], depth.shape[-1]
    d = depth.clone()
    d[~valid] = float('nan')
    per_frame = torch.nanmedian(d.reshape(*lead, T, H * W), dim=-1).values   # [..., T]
    box = torch.nanmedian(per_frame, dim=-1).values                          # [...]
    return torch.nan_to_num(box, nan=1.0).clamp(min=0.1)


def build_gt_geometry_from_frames(frames, anchor_mode='se3'):
    """Assemble chunk-local GT geometry from a training batch — for GT-geometry
    training (``geo_source='gt'``), which skips the frozen geo backbone entirely.

    Mirrors the chunk-local convention the geo backbone produces at inference:
    frame 0 is the reference (``c2w[0] = I``), so the returned poses and
    world_points are anchored to frame 0's camera frame.  This keeps the
    CausalBranch's geometry conditioning (pos3d / pose / plücker tokens) in the
    SAME coordinate convention whether geometry comes from GT (train / eval) or
    from DA3 (in-the-wild inference).

    Reads (all already batched to ``[B, T, ...]`` by the dust3r collate):
        frames['depthmap']          [B,T,H,W]      GT camera-frame depth (Z, metric)
        frames['camera_pose']       [B,T,4,4]      GT c2w (cam→world)
        frames['camera_intrinsics'] [B,T,3,3]      GT K
        frames['pts3d']             [B,T,H,W,3]    GT world points (dataset frame)
        frames['valid_mask']        [B,T,H,W]      valid-depth mask (GT has holes)

    Returns:
        depth      [B,T,H,W]    camera-frame Z (frame-invariant; holes kept as 0)
        c2w_rel    [B,T,4,4]    frame-0-anchored c2w (c2w_rel[:,0] = I)
        intrinsics [B,T,3,3]
        world_pts  [B,T,H,W,3]  in frame-0 camera frame; invalid pixels filled
                                with the per-frame valid mean (neutral for the
                                avg-pool that builds pos3d patch tokens)
        conf       [B,T,H,W,1]  valid mask as a [0,1] confidence
        box_scale  [B]          median of VALID depth (holes excluded), the
                                shared scale used to normalize depth/pos3d/pose
    """
    depth = frames['depthmap'].float()              # [B,T,H,W]
    c2w   = frames['camera_pose'].float()           # [B,T,4,4]  dataset-world c2w
    K     = frames['camera_intrinsics'].float()     # [B,T,3,3]
    wp    = frames['pts3d'].float()                 # [B,T,H,W,3] dataset-world
    valid = frames.get('valid_mask', None)
    valid = (depth > 0) if valid is None else valid.bool()
    B, T, H, W = depth.shape

    # Re-anchor to frame 0.
    #   'se3' (default): full SE3 into frame-0 CAMERA frame (c2w[0]=I) — matches
    #                    the geo backbone's chunk-local convention.
    #   'translate'    : keep dataset-world (GRAVITY-aligned) rotation; only move
    #                    the origin to frame-0's camera position.  World points
    #                    (and thus amodal box GT) stay axis-aligned to gravity →
    #                    tight, learnable AABBs (vs the loose, tilted boxes 'se3'
    #                    produces because it rotates the scene into a pitched
    #                    camera frame).  Requires GT poses at inference too (use_gt).
    if anchor_mode == 'translate':
        t0 = -c2w[:, 0, :3, 3]                               # [B,3]  (-cam0 world pos)
        wp = wp + t0.view(B, 1, 1, 1, 3)
        c2w_rel = c2w.clone()
        c2w_rel[:, :, :3, 3] = c2w[:, :, :3, 3] + t0.view(B, 1, 3)
    else:
        w2c0 = torch.linalg.inv(c2w[:, 0])                   # [B,4,4]
        R0, t0 = w2c0[:, :3, :3], w2c0[:, :3, 3]             # [B,3,3], [B,3]
        wp = torch.einsum('bij,bthwj->bthwi', R0, wp) + t0.view(B, 1, 1, 1, 3)
        c2w_rel = torch.einsum('bij,btjk->btik', w2c0, c2w)  # [B,T,4,4]

    # Fill invalid world_points with the per-frame valid mean so the pos3d
    # patch avg-pool isn't dragged toward a spurious origin by depth holes.
    wp = fill_invalid_world_points(wp, valid)

    # box_scale = per-frame valid-depth median, then median over T (holes
    # excluded). Shared with the SLAM GT-geo-input path for train/eval parity.
    box_scale = valid_depth_box_scale(depth, valid)                      # [B]

    conf = valid.float().unsqueeze(-1)                       # [B,T,H,W,1]
    return depth, c2w_rel, K, wp, conf, box_scale


class OpenSe3r(nn.Module):
    """
    Online Causal Semantic & Instance PCD-SLAM model.

    Three frozen branches feed a trainable CausalBranch:
      - VGGT (DINOv2 patch_embed): per-frame tokens, no cross-frame attention → online-safe
      - VGGT (aggregator + heads):  chunk-level depth + world_points (geometry only)
      - LSeg: per-frame dense semantic features (also used as pseudo-GT)

    Training:  full T-frame forward with causal mask (one pass, no BPTT).
    Inference: chunk-based with KV-cache preserved across chunks.
    Output:    per-pixel (3D point, semantic feature, instance feature).
    """

    def __init__(
        self,
        vggt_ckpt: str = './checkpoints/VGGT.pt',
        omega_ckpt: str = './checkpoints/vggt_omega/vggt_omega_1b_512.pt',
        lseg_ckpt: str = './checkpoints/lseg_minimal_e200.ckpt',
        # Semantic head
        sem_dim: int = 64,           # compact output dimension (< 512)
        sem_expand: Optional[int] = None,  # expanded dim; None → backbone.output_dim
        # Instance head
        ins_dim: int = 16,
        # Causal branch
        n_blocks: int = 12,
        depth_cross_every: int = 3,
        init_from_vggt: bool = True,
        patch_size: int = 14,
        # DINO patch-token source for the CausalBranch input:
        #   'vggt'  (default) — VGGT's DINOv2 encode_patch_tokens (patch_size=14).
        #                       Legacy / backward-compatible.
        #   'omega' — VGGT-Ω's DINOv3 patch_embed (patch_size=16; ~24% fewer
        #             tokens per frame → cheaper CausalBranch + DPT heads).
        #             VGGT geometry path is UNCHANGED (still DINOv2-based);
        #             only the CausalBranch input source changes.
        #   When 'omega', `patch_size` is auto-overridden to 16.
        dino_backbone: str = 'vggt',
        # Depth normalization mode for the DepthEncoder input.
        # 'chunk': single divisor for all T frames in a batch element (= box_scale).
        #          Cross-view consistent.  Default for new training.
        # 'frame': per-frame median divisor.  Legacy behaviour, kept for
        #          ckpt back-compat — same physical surface gets different
        #          normalized depths across frames.
        depth_norm_mode: str = 'chunk',
        # gt_anchor_mode (geo_source='gt'): 'se3' (full SE3 → tilted frame-0 cam
        # frame) or 'translate' (keep gravity-aligned world rotation, shift origin
        # to cam0 → tight axis-aligned amodal boxes).
        gt_anchor_mode: str = 'se3',
        # Optional conditioning streams (default off).  Each can be toggled
        # independently; combinations are valid.  See opense3r/cond_encoders.py.
        depth_cond:   bool = False,    # depth_tokens added at input (residual)
        point_cond:   bool = False,    # feed normalized world points (xyz) via PointEncoder → depth-cross-attn path
        rope_mode:        str   = '2d',   # '2d' | '3d_full' | '3d_cross'
        rope_voxel_size:  float = 0.1,
        pose_cond:    bool = False,    # per-frame pose token prepended at input
        plucker_cond: bool = False,    # per-patch Plücker ray tokens (residual)
        # 3D PE (Idea 1) + middle-depth gate + query init mode
        pos3d_cond:                  bool = False,
        use_depth_cross_attn:        bool = True,
        query_init_type:             str  = 'learnable',
        # Instance queries in CausalBranch (Idea 2)
        num_instance_queries:        int  = 0,
        instance_query_block_indices: Optional[List[int]] = None,
        instance_queries_init_type:  str  = 'learnable',
        # VGGT-Ω register attention (CausalBranch).  Default 0 = off (legacy).
        # Set num_register_tokens > 0 to enable; default block indices = 3
        # evenly-spaced blocks (25% of 12), tweak via register_block_indices.
        num_register_tokens:    int = 0,
        register_block_indices: Optional[List[int]] = None,
        plucker_num_freqs: int = 4,    # Fourier frequencies for PluckerEncoder
        # Semantic backbone (default = legacy LSeg behaviour).
        semantic_backbone: str = 'lseg',     # {'lseg', 'pe-core-b16-224', 'pe-core-l14-336', ...}
        prior_method:      str = 'lseg',     # {'lseg', 'proxy_clip'}
        sem_supervision:   str = 'lseg',     # {'lseg', 'textregion', 'mask_crops', 'patch_mean'}
        proxy_beta:    float = 1.2,          # ProxyCLIP adaptive-norm centring
        proxy_gamma:   float = 3.0,          # ProxyCLIP adaptive-norm scaling
        multi_res_crops: bool = False,       # Enable TextRegion multi-res tiling
        mr_grid_h: int = None,               # Override tile count (H direction); None = auto: max(H//S, 1)
        mr_grid_w: int = None,               # Override tile count (W direction); None = auto: max(W//S, 1)
        mask_crop_size: int = 224,           # Method 2 crop input size
        mask_crop_margin: float = 0.1,       # Method 2 bbox margin (fraction)
        mask_crop_mode: str = 'bbox',        # {'bbox', 'masked'}
        use_head_sem_cond: bool = True,      # Inject prior as DPT-layer cond in semantic head.
                                             # False = no-shortcut variant (head has no cond at all).
        head_pixel_shuffle: bool = False,    # FeatureDPTHead lightweight path (VGGT-Ω-style):
                                             # skip RefineNet cascade → MLP + pixel_shuffle.
                                             # ~30% training-memory saving; quality competitive.
        # Channel split for sem/ins heads (default = full 0..1024, backward-compatible).
        # Use e.g. sem_channels=(0,768), ins_channels=(512,1024) to force the two heads
        # to operate on different (partially overlapping) channel sub-spaces of the
        # CausalBranch token — encourages semantic and instance to decompose into
        # task-specific representations rather than competing for the same channels.
        sem_channels: Tuple[int, int] = (0, 1024),
        ins_channels: Tuple[int, int] = (0, 1024),
        # Instance head type — keep both implementations available.  Default
        # 'dense' = legacy FeatureDPTHead → instance_feat (no behaviour change
        # for existing checkpoints).  'query' = QueryInstanceHead → mask_logits +
        # objectness + descriptor (MaskFormer-style, class-agnostic).
        instance_head_type: str = 'dense',
        # Query-head specific (only used when instance_head_type='query').
        # `query_arch` selects between the two query-head implementations:
        #   'maskformer'  = MaskFormer-style decoder (legacy, single-scale,
        #                    separate descriptor head, no masked attn)
        #   'mask2former' = Mask2Former-style decoder (multi-scale, masked
        #                    attention, unified mask_embed-as-descriptor,
        #                    deep supervision aux outputs)
        query_arch:          str = 'maskformer',
        query_num_queries:   int = 100,
        query_d_model:       int = 256,
        query_descriptor_dim: int = 256,
        query_c_mask:        int = 256,
        query_n_heads:       int = 8,
        query_n_layers:      int = 6,
        # Mask2Former-specific extras (ignored for 'maskformer'):
        query_num_feature_levels: int = 3,
        query_return_aux:    bool = True,
        query_use_box_head:  bool = False,    # Optional 3D bbox prediction head
        query_box_mode:      str  = 'aabb',   # 'aabb'|'obb'|'world_aabb'|'world_amodal'|'world_amodal_grow'
        query_box_size_anchor: bool = False,  # world_amodal*: size=visible-extent×exp(Δ)
        query_box_query_only:  bool = False,  # box MLP input = q_norm only (no mask-pool)
        query_box_size_extent_aware: bool = False,  # feed log(extent) → extent-aware size mult
        query_use_fgbg_head:   bool = False,  # per-query thing/stuff (fg/bg) logit
        query_fastsam_mask_prior: bool = False,  # ESAM-style: layer-0 attn_mask = FastSAM region
        use_merge_head:      bool = False,
        query_descriptor_mlp_layers: int = 0,
        query_box_head_detach: bool = False,
        # Dual-path fused descriptor head (memory + appearance); see
        # Mask2FormerQueryInstanceHead.  Requires query_init_type='external'.
        query_desc_dual_path:  bool = False,
        query_desc_fuse_layers: int = 3,
        query_desc_pathB_aux:  bool = False,
        # ── FastSAM-region-seeded queries (ESAM-style instance head) ──────────
        # When True, instance queries are NOT a learnable set — every query is
        # seeded from one FastSAM region via GeoAwareRegionPooling (causal +
        # detached-semantic + geo-aware-pooled shape).  Requires
        # instance_head_type='query', query_arch='mask2former',
        # query_init_type='external'.  Mutually exclusive with the CausalBranch
        # instance queries (num_instance_queries>0).  Default off → fully
        # backward compatible (no new state_dict keys, no behaviour change).
        use_sem_conf_head:   bool = False,   # A': semantic-based mask-quality head
        fastsam_query:       bool = False,
        fastsam_ckpt:        str  = "/mnt/HDD4/ricky/sam_test/FastSAM-x.pt",
        fastsam_conf:        float = 0.2,
        fastsam_iou:         float = 0.9,
        fastsam_max_regions: int  = 100,
    ):
        super().__init__()
        if dino_backbone not in ('vggt', 'omega'):
            raise ValueError(f"dino_backbone must be 'vggt' or 'omega', got {dino_backbone!r}")
        self.dino_backbone = dino_backbone
        # omega backbone forces patch_size=16 (DINOv3 ViT-L default).
        if dino_backbone == 'omega' and patch_size != 16:
            print(f"[OpenSe3r] dino_backbone='omega' — overriding patch_size {patch_size} → 16")
            patch_size = 16
        self.patch_size = patch_size
        if depth_norm_mode not in ('chunk', 'frame', 'none'):
            raise ValueError(f"depth_norm_mode must be 'chunk' or 'frame', got {depth_norm_mode!r}")
        self.depth_norm_mode = depth_norm_mode
        if gt_anchor_mode not in ('se3', 'translate'):
            raise ValueError(f"gt_anchor_mode must be 'se3' or 'translate', got {gt_anchor_mode!r}")
        self.gt_anchor_mode = gt_anchor_mode

        # Conditioning flags (also stored on causal_branch for forward routing).
        self.depth_cond   = depth_cond
        self.pose_cond    = pose_cond
        self.plucker_cond = plucker_cond

        # Semantic backbone routing flags (used in forward).
        self.semantic_backbone_name = semantic_backbone
        self.prior_method     = prior_method
        self.sem_supervision  = sem_supervision
        self.multi_res_crops  = multi_res_crops
        self.mr_grid_h        = mr_grid_h
        self.mr_grid_w        = mr_grid_w
        self.mask_crop_size   = mask_crop_size
        self.mask_crop_margin = mask_crop_margin
        self.mask_crop_mode   = mask_crop_mode
        self.use_head_sem_cond = use_head_sem_cond

        # Channel split validation + storage.
        causal_dim = 1024   # CausalBranch token dim (DINOv2 ViT-L embed_dim)
        for name, ch in (('sem_channels', sem_channels), ('ins_channels', ins_channels)):
            s, e = ch
            if not (0 <= s < e <= causal_dim):
                raise ValueError(
                    f"{name}={ch} invalid: require 0 <= start < end <= {causal_dim}"
                )
        self.sem_channels = tuple(sem_channels)
        self.ins_channels = tuple(ins_channels)
        sem_causal_dim = sem_channels[1] - sem_channels[0]
        ins_causal_dim = ins_channels[1] - ins_channels[0]

        # Instance head type — store for forward dispatch.
        if instance_head_type not in ('dense', 'query'):
            raise ValueError(
                f"instance_head_type must be 'dense' or 'query', got '{instance_head_type}'"
            )
        self.instance_head_type = instance_head_type

        # Sanity: legal flag combinations.
        if semantic_backbone == 'lseg':
            if prior_method != 'lseg':
                raise ValueError("semantic_backbone='lseg' requires prior_method='lseg'.")
            if sem_supervision != 'lseg':
                raise ValueError("semantic_backbone='lseg' requires sem_supervision='lseg'.")
        else:
            if prior_method not in ('proxy_clip', 'native'):
                raise ValueError(
                    f"semantic_backbone='{semantic_backbone}' requires "
                    f"prior_method in {{'proxy_clip', 'native'}}, got '{prior_method}'."
                )

        # ------------------------------------------------------------------
        # Frozen VGGT (geometry backbone)
        # ------------------------------------------------------------------
        self.vggt = VGGT(return_depth_feat=False)
        if os.path.isfile(vggt_ckpt):
            ckpt = torch.load(vggt_ckpt, map_location='cpu')
            msg = self.vggt.load_state_dict(ckpt, strict=False)
            print(f"[OpenSe3r] VGGT loaded: {len(msg.missing_keys)} missing, "
                  f"{len(msg.unexpected_keys)} unexpected")
        for p in self.vggt.parameters():
            p.requires_grad_(False)

        # ------------------------------------------------------------------
        # Optional: VGGT-Omega DINOv3 patch_embed (frozen).
        # When dino_backbone='omega', the CausalBranch's input tokens come
        # from this DINOv3 backbone (patch_size=16) instead of VGGT's DINOv2
        # (patch_size=14).  VGGT geometry path unchanged.
        # ------------------------------------------------------------------
        # Stash omega state_dict for CausalBranch init below; loaded once here.
        omega_state_dict = None
        if dino_backbone == 'omega':
            # Lazy import — vggt_omega uses Python 3.10+ type hints and would
            # break on 3.9 environments if imported unconditionally.
            sys.path.append(os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'thirdparty', 'vggt-omega'
            ))
            from vggt_omega.models.aggregator import _build_patch_embed
            self.omega_patch_embed = _build_patch_embed(
                patch_size=16, embed_dim=1024,
            )
            for p in self.omega_patch_embed.parameters():
                p.requires_grad_(False)
            # Load VGGT-Omega pretrained weights into patch_embed (DINOv3 ViT-L).
            if os.path.isfile(omega_ckpt):
                ckpt = torch.load(omega_ckpt, map_location='cpu', weights_only=False)
                omega_state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
                # Extract aggregator.patch_embed.* → load into our omega_patch_embed.
                prefix = 'aggregator.patch_embed.'
                pe_sd = {k[len(prefix):]: v for k, v in omega_state_dict.items()
                          if k.startswith(prefix)}
                msg = self.omega_patch_embed.load_state_dict(pe_sd, strict=False)
                print(f"[OpenSe3r] Omega patch_embed loaded: "
                      f"{len(msg.missing_keys)} missing, "
                      f"{len(msg.unexpected_keys)} unexpected")
            else:
                print(f"[OpenSe3r] WARN: omega_ckpt not found at {omega_ckpt} — "
                      f"omega_patch_embed kept at random init")
            # ImageNet (ResNet-style) normalization, matching omega's aggregator.
            # Buffer convention: [1, 3, 1, 1] for broadcast over [B, 3, H, W].
            self.register_buffer(
                '_omega_mean',
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                '_omega_std',
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                persistent=False,
            )
        else:
            self.omega_patch_embed = None

        # ------------------------------------------------------------------
        # Frozen semantic backbone (LSeg / PE-Core / CLIP)
        # output_dim defines the prior-feature channel and the semantic GT
        # channel — must match through input_proj, DPT cond, expand_projector.
        # ------------------------------------------------------------------
        self.semantic_backbone = build_semantic_backbone(
            semantic_backbone,
            lseg_ckpt    = lseg_ckpt,
            proxy_beta   = proxy_beta,
            proxy_gamma  = proxy_gamma,
        )
        prior_dim = self.semantic_backbone.output_dim
        if sem_expand is None:
            sem_expand = prior_dim
        self.sem_expand = sem_expand
        # A' — semantic-based mask-quality head.  Input = mask-aggregated (soft-mask
        # pooled) STUDENT semantic feature (detached); output = per-query scalar
        # trained to regress a dense semantic-consistency target computed from the
        # FROZEN backbone feature (dominant-class fraction × mean max-cosine).
        self.use_sem_conf_head = bool(use_sem_conf_head)
        if self.use_sem_conf_head:
            self.sem_conf_head = nn.Sequential(
                nn.Linear(sem_expand, 256), nn.GELU(), nn.Linear(256, 1))
            self._sc_text_emb = None   # lazy [K, C], mean-subtracted + L2-normed
        # Legacy alias (some callsites in ckpt-loader code reference self.lseg).
        # For LSeg backbone we keep the underlying LSeg accessible.
        if semantic_backbone == 'lseg':
            self.lseg = self.semantic_backbone.lseg

        # DINOv2 ViT-L token dim (x_norm_patchtokens from encode_patch_tokens)
        # Per-frame, no cross-frame attention → safe for online inference
        dino_token_dim = 1024

        # ------------------------------------------------------------------
        # Trainable: DepthEncoder
        # ------------------------------------------------------------------
        self.depth_encoder = DepthEncoder(out_dim=1024)
        self.point_cond = point_cond
        if point_cond:
            from .depth_encoder import PointEncoder
            self.point_encoder = PointEncoder(out_dim=1024)

        # ------------------------------------------------------------------
        # Trainable: CausalBranch
        # ------------------------------------------------------------------
        self.causal_branch = CausalBranch(
            dino_token_dim=dino_token_dim,
            lseg_dim=prior_dim,            # input fusion dim = backbone output_dim
            causal_dim=1024,
            n_blocks=n_blocks,
            depth_cross_every=depth_cross_every,
            patch_size=patch_size,
            rope_mode=rope_mode,
            rope_voxel_size=rope_voxel_size,
            depth_cond=(depth_cond or point_cond),
            pose_cond=pose_cond,
            plucker_cond=plucker_cond,
            num_register_tokens    = num_register_tokens,
            register_block_indices = register_block_indices,
            pos3d_cond                  = pos3d_cond,
            use_depth_cross_attn        = use_depth_cross_attn,
            num_instance_queries        = num_instance_queries,
            instance_query_block_indices = instance_query_block_indices,
            instance_queries_init_type  = instance_queries_init_type,
        )

        self.pos3d_cond = pos3d_cond
        if pos3d_cond:
            from .pos_encoders import PosEncoder3D
            self.pos3d_encoder = PosEncoder3D(out_dim=1024)
        self.query_init_type = query_init_type

        if init_from_vggt:
            # When omega backbone + omega ckpt loaded, prefer omega's
            # aggregator blocks (they're trained on DINOv3 token statistics,
            # matching what CausalBranch will see at runtime).  Fall back to
            # VGGT init if omega state_dict wasn't available.
            if omega_state_dict is not None:
                print("[OpenSe3r] Initializing CausalBranch from VGGT-Omega weights...")
                self.causal_branch.init_from_omega(omega_state_dict)
            else:
                print("[OpenSe3r] Initializing CausalBranch from VGGT weights...")
                self.causal_branch.init_from_vggt(self.vggt)

        # VGGT geometry IS used in this backend; only track_head is unused.
        freed = trim_vggt_modules(self.vggt, keep_geometry=True)
        if freed:
            print(f"[OpenSe3r] Freed unused VGGT submodules: {freed}")

        # ------------------------------------------------------------------
        # Optional conditioning encoders (only built when their flag is set
        # → unused flags add no parameters and don't appear in state_dict).
        # ------------------------------------------------------------------
        if pose_cond:
            self.pose_encoder = PoseTokenEncoder(in_dim=13, dim=1024)
        if plucker_cond:
            self.plucker_encoder = PluckerEncoder(
                dim=1024, num_freqs=plucker_num_freqs, use_fourier=True,
            )

        # ------------------------------------------------------------------
        # Trainable: Semantic & Instance DPT heads
        # ------------------------------------------------------------------
        # intermediate_layer_idx: sample 4 evenly-spaced layers from 12
        dpt_layer_idx = [2, 5, 8, 11]

        self.semantic_head = FeatureDPTHead(
            causal_dim=sem_causal_dim,           # sliced channel sub-space (default = 1024)
            out_channels=sem_dim,
            expand_to=sem_expand,                # projector sem_dim → backbone.output_dim
            intermediate_layer_idx=dpt_layer_idx,
            patch_size=patch_size,
            # `use_head_sem_cond=False` → no cond mechanism built at all (smaller head,
            # forces CausalBranch to carry all semantic information).
            lseg_dim=prior_dim if use_head_sem_cond else None,
            lseg_proj_dim=128,
            use_pixel_shuffle_head=head_pixel_shuffle,
        )

        # Instance head: two variants, selected by instance_head_type.
        # Both implementations are imported eagerly only when their config calls
        # for them — keeps the unused branch out of the state_dict for clean
        # backward compatibility with existing 'dense' checkpoints.
        self.use_merge_head = bool(use_merge_head) and (instance_head_type == 'query')
        if instance_head_type == 'dense':
            # Legacy dense head — no LSeg conditioning.
            self.instance_head = FeatureDPTHead(
                causal_dim=ins_causal_dim,
                out_channels=ins_dim,
                expand_to=None,
                intermediate_layer_idx=dpt_layer_idx,
                patch_size=patch_size,
                use_pixel_shuffle_head=head_pixel_shuffle,
            )
        else:   # 'query'
            if query_arch == 'maskformer':
                from .instance_head_query import QueryInstanceHead
                self.instance_head = QueryInstanceHead(
                    causal_dim     = ins_causal_dim,
                    num_queries    = query_num_queries,
                    d_model        = query_d_model,
                    descriptor_dim = query_descriptor_dim,
                    c_mask         = query_c_mask,
                    n_heads        = query_n_heads,
                    n_layers       = query_n_layers,
                    intermediate_layer_idx = dpt_layer_idx,
                    patch_size     = patch_size,
                    use_pixel_shuffle_head = head_pixel_shuffle,
                )
            elif query_arch == 'mask2former':
                from .instance_head_query_m2f import Mask2FormerQueryInstanceHead
                # FastSAM seed queries arrive already at d_model (GeoAwareRegion-
                # Pooling output) → external_init_dim = d_model (no projection).
                # CausalBranch external queries arrive at 1024.
                ext_in_dim = (query_d_model if fastsam_query
                              else (1024 if query_init_type == 'external' else None))
                self.instance_head = Mask2FormerQueryInstanceHead(
                    causal_dim     = ins_causal_dim,
                    num_queries    = query_num_queries,
                    d_model        = query_d_model,
                    c_mask         = query_c_mask,
                    descriptor_dim = query_descriptor_dim,   # =c_mask unless a separate descriptor head is built
                    n_heads        = query_n_heads,
                    n_layers       = query_n_layers,
                    intermediate_layer_idx = dpt_layer_idx,
                    patch_size     = patch_size,
                    num_feature_levels = query_num_feature_levels,
                    return_aux     = query_return_aux,
                    use_box_head   = query_use_box_head,
                    box_mode       = query_box_mode,
                    query_init_type   = query_init_type,
                    external_init_dim = ext_in_dim,
                    descriptor_mlp_layers = query_descriptor_mlp_layers,
                    desc_dual_path    = query_desc_dual_path,
                    desc_fuse_layers  = query_desc_fuse_layers,
                    desc_pathB_aux    = query_desc_pathB_aux,
                    box_head_detach = query_box_head_detach,
                    box_size_anchor = query_box_size_anchor,
                    box_query_only  = query_box_query_only,
                    box_size_extent_aware = query_box_size_extent_aware,
                    use_fgbg_head   = query_use_fgbg_head,
                    fastsam_mask_prior = query_fastsam_mask_prior,
                    # metric mode (depth_norm_mode='none') → init box size ~1 m
                    box_init_log_size = (0.0 if depth_norm_mode == 'none' else -1.20397),
                )
            else:
                raise ValueError(f"query_arch must be 'maskformer' or 'mask2former', got '{query_arch}'")
        if self.use_merge_head:
            from .merge_head import MergeHead
            self.merge_head = MergeHead(desc_dim=query_descriptor_dim, sem_dim=sem_dim)
        self.query_arch = query_arch if instance_head_type == 'query' else None

        # ── FastSAM-region-seeded query modules (gated) ──────────────────────
        self.fastsam_query = bool(fastsam_query)
        self.fastsam_max_regions = int(fastsam_max_regions)
        if self.fastsam_query:
            if not (instance_head_type == 'query' and query_arch == 'mask2former'):
                raise ValueError("fastsam_query requires instance_head_type='query' "
                                 "and query_arch='mask2former'.")
            if query_init_type != 'external':
                raise ValueError("fastsam_query requires query_init_type='external' "
                                 "(seed queries are fed as init_queries).")
            if num_instance_queries and num_instance_queries > 0:
                raise ValueError("fastsam_query is mutually exclusive with the "
                                 "CausalBranch instance queries (num_instance_queries>0).")
            if query_num_queries != fastsam_max_regions:
                raise ValueError(f"query_num_queries ({query_num_queries}) must equal "
                                 f"fastsam_max_regions ({fastsam_max_regions}) — every "
                                 "query slot is one (padded) FastSAM region.")
            from .region_pool import GeoAwareRegionPooling
            from .fastsam_seg import FastSAMSegmenter
            # Trainable seed-query pooler (new state_dict keys appear only here).
            self.region_pool = GeoAwareRegionPooling(
                causal_dim=ins_causal_dim, sem_dim=sem_dim, d_model=query_d_model,
            )
            # Frozen FastSAM held as a PLAIN attribute → never enters state_dict.
            self._fastsam_seg = FastSAMSegmenter(
                ckpt_path=fastsam_ckpt, conf=fastsam_conf, iou=fastsam_iou,
                max_regions=fastsam_max_regions,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slice_token_channels(tokens_list: List[torch.Tensor],
                              channels: Tuple[int, int]) -> List[torch.Tensor]:
        """Slice the last channel dim of each token tensor in the list.
        Returns the original list (no copy) when channels span the full range.
        """
        s, e = channels
        if s == 0 and e == tokens_list[0].shape[-1]:
            return tokens_list
        return [t[..., s:e] for t in tokens_list]

    def _build_fastsam_seed(
        self,
        images_01:        torch.Tensor,            # [B, T, 3, H, W] in [0, 1]
        ins_tokens_list:  List[torch.Tensor],       # causal patch tokens (instance)
        sem_feat:         torch.Tensor,            # [B, T, sem_dim, H, W]
        world_points:     torch.Tensor,            # [B, T, H, W, 3]
        patch_start_idx:  int,
    ):
        """FastSAM region → geo-aware-pooled seed queries (patch resolution).

        Returns:
            seed       [B, T, Mmax, d_model]   query init for the M2F head
            region_id  [B, T, H, W] long       per-pixel FastSAM region id (-1=none)
            region_conf[B, T, Mmax] float       per-region FastSAM confidence
        """
        from .fastsam_seg import build_seed_queries
        return build_seed_queries(
            self._fastsam_seg, self.region_pool, self.patch_size,
            self.fastsam_max_regions, images_01, ins_tokens_list, sem_feat,
            world_points, patch_start_idx,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _sc_get_text_emb(self, device):
        """Lazy ScanNet200 text embeddings (frozen backbone), mean-subtracted +
        L2-normed — removes the CLIP/TIPS modality-gap common component."""
        if self._sc_text_emb is None:
            from amb3r.tools.semantic_vis_utils import build_text_embeddings_via_backbone
            from amb3r.tools.scannet200_constants import CLASS_LABELS_200
            t = build_text_embeddings_via_backbone(
                self.semantic_backbone, list(CLASS_LABELS_200)).float().to(device)
            t = F.normalize(t, dim=-1)
            self._sc_text_emb = F.normalize(t - t.mean(0, keepdim=True), dim=-1)  # [K, C]
        return self._sc_text_emb

    def _forward_sem_conf(self, sem_student, sem_gt, mask_logits, want_target):
        """A' semantic mask-quality head.
          sem_student : [B,T,C,H,W]  student sem feat (DETACHED for the head input)
          sem_gt      : [B,T,C,H,W]  frozen backbone feat (for the target) or None
          mask_logits : [B,T,N,Hm,Wm]
        Returns (sem_conf_logit [B,T,N], target [B,T,N] or None).
        Target = dominant-class-fraction × mean max-cosine over each soft mask."""
        B, T, C, _, _ = sem_student.shape
        N = mask_logits.shape[2]
        H = W = 64          # compute pooling + dense target on a small grid (bounds
                            # memory: sim is [B,T,K,H,W]; full-res K=200 would OOM)
        sem_student = F.interpolate(sem_student.reshape(B * T, C, *sem_student.shape[-2:]),
                                    size=(H, W), mode='bilinear', align_corners=False).reshape(B, T, C, H, W)
        ml = F.interpolate(mask_logits.reshape(B * T, N, *mask_logits.shape[-2:]),
                           size=(H, W), mode='bilinear', align_corners=False).reshape(B, T, N, H, W)
        soft = ml.sigmoid()                                            # [B,T,N,H,W]
        msum = soft.flatten(3).sum(-1).clamp(min=1.0)                  # [B,T,N]
        # head input: soft-mask pooled STUDENT feat (detached)
        pooled = torch.einsum('btnhw,btchw->btnc', soft, sem_student.detach()) / msum[..., None]
        sem_conf_logit = self.sem_conf_head(pooled).squeeze(-1)        # [B,T,N]
        target = None
        if want_target and sem_gt is not None:
            with torch.no_grad():
                txt = self._sc_get_text_emb(sem_gt.device)             # [K,C]
                sem_gt = F.interpolate(sem_gt.float().reshape(B * T, C, *sem_gt.shape[-2:]),
                                       size=(H, W), mode='bilinear', align_corners=False).reshape(B, T, C, H, W)
                sg = F.normalize(sem_gt, dim=2)                        # [B,T,C,H,W]
                sim = torch.einsum('kc,btchw->btkhw', txt, sg)         # [B,T,K,H,W]
                s_map, c_map = sim.max(2).values, sim.argmax(2)        # [B,T,H,W]
                mean_s = torch.einsum('btnhw,bthw->btn', soft, s_map) / msum
                K = txt.shape[0]
                oneh = F.one_hot(c_map.reshape(B, T, -1), K).to(soft.dtype)   # [B,T,HW,K]
                hist = torch.einsum('btnl,btlk->btnk', soft.reshape(B, T, N, -1), oneh)
                dom_frac = hist.max(-1).values / hist.sum(-1).clamp(min=1.0)  # [B,T,N]
                target = (dom_frac * mean_s.clamp(min=0.0)).clamp(0.0, 1.0)
        return sem_conf_logit, target

    def forward(
        self,
        frames: Dict[str, torch.Tensor],
        mode: str = 'causal',
        kv_cache_list: Optional[List] = None,
        return_poses: bool = True,
        extr_rel_override: Optional[torch.Tensor] = None,
        world_points_override: Optional[torch.Tensor] = None,
        box_scale_override:    Optional[torch.Tensor] = None,
        geo_source: str = 'model',
        debug_aabb_dump_dir:    Optional[str] = None,
        debug_aabb_dump_prefix: str = '',
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames:         dict with 'images' key: [B, T, 3, H, W] in [-1, 1]
            mode:           'causal'       (training; causal mask within chunk)
                          | 'full'         (non-streaming eval; fully bidirectional)
                          | 'chunk_bidir'  (streaming inference; bidirectional
                                            within current chunk, causal across
                                            chunks via kv-cache)
            kv_cache_list:  KV-cache for streaming inference (None = fresh start)
            return_poses:   if True, also return VGGT predicted c2w poses [B, T, 4, 4]
            extr_rel_override: Optional [B, T, 4, 4] c2w in the SLAM global frame
                          (anchored to sequence frame 0).  When set, skips
                          `make_relative_pose(extrinsics)`.  See OpenSe3rDA3.forward
                          docstring — required for streaming inference with
                          `pose_cond` to keep pose tokens consistent with
                          KV-cached past tokens.
            world_points_override: Optional [B, T, H, W, 3] world_points already
                          expressed in the SLAM global frame.  When set, replaces
                          VGGT's point-head output (which is in chunk-local
                          frame, c2w[0]=I) for `pos3d_cond`, the box-head
                          conditioning, and the returned
                          `predictions['world_points']`.  REQUIRED for streaming
                          inference with `pos3d_cond` so that 3D-PE tokens share
                          the same frame as the SLAM-global pose tokens.
            box_scale_override: Optional [B] scalar per batch to use in place
                          of the chunk-level depth median for depth /
                          world_points / pose-translation normalisation.  Use
                          this to pin a sequence-global scale (e.g. captured
                          from the init chunk) so the DepthEncoder / pos3d /
                          pose-token normalisation stays stable across chunks
                          instead of fluctuating with each chunk.

        Returns:
            predictions dict with:
                world_points          [B, T, H, W, 3]
                depth                 [B, T, H, W]
                poses                 [B, T, 4, 4]  (only when return_poses=True)
                semantic_feat         [B, T, sem_dim, H, W]
                semantic_feat_expanded [B, T, 512, H, W]
                semantic_conf         [B, T, 1, H, W]
                instance_feat         [B, T, ins_dim, H, W]
                instance_conf         [B, T, 1, H, W]
                _lseg_feat_gt         [B, T, 512, H, W]
                kv_cache              updated cache (only when kv_cache_list passed)
        """
        images_neg1 = frames['images']                            # [B, T, 3, H, W] in [-1,1]
        images_01   = (images_neg1 + 1.0) / 2.0
        B, T, _, H, W = images_01.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # ------------------------------------------------------------------
        # GT-geometry path: skip the VGGT aggregator decode + depth/point/camera
        # heads; build the geometry conditioning from GT keys in `frames`
        # (frame-0-anchored, matching VGGT's chunk-local frame). Only the frozen
        # per-frame DINOv2 patch embed is still needed (CausalBranch input).
        # ------------------------------------------------------------------
        if geo_source == 'gt':
            with torch.no_grad():
                if self.dino_backbone == 'omega':
                    imgs_flat = images_01.reshape(B * T, 3, H, W)
                    imgs_norm = (imgs_flat - self._omega_mean) / self._omega_std
                    dino_tokens_flat = self.omega_patch_embed(
                        imgs_norm, is_training=True)['x_norm_patchtokens']
                else:
                    dino_tokens_flat = self.vggt.aggregator.encode_patch_tokens(
                        images_01)['x_norm_patchtokens']
                dino_tokens = dino_tokens_flat.reshape(B, T, -1, dino_tokens_flat.shape[-1])
            depth_gt, c2w_rel, K_gt, wp_gt, conf_gt, box_scale_gt = \
                build_gt_geometry_from_frames(frames, anchor_mode=self.gt_anchor_mode)
            return self._run_semantic_branch(
                frames, images_01,
                dino_tokens        = dino_tokens,
                dino_tokens_flat   = dino_tokens_flat,
                depth              = depth_gt.unsqueeze(-1),     # [B,T,H,W,1]
                pts_conf           = conf_gt,
                extrinsics         = c2w_rel,
                intrinsics_pred    = K_gt,
                world_points       = wp_gt,
                kv_cache_list      = kv_cache_list,
                mode               = mode,
                return_poses       = return_poses,
                extr_rel_override  = c2w_rel,                    # frame-0 anchored
                box_scale_override = box_scale_gt,
            )
        if geo_source != 'model':
            raise ValueError(f"geo_source must be 'model' or 'gt', got {geo_source!r}")

        # ------------------------------------------------------------------
        # Frozen VGGT: DINOv2 per-frame patch tokens + geometry heads
        # encode_patch_tokens runs DINOv2 independently per frame — no
        # cross-frame attention, safe for online inference.
        # decode_patch_tokens runs the bidirectional VGGT aggregator only
        # for depth/point geometry outputs (frozen, chunk-level).
        # ------------------------------------------------------------------
        with torch.no_grad():
            # CausalBranch input tokens: either VGGT DINOv2 (patch=14, legacy)
            # or VGGT-Ω DINOv3 (patch=16, fewer tokens, ~24% cheaper).
            if self.dino_backbone == 'omega':
                # Omega DinoVisionTransformer expects ImageNet-normalized input.
                imgs_flat   = images_01.reshape(B * T, 3, H, W)
                imgs_norm   = (imgs_flat - self._omega_mean) / self._omega_std
                omega_out   = self.omega_patch_embed(imgs_norm, is_training=True)
                dino_tokens_flat = omega_out['x_norm_patchtokens']           # [B*T, N16, 1024]
                dino_tokens = dino_tokens_flat.reshape(B, T, -1, dino_tokens_flat.shape[-1])
                # VGGT geometry path still uses its OWN DINOv2 patch_embed
                # internally (encode + decode); call as usual.
                patch_tokens_dict = self.vggt.aggregator.encode_patch_tokens(images_01)
            else:
                patch_tokens_dict = self.vggt.aggregator.encode_patch_tokens(images_01)
                # DINOv2 per-frame patch tokens: [B*T, N_patch, 1024]
                dino_tokens_flat = patch_tokens_dict['x_norm_patchtokens']
                dino_tokens = dino_tokens_flat.reshape(B, T, -1, dino_tokens_flat.shape[-1])

            # Run VGGT aggregator for geometry (chunk-level, bidirectional)
            decoded = self.vggt.aggregator.decode_patch_tokens(
                patch_tokens_dict, images_01, mem_eff=True
            )
            agg_list = decoded['aggregated_tokens_list']
            ps_idx   = decoded['patch_start_idx']

            # Run heads in fp32 regardless of outer autocast — matches official VGGT.forward()
            # which wraps all heads in torch.cuda.amp.autocast(enabled=False).
            with torch.amp.autocast('cuda', enabled=False):
                depth, depth_conf = self.vggt.depth_head(
                    agg_list, images=images_01, patch_start_idx=ps_idx
                )
                # depth: [B, T, H, W, 1]

                world_points, pts_conf = self.vggt.point_head(
                    agg_list, images=images_01, patch_start_idx=ps_idx
                )
                # world_points: [B, T, H, W, 3]  in VGGT chunk-local frame

                # Compute pose whenever any consumer needs it: caller asked
                # via return_poses, or pose_cond / plucker_cond requires it
                # for conditioning.
                need_pose = return_poses or self.pose_cond or self.plucker_cond
                if need_pose:
                    pose_enc_list = self.vggt.camera_head(agg_list)
                    w2c_34, intrinsics_pred = pose_encoding_to_extri_intri(
                        pose_enc_list[-1], (H, W)
                    )
                    # w2c_34: [B, T, 3, 4]  world-to-camera (OpenCV)
                    # intrinsics_pred: [B, T, 3, 3]
                    # Pad to [B, T, 4, 4] and invert to get c2w
                    bottom = torch.tensor(
                        [0.0, 0.0, 0.0, 1.0],
                        dtype=w2c_34.dtype, device=w2c_34.device,
                    ).view(1, 1, 1, 4).expand(B, T, 1, 4)
                    w2c_44 = torch.cat([w2c_34, bottom], dim=2)   # [B, T, 4, 4]
                    extrinsics = torch.inverse(w2c_44)             # [B, T, 4, 4]  c2w

        # Pipeline-supplied world_points in SLAM-global frame override VGGT's
        # chunk-local point-head output (see docstring).  Applied here so every
        # downstream consumer (pos3d_cond, box head, predictions output) sees
        # the same SLAM-global world_points.
        if world_points_override is not None:
            if world_points_override.shape != world_points.shape:
                raise ValueError(
                    f"world_points_override shape {tuple(world_points_override.shape)} "
                    f"!= VGGT world_points shape {tuple(world_points.shape)}"
                )
            world_points = world_points_override.to(
                world_points.device, dtype=world_points.dtype,
            )

        # need_pose path leaves `extrinsics` / `intrinsics_pred` undefined when
        # neither return_poses nor pose_cond / plucker_cond is set — pass None
        # and let the helper short-circuit those branches.
        if not (return_poses or self.pose_cond or self.plucker_cond):
            extrinsics      = None
            intrinsics_pred = None

        return self._run_semantic_branch(
            frames, images_01,
            dino_tokens        = dino_tokens,
            dino_tokens_flat   = dino_tokens_flat,
            depth              = depth,                 # [B,T,H,W,1]
            pts_conf           = pts_conf,              # [B,T,H,W,1]
            extrinsics         = extrinsics,            # [B,T,4,4] c2w or None
            intrinsics_pred    = intrinsics_pred,
            world_points       = world_points,
            kv_cache_list      = kv_cache_list,
            mode               = mode,
            return_poses       = return_poses,
            extr_rel_override  = extr_rel_override,
            box_scale_override = box_scale_override,
            debug_aabb_dump_dir    = debug_aabb_dump_dir,
            debug_aabb_dump_prefix = debug_aabb_dump_prefix,
        )

    # ------------------------------------------------------------------
    # Streaming-SLAM helper — runs ONLY the trainable semantic branch.
    # Skips VGGT's aggregator decode + depth_head + point_head + camera_head;
    # caller supplies the already-aligned geometry from a preceding
    # `forward_geometry` (+ memory.update_geo in SLAM).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_semantic(
        self,
        frames: Dict[str, torch.Tensor],
        *,
        depth:              torch.Tensor,            # [B, T, H, W]  camera-frame Z (frame-invariant)
        intrinsics:         torch.Tensor,            # [B, T, 3, 3]
        c2w:                torch.Tensor,            # [B, T, 4, 4]  SLAM-global c2w
        world_points:       torch.Tensor,            # [B, T, H, W, 3]  SLAM-global
        world_points_conf:  Optional[torch.Tensor] = None,  # [B, T, H, W, 1]
        kv_cache_list:      Optional[List] = None,
        mode:               str = 'causal',
        box_scale_override: Optional[torch.Tensor] = None,
        return_poses:       bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Run the trainable semantic / instance branch on a pre-computed
        geometry chunk (VGGT backend).  Skips the expensive VGGT aggregator
        decode + depth/point/camera heads.  Use in streaming SLAM (pred-mode)
        to avoid the redundant per-chunk VGGT pass.

        Geometry inputs are expected to be in the SLAM-global frame already;
        this method passes the c2w through as `extr_rel_override` so the
        pose / plucker tokens use it directly (no `make_relative_pose`).

        Args:
            frames:        dict with 'images' [B,T,3,H,W] in [-1,1]
            depth:         [B,T,H,W]   from `geo_res['depth']`
            intrinsics:    [B,T,3,3]   from `geo_res['intrinsics']`
            c2w:           [B,T,4,4]   SLAM-global c2w (e.g. `memory.poses`)
            world_points:  [B,T,H,W,3] SLAM-global (e.g. `memory._pending_pts_new`)
            world_points_conf: [B,T,H,W,1]  optional, defaults to ones
        """
        images_neg1 = frames['images']
        images_01   = (images_neg1 + 1.0) / 2.0
        B, T, _, H, W = images_01.shape

        # DINOv2 patch tokens (per-frame, frozen) — the only frozen forward
        # we still need; VGGT aggregator decode + heads are skipped.
        with torch.no_grad():
            if self.dino_backbone == 'omega':
                imgs_flat   = images_01.reshape(B * T, 3, H, W)
                imgs_norm   = (imgs_flat - self._omega_mean) / self._omega_std
                omega_out   = self.omega_patch_embed(imgs_norm, is_training=True)
                dino_tokens_flat = omega_out['x_norm_patchtokens']
                dino_tokens = dino_tokens_flat.reshape(B, T, -1, dino_tokens_flat.shape[-1])
            else:
                patch_tokens_dict = self.vggt.aggregator.encode_patch_tokens(images_01)
                dino_tokens_flat  = patch_tokens_dict['x_norm_patchtokens']
                dino_tokens       = dino_tokens_flat.reshape(B, T, -1, dino_tokens_flat.shape[-1])

        if world_points_conf is None:
            world_points_conf = torch.ones(
                B, T, H, W, 1, device=depth.device, dtype=depth.dtype,
            )
        # Helper's `depth` arg expects the VGGT depth-head shape: [B,T,H,W,1].
        depth_5d = depth.unsqueeze(-1) if depth.ndim == 4 else depth

        return self._run_semantic_branch(
            frames, images_01,
            dino_tokens        = dino_tokens,
            dino_tokens_flat   = dino_tokens_flat,
            depth              = depth_5d,
            pts_conf           = world_points_conf,
            extrinsics         = c2w,
            intrinsics_pred    = intrinsics,
            world_points       = world_points,
            kv_cache_list      = kv_cache_list,
            mode               = mode,
            return_poses       = return_poses,
            # User's c2w is already SLAM-global → reuse it as the override
            # so pose_token / plucker skip `make_relative_pose`.
            extr_rel_override  = c2w,
            box_scale_override = box_scale_override,
        )

    # ------------------------------------------------------------------
    # Post-geometry shared body.  forward() and forward_semantic() both
    # converge here once dino tokens + geometry are in hand.
    # ------------------------------------------------------------------

    def _run_semantic_branch(
        self,
        frames:           Dict[str, torch.Tensor],
        images_01:        torch.Tensor,
        *,
        dino_tokens:      torch.Tensor,
        dino_tokens_flat: torch.Tensor,
        depth:            torch.Tensor,                     # [B,T,H,W,1]
        pts_conf:         torch.Tensor,                     # [B,T,H,W,1]
        extrinsics:       Optional[torch.Tensor],           # [B,T,4,4] c2w or None
        intrinsics_pred:  Optional[torch.Tensor],           # [B,T,3,3] or None
        world_points:     torch.Tensor,
        kv_cache_list:    Optional[List],
        mode:             str,
        return_poses:     bool,
        extr_rel_override:  Optional[torch.Tensor],
        box_scale_override: Optional[torch.Tensor],
        debug_aabb_dump_dir:    Optional[str] = None,
        debug_aabb_dump_prefix: str = '',
    ) -> Dict[str, torch.Tensor]:
        B, T, _, H, W = images_01.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # ------------------------------------------------------------------
        # Frozen semantic backbone: prior + (training-time) supervision GT.
        # Naming preserved as `lseg_feat` for state-dict compatibility, but
        # the dense feature now comes from whichever backbone was selected.
        # ------------------------------------------------------------------
        imgs_flat = images_01.reshape(B * T, 3, H, W)

        with torch.no_grad():
            if self.prior_method == 'proxy_clip':
                # Need DINOv2 patch grid (ph, pw) — same as model patch grid.
                lseg_feat = self.semantic_backbone.forward_prior(
                    imgs_flat,
                    dino_tokens = dino_tokens_flat,
                    dino_grid   = (patch_h, patch_w),
                )
            elif self.prior_method == 'native':
                # Native dense forward — best for TIPSv2 (patch-text aligned).
                lseg_feat = self.semantic_backbone.forward_prior_native(imgs_flat)
            else:                                         # 'lseg'
                lseg_feat = self.semantic_backbone.forward_prior(imgs_flat)
            # Upsample to (H, W) so downstream code (input fusion + DPT cond)
            # keeps its interface regardless of backbone's native grid size.
            if lseg_feat.shape[-2:] != (H, W):
                lseg_feat = F.interpolate(
                    lseg_feat, size=(H, W),
                    mode='bilinear', align_corners=False,
                )

        # ------------------------------------------------------------------
        # Depth encoder → patch-level geometry tokens for cross-attention.
        #
        # `depth_norm_mode` decides the per-batch divisor:
        #   - 'chunk' (default): same chunk-level median for every T frame.
        #     Preserves cross-view consistency inside a chunk so the
        #     CausalBranch's cross-frame attention sees coherent depth
        #     tokens.  Aligns with `box_scale` exactly.
        #   - 'frame'  : per-frame median.  Cheaper to compute but the
        #     same physical surface appears at different normalized
        #     depths across frames → cross-attention has to bridge an
        #     extra per-frame scale offset.  Kept for back-compat with
        #     ckpts trained under the original behaviour.
        # ------------------------------------------------------------------
        depth_2d = depth.squeeze(-1)                                              # [B, T, H, W]
        depth_med_per_frame = depth_2d.median(dim=-1, keepdim=True).values \
                                       .median(dim=-2, keepdim=True).values        # [B, T, 1, 1]
        # Chunk-level median (used as the scale-invariant box_scale either way).
        box_scale = depth_med_per_frame.flatten(1).median(dim=-1) \
                                         .values.clamp(min=0.1)                     # [B]

        # Pipeline-supplied sequence-global scale overrides the chunk-level
        # median.  Affects depth / pos3d / scene_scale (pose+plucker)
        # normalisation — see SLAM's `--global_scale_mode`.
        if box_scale_override is not None:
            if box_scale_override.shape != box_scale.shape:
                raise ValueError(
                    f"box_scale_override shape {tuple(box_scale_override.shape)} "
                    f"!= computed box_scale shape {tuple(box_scale.shape)}"
                )
            box_scale = box_scale_override.to(
                box_scale.device, dtype=box_scale.dtype,
            ).clamp(min=0.1)

        if self.depth_norm_mode == 'none':
            # Metric mode: box_scale → 1 disables ALL scale norm (depth / point /
            # pos3d / box-head + GT box) → everything stays METRIC, consistent.
            box_scale = torch.ones_like(box_scale)

        if self.depth_norm_mode in ('chunk', 'none'):
            depth_norm = depth_2d / box_scale.view(-1, 1, 1, 1)
        else:                                                                      # 'frame'
            depth_norm = depth_2d / (depth_med_per_frame + 1e-6)

        if self.point_cond:
            # Normalized per-pixel world points (xyz) → PointEncoder → depth-cross-attn path.
            wp_full = world_points.reshape(B * T, H, W, 3).permute(0, 3, 1, 2)     # [BT,3,H,W]
            bs_pt = box_scale.clamp(min=1e-3).view(B, 1, 1, 1).expand(B, T, 1, 1).reshape(B * T, 1, 1, 1)
            depth_tokens = self.point_encoder((wp_full / bs_pt).to(depth_2d.dtype), patch_h, patch_w)
        else:
            depth_tokens = self.depth_encoder(
                depth_norm.reshape(B * T, 1, H, W), patch_h, patch_w
            )                                                                     # [B*T, N_patch, 1024]

        # (Idea 1) Optional 3D position encoder — pool world_points to patch grid,
        # NORMALIZE by box_scale, then Fourier PE → MLP.
        pos3d_tokens = None
        if self.pos3d_cond:
            wp_full  = world_points.reshape(B * T, H, W, 3).permute(0, 3, 1, 2)
            wp_patch = F.adaptive_avg_pool2d(wp_full, (patch_h, patch_w))
            wp_patch = wp_patch.permute(0, 2, 3, 1).reshape(B * T, patch_h * patch_w, 3)
            bs = box_scale.clamp(min=1e-3).view(B, 1, 1, 1) \
                            .expand(B, T, 1, 1).reshape(B * T, 1, 1)
            wp_patch_norm = wp_patch / bs
            pos3d_tokens = self.pos3d_encoder(wp_patch_norm.to(depth_tokens.dtype))

        # ------------------------------------------------------------------
        # Optional: pose / Plücker conditioning tokens.
        # scene_scale is the clip-level depth median, shared with DepthEncoder's
        # per-frame median normalization → cross-backend (canonical / metric)
        # scale-invariance.
        # ------------------------------------------------------------------
        pose_token     = None
        plucker_tokens = None
        if self.pose_cond or self.plucker_cond:
            scene_scale = box_scale                                               # [B]

            if extr_rel_override is not None:
                extr_rel = extr_rel_override.to(extrinsics.device, dtype=extrinsics.dtype)
                if extr_rel.shape != extrinsics.shape:
                    raise ValueError(
                        f"extr_rel_override shape {tuple(extr_rel.shape)} != "
                        f"VGGT extrinsics shape {tuple(extrinsics.shape)}"
                    )
            else:
                extr_rel = make_relative_pose(extrinsics.detach())                # [B,T,4,4]
            intr_d   = intrinsics_pred.detach()

            if self.pose_cond:
                pose_feats = pose_to_features(
                    extr_rel, intr_d, H, W, scene_scale,
                )                                                                  # [B, T, 13]
                pose_token = self.pose_encoder(pose_feats)                         # [B, T, 1024]

            if self.plucker_cond:
                plucker = compute_plucker_rays(
                    extr_rel, intr_d, H, W, self.patch_size, scene_scale,
                )                                                                  # [B, T, N_patch, 6]
                plucker_tokens = self.plucker_encoder(plucker)                     # [B, T, N_patch, 1024]

        # Per-patch world coords for 3D RoPE (metric, NOT normalized).
        world_pts_patch = None
        if getattr(self.causal_branch, 'rope_mode', '2d') != '2d':
            wp_p = world_points.reshape(B * T, H, W, 3).permute(0, 3, 1, 2)
            wp_p = F.adaptive_avg_pool2d(wp_p, (patch_h, patch_w))
            world_pts_patch = wp_p.permute(0, 2, 3, 1).reshape(B, T, patch_h * patch_w, 3)

        # ------------------------------------------------------------------
        # Trainable: Causal branch
        # ------------------------------------------------------------------
        output_tokens_list, new_kv, instance_q_out = self.causal_branch(
            dino_tokens=dino_tokens,
            lseg_feat=lseg_feat,
            depth_tokens=depth_tokens,
            patch_h=patch_h,
            patch_w=patch_w,
            mode=mode,
            kv_cache_list=kv_cache_list,
            pose_token=pose_token,
            plucker_tokens=plucker_tokens,
            pos3d_tokens=pos3d_tokens,
            world_pts_patch=world_pts_patch,
        )

        # ------------------------------------------------------------------
        # DPT heads
        # semantic head receives LSeg features as spatial conditioning.
        # patch_start_idx = 1 when pose token is prepended at position 0.
        # ------------------------------------------------------------------
        ps_idx_dpt = self.causal_branch.patch_start_idx
        # Optional per-head channel slicing to encourage decomposition between
        # semantic and instance representations.  Default (0..1024, 0..1024)
        # is a no-op pass-through and is identical to the legacy single-head
        # behaviour (existing checkpoints load unchanged).
        sem_tokens_list = self._slice_token_channels(output_tokens_list, self.sem_channels)
        ins_tokens_list = self._slice_token_channels(output_tokens_list, self.ins_channels)

        # `use_head_sem_cond` lets you suppress the prior at the head even when
        # the head was built with a cond mechanism (e.g. for inference-time
        # ablation on a checkpoint trained with cond enabled).  When the head
        # was built without cond (lseg_dim=None), this kwarg is ignored.
        sem_feat, sem_conf, sem_feat_expanded = self.semantic_head(
            sem_tokens_list, images_01, patch_start_idx=ps_idx_dpt,
            semantic_cond=lseg_feat if self.use_head_sem_cond else None,
        )

        # Instance head dispatch: dense returns (feat, conf, _), query returns
        # a dict with keys mask_logits / objectness / descriptor.
        if self.instance_head_type == 'dense':
            ins_feat, ins_conf, _ = self.instance_head(
                ins_tokens_list, images_01, patch_start_idx=ps_idx_dpt,
            )
            query_out = None
        else:   # 'query'
            ins_feat, ins_conf = None, None
            # Pass world_points only to the mask2former head AND only when
            # the world_aabb box mode is active (other modes ignore it).
            head_kw = dict(
                output_tokens_list = ins_tokens_list,
                images             = images_01,
                patch_start_idx    = ps_idx_dpt,
            )
            if getattr(self.instance_head, 'box_mode', None) in ('world_aabb', 'world_amodal', 'world_amodal_grow'):
                head_kw['world_points'] = world_points
                head_kw['box_scale']    = box_scale
                if debug_aabb_dump_dir is not None:
                    head_kw['debug_dump_dir']    = debug_aabb_dump_dir
                    head_kw['debug_dump_prefix'] = debug_aabb_dump_prefix
            # (Idea 2) external init queries from CausalBranch
            if (getattr(self.instance_head, 'query_init_type', None) == 'external'
                    and instance_q_out is not None):
                head_kw['init_queries'] = instance_q_out
            # FastSAM-region-seeded queries (ESAM-style).  Overrides the init
            # source; also emits the region map for the sparse matcher.
            fastsam_region_id = None
            fastsam_region_conf = None
            if self.fastsam_query:
                seed, fastsam_region_id, fastsam_region_conf = self._build_fastsam_seed(
                    images_01, ins_tokens_list, sem_feat, world_points, ps_idx_dpt,
                )
                head_kw['init_queries'] = seed
                # ESAM-style mask prior: feed the FastSAM region map so the head
                # can start layer-0 cross-attn from each query's own region.
                if getattr(self.instance_head, 'fastsam_mask_prior', False):
                    head_kw['fastsam_region_id'] = fastsam_region_id
            query_out = self.instance_head(**head_kw)

        # ------------------------------------------------------------------
        # Semantic GT for the loss.
        #   - 'lseg':        prior dense feature itself (loss aggregates by inst)
        #   - 'textregion':  per-instance V-pooled features broadcast to mask area
        #   - 'mask_crops':  per-instance crop global emb broadcast to mask area
        #   - 'patch_mean':  hard-mask mean of backbone-native dense patch
        #                    features (needs patch-text-aligned backbone)
        #
        # textregion / mask_crops / patch_mean require an instance_mask in
        # `frames`. If not provided (e.g. inference), the GT key is omitted.
        # ------------------------------------------------------------------
        sem_gt = None
        instance_mask = frames.get('instance_mask') if isinstance(frames, dict) else None
        with torch.no_grad():
            if self.sem_supervision == 'lseg':
                sem_gt = lseg_feat.reshape(B, T, lseg_feat.shape[1], H, W)
            elif instance_mask is not None:
                sem_gt = self.semantic_backbone.forward_supervision(
                    imgs_flat,
                    instance_mask    = instance_mask,
                    sem_supervision  = self.sem_supervision,
                    multi_res_crops  = self.multi_res_crops,
                    mr_grid_h        = self.mr_grid_h,
                    mr_grid_w        = self.mr_grid_w,
                    mask_crop_size   = self.mask_crop_size,
                    mask_crop_margin = self.mask_crop_margin,
                    mask_crop_mode   = self.mask_crop_mode,
                )

        # ------------------------------------------------------------------
        # Assemble predictions
        # ------------------------------------------------------------------
        predictions = {
            'world_points':      world_points,                          # [B,T,H,W,3]
            'world_points_conf': pts_conf,                              # [B,T,H,W,1]
            'depth':             depth.squeeze(-1),                     # [B,T,H,W]
            'semantic_feat':     sem_feat,                              # [B,T,sem_dim,H,W]
            'semantic_conf':     sem_conf,                              # [B,T,1,H,W]
            # Dense instance outputs (only populated in 'dense' mode):
            'instance_feat':     ins_feat,                              # [B,T,ins_dim,H,W] or None
            'instance_conf':     ins_conf,                              # [B,T,1,H,W] or None
            'instance_head_type': self.instance_head_type,
            'is_metric':         False,
            'scale_factor':      1.0,
            # Always-on so SLAM can capture the first-chunk scale for the
            # `--global_scale_mode first_chunk` flag.
            'box_scale':         box_scale,                             # [B]
        }
        # Query-based instance outputs (only populated in 'query' mode):
        if query_out is not None:
            predictions['mask_logits'] = query_out['mask_logits']        # [B,T,N,H,W]
            predictions['objectness']  = query_out['objectness']         # [B,T,N]
            predictions['descriptor']  = query_out['descriptor']         # [B,T,N,D]
            _sc_in = sem_feat_expanded if sem_feat_expanded is not None else sem_feat
            if self.use_sem_conf_head and _sc_in is not None:
                sc_logit, sc_tgt = self._forward_sem_conf(
                    _sc_in, sem_gt, query_out['mask_logits'], want_target=self.training)
                predictions['sem_conf'] = sc_logit                       # [B,T,N]
                if sc_tgt is not None:
                    predictions['sem_conf_target'] = sc_tgt
            if 'fgbg_logits' in query_out:
                predictions['fgbg_logits'] = query_out['fgbg_logits']    # [B,T,N] thing/stuff
            # FastSAM region map → consumed by the sparse matcher (loss) and
            # by inference-time snap.  Only present when fastsam_query is on.
            if self.fastsam_query and fastsam_region_id is not None:
                predictions['fastsam_region_id']   = fastsam_region_id   # [B,T,H,W] long
                predictions['fastsam_region_conf'] = fastsam_region_conf # [B,T,Mmax]
            if query_out.get('descriptor_pathB') is not None:
                predictions['descriptor_pathB'] = query_out['descriptor_pathB']  # [B,T,N,D] (path-B aux)
            if getattr(self, 'use_merge_head', False):
                predictions['_merge_head'] = self.merge_head
            if 'box' in query_out:
                # Head outputs (cx, cy, cz, w, h, d) in CAMERA frame at
                # CANONICAL scale.  Loss compares against gt_box also at
                # canonical scale (divided by gt depth median per batch).
                # SLAM inference multiplies by box_scale to get the
                # current scene's metric (in DA3's native units).
                # `box_scale` itself is now always set in predictions above
                # (used by the `--global_scale_mode` SLAM path).
                predictions['box']       = query_out['box']              # [B,T,N,6|12]
                predictions['box_mode']  = query_out.get('box_mode', 'aabb')
            # Deep-supervision aux outputs (Mask2Former arch only):
            if 'aux_outputs' in query_out:
                predictions['aux_outputs'] = query_out['aux_outputs']
        if sem_gt is not None:
            predictions['_sem_feat_gt'] = sem_gt
            # Legacy alias — keeps old loss / inference scripts working.
            if self.sem_supervision == 'lseg':
                predictions['_lseg_feat_gt'] = sem_gt

        if sem_feat_expanded is not None:
            predictions['semantic_feat_expanded'] = sem_feat_expanded

        # DINO patch tokens (computed anyway; expose for Stage 2 refinement
        # and debug visualisation).  Native shape [B, T, N_patch, dino_dim];
        # downstream consumer reshapes to [B, T, dino_dim, ph, pw] if needed.
        predictions['dino_patches'] = dino_tokens

        if return_poses:
            predictions['poses'] = extrinsics   # [B, T, 4, 4]

        if new_kv is not None:
            predictions['kv_cache'] = new_kv

        return predictions

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_geometry(
        self,
        frames: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        VGGT-only forward pass: returns world_points, depth, poses, and
        world_points_conf without running the CausalBranch or DPT heads.
        Used for overlap-based chunk alignment in the predicted-geometry pipeline.

        Args:
            frames: dict with 'images' key: [B, T, 3, H, W] in [-1, 1]

        Returns:
            dict with:
                world_points      [B, T, H, W, 3]
                world_points_conf [B, T, H, W, 1]
                depth             [B, T, H, W]
                poses             [B, T, 4, 4]  c2w
                intrinsics        [B, T, 3, 3]
        """
        images_neg1 = frames['images']
        images_01   = (images_neg1 + 1.0) / 2.0
        B, T, _, H, W = images_01.shape

        with torch.no_grad():
            patch_tokens_dict = self.vggt.aggregator.encode_patch_tokens(images_01)
            decoded = self.vggt.aggregator.decode_patch_tokens(
                patch_tokens_dict, images_01, mem_eff=True
            )
            agg_list = decoded['aggregated_tokens_list']
            ps_idx   = decoded['patch_start_idx']

            with torch.amp.autocast('cuda', enabled=False):
                depth, _ = self.vggt.depth_head(
                    agg_list, images=images_01, patch_start_idx=ps_idx
                )
                world_points, pts_conf = self.vggt.point_head(
                    agg_list, images=images_01, patch_start_idx=ps_idx
                )

                pose_enc_list = self.vggt.camera_head(agg_list)
                w2c_34, intrinsics_pred = pose_encoding_to_extri_intri(pose_enc_list[-1], (H, W))
                bottom = torch.tensor(
                    [0.0, 0.0, 0.0, 1.0],
                    dtype=w2c_34.dtype, device=w2c_34.device,
                ).view(1, 1, 1, 4).expand(B, T, 1, 4)
                w2c_44   = torch.cat([w2c_34, bottom], dim=2)
                extrinsics = torch.inverse(w2c_44)

        return {
            'world_points':      world_points,
            'world_points_conf': pts_conf,
            'depth':             depth.squeeze(-1),
            'poses':             extrinsics,
            'intrinsics':        intrinsics_pred,
        }

    def init_kv_cache(self) -> List:
        return self.causal_branch.init_kv_cache()

    @torch.no_grad()
    def forward_chunk(
        self,
        frames: Dict[str, torch.Tensor],
        kv_cache_list: Optional[List] = None,
        return_poses: bool = True,
        mode: str = 'causal',
        extr_rel_override:      Optional[torch.Tensor] = None,
        world_points_override:  Optional[torch.Tensor] = None,
        box_scale_override:     Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Streaming inference: process a chunk of frames with KV-cache.

        Args:
            mode: 'causal' (default — matches training, frame i can't see j>i
                  inside the first chunk) or 'chunk_bidir' (frames within the
                  current chunk attend bidirectionally; chunk-to-chunk
                  causality preserved by kv_cache_list).
            extr_rel_override:      see `forward()` — SLAM-global c2w for
                  the current chunk; required for `pose_cond` streaming.
            world_points_override:  see `forward()` — SLAM-global
                  world_points for the current chunk; required for
                  `pos3d_cond` streaming.
            box_scale_override:     see `forward()` — sequence-global
                  scale to pin depth / pos3d / pose normalisation across
                  chunks; see SLAM's `--global_scale_mode`.
        """
        return self.forward(frames, mode=mode,
                            kv_cache_list=kv_cache_list,
                            return_poses=return_poses,
                            extr_rel_override=extr_rel_override,
                            world_points_override=world_points_override,
                            box_scale_override=box_scale_override)
