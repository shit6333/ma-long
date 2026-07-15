import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Dict, Tuple

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))

from vggt.models.vggt import VGGT

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import affine_inverse, as_homogeneous, unproject_depth

from .causal_branch import CausalBranch
from .depth_encoder import DepthEncoder, PointEncoder
from .dpt_heads import FeatureDPTHead
from .model import trim_vggt_modules, build_gt_geometry_from_frames
from .cond_encoders import (
    PoseTokenEncoder, PluckerEncoder,
    make_relative_pose, pose_to_features, compute_plucker_rays,
)
from .semantic_backbones import build_semantic_backbone


# ImageNet stats — DA3 expects ImageNet-normalized inputs.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def _safe_interp_bilinear(x: torch.Tensor, size) -> torch.Tensor:
    """F.interpolate(bilinear) that avoids the CUDA int32 INPUT-numel limit (2^31)
    by resampling per-batch-item when the input would overflow.  Needed for wide
    semantic features (RADIO/SigLIP 1536-d) at full resolution × many frames."""
    if x.numel() < (1 << 31):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    return torch.cat([F.interpolate(x[i:i + 1], size=size, mode='bilinear',
                                    align_corners=False) for i in range(x.shape[0])], dim=0)


class OpenSe3rDA3(nn.Module):
    """
    OpenSe3r variant where geometry comes from Depth Anything 3 (DA3) instead of VGGT.

    Frozen branches:
      - DA3:                 depth + camera intrinsics + (w2c) extrinsics
      - VGGT.encode_patch_tokens:  per-frame DINOv2 patch tokens (no cross-frame attn)
      - LSeg:                per-frame dense semantic features (also pseudo-GT)

    Trainable (unchanged from OpenSe3r):
      - DepthEncoder, CausalBranch (init from VGGT), SemanticDPTHead, InstanceDPTHead.

    World points are unprojected from DA3 depth + intrinsics + c2w (= inv(extrinsics))
    inside the model so downstream code keeps the same output schema as OpenSe3r.
    """

    def __init__(
        self,
        da3_ckpt: str = '/mnt/HDD1/ricky/slam/amb3r/checkpoints/da3_base',
        vggt_ckpt: str = './checkpoints/VGGT.pt',
        omega_ckpt: str = './checkpoints/vggt_omega/vggt_omega_1b_512.pt',
        lbotv_ckpt: str = './checkpoints/lingbot_vision_vitl',
        lseg_ckpt: str = './checkpoints/lseg_minimal_e200.ckpt',
        # Semantic head
        sem_dim: int = 64,
        sem_expand: Optional[int] = None,
        # Instance head
        ins_dim: int = 16,
        # Causal branch
        n_blocks: int = 12,
        depth_cross_every: int = 3,
        init_from_vggt: bool = True,
        patch_size: int = 14,
        dino_backbone: str = 'vggt',   # 'vggt' (DINOv2 ps14) / 'omega' (DINOv3 ps16) /
                                       # 'radio' (RADIO replaces DINO+semantic; needs radio-* backbone) /
                                       # 'lbotv' (LingBot-Vision ViT-L/16 replaces DINOv2; semantic
                                       #  backbone (e.g. TIPS) kept as dual-stream prior + GT)
        # See OpenSe3r for full doc.  'chunk' (default): single divisor for
        # all T frames in a batch element (cross-view consistent, matches
        # box_scale).  'frame': per-frame median (legacy).
        depth_norm_mode: str = 'chunk',
        # gt_anchor_mode (geo_source='gt' only): how to re-anchor dataset-world GT
        # geometry to frame 0.  'se3' (default) = full SE3 into the tilted frame-0
        # camera frame; 'translate' = keep gravity-aligned world rotation, only
        # shift the origin (→ tight axis-aligned amodal boxes; pairs with
        # --query_box_mode world_amodal + --point_cond).
        gt_anchor_mode: str = 'se3',
        # Optional conditioning streams (default off).  See model.py / cond_encoders.py.
        depth_cond:   bool = False,
        # point_cond: feed NORMALIZED per-pixel world_points (xyz) through a 3-ch
        # PointEncoder into the SAME depth-cross-attn injection path (replacing
        # depth as the geometry signal).  Gives the CausalBranch explicit 3D
        # position.  Mutually usable with depth_cond (point wins when both set).
        point_cond:   bool = False,
        pose_cond:    bool = False,
        plucker_cond: bool = False,
        # 3D RoPE injection: '2d' (default) | '3d_full' | '3d_cross'.
        rope_mode:        str   = '2d',
        rope_voxel_size:  float = 0.1,
        # 3D PE (Idea 1) + middle-depth gate
        pos3d_cond:                  bool = False,
        use_depth_cross_attn:        bool = True,
        query_init_type:             str  = 'learnable',
        # Instance queries in CausalBranch (Idea 2)
        num_instance_queries:        int  = 0,
        instance_query_block_indices: Optional[List[int]] = None,
        instance_queries_init_type:  str  = 'learnable',
        # VGGT-Ω register attention (CausalBranch).  Default 0 = off (legacy).
        num_register_tokens:    int = 0,
        register_block_indices: Optional[List[int]] = None,
        plucker_num_freqs: int = 4,
        # Semantic backbone (default = legacy LSeg).
        semantic_backbone: str = 'lseg',
        prior_method:      str = 'lseg',
        sem_supervision:   str = 'lseg',     # {'lseg', 'textregion', 'mask_crops', 'patch_mean'}
        proxy_beta:    float = 1.2,
        proxy_gamma:   float = 3.0,
        multi_res_crops: bool = False,
        mr_grid_h: int = None,
        mr_grid_w: int = None,
        mask_crop_size: int = 224,
        mask_crop_margin: float = 0.1,
        mask_crop_mode: str = 'bbox',
        use_head_sem_cond: bool = True,      # Inject prior as DPT-layer cond in semantic head.
        head_pixel_shuffle: bool = False,    # FeatureDPTHead lightweight path (VGGT-Ω-style).
                                             # False = no-shortcut variant (head has no cond at all).
        # Channel split for sem/ins heads — see OpenSe3r for details.  Default
        # (0..1024, 0..1024) is backward-compatible with existing checkpoints.
        sem_channels: Tuple[int, int] = (0, 1024),
        ins_channels: Tuple[int, int] = (0, 1024),
        # Instance head type — see OpenSe3r for details.
        instance_head_type: str = 'dense',
        query_arch:          str = 'maskformer',
        query_num_queries:   int = 100,
        query_d_model:       int = 256,
        query_descriptor_dim: int = 256,
        query_c_mask:        int = 256,
        query_n_heads:       int = 8,
        query_n_layers:      int = 6,
        query_num_feature_levels: int = 3,
        query_return_aux:    bool = True,
        query_use_box_head:  bool = False,
        query_box_mode:      str  = 'aabb',
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
        # FastSAM-region-seeded queries (see OpenSe3r for full doc).  Gated;
        # default off → backward compatible.
        use_sem_conf_head:   bool = False,   # A': semantic-based mask-quality head
        fastsam_query:       bool = False,
        fastsam_ckpt:        str  = "/mnt/HDD4/ricky/sam_test/FastSAM-x.pt",
        fastsam_conf:        float = 0.2,
        fastsam_iou:         float = 0.9,
        fastsam_max_regions: int  = 100,
        # Hook: when True, skip building the DA3 backbone (`self.da3` will
        # remain unset).  Subclasses (e.g. `OpenSe3rMapAnything`) use this to
        # plug in a different frozen geometry backbone, then override
        # `_da3_forward` to call it instead.  Default False keeps the legacy
        # behaviour bit-identical.
        _skip_geo_backbone:  bool = False,
    ):
        super().__init__()
        self._skip_geo_backbone = bool(_skip_geo_backbone)
        if dino_backbone not in ('vggt', 'omega', 'radio', 'lbotv'):
            raise ValueError(f"dino_backbone must be 'vggt', 'omega', 'radio' or 'lbotv', got {dino_backbone!r}")
        self.dino_backbone = dino_backbone
        if dino_backbone in ('omega', 'radio', 'lbotv') and patch_size != 16:
            print(f"[OpenSe3rDA3] dino_backbone='{dino_backbone}' — overriding patch_size {patch_size} → 16")
            patch_size = 16   # RADIO / LingBot-Vision are natively patch-16; use their native grid
        self.patch_size = patch_size
        if depth_norm_mode not in ('chunk', 'frame', 'none'):
            raise ValueError(f"depth_norm_mode must be 'chunk', 'frame' or 'none', got {depth_norm_mode!r}")
        self.depth_norm_mode = depth_norm_mode
        if gt_anchor_mode not in ('se3', 'translate'):
            raise ValueError(f"gt_anchor_mode must be 'se3' or 'translate', got {gt_anchor_mode!r}")
        self.gt_anchor_mode = gt_anchor_mode

        self.depth_cond   = depth_cond
        self.pose_cond    = pose_cond
        self.plucker_cond = plucker_cond

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

        # Channel split validation + storage (mirrors OpenSe3r).
        causal_dim = 1024
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

        if instance_head_type not in ('dense', 'query'):
            raise ValueError(
                f"instance_head_type must be 'dense' or 'query', got '{instance_head_type}'"
            )
        self.instance_head_type = instance_head_type

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
        # Frozen DA3 (geometry backbone) — gated by `_skip_geo_backbone` so
        # subclasses can plug in a different backbone (e.g. MapAnything).
        # ------------------------------------------------------------------
        if not self._skip_geo_backbone:
            if os.path.isdir(da3_ckpt) or '/' in da3_ckpt:
                self.da3 = DepthAnything3.from_pretrained(da3_ckpt)
                print(f"[OpenSe3rDA3] DA3 loaded from {da3_ckpt}")
            else:
                self.da3 = DepthAnything3(model_name=da3_ckpt)
                print(f"[OpenSe3rDA3] DA3 instantiated as preset '{da3_ckpt}' (no weights)")
            for p in self.da3.parameters():
                p.requires_grad_(False)

        # ImageNet normalization buffers — kept on both DA3 and MapAnything
        # subclass paths because MapAnything's DINOv2 encoder also uses
        # ImageNet stats; the buffers are tiny and harmless when unused.
        self.register_buffer(
            '_da3_mean', torch.tensor(_IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False,
        )
        self.register_buffer(
            '_da3_std',  torch.tensor(_IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False,
        )

        # ------------------------------------------------------------------
        # Frozen VGGT — only encode_patch_tokens is used (per-frame DINOv2).
        # CausalBranch is still initialized from VGGT weights when requested.
        # ------------------------------------------------------------------
        self.vggt = VGGT(return_depth_feat=False)
        if os.path.isfile(vggt_ckpt):
            ckpt = torch.load(vggt_ckpt, map_location='cpu')
            msg = self.vggt.load_state_dict(ckpt, strict=False)
            print(f"[OpenSe3rDA3] VGGT loaded: {len(msg.missing_keys)} missing, "
                  f"{len(msg.unexpected_keys)} unexpected")
        for p in self.vggt.parameters():
            p.requires_grad_(False)

        # ------------------------------------------------------------------
        # Optional: VGGT-Omega DINOv3 patch_embed (frozen) — see model.py
        # for the full design rationale.
        # ------------------------------------------------------------------
        omega_state_dict = None
        if dino_backbone == 'omega':
            sys.path.append(os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'thirdparty', 'vggt-omega'
            ))
            from vggt_omega.models.aggregator import _build_patch_embed
            self.omega_patch_embed = _build_patch_embed(patch_size=16, embed_dim=1024)
            for p in self.omega_patch_embed.parameters():
                p.requires_grad_(False)
            if os.path.isfile(omega_ckpt):
                ckpt = torch.load(omega_ckpt, map_location='cpu', weights_only=False)
                omega_state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
                prefix = 'aggregator.patch_embed.'
                pe_sd = {k[len(prefix):]: v for k, v in omega_state_dict.items()
                          if k.startswith(prefix)}
                msg = self.omega_patch_embed.load_state_dict(pe_sd, strict=False)
                print(f"[OpenSe3rDA3] Omega patch_embed loaded: "
                      f"{len(msg.missing_keys)} missing, "
                      f"{len(msg.unexpected_keys)} unexpected")
            else:
                print(f"[OpenSe3rDA3] WARN: omega_ckpt not found at {omega_ckpt}")
            # Omega uses ImageNet (ResNet) norm — same as DA3; reuse those
            # buffers, but expose under the new name for clarity / symmetry.
            self.register_buffer(
                '_omega_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                '_omega_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                persistent=False,
            )
        else:
            self.omega_patch_embed = None

        # ------------------------------------------------------------------
        # Optional: LingBot-Vision ViT-L/16 (frozen) — replaces DINOv2 as the
        # CausalBranch appearance stream.  Unlike 'radio' this stays
        # dual-stream: the semantic backbone (TIPS/...) still supplies the
        # input prior + DPT cond + distillation GT.  Loader freezes + evals
        # the model and returns its embed dim (ViT-L = 1024, matches the
        # CausalBranch DINO slot exactly — no projector).
        # ------------------------------------------------------------------
        if dino_backbone == 'lbotv':
            sys.path.append(os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'thirdparty', 'lingbot-vision'
            ))
            from lingbot_vision import load_pretrained_backbone
            self.lbotv_backbone, _lbotv_dim = load_pretrained_backbone(
                lbotv_ckpt, variant='large', device='cpu',
                dtype=torch.float32, verbose=True)
            if _lbotv_dim != 1024:
                raise ValueError(
                    f"lbotv backbone embed_dim {_lbotv_dim} != 1024; the "
                    f"CausalBranch DINO slot expects ViT-L (use variant='large')")
            # LingBot-Vision expects ImageNet-normalized input (same stats as omega/DA3).
            self.register_buffer(
                '_lbotv_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                '_lbotv_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
                persistent=False,
            )
        else:
            self.lbotv_backbone = None

        # ------------------------------------------------------------------
        # Frozen semantic backbone (LSeg / PE-Core / CLIP)
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
        # A' semantic-based mask-quality head (mirror of OpenSe3r; see model.py).
        self.use_sem_conf_head = bool(use_sem_conf_head)
        if self.use_sem_conf_head:
            self.sem_conf_head = nn.Sequential(
                nn.Linear(sem_expand, 256), nn.GELU(), nn.Linear(256, 1))
            self._sc_text_emb = None
        if semantic_backbone == 'lseg':
            self.lseg = self.semantic_backbone.lseg

        # dino_backbone='radio' → RADIO also supplies the CausalBranch appearance
        # tokens (replacing VGGT DINOv2), so the model runs a SINGLE encoder for
        # both appearance and semantics (no VGGT-DINO + TIPS dual pass).
        # radio mode: CausalBranch input is a SINGLE RADIO raw patch-feature stream
        # (no DINO⊕semantic-prior concat).  The SigLIP dense (SCGA+head_mlp) is used
        # ONLY as the distillation GT, never as an input → so we also drop the
        # semantic-prior concat (lseg) and the DPT head semantic conditioning.
        _causal_lseg_dim = prior_dim
        if self.dino_backbone == 'radio':
            if not hasattr(self.semantic_backbone, 'forward_features'):
                raise ValueError(
                    "dino_backbone='radio' requires a RADIO semantic_backbone "
                    "(--semantic_backbone radio-v4-*); it must expose "
                    "forward_features / appearance_dim.")
            dino_token_dim = int(self.semantic_backbone.appearance_dim)
            _causal_lseg_dim = None                       # single-source input
            if use_head_sem_cond:
                print("[OpenSe3rDA3] dino_backbone=radio → forcing use_head_sem_cond=False "
                      "(no separate semantic prior; semantics learned from RADIO raw + GT)")
            use_head_sem_cond = False
            self.use_head_sem_cond = False
            print(f"[OpenSe3rDA3] dino_backbone=radio → CausalBranch single input: RADIO raw "
                  f"({dino_token_dim}-d) resampled to the patch-{patch_size} grid; dense→GT only")
        else:
            dino_token_dim = 1024

        # ------------------------------------------------------------------
        # Trainable: DepthEncoder (+ optional PointEncoder for point_cond)
        # ------------------------------------------------------------------
        self.depth_encoder = DepthEncoder(out_dim=1024)
        self.point_cond = point_cond
        if point_cond:
            self.point_encoder = PointEncoder(out_dim=1024)

        # ------------------------------------------------------------------
        # Trainable: CausalBranch (unchanged from OpenSe3r)
        # ------------------------------------------------------------------
        self.causal_branch = CausalBranch(
            dino_token_dim=dino_token_dim,
            lseg_dim=_causal_lseg_dim,
            causal_dim=1024,
            n_blocks=n_blocks,
            depth_cross_every=depth_cross_every,
            patch_size=patch_size,
            rope_mode=rope_mode,
            rope_voxel_size=rope_voxel_size,
            depth_cond=(depth_cond or point_cond),   # point_cond reuses the depth-cross-attn path
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

        # Optional 3D position encoder for the CausalBranch input residual.
        self.pos3d_cond = pos3d_cond
        if pos3d_cond:
            from .pos_encoders import PosEncoder3D
            self.pos3d_encoder = PosEncoder3D(out_dim=1024)
        # Store query_init_type so forward() can pick instance head correctly.
        self.query_init_type = query_init_type

        if init_from_vggt and omega_state_dict is not None:
            print("[OpenSe3rDA3] Initializing CausalBranch from VGGT-Omega weights...")
            self.causal_branch.init_from_omega(omega_state_dict)
        elif init_from_vggt:
            print("[OpenSe3rDA3] Initializing CausalBranch from VGGT weights...")
            self.causal_branch.init_from_vggt(self.vggt)

        # ------------------------------------------------------------------
        # Free unused VGGT weights now that CausalBranch is initialized.
        # Geometry comes from DA3 in this backend, so VGGT is not used for geo.
        #   - dino_backbone == 'omega': VGGT is unused at runtime entirely
        #     (omega_patch_embed produces the DINO tokens) → drop it wholesale.
        #   - otherwise: runtime only calls aggregator.encode_patch_tokens, so
        #     keep aggregator.patch_embed and drop the attention trunk + heads.
        # Done before .to(device) so the dropped weights never reach the GPU.
        # ------------------------------------------------------------------
        if self.dino_backbone == 'omega':
            self.vggt = None
            print("[OpenSe3rDA3] dino_backbone=omega → dropped VGGT entirely "
                  "(unused at runtime; CausalBranch init'd from omega weights)")
        elif self.dino_backbone == 'radio':
            self.vggt = None
            print("[OpenSe3rDA3] dino_backbone=radio → dropped VGGT entirely "
                  "(RADIO supplies DINO tokens; CausalBranch already init'd from VGGT)")
        elif self.dino_backbone == 'lbotv':
            self.vggt = None
            print("[OpenSe3rDA3] dino_backbone=lbotv → dropped VGGT entirely "
                  "(LingBot-Vision supplies DINO tokens; CausalBranch already init'd from VGGT)")
        else:
            freed = trim_vggt_modules(self.vggt, keep_geometry=False)
            print(f"[OpenSe3rDA3] Freed unused VGGT submodules (geometry from DA3): {freed}")

        # ------------------------------------------------------------------
        # Optional conditioning encoders (only built when their flag is set).
        # ------------------------------------------------------------------
        if pose_cond:
            self.pose_encoder = PoseTokenEncoder(in_dim=13, dim=1024)
        if plucker_cond:
            self.plucker_encoder = PluckerEncoder(
                dim=1024, num_freqs=plucker_num_freqs, use_fourier=True,
            )

        # ------------------------------------------------------------------
        # Trainable: Semantic & Instance DPT heads (unchanged)
        # ------------------------------------------------------------------
        dpt_layer_idx = [2, 5, 8, 11]

        self.semantic_head = FeatureDPTHead(
            causal_dim=sem_causal_dim,
            out_channels=sem_dim,
            expand_to=sem_expand,
            intermediate_layer_idx=dpt_layer_idx,
            patch_size=patch_size,
            # `use_head_sem_cond=False` → no cond mechanism built at all (smaller head,
            # forces CausalBranch to carry all semantic information).
            lseg_dim=prior_dim if use_head_sem_cond else None,
            lseg_proj_dim=128,
            use_pixel_shuffle_head=head_pixel_shuffle,
        )

        if instance_head_type == 'query':
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
                # When query_init_type='external', queries come from CausalBranch
                # (dim = causal_dim = 1024) and must be projected to d_model.
                # FastSAM seed queries arrive already at d_model (no projection).
                ext_in_dim = (query_d_model if fastsam_query
                              else (1024 if query_init_type == 'external' else None))
                self.instance_head = Mask2FormerQueryInstanceHead(
                    causal_dim     = ins_causal_dim,
                    num_queries    = query_num_queries,
                    d_model        = query_d_model,
                    c_mask         = query_c_mask,
                    descriptor_dim = query_descriptor_dim,
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
            # Learned pairwise merge head (detached probe; same-instance classifier).
            self.use_merge_head = bool(use_merge_head)
            if self.use_merge_head:
                from .merge_head import MergeHead
                self.merge_head = MergeHead(desc_dim=query_descriptor_dim, sem_dim=sem_dim)
        else:
            self.use_merge_head = False
            self.instance_head = FeatureDPTHead(
                causal_dim=ins_causal_dim,
                out_channels=ins_dim,
                expand_to=None,
                intermediate_layer_idx=dpt_layer_idx,
                patch_size=patch_size,
                use_pixel_shuffle_head=head_pixel_shuffle,
            )

        # ── FastSAM-region-seeded query modules (gated; see OpenSe3r) ─────────
        self.fastsam_query = bool(fastsam_query)
        self.fastsam_max_regions = int(fastsam_max_regions)
        if self.fastsam_query:
            if not (instance_head_type == 'query' and query_arch == 'mask2former'):
                raise ValueError("fastsam_query requires instance_head_type='query' "
                                 "and query_arch='mask2former'.")
            if query_init_type != 'external':
                raise ValueError("fastsam_query requires query_init_type='external'.")
            if num_instance_queries and num_instance_queries > 0:
                raise ValueError("fastsam_query is mutually exclusive with "
                                 "num_instance_queries>0.")
            if query_num_queries != fastsam_max_regions:
                raise ValueError(f"query_num_queries ({query_num_queries}) must equal "
                                 f"fastsam_max_regions ({fastsam_max_regions}).")
            from .region_pool import GeoAwareRegionPooling
            from .fastsam_seg import FastSAMSegmenter
            self.region_pool = GeoAwareRegionPooling(
                causal_dim=ins_causal_dim, sem_dim=sem_dim, d_model=query_d_model,
            )
            self._fastsam_seg = FastSAMSegmenter(
                ckpt_path=fastsam_ckpt, conf=fastsam_conf, iou=fastsam_iou,
                max_regions=fastsam_max_regions,
            )

    # ------------------------------------------------------------------
    # Internal: DA3 geometry pass
    # ------------------------------------------------------------------

    def _da3_forward(self, images_01: torch.Tensor, **kwargs):
        """
        Run frozen DA3 net to obtain depth, intrinsics, and c2w poses.

        Extra kwargs (e.g. ``depth_z``, ``intrinsics``) are accepted but
        ignored here — DA3 is a pure image-only backbone.  Subclasses such
        as ``OpenSe3rMapAnything`` override this method and DO consume the
        extra inputs for multi-modal inference.

        Returns the geometry plus a small `meta` dict propagating DA3's
        `is_metric` / `scale_factor` fields when the loaded backbone is the
        nested (metric) variant. For canonical-only backbones these default
        to (False, 1.0).

        Reference view: forced to ``"first"`` so DA3 uses the user's first input
        frame as the world origin (i.e. ``c2w[0] ≈ identity``).  DA3's default
        is ``"saddle_balanced"``, which picks a feature-balanced view as origin
        — this would silently make ``c2w[k] ≈ identity`` for some k != 0,
        which breaks any SLAM/aligner that assumes first-input-as-world (VGGT,
        AMB3R coordinate_alignment, etc.).
        """
        images_norm = (images_01 - self._da3_mean) / self._da3_std

        with torch.no_grad():
            # Call the inner Net directly (DepthAnything3.forward uses
            # torch.inference_mode, which is incompatible with autograd contexts).
            out = self.da3.model(
                images_norm,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=[],
                infer_gs=False,
                use_ray_pose=False,
                ref_view_strategy="first",
            )

        depth      = out['depth']       # [B, T, H, W]
        depth_conf = out['depth_conf']  # [B, T, H, W]
        extrinsics = out['extrinsics']  # [B, T, 3, 4] or [B, T, 4, 4] (w2c)
        intrinsics = out['intrinsics']  # [B, T, 3, 3]

        # DA3's cam_dec returns extrinsics packed as (3, 4); pad to (4, 4) before inversion.
        c2w = affine_inverse(as_homogeneous(extrinsics))

        meta = {
            'is_metric':    bool(int(out.get('is_metric', 0))),
            'scale_factor': float(out.get('scale_factor', 1.0)),
        }
        return depth, depth_conf, c2w, intrinsics, meta

    # ------------------------------------------------------------------
    # Helpers (mirror OpenSe3r)
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _sc_get_text_emb(self, device):
        """ScanNet200 text emb (frozen backbone), mean-subtracted + L2-normed."""
        if self._sc_text_emb is None:
            from amb3r.tools.semantic_vis_utils import build_text_embeddings_via_backbone
            from amb3r.tools.scannet200_constants import CLASS_LABELS_200
            t = build_text_embeddings_via_backbone(
                self.semantic_backbone, list(CLASS_LABELS_200)).float().to(device)
            t = F.normalize(t, dim=-1)
            self._sc_text_emb = F.normalize(t - t.mean(0, keepdim=True), dim=-1)
        return self._sc_text_emb

    def _forward_sem_conf(self, sem_student, sem_gt, mask_logits, want_target):
        """A' semantic mask-quality head — mirror of OpenSe3r._forward_sem_conf."""
        B, T, C, _, _ = sem_student.shape
        N = mask_logits.shape[2]
        H = W = 64
        sem_student = _safe_interp_bilinear(
            sem_student.reshape(B * T, C, *sem_student.shape[-2:]),
            (H, W)).reshape(B, T, C, H, W)
        ml = F.interpolate(mask_logits.reshape(B * T, N, *mask_logits.shape[-2:]),
                           size=(H, W), mode='bilinear', align_corners=False).reshape(B, T, N, H, W)
        soft = ml.sigmoid()
        msum = soft.flatten(3).sum(-1).clamp(min=1.0)
        pooled = torch.einsum('btnhw,btchw->btnc', soft, sem_student.detach()) / msum[..., None]
        sem_conf_logit = self.sem_conf_head(pooled).squeeze(-1)
        target = None
        if want_target and sem_gt is not None:
            with torch.no_grad():
                txt = self._sc_get_text_emb(sem_gt.device)
                sem_gt = _safe_interp_bilinear(
                    sem_gt.float().reshape(B * T, C, *sem_gt.shape[-2:]),
                    (H, W)).reshape(B, T, C, H, W)
                sg = F.normalize(sem_gt, dim=2)
                sim = torch.einsum('kc,btchw->btkhw', txt, sg)
                s_map, c_map = sim.max(2).values, sim.argmax(2)
                mean_s = torch.einsum('btnhw,bthw->btn', soft, s_map) / msum
                K = txt.shape[0]
                oneh = F.one_hot(c_map.reshape(B, T, -1), K).to(soft.dtype)
                hist = torch.einsum('btnl,btlk->btnk', soft.reshape(B, T, N, -1), oneh)
                dom_frac = hist.max(-1).values / hist.sum(-1).clamp(min=1.0)
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
            geo_source:     'model' (default) runs the frozen DA3 geometry
                            backbone.  'gt' skips DA3 entirely and builds the
                            geometry conditioning from GT keys in `frames`
                            (depthmap / camera_pose / camera_intrinsics / pts3d /
                            valid_mask), re-anchored to frame 0.  Use 'gt' for
                            GT-geometry training (pair with `_skip_geo_backbone`
                            so DA3 is never built / never occupies VRAM).
            mode:           'causal'       (training; causal mask within chunk)
                          | 'full'         (non-streaming eval; fully bidirectional)
                          | 'chunk_bidir'  (streaming inference; bidirectional
                                            within current chunk, causal across
                                            chunks via kv-cache)
            kv_cache_list:  KV-cache for streaming inference (None = fresh start)
            return_poses:   if True, also return DA3 c2w poses [B, T, 4, 4]
            extr_rel_override: Optional [B, T, 4, 4] c2w in the SLAM global frame
                          (anchored to sequence frame 0).  When set, the model
                          uses this directly for `pose_to_features` /
                          `compute_plucker_rays` instead of running
                          `make_relative_pose(c2w)` on the chunk-local DA3 poses.
                          REQUIRED for streaming inference with `pose_cond`:
                          DA3 outputs c2w with the chunk's first frame as
                          identity, so without override every new chunk would
                          re-anchor pose tokens to its own first frame, which
                          contradicts the past pose tokens already in the
                          KV-cache.  Training is unaffected — training does a
                          single forward over the whole sequence so chunk[0]
                          coincides with the global origin.
            world_points_override: Optional [B, T, H, W, 3] world_points already
                          expressed in the SLAM global frame.  When set, replaces
                          the DA3-unprojected (chunk-local) world_points used for
                          `pos3d_cond`, the box-head conditioning, and the
                          returned `predictions['world_points']`.  REQUIRED for
                          streaming inference with `pos3d_cond` to keep the
                          3D-PE tokens consistent with the SLAM-global pose
                          tokens (otherwise pos3d uses DA3's chunk-local frame,
                          which disagrees with `extr_rel_override`'s frame).
            box_scale_override: Optional [B] scalar per batch element to use in
                          place of the chunk-level depth median for depth /
                          world_points / pose-translation normalisation.  Use
                          this to pin a sequence-global scale (e.g. captured
                          from the init chunk or derived from GT depths) so
                          the DepthEncoder / pos3d / pose-token normalisation
                          stays stable across chunks instead of fluctuating
                          with each chunk's local depth distribution.

        Returns:
            predictions dict matching OpenSe3r's schema.
        """
        images_neg1 = frames['images']                             # [B, T, 3, H, W] in [-1,1]
        images_01   = (images_neg1 + 1.0) / 2.0
        B, T, _, H, W = images_01.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # ------------------------------------------------------------------
        # GT-geometry path: skip DA3, build conditioning from GT keys in frames.
        # All geometry is frame-0-anchored (matching DA3's chunk-local frame),
        # so the trainable branch is identical whether geometry is GT or DA3.
        # ------------------------------------------------------------------
        if geo_source == 'gt':
            depth_gt, c2w_rel, K_gt, wp_gt, conf_gt, box_scale_gt = \
                build_gt_geometry_from_frames(frames, anchor_mode=self.gt_anchor_mode)
            return self._run_semantic_branch(
                frames, images_01,
                depth             = depth_gt,
                world_points_conf = conf_gt,
                c2w               = c2w_rel,
                intrinsics        = K_gt,
                world_points      = wp_gt,
                da3_meta          = {'is_metric': True, 'scale_factor': 1.0},
                kv_cache_list     = kv_cache_list,
                mode              = mode,
                return_poses      = return_poses,
                extr_rel_override = c2w_rel,          # already frame-0 anchored
                box_scale_override= box_scale_gt,
                debug_aabb_dump_dir    = debug_aabb_dump_dir,
                debug_aabb_dump_prefix = debug_aabb_dump_prefix,
            )
        if geo_source != 'model':
            raise ValueError(f"geo_source must be 'model' or 'gt', got {geo_source!r}")
        if getattr(self, 'da3', None) is None:
            raise RuntimeError(
                "geo_source='model' requires the DA3 backbone, but it was not "
                "built (_skip_geo_backbone=True). Use geo_source='gt' or rebuild "
                "the model without _skip_geo_backbone."
            )

        # ------------------------------------------------------------------
        # Frozen DA3 geometry: depth + c2w + intrinsics → unproject to world points.
        # Pass-through `depth_z` / `intrinsics` if present in `frames` so
        # multi-modal-capable subclasses (MapAnything) can consume them.
        # ------------------------------------------------------------------
        geo_kwargs = {
            k: frames[k] for k in ('depth_z', 'intrinsics') if k in frames
        }
        depth, depth_conf, c2w, intrinsics, da3_meta = self._da3_forward(
            images_01, **geo_kwargs,
        )
        # unproject_depth expects depth shape (B, T, H, W, 1)
        with torch.amp.autocast('cuda', enabled=False):
            world_points = unproject_depth(
                depth.unsqueeze(-1).float(),
                intrinsics.float(),
                c2w=c2w.float(),
            )  # [B, T, H, W, 3]   in DA3 chunk-local frame (c2w[0] = I)

        # Pipeline-supplied world_points in SLAM-global frame override the
        # chunk-local unprojection (see docstring).  Applied here so every
        # downstream consumer (pos3d_cond, box head, predictions output) sees
        # the same SLAM-global world_points.
        if world_points_override is not None:
            if world_points_override.shape != world_points.shape:
                raise ValueError(
                    f"world_points_override shape {tuple(world_points_override.shape)} "
                    f"!= DA3 world_points shape {tuple(world_points.shape)}"
                )
            world_points = world_points_override.to(
                world_points.device, dtype=world_points.dtype,
            )

        return self._run_semantic_branch(
            frames, images_01,
            depth             = depth,
            world_points_conf = depth_conf.unsqueeze(-1),     # [B,T,H,W,1]
            c2w               = c2w,
            intrinsics        = intrinsics,
            world_points      = world_points,
            da3_meta          = da3_meta,
            kv_cache_list     = kv_cache_list,
            mode              = mode,
            return_poses      = return_poses,
            extr_rel_override  = extr_rel_override,
            box_scale_override = box_scale_override,
            debug_aabb_dump_dir    = debug_aabb_dump_dir,
            debug_aabb_dump_prefix = debug_aabb_dump_prefix,
        )

    # ------------------------------------------------------------------
    # Streaming-SLAM helper — runs ONLY the trainable semantic branch.
    # Skips the frozen DA3 depth/pose pass; the caller is expected to
    # have already run `forward_geometry` and (in SLAM) aligned the
    # result to the SLAM-global frame before calling this.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_semantic(
        self,
        frames: Dict[str, torch.Tensor],
        *,
        depth:              torch.Tensor,            # [B, T, H, W]   camera-frame Z (frame-invariant)
        intrinsics:         torch.Tensor,            # [B, T, 3, 3]
        c2w:                torch.Tensor,            # [B, T, 4, 4]   SLAM-global c2w
        world_points:       torch.Tensor,            # [B, T, H, W, 3]  SLAM-global
        world_points_conf:  Optional[torch.Tensor] = None,  # [B, T, H, W, 1]
        kv_cache_list:      Optional[List] = None,
        mode:               str = 'causal',
        box_scale_override: Optional[torch.Tensor] = None,
        return_poses:       bool = True,
        is_metric:          bool = False,
        scale_factor:       float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Run the trainable semantic / instance branch on a pre-computed
        geometry chunk.  Use this in streaming SLAM (pred-mode) to avoid
        the redundant per-chunk DA3 pass that `forward_chunk` triggers.

        All geometry inputs are expected to be in the SLAM-global frame
        already — this method does NOT apply `make_relative_pose` and does
        NOT internally support `extr_rel_override` / `world_points_override`
        (the user-supplied c2w + world_points play that role directly).

        Args:
            frames:            dict with 'images' [B,T,3,H,W] in [-1,1]
            depth:             [B,T,H,W]   typically `geo_res['depth']` from
                               a preceding `forward_geometry` call
            intrinsics:        [B,T,3,3]   `geo_res['intrinsics']`
            c2w:               [B,T,4,4]   SLAM-global c2w of the chunk's frames
                               (e.g. `memory.poses[start:end+1]`)
            world_points:      [B,T,H,W,3] SLAM-global per-pixel points
                               (e.g. `memory._pending_pts_new` after update_geo)
            world_points_conf: [B,T,H,W,1] from `geo_res['world_points_conf']`;
                               defaults to ones if None
            box_scale_override / kv_cache_list / mode / return_poses:
                               same semantics as `forward()`
            is_metric / scale_factor:
                               propagate the geo backbone's metric flags into
                               the predictions dict (caller knows; the model
                               doesn't re-run DA3 here)
        """
        images_neg1 = frames['images']
        images_01   = (images_neg1 + 1.0) / 2.0
        B, T, _, H, W = images_01.shape

        if world_points_conf is None:
            world_points_conf = torch.ones(
                B, T, H, W, 1, device=depth.device, dtype=depth.dtype,
            )

        return self._run_semantic_branch(
            frames, images_01,
            depth             = depth,
            world_points_conf = world_points_conf,
            c2w               = c2w,
            intrinsics        = intrinsics,
            world_points      = world_points,
            da3_meta          = {'is_metric': is_metric,
                                 'scale_factor': scale_factor},
            kv_cache_list     = kv_cache_list,
            mode              = mode,
            return_poses      = return_poses,
            # User's c2w is already in SLAM-global → reuse it as the
            # override so pose_token / plucker skip `make_relative_pose`.
            extr_rel_override  = c2w,
            box_scale_override = box_scale_override,
        )

    # ------------------------------------------------------------------
    # Post-geometry shared body.  forward() and forward_semantic() both
    # converge here once depth / c2w / intrinsics / world_points are in
    # hand (whether predicted by DA3 or supplied by the caller).
    # ------------------------------------------------------------------

    def _run_semantic_branch(
        self,
        frames:            Dict[str, torch.Tensor],
        images_01:         torch.Tensor,
        *,
        depth:             torch.Tensor,                  # [B,T,H,W]
        world_points_conf: torch.Tensor,                  # [B,T,H,W,1]
        c2w:               torch.Tensor,
        intrinsics:        torch.Tensor,
        world_points:      torch.Tensor,
        da3_meta:          Dict,
        kv_cache_list:     Optional[List],
        mode:              str,
        return_poses:      bool,
        extr_rel_override:  Optional[torch.Tensor],
        box_scale_override: Optional[torch.Tensor],
        debug_aabb_dump_dir:    Optional[str] = None,
        debug_aabb_dump_prefix: str = '',
    ) -> Dict[str, torch.Tensor]:
        B, T, _, H, W = images_01.shape
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # ------------------------------------------------------------------
        # Per-frame patch tokens for CausalBranch (vggt DINOv2 or omega DINOv3).
        # Both are frozen and operate per-frame → online-safe.
        # ------------------------------------------------------------------
        _radio_dense = None
        with torch.no_grad():
            if self.dino_backbone == 'omega':
                imgs_flat   = images_01.reshape(B * T, 3, H, W)
                imgs_norm   = (imgs_flat - self._omega_mean) / self._omega_std
                omega_out   = self.omega_patch_embed(imgs_norm, is_training=True)
                dino_tokens_flat = omega_out['x_norm_patchtokens']                   # [B*T, N16, 1024]
                dino_tokens = dino_tokens_flat.reshape(B, T, -1, dino_tokens_flat.shape[-1])
            elif self.dino_backbone == 'lbotv':
                imgs_flat   = images_01.reshape(B * T, 3, H, W)
                imgs_norm   = (imgs_flat - self._lbotv_mean) / self._lbotv_std
                lbotv_out   = self.lbotv_backbone(imgs_norm, is_training=True)
                dino_tokens_flat = lbotv_out['x_norm_patchtokens']                   # [B*T, N16, 1024]
                dino_tokens = dino_tokens_flat.reshape(B, T, -1, dino_tokens_flat.shape[-1])
            elif self.dino_backbone == 'radio':
                # ONE RADIO forward → raw appearance (→ single-source DINO tokens) and,
                # in training, dense SigLIP (SCGA+head_mlp) kept ONLY as the distillation
                # GT (never an input).  Raw is resampled to the model's patch grid.
                imgs_flat = images_01.reshape(B * T, 3, H, W)
                # Dense SigLIP is the semantic-loss GT — needed whenever a loss is
                # computed (train AND val eval), signalled by instance_mask being
                # present.  NOT self.training-only (val eval runs with grad off but
                # still needs the GT), and NOT in SLAM inference (no instance_mask).
                _want_dense = self.training or (
                    isinstance(frames, dict) and frames.get('instance_mask') is not None)
                raw_chw, _radio_dense = self.semantic_backbone.forward_features(
                    imgs_flat, want_dense=_want_dense)
                # patch-16 native: with mult-16 input, RADIO's grid already == the
                # model's patch-16 grid → no resample (identity).  Fall back to a
                # bilinear resample only if they differ (non-mult-16 resolution).
                if raw_chw.shape[-2:] != (patch_h, patch_w):
                    raw_chw = F.interpolate(raw_chw, (patch_h, patch_w),
                                            mode='bilinear', align_corners=False)
                dino_tokens_flat = raw_chw.flatten(2).transpose(1, 2).contiguous()  # [BT, N, appearance_dim]
                dino_tokens = dino_tokens_flat.reshape(B, T, -1, dino_tokens_flat.shape[-1])
            else:
                patch_tokens_dict = self.vggt.aggregator.encode_patch_tokens(images_01)
                dino_tokens_flat = patch_tokens_dict['x_norm_patchtokens']
                dino_tokens = dino_tokens_flat.reshape(B, T, -1, dino_tokens_flat.shape[-1])

        # ------------------------------------------------------------------
        # Frozen semantic backbone: prior dense feature.
        #   - radio mode: NO input prior — the CausalBranch runs on RADIO raw only;
        #     the SigLIP dense (_radio_dense) is used solely as the distillation GT.
        #   - otherwise: run the semantic backbone for the input/DPT-cond prior.
        # ------------------------------------------------------------------
        imgs_flat = images_01.reshape(B * T, 3, H, W)
        if self.dino_backbone == 'radio':
            lseg_feat = None
        else:
            with torch.no_grad():
                if self.prior_method == 'proxy_clip':
                    lseg_feat = self.semantic_backbone.forward_prior(
                        imgs_flat,
                        dino_tokens = dino_tokens_flat,
                        dino_grid   = (patch_h, patch_w),
                    )
                elif self.prior_method == 'native':
                    lseg_feat = self.semantic_backbone.forward_prior_native(imgs_flat)
                else:                                         # 'lseg'
                    lseg_feat = self.semantic_backbone.forward_prior(imgs_flat)
                if lseg_feat.shape[-2:] != (H, W):
                    lseg_feat = F.interpolate(
                        lseg_feat, size=(H, W),
                        mode='bilinear', align_corners=False,
                    )

        # ------------------------------------------------------------------
        # Depth encoder → patch-level geometry tokens.
        # `depth_norm_mode`: 'chunk' (default) shares one divisor across all
        # T frames in a batch element (= box_scale) so cross-frame attention
        # in the CausalBranch sees consistent depth tokens; 'frame' uses a
        # per-frame divisor (legacy, kept for ckpt back-compat).  See
        # OpenSe3r for the rationale.
        # ------------------------------------------------------------------
        depth_med_per_frame = depth.median(dim=-1, keepdim=True).values \
                                    .median(dim=-2, keepdim=True).values        # [B, T, 1, 1]
        # Chunk-level median (also the box_scale used by the box head / SLAM).
        box_scale = depth_med_per_frame.flatten(1).median(dim=-1) \
                                         .values.clamp(min=0.1)                  # [B]

        # Pipeline-supplied sequence-global scale overrides the chunk-level
        # median.  Affects depth normalisation, pos3d normalisation, and
        # scene_scale used for pose/plucker translation normalisation — see
        # the `--global_scale_mode` SLAM flag.
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
            # Metric mode: disable ALL scale normalisation.  box_scale is the
            # single divisor for depth / point / pos3d / box-head (and the box
            # GT in the loss reads preds['box_scale']) → forcing it to 1 keeps
            # EVERYTHING in METRIC scale, and scale becomes an information
            # source.  pred & GT box both stay metric → loss is consistent.
            box_scale = torch.ones_like(box_scale)

        if self.depth_norm_mode in ('chunk', 'none'):
            depth_norm = depth / box_scale.view(-1, 1, 1, 1)
        else:                                                                   # 'frame'
            depth_norm = depth / (depth_med_per_frame + 1e-6)

        if self.point_cond:
            # Feed NORMALIZED per-pixel world points (xyz) through the PointEncoder,
            # routed into the same `depth_tokens` slot → DepthCrossAttn injects 3D
            # position instead of depth.  world_points: [B,T,H,W,3] (frame-0 anchored
            # for geo_source=gt; DA3-canonical otherwise).
            wp_full = world_points.reshape(B * T, H, W, 3).permute(0, 3, 1, 2)   # [BT,3,H,W]
            bs_pt = box_scale.clamp(min=1e-3).view(B, 1, 1, 1).expand(B, T, 1, 1).reshape(B * T, 1, 1, 1)
            wp_norm = wp_full / bs_pt
            depth_tokens = self.point_encoder(wp_norm.to(depth.dtype), patch_h, patch_w)
        else:
            depth_tokens = self.depth_encoder(
                depth_norm.reshape(B * T, 1, H, W), patch_h, patch_w
            )

        # ------------------------------------------------------------------
        # (Idea 1) Optional 3D positional encoding tokens.
        # Pool world_points to patch grid, normalize by box_scale, then PE.
        # ------------------------------------------------------------------
        pos3d_tokens = None
        if self.pos3d_cond:
            # world_points: [B, T, H, W, 3] in CANONICAL (DA3 frame).  Pool to patch.
            wp_full = world_points.reshape(B * T, H, W, 3).permute(0, 3, 1, 2)  # [BT, 3, H, W]
            wp_patch = F.adaptive_avg_pool2d(wp_full, (patch_h, patch_w))         # [BT, 3, ph, pw]
            wp_patch = wp_patch.permute(0, 2, 3, 1)                                # [BT, ph, pw, 3]
            wp_patch = wp_patch.reshape(B * T, patch_h * patch_w, 3)
            # NORMALIZE by box_scale → ~unit range (scale-agnostic).
            bs = box_scale.clamp(min=1e-3).view(B, 1, 1, 1)
            bs = bs.expand(B, T, 1, 1).reshape(B * T, 1, 1)
            wp_patch_norm = wp_patch / bs
            pos3d_tokens = self.pos3d_encoder(wp_patch_norm.to(depth_tokens.dtype))
                                                                                    # [BT, N_patch, 1024]

        # ------------------------------------------------------------------
        # Optional: pose / Plücker conditioning tokens.
        # DA3 already provides c2w + intrinsics directly (no camera_head call
        # needed).  scene_scale = clip-level depth median, shared with the
        # DepthEncoder normalization for cross-backend invariance.
        # ------------------------------------------------------------------
        pose_token     = None
        plucker_tokens = None
        if self.pose_cond or self.plucker_cond:
            scene_scale = box_scale                                          # [B]

            if extr_rel_override is not None:
                # Caller supplies poses already anchored to the SLAM global
                # frame (sequence[0] = identity).  Skip make_relative_pose.
                extr_rel = extr_rel_override.to(c2w.device, dtype=c2w.dtype)
                if extr_rel.shape != c2w.shape:
                    raise ValueError(
                        f"extr_rel_override shape {tuple(extr_rel.shape)} != "
                        f"DA3 c2w shape {tuple(c2w.shape)}"
                    )
            else:
                extr_rel = make_relative_pose(c2w.detach())                   # [B,T,4,4]
            intr_d   = intrinsics.detach()

            if self.pose_cond:
                pose_feats = pose_to_features(
                    extr_rel, intr_d, H, W, scene_scale,
                )                                                             # [B, T, 13]
                pose_token = self.pose_encoder(pose_feats)                    # [B, T, 1024]

            if self.plucker_cond:
                plucker = compute_plucker_rays(
                    extr_rel, intr_d, H, W, self.patch_size, scene_scale,
                )                                                             # [B, T, N_patch, 6]
                plucker_tokens = self.plucker_encoder(plucker)                # [B, T, N_patch, 1024]

        # ------------------------------------------------------------------
        # Trainable: Causal branch
        # ------------------------------------------------------------------
        # Per-patch world coords for 3D RoPE (metric, frame-0 anchored — NOT
        # normalized; voxelization uses the metric rope_voxel_size).
        world_pts_patch = None
        if getattr(self.causal_branch, 'rope_mode', '2d') != '2d':
            wp_p = world_points.reshape(B * T, H, W, 3).permute(0, 3, 1, 2)          # [BT,3,H,W]
            wp_p = F.adaptive_avg_pool2d(wp_p, (patch_h, patch_w))                    # [BT,3,ph,pw]
            world_pts_patch = wp_p.permute(0, 2, 3, 1).reshape(B, T, patch_h * patch_w, 3)

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
        # DPT heads.  patch_start_idx=1 when a pose token is prepended.
        # ------------------------------------------------------------------
        ps_idx_dpt = self.causal_branch.patch_start_idx
        # Optional per-head channel slicing (default = full passthrough).
        sem_tokens_list = self._slice_token_channels(output_tokens_list, self.sem_channels)
        ins_tokens_list = self._slice_token_channels(output_tokens_list, self.ins_channels)

        # `use_head_sem_cond` lets you suppress the prior at the head even when
        # the head was built with a cond mechanism (e.g. for inference-time
        # ablation on a checkpoint trained with cond enabled).  When the head
        # was built without cond (lseg_dim=None), this kwarg is ignored.
        # Activation checkpointing (training only, like CausalBranch): the DPT
        # heads' full-res upsampling activations are the multi-view memory killer
        # (esp. sem_dim=1536 × T × H×W).  Recompute them in backward instead of
        # storing.  Grad-carrying token tensors are passed positionally so
        # use_reentrant=False tracks them.
        _sem_cond = lseg_feat if self.use_head_sem_cond else None
        def _sem_head_fwd(*toks):
            return self.semantic_head(
                list(toks), images_01, patch_start_idx=ps_idx_dpt,
                semantic_cond=_sem_cond)
        if self.training:
            sem_feat, sem_conf, sem_feat_expanded = checkpoint(
                _sem_head_fwd, *sem_tokens_list, use_reentrant=False)
        else:
            sem_feat, sem_conf, sem_feat_expanded = _sem_head_fwd(*sem_tokens_list)

        # Instance head dispatch (mirror of OpenSe3r).
        if self.instance_head_type == 'dense':
            ins_feat, ins_conf, _ = self.instance_head(
                ins_tokens_list, images_01, patch_start_idx=ps_idx_dpt,
            )
            query_out = None
        else:
            ins_feat, ins_conf = None, None
            head_kw = dict(
                output_tokens_list = ins_tokens_list,
                images             = images_01,
                patch_start_idx    = ps_idx_dpt,
            )
            # World-aabb mode (mask2former only) needs world_points + box_scale
            # for the mask-pool + PCA-prior path (both used to NORMALIZE
            # world_points to median ~1 → head is scale-invariant).
            if getattr(self.instance_head, 'box_mode', None) in ('world_aabb', 'world_amodal', 'world_amodal_grow'):
                head_kw['world_points'] = world_points
                head_kw['box_scale']    = box_scale
                # Optional debug PLY dump (passed via model forward kwargs).
                if debug_aabb_dump_dir is not None:
                    head_kw['debug_dump_dir']    = debug_aabb_dump_dir
                    head_kw['debug_dump_prefix'] = debug_aabb_dump_prefix
            # (Idea 2) When head uses external init queries, feed in the
            # processed instance queries from CausalBranch.
            if (getattr(self.instance_head, 'query_init_type', None) == 'external'
                    and instance_q_out is not None):
                head_kw['init_queries'] = instance_q_out
            # FastSAM-region-seeded queries (ESAM-style); overrides init source
            # and emits the region map for the sparse matcher / snap.
            fastsam_region_id = None
            fastsam_region_conf = None
            if self.fastsam_query:
                from .fastsam_seg import build_seed_queries
                seed, fastsam_region_id, fastsam_region_conf = build_seed_queries(
                    self._fastsam_seg, self.region_pool, self.patch_size,
                    self.fastsam_max_regions, images_01, ins_tokens_list,
                    sem_feat, world_points, ps_idx_dpt,
                )
                head_kw['init_queries'] = seed
                if getattr(self.instance_head, 'fastsam_mask_prior', False):
                    head_kw['fastsam_region_id'] = fastsam_region_id
            # Activation checkpointing (training only): the Mask2Former head's
            # full-res mask_logits [B,T,Q,H,W] + upsampling are recomputed in
            # backward instead of stored.  Grad-carrying inputs (token list +
            # init_queries) passed positionally for use_reentrant=False.
            if self.training:
                _n_tok = len(ins_tokens_list)
                _has_iq = 'init_queries' in head_kw
                _grad_in = list(ins_tokens_list) + ([head_kw['init_queries']] if _has_iq else [])
                def _ins_head_fwd(*g):
                    kw = dict(head_kw)
                    kw['output_tokens_list'] = list(g[:_n_tok])
                    if _has_iq:
                        kw['init_queries'] = g[_n_tok]
                    return self.instance_head(**kw)
                query_out = checkpoint(_ins_head_fwd, *_grad_in, use_reentrant=False)
            else:
                query_out = self.instance_head(**head_kw)

        # ------------------------------------------------------------------
        # Semantic GT for the loss (training-time only when supervision != lseg).
        # ------------------------------------------------------------------
        sem_gt = None
        instance_mask = frames.get('instance_mask') if isinstance(frames, dict) else None
        with torch.no_grad():
            if self.dino_backbone == 'radio':
                # Reuse the SigLIP dense already computed in the single RADIO forward
                # (no 2nd encoder pass): per-instance hard-mask mean = the pseudo GT.
                if _radio_dense is not None and instance_mask is not None:
                    sem_gt = self.semantic_backbone.patch_mean_aggregate(
                        _radio_dense, instance_mask, B, T, H, W)
            elif self.sem_supervision == 'lseg':
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
        # Assemble predictions (same schema as OpenSe3r)
        # ------------------------------------------------------------------
        predictions = {
            'world_points':      world_points,                          # [B,T,H,W,3]
            'world_points_conf': world_points_conf,                     # [B,T,H,W,1]
            'depth':             depth,                                 # [B,T,H,W]
            'semantic_feat':     sem_feat,
            'semantic_conf':     sem_conf,
            # Dense instance outputs (only populated in 'dense' mode):
            'instance_feat':     ins_feat,
            'instance_conf':     ins_conf,
            'instance_head_type': self.instance_head_type,
            'is_metric':         da3_meta['is_metric'],
            'scale_factor':      da3_meta['scale_factor'],
            # Always-on so SLAM can capture the first-chunk scale for the
            # `--global_scale_mode first_chunk` flag.
            'box_scale':         box_scale,                             # [B]
        }
        # Query-based instance outputs (only populated in 'query' mode):
        if query_out is not None:
            predictions['mask_logits'] = query_out['mask_logits']
            predictions['objectness']  = query_out['objectness']
            _sc_in = sem_feat_expanded if sem_feat_expanded is not None else sem_feat
            if self.use_sem_conf_head and _sc_in is not None:
                sc_logit, sc_tgt = self._forward_sem_conf(
                    _sc_in, sem_gt, query_out['mask_logits'], want_target=self.training)
                predictions['sem_conf'] = sc_logit
                if sc_tgt is not None:
                    predictions['sem_conf_target'] = sc_tgt
            predictions['descriptor']  = query_out['descriptor']
            if 'fgbg_logits' in query_out:
                predictions['fgbg_logits'] = query_out['fgbg_logits']    # [B,T,N] thing/stuff
            if self.fastsam_query and fastsam_region_id is not None:
                predictions['fastsam_region_id']   = fastsam_region_id
                predictions['fastsam_region_conf'] = fastsam_region_conf
            if query_out.get('descriptor_pathB') is not None:
                predictions['descriptor_pathB'] = query_out['descriptor_pathB']
            # Expose the merge-head module so the loss can train it (detached
            # inputs → no trunk grad) and SLAM can call it. Not a tensor; the
            # loss reads preds['_merge_head'] and ignores it otherwise.
            if getattr(self, 'use_merge_head', False):
                predictions['_merge_head'] = self.merge_head
            if 'box' in query_out:
                # See model.py for the box / box_scale contract.  Head
                # outputs canonical-scale camera-frame box; SLAM/inference
                # multiplies by box_scale (depth median) for the current
                # scene's units.  `box_scale` itself is now always set
                # earlier in the predictions dict (used by the
                # `--global_scale_mode` SLAM path).
                predictions['box']       = query_out['box']                 # [B,T,N,6|12]
                predictions['box_mode']  = query_out.get('box_mode', 'aabb')
            if 'aux_outputs' in query_out:
                predictions['aux_outputs'] = query_out['aux_outputs']
        if sem_gt is not None:
            predictions['_sem_feat_gt'] = sem_gt
            if self.sem_supervision == 'lseg':
                predictions['_lseg_feat_gt'] = sem_gt

        if sem_feat_expanded is not None:
            predictions['semantic_feat_expanded'] = sem_feat_expanded

        # DINO patch tokens (computed anyway; expose for Stage 2 refinement
        # and debug visualisation).  Native shape [B, T, N_patch, dino_dim].
        predictions['dino_patches'] = dino_tokens

        if return_poses:
            predictions['poses'] = c2w                                 # [B,T,4,4] c2w

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
        """DA3-only forward pass for overlap-based chunk alignment.

        Also returns `intrinsics` so that downstream `forward_semantic`
        callers (streaming SLAM that wants to skip a second DA3 pass) can
        feed it back in directly without re-running the head.
        """
        images_neg1 = frames['images']
        images_01   = (images_neg1 + 1.0) / 2.0

        geo_kwargs = {
            k: frames[k] for k in ('depth_z', 'intrinsics') if k in frames
        }
        depth, depth_conf, c2w, intrinsics, da3_meta = self._da3_forward(
            images_01, **geo_kwargs,
        )
        with torch.amp.autocast('cuda', enabled=False):
            world_points = unproject_depth(
                depth.unsqueeze(-1).float(),
                intrinsics.float(),
                c2w=c2w.float(),
            )

        return {
            'world_points':      world_points,
            'world_points_conf': depth_conf.unsqueeze(-1),
            'depth':             depth,
            'poses':             c2w,
            'intrinsics':        intrinsics,
            'is_metric':         da3_meta['is_metric'],
            'scale_factor':      da3_meta['scale_factor'],
        }

    def init_kv_cache(self) -> List:
        # A fresh cache list marks a new streaming sequence — also wipe the
        # optional STAC/H2O temporal KV manager (no-op when not registered).
        self.causal_branch.reset_kv_mgr()
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
