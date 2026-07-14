"""
opense3r/dpt_heads_multiscale.py
==================================
Multi-scale pixel decoder for the Mask2Former-style query instance head.

Wraps the existing `FeatureDPTHead` (full-resolution dense feature output)
and emits FOUR things needed by the Mask2Former-style decoder:

  1. mask_features    [B*T, c_mask, H,    W]      ← full-res, used for
                                                    final mask-logit dot product
                                                    against mask_embed
  2. multi_scale_feats list of three tensors at 1/32, 1/16, 1/8 of (H, W):
       [B*T, c_attn, H/32, W/32]   ← coarse, big-object scale
       [B*T, c_attn, H/16, W/16]
       [B*T, c_attn,  H/8,  W/8]   ← fine, small-object scale
       Each has a learnable "level embedding" added so the decoder knows
       which scale a token comes from when it's flattened together.

The three multi-scale feature maps feed the decoder layers in a CYCLIC
fashion (layer i uses scale i % 3) so every query gets to attend to
every scale across the layer stack.

Design choices:
  * The three scales are produced by adaptive average pooling on the
    full-resolution F_mask (cheap, no extra params).  An optional 1×1
    conv per level projects c_mask → c_attn (typically c_mask == c_attn
    so the conv is identity-init friendly).
  * Level embeddings (one learnable d_model-dim vector per scale) get
    broadcast-added to each level's feature map.
  * 2D sinusoidal positional encoding is added separately inside the
    decoder, NOT here — this module only emits feature maps.

Backward compatibility:
  This module does NOT change `FeatureDPTHead` itself.  The legacy
  query head (`instance_head_query.py`) continues to use `FeatureDPTHead`
  directly; only the new `instance_head_query_m2f.py` uses this wrapper.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_heads import FeatureDPTHead


class MultiScaleFeatureDPTHead(nn.Module):
    """Wraps FeatureDPTHead to also emit multi-scale pooled features.

    Args:
        causal_dim:          input token dim (matches CausalBranch / ins_channels span)
        c_mask:              channel dim of full-res mask features (e.g. 256)
        c_attn:              channel dim of multi-scale attention features (e.g. 256)
        intermediate_layer_idx: DPT input layer indices ([2, 5, 8, 11] default)
        patch_size:          ViT patch size (14 for VGGT)
        scales:              downsample factors for multi-scale features
                             (sorted coarsest→finest by default).
                             Default [32, 16, 8] = Mask2Former's standard.
    """

    def __init__(
        self,
        causal_dim:            int,
        c_mask:                int = 256,
        c_attn:                int = 256,
        intermediate_layer_idx: Optional[List[int]] = None,
        patch_size:            int = 14,
        scales:                Optional[List[int]] = None,
    ):
        super().__init__()
        if intermediate_layer_idx is None:
            intermediate_layer_idx = [2, 5, 8, 11]
        if scales is None:
            scales = [32, 16, 8]

        # Sort coarsest → finest (Mask2Former convention).
        scales = sorted(scales, reverse=True)
        self.scales = scales
        self.num_feature_levels = len(scales)
        self.c_mask = c_mask
        self.c_attn = c_attn

        # Base full-resolution DPT — REQUIRED to be the multi-scale DPT
        # cascade (NOT the pixel-shuffle path).  Mask2Former's transformer
        # decoder relies on rich multi-scale features (input_proj on
        # pooled F_mask + per-level cross-attn); a single-scale
        # pixel-shuffle output would collapse the representation power
        # used for `query_num_feature_levels` cross-attn levels.
        self.base_dpt = FeatureDPTHead(
            causal_dim = causal_dim,
            out_channels = c_mask,
            expand_to = None,
            intermediate_layer_idx = intermediate_layer_idx,
            patch_size = patch_size,
            lseg_dim = None,
        )

        # Per-level 1×1 conv to project c_mask → c_attn.  When c_mask == c_attn,
        # we still keep a learnable projection so each level can specialise.
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_mask, c_attn, kernel_size=1),
                nn.GroupNorm(32, c_attn),
            )
            for _ in range(self.num_feature_levels)
        ])

        # Learnable level embedding (one d_model vector per scale).
        self.level_embed = nn.Embedding(self.num_feature_levels, c_attn)
        nn.init.normal_(self.level_embed.weight, std=0.02)

    def forward(
        self,
        output_tokens_list: List[torch.Tensor],
        images:             torch.Tensor,
        patch_start_idx:    int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Returns:
            mask_features:        [B*T, c_mask, H, W]   full-res
            multi_scale_features: list of [B*T, c_attn, H/s, W/s]
                                  for each s in self.scales (coarsest→finest)
            sizes:                list of (H_s, W_s) tuples (per level)
        """
        # Full-resolution mask features (re-uses your existing DPT decoder).
        F_mask, _, _ = self.base_dpt(
            output_tokens_list, images, patch_start_idx=patch_start_idx,
            semantic_cond=None,
        )
        # F_mask: [B, T, c_mask, H, W]
        B, T = F_mask.shape[:2]
        H, W = F_mask.shape[-2:]
        F_mask_flat = F_mask.reshape(B * T, self.c_mask, H, W)

        # Multi-scale features by adaptive average pooling + 1×1 conv.
        ms_feats: List[torch.Tensor] = []
        sizes:    List[tuple]         = []
        for level, scale in enumerate(self.scales):
            target_h = max(H // scale, 1)
            target_w = max(W // scale, 1)
            pooled = F.adaptive_avg_pool2d(F_mask_flat, (target_h, target_w))
            proj   = self.input_proj[level](pooled)               # [B*T, c_attn, h, w]
            # Add level embedding (broadcast over spatial dims).
            level_vec = self.level_embed.weight[level]            # [c_attn]
            proj = proj + level_vec[None, :, None, None]
            ms_feats.append(proj)
            sizes.append((target_h, target_w))

        return {
            'mask_features':        F_mask_flat,
            'multi_scale_features': ms_feats,
            'sizes':                sizes,
        }
