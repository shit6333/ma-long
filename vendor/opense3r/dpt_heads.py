import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty'))

from vggt.heads.dpt_head import DPTHead as VGGTDPTHead, custom_interpolate


class FeatureDPTHead(nn.Module):
    """
    DPT-based feature extraction head for the CausalBranch output.

    Reuses VGGT's DPTHead internals with a custom forward that optionally injects
    LSeg/CLIP conditioning at each DPT layer (for the semantic head).

    Conditioning pattern (from AMB3R PatchConditionedDPTHead):
      At each of the 4 DPT layers, after extracting token spatial maps:
        1. Pool lseg_feat to patch grid
        2. Project to lseg_proj_dim (128)
        3. Cat with token map and fuse back to causal_dim
        4. Continue normal DPT path

    For the instance head, set lseg_dim=None to skip conditioning entirely.

    Two head variants (selected by `use_pixel_shuffle_head`):
      - DPT (default, backward-compat): multi-scale FeatureFusionBlock chain
        from VGGT's RefineNet, with bilinear upsample at the end.  Expensive
        but accurate.
      - Pixel-shuffle (VGGT-Ω-style): take the deepest intermediate layer,
        optional LSeg fusion, MLP (Conv 1×1 ×2) → pixel-shuffle to full res.
        Skips the whole RefineNet cascade — ~30% training-memory saving;
        quality competitive (Mask2Former and VGGT-Ω both use this).
    """

    _DPT_FEAT_DIM = 256   # DPT's internal fusion output channels

    def __init__(
        self,
        causal_dim: int = 1024,
        out_channels: int = 64,
        expand_to: Optional[int] = None,
        intermediate_layer_idx: List[int] = None,
        patch_size: int = 14,
        # LSeg conditioning (set for semantic head, None for instance head)
        lseg_dim: Optional[int] = None,
        lseg_proj_dim: int = 128,
        # Lightweight head: MLP + pixel_shuffle replaces DPT RefineNet
        # (VGGT-Ω §2.1; ~30% training-memory saving over the DPT cascade).
        use_pixel_shuffle_head: bool = False,
        ps_hidden: int = 512,    # MLP hidden width in the pixel-shuffle path
    ):
        super().__init__()

        if intermediate_layer_idx is None:
            intermediate_layer_idx = [2, 5, 8, 11]

        self.intermediate_layer_idx = intermediate_layer_idx
        self.patch_size = patch_size
        self.use_pixel_shuffle_head = use_pixel_shuffle_head

        # DPT instance only built for legacy / RefineNet path.
        # (Pixel-shuffle path replaces the DPT cascade entirely.)
        if not use_pixel_shuffle_head:
            self.dpt = VGGTDPTHead(
                dim_in=causal_dim,
                patch_size=patch_size,
                feature_only=True,
                intermediate_layer_idx=intermediate_layer_idx,
                pos_embed=True,
            )
        else:
            # Pixel-shuffle path with bottleneck.  Naive design
            #   ps_hidden → (out_channels × patch_size²)   then pixel_shuffle
            # blows up when out_channels is large (e.g. sem_dim=1024 ×
            # patch_size²=196 → 200k channels, ~100M params for a single
            # Conv1×1).  Fix: pixel-shuffle to a SMALL intermediate
            # `ps_mid` channels, then a cheap Conv1×1 projects ps_mid →
            # out_channels at full resolution.
            #
            #   ps_mlp:        causal_dim → ps_hidden            (low-res)
            #   ps_feat_proj:  ps_hidden  → ps_mid × patch_size² (low-res)
            #   pixel_shuffle: → [B*T, ps_mid, H, W]
            #   ps_out_proj:   ps_mid → out_channels             (full-res, 1×1)
            #
            # ps_conf does the analogous thing for the 1-channel conf head.
            ps_mid = min(ps_hidden, 64)        # bottleneck channels
            self.ps_mlp = nn.Sequential(
                nn.Conv2d(causal_dim, ps_hidden, 1),
                nn.GELU(),
                nn.Conv2d(ps_hidden, ps_hidden, 1),
                nn.GELU(),
            )
            self.ps_feat_proj = nn.Conv2d(
                ps_hidden, ps_mid * (patch_size ** 2), 1,
            )
            self.ps_out_proj = nn.Conv2d(ps_mid, out_channels, 1)
            self.ps_conf_proj = nn.Conv2d(
                ps_hidden, 1 * (patch_size ** 2), 1,
            )
            self.ps_norm = nn.LayerNorm(causal_dim)

        # --- Optional LSeg conditioning ---
        self.lseg_dim = lseg_dim
        if lseg_dim is not None:
            self.lseg_proj = nn.Sequential(
                nn.Conv2d(lseg_dim, lseg_proj_dim, 1, bias=False),
                nn.BatchNorm2d(lseg_proj_dim),
                nn.GELU(),
                nn.Conv2d(lseg_proj_dim, lseg_proj_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(lseg_proj_dim),
                nn.GELU(),
            )
            # Number of LSeg fusion points: DPT path has one per layer;
            # pixel-shuffle path needs exactly one (before the MLP).
            n_fusion = 1 if use_pixel_shuffle_head else len(intermediate_layer_idx)
            fusion_in_dim = causal_dim + lseg_proj_dim
            self.fusion_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(fusion_in_dim, causal_dim, 1, bias=False),
                    nn.BatchNorm2d(causal_dim),
                    nn.GELU(),
                )
                for _ in range(n_fusion)
            ])
            # Fallback token when no LSeg cond is provided at inference
            self.null_sem_token = nn.Parameter(torch.zeros(1, lseg_proj_dim, 1, 1))

        # Out-proj / conf head only for DPT path; pixel-shuffle path's
        # ps_feat_proj / ps_conf_proj already produce final-channel output.
        if not use_pixel_shuffle_head:
            self.out_proj = nn.Sequential(
                nn.Conv2d(self._DPT_FEAT_DIM, out_channels, 1),
                nn.GELU(),
            )
            self.conf_head = nn.Sequential(
                nn.Conv2d(self._DPT_FEAT_DIM, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
                nn.Sigmoid(),
            )

        # Optional expander: out_channels → expand_to  (e.g. 64 → 512 for semantic head)
        self.expand_to = expand_to
        if expand_to is not None and expand_to != out_channels:
            self.expander = nn.Linear(out_channels, expand_to)
        else:
            self.expander = None

    def forward(
        self,
        output_tokens_list: List[torch.Tensor],   # list of [B, T, N_patch, causal_dim]
        images: torch.Tensor,                      # [B, T, 3, H, W]
        patch_start_idx: int = 0,
        semantic_cond: Optional[torch.Tensor] = None,  # [B*T, 512, H, W] or [B, T, 512, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            semantic_cond: LSeg features for conditioning.  Only used when
                           lseg_dim was set at construction. Pass None to use
                           the learned null token as fallback.

        Returns:
            feat:          [B, T, out_channels, H, W]
            conf:          [B, T, 1, H, W]
            feat_expanded: [B, T, expand_to, H, W]  or None
        """
        B, T = images.shape[:2]
        H, W = images.shape[-2:]
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size

        # --- Prepare LSeg conditioning patch feature ---
        sem_feat = None
        if self.lseg_dim is not None:
            if semantic_cond is not None:
                if semantic_cond.dim() == 5:
                    sem_flat = semantic_cond.reshape(B * T, *semantic_cond.shape[2:])
                else:
                    sem_flat = semantic_cond            # already [B*T, C, H, W]
                sem_patch = F.adaptive_avg_pool2d(sem_flat, (patch_h, patch_w))
                sem_feat = self.lseg_proj(sem_patch)   # [B*T, lseg_proj_dim, ph, pw]
            else:
                sem_feat = self.null_sem_token.expand(B * T, -1, patch_h, patch_w)

        if self.use_pixel_shuffle_head:
            # ── Lightweight path: deepest layer → (optional LSeg fuse) →
            #    MLP → pixel_shuffle to full res.  Skips RefineNet entirely.
            deepest = self.intermediate_layer_idx[-1]
            x = output_tokens_list[deepest][:, :, patch_start_idx:]  # [B, T, N, dim]
            x = x.reshape(B * T, -1, x.shape[-1])                    # [B*T, N, dim]
            x = self.ps_norm(x)
            x = x.permute(0, 2, 1).reshape(B * T, x.shape[-1], patch_h, patch_w)
            if sem_feat is not None:
                x = self.fusion_projs[0](torch.cat([x, sem_feat], dim=1))
            x = self.ps_mlp(x)                                       # [B*T, ps_hidden, ph, pw]
            # Pixel-shuffle to ps_mid channels at full res, THEN linear
            # project ps_mid → out_channels (cheap full-res 1×1 conv).
            feat_logits = self.ps_feat_proj(x)                       # [B*T, ps_mid·p², ph, pw]
            feat_mid    = F.pixel_shuffle(feat_logits, self.patch_size)   # [B*T, ps_mid, H, W]
            feat        = self.ps_out_proj(feat_mid)                 # [B*T, out_C, H, W]
            conf_logits = self.ps_conf_proj(x)                       # [B*T,    p², ph, pw]
            conf = F.pixel_shuffle(conf_logits, self.patch_size)     # [B*T, 1, H, W]
            conf = torch.sigmoid(conf) + 1e-6
            feat = feat.reshape(B, T, *feat.shape[1:])
            conf = conf.reshape(B, T, *conf.shape[1:])
        else:
            # ── Legacy DPT RefineNet path ──
            out = []
            for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
                x = output_tokens_list[layer_idx][:, :, patch_start_idx:]  # [B, T, N, dim]
                x = x.reshape(B * T, -1, x.shape[-1])                      # [B*T, N, dim]
                x = self.dpt.norm(x)
                x = x.permute(0, 2, 1).reshape(B * T, x.shape[-1], patch_h, patch_w)
                # [B*T, causal_dim, patch_h, patch_w]

                # ---- LSeg fusion (semantic head only) ----
                if sem_feat is not None:
                    x = self.fusion_projs[dpt_idx](torch.cat([x, sem_feat], dim=1))

                x = self.dpt.projects[dpt_idx](x)
                if self.dpt.pos_embed:
                    x = self.dpt._apply_pos_embed(x, W, H)
                x = self.dpt.resize_layers[dpt_idx](x)
                out.append(x)

            dpt_feat = self.dpt.scratch_forward(out)   # [B*T, 256, H', W']
            dpt_feat = custom_interpolate(
                dpt_feat,
                (patch_h * self.patch_size, patch_w * self.patch_size),
                mode='bilinear',
                align_corners=True,
            )
            if self.dpt.pos_embed:
                dpt_feat = self.dpt._apply_pos_embed(dpt_feat, W, H)
            # dpt_feat: [B*T, 256, H, W]

            feat = self.out_proj(dpt_feat)              # [B*T, out_channels, H, W]
            conf = self.conf_head(dpt_feat) + 1e-6      # [B*T, 1, H, W]

            feat = feat.reshape(B, T, *feat.shape[1:])
            conf = conf.reshape(B, T, *conf.shape[1:])

        feat_expanded = None
        if self.expander is not None:
            feat_perm = feat.permute(0, 1, 3, 4, 2)           # [B, T, H, W, C]
            feat_expanded = self.expander(feat_perm).permute(0, 1, 4, 2, 3)  # [B, T, expand_to, H, W]

        return feat, conf, feat_expanded
