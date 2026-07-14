"""
Common base for vision-language semantic backbones (PE-Core, OpenAI CLIP, ...).

A backbone has three forward paths used by OpenSe3r:

  1. forward_prior(images, dino_tokens, dino_grid)
     ProxyCLIP last-attention path: VFM (DINOv2) self-similarity drives the
     CLIP/PE last attention layer; CLIP/PE V is aggregated → dense feature
     map at the VFM patch grid.

  2. forward_supervision(images, instance_mask, sem_supervision, ...)
     - 'textregion': mask-weighted V pooling at last attention layer
       (optionally with multi-resolution crops for higher patch grid)
     - 'mask_crops': per-instance bbox/masked crop full forward + global emb

  3. encode_image_global / encode_text: standard CLIP-style image/text
     encoders. Used by Method 2 (mask_crops) and at inference for text query.

Subclasses implement four hooks:
  - _v_forward(prepped):       backbone PE/CLIP forward to last-1 layer + V
  - encode_image_global:       standard image encoder (pooled, projected)
  - encode_text:               standard text encoder
  - register img_mean/img_std buffers in __init__

Everything else (multi_res tiling, proxy attention, mask-attn pool, mask-crop
broadcast) is shared and lives here.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .proxy_attn import compute_proxy_attn_weights, apply_proxy_clip_last_attn
from .text_region_attn import textregion_supervision_per_frame
from .mask_crop_extractor import mask_crops_supervision_per_frame


class VLMBackboneBase(nn.Module):
    """Common base for PE-Core / OpenAI CLIP / TIPS semantic backbones."""

    # Subclass sets these in __init__:
    output_dim: int     # image-text shared dim (e.g. 1024 for PE-B16, 512 for CLIP-B/16, 768 for TIPS-B/14)
    input_size: int     # square input resolution (e.g. 224 / 336 / 448)
    patch_size: int     # 14 or 16
    n_special: int      # leading non-patch tokens in V (CLS=1; TIPS adds register tokens too)
    proxy_beta:  float
    proxy_gamma: float
    # Subclass registers buffers img_mean/img_std (3-channel) shaped [1, 3, 1, 1].

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _v_forward(self, images_prepped: torch.Tensor) -> dict:
        """
        Forward through transformer up to the last block's qkv linear, returning V.
        Subclass implements the backbone-specific positional / embedding code.

        Args:
            images_prepped: [N, 3, S, S] already resized + normalised

        Returns dict with the V tensor + the post-attention pipeline modules
        named canonically (different backbones have different attribute names
        — e.g. CLIP's ``out_proj`` vs DINOv2's ``proj`` — so we hide that here):

            v             : [N, num_heads, S_seq, head_dim]
            patch_h, patch_w : int     spatial patch grid (S // patch_size)
            attn_out_proj : nn.Module  V → x linear  (CLIP: blk.attn.out_proj; TIPS: blk.attn.proj)
            ls1           : nn.Module  LayerScale or Identity
            mlp_norm      : nn.Module  norm before MLP (CLIP: ln_2; TIPS: norm2)
            mlp           : nn.Module
            ls2           : nn.Module  LayerScale or Identity
            final_norm    : nn.Module  final LN (CLIP: ln_post; TIPS: top-level norm)
            final_proj    : Tensor     [width, output_dim] image-text projection,
                                       or None if the backbone has no projection
                                       (TIPS: output_dim == embed_dim → None)
            last_block    : the last ResidualAttentionBlock (kept for diagnostics)
        """
        raise NotImplementedError

    def encode_image_global(self, crops_01: torch.Tensor) -> torch.Tensor:
        """Standard backbone image encoder. Returns [N, output_dim] L2-normalised."""
        raise NotImplementedError

    def encode_text(self, prompts) -> torch.Tensor:
        """Encode text query. Returns [N, output_dim] L2-normalised."""
        raise NotImplementedError

    @torch.no_grad()
    def forward_prior_native(self, images_01: torch.Tensor) -> torch.Tensor:
        """
        Native dense prior — run the backbone's standard image encoder and
        return per-patch features (in the image-text shared space) at the
        backbone's natural patch grid.

        For TIPSv2 this is the 'right' answer: v2 has explicit patch-text
        alignment so dense patch tokens land directly in the shared space.
        For CLIP / PE-Core, native patch features are CLS-biased (CLIP-style
        contrastive training), so 'proxy_clip' (DINOv2-driven) usually gives
        cleaner spatial structure — we still expose this option for ablation.

        Args:
            images_01: [BT, 3, H, W] in [0, 1]
        Returns:
            [BT, output_dim, ph_native, pw_native]
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement forward_prior_native. "
            f"Either use prior_method='proxy_clip' or override this method."
        )

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _prep(self, images_01: torch.Tensor, target_size: int = None) -> torch.Tensor:
        """Resize to (S, S) then normalise. Default S = self.input_size."""
        S = target_size or self.input_size
        x = F.interpolate(images_01, size=(S, S), mode='bilinear', align_corners=False)
        return (x - self.img_mean) / self.img_std

    @torch.no_grad()
    def shared_v_forward(self, images_01: torch.Tensor) -> dict:
        """Resize+normalise then run _v_forward."""
        prepped = self._prep(images_01)
        return self._v_forward(prepped)

    # ------------------------------------------------------------------
    # Multi-resolution V forward (TextRegion-style)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def multi_res_v_forward(
        self,
        images_01: torch.Tensor,
        crop_num_h_override: int = None,
        crop_num_w_override: int = None,
    ) -> dict:
        """
        TextRegion multi-resolution V forward.

        For each frame, run 1 global forward (whole image resized to S×S) plus
        crop_num_h * crop_num_w tile forwards (each tile cropped from the
        original image then resized to S×S).  Combine into a higher-resolution
        V map at points_per_h × points_per_w grid.

        Tile count default: ``crop_num_{h,w} = max(H // S, 1)`` / ``max(W // S, 1)``
        — fits "image is large; tile to native S×S without resize".  For the
        more common "image smaller than S × tile_count" use case (we want to
        zoom into local regions to get finer V grid), pass
        ``crop_num_h_override`` / ``crop_num_w_override`` explicitly: each tile
        becomes ``ceil(H / crop_num_h) × ceil(W / crop_num_w)`` in the original
        image and gets bilinearly upsampled to S×S before the backbone forward.
        Backbone always sees its native input_size — no posemb interpolation.

        If ``crop_num_h * crop_num_w == 1`` (after applying the override),
        falls back to single-pass.

        Returns dict with the same fields as _v_forward, but:
          - v at [BT, num_heads, points_per_h*points_per_w, head_dim]  (CLS already stripped)
          - patch_h, patch_w = points_per_h, points_per_w
          - 'cls_stripped': True   (downstream consumers should not strip again)
        """
        BT, _, H, W = images_01.shape
        S = self.input_size
        P = self.patch_size
        patch_num = S // P

        crop_num_h = (crop_num_h_override
                      if crop_num_h_override is not None else max(H // S, 1))
        crop_num_w = (crop_num_w_override
                      if crop_num_w_override is not None else max(W // S, 1))

        if crop_num_h == 1 and crop_num_w == 1:
            # Image fits in one tile — fall back to single-pass.  Strip special
            # tokens (CLS / register) so the output schema matches multi-res.
            shared = self.shared_v_forward(images_01)
            v = shared['v']
            if self.n_special > 0:
                v = v[:, :, self.n_special:, :]
            shared = dict(shared)               # shallow copy, override v + flag
            shared['v'] = v
            shared['cls_stripped'] = True
            return shared

        points_per_h = patch_num * crop_num_h
        points_per_w = patch_num * crop_num_w

        # Image-domain tile size (ceil division — last tile may overlap with previous).
        crop_size_h = math.ceil(H / crop_num_h)
        crop_size_w = math.ceil(W / crop_num_w)

        # Build inputs.  Layout: [global_for_all_frames, crop_0_for_all, crop_1_for_all, ...]
        # → each chunk along batch dim is BT frames for the same input slot.
        global_input = F.interpolate(images_01, size=(S, S), mode='bilinear', align_corners=False)
        crop_inputs  = []
        crop_offsets = []  # patch-grid (y, x) offsets for each crop slot
        for hi in range(crop_num_h):
            for wi in range(crop_num_w):
                y1 = hi * crop_size_h
                x1 = wi * crop_size_w
                y2 = min(y1 + crop_size_h, H)
                x2 = min(x1 + crop_size_w, W)
                y1 = max(y2 - crop_size_h, 0)
                x1 = max(x2 - crop_size_w, 0)
                crop = images_01[:, :, y1:y2, x1:x2]
                crop = F.interpolate(crop, size=(S, S), mode='bilinear', align_corners=False)
                crop_inputs.append(crop)
                crop_offsets.append((hi * patch_num, wi * patch_num))

        N_inputs = 1 + crop_num_h * crop_num_w
        all_inputs = torch.cat([global_input] + crop_inputs, dim=0)   # [N_inputs * BT, 3, S, S]

        # Already at (S, S); just normalise.
        all_inputs = (all_inputs - self.img_mean) / self.img_std
        out = self._v_forward(all_inputs)
        v_all = out['v']                                              # [N_inputs*BT, h, S_seq, d]
        if self.n_special > 0:
            v_all = v_all[:, :, self.n_special:, :]

        num_heads = v_all.shape[1]
        head_dim  = v_all.shape[3]
        v_all = v_all.reshape(N_inputs, BT, num_heads, patch_num * patch_num, head_dim)

        # 1. Initialise v_multi from interpolated global V (input slot 0).
        v_global  = v_all[0]                                           # [BT, h, P*P, d]
        v_global  = v_global.permute(0, 1, 3, 2).reshape(
            BT * num_heads, head_dim, patch_num, patch_num,
        )
        v_global_up = F.interpolate(
            v_global, size=(points_per_h, points_per_w),
            mode='bilinear', align_corners=False,
        )
        v_multi = v_global_up.reshape(
            BT, num_heads, head_dim, points_per_h, points_per_w,
        ).clone()

        # 2. Add each crop's V to its tile location.
        for crop_id in range(crop_num_h * crop_num_w):
            y_off, x_off = crop_offsets[crop_id]
            v_crop = v_all[1 + crop_id]                                # [BT, h, P*P, d]
            v_crop = v_crop.permute(0, 1, 3, 2).reshape(
                BT, num_heads, head_dim, patch_num, patch_num,
            )
            v_multi[:, :, :, y_off:y_off + patch_num, x_off:x_off + patch_num] = (
                v_multi[:, :, :, y_off:y_off + patch_num, x_off:x_off + patch_num]
                + v_crop
            )

        # 3. Reshape to canonical [BT, num_heads, points_per_h*points_per_w, head_dim].
        v_out = v_multi.reshape(
            BT, num_heads, head_dim, points_per_h * points_per_w,
        ).permute(0, 1, 3, 2).contiguous()

        # Reuse all post-attention modules from `out`, just swap v + grid.
        result = dict(out)
        result['v']            = v_out
        result['patch_h']      = points_per_h
        result['patch_w']      = points_per_w
        result['cls_stripped'] = True
        return result

    # ------------------------------------------------------------------
    # Prior path: ProxyCLIP last-attn driven by DINOv2 self-similarity
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_prior(
        self,
        images_01:   torch.Tensor,                  # [BT, 3, H, W]
        dino_tokens: torch.Tensor,                  # [BT, N_patch, dino_dim]
        dino_grid:   tuple,                         # (ph_dino, pw_dino)
        **_unused,
    ) -> torch.Tensor:
        """
        Returns prior dense feature [BT, output_dim, ph_dino, pw_dino].
        """
        ph_dino, pw_dino = dino_grid
        BT = images_01.shape[0]

        shared = self.shared_v_forward(images_01)
        ex = dino_tokens.permute(0, 2, 1).reshape(BT, -1, ph_dino, pw_dino)
        attn_w = compute_proxy_attn_weights(
            ex, beta=self.proxy_beta, gamma=self.proxy_gamma,
        )

        return apply_proxy_clip_last_attn(
            v            = shared['v'],
            attn_weights = attn_w,
            shared       = shared,
            clip_hw      = (shared['patch_h'], shared['patch_w']),
            vfm_hw       = (ph_dino, pw_dino),
            n_special    = self.n_special,
        )

    # ------------------------------------------------------------------
    # Supervision path: TextRegion or mask_crops
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_supervision(
        self,
        images_01:        torch.Tensor,               # [BT, 3, H, W]
        instance_mask:    torch.Tensor,               # [B, T, H, W]
        sem_supervision:  str = 'textregion',
        multi_res_crops:  bool = False,
        mr_grid_h:        int = None,                 # override tile count (H direction)
        mr_grid_w:        int = None,                 # override tile count (W direction)
        mask_crop_size:   int = 224,
        mask_crop_margin: float = 0.1,
        mask_crop_mode:   str = 'bbox',
        **_unused,
    ) -> torch.Tensor:
        B, T, H, W = instance_mask.shape

        if sem_supervision == 'textregion':
            return self._supervision_textregion(
                images_01, instance_mask, B, T, H, W,
                multi_res_crops = multi_res_crops,
                mr_grid_h       = mr_grid_h,
                mr_grid_w       = mr_grid_w,
            )
        if sem_supervision == 'mask_crops':
            return self._supervision_mask_crops(
                images_01, instance_mask, B, T, H, W,
                crop_size = mask_crop_size,
                margin    = mask_crop_margin,
                crop_mode = mask_crop_mode,
            )
        if sem_supervision == 'patch_mean':
            return self._supervision_patch_mean(
                images_01, instance_mask, B, T, H, W,
            )
        raise ValueError(f"Unknown sem_supervision: {sem_supervision}")

    def _supervision_textregion(
        self, images_01, instance_mask, B, T, H, W,
        multi_res_crops: bool,
        mr_grid_h: int = None,
        mr_grid_w: int = None,
    ) -> torch.Tensor:
        BT = B * T

        if multi_res_crops:
            shared = self.multi_res_v_forward(
                images_01,
                crop_num_h_override = mr_grid_h,
                crop_num_w_override = mr_grid_w,
            )
            v_all  = shared['v']                                        # already special-stripped
        else:
            shared = self.shared_v_forward(images_01)
            v_all  = shared['v']
            if self.n_special > 0:
                v_all = v_all[:, :, self.n_special:, :]

        ph, pw = shared['patch_h'], shared['patch_w']

        gt = torch.zeros(B, T, self.output_dim, H, W,
                         device=v_all.device, dtype=v_all.dtype)

        for b in range(B):
            for t in range(T):
                idx = b * T + t
                gt[b, t] = textregion_supervision_per_frame(
                    v             = v_all[idx],
                    instance_mask = instance_mask[b, t],
                    shared        = shared,
                    patch_h       = ph,
                    patch_w       = pw,
                    output_dim    = self.output_dim,
                )
        return gt

    def _supervision_mask_crops(
        self, images_01, instance_mask, B, T, H, W,
        crop_size: int, margin: float, crop_mode: str,
    ) -> torch.Tensor:
        gt = torch.zeros(B, T, self.output_dim, H, W,
                         device=images_01.device, dtype=images_01.dtype)
        for b in range(B):
            for t in range(T):
                idx = b * T + t
                gt[b, t] = mask_crops_supervision_per_frame(
                    image         = images_01[idx],
                    instance_mask = instance_mask[b, t],
                    backbone      = self,
                    crop_size     = crop_size,
                    margin        = margin,
                    crop_mode     = crop_mode,
                )
        return gt

    # ------------------------------------------------------------------
    # patch_mean: hard-mask mean of backbone-native dense patch features.
    # Best for backbones with explicit patch-text alignment (TIPSv2,
    # SigLIP-v2 dense variants). For CLIP/PE the patch tokens are
    # CLS-biased, so this mode will give noisier GT than textregion.
    # ------------------------------------------------------------------
    def _supervision_patch_mean(
        self, images_01, instance_mask, B, T, H, W,
    ) -> torch.Tensor:
        # Native dense patch features in image-text shared space.
        feat = self.forward_prior_native(images_01)               # [BT, D, ph, pw]
        return self.patch_mean_aggregate(feat, instance_mask, B, T, H, W)

    @torch.no_grad()
    def patch_mean_aggregate(
        self, feat: torch.Tensor, instance_mask: torch.Tensor,
        B: int, T: int, H: int, W: int,
    ) -> torch.Tensor:
        """Per-instance hard-mask mean of a PRECOMPUTED dense feature map.

        feat: [BT, D, ph, pw] dense features (image-text shared space).
        Returns [B, T, D, H, W] — each pixel carries its instance's mean feature
        (background = 0).  Split out from `_supervision_patch_mean` so callers that
        already hold the dense map (e.g. RADIO's single forward) can build the GT
        without a second backbone pass.
        """
        BT, D, ph, pw = feat.shape
        N_patch = ph * pw
        device, dtype = feat.device, feat.dtype

        feat_flat = feat.reshape(BT, D, N_patch).permute(0, 2, 1) # [BT, N_patch, D]
        gt = torch.zeros(B, T, D, H, W, device=device, dtype=dtype)

        for b in range(B):
            for t in range(T):
                idx = b * T + t
                inst = instance_mask[b, t]                        # [H, W]
                ids = inst.unique()
                ids = ids[ids > 0]
                if ids.numel() == 0:
                    continue

                # Full-res one-hot for broadcast: [num_inst, H, W]
                mask_oh = (inst.unsqueeze(0) == ids.view(-1, 1, 1)).to(dtype)

                # Patch-grid hard mask via nearest-neighbour downsampling.
                # Avoids the soft bilinear weights leaking across instance
                # boundaries — user asked for hard mask here.
                mask_low = F.interpolate(
                    mask_oh.unsqueeze(0), (ph, pw), mode='nearest',
                ).squeeze(0).reshape(ids.numel(), N_patch)        # [num_inst, N_patch] in {0,1}

                # Drop instances that lost all coverage at patch resolution.
                weight = mask_low.sum(dim=-1, keepdim=True)        # [num_inst, 1]
                keep   = (weight.squeeze(-1) > 0)
                if keep.sum() == 0:
                    continue
                mask_low = mask_low[keep]
                weight   = weight[keep]
                mask_oh  = mask_oh[keep]

                # Mean patch feature per instance.
                region_feat = (mask_low @ feat_flat[idx]) / weight # [num_inst, D]

                # Broadcast back to dense map (full-res mask).
                gt[b, t] = torch.einsum('id,ihw->dhw', region_feat, mask_oh)

        return gt
