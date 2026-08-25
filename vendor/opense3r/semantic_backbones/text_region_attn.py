"""
TextRegion-style mask-weighted attention pooling.

For each instance mask, attention-pool CLIP/PE last-layer V using the mask
itself as attention weights. The pooled per-instance feature goes through the
backbone's post-attention path (out_proj + LayerScale + MLP residual +
final_norm + final_proj) to land in the image-text shared space, then is
broadcast back to mask area to form a dense GT feature map.

Differs from the original TextRegion in two ways:
  - SAM2 region masks replaced by the user's instance_mask (higher quality)
  - The per-region feature is broadcast back as **feature** (not classified
    to logits first). This forms a dense feature map for cosine distillation.

Reference: TextRegion https://arxiv.org/abs/2502.09921
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def textregion_regions(
    v: torch.Tensor,                # [num_heads, N_patch, head_dim]  V (no special tokens)
    masks_low: torch.Tensor,        # [N, N_patch]  per-region weights at the patch grid (soft or hard)
    shared: dict,                   # post-attention modules from _v_forward
) -> torch.Tensor:
    """Batched mask-weighted V pool → per-region text-aligned features [N, output_dim].

    This is the core of `textregion_supervision_per_frame` without the dense
    broadcast — usable for BOTH the per-query base (soft masks) and the
    per-GT-instance target (hard one-hot), from the SAME cached V.  The math is
    byte-identical to per_frame (weighted-sum `masks_low @ v_flat` → post-attn),
    so a GT-mask call reproduces the legacy textregion target exactly.
    """
    num_heads, N_patch, head_dim = v.shape
    v_flat = v.permute(1, 0, 2).reshape(N_patch, num_heads * head_dim)   # [N_patch, D]
    region_v = (masks_low @ v_flat).unsqueeze(0)                          # [1, N, D]
    out = shared['attn_out_proj'](region_v)
    out = shared['ls1'](out)
    out = out + shared['ls2'](shared['mlp'](shared['mlp_norm'](out)))
    out = shared['final_norm'](out)
    if shared.get('final_proj', None) is not None:
        out = out @ shared['final_proj']                                 # [1, N, output_dim]
    return out.squeeze(0)                                                 # [N, output_dim]


def textregion_supervision_per_frame(
    v: torch.Tensor,                # [num_heads, N_patch, head_dim]  V (no special tokens)
    instance_mask: torch.Tensor,    # [H, W]  long, instance IDs (0=unlabeled)
    shared: dict,                   # post-attention modules from _v_forward
    patch_h: int,
    patch_w: int,
    output_dim: int,
) -> torch.Tensor:
    """
    Build dense GT feature map for one frame via mask-weighted V pooling.

    Returns:
        dense_gt: [output_dim, H, W]  per-instance broadcast features
    """
    H, W = instance_mask.shape
    device = v.device
    dtype  = v.dtype

    ids = instance_mask.unique()
    ids = ids[ids > 0]
    num_inst = ids.numel()

    if num_inst == 0:
        return torch.zeros(output_dim, H, W, device=device, dtype=dtype)

    # Downsample one-hot to patch grid in CHUNKS: a full [num_inst, H, W]
    # one-hot hits ~90 GB on frames with thousands of raw instance ids
    # (ASE/Infinigen) — killed epoch-18 val twice (2026-07-29).
    _rows = []
    for _c0 in range(0, num_inst, 128):
        _chunk = ids[_c0:_c0 + 128]
        _oh = (instance_mask.unsqueeze(0) == _chunk.view(-1, 1, 1)).to(dtype)
        _rows.append(F.interpolate(
            _oh.unsqueeze(0), (patch_h, patch_w),
            mode='bilinear', align_corners=False,
        ).squeeze(0).reshape(_chunk.numel(), patch_h * patch_w))
    mask_low = torch.cat(_rows, 0).clamp(0, 1)

    # Drop instances that lost all coverage at low res.
    keep = mask_low.sum(dim=-1) > 0
    if keep.sum() == 0:
        return torch.zeros(output_dim, H, W, device=device, dtype=dtype)
    mask_low = mask_low[keep]
    ids      = ids[keep]
    num_inst = ids.numel()

    # Flatten V: [num_heads, N_patch, head_dim] → [N_patch, embed_dim]
    num_heads, N_patch, head_dim = v.shape
    embed_dim = num_heads * head_dim
    v_flat = v.permute(1, 0, 2).reshape(N_patch, embed_dim)

    # Mask-weighted pool: [num_inst, embed_dim]
    region_v = mask_low @ v_flat                                   # [num_inst, embed_dim]
    region_v = region_v.unsqueeze(0)                                # [1, num_inst, embed_dim]

    # ── Post-attention pipeline (backbone-specific modules from `shared`) ──
    out = shared['attn_out_proj'](region_v)                         # [1, num_inst, embed_dim]
    out = shared['ls1'](out)
    # MLP residual (TextRegion always adds it):
    out = out + shared['ls2'](shared['mlp'](shared['mlp_norm'](out)))
    out = shared['final_norm'](out)
    if shared.get('final_proj', None) is not None:
        out = out @ shared['final_proj']                            # [1, num_inst, output_dim]
    out = out.squeeze(0)                                            # [num_inst, output_dim]

    # Broadcast back to dense [output_dim, H, W] via per-pixel gather —
    # instance_mask is a label map (masks exclusive), no [n,H,W] one-hot.
    dense = torch.zeros(output_dim, H * W, device=device, dtype=dtype)
    inst_flat = instance_mask.reshape(-1)
    iidx = torch.searchsorted(ids, inst_flat.clamp_min(ids.min()))
    iidx = iidx.clamp_max(num_inst - 1)
    sel = ids[iidx] == inst_flat
    dense[:, sel] = out[iidx[sel]].T.to(dtype)
    return dense.reshape(output_dim, H, W)
