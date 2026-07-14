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

    # One-hot: [num_inst, H, W]
    mask_oh = (instance_mask.unsqueeze(0) == ids.view(-1, 1, 1)).to(dtype)

    # Downsample to patch grid → [num_inst, N_patch]
    mask_low = F.interpolate(
        mask_oh.unsqueeze(0), (patch_h, patch_w),
        mode='bilinear', align_corners=False,
    ).squeeze(0).reshape(num_inst, patch_h * patch_w)
    mask_low = mask_low.clamp(0, 1)

    # Drop instances that lost all coverage at low res.
    keep = mask_low.sum(dim=-1) > 0
    if keep.sum() == 0:
        return torch.zeros(output_dim, H, W, device=device, dtype=dtype)
    mask_low = mask_low[keep]
    ids      = ids[keep]
    mask_oh  = mask_oh[keep]
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

    # Broadcast back to dense [output_dim, H, W]
    dense = torch.einsum('id,ihw->dhw', out, mask_oh)
    return dense
