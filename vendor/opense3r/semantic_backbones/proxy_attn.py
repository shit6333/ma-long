"""
ProxyCLIP-style attention.

The key idea: at the last attention layer of CLIP/PE, replace the standard
Q·K^T attention map with a self-similarity matrix computed from an external
Vision Foundation Model (VFM, e.g. DINOv2) feature map. The CLIP/PE V is
then aggregated by VFM-driven attention. Output goes through the backbone's
post-attention path (out_proj + LayerScale + final_norm + final_proj) to
land in the image-text shared space.

The post-attention pipeline is fed in via a `shared` dict that names each
module canonically — backbones with different attribute names (CLIP's
``out_proj`` vs DINOv2's ``proj``) and different presence of LayerScale
populate the dict in their own ``_v_forward``.

Reference: ProxyCLIP (ECCV 2024) https://arxiv.org/abs/2408.04883
"""

import torch
import torch.nn.functional as F


def compute_proxy_attn_weights(
    ex_feats: torch.Tensor,
    beta: float = 1.2,
    gamma: float = 3.0,
) -> torch.Tensor:
    """
    Compute ProxyCLIP attention weights from VFM patch self-similarity.

    Always runs in fp32: bf16/fp16 makes `F.normalize` underflow on small
    rows and the large `[HW, HW]` softmax accumulate enough error to produce
    NaN.  The output is cast back to `ex_feats.dtype` for downstream matmul.

    Args:
        ex_feats: [B, C, H, W]  VFM features at patch grid (e.g. DINOv2)
        beta:     re-centering factor (subtract mean*beta)
        gamma:    scale factor before softmax

    Returns:
        attn_weights: [B, HW, HW]  softmax-normalized
    """
    orig_dtype = ex_feats.dtype
    ex_feats = ex_feats.float()                              # promote to fp32
    qk = F.normalize(ex_feats.flatten(2), dim=1, eps=1e-6)   # [B, C, HW]
    sim = torch.einsum('bcm,bcn->bmn', qk, qk)               # [B, HW, HW]
    # On very uniform scenes (e.g. wall/floor close-ups) DINOv2 patches are
    # nearly identical → off-diag mean → 1, so `sim.mean() * beta` can exceed
    # sim.max() (= 1 for cosine self-sim diagonals).  Then every entry is
    # masked → softmax produces all-NaN → output collapses to zero.
    # Clamp threshold to (max - eps) so each row's diagonal always survives.
    threshold = torch.minimum(
        sim.mean() * beta,
        sim.amax() - sim.new_tensor(1e-4),
    )
    sim = (sim - threshold) * gamma
    sim = sim.masked_fill(sim < 0, float('-inf'))
    attn = F.softmax(sim, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)                   # belt-and-suspenders
    return attn.to(orig_dtype)


def apply_proxy_clip_last_attn(
    v: torch.Tensor,                  # [B, num_heads, S_seq, head_dim]  CLIP/PE V
    attn_weights: torch.Tensor,       # [B, HW_vfm, HW_vfm]
    shared: dict,                     # post-attention modules from _v_forward
    clip_hw: tuple,                   # (H_clip, W_clip)  V grid before interp
    vfm_hw: tuple,                    # (H_vfm, W_vfm)
    n_special: int = 1,               # number of leading non-patch tokens to drop (CLS, register, ...)
) -> torch.Tensor:
    """
    Apply ProxyCLIP's custom last-layer attention.

    Returns:
        [B, output_dim, H_vfm, W_vfm]  dense feature in CLIP/PE image-text space
    """
    B, num_heads, N, head_dim = v.shape

    if n_special > 0:
        v = v[:, :, n_special:, :]                          # drop CLS / register tokens

    H_clip, W_clip = clip_hw
    H_vfm,  W_vfm  = vfm_hw

    # Reshape to spatial then interpolate V to VFM grid.
    v = v.permute(0, 1, 3, 2).reshape(B * num_heads, head_dim, H_clip, W_clip)
    v = F.interpolate(v, size=(H_vfm, W_vfm), mode='bilinear', align_corners=False)
    v = v.reshape(B, num_heads, head_dim, H_vfm * W_vfm).permute(0, 1, 3, 2)
    # v: [B, num_heads, HW_vfm, head_dim]

    attn_w = attn_weights.unsqueeze(1).to(v.dtype)          # [B, 1, HW_vfm, HW_vfm]
    out = torch.matmul(attn_w, v)                            # [B, num_heads, HW_vfm, head_dim]
    out = out.permute(0, 2, 1, 3).reshape(B, H_vfm * W_vfm, num_heads * head_dim)

    # ── Post-attention pipeline (backbone-specific modules from `shared`) ──
    out = shared['attn_out_proj'](out)        # CLIP: out_proj  / TIPS: proj
    out = shared['ls1'](out)                  # LayerScale (Identity for CLIP; active for TIPS)
    out = shared['final_norm'](out)           # ln_post / final norm
    if shared.get('final_proj', None) is not None:
        out = out @ shared['final_proj']      # image-text shared projection (PE/CLIP have it; TIPS doesn't)

    out = out.reshape(B, H_vfm, W_vfm, -1).permute(0, 3, 1, 2)
    return out                                # [B, output_dim, H_vfm, W_vfm]
