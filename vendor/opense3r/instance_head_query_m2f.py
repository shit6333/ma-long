"""
opense3r/instance_head_query_m2f.py
=====================================
Mask2Former-style query instance head for OpenSe3r.

Key differences from `instance_head_query.py` (MaskFormer-style):

  1. Masked attention  — cross-attention is restricted to pixels where
                          the previous layer's predicted mask sigmoid > 0.5
                          (the "mask gate" idea from Mask2Former).  Forces
                          each query to focus on its own object → Q becomes
                          instance-specific instead of competing for capacity.

  2. Multi-scale features — decoder layers cycle through 1/32, 1/16, 1/8
                             scales of the pixel-decoder output.  Each
                             query attends to a different scale at each
                             layer → handles object size variation.

  3. Cross-attn → Self-attn order — Mask2Former found this converges
                                     more stably than self-then-cross.

  4. Learnable query content + query positional embedding — query content
                                     starts from a learnable parameter
                                     (not zero), giving each slot a
                                     stronger prior identity.

  5. Unified mask_embed-as-descriptor — `mask_embed` (used for the dot
                                          product with F_mask to make
                                          masks) is L2-normalised and
                                          ALSO returned as `descriptor`.
                                          The NT-Xent loss is applied to
                                          this same vector → no separate
                                          descriptor head competing for
                                          Q's capacity.

  6. Deep supervision — every layer's predictions (mask + objectness +
                          descriptor) are returned in `aux_outputs`,
                          letting the loss apply supervision at every
                          decoder layer, not just the final one.

Inputs / outputs schema match the legacy head, plus `aux_outputs` for
deep supervision:

    predictions = {
        'mask_logits':  [B, T, N, H, W],         # final-layer mask
        'objectness':   [B, T, N],               # final-layer obj
        'descriptor':   [B, T, N, c_mask],       # final-layer L2-normed mask_embed
        'aux_outputs':  list of L-1 dicts with mask_logits/objectness/descriptor
                        for each of the earlier layers (and the initial
                        prediction before any decoder layer).
    }

Backward compatibility:
  This module is `--query_arch mask2former` (opt-in).  The default
  `--query_arch maskformer` continues to use `instance_head_query.py`.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_heads_multiscale import MultiScaleFeatureDPTHead
from .instance_head_query import _MLP, _PositionEmbeddingSine2D


# ─────────────────────────────────────────────────────────────────────────────
# Single decoder-layer building blocks (Mask2Former order: cross → self → ffn)
# ─────────────────────────────────────────────────────────────────────────────

class _CrossAttentionLayer(nn.Module):
    """Cross-attention with optional masked attention (key padding-style mask)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                            batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries:    torch.Tensor,                  # [B, N_q, d]
        memory:     torch.Tensor,                  # [B, N_kv, d]
        query_pos:  torch.Tensor,                  # [B, N_q, d]
        memory_pos: torch.Tensor,                  # [B, N_kv, d]
        attn_mask:  Optional[torch.Tensor] = None, # [B*n_heads, N_q, N_kv] bool, True=block
    ) -> torch.Tensor:
        q = queries  + query_pos
        k = memory   + memory_pos
        v = memory
        # If attn_mask is all-True in a row (every key blocked) softmax of -inf is NaN.
        # Detect and disable mask for those rows to avoid NaN gradient.
        if attn_mask is not None:
            all_blocked = attn_mask.all(dim=-1)                  # [B*nh, N_q]
            if all_blocked.any():
                attn_mask = attn_mask.clone()
                attn_mask[all_blocked] = False                    # let those queries attend everywhere
        out, _ = self.attn(q, k, v, attn_mask=attn_mask, need_weights=False)
        return self.norm(queries + self.dropout(out))


class _SelfAttentionLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                            batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, query_pos, key_padding_mask=None):
        # key_padding_mask [B, N] bool, True = exclude as KEY (merged followers:
        # leaders must not attend stale absorbed fragments; followers still act
        # as queries — their outputs are ignored downstream).
        q = k = queries + query_pos
        out, _ = self.attn(q, k, queries, need_weights=False,
                           key_padding_mask=key_padding_mask)
        return self.norm(queries + self.dropout(out))


class _FFNLayer(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries):
        out = self.linear2(self.dropout(F.relu(self.linear1(queries))))
        return self.norm(queries + self.dropout(out))


# ─────────────────────────────────────────────────────────────────────────────
# World-frame AABB box head
# ─────────────────────────────────────────────────────────────────────────────

class BoxAABBHead(nn.Module):
    """World-frame AABB head — PCA-prior + small delta — in NORMALIZE scale.

    Three scale spaces in play:
      metric     → original GT (real meters)
      canonical  → DA3 / geo model native (≈1 for da3-base/large/giant,
                                          ≈m for da3-nested-metric)
      normalize  → canonical / box_scale → median ~1 ALWAYS, both modes

    Head operates entirely in NORMALIZE scale (scale-invariant):
        world_pts_norm     = world_points / box_scale   ← divide here
        soft_mask          = sigmoid(mask_logits).detach()
        anchor_n, sh_n     = weighted_std_aabb(soft_mask, world_pts_norm)
        mask_pooled_feat   = Σ(soft_mask × mask_features) / Σ
        ref                = ||sh_n||
        size_norm          = sh_n / ref   (dimensionless aspect)
        delta              = MLP(q_norm, mask_pooled_feat, size_norm)
        center_n           = anchor_n + delta[:3]  × ref
        size_half_n        = sh_n × exp(delta[3:6]).clamp(0.5, 3)

    Output: [BT, N, 6] = (cx, cy, cz, sx, sy, sz)  in NORMALIZE scale.

    Downstream:
      - Loss: GT also normalized by SAME box_scale → DIoU in normalize scale
      - PLY:  pred_box × box_scale  → back to CANONICAL (= geo PLY scale)
      - SLAM: same multiply → canonical → consistent with map geometry
    """
    def __init__(self, q_dim: int = 256, mask_feat_dim: int = 256, hidden: int = 256):
        super().__init__()
        in_dim = q_dim + mask_feat_dim + 3
        self.mlp = _MLP(in_dim, hidden, 6, num_layers=3)
        # Init: zero last layer → delta=0 → final box = pure PCA prior
        with torch.no_grad():
            self.mlp.layers[-1].weight.zero_()
            self.mlp.layers[-1].bias.zero_()

    def forward(
        self,
        q_norm:        torch.Tensor,   # [BT, N, q_dim]
        mask_logits:   torch.Tensor,   # [BT, N, H, W]      (will be detached + sigmoided)
        mask_features: torch.Tensor,   # [BT, mask_feat_dim, H, W]
        world_points:  torch.Tensor,   # [BT, H, W, 3]      CANONICAL frame
        box_scale:     Optional[torch.Tensor] = None,    # [BT] or [B] scalar per batch
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """Returns box6 [BT, N, 6] FULL size, NORMALIZE scale.

        Multiply by box_scale to recover CANONICAL-scale box (for PLY / SLAM).
        """
        BT, N, H, W = mask_logits.shape
        HW = H * W

        # ── Normalize world_points by box_scale → median ~1 regardless of DA3 mode ──
        if box_scale is not None:
            # Expand box_scale → [BT, 1, 1, 1]; broadcast B → BT if needed.
            bs = box_scale.to(world_points.dtype).clamp(min=eps)
            if bs.numel() == 1:
                bs_btn = bs.view(1, 1, 1, 1).expand(BT, 1, 1, 1)
            elif bs.shape[0] == BT:
                bs_btn = bs.view(BT, 1, 1, 1)
            else:    # [B] expand to BT
                B = bs.shape[0]
                T = BT // B
                bs_btn = bs.view(B, 1, 1, 1, 1).expand(B, T, 1, 1, 1).reshape(BT, 1, 1, 1)
            wp_in = world_points / bs_btn                                  # NORMALIZE
        else:
            wp_in = world_points                                            # legacy fall-through

        # ── (a) Mask (detached — box loss must NOT backprop into mask) ──
        # Use HARD-thresholded sigmoid (> 0.5) for PCA to match GT
        # construction (which uses binary instance_mask).  Without this, the
        # mask's soft boundary (sigmoid 0.1–0.5) inflates the weighted std
        # → PCA prior becomes ~2× the true object extent → DIoU stuck at 0.8+.
        # Fall back to soft mask if the hard mask is too sparse (rare object
        # / unconverged mask) to keep gradients flowing.
        sm_soft = mask_logits.sigmoid().detach()                        # [BT, N, H, W]
        sm_hard = (sm_soft > 0.5).to(sm_soft.dtype)
        hard_n  = sm_hard.sum(dim=(-1, -2))                              # [BT, N]
        use_hard = (hard_n > 5.0).unsqueeze(-1).unsqueeze(-1).to(sm_soft.dtype)
        sm = use_hard * sm_hard + (1.0 - use_hard) * sm_soft             # [BT, N, H, W]
        sm_flat = sm.reshape(BT, N, HW)
        w_sum = sm_flat.sum(dim=-1).clamp(min=eps)                       # [BT, N]

        # ── (b) Mask-pooled world points: center + size_half (PCA prior) ──
        # All in NORMALIZE scale.
        wp_flat    = wp_in.reshape(BT, HW, 3)                            # [BT, HW, 3]
        wp_sq_flat = wp_flat.pow(2)                                      # [BT, HW, 3]
        center = torch.bmm(sm_flat, wp_flat) / w_sum.unsqueeze(-1)       # E[X]
        sq_sum = torch.bmm(sm_flat, wp_sq_flat) / w_sum.unsqueeze(-1)    # E[X²]
        # √3·σ matches the bmax-bmin of a UNIFORM distribution (which is
        # what mask-thresholded points typically look like inside an
        # object).  2σ overshoots by ~15%; ½·σ undershoots.
        # For DEGENERATE axes (flat surfaces → σ→0), clamp size_half to
        # 0.5e-3 so size_full ≥ 1e-3 — same lower bound as GT's
        # `(bmax-bmin).clamp(min=1e-3)`.  Without this, eps² lower bound
        # on var → size_full_z = 4e-3 vs GT_z = 1e-3 = 4× mismatch for
        # flat objects (tables / floors), pushes DIoU loss past 0.8.
        var = (sq_sum - center.pow(2)).clamp(min=0.0)                    # [BT, N, 3]
        size_half_prior = (var.sqrt() * math.sqrt(3.0)).clamp(min=0.5e-3) # [BT, N, 3]  √3·σ  (≈ uniform AABB)

        # ── (c) Mask-pooled image features (scale-free; no normalization needed) ──
        mf_flat = mask_features.reshape(BT, mask_features.shape[1], HW)  # [BT, C, HW]
        pooled = torch.bmm(sm_flat, mf_flat.transpose(1, 2)) / w_sum.unsqueeze(-1)

        # ── (d) Head input: aspect-ratio-only size (dimensionless) ──
        ref = size_half_prior.norm(dim=-1, keepdim=True).clamp(min=eps)   # [BT, N, 1]
        size_norm = size_half_prior / ref                                  # [BT, N, 3]
        x = torch.cat([q_norm, pooled, size_norm], dim=-1)                # [BT, N, q+C+3]
        delta = self.mlp(x)                                                # [BT, N, 6]

        # ── (e) Compose final box in NORMALIZE frame ──
        dc, dlog_s = delta[..., :3], delta[..., 3:6]
        center_final    = center + dc * ref
        size_half_final = size_half_prior * dlog_s.exp().clamp(0.5, 3.0)
        size_full       = size_half_final * 2.0
        return torch.cat([center_final, size_full], dim=-1)                # [BT, N, 6]


class BoxAmodalHead(nn.Module):
    """ESAM-style AMODAL world-frame AABB head — DIRECT prediction in NORMALIZE scale.

    Unlike BoxAABBHead (PCA-prior size + small delta), this head predicts the box
    DIRECTLY:
        anchor   = soft-mask-weighted centroid of (normalized) world points
                   — the only geometric prior (a location anchor, ESAM-style;
                   NOT a size prior).
        center   = anchor + MLP_offset(q_norm, mask_pooled_feat)
        size     = exp(MLP_logsize(...))            (FULL size, normalized)

    Because the GT is the FULL-object box (pre-generated across ALL scene frames),
    the head is trained to EXTRAPOLATE beyond the visible/partial extent — i.e. it
    learns amodal completion, exactly like ESAM's `out_reg`.

    Output: [BT, N, 6] = (cx, cy, cz, sx, sy, sz)  full size, NORMALIZE scale.
    Multiply by box_scale to recover CANONICAL scale (PLY / SLAM).
    """
    def __init__(self, q_dim: int = 256, mask_feat_dim: int = 256, hidden: int = 256,
                 detach_inputs: bool = False, init_log_size: float = math.log(0.3),
                 size_anchor: bool = False, query_only: bool = False,
                 size_extent_aware: bool = False):
        super().__init__()
        # detach_inputs: probe-only mode — q_norm / mask_features are detached so
        # box gradients never reach the shared decoder (suspected of degrading
        # mask/desc from ~ep10 in run2; unproven — ESAM trains fine without).
        # Default False = ESAM-style end-to-end gradients.
        self.detach_inputs = bool(detach_inputs)
        # size_anchor: predict size as a RESIDUAL on the visible point cloud's
        # extent (weighted-std AABB, absolute NORMALIZE scale) → size = extent ×
        # exp(Δlog-size).  Gives the head an absolute geometric size prior (the
        # query's own geo feature is diameter-normalized → scale-blind).  Best
        # paired with the growing-visible-box GT (residual = memory-driven
        # completion beyond the current view).  False = legacy exp(logsize).
        # query_only: MLP input = q_norm ONLY (drop the mask-pooled appearance
        #   feature).  The decoder query already carries appearance/geo/memory.
        # size_extent_aware: feed log(visible extent) into the MLP so the size
        #   multiplier is extent-DEPENDENT ("how much have I seen → how much to
        #   extend").  Requires size_anchor (forces it on).  Scale is translation/
        #   rotation invariant → safe to feed (unlike absolute center).
        self.size_anchor       = bool(size_anchor) or bool(size_extent_aware)
        self.query_only        = bool(query_only)
        self.size_extent_aware = bool(size_extent_aware)
        in_dim = q_dim + (0 if self.query_only else mask_feat_dim) \
                 + (3 if self.size_extent_aware else 0)
        self.mlp = _MLP(in_dim, hidden, 6, num_layers=3)
        # Init: offset=0 (center≈anchor).  size_anchor → Δlog-size bias 0 so
        # size = visible-extent anchor at init; else absolute size=exp(init_log_size)
        # (ln(0.3) for NORMALIZE median~1; ln(1.0)=0 for metric ~1 m).  Zero
        # weights so all queries start identical; gradient learns to deviate.
        with torch.no_grad():
            self.mlp.layers[-1].weight.zero_()
            b = torch.zeros(6)
            b[3:6] = 0.0 if self.size_anchor else float(init_log_size)
            self.mlp.layers[-1].bias.copy_(b)

    def forward(
        self,
        q_norm:        torch.Tensor,   # [BT, N, q_dim]
        mask_logits:   torch.Tensor,   # [BT, N, H, W]
        mask_features: torch.Tensor,   # [BT, mask_feat_dim, H, W]
        world_points:  torch.Tensor,   # [BT, H, W, 3]   (frame-0 anchored at train)
        box_scale:     Optional[torch.Tensor] = None,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        BT, N, H, W = mask_logits.shape
        HW = H * W

        if self.detach_inputs:
            # Probe-only mode: box gradients never reach the shared decoder.
            q_norm = q_norm.detach()
            mask_features = mask_features.detach()

        # ── Normalize world_points by box_scale → median ~1 (DA3-mode-invariant) ──
        if box_scale is not None:
            bs = box_scale.to(world_points.dtype).clamp(min=eps)
            if bs.numel() == 1:
                bs_btn = bs.view(1, 1, 1, 1).expand(BT, 1, 1, 1)
            elif bs.shape[0] == BT:
                bs_btn = bs.view(BT, 1, 1, 1)
            else:
                Bn = bs.shape[0]; Tn = BT // Bn
                bs_btn = bs.view(Bn, 1, 1, 1, 1).expand(Bn, Tn, 1, 1, 1).reshape(BT, 1, 1, 1)
            wp_in = world_points / bs_btn
        else:
            wp_in = world_points

        # ── Soft/hard mask (detached) → centroid anchor + pooled features ──
        sm_soft = mask_logits.sigmoid().detach()
        sm_hard = (sm_soft > 0.5).to(sm_soft.dtype)
        hard_n  = sm_hard.sum(dim=(-1, -2))
        use_hard = (hard_n > 5.0).unsqueeze(-1).unsqueeze(-1).to(sm_soft.dtype)
        sm = use_hard * sm_hard + (1.0 - use_hard) * sm_soft
        sm_flat = sm.reshape(BT, N, HW)
        w_sum = sm_flat.sum(dim=-1).clamp(min=eps)

        wp_flat = wp_in.reshape(BT, HW, 3)
        anchor = torch.bmm(sm_flat, wp_flat) / w_sum.unsqueeze(-1)          # [BT, N, 3] normalized centroid

        # Visible-extent anchor via weighted std (E[X²]−E[X]², bmm-based → no
        # [BT,N,HW,3] materialization).  full extent ≈ 2×(2σ)=4σ.  Computed
        # BEFORE the MLP so size_extent_aware can feed log(extent) as input.
        extent = None
        if self.size_anchor:
            ex2    = torch.bmm(sm_flat, wp_flat * wp_flat) / w_sum.unsqueeze(-1)   # E[X²] [BT,N,3]
            var    = (ex2 - anchor * anchor).clamp(min=1e-6)
            extent = 4.0 * var.sqrt()                                              # [BT,N,3] normalized

        feats = [q_norm]
        if not self.query_only:
            mf_flat = mask_features.reshape(BT, mask_features.shape[1], HW)
            pooled = torch.bmm(sm_flat, mf_flat.transpose(1, 2)) / w_sum.unsqueeze(-1)  # [BT, N, C]
            feats.append(pooled)
        if self.size_extent_aware:
            feats.append(extent.clamp(min=1e-3).log())                     # extent-aware size multiplier
        out = self.mlp(torch.cat(feats, dim=-1))                           # [BT, N, 6]

        center = anchor + out[..., :3]                                     # anchor + offset
        if self.size_anchor:
            size = (extent * out[..., 3:6].exp()).clamp(min=1e-3, max=20.0)        # residual on extent
        else:
            size = out[..., 3:6].exp().clamp(min=1e-3, max=20.0)           # full size, normalized
        return torch.cat([center, size], dim=-1)                           # [BT, N, 6]


# ─────────────────────────────────────────────────────────────────────────────
# DEBUG: dump per-frame PCA-prior / final-pred boxes + mask-pooled points
# Inside the model code → independent of any train-side / PLY post-processing.
# Lets us verify: (1) mask-pooled points look reasonable, (2) PCA prior
# matches those points, (3) head's final pred differs from prior how expected.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def dump_world_aabb_debug_per_frame(
    mask_logits:          torch.Tensor,   # [BT, N, H, W]
    objectness:           torch.Tensor,   # [BT, N]
    world_points:         torch.Tensor,   # [BT, H, W, 3]    CANONICAL
    box_scale:            torch.Tensor,   # [B] or [BT]      scalar per chunk
    final_box_normalize:  torch.Tensor,   # [BT, N, 6]       head output (NORMALIZE)
    save_dir:             str,
    prefix:               str = '',
    top_k:                int = 10,
    obj_threshold:        float = 0.4,
    B:                    int = 1,
    T:                    int = None,     # if None, T = BT/B
):
    """Per-frame debug: 3 PLY per frame, all in CANONICAL scale (overlays geo PLY).
        prefix_t{t}_pca_prior.ply   wireframe of PCA-prior boxes (top-K slots)
        prefix_t{t}_pred.ply        wireframe of head's final pred boxes
        prefix_t{t}_mask_pts.ply    point cloud of mask-pooled world_pts (colored by slot)

    Each slot's box / pts gets a consistent slot-id colour across files,
    so you can overlay them in CloudCompare / MeshLab.

    NOTHING about this function uses the train-side proposal/merge path —
    it directly reads the head's per-(t, slot) tensors.
    """
    import os
    import numpy as np
    import trimesh
    BT = mask_logits.shape[0]
    if T is None:
        T = BT // B
    N, H, W = mask_logits.shape[1], mask_logits.shape[2], mask_logits.shape[3]
    os.makedirs(save_dir, exist_ok=True)

    # broadcast box_scale → [BT]
    bs = box_scale.to(world_points.dtype).clamp(min=1e-3)
    if bs.numel() == 1:
        bs_bt = bs.expand(BT)
    elif bs.shape[0] == BT:
        bs_bt = bs
    else:
        bs_bt = bs.unsqueeze(1).expand(B, T).reshape(BT)

    # local import to avoid top-of-file cycles
    from amb3r.vis_query_instance import _box_edge_points, _labels_to_colors

    obj_prob = objectness.sigmoid()                                   # [BT, N]
    for bt in range(BT):
        b = bt // T
        t = bt %  T
        # Top-K objectness slots above threshold
        probs_bt = obj_prob[bt]
        top = probs_bt.topk(min(top_k, N))
        slot_ids = top.indices[top.values > obj_threshold].tolist()
        if not slot_ids:
            continue

        wp = world_points[bt]                                          # [H, W, 3]  CANONICAL
        bs_scalar = float(bs_bt[bt].item())

        prior_box_lst:   list = []
        prior_lbl_lst:   list = []
        pred_box_lst:    list = []
        pred_lbl_lst:    list = []
        pts_lst:         list = []
        pts_lbl_lst:     list = []

        for k_idx, slot in enumerate(slot_ids):
            # ── soft mask + hard threshold ──
            sm = mask_logits[bt, slot].sigmoid()                       # [H, W]
            sm_hard = (sm > 0.5).to(sm.dtype)
            sm_use  = sm_hard if sm_hard.sum() > 5 else sm

            # ── PCA prior (in canonical scale, since wp is canonical) ──
            w_sum  = sm_use.sum().clamp(min=1e-3)
            center = (sm_use.unsqueeze(-1) * wp).sum(dim=(0, 1)) / w_sum
            sq_sum = (sm_use.unsqueeze(-1) * wp.pow(2)).sum(dim=(0, 1)) / w_sum
            var    = (sq_sum - center.pow(2)).clamp(min=0.0)
            size_h = (var.sqrt() * math.sqrt(3.0)).clamp(min=0.5e-3)
            size_full = size_h * 2.0                                   # FULL
            prior_box6 = torch.cat([center, size_full]).cpu().numpy()
            prior_box_lst.append(prior_box6)
            prior_lbl_lst.append(slot)

            # ── Head final pred (×box_scale → CANONICAL for vis overlay) ──
            pred_norm = final_box_normalize[bt, slot]
            pred_canon = pred_norm * bs_scalar                          # [6] canonical
            pred_box_lst.append(pred_canon.cpu().numpy())
            pred_lbl_lst.append(slot)

            # ── Mask-pooled points (hard-thresholded if available) ──
            mask_bool = (sm_use > 0.5) if sm_hard.sum() > 5 else (sm > 0.5)
            pts = wp[mask_bool]                                        # [P, 3]  canonical
            if pts.shape[0] > 0:
                pts_np = pts.cpu().numpy().astype(np.float32)
                pts_lst.append(pts_np)
                pts_lbl_lst.append(np.full(pts_np.shape[0], slot, dtype=np.int32))

        # ── Save: PCA prior wireframe ──
        if prior_box_lst:
            edge_pts, edge_lbls = [], []
            for b6, lbl in zip(prior_box_lst, prior_lbl_lst):
                ep = _box_edge_points(b6, num_per_edge=30)
                edge_pts.append(ep)
                edge_lbls.append(np.full(ep.shape[0], lbl, dtype=np.int32))
            P = np.concatenate(edge_pts).astype(np.float32)
            L = np.concatenate(edge_lbls)
            C = _labels_to_colors(L)
            trimesh.points.PointCloud(P, colors=C).export(
                os.path.join(save_dir, f'{prefix}b{b}_t{t}_pca_prior.ply'))

        # ── Save: head final pred wireframe ──
        if pred_box_lst:
            edge_pts, edge_lbls = [], []
            for b6, lbl in zip(pred_box_lst, pred_lbl_lst):
                ep = _box_edge_points(b6, num_per_edge=30)
                edge_pts.append(ep)
                edge_lbls.append(np.full(ep.shape[0], lbl, dtype=np.int32))
            P = np.concatenate(edge_pts).astype(np.float32)
            L = np.concatenate(edge_lbls)
            C = _labels_to_colors(L)
            trimesh.points.PointCloud(P, colors=C).export(
                os.path.join(save_dir, f'{prefix}b{b}_t{t}_pred.ply'))

        # ── Save: mask-pooled points (per-slot coloured) ──
        if pts_lst:
            P_all = np.concatenate(pts_lst, axis=0)
            L_all = np.concatenate(pts_lbl_lst)
            C_all = _labels_to_colors(L_all)
            trimesh.points.PointCloud(P_all, colors=C_all).export(
                os.path.join(save_dir, f'{prefix}b{b}_t{t}_mask_pts.ply'))


# ─────────────────────────────────────────────────────────────────────────────
# Mask2Former-style query head
# ─────────────────────────────────────────────────────────────────────────────

class Mask2FormerQueryInstanceHead(nn.Module):
    """Mask2Former-style class-agnostic instance head with unified
    mask_embed-as-descriptor and deep supervision."""

    def __init__(
        self,
        causal_dim:    int = 1024,
        num_queries:   int = 100,
        d_model:       int = 256,
        c_mask:        int = 256,
        descriptor_dim: int = 256,   # kept for API compat; should equal c_mask
        n_heads:       int = 8,
        n_layers:      int = 9,      # 3 scales × 3 cycles (Mask2Former default)
        dim_feedforward: int = 2048,
        dropout:       float = 0.0,
        intermediate_layer_idx: Optional[List[int]] = None,
        patch_size:    int = 14,
        mask_mlp_layers: int = 3,
        num_feature_levels: int = 3,
        scales:        Optional[List[int]] = None,    # [32, 16, 8] default
        return_aux:    bool = True,
        # Optional auxiliary box head — predicts 6-dim axis-aligned 3D bbox per
        # query, format [cx, cy, cz, sx, sy, sz] (center + full size).  When
        # enabled, also added to predictions dict as 'box' and (per-aux) in
        # 'aux_outputs'.  Default False; turn on via --query_use_box_head.
        use_box_head:  bool = False,
        box_mlp_layers: int = 3,
        # 'aabb' (legacy) = 6-dim AABB (cx,cy,cz,w,h,d);  size via softplus.
        # 'obb'  (new)    = 12-dim OBB (cx,cy,cz, sx,sy,sz, r1..r6);
        #                    size via exp (log-space), rot via 6d → SO(3)
        #                    at loss time.  Center is absolute in
        #                    chunk-local frame (matches DA3 world_points).
        box_mode:      str = 'aabb',
        # ── Query init source (Idea 1 + Idea 2) ──
        #   'learnable' (default, current): self.query_feat = nn.Embedding
        #   '3d_anchor': learnable 3D reference points in NORMALIZE space
        #                → pos2posemb3d → MLP → init query feat (3DPPE-style).
        #                Requires `causal_dim_external_init` to know the input dim.
        #   'external':  caller passes init_queries to forward (e.g. from
        #                CausalBranch instance_queries).  No self.query_feat built.
        query_init_type:           str = 'learnable',
        # Used by 'external' mode — input init queries can come at causal_dim
        # (from CausalBranch) and need to be projected to d_model.  Set None
        # to skip projection (caller already provides d_model-sized queries).
        external_init_dim:         Optional[int] = None,
        # Descriptor projection head (ESAM MergeHead-style):
        #   0 (default) — descriptor = L2-norm(mask_embed), entangled with the
        #                 mask-generating embedding (legacy behavior).
        #   >0          — descriptor = L2-norm(MLP_k(q_norm)): a DEDICATED
        #                 identity head decoupled from mask generation, so the
        #                 instance feature can specialize for cross-view matching
        #                 (descriptor_dim may then differ from c_mask).
        descriptor_mlp_layers:     int = 0,
        # ── Dual-path fused descriptor head (memory + appearance) ──
        # When True (requires query_init_type='external'), the descriptor is
        #   d = normalize( MLP_B(LN(q_mem))                        # path B: base
        #                + MLP_int(LN(q_mem) ‖ LN(q_norm)) )       # interaction
        # where q_mem = the external (CausalBranch, memory-aware) init queries
        # and q_norm = the per-frame mask-refined query.  MLP_int's final layer
        # is zero-init → at start d = normalize(path B) so the memory path is
        # forced to carry identity before the appearance interaction ramps in.
        # Default False → legacy descriptor (mask_embed / descriptor_proj),
        # fully backward compatible.
        desc_dual_path:            bool = False,
        desc_fuse_layers:          int  = 3,
        # When True (and desc_dual_path), also expose `descriptor_pathB` =
        # normalize(MLP_B(LN(q_mem))) so the loss can supervise the memory path
        # on its own (path-B aux contrastive).
        desc_pathB_aux:            bool = False,
        # world_amodal only: detach box-head inputs (probe-only; see BoxAmodalHead).
        box_head_detach:           bool = False,
        box_init_log_size:         float = math.log(0.3),   # metric mode → pass log(1.0)
        box_size_anchor:           bool = False,            # world_amodal*: size=visible-extent×exp(Δ)
        box_query_only:            bool = False,            # box MLP input = q_norm only (no mask-pool)
        box_size_extent_aware:     bool = False,            # feed log(extent) → extent-aware size mult
        # thing/stuff (foreground/background) head — per-query binary logit from
        # q_norm; BCE-supervised by the matched GT instance's is_bg flag.  Used to
        # filter stuff (wall/floor/ceiling) proposals at SLAM/eval.  Default off.
        use_fgbg_head:             bool = False,
        # FastSAM mask prior (ESAM-style): query i is seeded from FastSAM region i;
        # use that region's mask as the layer-0 masked-cross-attn restriction
        # (instead of the predicted mask) → each query STARTS from its strong,
        # in-domain-robust FastSAM region, then the decoder refines.  Requires
        # fastsam_query + fastsam_region_id passed to forward().  Default off.
        fastsam_mask_prior:        bool = False,
        # ── Mid-decoder query merge (DESIGN_INSTANCE_HEAD_OTM_MERGE.md §3) ──
        # After `merge_layer`'s prediction, cluster active queries by descriptor
        # cosine (leaders = high objectness, greedy, leader-only affinity, 3D
        # centroid veto), absorb followers into leaders (zero-init residual so
        # activation == identity), union the cluster masks into the next layer's
        # cross-attn prior, and deactivate followers (self-attn key padding +
        # excluded from final output via 'active_mask').  Training curriculum:
        # the runtime switch `self.merge_active` (set per-epoch by the trainer,
        # always True in SLAM) gates the whole block; merge decisions (affinity,
        # leader score, weights) are fully detached.
        merge_enable:              bool  = False,
        merge_layer:               int   = 5,     # merge after this layer's _predict
        merge_tau:                 float = 0.9,   # desc-cosine threshold (precision-first)
        merge_dist_veto:           float = 1.5,   # max centroid distance (m); 0/None = off
        merge_act_threshold:       float = 0.1,   # min σ(obj) to participate
        # P/Q head support (opense3r/pq_head.py): when True, stash the
        # FINAL-layer decoder-normed queries + pixel-decoder mask features
        # (both DETACHED) into the output dict ('_pq_qnorm'/'_pq_fmask') so
        # the external PQ scorer can inspect the head's own work.  Zero
        # behavior change when off.
        expose_pq_inputs:          bool  = False,
        # ── Instance-aware semantic head (RESIDUAL on a frozen-TIPS base) ──
        # When True, a gated 3-layer MLP on the FINAL decoder query produces a
        # per-query semantic RESIDUAL `inst_sem_raw` [B,T,N,inst_sem_dim].  The
        # model adds the frozen-TIPS textregion base (pooled over the query's
        # predicted soft mask) → sem = base + SemProj(q).  Last layer zero-init
        # so training starts at sem = base.  inst_sem_dim MUST equal the
        # semantic_backbone.output_dim (TIPS-v2-l14 = 1024).
        inst_sem_head:             bool  = False,
        inst_sem_dim:              int   = 1024,
    ):
        super().__init__()
        if intermediate_layer_idx is None:
            intermediate_layer_idx = [2, 5, 8, 11]
        # Dual-path head owns a dedicated descriptor projection, so descriptor_dim
        # is free (need NOT equal c_mask).  Only force the equality for the legacy
        # mask_embed-as-descriptor path.
        if (not desc_dual_path) and descriptor_mlp_layers <= 0 and descriptor_dim != c_mask:
            # Unified mask_embed-as-descriptor REQUIRES descriptor_dim == c_mask.
            print(f"[Mask2FormerQueryInstanceHead] descriptor_dim ({descriptor_dim}) != "
                  f"c_mask ({c_mask}); forcing descriptor_dim = c_mask = {c_mask} "
                  f"because the head uses mask_embed itself as descriptor "
                  f"(set descriptor_mlp_layers>0 for a separate descriptor head).")
            descriptor_dim = c_mask

        self.causal_dim     = causal_dim
        self.num_queries    = num_queries
        self.d_model        = d_model
        self.c_mask         = c_mask
        self.descriptor_dim = descriptor_dim
        self.n_heads        = n_heads
        self.n_layers       = n_layers
        self.num_feature_levels = num_feature_levels
        self.return_aux     = return_aux
        self.patch_size     = patch_size

        # Multi-scale pixel decoder
        self.pixel_decoder = MultiScaleFeatureDPTHead(
            causal_dim = causal_dim,
            c_mask     = c_mask,
            c_attn     = d_model,
            intermediate_layer_idx = intermediate_layer_idx,
            patch_size = patch_size,
            scales     = scales,
        )
        assert self.pixel_decoder.num_feature_levels == num_feature_levels, (
            f"pixel_decoder built with {self.pixel_decoder.num_feature_levels} "
            f"scales but num_feature_levels={num_feature_levels}"
        )

        # Query init source — three modes.  query_pos (positional embed)
        # is ALWAYS learnable [num_queries, d_model] regardless of source.
        if query_init_type not in ('learnable', '3d_anchor', 'external'):
            raise ValueError(
                f"query_init_type must be 'learnable' | '3d_anchor' | 'external', "
                f"got {query_init_type!r}"
            )
        self.query_init_type = query_init_type
        self.query_pos = nn.Embedding(num_queries, d_model)
        nn.init.normal_(self.query_pos.weight, std=0.02)
        if query_init_type == 'learnable':
            self.query_feat = nn.Embedding(num_queries, d_model)
            nn.init.normal_(self.query_feat.weight, std=0.02)
        elif query_init_type == '3d_anchor':
            from .pos_encoders import PosEncoder3D
            # Learnable 3D anchor in NORMALIZE space (≈ chunk-local unit cube).
            # User specified init range [-1, 1] (chunk-anchor frame typically
            # spans roughly that range for indoor scenes).
            self.query_anchor = nn.Parameter(torch.empty(1, num_queries, 3))
            nn.init.uniform_(self.query_anchor, -1.0, 1.0)
            self.query_pos3d_encoder = PosEncoder3D(out_dim=d_model)
        elif query_init_type == 'external':
            # Caller will pass init_queries to forward().  Optionally include
            # a projection if their dim ≠ d_model (e.g. CausalBranch=1024,
            # head d_model=256).
            if external_init_dim is not None and external_init_dim != d_model:
                self.query_external_proj = nn.Linear(external_init_dim, d_model)
            else:
                self.query_external_proj = None

        # 2D sinusoidal positional encoding for memory tokens.
        # We share one PE module — at call time we compute per-level pos.
        assert d_model % 2 == 0
        self.pe_layer = _PositionEmbeddingSine2D(num_pos_feats=d_model // 2)

        # Decoder layers — cross-attn, self-attn, FFN per layer (post-norm).
        # Order INSIDE forward: cross → self → ffn  (Mask2Former order).
        self.cross_attn_layers = nn.ModuleList([
            _CrossAttentionLayer(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.self_attn_layers = nn.ModuleList([
            _SelfAttentionLayer(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            _FFNLayer(d_model, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final decoder layer-norm (applied BEFORE prediction heads).
        self.decoder_norm = nn.LayerNorm(d_model)

        # Output heads:
        #  - mask_embed: per-query template in F_mask space.  Also serves
        #    as the L2-normalised descriptor.
        #  - objectness: scalar logit per slot.
        #  - box (optional): 6-dim axis-aligned 3D bounding box.
        self.mask_embed = _MLP(d_model, d_model, c_mask, num_layers=mask_mlp_layers)
        self.objectness = nn.Linear(d_model, 1)
        # Optional dedicated descriptor (identity) head — decoupled from mask_embed.
        self.descriptor_mlp_layers = int(descriptor_mlp_layers)
        self.descriptor_proj = (
            _MLP(d_model, d_model, descriptor_dim, num_layers=descriptor_mlp_layers)
            if descriptor_mlp_layers > 0 else None
        )

        # ── Dual-path fused descriptor head ──
        self.desc_dual_path = bool(desc_dual_path)
        self.desc_pathB_aux = bool(desc_pathB_aux)
        if self.desc_dual_path:
            if query_init_type != 'external':
                raise ValueError(
                    "desc_dual_path=True requires query_init_type='external' "
                    "(path B needs the CausalBranch memory-aware queries)."
                )
            # path B input dim = the raw external (CausalBranch) query dim.
            D_mem = external_init_dim if external_init_dim is not None else d_model
            self.desc_mem_dim = int(D_mem)
            self.desc_ln_mem  = nn.LayerNorm(D_mem)
            self.desc_ln_mask = nn.LayerNorm(d_model)
            self.desc_mlp_B   = _MLP(D_mem, d_model, descriptor_dim,
                                     num_layers=desc_fuse_layers)
            self.desc_mlp_int = _MLP(D_mem + d_model, d_model, descriptor_dim,
                                     num_layers=desc_fuse_layers)
            # Zero-init interaction final layer → starts as no-op (d = path B).
            nn.init.zeros_(self.desc_mlp_int.layers[-1].weight)
            nn.init.zeros_(self.desc_mlp_int.layers[-1].bias)
        self.use_box_head = use_box_head
        # box_mode:
        #   'aabb'       = legacy 6-dim (cx,cy,cz,w,h,d) AABB in CAMERA frame
        #   'obb'        = 12-dim (cx,cy,cz, sx,sy,sz, r1..r6) OBB in CAMERA frame
        #   'world_aabb' = ESAM-style 6-dim AABB in WORLD frame directly.
        #                  Uses BoxAABBHead (PCA prior + small delta on
        #                  mask-pooled image features + 3D moments). No
        #                  camera→world transform anywhere in train / infer.
        # Output dim is decided here; loss function picks the matching path.
        self.box_mode = box_mode
        if use_box_head:
            if box_mode == 'world_aabb':
                # New head: needs (q_norm, mask_logits, mask_features, world_points).
                # Output is FULL-size world AABB; zero-init last layer → init box
                # equals pure PCA prior (already a sensible bbox even at step 0).
                self.box_head = BoxAABBHead(
                    q_dim=d_model, mask_feat_dim=c_mask, hidden=d_model,
                )
            elif box_mode in ('world_amodal', 'world_amodal_grow'):
                # ESAM-style DIRECT amodal head: anchor(centroid)+offset, size.
                # world_amodal      → static pre-generated full-object GT boxes.
                # world_amodal_grow → causal growing-visible GT (box_size_anchor
                #                     recommended: size = visible-extent × exp(Δ)).
                self.box_head = BoxAmodalHead(
                    q_dim=d_model, mask_feat_dim=c_mask, hidden=d_model,
                    detach_inputs=box_head_detach,
                    init_log_size=box_init_log_size,
                    size_anchor=box_size_anchor,
                    query_only=box_query_only,
                    size_extent_aware=box_size_extent_aware,
                )
            else:
                out_dim = 6 if box_mode == 'aabb' else 12
                self.box_head = _MLP(d_model, d_model, out_dim, num_layers=box_mlp_layers)
            # ─── Sensible bias init for OBB head ───
            # Without this, raw output ≈ 0 →  size_half = exp(0) = 1.0  (2m
            # cubes) AND  rot_6d = (0,0,0,0,0,0)  → R = zero matrix (det=0,
            # NOT SO(3); Gram-Schmidt on the zero vector hits the 1e-6 eps
            # singularity → unstable gradients).
            # Bias the LAST linear layer so init output is:
            #   center delta = 0       (anchor mode pred ≈ anchor at init)
            #   log_size    ≈ ln(0.2) → size_half ≈ 0.2 m  (typical object)
            #   rot_6d      = (1,0,0, 0,1,0) → identity rotation matrix
            # Zero the LAST layer's WEIGHTS too so all queries start at the
            # same well-defined identity OBB (delta from anchor); the model
            # then learns to deviate from this base via gradient.
            if box_mode == 'obb':
                with torch.no_grad():
                    last_lin = self.box_head.layers[-1]
                    last_lin.weight.zero_()
                    # bias = [center_delta(3), log_size(3), rot_6d(6)]
                    bias = torch.zeros(out_dim)
                    bias[3:6] = float(torch.log(torch.tensor(0.2)))   # ≈ -1.61
                    bias[6:9]  = torch.tensor([1.0, 0.0, 0.0])         # R col 0
                    bias[9:12] = torch.tensor([0.0, 1.0, 0.0])         # R col 1
                    last_lin.bias.copy_(bias)
            # world_aabb: BoxAABBHead does its own zero-init in __init__.
        else:
            self.box_head = None

        # Thing/stuff (foreground/background) head — per-query binary logit.
        self.use_fgbg_head = bool(use_fgbg_head)
        self.fgbg_head = nn.Linear(d_model, 1) if use_fgbg_head else None

        # ── Instance-aware semantic head (residual MLP) ──
        # 3-layer MLP q → inst_sem_dim; LAST layer zero-init so the head emits a
        # zero residual at start (sem = base).  Only built when enabled.
        self.inst_sem_head = bool(inst_sem_head)
        self.inst_sem_dim  = int(inst_sem_dim)
        if self.inst_sem_head:
            self.sem_proj = _MLP(d_model, d_model, self.inst_sem_dim, num_layers=3)
            with torch.no_grad():
                self.sem_proj.layers[-1].weight.zero_()
                self.sem_proj.layers[-1].bias.zero_()
        else:
            self.sem_proj = None
        self.fastsam_mask_prior = bool(fastsam_mask_prior)   # ESAM-style SAM mask prior (no params)

        # ── Mid-decoder query merge (§3) ──
        self.merge_enable        = bool(merge_enable)
        self.merge_layer         = int(merge_layer)
        self.merge_tau           = float(merge_tau)
        self.merge_dist_veto     = float(merge_dist_veto or 0.0)
        self.merge_act_threshold = float(merge_act_threshold)
        self.merge_active        = False    # runtime curriculum switch (trainer/SLAM sets)
        self.expose_pq_inputs    = bool(expose_pq_inputs)
        if self.merge_enable:
            if not (0 <= self.merge_layer < n_layers - 1):
                raise ValueError(f"merge_layer must leave >=1 refinement layer: "
                                 f"got {self.merge_layer} with n_layers={n_layers}")
            # Follower-feature projection; ZERO-init so merge activation is an
            # exact identity (z = q_leader + Σ w·W·q_follower, W=0 at start).
            # NOTE deliberate deviation from the design doc's LN(q + Σ...):
            # LN(q) != q would break identity-at-activation.
            self.merge_proj = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.merge_proj.weight)
        else:
            self.merge_proj = None

    # ─────────────────────────────────────────────────────────────────────
    # Prediction heads: shared between initial pred and every layer's pred.
    # ─────────────────────────────────────────────────────────────────────
    def _predict(
        self,
        query_features: torch.Tensor,                # [BT, N, d_model]
        F_mask:         torch.Tensor,                # [BT, c_mask, H, W]
        attn_mask_target_size: Optional[tuple],      # (h, w) for the NEXT layer's cross-attn
        # world_aabb head extras — only used when self.box_mode == 'world_aabb'
        # AND `compute_box=True`.  For aux layers we skip the box pred to save
        # PCA / mask-pool cost (aux loss usefully trains mask/obj/desc anyway).
        world_points:   Optional[torch.Tensor] = None,   # [BT, H, W, 3]
        box_scale:      Optional[torch.Tensor] = None,   # [B] or [BT] scalar per chunk
        compute_box:    bool = True,
        q_mem:          Optional[torch.Tensor] = None,   # [BT, N, D_mem]  external memory queries (dual-path desc)
    ):
        """Run mask_embed / objectness on queries, compute mask logits,
        and build the attn_mask for the next decoder layer."""
        BT, N, _ = query_features.shape
        H, W = F_mask.shape[-2:]

        q_norm = self.decoder_norm(query_features)
        mask_embed = self.mask_embed(q_norm)                            # [BT, N, c_mask]
        objectness = self.objectness(q_norm).squeeze(-1)                # [BT, N]

        # Box head — output depends on `box_mode`:
        #   'aabb' (legacy):  [BT, N, 6]  (cx, cy, cz, w, h, d), size via softplus
        #   'obb'  (new):     [BT, N, 12] (cx, cy, cz, sx, sy, sz, r1..r6)
        #                     size via exp (log-space), rot_6d raw (Gram-
        #                     Schmidt at loss / inference time).
        # Box is per-(b, t, n) in frame t's CAMERA coord at canonical scale.
        # Loss transforms pred → world frame (chunk-anchor metric) before
        # comparing with the world OBB GT.  Center semantics depend on the
        # loss-side `box_anchor_mode`:
        #   'abs'    – first 3 dims are absolute center; transformed via
        #              R_c2w @ (c × box_scale) + t_c2w
        #   'anchor' – first 3 dims are a DELTA from the soft-mask centroid;
        #              transformed via R_c2w @ (Δ × box_scale) + anchor_world
        #              (anchor_world = soft_mask · world_pts weighted mean,
        #               computed at loss time — head is unaware of the choice)
        # Mask logits at full F_mask resolution
        mask_logits = torch.einsum('bnc,bchw->bnhw', mask_embed, F_mask)   # [BT, N, H, W]

        # ── Box head ──
        # 'aabb' / 'obb': camera-frame, simple MLP from q_norm
        # 'world_aabb':   world-frame, needs mask_logits + F_mask + world_points
        if self.box_head is None or not compute_box:
            box = None
        elif self.box_mode in ('world_aabb', 'world_amodal', 'world_amodal_grow'):
            assert world_points is not None, \
                f"box_mode='{self.box_mode}' requires world_points to be passed to _predict"
            # Both heads normalize world_points by box_scale internally → output
            # is in NORMALIZE scale (median ~1, DA3-mode-invariant).
            box = self.box_head(q_norm, mask_logits, F_mask, world_points,
                                  box_scale=box_scale)
        else:
            box_raw = self.box_head(q_norm)
            if self.box_mode == 'aabb':
                box_center = box_raw[..., :3]
                box_size   = F.softplus(box_raw[..., 3:6])
                box        = torch.cat([box_center, box_size], dim=-1)     # [BT, N, 6]
            else:  # 'obb'
                box_center    = box_raw[..., 0:3]                          # absolute center
                box_size_half = box_raw[..., 3:6].exp().clamp(max=10.0)    # positive, safety clamp
                box_rot6      = box_raw[..., 6:12]                         # raw 6d (Gram-Schmidt later)
                box           = torch.cat([box_center, box_size_half, box_rot6], dim=-1)   # [BT, N, 12]

        # Descriptor: dual-path fuse (memory + appearance) > dedicated head >
        # L2-normed mask_embed (legacy).
        if self.desc_dual_path and q_mem is not None:
            m  = self.desc_ln_mem(q_mem)                                     # [BT, N, D_mem]
            a  = self.desc_ln_mask(q_norm)                                   # [BT, N, d_model]
            zB  = self.desc_mlp_B(m)                                         # [BT, N, descriptor_dim]
            zAB = self.desc_mlp_int(torch.cat([m, a], dim=-1))              # zero-init final → 0 at start
            descriptor = F.normalize(zB + zAB, dim=-1)                       # [BT, N, descriptor_dim]
        elif self.descriptor_proj is not None:
            descriptor = F.normalize(self.descriptor_proj(q_norm), dim=-1)   # [BT, N, descriptor_dim]
        else:
            descriptor = F.normalize(mask_embed, dim=-1)                    # [BT, N, c_mask]

        # Build attn_mask for next layer's cross-attention.
        if attn_mask_target_size is not None:
            attn_mask = F.interpolate(
                mask_logits, size=attn_mask_target_size,
                mode='bilinear', align_corners=False,
            )
            # σ < 0.5 → block (True); σ ≥ 0.5 → allow (False).
            attn_mask = (attn_mask.sigmoid() < 0.5)
            # Reshape to MultiheadAttention's expected shape:
            #   [BT * num_heads, N_q, h*w]
            h, w = attn_mask_target_size
            attn_mask = attn_mask.flatten(2)                                # [BT, N, h*w]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            attn_mask = attn_mask.flatten(0, 1)                              # [BT*nh, N, h*w]
            attn_mask = attn_mask.detach()                                   # don't propagate
        else:
            attn_mask = None

        return mask_logits, objectness, descriptor, attn_mask, box

    def _fastsam_attn_mask(self, region_id: torch.Tensor,
                           target_size) -> torch.Tensor:
        """Build a layer-0 masked-cross-attention mask from FastSAM regions.

        region_id: [BT, H, W] long — per-pixel FastSAM region index (query i ↔
                   region i; -1 = no region).
        target_size: (h, w) of the level-0 memory feature map.

        Returns attn_mask [BT*n_heads, N, h*w] bool, True = BLOCK (same convention
        as `_predict`): each query is restricted to its own FastSAM region.  A
        query whose region is empty at this resolution is left UNRESTRICTED
        (all-False row) to avoid a NaN softmax over an all-blocked row.
        """
        h, w = target_size
        N  = self.num_queries
        # nearest-downsample the integer id map to the memory resolution
        rid = F.interpolate(region_id.float().unsqueeze(1), size=(h, w),
                            mode='nearest').long().squeeze(1)          # [BT, h, w]
        qidx = torch.arange(N, device=rid.device).view(1, N, 1, 1)
        block = (rid.unsqueeze(1) != qidx)                            # [BT, N, h, w] True=outside region
        block = block.flatten(2)                                       # [BT, N, h*w]
        allblk = block.all(dim=-1)                                     # query with no region pixels here
        block[allblk] = False                                          # → unrestrict (avoid NaN)
        block = block.unsqueeze(1).repeat(1, self.n_heads, 1, 1).flatten(0, 1)
        return block.detach()                                          # [BT*nh, N, h*w]

    def _merge_queries(
        self,
        queries:      torch.Tensor,           # [BT, N, d]   post-merge_layer features
        mask_logits:  torch.Tensor,           # [BT, N, H, W] merge_layer prediction
        objectness:   torch.Tensor,           # [BT, N]
        descriptor:   torch.Tensor,           # [BT, N, D]   L2-normed
        attn_mask:    torch.Tensor,           # [BT*nh, N, h*w] next layer's prior (True=block)
        attn_size:    tuple,                  # (h, w) of attn_mask
        world_points: Optional[torch.Tensor], # [BT, H, W, 3] or None (veto off)
    ):
        """Within-frame leader-based query merge (§3.3–3.6).

        query-scan sequential greedy: iterate active queries by descending
        σ(obj); affinity is computed ONLY against existing leaders (no
        transitive chains); a follower joins the argmax leader iff cosine >
        merge_tau AND centroid distance ≤ merge_dist_veto (both centroids
        valid).  All decisions detached.  Returns (queries, attn_mask,
        follower_mask [BT,N] True=absorbed, assign [BT,N] leader index).
        """
        BT, N, _ = queries.shape
        dev = queries.device
        r     = objectness.detach().sigmoid()                          # [BT, N]
        desc  = descriptor.detach()                                    # unit vecs
        probs = mask_logits.detach().sigmoid()                         # [BT, N, H, W]

        # per-query 3D centroid from thresholded mask (invalid → no veto → no merge)
        cent = None
        cent_ok = torch.zeros(BT, N, dtype=torch.bool, device=dev)
        if world_points is not None and self.merge_dist_veto > 0:
            hard = (probs > 0.5).float()                               # [BT, N, H, W]
            wp = world_points
            if wp.shape[1:3] != probs.shape[2:4]:
                wp = F.interpolate(wp.permute(0, 3, 1, 2),
                                   size=probs.shape[2:4], mode='nearest'
                                   ).permute(0, 2, 3, 1)
            npix = hard.sum(dim=(2, 3))                                # [BT, N]
            cent = torch.einsum('bnhw,bhwc->bnc', hard, wp) / npix.clamp_min(1).unsqueeze(-1)
            cent_ok = npix > 0

        follower_mask = torch.zeros(BT, N, dtype=torch.bool, device=dev)
        assign = torch.arange(N, device=dev).unsqueeze(0).repeat(BT, 1)  # [BT, N]
        w_flat = torch.zeros(BT, N, device=dev)     # follower weight w̃ = cos·r (0 for non-followers)

        for b in range(BT):
            act = (r[b] > self.merge_act_threshold).nonzero(as_tuple=True)[0]
            if act.numel() < 2:
                continue
            order = act[torch.argsort(r[b, act], descending=True)]
            leaders: List[int] = []
            for j in order.tolist():
                merged = False
                if leaders:
                    lead_t = torch.tensor(leaders, device=dev)
                    cos = desc[b, lead_t] @ desc[b, j]                 # [n_lead]
                    if self.merge_dist_veto > 0:
                        if cent is None or not bool(cent_ok[b, j]):
                            cos = torch.full_like(cos, -1.0)           # unverifiable → no merge
                        else:
                            dist = (cent[b, lead_t] - cent[b, j]).norm(dim=-1)
                            veto = (~cent_ok[b, lead_t]) | (dist > self.merge_dist_veto)
                            cos = cos.masked_fill(veto, -1.0)
                    best = int(cos.argmax())
                    if float(cos[best]) > self.merge_tau:
                        li = leaders[best]
                        follower_mask[b, j] = True
                        assign[b, j] = li
                        w_flat[b, j] = float(cos[best]) * float(r[b, j])
                        merged = True
                if not merged:
                    leaders.append(j)

        if not follower_mask.any():
            return queries, attn_mask, follower_mask, assign

        # ── feature merge: z_l = q_l + Σ_j w̄_lj · W · q_j  (W zero-init) ──
        # scatter follower contributions onto their leaders; weights normalized
        # per leader over its followers.  All weights detached scalars.
        proj = self.merge_proj(queries)                                # [BT, N, d]
        wsum = torch.zeros(BT, N, device=dev)
        wsum.scatter_add_(1, assign, w_flat)                           # per-leader Σw
        w_norm = w_flat / wsum.gather(1, assign).clamp_min(1e-6)       # [BT, N]
        w_norm = w_norm * follower_mask.float()
        contrib = torch.zeros_like(queries)
        contrib.scatter_add_(1, assign.unsqueeze(-1).expand_as(proj),
                             proj * w_norm.unsqueeze(-1))
        queries = queries + contrib

        # ── attention prior union (weighted max, §3.5) at attn resolution ──
        # follower prob weighted by w̃ = cos·r (UN-normalized, ≤1); leader = 1.
        h, w = attn_size
        p = F.interpolate(probs, size=(h, w), mode='bilinear',
                          align_corners=False)                         # [BT, N, h, w]
        wp_ = torch.where(follower_mask, w_flat,
                          torch.ones_like(w_flat))                     # [BT, N]
        p_w = p * wp_.view(BT, N, 1, 1)
        union = torch.zeros_like(p)
        union.scatter_reduce_(1, assign.view(BT, N, 1, 1).expand_as(p_w),
                              p_w, reduce='amax')
        # leader rows get the union; follower rows keep their own prior (they
        # still run through late layers but are ignored downstream).
        _nfol = torch.zeros(BT, N, device=dev)
        _nfol.scatter_add_(1, assign, follower_mask.float())
        is_leader_merged = _nfol > 0                                   # [BT, N]
        new_block = (union < 0.5).flatten(2)                           # [BT, N, h*w]
        old_block = attn_mask.view(BT, self.n_heads, N, h * w)[:, 0]   # [BT, N, h*w]
        take = is_leader_merged.unsqueeze(-1)
        blk = torch.where(take, new_block, old_block)
        allblk = blk.all(dim=-1)
        if allblk.any():
            blk = blk.clone()
            blk[allblk] = False
        attn_mask = blk.unsqueeze(1).repeat(1, self.n_heads, 1, 1).flatten(0, 1).detach()
        return queries, attn_mask, follower_mask, assign

    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        output_tokens_list: List[torch.Tensor],
        images:             torch.Tensor,             # [B, T, 3, H, W]
        patch_start_idx:    int = 0,
        world_points:       Optional[torch.Tensor] = None,   # [B, T, H, W, 3]  (world_aabb only)
        box_scale:          Optional[torch.Tensor] = None,   # [B] per-chunk normalizer (world_aabb only)
        debug_dump_dir:     Optional[str] = None,            # if set, dumps per-frame PCA / pred / mask-pts PLY (world_aabb only)
        debug_dump_prefix:  str = '',
        init_queries:       Optional[torch.Tensor] = None,   # [B, N_q, *] external init (Idea 2)
        pq_module:          Optional[torch.nn.Module] = None,  # model-level PQHead: compute P/Q
                                                               # IN-HEAD (small outputs cross the
                                                               # checkpoint boundary, not F_mask)
        pq_sem_feat:        Optional[torch.Tensor] = None,     # [B,T,C,Hf,Wf] detached sem feat
        pq_sem_backbone     = None,   # TextRegion P/Q: frozen teacher (None = dense pooling)
        pq_shared:          Optional[dict] = None,   # TextRegion P/Q: cached shared_v_forward
        fastsam_region_id:  Optional[torch.Tensor] = None,   # [B,T,H,W] or [BT,H,W] long: per-pixel
                                                             # FastSAM region id (query i ↔ region i,
                                                             # -1=none).  Used iff fastsam_mask_prior.
    ) -> Dict[str, torch.Tensor]:
        B, T = images.shape[:2]
        H, W = images.shape[-2:]
        BT = B * T
        # Flatten world_points for _predict (which works in BT dimension)
        wp_bt = (world_points.reshape(BT, *world_points.shape[2:])
                  if world_points is not None else None)

        # ── Pixel decoder ──
        pix = self.pixel_decoder(output_tokens_list, images,
                                   patch_start_idx=patch_start_idx)
        F_mask  = pix['mask_features']                                       # [BT, c_mask, H, W]
        ms_feat = pix['multi_scale_features']                                # list of L tensors
        sizes   = pix['sizes']                                                # list of (h, w)
        L = self.num_feature_levels

        # ── Prepare per-level memory + positional encoding ──
        memory_per_level     = []
        memory_pos_per_level = []
        for lv in range(L):
            f = ms_feat[lv]                                                    # [BT, d_model, h, w]
            h, w = sizes[lv]
            memory = f.flatten(2).transpose(1, 2)                              # [BT, h*w, d_model]
            pos = self.pe_layer(BT, h, w, device=memory.device, dtype=memory.dtype)
            pos = pos.flatten(2).transpose(1, 2)                                # [BT, h*w, d_model]
            memory_per_level.append(memory)
            memory_pos_per_level.append(pos)

        # ── Initial queries ──
        # ── Initial queries: three modes (see __init__) ──
        # q_mem_bt: the raw external (CausalBranch, memory-aware) queries kept at
        # their native dim [BT, N, D_mem] for the dual-path descriptor (path B).
        # None for non-external modes.
        q_mem_bt = None
        if self.query_init_type == 'learnable':
            queries = self.query_feat.weight.unsqueeze(0).expand(BT, -1, -1)   # [BT, N, d]
        elif self.query_init_type == '3d_anchor':
            # query_anchor: [1, N, 3] normalize-space (~ unit cube).  Fourier PE → MLP → [1, N, d_model].
            q_init = self.query_pos3d_encoder(self.query_anchor)               # [1, N, d_model]
            queries = q_init.expand(BT, -1, -1).contiguous()
        else:   # 'external'
            assert init_queries is not None, \
                "query_init_type='external' requires init_queries=… in forward()"
            ext = init_queries
            # Accept per-frame [B, T, N, D_in] (current CausalBranch) or legacy
            # chunk-level [B, N, D_in] (broadcast to all T).
            if ext.dim() == 4:
                q_mem_bt = ext.reshape(BT, self.num_queries, ext.shape[-1]).contiguous()
            else:
                q_mem_bt = ext.unsqueeze(1).expand(B, T, self.num_queries, ext.shape[-1]) \
                              .reshape(BT, self.num_queries, ext.shape[-1]).contiguous()
            # Path-A init = q_mem projected to d_model (decoder operates at d_model).
            if hasattr(self, 'query_external_proj') and self.query_external_proj is not None:
                queries = self.query_external_proj(q_mem_bt)                    # [BT, N, d_model]
            else:
                queries = q_mem_bt
        query_pos = self.query_pos.weight.unsqueeze(0).expand(BT, -1, -1)      # [BT, N, d]

        # ── Initial prediction (BEFORE any decoder layer) to get attn_mask for layer 0 ──
        # Init prediction never needs box (skipped via compute_box=False) to avoid
        # running the heavy PCA / mask-pool on raw query_feat tokens.
        m_logits, obj, desc, attn_mask, box = self._predict(
            queries, F_mask, attn_mask_target_size=sizes[0],
            world_points=wp_bt, box_scale=box_scale, compute_box=False,
            q_mem=q_mem_bt,
        )

        # ── FastSAM mask prior (ESAM-style): override layer-0 attn_mask with each
        # query's own FastSAM region, so the first decoder layer's masked
        # cross-attention STARTS from the strong in-domain FastSAM segment rather
        # than the (possibly poor, OOD) initial prediction.  Layers 1..L-1 then
        # refine using the model's own predicted masks. ──
        if self.fastsam_mask_prior and fastsam_region_id is not None:
            rid = fastsam_region_id
            if rid.dim() == 4:                                   # [B,T,H,W] → [BT,H,W]
                rid = rid.reshape(BT, *rid.shape[-2:])
            attn_mask = self._fastsam_attn_mask(rid, sizes[0])

        all_mask_logits = [m_logits]
        all_objectness  = [obj]
        all_descriptors = [desc]
        all_boxes       = [box]

        # ── Decoder layer stack (with cyclic multi-scale + masked attention) ──
        follower_mask = None                   # [BT, N] bool, set at merge point
        merge_assign  = None                   # [BT, N] long leader index
        do_merge = (self.merge_enable and self.merge_active
                    and self.merge_proj is not None)
        for i in range(self.n_layers):
            level = i % L

            queries = self.cross_attn_layers[i](
                queries=queries, memory=memory_per_level[level],
                query_pos=query_pos, memory_pos=memory_pos_per_level[level],
                attn_mask=attn_mask,
            )
            queries = self.self_attn_layers[i](queries, query_pos,
                                               key_padding_mask=follower_mask)
            queries = self.ffn_layers[i](queries)

            next_level = (i + 1) % L
            # For world_aabb mode: only compute box at the FINAL layer to save
            # the heavy PCA / mask-pool compute (~9× speedup on box path).
            # Other modes ('aabb', 'obb'): compute every layer (cheap MLP).
            is_final = (i == self.n_layers - 1)
            # world_aabb / world_amodal: only compute box at the FINAL layer (heavy
            # mask-pool / centroid + GT lookup is final-only).  aabb/obb: every layer.
            compute_box = (self.box_head is not None) and (
                self.box_mode not in ('world_aabb', 'world_amodal', 'world_amodal_grow') or is_final
            )
            m_logits, obj, desc, attn_mask, box = self._predict(
                queries, F_mask, attn_mask_target_size=sizes[next_level],
                world_points=wp_bt, box_scale=box_scale, compute_box=compute_box,
                q_mem=q_mem_bt,
            )
            all_mask_logits.append(m_logits)
            all_objectness .append(obj)
            all_descriptors.append(desc)
            all_boxes      .append(box)

            # ── Mid-decoder merge (§3): consume THIS layer's (supervised)
            # prediction, consolidate N fragment queries → K object queries,
            # union the cluster priors, deactivate followers for layers > i. ──
            if do_merge and i == self.merge_layer and not is_final:
                queries, attn_mask, follower_mask, merge_assign = \
                    self._merge_queries(
                        queries, m_logits, obj, desc,
                        attn_mask, sizes[next_level], wp_bt,
                    )
                if not follower_mask.any():
                    follower_mask = None       # keep self-attn fast path

        # ── Assemble outputs ──
        # Last entry = final prediction; earlier entries = auxiliary supervision.
        def _shape_bt_to_bt(x, *rest):
            # Reshape [BT, ...] → [B, T, ...]
            return x.reshape(B, T, *rest)

        final = {
            'mask_logits': _shape_bt_to_bt(all_mask_logits[-1], self.num_queries, H, W),
            'objectness':  _shape_bt_to_bt(all_objectness [-1], self.num_queries),
            'descriptor':  _shape_bt_to_bt(all_descriptors[-1], self.num_queries, self.descriptor_dim),
        }
        # Thing/stuff logit from the FINAL decoder queries (1 = foreground thing).
        if self.fgbg_head is not None:
            final['fgbg_logits'] = _shape_bt_to_bt(
                self.fgbg_head(queries).squeeze(-1), self.num_queries)            # [B, T, N]
        # Instance-aware semantic RESIDUAL from the FINAL decoder queries.
        # The frozen-TIPS base is added in the model → final sem = base + this.
        if self.sem_proj is not None:
            final['inst_sem_raw'] = _shape_bt_to_bt(
                self.sem_proj(queries), self.num_queries, self.inst_sem_dim)      # [B, T, N, inst_sem_dim]
        # ── Merge outputs (§3.6/§3.7): active_mask True = alive proposal
        # (leaders + unmerged); merge_assign = leader index per query (own
        # index if not a follower).  Consumers: loss (post-merge OTO with
        # allowed-OR + follower exclusion) and inference proposal filtering.
        if pq_module is not None:
            # P/Q computed HERE (inside the possibly-checkpointed head) so only
            # [B,T,N] logits leave the boundary — the previous design exported
            # full-res F_mask through the checkpoint (prime NaN-trigger suspect).
            from .pq_head import build_pq_inputs, pooled_matmul
            _dev_t = 'cuda' if queries.is_cuda else 'cpu'
            with torch.autocast(device_type=_dev_t, enabled=False):
                qn_f = self.decoder_norm(queries).detach().float()        # [BT,N,D]
                mlf  = all_mask_logits[-1].detach().float()               # [BT,N,H,W]
                Fm_f = F_mask.detach().float()
                semf = (pq_sem_feat.reshape(BT, *pq_sem_feat.shape[2:]).detach().float()
                        if pq_sem_feat is not None else None)   # None in TextRegion mode
                _pps, _pqs, _pvs = [], [], []
                for _i in range(BT):
                    # feat_1x1 stays inside @no_grad build (it stays a FROZEN
                    # random projection): routing the raw 256ch crop through
                    # pq_module.forward to train it re-triggered the trunk-NaN
                    # memory corruption (official run died 06:06 iter<400;
                    # the 17ch-crop bf16 config survived 400+). JL-projection
                    # crops are what offline dump-era heads learned on anyway.
                    _inp, _val = build_pq_inputs(mlf[_i], Fm_f[_i], qn_f[_i],
                                                 feat_1x1=pq_module.feat_1x1)
                    _semp = None
                    if pq_module.proj_sem is not None:
                        if pq_sem_backbone is not None and pq_shared is not None:
                            # TextRegion P/Q: pool the frozen teacher's V under
                            # this query's soft mask.  Same quantity the dense
                            # branch below approximates ("what does this query's
                            # region mean"), but mask-aware -- and it is what the
                            # instance semantics are read out with, so P/Q now
                            # scores the representation it actually ranks.  It
                            # also lets the dense head be dropped entirely.
                            _th, _tw = pq_shared['patch_h'], pq_shared['patch_w']
                            _pr = F.interpolate(mlf[_i].sigmoid().unsqueeze(1),
                                                size=(_th, _tw), mode='bilinear',
                                                align_corners=False)[:, 0]
                            # The teacher's cached V is bf16 while this whole P/Q
                            # block runs with autocast DISABLED (fp32), so the
                            # pooling has to be re-entered under bf16 or the
                            # post-attention matmuls hit mixed dtypes.
                            with torch.autocast(device_type=_dev_t,
                                                dtype=torch.bfloat16, enabled=True):
                                _semp = pq_sem_backbone.textregion_from_shared(
                                    pq_shared, _pr.flatten(1).clamp(min=0), frame_idx=_i)
                            _semp = F.normalize(_semp.float(), dim=-1)
                        else:
                            _sf = semf[_i]
                            _pr = F.interpolate(mlf[_i].sigmoid().unsqueeze(1),
                                                size=_sf.shape[-2:], mode='bilinear',
                                                align_corners=False)[:, 0]
                            _w = _pr.flatten(1)
                            # [ma-slam local fix] K-chunked GEMM — see pq_head.pooled_matmul
                            _semp = pooled_matmul(_w, _sf.flatten(1).t()) / _w.sum(1, keepdim=True).clamp_min(1e-6)
                            _semp = F.normalize(_semp, dim=-1)
                    # bf16 for the pq_module call ONLY: the fp32 conv
                    # backward kernels here corrupt memory on this rig
                    # (bisect 2026-07-28: loss-detached survives, backward
                    # kills trunk) — bf16 dispatches different kernels.
                    with torch.autocast(device_type=_dev_t, dtype=torch.bfloat16,
                                        enabled=True):
                        _pl, _ql = pq_module(_inp['q_norm'], _inp['pooled'],
                                             _inp['crop'], _inp['scale'],
                                             sem_pooled=_semp)
                    _pl, _ql = _pl.float(), _ql.float()
                    _pps.append(_pl); _pqs.append(_ql); _pvs.append(_val)
            final['pq_p'] = _shape_bt_to_bt(torch.stack(_pps), self.num_queries)
            final['pq_q'] = _shape_bt_to_bt(torch.stack(_pqs), self.num_queries)
            final['pq_valid'] = _shape_bt_to_bt(torch.stack(_pvs), self.num_queries)
            # Targeted sync: 6/6 async runs NaN'd stochastically in the
            # CausalBranch while fully-synchronized runs (forensic /
            # CUDA_LAUNCH_BLOCKING) survive — a kernel in/near this block
            # (roi_align / index ops) appears to corrupt memory when overlapped.
            # This fence costs ~nothing per chunk and closes that window.
            if queries.is_cuda:
                torch.cuda.synchronize()
        if self.expose_pq_inputs:
            # final-layer q_norm (post decoder_norm, matching _predict's view)
            final['_pq_qnorm'] = _shape_bt_to_bt(
                self.decoder_norm(queries).detach(), self.num_queries, -1)
            final['_pq_fmask'] = F_mask.detach().reshape(
                B, T, *F_mask.shape[1:])                                     # [B,T,C,H,W]
        if self.merge_enable:
            if follower_mask is None:
                follower_mask = torch.zeros(BT, self.num_queries, dtype=torch.bool,
                                            device=queries.device)
                merge_assign = torch.arange(self.num_queries, device=queries.device
                                            ).unsqueeze(0).repeat(BT, 1)
            final['active_mask']  = _shape_bt_to_bt(~follower_mask, self.num_queries)
            final['merge_assign'] = _shape_bt_to_bt(merge_assign, self.num_queries)
        # ── Path-B-only descriptor (memory path) for the aux contrastive loss.
        # Depends only on q_mem (constant across decoder layers) → computed once.
        if self.desc_dual_path and self.desc_pathB_aux and q_mem_bt is not None:
            zB = self.desc_mlp_B(self.desc_ln_mem(q_mem_bt))                     # [BT, N, D_desc]
            descriptor_pathB = F.normalize(zB, dim=-1)
            final['descriptor_pathB'] = _shape_bt_to_bt(
                descriptor_pathB, self.num_queries, self.descriptor_dim)
        _box_dim = {'aabb': 6, 'obb': 12, 'world_aabb': 6, 'world_amodal': 6,
                    'world_amodal_grow': 6}[self.box_mode]
        if all_boxes[-1] is not None:
            final['box'] = _shape_bt_to_bt(
                all_boxes[-1], self.num_queries, _box_dim,
            )
            final['box_mode'] = self.box_mode

            # ── Optional debug PLY dump (model-internal, world_aabb only) ──
            if (debug_dump_dir is not None
                    and self.box_mode in ('world_aabb', 'world_amodal', 'world_amodal_grow')
                    and wp_bt is not None
                    and box_scale is not None):
                # all_boxes[-1]: [BT, N, 6]  NORMALIZE scale
                # objectness:    [BT, N]    raw logits
                dump_world_aabb_debug_per_frame(
                    mask_logits         = all_mask_logits[-1],
                    objectness          = all_objectness[-1],
                    world_points        = wp_bt,
                    box_scale           = box_scale,
                    final_box_normalize = all_boxes[-1],
                    save_dir            = debug_dump_dir,
                    prefix              = debug_dump_prefix,
                    B                   = B,
                    T                   = T,
                )
        if self.return_aux and len(all_mask_logits) > 1:
            aux_list = []
            for k in range(len(all_mask_logits) - 1):
                aux = {
                    'mask_logits': _shape_bt_to_bt(all_mask_logits[k], self.num_queries, H, W),
                    'objectness':  _shape_bt_to_bt(all_objectness [k], self.num_queries),
                    'descriptor':  _shape_bt_to_bt(all_descriptors[k], self.num_queries, self.descriptor_dim),
                }
                if all_boxes[k] is not None:
                    aux['box'] = _shape_bt_to_bt(all_boxes[k], self.num_queries, _box_dim)
                # world_aabb: all_boxes[k] is None for non-final layers (skipped)
                # → aux dict simply omits 'box' → loss skips aux box loss for this layer.
                aux_list.append(aux)
            final['aux_outputs'] = aux_list
        return final
