"""
opense3r/loss_query.py
=======================
Loss module for the query-based instance head.

Inputs (from the head):
    mask_logits  [B, T, N, H, W]
    objectness   [B, T, N]
    descriptor   [B, T, N, D]              L2-normalized

GT (from dataset):
    instance_mask  [B, T, H, W]  long.  0 = unlabeled / void.
                                Positive values are scene-consistent IDs.

Pipeline:
    1. Build per-frame GT masks + IDs from `instance_mask`.
    2. Hungarian matching per frame (point-sampled).
    3. Compute three loss terms:
         L_mask = w_bce·BCE + w_dice·Dice  on matched pairs (FULL resolution)
         L_obj  = BCE on objectness with target = (matched? 1 : 0) for ALL N queries
         L_desc = NT-Xent supervised contrastive on matched descriptors
                  across all frames in the chunk (warm-up after K1 epochs)

Loss schedule:
    Phase 1 (epoch < descriptor_warmup_epochs):
        L = w_mask·L_mask + w_obj·L_obj            (descriptor weight = 0)
    Phase 2 (epoch >= descriptor_warmup_epochs):
        L = w_mask·L_mask + w_obj·L_obj + w_desc·L_desc

The criterion exposes `set_epoch(epoch)` so the training loop can update
the schedule on each epoch.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .query_matcher import HungarianMatcherPointSampled, build_per_frame_gt_masks
from .loss import OpenSe3rLoss
from .box_gt   import (
    build_gt_camera_boxes, build_gt_world_obbs, build_gt_world_aabbs,
    build_amodal_gt_aabbs, build_growing_gt_aabbs,
    rot_6d_to_matrix, obb_12d_to_dict,
    transform_obb_cam_to_world,
    compute_anchor_world, obb_corner_match_loss,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3D AABB DIoU loss (cx,cy,cz,w,h,d format)
# ─────────────────────────────────────────────────────────────────────────────

def box6_to_bminmax(box: torch.Tensor) -> tuple:
    """[P, 6] (cx,cy,cz,w,h,d) → (bmin [P,3], bmax [P,3])."""
    center = box[..., :3]
    half   = box[..., 3:6] * 0.5
    return center - half, center + half


def diou_3d_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Distance-IoU loss for axis-aligned 3D bboxes.

    Args:
        pred / gt: [P, 6] in (cx, cy, cz, w, h, d) format, same scale.
    Returns:
        Scalar mean loss = mean(1 - DIoU).

    DIoU_3D = IoU - (d²_centers) / (c²_diagonal_enclosing).
    Always has a non-zero gradient (even when IoU = 0) thanks to the
    centre-distance term, which is the whole point vs vanilla 1-IoU.
    """
    if pred.shape[0] == 0:
        return pred.new_zeros(())

    p_min, p_max = box6_to_bminmax(pred)
    g_min, g_max = box6_to_bminmax(gt)

    # IoU
    inter_min = torch.maximum(p_min, g_min)
    inter_max = torch.minimum(p_max, g_max)
    inter     = (inter_max - inter_min).clamp(min=0).prod(dim=-1)            # [P]
    vol_p     = (p_max - p_min).clamp(min=1e-6).prod(dim=-1)
    vol_g     = (g_max - g_min).clamp(min=1e-6).prod(dim=-1)
    union     = vol_p + vol_g - inter
    iou       = inter / union.clamp(min=1e-8)                                # [P]

    # Enclosing box diagonal²
    enc_min = torch.minimum(p_min, g_min)
    enc_max = torch.maximum(p_max, g_max)
    enc_diag2 = ((enc_max - enc_min) ** 2).sum(dim=-1).clamp(min=1e-8)        # [P]

    # Centre-distance²
    c_p = (p_min + p_max) * 0.5
    c_g = (g_min + g_max) * 0.5
    cdist2 = ((c_p - c_g) ** 2).sum(dim=-1)                                   # [P]

    diou = iou - cdist2 / enc_diag2
    return (1.0 - diou).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Box loss helper: axis-aligned 3D bbox L1
# ─────────────────────────────────────────────────────────────────────────────

def obb_world_loss(
    pred12: torch.Tensor,            # [P, 12]  matched preds (chunk-local frame)
    gt12:   torch.Tensor,            # [P, 12]  matched GTs   (same frame)
    w_center: float = 1.0,
    w_size:   float = 1.0,
    w_rot:    float = 1.0,
) -> Dict[str, torch.Tensor]:
    """OBB loss in a shared (chunk-local / world) frame.

    Pred and GT are expected to already be in the same frame — for our
    setup, both are in the chunk-anchor frame (DA3's output frame at
    training; coord-aligned global frame at SLAM-time inference).

    Three components:
      - L_center : smooth-L1 on box center
      - L_size   : smooth-L1 in log-space on half-extents (log positivity)
      - L_rot    : squared Frobenius distance between rotation matrices

    Returns dict with the three terms + summed 'L_box'.
    """
    if pred12.shape[0] == 0:
        z = pred12.new_zeros(())
        return {'L_center': z, 'L_size': z, 'L_rot': z, 'L_box': z}

    pc, ps, pr6 = pred12[..., :3], pred12[..., 3:6], pred12[..., 6:12]
    gc, gs, gr6 = gt12[...,   :3], gt12[...,   3:6], gt12[...,   6:12]

    L_center = F.smooth_l1_loss(pc, gc)
    L_size   = F.smooth_l1_loss(
        torch.log(ps.clamp(min=1e-3)),
        torch.log(gs.clamp(min=1e-3)),
    )
    pR = rot_6d_to_matrix(pr6)
    gR = rot_6d_to_matrix(gr6)
    L_rot = ((pR - gR) ** 2).sum(dim=(-2, -1)).mean()

    L_box = w_center * L_center + w_size * L_size + w_rot * L_rot
    return {'L_center': L_center, 'L_size': L_size, 'L_rot': L_rot,
            'L_box': L_box}


def gt_box_from_mask_pts(
    gt_mask:       torch.Tensor,    # [H, W] binary
    world_points:  torch.Tensor,    # [H, W, 3]
) -> Optional[torch.Tensor]:
    """Compute axis-aligned 3D bbox (cx, cy, cz, sx, sy, sz) from the
    world_points inside a GT binary mask.  Returns None if mask empty.

    Format: [cx, cy, cz] = center, [sx, sy, sz] = full size (max-min).
    """
    if not gt_mask.any():
        return None
    pts = world_points[gt_mask.bool()]
    if pts.numel() == 0:
        return None
    finite = torch.isfinite(pts).all(dim=-1)
    pts = pts[finite]
    if pts.numel() == 0:
        return None
    bmin = pts.min(dim=0).values     # [3]
    bmax = pts.max(dim=0).values
    center = (bmin + bmax) * 0.5
    size   = (bmax - bmin).clamp(min=1e-3)   # avoid degenerate zero-size
    return torch.cat([center, size], dim=-1)  # [6]


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair mask losses (full resolution, only on matched pairs)
# ─────────────────────────────────────────────────────────────────────────────

def _mask_bce_loss(pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits averaged over pixels (and pairs).

    Args:
        pred_logits: [P, H, W]  matched-pair predicted mask logits
        gt:          [P, H, W]  binary GT masks
    """
    return F.binary_cross_entropy_with_logits(
        pred_logits, gt, reduction='mean',
    )


def _sigmoid_focal_loss(
    logits:  torch.Tensor,
    targets: torch.Tensor,
    alpha:   float = 0.25,
    gamma:   float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Standard sigmoid focal loss (mean over all entries).

    L = -α_t · (1 - p_t)^γ · log p_t

    Where p_t = p if target=1 else (1-p);  α_t = α if target=1 else (1-α).
    Reduces BCE's gradient contribution from easy negatives (correctly
    predicted background queries) → matched-vs-unmatched imbalance no
    longer dominates the loss.  Standard Mask2Former / DETR setting:
    α=0.25, γ=2.0.
    """
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss if reduction == 'none' else loss.mean()


def _mask_dice_loss(pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Soft Dice loss averaged over pairs (per-pair Dice then mean).

    Args:
        pred_logits: [P, H, W]
        gt:          [P, H, W]
    """
    P = pred_logits.shape[0]
    if P == 0:
        return pred_logits.new_zeros(())
    pred = pred_logits.sigmoid().flatten(1)              # [P, H*W]
    target = gt.flatten(1)                                # [P, H*W]
    numerator   = 2 * (pred * target).sum(dim=-1)         # [P]
    denominator = pred.sum(dim=-1) + target.sum(dim=-1)
    dice = (numerator + 1) / (denominator + 1)            # [P]
    return (1 - dice).mean()


def _mask_iou_loss(pred_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Soft mask-IoU (Jaccard) loss = 1 - soft_IoU, averaged over pairs.
    Complements Dice (ESAM uses BCE + Dice + IoU): IoU's union denominator
    penalises over-shoot harder than Dice → sharper mask boundaries.
        pred_logits / gt: [P, H, W]
    """
    P = pred_logits.shape[0]
    if P == 0:
        return pred_logits.new_zeros(())
    pred   = pred_logits.sigmoid().flatten(1)             # [P, H*W]
    target = gt.flatten(1)
    inter = (pred * target).sum(dim=-1)                   # [P]
    union = pred.sum(dim=-1) + target.sum(dim=-1) - inter
    iou   = (inter + 1) / (union + 1)
    return (1 - iou).mean()


# ── PointRend / Mask2Former importance point sampling (boundary-focused mask loss) ──
def _point_sample(feat, coords):
    """Bilinear-sample feat [P,C,H,W] at normalized coords [P,K,2] in [0,1] → [P,C,K]."""
    grid = 2.0 * coords - 1.0                                   # [0,1] → [-1,1]
    out = F.grid_sample(feat, grid.unsqueeze(2), align_corners=False)   # [P,C,K,1]
    return out.squeeze(-1)


def _uncertain_point_coords(mask_logits, num_points, oversample=3.0, imp_ratio=0.75):
    """Mask2Former sampling: pick `num_points` coords, ~imp_ratio of them at the
    MOST UNCERTAIN locations (|logit|≈0 → the boundary), rest uniform-random.
    mask_logits [P,1,H,W] → coords [P,num_points,2] in [0,1]."""
    P = mask_logits.shape[0]
    dev = mask_logits.device
    n_over = int(num_points * oversample)
    coords = torch.rand(P, n_over, 2, device=dev)
    logits = _point_sample(mask_logits, coords)                # [P,1,n_over]
    unc = -(logits.abs())[:, 0, :]                             # uncertainty [P,n_over]
    n_unc = int(imp_ratio * num_points)
    idx = unc.topk(n_unc, dim=1)[1]                            # [P,n_unc]
    shift = (n_over * torch.arange(P, device=dev))[:, None]
    coords = coords.view(-1, 2)[(idx + shift).view(-1)].view(P, n_unc, 2)
    n_rand = num_points - n_unc
    if n_rand > 0:
        coords = torch.cat([coords, torch.rand(P, n_rand, 2, device=dev)], dim=1)
    return coords


def _mask_losses_pointsampled(pred_logits, gt, num_points, oversample, imp_ratio, want_iou):
    """BCE + Dice (+ optional soft-IoU) computed on importance-sampled points
    instead of the full mask.  pred_logits / gt: [P,H,W].  Focuses gradient on
    uncertain boundary pixels → sharper, less-fragmented masks (Mask2Former)."""
    P = pred_logits.shape[0]
    if P == 0:
        z = pred_logits.new_zeros(())
        return z, z, z
    pl = pred_logits[:, None]                                  # [P,1,H,W]
    gm = gt[:, None].float()
    with torch.no_grad():
        coords = _uncertain_point_coords(pl, num_points, oversample, imp_ratio)
        pt_gt = _point_sample(gm, coords).squeeze(1)           # [P,K] (0/1, bilinear)
    pt_logit = _point_sample(pl, coords).squeeze(1)            # [P,K]
    l_bce  = _mask_bce_loss(pt_logit, pt_gt)
    l_dice = _mask_dice_loss(pt_logit, pt_gt)
    l_iou  = _mask_iou_loss(pt_logit, pt_gt) if want_iou else pred_logits.new_zeros(())
    return l_bce, l_dice, l_iou


# ─────────────────────────────────────────────────────────────────────────────
# Descriptor contrastive loss
# ─────────────────────────────────────────────────────────────────────────────

def _nt_xent_supervised(
    descriptors: torch.Tensor,         # [P, D]  L2-normalized
    ids:         torch.Tensor,         # [P]     long, scene-consistent IDs
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive (NT-Xent variant).

    For each anchor i:
        positives = {j : j != i AND ids[j] == ids[i]}
        negatives = {j : j != i AND ids[j] != ids[i]}
    Loss_i = -log( sum_{j in pos} exp(sim(i,j)/τ)
                  / sum_{j in pos ∪ neg} exp(sim(i,j)/τ) )

    Anchors with no positive partners are skipped.  Returns the mean over
    valid anchors.  If no valid anchor exists, returns zero (no signal).
    """
    P = descriptors.shape[0]
    if P < 2:
        return descriptors.new_zeros(())

    sim = descriptors @ descriptors.t() / temperature                   # [P, P]

    # Build (P, P) mask: True at same-id off-diagonal positions.
    same_id = ids.unsqueeze(0) == ids.unsqueeze(1)                       # [P, P]
    eye_mask = torch.eye(P, dtype=torch.bool, device=descriptors.device)
    pos_mask = same_id & ~eye_mask                                       # [P, P]
    valid_mask = ~eye_mask                                               # all off-diagonal

    # Numeric stability — subtract row max before exp.
    sim_masked = sim.masked_fill(eye_mask, float('-inf'))
    row_max = sim_masked.max(dim=1, keepdim=True).values
    sim_shift = sim_masked - row_max.detach()

    exp_sim = sim_shift.exp() * valid_mask                               # zero on diag
    denom = exp_sim.sum(dim=1)                                            # [P]
    numer = (exp_sim * pos_mask).sum(dim=1)                              # [P]

    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return descriptors.new_zeros(())

    # Log( numer / denom ) only where has_pos.
    log_prob = (numer[has_pos] + 1e-12).log() - (denom[has_pos] + 1e-12).log()
    return -log_prob.mean()


def _desc_push_only(
    descriptors: torch.Tensor,         # [P, D]  L2-normalized
    ids:         torch.Tensor,         # [P]     long, per-frame instance IDs
    margin:      float = 0.1,
) -> torch.Tensor:
    """Within-frame DISCRIMINATIVE descriptor loss (push-only).

    Single-frame has ~1 matched descriptor per instance → no same-id positive
    pairs → NT-Xent yields 0.  This instead penalizes high cosine similarity
    between descriptors of DIFFERENT instances (hinge above `margin`), so the
    head learns to SEPARATE instances within the frame.  No pull term — the
    cross-view "same instance" consistency is left for multi-view training.

    Returns the mean hinge over different-id off-diagonal pairs; 0 if none.
    """
    P = descriptors.shape[0]
    if P < 2:
        return descriptors.new_zeros(())
    cos = descriptors @ descriptors.t()                                  # [P,P] cosine (L2-normed)
    eye = torch.eye(P, dtype=torch.bool, device=descriptors.device)
    diff_id = (ids.unsqueeze(0) != ids.unsqueeze(1)) & ~eye              # different instance, off-diag
    if not diff_id.any():
        return descriptors.new_zeros(())
    penalty = F.relu(cos - margin) * diff_id                             # only when too similar
    return penalty.sum() / diff_id.sum().clamp(min=1)


def _momentum_contrastive(
    descs:       torch.Tensor,   # [P, D]  L2-normalized matched descriptors
    frame_ids:   torch.Tensor,   # [P]     frame index t (0..T-1) of each descriptor
    inst_ids:    torch.Tensor,   # [P]     scene-consistent instance id
    momentum:    float = 0.75,
    sim_guided:  bool  = True,
    temperature: float = 0.07,
):
    """CTVIS-style CAUSAL momentum contrastive (ported from CTVIS ct_cl_plugin).

    Per instance, accumulate a (similarity-guided) EMA reference across frames in
    TEMPORAL order.  At frame t, the matched descriptor (anchor) is contrasted
    against the memory bank = {all instances' EMA built from frames < t}; its own
    id is the positive, others are negatives.  This mirrors SLAM inference (match
    a current observation against the past-accumulated per-instance reference) →
    train/infer aligned.  Causal: an anchor only ever sees the PAST (frames < t).

    EMA update (CTVIS):
      - first appearance: ema = desc
      - sim_guided=True : beta = clamp(cos(ema, desc), 0, 1) (DETACHED gate) →
                          ema = (1-beta)*ema + beta*desc  (outliers down-weighted)
      - sim_guided=False: ema = (1-m)*ema + m*desc

    Loss = QDTrack-style contrastive  logsumexp(neg_sim - pos_sim)
         + aux cosine calibration     |cos - label|^2   (pos→1, neg→0; enables
           stable similarity thresholding at inference, like CTVIS bi-softmax).

    Returns the contrastive loss (scalar mean over valid has-past anchors).
    The cosine-calibration aux is now a SEPARATE term (`_desc_aux_calibration`)
    so it can be applied to either contrastive mode independently of the EMA.
    """
    z = descs.new_zeros(())
    P = descs.shape[0]
    if P < 2:
        return z

    ema: dict = {}                                   # inst_id -> current EMA vec (grad-carrying)
    contras_terms = []

    frames = torch.unique(frame_ids)
    frames = frames[torch.argsort(frames)]
    for t in frames.tolist():
        idx_t = (frame_ids == t).nonzero(as_tuple=True)[0].tolist()
        # snapshot the memory bank = per-instance EMA from frames < t.  The bank
        # KEEPS gradient (built from grad-carrying descs, like CTVIS) → the loss
        # trains both the anchor AND the accumulated reference.  Only the beta
        # gate (below) is detached.
        bank_ids = list(ema.keys())
        if bank_ids:
            bank_vecs = torch.stack([ema[i] for i in bank_ids], dim=0)        # [Nb, D]
            bank_ids_t = torch.tensor(bank_ids, device=descs.device)
        # (1) contrast each anchor at frame t against the bank (frames < t)
        for j in idx_t:
            iid = int(inst_ids[j].item())
            if not bank_ids:
                continue
            label = (bank_ids_t == iid).float()                              # [Nb]
            if label.sum() == 0:                                            # no past positive
                continue
            d = descs[j]                                                    # [D]
            cos  = bank_vecs @ d                                            # [Nb] cosine (unit vecs)
            sims = cos / temperature
            pos = sims[label > 0]
            neg = sims[label == 0]
            if neg.numel() > 0:
                diff = neg.unsqueeze(0) - pos.unsqueeze(1)                   # [npos, nneg]
                x = torch.cat([diff.reshape(-1), diff.new_zeros(1)])         # +0 pad (QDTrack)
                contras_terms.append(torch.logsumexp(x, dim=0))
        # (2) AFTER contrasting, update EMA with frame-t observations (causal)
        for j in idx_t:
            iid = int(inst_ids[j].item())
            d = descs[j]
            if iid not in ema:
                ema[iid] = d
            elif sim_guided:
                beta = (ema[iid].detach() * d.detach()).sum().clamp(min=0.0, max=1.0)
                ema[iid] = (1.0 - beta) * ema[iid] + beta * d
            else:
                ema[iid] = (1.0 - momentum) * ema[iid] + momentum * d

    if not contras_terms:
        return z
    return torch.stack(contras_terms).mean()


def _desc_aux_calibration(descriptors: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Cosine ABSOLUTE-calibration aux (decoupled from the contrastive mode):
        mean over off-diagonal pairs of (cos(i,j) - 1[id_i==id_j])^2
    → pushes same-instance cosine→1 and different-instance→0.  Complements either
    ntxent (relative ranking) or momentum (causal) — gives the absolute scale that
    makes a fixed match-threshold meaningful.  Works on the same matched pool, no EMA.
    """
    P = descriptors.shape[0]
    if P < 2:
        return descriptors.new_zeros(())
    sim = descriptors @ descriptors.t()                                  # [P,P] cosine (unit vecs)
    label = (ids.unsqueeze(0) == ids.unsqueeze(1)).to(sim.dtype)         # [P,P]
    off = ~torch.eye(P, dtype=torch.bool, device=descriptors.device)
    return ((sim[off] - label[off]) ** 2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Main criterion
# ─────────────────────────────────────────────────────────────────────────────

def _pair_auroc(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """AUROC via the Mann–Whitney U statistic (rank-based, no sklearn).

    AUROC = P(score_pos > score_neg).  Returns 0.5 if a class is empty.
    """
    pos = logits[y > 0.5]
    neg = logits[y <= 0.5]
    np_, nn_ = pos.numel(), neg.numel()
    if np_ == 0 or nn_ == 0:
        return logits.new_tensor(0.5)
    order = torch.argsort(logits)                       # ascending
    ranks = torch.empty_like(order, dtype=logits.dtype)
    ranks[order] = torch.arange(1, logits.numel() + 1, device=logits.device,
                                dtype=logits.dtype)
    sum_pos_ranks = ranks[y > 0.5].sum()
    auc = (sum_pos_ranks - np_ * (np_ + 1) / 2.0) / (np_ * nn_)
    return auc


class QueryInstanceLoss(nn.Module):
    """Full loss for the query-based instance head.

    Args:
        w_mask_bce, w_mask_dice, w_obj, w_desc:  loss weights.
        descriptor_warmup_epochs:  before this epoch, w_desc is forced to 0.
                                    Set 0 to enable descriptor from the start.
        matcher_cost_bce / dice / obj:  weights in the matcher cost.
        matcher_num_points:  K sampled pixels per frame for matching.
        descriptor_temperature:  NT-Xent τ.
        min_gt_pixels:  drop GT instances smaller than this.
        deep_supervision:  if True and predictions carry `aux_outputs` (one per
                            intermediate decoder layer), compute the same loss
                            on each aux output and add to the final-layer loss
                            (Mask2Former-style deep supervision).  Each aux
                            output gets its own Hungarian matching.
        aux_loss_weight:    multiplier on each aux loss term (1.0 = same weight
                            as final-layer loss).
        aux_loss_layer_indices: list of decoder-layer indices (0-based, with
                            the init prediction ALWAYS dropped) to supervise.
                            None (default) → supervise every layer (canonical
                            M2F).  Example: with n_layers=9, num_feature_levels=3
                            (Mask2Former), [2, 5] supervises only the cycle
                            ends (1/8 fine scale, matching the final layer's
                            scale).  Final-layer loss is unaffected.
    """

    def __init__(
        self,
        w_mask_bce:                 float = 2.0,
        w_mask_dice:                float = 5.0,
        w_mask_iou:                 float = 0.0,     # soft mask-IoU (ESAM L_iou); 0 = off
        w_obj:                      float = 1.0,
        w_desc:                     float = 0.5,
        # Path-B-only descriptor aux contrastive (dual-path desc head).  When
        # >0 and predictions carry 'descriptor_pathB', supervise the memory-only
        # descriptor with the same NT-Xent (same matched pairs) → forces the
        # memory path to carry identity on its own.  0 = off (back-compat).
        # Gated by the same descriptor warmup as the main descriptor loss.
        w_desc_pathB:               float = 0.0,
        descriptor_warmup_epochs:   int   = 4,
        matcher_cost_bce:           float = 2.0,
        matcher_cost_dice:          float = 5.0,
        matcher_cost_obj:           float = 1.0,
        matcher_num_points:         int   = 12544,
        # Matcher type — 'hungarian' (default, free 1-to-1) or 'sparse' (ESAM
        # constrained per-GT top-k; requires FastSAM-region query_masks at
        # forward time via preds['fastsam_region_id']).  Default keeps every
        # existing run identical (backward compatible).
        matcher_type:               str   = 'hungarian',
        sparse_topk:                int   = 1,
        descriptor_temperature:     float = 0.07,
        # Descriptor contrastive scheme:
        #   'ntxent'   (default) — flat all-frame supervised NT-Xent (legacy).
        #   'momentum' — CTVIS-style CAUSAL: anchor at frame t vs the per-instance
        #                similarity-guided EMA accumulated from frames < t (memory
        #                bank), own-id positive / others negative.  Train/infer
        #                aligned (mirrors the SLAM running reference).
        desc_contrastive_mode:      str   = 'ntxent',
        desc_push_margin:           float = 0.1,    # 'within_frame_push': hinge margin
        # Mask2Former importance point sampling for the mask loss (boundary-focused
        # → sharper, less-fragmented masks).  Default off = full-mask BCE/Dice.
        mask_point_sample:          bool  = False,
        mask_num_points:            int   = 12544,
        mask_oversample_ratio:      float = 3.0,
        mask_importance_ratio:      float = 0.75,
        desc_momentum:              float = 0.75,
        desc_sim_guided:            bool  = True,
        # Cosine-calibration aux (|cos-label|^2, push same→1/diff→0).  Independent
        # of the contrastive mode — works with BOTH ntxent and momentum (no EMA).
        # Default 0 = OFF (legacy).  Set >0 to add the absolute-separation term.
        desc_aux_weight:            float = 0.0,
        min_gt_pixels:              int   = 10,
        deep_supervision:           bool  = True,
        aux_loss_weight:            float = 1.0,
        aux_loss_layer_indices:     Optional[List[int]] = None,
        # Optional 3D box DIoU loss.  Default off (w_box=0) — only enabled
        # by the train script when `--query_use_box_head` is set and
        # `--w_box > 0` (caller is also responsible for passing gt_depth /
        # gt_intrinsic / gt_c2w into forward()).
        w_box:                      float = 0.0,
        w_fgbg:                     float = 0.0,         # thing/stuff BCE weight (0 = off)
        box_iqr_mult:               float = 1.5,         # GT outlier filter strength
        box_min_pix:                int   = 10,          # min visible pixels per (inst, frame)
        # Objectness loss type — 'bce' (default) or 'focal'.  Focal handles
        # the matched-vs-unmatched query imbalance (typically 15:85) better
        # at train time, but it down-weights confident positives and so
        # compresses the predicted objectness range (~0.4–0.6 instead of
        # ~0.75–0.98).  That compressed range is a poor confidence signal for
        # SLAM instance AP ranking (mean-aggregated over many observations it
        # collapses to ~0.5, interleaving TP/FP), so BCE is the default.
        obj_loss_type:              str   = 'bce',
        obj_focal_alpha:            float = 0.25,
        obj_focal_gamma:            float = 2.0,
        # Objectness TARGET design:
        #   'hard' (default) — matched query → 1.0, unmatched → 0.0 (binary
        #                      matched/unmatched; carries NO mask-quality info).
        #   'iou'            — quality-aware: matched query with mask IoU >
        #                      `obj_iou_floor` → target = that IoU (so the
        #                      objectness/confidence reflects mask quality and
        #                      ranks good vs mediocre masks); matched query with
        #                      IoU <= floor → 0 (treated as a failed detection);
        #                      unmatched → 0.  Single head does background
        #                      rejection (the 0s) AND quality ranking (the IoU
        #                      targets) — suited to open-vocab (no class head).
        #                      IoU is computed no-grad from the predicted mask.
        obj_target_mode:            str   = 'hard',
        obj_iou_floor:              float = 0.5,
        # (obj_target='iou' only) how to treat MATCHED queries whose mask IoU
        # is <= obj_iou_floor:
        #   'zero'   (default) — target 0 (actively push confidence down; safe
        #                        for a single objectness head — it doubles as
        #                        background rejection, since there is no class
        #                        head to suppress them otherwise).
        #   'ignore'           — no objectness supervision at all (masked out of
        #                        the loss), ESAM-style.  Only UNMATCHED queries
        #                        are pushed to 0.  Smoother near the boundary &
        #                        no conflict with the still-improving mask, BUT
        #                        these queries' objectness is then unconstrained
        #                        → risk of high-confidence false positives in an
        #                        open-vocab (no class head) setup.
        obj_iou_below:              str   = 'zero',
        # ESAM-style ∅ (no-object) down-weight: weight on UNMATCHED queries in the
        # objectness loss (matched stay 1.0).  ESAM's `non_object_weight`/eos_coef
        # ≈ 0.1 stops the abundant unmatched negatives (~85%) from dominating the
        # loss.  Default 1.0 = legacy (all queries equal weight).
        obj_nonobject_weight:       float = 1.0,
        # Use is_bg to sharpen the ∅ signal: a query matched to a BG instance
        # (wall/floor/ceiling) is set objectness target 0 (no-object) at FULL
        # weight — a confident background-rejection signal (vs the ambiguous
        # down-weighted unmatched negatives).  Needs bg_ids (derived from the
        # dataset is_bg mask).  Default False = legacy.
        obj_bg_negative:            bool  = False,
        # A' semantic-conf head: BCE(sigmoid(pred['sem_conf']), pred['sem_conf_target'])
        # over ALL queries.  w=0 → off.  Warmup delays it until the student semantic
        # feature (the head's input) is meaningful.
        w_sem_conf:                 float = 0.0,
        sem_conf_warmup_epochs:     int   = 1,
        # Box mode — 'aabb' (legacy, 6-d AABB DIoU loss, per-(b,t,iid) GT)
        # or 'obb' (12-d OBB world-frame, per-(b,iid) GT).  Must match the
        # head's box_mode at construction.
        box_mode:                   str   = 'aabb',
        # OBB sub-loss weights (used only when box_mode='obb' AND
        # box_loss_form='frobenius').
        w_box_center:               float = 1.0,
        w_box_size:                 float = 1.0,
        w_box_rot:                  float = 1.0,
        w_box_iou:                  float = 1.0,    # world_amodal*: DIoU term weight
        # OBB extras (only used when box_mode='obb'):
        #   box_warmup_epochs: number of epochs to skip box loss entirely
        #                      (mask/obj/desc/sem still trained → soft mask
        #                      becomes meaningful before box head sees signal)
        #   box_anchor_mode:   'abs'    – head outputs absolute center (legacy)
        #                      'anchor' – head outputs delta-from-mask-centroid
        #   box_loss_form:     'frobenius' – legacy (center+size+rot, ambiguous)
        #                      'corner'    – V_4 sym-aware corner SmoothL1
        box_warmup_epochs:          int   = 0,
        box_anchor_mode:            str   = 'abs',
        box_loss_form:              str   = 'frobenius',
        # MUST match the model's gt_anchor_mode so amodal box GT is built in the
        # same frame the head predicts in ('se3' tilted-cam | 'translate' gravity).
        gt_anchor_mode:             str   = 'se3',
        # world_amodal: extra L1 term on the 6-d box (center+size, NORMALIZE
        # scale) ADDED to DIoU — stabilises absolute regression (pure IoU-family
        # is slow / no-gradient when pred & GT are disjoint early in training).
        w_box_l1:                   float = 1.0,
        # Pre-generated FULL-object (amodal) world-box GT, for box_mode=
        # 'world_amodal'.  Path / comma-list / list of .pt files produced by
        # gen_gt_world_boxes.py.  None → fall back to the in-chunk world_aabb GT.
        box_gt_paths=None,
        # Drop GT boxes whose METRIC max dimension exceeds this (walls/floors/
        # ceilings — unlearnable amodal targets from a partial chunk view).
        # 0 = no filter.
        box_gt_max_dim_m:           float = 3.0,
        # world_amodal_grow: snap the growing box to the full amodal box once it
        # covers > box_snap_cover of the full volume (None = never snap, pure
        # accumulated-visible target).
        box_snap_cover:             Optional[float] = None,
        # ── Learned pairwise merge head (v1) ──────────────────────────────
        # When the model carries a `merge_head` (passed via preds['_merge_head'])
        # and w_merge>0, train a same-instance classifier on DETACHED features
        # (desc / sem / vox-IoU / center-dist) using leave-one-out positives +
        # hard negatives.  Gradients never reach the trunk (inputs detached).
        w_merge:                    float = 0.0,
        merge_warmup_epochs:        int   = 4,    # like descriptor warmup
        merge_voxel_size:           float = 0.05, # == SLAM inst_voxel_size
        merge_max_pos:              int   = 64,   # cap positive pairs / step
        merge_neg_per_pos:          int   = 4,    # negatives sampled per positive
        merge_hard_frac:            float = 0.75, # fraction of negs that are hard
    ):
        super().__init__()
        self.w_mask_bce   = w_mask_bce
        self.w_mask_dice  = w_mask_dice
        self.w_mask_iou   = float(w_mask_iou)
        self.w_obj        = w_obj
        self.w_desc       = w_desc
        self.w_desc_pathB = float(w_desc_pathB)
        self.descriptor_warmup_epochs = descriptor_warmup_epochs
        self.descriptor_temperature   = descriptor_temperature
        if desc_contrastive_mode not in ('ntxent', 'momentum', 'within_frame_push'):
            raise ValueError(f"desc_contrastive_mode must be 'ntxent' | 'momentum' | "
                             f"'within_frame_push', got {desc_contrastive_mode!r}")
        self.desc_contrastive_mode = desc_contrastive_mode
        self.desc_push_margin = float(desc_push_margin)
        self.mask_point_sample = bool(mask_point_sample)
        self.mask_num_points = int(mask_num_points)
        self.mask_oversample_ratio = float(mask_oversample_ratio)
        self.mask_importance_ratio = float(mask_importance_ratio)
        self.desc_momentum         = float(desc_momentum)
        self.desc_sim_guided       = bool(desc_sim_guided)
        self.desc_aux_weight       = float(desc_aux_weight)
        self.min_gt_pixels            = min_gt_pixels
        self.deep_supervision         = deep_supervision
        self.aux_loss_weight          = aux_loss_weight
        if aux_loss_layer_indices is not None:
            for k in aux_loss_layer_indices:
                if not isinstance(k, int) or k < 0:
                    raise ValueError(
                        f"aux_loss_layer_indices must be a list of non-negative ints; "
                        f"got element {k!r}"
                    )
        self.aux_loss_layer_indices = (list(aux_loss_layer_indices)
                                         if aux_loss_layer_indices is not None else None)
        self.w_box        = w_box
        self.w_fgbg       = float(w_fgbg)
        self.box_iqr_mult = float(box_iqr_mult)
        self.box_min_pix  = int(box_min_pix)
        if obj_loss_type not in ('focal', 'bce'):
            raise ValueError(f"obj_loss_type must be 'focal' or 'bce' "
                             f"(got {obj_loss_type!r})")
        self.obj_loss_type   = obj_loss_type
        self.obj_focal_alpha = float(obj_focal_alpha)
        self.obj_focal_gamma = float(obj_focal_gamma)
        if obj_target_mode not in ('hard', 'iou'):
            raise ValueError(f"obj_target_mode must be 'hard' or 'iou' "
                             f"(got {obj_target_mode!r})")
        if obj_iou_below not in ('zero', 'ignore'):
            raise ValueError(f"obj_iou_below must be 'zero' or 'ignore' "
                             f"(got {obj_iou_below!r})")
        self.obj_target_mode = obj_target_mode
        self.obj_iou_floor   = float(obj_iou_floor)
        self.obj_iou_below   = obj_iou_below
        self.obj_nonobject_weight = float(obj_nonobject_weight)
        self.obj_bg_negative = bool(obj_bg_negative)
        self.w_sem_conf = float(w_sem_conf)
        self.sem_conf_warmup_epochs = int(sem_conf_warmup_epochs)
        if box_mode not in ('aabb', 'obb', 'world_aabb', 'world_amodal', 'world_amodal_grow'):
            raise ValueError(f"box_mode must be 'aabb', 'obb', 'world_aabb', 'world_amodal', or "
                             f"'world_amodal_grow' (got {box_mode!r})")
        if box_anchor_mode not in ('abs', 'anchor'):
            raise ValueError(f"box_anchor_mode must be 'abs' or 'anchor' (got {box_anchor_mode!r})")
        if box_loss_form not in ('frobenius', 'corner'):
            raise ValueError(f"box_loss_form must be 'frobenius' or 'corner' (got {box_loss_form!r})")
        self.box_mode          = box_mode
        self.w_box_center      = float(w_box_center)
        self.w_box_size        = float(w_box_size)
        self.w_box_rot         = float(w_box_rot)
        self.w_box_iou         = float(w_box_iou)
        self.box_warmup_epochs = int(box_warmup_epochs)
        self.box_anchor_mode   = box_anchor_mode
        self.gt_anchor_mode    = gt_anchor_mode
        self.box_loss_form     = box_loss_form
        self.w_box_l1          = float(w_box_l1)
        self.box_gt_max_dim_m  = float(box_gt_max_dim_m)
        self.box_snap_cover    = box_snap_cover
        # Learned merge-head schedule / sampling config.
        self.w_merge             = float(w_merge)
        self.merge_warmup_epochs = int(merge_warmup_epochs)
        self.merge_voxel_size    = float(merge_voxel_size)
        self.merge_max_pos       = int(merge_max_pos)
        self.merge_neg_per_pos   = int(merge_neg_per_pos)
        self.merge_hard_frac     = float(merge_hard_frac)
        # Pre-generated amodal box table (loaded once → in-memory dict lookup).
        self.box_table = None
        if box_gt_paths:
            from .box_gt import load_pregen_boxes
            self.box_table = load_pregen_boxes(box_gt_paths)
            n_sc = sum(len(v) for v in self.box_table.values())
            print(f"[QueryInstanceLoss] loaded amodal box GT: "
                  f"{list(self.box_table.keys())} ({n_sc} scenes)")

        if matcher_type not in ('hungarian', 'sparse'):
            raise ValueError(f"matcher_type must be 'hungarian' or 'sparse', got '{matcher_type}'")
        self.matcher_type = matcher_type
        if matcher_type == 'sparse':
            from .query_matcher import SparseMatcherPointSampled
            self.matcher = SparseMatcherPointSampled(
                cost_bce=matcher_cost_bce, cost_dice=matcher_cost_dice,
                cost_obj=matcher_cost_obj, num_points=matcher_num_points,
                topk=sparse_topk,
            )
        else:
            self.matcher = HungarianMatcherPointSampled(
                cost_bce=matcher_cost_bce, cost_dice=matcher_cost_dice,
                cost_obj=matcher_cost_obj, num_points=matcher_num_points,
            )

        # Mutable schedule state (updated externally via set_epoch).
        self._current_epoch: int = 0

    def set_epoch(self, epoch: int):
        self._current_epoch = int(epoch)

    @property
    def descriptor_active(self) -> bool:
        return self._current_epoch >= self.descriptor_warmup_epochs

    @property
    def sem_conf_active(self) -> bool:
        return self.w_sem_conf > 0 and self._current_epoch >= self.sem_conf_warmup_epochs

    @property
    def box_warmup_active(self) -> bool:
        """True while box loss should be SUPPRESSED (epoch < warmup)."""
        return self._current_epoch < self.box_warmup_epochs

    @property
    def merge_active(self) -> bool:
        """True once the merge-head loss is enabled (epoch >= warmup)."""
        return (self.w_merge > 0.0
                and self._current_epoch >= self.merge_warmup_epochs)

    # ─────────────────────────────────────────────────────────────────────
    def _compute_one(
        self,
        mask_logits:   torch.Tensor,                # [B, T, N, H, W]
        objectness:    torch.Tensor,                # [B, T, N]
        descriptor:    torch.Tensor,                # [B, T, N, D]
        instance_mask: torch.Tensor,                # [B, T, H, W] long
        gt_masks_btM:  List[List[torch.Tensor]],     # cached, per-frame binary masks
        gt_ids_btM:    List[List[torch.Tensor]],     # cached, per-frame instance IDs
        descriptor_pathB: Optional[torch.Tensor] = None,  # [B, T, N, D] path-B-only (dual-path desc)
        box:           Optional[torch.Tensor] = None,           # [B, T, N, 6 | 12]
        gt_camera_boxes: Optional[Dict] = None,                  # AABB mode: dict[(b,t,inst_id) → [6]]
        gt_world_obbs:   Optional[Dict] = None,                  # OBB  mode: dict[(b,inst_id)   → [12]]
        gt_world_aabbs:  Optional[Dict] = None,                  # world_aabb mode: dict[(b,inst_id) → [6]]
        box_scale:     Optional[torch.Tensor] = None,            # [B]            canonical→metric (OBB)
        gt_c2w:        Optional[torch.Tensor] = None,            # [B, T, 4, 4]   camera→world (OBB)
        world_pts:     Optional[torch.Tensor] = None,            # [B, T, H, W, 3] (OBB anchor mode)
        query_masks:   Optional[List] = None,                    # B×T [M_bt, N] bool (sparse matcher)
        fgbg_logits:   Optional[torch.Tensor] = None,            # [B, T, N] thing/stuff logit
        bg_ids:        Optional[list] = None,                    # len-B list of bg inst-id iterables
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for one set of predictions (final layer or one aux).
        Matching is done independently here — so each aux output gets its own
        assignment."""
        B, T, N, H, W = mask_logits.shape
        device = mask_logits.device

        matches = self.matcher(
            mask_logits=mask_logits,
            objectness=objectness,
            gt_masks=gt_masks_btM,
            query_masks=query_masks,
        )

        pred_mask_pairs: List[torch.Tensor] = []
        gt_mask_pairs:   List[torch.Tensor] = []
        obj_target = torch.zeros(B, T, N, device=device, dtype=mask_logits.dtype)
        # Per-query objectness loss weight.  UNMATCHED (∅) queries start at
        # obj_nonobject_weight (ESAM eos_coef; 1.0 = legacy); MATCHED set to 1.0
        # below.  Set to 0 only by obj_target='iou'+obj_iou_below='ignore'.
        obj_weight = torch.full((B, T, N), self.obj_nonobject_weight,
                                device=device, dtype=mask_logits.dtype)
        desc_pool:    List[torch.Tensor] = []
        desc_ids_pool: List[torch.Tensor] = []
        desc_frame_pool: List[torch.Tensor] = []   # frame index t per matched desc (momentum mode)
        desc_pathB_pool: List[torch.Tensor] = []
        pred_box_pairs: List[torch.Tensor] = []
        gt_box_pairs:   List[torch.Tensor] = []
        fgbg_logit_pool:  List[torch.Tensor] = []   # thing/stuff: matched-query logits
        fgbg_target_pool: List[torch.Tensor] = []   # 1 = thing (fg), 0 = stuff (bg)
        fgbg_active = (fgbg_logits is not None) and self.w_fgbg > 0
        # OBB-only: per-pair (box_scale_scalar, c2w_4x4) for camera→world transform.
        pair_box_scale: List[torch.Tensor] = []
        pair_c2w:       List[torch.Tensor] = []
        # OBB anchor mode: per-pair world-frame mask centroid.
        pair_anchor:    List[torch.Tensor] = []

        # Box loss is active iff: head outputs `box`, GT dict is provided,
        # the global box weight is > 0, and (OBB / world_aabb) we're past warmup.
        if self.box_mode == 'aabb':
            box_active = (box is not None) and (gt_camera_boxes is not None) \
                          and (self.w_box > 0)
        elif self.box_mode == 'obb':
            box_active = (box is not None) and (gt_world_obbs is not None) \
                          and (self.w_box > 0) \
                          and (box_scale is not None) and (gt_c2w is not None) \
                          and (not self.box_warmup_active)
            # Anchor mode additionally needs world_pts to compute the mask
            # centroid per (b, t, n).
            if box_active and self.box_anchor_mode == 'anchor' and world_pts is None:
                box_active = False
        else:   # 'world_aabb' — head outputs NORMALIZE-scale, GT same scale
            box_active = (box is not None) and (gt_world_aabbs is not None) \
                          and (self.w_box > 0) \
                          and (box_scale is not None) \
                          and (not self.box_warmup_active)

        # Pre-compute soft-mask anchors for the WHOLE [B,T,N] grid in one
        # einsum (cheap; H·W ~ 84·84) → then per-pair index into it.
        anchor_world_all: Optional[torch.Tensor] = None
        if box_active and self.box_mode == 'obb' and self.box_anchor_mode == 'anchor':
            with torch.no_grad():
                soft_mask = torch.sigmoid(mask_logits.detach())          # [B,T,N,H,W]
            anchor_world_all = compute_anchor_world(soft_mask, world_pts)  # [B,T,N,3]


        for b in range(B):
            for t in range(T):
                pred_i, gt_j = matches[b][t]
                if pred_i.numel() == 0:
                    continue
                pm = mask_logits[b, t, pred_i]                # [M, H, W]
                gm = gt_masks_btM[b][t][gt_j]                 # [M, H, W]
                pred_mask_pairs.append(pm)
                gt_mask_pairs  .append(gm)
                obj_weight[b, t, pred_i] = 1.0        # matched → full weight (ESAM: 1.0 vs ∅'s eos_coef)
                if self.obj_target_mode == 'iou':
                    # Quality-aware target: matched query's objectness regresses
                    # toward its predicted-mask IoU.  Matched queries with IoU
                    # <= floor are either pushed to 0 ('zero') or dropped from
                    # the objectness loss ('ignore').
                    with torch.no_grad():
                        pm_b = (pm.sigmoid() >= 0.5).flatten(1).float()   # [M, H*W]
                        gm_b = (gm > 0.5).flatten(1).float()
                        inter = (pm_b * gm_b).sum(1)
                        union = pm_b.sum(1) + gm_b.sum(1) - inter
                        iou   = inter / (union + 1e-6)                    # [M]
                        above = iou > self.obj_iou_floor                  # [M]
                    obj_target[b, t, pred_i] = torch.where(
                        above, iou, torch.zeros_like(iou)).to(obj_target.dtype)
                    if self.obj_iou_below == 'ignore':
                        # Drop the below-floor matched queries from the obj loss.
                        below_idx = pred_i[~above]
                        obj_weight[b, t, below_idx] = 0.0
                else:
                    obj_target[b, t, pred_i] = 1.0
                if self.obj_bg_negative and bg_ids is not None and b < len(bg_ids):
                    # query matched to a BG instance → ∅ (no-object), full weight
                    bg_b = set(int(x) for x in bg_ids[b])
                    is_bg_match = torch.tensor(
                        [int(i) in bg_b for i in gt_ids_btM[b][t][gt_j].tolist()],
                        device=device)
                    if bool(is_bg_match.any()):
                        bg_q = pred_i[is_bg_match]
                        obj_target[b, t, bg_q] = 0.0
                        obj_weight[b, t, bg_q] = 1.0
                desc_pool   .append(descriptor[b, t, pred_i])
                desc_ids_pool.append(gt_ids_btM[b][t][gt_j])

                if fgbg_active:
                    # thing/stuff target per matched query: 0 if its GT instance
                    # is bg (wall/floor/ceiling), else 1.
                    bg_b = set(int(x) for x in bg_ids[b]) if (bg_ids is not None and b < len(bg_ids)) else set()
                    tgt = torch.tensor(
                        [0.0 if int(i) in bg_b else 1.0 for i in gt_ids_btM[b][t][gt_j].tolist()],
                        device=device, dtype=mask_logits.dtype)
                    fgbg_logit_pool.append(fgbg_logits[b, t, pred_i])
                    fgbg_target_pool.append(tgt)
                desc_frame_pool.append(torch.full((pred_i.numel(),), t,
                                                  dtype=torch.long, device=descriptor.device))
                if descriptor_pathB is not None:
                    desc_pathB_pool.append(descriptor_pathB[b, t, pred_i])

                if box_active:
                    # AABB:       GT keyed by (b, t, inst_id) — per-frame target.
                    # OBB:        GT keyed by (b, inst_id)    — same target across frames.
                    #             Pred is camera frame, transformed via box_scale × c2w.
                    # world_aabb: GT keyed by (b, inst_id)    — same target across frames.
                    #             Pred is ALREADY in world frame (head outputs world).
                    inst_ids = gt_ids_btM[b][t][gt_j].tolist()
                    # world_amodal pair gate: pred-mask ∩ GT-mask pixel count.
                    # The head's box anchor is the PRED-mask-pooled centroid; a
                    # matched query whose mask doesn't actually cover this GT
                    # (Hungarian must assign every visible GT, even slivers)
                    # yields an off-object anchor 3–4 normalized units away →
                    # DIoU saturates ~2 and poisons the size gradient.  Require
                    # the anchor to be grounded on the object before the pair
                    # contributes box loss; pairs re-enter as masks improve.
                    pair_inter = None
                    if self.box_mode in ('world_amodal', 'world_amodal_grow'):
                        with torch.no_grad():
                            _hard = pm.detach().sigmoid() > 0.5            # [M, H, W]
                            _gtb  = gm > 0.5
                            pair_inter = (_hard & _gtb).sum(dim=(-1, -2))   # [M]
                    for k, iid in enumerate(inst_ids):
                        if self.box_mode == 'aabb':
                            gt_box = gt_camera_boxes.get((b, t, int(iid)))
                        elif self.box_mode == 'obb':
                            gt_box = gt_world_obbs.get((b, int(iid)))
                        elif self.box_mode == 'world_amodal_grow':
                            # per-frame causal growing box keyed (b, t, iid)
                            gt_box = gt_world_aabbs.get((b, t, int(iid)))
                        else:   # 'world_aabb' / 'world_amodal'
                            gt_box = gt_world_aabbs.get((b, int(iid)))
                        if gt_box is None:
                            continue
                        if (pair_inter is not None
                                and float(pair_inter[k]) < self.box_min_pix):
                            continue    # anchor not grounded on this object → skip pair
                        q_idx = int(pred_i[k].item())
                        pred_box_pairs.append(box[b, t, q_idx])
                        gt_box_pairs.append(gt_box.to(box.dtype))
                        if self.box_mode == 'obb':
                            pair_box_scale.append(box_scale[b])
                            pair_c2w      .append(gt_c2w[b, t])
                            if anchor_world_all is not None:
                                pair_anchor.append(anchor_world_all[b, t, q_idx])

        if pred_mask_pairs:
            pred_cat = torch.cat(pred_mask_pairs, dim=0)
            gt_cat   = torch.cat(gt_mask_pairs,   dim=0).to(pred_cat.dtype)
            if self.mask_point_sample:
                # Mask2Former boundary-focused loss on importance-sampled points.
                L_bce, L_dice, L_mask_iou = _mask_losses_pointsampled(
                    pred_cat, gt_cat, self.mask_num_points,
                    self.mask_oversample_ratio, self.mask_importance_ratio,
                    want_iou=self.w_mask_iou > 0)
            else:
                L_bce  = _mask_bce_loss(pred_cat, gt_cat)
                L_dice = _mask_dice_loss(pred_cat, gt_cat)
                L_mask_iou = _mask_iou_loss(pred_cat, gt_cat) if self.w_mask_iou > 0 \
                             else mask_logits.new_zeros(())
        else:
            L_bce  = mask_logits.new_zeros(())
            L_dice = mask_logits.new_zeros(())
            L_mask_iou = mask_logits.new_zeros(())

        # `obj_weight` is all-ones except when obj_iou_below='ignore' masks out
        # matched low-IoU queries.  Use weighted mean when anything is masked.
        masked_obj = bool((obj_weight != 1.0).any())   # weighted mean if ignore-masked OR ∅ down-weighted
        if self.obj_loss_type == 'focal':
            if masked_obj:
                per = _sigmoid_focal_loss(
                    objectness, obj_target,
                    alpha=self.obj_focal_alpha, gamma=self.obj_focal_gamma,
                    reduction='none',
                )
                L_obj = (per * obj_weight).sum() / obj_weight.sum().clamp(min=1.0)
            else:
                L_obj = _sigmoid_focal_loss(
                    objectness, obj_target,
                    alpha=self.obj_focal_alpha, gamma=self.obj_focal_gamma,
                )
        else:
            if masked_obj:
                per = F.binary_cross_entropy_with_logits(
                    objectness, obj_target, reduction='none',
                )
                L_obj = (per * obj_weight).sum() / obj_weight.sum().clamp(min=1.0)
            else:
                L_obj = F.binary_cross_entropy_with_logits(
                    objectness, obj_target, reduction='mean',
                )

        L_desc_aux = mask_logits.new_zeros(())
        if desc_pool and self.descriptor_active:
            desc_cat = torch.cat(desc_pool,    dim=0)
            ids_cat  = torch.cat(desc_ids_pool, dim=0)
            if self.desc_contrastive_mode == 'momentum':
                frame_cat = torch.cat(desc_frame_pool, dim=0)
                L_desc = _momentum_contrastive(
                    desc_cat, frame_cat, ids_cat,
                    momentum=self.desc_momentum, sim_guided=self.desc_sim_guided,
                    temperature=self.descriptor_temperature,
                )
            elif self.desc_contrastive_mode == 'within_frame_push':
                # single-frame: no same-id positives → push different instances apart
                L_desc = _desc_push_only(desc_cat, ids_cat, margin=self.desc_push_margin)
            else:  # 'ntxent' (legacy)
                L_desc = _nt_xent_supervised(
                    desc_cat, ids_cat, temperature=self.descriptor_temperature,
                )
            # Cosine-calibration aux — independent of the contrastive mode; works
            # with ntxent OR momentum (no EMA needed).  Enabled by desc_aux_weight>0.
            if self.desc_aux_weight > 0:
                L_desc_aux = _desc_aux_calibration(desc_cat, ids_cat)
        else:
            L_desc = mask_logits.new_zeros(())
            ids_cat = None

        # Path-B-only descriptor aux contrastive (same matched pairs / ids).
        if (descriptor_pathB is not None and desc_pathB_pool
                and self.descriptor_active and ids_cat is not None):
            descB_cat = torch.cat(desc_pathB_pool, dim=0)
            L_desc_pathB = _nt_xent_supervised(
                descB_cat, ids_cat, temperature=self.descriptor_temperature,
            )
        else:
            L_desc_pathB = mask_logits.new_zeros(())

        # 3D box loss — branch on box_mode:
        #   'aabb'       camera-frame DIoU on 6-d
        #   'obb'        world-frame, transform + corner/Frobenius on 12-d
        #   'world_aabb' world-frame DIoU on 6-d (head outputs world directly)
        L_box_center = mask_logits.new_zeros(())
        L_box_size   = mask_logits.new_zeros(())
        L_box_rot    = mask_logits.new_zeros(())
        L_box_iou    = mask_logits.new_zeros(())
        if box_active and pred_box_pairs:
            pb = torch.stack(pred_box_pairs, dim=0).float()           # [P, 6 or 12]
            gb = torch.stack(gt_box_pairs,   dim=0).to(pb.dtype)
            if self.box_mode == 'aabb':
                L_box = diou_3d_loss(pb, gb)
            elif self.box_mode in ('world_aabb', 'world_amodal', 'world_amodal_grow'):
                # Pred and GT both in WORLD (frame-0) NORMALIZE frame, both [P, 6]
                # (cx,cy,cz,sx,sy,sz) FULL-size.  Three SEPARATELY-weighted +
                # SEPARATELY-LOGGED terms: center-L1 (mean over xyz) + size-L1
                # (mean over whd) + DIoU (stabilises overlap; L1 gives gradient on
                # disjoint boxes).  NB each L1 is a per-component mean → this is
                # NOT identical to the old DIoU + w_box_l1·L1(all 6); tune
                # w_box_center / w_box_size / w_box_iou.
                L_box_iou    = diou_3d_loss(pb, gb)
                L_box_center = F.l1_loss(pb[..., :3], gb[..., :3])
                L_box_size   = F.l1_loss(pb[..., 3:6], gb[..., 3:6])
                L_box = (self.w_box_iou    * L_box_iou
                         + self.w_box_center * L_box_center
                         + self.w_box_size   * L_box_size)
            else:  # 'obb'
                # Pred is in CAMERA frame (canonical scale).  Transform to
                # world (metric) using per-pair box_scale × c2w, then compare
                # against the world-frame GT OBB.
                ps_t  = torch.stack(pair_box_scale, dim=0).to(pb.dtype)   # [P]
                c2w_t = torch.stack(pair_c2w,       dim=0).to(pb.dtype)   # [P, 4, 4]
                anchor_t = (torch.stack(pair_anchor, dim=0).to(pb.dtype)
                              if pair_anchor else None)                   # [P, 3] or None
                pb_world = transform_obb_cam_to_world(
                    pb, ps_t, c2w_t, anchor=anchor_t,
                )                                                          # [P, 12]
                if self.box_loss_form == 'corner':
                    L_box = obb_corner_match_loss(pb_world, gb)
                else:
                    parts = obb_world_loss(
                        pb_world, gb,
                        w_center=self.w_box_center,
                        w_size=self.w_box_size,
                        w_rot=self.w_box_rot,
                    )
                    L_box        = parts['L_box']
                    L_box_center = parts['L_center']
                    L_box_size   = parts['L_size']
                    L_box_rot    = parts['L_rot']
        else:
            L_box = mask_logits.new_zeros(())

        # ── Thing/stuff (foreground/background) BCE on matched queries ──
        L_fgbg = mask_logits.new_zeros(())
        if fgbg_active and fgbg_logit_pool:
            fl = torch.cat(fgbg_logit_pool, dim=0).float()       # [P]
            ft = torch.cat(fgbg_target_pool, dim=0).to(fl.dtype)
            L_fgbg = F.binary_cross_entropy_with_logits(fl, ft)

        return {
            'L_bce':         L_bce,
            'L_dice':        L_dice,
            'L_mask_iou':    L_mask_iou,
            'L_obj':         L_obj,
            'L_desc':        L_desc,
            'L_desc_pathB':  L_desc_pathB,
            'L_desc_aux':    L_desc_aux,
            'L_box':         L_box,
            'L_box_center':  L_box_center,
            'L_box_size':    L_box_size,
            'L_box_rot':     L_box_rot,
            'L_box_iou':     L_box_iou,
            'L_fgbg':        L_fgbg,
            'n_matched':     obj_target.sum(),
        }

    def _weighted_objective(self, parts: Dict[str, torch.Tensor]) -> torch.Tensor:
        objective = (
            self.w_mask_bce  * parts['L_bce']
          + self.w_mask_dice * parts['L_dice']
          + self.w_mask_iou  * parts['L_mask_iou']
          + self.w_obj       * parts['L_obj']
        )
        if self.descriptor_active:
            objective = objective + self.w_desc * parts['L_desc']
            if self.w_desc_pathB > 0 and 'L_desc_pathB' in parts:
                objective = objective + self.w_desc_pathB * parts['L_desc_pathB']
            # CTVIS aux cosine-calibration (momentum mode only; zero otherwise)
            if self.desc_aux_weight > 0 and 'L_desc_aux' in parts:
                objective = objective + self.desc_aux_weight * parts['L_desc_aux']
        if self.w_box > 0:
            objective = objective + self.w_box * parts['L_box']
        if self.w_fgbg > 0 and 'L_fgbg' in parts:
            objective = objective + self.w_fgbg * parts['L_fgbg']
        return objective

    # ─────────────────────────────────────────────────────────────────────
    # Learned merge head (v1): pred-features + GT-id labels, leave-one-out
    # positives + hard negatives, BCE.  Inputs DETACHED → no trunk grad.
    # ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _collect_merge_obs(self, mask_logits, objectness, descriptor, sem_feat, world_pts,
                           gt_depth, gt_masks_btM, gt_ids_btM):
        """Per-batch list of observation dicts from matched (pred query ↔ GT id).

        Returns obs_by_b: list (len B) of list of dict(id, desc, sem, vox, ctr).
        Everything detached / cpu-light.  Features are POOLED from PREDICTED
        masks (matches SLAM distribution); GT id is only the grouping label.
        """
        from .merge_head import voxel_codes
        B, T, N, Hm, Wm = mask_logits.shape
        matches = self.matcher(mask_logits=mask_logits, objectness=objectness,
                               gt_masks=gt_masks_btM)
        # Independent re-match (same matcher/cost as the loss) → identical pred↔GT
        # assignment; we only need the pairing to group observations by GT id.
        obs_by_b: List[List[dict]] = [[] for _ in range(B)]
        Hd, Wd = gt_depth.shape[-2], gt_depth.shape[-1]
        for b in range(B):
            for t in range(T):
                pred_i, gt_j = matches[b][t]
                if pred_i.numel() == 0:
                    continue
                ids_t = gt_ids_btM[b][t][gt_j]                       # [M]
                # masks → depth res (pool geo/sem where world_pts is exact)
                ml = mask_logits[b, t, pred_i].unsqueeze(1)          # [M,1,Hm,Wm]
                if (Hm, Wm) != (Hd, Wd):
                    ml = F.interpolate(ml.float(), size=(Hd, Wd),
                                       mode='bilinear', align_corners=False)
                hard = (ml.squeeze(1).sigmoid() > 0.5)               # [M,Hd,Wd]
                wp_t  = world_pts[b, t]                               # [Hd,Wd,3]
                valid = (gt_depth[b, t] > 1e-3)                       # [Hd,Wd]
                sem_t = sem_feat[b, t]                                # [Ds,Hs,Ws]
                if sem_t.shape[-2:] != (Hd, Wd):
                    sem_t = F.interpolate(sem_t.unsqueeze(0).float(), size=(Hd, Wd),
                                          mode='bilinear', align_corners=False)[0]
                sem_t = sem_t.permute(1, 2, 0)                       # [Hd,Wd,Ds]
                for k in range(pred_i.numel()):
                    m = hard[k] & valid                              # [Hd,Wd]
                    if int(m.sum()) < 10:
                        continue
                    pts = wp_t[m]                                    # [P,3]
                    sem_q = F.normalize(sem_t[m].mean(0), dim=-1)    # [Ds]
                    obs_by_b[b].append(dict(
                        id   = int(ids_t[k].item()),
                        desc = descriptor[b, t, pred_i[k]].detach(),  # [Dd] (already L2)
                        sem  = sem_q.detach(),
                        vox  = voxel_codes(pts, self.merge_voxel_size),
                        ctr  = pts.median(0).values.detach(),         # [3]
                    ))
        return obs_by_b

    def _merge_pair_loss(self, preds, gt_depth, gt_masks_btM, gt_ids_btM,
                         world_pts, merge_head, device):
        """BCE over leave-one-out positive + hard-negative pairs."""
        from .merge_head import build_pair_features, vox_iou_min
        import random
        sem_feat = preds.get('semantic_feat', None)
        if sem_feat is None or merge_head is None or world_pts is None:
            return None, None
        obs_by_b = self._collect_merge_obs(
            preds['mask_logits'], preds['objectness'], preds['descriptor'], sem_feat,
            world_pts, gt_depth, gt_masks_btM, gt_ids_btM)

        # Build feature tensors for sampled pairs (proposal P vs record R).
        ad, asem, gg, labels = [], [], [], []

        def aggregate(obs_list):
            desc = F.normalize(torch.stack([o['desc'] for o in obs_list]).mean(0), dim=-1)
            sem  = F.normalize(torch.stack([o['sem']  for o in obs_list]).mean(0), dim=-1)
            vox  = torch.unique(torch.cat([o['vox'] for o in obs_list])) \
                   if any(o['vox'].numel() for o in obs_list) else obs_list[0]['vox']
            ctr  = torch.stack([o['ctr'] for o in obs_list]).mean(0)
            return dict(desc=desc, sem=sem, vox=vox, ctr=ctr)

        n_pos = 0
        for b in range(len(obs_by_b)):
            obs = obs_by_b[b]
            if len(obs) < 2:
                continue
            by_id: Dict[int, List[dict]] = {}
            for o in obs:
                by_id.setdefault(o['id'], []).append(o)
            ids = [i for i, v in by_id.items() if len(v) >= 1]
            if len(ids) < 2:
                continue
            # full per-id aggregate (record candidate pool for negatives)
            full_agg = {i: aggregate(v) for i, v in by_id.items()}
            mergeable = [i for i, v in by_id.items() if len(v) >= 2]
            random.shuffle(mergeable)
            for iid in mergeable:
                if n_pos >= self.merge_max_pos:
                    break
                group = by_id[iid]
                pj = random.randrange(len(group))
                prop = group[pj]
                rec  = aggregate([o for j, o in enumerate(group) if j != pj])
                # ── positive ──
                ad_p, as_p, g_p = build_pair_features(
                    prop['desc'], rec['desc'], prop['sem'], rec['sem'],
                    torch.tensor(vox_iou_min(prop['vox'], rec['vox']), device=device),
                    (prop['ctr'] - rec['ctr']).norm())
                ad.append(ad_p); asem.append(as_p); gg.append(g_p); labels.append(1.0)
                n_pos += 1
                # ── negatives (hard = same-ish sem & near; + random) ──
                others = [j for j in ids if j != iid]
                if not others:
                    continue
                # rank others by sem-cos to the proposal (hard = high sem sim)
                sem_sim = {j: float((prop['sem'] * full_agg[j]['sem']).sum())
                           for j in others}
                hard_sorted = sorted(others, key=lambda j: -sem_sim[j])
                n_hard = max(1, int(round(self.merge_neg_per_pos * self.merge_hard_frac)))
                neg_ids = hard_sorted[:n_hard]
                rand_pool = [j for j in others if j not in neg_ids]
                if rand_pool:
                    neg_ids = neg_ids + random.sample(
                        rand_pool, min(self.merge_neg_per_pos - len(neg_ids), len(rand_pool)))
                for nj in neg_ids:
                    rj = full_agg[nj]
                    ad_n, as_n, g_n = build_pair_features(
                        prop['desc'], rj['desc'], prop['sem'], rj['sem'],
                        torch.tensor(vox_iou_min(prop['vox'], rj['vox']), device=device),
                        (prop['ctr'] - rj['ctr']).norm())
                    ad.append(ad_n); asem.append(as_n); gg.append(g_n); labels.append(0.0)

        if n_pos == 0 or len(labels) < 2:
            # No valid pairs this step.  Return a grad-connected ZERO (touches
            # merge_head) instead of None, so the objective always carries grad
            # — in probe mode the trunk is frozen, so without this the objective
            # would have no grad_fn and backward() would crash.  Contributes 0.
            dd, ds = 2 * merge_head.desc_dim, 2 * merge_head.sem_dim
            dummy = merge_head(torch.zeros(1, dd, device=device),
                               torch.zeros(1, ds, device=device),
                               torch.zeros(1, 2, device=device)).sum() * 0.0
            return dummy, torch.full((), float('nan'), device=device)
        ad   = torch.stack(ad).to(device)
        asem = torch.stack(asem).to(device)
        gg   = torch.stack(gg).to(device)
        y    = torch.tensor(labels, device=device, dtype=ad.dtype)
        logits = merge_head(ad, asem, gg)                            # [Npair]
        # class-balanced BCE (pos rarer than neg)
        n_p = float((y > 0.5).sum()); n_n = float((y <= 0.5).sum())
        pos_w = torch.tensor(max(n_n, 1.0) / max(n_p, 1.0), device=device)
        L = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_w)
        # AUROC (Mann–Whitney) for monitoring — detached.
        with torch.no_grad():
            auroc = _pair_auroc(logits.detach(), y)
        return L, auroc

    def forward(
        self,
        preds:        Dict[str, torch.Tensor],
        instance_mask: torch.Tensor,                # [B, T, H, W] long
        gt_depth:      Optional[torch.Tensor] = None,    # [B, T, H, W]    GT depth (metric)
        gt_intrinsic:  Optional[torch.Tensor] = None,    # [B, T, 3, 3]    GT intrinsics
        gt_c2w:        Optional[torch.Tensor] = None,    # [B, T, 4, 4]    GT camera-to-world (dataset-world)
        scene_ids=None,                                  # len-B list (world_amodal box lookup)
        dataset_names=None,                              # len-B list (world_amodal box lookup)
        bg_ids=None,                                     # len-B list of bg inst-id iterables
        bg_mask=None,                                    # [B,T,H,W] bool per-pixel is_bg → derives
                                                         # bg_ids (wall/floor/ceiling) if bg_ids None
    ) -> Dict[str, torch.Tensor]:
        device = preds['mask_logits'].device

        # Derive per-(b) bg instance-id sets from the per-pixel is_bg mask
        # (dataset provides bg_mask, not an id list).  Used to skip bg from box
        # GT + as the thing/stuff target.
        if bg_ids is None and bg_mask is not None:
            bm = bg_mask.bool()
            bg_ids = [torch.unique(instance_mask[b][bm[b]]).tolist()
                      for b in range(instance_mask.shape[0])]

        # ── 1. Build per-frame GT masks + IDs (shared across final + aux) ──
        gt_masks_btM, gt_ids_btM = build_per_frame_gt_masks(
            instance_mask, min_pixels=self.min_gt_pixels,
        )

        # ── 1a. (sparse matcher) Build allowed (GT, query) masks from the
        # FastSAM region map.  Shared across final + every aux layer.  Only when
        # the sparse matcher is active AND the model emitted region ids.
        query_masks_btM = None
        if self.matcher_type == 'sparse':
            region_id = preds.get('fastsam_region_id', None)
            if region_id is None:
                raise ValueError(
                    "matcher_type='sparse' but preds has no 'fastsam_region_id' "
                    "(enable --fastsam_query on the model).")
            from .query_matcher import build_query_masks
            query_masks_btM = build_query_masks(
                region_id, gt_masks_btM, max_regions=preds['mask_logits'].shape[2],
            )

        # ── 1b. Build GT boxes (mode-dependent) ──
        # Only when caller passes the GT geometry AND box loss is enabled.
        # Computed once per chunk and reused for final + every aux layer.
        gt_camera_boxes = None
        gt_world_obbs   = None
        gt_world_aabbs  = None
        world_pts       = None      # shared between GT builder and anchor compute
        box_scale = preds.get('box_scale', None)
        have_geo = (gt_depth is not None and gt_intrinsic is not None
                    and gt_c2w is not None)
        if self.w_box > 0 and have_geo:
            if self.box_mode == 'aabb' and box_scale is not None:
                gt_camera_boxes = build_gt_camera_boxes(
                    depth         = gt_depth,
                    intrinsic     = gt_intrinsic,
                    c2w           = gt_c2w,
                    instance_mask = instance_mask,
                    box_scale     = box_scale,
                    iqr_mult      = self.box_iqr_mult,
                    min_pix       = self.box_min_pix,
                )
            elif self.box_mode == 'obb':
                # Unproject GT depth → world points (in c2w's world frame
                # = chunk-anchor frame for training data).  Reused for:
                #   - GT OBB PCA fitting (build_gt_world_obbs)
                #   - Per-(b,t,n) anchor centroid (anchor mode in _compute_one)
                from .box_gt import unproject_to_world
                world_pts = unproject_to_world(gt_depth, gt_intrinsic, gt_c2w)
                valid     = gt_depth > 1e-3
                gt_world_obbs = build_gt_world_obbs(
                    world_pts        = world_pts,
                    instance_mask    = instance_mask,
                    valid_mask       = valid,
                    iqr_mult         = self.box_iqr_mult,
                    min_pix_per_inst = self.box_min_pix,
                )
            elif self.box_mode == 'world_aabb':
                # ── Three-scale system (see BoxAABBHead docstring) ──
                #   pred_world_points  = canonical (DA3 native)
                #   box_scale          = depth median → canonical→normalize divisor
                #   normalize          = canonical / box_scale  (median ~1)
                #
                # Head outputs in NORMALIZE scale (PCA prior + delta, all
                # internally divided by box_scale).  GT must match → also
                # build in normalize scale by dividing pred world_pts by
                # box_scale BEFORE aggregating per-instance AABB.
                world_pts_canon = preds.get('world_points', None)
                if (world_pts_canon is not None
                        and box_scale is not None
                        and self.w_box > 0):
                    bs = box_scale.detach().to(world_pts_canon.dtype) \
                                          .clamp(min=1e-3)              # [B]
                    # [B, T, H, W, 3] / [B, 1, 1, 1, 1]  → normalize scale
                    world_pts_norm = world_pts_canon.detach() / bs.view(-1, 1, 1, 1, 1)
                    if 'world_points_conf' in preds and preds['world_points_conf'] is not None:
                        valid = preds['world_points_conf'].squeeze(-1) > 0
                    elif gt_depth is not None:
                        valid = gt_depth > 1e-3
                    else:
                        valid = None
                    gt_world_aabbs = build_gt_world_aabbs(
                        world_pts        = world_pts_norm,
                        instance_mask    = instance_mask,
                        valid_mask       = valid,
                        iqr_mult         = self.box_iqr_mult,
                        min_pix_per_inst = self.box_min_pix,
                    )
            elif self.box_mode == 'world_amodal':
                # Pre-generated FULL-object (amodal) world boxes, transformed to
                # the SAME frame-0-anchored NORMALIZE space the head predicts in.
                # gt_c2w is the dataset-world camera_pose (absolute); box_scale is
                # the per-chunk divisor.  Output reuses the gt_world_aabbs slot.
                if (self.box_table is not None and box_scale is not None
                        and scene_ids is not None and dataset_names is not None
                        and gt_c2w is not None):
                    gt_world_aabbs = build_amodal_gt_aabbs(
                        box_table     = self.box_table,
                        dataset_names = dataset_names,
                        scene_ids     = scene_ids,
                        instance_mask = instance_mask,
                        camera_pose   = gt_c2w,
                        box_scale     = box_scale,
                        anchor_mode   = self.gt_anchor_mode,
                        max_dim_m     = self.box_gt_max_dim_m,
                    )
            elif self.box_mode == 'world_amodal_grow':
                # CAUSAL growing visible box (DetAny4D-style), built in-loss from
                # the chunk's own world points (NORMALIZE, frame-0 anchored — same
                # space as world_aabb), keyed per-(b,t,iid).  Optionally clamped/
                # snapped to the pre-generated full amodal box (box_table) so depth
                # noise can't push the visible AABB past the true extent.
                world_pts_canon = preds.get('world_points', None)
                if (world_pts_canon is not None and box_scale is not None
                        and self.w_box > 0):
                    bs = box_scale.detach().to(world_pts_canon.dtype).clamp(min=1e-3)
                    world_pts_norm = world_pts_canon.detach() / bs.view(-1, 1, 1, 1, 1)
                    if 'world_points_conf' in preds and preds['world_points_conf'] is not None:
                        valid = preds['world_points_conf'].squeeze(-1) > 0
                    elif gt_depth is not None:
                        valid = gt_depth > 1e-3
                    else:
                        valid = None
                    full_boxes = None
                    if (self.box_table is not None and scene_ids is not None
                            and dataset_names is not None and gt_c2w is not None):
                        full_boxes = build_amodal_gt_aabbs(
                            box_table=self.box_table, dataset_names=dataset_names,
                            scene_ids=scene_ids, instance_mask=instance_mask,
                            camera_pose=gt_c2w, box_scale=box_scale,
                            anchor_mode=self.gt_anchor_mode, max_dim_m=self.box_gt_max_dim_m,
                        )
                    gt_world_aabbs = build_growing_gt_aabbs(
                        world_pts=world_pts_norm, instance_mask=instance_mask,
                        valid_mask=valid, iqr_mult=self.box_iqr_mult,
                        min_pix_per_inst=self.box_min_pix, bg_ids=bg_ids,
                        full_boxes=full_boxes, snap_cover=self.box_snap_cover,
                    )

        # ── 2. Final-layer loss ──
        final_parts = self._compute_one(
            mask_logits   = preds['mask_logits'],
            objectness    = preds['objectness'],
            descriptor    = preds['descriptor'],
            instance_mask = instance_mask,
            gt_masks_btM  = gt_masks_btM,
            gt_ids_btM    = gt_ids_btM,
            descriptor_pathB = preds.get('descriptor_pathB', None),
            box           = preds.get('box', None),
            gt_camera_boxes = gt_camera_boxes,
            gt_world_obbs   = gt_world_obbs,
            gt_world_aabbs  = gt_world_aabbs,
            box_scale       = box_scale,
            gt_c2w          = gt_c2w,
            world_pts       = world_pts,
            query_masks     = query_masks_btM,
            fgbg_logits     = preds.get('fgbg_logits', None),
            bg_ids          = bg_ids,
        )
        objective = self._weighted_objective(final_parts)

        # ── 2b. Learned merge-head loss (final layer only, detached inputs) ──
        L_merge   = torch.zeros((), device=device)
        merge_auroc = torch.full((), float('nan'), device=device)
        merge_head = preds.get('_merge_head', None)
        if self.merge_active and merge_head is not None and have_geo:
            from .box_gt import unproject_to_world
            merge_world_pts = unproject_to_world(gt_depth, gt_intrinsic, gt_c2w)  # dataset-world metric
            Lm, auroc = self._merge_pair_loss(
                preds, gt_depth, gt_masks_btM, gt_ids_btM,
                merge_world_pts, merge_head, device)
            if Lm is not None:
                L_merge = Lm
                merge_auroc = auroc
                objective = objective + self.w_merge * L_merge

        # ── 3. Aux losses (deep supervision) ──
        aux_outs = preds.get('aux_outputs', None) if self.deep_supervision else None
        if aux_outs:
            # Always drop the init prediction (index 0) — that prediction is
            # made BEFORE any decoder layer runs, from the raw learnable
            # query_feat.  It carries no decoder-layer training signal so
            # supervising it doesn't help mask refinement.
            aux_outs = aux_outs[1:]
            # Apply explicit layer index selection (None = supervise all).
            if self.aux_loss_layer_indices is not None:
                idx = [k for k in self.aux_loss_layer_indices if 0 <= k < len(aux_outs)]
                aux_outs_used = [aux_outs[k] for k in idx]
            else:
                aux_outs_used = aux_outs
            # Each aux gets its own Hungarian + loss; we sum.
            for k, aux in enumerate(aux_outs_used):
                aux_parts = self._compute_one(
                    mask_logits   = aux['mask_logits'],
                    objectness    = aux['objectness'],
                    descriptor    = aux['descriptor'],
                    instance_mask = instance_mask,
                    gt_masks_btM  = gt_masks_btM,
                    gt_ids_btM    = gt_ids_btM,
                    box           = aux.get('box', None),
                    gt_camera_boxes = gt_camera_boxes,
                    gt_world_obbs   = gt_world_obbs,
                    box_scale       = box_scale,
                    gt_c2w          = gt_c2w,
                    world_pts       = world_pts,
                    query_masks     = query_masks_btM,
                    fgbg_logits     = aux.get('fgbg_logits', None),
                    bg_ids          = bg_ids,
                )
                aux_obj = self._weighted_objective(aux_parts)
                objective = objective + self.aux_loss_weight * aux_obj
        else:
            aux_outs_used = []

        # ── A' semantic-conf head: BCE(sem_conf, dense-consistency target) ──
        L_sem_conf = torch.zeros((), device=device)
        if self.sem_conf_active and preds.get('sem_conf') is not None \
                and preds.get('sem_conf_target') is not None:
            L_sem_conf = F.binary_cross_entropy_with_logits(
                preds['sem_conf'].float(), preds['sem_conf_target'].float())
            objective = objective + self.w_sem_conf * L_sem_conf

        # Compose return dict.  Logged metrics report FINAL layer's components
        # (aux losses are aggregated into 'objective' but not separately surfaced
        # to keep the log compact).
        L_mask_total = (self.w_mask_bce  * final_parts['L_bce']
                       + self.w_mask_dice * final_parts['L_dice']
                       + self.w_mask_iou  * final_parts['L_mask_iou'])
        n_aux = len(aux_outs_used) if aux_outs else 0
        return {
            'objective':   objective,
            'L_mask':      L_mask_total.detach(),
            'L_bce':       final_parts['L_bce'].detach(),
            'L_dice':      final_parts['L_dice'].detach(),
            'L_mask_iou':  final_parts['L_mask_iou'].detach(),
            'L_obj':       final_parts['L_obj'].detach(),
            'L_desc':      final_parts['L_desc'].detach(),
            'L_desc_pathB': final_parts['L_desc_pathB'].detach(),
            'L_desc_aux':  final_parts['L_desc_aux'].detach(),
            'L_box':       final_parts['L_box'].detach(),
            'L_box_center': final_parts['L_box_center'].detach(),
            'L_box_size':  final_parts['L_box_size'].detach(),
            'L_box_iou':   final_parts['L_box_iou'].detach(),
            'L_fgbg':      final_parts['L_fgbg'].detach(),
            'L_sem_conf':  L_sem_conf.detach(),
            'desc_active': torch.tensor(float(self.descriptor_active),
                                          device=device).detach(),
            'n_matched':   final_parts['n_matched'].detach(),
            'n_aux_layers': torch.tensor(float(n_aux), device=device).detach(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Composite loss: semantic (legacy OpenSe3rLoss) + query instance
# ─────────────────────────────────────────────────────────────────────────────

class CompositeQueryLoss(nn.Module):
    """Combines semantic distillation (from OpenSe3rLoss, dense-instance terms
    auto-skipped) with the query-based instance loss.  Used when
    `--instance_head_type query`.

    Output dict carries the original keys from both sub-losses; query loss
    keys are prefixed with `q_` to avoid collision.  `objective` is the sum
    of both sub-loss objectives.
    """

    def __init__(self, sem_loss: OpenSe3rLoss, query_loss: QueryInstanceLoss):
        super().__init__()
        self.sem_loss   = sem_loss
        self.query_loss = query_loss

    def set_epoch(self, epoch: int):
        self.query_loss.set_epoch(epoch)

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Semantic part — OpenSe3rLoss auto-skips ins/consistency when
        # predictions['instance_feat'] is None (query mode).
        sem_out = self.sem_loss(predictions, targets)
        # Query part — needs predictions[{mask_logits,objectness,descriptor}]
        # and targets['instance_mask'].  When the box head is enabled also
        # forward GT geometry so the loss can build per-(inst, frame) GT
        # camera boxes.  These keys come from views_all in train_opense3r.
        q_out = self.query_loss(
            predictions, targets['instance_mask'],
            gt_depth      = targets.get('depthmap',         None),
            gt_intrinsic  = targets.get('camera_intrinsics', None),
            gt_c2w        = targets.get('camera_pose',       None),
            scene_ids     = targets.get('scene_id',     None),
            dataset_names = targets.get('dataset_name', None),
            bg_mask       = targets.get('is_bg',        None),
        )

        out = dict(sem_out)
        for k, v in q_out.items():
            if k == 'objective':
                out['q_objective'] = v
            else:
                out[f'q_{k}'] = v
        out['objective'] = sem_out['objective'] + q_out['objective']
        return out
