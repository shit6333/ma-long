"""
opense3r/instance_post.py
==========================
Reusable post-processing utilities for the query-based instance head.

These primitives are written so the **eval-time PLY export** and the
**online SLAM matching** can share the exact same algorithms — same
hyperparameters, same scoring, same merging behaviour.

Pipeline pieces:

  1. `mask_nms`              — remove duplicate proposals within a frame
  2. `resolve_per_pixel`     — argmax over kept slots with objectness gating
  3. `lift_proposal_to_3d`   — turn a 2D mask + world_points into a 3D
                                proposal (center, voxel set, point count)
  4. `proposal_pair_score`   — descriptor + voxel + center score, mirrors
                                the SLAM matching cost
  5. `union_find_merge`      — connect-components merge of (frame, slot)
                                proposals into global cluster IDs

Designed for *one chunk* of frames at a time.  All operations are
agnostic to the underlying device — set / tuple operations run on CPU
(unavoidable for Python set intersection), other ops stay on GPU.

This module deliberately does NOT depend on HDBSCAN.  The legacy
dense (FeatureDPTHead) path keeps using `vis_instance_hdbscan.py`
unchanged.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set, Iterable
import math

import numpy as np
import torch

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    linear_sum_assignment = None
    _HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mask NMS within a single frame
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_mask_iou(masks: torch.Tensor) -> torch.Tensor:
    """Vectorised pairwise IoU between N binary masks.

    Args:
        masks: [N, H, W] bool/float
    Returns:
        iou:   [N, N]  float
    """
    N = masks.shape[0]
    flat = masks.reshape(N, -1).float()
    inter = flat @ flat.t()                            # [N, N]
    sums = flat.sum(dim=1)                              # [N]
    union = sums.unsqueeze(0) + sums.unsqueeze(1) - inter
    return inter / union.clamp(min=1.0)


def mask_nms(
    masks: torch.Tensor,            # [N, H, W] bool
    scores: torch.Tensor,           # [N]  higher = better (e.g. objectness)
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Greedy mask NMS.  Returns kept indices (long) in DESCENDING score order.

    Used to drop duplicate query slots that predicted overlapping masks of
    the same object (a common artefact of unconstrained 100-query decoders).
    """
    N = masks.shape[0]
    if N == 0:
        return torch.empty(0, dtype=torch.long, device=masks.device)

    iou_mat = pairwise_mask_iou(masks).cpu()
    order = scores.argsort(descending=True).cpu().tolist()
    suppressed = [False] * N
    keep: List[int] = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        # Suppress later-in-order slots that overlap with i.
        for j in order:
            if j == i or suppressed[j]:
                continue
            if iou_mat[i, j].item() > iou_threshold:
                suppressed[j] = True
    return torch.as_tensor(keep, dtype=torch.long, device=masks.device)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-pixel argmax (resolve overlap among kept slots)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_per_pixel(
    mask_logits: torch.Tensor,      # [N, H, W]   pre-sigmoid logits of kept slots
    objectness:  torch.Tensor,      # [N]         pre-sigmoid logits
    keep_mask:   Optional[torch.Tensor] = None,    # [H, W] bool  — geometry conf filter
    score_threshold: float = 0.25,
    sem_conf:    Optional[torch.Tensor] = None,    # [N] logits; if given, per-pixel
                                                   # ownership score = σ(obj)·σ(sem_conf)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve which kept slot owns each pixel using `objectness × mask_prob`
    (× sem_conf when given → A' helps pick which slot owns contested pixels).

    Returns:
        best_slot_local:  [H, W] long.  Index into the kept-slot list, -1 if
                          no slot passes `score_threshold` (or geometry conf).
        best_score:       [H, W] float.  Per-pixel max score.
        pixel_active:     [H, W] bool.   True where best_slot_local >= 0 AND
                          geometry confidence passes.
    """
    H, W = mask_logits.shape[-2:]
    if mask_logits.numel() == 0:
        device = mask_logits.device
        return (
            torch.full((H, W), -1, dtype=torch.long, device=device),
            torch.zeros((H, W), device=device),
            torch.zeros((H, W), dtype=torch.bool, device=device),
        )

    mask_probs = mask_logits.sigmoid()                          # [N, H, W]
    obj_probs  = objectness.sigmoid()                           # [N]
    gate_scores = obj_probs.view(-1, 1, 1) * mask_probs         # [N, H, W]  obj-only (recall)
    if sem_conf is not None:
        # sem_conf picks WHICH slot owns each pixel (argmax), but the activity
        # threshold stays on the obj-only score so recall is unchanged — else the
        # low-valued σ(sem_conf) drags the product under score_threshold and pixels
        # (hence whole proposals) vanish.
        sel_scores = (obj_probs * sem_conf.sigmoid()).view(-1, 1, 1) * mask_probs
        best_slot  = sel_scores.argmax(dim=0)                   # owner = argmax(obj×sem)
        best_score = gate_scores.max(dim=0).values             # recall by obj-only
    else:
        best_score, best_slot = gate_scores.max(dim=0)         # both [H, W]
    above_threshold = best_score > score_threshold
    if keep_mask is not None:
        pixel_active = above_threshold & keep_mask.to(above_threshold.device)
    else:
        pixel_active = above_threshold
    best_slot[~pixel_active] = -1
    return best_slot, best_score, pixel_active


# ─────────────────────────────────────────────────────────────────────────────
# 3. Lift a 2D mask to a 3D proposal (center + voxel set)
# ─────────────────────────────────────────────────────────────────────────────

def lift_proposal_to_3d(
    mask:         torch.Tensor,      # [H, W]    bool  pixels belonging to this proposal
    world_points: torch.Tensor,      # [H, W, 3] world-space coords
    voxel_size:   float = 0.05,
    sem_feat:     Optional[torch.Tensor] = None,    # [D, H, W]  dense semantic feature (optional)
    outlier_iqr_mult: float = 1.5,                   # Tukey-fence multiplier (0 = filter off)
    min_pixels_for_filter: int = 50,                 # only filter when enough pts
) -> Optional[Dict]:
    """Aggregate the 3D points inside a mask into a sparse proposal.

    Outlier handling: when ``outlier_iqr_mult > 0`` AND there are enough
    valid pixels, applies a per-axis Tukey fence filter: keep only points
    in ``[q1 - m·IQR, q3 + m·IQR]`` (default m = 1.5).  Robust to varying
    outlier fractions — for clean data the fences are wider than the
    actual range (so nothing is cut), while sparse extreme outliers fall
    outside the fences and are dropped.  This stops drifting mask-edge
    pixels (typically DA3 depth bleed into the background) from inflating
    the bbox or polluting the voxel set.

    Center is computed as the **median** of the filtered points (also
    robust to outliers; survives even when filtering is disabled).

    Returns dict with:
        center:     [3] tensor       median 3D position
        voxels:     set of (vx, vy, vz) int tuples
        num_voxels: int
        num_pixels: int
        sem_query:  [D] tensor       L2-normalised per-query semantic feature
                                       (mean of sem_feat inside mask).  Only
                                       present when sem_feat was provided.
        bbox_min, bbox_max: [3] tensor  axis-aligned 3D bbox of inlier points.
    or None if the mask is empty / all points are non-finite.
    """
    if not mask.any():
        return None
    mask_bool = mask.bool()
    pts = world_points[mask_bool]                                # [P, 3]
    finite = torch.isfinite(pts).all(dim=-1)
    pts = pts[finite]
    P = pts.shape[0]
    if P == 0:
        return None

    # Outlier trimming: per-axis Tukey-fence (IQR) filter.  Adaptive to
    # data spread, so clean data passes untouched while sparse far
    # outliers (DA3 depth bleed at mask edges) are dropped.
    if outlier_iqr_mult > 0 and P >= min_pixels_for_filter:
        q1 = pts.quantile(0.25, dim=0)
        q3 = pts.quantile(0.75, dim=0)
        iqr = q3 - q1
        lo = q1 - outlier_iqr_mult * iqr
        hi = q3 + outlier_iqr_mult * iqr
        in_range = ((pts >= lo) & (pts <= hi)).all(dim=-1)
        pts = pts[in_range]
        if pts.shape[0] == 0:
            return None

    center = pts.median(dim=0).values                            # [3]  robust to outliers
    bmin = pts.min(dim=0).values
    bmax = pts.max(dim=0).values
    voxel_idx = (pts / float(voxel_size)).floor().long()         # [P, 3]
    voxel_keys: Set[Tuple[int, int, int]] = set(
        map(tuple, voxel_idx.cpu().tolist())
    )

    out: Dict = {
        'center':     center,
        'voxels':     voxel_keys,
        'num_voxels': len(voxel_keys),
        'num_pixels': int(P),
        'bbox_min':   bmin,
        'bbox_max':   bmax,
    }

    # Optional: per-query semantic descriptor (mean sem_feat within mask).
    if sem_feat is not None:
        # sem_feat: [D, H, W];  pick sem at mask pixels, average.
        # Use the same mask (without finite-points sub-filter) for simplicity —
        # the difference is at most a few NaN-depth pixels, negligible.
        D = sem_feat.shape[0]
        sem_pixels = sem_feat.permute(1, 2, 0)[mask_bool]        # [P_pix, D]
        if sem_pixels.numel() > 0:
            sem_query = sem_pixels.float().mean(dim=0)
            sem_query = torch.nn.functional.normalize(sem_query, dim=-1)
        else:
            sem_query = torch.zeros(D, device=sem_feat.device, dtype=torch.float32)
        out['sem_query'] = sem_query

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pairwise score (mirrors SLAM matching cost)
# ─────────────────────────────────────────────────────────────────────────────

def proposal_pair_score(
    prop_a: Dict,
    prop_b: Dict,
    desc_a: torch.Tensor,           # [D] L2-normalised
    desc_b: torch.Tensor,
    w_desc:     float = 0.5,
    w_vox:      float = 0.3,
    w_geo:      float = 0.2,
    sigma_dist: float = 0.5,        # metres
) -> float:
    """Combined similarity score in [-w_desc, +1].  Used as the matching cost
    to decide whether two proposals refer to the same object."""
    # Descriptor cosine (both L2-norm'd → dot product = cosine)
    sim_desc = float((desc_a * desc_b).sum().item())

    # Voxel overlap (|A ∩ B| / min(|A|, |B|) — friendly to partial views)
    if prop_a['num_voxels'] > 0 and prop_b['num_voxels'] > 0:
        inter = len(prop_a['voxels'] & prop_b['voxels'])
        min_sz = min(prop_a['num_voxels'], prop_b['num_voxels'])
        sim_vox = inter / max(min_sz, 1)
    else:
        sim_vox = 0.0

    # Center distance → Gaussian-style score
    dist = float((prop_a['center'] - prop_b['center']).norm().item())
    sim_geo = math.exp(-dist / max(sigma_dist, 1e-6))

    return w_desc * sim_desc + w_vox * sim_vox + w_geo * sim_geo


# ─────────────────────────────────────────────────────────────────────────────
# 5. Union-find merging of proposals
# ─────────────────────────────────────────────────────────────────────────────

class _UnionFind:
    """Plain int union-find with path compression + union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


def union_find_merge(
    score_matrix: torch.Tensor,       # [U, U]  symmetric pairwise score
    threshold:    float,
) -> List[int]:
    """Connect-components merge: union pairs with score > threshold.

    Returns a list of U cluster labels in 0..K-1 (compact).
    """
    U = score_matrix.shape[0]
    if U == 0:
        return []
    uf = _UnionFind(U)
    # Build edges; iterate upper triangle for efficiency.
    triu_i, triu_j = torch.triu_indices(U, U, offset=1)
    edges = score_matrix[triu_i, triu_j]
    edge_mask = edges > threshold
    for k in edge_mask.nonzero(as_tuple=True)[0].tolist():
        i = int(triu_i[k].item())
        j = int(triu_j[k].item())
        uf.union(i, j)

    raw_labels = [uf.find(i) for i in range(U)]
    # Compactify to 0..K-1
    remap: Dict[int, int] = {}
    out: List[int] = []
    for r in raw_labels:
        if r not in remap:
            remap[r] = len(remap)
        out.append(remap[r])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Convenience: full pipeline for a single chunk
# ─────────────────────────────────────────────────────────────────────────────

def _matrix_nms_decay(masks_bin: torch.Tensor, scores: torch.Tensor,
                      kernel: str = 'linear', sigma: float = 2.0) -> torch.Tensor:
    """Class-agnostic Matrix NMS (ESAM `mask_matrix_nms`, single label): instead
    of deleting overlapping masks, DECAY each mask's score by its IoU overlap
    with higher-scoring masks.  Returns decayed scores aligned to the INPUT order
    (caller then thresholds).  masks_bin: [M,H,W] or [M,P] binary float; scores [M].
    """
    M = masks_bin.shape[0]
    if M == 0:
        return scores
    flat = masks_bin.reshape(M, -1).float()
    s_sorted, order = torch.sort(scores, descending=True)
    flat = flat[order]
    area = flat.sum(1)
    inter = flat @ flat.t()
    union = area[:, None] + area[None, :] - inter
    iou = (inter / union.clamp(min=1e-6)).triu(diagonal=1)            # [M,M] (CA: all same label)
    compensate_iou, _ = iou.max(dim=0)                               # [M]
    compensate_iou = compensate_iou[None, :].expand(M, M).transpose(0, 1)
    if kernel == 'gaussian':
        decay = torch.exp(-sigma * iou ** 2) / torch.exp(-sigma * compensate_iou ** 2)
        coef, _ = decay.min(dim=0)
    else:  # linear
        decay = (1.0 - iou) / (1.0 - compensate_iou).clamp(min=1e-6)
        coef, _ = decay.min(dim=0)
    s_decayed = s_sorted * coef
    out = torch.empty_like(scores)
    out[order] = s_decayed                                           # map back to input order
    return out


def process_chunk_to_proposals(
    mask_logits:     torch.Tensor,    # [T, N, H, W]
    objectness:      torch.Tensor,    # [T, N]
    descriptor:      torch.Tensor,    # [T, N, D]   L2-normalised
    world_points:    torch.Tensor,    # [T, H, W, 3]
    keep_mask:       torch.Tensor,    # [T, H, W]
    obj_threshold:   float = 0.5,
    obj_top_k:       Optional[int] = None,             # if set, keep only top-K obj slots per frame (AND obj_threshold)
    nms_iou:         float = 0.5,
    score_threshold: float = 0.25,
    voxel_size:      float = 0.05,
    sem_feat:        Optional[torch.Tensor] = None,   # [T, D, H, W]  per-pixel sem (optional)
    box:             Optional[torch.Tensor] = None,   # [T, N, 6]     predicted box (camera frame, canonical)
    box_scale:       Optional[float] = None,           # scalar:        canonical → metric multiplier
    poses:           Optional[torch.Tensor] = None,    # [T, 4, 4]      c2w per frame, for camera → world
    box_anchor_mode: str = 'abs',                      # 'abs' or 'anchor' — MUST match the head's training-time mode
    box_world_frame: bool = False,                     # True when head already outputs world-frame AABB (box_mode='world_aabb')
    outlier_iqr_mult: float = 1.5,                     # Tukey-fence multiplier for `lift_proposal_to_3d`
    min_points:      int = 0,                          # drop proposals with < this many 3D-lifted points (ESAM npoint_thr analog); 0 = off
    # ── ESAM-style proposal post-processing (all default = legacy behaviour) ──
    nms_mode:        str = 'hard',                      # 'hard' (mask-IoU NMS, delete) | 'matrix' (ESAM soft score-decay)
    matrix_nms_kernel: str = 'linear',                 # 'linear' | 'gaussian' (matrix mode)
    panoptic:        bool = True,                       # True: per-pixel argmax (1 owner/pixel); False: each query's own mask (ESAM, masks may overlap)
    sp_score_thr:    float = 0.5,                       # mask binarize thr when panoptic=False
    obj_normalize:   bool = False,                      # re-score objectness by mask quality (ESAM obj_normalization)
    sem_conf:        Optional[torch.Tensor] = None,     # [T, N] A' semantic mask-quality logits.  Folded into gating
                                                        # (sem_conf_gate) and/or cluster_score (sem_conf_rank), independently.
    sem_conf_gate:   bool = False,                      # fold σ(sem_conf) into GATING/top-K/NMS (needs a matched-budget
                                                        # obj_threshold).  Decoupled from ranking below.
    sem_conf_rank:   bool = True,                       # fold σ(sem_conf) into the STORED cluster_score (ranking).
    sem_conf_select: bool = False,                      # fold σ(sem_conf) into SELECTION only — per-pixel ownership,
                                                        # top-K, NMS winner — but NOT the recall threshold (which stays
                                                        # pure obj).  Lets A' pick the better mask among competitors
                                                        # without tightening recall.  Independent of gate/rank.
) -> Tuple[List[Dict], torch.Tensor]:
    """Per-frame: objectness filter → NMS → argmax → 3D lift.

    When `box` + `box_scale` + `poses` are all given, the head-predicted
    camera-frame canonical box is converted to a world AABB and stored
    as `proposal['pred_bbox_min']` / `pred_bbox_max'`.  This is kept
    SEPARATE from the mask-derived `bbox_min` / `bbox_max` — the observed
    bbox stays as the tight "what we actually saw" extent (used for
    storage / vis / export), while the predicted bbox is the head's
    estimate of the FULL extent (used as the matching cue, since it can
    represent occluded parts).

    Returns:
        proposals: list of dicts (descriptor, center, voxels, frame, slot, ...)
        pixel_to_proposal: [T, H, W] long, -1 where no proposal claims the pixel
    """
    T, N, H, W = mask_logits.shape
    device = mask_logits.device
    proposals: List[Dict] = []
    pixel_to_proposal = torch.full((T, H, W), -1, dtype=torch.long, device=device)

    for t in range(T):
        # 1) Objectness filter — top_k AND threshold (intersection).
        #    Focal-loss-trained heads compress the obj range (peak ~0.5–0.7),
        #    so the legacy absolute 0.5 floor drops too many real masks.
        #    Top_k bounds the per-frame Hungarian size; obj_threshold floors
        #    out clearly-bad ones.  Both must hold to keep a slot.
        base_t = objectness[t].sigmoid()
        # (ESAM obj_normalization) re-score by mask quality = sigmoid-weighted
        # coverage of the positive mask region → confidence reflects mask quality.
        if obj_normalize:
            ml_t = mask_logits[t]                                      # [N, H, W]
            pos  = ml_t > 0
            mq   = (ml_t.sigmoid() * pos).sum(dim=(-1, -2)) / (pos.sum(dim=(-1, -2)) + 1e-6)
            base_t = base_t * mq
        # A' sem_conf enters two DECOUPLED scores:
        #   GATE  (recall threshold): base [× sem_conf if sem_conf_gate].  Keep this
        #         pure-obj (gate=False) so A''s miscalibrated absolute value doesn't
        #         drop real masks.
        #   SELECT(per-pixel ownership / top-K / NMS winner): base [× sem_conf if
        #         sem_conf_select].  Lets A' pick the better mask among COMPETITORS
        #         without tightening recall — the untested "A' improves selection" path.
        obj_probs_gate = base_t
        if sem_conf is not None and sem_conf_gate:
            obj_probs_gate = obj_probs_gate * sem_conf[t].sigmoid()
        obj_probs_sel = base_t
        if sem_conf is not None and sem_conf_select:
            obj_probs_sel = obj_probs_sel * sem_conf[t].sigmoid()
        active = obj_probs_gate > obj_threshold                       # [N] bool (recall = obj)
        if obj_top_k is not None and obj_top_k < N:
            top_idx = obj_probs_sel.topk(int(obj_top_k)).indices       # top-K by SELECT score
            top_mask = torch.zeros_like(active)
            top_mask[top_idx] = True
            active = active & top_mask
        if not active.any():
            continue
        active_idx = active.nonzero(as_tuple=True)[0]                  # global slot indices

        # 2) NMS among active slots — 'hard' (delete) or 'matrix' (soft decay).
        #    Ranked by the SELECT score → A' picks the winning duplicate.
        if nms_mode == 'matrix':
            mbin = (mask_logits[t, active_idx].sigmoid() > 0.5).float()    # [N_act, H, W]
            decayed = _matrix_nms_decay(mbin, obj_probs_sel[active_idx],
                                        kernel=matrix_nms_kernel)
            kept_local = (decayed > obj_threshold).nonzero(as_tuple=True)[0]
        else:  # 'hard'
            mbin = mask_logits[t, active_idx].sigmoid() > 0.5
            kept_local = mask_nms(mbin, obj_probs_sel[active_idx], iou_threshold=nms_iou)
        if kept_local.numel() == 0:
            continue
        kept_slots = active_idx[kept_local]                            # [N_kept] global slot ids

        # 3) Pixel assignment — panoptic (per-pixel argmax, 1 owner/pixel) or
        #    per-query own mask (ESAM: masks may overlap → fuller per-proposal geometry).
        sem_feat_t = sem_feat[t] if sem_feat is not None else None    # [D, H, W] or None
        if panoptic:
            best_slot_local, _, pixel_active = resolve_per_pixel(
                mask_logits[t, kept_slots], objectness[t, kept_slots], keep_mask[t],
                score_threshold=score_threshold,
                sem_conf=(sem_conf[t, kept_slots]
                          if (sem_conf is not None and sem_conf_select) else None),
            )
            def _slot_pixels(kl):
                return (best_slot_local == kl) & pixel_active
        else:
            kept_sig = mask_logits[t, kept_slots].sigmoid()            # [N_kept, H, W]
            def _slot_pixels(kl):
                return (kept_sig[kl] > sp_score_thr) & keep_mask[t]

        # 4) For each kept slot that owns ≥1 pixel, lift to 3D
        for k_local in range(kept_slots.numel()):
            slot_pixels = _slot_pixels(k_local)
            if not slot_pixels.any():
                continue
            prop = lift_proposal_to_3d(
                slot_pixels, world_points[t], voxel_size=voxel_size,
                sem_feat=sem_feat_t,
                outlier_iqr_mult=outlier_iqr_mult,
            )
            if prop is None:
                continue
            # Per-frame min-points filter (ESAM npoint_thr analog): drop tiny
            # proposals BEFORE they enter the InstanceTable, so they don't
            # pollute cross-chunk matching.  Counts 3D-lifted points (= valid
            # mask pixels).  0 = off (default → unchanged behaviour).
            if min_points > 0 and prop['num_pixels'] < min_points:
                continue
            slot_global = int(kept_slots[k_local].item())
            prop['frame']      = int(t)
            prop['slot']       = slot_global
            prop['descriptor'] = descriptor[t, slot_global].detach()
            _obj = float(objectness[t, slot_global].sigmoid().item())
            if sem_conf is not None and sem_conf_rank:
                _obj *= float(sem_conf[t, slot_global].sigmoid().item())   # obj×A' → cluster_score
            prop['objectness'] = _obj

            # Predicted bbox: convert camera-canonical → world AABB and
            # store as a separate field (does NOT overwrite the mask-
            # derived observed bbox).  Downstream uses the observed one
            # for storage / vis (tight, matches what we saw) and the
            # predicted one for the matching cue (loose but estimates
            # the full extent including unseen parts).
            # Box → pred_bbox_min/max for SLAM merging.  Three paths:
            #   world_frame=True: box is in NORMALIZE scale (from BoxAABBHead).
            #                     Multiply by box_scale to recover CANONICAL
            #                     (= same scale as world_points / geo PLY)
            #                     so the box overlays the scene geometry.
            #   else (legacy):    camera-canonical box, transform via box_scale × c2w.
            if box is not None and box_world_frame:
                box_t_n = box[t, slot_global].detach()
                assert box_t_n.numel() == 6, \
                    f"box_world_frame=True expects 6-d AABB, got {box_t_n.numel()}-d"
                # Convert NORMALIZE → CANONICAL by multiplying by box_scale.
                s = float(box_scale) if box_scale is not None else 1.0
                center = box_t_n[:3] * s
                size   = (box_t_n[3:6] * s).clamp(min=1e-3)
                pred_bmin = center - size * 0.5
                pred_bmax = center + size * 0.5
                prop['pred_bbox_min'] = pred_bmin
                prop['pred_bbox_max'] = pred_bmax
                prop['box']           = box_t_n     # raw normalize-scale 6-d for debug

                # ── Also compute the PCA-PRIOR box (canonical) for vis only ──
                # Same math as BoxAABBHead's prior path (hard-threshold mask,
                # √3·σ, clamp).  Lets us export an extra PLY to compare:
                # GT vs PCA-prior vs head's final pred.  No grad, cpu-friendly.
                import math as _math
                sm = mask_logits[t, slot_global].sigmoid().detach()       # [H, W]
                sm_hard = (sm > 0.5).to(sm.dtype)
                sm_use  = sm_hard if sm_hard.sum() > 5 else sm
                w_sum   = sm_use.sum().clamp(min=1e-3)
                wp_t    = world_points[t]                                   # [H, W, 3]  CANONICAL
                ctr_p   = (sm_use.unsqueeze(-1) * wp_t).sum(dim=(0, 1)) / w_sum
                sq_p    = (sm_use.unsqueeze(-1) * wp_t.pow(2)).sum(dim=(0, 1)) / w_sum
                var_p   = (sq_p - ctr_p.pow(2)).clamp(min=0.0)
                sh_p    = (var_p.sqrt() * _math.sqrt(3.0)).clamp(min=0.5e-3)
                prop['pred_prior_bbox_min'] = ctr_p - sh_p
                prop['pred_prior_bbox_max'] = ctr_p + sh_p
            elif box is not None and box_scale is not None and poses is not None:
                box_t_n = box[t, slot_global].detach()
                # ── Anchor-mode: compute world-frame mask centroid for this
                #    (t, slot_global) and pass to the transform so the
                #    PLY-side box position matches training-loss convention.
                anchor_w = None
                if box_anchor_mode == 'anchor':
                    sm = mask_logits[t, slot_global].sigmoid()        # [H, W]
                    w_sum = sm.sum().clamp(min=1e-3)
                    # world_points[t]: [H, W, 3]
                    anchor_w = (sm.unsqueeze(-1) * world_points[t]).sum(dim=(0, 1)) / w_sum
                pred_bmin, pred_bmax = pred_box_to_world_aabb(
                    box_t_n, float(box_scale), poses[t], anchor_world=anchor_w,
                )
                prop['pred_bbox_min'] = pred_bmin
                prop['pred_bbox_max'] = pred_bmax
                prop['box']           = box_t_n     # raw canonical for debug
                # For OBB (12-d), also stash the 8 oriented world corners so
                # downstream PLY export can draw the actual OBB wireframe
                # instead of the AABB encloser (which loses rotation info).
                if box_t_n.numel() == 12:
                    prop['pred_obb_corners'] = pred_obb_to_world_corners(
                        box_t_n, float(box_scale), poses[t], anchor_world=anchor_w,
                    )

            prop_idx = len(proposals)
            proposals.append(prop)
            pixel_to_proposal[t][slot_pixels] = prop_idx

    return proposals, pixel_to_proposal


def pred_box_to_world_aabb(
    box_canonical: torch.Tensor,    # [6] AABB (cx,cy,cz,w,h,d)  OR  [12] OBB (cx,cy,cz, sx,sy,sz, r1..r6)
    box_scale:     float,
    c2w:           torch.Tensor,    # [4, 4]
    anchor_world:  Optional[torch.Tensor] = None,   # [3]  world-frame mask centroid
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Camera-canonical box → world AABB (re-axis-aligned after rotation).

    Two coord-transform modes (must match training-time loss convention):
      - anchor_world=None   (abs):     center_w = R_c2w @ (delta_c × s) + t_c2w
      - anchor_world=given  (anchor):  center_w = R_c2w @ (delta_c × s) + anchor_world
                                       (loss does this when --box_anchor_mode anchor)

    Steps:
        1. Multiply canonical → metric (× box_scale) on center + size only.
        2. Build the 8 corners in camera frame (AABB-style for 6-d,
           R-rotated for 12-d).
        3. Apply R_c2w to rotate corners; translate by anchor_world (if given)
           else by t_c2w.
        4. Take min/max → world AABB.

    Returns (world_bmin [3], world_bmax [3]).
    """
    signs = torch.tensor([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=box_canonical.dtype, device=box_canonical.device)

    s = float(box_scale)
    if box_canonical.numel() == 12:
        # OBB: scale center + size_half (rot is unit-less).
        from .box_gt import rot_6d_to_matrix
        center    = box_canonical[:3] * s                                # [3]
        size_half = box_canonical[3:6] * s                               # [3]
        R         = rot_6d_to_matrix(box_canonical[6:12])                # [3, 3]
        local     = signs * size_half[None, :]                           # [8, 3]
        # Rotate then translate: cam_corner = R @ local + center.
        corners_cam = (R @ local.T).T + center[None, :]                  # [8, 3]
    else:
        # 6-d AABB legacy: full size → half = size * 0.5.
        box_metric = box_canonical * s
        center     = box_metric[:3]
        half       = box_metric[3:6] * 0.5
        corners_cam = center[None, :] + signs * half[None, :]            # [8, 3]

    R_c2w = c2w[:3, :3].to(corners_cam.dtype)
    rotated = (R_c2w @ corners_cam.T).T                                  # [8, 3]
    if anchor_world is not None:
        t = anchor_world.to(corners_cam.dtype)
    else:
        t = c2w[:3, 3].to(corners_cam.dtype)
    corners_w = rotated + t[None, :]
    bmin = corners_w.min(dim=0).values
    bmax = corners_w.max(dim=0).values
    return bmin, bmax


def pred_obb_to_world_corners(
    box_canonical: torch.Tensor,    # [12]  (cx,cy,cz, sx,sy,sz, r1..r6)
    box_scale:     float,
    c2w:           torch.Tensor,    # [4, 4]
    anchor_world:  Optional[torch.Tensor] = None,   # [3] world-frame mask centroid
) -> torch.Tensor:
    """Compute the 8 oriented-bbox corners in world frame (metric).

    MUST match the training-time transform in `transform_obb_cam_to_world`:
      - anchor_world=None  (abs):     center_w = R_c2w @ (delta × s) + t_c2w
      - anchor_world=given (anchor):  center_w = R_c2w @ (delta × s) + anchor_world

    Mismatch causes pred boxes to be drawn at the camera origin instead of
    the actual object location (visible as boxes floating outside the scene
    geometry in the PLY).

    Returns: [8, 3] tensor on `box_canonical.device`.
    """
    from .box_gt import rot_6d_to_matrix
    signs = torch.tensor([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=box_canonical.dtype, device=box_canonical.device)
    s = float(box_scale)
    center    = box_canonical[:3] * s                                  # [3]
    size_half = box_canonical[3:6] * s                                 # [3]
    R         = rot_6d_to_matrix(box_canonical[6:12])                  # [3, 3]
    local     = signs * size_half[None, :]                             # [8, 3]
    corners_cam = (R @ local.T).T + center[None, :]                    # [8, 3]
    R_c2w = c2w[:3, :3].to(corners_cam.dtype)
    rotated = (R_c2w @ corners_cam.T).T                                # [8, 3]
    if anchor_world is not None:
        t = anchor_world.to(corners_cam.dtype)
    else:
        t = c2w[:3, 3].to(corners_cam.dtype)
    return rotated + t[None, :]                                        # [8, 3]


def point_to_aabb_distance(point: torch.Tensor,
                              bmin: torch.Tensor,
                              bmax: torch.Tensor) -> float:
    """Euclidean distance from a 3D point to an axis-aligned bbox.

    Returns 0 when `point` lies inside `[bmin, bmax]`.  Returns the
    distance to the nearest bbox face when `point` is outside.

    Args:
        point: [3] tensor
        bmin / bmax: [3] tensors, bbox corner extents

    Why we use this for spatial gating + the geo score cue (instead of
    the legacy point-to-point center distance):
        For a long object accumulated across views the cluster `center`
        is the running average of partial-view centers, which settles to
        the middle of observed extent.  A new view at the edge of the
        cluster is then "center-far" but physically still inside the
        cluster's spatial occupancy.  Using point-to-bbox naturally
        handles this — the metric scales with the cluster's physical
        extent rather than collapsing it to a single point.
    """
    zero = torch.zeros_like(point)
    delta = torch.maximum(bmin - point, zero) + torch.maximum(point - bmax, zero)
    return float(delta.norm().item())


def _spatial_distance_for_pair(p: Dict, q: Dict) -> float:
    """Symmetric spatial distance used by the geo score cue.

    Takes the min of (p.center → q.bbox) and (q.center → p.bbox).  Either
    proposal being "spatially inside" the other yields distance 0.  When
    a proposal has no bbox (legacy callers), falls back to point-to-point
    center distance.
    """
    p_has_bbox = ('bbox_min' in p and 'bbox_max' in p)
    q_has_bbox = ('bbox_min' in q and 'bbox_max' in q)
    if p_has_bbox and q_has_bbox:
        d1 = point_to_aabb_distance(p['center'], q['bbox_min'], q['bbox_max'])
        d2 = point_to_aabb_distance(q['center'], p['bbox_min'], p['bbox_max'])
        return min(d1, d2)
    if q_has_bbox:
        return point_to_aabb_distance(p['center'], q['bbox_min'], q['bbox_max'])
    if p_has_bbox:
        return point_to_aabb_distance(q['center'], p['bbox_min'], p['bbox_max'])
    return float((p['center'] - q['center']).norm().item())


def aabb_iou_min(bmin_a: torch.Tensor, bmax_a: torch.Tensor,
                   bmin_b: torch.Tensor, bmax_b: torch.Tensor) -> float:
    """3D AABB intersection-over-min-volume.

    Like standard IoU but uses ``min(vol_a, vol_b)`` as the denominator
    instead of ``union``.  This gives a strong signal when a small box is
    fully contained inside a larger one — common in SLAM, where an
    instance's accumulated bbox is much larger than a single new
    observation, but matches the same object.  Used for the bbox-IoU
    matching cue (Phase 3 replacement for the geo cue when both
    proposals carry predicted bboxes).
    """
    inter_min = torch.maximum(bmin_a, bmin_b)
    inter_max = torch.minimum(bmax_a, bmax_b)
    inter = (inter_max - inter_min).clamp(min=0).prod()
    vol_a = (bmax_a - bmin_a).clamp(min=1e-6).prod()
    vol_b = (bmax_b - bmin_b).clamp(min=1e-6).prod()
    denom = torch.minimum(vol_a, vol_b).clamp(min=1e-6)
    return float((inter / denom).item())


def aabb_iou_union(bmin_a: torch.Tensor, bmax_a: torch.Tensor,
                   bmin_b: torch.Tensor, bmax_b: torch.Tensor) -> float:
    """Standard 3D AABB IoU (intersection-over-UNION) — ESAM's box cue.

    Unlike `aabb_iou_min`, a small box fully inside a large one scores LOW
    (small/large), so it does NOT over-merge nested objects (part-in-whole).
    ESAM uses this (AxisAlignedBboxOverlaps3D, mode='iou') as a multiplicative
    factor: query_feat_cos × box_IoU × class_gate.
    """
    inter_min = torch.maximum(bmin_a, bmin_b)
    inter_max = torch.minimum(bmax_a, bmax_b)
    inter = (inter_max - inter_min).clamp(min=0).prod()
    vol_a = (bmax_a - bmin_a).clamp(min=1e-6).prod()
    vol_b = (bmax_b - bmin_b).clamp(min=1e-6).prod()
    union = (vol_a + vol_b - inter).clamp(min=1e-6)
    return float((inter / union).item())


def _aabb_iou(mode: str, bmin_a, bmax_a, bmin_b, bmax_b) -> float:
    """Dispatch box-IoU by mode: 'min' (inter/min, default) or 'union' (ESAM)."""
    if mode == 'union':
        return aabb_iou_union(bmin_a, bmax_a, bmin_b, bmax_b)
    return aabb_iou_min(bmin_a, bmax_a, bmin_b, bmax_b)


def box_iou_pair(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """Axis-aligned 3D bbox IoU between two boxes in (cx, cy, cz, sx, sy, sz)."""
    a_min = box_a[:3] - box_a[3:].clamp(min=1e-3) * 0.5
    a_max = box_a[:3] + box_a[3:].clamp(min=1e-3) * 0.5
    b_min = box_b[:3] - box_b[3:].clamp(min=1e-3) * 0.5
    b_max = box_b[:3] + box_b[3:].clamp(min=1e-3) * 0.5
    inter_min = torch.maximum(a_min, b_min)
    inter_max = torch.minimum(a_max, b_max)
    inter = (inter_max - inter_min).clamp(min=0).prod()
    a_vol = (a_max - a_min).clamp(min=1e-6).prod()
    b_vol = (b_max - b_min).clamp(min=1e-6).prod()
    union = a_vol + b_vol - inter
    return float((inter / union.clamp(min=1e-6)).item())


def _proposal_pair_score_dict(
    p:  Dict, q: Dict,
    w_desc: float, w_vox: float, w_geo: float,
    w_sem:  float, w_box: float,
    sigma_dist: float,
    vox_gate:  Optional[float] = None,
    desc_gate: Optional[float] = None,
    sem_gate_thr: float = 0.0,
    combine:      str   = 'add',
    vox_bonus_alpha: float = 1.0,
    desc_pow:        float = 1.0,
    vox_or_tau:      float = 0.5,
    box_iou_mode:    str   = 'min',
    merge_head=None,
) -> float:
    """Like `proposal_pair_score` but with dict access + optional sem/box cues.

    Each cue is gated on availability — if the proposals don't carry it (or
    weight is 0) the term contributes 0.

    `combine` selects how the cues are fused:
      - `'add'` (default): weighted linear sum (original behaviour).
      - `'mul'` : ESAM-style product `∏ cue` over the cues whose weight > 0.
        In mul mode the weight MAGNITUDE is irrelevant (scaling a product just
        shifts the merge threshold) — `w_desc/w_vox/w_geo` act as ON/OFF
        switches (>0 = include the factor).  Descriptor cosine is clamped to
        [0,1] first: a negative cosine would flip the product's sign and a
        neg×neg would fake a high score.  The semantic cue is NEVER a factor
        (see `sem_gate_thr`).
      - `'mul_voxbonus'` (A): `d·geo·(1 + vox_bonus_alpha·vox)`.  vox is a
        BONUS, not a veto — neutral (×1) when vox=0 (our vox cue is sparse so
        =0 is "no info" not "different object"), boosted when voxels overlap.
        Range [0, 1+alpha].
      - `'geo_gated'`  (C): `(w_desc·d + w_vox·vox)·geo`.  Additive desc+vox
        core (cues compensate, like 'add') but spatially GATED by geo (far →
        ≈0).  Here weight MAGNITUDES matter.  Range [0, w_desc+w_vox].
      - `'mul_descpow'`(D): `d**desc_pow · geo`.  desc_pow>1 sharpens the
        descriptor — crushes mediocre (0.5–0.6 same-class) cosines while
        keeping true matches (d>0.85).  Range [0, 1].
      - `'or_dual'`    (E): `max(d·geo, vox if vox>vox_or_tau else 0)`.  Two
        INDEPENDENT merge evidences: appearance+spatial OR strong voxel
        overlap.  Range [0, 1].
      where `d = clamp(sim_desc, 0, 1)`.

    `sem_gate_thr` (applies to BOTH modes, default 0 = off): a soft semantic
    gate.  If both proposals carry `sem_query` and their cosine is BELOW this
    threshold, the pair is a different semantic category → return 0 (veto the
    merge).  Kept as a gate rather than a scored cue because adding sem to the
    score inflates same-class-different-instance pairs (two chairs have high
    sem sim); gate-only avoids that while still blocking cross-category merges.

    Multi-gate overrides (optional, default None = off; `'add'` mode only):
      - `vox_gate`  (e.g. 0.5): if voxel intersection ratio exceeds this,
        force score = 1.0 (geometric evidence strong on its own).  Targets
        the "two records claim the same surface" pattern that produces
        red-blue checkerboards in voxel PLYs.
      - `desc_gate` (e.g. 0.88): if descriptor cosine exceeds this, force
        score = 1.0.  Targets long-distance loop closures where vox=0
        but descriptor is confident.
    """
    sim_desc = float((p['descriptor'] * q['descriptor']).sum().item())

    if p['num_voxels'] > 0 and q['num_voxels'] > 0:
        inter = len(p['voxels'] & q['voxels'])
        sim_vox = inter / max(min(p['num_voxels'], q['num_voxels']), 1)
    else:
        sim_vox = 0.0

    # ── Learned merge head (overrides all hand-weighted cues / gates) ───────
    # combine='learned': P(same instance) from the trained MergeHead on the
    # SAME features as training (desc / sem / vox-IoU / center-dist).  Builds
    # the pair feature with the shared `build_pair_features` so train↔SLAM
    # layouts match.  Needs sem on both sides; if missing → don't merge (0.0).
    if combine == 'learned' and merge_head is not None:
        sp, sq = p.get('sem_query'), q.get('sem_query')
        if sp is None or sq is None:
            return 0.0
        import torch as _torch
        from .merge_head import build_pair_features as _bpf
        dev = next(merge_head.parameters()).device
        # p (proposal) may be on GPU, q (record) on CPU → move both to dev.
        cdist = float((p['center'].to(dev) - q['center'].to(dev)).norm().item())
        with _torch.no_grad():
            a_desc, a_sem, g = _bpf(
                p['descriptor'].to(dev), q['descriptor'].to(dev),
                sp.to(dev), sq.to(dev),
                _torch.tensor(sim_vox, device=dev),
                _torch.tensor(cdist, device=dev),
            )
            return float(_torch.sigmoid(merge_head(a_desc, a_sem, g)).item())

    # Soft semantic gate (both modes): different-category pairs are vetoed
    # outright.  Cheap, so do it first.
    if sem_gate_thr > 0.0 and 'sem_query' in p and 'sem_query' in q:
        sim_sem_gate = float((p['sem_query'] * q['sem_query']).sum().item())
        if sim_sem_gate < sem_gate_thr:
            return 0.0

    # Multi-gate (cheap, before geo computation): strong single-cue evidence
    # bypasses the linear formula.  vox_gate handles the red-blue case;
    # desc_gate handles long-distance loop closure.  Only meaningful for the
    # additive formula — in 'mul' mode the product itself is the affinity.
    if combine == 'add':
        if vox_gate is not None and sim_vox > vox_gate:
            return 1.0
        if desc_gate is not None and sim_desc > desc_gate:
            return 1.0

    # Spatial cue: when BOTH proposals carry a head-predicted bbox
    # (`pred_bbox_min/max`), use 3D AABB min-IoU on the predicted
    # bboxes — predicted boxes estimate the FULL object extent so a
    # partial-view bbox is contained in the cluster bbox → IoU ≈ 1.
    # Otherwise fall back to point-to-bbox exp decay on the OBSERVED
    # bbox (the mask-derived one, which everyone has).
    use_pred_bbox_cue = (
        'pred_bbox_min' in p and 'pred_bbox_max' in p
        and 'pred_bbox_min' in q and 'pred_bbox_max' in q
    )
    if use_pred_bbox_cue:
        sim_geo = _aabb_iou(box_iou_mode,
                            p['pred_bbox_min'], p['pred_bbox_max'],
                            q['pred_bbox_min'], q['pred_bbox_max'])
    else:
        dist    = _spatial_distance_for_pair(p, q)
        sim_geo = math.exp(-dist / max(sigma_dist, 1e-6))

    dcl = max(sim_desc, 0.0)          # clamped descriptor cosine ∈ [0,1]
    svox = float(sim_vox)
    sgeo = float(sim_geo)

    # ── Multiplicative (ESAM-style) affinity ───────────────────────────────
    if combine == 'mul':
        factors = []
        if w_desc > 0:
            factors.append(dcl)                  # clamp cosine to [0,1]
        if w_vox > 0:
            factors.append(svox)                 # ∈[0,1]; =0 → veto (no overlap)
        if w_geo > 0:
            factors.append(sgeo)                 # exp(-dist/σ)∈(0,1]; smooth prior
        if not factors:
            return 0.0
        score = 1.0
        for f in factors:
            score *= f
        return score

    # ── A: vox-as-multiplicative-bonus  d·geo·(1 + α·vox) ───────────────────
    if combine == 'mul_voxbonus':
        return dcl * sgeo * (1.0 + vox_bonus_alpha * svox)

    # ── C: geo-gated additive  (w_desc·d + w_vox·vox)·geo ───────────────────
    if combine == 'geo_gated':
        return (w_desc * dcl + w_vox * svox) * sgeo

    # ── D: sharpened descriptor  d**p · geo ─────────────────────────────────
    if combine == 'mul_descpow':
        return (dcl ** desc_pow) * sgeo

    # ── E: dual-path OR  max(d·geo, vox if vox>τ else 0) ────────────────────
    if combine == 'or_dual':
        vox_path = svox if svox > vox_or_tau else 0.0
        return max(dcl * sgeo, vox_path)

    # ── Additive (default) ─────────────────────────────────────────────────
    score = w_desc * sim_desc + w_vox * sim_vox + w_geo * sim_geo

    if w_sem > 0 and 'sem_query' in p and 'sem_query' in q:
        sim_sem = float((p['sem_query'] * q['sem_query']).sum().item())
        score += w_sem * sim_sem

    if w_box > 0 and 'box' in p and 'box' in q:
        # Legacy AABB-on-canonical-box IoU cue.  Only valid for 6-d
        # (cx,cy,cz,w,h,d).  Under OBB (12-d) the analogous information is
        # already covered by `sim_geo` via the pred_bbox AABB-IoU above,
        # so we just skip (don't error, don't double-count).
        if p['box'].numel() == 6 and q['box'].numel() == 6:
            sim_box = box_iou_pair(p['box'], q['box'])
            score += w_box * sim_box

    return score


def _init_instance_from_proposal(p: Dict) -> Dict:
    """Initialise an accumulated instance representation from a single proposal.

    Bbox fields (observed `bbox_min/max` and head-predicted
    `pred_bbox_min/max`) are propagated when present — they are needed by
    `_spatial_distance_for_pair` so the geo cue becomes center-to-cluster-bbox
    instead of falling back to center-to-center.
    """
    inst = {
        'descriptor': p['descriptor'].clone(),
        'center':     p['center'].clone(),
        'voxels':     set(p['voxels']),
        'num_voxels': p['num_voxels'],
        'count':      1,
        'member_proposals': [],     # list of proposal indices (filled by caller)
    }
    if 'sem_query' in p:
        inst['sem_query'] = p['sem_query'].clone()
    if 'box' in p:
        inst['box'] = p['box'].clone()
    # Observed (mask-derived) bbox — taken from the single proposal at init.
    if 'bbox_min' in p and 'bbox_max' in p:
        inst['bbox_min'] = p['bbox_min'].clone()
        inst['bbox_max'] = p['bbox_max'].clone()
    # Predicted (head full-extent) bbox — running-average init.
    if 'pred_bbox_min' in p and 'pred_bbox_max' in p:
        inst['pred_bbox_min'] = p['pred_bbox_min'].clone()
        inst['pred_bbox_max'] = p['pred_bbox_max'].clone()
    if 'pred_prior_bbox_min' in p and 'pred_prior_bbox_max' in p:
        inst['pred_prior_bbox_min'] = p['pred_prior_bbox_min'].clone()
        inst['pred_prior_bbox_max'] = p['pred_prior_bbox_max'].clone()
    if 'pred_obb_corners' in p:
        inst['pred_obb_corners'] = p['pred_obb_corners'].clone()
    return inst


def _update_instance_with_proposal(inst: Dict, p: Dict):
    """Running-average update of an instance representation (Mask2Former /
    ESAM convention).  Voxel sets accumulate by union; observed bboxes
    take the UNION (cluster bbox = enclosing of all member bboxes); predicted
    bbox uses running-average (head's per-view estimate of FULL extent — averaging
    is more robust than min/max which would amplify any single-view noise)."""
    n = inst['count']
    a = n / (n + 1)
    b = 1.0 / (n + 1)
    inst['descriptor'] = torch.nn.functional.normalize(
        a * inst['descriptor'] + b * p['descriptor'], dim=-1,
    )
    inst['center'] = a * inst['center'] + b * p['center']
    inst['voxels'] = inst['voxels'] | p['voxels']
    inst['num_voxels'] = len(inst['voxels'])
    if 'sem_query' in inst and 'sem_query' in p:
        inst['sem_query'] = torch.nn.functional.normalize(
            a * inst['sem_query'] + b * p['sem_query'], dim=-1,
        )
    if 'box' in inst and 'box' in p:
        inst['box'] = a * inst['box'] + b * p['box']
    # Observed bbox: take elementwise min/max → enclosing cluster bbox.
    if 'bbox_min' in inst and 'bbox_min' in p:
        inst['bbox_min'] = torch.minimum(inst['bbox_min'], p['bbox_min'])
        inst['bbox_max'] = torch.maximum(inst['bbox_max'], p['bbox_max'])
    # Predicted bbox: running-average (matches `box`/`sem_query` convention).
    if 'pred_bbox_min' in inst and 'pred_bbox_min' in p:
        inst['pred_bbox_min'] = a * inst['pred_bbox_min'] + b * p['pred_bbox_min']
        inst['pred_bbox_max'] = a * inst['pred_bbox_max'] + b * p['pred_bbox_max']
    # PCA prior bbox: same running-average update (vis-only field).
    if 'pred_prior_bbox_min' in inst and 'pred_prior_bbox_min' in p:
        inst['pred_prior_bbox_min'] = a * inst['pred_prior_bbox_min'] + b * p['pred_prior_bbox_min']
        inst['pred_prior_bbox_max'] = a * inst['pred_prior_bbox_max'] + b * p['pred_prior_bbox_max']
    # Predicted OBB corners: running-average over the 8 oriented corners.
    # Note: averaging corners is approximate (the average of two rotated
    # boxes is not a "valid" box), but for visualization it gives a
    # reasonable interpolation between view-specific predictions.
    if 'pred_obb_corners' in inst and 'pred_obb_corners' in p:
        inst['pred_obb_corners'] = a * inst['pred_obb_corners'] + b * p['pred_obb_corners']
    inst['count'] += 1


def match_proposals_sequential_hungarian(
    proposals:  List[Dict],
    w_desc:     float = 0.5,
    w_vox:      float = 0.3,
    w_geo:      float = 0.2,
    w_sem:      float = 0.0,
    w_box:      float = 0.0,
    sigma_dist: float = 0.5,
    match_threshold: float = 0.35,
) -> List[int]:
    """Streaming-style merging that mirrors SLAM online inference.

    Process proposals frame-by-frame.  For each new frame, compute a similarity
    matrix against the accumulated instance table and solve **bipartite
    Hungarian matching** — every current proposal matches at most ONE existing
    instance.  Matched pairs merge (running-average update).  Unmatched
    proposals become new instances.

    Avoids the transitive-chain collapse of union-find: even if descriptors are
    weak, two different physical objects can no longer merge through a chain of
    intermediates.

    Returns a list of cluster-id per proposal (same length as `proposals`).
    """
    if not _HAS_SCIPY:
        # Fall back to union-find if scipy unavailable.
        return union_find_merge(
            _build_score_matrix(proposals, w_desc, w_vox, w_geo, w_sem, w_box, sigma_dist),
            threshold=match_threshold,
        )

    U = len(proposals)
    if U == 0:
        return []

    # Group proposals by frame, keep original indices.
    by_frame: Dict[int, List[int]] = {}
    for i, p in enumerate(proposals):
        by_frame.setdefault(int(p['frame']), []).append(i)
    sorted_frames = sorted(by_frame.keys())

    instance_table: List[Dict] = []
    proposal_to_instance: List[int] = [-1] * U

    for t in sorted_frames:
        frame_idxs = by_frame[t]

        # No existing instances yet → every proposal is new.
        if not instance_table:
            for pi in frame_idxs:
                instance_table.append(_init_instance_from_proposal(proposals[pi]))
                proposal_to_instance[pi] = len(instance_table) - 1
                instance_table[-1]['member_proposals'].append(pi)
            continue

        # Score matrix [current, existing].
        n_cur = len(frame_idxs)
        n_inst = len(instance_table)
        score_mat = np.zeros((n_cur, n_inst), dtype=np.float32)
        for r, pi in enumerate(frame_idxs):
            for c, inst in enumerate(instance_table):
                score_mat[r, c] = _proposal_pair_score_dict(
                    proposals[pi], inst,
                    w_desc=w_desc, w_vox=w_vox, w_geo=w_geo,
                    w_sem=w_sem, w_box=w_box,
                    sigma_dist=sigma_dist,
                )
        # Block matches below threshold.
        cost = -score_mat.copy()
        below = score_mat < match_threshold
        cost[below] = 1e6      # forbid matching

        row_ind, col_ind = linear_sum_assignment(cost)
        # Sweep matches; below-threshold (cost==1e6) → treat as unmatched.
        matched_cur: Set[int] = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= 1e5:
                continue          # not a real match
            pi = frame_idxs[r]
            _update_instance_with_proposal(instance_table[c], proposals[pi])
            instance_table[c]['member_proposals'].append(pi)
            proposal_to_instance[pi] = c
            matched_cur.add(r)

        # Unmatched current proposals become new instances.
        for r, pi in enumerate(frame_idxs):
            if r in matched_cur:
                continue
            instance_table.append(_init_instance_from_proposal(proposals[pi]))
            instance_table[-1]['member_proposals'].append(pi)
            proposal_to_instance[pi] = len(instance_table) - 1

    return proposal_to_instance


def _build_score_matrix(
    proposals, w_desc, w_vox, w_geo, w_sem, w_box, sigma_dist,
) -> torch.Tensor:
    """Build U×U symmetric score matrix (used by legacy union-find path)."""
    U = len(proposals)
    if U == 0:
        return torch.zeros(0, 0)
    score = torch.zeros(U, U)
    for i in range(U):
        for j in range(i + 1, U):
            s = _proposal_pair_score_dict(
                proposals[i], proposals[j],
                w_desc=w_desc, w_vox=w_vox, w_geo=w_geo,
                w_sem=w_sem, w_box=w_box, sigma_dist=sigma_dist,
            )
            score[i, j] = s
            score[j, i] = s
    score.fill_diagonal_(-1.0)
    return score


def match_proposals(
    proposals:  List[Dict],
    w_desc:     float = 0.5,
    w_vox:      float = 0.3,
    w_geo:      float = 0.2,
    sigma_dist: float = 0.5,
    match_threshold: float = 0.35,
    # New optional cues (default off):
    w_sem:      float = 0.0,
    w_box:      float = 0.0,
    method:     str   = 'sequential_hungarian',     # or 'union_find' (legacy)
) -> List[int]:
    """Group proposals into cross-frame instance clusters.

    Two algorithms are available via `method`:
      - 'sequential_hungarian' (default, **recommended**):
            Mirror of the SLAM online merger.  Process proposals frame-by-frame
            and bipartite-match against an accumulated instance table.  Avoids
            the transitive-chain collapse of union-find on noisy scores.
      - 'union_find' (legacy):
            Build U×U score matrix and union all pairs above threshold.
            Vulnerable to chain explosion when descriptors are weak.

    Returns: list of U cluster IDs (each proposal → global instance ID).
    """
    U = len(proposals)
    if U == 0:
        return []
    if U == 1:
        return [0]

    if method == 'sequential_hungarian':
        return match_proposals_sequential_hungarian(
            proposals,
            w_desc=w_desc, w_vox=w_vox, w_geo=w_geo,
            w_sem=w_sem, w_box=w_box,
            sigma_dist=sigma_dist,
            match_threshold=match_threshold,
        )

    # ── Legacy union-find path ───────────────────────────────────────────
    score = _build_score_matrix(proposals, w_desc, w_vox, w_geo, w_sem, w_box,
                                  sigma_dist)
    return union_find_merge(score, threshold=match_threshold)
