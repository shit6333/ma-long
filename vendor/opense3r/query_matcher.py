"""
opense3r/query_matcher.py
==========================
Point-sampled Hungarian matcher for the query-based instance head.

For each frame independently, build an N×M cost matrix between N predicted
masks and M ground-truth instance masks, then solve the linear-sum
assignment problem.  To avoid O(NM·HW) cost (~tens of GB at training res),
we follow Mask2Former and compute mask costs on K randomly sampled pixel
locations (default K=12544) shared between all (pred, GT) pairs.

The matcher is pure inference (no_grad) — it only produces the
assignment; the actual loss is computed downstream at full resolution.

Cost formulation (per frame):
    cost(i,j) = λ_bce · BCE_sampled(pred_i, gt_j)
              + λ_dice · Dice_sampled(pred_i, gt_j)
              - λ_obj · log σ(pred_obj_i)            # objectness bias

The objectness term encourages slots with high objectness score to be
preferred for matching (matched slots should be confident).

References:
    MaskFormer (Cheng et al. 2021)  for the framework
    Mask2Former (Cheng et al. 2022) for the point-sampling trick
"""

from __future__ import annotations
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Point sampling utilities
# ─────────────────────────────────────────────────────────────────────────────

def sample_point_coords(num_points: int, H: int, W: int,
                          device, generator: Optional[torch.Generator] = None
                          ) -> torch.Tensor:
    """Uniform-random pixel coordinates (continuous float in [0,1]) for grid_sample.

    Returns: [num_points, 2]  (x, y) in [0, 1] — convention matches Mask2Former
             (we'll convert to [-1, 1] for grid_sample at call time).
    """
    return torch.rand(num_points, 2, device=device, generator=generator)


def sample_mask_at_points(mask: torch.Tensor, points_xy01: torch.Tensor
                          ) -> torch.Tensor:
    """Sample mask values at floating-point coordinates with bilinear interp.

    Args:
        mask:      [N, H, W] (or [N, 1, H, W])  float mask values (logits or probs)
        points_xy01: [K, 2]  (x, y) in [0, 1]

    Returns: [N, K]  sampled values.
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)                    # [N, 1, H, W]
    N = mask.shape[0]
    K = points_xy01.shape[0]
    # grid_sample expects [N, K, 1, 2] in [-1, 1]
    grid = (points_xy01 * 2 - 1).unsqueeze(0).unsqueeze(2).expand(N, K, 1, 2)
    out = F.grid_sample(mask, grid, mode='bilinear', align_corners=False)
    return out.squeeze(-1).squeeze(1)                # [N, K]


def batch_dice_cost(pred_probs: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """All-pairs dice cost.  Lower = better match.

    Args:
        pred_probs: [N, K]  sigmoid'd predicted probabilities at sampled points
        gt:         [M, K]  binary GT at the same sampled points

    Returns: [N, M] cost matrix.
    """
    numerator   = 2 * pred_probs @ gt.t()                                  # [N, M]
    denominator = pred_probs.sum(-1, keepdim=True) + gt.sum(-1)[None, :]    # [N, M]
    cost = 1 - (numerator + 1) / (denominator + 1)
    return cost


def batch_sigmoid_bce_cost(pred_logits: torch.Tensor, gt: torch.Tensor
                            ) -> torch.Tensor:
    """All-pairs sigmoid BCE cost.  Lower = better.

    Computed pairwise via the identity:
       BCE(p, y) = (1-y) * log(1+e^p) + y * log(1+e^-p)
                 = log(1+e^p) - y*p
       For pairs: pos cost = log(1+e^-p);  neg cost = log(1+e^p)
    Sum across the K sampled points; lower means pred matches gt better.

    Args:
        pred_logits: [N, K]
        gt:          [M, K]  binary 0/1

    Returns: [N, M] cost matrix, normalized by K.
    """
    K = pred_logits.shape[1]
    # Positive cost: -log σ(p)        (treating gt position as label=1)
    pos = F.binary_cross_entropy_with_logits(
        pred_logits, torch.ones_like(pred_logits), reduction='none',
    )                                                                       # [N, K]
    # Negative cost: -log(1-σ(p))     (label=0)
    neg = F.binary_cross_entropy_with_logits(
        pred_logits, torch.zeros_like(pred_logits), reduction='none',
    )                                                                       # [N, K]
    # All-pairs: cost[n,m] = sum_k gt[m,k]*pos[n,k] + (1-gt[m,k])*neg[n,k]
    cost = pos @ gt.t() + neg @ (1 - gt).t()                                # [N, M]
    return cost / K


# ─────────────────────────────────────────────────────────────────────────────
# Hungarian matcher
# ─────────────────────────────────────────────────────────────────────────────

class HungarianMatcherPointSampled:
    """Point-sampled per-frame Hungarian matcher.

    Stateless (callable).  Not an `nn.Module` because it has no parameters.

    Args:
        cost_bce:   weight on point-sampled BCE cost
        cost_dice:  weight on point-sampled Dice cost
        cost_obj:   weight on -log σ(objectness) term (encourages high-obj slots to match)
        num_points: K sampled pixels per frame
    """

    def __init__(
        self,
        cost_bce:   float = 2.0,
        cost_dice:  float = 5.0,
        cost_obj:   float = 1.0,
        num_points: int   = 12544,
    ):
        if not _HAS_SCIPY:
            raise ImportError(
                "HungarianMatcherPointSampled needs scipy.optimize.linear_sum_assignment. "
                "Install with: pip install scipy"
            )
        self.cost_bce   = cost_bce
        self.cost_dice  = cost_dice
        self.cost_obj   = cost_obj
        self.num_points = num_points

    @torch.no_grad()
    def __call__(
        self,
        mask_logits: torch.Tensor,                # [B, T, N, H, W]
        objectness:  torch.Tensor,                # [B, T, N]
        gt_masks:    List[List[torch.Tensor]],     # B × T list of [M_bt, H, W] binary
        **kwargs,                                  # ignores query_masks (sparse-only)
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run matching per frame.

        Returns: B × T list of (pred_idx, gt_idx) torch.long tensors of shape
                 [min(N, M_bt)].  Where M_bt == 0 (no GT in that frame), both
                 indices are empty tensors.
        """
        B, T, N, H, W = mask_logits.shape
        assert len(gt_masks) == B
        device = mask_logits.device

        # Sample K points once per frame (could share across frames; using
        # per-frame keeps it simple and avoids data race when batching).
        all_matches: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []

        for b in range(B):
            row_for_b: List[Tuple[torch.Tensor, torch.Tensor]] = []
            assert len(gt_masks[b]) == T, (
                f"gt_masks[{b}] has {len(gt_masks[b])} frames, expected {T}"
            )

            for t in range(T):
                gt_bt = gt_masks[b][t]                                # [M_bt, H, W]
                M_bt  = gt_bt.shape[0] if gt_bt is not None else 0

                if M_bt == 0:
                    row_for_b.append((
                        torch.empty(0, dtype=torch.long, device=device),
                        torch.empty(0, dtype=torch.long, device=device),
                    ))
                    continue

                # Sample K points
                pts = sample_point_coords(self.num_points, H, W, device=device)
                pred_at_pts = sample_mask_at_points(
                    mask_logits[b, t].float(), pts,                  # [N, K]
                )
                gt_at_pts = sample_mask_at_points(
                    gt_bt.float(), pts,                                # [M, K]
                ).clamp(0.0, 1.0)
                # Bilinear of a 0/1 mask can yield 0.5 at boundary — binarise.
                gt_at_pts = (gt_at_pts > 0.5).float()

                # Costs
                pred_probs = pred_at_pts.sigmoid()
                C_bce  = batch_sigmoid_bce_cost(pred_at_pts, gt_at_pts)            # [N, M]
                C_dice = batch_dice_cost      (pred_probs,  gt_at_pts)             # [N, M]
                C_obj  = (-F.logsigmoid(objectness[b, t])).unsqueeze(1).expand_as(C_bce)
                # cost_obj is the same value across columns (M dim) — only used to
                # bias the matcher toward high-objectness slots.

                C = (
                    self.cost_bce  * C_bce
                    + self.cost_dice * C_dice
                    + self.cost_obj  * C_obj
                )                                                                   # [N, M]
                C_np = C.cpu().numpy()
                # Handle NaN / inf defensively
                C_np = np.nan_to_num(C_np, nan=1e6, posinf=1e6, neginf=-1e6)

                pred_i, gt_j = linear_sum_assignment(C_np)
                row_for_b.append((
                    torch.as_tensor(pred_i, dtype=torch.long, device=device),
                    torch.as_tensor(gt_j,   dtype=torch.long, device=device),
                ))
            all_matches.append(row_for_b)
        return all_matches


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build per-frame GT mask list from instance_mask tensor
# ─────────────────────────────────────────────────────────────────────────────

class SparseMatcherPointSampled:
    """ESAM-style constrained matcher for FastSAM-region-seeded queries.

    Port of thirdparty/ESAM/oneformer3d/instance_criterion.py::SparseMatcher to
    our 2D point-sampled setting.  Each query is seeded from one FastSAM region;
    `query_masks[b][t]` ([M_gt, N]) marks which (GT, query) pairs are allowed —
    a query may only match the GT instance its source region overlaps (>50%).

    Unlike Hungarian (one-to-one), this is per-GT top-k (default 1, ESAM): each
    GT is matched to its `topk` lowest-cost ALLOWED queries.  Disallowed pairs
    get +inf cost.  Returns the SAME (pred_idx, gt_idx) format as the Hungarian
    matcher, so the downstream loss is unchanged.

    Args:
        cost_bce / cost_dice / cost_obj / num_points: same cost as Hungarian.
        topk: queries matched per GT (ESAM default 1).
    """

    def __init__(self, cost_bce: float = 2.0, cost_dice: float = 5.0,
                 cost_obj: float = 1.0, num_points: int = 12544, topk: int = 1):
        self.cost_bce = cost_bce
        self.cost_dice = cost_dice
        self.cost_obj = cost_obj
        self.num_points = num_points
        self.topk = int(topk)
        self.inf = 1e8

    @torch.no_grad()
    def __call__(
        self,
        mask_logits: torch.Tensor,                # [B, T, N, H, W]
        objectness:  torch.Tensor,                # [B, T, N]
        gt_masks:    List[List[torch.Tensor]],     # B × T list of [M_bt, H, W]
        query_masks: Optional[List[List[torch.Tensor]]] = None,  # B×T [M_bt, N] bool
    ) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        assert query_masks is not None, "SparseMatcher requires query_masks"
        B, T, N, H, W = mask_logits.shape
        device = mask_logits.device
        all_matches: List[List[Tuple[torch.Tensor, torch.Tensor]]] = []

        for b in range(B):
            row_for_b: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for t in range(T):
                gt_bt = gt_masks[b][t]
                M_bt = gt_bt.shape[0] if gt_bt is not None else 0
                if M_bt == 0:
                    row_for_b.append((
                        torch.empty(0, dtype=torch.long, device=device),
                        torch.empty(0, dtype=torch.long, device=device),
                    ))
                    continue

                pts = sample_point_coords(self.num_points, H, W, device=device)
                pred_at_pts = sample_mask_at_points(mask_logits[b, t].float(), pts)   # [N,K]
                gt_at_pts = sample_mask_at_points(gt_bt.float(), pts).clamp(0, 1)
                gt_at_pts = (gt_at_pts > 0.5).float()                                 # [M,K]

                pred_probs = pred_at_pts.sigmoid()
                C_bce  = batch_sigmoid_bce_cost(pred_at_pts, gt_at_pts)               # [N,M]
                C_dice = batch_dice_cost(pred_probs, gt_at_pts)                       # [N,M]
                C_obj  = (-F.logsigmoid(objectness[b, t])).unsqueeze(1).expand_as(C_bce)
                C = self.cost_bce * C_bce + self.cost_dice * C_dice + self.cost_obj * C_obj  # [N,M]

                # Constrain to allowed (GT, query) pairs.  query_masks[b][t]:[M,N].
                allowed = query_masks[b][t].to(device).bool().t()                    # [N,M]
                C = torch.where(allowed, C, C.new_full((), self.inf))
                # Per-GT top-k smallest-cost queries (ESAM: topk over the N dim).
                k = min(self.topk + 1, N)
                thresh = torch.topk(C, k, dim=0, largest=False).values[-1:, :]        # [1,M]
                ids = torch.argwhere(C < thresh)                                      # [P,2] (q, gt)
                # Drop matches that landed on disallowed (inf) pairs (can happen
                # when a GT has fewer than k allowed queries → thresh is inf).
                keep = allowed[ids[:, 0], ids[:, 1]]
                ids = ids[keep]
                row_for_b.append((ids[:, 0].long(), ids[:, 1].long()))
            all_matches.append(row_for_b)
        return all_matches


@torch.no_grad()
def build_query_masks(
    region_id:    torch.Tensor,                  # [B, T, H, W] long, -1 = none
    gt_masks_btM: List[List[torch.Tensor]],       # B×T list of [M_bt, H, W]
    max_regions:  int,
) -> List[List[torch.Tensor]]:
    """Per-frame allowed (GT, query) matrix from FastSAM region ↔ GT overlap.

    A query (= region id r) is allowed to match GT g if >50% of region r's
    pixels fall inside g.  Query slot index == region id, so slots with no
    region (padding) get an all-False column → never matched (→ objectness 0).

    Returns: B×T list of [M_bt, max_regions] bool.
    """
    B, T, H, W = region_id.shape
    device = region_id.device
    out: List[List[torch.Tensor]] = []
    for b in range(B):
        row: List[torch.Tensor] = []
        for t in range(T):
            gt = gt_masks_btM[b][t]                       # [M, H, W] float
            M = gt.shape[0] if gt is not None else 0
            if M == 0:
                row.append(torch.zeros(0, max_regions, dtype=torch.bool, device=device))
                continue
            rid = region_id[b, t].reshape(-1)             # [HW]
            valid = rid >= 0
            onehot = torch.zeros(H * W, max_regions, device=device)
            if valid.any():
                idx = rid[valid].clamp_max(max_regions - 1)
                onehot[valid.nonzero(as_tuple=True)[0], idx] = 1.0    # [HW, R]
            gt_flat = (gt.reshape(M, -1) > 0.5).float()   # [M, HW]
            inter = gt_flat @ onehot                       # [M, R]  pixels of region∩GT
            region_size = onehot.sum(0).clamp_min(1.0)     # [R]
            frac = inter / region_size.unsqueeze(0)        # [M, R]
            row.append(frac > 0.5)
        out.append(row)
    return out


def build_per_frame_gt_masks(
    instance_mask: torch.Tensor,           # [B, T, H, W]  long, 0 = unlabeled
    min_pixels: int = 10,
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
    """Split a dense instance_mask into per-frame lists of binary masks and IDs.

    Args:
        instance_mask: [B, T, H, W] int tensor.  0 is treated as background and
                       dropped (no GT mask for class 0).  Positive values are
                       scene-consistent instance IDs.
        min_pixels:    Drop GT instances smaller than this (e.g. partial occlusion
                       remnants that hurt training).

    Returns:
        gt_masks: B × T list of [M_bt, H, W] binary tensors (float)
        gt_ids:   B × T list of [M_bt]      long tensors
    """
    B, T, H, W = instance_mask.shape
    device = instance_mask.device
    masks_all: List[List[torch.Tensor]] = []
    ids_all:   List[List[torch.Tensor]] = []

    for b in range(B):
        masks_b: List[torch.Tensor] = []
        ids_b:   List[torch.Tensor] = []
        for t in range(T):
            inst = instance_mask[b, t]                  # [H, W]
            unique_ids = inst.unique()
            unique_ids = unique_ids[unique_ids > 0]
            kept_masks: List[torch.Tensor] = []
            kept_ids:   List[int]          = []
            for iid in unique_ids.tolist():
                m = (inst == iid)
                if int(m.sum()) >= min_pixels:
                    kept_masks.append(m.float())
                    kept_ids.append(iid)
            if kept_masks:
                masks_b.append(torch.stack(kept_masks, dim=0))   # [M, H, W]
                ids_b.append(torch.as_tensor(kept_ids, dtype=torch.long, device=device))
            else:
                masks_b.append(torch.zeros(0, H, W, device=device, dtype=torch.float))
                ids_b.append(torch.zeros(0, dtype=torch.long, device=device))
        masks_all.append(masks_b)
        ids_all.append(ids_b)
    return masks_all, ids_all
