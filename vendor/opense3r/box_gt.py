"""
opense3r/box_gt.py
==================
Full-object GT bbox generation for the query-head box supervision.

Algorithm (mirrors scripts/toy_gt_bbox.py but in torch / batched):

  1. Unproject gt_depth + gt_intrinsic + gt_c2w → per-pixel world points.
  2. For each instance ID in the chunk, gather world points across all
     frames where the mask matches and depth is valid.
  3. IQR-fence outlier filter on the aggregated points → robust world
     AABB per instance.
  4. For each (frame, inst) with ≥ `min_pix` visible pixels: take the
     8 world-AABB corners → transform to camera frame via inverse(c2w)
     → camera AABB (bmin, bmax).
  5. Normalize the camera AABB by `box_scale[b]` (per-batch depth
     median) so the GT lives in the same canonical scale as the head's
     output.

Returned schema for loss-side consumption:
    gt_boxes_canonical: dict[(b, t, inst_id) → torch.Tensor [6]]
                        in (cx, cy, cz, w, h, d) format, canonical scale,
                        camera frame.  Loss is computed only for pairs
                        present in this dict (= the "unseen-frame mask"
                        is naturally encoded via dict membership).

Notes on scale (critical for canonical DA3 training):
    The head learns a SCENE-NORMALIZED box.  At training, the GT is
    divided by `median(gt_depth)` — the clean, scene-specific metric
    scale.  This makes the target unitless (extent / scene-scale) and
    invariant to whether the model sees canonical or metric depth from
    DA3.  Without this normalization, GT would be in absolute metric
    units while DA3's canonical features carry no absolute-scale cue,
    so the head would have to learn per-scene scale-up — slow / no
    convergence (the symptom we saw on first attempt with `box_scale`
    = median(da3_depth) as the divisor).

    At inference, the SLAM side multiplies the head output by
    `median(da3_depth)`:
      - Metric DA3: median(da3) ≈ median(gt_depth used at train) →
        pred × scale ≈ true metric box.  ✓
      - Canonical DA3: median(da3) ≈ 1 (per-scene normalization built
        into the backbone), so pred × 1 ≈ scene-normalized box. All
        instances live in the same scale → matching works internally
        but absolute metric is not recovered.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def unproject_to_world(
    depth:     torch.Tensor,    # [B, T, H, W]
    intrinsic: torch.Tensor,    # [B, T, 3, 3]
    c2w:       torch.Tensor,    # [B, T, 4, 4]
) -> torch.Tensor:
    """Unproject pixel grid to world-coord points.  Returns [B, T, H, W, 3]."""
    B, T, H, W = depth.shape
    device, dtype = depth.device, depth.dtype

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij',
    )
    ones = torch.ones_like(xx)
    pix = torch.stack([xx, yy, ones], dim=-1)                  # [H, W, 3]

    # K_inv per (b, t): [B, T, 3, 3]
    K_inv = torch.linalg.inv(intrinsic.to(dtype=torch.float32)).to(dtype)
    # rays_cam = K_inv @ pix_h
    rays = torch.einsum('btij,hwj->bthwi', K_inv, pix)          # [B, T, H, W, 3]
    cam  = rays * depth.unsqueeze(-1)                           # [B, T, H, W, 3]

    R = c2w[..., :3, :3]                                        # [B, T, 3, 3]
    t = c2w[..., :3, 3]                                         # [B, T, 3]
    # world = R @ cam + t
    world = torch.einsum('btij,bthwj->bthwi', R, cam) + t[:, :, None, None, :]
    return world


def aabb_corners(bmin: torch.Tensor, bmax: torch.Tensor) -> torch.Tensor:
    """[3], [3] → [8, 3] corners.  Order matches scripts/toy_gt_bbox.py."""
    return torch.stack([
        torch.stack([bmin[0], bmin[1], bmin[2]]),
        torch.stack([bmax[0], bmin[1], bmin[2]]),
        torch.stack([bmax[0], bmax[1], bmin[2]]),
        torch.stack([bmin[0], bmax[1], bmin[2]]),
        torch.stack([bmin[0], bmin[1], bmax[2]]),
        torch.stack([bmax[0], bmin[1], bmax[2]]),
        torch.stack([bmax[0], bmax[1], bmax[2]]),
        torch.stack([bmin[0], bmax[1], bmax[2]]),
    ], dim=0)                                                   # [8, 3]


def iqr_filter(pts: torch.Tensor, mult: float, min_pts: int = 50) -> torch.Tensor:
    """Per-axis Tukey-fence filter on [P, 3] pts.  Returns kept rows."""
    if mult <= 0 or pts.shape[0] < min_pts:
        return pts
    q1 = pts.quantile(0.25, dim=0)
    q3 = pts.quantile(0.75, dim=0)
    iqr = q3 - q1
    lo, hi = q1 - mult * iqr, q3 + mult * iqr
    keep = ((pts >= lo) & (pts <= hi)).all(dim=-1)
    return pts[keep]


# ─────────────────────────────────────────────────────────────────────────────
# Rotation 6d helpers (Zhou et al. CVPR'19, continuous SO(3) representation)
# ─────────────────────────────────────────────────────────────────────────────

def matrix_to_rot_6d(R: torch.Tensor) -> torch.Tensor:
    """[..., 3, 3] SO(3) → [..., 6]  (first two columns concatenated).

    Layout: d6 = [R[..., :, 0], R[..., :, 1]] (column-stacked, not
    row-stacked — matches `rot_6d_to_matrix` which reads d6[..., :3] as
    the first column).
    """
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def rot_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """[..., 6] → [..., 3, 3] SO(3) via Gram-Schmidt."""
    a1 = d6[..., :3]
    a2 = d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-6)
    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2 - proj, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)              # [..., 3, 3]


# ─────────────────────────────────────────────────────────────────────────────
# PCA OBB fit
# ─────────────────────────────────────────────────────────────────────────────

def fit_obb_pca(pts: torch.Tensor) -> Dict[str, torch.Tensor]:
    """PCA-based oriented bbox fit on [N, 3] points.

    Returns:
        {
          'center':    [3]     world-frame OBB centre
          'R':         [3, 3]  rotation (SO(3); columns = principal axes,
                               sorted by descending eigenvalue)
          'size_half': [3]     half-extents along R's columns
        }

    For degenerate inputs (< 3 points), returns a zero-size identity OBB.
    """
    device, dtype = pts.device, pts.dtype
    N = pts.shape[0]
    if N < 3:
        return {
            'center':    (pts.mean(0) if N > 0 else
                          torch.zeros(3, device=device, dtype=dtype)),
            'R':         torch.eye(3, device=device, dtype=dtype),
            'size_half': torch.zeros(3, device=device, dtype=dtype),
        }

    c = pts.mean(0)                                       # [3]
    pts_c = (pts - c).to(torch.float32)                   # [N, 3]
    cov = (pts_c.T @ pts_c) / float(N)                    # [3, 3]
    eigvals, eigvecs = torch.linalg.eigh(cov.double())    # ascending
    # Sort descending → principal axis first.
    idx = torch.argsort(eigvals, descending=True)
    R = eigvecs[:, idx].to(dtype)                          # [3, 3]
    # Force det(R) = +1 (SO(3) not O(3))
    if torch.det(R) < 0:
        R = R.clone()
        R[:, -1] = -R[:, -1]

    pts_rot = pts_c @ R                                    # [N, 3] in rotated frame
    bmin = pts_rot.min(0).values
    bmax = pts_rot.max(0).values
    size_half = ((bmax - bmin) * 0.5).clamp(min=1e-3)
    c_offset_rot = (bmin + bmax) * 0.5
    center = c + R @ c_offset_rot                          # [3]

    return {
        'center':    center.to(dtype),
        'R':         R,
        'size_half': size_half.to(dtype),
    }


@torch.no_grad()
def compute_aabb_prior_weighted_std(
    soft_mask: torch.Tensor,    # [..., H, W]  sigmoid(mask_logits), in [0, 1]
    world_pts: torch.Tensor,    # [..., H, W, 3]
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weighted mean + 2×weighted-std AABB on mask-pooled world points.

    Robust alternative to weighted min/max — outliers (noisy mask pixels at
    surface boundaries) don't blow up the size.  Size_half = 2σ covers ~95%
    of the mass for a Gaussian-ish distribution.

    Returns:
        center    [..., 3]   weighted centroid (anchor)
        size_half [..., 3]   per-axis half-extent in same units as world_pts
    """
    w = soft_mask.unsqueeze(-1)                                # [..., H, W, 1]
    w_sum = soft_mask.sum(dim=(-1, -2)).clamp(min=eps)         # [...]
    # E[X]
    center = (w * world_pts).sum(dim=(-2, -3)) / w_sum.unsqueeze(-1)         # [..., 3]
    # Var = E[X²] - E[X]²   (memory-efficient; never materializes per-pixel diff)
    sq_sum = (w * world_pts.pow(2)).sum(dim=(-2, -3)) / w_sum.unsqueeze(-1)  # [..., 3]
    var = (sq_sum - center.pow(2)).clamp(min=eps * eps)                       # [..., 3]
    size_half = var.sqrt() * 2.0                                              # 2σ
    return center, size_half


@torch.no_grad()
def build_gt_world_aabbs(
    world_pts:     torch.Tensor,    # [B, T, H, W, 3]   world-frame points
    instance_mask: torch.Tensor,    # [B, T, H, W]      long, 0 = unlabeled
    valid_mask:    Optional[torch.Tensor] = None,
    iqr_mult:      float = 1.5,
    min_pix_per_inst: int = 30,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Per-(b, inst_id) world-frame AABB GT (full extent across chunk).

    Aggregates all valid world points belonging to each instance across all
    T frames, IQR-filters outliers, then takes axis-aligned min/max.

    Returns: dict[(b, inst_id) → [6] tensor]   format (cx, cy, cz, sx, sy, sz)
                                                  with FULL size (sx = bmax - bmin),
                                                  matching legacy AABB convention.
    """
    B, T, H, W, _ = world_pts.shape
    device, dtype = world_pts.device, world_pts.dtype
    if valid_mask is None:
        valid_mask = torch.ones((B, T, H, W), device=device, dtype=torch.bool)

    out: Dict[Tuple[int, int], torch.Tensor] = {}
    for b in range(B):
        ids = torch.unique(instance_mask[b])
        ids = ids[ids > 0]
        for iid in ids.tolist():
            mask_all = (instance_mask[b] == int(iid)) & valid_mask[b]
            if int(mask_all.sum().item()) < min_pix_per_inst:
                continue
            pts = world_pts[b][mask_all]
            finite = torch.isfinite(pts).all(dim=-1)
            pts = pts[finite]
            if pts.shape[0] < 3:
                continue
            pts = iqr_filter(pts, iqr_mult)
            if pts.shape[0] < 3:
                continue
            bmin = pts.min(dim=0).values
            bmax = pts.max(dim=0).values
            center = (bmin + bmax) * 0.5
            size   = (bmax - bmin).clamp(min=1e-3)
            out[(b, int(iid))] = torch.cat([center, size], dim=-1).to(dtype)
    return out


def build_growing_gt_aabbs(
    world_pts:     torch.Tensor,        # [B, T, H, W, 3]  NORMALIZE-scale, frame-0 anchored
    instance_mask: torch.Tensor,        # [B, T, H, W]     long, 0 = unlabeled
    valid_mask:    Optional[torch.Tensor] = None,
    iqr_mult:      float = 1.5,
    min_pix_per_inst: int = 30,
    bg_ids:        Optional[list] = None,   # len-B list of iterables of bg inst ids to skip
    full_boxes:    Optional[Dict[Tuple[int, int], torch.Tensor]] = None,  # (b,iid)->[6] amodal clamp
    snap_cover:    Optional[float] = None,  # vol(box)/vol(full) > snap_cover → snap to full box
) -> Dict[Tuple[int, int, int], torch.Tensor]:
    """Per-(b, t, inst_id) CAUSAL *growing* world-frame AABB GT.

    The box at frame t is the AABB of all instance points VISIBLE IN FRAMES 0..t
    of this chunk (IQR-filtered) — i.e. it grows as more of the object is seen,
    matching the DetAny4D "accumulated visible b-box" target and the model's own
    causal attention order. Same space/convention as `build_gt_world_aabbs`
    (NORMALIZE scale, frame-0 anchored), but keyed per-frame.

    Options:
      bg_ids:     skip these instance ids (wall/floor/ceiling) → no box GT.
      full_boxes: pre-generated amodal full box (same space) — clamp the growing
                  box to it (depth-noise can push the visible AABB past the true
                  extent); also enables snap.
      snap_cover: once the (clamped) box covers > snap_cover of the full box
                  volume, snap exactly to the full amodal box.

    Returns: dict[(b, t, inst_id) -> Tensor[6]]  (cx,cy,cz,sx,sy,sz), full size.
    """
    B, T, H, W, _ = world_pts.shape
    device, dtype = world_pts.device, world_pts.dtype
    if valid_mask is None:
        valid_mask = torch.ones((B, T, H, W), device=device, dtype=torch.bool)

    out: Dict[Tuple[int, int, int], torch.Tensor] = {}
    for b in range(B):
        skip = set(int(x) for x in bg_ids[b]) if (bg_ids is not None and b < len(bg_ids)) else set()
        ids = torch.unique(instance_mask[b])
        ids = [int(i) for i in ids[ids > 0].tolist() if int(i) not in skip]
        acc: Dict[int, torch.Tensor] = {}                       # iid -> [P,3] accumulated pts
        for t in range(T):
            mt = instance_mask[b, t]
            vt = valid_mask[b, t]
            for iid in ids:
                m = (mt == iid) & vt
                if m.any():
                    p = world_pts[b, t][m]
                    p = p[torch.isfinite(p).all(dim=-1)]
                    if p.shape[0] > 0:
                        acc[iid] = p if iid not in acc else torch.cat([acc[iid], p], dim=0)
                pts = acc.get(iid, None)
                if pts is None or pts.shape[0] < min_pix_per_inst:
                    continue
                ptsf = iqr_filter(pts, iqr_mult)
                if ptsf.shape[0] < 3:
                    continue
                bmin = ptsf.min(dim=0).values
                bmax = ptsf.max(dim=0).values
                if full_boxes is not None and (b, iid) in full_boxes:
                    fb = full_boxes[(b, iid)].to(device=bmin.device, dtype=bmin.dtype)
                    fmin, fmax = fb[:3] - fb[3:6] * 0.5, fb[:3] + fb[3:6] * 0.5
                    bmin = torch.maximum(bmin, fmin)            # clamp growing box ⊆ full
                    bmax = torch.minimum(bmax, fmax)
                    if snap_cover is not None:
                        vol  = (bmax - bmin).clamp(min=1e-6).prod()
                        fvol = (fmax - fmin).clamp(min=1e-6).prod()
                        if float(vol / fvol) > snap_cover:
                            bmin, bmax = fmin, fmax
                center = (bmin + bmax) * 0.5
                size   = (bmax - bmin).clamp(min=1e-3)
                out[(b, t, iid)] = torch.cat([center, size], dim=-1).to(dtype)
    return out


def obb_to_corners(box12: torch.Tensor) -> torch.Tensor:
    """Convert OBB 12-d → 8 world-frame corner positions.

    box12: [..., 12]   (cx,cy,cz, sx,sy,sz, r1..r6)   -- size_half!
    Returns: [..., 8, 3]   corners in the SAME frame as the input box.

    Corner ordering matches `aabb_corners` (and toy_gt_bbox.py):
        0: (-,-,-), 1: (+,-,-), 2: (+,+,-), 3: (-,+,-),
        4: (-,-,+), 5: (+,-,+), 6: (+,+,+), 7: (-,+,+)
    """
    center    = box12[..., 0:3]                                    # [..., 3]
    size_half = box12[..., 3:6]                                    # [..., 3]
    R         = rot_6d_to_matrix(box12[..., 6:12])                  # [..., 3, 3]

    device, dtype = box12.device, box12.dtype
    unit = torch.tensor([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1],
    ], device=device, dtype=dtype)                                  # [8, 3]

    # local[k] = unit[k] * size_half    (broadcast over leading dims)
    local = unit * size_half.unsqueeze(-2)                          # [..., 8, 3]
    # world[k] = R @ local[k] = local[k] @ R^T   (row-vector convention)
    rotated = torch.einsum('...ij,...kj->...ki', R, local)          # [..., 8, 3]
    return rotated + center.unsqueeze(-2)                           # [..., 8, 3]


# V_4 corner permutations (proper-rotation symmetries of a box).
# Generated by 180° rotations around the box's own axes:
#   I              → [0,1,2,3,4,5,6,7]
#   R_x(π)  flips y,z → corner k maps to k' where unit[k'] = (1,-1,-1) * unit[k]
#   R_y(π)  flips x,z
#   R_z(π)  flips x,y
_V4_CORNER_PERMS = (
    (0, 1, 2, 3, 4, 5, 6, 7),
    (7, 6, 5, 4, 3, 2, 1, 0),   # R_x(π)
    (5, 4, 7, 6, 1, 0, 3, 2),   # R_y(π)
    (2, 3, 0, 1, 6, 7, 4, 5),   # R_z(π)
)


def obb_corner_match_loss(
    pred12: torch.Tensor,    # [P, 12]  in WORLD frame, metric scale
    gt12:   torch.Tensor,    # [P, 12]  same frame, metric scale
) -> torch.Tensor:
    """Permutation-aware SmoothL1 over 8 corners (V_4 symmetry-aware).

    For each matched pair, computes 4 candidate losses corresponding to the
    4 proper-rotation symmetries of a box (identity + 3 axis-flip pairs),
    then takes the minimum.  This makes the loss invariant to the PCA sign
    ambiguity / axis-permutation in the GT R, so the network can converge
    to ANY valid representation of the right physical box.

    Scalar output (mean over pairs).  Zero pairs → zero scalar.
    """
    if pred12.shape[0] == 0:
        return pred12.new_zeros(())
    pred_c = obb_to_corners(pred12)                                  # [P, 8, 3]
    gt_c   = obb_to_corners(gt12)                                    # [P, 8, 3]

    candidates = []
    for perm in _V4_CORNER_PERMS:
        idx   = torch.tensor(perm, device=gt_c.device, dtype=torch.long)
        gt_p  = gt_c.index_select(dim=1, index=idx)                  # [P, 8, 3]
        # per-pair mean over corners & xyz
        l_p   = torch.nn.functional.smooth_l1_loss(
            pred_c, gt_p, reduction='none').mean(dim=(1, 2))         # [P]
        candidates.append(l_p)
    stacked = torch.stack(candidates, dim=0)                         # [4, P]
    return stacked.min(dim=0).values.mean()


def compute_anchor_world(
    soft_mask:  torch.Tensor,    # [B, T, N, H, W]   sigmoid probs ∈ [0,1]
    world_pts:  torch.Tensor,    # [B, T, H, W, 3]
    eps:        float = 1e-3,
) -> torch.Tensor:
    """Soft-mask-weighted centroid of world points, per (b, t, n).

    Anchor lives in the SAME frame as world_pts (chunk-anchor at training).
    Returns: [B, T, N, 3].

    Mask weights sum < eps → returns the chunk's centroid (mean of all
    valid world_pts in that frame) as a safe fallback — avoids div-by-zero
    but downstream loss should also mask these out via gt-presence check.
    """
    B, T, N, H, W = soft_mask.shape
    # einsum: w[b,t,n,h,w] * p[b,t,h,w,c] → [b,t,n,c]
    weighted = torch.einsum('btnhw,bthwc->btnc', soft_mask, world_pts)   # [B,T,N,3]
    w_sum    = soft_mask.sum(dim=(-1, -2)).unsqueeze(-1).clamp(min=eps)  # [B,T,N,1]
    return weighted / w_sum                                              # [B,T,N,3]


def transform_obb_cam_to_world(
    box12_cam: torch.Tensor,    # [P, 12]  (cx,cy,cz, sx,sy,sz, r1..r6)
    box_scale: torch.Tensor,    # [P]      canonical → metric scalar per pair
    c2w:        torch.Tensor,   # [P, 4, 4] camera → world per pair
    anchor:     Optional[torch.Tensor] = None,   # [P, 3]  world-frame mask centroid
) -> torch.Tensor:
    """Transform a batch of camera-frame canonical-scale OBBs to world frame
    (metric).

    Two modes:
      anchor=None  (abs):     center_w = R_c2w @ (center_c * scale) + t_c2w
      anchor=given (anchor):  center_w = R_c2w @ (delta_c * scale) + anchor
                              (head's first 3 dims are interpreted as a
                               small DELTA from the soft-mask centroid)

    Size and rotation transform the same in both modes:
        size_w = size_c * scale                       # rotation-invariant
        R_w    = R_c2w @ R_c                           # box axes rotate

    Returns: [P, 12] in world frame, metric scale.
    """
    center_c = box12_cam[..., :3]
    size_h   = box12_cam[..., 3:6]
    rot6_c   = box12_cam[..., 6:12]

    s = box_scale.unsqueeze(-1).to(box12_cam.dtype)        # [P, 1]
    center_c = center_c * s
    size_h   = size_h   * s

    R_cam   = rot_6d_to_matrix(rot6_c)                     # [P, 3, 3]
    R_c2w   = c2w[..., :3, :3].to(box12_cam.dtype)         # [P, 3, 3]

    rotated_center = torch.einsum('pij,pj->pi', R_c2w, center_c)
    if anchor is None:
        t_c2w    = c2w[..., :3, 3].to(box12_cam.dtype)     # [P, 3]
        center_w = rotated_center + t_c2w
    else:
        center_w = rotated_center + anchor.to(box12_cam.dtype)

    R_w      = torch.einsum('pij,pjk->pik', R_c2w, R_cam)
    rot6_w   = matrix_to_rot_6d(R_w)

    return torch.cat([center_w, size_h, rot6_w], dim=-1)   # [P, 12]


def obb_dict_to_12d(obb: Dict[str, torch.Tensor]) -> torch.Tensor:
    """{center, R, size_half} → [12]  (cx,cy,cz, sx,sy,sz, r1..r6)."""
    return torch.cat([obb['center'], obb['size_half'],
                       matrix_to_rot_6d(obb['R'])], dim=-1)


def obb_12d_to_dict(box12: torch.Tensor) -> Dict[str, torch.Tensor]:
    """[..., 12] → {center, R, size_half}."""
    return {
        'center':    box12[..., :3],
        'size_half': box12[..., 3:6],
        'R':         rot_6d_to_matrix(box12[..., 6:12]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GT bbox builder
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def build_gt_world_obbs(
    world_pts:     torch.Tensor,    # [B, T, H, W, 3]   pre-computed world points
    instance_mask: torch.Tensor,    # [B, T, H, W]      long (0 = unlabeled)
    valid_mask:    Optional[torch.Tensor] = None,   # [B, T, H, W] bool
    iqr_mult:      float = 1.5,
    min_pix_per_inst: int = 30,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Per-(b, inst_id) world-frame OBB GT (PCA-fit on aggregated points).

    Returns:
        dict[(b, inst_id) → torch.Tensor [12]]   (cx, cy, cz, sx, sy, sz, r1..r6)
        All on `world_pts.device`, same dtype, in WORLD frame (the frame
        of `world_pts` — typically chunk-anchor frame at training).

    A (b, inst_id) pair is included iff:
        - The instance has ≥ `min_pix_per_inst` total visible pixels across
          the chunk (with valid_mask=True if provided).
        - The IQR-filtered aggregate has ≥ 3 surviving points.

    The OBB is the same for every frame in the chunk (world-side
    supervision) — no per-frame transform.  Callers do the loss in the
    world frame: transform pred (in same chunk-anchor frame) to world is a
    no-op; pred and GT both already share that frame.
    """
    B, T, H, W, _ = world_pts.shape
    device, dtype = world_pts.device, world_pts.dtype
    if valid_mask is None:
        valid_mask = torch.ones((B, T, H, W), device=device, dtype=torch.bool)

    out: Dict[Tuple[int, int], torch.Tensor] = {}

    for b in range(B):
        ids_in_b = torch.unique(instance_mask[b])
        ids_in_b = ids_in_b[ids_in_b > 0]
        if ids_in_b.numel() == 0:
            continue

        for iid in ids_in_b.tolist():
            # Aggregate all (b, t, h, w) world points belonging to this inst.
            mask_all = (instance_mask[b] == int(iid)) & valid_mask[b]   # [T, H, W]
            if int(mask_all.sum().item()) < min_pix_per_inst:
                continue
            pts = world_pts[b][mask_all]                                  # [P, 3]
            finite = torch.isfinite(pts).all(dim=-1)
            pts = pts[finite]
            if pts.shape[0] < 3:
                continue
            pts = iqr_filter(pts, iqr_mult)
            if pts.shape[0] < 3:
                continue

            obb = fit_obb_pca(pts)
            out[(b, int(iid))] = obb_dict_to_12d(obb).to(dtype)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Pre-generated AMODAL world box GT (full-scene extent, from gen_gt_world_boxes.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_pregen_boxes(paths) -> Dict[str, Dict[str, Dict[int, "torch.Tensor"]]]:
    """Load one or more pre-generated GT-box files (gen_gt_world_boxes.py output).

    `paths`: a single path, a comma-separated string, or a list of paths.  Each
    file is `{'boxes': {scene_id: {inst_id: Tensor[6]}}, 'meta': {'dataset': ...}}`
    in METRIC, dataset-world frame, FULL size (cx,cy,cz,sx,sy,sz).

    Returns: {dataset_name → {scene_id → {inst_id(int) → Tensor[6] cpu float32}}}.
    Files are keyed by their meta['dataset']; if absent, the filename stem is used.
    """
    import os
    if isinstance(paths, str):
        paths = [p for p in paths.split(',') if p]
    table: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {}
    for p in paths:
        blob = torch.load(p, map_location='cpu')
        boxes = blob['boxes'] if isinstance(blob, dict) and 'boxes' in blob else blob
        meta = blob.get('meta', {}) if isinstance(blob, dict) else {}
        ds = meta.get('dataset') or os.path.splitext(os.path.basename(p))[0].split('_')[0]
        dst = table.setdefault(ds, {})
        for scene_id, inst_map in boxes.items():
            dst[scene_id] = {int(k): v.float().cpu() for k, v in inst_map.items()}
    return table


@torch.no_grad()
def build_amodal_gt_aabbs(
    box_table:     Dict[str, Dict[str, Dict[int, torch.Tensor]]],
    dataset_names: list,            # len B  (e.g. ['scannetpp', ...])
    scene_ids:     list,            # len B
    instance_mask: torch.Tensor,    # [B, T, H, W]  long (0 = unlabeled)
    camera_pose:   torch.Tensor,    # [B, T, 4, 4]  dataset-world c2w (GT)
    box_scale:     torch.Tensor,    # [B]           per-chunk depth-median scale
    eps: float = 1e-6,
    anchor_mode: str = 'se3',       # MUST match build_gt_geometry_from_frames:
                                    #   'se3'       full SE3 → tilted frame-0 cam frame
                                    #   'translate' gravity-aligned (origin@cam0, no rot)
    # Drop structural / stuff-scale instances (walls, floors, ceilings) from
    # box supervision: their amodal full-scene box center can lie far outside
    # the chunk's view (a chunk sees one corner of a wall), making the target
    # unlearnable — DIoU saturates ~2 and the huge-size gradient inflates the
    # head's exp(size) for everything else.  ESAM never faces this because
    # ScanNet instance GT excludes wall/floor; ScanNetpp arrow masks include
    # them.  0 = no filter.
    max_dim_m: float = 3.0,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Look up pre-generated full-scene (amodal) metric world boxes and transform
    them into the SAME space the head/loss use: frame-0-anchored + NORMALIZE
    (÷box_scale).  Output matches `build_gt_world_aabbs`:
        dict[(b, inst_id) → Tensor[6]]   (cx,cy,cz,sx,sy,sz), full size.

    Anchoring mirrors `build_gt_geometry_from_frames` exactly:
        w2c0 = inv(camera_pose[b, 0]);  p_anchored = w2c0 @ p_world
    Only instances actually present in this chunk's instance_mask[b] are emitted
    (so they line up with the chunk's GT masks used by Hungarian matching).
    """
    B, T, H, W = instance_mask.shape
    device, dtype = instance_mask.device, camera_pose.dtype
    out: Dict[Tuple[int, int], torch.Tensor] = {}

    for b in range(B):
        ds = dataset_names[b] if b < len(dataset_names) else None
        sc = scene_ids[b] if b < len(scene_ids) else None
        scene_boxes = box_table.get(ds, {}).get(sc, None)
        if not scene_boxes:
            continue
        # frame-0 anchor (dataset-world → chunk frame), exactly as the model does.
        #   'se3'       : R0,t0 = full inv(c2w[0]) → boxes rotated into tilted cam
        #                 frame, then re-AABB'd → LOOSE enclosing box.
        #   'translate' : R0 = I (keep gravity-aligned world rotation), t0 = -cam0
        #                 position → boxes stay axis-aligned → TIGHT amodal AABB.
        if anchor_mode == 'translate':
            R0 = torch.eye(3, dtype=torch.float32, device=camera_pose.device)
            t0 = -camera_pose[b, 0, :3, 3].to(torch.float32)
        else:
            w2c0 = torch.linalg.inv(camera_pose[b, 0].to(torch.float32))   # [4,4]
            R0, t0 = w2c0[:3, :3], w2c0[:3, 3]
        bs = float(box_scale[b].clamp(min=eps).item()) if torch.is_tensor(box_scale) \
            else float(max(box_scale[b], eps))

        ids = torch.unique(instance_mask[b])
        ids = ids[ids > 0].tolist()
        for iid in ids:
            box6 = scene_boxes.get(int(iid), None)
            if box6 is None:
                continue
            box6 = box6.to(device=R0.device, dtype=torch.float32)
            if max_dim_m > 0 and float(box6[3:6].max()) > max_dim_m:
                continue        # structural/stuff-scale instance → no box supervision
            center, size = box6[:3], box6[3:6]
            bmin, bmax = center - size * 0.5, center + size * 0.5
            corners = aabb_corners(bmin, bmax).to(torch.float32)        # [8,3] world
            corners_a = corners @ R0.T + t0                             # [8,3] frame-0
            amin = corners_a.min(dim=0).values
            amax = corners_a.max(dim=0).values
            c_n = (amin + amax) * 0.5 / bs                              # NORMALIZE
            s_n = (amax - amin).clamp(min=1e-3) / bs
            out[(b, int(iid))] = torch.cat([c_n, s_n], dim=-1).to(device=device, dtype=dtype)
    return out


@torch.no_grad()
def build_gt_camera_boxes(
    depth:         torch.Tensor,    # [B, T, H, W]   GT depth (metric)
    intrinsic:     torch.Tensor,    # [B, T, 3, 3]   GT intrinsics
    c2w:           torch.Tensor,    # [B, T, 4, 4]   GT camera-to-world
    instance_mask: torch.Tensor,    # [B, T, H, W]   long (0 = unlabeled)
    box_scale:     Optional[torch.Tensor] = None,    # IGNORED (kept for API compat)
    iqr_mult:      float = 1.5,
    min_pix:       int = 10,
) -> Dict[Tuple[int, int, int], torch.Tensor]:
    """Produce per-(b, t, inst_id) camera-frame canonical GT box.

    Returned dict:
        keys:   (b, t, int(inst_id)) where inst_id > 0
        values: torch.Tensor [6] on the same device as `depth`,
                in (cx, cy, cz, w, h, d) format, CANONICAL scale,
                CAMERA frame.

    A (b, t, inst_id) pair is included iff:
        - That instance has ≥ min_pix visible pixels in frame (b, t)
          (with depth > 0).
        - The cross-frame aggregate for that instance has ≥ 1 surviving
          point after IQR filter.

    Devices: stays on the input device (no cpu round-trip).
    """
    B, T, H, W = depth.shape
    valid_depth = depth > 1e-3                                  # [B, T, H, W]
    world_pts = unproject_to_world(depth, intrinsic, c2w)       # [B, T, H, W, 3]

    out: Dict[Tuple[int, int, int], torch.Tensor] = {}

    # Per-batch scope: instance IDs are scene-consistent within a single
    # ScanNetpp scene; batches are independent.
    for b in range(B):
        # Collect unique instance IDs across all frames in this batch.
        ids_in_batch = torch.unique(instance_mask[b])
        ids_in_batch = ids_in_batch[ids_in_batch > 0]            # drop 0 = unlabeled
        if ids_in_batch.numel() == 0:
            continue

        # Scene-scale normalizer: median of all valid GT depth pixels in
        # this batch.  Using GT (not DA3 predicted) keeps the target
        # noise-free and scale-invariant for canonical DA3 training.
        valid_b = valid_depth[b]
        if not valid_b.any():
            continue
        scale_b = float(depth[b][valid_b].median().item())
        if scale_b <= 1e-6:
            continue

        # Per-(b, t) visibility count per instance — needed for the
        # min_pix gate.  Loop over frames once.
        per_frame_visible: Dict[Tuple[int, int], int] = {}        # (t, inst) → pix count
        per_inst_world_pts: Dict[int, list] = {}                  # inst → list of [P_t, 3]

        for t in range(T):
            mask_t  = instance_mask[b, t]                         # [H, W]
            vdepth_t = valid_depth[b, t]                          # [H, W]
            wpts_t  = world_pts[b, t]                             # [H, W, 3]

            for iid in ids_in_batch.tolist():
                vis = (mask_t == iid) & vdepth_t
                n   = int(vis.sum().item())
                per_frame_visible[(t, iid)] = n
                if n > 0:
                    pts = wpts_t[vis]                              # [P_t, 3]
                    per_inst_world_pts.setdefault(iid, []).append(pts)

        # Aggregate cross-frame → world AABB per instance
        for iid, pts_list in per_inst_world_pts.items():
            all_pts = torch.cat(pts_list, dim=0)                  # [P_total, 3]
            # Drop non-finite (safety)
            finite = torch.isfinite(all_pts).all(dim=-1)
            all_pts = all_pts[finite]
            if all_pts.shape[0] == 0:
                continue
            filtered = iqr_filter(all_pts, iqr_mult)
            if filtered.shape[0] == 0:
                continue
            world_bmin = filtered.min(dim=0).values                # [3]
            world_bmax = filtered.max(dim=0).values                # [3]

            # Project world AABB → each frame's camera AABB.
            corners_w = aabb_corners(world_bmin, world_bmax)       # [8, 3]
            corners_w_h = torch.cat([corners_w,
                                      torch.ones(8, 1, device=corners_w.device,
                                                  dtype=corners_w.dtype)], dim=-1)  # [8, 4]

            for t in range(T):
                if per_frame_visible.get((t, iid), 0) < min_pix:
                    continue
                w2c_t = torch.linalg.inv(c2w[b, t].to(dtype=torch.float32)).to(corners_w.dtype)
                corners_c = (w2c_t @ corners_w_h.T).T[:, :3]       # [8, 3] camera frame
                cam_bmin = corners_c.min(dim=0).values              # [3]
                cam_bmax = corners_c.max(dim=0).values
                center = (cam_bmin + cam_bmax) * 0.5
                size   = (cam_bmax - cam_bmin).clamp(min=1e-3)
                # Convert metric → canonical (head's output space).
                box6 = torch.cat([center, size], dim=-1) / scale_b   # [6]
                out[(b, t, int(iid))] = box6

    return out
