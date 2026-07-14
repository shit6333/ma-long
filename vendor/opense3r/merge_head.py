"""Learned pairwise merge head (instance-association classifier).

Replaces the hand-weighted SLAM merge score
``w_desc·desc + w_vox·vox + w_geo·geo`` with a small network that, given two
instance representations (a current-chunk *proposal* and an accumulated
*record*, or any two observations), predicts ``P(same physical instance)``.

Design (v1):
  - **Inputs** (all DETACHED by the caller — gradients never reach the trunk):
        desc  : [.., 256]  L2-normed query descriptor   (appearance)
        sem   : [.., 1024] L2-normed mask-pooled dense semantic feature
        geo   : (vox_iou, center_dist)  scalars
    Compared SYMMETRICALLY (Hadamard ⊙ and |Δ|) so the head is order-invariant
    by construction — no need to learn symmetry.
  - **Per-stream encoders** map each stream (esp. the 2-d geo, low→high) to a
    fixed embedding, THEN a fusion MLP → 1 logit.  Encoding the scalar geo up
    is necessary so it isn't drowned by the high-dim appearance/semantic
    streams in the fusion.
  - Output is a LOGIT (use BCEWithLogits at train; ``.sigmoid()`` at SLAM).

The same `build_pair_features` is used at train AND SLAM so the feature layout
is identical on both sides — the single most important consistency requirement.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dims, dropout: float = 0.0) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:                       # no norm/act on the last layer
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def build_pair_features(
    desc_a: torch.Tensor, desc_b: torch.Tensor,     # [.., Dd]  L2-normed
    sem_a:  torch.Tensor, sem_b:  torch.Tensor,     # [.., Ds]  L2-normed
    vox_iou:     torch.Tensor,                       # [..]      ∈[0,1]
    center_dist: torch.Tensor,                       # [..]      metres
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric pair-feature builder shared by train + SLAM.

    Returns (a_desc [.., 2Dd], a_sem [.., 2Ds], g [.., 2]).
    All ops are symmetric in (a, b) → the resulting score is order-invariant.
    """
    a_desc = torch.cat([desc_a * desc_b, (desc_a - desc_b).abs()], dim=-1)
    a_sem  = torch.cat([sem_a * sem_b,  (sem_a  - sem_b ).abs()], dim=-1)
    g = torch.stack([vox_iou, center_dist], dim=-1)
    return a_desc, a_sem, g


class MergeHead(nn.Module):
    """Pairwise same-instance classifier.  ~0.4M params."""

    def __init__(self, desc_dim: int = 256, sem_dim: int = 1024,
                 desc_hidden: int = 128, sem_hidden: int = 128,
                 geo_hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.desc_dim = int(desc_dim)
        self.sem_dim  = int(sem_dim)
        self.enc_desc = _mlp([2 * desc_dim, 256, desc_hidden], dropout)
        self.enc_sem  = _mlp([2 * sem_dim, 512, sem_hidden], dropout)
        self.enc_geo  = _mlp([2, geo_hidden, geo_hidden], dropout)
        fused = desc_hidden + sem_hidden + geo_hidden
        self.fusion = _mlp([fused, 256, 128, 1], dropout)

    def forward(self, a_desc: torch.Tensor, a_sem: torch.Tensor,
                g: torch.Tensor) -> torch.Tensor:
        """Returns a LOGIT of shape [..]  (squeeze last dim)."""
        e = torch.cat([self.enc_desc(a_desc), self.enc_sem(a_sem), self.enc_geo(g)], dim=-1)
        return self.fusion(e).squeeze(-1)

    @torch.no_grad()
    def score(self, desc_a, desc_b, sem_a, sem_b, vox_iou, center_dist) -> torch.Tensor:
        """Convenience for SLAM: build features → forward → sigmoid → P(same)."""
        a_desc, a_sem, g = build_pair_features(
            desc_a, desc_b, sem_a, sem_b, vox_iou, center_dist)
        return torch.sigmoid(self.forward(a_desc, a_sem, g))


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (shared layout with SLAM's voxelisation: inter/min IoU)
# ─────────────────────────────────────────────────────────────────────────────

def voxel_codes(pts: torch.Tensor, voxel_size: float, grid: int = 4096) -> torch.Tensor:
    """[P,3] world points → unique int64 voxel codes (fixed-origin, order-free)."""
    if pts.numel() == 0:
        return pts.new_zeros((0,), dtype=torch.long)
    idx = torch.floor(pts / float(voxel_size)).long() + grid // 2
    idx = idx.clamp_(0, grid - 1)
    return torch.unique((idx[:, 0] * grid + idx[:, 1]) * grid + idx[:, 2])


def vox_iou_min(codes_a: torch.Tensor, codes_b: torch.Tensor) -> float:
    """Intersection-over-min-count of two unique voxel-code sets (== SLAM sim_vox)."""
    na, nb = codes_a.numel(), codes_b.numel()
    if na == 0 or nb == 0:
        return 0.0
    inter = torch.isin(codes_a, codes_b).sum().item()
    return float(inter) / float(max(min(na, nb), 1))
