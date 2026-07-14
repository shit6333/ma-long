"""GeoAwareRegionPooling — FastSAM-region → seed-query pooling (patch resolution).

This is the 2D / patch-grid adaptation of ESAM's `GeoAwarePooling`
(thirdparty/ESAM/oneformer3d/geo_aware_pool.py).  ESAM pools per-POINT 3D
U-Net features into per-SUPERPOINT query vectors using a geometry-aware
weighted mean.  Here the analog is:

    point     → image patch (causal-transformer patch token, native resolution)
    superpoint→ FastSAM region (a 2D mask)
    point xyz → per-patch world_points (avg-pooled to the patch grid)

Everything runs at PATCH resolution (≈ H/14 × W/14) because the causal feature
is patch-constant — pooling at pixel resolution would cost ~700 MB/frame for the
1024-d feature with no information gain.  The output mask / snap still happen at
full resolution elsewhere, so edge precision is unaffected.

Faithful to ESAM:
  • per-region center + diameter normalization of the local 3D shape
    (translation/scale invariant — NOT an absolute 3D positional encoding, which
    was shown to hurt instance separation; see memory 3dpe-hurts-instance);
  • per-patch sigmoid geometric-saliency weight (∈[0,2]) → weighted mean;
  • explicit global shape descriptor (z_global) added to the pooled feature.

The semantic feature fed in here MUST be detached by the caller so the seed-query
path does not backprop into the semantic head.

All ops are native torch (no torch_scatter dependency).  Per-frame loop over the
(small) B*T dimension keeps the region bookkeeping simple and correct for the
variable per-frame region count.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean of `src` rows grouped by `index`.

    Args:
        src:   [N, C]
        index: [N]   long, values in [0, dim_size)
        dim_size: number of output groups M
    Returns:
        [M, C]  (groups with no member → 0)
    """
    C = src.shape[1]
    out = src.new_zeros((dim_size, C))
    out.index_add_(0, index, src)
    cnt = src.new_zeros((dim_size, 1))
    cnt.index_add_(0, index, torch.ones_like(src[:, :1]))
    return out / cnt.clamp_min(1.0)


def _scatter_reduce(src: torch.Tensor, index: torch.Tensor, dim_size: int,
                    reduce: str) -> torch.Tensor:
    """Per-group amax / amin of `src` (native torch scatter_reduce)."""
    C = src.shape[1]
    init = float('-inf') if reduce == 'amax' else float('inf')
    out = src.new_full((dim_size, C), init)
    idx = index.unsqueeze(1).expand(-1, C)
    out = out.scatter_reduce(0, idx, src, reduce=reduce, include_self=True)
    # Empty groups keep ±inf → zero them so downstream math stays finite.
    # Use torch.where (out-of-place) to stay autograd-safe.
    return torch.where(out.isinf(), torch.zeros_like(out), out)


class GeoAwareRegionPooling(nn.Module):
    """Pool patch features into per-FastSAM-region seed queries.

    Args:
        causal_dim:    channel dim of the causal patch feature (e.g. 512).
        sem_dim:       channel dim of the (detached) semantic patch feature.
        d_model:       output seed-query dim (== head d_model, e.g. 256).
        geo_channels:  width of the geometry MLP (ESAM `channel_proj`, default 32).
        seed_hidden:   hidden width of the final SeedMLP.
    """

    def __init__(self, causal_dim: int, sem_dim: int, d_model: int,
                 geo_channels: int = 32, seed_hidden: int = 256):
        super().__init__()
        self.causal_dim = int(causal_dim)
        self.sem_dim    = int(sem_dim)
        self.d_model    = int(d_model)
        self.geo_channels = int(geo_channels)

        # ESAM pts_proj1 : 3 → C  (local shape feature from normalized xyz)
        self.pts_proj1 = nn.Sequential(
            nn.Linear(3, geo_channels),
            nn.LayerNorm(geo_channels),
            nn.ReLU(inplace=True),
            nn.Linear(geo_channels, geo_channels),
            nn.LayerNorm(geo_channels),
        )
        # ESAM pts_proj2 : 2C → 1  (per-patch geometric-saliency weight)
        self.pts_proj2 = nn.Sequential(
            nn.Linear(2 * geo_channels, geo_channels),
            nn.LayerNorm(geo_channels),
            nn.ReLU(inplace=True),
            nn.Linear(geo_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        # Final seed-query MLP : [causal ; sem ; z_global] → d_model
        in_dim = self.causal_dim + self.sem_dim + self.geo_channels
        self.seed_mlp = nn.Sequential(
            nn.Linear(in_dim, seed_hidden),
            nn.LayerNorm(seed_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(seed_hidden, d_model),
        )

    def _scatter_norm(self, xyz: torch.Tensor, index: torch.Tensor,
                      dim_size: int) -> torch.Tensor:
        """Per-region center + diameter normalization (ESAM scatter_norm).

        Returns normalized per-patch xyz of shape [N, 3] (unit-ish local shape).
        """
        center = _scatter_mean(xyz, index, dim_size)                 # [M,3]
        mx = _scatter_reduce(xyz, index, dim_size, 'amax')           # [M,3]
        mn = _scatter_reduce(xyz, index, dim_size, 'amin')           # [M,3]
        diameter = (mx - mn).max(dim=1).values                       # [M]
        c = center[index]                                            # [N,3]
        d = diameter[index].unsqueeze(1)                             # [N,1]
        return (xyz - c) / (d + 1e-2)

    def _forward_one(self, causal: torch.Tensor, sem: torch.Tensor,
                     xyz: torch.Tensor, region_id: torch.Tensor,
                     num_regions: int) -> torch.Tensor:
        """Single frame.

        Args:
            causal:    [Np, causal_dim]
            sem:       [Np, sem_dim]  (already detached by caller)
            xyz:       [Np, 3]
            region_id: [Np]  long in [0, num_regions); patches with id<0 dropped
            num_regions: M (>0)
        Returns:
            [M, d_model] seed queries
        """
        valid = region_id >= 0
        if valid.sum() == 0:
            return causal.new_zeros((num_regions, self.d_model))
        causal = causal[valid]
        sem    = sem[valid]
        xyz    = xyz[valid]
        idx    = region_id[valid]

        xyz_n   = self._scatter_norm(xyz, idx, num_regions)          # [n,3]
        z_local = self.pts_proj1(xyz_n)                              # [n,C]
        z_glob  = _scatter_reduce(z_local, idx, num_regions, 'amax') # [M,C]
        w = self.pts_proj2(torch.cat([z_local, z_glob[idx]], dim=-1)) * 2.0  # [n,1]

        f_app = _scatter_mean(causal * w, idx, num_regions)          # [M, causal_dim]
        f_sem = _scatter_mean(sem * w,    idx, num_regions)          # [M, sem_dim]
        q = self.seed_mlp(torch.cat([f_app, f_sem, z_glob], dim=-1)) # [M, d_model]
        return q

    def forward(self, causal: torch.Tensor, sem: torch.Tensor,
                xyz: torch.Tensor, region_id: torch.Tensor,
                max_queries: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool a batch of frames into padded seed queries.

        Args:
            causal:    [BT, Np, causal_dim]   causal patch tokens
            sem:       [BT, Np, sem_dim]      semantic patch feature (DETACHED)
            xyz:       [BT, Np, 3]            per-patch world_points
            region_id: [BT, Np]  long         per-patch FastSAM region id
                                              (0..M_t-1, or -1 for none/invalid).
                                              Region ids must be 0-based & dense
                                              per frame; ids >= max_queries are
                                              treated as invalid (-1).
            max_queries: Mmax — pad/truncate every frame to this many queries.
        Returns:
            seed:  [BT, Mmax, d_model]
            valid: [BT, Mmax]  bool — True where a real region exists
        """
        BT, Np, _ = causal.shape
        device = causal.device
        seed  = causal.new_zeros((BT, max_queries, self.d_model))
        valid = torch.zeros((BT, max_queries), dtype=torch.bool, device=device)

        for b in range(BT):
            rid = region_id[b].clone()
            rid[rid >= max_queries] = -1
            present = rid[rid >= 0]
            if present.numel() == 0:
                continue
            M = int(present.max().item()) + 1
            q = self._forward_one(causal[b], sem[b], xyz[b], rid, M)   # [M, d_model]
            m = min(M, max_queries)
            seed[b, :m] = q[:m]
            # A slot is valid only if at least one patch was assigned to it.
            cnt = torch.zeros(M, dtype=torch.long, device=device)
            cnt.index_add_(0, present, torch.ones_like(present))
            valid[b, :m] = cnt[:m] > 0
        return seed, valid
