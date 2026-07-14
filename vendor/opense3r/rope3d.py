"""3D Rotary Position Embedding for the CausalBranch.

A direct 3-axis extension of stream3r's RotaryPositionEmbedding2D (same
table-based mechanism), injecting per-patch 3D world position into attention so
that cross-frame attention sees the TRUE 3D layout instead of the repeated 2D
image grid.  Inspired by Volt (Volume Transformer, RWTH) which RoPEs voxel
(x,y,z) indices for 3D scene understanding.

Mechanics (identical principle to the 2D version):
    θ(pos) = pos · inv_freq ;  attention(i,j) depends only on (pos_i − pos_j)
    → relative / translation-invariant.  Feeding frame-0-anchored world voxel
    indices makes two physically-near patches (from any frames) get near indices
    → near rotation → attended as near.

head_dim (=64 here) is split across the 3 axes by `dims_split` (each even).
Default (24, 24, 16) mirrors Volt's (12,12,8) frequency split (z gets fewer).
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def voxelize_world_points(
    world_pts: torch.Tensor,        # [B, T, P, 3]  frame-0 anchored world coords (metric)
    voxel_size: float = 0.1,
    max_grid: int = 4096,
) -> torch.Tensor:
    """Per-patch world coords → non-negative int voxel indices [B, T, P, 3] (long).

    Origin is a FIXED offset (max_grid // 2), NOT a per-chunk min.  This makes
    the index a deterministic function of the (frame-0 / SLAM-global) coordinate:
    the SAME physical point gets the SAME index in EVERY chunk → cross-chunk RoPE
    stays correct under KV-cache streaming (where the cached K from a previous
    chunk must share the new query's origin).

    Why a fixed origin is also fine for training: RoPE only sees RELATIVE
    positions, and within a single chunk a constant offset cancels
    ((a+C)-(b+C) = a-b), so per-chunk-min and fixed-offset are identical for
    training (single forward, no cross-chunk cache).  Indoor scenes at 0.1 m
    voxel span well within ±(max_grid/2) voxels, so clamping is rarely hit.

    Non-finite coords (depth holes) map to the origin (index = offset).
    """
    off = max_grid // 2
    wp = world_pts.float()
    finite = torch.isfinite(wp).all(dim=-1, keepdim=True)
    wp_filled = torch.where(finite, wp, torch.zeros_like(wp))         # holes → origin
    idx = torch.floor(wp_filled / voxel_size) + off
    idx = idx.clamp_(min=0, max=max_grid - 1)
    return idx.long()


class RotaryPositionEmbedding3D(nn.Module):
    """3-axis RoPE.  forward(tokens [B,nh,N,head_dim], positions [B,N,3] long)."""

    def __init__(self, frequency: float = 100.0, dims_split: Tuple[int, int, int] = (24, 24, 16)):
        super().__init__()
        assert all(d % 2 == 0 for d in dims_split), "each axis dim must be even"
        self.base_frequency = frequency
        self.dims_split = tuple(dims_split)
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(self, dim, seq_len, device, dtype):
        key = (dim, seq_len, device, dtype)
        if key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq).to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            self.frequency_cache[key] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self.frequency_cache[key]

    @staticmethod
    def _rotate_features(x):
        d = x.shape[-1]
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(self, tokens, positions, cos_comp, sin_comp):
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert positions.ndim == 3 and positions.shape[-1] == 3, \
            "positions must be (B, N, 3)"
        assert tokens.size(-1) == sum(self.dims_split), \
            f"head_dim {tokens.size(-1)} != sum(dims_split) {sum(self.dims_split)}"
        chunks = tokens.split(self.dims_split, dim=-1)               # 3 × [B,nh,N,d_ax]
        out = []
        for ax in range(3):
            d = self.dims_split[ax]
            pos_ax = positions[..., ax]
            max_pos = int(pos_ax.max()) + 1
            cos_c, sin_c = self._compute_frequency_components(d, max_pos, tokens.device, tokens.dtype)
            out.append(self._apply_1d_rope(chunks[ax], pos_ax, cos_c, sin_c))
        return torch.cat(out, dim=-1)
