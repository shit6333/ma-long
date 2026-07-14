"""
opense3r/pos_encoders.py
========================
3DPPE-style sinusoidal Fourier encoder for normalized 3D coordinates.

Adapted from PETR / PETRv2 family (3DPPE).  Maps per-point (x, y, z) to a
fixed-dim token suitable as a PE residual on transformer inputs.

Two usage sites in OpenSe3r:
  - CausalBranch input residual (per-patch 3D coord PE)
  - Mask2Former query init (per-query 3D anchor PE → MLP → init query feat)

Always operates on NORMALIZED coordinates (world_pts / box_scale → median
≈ 1), so it is scale-invariant across DA3 canonical / nested-metric modes
AND avoids OOD when the SLAM trajectory grows beyond training-time chunk
extents.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


def pos2posemb3d(
    pos: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
) -> torch.Tensor:
    """3D sinusoidal positional encoding on (x, y, z) tensors.

    Args:
        pos:           [..., 3]   coordinates in NORMALIZED scale (~ unit range)
        num_pos_feats: per-axis hidden dim → output dim = num_pos_feats * 3
        temperature:   sinusoid temperature (standard 10000)

    Returns:
        posemb: [..., num_pos_feats * 3]
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0:1] / dim_t
    pos_y = pos[..., 1:2] / dim_t
    pos_z = pos[..., 2:3] / dim_t

    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    return torch.cat((pos_y, pos_x, pos_z), dim=-1)


class PosEncoder3D(nn.Module):
    """3D position → model-dim token via Fourier PE + 2-layer MLP.

    Mirrors the `query_embedding` MLP in 3DPPE's PETRHead but exposed as a
    standalone module so the same instance can be used for both:
      - per-patch PE residual (input world_pts)
      - per-query init (learnable 3D anchor)
    """
    def __init__(
        self,
        out_dim:       int = 1024,
        num_pos_feats: int = 128,
        temperature:   int = 10000,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature   = temperature
        in_dim = num_pos_feats * 3
        # Match 3DPPE: Linear→ReLU→Linear (same width)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, pos_normalized: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_normalized: [..., 3]  in NORMALIZE scale (≈ unit)
        Returns:
            token:          [..., out_dim]
        """
        pe = pos2posemb3d(pos_normalized, self.num_pos_feats, self.temperature)
        return self.mlp(pe.to(pos_normalized.dtype))
