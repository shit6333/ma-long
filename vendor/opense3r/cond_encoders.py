"""
Conditioning encoders for OpenSe3r.

Three optional conditioning streams (all default off, controlled by flags on
the OpenSe3r model):

  - depth_cond:    depth_tokens (from DepthEncoder) added at CausalBranch input
                   via a zero-init residual.  Lives in CausalBranch (no encoder
                   here — DepthEncoder already exists).

  - pose_cond:     per-frame pose token = MLP(rot6d + tanh(t/scene_scale) +
                   normalized intrinsics).  Prepended to each frame's input.
                   Encoder = PoseTokenEncoder.

  - plucker_cond:  per-patch Plücker rays = (origin_normalized, unit_direction)
                   passed through Fourier-feature MLP.  Added at CausalBranch
                   input via residual (same pattern as depth).
                   Encoder = PluckerEncoder.

Design notes:
  - scene_scale = clip-level depth median.  Shared with DepthEncoder's
    median normalization so origin / depth use the *same* scale-invariant
    convention → cross-backend (VGGT canonical / DA3 metric) weight transfer.
  - Reference frame for both pose_cond and plucker_cond is clip[0] (relative
    poses).  Frame 0 is identity, frames 1..T-1 are non-identity.
  - All extrinsics / intrinsics inputs are expected to be already detached
    by the caller (they originate from frozen backbones, but detach is
    belt-and-suspenders).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def make_relative_pose(extrinsics: torch.Tensor) -> torch.Tensor:
    """
    Convert per-frame c2w extrinsics into clip[0]-relative poses.
    Frame 0 → identity, frames 1..T-1 → c2w relative to frame 0.

    Args:
        extrinsics: [B, T, 4, 4]  c2w
    Returns:
        rel: [B, T, 4, 4]  c2w relative to frame 0
    """
    T0_inv = torch.inverse(extrinsics[:, 0:1])           # [B, 1, 4, 4]
    return T0_inv @ extrinsics                           # [B, T, 4, 4]


def rotation_to_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Zhou et al. 2019: take first two columns of the rotation matrix.
    Continuous, singularity-free, suitable for MLP input.

    Args:
        R: [..., 3, 3]
    Returns:
        [..., 6]
    """
    return R[..., :, :2].reshape(*R.shape[:-2], 6)


def pose_to_features(
    extrinsics_rel: torch.Tensor,        # [B, T, 4, 4]  c2w relative to clip[0]
    intrinsics:     torch.Tensor,        # [B, T, 3, 3]
    H: int, W: int,
    scene_scale:    torch.Tensor,        # [B]  clip-level depth median
) -> torch.Tensor:
    """
    Build the 13-dim pose feature vector per frame:
        [rot6d (6) | tanh(trans / scene_scale) (3) | normalized intrinsics (4)]

    Returns:
        [B, T, 13]
    """
    R = extrinsics_rel[..., :3, :3]                                # [B, T, 3, 3]
    rot6d = rotation_to_6d(R)                                       # [B, T, 6]

    t = extrinsics_rel[..., :3, 3]                                  # [B, T, 3]
    trans_n = torch.tanh(t / (scene_scale[:, None, None] + 1e-6))   # [B, T, 3]

    fx = intrinsics[..., 0, 0] / W
    fy = intrinsics[..., 1, 1] / H
    cx = intrinsics[..., 0, 2] / W
    cy = intrinsics[..., 1, 2] / H
    intr = torch.stack([fx, fy, cx, cy], dim=-1)                    # [B, T, 4]

    return torch.cat([rot6d, trans_n, intr], dim=-1)                # [B, T, 13]


def compute_plucker_rays(
    extrinsics_rel: torch.Tensor,        # [B, T, 4, 4]  c2w relative to clip[0]
    intrinsics:     torch.Tensor,        # [B, T, 3, 3]
    H: int, W: int,
    patch_size: int,
    scene_scale: torch.Tensor,           # [B]
) -> torch.Tensor:
    """
    Per-patch Plücker rays = (scene-scale-normalized origin, unit direction).

    Origin is the camera centre in clip[0]-relative world coords, normalized by
    scene_scale and squashed with tanh for bounded long-trajectory behaviour.
    Direction is the unit-norm world-frame ray through the patch centre pixel.

    Returns:
        plucker: [B, T, N_patch, 6]   where N_patch = (H/patch_size) * (W/patch_size)
    """
    B, T = extrinsics_rel.shape[:2]
    device = extrinsics_rel.device
    dtype  = extrinsics_rel.dtype
    ph, pw = H // patch_size, W // patch_size

    # 1. Patch-centre pixel coordinates [N_patch, 3] homogeneous.
    j = torch.arange(pw, device=device, dtype=dtype)
    i = torch.arange(ph, device=device, dtype=dtype)
    u = (j + 0.5) * patch_size                                        # [pw]
    v = (i + 0.5) * patch_size                                        # [ph]
    vv, uu = torch.meshgrid(v, u, indexing='ij')                      # [ph, pw]
    pix = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)          # [ph, pw, 3]
    pix = pix.reshape(ph * pw, 3)                                     # [N_patch, 3]

    # 2. Camera-frame ray (un-normalized): K^-1 @ pix.
    K_inv = torch.inverse(intrinsics)                                  # [B, T, 3, 3]
    cam_rays = torch.einsum('btij,nj->btni', K_inv, pix)               # [B, T, N_patch, 3]

    # 3. Rotate to world frame, then unit-normalize.
    R = extrinsics_rel[..., :3, :3]                                    # [B, T, 3, 3]
    world_dirs = torch.einsum('btij,btnj->btni', R, cam_rays)          # [B, T, N_patch, 3]
    world_dirs = F.normalize(world_dirs, dim=-1)

    # 4. Origin: scene-scale normalized + tanh.  Same convention as pose token.
    origin = extrinsics_rel[..., :3, 3]                                # [B, T, 3]
    origin_n = torch.tanh(origin / (scene_scale[:, None, None] + 1e-6))
    origin_n = origin_n.unsqueeze(-2).expand(-1, -1, ph * pw, -1)      # [B, T, N_patch, 3]

    return torch.cat([origin_n, world_dirs], dim=-1)                   # [B, T, N_patch, 6]


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class PoseTokenEncoder(nn.Module):
    """
    13-D pose feature → causal_dim per-frame token.

    Default kaiming init (no zero-init): pose conditioning takes effect from
    iteration 1.  Loading a ckpt trained without pose_cond will see a small
    transient loss spike that recovers within a few epochs.
    """

    def __init__(self, in_dim: int = 13, dim: int = 1024):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, 512),    nn.GELU(),
            nn.Linear(512, dim),
        )

    def forward(self, pose_feats: torch.Tensor) -> torch.Tensor:
        """
        pose_feats: [B, T, 13]
        Returns:    [B, T, dim]
        """
        return self.mlp(pose_feats)


class PluckerEncoder(nn.Module):
    """
    6-D per-patch Plücker rays → causal_dim per-patch tokens, with optional
    Fourier features (NeRF-style) for high-frequency capacity.

    With use_fourier=True and num_freqs=K:
        in_dim = 6 + 6 * 2 * K   (raw + sin/cos at K log-spaced frequencies)
        e.g. K=4 → in_dim=54,  K=8 → in_dim=102.

    Default init = PyTorch default (kaiming uniform for Linear).
    """

    def __init__(self, dim: int = 1024, num_freqs: int = 4,
                 use_fourier: bool = True):
        super().__init__()
        self.use_fourier = use_fourier and num_freqs > 0
        self.num_freqs   = num_freqs
        in_dim = 6 + (6 * 2 * num_freqs if self.use_fourier else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, 512),    nn.GELU(),
            nn.Linear(512, dim),
        )

    def forward(self, plucker: torch.Tensor) -> torch.Tensor:
        """
        plucker: [B, T, N_patch, 6]
        Returns: [B, T, N_patch, dim]
        """
        if self.use_fourier:
            freqs = (2.0 ** torch.arange(
                self.num_freqs, device=plucker.device, dtype=plucker.dtype
            )) * math.pi                                              # [num_freqs]
            scaled = plucker.unsqueeze(-1) * freqs                    # [..., 6, num_freqs]
            ff = torch.cat([torch.sin(scaled), torch.cos(scaled)],
                           dim=-1)                                    # [..., 6, 2*num_freqs]
            ff = ff.flatten(-2)                                       # [..., 6 * 2*num_freqs]
            x = torch.cat([plucker, ff], dim=-1)
        else:
            x = plucker
        return self.mlp(x)
