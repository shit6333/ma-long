import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthEncoder(nn.Module):
    """
    Encode a per-frame depth map into patch-level tokens for depth-guided cross-attention.

    depth [B*T, 1, H, W]  →  [B*T, N_patch, out_dim]

    The output projection is zero-initialized so the fusion starts as a no-op
    and learns to inject geometry signal over training.
    """

    def __init__(self, out_dim: int = 1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, out_dim, 1),
        )
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, depth: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        """
        Args:
            depth:   [B*T, 1, H, W]
            patch_h: target patch-grid height  (H // patch_size)
            patch_w: target patch-grid width   (W // patch_size)
        Returns:
            [B*T, N_patch, out_dim]  where N_patch = patch_h * patch_w
        """
        x = self.encoder(depth)                             # [B*T, out_dim, H, W]
        x = F.adaptive_avg_pool2d(x, (patch_h, patch_w))   # [B*T, out_dim, ph, pw]
        return x.flatten(2).permute(0, 2, 1)               # [B*T, N_patch, out_dim]


class PointEncoder(nn.Module):
    """
    Encode a per-frame NORMALIZED world-point map (xyz) into patch-level tokens
    for geometry-guided cross-attention — the 3-channel sibling of DepthEncoder.

    world_points_norm [B*T, 3, H, W]  →  [B*T, N_patch, out_dim]

    Unlike DepthEncoder, the output projection is NOT zero-initialized: the
    pretrained backbone and this task differ substantially, so the geometry
    signal is allowed to contribute from step 0 rather than ramping from a no-op.

    The caller is expected to feed world_points ALREADY normalized by box_scale
    (median ≈ 1), so the input distribution is scale-invariant across DA3
    canonical / nested-metric backends — same convention as pos3d_cond.
    """

    def __init__(self, out_dim: int = 1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, out_dim, 1),
        )

    def forward(self, world_pts: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        """
        Args:
            world_pts: [B*T, 3, H, W]   NORMALIZED world points (÷ box_scale)
            patch_h, patch_w: target patch-grid size
        Returns:
            [B*T, N_patch, out_dim]
        """
        x = self.encoder(world_pts)                         # [B*T, out_dim, H, W]
        x = F.adaptive_avg_pool2d(x, (patch_h, patch_w))   # [B*T, out_dim, ph, pw]
        return x.flatten(2).permute(0, 2, 1)               # [B*T, N_patch, out_dim]
