"""Confidence-weighted pose blending — compact port of AMB3R `pose_interp.interpolate_poses`.

Blends two c2w pose sets (e.g. the existing map pose and a fresh estimate) per frame,
weighting by each side's mean confidence: SLERP on rotation, linear on translation.
Used so a keyframe's pose is refined (not overwritten) as it is re-observed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _slerp(R0: np.ndarray, R1: np.ndarray, w: float) -> np.ndarray:
    """Spherical-linear interpolation between two rotations, fraction w toward R1."""
    from scipy.spatial.transform import Rotation as Rot, Slerp
    key = Rot.from_matrix(np.stack([R0, R1]))
    return Slerp([0.0, 1.0], key)([w])[0].as_matrix()


def interpolate_poses(poses_a: np.ndarray, poses_b: np.ndarray,
                      conf_a: Optional[np.ndarray] = None,
                      conf_b: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-frame confidence-weighted blend of two (N,4,4) c2w pose sets (a=old, b=new)."""
    n = len(poses_a)
    ca = conf_a.reshape(n, -1).mean(1) if conf_a is not None else np.ones(n)
    cb = conf_b.reshape(n, -1).mean(1) if conf_b is not None else np.ones(n)
    out = np.tile(np.eye(4), (n, 1, 1)).astype(np.float64)
    for i in range(n):
        wb = float(cb[i] / (ca[i] + cb[i] + 1e-9))   # weight toward the new estimate
        if wb <= 1e-6:
            out[i] = poses_a[i]; continue
        if wb >= 1 - 1e-6:
            out[i] = poses_b[i]; continue
        out[i, :3, :3] = _slerp(poses_a[i, :3, :3], poses_b[i, :3, :3], wb)
        out[i, :3, 3] = (1 - wb) * poses_a[i, :3, 3] + wb * poses_b[i, :3, 3]
    return out
