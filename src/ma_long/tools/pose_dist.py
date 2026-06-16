"""Extrinsic (pose) distance — ported from AMB3R `amb3r/tools/pose_dist.py` (numpy).

Distance between two c2w poses = normalized rotation angle + lambda_t * translation
distance:  d(A, B) = angle(R_A, R_B)/180 + lambda_t * ||t_A - t_B||.

Used for content-adaptive keyframe selection / chunk anchoring.
"""

from __future__ import annotations

import numpy as np


def rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle (degrees) between two 3x3 rotations."""
    R = R1.T @ R2
    val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


def extrinsic_distance(e1: np.ndarray, e2: np.ndarray, lambda_t: float = 1.0) -> float:
    """Distance between two 4x4 poses."""
    rot = rotation_angle(e1[:3, :3], e2[:3, :3]) / 180.0
    trans = float(np.linalg.norm(e1[:3, 3] - e2[:3, 3]))
    return rot + lambda_t * trans


def _rotation_angle_batch(R: np.ndarray) -> np.ndarray:
    """Pairwise normalized rotation angle for (N,3,3) -> (N,N) in [0,1]."""
    Rt = np.transpose(R, (0, 2, 1))[:, None]        # (N,1,3,3)
    M = np.matmul(Rt, R[None])                       # (N,N,3,3)
    tr = M[..., 0, 0] + M[..., 1, 1] + M[..., 2, 2]
    val = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(val)) / 180.0


def pairwise_extrinsic_distance(poses: np.ndarray, lambda_t: float = 1.0,
                                normalize: bool = True) -> np.ndarray:
    """(N,4,4) -> (N,N) pairwise pose distances.

    normalize: divide translations by the mean camera-centre norm so lambda_t is
    scale-independent (matches AMB3R's `compute_ranking(normalize=True)`).
    """
    poses = np.asarray(poses, dtype=np.float64)
    R = poses[:, :3, :3]
    t = poses[:, :3, 3].copy()
    if normalize:
        scale = float(np.mean(np.linalg.norm(t, axis=1))) + 1e-9
        t = t / scale
    rot = _rotation_angle_batch(R)                          # (N,N)
    trans = np.linalg.norm(t[:, None] - t[None], axis=2)    # (N,N)
    return rot + lambda_t * trans
