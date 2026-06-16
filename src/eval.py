"""Trajectory ATE evaluation against TUM-style ground truth.

Compares estimated per-frame c2w (camera_poses.txt, 16 numbers/line) to a GT file
with header ``# timestamp tx ty tz qx qy qz qw`` (one row per frame, e.g. ScanNet
``gt_pose.txt``). Uses Umeyama Sim3 alignment of the two position tracks before
computing ATE RMSE, since monocular reconstruction is only defined up to a
similarity transform.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def load_estimated_positions(path: str) -> np.ndarray:
    poses = np.loadtxt(path).reshape(-1, 4, 4)
    return poses[:, :3, 3]


def load_gt_positions(path: str) -> np.ndarray:
    data = np.loadtxt(path)  # comment lines (#...) are skipped by loadtxt
    return data[:, 1:4].astype(np.float64)  # tx ty tz


def umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity (s, R, t) mapping src -> dst (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = (d_c.T @ s_c) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (s_c ** 2).sum() / src.shape[0]
    s = np.trace(np.diag(D) @ S) / var
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate_rmse(est_path: str, gt_path: str, align: str = "sim3") -> dict:
    """ATE between estimated and GT trajectories.

    align:
        'sim3' — full similarity alignment (scale-invariant; classic monocular ATE,
                 but HIDES metric-scale error — not ideal for judging depth input).
        'se3'  — rotation+translation only (metric ATE; reveals depth's scale benefit).
        'none' — no alignment (raw world-frame error).
    Also reports ``scale`` = the Sim3 scale that best maps est->gt (how far the
    estimated trajectory scale is from metric; 1.0 == perfectly metric).
    """
    est = load_estimated_positions(est_path)
    gt = load_gt_positions(gt_path)
    k = min(len(est), len(gt))
    est, gt = est[:k], gt[:k]
    valid = np.isfinite(est).all(1) & np.isfinite(gt).all(1)
    est, gt = est[valid], gt[valid]

    s, R, t = umeyama_sim3(est, gt)  # always computed, for the `scale` diagnostic
    if align == "sim3":
        aligned = (s * (R @ est.T).T) + t
    elif align == "se3":
        Rr, tr = umeyama_sim3(est, gt)[1], None
        mu_s, mu_d = est.mean(0), gt.mean(0)
        U, _, Vt = np.linalg.svd(((gt - mu_d).T @ (est - mu_s)) / len(est))
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1
        Rr = U @ S @ Vt
        aligned = (Rr @ est.T).T + (mu_d - Rr @ mu_s)
    elif align == "none":
        aligned = est
    else:
        raise ValueError(f"align must be 'sim3'|'se3'|'none', got {align!r}")

    err = np.linalg.norm(aligned - gt, axis=1)
    return {"n": int(len(err)), "align": align, "scale": float(s),
            "ate_rmse": float(np.sqrt((err ** 2).mean())),
            "ate_mean": float(err.mean()), "ate_median": float(np.median(err))}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--est", required=True)
    ap.add_argument("--gt", required=True)
    a = ap.parse_args()
    sim3 = ate_rmse(a.est, a.gt, align="sim3")
    se3 = ate_rmse(a.est, a.gt, align="se3")
    print(f"Sim3-ATE: {sim3['ate_rmse']:.4f} m (scale-invariant) | "
          f"SE3-ATE(metric): {se3['ate_rmse']:.4f} m | "
          f"est scale vs gt: {sim3['scale']:.4f} | n={sim3['n']}")
