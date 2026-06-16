"""Keyframe selection & resampling — ported/adapted from AMB3R (numpy, model-agnostic).

- `select_keyframes`: greedy content-adaptive selection by pose distance + confidence
  (AMB3R `keyframes.py:select_keyframes_iteratively`). Picks frames that are far (in pose
  space) from all current keyframes, breaking ties by highest mean confidence then earliest.
- `resample_keyframes`: keep a bounded, diverse set of keyframes — newest + diverse core +
  transition fillers + bridge (AMB3R `memory.py:resample_keyframes`). For an online/streaming
  ma_long with a fixed keyframe budget.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ma_long.tools.pose_dist import pairwise_extrinsic_distance


def select_keyframes(
    poses: np.ndarray,
    conf_mean: np.ndarray,
    threshold: float = 0.15,
    *,
    lambda_t: float = 1.0,
    init: Sequence[int] = (0,),
    tolerance: float = 5e-3,
    dists: Optional[np.ndarray] = None,
) -> List[int]:
    """Greedily select keyframe indices.

    Args:
        poses: (N,4,4) c2w poses.
        conf_mean: (N,) per-frame mean confidence.
        threshold: a frame becomes a keyframe only if its pose distance to *every*
            existing keyframe exceeds this.
        init: indices to seed the keyframe set.
        tolerance: frames within (1-tolerance)*max_conf of the best candidate are
            considered tied; the earliest of those is chosen.
        dists: optional precomputed (N,N) pose-distance matrix.

    Returns:
        Sorted list of keyframe indices.
    """
    n = len(poses)
    if dists is None:
        dists = pairwise_extrinsic_distance(poses, lambda_t=lambda_t)
    conf_mean = np.asarray(conf_mean, dtype=np.float64)

    kf = list(init)
    candidates = set(range(n)) - set(kf)
    while True:
        far = [c for c in candidates if all(dists[c, k] > threshold for k in kf)]
        if not far:
            break
        max_conf = max(conf_mean[i] for i in far)
        nxt = min(i for i in far if conf_mean[i] >= max_conf * (1 - tolerance))
        kf.append(nxt)
        candidates.discard(nxt)
    return sorted(kf)


def adaptive_chunk_indices(
    poses: np.ndarray,
    target_motion: float,
    *,
    max_size: int,
    min_size: int = 4,
    overlap: int = 8,
    lambda_t: float = 1.0,
    normalize: bool = True,
) -> List[tuple]:
    """Content-adaptive chunk boundaries from a (provisional) trajectory.

    Each chunk grows from its start frame until the pose distance to the start
    reaches `target_motion` (so chunks span roughly equal *motion*, not equal frame
    counts), bounded by [min_size, max_size]. Consecutive chunks share `overlap` frames.

    Intended for a second pass (or streaming mode) once provisional poses exist; lets
    fast-motion segments get shorter chunks and slow/static segments longer ones,
    avoiding fixed-split misalignments. Returns [(start, end), ...] tiling [0, N).
    """
    poses = np.asarray(poses, dtype=np.float64)
    n = len(poses)
    min_size = max(min_size, overlap + 1)  # guarantee forward progress (chunk > overlap)
    if n <= min_size:
        return [(0, n)]
    t = poses[:, :3, 3].copy()
    if normalize:
        t = t / (float(np.mean(np.linalg.norm(t, axis=1))) + 1e-9)
    R = poses[:, :3, :3]

    def dist(i, j):
        Rr = R[i].T @ R[j]
        ang = np.degrees(np.arccos(np.clip((np.trace(Rr) - 1) / 2, -1, 1))) / 180.0
        return ang + lambda_t * float(np.linalg.norm(t[i] - t[j]))

    chunks, s = [], 0
    while s < n:
        e = min(s + min_size, n)
        while e < n and (e - s) < max_size and dist(s, e) < target_motion:
            e += 1
        chunks.append((s, e))
        if e >= n:
            break
        s = e - overlap  # fixed overlap; min_size > overlap ensures e-overlap > s
    return chunks


def resample_keyframes(
    kf_indices: Sequence[int],
    poses: np.ndarray,
    num_keep: int,
    *,
    lambda_t: float = 1.0,
    top_k: int = 4,
    bridge: int = 1,
    thr_min: float = 0.1,
    thr_fill: float = 0.4,
    thr_fill_max: float = 1.2,
    loop_gap: int = 200,
) -> List[int]:
    """Reduce an active keyframe set to `num_keep` diverse keyframes (AMB3R strategy).

    Always keeps the newest; adds up to `top_k` closest-yet-diverse (or loop-closure,
    >`loop_gap` frames apart) keyframes; fills remaining slots with "transition" frames
    (within [thr_min, thr_fill] of the set and < thr_fill_max from some member); reserves
    `bridge` slot(s) for the frame(s) farthest (sum of distances) from the kept set.

    Args:
        kf_indices: global indices of the currently-active keyframes (last = newest).
        poses: (M,4,4) poses aligned 1:1 with kf_indices.
    """
    kf = [int(i) for i in kf_indices]
    if len(kf) <= num_keep:
        return sorted(kf)

    D = pairwise_extrinsic_distance(poses, lambda_t=lambda_t)  # (M,M) over kf set
    pos = {g: p for p, g in enumerate(kf)}

    newest = kf[-1]
    kept = [newest]
    others = sorted(kf[:-1], key=lambda g: D[pos[g], pos[newest]])  # nearest-first

    for g in others:
        if len(kept) >= top_k + 1:
            break
        if abs(g - newest) > loop_gap:          # loop-closure anchor: always keep
            kept.append(g); continue
        if all(D[pos[g], pos[s]] > thr_min for s in kept):  # diverse enough
            kept.append(g)

    pool = sorted(g for g in kf if g not in kept)
    while len(kept) < num_keep - bridge:
        add = None
        for g in pool:
            d = [D[pos[g], pos[s]] for s in kept]
            if min(d) <= thr_fill and max(d) <= thr_fill_max:
                add = g; break
        if add is None:
            break
        kept.append(add); pool.remove(add)

    if bridge > 0:
        rest = [g for g in kf if g not in kept]
        if rest:
            # bridge = frames CLOSEST (smallest summed distance) to the kept set, for
            # smooth local transitions (AMB3R `sum_min=True`, topk largest=False).
            sums = [sum(D[pos[g], pos[s]] for s in kept) for g in rest]
            order = np.argsort(sums)[:bridge]
            kept += [rest[i] for i in order]

    return sorted(set(kept))
