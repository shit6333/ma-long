"""Instance-level loop-candidate verification (semantic-assisted LC).

Standalone pipeline: per-frame bags of NMS'd instance-region descriptors (from
``make_proposals`` output, 2D only — no dependency on SLAM geometry, drift or
scale) verify SALAD loop candidates before the expensive coloc re-inference.
Consumed by ``ma_slam.solver`` through ``SemanticHooks.verify_loop_candidate``.

Two-step check (premise: the 256-D query descriptor is discriminative — measured
same-instance cos ~0.85-0.93 vs different-instance ~0.14-0.28, hard same-class
AUROC 0.97+; occasional twin-object mismatches are independent events killed by
relation voting):

- Step 1  desc distribution: Hungarian one-to-one on desc cosine — do the two
  frames see the same instance set?  (one-to-one also stops over-segmentation
  double counting; upstream NMS already removed duplicate proposals.)
- Step 2  desc distribution in space: for every matched pair, the within-frame
  displacement vector between the two instances must keep its direction and
  approximate length across frames ("cup above table, chair in front" — the
  continuous form of arrangement predicates; translation/zoom invariant, large
  viewpoint change rotates the vectors and fails, which is intended: those
  pairs can't coloc-verify from 2 frames anyway).  A wrongly matched twin sits
  at a random position, loses its relation votes, and is removed by
  leave-one-out before the final inlier count.

Decision is a COUNT (inliers >= min_inliers), not an average: unmatched clutter
(objects entering/leaving view, segmentation drift) is normal and must not
dilute the score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


# --------------------------------------------------------------------------- config
@dataclass
class InstanceLCConfig:
    # bag formation (aligned with the desc measurement population: clearly seen,
    # mid/large objects — objectness + area gates pull the online distribution
    # back to the measured one)
    bag_top_k: int = 12
    bag_min_obj: float = 0.5
    # area band calibrated on s0231/s0011 shadow runs (2026-07): proposal area
    # median is only ~1.5-2.8%, so the old 0.005 floor killed 28% of proposals and
    # left 9% of frames with <3 regions (ABSTAIN/strict-2 territory); 0.002/0.40
    # halves that and doubled rescued tier-B loops on s0231 at the cost of a few
    # extra dry colocs (bounded by max_tier_b).
    bag_min_area: float = 0.002      # area_frac below -> unstable desc/centroid
    bag_max_area: float = 0.40       # above -> stuff-like (wall/floor), aliasing fuel

    # step 1: desc matching
    tau_hi: float = 0.60             # matched pair := assigned & cos >= tau_hi
    min_matched: int = 2             # below -> FAIL outright

    # step 2: spatial-relation voting
    theta_tol_deg: float = 35.0      # max direction change of within-frame vectors
    len_ratio_lo: float = 0.6        # ||v_m|| / ||v_q|| band (tolerates zoom,
    len_ratio_hi: float = 1.7        #  rejects random placements)
    min_vec_len: float = 0.03        # skip relation if either vector shorter (units
                                     #  of image WIDTH; near-coincident centroids
                                     #  have undefined direction)
    loo_keep_ratio: float = 0.5      # per-instance consistent-relation ratio to survive

    # final gate
    min_inliers: int = 3
    strict2_angle_deg: float = 15.0  # inliers==2 branch: single relation must be tight
    strict2_obj: float = 0.7         #  and both instances confidently objects


# --------------------------------------------------------------------------- bags
@dataclass
class FrameBag:
    """Per-frame instance regions. ``cent`` is in units of image WIDTH for BOTH
    axes (aspect-preserving — angles between centroid vectors stay geometric)."""
    descs: np.ndarray                # (K, D) float32, rows L2-normalized
    cent: np.ndarray                 # (K, 2) float32, (u, v) / W
    area: np.ndarray                 # (K,)   float32, mask pixels / (H*W)
    obj: np.ndarray                  # (K,)   float32


def build_bags(props: Sequence[dict], pix_map, paths: Sequence[str],
               cfg: InstanceLCConfig, stats: Optional[dict] = None) -> Dict[str, FrameBag]:
    """``make_proposals`` output (+ the chunk's per-frame image paths) -> bags.

    ``props[k]['frame']`` indexes into ``paths``; ``pix_map`` is [T,H,W] with the
    per-pixel owning proposal index (conf-gated panoptic resolve) — centroids and
    areas are computed from it so they match exactly what the descriptor pooled.
    ``stats`` (optional dict) accumulates per-proposal drop reasons for calibration.
    """
    pm = pix_map.cpu().numpy() if hasattr(pix_map, "cpu") else np.asarray(pix_map)
    T, H, W = pm.shape

    def tally(key, n=1):
        if stats is not None:
            stats[key] = stats.get(key, 0) + n

    per_frame: Dict[int, List[tuple]] = {}
    for p_idx, prop in enumerate(props):
        t = int(prop["frame"])
        o = float(prop["objectness"])
        if o < cfg.bag_min_obj:
            tally("drop_low_obj")
            continue
        ys, xs = np.nonzero(pm[t] == p_idx)
        if ys.size == 0:
            tally("drop_no_pixels")
            continue
        area = ys.size / float(H * W)
        if stats is not None:
            stats.setdefault("areas", []).append(round(area, 4))
        if area < cfg.bag_min_area:
            tally("drop_too_small")
            continue
        if area > cfg.bag_max_area:
            tally("drop_too_big")
            continue
        tally("kept")
        d = prop["descriptor"]
        d = d.detach().cpu().numpy() if hasattr(d, "detach") else np.asarray(d)
        d = d.astype(np.float32)
        d /= (np.linalg.norm(d) + 1e-12)
        cent = np.array([xs.mean() / W, ys.mean() / W], dtype=np.float32)
        per_frame.setdefault(t, []).append((o, d, cent, area))

    bags: Dict[str, FrameBag] = {}
    for t, items in per_frame.items():
        items.sort(key=lambda it: -it[0])
        tally("drop_topk", max(0, len(items) - cfg.bag_top_k))
        items = items[:cfg.bag_top_k]
        if stats is not None:
            stats.setdefault("bag_sizes", []).append(len(items))
        bags[paths[t]] = FrameBag(
            descs=np.stack([it[1] for it in items]),
            cent=np.stack([it[2] for it in items]),
            area=np.array([it[3] for it in items], dtype=np.float32),
            obj=np.array([it[0] for it in items], dtype=np.float32))
    return bags


class BagStore:
    """path -> FrameBag. Keyed by image path because graph keys are ambiguous
    (overlap frames own two nodes and the bridge dedups the shared frame; the
    physical frame's path is identical in both submaps)."""

    def __init__(self):
        self._bags: Dict[str, FrameBag] = {}

    def add(self, bags: Dict[str, FrameBag]) -> None:
        self._bags.update(bags)

    def get(self, path: str) -> Optional[FrameBag]:
        return self._bags.get(path)

    def __len__(self):
        return len(self._bags)


# --------------------------------------------------------------------------- verdict
@dataclass
class Verdict:
    status: str                      # 'PASS' | 'FAIL' | 'ABSTAIN'
    n_q: int = 0
    n_m: int = 0
    n_matched: int = 0
    n_inliers: int = 0
    score: float = 0.0               # sum(sim * obj) over inliers (ranking only)
    reason: str = ""
    # per matched pair: dict(q, m, sim, obj, aq, am, inlier) — obj = min of the two
    # sides, aq/am = area fractions. For offline mismatch analysis (SEMLCPAIRS log).
    pairs: list = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.status} inliers={self.n_inliers} matched={self.n_matched} "
                f"bags={self.n_q}/{self.n_m} score={self.score:.2f}"
                + (f" ({self.reason})" if self.reason else ""))

    def pairs_str(self) -> str:
        return " ".join(
            f"q{p['q']}~m{p['m']}:s{p['sim']:.2f}:o{p['obj']:.2f}"
            f":a{p['aq']:.3f}/{p['am']:.3f}:{'IN' if p['inlier'] else 'out'}"
            for p in self.pairs)


def _relation_consistency(idx: List[int], qc: np.ndarray, mc: np.ndarray,
                          cfg: InstanceLCConfig):
    """Pairwise within-frame vector agreement over the matched instances in
    ``idx``. Returns (consistent, evaluated) as n x n bool matrices (upper used
    symmetrically); a relation is skipped (not evaluated) when either vector is
    too short for a stable direction."""
    n = len(idx)
    cons = np.zeros((n, n), dtype=bool)
    evald = np.zeros((n, n), dtype=bool)
    cos_tol = np.cos(np.deg2rad(cfg.theta_tol_deg))
    for a in range(n):
        for b in range(a + 1, n):
            vq = qc[b] - qc[a]
            vm = mc[b] - mc[a]
            lq, lm = np.linalg.norm(vq), np.linalg.norm(vm)
            if lq < cfg.min_vec_len or lm < cfg.min_vec_len:
                continue
            evald[a, b] = evald[b, a] = True
            ratio = lm / lq
            ok = (float(vq @ vm) / (lq * lm) >= cos_tol
                  and cfg.len_ratio_lo <= ratio <= cfg.len_ratio_hi)
            cons[a, b] = cons[b, a] = ok
    return cons, evald


def verify(bag_q: Optional[FrameBag], bag_m: Optional[FrameBag],
           cfg: InstanceLCConfig) -> Verdict:
    """Two-step instance verification of one loop candidate (see module doc)."""
    if bag_q is None or bag_m is None or len(bag_q.obj) < 2 or len(bag_m.obj) < 2:
        return Verdict("ABSTAIN",
                       n_q=0 if bag_q is None else len(bag_q.obj),
                       n_m=0 if bag_m is None else len(bag_m.obj),
                       reason="bag missing or <2 regions")

    # ---- step 1: one-to-one desc assignment
    from scipy.optimize import linear_sum_assignment
    sims = bag_q.descs @ bag_m.descs.T                       # cosine (rows normalized)
    rows, cols = linear_sum_assignment(-sims)
    matched = [(int(r), int(c)) for r, c in zip(rows, cols) if sims[r, c] >= cfg.tau_hi]
    nq, nm = len(bag_q.obj), len(bag_m.obj)
    pair_info = [dict(q=r, m=c, sim=float(sims[r, c]),
                      obj=float(min(bag_q.obj[r], bag_m.obj[c])),
                      aq=float(bag_q.area[r]), am=float(bag_m.area[c]),
                      inlier=False)
                 for r, c in matched]
    if len(matched) < cfg.min_matched:
        return Verdict("FAIL", n_q=nq, n_m=nm, n_matched=len(matched),
                       reason="instance sets differ", pairs=pair_info)

    # ---- step 2: arrangement voting + leave-one-out
    qi = [r for r, _ in matched]
    mi = [c for _, c in matched]
    qc, mc = bag_q.cent[qi], bag_m.cent[mi]

    def survivors(indices: List[int]) -> List[int]:
        cons, evald = _relation_consistency(indices, qc[indices], mc[indices], cfg)
        keep = []
        for a in range(len(indices)):
            ne = int(evald[a].sum())
            if ne == 0:
                continue                                     # no spatial evidence -> drop
            if cons[a].sum() / ne >= cfg.loo_keep_ratio:
                keep.append(indices[a])
        return keep

    # pass 1 removes gross mismatches (twin objects at random positions); pass 2
    # recounts among survivors so a removed instance stops poisoning the ratios.
    alive = survivors(list(range(len(matched))))
    if len(alive) >= 2:
        alive = survivors(alive)

    n_in = len(alive)
    for a in alive:
        pair_info[a]["inlier"] = True
    score = float(sum(sims[qi[a], mi[a]] * min(bag_q.obj[qi[a]], bag_m.obj[mi[a]])
                      for a in alive))
    v = Verdict("FAIL", n_q=nq, n_m=nm, n_matched=len(matched),
                n_inliers=n_in, score=score, pairs=pair_info)
    if n_in >= cfg.min_inliers:
        v.status = "PASS"
    elif n_in == 2:
        a, b = alive
        vq = qc[b] - qc[a]
        vm = mc[b] - mc[a]
        lq, lm = np.linalg.norm(vq), np.linalg.norm(vm)
        tight = (lq >= cfg.min_vec_len and lm >= cfg.min_vec_len
                 and float(vq @ vm) / (lq * lm) >= np.cos(np.deg2rad(cfg.strict2_angle_deg))
                 and cfg.len_ratio_lo <= lm / lq <= cfg.len_ratio_hi)
        objs_ok = all(min(bag_q.obj[qi[k]], bag_m.obj[mi[k]]) >= cfg.strict2_obj
                      for k in alive)
        if tight and objs_ok:
            v.status = "PASS"
            v.reason = "strict-2"
        else:
            v.reason = "2 inliers, strict branch failed"
    else:
        v.reason = "arrangement inconsistent"
    return v
