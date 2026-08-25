"""Cluster confidence — the score the AP evaluator ranks predictions by.

Ablation surface #3.  Every ranker maps one record's member observations to a
single scalar; nothing else in the pipeline changes when you switch.

Why this matters beyond a benchmark number: the score is the map's statement
about its own reliability, which is what downstream consumers (what to act on,
where to look again, what to keep in a bounded memory) actually read.

Verdicts (EXPERIMENTS.md 2026-08-05/06):
    p_top25q  meanP x mean(top-25% Q)   ← default, +2.9 AP50 ScanNet / +3.6 SceneNN
    pq        mean(P*Q)                 ≈ p_top25q (statistical tie)
    obj       mean(objectness)          the pre-P/Q baseline
    p, q      single-factor diagnostics

Why the top quartile for Q and the mean for P: P (thingness) is view-stable, so
its mean is well behaved; Q (mask quality) is view-dependent, so a real object's
mean is dragged down by its many awkward views.  Asking "how good are this
record's BEST views" separates real objects from junk — junk has no good views.
An EMA over the observation order was tried and lands in the same noise band,
with the drawback of weighting by recency (objects often exit the frame on a bad
view), so the quantile is kept.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np


def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if len(xs) else 0.0


def _top_quantile_mean(xs: Sequence[float], frac: float = 0.25) -> float:
    """Mean of the best `frac` of the values (at least one)."""
    if not len(xs):
        return 0.0
    k = max(1, int(len(xs) * frac))
    return float(np.mean(sorted(xs)[-k:]))


def rank_obj(rec) -> float:
    return _mean(rec.member_objectness)


def rank_p(rec) -> float:
    return _mean(rec.member_pq_p) if rec.member_pq_p else rank_obj(rec)


def rank_q(rec) -> float:
    return _mean(rec.member_pq_q) if rec.member_pq_q else rank_obj(rec)


def rank_pq(rec) -> float:
    if not (rec.member_pq_p and rec.member_pq_q):
        return rank_obj(rec)
    return float(np.mean(np.asarray(rec.member_pq_p) * np.asarray(rec.member_pq_q)))


def rank_p_top25q(rec) -> float:
    if not (rec.member_pq_p and rec.member_pq_q):
        return rank_obj(rec)
    return _mean(rec.member_pq_p) * _top_quantile_mean(rec.member_pq_q, 0.25)


RANKERS: Dict[str, Callable] = {
    'obj': rank_obj,
    'p': rank_p,
    'q': rank_q,
    'pq': rank_pq,
    'p_top25q': rank_p_top25q,
}


def score_record(rec, ranker: str = 'p_top25q') -> float:
    """Confidence for one record.  Falls back to objectness when the record
    carries no P/Q members (e.g. a checkpoint without the P/Q head)."""
    try:
        fn = RANKERS[ranker]
    except KeyError:
        raise ValueError(f'unknown ranker {ranker!r} (expected {sorted(RANKERS)})')
    return float(fn(rec))
