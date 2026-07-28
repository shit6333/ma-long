"""Bi-Desc LC (v2) — bidirectional instance-descriptor verification of SALAD loop
candidates, VGGT-SLAM-2.0-inspired redesign of instance_lc (which is retired).

Per-frame storage is just the GLOBAL INSTANCE IDS the frame's proposals were
associated to (``FrameIndex``); descriptors are looked up at verify time from the
InstanceTable's running-mean memory — no per-frame descriptor bags, no extra
model forward, no area filtering (debug-friendly: what the model proposes is
what the gate sees, capped at ``k_bag`` by objectness).

Verdict (three-state, shadow-first):
  S = D_q D_r^T (cosine); directional scores a_qr / a_rq = mean of the top-K row/
  column maxima; a_bi = harmonic mean (punishes one-sided support); N_uniq =
  mutual-nearest-neighbour pairs above tau_pair (kills many-to-one inflation).
  UNKNOWN if either side has < n_min instances (abstain, never veto);
  PASS if N_uniq >= n_uniq and a_qr > tau_dir and a_rq > tau_dir and a_bi > tau_bi;
  FAIL otherwise.
All MNN pairs (also sub-threshold ones) are kept on the verdict for BIDESCPAIRS
logging — the "what sim did the rejected assignment have" debugging blind spot
of v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- config
@dataclass
class BiDescConfig:
    n_min: int = 2          # < n_min instances on either side -> UNKNOWN
    tau_pair: float = 0.85  # MNN unique-match cos (calibrated 2026-07: PASS must sit in the id-consistent ~1.0 cluster; 0.7-0.85 is the same-type aliasing band)
    tau_dir: float = 0.85   # per-direction MeanTopK threshold (calibrated, see tau_pair)
    tau_bi: float = 0.85    # harmonic-mean threshold (calibrated, see tau_pair)
    n_uniq: int = 2         # min unique MNN matches for PASS
    k_lo: int = 2           # adaptive top-K = clip(ceil(N*k_frac), k_lo, k_hi)
    k_hi: int = 4
    k_frac: float = 0.5
    k_bag: int = 16         # per-frame id cap (by objectness); no area filtering
    eps: float = 1e-6


def adaptive_k(n: int, cfg: BiDescConfig) -> int:
    return max(1, min(n, max(cfg.k_lo, min(cfg.k_hi, int(np.ceil(cfg.k_frac * n))))))


# --------------------------------------------------------------------------- storage
class FrameIndex:
    """path -> [(inst_id, objectness)] recorded at mapping time (Obs-equivalent
    provenance, inverted). The ONLY per-frame storage of v2 (~bytes per frame)."""

    def __init__(self, k_bag: int = 16):
        self.k_bag = k_bag
        self._ids: Dict[str, List[Tuple[int, float]]] = {}

    def add(self, path: str, items: Sequence[Tuple[int, float]]) -> None:
        kept = sorted(items, key=lambda t: -t[1])[: self.k_bag]
        self._ids[path] = [(int(i), float(o)) for i, o in kept]

    def get(self, path: str) -> Optional[List[Tuple[int, float]]]:
        return self._ids.get(path)

    def __len__(self):
        return len(self._ids)


def lookup_descs(ids: Sequence[Tuple[int, float]], records) -> np.ndarray:
    """(inst_id, obj) list -> (N,256) L2-normalized running-mean descriptors.
    Ids missing from the table (shouldn't happen mid-run) are skipped."""
    ds = []
    for iid, _ in ids:
        rec = records.get(iid)
        if rec is None:
            continue
        d = rec.descriptor.numpy().astype(np.float32)
        ds.append(d / (np.linalg.norm(d) + 1e-12))
    return np.stack(ds) if ds else np.zeros((0, 256), np.float32)


# --------------------------------------------------------------------------- verdict
@dataclass
class BiDescVerdict:
    status: str             # 'PASS' | 'FAIL' | 'UNKNOWN'
    n_q: int = 0
    n_r: int = 0
    a_qr: float = 0.0
    a_rq: float = 0.0
    a_bi: float = 0.0
    n_uniq: int = 0
    reason: str = ""
    # every mutual-NN pair: dict(q, r, sim, uniq) — uniq means sim > tau_pair
    pairs: list = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.status} a_qr={self.a_qr:.2f} a_rq={self.a_rq:.2f} "
                f"a_bi={self.a_bi:.2f} nuniq={self.n_uniq} nq/nr={self.n_q}/{self.n_r}"
                + (f" ({self.reason})" if self.reason else ""))

    def pairs_str(self) -> str:
        return " ".join(f"q{p['q']}~r{p['r']}:s{p['sim']:.2f}"
                        f":{'IN' if p['uniq'] else 'out'}" for p in self.pairs)


def verify(D_q: np.ndarray, D_r: np.ndarray, cfg: BiDescConfig) -> BiDescVerdict:
    n_q, n_r = len(D_q), len(D_r)
    if n_q < cfg.n_min or n_r < cfg.n_min:
        return BiDescVerdict("UNKNOWN", n_q=n_q, n_r=n_r,
                             reason=f"insufficient instances (n_min={cfg.n_min})")

    S = D_q @ D_r.T                                        # (n_q, n_r) cosine

    row_max = S.max(axis=1)                                # current -> history
    col_max = S.max(axis=0)                                # history -> current
    kq, kr = adaptive_k(n_q, cfg), adaptive_k(n_r, cfg)
    a_qr = float(np.sort(row_max)[::-1][:kq].mean())
    a_rq = float(np.sort(col_max)[::-1][:kr].mean())
    a_bi = float(2.0 * a_qr * a_rq / (a_qr + a_rq + cfg.eps))

    # mutual nearest neighbours (all logged; unique iff above tau_pair)
    row_arg = S.argmax(axis=1)
    col_arg = S.argmax(axis=0)
    pairs = []
    n_uniq = 0
    for i in range(n_q):
        j = int(row_arg[i])
        if int(col_arg[j]) == i:
            uniq = bool(S[i, j] > cfg.tau_pair)
            n_uniq += int(uniq)
            pairs.append(dict(q=i, r=j, sim=float(S[i, j]), uniq=uniq))

    v = BiDescVerdict("FAIL", n_q=n_q, n_r=n_r, a_qr=a_qr, a_rq=a_rq,
                      a_bi=a_bi, n_uniq=n_uniq, pairs=pairs)
    if (n_uniq >= cfg.n_uniq and a_qr > cfg.tau_dir
            and a_rq > cfg.tau_dir and a_bi > cfg.tau_bi):
        v.status = "PASS"
    else:
        why = []
        if n_uniq < cfg.n_uniq:
            why.append(f"nuniq<{cfg.n_uniq}")
        if a_qr <= cfg.tau_dir or a_rq <= cfg.tau_dir:
            why.append("dir")
        if a_bi <= cfg.tau_bi:
            why.append("bi")
        v.reason = "+".join(why)
    return v
