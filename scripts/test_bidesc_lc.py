"""Synthetic tests for ma_slam_semantic/bidesc_lc.py.
Run:  PYTHONPATH=src python scripts/test_bidesc_lc.py"""
import sys

import numpy as np

from ma_slam_semantic.bidesc_lc import (BiDescConfig, FrameIndex, adaptive_k,
                                        lookup_descs, verify)

rng = np.random.default_rng(0)
CFG = BiDescConfig()
FAILED = []


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def make(n):
    return np.stack([unit(rng.standard_normal(256)).astype(np.float32) for _ in range(n)])


def noisy(D, eps=0.03):   # cos ~0.9, the measured same-inst level
    return np.stack([unit(d + eps * rng.standard_normal(256)).astype(np.float32) for d in D])


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got!r} (want {want!r})")
    if not ok:
        FAILED.append(name)


D6 = make(6)

# 1. same instance set (noisy re-observation) -> PASS
v = verify(noisy(D6), D6, CFG)
check("same set", v.status, "PASS")
check("same set nuniq", v.n_uniq, 6)

# 2. disjoint sets -> FAIL (random cos ~0)
v = verify(make(5), make(5), CFG)
check("disjoint sets", v.status, "FAIL")

# 3. asymmetric: history is subset of current (3 of 8 shared) — harmonic penalizes
Dq = np.concatenate([noisy(D6[:3]), make(5)])            # 8 current, 3 shared
Dr = D6[:3]                                              # 3 historical, all shared
v = verify(Dq, Dr, CFG)
# a_rq high (all history explained), a_qr diluted by 5 novel instances but topK
# picks the strongest -> still passes; the point is it does NOT fail on asymmetry
check("subset overlap", v.status, "PASS")

# 4. one-sided support only: current has 1 shared + 7 novel, history 1 shared + 5 novel
Dq = np.concatenate([noisy(D6[:1]), make(7)])
Dr = np.concatenate([D6[:1], make(5)])
v = verify(Dq, Dr, CFG)
check("single shared instance", v.status, "FAIL")
check("single shared nuniq", v.n_uniq <= 1 or v.status == "FAIL", True)

# 5. many-to-one clones: 4 current fragments all near ONE history desc
base = D6[0]
Dq = np.stack([unit(base + 0.03 * rng.standard_normal(256)).astype(np.float32)
               for _ in range(4)])
Dr = np.stack([base] + [unit(rng.standard_normal(256)).astype(np.float32) for _ in range(3)])
v = verify(Dq, Dr, CFG)
check("many-to-one clones nuniq", v.n_uniq, 1)     # MNN collapses to one unique pair
check("many-to-one clones", v.status, "FAIL")

# 6. too few instances -> UNKNOWN
check("nq<n_min", verify(make(1), make(6), CFG).status, "UNKNOWN")
check("nr<n_min", verify(make(6), np.zeros((0, 256), np.float32), CFG).status, "UNKNOWN")

# 7. adaptive K
check("adaptive_k(2)", adaptive_k(2, CFG), 2)
check("adaptive_k(6)", adaptive_k(6, CFG), 3)
check("adaptive_k(12)", adaptive_k(12, CFG), 4)

# 8. pairs logging includes sub-threshold MNN pairs
Dq = np.concatenate([noisy(D6[:2]), make(2)])
Dr = np.concatenate([D6[:2], make(2)])
v = verify(Dq, Dr, CFG)
subthr = [p for p in v.pairs if not p["uniq"]]
check("sub-threshold pairs logged", len(v.pairs) >= v.n_uniq, True)
check("pairs_str nonempty", len(v.pairs_str()) > 0, True)

# 9. FrameIndex + lookup
class Rec:
    def __init__(self, d):
        import torch
        self.descriptor = torch.from_numpy(d)
records = {i: Rec(D6[i]) for i in range(6)}
idx = FrameIndex(k_bag=3)
idx.add("f0.png", [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.99)])   # cap 3 by objectness
ids = idx.get("f0.png")
check("k_bag cap", len(ids), 3)
check("cap keeps highest obj", ids[0][0], 3)
D = lookup_descs(ids + [(99, 0.5)], records)                    # 99 missing -> skipped
check("lookup shape", D.shape, (3, 256))
check("lookup normalized", bool(abs(np.linalg.norm(D[0]) - 1) < 1e-5), True)

print(f"\n{'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}'}")
sys.exit(1 if FAILED else 0)
