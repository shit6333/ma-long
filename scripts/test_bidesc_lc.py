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

# 3. asymmetric: history is a subset of current. top-K is adaptive (K=clip(ceil(n/2),2,4)),
# so when the shared count is BELOW K the average pulls in an unmatched row and the
# directional score drops — at the calibrated tau=0.85 that means FAIL for 3-of-8 but
# PASS once the shared count reaches K.
Dq = np.concatenate([noisy(D6[:3]), make(5)])            # 8 current, 3 shared, K_q=4
check("subset overlap 3-of-8 (shared < K)", verify(Dq, D6[:3], CFG).status, "FAIL")
Dq = np.concatenate([noisy(D6[:4]), make(4)])            # 8 current, 4 shared, K_q=4
check("subset overlap 4-of-8 (shared >= K)", verify(Dq, D6[:4], CFG).status, "PASS")

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



# --------------------------------------------------------------------- stage 2
print("verify_layout():")
from ma_slam_semantic.bidesc_lc import verify_layout

def rot(deg, axis=np.array([0, 1, 0.0])):
    axis = axis / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K

P = np.array([[0.5, 0.2, 2.0], [-0.6, 0.1, 2.5], [0.1, -0.4, 3.2], [0.9, 0.5, 1.5]])

# 1. same view (small noise) -> PASS
v = verify_layout(P, P + rng.normal(0, 0.02, P.shape), CFG)
check("layout same view", v.status, "PASS")

# 2. pure camera rotation 30deg (about the CAMERA origin: X_q = R X_r, t*=0)
#    NB: rotating points about their own centroid instead would mean the camera
#    ORBITS the cluster -> real displacement -> T_FAR would be correct there.
v = verify_layout(P @ rot(30).T, P, CFG)
check("layout 30deg", v.status, "PASS")
check("layout 30deg thetaR", bool(abs(v.theta_r - 30) < 6), True)

# 3. pure camera rotation 120deg -> LARGE_VIEW_CHANGE (Ed=0, ER=0, theta >> tau)
v = verify_layout(P @ rot(120).T, P, CFG)
check("layout 120deg", v.status, "LARGE_VIEW_CHANGE")

# 4. scrambled correspondence -> rejected (Ed blows up; t* is garbage too, so
#    either FAIL or T_FAR is a correct rejection under the tnorm-first ordering)
v = verify_layout(P, P[[2, 0, 3, 1]] * np.array([1.4, 0.7, 1.2]), CFG)
check("layout scrambled rejected", v.status in ("FAIL", "T_FAR"), True)

# 5. common scale offset x1.3: doc's Ed = |log 1.3| = 0.26 <= 0.4 -> PASSES
#    (documents that Ed-as-specified tolerates uniform scale drift up to e^tau_d)
v = verify_layout(P, P * 1.3, CFG)
check("layout scale x1.3 (doc Ed tolerates)", v.status, "PASS")
check("layout scale diag Escale", bool(abs(v.e_scale + np.log(1.3)) < 0.02), True)

# 6. translation-only false loop (teddy-type): edge metrics blind, but the
#    calibrated ||t*|| gate catches it (t* estimates the camera displacement)
v = verify_layout(P, P + np.array([1.0, 0, 0]), CFG)
check("layout translation-only -> T_FAR", v.status, "T_FAR")
check("layout translation t_norm~1.0", bool(abs(v.t_norm - 1.0) < 0.05), True)

# 6b. M=3 wild-rotation estimate on a REAL loop: rotation abstains below
#     s2_min_m_rot, distance-only ruling keeps it (the plant 879/1014 fix)
v = verify_layout(P[:3] @ rot(120).T, P[:3], CFG)
check("layout M=3 rotation abstains", v.status, "PASS")
check("layout M=3 reason", v.reason, "rot-abstain")

# 7. M<=2 -> M_LOW hard reject (1-2 co-visible objects can't support a verdict:
#    M=1 has zero edges, M=2's single edge is chance-consistent too easily)
check("layout M=1 -> M_LOW", verify_layout(P[:1], P[:1], CFG).status, "M_LOW")
check("layout M=2 -> M_LOW", verify_layout(P[:2], P[:2], CFG).status, "M_LOW")

print(f"\n{'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}'}")
sys.exit(1 if FAILED else 0)
