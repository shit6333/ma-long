"""Synthetic-bag unit tests for ma_slam_semantic/instance_lc.py.

Run:  PYTHONPATH=src python scripts/test_instance_lc.py
No model, no GPU — bags are built by hand; descs are random unit vectors where
"same instance" means same vector + small noise (mirrors the measured desc
stats: same-inst cos ~0.9, diff-inst ~0.2).
"""
import sys

import numpy as np

from ma_slam_semantic.instance_lc import (InstanceLCConfig, FrameBag, BagStore,
                                          build_bags, verify)

rng = np.random.default_rng(0)
CFG = InstanceLCConfig()


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def make_desc(n):
    return np.stack([unit(rng.standard_normal(256)).astype(np.float32) for _ in range(n)])


def noisy(descs, eps=0.03):    # ||eps*n|| ~ eps*16 -> cos ~0.9, the measured same-inst level
    return np.stack([unit(d + eps * rng.standard_normal(256)).astype(np.float32)
                     for d in descs])


def bag(descs, cent, obj=None, area=None):
    k = len(descs)
    return FrameBag(descs=np.asarray(descs, np.float32),
                    cent=np.asarray(cent, np.float32),
                    area=np.asarray(area if area is not None else [0.05] * k, np.float32),
                    obj=np.asarray(obj if obj is not None else [0.9] * k, np.float32))


FAILED = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {got} (want {want})")
    if not ok:
        FAILED.append(name)


# --------------------------------------------------------------------- verify()
print("verify():")
D4 = make_desc(4)
C4 = np.array([[0.20, 0.15], [0.55, 0.20], [0.40, 0.45], [0.70, 0.50]])

# 1. same place, slight desc noise -> PASS
v = verify(bag(D4, C4), bag(noisy(D4), C4 + rng.normal(0, 0.01, C4.shape)), CFG)
check("identical scene", v.status, "PASS")

# 2. camera pan: common centroid shift -> vectors unchanged -> PASS
v = verify(bag(D4, C4), bag(noisy(D4), C4 + np.array([0.22, 0.06])), CFG)
check("common shift (pan)", v.status, "PASS")

# 3. zoom 1.3x about the centroid -> lengths scale within band -> PASS
mid = C4.mean(0)
v = verify(bag(D4, C4), bag(noisy(D4), mid + 1.3 * (C4 - mid)), CFG)
check("zoom 1.3x", v.status, "PASS")

# 4. zoom 2.5x -> outside len_ratio band -> FAIL
v = verify(bag(D4, C4), bag(noisy(D4), mid + 2.5 * (C4 - mid)), CFG)
check("zoom 2.5x", v.status, "FAIL")

# 5. 90-deg viewpoint rotation of the arrangement -> FAIL (intended: side view)
R = np.array([[0.0, -1.0], [1.0, 0.0]])
v = verify(bag(D4, C4), bag(noisy(D4), mid + (C4 - mid) @ R.T), CFG)
check("90deg arrangement rotation", v.status, "FAIL")

# 6. twin mismatch: 4 matched, one at a random position -> LOO drops it, 3 survive
C_twin = C4 + np.array([0.10, 0.02])
C_twin[2] = [0.05, 0.55]                       # instance 2 landed on the wrong twin
v = verify(bag(D4, C4), bag(noisy(D4), C_twin), CFG)
check("twin mismatch LOO status", v.status, "PASS")
check("twin mismatch LOO inliers", v.n_inliers, 3)

# 7. aliasing: same-ish descs, scrambled arrangement -> FAIL
v = verify(bag(D4, C4), bag(noisy(D4), C4[[2, 3, 0, 1]]), CFG)
check("scrambled arrangement", v.status, "FAIL")

# 8. different instance sets -> FAIL at step 1
v = verify(bag(D4, C4), bag(make_desc(4), C4), CFG)
check("different instances", v.status, "FAIL")

# 9. tiny bags / missing bag -> ABSTAIN
v = verify(bag(D4[:1], C4[:1]), bag(D4, C4), CFG)
check("K<2 abstain", v.status, "ABSTAIN")
check("missing bag abstain", verify(None, bag(D4, C4), CFG).status, "ABSTAIN")

# 10. strict-2 branch: two consistent, tight, confident -> PASS
D2, C2 = D4[:2], C4[:2]
v = verify(bag(D2, C2, obj=[0.9, 0.9]), bag(noisy(D2), C2 + 0.01, obj=[0.9, 0.9]), CFG)
check("strict-2 tight", v.status, "PASS")
# ... but with low objectness -> FAIL
v = verify(bag(D2, C2, obj=[0.6, 0.6]), bag(noisy(D2), C2 + 0.01, obj=[0.6, 0.6]), CFG)
check("strict-2 low obj", v.status, "FAIL")
# ... and with a 25-deg rotation (passes 35-deg normal tol, fails 15-deg strict) -> FAIL
th = np.deg2rad(25)
R25 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
v = verify(bag(D2, C2, obj=[0.9, 0.9]),
           bag(noisy(D2), C2.mean(0) + (C2 - C2.mean(0)) @ R25.T, obj=[0.9, 0.9]), CFG)
check("strict-2 25deg", v.status, "FAIL")

# 11. coincident centroids -> relation skipped -> no spatial evidence -> not PASS
Cc = np.tile(np.array([[0.4, 0.3]]), (4, 1))
v = verify(bag(D4, Cc), bag(noisy(D4), Cc), CFG)
check("coincident centroids", v.status, "FAIL")

# --------------------------------------------------------------------- build_bags()
print("build_bags():")
H, W = 24, 37
pix = np.full((2, H, W), -1, dtype=np.int64)
pix[0, 2:8, 2:8] = 0          # 36 px  -> area 0.041  keep
pix[0, 10:22, 5:30] = 1       # 300 px -> area 0.338  keep (under the 0.40 cap)
pix[0, 3, 30] = 2             # 1 px   -> area 0.001  too small
pix[0, 15:19, 32:36] = 3      # kept but low obj
pix[1, 4:10, 20:26] = 4       # frame 1 keep
props = [
    dict(frame=0, descriptor=make_desc(1)[0], objectness=0.9),
    dict(frame=0, descriptor=make_desc(1)[0], objectness=0.9),
    dict(frame=0, descriptor=make_desc(1)[0], objectness=0.9),
    dict(frame=0, descriptor=make_desc(1)[0], objectness=0.3),
    dict(frame=1, descriptor=make_desc(1)[0], objectness=0.8),
]
bags = build_bags(props, pix, ["f0.png", "f1.png"], CFG)
check("frame0 regions kept", len(bags["f0.png"].obj), 2)
check("frame1 regions kept", len(bags["f1.png"].obj), 1)
b0 = bags["f0.png"]
cu, cv = b0.cent[0]
check("centroid units (u/W)", bool(abs(cu - 4.5 / W) < 1e-5 and abs(cv - 4.5 / W) < 1e-5), True)
check("desc renormalized", bool(abs(np.linalg.norm(b0.descs[0]) - 1.0) < 1e-5), True)

store = BagStore()
store.add(bags)
check("store lookup", store.get("f1.png") is not None, True)
check("store miss -> None", store.get("nope.png") is None, True)

print(f"\n{'ALL PASS' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}'}")
sys.exit(1 if FAILED else 0)
