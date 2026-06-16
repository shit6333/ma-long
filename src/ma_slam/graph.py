"""Global pose-graph backend for ma_slam (VGGT-SLAM-style, metric SE3).

VGGT-SLAM optimizes a factor graph whose nodes are **SL(4) homographies**, because
VGGT is uncalibrated and its submaps differ by a 3D *projective* transform. MapAnything
is metric and calibrated, so its submaps differ only by **SE(3)** (with depth/intrinsics)
or **Sim(3)** (rgb-only scale drift, which ma_slam removes by pre-scaling submaps). We
therefore optimize on SE(3) with stock upstream ``gtsam`` (``Pose3`` + ``BetweenFactorPose3``).

A thin ``manifold`` seam is kept so a Sim(3) backend (vendored pypose ``Sim3LoopOptimizer``)
or a faithful SL(4) backend (MIT-SPARK's custom gtsam fork) can be slotted in later without
touching the solver — every node is a global camera-to-world pose ``T_wc`` regardless.

Nodes are keyed by integer ``base + frame_index`` exactly like VGGT-SLAM, so the i-th frame
of the submap whose base id is ``base`` is ``X(base + i)``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    import gtsam
    from gtsam.symbol_shorthand import X
    _HAVE_GTSAM = True
except Exception:  # pragma: no cover - surfaced lazily in __init__
    _HAVE_GTSAM = False


class PoseGraph:
    """Incremental SE(3) factor graph re-optimized after every submap (gtsam LM).

    All public poses are 4x4 camera-to-world matrices. Relative measurements follow
    gtsam's ``Between`` convention: ``between(T_a, T_b) = inv(T_a) @ T_b`` (b expressed
    in a's frame).
    """

    def __init__(self, manifold: str = "se3",
                 inner_sigma: float = 0.05, intra_sigma: float = 0.05,
                 anchor_sigma: float = 1e-6, loop_sigma: float = 0.10,
                 loop_robust: Optional[str] = None, loop_robust_k: float = 1.345):
        if manifold != "se3":
            raise NotImplementedError(
                f"manifold={manifold!r} not wired yet. The graph keeps an SE3/Pose3 "
                "backend; Sim3 (vendored pypose Sim3LoopOptimizer) and SL4 (MIT-SPARK "
                "gtsam fork) are the intended future backends behind this same seam."
            )
        if not _HAVE_GTSAM:
            raise ImportError(
                "ma_slam's SE3 backend needs `gtsam` (`pip install gtsam`). It is a "
                "self-contained wheel; only numpy/pyparsing deps."
            )
        self.manifold = manifold
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.initialized: set = set()
        self.num_loops = 0
        # Pose3 tangent order is [rx, ry, rz, tx, ty, tz] (6-dim).
        self.inner_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, inner_sigma))
        self.intra_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, intra_sigma))
        self.anchor_noise = gtsam.noiseModel.Diagonal.Sigmas(np.full(6, anchor_sigma))
        # Loop factors get an optional robust m-estimator (Huber/Cauchy) so a few wrong
        # loops (e.g. perceptual aliasing that slips past geo verification) are
        # auto-downweighted instead of warping the whole trajectory. Odometry
        # (inner/intra) stays a plain Gaussian — it is trusted, no outliers expected.
        self.loop_noise = self._robustify(
            gtsam.noiseModel.Diagonal.Sigmas(np.full(6, loop_sigma)), loop_robust, loop_robust_k)

    @staticmethod
    def _robustify(base, kind, k):
        if not kind or str(kind).lower() == "none":
            return base
        me = gtsam.noiseModel.mEstimator
        est = {"huber": me.Huber, "cauchy": me.Cauchy}[kind.lower()].Create(k)
        return gtsam.noiseModel.Robust.Create(est, base)

    # ------------------------------------------------------------------ nodes
    def add_pose(self, key: int, T_wc: np.ndarray):
        """Insert a node (global cam->world pose) if not already present."""
        if key in self.initialized:
            return
        self.values.insert(X(key), gtsam.Pose3(np.asarray(T_wc, dtype=np.float64)))
        self.initialized.add(key)

    def has(self, key: int) -> bool:
        return key in self.initialized

    # ------------------------------------------------------------------ factors
    def add_prior(self, key: int, T_wc: np.ndarray):
        """Anchor a node to a fixed global pose (fixes the world gauge)."""
        if key not in self.initialized:
            raise ValueError(f"prior on missing node {key}")
        self.graph.add(gtsam.PriorFactorPose3(
            X(key), gtsam.Pose3(np.asarray(T_wc, dtype=np.float64)), self.anchor_noise))

    def add_between(self, key_a: int, key_b: int, T_ab: np.ndarray, kind: str = "inner"):
        """Add a relative-pose constraint ``T_ab = inv(T_a) @ T_b`` between two nodes."""
        if key_a not in self.initialized or key_b not in self.initialized:
            raise ValueError(f"between factor on missing node(s) {key_a},{key_b}")
        noise = {"inner": self.inner_noise, "intra": self.intra_noise,
                 "loop": self.loop_noise}[kind]
        self.graph.add(gtsam.BetweenFactorPose3(
            X(key_a), X(key_b), gtsam.Pose3(np.asarray(T_ab, dtype=np.float64)), noise))
        if kind == "loop":
            self.num_loops += 1

    # ------------------------------------------------------------------ solve / read
    def optimize(self, verbose: bool = False) -> float:
        params = gtsam.LevenbergMarquardtParams()
        if verbose:
            params.setVerbosityLM("SUMMARY")
        opt = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, params)
        self.values = opt.optimize()
        return float(self.graph.error(self.values))

    def get_pose(self, key: int) -> np.ndarray:
        """Optimized global cam->world 4x4 pose for a node."""
        return self.values.atPose3(X(key)).matrix()

    def get_num_loops(self) -> int:
        return self.num_loops


# --------------------------------------------------------------------------- Sim3 backend
def _sim3_inv(s, R, t):
    return 1.0 / s, R.T, -(R.T @ t) / s


def _sim3_mul(a, b):
    sa, Ra, ta = a; sb, Rb, tb = b
    return sa * sb, Ra @ Rb, sa * (Ra @ tb) + ta


def _mat_to_sim3(M):
    """4x4 (with scale folded as sR in the top-left) -> (s, R, t)."""
    A = M[:3, :3]
    s = float(np.cbrt(max(np.linalg.det(A), 1e-12)))
    return s, A / s, M[:3, 3].copy()


def _sim3_to_mat(s, R, t):
    M = np.eye(4); M[:3, :3] = s * R; M[:3, 3] = t
    return M


class Sim3PoseGraph:
    """Sim(3) pose graph (same interface as PoseGraph) over the vendored pypose
    ``Sim3LoopOptimizer``. Nodes are absolute Sim(3) poses; the sequential chain is rebuilt
    from the (consistent) node placements, loop factors are the extra constraints. Without a
    loop the chain is exactly satisfiable, so ``optimize`` is a no-op (matches the SE3 case).

    ``get_pose`` returns a 4x4 with scale folded into the top-left block (``s·R``), so point
    transforms scale correctly and the translation is the metric camera centre (ATE-ready).
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.abs: Dict[int, tuple] = {}      # key -> (s, R, t) absolute (world<-cam)
        self.loops: list = []                # (key_i, key_j, (s,R,t)) relative
        self.initialized: set = set()
        self.num_loops = 0

    def add_pose(self, key: int, T_wc: np.ndarray):
        if key in self.initialized:
            return
        self.abs[key] = _mat_to_sim3(np.asarray(T_wc, dtype=np.float64))
        self.initialized.add(key)

    def has(self, key: int) -> bool:
        return key in self.initialized

    def add_prior(self, key: int, T_wc: np.ndarray):
        pass   # node 0 is the chain origin; Sim3LoopOptimizer anchors it implicitly

    def add_between(self, key_a: int, key_b: int, T_ab: np.ndarray, kind: str = "inner"):
        # intra/inter ties are encoded by the node placements (chain rebuilt in optimize);
        # only loop factors are extra constraints worth storing.
        if kind == "loop":
            # The solver passes T_ab in gtsam's BetweenFactor convention `inv(P_a) @ P_b`,
            # but Sim3LoopOptimizer's edge (i, j) constraint is the REVERSED `inv(P_j) @ P_i`
            # = inv(T_ab). Storing T_ab un-inverted makes every loop pull the wrong way and
            # blows the trajectory up (matches the slam.py loop-direction gotcha).
            self.loops.append((key_a, key_b, _sim3_inv(*_mat_to_sim3(np.asarray(T_ab, dtype=np.float64)))))
            self.num_loops += 1

    def optimize(self, verbose: bool = False) -> float:
        if not self.loops:
            return 0.0                                  # chain is exact -> no-op
        from align.sim3loop import Sim3LoopOptimizer
        from align.sim3utils import accumulate_sim3_transforms
        nodes = sorted(self.initialized)
        idx = {g: k for k, g in enumerate(nodes)}
        absG = [self.abs[g] for g in nodes]
        seq = [_sim3_mul(_sim3_inv(*absG[k]), absG[k + 1]) for k in range(len(nodes) - 1)]
        loops = [(idx[i], idx[j], con) for (i, j, con) in self.loops if i in idx and j in idx]
        if not loops:
            return 0.0
        opt = Sim3LoopOptimizer(self.cfg, device="cpu").optimize(seq, loops)
        opt_rel = [(1.0, np.eye(3), np.zeros(3))] + accumulate_sim3_transforms(opt)
        abs0 = absG[0]
        for k, g in enumerate(nodes):
            self.abs[g] = _sim3_mul(abs0, opt_rel[k])
        return 0.0

    def get_pose(self, key: int) -> np.ndarray:
        return _sim3_to_mat(*self.abs[key])

    def get_num_loops(self) -> int:
        return self.num_loops
