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
