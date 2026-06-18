"""Real-time point-cloud / trajectory visualisation for ma_slam streaming, via rerun.

Decouples viewing from compute: the server logs, you watch on any machine. Three sinks
(``--viz``):

* ``spawn``   — launch a native rerun viewer on THIS machine (server needs a display).
* ``web``     — serve the rerun web viewer from the server; open the printed URL in a browser
                anywhere (the laptop needs nothing but a browser).
* ``connect`` — stream to an already-running rerun viewer elsewhere (e.g. ``rerun`` on your
                laptop); point ``--viz_addr`` at it (``rr+http://LAPTOP_IP:9876``).

Incremental strategy (global PGO moves poses every submap; loop closure moves everything):
each new submap's confident points are logged once under ``world/points/submap_<id>``; the
camera trajectory (cheap) is re-logged every submap; when a loop closes, ALL submaps' points
are re-logged so the drift correction is reflected.

Depends only on ``rerun-sdk`` (already in the repo env, 0.26.2). No-op if ``--viz none``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class RerunViz:
    def __init__(self, mode: str, addr: Optional[str] = None, *, conf_coef: float = 1.0,
                 max_points: int = 500_000, app_id: str = "ma_slam_stream"):
        import rerun as rr
        self.rr = rr
        self.conf_coef = conf_coef
        self.max_points = max_points
        self._logged: set = set()              # submap base_ids already logged (points)
        self._rng = np.random.default_rng(0)

        rr.init(app_id)
        if mode == "spawn":
            rr.spawn()
        elif mode == "web":
            rr.serve_web(open_browser=False)
            print("[viz] rerun web viewer served — open the URL above (or http://<server>:9090) "
                  "in a browser")
        elif mode == "connect":
            url = addr or "rr+http://127.0.0.1:9876"
            rr.connect_grpc(url)
            print(f"[viz] streaming to rerun viewer at {url}")
        else:
            raise ValueError(f"unknown viz mode {mode!r}")
        # MapAnything poses are OpenCV right-down-forward.
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    # ------------------------------------------------------------------ logging
    def _log_submap_points(self, sm, graph, budget):
        p, c = sm.world_points_and_colors(graph, conf_coef=self.conf_coef,
                                          max_points=budget, rng=self._rng)
        if len(p):
            self.rr.log(f"world/points/submap_{sm.base_id}",
                        self.rr.Points3D(p, colors=c.astype(np.uint8)))

    def _log_trajectory(self, slam):
        centers = []
        for sm in slam.map.ordered():
            if getattr(sm, "is_lc", False):
                continue
            for i in range(sm.n):
                centers.append(graph_center(slam.graph, sm.key(i)))
        if len(centers) >= 2:
            path = np.asarray(centers, dtype=np.float32)
            self.rr.log("world/trajectory", self.rr.LineStrips3D([path]))
            last = path[-1]
            self.rr.log("world/camera", self.rr.Transform3D(translation=last))

    def update(self, slam, n_frames: int, loop_happened: bool):
        rr = self.rr
        rr.set_time("frame", sequence=n_frames)
        nonlc = [sm for sm in slam.map.ordered() if not getattr(sm, "is_lc", False)]
        budget = None if not self.max_points else max(1, self.max_points // max(1, len(nonlc)))
        if loop_happened:                      # a loop moved everything -> re-log all clouds
            self._logged.clear()
        for sm in nonlc:
            if sm.base_id in self._logged:
                continue
            self._log_submap_points(sm, slam.graph, budget)
            self._logged.add(sm.base_id)
        self._log_trajectory(slam)


def graph_center(graph, key) -> np.ndarray:
    return graph.get_pose(key)[:3, 3].astype(np.float32)


def make_viz(mode: str, addr: Optional[str], *, conf_coef: float = 1.0,
             max_points: int = 500_000) -> Optional[RerunViz]:
    """Return a RerunViz, or None when viz is disabled."""
    if not mode or mode == "none":
        return None
    return RerunViz(mode, addr, conf_coef=conf_coef, max_points=max_points)
