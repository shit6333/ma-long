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
            import urllib.parse
            web_port, grpc_port = 9090, 9876
            rr.serve_web(open_browser=False, web_port=web_port, grpc_port=grpc_port)
            ip = _primary_ip()
            grpc = f"rerun+http://{ip}:{grpc_port}/proxy"
            self.web_url = f"http://{ip}:{web_port}/?url=" + urllib.parse.quote(grpc, safe="")
            # The bare http://host:9090 only shows rerun's welcome screen — the viewer needs the
            # ?url=... that points back at THIS server's data stream, so print the full URL.
            print("\n[viz] ============ OPEN THIS URL IN A BROWSER ============")
            print(f"[viz]   {self.web_url}")
            print("[viz] ===================================================\n")
        elif mode == "connect":
            url = addr or "rr+http://127.0.0.1:9876"
            rr.connect_grpc(url)
            print(f"[viz] streaming to rerun viewer at {url}")
        else:
            raise ValueError(f"unknown viz mode {mode!r}")
        # MapAnything poses are OpenCV right-down-forward.
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

        # Blueprint: three independent, individually-toggleable layers —
        #   world/points/**        RGB geometry cloud (per-submap children)
        #   world/instances/voxels instance-colored semantic voxels
        #   world/instances/boxes  instance bboxes — hidden by default (large
        #                          floor/wall boxes occlude the scene; flip the
        #                          eye icon in the entity tree to show them)
        try:
            import rerun.blueprint as rrb
            rr.send_blueprint(rrb.Blueprint(
                rrb.Spatial3DView(
                    origin="world",
                    overrides={
                        "world/instances/boxes": rrb.EntityBehavior(visible=False),
                    },
                ),
                collapse_panels=False,
            ))
        except Exception as e:                 # blueprint is cosmetic — never fatal
            print(f"[viz] blueprint skipped ({e})")

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

    # ------------------------------------------------------------------
    # Semantic instance layer (ma_slam_semantic hooks)
    # ------------------------------------------------------------------

    @staticmethod
    def _instance_color(inst_id: int) -> np.ndarray:
        """Stable, well-separated per-id color (golden-ratio hue walk)."""
        import colorsys
        h = (inst_id * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
        return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)

    def update_instances(self, table, n_frames: int, *,
                         max_extent_m: float = 4.0,
                         min_voxels: int = 5,
                         stale_grace: int = 40,
                         vote_min: int = 2,
                         radius_scale: float = 0.25):
        """Log the InstanceTable as an instance-colored voxel cloud + bboxes.

        Cheap enough to re-log wholesale every submap (tens of thousands of
        0.05 m voxel centers), which also keeps the view consistent with the
        table's true state after loop closure: records re-lift lazily via the
        M-signature machinery, so their voxels simply move on a later update
        once re-observed — no special LC handling needed here.

        Display-only denoising (the table itself is untouched — the export
        applies the real re-lift/merge/filter at stop; these mirror its spirit
        so the live view resembles the exported PLY):
          * hide records larger than ``max_extent_m`` or with < ``min_voxels``
          * hide single-observation records not re-seen within ``stale_grace``
            key-steps (transient one-frame noise)
          * for re-observed records, draw only voxels with votes >= ``vote_min``
            (drops one-frame spurious voxels / pre-LC drift smear)
        """
        if table is None or len(table) == 0:
            return
        rr = self.rr
        rr.set_time("frame", sequence=n_frames)
        vs = float(table.voxel_size)
        now = max((rec.last_seen for rec in table.records.values()), default=0)

        pts, cols = [], []
        box_centers, box_half, box_cols, labels = [], [], [], []
        for iid, rec in table.records.items():
            bmin = rec.bbox_min.numpy().astype(np.float32)
            bmax = rec.bbox_max.numpy().astype(np.float32)
            # -- display filter (mirrors the export noise filter's spirit) --
            if float((bmax - bmin).max()) > max_extent_m:
                continue
            if len(rec.voxels) < min_voxels:
                continue
            if rec.count <= 1 and (now - rec.last_seen) > stale_grace:
                continue
            if rec.count >= 2:
                vox = [v for v, n in rec.voxel_votes.items() if n >= vote_min]
            else:
                vox = list(rec.voxels)        # fresh singleton: show as-is
            if not vox:
                continue

            c = self._instance_color(int(iid))
            v = np.asarray(vox, dtype=np.float32)
            pts.append((v + 0.5) * vs)
            cols.append(np.repeat(c[None, :], len(v), axis=0))
            box_centers.append((bmin + bmax) * 0.5)
            box_half.append(np.maximum((bmax - bmin) * 0.5, 1e-3))
            box_cols.append(c)
            labels.append(str(int(iid)))

        if pts:
            rr.log("world/instances/voxels",
                   rr.Points3D(np.concatenate(pts),
                               colors=np.concatenate(cols),
                               radii=vs * radius_scale))
        if box_centers:
            rr.log("world/instances/boxes",
                   rr.Boxes3D(centers=np.stack(box_centers),
                              half_sizes=np.stack(box_half),
                              colors=np.stack(box_cols),
                              labels=labels,
                              show_labels=False))   # ids visible on hover/toggle


def _primary_ip() -> str:
    """Best-effort primary outbound IPv4 (so the browser can reach this server's data stream)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def graph_center(graph, key) -> np.ndarray:
    return graph.get_pose(key)[:3, 3].astype(np.float32)


def make_viz(mode: str, addr: Optional[str], *, conf_coef: float = 1.0,
             max_points: int = 500_000) -> Optional[RerunViz]:
    """Return a RerunViz, or None when viz is disabled."""
    if not mode or mode == "none":
        return None
    return RerunViz(mode, addr, conf_coef=conf_coef, max_points=max_points)
