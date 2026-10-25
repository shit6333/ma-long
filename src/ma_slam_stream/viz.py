"""Real-time point-cloud / trajectory visualisation for ma_slam streaming, via rerun.

Decouples viewing from compute: the server logs, you watch on any machine. Three sinks
(``--viz``):

* ``spawn``   — launch a native rerun viewer on THIS machine (server needs a display).
* ``web``     — serve the rerun web viewer from the server; open the printed URL in a browser
                anywhere (the laptop needs nothing but a browser).
* ``connect`` — stream to an already-running rerun viewer elsewhere (e.g. ``rerun`` on your
                laptop); point ``--viz_addr`` at it (``rr+http://LAPTOP_IP:9876``).

Per-frame streaming view (default ``reveal='paced'``):
* every arriving keyframe is shown immediately in the ``live/rgb`` panel (``log_live_frame``,
  called from a tap thread, so it never waits for reconstruction);
* reconstruction is per submap, but its result is revealed FRAME BY FRAME on the ``frame``
  timeline: per-frame points ``world/points/f<idx>``, the camera frustum with its image, the
  trajectory, and an instance-layer snapshot after every semantic chunk. A background thread
  spreads one submap's frames over the time the submap took to compute, so the 3D view advances
  continuously, one submap behind live. ``reveal='instant'`` logs them without pacing.
* when a loop closes, every frame's points are re-logged with the corrected poses.

Depends only on ``rerun-sdk`` (already in the repo env, 0.26.2). No-op if ``--viz none``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class RerunViz:
    def __init__(self, mode: str, addr: Optional[str] = None, *, conf_coef: float = 1.0,
                 max_points: int = 500_000, app_id: str = "ma_slam_stream",
                 reveal: str = "paced", inst_style: str = "points", inst_alpha: float = 0.5,
                 inst_radius: float = 0.25):
        import queue
        import threading
        import rerun as rr
        self.rr = rr
        self.conf_coef = conf_coef
        self.max_points = max_points
        self._logged: set = set()              # submap base_ids already logged (points)
        self._rng = np.random.default_rng(0)
        # instance layer style: 'points' = coloured voxel-centre points with radius
        # inst_radius x voxel (0.25 default; 0.5 = neighbouring points touch);
        # 'voxels' = solid translucent cubes (alpha = inst_alpha; nicer but heavier)
        self.inst_style = inst_style
        self.inst_radius = float(inst_radius)
        self.inst_alpha = int(round(255 * float(np.clip(inst_alpha, 0.0, 1.0))))
        # per-frame reveal state
        self.reveal = reveal
        self.frame_points = max(2000, max_points // 64)   # points per revealed frame
        self.arrival: dict = {}                # image path -> arrival index (set by the tap)
        self._frames: dict = {}                # image path -> (submap, frame index), latest owner
        self._shown: set = set()               # image paths already revealed
        self._pending_inst: list = []          # (idx, instance payload) from semantic chunks
        self._centers: dict = {}               # idx -> camera centre (trajectory)
        self._q: "queue.Queue" = queue.Queue()
        self._worker = threading.Thread(target=self._reveal_loop, daemon=True)
        self._worker.start()

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
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        origin="world",
                        overrides={
                            "world/instances/boxes": rrb.EntityBehavior(visible=False),
                        },
                    ),
                    rrb.Spatial2DView(origin="live/rgb", name="live keyframes"),
                    column_shares=[3, 1],
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
                         radius_scale: Optional[float] = None):
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
        p = self._instance_payload(table, max_extent_m=max_extent_m, min_voxels=min_voxels,
                                   stale_grace=stale_grace, vote_min=vote_min,
                                   radius_scale=radius_scale)
        if p is None:
            return
        self.rr.set_time("frame", sequence=n_frames)
        self._log_instances(p)

    def _instance_payload(self, table, *, max_extent_m: float = 4.0, min_voxels: int = 5,
                          stale_grace: int = 40, vote_min: int = 2,
                          radius_scale: Optional[float] = None):
        """Snapshot the display arrays of the instance layer (see ``update_instances``)."""
        if table is None or len(table) == 0:
            return None
        if radius_scale is None:
            radius_scale = self.inst_radius
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

        if not pts:
            return None
        return dict(pts=np.concatenate(pts), cols=np.concatenate(cols), radius=vs * radius_scale,
                    voxel=vs,
                    box_centers=np.stack(box_centers), box_half=np.stack(box_half),
                    box_cols=np.stack(box_cols), labels=labels)

    def _log_instances(self, p):
        rr = self.rr
        if self.inst_style == "voxels":
            # one solid cube per voxel, slightly shrunk so the grid stays readable;
            # the alpha channel is the transparency in rerun's solid fill mode
            n = len(p["pts"])
            rgba = np.concatenate([p["cols"], np.full((n, 1), self.inst_alpha, np.uint8)], 1)
            rr.log("world/instances/voxels",
                   rr.Boxes3D(centers=p["pts"], half_sizes=np.full((n, 3), 0.45 * p["voxel"], np.float32),
                              colors=rgba, fill_mode="solid"))
        else:
            rr.log("world/instances/voxels",
                   rr.Points3D(p["pts"], colors=p["cols"], radii=p["radius"]))
        rr.log("world/instances/boxes",
               rr.Boxes3D(centers=p["box_centers"], half_sizes=p["box_half"],
                          colors=p["box_cols"], labels=p["labels"],
                          show_labels=False))       # ids visible on hover/toggle

    # ------------------------------------------------------------------
    # Per-frame streaming view
    # ------------------------------------------------------------------
    def log_live_frame(self, path: str, idx: int):
        """Show an arriving keyframe right away (called from the source tap thread)."""
        rr = self.rr
        rr.set_time("frame", sequence=idx)    # rerun time context is thread-local
        rr.log("live/rgb", rr.EncodedImage(path=path))

    def collect_instances(self, table, paths):
        """Semantic-hook callback after every chunk: snapshot the instance layer at the
        chunk's last frame so it is revealed in step with the geometry."""
        idx = max((self.arrival.get(p, -1) for p in paths), default=-1)
        p = self._instance_payload(table)
        if idx >= 0 and p is not None:
            self._pending_inst.append((idx, p))

    def _frame_payload(self, sm, graph, i: int):
        G = graph.get_pose(sm.key(i))
        M = G @ np.linalg.inv(sm.poses_local[i])          # submap-local -> global
        m = sm.conf[i] > sm.conf_threshold * self.conf_coef
        if sm.mask is not None:
            m &= sm.mask[i]
        Q, C = sm.points[i][m], sm.colors[i][m]
        if len(Q) > self.frame_points:
            sel = self._rng.choice(len(Q), self.frame_points, replace=False)
            Q, C = Q[sel], C[sel]
        P = (Q @ M[:3, :3].T + M[:3, 3]).astype(np.float32)
        return dict(P=P, C=np.asarray(C).astype(np.uint8), G=G,
                    K=np.asarray(sm.intrinsics[i], dtype=np.float32),
                    img=np.asarray(sm.colors[i]).astype(np.uint8))

    def push_submap(self, slam, sm, *, loop_happened: bool, compute_s: float):
        """Queue one reconstructed submap for frame-by-frame reveal (call after
        ``process_submap``; the semantic hook has already filled ``_pending_inst``)."""
        for i in range(sm.n):
            self._frames[sm.image_paths[i]] = (sm, i)
        events = []
        for i in range(sm.n):
            path = sm.image_paths[i]
            if path in self._shown:                        # shared overlap frame
                continue
            self._shown.add(path)
            idx = self.arrival.get(path)
            if idx is not None:
                events.append((idx, 0, "frame", self._frame_payload(sm, slam.graph, i)))
        events += [(idx, 1, "inst", p) for idx, p in self._pending_inst]
        self._pending_inst = []
        events.sort(key=lambda e: (e[0], e[1]))
        if loop_happened and events:                       # poses moved -> re-log everything
            allf = [(self.arrival[p], self._frame_payload(s, slam.graph, i))
                    for p, (s, i) in self._frames.items() if p in self.arrival and p in self._shown]
            events.append((events[-1][0], 2, "relog", allf))
        n = sum(1 for e in events if e[2] == "frame")
        delay = 0.0 if self.reveal == "instant" else float(np.clip(compute_s / max(1, n), 0.03, 1.5))
        self._q.put((events, delay))

    def flush(self):
        """Block until every queued frame has been revealed (call before shutdown)."""
        self._q.join()

    def _reveal_loop(self):
        import time
        while True:
            events, delay = self._q.get()
            try:
                for idx, _, kind, p in events:
                    self._emit(idx, kind, p)
                    if kind == "frame" and delay:
                        time.sleep(delay)
            except Exception as e:                         # display must never kill SLAM
                print(f"[viz] reveal error: {e}")
            finally:
                self._q.task_done()

    def _emit(self, idx, kind, p):
        rr = self.rr
        rr.set_time("frame", sequence=idx)
        if kind == "frame":
            rr.log(f"world/points/f{idx:05d}", rr.Points3D(p["P"], colors=p["C"]))
            G, K, img = p["G"], p["K"], p["img"]
            rr.log("world/camera", rr.Transform3D(translation=G[:3, 3], mat3x3=G[:3, :3]))
            h, w = img.shape[:2]
            rr.log("world/camera/image", rr.Pinhole(image_from_camera=K, resolution=[w, h],
                                                    image_plane_distance=0.25))
            rr.log("world/camera/image", rr.Image(img).compress(jpeg_quality=80))
            self._centers[idx] = G[:3, 3].astype(np.float32)
            self._log_traj()
        elif kind == "inst":
            self._log_instances(p)
        elif kind == "relog":
            for i, fp in p:
                rr.log(f"world/points/f{i:05d}", rr.Points3D(fp["P"], colors=fp["C"]))
                self._centers[i] = fp["G"][:3, 3].astype(np.float32)
            self._log_traj()

    def _log_traj(self):
        if len(self._centers) >= 2:
            path = np.stack([self._centers[k] for k in sorted(self._centers)])
            self.rr.log("world/trajectory", self.rr.LineStrips3D([path]))


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
             max_points: int = 500_000, reveal: str = "paced", inst_style: str = "points",
             inst_alpha: float = 0.5, inst_radius: float = 0.25) -> Optional[RerunViz]:
    """Return a RerunViz, or None when viz is disabled."""
    if not mode or mode == "none":
        return None
    return RerunViz(mode, addr, conf_coef=conf_coef, max_points=max_points, reveal=reveal,
                    inst_style=inst_style, inst_alpha=inst_alpha, inst_radius=inst_radius)
