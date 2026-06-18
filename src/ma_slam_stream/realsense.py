"""RealSense capture for ma_slam streaming — official ``pyrealsense2`` API.

Shared by both the in-process *local* source (``sources.LocalRealSenseSource``, camera on the
server) and the *laptop* client (``client.py``). Wraps the standard Intel pipeline:

* color (BGR8) + depth (Z16), depth **aligned to color** so a single ``K`` and a single
  resolution apply to both;
* optional **infrared** streams (left/right Y8) and **projector/emitter** control;
* the official **post-processing filter chain** applied to the (aligned) depth frame —
  threshold → depth→disparity → spatial → temporal → disparity→depth → hole-filling
  (the order Intel recommends in their post-processing tutorial). Decimation is intentionally
  omitted: it lowers depth resolution, which would no longer match the color ``K`` we export.

Depth is normalised to **uint16 millimetres** regardless of the sensor's native depth unit,
so downstream always uses ``depth_scale=1000``.

This module depends only on ``pyrealsense2``, ``numpy`` and ``opencv-python`` — no repo
imports — so ``realsense.py`` + ``client.py`` can be copied to the laptop on their own.
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterator, Optional

import cv2
import numpy as np


def laplacian_var(bgr: np.ndarray) -> float:
    """Sharpness score: variance of the Laplacian on grayscale (higher = sharper / less blur)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class RealSenseCapture:
    """Thin wrapper over the official RealSense pipeline producing aligned RGBD (+ optional IR).

    Usage::

        cap = RealSenseCapture(want_depth=True, postprocess=True, depth_max=6.0)
        cap.start()                       # sets cap.K (3x3) and cap.depth_scale (=1000.0)
        for f in cap.frames():            # f = {"bgr", "depth_mm", "ir"}
            ...
        cap.stop()
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, *,
                 want_depth: bool = True, enable_ir: bool = False, emitter: bool = True,
                 postprocess: bool = True, depth_max: Optional[float] = None,
                 depth_min: float = 0.1, laser_power: Optional[float] = None):
        self.width, self.height, self.fps = width, height, fps
        self.want_depth = want_depth
        self.enable_ir = enable_ir
        self.emitter = emitter
        self.postprocess = postprocess
        self.depth_max = depth_max
        self.depth_min = depth_min
        self.laser_power = laser_power
        # populated by start()
        self.K: Optional[np.ndarray] = None
        self.depth_scale: float = 1000.0          # we always export mm
        self._pipeline = None
        self._align = None
        self._filters = []
        self._depth_units_m = 0.001

    # ------------------------------------------------------------------ lifecycle
    def start(self):
        import pyrealsense2 as rs
        self._rs = rs
        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.want_depth:
            cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        if self.enable_ir:
            cfg.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
            cfg.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps)
        profile = self._pipeline.start(cfg)

        # intrinsics from the COLOR stream (depth is aligned to color).
        cprof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        ci = cprof.get_intrinsics()
        self.K = np.array([[ci.fx, 0, ci.ppx], [0, ci.fy, ci.ppy], [0, 0, 1]], dtype=np.float32)

        if self.want_depth:
            self._align = rs.align(rs.stream.color)
            dsensor = profile.get_device().first_depth_sensor()
            self._depth_units_m = dsensor.get_depth_scale()
            # projector / IR emitter (dot pattern improves stereo depth; turn off for clean IR).
            if dsensor.supports(rs.option.emitter_enabled):
                dsensor.set_option(rs.option.emitter_enabled, 1.0 if self.emitter else 0.0)
            if self.laser_power is not None and dsensor.supports(rs.option.laser_power):
                dsensor.set_option(rs.option.laser_power, float(self.laser_power))
            if self.postprocess:
                self._build_filters()
        return self

    def _build_filters(self):
        rs = self._rs
        thresh = rs.threshold_filter()
        thresh.set_option(rs.option.min_distance, float(self.depth_min))
        if self.depth_max:
            thresh.set_option(rs.option.max_distance, float(self.depth_max))
        # official recommended chain (spatial/temporal operate in the disparity domain).
        self._filters = [
            thresh,
            rs.disparity_transform(True),    # depth -> disparity
            rs.spatial_filter(),             # edge-preserving smoothing
            rs.temporal_filter(),            # multi-frame smoothing
            rs.disparity_transform(False),   # disparity -> depth
            rs.hole_filling_filter(),
        ]

    def stop(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    # ------------------------------------------------------------------ frames
    def frames(self) -> Iterator[Dict]:
        rs = self._rs
        while True:
            fs = self._pipeline.wait_for_frames()
            if self._align is not None:
                fs = self._align.process(fs)        # align depth into the color frame
            color = fs.get_color_frame()
            if not color:
                continue
            out: Dict = {"bgr": np.asanyarray(color.get_data()), "depth_mm": None, "ir": {}}
            if self.want_depth:
                depth = fs.get_depth_frame()
                if not depth:
                    continue
                for f in self._filters:             # post-processing on the aligned depth frame
                    depth = f.process(depth)
                raw = np.asanyarray(depth.get_data()).astype(np.float32)
                out["depth_mm"] = np.clip(raw * self._depth_units_m * 1000.0,
                                          0, 65535).astype(np.uint16)
            if self.enable_ir:
                for idx in (1, 2):
                    irf = fs.get_infrared_frame(idx)
                    if irf:
                        out["ir"][idx] = np.asanyarray(irf.get_data())
            yield out


def blur_select(frames: Iterator[Dict], k: int) -> Iterator[Dict]:
    """Keep the sharpest frame of every ``k`` (k=1 -> pass every frame through).

    Wraps any iterator of capture dicts; the chosen dict carries an added ``"sharpness"`` key.
    """
    if k <= 1:
        for f in frames:
            f["sharpness"] = laplacian_var(f["bgr"])
            yield f
        return
    best, win = None, 0
    for f in frames:
        s = laplacian_var(f["bgr"])
        if best is None or s > best[0]:
            best = (s, f)
        win += 1
        if win >= k:
            chosen = best[1]
            chosen["sharpness"] = best[0]
            yield chosen
            best, win = None, 0


# ---------------------------------------------------------------------- shared CLI args
def add_capture_args(ap: argparse.ArgumentParser):
    """RealSense capture flags shared by the local source (run_stream) and the laptop client."""
    g = ap.add_argument_group("RealSense capture")
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=480)
    g.add_argument("--fps", type=int, default=15)
    g.add_argument("--k", type=int, default=5,
                   help="blur window: keep the sharpest of every k frames (1 = every frame)")
    g.add_argument("--enable_ir", action="store_true",
                   help="also capture left/right infrared streams (saved/previewed, not fed to SLAM)")
    g.add_argument("--no_emitter", action="store_true",
                   help="turn the IR projector OFF (cleaner IR image, worse stereo depth)")
    g.add_argument("--no_postprocess", action="store_true",
                   help="disable the official depth post-processing filter chain")
    g.add_argument("--laser_power", type=float, default=None, help="depth laser power (mW)")
    g.add_argument("--rs_depth_min", type=float, default=0.1, help="threshold-filter min distance (m)")
    g.add_argument("--rs_depth_max", type=float, default=0.0,
                   help="threshold-filter max distance (m); 0 = filter off (still send raw depth)")


def capture_from_args(a, want_depth: bool) -> RealSenseCapture:
    return RealSenseCapture(
        width=a.width, height=a.height, fps=a.fps, want_depth=want_depth,
        enable_ir=a.enable_ir, emitter=not a.no_emitter, postprocess=not a.no_postprocess,
        depth_max=(a.rs_depth_max if a.rs_depth_max > 0 else None),
        depth_min=a.rs_depth_min, laser_power=a.laser_power)
