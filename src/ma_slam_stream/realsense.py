"""RealSense capture for ma_slam streaming — official ``pyrealsense2`` API.

Shared by both the in-process *local* source (``sources.LocalRealSenseSource``, camera on the
server) and the *laptop* client (``client.py``). Wraps the standard Intel pipeline:

* color (BGR8) + depth (Z16) at 848x480, depth **aligned to color** so a single ``K`` and a
  single resolution apply to both; the frame is then **center-cropped to 640x480** before
  export (crop, not resize — keeps fx/fy and depth values exact, only shifts cx/cy). ``K`` is
  adjusted for the crop, so what the server receives is already correct;
* a depth **visual preset** (default ``Medium Density``) + **projector/emitter** control;
* optional **infrared** streams (left/right Y8);
* the official **post-processing filter chain** on the aligned depth frame, in Intel's
  recommended order — threshold → [decimation] → depth→disparity → spatial → temporal →
  disparity→depth → [hole-filling]. Defaults (validated on the user's D4xx): **spatial +
  temporal ON, decimation + hole-filling OFF**, spatial mag=1/α=0.7/δ=8, temporal α=0.5/δ=12,
  range 0.2–4.0 m. Decimation off keeps depth at the color resolution the exported ``K`` assumes.

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

        cap = RealSenseCapture(want_depth=True, depth_max=4.0)
        cap.start()                       # sets cap.K (3x3) and cap.depth_scale (=1000.0)
        for f in cap.frames():            # f = {"bgr", "depth_mm", "ir"}
            ...
        cap.stop()

    Filter defaults match the user's validated D4xx config (Medium Density preset, spatial +
    temporal only). Override via the constructor or ``add_capture_args`` / ``capture_from_args``.
    """

    def __init__(self, width: int = 848, height: int = 480, fps: int = 30, *,
                 out_width: int = 640, out_height: int = 480,
                 want_depth: bool = True, enable_ir: bool = False, emitter: bool = True,
                 preset: Optional[str] = "Medium Density",
                 depth_max: Optional[float] = 4.0, depth_min: float = 0.2,
                 laser_power: Optional[float] = None,
                 spatial: bool = True, temporal: bool = True,
                 hole_filling: bool = False, decimation: bool = False,
                 spatial_mag: int = 1, spatial_alpha: float = 0.7, spatial_delta: float = 8.0,
                 temporal_alpha: float = 0.5, temporal_delta: float = 12.0,
                 decimation_mag: int = 2, hole_mode: int = 1):
        self.width, self.height, self.fps = width, height, fps
        # capture at (width,height) with the tuned config, then CENTER-CROP to (out_*) before
        # export. Crop (not resize) keeps fx/fy and depth values exact; only cx/cy shift.
        self.out_width, self.out_height = out_width, out_height
        self.want_depth = want_depth
        self.enable_ir = enable_ir
        self.emitter = emitter
        self.preset = preset
        self.depth_max = depth_max
        self.depth_min = depth_min
        self.laser_power = laser_power
        self.en_spatial, self.en_temporal = spatial, temporal
        self.en_hole, self.en_decimation = hole_filling, decimation
        self.spatial_mag, self.spatial_alpha, self.spatial_delta = spatial_mag, spatial_alpha, spatial_delta
        self.temporal_alpha, self.temporal_delta = temporal_alpha, temporal_delta
        self.decimation_mag, self.hole_mode = decimation_mag, hole_mode
        # populated by start()
        self.K: Optional[np.ndarray] = None
        self.depth_scale: float = 1000.0          # we always export mm
        self._pipeline = None
        self._align = None
        self._depth_units_m = 0.001
        self._crop = None                         # (x_off, y_off, w, h) or None

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

        # center-crop setup: shift the principal point by the crop offset; fx/fy unchanged.
        ow = self.out_width or self.width
        oh = self.out_height or self.height
        if ow < self.width or oh < self.height:
            x_off, y_off = (self.width - ow) // 2, (self.height - oh) // 2
            self._crop = (x_off, y_off, ow, oh)
            self.K[0, 2] -= x_off
            self.K[1, 2] -= y_off
            print(f"[realsense] capture {self.width}x{self.height} -> center-crop {ow}x{oh} "
                  f"(cx,cy -> {self.K[0,2]:.1f},{self.K[1,2]:.1f})")

        if self.want_depth:
            self._align = rs.align(rs.stream.color)
            dsensor = profile.get_device().first_depth_sensor()
            self._depth_units_m = dsensor.get_depth_scale()
            self._set_preset(dsensor)
            # projector / IR emitter (dot pattern improves stereo depth; turn off for clean IR).
            if dsensor.supports(rs.option.emitter_enabled):
                dsensor.set_option(rs.option.emitter_enabled, 1.0 if self.emitter else 0.0)
            if self.laser_power is not None and dsensor.supports(rs.option.laser_power):
                dsensor.set_option(rs.option.laser_power, float(self.laser_power))
            self._build_filters()
        return self

    def _set_preset(self, dsensor):
        """Select a depth visual_preset by matching its description (e.g. 'Medium Density')."""
        rs = self._rs
        if not self.preset or not dsensor.supports(rs.option.visual_preset):
            return
        rng = dsensor.get_option_range(rs.option.visual_preset)
        want = self.preset.lower()
        for i in range(int(rng.min), int(rng.max) + 1):
            try:
                desc = dsensor.get_option_value_description(rs.option.visual_preset, i)
            except Exception:
                continue
            if want in desc.lower():
                dsensor.set_option(rs.option.visual_preset, float(i))
                print(f"[realsense] visual_preset = {desc}")
                return
        print(f"[realsense] preset not found: {self.preset!r}")

    def _build_filters(self):
        rs = self._rs
        self._thresh = rs.threshold_filter()
        self._thresh.set_option(rs.option.min_distance, float(self.depth_min))
        if self.depth_max:
            self._thresh.set_option(rs.option.max_distance, float(self.depth_max))
        self._decimation = rs.decimation_filter()
        self._decimation.set_option(rs.option.filter_magnitude, self.decimation_mag)
        self._to_disp = rs.disparity_transform(True)
        self._to_depth = rs.disparity_transform(False)
        self._spatial = rs.spatial_filter()
        self._spatial.set_option(rs.option.filter_magnitude, self.spatial_mag)
        self._spatial.set_option(rs.option.filter_smooth_alpha, self.spatial_alpha)
        self._spatial.set_option(rs.option.filter_smooth_delta, self.spatial_delta)
        self._temporal = rs.temporal_filter()
        self._temporal.set_option(rs.option.filter_smooth_alpha, self.temporal_alpha)
        self._temporal.set_option(rs.option.filter_smooth_delta, self.temporal_delta)
        self._hole = rs.hole_filling_filter(self.hole_mode)

    def _apply_filters(self, depth):
        """Official order: threshold → [decimation] → disparity(spatial,temporal) → [hole]."""
        depth = self._thresh.process(depth)
        if self.en_decimation:
            depth = self._decimation.process(depth)
        if self.en_spatial or self.en_temporal:
            depth = self._to_disp.process(depth)
            if self.en_spatial:
                depth = self._spatial.process(depth)
            if self.en_temporal:
                depth = self._temporal.process(depth)
            depth = self._to_depth.process(depth)
        if self.en_hole:
            depth = self._hole.process(depth)
        return depth.as_depth_frame()

    def stop(self):
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def _crop_img(self, img):
        if self._crop is None or img is None:
            return img
        x, y, w, h = self._crop
        return img[y:y + h, x:x + w]

    # ------------------------------------------------------------------ frames
    def frames(self) -> Iterator[Dict]:
        while True:
            fs = self._pipeline.wait_for_frames()
            if self._align is not None:
                fs = self._align.process(fs)        # align depth into the color frame
            color = fs.get_color_frame()
            if not color:
                continue
            out: Dict = {"bgr": self._crop_img(np.asanyarray(color.get_data())),
                         "depth_mm": None, "ir": {}}
            if self.want_depth:
                depth = fs.get_depth_frame()
                if not depth:
                    continue
                depth = self._apply_filters(depth)  # post-processing on the aligned depth frame
                if not depth:
                    continue
                raw = np.asanyarray(depth.get_data()).astype(np.float32)
                dmm = np.clip(raw * self._depth_units_m * 1000.0, 0, 65535).astype(np.uint16)
                out["depth_mm"] = self._crop_img(dmm)
            if self.enable_ir:
                for idx in (1, 2):
                    irf = fs.get_infrared_frame(idx)
                    if irf:
                        out["ir"][idx] = self._crop_img(np.asanyarray(irf.get_data()))
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


# ------------------------------------------------------- keyframe gate (LK disparity)
# Standalone mirror of ``ma_slam/keyframe_flow.py`` so realsense.py + client.py stay
# copy-to-laptop with no repo imports. VGGT-SLAM-style: promote a frame to a keyframe when
# the mean LK optical-flow displacement of tracked corners vs the last keyframe exceeds
# ``min_disparity`` px — drops redundant low-parallax frames. Causal → streaming-safe.
_KF_MAX_CORNERS, _KF_QUALITY, _KF_MIN_DIST, _KF_BLOCK, _KF_MIN_TRACKED = 1000, 0.01, 8, 7, 10
_KF_LK = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


class DisparityKeyframer:
    """Stateful, causal LK-disparity keyframe selector; ``step(bgr)`` True iff keyframe."""

    def __init__(self, min_disparity: float = 25.0):
        self.min_disp = float(min_disparity)
        self.kf_gray: Optional[np.ndarray] = None
        self.kf_pts: Optional[np.ndarray] = None

    def _reinit(self, gray):
        self.kf_gray = gray
        self.kf_pts = cv2.goodFeaturesToTrack(gray, _KF_MAX_CORNERS, _KF_QUALITY,
                                              _KF_MIN_DIST, blockSize=_KF_BLOCK)

    def step(self, bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.kf_pts is None or len(self.kf_pts) < _KF_MIN_TRACKED:
            self._reinit(gray)
            return True
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.kf_gray, gray, self.kf_pts, None, **_KF_LK)
        st = st.flatten()
        good_kf, good_nxt = self.kf_pts[st == 1], nxt[st == 1]
        if len(good_kf) < _KF_MIN_TRACKED:
            self._reinit(gray)
            return True
        mean_disp = float(np.mean(np.linalg.norm(good_nxt - good_kf, axis=1)))
        if mean_disp > self.min_disp:
            self._reinit(gray)           # select + re-anchor onto this frame
            return True
        return False


def disparity_gate(frames: Iterator[Dict], min_disparity: float) -> Iterator[Dict]:
    """Yield only keyframes from a capture-dict iterator. ``min_disparity <= 0`` -> gate off."""
    if min_disparity <= 0:
        yield from frames
        return
    kf = DisparityKeyframer(min_disparity)
    for f in frames:
        if kf.step(f["bgr"]):
            yield f


def select_stream(cap, k: int, keyframe_disparity: float) -> Iterator[Dict]:
    """Producer selection pipeline: blur-select (sharpest of k) then LK keyframe gate."""
    return disparity_gate(blur_select(cap.frames(), k), keyframe_disparity)


# ---------------------------------------------------------------------- shared CLI args
def add_capture_args(ap: argparse.ArgumentParser):
    """RealSense capture flags shared by the local source (run_stream) and the laptop client.

    Defaults match the user's validated D4xx config (848x480, Medium Density, spatial+temporal,
    0.2–4.0 m). Only override when you need to.
    """
    g = ap.add_argument_group("RealSense capture")
    g.add_argument("--width", type=int, default=848, help="native capture width (tuned config)")
    g.add_argument("--height", type=int, default=480, help="native capture height")
    g.add_argument("--out_width", type=int, default=640,
                   help="center-crop width fed to SLAM (0 = keep native; fx/fy unchanged, cx shifts)")
    g.add_argument("--out_height", type=int, default=480, help="center-crop height fed to SLAM (0 = native)")
    g.add_argument("--fps", type=int, default=30)
    g.add_argument("--k", type=int, default=3,
                   help="blur window: keep the sharpest of every k frames (1 = every frame)")
    g.add_argument("--keyframe_disparity", type=float, default=25.0,
                   help="LK keyframe gate (px): drop frames whose mean optical-flow displacement "
                        "vs the last keyframe is below this. Default 25 (on, follows ma_slam); "
                        "0 = off. Runs after blur-select; submap_size then counts keyframes.")
    g.add_argument("--preset", default="Medium Density",
                   help="depth visual_preset (e.g. 'Medium Density', 'High Accuracy', 'High Density')")
    g.add_argument("--enable_ir", action="store_true",
                   help="also capture left/right infrared streams (saved/previewed, not fed to SLAM)")
    g.add_argument("--no_emitter", action="store_true",
                   help="turn the IR projector OFF (cleaner IR image, worse stereo depth)")
    g.add_argument("--laser_power", type=float, default=None, help="depth laser power (mW)")
    g.add_argument("--rs_depth_min", type=float, default=0.2, help="threshold-filter min distance (m)")
    g.add_argument("--rs_depth_max", type=float, default=4.0,
                   help="threshold-filter max distance (m); 0 = no far clip")
    # filter chain (defaults: spatial+temporal on, decimation+hole off)
    g.add_argument("--no_spatial", action="store_true", help="disable the spatial filter")
    g.add_argument("--no_temporal", action="store_true", help="disable the temporal filter")
    g.add_argument("--hole", action="store_true", help="enable hole-filling (off by default)")
    g.add_argument("--decimation", action="store_true", help="enable decimation (off; lowers depth res)")
    g.add_argument("--spatial_mag", type=int, default=1)
    g.add_argument("--spatial_alpha", type=float, default=0.7)
    g.add_argument("--spatial_delta", type=float, default=8.0)
    g.add_argument("--temporal_alpha", type=float, default=0.5)
    g.add_argument("--temporal_delta", type=float, default=12.0)


def capture_from_args(a, want_depth: bool) -> RealSenseCapture:
    return RealSenseCapture(
        width=a.width, height=a.height, out_width=a.out_width, out_height=a.out_height,
        fps=a.fps, want_depth=want_depth,
        enable_ir=a.enable_ir, emitter=not a.no_emitter, preset=a.preset,
        depth_max=(a.rs_depth_max if a.rs_depth_max > 0 else None), depth_min=a.rs_depth_min,
        laser_power=a.laser_power,
        spatial=not a.no_spatial, temporal=not a.no_temporal,
        hole_filling=a.hole, decimation=a.decimation,
        spatial_mag=a.spatial_mag, spatial_alpha=a.spatial_alpha, spatial_delta=a.spatial_delta,
        temporal_alpha=a.temporal_alpha, temporal_delta=a.temporal_delta)
