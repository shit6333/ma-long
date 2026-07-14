"""VGGT-SLAM-style disparity keyframe gate (borrowed from ma_slam).

A frame is promoted to a keyframe when the mean Lucas-Kanade optical-flow
displacement of tracked Shi-Tomasi corners (w.r.t. the last keyframe) exceeds
``min_disparity`` pixels.  This drops redundant low-parallax frames so each frame
fed to the backbone carries genuine multi-view information — VGGT-SLAM Sec 4.1.

Borrowed verbatim from /mnt/HDD4/ricky/slam/ma_long/src/ma_slam/keyframe_flow.py
and extended with `select_keyframes_from_tensor` so OpenSe3r's SLAM run (which
holds a normalized [1,T,3,H,W] image tensor, not file paths) can use it.

The decision is CAUSAL (depends only on past frames): the streaming front-end and
the offline pipeline get identical selections.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import torch

# corner-detection / LK params (match VGGT-SLAM's FrameTracker)
_MAX_CORNERS = 1000
_QUALITY = 0.01
_MIN_DIST = 8
_BLOCK = 7
_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
_MIN_TRACKED = 10   # below this, force a keyframe + re-anchor (texture loss / heavy blur)


class DisparityKeyframer:
    """Stateful, causal LK-disparity keyframe selector.

    ``step(gray)`` returns True iff the frame is a keyframe (and re-anchors onto it).
    The very first frame is always a keyframe.
    """

    def __init__(self, min_disparity: float = 25.0):
        self.min_disp = float(min_disparity)
        self.kf_gray: Optional[np.ndarray] = None
        self.kf_pts: Optional[np.ndarray] = None

    def _reinit(self, gray: np.ndarray):
        self.kf_gray = gray
        self.kf_pts = cv2.goodFeaturesToTrack(gray, _MAX_CORNERS, _QUALITY, _MIN_DIST,
                                              blockSize=_BLOCK)

    def step(self, gray: np.ndarray) -> bool:
        if self.kf_pts is None or len(self.kf_pts) < _MIN_TRACKED:
            self._reinit(gray)
            return True
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.kf_gray, gray, self.kf_pts, None, **_LK)
        st = st.flatten()
        good_kf, good_nxt = self.kf_pts[st == 1], nxt[st == 1]
        if len(good_kf) < _MIN_TRACKED:
            self._reinit(gray)
            return True
        mean_disp = float(np.mean(np.linalg.norm(good_nxt - good_kf, axis=1)))
        if mean_disp > self.min_disp:
            self._reinit(gray)          # select + re-anchor onto this frame
            return True
        return False


def _tensor_to_gray_uint8(frame_chw: torch.Tensor) -> np.ndarray:
    """[3,H,W] normalized (~[-1,1]) → HxW uint8 gray.  Robust to the exact norm:
    LK tracks local gradients, so a fixed [-1,1]→[0,255] denorm is sufficient."""
    img = frame_chw.detach().float().cpu()
    img = (img * 0.5 + 0.5).clamp(0.0, 1.0)          # [-1,1] → [0,1]
    rgb = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)   # H,W,3 (RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def select_keyframes_from_tensor(images: torch.Tensor,
                                 min_disparity: float = 25.0) -> List[int]:
    """images: [1,T,3,H,W] or [T,3,H,W] normalized → indices of selected keyframes.

    First frame always kept; causal, so identical to a streaming decision."""
    if images.dim() == 5:
        assert images.shape[0] == 1, "batched keyframe selection not supported"
        images = images[0]
    kf = DisparityKeyframer(min_disparity)
    idx: List[int] = []
    for t in range(images.shape[0]):
        if kf.step(_tensor_to_gray_uint8(images[t])):
            idx.append(t)
    return idx


def select_keyframes_by_disparity(image_paths: List[str],
                                  min_disparity: float = 25.0) -> List[int]:
    """Offline file-path wrapper (kept for parity with ma_slam)."""
    kf = DisparityKeyframer(min_disparity)
    idx: List[int] = []
    for i, p in enumerate(image_paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read {p}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if kf.step(gray):
            idx.append(i)
    return idx
