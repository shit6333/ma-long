"""VGGT-SLAM-style disparity keyframe gate for ma_slam.

A frame is promoted to a keyframe when the mean Lucas-Kanade optical-flow displacement
of tracked Shi-Tomasi corners (w.r.t. the last keyframe) exceeds ``min_disparity`` pixels.
This drops redundant low-parallax frames so each frame fed to the backbone carries genuine
multi-view information — VGGT-SLAM Sec 4.1 (``thirdparty/VGGT-SLAM/vggt_slam/frame_overlap.py``).

The decision is CAUSAL (depends only on past frames), so the stateful ``DisparityKeyframer``
serves both the streaming front-end (call ``step`` per arriving frame) and the offline
pipeline (``select_keyframes_by_disparity`` loops it over a fixed list — identical result).
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

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

    ``step(frame_bgr)`` returns True iff the frame is a keyframe (and re-anchors onto it).
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

    def step(self, frame_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
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


def select_keyframes_by_disparity(image_paths: List[str], min_disparity: float = 25.0) -> List[int]:
    """Offline wrapper: indices of selected keyframes (causal → identical to streaming)."""
    kf = DisparityKeyframer(min_disparity)
    idx: List[int] = []
    for i, p in enumerate(image_paths):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"cannot read {p}")
        if kf.step(img):
            idx.append(i)
    return idx
