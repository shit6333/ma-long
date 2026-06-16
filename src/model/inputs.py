"""Multi-modal input assembly for ma_long.

Builds MapAnything-ready ``views`` (with consistent resolution across RGB / depth /
intrinsics) for the four supported input modes:

    rgb            — image only
    rgb+intr       — image + camera intrinsics (calibration hint)
    rgb+depth      — image + metric depth (intrinsics still needed to interpret
                     depth as a ray-scaled point map; supplied but not used as a
                     pose/calibration hint — see ma_infer.MODE_FLAGS)
    rgb+depth+intr — image + metric depth + intrinsics

The heavy lifting (aspect-ratio selection, joint resize of image+depth, intrinsic
rescaling, normalization, batch-dim insertion) is delegated to MapAnything's own
``preprocess_inputs`` so the depth/intrinsics stay pixel-consistent with the
resized RGB the encoder actually sees.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

# Modes that require depth / intrinsics to be available.
MODES = ("rgb", "rgb+intr", "rgb+depth", "rgb+depth+intr")
_MODES_NEED_DEPTH = ("rgb+depth", "rgb+depth+intr")
_MODES_NEED_INTR = ("rgb+intr", "rgb+depth", "rgb+depth+intr")  # depth needs K too


def load_intrinsics(path: str) -> np.ndarray:
    """Load a 3x3 pinhole K from a whitespace-separated matrix file.

    Accepts 3x3 or 4x4 (e.g. ScanNet ``intrinsic.txt``); the top-left 3x3 is used.
    """
    mat = np.loadtxt(path).astype(np.float32)
    if mat.shape[0] >= 3 and mat.shape[1] >= 3:
        return np.ascontiguousarray(mat[:3, :3])
    raise ValueError(f"Intrinsics file {path} has unexpected shape {mat.shape}")


def load_depth(path: str, scale: float = 1000.0, max_depth: Optional[float] = None) -> np.ndarray:
    """Load a single-channel depth map as float32 metres.

    Defaults to the ScanNet convention: 16-bit PNG in millimetres (``scale=1000``).
    Zero (missing) pixels are preserved as 0.0 — MapAnything treats them as no-op.
    ``max_depth`` (metres) zeros out farther pixels — useful to drop unreliable far-range
    sensor noise (e.g. RealSense beyond ~5-6 m), which MapAnything would otherwise trust.
    """
    depth = np.array(Image.open(path)).astype(np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth {path} expected (H, W), got shape {depth.shape}")
    depth = depth / float(scale)
    if max_depth is not None:
        depth[depth > max_depth] = 0.0
    return depth


def load_rgb(path: str) -> np.ndarray:
    """Load an RGB image as uint8 (H, W, 3)."""
    return np.array(Image.open(path).convert("RGB"))


def build_chunk_views(
    image_paths: Sequence[str],
    mode: str = "rgb",
    *,
    depth_paths: Optional[Sequence[str]] = None,
    intrinsics: Optional[np.ndarray] = None,
    depth_scale: float = 1000.0,
    depth_max: Optional[float] = None,
    resolution_set: int = 518,
    norm_type: str = "dinov2",
) -> List[dict]:
    """Assemble model-ready MapAnything views for one chunk.

    Args:
        image_paths: RGB frame paths for this chunk.
        mode: one of :data:`MODES`.
        depth_paths: per-frame depth paths (required for ``*depth*`` modes); must
            align 1:1 with ``image_paths``.
        intrinsics: a single (3, 3) K shared by the chunk, or an (N, 3, 3) array of
            per-frame K. Required for any mode using depth or intrinsics.
        depth_scale: divisor to convert raw depth to metres (ScanNet: 1000).
        resolution_set / norm_type: MapAnything preprocessing knobs.

    Returns:
        List of view dicts (img/depth_z/intrinsics already resized + batched),
        ready to pass straight to ``MapAnything.infer``.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")

    n = len(image_paths)
    need_depth = mode in _MODES_NEED_DEPTH
    need_intr = mode in _MODES_NEED_INTR

    if need_depth and (depth_paths is None or len(depth_paths) != n):
        raise ValueError(f"mode {mode!r} needs one depth path per image ({n}).")
    if need_intr and intrinsics is None:
        raise ValueError(
            f"mode {mode!r} needs intrinsics (depth is interpreted via K)."
        )

    # Normalize intrinsics to per-frame (N, 3, 3).
    K = None
    if need_intr:
        K = np.asarray(intrinsics, dtype=np.float32)
        if K.shape == (3, 3):
            K = np.broadcast_to(K, (n, 3, 3)).copy()
        elif K.shape != (n, 3, 3):
            raise ValueError(f"intrinsics must be (3,3) or ({n},3,3), got {K.shape}")

    # Defer to MapAnything's joint preprocessing for consistent resizing.
    from mapanything.utils.image import preprocess_inputs

    raw_views: List[dict] = []
    for i, img_path in enumerate(image_paths):
        view: dict = {"img": load_rgb(img_path), "idx": i, "instance": str(i)}
        if need_intr:
            view["intrinsics"] = K[i]
        if need_depth:
            view["depth_z"] = load_depth(depth_paths[i], scale=depth_scale, max_depth=depth_max)
            # Must be a pre-batched (1,) bool tensor: preprocess_inputs copies this
            # key verbatim (no batch-dim insertion) and the model indexes it as (B,).
            view["is_metric_scale"] = torch.ones(1, dtype=torch.bool)
        raw_views.append(view)

    views = preprocess_inputs(
        raw_views,
        resize_mode="fixed_mapping",
        resolution_set=resolution_set,
        norm_type=norm_type,
    )
    return views
