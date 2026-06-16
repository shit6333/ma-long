"""Per-chunk MapAnything inference for ma_long.

`MaChunkModel` wraps a frozen MapAnything backbone and runs one *chunk* (a window
of consecutive frames) through it, returning a standardized dict of point maps,
confidences, depth, c2w poses and intrinsics — the common currency the long
pipeline (chunking / overlap alignment / loop closure) consumes, independent of
which input modality was used.

Input modes (see :data:`MODE_FLAGS`) decide which optional signals MapAnything is
told to *use* via its ``ignore_*`` flags; the depth/intrinsics themselves are fed
through :func:`model.inputs.build_chunk_views`.

Coordinate convention (matches DA3 / OpenSe3r): without pose inputs, MapAnything
returns poses "in the frame of reference view 0", so ``poses[0] ≈ identity`` and
all geometry is chunk-local. ``camera_poses`` is OpenCV cam2world.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np
import torch

# Make the bundled map-anything importable both standalone (this repo) and when
# ma_long is vendored into another repo's thirdparty/ (same pattern OpenSe3r uses).
for _cand in (
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "thirdparty", "map-anything"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))), "map-anything"),
):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

from model.inputs import build_chunk_views

_DEFAULT_MODEL_ID = "facebook/map-anything"
# Repo-local HF cache (on the big disk) so ma_long is self-contained and doesn't fill the
# default ~/.cache. Override per call with cache_dir=...
_DEFAULT_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "hf_cache")

# mode -> which optional inputs MapAnything should USE.
# (depth always carries K for interpretation, but a mode may still ignore K as a
#  calibration/pose hint — that's the rgb+depth case.)
MODE_FLAGS: Dict[str, Dict[str, bool]] = {
    "rgb":            {"ignore_calibration_inputs": True,  "ignore_depth_inputs": True},
    "rgb+intr":       {"ignore_calibration_inputs": False, "ignore_depth_inputs": True},
    "rgb+depth":      {"ignore_calibration_inputs": True,  "ignore_depth_inputs": False},
    "rgb+depth+intr": {"ignore_calibration_inputs": False, "ignore_depth_inputs": False},
}


class MaChunkModel:
    """Frozen MapAnything backbone with a chunk-level multi-modal inference API."""

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        device: str = "cuda",
        *,
        cache_dir: Optional[str] = None,
        memory_efficient_inference: bool = True,
        apply_mask: bool = False,
        mask_edges: bool = False,
        apply_confidence_mask: bool = False,
        confidence_percentile: float = 10.0,
        amp_dtype: str = "bf16",
    ):
        from mapanything.models import MapAnything

        self.device = device
        self.model_id = model_id
        cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        self.model = MapAnything.from_pretrained(
            model_id, cache_dir=cache_dir).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # `infer` knobs shared across modes. Masking is OFF by default so dense
        # point maps are preserved for overlap alignment / pointcloud export;
        # we filter by confidence downstream instead.
        self._base_infer_kwargs = dict(
            memory_efficient_inference=memory_efficient_inference,
            use_amp=True,
            amp_dtype=amp_dtype,
            apply_mask=apply_mask,
            mask_edges=mask_edges,
            apply_confidence_mask=apply_confidence_mask,
            confidence_percentile=confidence_percentile,
            ignore_pose_inputs=True,        # we never feed poses (chunk-local output)
            ignore_pose_scale_inputs=True,
            ignore_depth_scale_inputs=False,  # honour metric depth scale when given
        )

    def is_metric(self, mode: str) -> bool:
        """MapAnything is metric only when depth is fed (rgb-only output drifts in scale)."""
        return "depth" in mode

    @torch.no_grad()
    def infer_chunk(
        self,
        image_paths: Sequence[str],
        *,
        mode: str = "rgb",
        depth_paths: Optional[Sequence[str]] = None,
        intrinsics: Optional[np.ndarray] = None,
        depth_scale: float = 1000.0,
        depth_max: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run one chunk through MapAnything.

        Returns a dict of CPU float32 tensors:
            world_points      (N, H, W, 3)  chunk-local world frame
            world_points_conf (N, H, W)
            depth             (N, H, W)     camera-frame Z, metric
            poses             (N, 4, 4)     c2w (OpenCV), poses[0] ≈ I
            intrinsics        (N, 3, 3)
            images            (N, H, W, 3)  uint8 RGB (for pointcloud colour)
            mask              (N, H, W)     bool, or None if not produced
        """
        if mode not in MODE_FLAGS:
            raise ValueError(f"Unknown mode {mode!r}; expected {tuple(MODE_FLAGS)}")

        views = build_chunk_views(
            image_paths, mode,
            depth_paths=depth_paths, intrinsics=intrinsics, depth_scale=depth_scale,
            depth_max=depth_max,
        )
        for v in views:  # move tensors onto the model device
            for k, val in v.items():
                if torch.is_tensor(val):
                    v[k] = val.to(self.device)

        infer_kwargs = {**self._base_infer_kwargs, **MODE_FLAGS[mode]}
        # MapAnything's post-processing calls torch.linalg.solve (no bf16 cuSOLVER
        # path); disable any outer autocast and let infer manage its own AMP.
        with torch.amp.autocast("cuda", enabled=False):
            preds = self.model.infer(views, **infer_kwargs)

        def stack(key, fn=lambda t: t):
            return torch.stack([fn(p[key])[0].float().cpu() for p in preds], dim=0)

        out = {
            "world_points": stack("pts3d"),                         # (N,H,W,3)
            "world_points_conf": stack("conf"),                     # (N,H,W)
            "depth": stack("depth_z", lambda t: t.squeeze(-1)),     # (N,H,W)
            "poses": stack("camera_poses"),                         # (N,4,4)
            "intrinsics": stack("intrinsics"),                      # (N,3,3)
            "images": stack("img_no_norm",
                            lambda t: (t.clamp(0, 1) * 255).round().to(torch.uint8)),
        }
        out["mask"] = (
            stack("mask", lambda t: t.squeeze(-1)).bool()
            if "mask" in preds[0] and preds[0]["mask"] is not None else None
        )
        return out
