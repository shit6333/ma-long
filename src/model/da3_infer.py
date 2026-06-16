"""Per-chunk Depth-Anything-3 inference for ma_long (drop-in backbone alternative).

`Da3ChunkModel` mirrors :class:`model.ma_infer.MaChunkModel` — same
``infer_chunk(...) -> dict`` contract — so ma_slam (and the pipelines) can swap backbones.

It wraps the **DA3NESTED-GIANT-LARGE-1.1** model, which predicts **metric** depth + camera
poses + intrinsics from RGB alone (``is_metric=1``). This is the motivation: in **rgb mode**
DA3-nested is metric where MapAnything is not, so no per-submap scale correction is needed
(SE3 throughout). DA3 does **not** ingest depth, so only ``rgb`` / ``rgb+intr`` are supported.

The MapAnything-bundled ``DA3Wrapper`` is training-only (no ``.infer``); we use DA3's native
API (``thirdparty/Depth-Anything-3/src/depth_anything_3/api.py``).
"""

from __future__ import annotations

import os
import sys
import types
from typing import Dict, Optional, Sequence

import numpy as np
import torch

_DEFAULT_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def _ensure_da3_importable():
    # Optional export-only deps that break import but are unused for inference.
    for m in ("moviepy", "moviepy.editor"):
        sys.modules.setdefault(m, types.ModuleType(m))
    for _cand in (
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "thirdparty", "Depth-Anything-3", "src"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))), "Depth-Anything-3", "src"),
    ):
        if os.path.isdir(_cand) and _cand not in sys.path:
            sys.path.insert(0, _cand)


class Da3ChunkModel:
    """Frozen DA3-nested backbone with the MaChunkModel chunk-inference API (rgb / rgb+intr)."""

    SUPPORTED_MODES = ("rgb", "rgb+intr")

    def __init__(self, model_id: str = _DEFAULT_MODEL_ID, device: str = "cuda",
                 *, process_res: int = 504):
        _ensure_da3_importable()
        from depth_anything_3.api import DepthAnything3

        self.device = device
        self.model_id = model_id
        self.process_res = process_res
        self.model = DepthAnything3.from_pretrained(model_id).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def is_metric(self, mode: str) -> bool:
        return True   # DA3-nested predicts metric scale (incl. rgb)

    @staticmethod
    def _unproject(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
        """Dense (N,H,W,3) world points from metric depth + per-frame K + c2w (OpenCV)."""
        N, H, W = depth.shape
        vs, us = np.meshgrid(np.arange(H, dtype=np.float32),
                             np.arange(W, dtype=np.float32), indexing="ij")
        out = np.empty((N, H, W, 3), np.float32)
        for i in range(N):
            z = depth[i]
            fx, fy, cx, cy = K[i, 0, 0], K[i, 1, 1], K[i, 0, 2], K[i, 1, 2]
            cam = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], axis=-1)  # (H,W,3)
            camh = np.concatenate([cam, np.ones((H, W, 1), np.float32)], axis=-1)  # (H,W,4)
            out[i] = (camh @ c2w[i].T)[..., :3]
        return out

    @torch.no_grad()
    def infer_chunk(self, image_paths: Sequence[str], *, mode: str = "rgb",
                    depth_paths: Optional[Sequence[str]] = None,
                    intrinsics: Optional[np.ndarray] = None,
                    depth_scale: float = 1000.0,
                    depth_max: Optional[float] = None) -> Dict[str, torch.Tensor]:
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Da3ChunkModel supports {self.SUPPORTED_MODES} (no depth input); got {mode!r}")
        K_in = None
        if "intr" in mode and intrinsics is not None:
            K = np.asarray(intrinsics, dtype=np.float32)
            n = len(image_paths)
            K_in = np.broadcast_to(K, (n, 3, 3)).copy() if K.shape == (3, 3) else K

        pred = self.model.inference(list(image_paths), intrinsics=K_in,
                                    process_res=self.process_res)
        depth = np.ascontiguousarray(pred.depth, dtype=np.float32)          # (N,H,W) metric Z
        conf = np.ascontiguousarray(pred.conf, dtype=np.float32)            # (N,H,W)
        Kp = np.ascontiguousarray(pred.intrinsics, dtype=np.float64)        # (N,3,3) at proc res
        imgs = np.ascontiguousarray(pred.processed_images).astype(np.uint8)  # (N,H,W,3)

        N = depth.shape[0]
        w2c = np.tile(np.eye(4), (N, 1, 1))
        w2c[:, :3, :4] = pred.extrinsics                                   # DA3 extrinsics are w2c (3x4)
        c2w = np.linalg.inv(w2c)
        # canonicalize so the chunk is in frame-0's coordinates (poses[0] ≈ I), matching MapAnything.
        poses_local = (np.linalg.inv(c2w[0])[None] @ c2w).astype(np.float64)
        world_points = self._unproject(depth, Kp, poses_local)

        t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        return {
            "world_points": t(world_points).float(),
            "world_points_conf": t(conf).float(),
            "depth": t(depth).float(),
            "poses": t(poses_local).float(),
            "intrinsics": t(Kp).float(),
            "images": t(imgs),
            "mask": None,
        }
