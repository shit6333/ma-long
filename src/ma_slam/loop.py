"""Online visual-place-recognition loop detection for ma_slam.

VGGT-SLAM keeps a growing database of per-frame SALAD/DINOv2 descriptors and, for each
new submap, retrieves the most similar earlier frame as a loop candidate. We reuse the
SALAD ``VPRModel`` already vendored in ``ma_long/loop/`` and maintain an incremental
faiss inner-product index (descriptors are L2-normalized → cosine similarity).

Returns candidate (query_frame, matched_frame) pairs; geometric verification (re-infer
the pair and check co-location) is done in the solver, mirroring VGGT-SLAM's
``image_match_ratio`` gate but on metric MapAnything geometry.
"""

from __future__ import annotations

import os
from typing import List, NamedTuple, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from loop.loop_detector import VPRModel

_DEF_CKPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights", "dino_salad.ckpt")


class LoopCandidate(NamedTuple):
    similarity: float
    query_base: int       # submap base id of the query (current) submap
    query_frame: int      # frame index within the query submap
    match_base: int       # submap base id of the retrieved (older) submap
    match_frame: int      # frame index within the retrieved submap


class ImageRetrieval:
    def __init__(self, ckpt_path: str = _DEF_CKPT, image_size=(322, 322),
                 device: str = "cuda", min_submap_gap: int = 1):
        self.ckpt_path = ckpt_path
        self.image_size = tuple(image_size)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.min_submap_gap = min_submap_gap
        self.model: Optional[VPRModel] = None
        self._tf = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # database
        self._descs: List[np.ndarray] = []        # each (D,)
        self._meta: List[tuple] = []              # (base_id, frame_idx)

    def _load(self):
        model = VPRModel(
            backbone_arch="dinov2_vitb14",
            backbone_config={"num_trainable_blocks": 4, "return_token": True, "norm_layer": True},
            agg_arch="SALAD",
            agg_config={"num_channels": 768, "num_clusters": 64, "cluster_dim": 128, "token_dim": 256},
        )
        model.load_state_dict(torch.load(self.ckpt_path, map_location="cpu"))
        self.model = model.eval().to(self.device)

    @torch.no_grad()
    def describe(self, image_paths) -> np.ndarray:
        """L2-normalized SALAD descriptors for a list of image paths -> (N, D)."""
        if self.model is None:
            self._load()
        imgs = torch.stack([self._tf(Image.open(p).convert("RGB")) for p in image_paths]).to(self.device)
        with torch.autocast(device_type="cuda" if self.device == "cuda" else "cpu", dtype=torch.float16):
            d = self.model(imgs).float().cpu().numpy()
        d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
        return d

    def query(self, descs: np.ndarray, query_base: int, sim_threshold: float,
              recent_bases: set) -> List[LoopCandidate]:
        """Best loop candidate per query frame against the database (excluding recent submaps)."""
        if not self._descs:
            return []
        db = np.stack(self._descs)                     # (M, D)
        sims = descs @ db.T                            # cosine (M normalized)
        cands: List[LoopCandidate] = []
        for qi in range(descs.shape[0]):
            order = np.argsort(-sims[qi])
            for mi in order:
                base, fidx = self._meta[mi]
                if base in recent_bases:
                    continue
                s = float(sims[qi, mi])
                if s >= sim_threshold:
                    cands.append(LoopCandidate(s, query_base, qi, base, fidx))
                break
        return cands

    def add(self, descs: np.ndarray, base_id: int):
        for i in range(descs.shape[0]):
            self._descs.append(descs[i])
            self._meta.append((base_id, i))
