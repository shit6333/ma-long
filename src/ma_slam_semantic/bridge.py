"""Geometry -> semantic-model tensor bridge.

Two sources produce the same chunk contract (see ``sem_model``):

- ``gt_chunks`` — golden-test mode: chunks straight from an OpenSe3r GT dataset batch
  (bypasses ma_slam entirely; must reproduce the reference 005 GT run).
- ``SubmapBridge`` — ma_slam mode: converts a post-optimize ``Submap`` + pose graph
  into chunks (the old ``update_gt`` external-geometry contract), handling resolution
  remap (MA grid -> 518x336), conf->validity remap, gauge anchoring (sequence frame 0
  == identity) and overlap-frame dedup (submaps share 1 frame; feed frames [1:] of
  every submap after the first so the causal stream has no duplicates).

Conditioning geometry (c2w / world_points) is ANCHORED to sequence frame 0 (matches
GT-geometry training); proposal geometry (``world_points_global``) stays in the
native global frame — the instance map lives there.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import PIL.Image
    _LANCZOS = PIL.Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    _LANCZOS = PIL.Image.LANCZOS


# --------------------------------------------------------------------------- helpers
def _center_crop_to_aspect(h: int, w: int, target_w: int, target_h: int) -> Tuple[int, int, int, int]:
    """(top, left, crop_h, crop_w) of the center crop matching target aspect."""
    ta = target_w / target_h
    if w / h > ta:                       # too wide -> crop width
        cw = int(round(h * ta))
        return 0, (w - cw) // 2, h, cw
    if w / h < ta:                       # too tall -> crop height
        ch = int(round(w / ta))
        return (h - ch) // 2, 0, ch, w
    return 0, 0, h, w


def _crop_resize_map(arr: np.ndarray, res: Tuple[int, int]) -> np.ndarray:
    """Center-crop to aspect + NEAREST resize a (H,W) or (H,W,C) map to (W,H) res."""
    import cv2
    tw, th = res
    t, l, ch, cw = _center_crop_to_aspect(arr.shape[0], arr.shape[1], tw, th)
    arr = arr[t:t + ch, l:l + cw]
    out = cv2.resize(arr, (tw, th), interpolation=cv2.INTER_NEAREST)
    return out


def _crop_resize_K(K: np.ndarray, hw: Tuple[int, int], res: Tuple[int, int]) -> np.ndarray:
    """Adjust a (3,3) K calibrated for grid ``hw`` to the cropped+resized ``res``."""
    tw, th = res
    t, l, ch, cw = _center_crop_to_aspect(hw[0], hw[1], tw, th)
    K = K.copy().astype(np.float32)
    K[0, 2] -= l
    K[1, 2] -= t
    sx, sy = tw / float(cw), th / float(ch)
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K


def load_image_norm(path: str, res: Tuple[int, int]) -> np.ndarray:
    """RGB -> center-crop to aspect -> LANCZOS resize -> [-1,1] float32 (3,H,W).

    Mirrors the OpenSe3r dataset convention (``/127.5 - 1``; DINOv2 mean/std norm
    happens inside the model).
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    tw, th = res
    t, l, ch, cw = _center_crop_to_aspect(h, w, tw, th)
    img = img.crop((l, t, l + cw, t + ch)).resize((tw, th), resample=_LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    return arr.transpose(2, 0, 1)


def _apply_T(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Rigid-transform an (...,3) point array by a 4x4."""
    return pts @ T[:3, :3].T + T[:3, 3]


# --------------------------------------------------------------------------- golden GT source
def gt_seq_box_scale(batch: dict) -> torch.Tensor:
    """Sequence-global GT scale — delegates to the reference implementation."""
    from slam_opense3r.pipeline_query_gt import _compute_gt_seq_box_scale
    return _compute_gt_seq_box_scale(batch["pts_gt"].float(), batch["camera_pose"].float())


def gt_chunks(batch: dict, chunk_size: int,
              max_frames: Optional[int] = None) -> Iterator[Dict]:
    """Chunk stream from an OpenSe3r GT dataset batch (golden-test geometry).

    Replicates ``_forward_semantic_gt``'s prep: c2w anchored to ``poses[0]``,
    world points anchored + invalid zeroed (mean-fill happens in ``sem_model``),
    validity = ``norm > 1e-3``.
    """
    images = batch["images"].float()                    # [T,3,H,W] in [-1,1]
    poses = batch["camera_pose"].float()                # [T,4,4]
    K = batch["camera_intrinsics"].float()              # [T,3,3]
    pts = batch["pts_gt"].float()                       # [T,H,W,3] GT world
    depth = batch["depth_gt"].float()                   # [T,H,W]
    T = images.shape[0] if max_frames is None else min(images.shape[0], max_frames)
    anchor_inv = torch.inverse(poses[0])

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        pts_raw = pts[start:end]
        valid = pts_raw.norm(dim=-1) > 1e-3             # [Tc,H,W]
        pts_h = torch.cat([pts_raw, torch.ones_like(pts_raw[..., :1])], dim=-1)
        pts_anch = torch.einsum("ij,thwj->thwi", anchor_inv, pts_h)[..., :3]
        pts_anch[~valid] = 0.0
        yield dict(
            images=images[start:end].unsqueeze(0),
            depth=depth[start:end].unsqueeze(0),
            intrinsics=K[start:end].unsqueeze(0),
            c2w=(anchor_inv @ poses[start:end]).unsqueeze(0),
            world_points_anchored=pts_anch.unsqueeze(0),
            valid=valid.unsqueeze(0),
            world_points_global=pts_raw,
            conf=valid.float(),
            frame_indices=list(range(start, end)),
            frame_keys=list(range(start, end)),
        )


# --------------------------------------------------------------------------- ma_slam source
class SubmapBridge:
    """ma_slam ``Submap`` (+ current graph) -> semantic chunk.

    Gotchas handled here: resolution remap (nearest for points/depth/conf, K
    rescaled, RGB reloaded from disk at the semantic res), MA-conf -> validity
    (MapAnything conf is not (0,1): gate at the per-submap mean, the same basis as
    ``Submap.conf_threshold``), gauge normalization, overlap-frame dedup.
    """

    def __init__(self, res: Tuple[int, int] = (518, 336)):
        self.res = res
        self.T0_inv: Optional[np.ndarray] = None        # sequence gauge anchor

    def chunk(self, sm, graph) -> Dict:
        i0 = 0 if self.T0_inv is None else 1            # drop shared overlap frame
        idx = list(range(i0, sm.n))
        tw, th = self.res
        imgs, c2w, wp_anch, wp_glob, valid, depth, Ks = [], [], [], [], [], [], []
        for i in idx:
            G = sm.frame_pose_global(graph, i)
            if self.T0_inv is None:
                self.T0_inv = np.linalg.inv(G)          # lock gauge: frame 0 == identity
            c2w.append(self.T0_inv @ G)

            P = _crop_resize_map(sm.frame_points_global(graph, i).astype(np.float32), self.res)
            v = sm.conf[i] > sm.conf_threshold          # mean-conf gate (MA conf remap)
            if sm.mask is not None:
                v = v & sm.mask[i]
            v = _crop_resize_map(v.astype(np.uint8), self.res).astype(bool)
            Pa = _apply_T(self.T0_inv, P)
            Pa[~v] = 0.0
            wp_glob.append(P)
            wp_anch.append(Pa)
            valid.append(v)
            depth.append(_crop_resize_map(sm.depth[i].astype(np.float32), self.res))
            Ks.append(_crop_resize_K(sm.intrinsics[i], sm.depth[i].shape, self.res))
            imgs.append(load_image_norm(sm.image_paths[i], self.res))

        f32 = lambda x: torch.from_numpy(np.stack(x).astype(np.float32))
        return dict(
            images=f32(imgs).unsqueeze(0),
            depth=f32(depth).unsqueeze(0),
            intrinsics=f32(Ks).unsqueeze(0),
            c2w=f32(c2w).unsqueeze(0),
            world_points_anchored=f32(wp_anch).unsqueeze(0),
            valid=torch.from_numpy(np.stack(valid)).unsqueeze(0),
            world_points_global=f32(wp_glob),
            conf=f32([v.astype(np.float32) for v in valid]),
            frame_indices=[sm.base_id + i for i in idx],   # graph keys are global frame ordinals
            frame_keys=[sm.key(i) for i in idx],
            submap_base=sm.base_id,
            frame_idx=idx,
        )


class PoseTracker:
    """Snapshot each frame's M_i = G_i @ inv(L_i) per submap; report the max
    translation delta of previously-seen frames since the last call. Drives the
    (default-off) KV reset valve; the instance-voxel cache uses its own
    per-instance M-signature instead."""

    def __init__(self):
        self._M: Dict[int, np.ndarray] = {}

    def update(self, gmap, graph) -> float:
        max_delta = 0.0
        for sm in gmap.ordered():
            if sm.is_lc:
                continue
            for i in range(sm.n):
                k = sm.key(i)
                M = graph.get_pose(k) @ np.linalg.inv(sm.poses_local[i])
                prev = self._M.get(k)
                if prev is not None:
                    max_delta = max(max_delta, float(np.linalg.norm(M[:3, 3] - prev[:3, 3])))
                self._M[k] = M
        return max_delta
