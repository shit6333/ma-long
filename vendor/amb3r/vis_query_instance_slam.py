"""
amb3r/vis_query_instance_slam.py
=================================
Export helpers for query-head SLAM memory.

Three PLYs + one JSON metadata dump, all coloured by `inst_id` and sharing
the **same** color map within a single memory (so a given inst_id renders
the same color in every PLY for visual cross-reference):

  - `export_query_instance_voxels_ply`     : voxel-resolution PLY
  - `export_query_instance_per_point_ply`  : pixel-resolution PLY
  - `export_query_instance_boxes_ply`      : per-instance wireframe AABB
  - `dump_instance_table_json`             : per-instance metadata

## Color modes

`color_map_for_table(table, mode=...)` returns the per-inst_id RGB used
across all exports.  Two modes:

  - 'rank' (legacy): sort by inst_id, step hues by golden-angle.  Adjacent
                     inst_ids get maximally-different hues; descriptor
                     similarity is not considered.

  - 'desc' (default, descriptor-aware): project the per-instance descriptor
                     to 1D via PCA, sort by projection, then step hues by
                     golden-angle.  Two instances with similar descriptors
                     end up adjacent in the sort order, so they get hues
                     differing by ≈0.618 (near-complementary) — i.e.
                     **the more similar the descriptors, the more distinct
                     the colours**.  This helps disambiguate visually
                     similar objects (e.g. multiple sofas of the same
                     class but at different locations).

  Falls back to 'rank' silently when fewer than 2 instances have valid
  descriptors.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Set
import colorsys
import json
import warnings

import numpy as np
import torch

# Legacy palette used as the fallback path / for the rank mode.
from .vis_instance_hdbscan import _labels_to_colors


# ─────────────────────────────────────────────────────────────────────────────
# Minimal binary PLY writer
# ─────────────────────────────────────────────────────────────────────────────
#
# We bypass trimesh / open3d for the SLAM-side instance exports for two
# reasons:
#   1. trimesh's `points.PointCloud` always pads colours to RGBA (alpha=255).
#      Several PC visualisers (CloudCompare, MeshLab) then enable
#      transparency rendering when they see the alpha property, which
#      noticeably slows down opening large clouds even when the alpha is a
#      constant.
#   2. open3d writes XYZ as float64, doubling the geometry size; for
#      instance PLYs we only need cm-level precision.
# This writer produces the most compact, viewer-friendly binary PLY:
# float32 xyz + uchar rgb, no alpha, no normals, no faces.

_PLY_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('r', 'u1'),  ('g', 'u1'),  ('b', 'u1'),
])


def _write_binary_ply(path: str,
                       points: np.ndarray,         # [N, 3] floatable
                       colors: np.ndarray) -> int: # [N, 3] or [N, 4] uint8
    """Write a binary little-endian PLY with float32 xyz + uchar rgb.

    Returns the vertex count written.
    """
    n = int(points.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode('ascii')

    arr = np.empty(n, dtype=_PLY_DTYPE)
    if n > 0:
        pts_f32 = np.ascontiguousarray(points[:, :3], dtype=np.float32)
        cols_u8 = np.ascontiguousarray(colors[:, :3], dtype=np.uint8)
        arr['x'] = pts_f32[:, 0]
        arr['y'] = pts_f32[:, 1]
        arr['z'] = pts_f32[:, 2]
        arr['r'] = cols_u8[:, 0]
        arr['g'] = cols_u8[:, 1]
        arr['b'] = cols_u8[:, 2]

    with open(path, 'wb') as f:
        f.write(header)
        f.write(arr.tobytes())
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Noise filter
# ─────────────────────────────────────────────────────────────────────────────

def filter_noise_instances(
    table,                                 # InstanceTable
    total_frames:                int,
    # Rule 1 — count too low
    min_count:                   int   = 0,        # 0 = off; 1 = drop count==1
    end_grace:                   int   = 10,       # but spare end-of-sequence
    # Rule 2 — oversized cluster with few observations
    max_size_m3:                 float = 0.0,      # 0 = off
    oversized_max_count:         int   = 3,
    # Rule 3 — tiny voxel count + few observations (rare, easy to misclassify)
    tiny_voxels:                 int   = 0,        # 0 = off
    tiny_max_count:              int   = 2,
    # Rule 4 — head was never confident
    min_avg_objectness:          float = 0.0,      # 0 = off
    # Rule 5 — elongated 1D extent (catches long thin wall/floor fragments
    # that slip past Rule 2 because their bbox VOLUME is small)
    max_extent_m:                float = 0.0,      # 0 = off
    elongated_max_count:         int   = 10,
) -> Set[int]:
    """Return the set of inst_ids classified as noise by these rules.

    Each rule is independent and only fires when its own threshold is > 0
    (or > 0.0).  Defaults are all OFF so this function is a no-op until the
    caller opts in.  Reasoning behind each rule:

      - **min_count + end_grace**: ghost masks appear once and never get
        re-observed.  But real objects at the very end of the sequence
        have not had a chance to be re-observed either, so we spare them.

      - **max_size_m3 + oversized_max_count**: a cluster spanning > 20m³
        that has only ≤ 3 merges is almost certainly the head latching
        onto "all of floor" or "all of wall" briefly.  Real big objects
        (sofa, table) get merged dozens of times.

      - **tiny_voxels + tiny_max_count**: tiny clusters that were also
        observed only 1-2 times are likely ghosts.  Joint with count to
        avoid killing legitimate small objects (cups, books, etc).

      - **min_avg_objectness**: if the head's confidence (post-σ) averaged
        over all merged proposals is low, the cluster is probably noise.
    """
    dropped: Set[int] = set()
    for inst_id, rec in table.records.items():
        # Rule 1: drop instances whose count is at-most `min_count` (so
        # min_count=1 drops count==1, min_count=2 drops count<=2, etc).
        # Spare those near the end of the sequence (no chance to be
        # re-observed yet).
        if min_count > 0 and rec.count <= min_count:
            if rec.last_seen < total_frames - end_grace:
                dropped.add(inst_id)
                continue
        # Rule 2
        if max_size_m3 > 0 and rec.bbox_volume > max_size_m3 \
                and rec.count <= oversized_max_count:
            dropped.add(inst_id)
            continue
        # Rule 3
        if tiny_voxels > 0 and rec.num_voxels < tiny_voxels \
                and rec.count <= tiny_max_count:
            dropped.add(inst_id)
            continue
        # Rule 4
        if min_avg_objectness > 0 and rec.mean_objectness < min_avg_objectness:
            dropped.add(inst_id)
            continue
        # Rule 5 — long thin elongated cluster (e.g. wall / floor strip).
        # Volume rule (Rule 2) misses these because a 5m×0.1m×3m wall is
        # only 1.5 m³ but extent_max=5m is still clearly an artifact.
        if max_extent_m > 0:
            extent = float((rec.bbox_max - rec.bbox_min).max().item())
            if extent > max_extent_m and rec.count <= elongated_max_count:
                dropped.add(inst_id)
                continue
    return dropped


# ─────────────────────────────────────────────────────────────────────────────
# Color map
# ─────────────────────────────────────────────────────────────────────────────

_GOLDEN_ANGLE = 0.6180339887


def _hue_to_rgb_u8(hue: float, sat: float = 0.85, val: float = 0.95) -> np.ndarray:
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, sat, val)
    return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


def color_map_for_table(
    table,                         # InstanceTable (duck-typed: needs .records)
    mode: str = 'desc',
) -> Dict[int, np.ndarray]:
    """Build a per-inst_id RGB color map shared across all PLY exports.

    Args:
        table:  an `InstanceTable`-like object with a `.records` dict.
        mode:   'desc' (descriptor-aware via 1D PCA + golden angle) or
                'rank' (sort by inst_id + golden angle, legacy behaviour).

    Returns:
        Dict[inst_id → np.ndarray (3,) uint8]
    """
    if mode not in ('rank', 'desc'):
        raise ValueError(f"mode must be 'rank' or 'desc', got {mode!r}")

    ids_sorted = sorted(table.records.keys())
    if not ids_sorted:
        return {}

    if mode == 'desc':
        # Collect descriptors; if missing or < 2 instances, fallback to rank.
        descs = []
        descs_ok = True
        for iid in ids_sorted:
            d = table.records[iid].descriptor
            if d is None:
                descs_ok = False
                break
            descs.append(d.detach().cpu().float().numpy())
        if descs_ok and len(descs) >= 2:
            D = np.stack(descs, axis=0)                                 # [N, dim]
            # 1D PCA = projection onto the top right-singular vector.
            # Use np.linalg.svd directly to avoid an sklearn dependency.
            D_centred = D - D.mean(axis=0, keepdims=True)
            try:
                # full_matrices=False keeps memory low for large D.
                _, _, Vt = np.linalg.svd(D_centred, full_matrices=False)
                proj = D_centred @ Vt[0]                                # [N]
                order = np.argsort(proj)
                ids_sorted = [ids_sorted[i] for i in order]
            except np.linalg.LinAlgError:
                warnings.warn(
                    "color_map_for_table: PCA failed, falling back to 'rank' mode."
                )

    # Apply golden-angle hue stepping over the (possibly-reordered) sequence.
    out: Dict[int, np.ndarray] = {}
    for rank, iid in enumerate(ids_sorted):
        out[int(iid)] = _hue_to_rgb_u8(rank * _GOLDEN_ANGLE)
    return out


def _colors_from_map(
    inst_ids: np.ndarray,
    color_map: Dict[int, np.ndarray],
    noise_color: Tuple[int, int, int] = (80, 80, 80),
) -> np.ndarray:
    """Vectorised lookup: inst_ids [N] → colors [N, 3] uint8."""
    nc = np.array(noise_color, dtype=np.uint8)
    out = np.broadcast_to(nc, (len(inst_ids), 3)).copy()
    for iid, col in color_map.items():
        out[inst_ids == iid] = col
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Common helpers
# ─────────────────────────────────────────────────────────────────────────────

def _voxel_centers(voxel_size: float, voxel_to_inst: dict) -> Tuple[np.ndarray, np.ndarray]:
    """voxel_to_inst : dict[(vx,vy,vz) → inst_id]  →  (centers [N,3], inst_ids [N])."""
    if not voxel_to_inst:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.int32)
    keys   = list(voxel_to_inst.keys())
    ids    = np.array([voxel_to_inst[k] for k in keys], dtype=np.int32)
    coords = np.array(keys, dtype=np.float32)
    centers = (coords + 0.5) * voxel_size
    return centers, ids


def _apply_transform(points_np: np.ndarray, T: Optional[torch.Tensor]) -> np.ndarray:
    if T is None:
        return points_np
    R = T[:3, :3].cpu().numpy().astype(np.float32)
    t = T[:3,  3].cpu().numpy().astype(np.float32)
    return points_np @ R.T + t[None, :]


def _invert_transform(points_np: np.ndarray, T: Optional[torch.Tensor]) -> np.ndarray:
    """Inverse of `_apply_transform` — final frame → local voxel frame."""
    if T is None:
        return points_np
    R = T[:3, :3].cpu().numpy().astype(np.float32)
    t = T[:3,  3].cpu().numpy().astype(np.float32)
    return (points_np - t[None, :]) @ R


def _box_edge_points(bmin: np.ndarray, bmax: np.ndarray,
                       num_per_edge: int = 30) -> np.ndarray:
    """Axis-aligned bbox (bmin/bmax) → 12 edges × N points.

    Returns [12 * num_per_edge, 3] float32.
    """
    corners = np.array([
        [bmin[0], bmin[1], bmin[2]],   # 0: ---
        [bmax[0], bmin[1], bmin[2]],   # 1: +--
        [bmax[0], bmax[1], bmin[2]],   # 2: ++-
        [bmin[0], bmax[1], bmin[2]],   # 3: -+-
        [bmin[0], bmin[1], bmax[2]],   # 4: --+
        [bmax[0], bmin[1], bmax[2]],   # 5: +-+
        [bmax[0], bmax[1], bmax[2]],   # 6: +++
        [bmin[0], bmax[1], bmax[2]],   # 7: -++
    ], dtype=np.float32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    t = np.linspace(0.0, 1.0, num_per_edge, dtype=np.float32)
    pts = []
    for i, j in edges:
        line = corners[i][None, :] + t[:, None] * (corners[j] - corners[i])[None, :]
        pts.append(line)
    return np.concatenate(pts, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Voxel-level PLY
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def export_query_instance_voxels_ply(
    memory,                                          # OpenSe3rQueryMemory
    save_file:    str,
    transformation: Optional[torch.Tensor] = None,
    color_mode:   str = 'desc',                       # 'desc' or 'rank'
    color_map:    Optional[Dict[int, np.ndarray]] = None,
    dropped_ids:  Optional[Set[int]]              = None,
    noise_color:  Tuple[int, int, int] = (80, 80, 80),
    verbose:      bool = False,
) -> dict:
    """Voxel-resolution instance PLY.

    Each voxel in `memory.instance_table.voxel_to_inst_map()` becomes one
    point at the voxel centre.  Pass an explicit `color_map` to keep
    colours consistent across multiple PLYs (e.g. voxels + boxes); when
    omitted, one is computed from the table using `color_mode`.
    """
    table = memory.instance_table
    voxel_to_inst = table.voxel_to_inst_map()
    centers, ids = _voxel_centers(table.voxel_size, voxel_to_inst)
    if centers.shape[0] == 0:
        warnings.warn(f"export_query_instance_voxels_ply: empty table — "
                      f"writing empty PLY: {save_file}")
        _write_binary_ply(save_file,
                            np.zeros((0, 3), np.float32),
                            np.zeros((0, 3), np.uint8))
        return {'num_voxels': 0, 'num_instances': 0}

    if color_map is None:
        color_map = color_map_for_table(table, mode=color_mode)
    # Override dropped inst_ids to noise_color so they render as grey
    # (we keep their voxels in the PLY for diagnostics — comment them out
    # of the cluster count in stdout instead of silently disappearing).
    effective_cmap = dict(color_map)
    if dropped_ids:
        noise_arr = np.array(noise_color, dtype=np.uint8)
        for iid in dropped_ids:
            effective_cmap[iid] = noise_arr
    centers = _apply_transform(centers, transformation)
    colors  = _colors_from_map(ids, effective_cmap, noise_color=noise_color)
    _write_binary_ply(save_file, centers, colors)
    n_inst        = int(np.unique(ids).size)
    n_inst_kept   = int(np.unique([i for i in ids if i not in (dropped_ids or set())]).size)
    if verbose:
        print(f"[query-vox-ply] {save_file}  voxels={centers.shape[0]:,}  "
              f"instances={n_inst} (kept={n_inst_kept})  color_mode={color_mode}")
    return {'num_voxels': int(centers.shape[0]),
            'num_instances': n_inst,
            'num_instances_kept': n_inst_kept}


# ─────────────────────────────────────────────────────────────────────────────
# Per-point PLY
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def export_query_instance_per_point_ply(
    memory,
    save_file:       str,
    transformation:  Optional[torch.Tensor] = None,
    color_mode:      str = 'desc',
    color_map:       Optional[Dict[int, np.ndarray]] = None,
    dropped_ids:     Optional[Set[int]]              = None,
    noise_color:     Tuple[int, int, int]    = (80, 80, 80),
    point_conf_min:  Optional[float]         = None,
    verbose:         bool                    = False,
) -> dict:
    """Pixel-resolution instance PLY (each valid pixel coloured by voxel inst_id)."""
    table = memory.instance_table
    voxel_to_inst = table.voxel_to_inst_map()
    vsize = table.voxel_size

    pts  = memory.pts.cpu().numpy().astype(np.float32)
    conf = memory.conf.cpu().numpy().astype(np.float32)
    T, H, W, _ = pts.shape

    valid = np.linalg.norm(pts, axis=-1) > 1e-3
    if point_conf_min is not None:
        valid = valid & (conf >= float(point_conf_min))

    valid_idx = np.nonzero(valid.reshape(-1))[0]
    if valid_idx.size == 0:
        warnings.warn(f"export_query_instance_per_point_ply: no valid pts — "
                      f"writing empty PLY: {save_file}")
        _write_binary_ply(save_file,
                            np.zeros((0, 3), np.float32),
                            np.zeros((0, 3), np.uint8))
        return {'num_points': 0, 'num_instances': 0}

    pts_flat = pts.reshape(-1, 3)[valid_idx]

    # Frame convention (matches `export_semantic_per_point_ply`):
    #   - `memory.pts` is already in FINAL (post-T_align) frame because
    #     `slam_opense3r/run.py` applies T_align in-place to memory.pts
    #     when `--align_to_gt_first_frame` is set.
    #   - `instance_table.voxel_to_inst` keys live in LOCAL (pre-T_align)
    #     frame — the voxel store is never touched by T_align.
    #   - So we invert `transformation` to bring pts back to local frame
    #     for the floor() lookup, then write pts AS-IS (already final).
    pts_local = _invert_transform(pts_flat, transformation)
    vox_keys = np.floor(pts_local / vsize).astype(np.int64)
    ids = np.full(pts_flat.shape[0], -1, dtype=np.int32)
    for i, k in enumerate(map(tuple, vox_keys.tolist())):
        ids[i] = voxel_to_inst.get(k, -1)

    if color_map is None:
        color_map = color_map_for_table(table, mode=color_mode)
    effective_cmap = dict(color_map)
    if dropped_ids:
        noise_arr = np.array(noise_color, dtype=np.uint8)
        for iid in dropped_ids:
            effective_cmap[iid] = noise_arr
    colors   = _colors_from_map(ids, effective_cmap, noise_color=noise_color)
    _write_binary_ply(save_file, pts_flat, colors)
    n_inst = int(np.unique(ids[ids >= 0]).size)
    if verbose:
        print(f"[query-pp-ply] {save_file}  pts={pts_flat.shape[0]:,}  "
              f"instances={n_inst}  color_mode={color_mode}")
    return {'num_points': int(pts_flat.shape[0]), 'num_instances': n_inst}


# ─────────────────────────────────────────────────────────────────────────────
# Bounding-box PLY  (NEW — wireframe per instance)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def export_query_instance_boxes_ply(
    memory,
    save_file:       str,
    transformation:  Optional[torch.Tensor] = None,
    color_mode:      str = 'desc',
    color_map:       Optional[Dict[int, np.ndarray]] = None,
    dropped_ids:     Optional[Set[int]]              = None,
    num_per_edge:    int = 30,
    min_voxels:      int = 1,     # skip pathologically small instances
    verbose:         bool = False,
) -> dict:
    """Per-instance wireframe AABB PLY.

    Reads `bbox_min` / `bbox_max` from each `InstanceRecord` and writes a
    12-edge × N points wireframe in the same color palette as the
    voxel / per-point PLYs (when `color_map` is shared).

    Suitable for overlaying in CloudCompare / MeshLab on top of the
    voxel PLY: same color → same instance.
    """
    table = memory.instance_table
    if color_map is None:
        color_map = color_map_for_table(table, mode=color_mode)

    pts_all, col_all = [], []
    n_drawn = 0
    drop_set = dropped_ids or set()
    for inst_id in sorted(table.records.keys()):
        if inst_id in drop_set:
            # Don't clutter the wireframe PLY with noise — skip instead of
            # rendering as grey (boxes are visually heavy enough).
            continue
        rec = table.records[inst_id]
        if rec.num_voxels < min_voxels:
            continue
        bmin = rec.bbox_min.cpu().numpy().astype(np.float32)
        bmax = rec.bbox_max.cpu().numpy().astype(np.float32)
        if np.any(bmax - bmin < 1e-3):     # degenerate
            continue
        edge_pts = _box_edge_points(bmin, bmax, num_per_edge=num_per_edge)
        edge_pts = _apply_transform(edge_pts, transformation)
        col_one  = color_map.get(int(inst_id),
                                  np.array([80, 80, 80], dtype=np.uint8))
        col_pts  = np.broadcast_to(col_one, (edge_pts.shape[0], 3)).copy()
        pts_all.append(edge_pts)
        col_all.append(col_pts)
        n_drawn += 1

    if not pts_all:
        warnings.warn(f"export_query_instance_boxes_ply: no valid bboxes — "
                      f"writing empty PLY: {save_file}")
        _write_binary_ply(save_file,
                            np.zeros((0, 3), np.float32),
                            np.zeros((0, 3), np.uint8))
        return {'num_boxes': 0, 'num_points': 0}

    pts_np = np.concatenate(pts_all, axis=0).astype(np.float32)
    col_np = np.concatenate(col_all, axis=0).astype(np.uint8)
    _write_binary_ply(save_file, pts_np, col_np)
    if verbose:
        print(f"[query-box-ply] {save_file}  boxes={n_drawn}  "
              f"pts={pts_np.shape[0]:,}  color_mode={color_mode}")
    return {'num_boxes': n_drawn, 'num_points': int(pts_np.shape[0])}


# ─────────────────────────────────────────────────────────────────────────────
# Metadata dump
# ─────────────────────────────────────────────────────────────────────────────

def dump_instance_labels_npz(
    memory,                                          # OpenSe3rQueryMemory
    save_file:      str,
    transformation: Optional[torch.Tensor] = None,    # [4,4] world-frame shift
    dropped_ids:    Optional[Set[int]]      = None,    # mapped to -1 (noise)
    verbose:        bool                    = False,
    objectness_agg: str                     = 'mean',   # 'mean' | 'max' — how to
                                                        # aggregate member objectness into cluster_score
) -> dict:
    """Export the SLAM voxel → inst_id map in the NPZ format expected by
    `eval_instance_npz_scannet.py`.

    Produces:
        centers        [V, 3] float32   voxel centre xyz (world coords, with
                                        `transformation` already applied)
        cluster_id     [V]    int32     inst_id per voxel; dropped (filtered)
                                        instances are encoded as -1 to match
                                        HDBSCAN's noise convention.
        score_ids      [K]    int32     kept inst_id (parallel to cluster_score)
        cluster_score  [K]    float32   per-instance ScanNet-style confidence
                                        = `mean_objectness`.  Used by AP eval
                                        when present (else eval falls back to
                                        cluster-size proxy).
    """
    table = memory.instance_table
    vmap  = table.voxel_to_inst_map()         # dict[(vx,vy,vz) → inst_id]
    centers_np, ids_np = _voxel_centers(table.voxel_size, vmap)

    if centers_np.shape[0] == 0:
        np.savez_compressed(save_file,
            centers       = np.zeros((0, 3), np.float32),
            cluster_id    = np.zeros((0,),  np.int32),
            score_ids     = np.zeros((0,),  np.int32),
            cluster_score = np.zeros((0,),  np.float32),
        )
        if verbose:
            print(f"[inst-npz] {save_file}  (empty)")
        return {'num_voxels': 0, 'num_instances': 0}

    if transformation is not None:
        centers_np = _apply_transform(centers_np, transformation)

    drop_set = dropped_ids or set()
    if drop_set:
        ids_np = ids_np.copy()
        for iid in drop_set:
            ids_np[ids_np == iid] = -1

    # Per-instance confidence — uses `mean_objectness` (head's per-frame
    # objectness post-σ, running-averaged across all merged proposals).
    # Skip dropped instances so they don't pollute the eval's pred list.
    score_ids, score_vals = [], []
    for iid in sorted(table.records.keys()):
        if iid in drop_set:
            continue
        rec = table.records[iid]
        # cluster_score in (0, 1) (sigmoid space).  'max' = best-view (avoids A'
        # obj×sem_conf being diluted by the per-instance mean over bad-view frames).
        score_ids.append(int(iid))
        score_vals.append(float(rec.max_objectness if objectness_agg == 'max'
                                else rec.mean_objectness))

    np.savez_compressed(save_file,
        centers       = centers_np.astype(np.float32),
        cluster_id    = ids_np.astype(np.int32),
        score_ids     = np.asarray(score_ids,  dtype=np.int32),
        cluster_score = np.asarray(score_vals, dtype=np.float32),
    )
    n_inst_kept = int((np.unique(ids_np[ids_np >= 0])).size)
    if verbose:
        score_range = (f"score[min={min(score_vals):.3f} "
                       f"max={max(score_vals):.3f}]") if score_vals else "(no scores)"
        print(f"[inst-npz] {save_file}  voxels={centers_np.shape[0]:,}  "
              f"instances={n_inst_kept}  {score_range}")
    return {'num_voxels': int(centers_np.shape[0]),
            'num_instances': n_inst_kept}


@torch.no_grad()
def dump_instance_labels_per_point_npz(
    memory,                                          # OpenSe3rQueryMemory
    save_file:      str,
    pts_world:      np.ndarray,                       # [N, 3] points in FINAL frame
    transformation: Optional[torch.Tensor] = None,    # [4, 4] local→final (same as voxel dump)
    dropped_ids:    Optional[Set[int]]      = None,
    verbose:        bool                    = False,
) -> dict:
    """Per-point version of `dump_instance_labels_npz` — same NPZ schema but
    `centers` are the actual 3D points (typically post-conf-filter geo pts)
    instead of voxel centres.  Eliminates the 5-cm voxel quantisation that
    inflates per-voxel IoU boundary error vs ESAM-style per-mesh-vert dumps.

    The cluster_id lookup is still voxel-based (model output resolution
    unchanged) — we just give each GT vertex a finer neighbour to match
    against during the eval's kNN projection.

    Produces (same as voxel version):
        centers        [N, 3] float32   per-point xyz in FINAL frame
        cluster_id     [N]    int32     inst_id per point; -1 if no voxel hit
        score_ids      [K]    int32     kept inst_ids
        cluster_score  [K]    float32   per-instance mean_objectness
    """
    table = memory.instance_table
    voxel_to_inst = table.voxel_to_inst_map()
    vsize = table.voxel_size

    pts_world = np.asarray(pts_world, dtype=np.float32)
    if pts_world.shape[0] == 0:
        np.savez_compressed(save_file,
            centers       = np.zeros((0, 3), np.float32),
            cluster_id    = np.zeros((0,),  np.int32),
            score_ids     = np.zeros((0,),  np.int32),
            cluster_score = np.zeros((0,),  np.float32),
        )
        if verbose:
            print(f"[inst-pp-npz] {save_file}  (empty)")
        return {'num_points': 0, 'num_instances': 0}

    # Invert FINAL → LOCAL frame for voxel-key lookup (voxel keys live in
    # LOCAL/pred frame; pts_world is post T_align).  See per_point PLY
    # exporter — same convention.
    pts_local = _invert_transform(pts_world, transformation)
    vox_keys = np.floor(pts_local / vsize).astype(np.int64)

    drop_set = dropped_ids or set()
    cluster_id = np.full(pts_world.shape[0], -1, dtype=np.int32)
    if voxel_to_inst:
        for i, k in enumerate(map(tuple, vox_keys.tolist())):
            iid = voxel_to_inst.get(k, -1)
            cluster_id[i] = -1 if iid in drop_set else iid

    # Per-instance scores (same as voxel version; skip dropped).
    score_ids, score_vals = [], []
    for iid in sorted(table.records.keys()):
        if iid in drop_set:
            continue
        rec = table.records[iid]
        score_ids.append(int(iid))
        score_vals.append(float(rec.mean_objectness))

    np.savez_compressed(save_file,
        centers       = pts_world.astype(np.float32),
        cluster_id    = cluster_id,
        score_ids     = np.asarray(score_ids,  dtype=np.int32),
        cluster_score = np.asarray(score_vals, dtype=np.float32),
    )
    n_assigned = int((cluster_id >= 0).sum())
    n_inst_kept = int(np.unique(cluster_id[cluster_id >= 0]).size)
    if verbose:
        pct = 100.0 * n_assigned / max(pts_world.shape[0], 1)
        print(f"[inst-pp-npz] {save_file}  pts={pts_world.shape[0]:,}  "
              f"assigned={n_assigned:,} ({pct:.1f}%)  instances={n_inst_kept}")
    return {'num_points': int(pts_world.shape[0]),
            'num_assigned': n_assigned,
            'num_instances': n_inst_kept}


@torch.no_grad()
def dump_instance_sem_queries_npz(
    memory,                                          # OpenSe3rQueryMemory
    save_file:      str,
    dropped_ids:    Optional[Set[int]] = None,
    verbose:        bool               = False,
) -> dict:
    """Dump per-instance aggregated semantic features (`sem_query`) → NPZ.

    `sem_query` is the mask-region mean of dense semantic features, kept
    as a running L2-normed average across all merged proposals (see
    `InstanceRecord.merge`).  We dump one D-dim vector per instance for
    offline open-vocabulary analysis: broadcast back to voxels/points and
    text-align against ScanNet class prompts.

    Produces:
        inst_ids:    [N]    int32     stable instance id (same as voxel
                                       NPZ's `cluster_id`).  Dropped
                                       instances are excluded entirely
                                       (cluster_id == -1 in voxel NPZ
                                       won't find a match here, which is
                                       the correct semantics for noise).
        sem_queries: [N, D] float32   L2-normed mean sem feature.
        voxel_size:  scalar float32   for reference (matches voxel NPZ).
    """
    table = memory.instance_table
    drop_set = dropped_ids or set()
    ids, feats = [], []
    for iid in sorted(table.records.keys()):
        if iid in drop_set:
            continue
        rec = table.records[iid]
        if rec.sem_query is None:
            continue
        ids.append(int(iid))
        feats.append(rec.sem_query.detach().cpu().float().numpy())

    if not feats:
        np.savez_compressed(save_file,
            inst_ids    = np.zeros((0,),  dtype=np.int32),
            sem_queries = np.zeros((0, 0), dtype=np.float32),
            voxel_size  = np.float32(table.voxel_size),
        )
        if verbose:
            print(f"[inst-sem-q] {save_file}  (no sem_query records found)")
        return {'num_instances': 0, 'sem_dim': 0}

    feats_np = np.stack(feats, axis=0).astype(np.float32)   # [N, D]
    ids_np   = np.asarray(ids, dtype=np.int32)              # [N]
    np.savez_compressed(save_file,
        inst_ids    = ids_np,
        sem_queries = feats_np,
        voxel_size  = np.float32(table.voxel_size),
    )
    if verbose:
        print(f"[inst-sem-q] {save_file}  instances={len(ids)}  "
              f"sem_dim={feats_np.shape[-1]}")
    return {'num_instances': len(ids), 'sem_dim': int(feats_np.shape[-1])}


def dump_instance_table_json(memory, save_file: str,
                              transformation: Optional[torch.Tensor] = None,
                              dropped_ids:    Optional[Set[int]]      = None,
                              verbose: bool = False) -> dict:
    meta = memory.instance_table.to_metadata_list()

    # Tag dropped instances so downstream analysis can spot them.
    drop_set = dropped_ids or set()
    for m in meta:
        m['filtered'] = bool(m['inst_id'] in drop_set)

    if transformation is not None and meta:
        R = transformation[:3, :3].cpu().numpy()
        t = transformation[:3,  3].cpu().numpy()
        for m in meta:
            c    = np.array(m['center'],   dtype=np.float64)
            bmin = np.array(m['bbox_min'], dtype=np.float64)
            bmax = np.array(m['bbox_max'], dtype=np.float64)
            m['center']   = (R @ c    + t).tolist()
            m['bbox_min'] = (R @ bmin + t).tolist()
            m['bbox_max'] = (R @ bmax + t).tolist()

    payload = {
        'num_instances':         len(meta),
        'num_instances_filtered': int(sum(1 for m in meta if m['filtered'])),
        'voxel_size':            memory.instance_table.voxel_size,
        'instances':             meta,
    }
    with open(save_file, 'w') as f:
        json.dump(payload, f, indent=2)
    if verbose:
        print(f"[inst-meta] {save_file}  num_instances={len(meta)}")
    return payload
