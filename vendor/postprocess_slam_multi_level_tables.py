#!/usr/bin/env python3
"""
postprocess_slam_multi_level_tables.py
======================================
Wrapper around `postprocess_slam_for_robot.build_instance_table` that
classifies the SLAM-produced instances against **multiple label sets at
different granularity levels** in a single pass, sharing the semantic
backbone load across levels.

Three default label sets, tuned for indoor home / residential scenes:

  - **coarse** (~10 classes): structural + main furniture only — what a
    robot needs for navigation / spatial reasoning.
  - **medium** (~29 classes): the default ScanNet-20 + ceiling + common
    indoor extras (same as `postprocess_slam_for_robot.py`'s default).
  - **fine** (~57 classes): medium ∪ commonly-seen small objects
    (kitchenware, electronics, decor, …) for detailed scene understanding.

Per-instance outputs:

  <out>/<robot_deploy_<scene>_instance_semantic_table_coarse.json
  <out>/<robot_deploy_<scene>_instance_semantic_table_medium.json
  <out>/<robot_deploy_<scene>_instance_semantic_table_fine.json
  <out>/<robot_deploy_<scene>_instance_semantic_table_all_levels.json  (combined)

The combined JSON contains one entry per instance with `class_coarse`,
`class_medium`, `class_fine` (each: `class_name` + `cosine_similarity`)
so downstream consumers can pick the level they need without re-loading
three files.

Usage
-----
    python postprocess_slam_multi_level_tables.py \
        --slam_dir   ./outputs/deploy/scene_xxx \
        --scene_name scene_xxx \
        [--output_dir ./outputs/deploy/scene_xxx] \
        [--backbone tips-v2-l14] \
        [--levels coarse,medium,fine] \
        [--coarse_labels ...] [--medium_labels ...] [--fine_labels ...] \
        [--include_filtered] [--no_combined]
"""

from __future__ import annotations
import argparse
import colorsys
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'thirdparty'))

# Reuse the original postprocess helpers:
#   - build_instance_table : (sem_queries + slam table json + label list) → dict
#   - resolve_label_set    : parse 'default' / path / comma-list / preset name
#   - DEFAULT_LABEL_SET    : the 29-class medium baseline
#   - _get_cached_backbone : memoizes the semantic backbone across levels
from postprocess_slam_for_robot import (
    DEFAULT_LABEL_SET,
    build_instance_table,
    resolve_label_set,
    _get_cached_backbone,
    palette_from_ids,
    write_ply_xyz_rgb_inst,
    load_geo_per_point,
    inst_id_per_point,
    subsample_points,
)
from amb3r.tools.semantic_vis_utils import save_semantic_color_legend


# ─────────────────────────────────────────────────────────────────────────────
# Default label sets for indoor home / residential scenes
# ─────────────────────────────────────────────────────────────────────────────

# COARSE — minimum viable structural + main-furniture set.  Useful when the
# robot only needs "where is the wall / floor / a sittable thing" and you
# want maximum classification robustness (fewer classes → less confusion
# in CLIP-text matching).
LABELS_COARSE: List[str] = [
    # Structural
    'wall', 'floor', 'ceiling',
    # Openings
    'door', 'window',
    # Storage
    'cabinet',
    # Main furniture
    'table', 'chair', 'sofa', 'bed',
    # Catch-all (last)
    'otherfurniture',
]

# MEDIUM — same as the original postprocess default (ScanNet-20 + ceiling
# + a handful of common indoor extras).  Already a reasonable trade-off
# between coverage and discriminability.
LABELS_MEDIUM: List[str] = list(DEFAULT_LABEL_SET)

# FINE additions on top of MEDIUM — common small objects seen in home
# environments (kitchen, living room, bedroom, bathroom, office).  Keep
# the list focused: too many similar classes (bottle vs cup vs glass)
# starts to hurt classification accuracy via CLIP-text confusion.
_FINE_EXTRA: List[str] = [
    # Kitchen appliances
    'oven', 'stove', 'kettle', 'toaster', 'dishwasher', 'coffee machine',
    # Tableware
    'bottle', 'cup', 'mug', 'plate', 'bowl',
    # Living room decor / small items
    'remote control', 'speaker', 'vase', 'plant', 'clock', 'mirror',
    'cushion', 'blanket',
    # Bedroom storage
    'wardrobe', 'nightstand', 'hanger',
    # Bathroom
    'towel',
    # Office / electronics
    'laptop', 'keyboard', 'mouse', 'phone', 'headphones', 'printer',
    # Misc commonly-seen
    'backpack', 'bag', 'fan', 'book', 'shoes',
]


def _build_fine_labels(medium: List[str], extras: List[str]) -> List[str]:
    """Fine = medium ∪ extras (deduped, preserves order).  We want fine to
    SUPERSET medium so that the model can still route e.g. a chair to
    'chair' rather than being forced into a less-appropriate fine class."""
    seen = set()
    out = []
    for x in list(medium) + list(extras):
        if x not in seen:
            out.append(x); seen.add(x)
    return out


LABELS_FINE: List[str] = _build_fine_labels(LABELS_MEDIUM, _FINE_EXTRA)


LEVEL_DEFAULTS = {
    'coarse': LABELS_COARSE,
    'medium': LABELS_MEDIUM,
    'fine':   LABELS_FINE,
}


# ─────────────────────────────────────────────────────────────────────────────
# Combined-table builder
# ─────────────────────────────────────────────────────────────────────────────

def _combine_level_tables(per_level: Dict[str, dict]) -> dict:
    """Merge per-level table dicts into a single table where each instance
    carries its classification at every requested level.  Geometry metadata
    (center / bbox / num_voxels / …) is read from the first level's table.

    `per_level`: {level_name: <table dict from build_instance_table>}
    """
    if not per_level:
        return {'levels': [], 'instances': []}

    # Pick any level as the geometry reference; they should all agree on
    # which inst_ids are kept (build_instance_table applies the same
    # drop_filtered logic regardless of label set).
    ref_level = next(iter(per_level))
    ref_inst  = {int(e['inst_id']): e for e in per_level[ref_level]['instances']}

    # Verify the kept inst_id set is identical across levels.
    for lvl, t in per_level.items():
        ids = {int(e['inst_id']) for e in t['instances']}
        if ids != set(ref_inst):
            print(f"[combined] WARN: level '{lvl}' kept inst_ids differ from "
                  f"reference level '{ref_level}'  "
                  f"(only in {lvl}: {sorted(ids - set(ref_inst))[:5]}…,  "
                  f"missing: {sorted(set(ref_inst) - ids)[:5]}…)")

    merged_instances = []
    for iid, ref in ref_inst.items():
        entry = {
            'inst_id':         iid,
            'center':          ref.get('center'),
            'bbox_min':        ref.get('bbox_min'),
            'bbox_max':        ref.get('bbox_max'),
            'bbox_volume_m3':  ref.get('bbox_volume_m3'),
            'num_voxels':      ref.get('num_voxels'),
            'mean_objectness': ref.get('mean_objectness'),
            'first_seen':      ref.get('first_seen'),
            'last_seen':       ref.get('last_seen'),
            'count':           ref.get('count'),
            'filtered':        ref.get('filtered', False),
            # Per-level classification:
            'class_per_level': {},
        }
        for lvl, t in per_level.items():
            by_id = {int(e['inst_id']): e for e in t['instances']}
            e = by_id.get(iid)
            if e is None:
                entry['class_per_level'][lvl] = None
            else:
                entry['class_per_level'][lvl] = {
                    'class_id':          e['class_id'],
                    'class_name':        e['class_name'],
                    'cosine_similarity': e['cosine_similarity'],
                }
        merged_instances.append(entry)

    return {
        'levels':              list(per_level.keys()),
        'labels_per_level':    {lvl: t['labels']     for lvl, t in per_level.items()},
        'voxel_size':          per_level[ref_level].get('voxel_size'),
        'backbone':            per_level[ref_level].get('backbone'),
        'num_instances_total':        per_level[ref_level].get('num_instances_total'),
        'num_instances_after_filter': len(merged_instances),
        'instances':           merged_instances,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-level semantic-colour PLY + legend PNG
# ─────────────────────────────────────────────────────────────────────────────
# Mirrors the `_sem_color.ply` path of postprocess_slam_for_robot.py:
#   - per-point cloud from <scene>_geo.ply (sub-sampled to --max_points)
#   - each point inherits the class of its instance (instance broadcast),
#     so all points of one instance share a single class colour
#   - colour comes from `palette_from_ids` (golden-angle HSV, sat=0.7)
# Difference: we do the geo-load / KDTree / filter / subsample ONCE and
# loop the cheap per-level relabelling so 3 levels ≈ same wall-clock as 1.

def _class_color_table_matching_palette_from_ids(
    num_classes: int,
    saturation:  float = 0.7,
    value:       float = 0.95,
) -> np.ndarray:
    """Same colour formula as `postprocess_slam_for_robot.palette_from_ids`
    (golden-angle HSV) but applied to a CONTIGUOUS range of class ids
    [0, num_classes).  We use this to build the legend PNG so the swatch
    colour for class i matches the colour `palette_from_ids` produces for
    a point whose class_id == i.

    Truncation (int(.*255)) — not round — to match the float-to-uint8 path
    in `write_ply_xyz_rgb_inst`'s `(colors_01 * 255).astype(uint8)`.
    """
    GOLDEN = 0.61803398875
    out = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        h = (i * GOLDEN) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, saturation, value)
        out[i] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def export_sem_color_ply_for_level(
    pts:           np.ndarray,        # [N_pt, 3] geo-PLY points (filtered + sub-sampled)
    inst_per_pt:   np.ndarray,        # [N_pt]    inst_id per point (from instance_labels)
    level_table:   dict,              # output of build_instance_table for this level
    labels:        List[str],         # this level's full label list
    out_ply:       str,
    out_legend:    str,
) -> None:
    """Write `_sem_color_<level>.ply` (per-point, instance-broadcast colour)
    and `_semantic_legend_<level>.png` with matching swatches.
    """
    # ── Per-point class_id via instance broadcast (one class per instance) ──
    iid_to_class = {int(e['inst_id']): int(e['class_id'])
                    for e in level_table['instances']}
    class_per_pt = np.array(
        [iid_to_class.get(int(i), -1) for i in inst_per_pt],
        dtype=np.int32,
    )

    # ── Palette via golden-angle (id < 0 = grey, matches palette_from_ids) ──
    sem_palette = palette_from_ids(class_per_pt, saturation=0.7, value=0.95)
    sem_colors_01 = np.array(
        [sem_palette[int(c)] for c in class_per_pt],
        dtype=np.float32,
    )

    # ── Write per-point PLY (xyz + rgb + inst_id) — IDENTICAL schema to
    #    postprocess_slam_for_robot's `_sem_color.ply`.
    write_ply_xyz_rgb_inst(pts, sem_colors_01, inst_per_pt, out_ply)

    # ── Legend PNG: one swatch per class, same colour formula as the PLY.
    color_table = _class_color_table_matching_palette_from_ids(len(labels))
    save_semantic_color_legend(
        labels      = labels,
        color_table = color_table,
        save_file   = out_legend,
        title       = f"Semantic Color Legend ({len(labels)} classes)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split('\n\n')[0],
    )
    p.add_argument('--slam_dir',   required=True,
                   help='SLAM output dir produced by slam_opense3r/run.py')
    p.add_argument('--scene_name', required=True,
                   help='Scene prefix used at SLAM time (matches the file '
                        'prefix in --slam_dir, e.g. slam_example0).')
    p.add_argument('--output_dir', default=None,
                   help='Output dir for the per-level table JSONs.  '
                        'Default = --slam_dir.')
    p.add_argument('--backbone',   default='tips-v2-l14',
                   help='Semantic backbone for text encoding.  Must match '
                        'the one used at training time.')
    p.add_argument('--levels',     default='coarse,medium,fine',
                   help='Comma-separated subset of {coarse, medium, fine}. '
                        'Default = all three.')
    p.add_argument('--coarse_labels', default=None,
                   help='Override LABELS_COARSE.  Same format as the '
                        'original postprocess --label_set: '
                        "'default' / path / comma-separated.")
    p.add_argument('--medium_labels', default=None,
                   help='Override LABELS_MEDIUM (same format as above).')
    p.add_argument('--fine_labels',   default=None,
                   help='Override LABELS_FINE (same format as above).')
    p.add_argument('--include_filtered', action='store_true', default=False,
                   help='Keep SLAM-flagged `filtered` instances in every '
                        'table (default: drop them).')
    p.add_argument('--no_combined', action='store_true', default=False,
                   help='Skip the combined all-levels JSON.')
    p.add_argument('--no_sem_maps', action='store_true', default=False,
                   help='Skip the per-level `_sem_color_<lvl>.ply` + '
                        '`_semantic_legend_<lvl>.png`.  By default both are '
                        'written for every requested level (per-point, '
                        'instance-broadcast colour — same scheme as '
                        'postprocess_slam_for_robot._sem_color.ply).')
    p.add_argument('--max_points', type=int, default=2_000_000,
                   help='Cap the per-point sem PLYs to this many points '
                        '(matches the original postprocess default).  0 = no cap.')
    p.add_argument('--knn_max_dist_factor', type=float, default=1.5,
                   help='Per-point inst_id lookup: any point whose nearest '
                        'voxel centre is farther than this * voxel_size becomes '
                        'inst_id = -1 (orphan).  Matches the original postprocess.')
    p.add_argument('--device',     default='cuda')
    args = p.parse_args()

    # ── Resolve level selection & label sets ──────────────────────────────
    requested = [s.strip() for s in args.levels.split(',') if s.strip()]
    for lvl in requested:
        if lvl not in LEVEL_DEFAULTS:
            raise SystemExit(
                f"unknown level '{lvl}'  (valid: {list(LEVEL_DEFAULTS)})")

    level_to_labels: Dict[str, List[str]] = {}
    overrides = {
        'coarse': args.coarse_labels,
        'medium': args.medium_labels,
        'fine':   args.fine_labels,
    }
    for lvl in requested:
        if overrides[lvl] is not None:
            level_to_labels[lvl] = resolve_label_set(overrides[lvl])
        else:
            level_to_labels[lvl] = list(LEVEL_DEFAULTS[lvl])
        # Ensure 'ceiling' is present so wall/floor/ceiling stuff filtering
        # at downstream eval stays consistent (matches original postproc
        # convention).
        if 'ceiling' not in level_to_labels[lvl]:
            print(f"[postproc-multi] level '{lvl}': adding 'ceiling' "
                  f"(missing from label set)")
            level_to_labels[lvl] = level_to_labels[lvl] + ['ceiling']

    for lvl, labs in level_to_labels.items():
        print(f"[postproc-multi] level '{lvl}' ({len(labs)} classes): "
              f"{labs[:10]}{'…' if len(labs) > 10 else ''}")

    # ── Locate input artefacts ────────────────────────────────────────────
    slam_dir = args.slam_dir
    scene    = args.scene_name
    out_dir  = args.output_dir or slam_dir
    os.makedirs(out_dir, exist_ok=True)

    slam_prefix = os.path.join(slam_dir, scene)
    out_prefix  = os.path.join(out_dir, f"robot_deploy_{scene}")

    table_json = f"{slam_prefix}_query_instance_table.json"
    sem_q_npz  = f"{slam_prefix}_instance_sem_queries.npz"
    for path in (table_json, sem_q_npz):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing {path}")

    # ── Build per-level tables (backbone is memo-cached across calls) ─────
    per_level: Dict[str, dict] = {}
    for lvl in requested:
        print(f"\n[postproc-multi] === level: {lvl} ===")
        per_level[lvl] = build_instance_table(
            instance_table_json = table_json,
            sem_queries_npz     = sem_q_npz,
            label_names         = level_to_labels[lvl],
            backbone_name       = args.backbone,
            drop_filtered       = (not args.include_filtered),
        )

        out_path = f"{out_prefix}_instance_semantic_table_{lvl}.json"
        with open(out_path, 'w') as f:
            json.dump(per_level[lvl], f, indent=2)
        print(f"[postproc-multi] wrote {lvl:6s} table → {out_path}  "
              f"({per_level[lvl]['num_instances_after_filter']}"
              f"/{per_level[lvl]['num_instances_total']} kept)")

    # ── Per-level sem-colour PLY + legend PNG ─────────────────────────────
    # Same scheme as postprocess_slam_for_robot's `_sem_color.ply` — per-point
    # cloud, each point gets the colour of its instance's class.  We do the
    # shared geo-load / KDTree / filter / subsample ONCE and loop the cheap
    # per-level relabelling so N levels ≈ same wall-clock as 1.
    if not args.no_sem_maps:
        geo_ply    = f"{slam_prefix}_geo.ply"
        labels_npz = f"{slam_prefix}_instance_labels.npz"
        missing = [p for p in (geo_ply, labels_npz) if not os.path.isfile(p)]
        if missing:
            print(f"\n[postproc-multi] WARN: missing {missing} — skipping "
                  f"per-level sem-colour PLYs.")
        else:
            print(f"\n[postproc-multi] loading geo per-point cloud + instance voxels")
            pts, rgb = load_geo_per_point(geo_ply)
            voxel_data = np.load(labels_npz)
            voxel_centers = voxel_data['centers'].astype(np.float32)
            voxel_inst    = voxel_data['cluster_id'].astype(np.int32)
            voxel_size    = float(per_level[requested[0]].get('voxel_size') or 0.05)

            inst_per_pt, _ = inst_id_per_point(
                pts, voxel_centers, voxel_inst,
                voxel_size       = voxel_size,
                max_dist_factor  = args.knn_max_dist_factor,
            )
            n_orphan = int((inst_per_pt < 0).sum())
            print(f"[postproc-multi] {len(pts):,} pts, {n_orphan:,} orphans "
                  f"(> {args.knn_max_dist_factor:.1f}x voxel)")

            # Filter (drop unmatched + filtered instances) using the FIRST
            # level's kept inst_id set — build_instance_table applies the
            # same drop_filtered logic across all levels, so the kept set
            # is identical for every level (verified inside _combine).
            ref_lvl = requested[0]
            if not args.include_filtered:
                kept_iids = {int(e['inst_id']) for e in per_level[ref_lvl]['instances']}
                mask = np.array([int(i) in kept_iids for i in inst_per_pt], dtype=bool)
                n_before = len(pts)
                pts         = pts[mask]
                inst_per_pt = inst_per_pt[mask]
                print(f"[postproc-multi] PLY filter: {len(pts):,}/{n_before:,} "
                      f"pts kept ({n_before - len(pts):,} dropped: filtered "
                      f"insts + orphans)")
            else:
                print(f"[postproc-multi] PLY filter: keeping all {len(pts):,} "
                      f"pts (--include_filtered)")

            # Sub-sample
            if args.max_points > 0 and len(pts) > args.max_points:
                pts, inst_per_pt = subsample_points(
                    pts, inst_per_pt, max_points=args.max_points,
                )
                print(f"[postproc-multi] sub-sampled to {len(pts):,} pts "
                      f"(--max_points={args.max_points})")

            # Per-level: cheap relabelling + palette + PLY + legend
            for lvl in requested:
                print(f"\n[postproc-multi] === sem map: {lvl} ===")
                out_ply    = f"{out_prefix}_sem_color_{lvl}.ply"
                out_legend = f"{out_prefix}_semantic_legend_{lvl}.png"
                export_sem_color_ply_for_level(
                    pts          = pts,
                    inst_per_pt  = inst_per_pt,
                    level_table  = per_level[lvl],
                    labels       = level_to_labels[lvl],
                    out_ply      = out_ply,
                    out_legend   = out_legend,
                )
                # Brief class-frequency summary (which classes got most points)
                iid_to_class = {int(e['inst_id']): int(e['class_id'])
                                for e in per_level[lvl]['instances']}
                cls_pt = np.array(
                    [iid_to_class.get(int(i), -1) for i in inst_per_pt],
                    dtype=np.int32,
                )
                uniq, cnt = np.unique(cls_pt[cls_pt >= 0], return_counts=True)
                top = sorted(zip(uniq.tolist(), cnt.tolist()),
                             key=lambda x: -x[1])[:5]
                top_str = ", ".join(
                    f"{level_to_labels[lvl][i]}({c:,})" for i, c in top
                )
                print(f"[postproc-multi] wrote {lvl:6s} sem PLY → {out_ply}")
                print(f"[postproc-multi] wrote {lvl:6s} legend  → {out_legend}")
                print(f"[postproc-multi]   top-5 classes by point count: {top_str}")

    # ── Combined JSON ─────────────────────────────────────────────────────
    if not args.no_combined and len(per_level) >= 2:
        combined = _combine_level_tables(per_level)
        combined_path = f"{out_prefix}_instance_semantic_table_all_levels.json"
        with open(combined_path, 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"\n[postproc-multi] wrote combined → {combined_path}  "
              f"({combined['num_instances_after_filter']} instances × "
              f"{len(combined['levels'])} levels)")
    elif args.no_combined:
        print("\n[postproc-multi] skipping combined JSON (--no_combined)")
    else:
        print("\n[postproc-multi] only 1 level requested → no combined JSON")


if __name__ == '__main__':
    main()
