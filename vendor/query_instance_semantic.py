"""
query_instance_semantic.py
==========================
Broadcast per-instance aggregated semantic features (sem_query) back to
voxels / points, text-align against ScanNet20 / 200 class prompts, and
write PLY / NPZ outputs compatible with the existing dense-path eval.

Why
---
The SLAM run already exports a *dense* per-point semantic prediction
(`<scene>_semantic_per_point.ply` + `<scene>_semantic_feats.npz`) by
storing per-voxel mean dense features in `memory.sem_store` and matching
each voxel against text prompts independently.

This script tests an alternative:
    1. Take the per-INSTANCE aggregated semantic feature (`sem_query` —
       the mask-region mean kept on `InstanceRecord` and merged with a
       running average across all proposals).
    2. Broadcast that vector back to every voxel/point owned by the
       instance.
    3. Text-align as usual.

Difference vs the dense path: this enforces "all pixels of one instance
get the same class label", so it can't produce intra-instance label
noise.  It also tests how good the head's instance aggregation is as a
classifier feature (since `sem_query` is the same vector used for the
cross-chunk matching cue).

Inputs (all produced by `slam_opense3r/run.py --save_npz`):
    --instance_labels_npz       <scene>_instance_labels.npz
                                 keys: centers [V,3] + cluster_id [V] int32
                                       (-1 = noise / dropped)
    --instance_sem_queries_npz  <scene>_instance_sem_queries.npz
                                 keys: inst_ids [N] + sem_queries [N,D]
                                       L2-normed + voxel_size scalar
    --geo_ply                   <scene>_geo.ply
                                 optional; needed for per-point PLY only

Outputs:
    <out>_instance_semantic_voxels.ply
        Voxel-resolution PLY, coloured by predicted ScanNet class.
        Voxels whose instance has no sem_query (or is in -1 noise group)
        are coloured grey.
    <out>_instance_semantic_per_point.ply
        Per-point PLY, each point's colour = nearest voxel's class.
        Only written when --geo_ply is provided.
    <out>_instance_semantic_feats.npz
        Drop-in replacement for `<scene>_semantic_feats.npz`:
            centers  [V, 3] float32
            features [V, D] float32 — broadcast sem_query per voxel
        Pass to `eval_semantic_npz_scannet.py` for direct numerical
        comparison vs the dense path.

Usage:
    python query_instance_semantic.py \\
        --instance_labels_npz       outputs/.../<scene>_instance_labels.npz \\
        --instance_sem_queries_npz  outputs/.../<scene>_instance_sem_queries.npz \\
        --geo_ply                   outputs/.../<scene>_geo.ply \\
        --benchmark 20 \\
        --backbone tips-v2-l14 \\
        --output_dir outputs/.../instance_sem
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'thirdparty'))

# Reuse the eval script's text-encoding helpers so backbone/template
# behaviour exactly matches the dense-path eval.
from eval_semantic_npz_scannet import (
    INDOOR_TEMPLATES,
    load_constants,
    encode_text_lseg,
    encode_text_via_backbone,
    save_labeled_ply,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        __doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--instance_labels_npz',      type=str, required=True,
                   help='<scene>_instance_labels.npz: centers + cluster_id')
    p.add_argument('--instance_sem_queries_npz', type=str, required=True,
                   help='<scene>_instance_sem_queries.npz: inst_ids + sem_queries')
    p.add_argument('--geo_ply', type=str, default=None,
                   help='Optional: <scene>_geo.ply — enables the per-point PLY.')

    p.add_argument('--benchmark', type=int, choices=[20, 200], required=True,
                   help='ScanNet benchmark (20 or 200) for the class prompt set.')
    p.add_argument('--constants_path', type=str,
                   default=str(REPO_ROOT / 'scannet_200' / 'scannet200_constants.py'))

    p.add_argument('--backbone', type=str, default='tips-v2-l14',
                   choices=['lseg', 'tips-v2-b14', 'tips-v2-l14',
                            'tips-v2-so14', 'tips-v2-g14',
                            'radio-v4-so400m',   # keep in sync with eval_semantic_npz_scannet.py
                            'pe-core-b16-224', 'pe-core-l14-336',
                            'pe-core-g14-448',
                            'clip-vit-b-32', 'clip-vit-b-16',
                            'clip-vit-l-14', 'clip-vit-l-14-336'],
                   help='Text encoder backbone — must match the model that '
                        'produced sem_query (semantic_feat_expanded dim).')
    p.add_argument('--lseg_ckpt', type=str,
                   default='./checkpoints/demo_e200.ckpt')
    p.add_argument('--prompt_template', type=str, default=None,
                   help='Single template, e.g. "a photo of a {}".  '
                        'Default uses the same indoor list as the dense eval.')
    p.add_argument('--class_prior_sub', action='store_true', default=False,
                   help='Zero-shot CLIP trick: subtract mean text emb before '
                        'cosine matching.')

    p.add_argument('--noise_color', type=int, nargs=3, default=[80, 80, 80],
                   help='RGB for voxels with no instance (cluster_id=-1) or '
                        'instances missing a sem_query.')
    p.add_argument('--per_point_max_dist', type=float, default=None,
                   help='Drop per-point pixels whose nearest voxel center is '
                        'farther than this (m).  Default: no drop.')

    p.add_argument('--output_dir', type=str, required=True)
    p.add_argument('--demo_name', type=str, default=None,
                   help='Output file prefix.  Defaults to the basename of '
                        '--instance_labels_npz minus "_instance_labels.npz".')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_voxel_npz(path: str):
    d = np.load(path)
    if 'centers' not in d or 'cluster_id' not in d:
        raise ValueError(f"{path}: missing 'centers' / 'cluster_id' "
                         f"(got {list(d.keys())})")
    return (d['centers'].astype(np.float32),
            d['cluster_id'].astype(np.int64))


def load_sem_query_npz(path: str):
    d = np.load(path)
    if 'inst_ids' not in d or 'sem_queries' not in d:
        raise ValueError(f"{path}: missing 'inst_ids' / 'sem_queries' "
                         f"(got {list(d.keys())})")
    return (d['inst_ids'].astype(np.int64),
            d['sem_queries'].astype(np.float32))


def broadcast_sem_to_voxels(
    cluster_id:    np.ndarray,   # [V]    int64
    inst_ids:      np.ndarray,   # [N]    int64
    sem_queries:   np.ndarray,   # [N, D] float32 (L2-normed)
):
    """Map each voxel's cluster_id → that instance's sem_query.

    Voxels with cluster_id=-1 (noise/dropped) or whose inst_id isn't in
    `inst_ids` (e.g. dropped instance was excluded from sem_query dump)
    get a zero feature + valid_mask=False.
    """
    V    = cluster_id.shape[0]
    D    = sem_queries.shape[1] if sem_queries.size > 0 else 0
    feat = np.zeros((V, D), dtype=np.float32)
    valid = np.zeros(V, dtype=bool)
    if D == 0:
        return feat, valid

    # Build inst_id → row dict for O(V) lookup.
    id_to_row = {int(iid): r for r, iid in enumerate(inst_ids.tolist())}
    # Vectorised: build a max-id+1 LUT (memory cheap, ids are dense ints).
    max_id = int(max(inst_ids.max() if inst_ids.size else -1,
                     cluster_id.max() if cluster_id.size else -1))
    if max_id < 0:
        return feat, valid
    lut = np.full(max_id + 2, -1, dtype=np.int64)
    for iid, r in id_to_row.items():
        if iid <= max_id:
            lut[iid] = r
    safe = np.clip(cluster_id, -1, max_id)
    rows = np.where(safe >= 0, lut[safe], -1)             # [V]
    hit  = rows >= 0
    feat[hit] = sem_queries[rows[hit]]
    valid[hit] = True
    return feat, valid


def load_geo_ply(path: str) -> np.ndarray:
    ply = PlyData.read(path)
    v = ply['vertex']
    return np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.demo_name is None:
        base = os.path.basename(args.instance_labels_npz)
        for suffix in ('_instance_labels.npz', '.npz'):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        args.demo_name = base
    print(f"[inst-sem] demo_name = {args.demo_name}")

    # ── Load inputs ────────────────────────────────────────────────────────
    centers, cluster_id = load_voxel_npz(args.instance_labels_npz)
    inst_ids, sem_queries = load_sem_query_npz(args.instance_sem_queries_npz)
    V = centers.shape[0]
    D = sem_queries.shape[1] if sem_queries.size > 0 else 0
    print(f"[inst-sem] voxels = {V:,}   instances_with_sem = {inst_ids.size}   "
          f"sem_dim = {D}")
    if V == 0 or D == 0:
        print("[inst-sem] nothing to do — empty input.")
        return

    # ── Broadcast sem_query to per-voxel features ──────────────────────────
    vox_feat, vox_valid = broadcast_sem_to_voxels(
        cluster_id, inst_ids, sem_queries,
    )
    n_valid = int(vox_valid.sum())
    print(f"[inst-sem] voxels with sem feature: {n_valid:,} / {V:,} "
          f"({100.0 * n_valid / V:.1f}%)")

    # Save NPZ in the same schema as <scene>_semantic_feats.npz so the
    # existing eval script eats it directly.
    feats_npz_path = os.path.join(
        args.output_dir,
        f"{args.demo_name}_instance_semantic_feats.npz",
    )
    np.savez_compressed(
        feats_npz_path,
        centers  = centers.astype(np.float32),
        features = vox_feat.astype(np.float32),
    )
    print(f"[inst-sem] Saved feats NPZ → {feats_npz_path}")

    # ── Text embeddings ────────────────────────────────────────────────────
    valid_ids, class_labels, color_map = load_constants(
        args.constants_path, args.benchmark,
    )
    templates = [args.prompt_template] if args.prompt_template else INDOOR_TEMPLATES
    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.backbone == 'lseg':
        text_emb = encode_text_lseg(
            class_labels, templates, args.lseg_ckpt, device,
        )
    else:
        text_emb = encode_text_via_backbone(
            class_labels, templates, args.backbone, device,
        )
    if text_emb.shape[-1] != D:
        raise ValueError(
            f"text emb dim ({text_emb.shape[-1]}) ≠ sem_query dim ({D}). "
            f"--backbone='{args.backbone}' does not match the model that "
            f"produced sem_query.  See `eval_semantic_npz_scannet.py` "
            f"for the standard backbone→dim pairings."
        )
    if args.class_prior_sub:
        text_emb = text_emb - text_emb.mean(0, keepdim=True)
        text_emb = F.normalize(text_emb, dim=-1)
        print("[inst-sem] applied class prior subtraction")

    # ── Voxel → predicted class (cosine argmax) ────────────────────────────
    feats_t = torch.from_numpy(vox_feat).to(device).float()
    # sem_query is stored L2-normed already, but renorm defensively
    # in case dumps come from a different code path.
    feats_t = F.normalize(feats_t, dim=-1)
    text_t  = text_emb.to(device).float()
    cos     = feats_t @ text_t.T                                  # [V, N_cls]
    pred_cls_idx = cos.argmax(dim=-1).cpu().numpy()
    pred_labels_raw = np.asarray(valid_ids, dtype=np.int64)[pred_cls_idx]

    # Mask invalid voxels so save_labeled_ply renders them grey (we do
    # this by setting their raw label to 0 — `save_labeled_ply`'s
    # color_map.get(0, (128,128,128)) falls back to grey).
    pred_labels_raw[~vox_valid] = 0

    # ── Voxel PLY ──────────────────────────────────────────────────────────
    vox_ply = os.path.join(
        args.output_dir,
        f"{args.demo_name}_instance_semantic_voxels.ply",
    )
    save_labeled_ply(centers, pred_labels_raw, color_map, vox_ply)
    print(f"[inst-sem] Saved voxel PLY  → {vox_ply}")

    # ── Per-point PLY (optional) ───────────────────────────────────────────
    if args.geo_ply is not None and os.path.exists(args.geo_ply):
        print(f"[inst-sem] reading geo PLY: {args.geo_ply}")
        pts = load_geo_ply(args.geo_ply)
        if pts.shape[0] == 0:
            print("[inst-sem] empty geo PLY — skipping per-point export.")
        else:
            tree = cKDTree(centers)
            if args.per_point_max_dist is not None:
                dists, idxs = tree.query(
                    pts, k=1, distance_upper_bound=float(args.per_point_max_dist),
                )
                hit = np.isfinite(dists)
            else:
                dists, idxs = tree.query(pts, k=1)
                hit = np.ones(pts.shape[0], dtype=bool)
            per_point_labels = np.zeros(pts.shape[0], dtype=np.int64)
            per_point_labels[hit] = pred_labels_raw[idxs[hit]]
            n_pp_valid = int(hit.sum())
            print(f"[inst-sem] per-point voxel match: {n_pp_valid:,} / "
                  f"{pts.shape[0]:,} ({100.0 * n_pp_valid / pts.shape[0]:.1f}%)")

            pp_ply = os.path.join(
                args.output_dir,
                f"{args.demo_name}_instance_semantic_per_point.ply",
            )
            save_labeled_ply(pts, per_point_labels, color_map, pp_ply)
            print(f"[inst-sem] Saved per-point PLY → {pp_ply}")
    elif args.geo_ply is not None:
        print(f"[inst-sem] WARN: --geo_ply '{args.geo_ply}' not found — "
              f"skipping per-point export.")

    # ── Quick stat: predicted-class distribution ───────────────────────────
    valid_pred = pred_labels_raw[vox_valid]
    if valid_pred.size > 0:
        uniq, cnts = np.unique(valid_pred, return_counts=True)
        order = np.argsort(-cnts)[:10]
        # Build raw-id → name map for printing.
        rid_to_name = {int(rid): name for rid, name in zip(valid_ids, class_labels)}
        print(f"[inst-sem] top-10 predicted classes (by voxel count):")
        for k in order:
            rid = int(uniq[k])
            print(f"  {rid_to_name.get(rid, f'<rid={rid}>'):<24} "
                  f"voxels={int(cnts[k]):>6}")

    print("[inst-sem] done.")


if __name__ == '__main__':
    main()
