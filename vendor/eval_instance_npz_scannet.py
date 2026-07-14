"""
3D instance segmentation evaluation (AP / AP50 / AP25) -- CLASS-AGNOSTIC.

Why class-agnostic: class quality is already covered by
`eval_semantic_npz_scannet.py` (mIoU / mAcc).  Instance metric should
isolate "can the model SEPARATE distinct instances", regardless of which
class label gets assigned to each cluster.

Workflow:
  1. Load per-voxel cluster assignments from `instance_hdbscan_labels.npz`
     (saved by `instance_hdbscan_cluster.py`).  Semantic NPZ is only used
     in the optional `--also_class_aware` breakdown.
  2. Build per-vertex GT (class_id, instance_id) from
        - `gt_scannet200_rawid_<scene>.ply` (per-vertex `label`)
        - `<scene>.aggregation.json`        (segment groups → object IDs)
        - `<scene>_vh_clean_2.0.010000.segs.json`  (per-vertex segment IDs)
  3. Drop GT instances whose dominant class is in {wall, floor, ceiling}
     (stuff — no instance concept).
  4. kNN-project pred clusters onto GT vertices → per-vertex cluster_id.
  5. For each pred cluster, find best-IoU match against ANY remaining GT
     instance (no class restriction).  Confidence = cluster size
     (n_voxels normalised to [0, 1]).
  6. Standard COCO/ScanNet AP with greedy matching per IoU threshold:
     sort preds by confidence desc; mark TP only when an unmatched GT has
     IoU above the threshold.  AP averaged over tau in [0.5, 0.55, ..., 0.95];
     AP50 / AP25 are fixed-threshold versions.
  7. Optional `--also_class_aware` flag adds per-class AP table (uses
     semantic NPZ + text encoder for class assignment + restricts matching
     to same-class GTs) — for diagnosis, not the main metric.

Example::

    python eval_instance_npz_scannet.py \\
        --hdbscan_labels_npz <out>/clustering/instance_hdbscan_labels.npz \\
        --semantic_npz       <out>/<scene>_semantic_feats.npz \\
        --gt_ply             scannet_200/<scene>/gt_scannet200_rawid_<scene>.ply \\
        --aggregation_json   scannet_200/<scene>/<scene>.aggregation.json \\
        --segs_json          scannet_200/<scene>/<scene>_vh_clean_2.0.010000.segs.json \\
        --benchmark 200 --backbone lseg \\
        --save_csv <out>/eval_instance_scannet200.csv
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'thirdparty'))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

INDOOR_TEMPLATES = ['a photo of a {}']            # match eval_semantic_npz_scannet default

# Classes excluded from instance evaluation (no instance concept).  Matches
# the ScanNet benchmark convention.
STUFF_CLASS_NAMES = {'wall', 'floor'}

IOU_THRESHOLDS_AP = np.arange(0.5, 0.95 + 1e-9, 0.05)   # 10 thresholds for AP


def get_args():
    p = argparse.ArgumentParser(__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hdbscan_labels_npz', type=str, required=True,
                   help="instance_hdbscan_labels.npz from instance_hdbscan_cluster.py")
    p.add_argument('--gt_ply',             type=str, default=None,
                   help="GT PLY with per-vertex `label` (e.g. "
                        "gt_scannet200_rawid_<scene>.ply or "
                        "<scene>_vh_clean_2.labels.ply). Omit if --gt_scenenn_npz.")
    p.add_argument('--aggregation_json',   type=str, default=None,
                   help="<scene>.aggregation.json")
    p.add_argument('--segs_json',          type=str, default=None,
                   help="<scene>_vh_clean_2.0.010000.segs.json")
    p.add_argument('--gt_scenenn_npz',     type=str, default=None,
                   help="SceneNN GT from build_scenenn_gt.py (centers/instance/"
                        "semantic nyu40) — replaces gt_ply/aggregation/segs; "
                        "stuff = nyu40 wall(1)/floor(2)/ceiling(22).")
    p.add_argument('--scenenn_keep_ceiling', action='store_true', default=False,
                   help="SceneNN only: drop ceiling(22) from the stuff set so "
                        "ceiling is a countable thing — MATCHES ESAM's SceneNN "
                        "config (stuff = wall/floor). Use for ESAM-comparable "
                        "SceneNN numbers.")
    p.add_argument('--benchmark',          type=int, choices=[20, 200], default=20,
                   help="Only used to look up STUFF class names; the main "
                        "class-agnostic AP doesn't depend on benchmark size.")
    p.add_argument('--constants_path',     type=str,
                   default=str(REPO_ROOT / 'scannet_200' / 'scannet200_constants.py'))

    p.add_argument('--max_dist',           type=float, default=None,
                   help='Drop GT vertices whose nearest voxel is > this (m). '
                        '(They count as unassigned, neither TP nor FN here.)')

    # Optional class-aware breakdown (uses semantic NPZ + text encoder)
    p.add_argument('--also_class_aware',   action='store_true', default=False,
                   help='In addition to the main class-agnostic AP, compute '
                        'a per-class breakdown using semantic NPZ + text '
                        'encoder for class assignment + match restricted to '
                        'same-class GTs.')
    p.add_argument('--semantic_npz',       type=str, default=None,
                   help="<scene>_semantic_feats.npz (required for "
                        "--also_class_aware and --filter_pred_stuff).")
    p.add_argument('--backbone',           type=str, default='lseg',
                   choices=['lseg', 'tips-v2-b14', 'tips-v2-l14',
                            'tips-v2-so14', 'tips-v2-g14',
                            'radio-v4-so400m',
                            'pe-core-b16-224', 'pe-core-l14-336',
                            'pe-core-g14-448',
                            'clip-vit-b-32', 'clip-vit-b-16',
                            'clip-vit-l-14', 'clip-vit-l-14-336'],
                   help='Used with --also_class_aware / --filter_pred_stuff. '
                        'Keep in sync with eval_semantic_npz_scannet.py.')
    p.add_argument('--prompt_template',    type=str, default=None,
                   help='Used with --also_class_aware / --filter_pred_stuff. '
                        'Default = single "a photo of a {}".')

    # Optional: also drop pred clusters classified as stuff (wall/floor/ceiling).
    # Default-off so it doesn't silently change the class-agnostic numbers.
    p.add_argument('--filter_pred_stuff', action='store_true', default=False,
                   help='Also drop pred clusters whose predicted class is '
                        'in {wall, floor, ceiling} (decided by argmax of '
                        'mean per-cluster sem feature vs text embeddings of '
                        'the full ScanNet20/200 class set).  Requires '
                        '--semantic_npz.  Reduces over-prediction caused by '
                        'the head occasionally emitting stuff-region masks.')

    # Mesh-segment snap (ESAM `segment_smooth` analog): after projecting pred
    # clusters to GT vertices, re-label whole precomputed mesh segments by
    # majority vote (>0.5 of the segment's verts, else cleared to no-instance).
    # Segments come from --segs_json (`segIndices`).  If the segs vertex count
    # differs from the GT basis (e.g. SceneNN voxelized GT cloud), pass
    # --snap_mesh_ply so segment ids are transferred by nearest mesh vertex;
    # GT points farther than --snap_far_cutoff from the mesh are left
    # un-snapped.  Validated: ScanNet 5-scene +13.4 AP, SceneNN 12-scene
    # +2.9 AP under --benchmark_void (see EXPERIMENTS.md 2026-07-11).
    p.add_argument('--segment_snap', action='store_true', default=False,
                   help='Snap pred labels to mesh segments (majority >0.5 per '
                        'segment) before AP — matches ESAM eval protocol. '
                        'Uses --segs_json; needs --snap_mesh_ply when the GT '
                        'basis is not the mesh vertices.')
    p.add_argument('--snap_mesh_ply', type=str, default=None,
                   help='Mesh PLY whose vertices the segs_json indexes '
                        '(required for --segment_snap on SceneNN-style '
                        'voxelized GT).')
    p.add_argument('--snap_far_cutoff', type=float, default=0.10,
                   help='Max GT-point→mesh-vertex distance (m) for segment-id '
                        'transfer; farther points are never snapped.')

    # Official ScanNet-benchmark FP convention (what ESAM & most published
    # numbers use, via mmdet3d `scannet_eval`):
    #   - GT instances with < benchmark_min_region verts are IGNORED (removed
    #     from n_gt; overlapping them doesn't penalize).
    #   - An unmatched pred is NOT counted as FP at threshold tau if the
    #     fraction of its verts on IGNORE regions (stuff classes + verts with
    #     no instance label + small-GT verts) is > tau ("void absorption").
    # Default-off: the stricter legacy protocol (every GT-overlapping
    # unmatched pred is an FP) stays byte-identical.
    p.add_argument('--benchmark_void', action='store_true', default=False,
                   help='Use the official ScanNet-benchmark FP convention: '
                        'unmatched preds mostly covering unlabeled/stuff/'
                        'small-GT regions are ignored instead of counted as '
                        'FP, and GT instances below --benchmark_min_region '
                        'verts are dropped from n_gt. Matches mmdet3d '
                        'scannet_eval (used by ESAM et al.).')
    p.add_argument('--benchmark_min_region', type=int, default=100,
                   help='min_region_sizes of the official benchmark '
                        '(GT instances smaller than this are ignored). '
                        'Only used with --benchmark_void.')

    p.add_argument('--save_csv',           type=str, default=None)
    p.add_argument('--save_json',          type=str, default=None)

    p.add_argument('--device',             type=str, default='cuda')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Constants loader
# ─────────────────────────────────────────────────────────────────────────────

def load_constants(constants_path: str, benchmark: int):
    spec = importlib.util.spec_from_file_location('sc_constants', constants_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if benchmark == 20:
        return (tuple(m.VALID_CLASS_IDS_20),
                tuple(m.CLASS_LABELS_20))
    if benchmark == 200:
        return (tuple(m.VALID_CLASS_IDS_200),
                tuple(m.CLASS_LABELS_200))
    raise ValueError(f'Unsupported benchmark: {benchmark}')


# ─────────────────────────────────────────────────────────────────────────────
# Text encoders (shared with eval_semantic_npz_scannet)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _ensemble(class_labels, templates, encode_fn) -> torch.Tensor:
    feats = []
    for tmpl in templates:
        prompts = [tmpl.format(c) for c in class_labels]
        t = encode_fn(prompts).float()
        feats.append(F.normalize(t, dim=-1))
    pooled = torch.stack(feats, 0).mean(0)
    return F.normalize(pooled, dim=-1)


@torch.no_grad()
def encode_text_lseg(class_labels, templates, device):
    import clip
    model, _ = clip.load('ViT-B/32', device=device, jit=False)
    model.eval()
    def enc(prompts):
        toks = clip.tokenize(prompts).to(device)
        return model.encode_text(toks)
    out = _ensemble(class_labels, templates, enc)
    del model
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def encode_text_via_backbone(class_labels, templates, backbone_name, device):
    from opense3r.semantic_backbones import build_semantic_backbone
    bb = build_semantic_backbone(backbone_name).to(device)
    if backbone_name.startswith('tips-'):
        bb = bb.to(torch.bfloat16)
    bb.eval()
    out = _ensemble(class_labels, templates, bb.encode_text)
    del bb
    torch.cuda.empty_cache()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# GT instance assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_gt_per_vertex_instance(
    aggregation_json_path: str,
    segs_json_path: str,
    num_vertices: int,
) -> np.ndarray:
    """Return per-vertex instance_id (object id from aggregation, -1 = no instance)."""
    with open(aggregation_json_path) as f:
        agg = json.load(f)
    with open(segs_json_path) as f:
        segs = json.load(f)
    seg_idx = np.asarray(segs['segIndices'], dtype=np.int64)
    if seg_idx.shape[0] != num_vertices:
        raise ValueError(
            f"segIndices len ({seg_idx.shape[0]}) != GT PLY vertex count "
            f"({num_vertices})."
        )
    seg_to_inst: Dict[int, int] = {}
    for grp in agg['segGroups']:
        iid = int(grp['objectId'])
        for sid in grp['segments']:
            seg_to_inst[int(sid)] = iid
    out = np.full(num_vertices, -1, dtype=np.int64)
    for i in range(num_vertices):
        out[i] = seg_to_inst.get(int(seg_idx[i]), -1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# AP helpers
# ─────────────────────────────────────────────────────────────────────────────

def voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Pascal-VOC area-under-curve AP (max precision envelope)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def greedy_pr_at_threshold(
    sorted_pred_records: List[Tuple[float, float, int]],   # (conf, max_iou, matched_gt_id)
    n_gt:                int,
    tau:                 float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standard COCO/ScanNet greedy matching at IoU threshold τ:
    walk preds from high to low confidence.  A pred is TP iff its best-match
    GT (the one with `max_iou`) is (a) IoU > τ AND (b) not yet matched by a
    higher-confidence pred."""
    matched_gts = set()
    tp = np.zeros(len(sorted_pred_records), dtype=np.float64)
    for i, (_, iou, gt_id) in enumerate(sorted_pred_records):
        if gt_id < 0:
            continue
        if iou > tau and gt_id not in matched_gts:
            tp[i] = 1.0
            matched_gts.add(gt_id)
    fp = 1.0 - tp
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    recall = cum_tp / max(n_gt, 1)
    return precision, recall


def greedy_pr_at_threshold_void(
    sorted_pred_records: List[Tuple[float, float, int, float]],
    #                      (conf, max_iou, matched_gt_id, ignore_frac)
    n_gt:                int,
    tau:                 float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Official ScanNet-benchmark variant of `greedy_pr_at_threshold`:
    an unmatched pred whose ignore_frac (fraction of its verts on
    void / stuff / small-GT regions) is > τ is SKIPPED (neither TP nor FP)
    — mirrors mmdet3d scannet_eval `evaluate_matches`."""
    matched_gts = set()
    tp, fp = [], []
    for _, iou, gt_id, ignore_frac in sorted_pred_records:
        if gt_id >= 0 and iou > tau and gt_id not in matched_gts:
            matched_gts.add(gt_id)
            tp.append(1.0)
            fp.append(0.0)
        elif ignore_frac > tau:
            continue                       # void-absorbed: not an FP
        else:
            tp.append(0.0)
            fp.append(1.0)
    if not tp:
        return np.zeros(1), np.zeros(1)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    recall = cum_tp / max(n_gt, 1)
    return precision, recall


def compute_ap(
    pred_records: List[Tuple[float, float, int]],   # (confidence, max_iou, matched_gt_id)
    n_gt:         int,
    benchmark_void: bool = False,   # records then carry a 4th field: ignore_frac
) -> Dict[str, float]:
    """AP (avg over [0.5..0.95]) + AP50 + AP25, with greedy matching."""
    if n_gt == 0:
        return {'AP': float('nan'), 'AP50': float('nan'), 'AP25': float('nan'),
                'n_gt': 0, 'n_pred': len(pred_records)}
    if len(pred_records) == 0:
        return {'AP': 0.0, 'AP50': 0.0, 'AP25': 0.0,
                'n_gt': n_gt, 'n_pred': 0}
    pr_fn = greedy_pr_at_threshold_void if benchmark_void \
        else greedy_pr_at_threshold
    sorted_pr = sorted(pred_records, key=lambda x: -x[0])
    ap_per_tau = []
    for tau in IOU_THRESHOLDS_AP:
        p, r = pr_fn(sorted_pr, n_gt, float(tau))
        ap_per_tau.append(voc_ap(r, p))
    p50, r50 = pr_fn(sorted_pr, n_gt, 0.5)
    p25, r25 = pr_fn(sorted_pr, n_gt, 0.25)
    return {
        'AP':    float(np.mean(ap_per_tau)),
        'AP50':  voc_ap(r50, p50),
        'AP25':  voc_ap(r25, p25),
        'n_gt':  n_gt,
        'n_pred': len(pred_records),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mesh-segment snap (ESAM segment_smooth analog, exclusive-label variant)
# ─────────────────────────────────────────────────────────────────────────────

def snap_labels_to_segments(
    labels:  np.ndarray,     # [N] int64 per-vertex pred label (-1 = none)
    seg_ids: np.ndarray,     # [N] int64 segment id per vertex (negative = never snap)
) -> np.ndarray:
    """Majority-snap exclusive labels onto mesh segments: if one label
    (including -1) owns > 0.5 of a segment's vertices the whole segment
    takes it; otherwise the segment is cleared to -1."""
    out = labels.copy()
    order = np.argsort(seg_ids, kind='stable')
    ss = seg_ids[order]
    bounds = np.flatnonzero(np.diff(ss)) + 1
    for chunk in np.split(order, bounds):
        if chunk.size < 2 or seg_ids[chunk[0]] < 0:
            continue
        vals, cnts = np.unique(labels[chunk], return_counts=True)
        k = cnts.argmax()
        out[chunk] = vals[k] if cnts[k] > 0.5 * chunk.size else -1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Per-vertex cluster mapping
# ─────────────────────────────────────────────────────────────────────────────

def project_clusters_to_vertices(
    voxel_centers: np.ndarray,           # [V, 3]
    voxel_cluster_id: np.ndarray,        # [V] int64
    gt_xyz: np.ndarray,                  # [N_v, 3]
    max_dist: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Each GT vertex → nearest voxel's cluster_id.  Returns (cluster_id, valid_mask)."""
    tree = cKDTree(voxel_centers)
    dists, idx = tree.query(gt_xyz, k=1)
    per_vert_cluster = voxel_cluster_id[idx]
    if max_dist is not None:
        valid = dists <= max_dist
    else:
        valid = np.ones(gt_xyz.shape[0], dtype=bool)
    return per_vert_cluster, valid


def classify_clusters_via_text(
    semantic_npz_path:  str,
    voxel_centers:      np.ndarray,      # [V, 3]
    voxel_cluster_id:   np.ndarray,      # [V] int64
    valid_class_ids:    Tuple[int, ...],
    class_labels:       Tuple[str, ...],
    templates:          List[str],
    backbone_name:      str,
    device:             str,
) -> Dict[int, int]:
    """Assign each pred cluster a raw ScanNet class via majority vote of
    per-voxel argmax (sem feature ⋅ text embedding).  Returns
    `{cluster_id: dominant_raw_class_id}` for cluster_id >= 0.

    Mirrors the first half of `_class_aware_breakdown` so the main path
    can reuse the classification for the `--filter_pred_stuff` filter
    without depending on the breakdown being enabled.
    """
    sem = np.load(semantic_npz_path)
    if 'centers' not in sem or 'features' not in sem:
        raise ValueError(f"{semantic_npz_path}: need 'centers' + 'features'")
    sem_centers  = sem['centers'].astype(np.float32)
    sem_features = sem['features'].astype(np.float32)

    # Map sem voxels back to cluster voxels (sem_store and instance_table
    # use independent voxel grids in general).
    if sem_centers.shape[0] != voxel_centers.shape[0]:
        tree_c = cKDTree(voxel_centers)
        _, c_idx = tree_c.query(sem_centers, k=1)
        cluster_id_for_sem = voxel_cluster_id[c_idx]
    else:
        cluster_id_for_sem = voxel_cluster_id

    # Text embeddings — uses the same encoder routines as the main path.
    if backbone_name == 'lseg':
        text_emb = encode_text_lseg(class_labels, templates, device)
    else:
        text_emb = encode_text_via_backbone(
            class_labels, templates, backbone_name, device,
        )
    if text_emb.shape[-1] != sem_features.shape[1]:
        raise SystemExit(
            f"text emb dim ({text_emb.shape[-1]}) != sem feat dim "
            f"({sem_features.shape[1]}).  --backbone={backbone_name} "
            f"doesn't match the model that produced the semantic NPZ."
        )

    feat_t = F.normalize(torch.from_numpy(sem_features).to(device), dim=-1)
    cos    = feat_t @ text_emb.to(device).T
    _, top_idx = cos.max(dim=-1)
    per_vox_class = np.asarray(valid_class_ids, dtype=np.int64)[
        top_idx.cpu().numpy()
    ]

    # Majority vote per cluster.
    cluster_class: Dict[int, int] = {}
    for cid in np.unique(cluster_id_for_sem):
        if cid < 0:
            continue
        votes = per_vox_class[cluster_id_for_sem == cid]
        if votes.size == 0:
            continue
        u, c = np.unique(votes, return_counts=True)
        cluster_class[int(cid)] = int(u[c.argmax()])
    return cluster_class


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()

    if args.gt_scenenn_npz:
        # SceneNN GT uses NYU40 labels; default stuff = wall(1)/floor(2)/
        # ceiling(22).  --scenenn_keep_ceiling drops ceiling from the stuff set
        # (ceiling becomes a countable thing) to MATCH ESAM's SceneNN config
        # (stuff = wall/floor only).
        valid_class_ids, class_labels = [], []
        name_of = {}
        stuff_class_ids = {1, 2} if args.scenenn_keep_ceiling else {1, 2, 22}
        print(f"[inst-eval] SceneNN GT (NYU40)  stuff_class_ids={sorted(stuff_class_ids)} "
              f"(class-agnostic — dropped from GT instance set"
              f"{'; ceiling KEPT as thing = ESAM-matched' if args.scenenn_keep_ceiling else ''})")
    else:
        valid_class_ids, class_labels = load_constants(args.constants_path, args.benchmark)
        name_of = {cid: nm for cid, nm in zip(valid_class_ids, class_labels)}
        stuff_class_ids = {cid for cid, nm in zip(valid_class_ids, class_labels)
                           if nm.lower() in STUFF_CLASS_NAMES}
        print(f"[inst-eval] benchmark=ScanNet{args.benchmark}  "
              f"stuff_class_ids={sorted(stuff_class_ids)}  "
              f"(class-agnostic — these are dropped from GT instance set)")

    # ── 1. Load HDBSCAN voxel cluster IDs.
    lab = np.load(args.hdbscan_labels_npz)
    if 'centers' not in lab or 'cluster_id' not in lab:
        raise ValueError(
            f"{args.hdbscan_labels_npz} must have keys 'centers' + 'cluster_id'. "
            f"Re-run instance_hdbscan_cluster.py to regenerate it."
        )
    voxel_centers    = lab['centers'].astype(np.float32)
    voxel_cluster_id = lab['cluster_id'].astype(np.int64)
    n_clusters_total = int((voxel_cluster_id >= 0).max(initial=-1)) + 1
    print(f"[inst-eval] voxels: {voxel_centers.shape[0]:,}  "
          f"clusters: {n_clusters_total}  "
          f"noise: {(voxel_cluster_id == -1).sum():,}")
    if voxel_centers.shape[0] == 0 or n_clusters_total == 0:
        print("[inst-eval] ⚠️⚠️⚠️  EMPTY prediction NPZ (0 voxels/clusters) — "
              "AP = 0 by definition. Usual cause: SLAM produced an empty "
              "InstanceTable because --inst_obj_threshold gates out every "
              "proposal for this ckpt (objectness range is model-specific; "
              "calibrate with a low-threshold smoke run first).")
        if args.save_csv:
            Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['metric', 'AP(%)', 'AP50(%)', 'AP25(%)',
                            'n_gt', 'n_pred'])
                w.writerow(['class_agnostic', '0.0000', '0.0000', '0.0000',
                            -1, 0])
            print(f"[inst-eval] CSV saved → {args.save_csv} (zeros)")
        return

    # ── 2. GT per-vertex (class, instance): SceneNN npz OR ScanNet ply+agg.
    if args.gt_scenenn_npz:
        g = np.load(args.gt_scenenn_npz)
        gt_xyz       = g['centers'].astype(np.float32)
        gt_inst_id   = g['instance'].astype(np.int64)
        gt_class_raw = g['semantic'].astype(np.int64)          # nyu40
        N_v = gt_xyz.shape[0]
    else:
        ply = PlyData.read(args.gt_ply)
        v = ply['vertex']
        gt_xyz       = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float32)
        gt_class_raw = np.asarray(v['label']).astype(np.int64)
        N_v = gt_xyz.shape[0]
        gt_inst_id   = build_gt_per_vertex_instance(
            args.aggregation_json, args.segs_json, N_v,
        )
    print(f"[inst-eval] GT vertices: {N_v:,}  "
          f"with-instance: {(gt_inst_id >= 0).sum():,}  "
          f"unique instances: {len(np.unique(gt_inst_id[gt_inst_id >= 0]))}")

    # ── 3. Per-vertex predicted cluster via kNN, masked by max_dist if set.
    per_vert_cluster, valid_mask = project_clusters_to_vertices(
        voxel_centers, voxel_cluster_id, gt_xyz, max_dist=args.max_dist,
    )
    if args.max_dist is not None:
        print(f"[inst-eval] valid GT verts (within {args.max_dist}m of voxel): "
              f"{valid_mask.sum():,}/{N_v:,}")

    # ── 3b. Optional mesh-segment snap (ESAM segment_smooth analog).
    if args.segment_snap:
        if not args.segs_json:
            raise SystemExit("--segment_snap requires --segs_json")
        seg_raw = np.asarray(
            json.load(open(args.segs_json))['segIndices'], dtype=np.int64)
        if seg_raw.shape[0] == N_v:
            seg_ids = seg_raw
            print(f"[inst-eval] --segment_snap: segs align with GT verts "
                  f"({N_v:,}), {len(np.unique(seg_ids)):,} segments")
        else:
            if not args.snap_mesh_ply:
                raise SystemExit(
                    f"--segment_snap: segs verts ({seg_raw.shape[0]:,}) != GT "
                    f"basis ({N_v:,}); pass --snap_mesh_ply to transfer ids.")
            mply = PlyData.read(args.snap_mesh_ply)['vertex']
            mv = np.stack([mply['x'], mply['y'], mply['z']], 1).astype(np.float32)
            if mv.shape[0] != seg_raw.shape[0]:
                raise SystemExit(
                    f"--snap_mesh_ply verts ({mv.shape[0]:,}) != segs "
                    f"({seg_raw.shape[0]:,})")
            d, idx = cKDTree(mv).query(gt_xyz, k=1)
            seg_ids = seg_raw[idx]
            far = d > args.snap_far_cutoff
            seg_ids[far] = -(np.arange(int(far.sum())) + 1)  # never snapped
            print(f"[inst-eval] --segment_snap: transferred via {args.snap_mesh_ply} "
                  f"(NN med {np.median(d)*100:.1f}cm, far>{args.snap_far_cutoff}m: "
                  f"{far.mean():.1%} un-snapped), "
                  f"{len(np.unique(seg_ids[seg_ids >= 0])):,} segments")
        labels_eff = np.where(valid_mask, per_vert_cluster, -1)
        per_vert_cluster = snap_labels_to_segments(labels_eff, seg_ids)
        valid_mask = np.ones_like(valid_mask)   # snapped labels supersede max_dist mask

    # ── 4. Build GT instance masks (skip stuff classes; instance is identified
    #       by GLOBAL instance_id, decoupled from class).
    gt_instances: List[Dict] = []
    for iid in np.unique(gt_inst_id[gt_inst_id >= 0]):
        mask = (gt_inst_id == iid)
        if not mask.any():
            continue
        # Dominant class of this instance
        cls_in_inst = gt_class_raw[mask]
        u, c = np.unique(cls_in_inst, return_counts=True)
        dom_class = int(u[c.argmax()])
        if dom_class in stuff_class_ids:
            continue
        gt_instances.append({
            'instance_id': int(iid),
            'dom_class':   dom_class,
            'mask':        mask & valid_mask,
            'n_verts':     int((mask & valid_mask).sum()),
        })
    # Drop empty-after-mask instances
    gt_instances = [g for g in gt_instances if g['n_verts'] > 0]

    # ── 4b. Official-benchmark ignore handling (--benchmark_void):
    #        GT instances < min_region verts are ignored (out of n_gt), and
    #        `ignore_vert` marks every vertex whose overlap must not create
    #        FPs: stuff classes, no-instance-label verts, small-GT verts.
    ignore_vert = None
    if args.benchmark_void:
        small = [g for g in gt_instances
                 if g['n_verts'] < args.benchmark_min_region]
        gt_instances = [g for g in gt_instances
                        if g['n_verts'] >= args.benchmark_min_region]
        ignore_vert = np.isin(gt_class_raw, list(stuff_class_ids)) \
            | (gt_inst_id < 0)
        for g in small:
            ignore_vert |= g['mask']
        print(f"[inst-eval] --benchmark_void: official ScanNet-benchmark FP "
              f"convention ON — {len(small)} small GT (<"
              f"{args.benchmark_min_region} verts) ignored, "
              f"ignore verts: {int(ignore_vert.sum()):,}/{ignore_vert.size:,}")
    n_gt = len(gt_instances)
    print(f"[inst-eval] GT instances (class-agnostic, no-stuff): {n_gt}")

    # ── 5. Build pred clusters with cluster-size confidence + class-agnostic
    #       best-IoU match against the full GT instance pool.
    unique_clusters = np.unique(voxel_cluster_id)
    unique_clusters = unique_clusters[unique_clusters >= 0]
    cluster_sizes = {int(cid): int((voxel_cluster_id == cid).sum())
                     for cid in unique_clusters}
    max_size = max(cluster_sizes.values()) if cluster_sizes else 1

    # ── 5a. (Optional) Drop pred clusters classified as stuff
    #        (wall/floor/ceiling) via sem-feature → text argmax.  Default
    #        off so the default behaviour matches "raw" class-agnostic AP.
    if args.filter_pred_stuff:
        if args.semantic_npz is None:
            raise SystemExit(
                "--filter_pred_stuff requires --semantic_npz so each cluster "
                "can be assigned a class via sem feature ⋅ text embeddings."
            )
        templates = ([args.prompt_template] if args.prompt_template
                     else INDOOR_TEMPLATES)
        cluster_class = classify_clusters_via_text(
            args.semantic_npz, voxel_centers, voxel_cluster_id,
            valid_class_ids, class_labels, templates,
            backbone_name = args.backbone,
            device        = args.device,
        )
        stuff_clusters = {cid for cid, cls in cluster_class.items()
                          if cls in stuff_class_ids}
        n_before = unique_clusters.size
        keep_mask = np.array(
            [int(cid) not in stuff_clusters for cid in unique_clusters],
            dtype=bool,
        )
        unique_clusters = unique_clusters[keep_mask]
        print(f"[inst-eval] --filter_pred_stuff: dropped "
              f"{n_before - unique_clusters.size} pred clusters as "
              f"stuff (kept {unique_clusters.size} of {n_before})")

    # Per-instance confidence source.  Prefer `cluster_score` (mean
    # objectness from the head, written by `dump_instance_labels_npz`);
    # fall back to a normalised cluster-size proxy when absent (older
    # NPZs from HDBSCAN pipeline / legacy SLAM dumps).
    cluster_score_map: Dict[int, float] = {}
    if 'cluster_score' in lab and 'score_ids' in lab:
        sids = lab['score_ids'].astype(np.int64)
        svals = lab['cluster_score'].astype(np.float32)
        cluster_score_map = {int(i): float(s) for i, s in zip(sids, svals)}
    score_source = 'head_mean_objectness' if cluster_score_map else 'size_proxy'
    print(f"[inst-eval] confidence source: {score_source}  "
          f"(n_scored={len(cluster_score_map)})")

    pred_records_agnostic: List[Tuple[float, float, int]] = []
    pred_details: List[Dict] = []
    for cid in unique_clusters:
        pmask = (per_vert_cluster == int(cid)) & valid_mask
        n_pred_verts = int(pmask.sum())
        if n_pred_verts == 0:
            continue
        if cluster_score_map:
            # Use head's mean objectness; instances without an entry
            # (shouldn't happen since dump skips dropped ones) get 0.
            conf = cluster_score_map.get(int(cid), 0.0)
        else:
            conf = cluster_sizes[int(cid)] / max(max_size, 1)
        best_iou = 0.0
        best_gt  = -1
        for g in gt_instances:
            gmask = g['mask']
            inter = int((pmask & gmask).sum())
            union = int((pmask | gmask).sum())
            if union == 0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_gt  = g['instance_id']
        if args.benchmark_void:
            ignore_frac = float((pmask & ignore_vert).sum()) / n_pred_verts
            pred_records_agnostic.append((conf, best_iou, best_gt, ignore_frac))
        else:
            ignore_frac = None
            pred_records_agnostic.append((conf, best_iou, best_gt))
        pred_details.append({
            'cluster_id':   int(cid),
            'n_voxels':     cluster_sizes[int(cid)],
            'n_verts':      n_pred_verts,
            'confidence':   conf,
            'best_iou':     best_iou,
            'matched_gt':   best_gt,
            **({'ignore_frac': ignore_frac} if ignore_frac is not None else {}),
        })

    # ── 6. Class-agnostic AP.
    agnostic = compute_ap(pred_records_agnostic, n_gt,
                          benchmark_void=args.benchmark_void)

    print('=' * 80)
    conv = ('official-benchmark (void absorption)' if args.benchmark_void
            else 'strict-legacy (every GT-overlapping unmatched pred = FP)')
    print(f"  CLASS-AGNOSTIC instance AP   [{conv}]")
    print(f"    AP (avg 0.5..0.95): {agnostic['AP']  *100:7.4f}")
    print(f"    AP50              : {agnostic['AP50']*100:7.4f}")
    print(f"    AP25              : {agnostic['AP25']*100:7.4f}")
    print(f"    n_gt={agnostic['n_gt']}  n_pred={agnostic['n_pred']}")
    print('=' * 80)

    # ── 7. Optional class-aware breakdown.
    per_class_metrics: Dict[int, Dict] = {}
    mAP = mAP50 = mAP25 = float('nan')
    if args.also_class_aware:
        if args.semantic_npz is None:
            raise SystemExit(
                "--also_class_aware requires --semantic_npz so each cluster "
                "can be assigned a class via cosine vs text embeddings."
            )
        per_class_metrics, mAP, mAP50, mAP25 = _class_aware_breakdown(
            args, voxel_centers, voxel_cluster_id, per_vert_cluster, valid_mask,
            valid_class_ids, class_labels, name_of, stuff_class_ids,
            gt_class_raw, gt_inst_id, cluster_sizes, max_size,
            cluster_score_map,
        )

    # ── 8. CSV / JSON output.
    if args.save_csv:
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['metric', 'AP(%)', 'AP50(%)', 'AP25(%)',
                        'n_gt', 'n_pred'])
            w.writerow(['class_agnostic',
                        f"{agnostic['AP']  *100:.4f}",
                        f"{agnostic['AP50']*100:.4f}",
                        f"{agnostic['AP25']*100:.4f}",
                        agnostic['n_gt'], agnostic['n_pred']])
            if args.also_class_aware:
                w.writerow([])
                w.writerow(['class-aware breakdown (mAP across classes)'])
                w.writerow(['', 'mAP', f"{mAP  *100:.4f}",
                            f"mAP50={mAP50*100:.4f}",
                            f"mAP25={mAP25*100:.4f}"])
                w.writerow(['class_id', 'class_name', 'AP(%)', 'AP50(%)',
                            'AP25(%)', 'n_gt', 'n_pred'])
                for cid in sorted(per_class_metrics.keys()):
                    m = per_class_metrics[cid]
                    AP   = 'nan' if np.isnan(m['AP'])   else f"{m['AP']  *100:.4f}"
                    AP50 = 'nan' if np.isnan(m['AP50']) else f"{m['AP50']*100:.4f}"
                    AP25 = 'nan' if np.isnan(m['AP25']) else f"{m['AP25']*100:.4f}"
                    w.writerow([cid, name_of.get(cid, '?'), AP, AP50, AP25,
                                m['n_gt'], m['n_pred']])
        print(f"[inst-eval] CSV saved → {args.save_csv}")

    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        out = {
            'benchmark':       args.benchmark,
            'class_agnostic':  agnostic,
            'pred_details':    pred_details,
        }
        if args.also_class_aware:
            out['class_aware'] = {
                'mAP':       mAP, 'mAP50': mAP50, 'mAP25': mAP25,
                'per_class': {int(cid): m for cid, m in per_class_metrics.items()},
            }
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2)
        print(f"[inst-eval] JSON saved → {args.save_json}")


# ─────────────────────────────────────────────────────────────────────────────
# Optional class-aware breakdown (for diagnosis only)
# ─────────────────────────────────────────────────────────────────────────────

def _class_aware_breakdown(
    args, voxel_centers, voxel_cluster_id, per_vert_cluster, valid_mask,
    valid_class_ids, class_labels, name_of, stuff_class_ids,
    gt_class_raw, gt_inst_id, cluster_sizes, max_size,
    cluster_score_map: Optional[Dict[int, float]] = None,
):
    """Per-class AP with same-class matching.  Uses semantic NPZ + text
    encoder to assign each cluster a class, then matches only same-class
    GTs.  Confidence source matches the class-agnostic main metric (head
    mean_objectness if available, else cluster-size proxy)."""
    sem = np.load(args.semantic_npz)
    sem_centers  = sem['centers'].astype(np.float32)
    sem_features = sem['features'].astype(np.float32)
    if sem_centers.shape[0] != voxel_centers.shape[0]:
        # KD-tree map sem voxels to cluster voxels
        tree_c = cKDTree(voxel_centers)
        _, c_idx = tree_c.query(sem_centers, k=1)
        cluster_id_for_sem = voxel_cluster_id[c_idx]
    else:
        cluster_id_for_sem = voxel_cluster_id
    D_sem = sem_features.shape[1]

    # Text embeddings
    templates = [args.prompt_template] if args.prompt_template else INDOOR_TEMPLATES
    device = args.device
    if args.backbone == 'lseg':
        text_emb = encode_text_lseg(class_labels, templates, device)
    else:
        text_emb = encode_text_via_backbone(class_labels, templates,
                                             args.backbone, device)
    if text_emb.shape[-1] != D_sem:
        raise SystemExit(
            f"text emb dim ({text_emb.shape[-1]}) != sem feat dim ({D_sem}). "
            f"--backbone={args.backbone} doesn't match the model that produced "
            f"the semantic NPZ."
        )

    feat_t = F.normalize(torch.from_numpy(sem_features).to(device), dim=-1)
    cos    = feat_t @ text_emb.to(device).T
    _, top_idx = cos.max(dim=-1)
    per_vox_class_idx = top_idx.cpu().numpy()
    valid_ids_arr = np.asarray(valid_class_ids, dtype=np.int64)
    per_vox_class    = valid_ids_arr[per_vox_class_idx]

    # Per-cluster class via majority vote
    cluster_class: Dict[int, int] = {}
    unique_clusters = np.unique(cluster_id_for_sem)
    unique_clusters = unique_clusters[unique_clusters >= 0]
    for cid in unique_clusters:
        votes = per_vox_class[cluster_id_for_sem == cid]
        u, c = np.unique(votes, return_counts=True)
        cluster_class[int(cid)] = int(u[c.argmax()])

    instance_class_ids = [cid for cid in valid_class_ids
                          if cid not in stuff_class_ids]

    # GT instances grouped by class
    gt_by_class: Dict[int, List[Dict]] = defaultdict(list)
    for iid in np.unique(gt_inst_id[gt_inst_id >= 0]):
        mask = (gt_inst_id == iid) & valid_mask
        if not mask.any():
            continue
        cls_in_inst = gt_class_raw[mask]
        u, c = np.unique(cls_in_inst, return_counts=True)
        dom_class = int(u[c.argmax()])
        if dom_class in stuff_class_ids:
            continue
        gt_by_class[dom_class].append({
            'instance_id': int(iid), 'mask': mask,
        })

    pred_per_class: Dict[int, List[Tuple[float, float, int]]] = defaultdict(list)
    for cid in unique_clusters:
        cls = cluster_class[int(cid)]
        if cls not in gt_by_class and cls not in instance_class_ids:
            continue
        pmask = (per_vert_cluster == int(cid)) & valid_mask
        if not pmask.any():
            continue
        if cluster_score_map:
            conf = cluster_score_map.get(int(cid), 0.0)
        else:
            conf = cluster_sizes[int(cid)] / max(max_size, 1)
        best_iou = 0.0
        best_gt  = -1
        for g in gt_by_class.get(cls, []):
            inter = int((pmask & g['mask']).sum())
            union = int((pmask | g['mask']).sum())
            if union == 0: continue
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_gt  = g['instance_id']
        pred_per_class[cls].append((conf, best_iou, best_gt))

    per_class_metrics: Dict[int, Dict] = {}
    APs, AP50s, AP25s = [], [], []
    for cid in instance_class_ids:
        m = compute_ap(pred_per_class.get(cid, []), len(gt_by_class.get(cid, [])))
        per_class_metrics[cid] = m
        if not np.isnan(m['AP']):    APs.append(m['AP'])
        if not np.isnan(m['AP50']):  AP50s.append(m['AP50'])
        if not np.isnan(m['AP25']):  AP25s.append(m['AP25'])
    mAP   = float(np.mean(APs))   if APs   else 0.0
    mAP50 = float(np.mean(AP50s)) if AP50s else 0.0
    mAP25 = float(np.mean(AP25s)) if AP25s else 0.0

    print('-' * 80)
    print(f"  CLASS-AWARE breakdown  mAP={mAP*100:.4f}  "
          f"mAP50={mAP50*100:.4f}  mAP25={mAP25*100:.4f}")
    print(f"  per-class (only non-empty):")
    for cid in instance_class_ids:
        m = per_class_metrics[cid]
        if m['n_gt'] == 0 and m['n_pred'] == 0:
            continue
        AP   = 'nan' if np.isnan(m['AP'])   else f"{m['AP']  *100:6.2f}"
        AP50 = 'nan' if np.isnan(m['AP50']) else f"{m['AP50']*100:6.2f}"
        AP25 = 'nan' if np.isnan(m['AP25']) else f"{m['AP25']*100:6.2f}"
        print(f"    [{cid:>3}] {name_of[cid]:<26}"
              f" {AP:>6}  {AP50:>6}  {AP25:>6}  "
              f"n_gt={m['n_gt']:>3}  n_pred={m['n_pred']:>5}")
    print('-' * 80)

    return per_class_metrics, mAP, mAP50, mAP25


if __name__ == '__main__':
    main()
