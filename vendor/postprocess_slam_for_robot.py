"""
postprocess_slam_for_robot.py
=============================
Post-process a `slam_opense3r/run.py --instance_head_type query` output
directory into three robot-ready artifacts:

    1. robot_deploy_<scene>_camera_trajectory.json
        Per-frame camera pose (c2w) in the format expected by the
        downstream ROS bridge.  Fields:
            timestamp, basename, tx/ty/tz, qx/qy/qz/qw (xyzw),
            transform_matrix (4x4 c2w as list of lists)

    2. robot_deploy_<scene>_instance_semantic_table.json
        One entry per instance after the SLAM-time noise filter, with
        the predicted semantic class assigned by cosine-matching the
        instance-aggregated `sem_query` feature against text embeddings
        of the chosen label set.  Default label set = ScanNet-20 +
        'ceiling' + a few common indoor classes (lamp, tv, monitor,
        microwave, box, pillow, trash can, shelf).

    3. robot_deploy_<scene>_inst_color.ply
       robot_deploy_<scene>_rgb_color.ply
       robot_deploy_<scene>_sem_color.ply
        Three point clouds, all carrying an integer `inst_id` property
        per vertex that indexes into the table from (2).  Point set is
        the per-point cloud from `<scene>_geo.ply` (≈ millions of pts,
        post-conf-filtered); each point's `inst_id` is looked up via
        KDTree from the voxel centres in `<scene>_instance_labels.npz`.
        The three PLYs differ only in colour:
            inst_color → golden-angle palette keyed by inst_id
            rgb_color  → original RGB from <scene>_geo.ply
            sem_color  → golden-angle palette keyed by class_id
        Points whose nearest voxel is farther than 1.5 × voxel_size, or
        whose instance was dropped by the SLAM filter, get inst_id=-1
        and are removed (unless --include_filtered).  Use --max_points
        to cap the cloud size (default 2,000,000).

Inputs (all produced by slam_opense3r/run.py --save_npz on the query head):
    <slam_dir>/<scene>_results.npz                   ← poses
    <slam_dir>/<scene>_instance_labels.npz           ← centers + cluster_id
    <slam_dir>/<scene>_instance_sem_queries.npz      ← per-instance sem feat
    <slam_dir>/<scene>_query_instance_table.json     ← per-instance metadata + filter flag
    <slam_dir>/<scene>_geo.ply                       ← RGB source (optional)

Usage:
    python postprocess_slam_for_robot.py \
        --slam_dir   outputs/slam_opense3r_query/.../scene0011_00_da3_..._pred \
        --scene_name scene0011_00 \
        --backbone   tips-v2-l14

The semantic backbone must match the one used at training time
(currently tips-v2-l14 for the delivered checkpoint).
"""

import argparse
import colorsys
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'thirdparty'))


# ─────────────────────────────────────────────────────────────────────────────
# Default label sets
# ─────────────────────────────────────────────────────────────────────────────

# ScanNet-20 + 'ceiling' (explicitly requested) + a handful of common
# indoor classes that ScanNet's 20-class set omits but show up often
# in real-world deployment.  Keep 'otherfurniture' last as the
# catch-all class — text-alignment will route ambiguous instances here.
DEFAULT_LABEL_SET = [
    # ── ScanNet-20 base ────────────────────────────────────────────────
    'wall', 'floor',
    'ceiling',                         # ← explicitly requested
    'cabinet', 'bed', 'chair', 'sofa', 'table',
    'door', 'window', 'bookshelf', 'picture', 'counter',
    'desk', 'curtain', 'refrigerator', 'shower curtain',
    'toilet', 'sink', 'bathtub',
    # ── Common indoor extras ───────────────────────────────────────────
    'shelf', 'tv', 'monitor', 'microwave',
    'box', 'pillow', 'lamp', 'trash can',
    # ── Catch-all (last) ───────────────────────────────────────────────
    'otherfurniture',
]

SCANNET20_LABELS = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
    'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
    'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
    'otherfurniture',
]


def resolve_label_set(spec: str):
    """`--label_set` accepts 'default' / 'scannet20' / path-to-txt /
    comma-separated names."""
    if spec == 'default':
        return list(DEFAULT_LABEL_SET)
    if spec == 'scannet20':
        return list(SCANNET20_LABELS)
    if os.path.isfile(spec):
        with open(spec) as f:
            return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith('#')]
    return [x.strip() for x in spec.split(',') if x.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# (1) Camera trajectory
# ─────────────────────────────────────────────────────────────────────────────

def export_trajectory(poses_c2w: np.ndarray, out_path: str, fps: float):
    """Write per-frame c2w pose as JSON list compatible with the ROS
    bridge's expected schema (see reference_pred_trajectory.json).

    `poses_c2w`: [T, 4, 4] float — saved by slam_opense3r/run.py as
                 `memory.poses` (camera-to-world, first frame ≈ identity).
    `memory.poses` is pre-allocated to the chunk-padded buffer size, so
    trailing entries past the actually-processed frames are all-zero.
    Skip those so scipy doesn't choke on a degenerate rotation matrix.
    """
    items = []
    skipped = 0
    for t, T_c2w in enumerate(poses_c2w):
        T_c2w = T_c2w.astype(np.float64)
        # All-zero (unwritten) or non-finite frames → skip.
        if not np.isfinite(T_c2w).all() or np.allclose(T_c2w[:3, :3], 0):
            skipped += 1
            continue
        tx, ty, tz = T_c2w[:3, 3]
        qx, qy, qz, qw = R.from_matrix(T_c2w[:3, :3]).as_quat()  # xyzw order
        items.append({
            "timestamp": float(t) / float(fps) if fps > 0 else float(t),
            "basename":  f"{t:06d}",
            "tx": float(tx), "ty": float(ty), "tz": float(tz),
            "qx": float(qx), "qy": float(qy), "qz": float(qz), "qw": float(qw),
            "transform_matrix": T_c2w.tolist(),
        })
    if skipped > 0:
        print(f"[trajectory] skipped {skipped}/{len(poses_c2w)} unwritten "
              f"(all-zero) frames; wrote {len(items)} valid poses")
    with open(out_path, 'w') as f:
        json.dump(items, f, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
# (2) Instance semantic table
# ─────────────────────────────────────────────────────────────────────────────

# Module-level cache so multi-level callers (postprocess_slam_multi_level_*)
# don't pay the backbone-build cost N times for the same `backbone_name`.
_BACKBONE_CACHE: dict = {}


def _get_cached_backbone(backbone_name: str, device: str):
    key = (backbone_name, device)
    if key not in _BACKBONE_CACHE:
        from opense3r.semantic_backbones import build_semantic_backbone
        print(f"[postproc] building semantic backbone '{backbone_name}' for text encoding")
        bb = build_semantic_backbone(backbone_name).to(device)
        bb.eval()
        _BACKBONE_CACHE[key] = bb
    return _BACKBONE_CACHE[key]


def classify_instances(sem_queries: np.ndarray,
                       label_names: list,
                       backbone_name: str,
                       device: str = 'cuda'):
    """sem_queries: [K, D] (already L2-normed at training time, but we
    re-normalise to be safe).  Returns (class_idx [K], max_sim [K]).

    The semantic backbone is memo-cached by (backbone_name, device) so
    multi-level callers can reuse it across many label sets at zero cost.
    """
    from amb3r.tools.semantic_vis_utils import build_text_embeddings_via_backbone

    bb = _get_cached_backbone(backbone_name, device)

    text_feat = build_text_embeddings_via_backbone(
        bb, labels=label_names, device=device,
    ).float()                                              # [C, D]
    text_feat = F.normalize(text_feat, dim=-1)

    sem_q = torch.as_tensor(sem_queries, device=device).float()
    sem_q = F.normalize(sem_q, dim=-1)

    sim = sem_q @ text_feat.t()                            # [K, C]
    max_sim, idx = sim.max(dim=-1)
    return idx.cpu().numpy(), max_sim.cpu().numpy()


def build_instance_table(instance_table_json: str,
                          sem_queries_npz: str,
                          label_names: list,
                          backbone_name: str,
                          drop_filtered: bool = True):
    with open(instance_table_json) as f:
        meta = json.load(f)
    sq = np.load(sem_queries_npz)
    inst_ids_sq = sq['inst_ids'].astype(np.int64)          # [K]
    sem_queries = sq['sem_queries'].astype(np.float32)     # [K, D]

    by_id = {int(e['inst_id']): e for e in meta.get('instances', [])}

    cls_idx, sims = classify_instances(sem_queries, label_names, backbone_name)

    table = []
    n_dropped = 0
    for k, iid in enumerate(inst_ids_sq):
        iid = int(iid)
        entry = by_id.get(iid, {})
        is_filtered = bool(entry.get('filtered', False))
        if drop_filtered and is_filtered:
            n_dropped += 1
            continue
        table.append({
            'inst_id': iid,
            'class_id': int(cls_idx[k]),
            'class_name': label_names[int(cls_idx[k])],
            'cosine_similarity': float(sims[k]),
            'filtered': is_filtered,
            # Geometry metadata copied from the SLAM instance table
            'center':         entry.get('center'),
            'bbox_min':       entry.get('bbox_min'),
            'bbox_max':       entry.get('bbox_max'),
            'bbox_volume_m3': entry.get('bbox_volume_m3'),
            'num_voxels':     entry.get('num_voxels'),
            'mean_objectness': entry.get('mean_objectness'),
            'first_seen':     entry.get('first_seen'),
            'last_seen':      entry.get('last_seen'),
            'count':          entry.get('count'),
        })

    print(f"[postproc] classified {len(inst_ids_sq)} instances, "
          f"dropped {n_dropped} filtered, kept {len(table)}")

    return {
        'labels': label_names,
        'voxel_size': meta.get('voxel_size'),
        'backbone': backbone_name,
        'num_instances_total': len(meta.get('instances', [])),
        'num_instances_after_filter': len(table),
        'instances': table,
    }


# ─────────────────────────────────────────────────────────────────────────────
# (3, 4) Point cloud PLYs with inst_id property
# ─────────────────────────────────────────────────────────────────────────────

def palette_from_ids(ids: np.ndarray, saturation: float = 0.85, value: float = 0.95):
    """Stable golden-angle palette keyed by unique id.  id < 0 → grey."""
    uniq = np.unique(ids)
    GOLDEN = 0.61803398875
    palette = {}
    for k, iid in enumerate(uniq.tolist()):
        if iid < 0:
            palette[iid] = (0.5, 0.5, 0.5)
        else:
            h = (int(iid) * GOLDEN) % 1.0
            palette[iid] = colorsys.hsv_to_rgb(h, saturation, value)
    return palette


def write_ply_xyz_rgb_inst(centers: np.ndarray,
                            colors_01: np.ndarray,
                            inst_ids: np.ndarray,
                            out_path: str):
    """Binary little-endian PLY with x,y,z (float32), red,green,blue
    (uchar) and inst_id (int32)."""
    n = len(centers)
    assert colors_01.shape == (n, 3) and inst_ids.shape == (n,)
    rgb = (np.clip(colors_01, 0.0, 1.0) * 255).astype(np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property int inst_id\n"
        "end_header\n"
    )
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                       ('inst_id', '<i4')])
    arr = np.empty(n, dtype=dtype)
    arr['x'], arr['y'], arr['z'] = centers[:, 0], centers[:, 1], centers[:, 2]
    arr['red'], arr['green'], arr['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    arr['inst_id'] = inst_ids.astype(np.int32)
    with open(out_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(arr.tobytes())


def load_geo_per_point(geo_ply_path: str):
    """Read per-point (xyz, rgb) from `<scene>_geo.ply`."""
    if not (geo_ply_path and os.path.isfile(geo_ply_path)):
        raise FileNotFoundError(f"geo PLY not found: {geo_ply_path}")
    pcd = o3d.io.read_point_cloud(geo_ply_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    rgb = np.asarray(pcd.colors, dtype=np.float32)
    if pts.size == 0:
        raise ValueError(f"{geo_ply_path} has zero points")
    if rgb.size == 0 or rgb.shape[0] != pts.shape[0]:
        print(f"[postproc] WARN: {geo_ply_path} has no per-vertex colour — RGB ply will be grey")
        rgb = np.full_like(pts, 0.7)
    return pts, rgb


def inst_id_per_point(pts: np.ndarray,
                       voxel_centers: np.ndarray,
                       voxel_inst_id: np.ndarray,
                       voxel_size: float,
                       max_dist_factor: float = 1.5):
    """For each point in `pts`, find the nearest voxel centre and copy
    its inst_id.  Points whose nearest-voxel distance is greater than
    `max_dist_factor * voxel_size` get inst_id = -1 (treated as orphan
    / outside any cluster)."""
    tree = cKDTree(voxel_centers)
    dist, idx = tree.query(pts, k=1, workers=-1)
    inst = voxel_inst_id[idx].astype(np.int32)
    too_far = dist > (max_dist_factor * voxel_size)
    inst[too_far] = -1
    return inst, dist


def subsample_points(*arrays, max_points: int, seed: int = 0):
    """Randomly sub-sample a set of co-indexed arrays to <= max_points."""
    if max_points <= 0 or arrays[0].shape[0] <= max_points:
        return arrays
    n = arrays[0].shape[0]
    rng = np.random.default_rng(seed)
    sel = rng.choice(n, size=max_points, replace=False)
    sel.sort()
    return tuple(a[sel] for a in arrays)


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
                   help='Scene prefix used at SLAM time (e.g. scene0011_00 / '
                        'your in-the-wild --demo_name).')
    p.add_argument('--output_dir', default=None,
                   help="Output dir for the three artefacts.  Default = --slam_dir.")
    p.add_argument('--label_set',  default='default',
                   help="'default' (ScanNet20 + ceiling + extras) | 'scannet20' | "
                        "path to a txt file (one class per line, # for comments) | "
                        "comma-separated class list.")
    p.add_argument('--backbone',   default='tips-v2-l14',
                   help='Semantic backbone for text encoding.  Must match the '
                        'one used at training time.')
    p.add_argument('--fps',        type=float, default=30.0,
                   help='Fallback FPS for the trajectory timestamps when the '
                        'SLAM results.npz does not record one.')
    p.add_argument('--include_filtered', action='store_true', default=False,
                   help='Keep instances flagged as `filtered` in the SLAM table '
                        '(they will appear with their assigned class and stay '
                        'in the PLY).  Default: drop them entirely.')
    p.add_argument('--max_points', type=int, default=2_000_000,
                   help='Cap the per-point PLYs to this many points (random '
                        'sub-sample).  0 = no limit; geo.ply itself is capped '
                        'at SLAM time (default 6 M).  2 M is fine for most '
                        'downstream viewers / robot consumers.')
    p.add_argument('--knn_max_dist_factor', type=float, default=1.5,
                   help='Per-point inst_id lookup: any point whose nearest voxel '
                        'centre is farther than this * voxel_size gets '
                        'inst_id = -1 (orphan).')
    p.add_argument('--device',     default='cuda')
    args = p.parse_args()

    slam_dir = args.slam_dir
    scene    = args.scene_name
    out_dir  = args.output_dir or slam_dir
    os.makedirs(out_dir, exist_ok=True)

    slam_prefix = os.path.join(slam_dir, scene)
    # All deploy artefacts use a `robot_deploy_<scene>_*` prefix so they
    # are easy to glob / separate from the intermediate SLAM dumps in the
    # same directory.
    out_prefix  = os.path.join(out_dir, f"robot_deploy_{scene}")

    # ── Resolve label set ────────────────────────────────────────────────
    labels = resolve_label_set(args.label_set)
    if 'ceiling' not in labels:
        print("[postproc] WARN: 'ceiling' missing from label set — adding it")
        labels = labels + ['ceiling']
    print(f"[postproc] label set ({len(labels)}): {labels}")

    # ─────────────────────────────────────────────────────────────────────
    # (1) Trajectory
    # ─────────────────────────────────────────────────────────────────────
    results_npz = f"{slam_prefix}_results.npz"
    if not os.path.isfile(results_npz):
        raise FileNotFoundError(f"missing {results_npz}")
    results = np.load(results_npz)
    poses = results['poses']                                       # [T,4,4] c2w
    fps   = float(results['fps']) if 'fps' in results.files else args.fps

    # Defensive truncation against legacy SLAM runs whose results.npz still
    # carries a chunk-padded buffer with trailing all-zero frames (newer
    # slam_opense3r/run.py truncates at save time; this guard handles old
    # NPZs).  Valid c2w has [3,3]==1; zero-init buffer rows have [3,3]==0.
    valid_mask = poses[:, 3, 3] != 0
    if valid_mask.any() and not valid_mask.all():
        N_valid = int(valid_mask.sum())
        if np.all(valid_mask[:N_valid]):
            # Contiguous prefix → safe to slice.
            print(f"[postproc] truncating results.npz arrays "
                  f"{len(poses)} → {N_valid} (trailing zero frames from "
                  f"a pre-padded SLAM buffer)")
            poses = poses[:N_valid]
        else:
            print(f"[postproc] WARN: {(~valid_mask).sum()} zero-rows scattered "
                  f"inside the {len(poses)}-frame buffer (non-contiguous); "
                  f"export_trajectory will skip them frame-by-frame.")

    traj_path = f"{out_prefix}_camera_trajectory.json"
    export_trajectory(poses, traj_path, fps=fps)
    print(f"[postproc] wrote trajectory       → {traj_path}  "
          f"({len(poses)} frames @ {fps:.2f} fps)")

    # ─────────────────────────────────────────────────────────────────────
    # (2) Instance semantic table
    # ─────────────────────────────────────────────────────────────────────
    table_json = f"{slam_prefix}_query_instance_table.json"
    sem_q_npz  = f"{slam_prefix}_instance_sem_queries.npz"
    for path in (table_json, sem_q_npz):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing {path}")

    table = build_instance_table(
        instance_table_json=table_json,
        sem_queries_npz=sem_q_npz,
        label_names=labels,
        backbone_name=args.backbone,
        drop_filtered=(not args.include_filtered),
    )
    table_path = f"{out_prefix}_instance_semantic_table.json"
    with open(table_path, 'w') as f:
        json.dump(table, f, indent=2)
    print(f"[postproc] wrote semantic table   → {table_path}  "
          f"({table['num_instances_after_filter']}/{table['num_instances_total']} kept)")

    # ─────────────────────────────────────────────────────────────────────
    # (3) Per-point PLY exports — inst_color / rgb_color / sem_color
    # ─────────────────────────────────────────────────────────────────────
    # Per-point source: <scene>_geo.ply (post-conf-filtered, up to 6 M pts).
    # inst_id source:   <scene>_instance_labels.npz (voxel centres + cluster_id).
    # We KDTree-lookup each geo point's nearest voxel centre and copy its
    # cluster_id; points beyond knn_max_dist_factor * voxel_size become -1.
    geo_ply    = f"{slam_prefix}_geo.ply"
    labels_npz = f"{slam_prefix}_instance_labels.npz"
    for path in (geo_ply, labels_npz):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing {path}")

    pts, rgb = load_geo_per_point(geo_ply)
    print(f"[postproc] loaded per-point cloud: {len(pts):,} points")

    voxel_data = np.load(labels_npz)
    voxel_centers = voxel_data['centers'].astype(np.float32)
    voxel_inst    = voxel_data['cluster_id'].astype(np.int32)
    voxel_size    = float(table.get('voxel_size') or 0.05)

    inst_per_pt, _ = inst_id_per_point(
        pts, voxel_centers, voxel_inst, voxel_size,
        max_dist_factor=args.knn_max_dist_factor,
    )
    n_orphan = int((inst_per_pt < 0).sum())
    print(f"[postproc] KNN voxel→point: voxel_size={voxel_size:.3f}, "
          f"{n_orphan:,} orphan points (> {args.knn_max_dist_factor:.1f}x voxel)")

    # Build inst_id → class_id lookup from the table.  Instances missing
    # from the table (e.g. dropped by the SLAM filter and --include_filtered
    # disabled in build_instance_table) won't appear here.
    iid_to_class = {int(e['inst_id']): int(e['class_id']) for e in table['instances']}

    if not args.include_filtered:
        # Drop any point whose inst_id is not in the kept set (covers both
        # filtered instances and orphan points with inst_id = -1).
        kept_iids = set(iid_to_class.keys())
        mask = np.array([int(i) in kept_iids for i in inst_per_pt], dtype=bool)
        n_before = len(pts)
        pts, rgb, inst_per_pt = pts[mask], rgb[mask], inst_per_pt[mask]
        print(f"[postproc] PLY filter: {len(pts):,}/{n_before:,} points kept "
              f"({n_before - len(pts):,} dropped: filtered insts + orphans)")
    else:
        print(f"[postproc] PLY filter: keeping all {len(pts):,} points "
              f"(--include_filtered; orphans stay as inst_id = -1)")

    # Sub-sample to --max_points
    if args.max_points > 0 and len(pts) > args.max_points:
        pts, rgb, inst_per_pt = subsample_points(
            pts, rgb, inst_per_pt, max_points=args.max_points,
        )
        print(f"[postproc] sub-sampled to {len(pts):,} points "
              f"(--max_points={args.max_points})")

    # ── Build per-point class_id (for sem colouring) ─────────────────────
    class_per_pt = np.array(
        [iid_to_class.get(int(i), -1) for i in inst_per_pt],
        dtype=np.int32,
    )

    # ── (a) Instance-coloured PLY ────────────────────────────────────────
    inst_palette = palette_from_ids(inst_per_pt)
    inst_colors  = np.array([inst_palette[int(i)] for i in inst_per_pt], dtype=np.float32)
    inst_ply_out = f"{out_prefix}_inst_color.ply"
    write_ply_xyz_rgb_inst(pts, inst_colors, inst_per_pt, inst_ply_out)
    print(f"[postproc] wrote inst-colour PLY  → {inst_ply_out}")

    # ── (b) RGB-coloured PLY ─────────────────────────────────────────────
    rgb_ply_out = f"{out_prefix}_rgb_color.ply"
    write_ply_xyz_rgb_inst(pts, rgb, inst_per_pt, rgb_ply_out)
    print(f"[postproc] wrote rgb-colour PLY   → {rgb_ply_out}")

    # ── (c) Semantic-coloured PLY ────────────────────────────────────────
    sem_palette = palette_from_ids(class_per_pt, saturation=0.7, value=0.95)
    sem_colors  = np.array([sem_palette[int(c)] for c in class_per_pt], dtype=np.float32)
    sem_ply_out = f"{out_prefix}_sem_color.ply"
    write_ply_xyz_rgb_inst(pts, sem_colors, inst_per_pt, sem_ply_out)
    print(f"[postproc] wrote sem-colour PLY   → {sem_ply_out}")

    print("[postproc] done.")


if __name__ == '__main__':
    main()
