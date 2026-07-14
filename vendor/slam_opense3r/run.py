"""
run.py — OpenSe3r SLAM entry point.
=====================================
Usage (predicted geometry, VGGT backend, ScanNet RGB+poses):
    python slam_opense3r/run.py \\
        --backend vggt \\
        --data_path  /path/to/scannet/scene \\
        --ckpt_path  ./checkpoints/opense3r.pt \\
        --results_path ./outputs/slam_opense3r/scene \\
        --demo_type scannet \\
        --resolution 518 336

Usage (GT point cloud + poses):
    python slam_opense3r/run.py \\
        --data_path  /path/to/scannet/scene \\
        --ckpt_path  ./checkpoints/opense3r.pt \\
        --results_path ./outputs/slam_opense3r/scene \\
        --demo_type scannet_gt \\
        --resolution 518 336 \\
        --use_gt

Usage (DA3 backend, in-the-wild RGB folder):
    python slam_opense3r/run.py \\
        --backend da3 \\
        --da3_ckpt /path/to/da3_giant_metric \\
        --data_path  /path/to/folder/of/jpgs \\
        --ckpt_path  ./outputs/opense3r/checkpoint-best.pth \\
        --results_path ./outputs/slam_opense3r/wild_scene \\
        --demo_type in_the_wild \\
        --resolution 518 336

Key args:
    --backend             'vggt' (default) or 'da3'.
    --da3_ckpt            DA3 checkpoint dir (HF format) when --backend=da3.
    --demo_type           'scannet' | 'scannet_gt' | 'in_the_wild'.
    --use_gt              Use GT pts + poses (requires scannet_gt demo_type).
    --feat_storage        'voxel' (default) or 'perpoint'.
    --sem_feat_mode       'expanded' (512-dim, default) or 'compact'.
    --kv_window_size      KV-cache window (int). Default: None (no pruning).
    --save_semantic       Export semantic PLY + text-matched colour map.
    --save_instance       Export instance PLY (PCA coloured).
    --save_npz            Export raw feature .npz files.
"""

import os
import sys
import time
import argparse
import numpy as np
# open3d MUST be imported before torch: open3d bundles its own libstdc++ /
# libgomp, and if torch loads first those symbols resolve to torch's versions,
# causing a SIGSEGV inside open3d at PLY-write time (geometry export) in the
# full SLAM process.  (Isolated o3d writes work fine — only the torch-first
# ordering triggers it.)
import open3d as o3d
import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, 'thirdparty'))
sys.path.append(os.path.join(_root, 'thirdparty', 'STream3R'))

import clip as clip_lib
from torch.utils.data import DataLoader

from opense3r.model import OpenSe3r
from opense3r.model_da3 import OpenSe3rDA3
from slam_opense3r.pipeline import OpenSe3rSLAM
from slam_opense3r.pipeline_gt import OpenSe3rSLAM_GT
from slam_opense3r.pipeline_voxel_kv import (
    OpenSe3rSLAM_VoxelKV, OpenSe3rSLAM_VoxelKV_GT,
)
from slam_opense3r.voxel_kv_memory import VoxelKVConfig
from slam_opense3r.datasets.scannet import (
    ScannetSLAMDataset, ScannetSLAMGTDataset, InTheWildSLAMDataset,
)
from slam_opense3r.memory import PerPointFeatureStore
from slam_semantic.semantic_voxel_map import VoxelFeatureMap

from amb3r.tools.pts_vis import get_pts_mask
from amb3r.tools.semantic_vis_utils import (
    get_scannet_label_and_color_map,
    build_text_embeddings,
    build_text_embeddings_via_backbone,
    save_semantic_color_legend,
    export_semantic_voxels_ply,
    export_semantic_per_point_ply,
    export_instance_voxels_ply,
    export_instance_per_point_ply,
)
from amb3r.vis_instance_hdbscan import export_instance_hdbscan_ply

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _write_ply_xyzrgb(path: str, xyz, rgb):
    """Write a colored point cloud to binary-little-endian PLY WITHOUT open3d.

    The open3d 0.18 wheel links the numpy 1.x C-ABI; under numpy 2.x (this
    env was rebuilt to numpy 2.2) feeding it a large cloud from inside the
    fully-loaded SLAM process SIGSEGVs in its numpy buffer path. This writer
    is pure numpy + a file handle, so it is immune. Output is a standard PLY
    (float32 xyz + uint8 rgb) that any viewer opens.

    xyz: [N,3] float; rgb: [N,3] in [0,1] (float) or [0,255] (uint).
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb)
    if rgb.dtype.kind == 'f':
        rgb = np.clip(rgb * 255.0 + 0.5, 0.0, 255.0)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    n = int(xyz.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode('ascii')
    vert = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                              ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    vert['x'], vert['y'], vert['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vert['r'], vert['g'], vert['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, 'wb') as f:
        f.write(header)
        vert.tofile(f)


def _apply_seed(seed: int):
    """Seed Python / NumPy / Torch and force deterministic cuDNN convs.

    Trade-off: cuDNN's deterministic algorithm is ~5-30% slower than the
    heuristic-selected default; SDPA backends are left at PyTorch defaults
    so flash / mem-efficient attention still applies (they're forward-
    deterministic given the same inputs).  This gets us ~bit-exact reruns
    in practice without paying the math-SDPA penalty.
    """
    import random as _py_random
    _py_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description='OpenSe3r SLAM — semantic + instance mapping',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Determinism
    p.add_argument('--seed', type=int, default=0,
                   help='Python / NumPy / Torch seed.  Also enables '
                        'cudnn.deterministic and disables cudnn.benchmark '
                        '(small perf hit, large reproducibility gain).')

    # Data / model
    p.add_argument('--data_path',    type=str, required=True)
    p.add_argument('--demo_name',    type=str, default='scene',
                   help='Scene name used for output file naming.')
    p.add_argument('--results_path', type=str, default='./outputs/slam_opense3r')
    p.add_argument('--demo_type',    type=str, default='scannet',
                   choices=['scannet', 'scannet_gt', 'in_the_wild', 'scenenn_gt'],
                   help="'scannet' = RGB+poses; 'scannet_gt' = RGB+poses+GT depth; "
                        "'in_the_wild' = RGB folder only (no GT).")
    p.add_argument('--ckpt_path',    type=str,
                   default='./checkpoints/opense3r.pt')
    p.add_argument('--resolution',   type=int, nargs=2, default=[518, 336],
                   metavar=('W', 'H'))
    p.add_argument('--cfg_path',     type=str,
                   default='./slam_opense3r/config.yaml')
    p.add_argument('--vggt_ckpt',      type=str, default='./checkpoints/VGGT.pt',)
    p.add_argument('--lseg_ckpt',      type=str, default='./checkpoints/demo_e200.ckpt',)
    p.add_argument('--backend',        type=str, default='vggt',
                   choices=['vggt', 'da3', 'mapanything'],
                   help='OpenSe3r geometry backbone: vggt (default), da3, or '
                        'mapanything (Meta\'s MapAnything — metric, multi-modal).')
    p.add_argument('--da3_ckpt',       type=str,
                   default='/mnt/HDD1/ricky/slam/amb3r/checkpoints/da3_base',
                   help='DA3 checkpoint dir (HF format), only used when --backend=da3.')

    # ── MapAnything backend ─────────────────────────────────────────────────
    p.add_argument('--ma_model_id',    type=str,
                   default='facebook/map-anything',
                   help='HF hub model id for MapAnything (full variant by '
                        'default; use "facebook/map-anything-apache" for the '
                        'Apache-2.0-licensed release).')
    p.add_argument('--ma_cache_dir',   type=str,
                   default='./checkpoints/mapanything',
                   help='Local HF cache_dir for MapAnything weights.  '
                        'Defaults to ./checkpoints/mapanything so the model '
                        'is downloaded alongside other project checkpoints '
                        'instead of into ~/.cache/huggingface/hub.  Pass an '
                        'empty string ("") to use the default HF cache.')
    p.add_argument('--use_depth_input', action='store_true', default=False,
                   help='Feed depth as a `depth_z` input to MapAnything for '
                        'depth-conditioned multi-modal geometry.  Final '
                        'mapping still uses MapAnything\'s OUTPUT geometry '
                        '(depth-refined).  Requires --backend mapanything '
                        'AND a depth source: --demo_type scannet_gt (GT '
                        'depth), OR --demo_type in_the_wild + --depth_folder '
                        '+ --intrinsics_path (RGBD).')

    # In-the-wild RGBD inputs (consumed by InTheWildSLAMDataset when set).
    p.add_argument('--depth_folder', type=str, default=None,
                   help='Optional depth folder for --demo_type in_the_wild. '
                        'Same file count + ordering as color.  Formats: '
                        'uint16 PNG (millimetres), 32-bit TIFF, or .npy/.npz '
                        '(field "depth", float32 metres).')
    p.add_argument('--intrinsics_path', type=str, default=None,
                   help='Optional intrinsics text file (9 floats = 3x3 K, OR '
                        '4 floats = fx fy cx cy).  K is at depth\'s native '
                        'resolution and rescaled to --resolution.  Required '
                        'whenever --depth_folder is set.')
    p.add_argument('--depth_scale', type=float, default=1000.0,
                   help='Divisor applied to raw depth values to get metres. '
                        '1000.0 (default) for ScanNet-style uint16 PNG in '
                        'millimetres; use 1.0 when depth is already in metres.')

    # ── Geometry backbone selection ─────────────────────────────────────────
    # 'da3' (default): legacy pipeline — DA3/VGGT geometry forward per chunk +
    #                  coordinate_alignment + keyframe management.
    # 'lingbot':       streaming geometry via LingBot-Map (GCTStream).  Skips
    #                  coord alignment / KF mgmt (LingBot's anchor context +
    #                  paged KV handles those).  Pairs with the OpenSe3r
    #                  forward_semantic path only — query-head instance mode.
    p.add_argument('--geo_backend', type=str, default='da3',
                   choices=['da3', 'lingbot'],
                   help='Geometry source.  da3 (default): per-chunk DA3/VGGT '
                        'pipeline.  lingbot: streaming LingBot-Map (GCTStream) '
                        'paired with OpenSe3r forward_semantic.  When '
                        '`lingbot` is selected: --use_gt is rejected, '
                        '--memory_mode is forced to default, '
                        '--instance_head_type must be `query`.')
    p.add_argument('--lingbot_ckpt', type=str,
                   default='./checkpoints/lingbot_map_long/lingbot-map-long.pt',
                   help='LingBot-Map checkpoint path (only used when '
                        '--geo_backend=lingbot).')
    p.add_argument('--lingbot_num_scale_frames', type=int, default=8,
                   help='Number of anchor / scale-phase frames for LingBot. '
                        'Default 8 matches the LingBot demo; too few (e.g. 3) '
                        'weakens coordinate/scale grounding.')
    p.add_argument('--lingbot_chunk_size', type=int, default=8,
                   help='Frames per OpenSe3r semantic chunk after the scale '
                        'phase.  Matches the CausalBranch training chunk size. '
                        '(In geo_mode=whole_seq the geometry is driven '
                        'frame-by-frame by inference_streaming regardless.)')
    p.add_argument('--lingbot_geo_mode', type=str, default='whole_seq',
                   choices=['whole_seq', 'interleaved'],
                   help="LingBot geometry driving. 'whole_seq' (default): one "
                        "demo-exact inference_streaming over the full sequence "
                        "(geometry identical to the LingBot reference; geo "
                        "CPU-offloaded, sliced per chunk).  'interleaved': "
                        "per-chunk forward (online, experimental).")
    p.add_argument('--lingbot_kv_window', type=int, default=64,
                   help="LingBot global-attention KV cache sliding window "
                        "(in tokens / blocks; see GCTStream).")
    p.add_argument('--lingbot_enable_3d_rope', action='store_true',
                   default=True,
                   help='Enable 3D RoPE on LingBot global attention (default '
                        'True — matches the released long-video model).')
    p.add_argument('--lingbot_no_3d_rope', dest='lingbot_enable_3d_rope',
                   action='store_false',
                   help='Disable 3D RoPE on LingBot (use only when ckpt '
                        "doesn't support it).")
    p.add_argument('--lingbot_max_frame_num', type=int, default=4096,
                   help='LingBot 3D-RoPE max frame index (raise if sequence > 4096).')
    p.add_argument('--lingbot_use_sdpa', action='store_true', default=True,
                   help='Use the SDPA attention backend instead of FlashInfer. '
                        'Default True — avoids the FlashInfer dependency. '
                        'Set `--lingbot_no_sdpa` only when FlashInfer is '
                        'installed and you want faster paged KV cache.')
    p.add_argument('--lingbot_no_sdpa', dest='lingbot_use_sdpa',
                   action='store_false',
                   help='Use FlashInfer attention backend for LingBot.')
    p.add_argument('--lingbot_image_size', type=int, default=518,
                   help='Image size LingBot was trained on (must divide patch_size).')
    p.add_argument('--lingbot_patch_size', type=int, default=14,
                   help='Patch size for LingBot DINOv2 backbone.')

    # In-the-wild specific
    p.add_argument('--max_frames',     type=int, default=None,
                   help='Limit number of frames (in_the_wild only).')
    p.add_argument('--frame_stride',   type=int, default=1,
                   help='Subsample every N-th frame (in_the_wild only).')
    p.add_argument('--keyframe_disparity', type=float, default=0.0,
                   help='VGGT-SLAM-style LK-disparity keyframe selection: keep a '
                        'frame only when mean optical-flow displacement vs the last '
                        'keyframe exceeds this many pixels (0 = off, use all frames). '
                        'Drops redundant low-parallax frames before SLAM. Applies to '
                        'ALL per-frame batch tensors (images + GT depth/pts/poses) so '
                        'GT stays aligned. Typical: 20-30 for ScanNet.')

    # Model dims (must match checkpoint)
    p.add_argument('--sem_dim',  type=int, default=64,
                   help='Compact semantic feature dim used during training.')
    p.add_argument('--ins_dim',  type=int, default=16,
                   help='Instance feature dim.')
    p.add_argument('--n_blocks', type=int, default=12,
                   help='Number of CausalBranch blocks.')

    # ── Instance head type ──────────────────────────────────────────────────
    # 'dense' (default): legacy FeatureDPTHead → per-pixel ins_feat (16-d).
    #                    Uses the legacy SLAM pipeline + HDBSCAN export.
    # 'query':           Mask2Former-style head → mask+obj+desc per slot.
    #                    Uses OpenSe3rQuerySLAM with persistent InstanceTable.
    p.add_argument('--instance_head_type', type=str, default='dense',
                   choices=['dense', 'query'],
                   help='Must match the checkpoint.  See model.py for arch.')
    p.add_argument('--query_arch', type=str, default='mask2former',
                   choices=['maskformer', 'mask2former'])
    p.add_argument('--query_num_queries',        type=int, default=100)
    p.add_argument('--query_d_model',            type=int, default=256)
    p.add_argument('--query_c_mask',             type=int, default=256)
    p.add_argument('--query_descriptor_dim',     type=int, default=256)
    p.add_argument('--query_descriptor_mlp_layers', type=int, default=0)
    p.add_argument('--query_use_box_head',       action='store_true', default=False)
    p.add_argument('--use_merge_head',           action='store_true', default=False,
                   help='Build the learned merge head so its trained weights load '
                        'from the ckpt (required for --inst_score_combine learned).')
    p.add_argument('--query_box_mode',           type=str, default='aabb',
                   choices=['aabb', 'obb', 'world_aabb', 'world_amodal', 'world_amodal_grow'])
    # Box-head sub-design flags — MUST match training (box head input dim
    # depends on query_only + extent_aware; mismatch → ckpt box head won't load).
    p.add_argument('--query_box_size_anchor', action='store_true', default=False)
    p.add_argument('--query_box_query_only', action='store_true', default=False)
    p.add_argument('--query_box_size_extent_aware', action='store_true', default=False)
    p.add_argument('--query_fastsam_mask_prior', action='store_true', default=False,
                   help='ESAM-style FastSAM mask prior (layer-0 attn_mask = region); '
                        'must match training.')
    p.add_argument('--gt_anchor_mode', type=str, default='se3',
                   choices=['se3', 'translate'],
                   help="GT geometry frame-0 anchoring — MUST match training "
                        "(translate = gravity-aligned tight boxes).")
    p.add_argument('--query_n_heads',            type=int, default=8)
    p.add_argument('--query_n_layers',           type=int, default=9)
    p.add_argument('--query_num_feature_levels', type=int, default=3)
    # ── FastSAM-region-seeded queries (ESAM-style) ──────────────────────────
    # MUST match training: a ckpt trained with --fastsam_query has external,
    # FastSAM-seeded queries (no learnable query_feat) + region_pool weights.
    # Building the SLAM model WITHOUT these → query_init falls back to
    # 'learnable', region_pool loads into nothing, no FastSAM seeding at
    # inference → silent garbage.  Requires --query_init_type external and
    # --query_num_queries == --fastsam_max_regions.
    p.add_argument('--fastsam_query', action='store_true', default=False,
                   help='Seed instance queries from FastSAM regions at inference '
                        '(must match training).')
    p.add_argument('--fastsam_ckpt', type=str,
                   default='/mnt/HDD4/ricky/sam_test/FastSAM-x.pt',
                   help='FastSAM-x weights (ultralytics).')
    p.add_argument('--fastsam_conf', type=float, default=0.2,
                   help='FastSAM confidence threshold (region keep).')
    p.add_argument('--fastsam_iou',  type=float, default=0.9,
                   help='FastSAM NMS IoU threshold.')
    p.add_argument('--fastsam_max_regions', type=int, default=100,
                   help='Max FastSAM regions / query slots (must == query_num_queries).')
    # Descriptor-strengthening design (3D PE + instance queries in CausalBranch).
    # All default to legacy/off so old checkpoints load unchanged.  Must match
    # the values used at training time.
    p.add_argument('--query_init_type', type=str, default='learnable',
                   choices=['learnable', '3d_anchor', 'external'],
                   help='Mask2Former query init.  external = queries come from '
                        'the CausalBranch (requires --num_instance_queries>0).')
    p.add_argument('--pos3d_cond', action='store_true', default=False,
                   help='3DPPE-style 3D positional encoding residual at the '
                        'CausalBranch input.')
    p.add_argument('--point_cond', action='store_true', default=False,
                   help='Feed normalized world points (xyz) via PointEncoder into '
                        'the depth-cross-attn path (must match training).')
    p.add_argument('--rope_mode', type=str, default='2d',
                   choices=['2d', '3d_full', '3d_cross'],
                   help='CausalBranch RoPE mode (must match training).')
    p.add_argument('--rope_voxel_size', type=float, default=0.1)
    p.add_argument('--no_depth_cross_attn', action='store_true', default=False,
                   help='Disable the middle-layer DepthCrossAttn in CausalBranch '
                        '(use_depth_cross_attn = not this).')
    p.add_argument('--num_instance_queries', type=int, default=0,
                   help='Instance queries routed through the CausalBranch '
                        '(Idea 2).  0 = off.')
    p.add_argument('--instance_query_block_indices', type=str, default=None,
                   help='Comma-separated CausalBranch block indices at which '
                        'instance queries are refined.  None = default [2,5,8,11].')
    p.add_argument('--instance_queries_init_type', type=str, default='learnable',
                   choices=['learnable', '3d_anchor'],
                   help='Init of the CausalBranch-internal instance queries.')
    # InstanceTable / merging knobs (only used in query mode).
    p.add_argument('--inst_voxel_size',          type=float, default=0.05)
    p.add_argument('--inst_knn_radius',          type=float, default=2.0,
                   help='Spatial KNN shortlist radius (m) for cross-chunk matching. '
                        'Set huge (e.g. 1e9) to disable the spatial gate (match all).')
    p.add_argument('--inst_max_candidates',      type=int, default=32,
                   help='Cap on KNN candidate count (hungarian path). Set large '
                        '(e.g. 9999) WITH a huge --inst_knn_radius to match ALL instances.')
    p.add_argument('--inst_match_threshold',     type=float, default=0.35)
    # CTVIS-aligned instance accumulation + matching (defaults = legacy).
    p.add_argument('--inst_embed_update', type=str, default='mean',
                   choices=['mean', 'momentum', 'sim_guided'],
                   help="InstanceTable descriptor accumulation across merges. "
                        "'mean' (default) cumulative mean; 'momentum' EMA; "
                        "'sim_guided' outlier-robust EMA (mirrors CTVIS training "
                        "--desc_contrastive_mode momentum). Match the trained ckpt.")
    p.add_argument('--inst_embed_momentum', type=float, default=0.75,
                   help="EMA momentum when --inst_embed_update momentum.")
    p.add_argument('--inst_match_metric', type=str, default='hungarian',
                   choices=['hungarian', 'bisoftmax'],
                   help="Matching score: 'hungarian' (raw fused score) or "
                        "'bisoftmax' (CTVIS mutual (d2t+t2d)/2 before threshold; "
                        "use a LOWER --inst_match_threshold, softmax normalises).")
    # ── NEW matching path: greedy + amodal-box overlap gate + desc dead-zone ──
    p.add_argument('--inst_match_mode', type=str, default='hungarian',
                   choices=['hungarian', 'greedy_deadzone'],
                   help="'hungarian' (legacy weighted-fusion 1-to-1) or "
                        "'greedy_deadzone' (greedy many-to-one: amodal-box ∩/min "
                        "GEO gate → desc 3-band merge/discard/spawn). The latter "
                        "uses ONLY --inst_gate_overlap / --inst_merge_desc_high / "
                        "--inst_spawn_desc_low (ignores the w_* fusion weights).")
    p.add_argument('--inst_gate_overlap', type=float, default=0.10,
                   help="greedy_deadzone τ_geo: amodal-box ∩/min gate (loose prune).")
    p.add_argument('--inst_merge_desc_high', type=float, default=0.75,
                   help="greedy_deadzone τ_high: desc-cos > → merge.")
    p.add_argument('--inst_spawn_desc_low', type=float, default=0.40,
                   help="greedy_deadzone τ_low: desc-cos < → spawn new; "
                        "in-between → discard (ambiguous).")
    p.add_argument('--inst_greedy_gate_observed', action='store_true', default=False,
                   help="greedy_deadzone GEO gate uses the OBSERVED (voxel) box "
                        "instead of the amodal pred box (reliable but partial). "
                        "Use when the box head's amodal box is too noisy.")
    # ── ESAM-style proposal post-processing (process_chunk_to_proposals) ──
    p.add_argument('--inst_nms_mode', type=str, default='hard',
                   choices=['hard', 'matrix'],
                   help="'hard' (mask-IoU NMS, delete) | 'matrix' (ESAM soft "
                        "Matrix-NMS score-decay, then re-threshold by obj).")
    p.add_argument('--inst_matrix_nms_kernel', type=str, default='linear',
                   choices=['linear', 'gaussian'])
    p.add_argument('--inst_no_panoptic', action='store_true', default=False,
                   help="Drop per-pixel argmax → each query keeps its own mask "
                        "(masks may overlap; fuller per-proposal geometry; ESAM-style).")
    p.add_argument('--inst_sp_score_thr', type=float, default=0.5,
                   help="mask binarize threshold when --inst_no_panoptic.")
    p.add_argument('--inst_obj_normalize', action='store_true', default=False,
                   help="Re-score objectness by mask quality (ESAM obj_normalization).")
    p.add_argument('--inst_use_sem_conf', action='store_true', default=False,
                   help="A': fold σ(sem_conf) into proposal confidence "
                        "(σ(obj)·σ(sem_conf) → cluster_score / ranking). Needs a "
                        "checkpoint trained with --use_sem_conf_head.")
    p.add_argument('--inst_sem_conf_gate', action='store_true', default=False,
                   help="A': ALSO fold σ(sem_conf) into gating/top-K (not just "
                        "ranking). Use a LOW --inst_obj_threshold + --inst_obj_top_k.")
    p.add_argument('--inst_sem_conf_select', action='store_true', default=False,
                   help="A': fold σ(sem_conf) into SELECTION only — per-pixel "
                        "ownership, top-K, NMS winner — while the recall threshold "
                        "stays pure obj. Tests whether A' picking the better mask "
                        "among competitors improves 3D (without losing recall).")
    p.add_argument('--inst_cluster_score_agg', type=str, default='mean',
                   choices=['mean', 'max'],
                   help="How to aggregate member objectness into the per-instance "
                        "cluster_score (AP ranking). 'max' = best-view (avoids A' "
                        "obj×sem_conf being diluted by the per-instance mean).")
    p.add_argument('--inst_w_desc',              type=float, default=0.35)
    p.add_argument('--inst_w_sem',               type=float, default=0.35)
    p.add_argument('--inst_w_vox',               type=float, default=0.15)
    p.add_argument('--inst_w_geo',               type=float, default=0.15)
    p.add_argument('--inst_sigma_dist',          type=float, default=0.5)
    p.add_argument('--inst_obj_threshold',       type=float, default=0.5)
    p.add_argument('--inst_obj_top_k',           type=int,   default=0,
                   help='Per-frame cap on proposals by top-K objectness, '
                        'AND-ed with --inst_obj_threshold.  0 = off (threshold '
                        'only).  Useful for focal-loss heads whose objectness '
                        'range is compressed — set a small K and lower the '
                        'threshold.')
    p.add_argument('--inst_min_points',          type=int,   default=0,
                   help='Per-frame: drop proposals with fewer than this many '
                        '3D-lifted points BEFORE they enter the InstanceTable '
                        '(ESAM npoint_thr analog — keeps tiny noisy masks out '
                        'of cross-chunk matching). 0 = off (default, unchanged).')
    p.add_argument('--inst_nms_iou',             type=float, default=0.5)
    p.add_argument('--inst_score_threshold',     type=float, default=0.25)
    p.add_argument('--inst_vox_gate',  type=float, default=None,
                   help='Voxel-overlap hard merge gate (None=off). When '
                        'sim_vox > this, force match score = 1.0 (bypass '
                        'linear combination). Targets the red-blue-checkerboard '
                        'pattern (two records claiming the same surface). '
                        'Typical: 0.5')
    p.add_argument('--inst_desc_gate', type=float, default=None,
                   help='Descriptor-cosine hard merge gate (None=off). When '
                        'sim_desc > this, force match score = 1.0. Targets '
                        'long-distance loop closures where voxel overlap is 0 '
                        'but descriptor is confident. Typical: 0.88')
    p.add_argument('--inst_box_iou_mode', type=str, default='min', choices=['min','union'],
                   help="box-IoU geo cue denominator: 'min' (inter/min, containment-friendly) or 'union' (ESAM standard IoU).")
    p.add_argument('--inst_score_combine', type=str, default='add',
                   choices=['add', 'mul', 'mul_voxbonus', 'geo_gated',
                            'mul_descpow', 'or_dual', 'learned'],
                   help="Cue fusion for instance match score. 'add'=weighted sum; "
                        "'mul'=product clamp(d)·vox·geo (w_* = ON/OFF); "
                        "'mul_voxbonus'(A)=d·geo·(1+alpha·vox); "
                        "'geo_gated'(C)=(w_desc·d+w_vox·vox)·geo; "
                        "'mul_descpow'(D)=d**p·geo; "
                        "'or_dual'(E)=max(d·geo, vox if vox>tau else 0). "
                        "NOTE: each mode has a different score scale → re-tune "
                        "--inst_match_threshold per mode.")
    p.add_argument('--inst_vox_bonus_alpha', type=float, default=1.0,
                   help="mode A (mul_voxbonus): vox bonus weight alpha in "
                        "d·geo·(1+alpha·vox).")
    p.add_argument('--inst_desc_pow', type=float, default=1.0,
                   help="mode D (mul_descpow): descriptor power p in d**p·geo "
                        "(>1 sharpens, suppresses mediocre same-class cosines).")
    p.add_argument('--inst_vox_or_tau', type=float, default=0.5,
                   help="mode E (or_dual): voxel-overlap threshold tau; the vox "
                        "OR-path fires only when sim_vox > tau.")
    p.add_argument('--inst_sem_gate_thr', type=float, default=0.0,
                   help='Soft semantic gate (0=off, both add/mul modes). If two '
                        "proposals' sem_query cosine < this, veto the merge "
                        '(different category). Kept as a gate, NOT a scored cue, '
                        'because adding sem to the score inflates same-class '
                        'different-instance pairs. Typical: 0.5')
    p.add_argument('--inst_iqr_mult',            type=float, default=1.5,
                   help='Tukey-fence multiplier in lift_proposal_to_3d.  '
                        'Smaller → tighter bbox/voxel filter (drops more '
                        'mask-edge outliers).  0 disables the filter.')
    p.add_argument('--inst_color_mode',          type=str, default='desc',
                   choices=['rank', 'desc'],
                   help="'desc' (default): assign instance colours via 1D PCA "
                        "on descriptors + golden-angle, so visually similar "
                        "instances get maximally distinct hues.  'rank': "
                        "legacy behaviour (sort by inst_id, golden-angle).")
    # ─── Noise-instance filter (post-SLAM, applies only to query exports) ──
    # Each rule fires only when its threshold > 0; default off.  See
    # amb3r/vis_query_instance_slam.py::filter_noise_instances for details.
    p.add_argument('--inst_filter_min_count',          type=int,   default=0,
                   help='Drop instances whose `count` is below this AND whose '
                        '`last_seen` is more than `--inst_filter_end_grace` '
                        'frames before sequence end.  0 = filter off.  '
                        '1 = drop one-shot mid-sequence ghosts.')
    p.add_argument('--inst_filter_end_grace',          type=int,   default=10,
                   help='Spare instances whose last observation is in the '
                        'final N frames (they had no chance to be re-matched).')
    p.add_argument('--inst_filter_max_size_m3',        type=float, default=0.0,
                   help='Drop instances whose bbox volume exceeds this AND '
                        'whose `count` is at most `--inst_filter_oversized_max_count`.  '
                        '0 = off.  ~20 m³ typically rejects "whole-floor" or '
                        '"whole-wall" head failures.')
    p.add_argument('--inst_filter_oversized_max_count', type=int, default=3)
    p.add_argument('--inst_filter_tiny_voxels',        type=int,   default=0,
                   help='Drop instances with fewer voxels than this AND '
                        '`count` at most `--inst_filter_tiny_max_count`.  '
                        '0 = off (risky — can kill small real objects).')
    p.add_argument('--inst_filter_tiny_max_count',     type=int,   default=2)
    p.add_argument('--inst_filter_max_extent_m',       type=float, default=0.0,
                   help='Drop instances whose max bbox extent (max over '
                        'x/y/z) > this AND count <= '
                        '`--inst_filter_elongated_max_count`.  Catches '
                        'long thin wall/floor strips that pass the '
                        'volume filter.  0 = off.')
    p.add_argument('--inst_filter_elongated_max_count', type=int, default=10)
    p.add_argument('--inst_filter_min_avg_objectness', type=float, default=0.0,
                   help='Drop instances whose average per-merge objectness '
                        '(post-σ, in [0,1]) is below this.  0 = off.  '
                        '0.6 typically rejects low-confidence ghosts.')
    # Channel split (must match training; pass-through to model build).
    p.add_argument('--sem_channels', type=str, default='0,1024')
    p.add_argument('--ins_channels', type=str, default='0,1024')

    # Semantic-branch architecture (must match training)
    p.add_argument('--dino_backbone', type=str, default='vggt',
                   choices=['vggt', 'omega', 'radio', 'lbotv'],
                   help="CausalBranch appearance-token source.  Must match training. "
                        "'radio' = RADIO single encoder replaces DINOv2+semantic "
                        "(forces patch_size=16; needs a radio-* semantic_backbone). "
                        "'lbotv' = LingBot-Vision ViT-L/16 replaces DINOv2, semantic "
                        "backbone kept (forces patch_size=16; da3 backend only; "
                        "--resolution must be multiples of 16, e.g. 512 336).")
    p.add_argument('--lbotv_ckpt', type=str,
                   default='./checkpoints/lingbot_vision_vitl',
                   help="LingBot-Vision weight dir (contains model.pt); used when "
                        "--dino_backbone=lbotv.  (Unrelated to --lingbot_ckpt, "
                        "which is the LingBot-Map geometry backend.)")
    p.add_argument('--semantic_backbone', type=str, default='lseg',
                   help="Frozen 2D backbone for semantic GT/prior.  Must match "
                        "the backbone used at training time (e.g. 'tips-v2-b14').")
    p.add_argument('--prior_method', type=str, default='lseg',
                   choices=['lseg', 'native', 'proxy_clip'],
                   help='Prior extraction mode.  Must match training.')
    p.add_argument('--sem_supervision', type=str, default='lseg',
                   choices=['lseg', 'textregion', 'mask_crops', 'patch_mean'],
                   help='Supervision mode used at training (only for model build; '
                        'inference does not run supervision).')
    p.add_argument('--depth_norm_mode', type=str, default='chunk',
                   choices=['chunk', 'frame', 'none'],
                   help="DepthEncoder input normalization.  Must match the "
                        "training-time setting of the ckpt being loaded — "
                        "use 'frame' for ckpts trained before the chunk-level "
                        "switch, 'chunk' for new training.")
    p.add_argument('--depth_cond', action='store_true', default=False,
                   help='Enable DepthEncoder cross-attention (must match training).')
    p.add_argument('--pose_cond',  action='store_true', default=False,
                   help='Enable pose token conditioning (must match training).')
    p.add_argument('--no_head_sem_cond', action='store_true', default=False,
                   help='Build semantic head WITHOUT prior conditioning at the '
                        'DPT layers (must match training).')
    p.add_argument('--use_sem_conf_head', action='store_true', default=False,
                   help="Build the A' semantic mask-quality head (must match "
                        "training; exposes predictions['sem_conf']).")

    # Pipeline mode
    p.add_argument('--use_gt', action='store_true', default=False,
                   help='Use GT point cloud + poses (requires demo_type=scannet_gt).')
    p.add_argument('--gt_geo_input', action='store_true', default=False,
                   help="(mode B / eval) Feed GT depth + world_points + poses + "
                        "intrinsics directly into the model's geometry "
                        "conditioning via forward_semantic, skipping the geo "
                        "backbone (DA3) entirely. Matches --geo_source=gt "
                        "training. Requires --use_gt. Forces "
                        "--global_scale_mode=gt_seq_median and (da3 backend) "
                        "builds the model without DA3 to save VRAM.")
    p.add_argument('--attn_mode', type=str, default='causal',
                   choices=['causal', 'chunk_bidir'],
                   help='Within-chunk attention for the semantic / instance '
                        'CausalBranch.  "causal" (default): frame i cannot see '
                        'frame j > i within the same chunk — matches training. '
                        '"chunk_bidir": full bidirectional attn within the '
                        'current chunk (cross-chunk causality still preserved '
                        'by KV-cache); usually slightly stronger features at '
                        'the cost of strict causality.')
    p.add_argument('--global_scale_mode', type=str, default='auto',
                   choices=['auto', 'per_chunk', 'first_chunk', 'gt_seq_median'],
                   help="Source of the scale used to normalise depth / "
                        "world_points (pos3d_cond) / pose translation inside "
                        "the model. "
                        "'auto' (default): per backbone — 'per_chunk' for the "
                        "DA3/VGGT pipeline (legacy behaviour), 'first_chunk' "
                        "for --geo_backend lingbot (LingBot's anchor frames "
                        "already fix the scale, so reusing the anchor median "
                        "keeps cross-chunk tokens consistent). "
                        "'per_chunk' (legacy): each chunk's own depth median; "
                        "drifts across chunks. "
                        "'first_chunk': capture the first chunk's depth median "
                        "and reuse it for the whole sequence. "
                        "'gt_seq_median': compute once from GT depths over the "
                        "whole sequence (only with --use_gt; else falls back "
                        "to 'first_chunk').")

    # Output coord-system alignment
    p.add_argument('--align_to_gt_first_frame', action='store_true', default=False,
                   help='Pred-mode only.  Multiplies the predicted map (pts + '
                        'poses) by the GT first-frame c2w so the output sits '
                        'in the dataset GT world frame (instead of the '
                        '"first-input-frame world" the model produces).  '
                        'Useful for ScanNet eval where the GT mesh is in a '
                        'different coordinate system than first-frame=identity.'
                        '  Auto-skipped when --use_gt or when the dataset has '
                        'no camera_pose available.')

    # KV cache
    p.add_argument('--no_cross_chunk_kv', action='store_true', default=False,
                   help='Reset KV cache to empty before every chunk → within-chunk '
                        'attention only, no cross-chunk memory (ablation to test if '
                        'the persistent KV memory helps). Query-GT default pipeline.')
    p.add_argument('--kv_window_size', type=int, default=None,
                   help='KV-cache window (number of recent frames to keep). '
                        'None = keep all (may OOM on long sequences). '
                        'Used only by the default flat-cache pipeline.')
    p.add_argument('--kv_keyframe_stride', type=int, default=None,
                   help='Sparse-keyframe KV mode: keep frames {S, 2S, 3S, ...} '
                        'from absolute index 0, IN ADDITION to anchor (frame 0) '
                        'and the sliding window (if --kv_window_size set).  '
                        'Useful for retaining long-horizon context without '
                        'unbounded KV growth.  None = disabled.  Stride is in '
                        'FRAMES, so e.g. with map_every=4, stride=12 = one '
                        'keyframe per 3 chunks.  Used only by the default '
                        'flat-cache pipeline.')

    # Voxel KV memory (experimental, --memory_mode voxel_kv)
    p.add_argument('--memory_mode', type=str, default='default',
                   choices=['default', 'voxel_kv'],
                   help="'default': flat KV cache (anchor=frame0 + sliding window). "
                        "'voxel_kv': 3-tier (first_frame + keyframe anchor pool + "
                        "window + spatial voxel pivots).  Works in BOTH pred and "
                        "GT modes; GT mode adds inline pose-distance keyframe "
                        "selection so anchor promotion still works.")
    p.add_argument('--vkv_anchor_cap',       type=int,   default=10)
    p.add_argument('--vkv_anchor_mode',      type=str,   default='sticky',
                   choices=['sticky', 'rotate'])
    p.add_argument('--vkv_window_chunks',    type=int,   default=6)
    p.add_argument('--vkv_voxel_size',       type=float, default=0.1)
    p.add_argument('--vkv_pivot_cap',        type=int,   default=4)
    p.add_argument('--vkv_retrieval_radius', type=int,   default=1)
    p.add_argument('--vkv_voxel_layers',     type=int,   nargs='+', default=None,
                   help='Subset of CausalBranch layer indices that participate in '
                        'voxel store.  Default None → all 12 layers.')
    p.add_argument('--vkv_pivot_signal',     type=str,   default='instance_feat',
                   choices=['instance_feat', 'K', 'V'])
    p.add_argument('--vkv_merge_threshold',  type=float, default=0.85)
    p.add_argument('--vkv_update_p',         type=float, default=1.0,
                   help='Count-saturated update exponent.  α = 1/(1+count)^p.')
    p.add_argument('--vkv_update_positions_on_alignment', action='store_true',
                   default=False,
                   help='Re-bucket voxel pivots after SLAM coordinate_alignment '
                        'refines global poses.  Default off (assumes refinement '
                        'magnitude < voxel size).')
    p.add_argument('--vkv_storage', type=str, default='hybrid',
                   choices=['cpu', 'gpu', 'hybrid'],
                   help="Where to hold KV tensors.  'cpu' = all on host (smallest "
                        "VRAM, slower per-step copies).  'gpu' = everything on "
                        "device (fastest, biggest VRAM).  'hybrid' (default) = "
                        "hot tier (first_frame + anchor + window) on GPU, voxel "
                        "pool on CPU (good balance for long sequences).")

    # Feature storage
    p.add_argument('--feat_storage', type=str, default='voxel',
                   choices=['voxel', 'perpoint'],
                   help="'voxel': confidence-weighted voxel map (compact). "
                        "'perpoint': store every projected point (larger).")
    p.add_argument('--sem_feat_mode', type=str, default='expanded',
                   choices=['expanded', 'compact'],
                   help="'expanded': 512-dim LSeg space. "
                        "'compact': model output sem_dim.")

    # Export
    p.add_argument('--save_semantic', action='store_true', default=True,
                   help='Export semantic PLY (text-matched colour + voxel/per-point).')
    p.add_argument('--save_instance', action='store_true', default=True,
                   help='Export instance PLY (PCA colour + HDBSCAN clusters).')
    p.add_argument('--save_npz',      action='store_true', default=False,
                   help='Export raw feature .npz files.')
    p.add_argument('--save_npz_per_point', action='store_true', default=False,
                   help='Also dump a per-point version of '
                        '<scene>_instance_labels.npz alongside the voxel one. '
                        'centers = filtered geo pts (FINAL frame), cluster_id '
                        'looked up via voxel. Closes the voxel-quantisation '
                        'IoU gap (matches per-mesh-vert dumps like ESAM). '
                        'Implies --save_npz + --save_geo.')
    p.add_argument('--save_geo', action=argparse.BooleanOptionalAction, default=True,
                   help='Export geometry point cloud PLY. Pass --no-save_geo to '
                        'skip it: the geo export downsamples the full ~32M-point '
                        'cloud and pushes it through open3d, which under CPU '
                        'memory pressure on large scenes can SIGSEGV and take the '
                        '(later, much smaller) instance/semantic NPZ exports down '
                        'with it. Skipping is safe — the eval-critical NPZ dumps '
                        'do not depend on the geo cloud.')
    # ── Debug: per-frame 2D instance-mask grids (query head only) ────────────
    p.add_argument('--save_2d_inst_masks', action='store_true', default=False,
                   help='Debug: during SLAM, save one grid PNG per frame of '
                        'the query head\'s top-K objectness instance masks '
                        '(each overlaid as a constant-red alpha mask on the '
                        'original RGB frame).  Query head only; default off.  '
                        'PNGs land in <results_path>/2d_inst_masks/.')
    p.add_argument('--save_2d_inst_topk', type=int, default=15,
                   help='Number of top-objectness instance slots to render '
                        'per frame for --save_2d_inst_masks.')

    p.add_argument('--label_set',     type=str, default='scannet20',
                   choices=['scannet20', 'scannet200'])
    p.add_argument('--conf_threshold', type=float, default=None,
                   help='Geometry confidence threshold for masking export pts. '
                        'If unset (default), --conf_threshold_percentile is used '
                        'to derive a per-scene threshold from the conf distribution.')
    p.add_argument('--conf_threshold_percentile', type=float, default=10.0,
                   help='Percentile (0-100) of valid confidences to drop. Only '
                        'used when --conf_threshold is not set. Default 10 = '
                        'cut the bottom 10%% of nonzero conf values.')
    p.add_argument('--max_points', type=int, default=3_000_000,
                   help='Max points in exported geometry / per-point PLYs (0 = no limit).')
    p.add_argument('--edge_mask',     action='store_true', default=False,
                   help='Apply edge / sky masking on geometry export.')

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(args):
    """Build OpenSe3r (vggt) or OpenSe3rDA3 (da3) and load semantic-branch weights.

    Trainable submodule names match across backends, so a checkpoint trained
    on one backbone can be loaded into the other via strict=False.
    """
    sem_ch = tuple(int(x) for x in args.sem_channels.split(','))
    ins_ch = tuple(int(x) for x in args.ins_channels.split(','))
    common_kwargs = dict(
        sem_dim    = args.sem_dim,
        ins_dim    = args.ins_dim,
        n_blocks   = args.n_blocks,
        init_from_vggt    = True,
        dino_backbone     = args.dino_backbone,
        semantic_backbone = args.semantic_backbone,
        prior_method      = args.prior_method,
        sem_supervision   = args.sem_supervision,
        depth_norm_mode   = args.depth_norm_mode,
        gt_anchor_mode    = getattr(args, 'gt_anchor_mode', 'se3'),
        depth_cond        = args.depth_cond,
        point_cond        = getattr(args, 'point_cond', False),
        rope_mode         = getattr(args, 'rope_mode', '2d'),
        rope_voxel_size   = getattr(args, 'rope_voxel_size', 0.1),
        pose_cond         = args.pose_cond,
        use_head_sem_cond = (not args.no_head_sem_cond),
        use_sem_conf_head = getattr(args, 'use_sem_conf_head', False),
        sem_channels      = sem_ch,
        ins_channels      = ins_ch,
        instance_head_type = args.instance_head_type,
        # Descriptor-strengthening design (default off → legacy ckpts unchanged).
        pos3d_cond                   = args.pos3d_cond,
        use_depth_cross_attn         = (not args.no_depth_cross_attn),
        query_init_type              = args.query_init_type,
        num_instance_queries         = args.num_instance_queries,
        instance_query_block_indices = (
            None if args.instance_query_block_indices is None
            else [int(x) for x in args.instance_query_block_indices.split(',') if x.strip()]
        ),
        instance_queries_init_type   = args.instance_queries_init_type,
    )
    if args.instance_head_type == 'query':
        common_kwargs.update(dict(
            query_arch              = args.query_arch,
            query_num_queries       = args.query_num_queries,
            query_d_model           = args.query_d_model,
            query_c_mask            = args.query_c_mask,
            query_descriptor_dim    = args.query_descriptor_dim,
            query_n_heads           = args.query_n_heads,
            query_n_layers          = args.query_n_layers,
            query_num_feature_levels = args.query_num_feature_levels,
            query_descriptor_mlp_layers = getattr(args, 'query_descriptor_mlp_layers', 0),
            query_use_box_head      = getattr(args, 'query_use_box_head', False),
            query_box_mode          = getattr(args, 'query_box_mode', 'aabb'),
            query_box_size_anchor   = getattr(args, 'query_box_size_anchor', False),
            query_box_query_only    = getattr(args, 'query_box_query_only', False),
            query_fastsam_mask_prior = getattr(args, 'query_fastsam_mask_prior', False),
            query_box_size_extent_aware = getattr(args, 'query_box_size_extent_aware', False),
            use_merge_head          = getattr(args, 'use_merge_head', False),
        ))
        # FastSAM-region-seeded queries (must match training).  Only forwarded
        # when enabled so legacy (non-fastsam) ckpts build unchanged.
        if getattr(args, 'fastsam_query', False):
            common_kwargs.update(dict(
                fastsam_query       = True,
                fastsam_ckpt        = args.fastsam_ckpt,
                fastsam_conf        = args.fastsam_conf,
                fastsam_iou         = args.fastsam_iou,
                fastsam_max_regions = args.fastsam_max_regions,
            ))
    if args.backend == 'vggt':
        model = OpenSe3r(
            vggt_ckpt = args.vggt_ckpt,
            lseg_ckpt = args.lseg_ckpt,
            **common_kwargs,
        )
    elif args.backend == 'da3':
        # gt_geo_input (mode B) never runs DA3 → skip building it (VRAM save).
        skip_da3 = getattr(args, 'gt_geo_input', False)
        model = OpenSe3rDA3(
            da3_ckpt  = args.da3_ckpt,
            vggt_ckpt = args.vggt_ckpt,
            lbotv_ckpt = args.lbotv_ckpt,
            lseg_ckpt = args.lseg_ckpt,
            _skip_geo_backbone = skip_da3,
            **common_kwargs,
        )
        if skip_da3:
            print('[run] --gt_geo_input → DA3 backbone NOT built '
                  '(geometry comes from GT batch keys via forward_semantic)')
    elif args.backend == 'mapanything':
        from opense3r.model_mapanything import OpenSe3rMapAnything
        # Empty string ("") on the CLI → fall back to the default HF cache
        # (i.e. ~/.cache/huggingface/hub) instead of project-local override.
        ma_cache = (args.ma_cache_dir or None) if args.ma_cache_dir != "" else None
        model = OpenSe3rMapAnything(
            ma_model_id  = args.ma_model_id,
            ma_cache_dir = ma_cache,
            vggt_ckpt    = args.vggt_ckpt,
            lseg_ckpt    = args.lseg_ckpt,
            **common_kwargs,
        )
    else:
        raise ValueError(f"unknown backend: {args.backend}")
    print(f"[run] backend={args.backend} model_class={type(model).__name__}")

    if os.path.isfile(args.ckpt_path):
        state = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
        if 'model' in state:
            state = state['model']
        msg = model.load_state_dict(state, strict=False)
        print(f"[run] Loaded checkpoint: {args.ckpt_path}")
        # Frozen backbones (vggt.*, da3.*, lseg.*, mapanything.*) are loaded
        # inside __init__, so they appear as 'missing' here — that's expected.
        # Only count the TOP-LEVEL trainable submodules to flag real load
        # problems.  Match by `startswith` so nested modules like
        # `mapanything.depth_encoder.*` (frozen, inside MapAnything) are NOT
        # misclassified as trainable.
        trainable_prefixes = ('depth_encoder.', 'causal_branch.',
                              'semantic_head.', 'instance_head.',
                              'pos3d_encoder.', 'pose_encoder.',
                              'plucker_encoder.', 'region_pool.')
        miss_train  = [k for k in msg.missing_keys
                       if any(k.startswith(p) for p in trainable_prefixes)]
        unexp_train = [k for k in msg.unexpected_keys
                       if any(k.startswith(p) for p in trainable_prefixes)]
        miss_frozen  = len(msg.missing_keys)    - len(miss_train)
        unexp_frozen = len(msg.unexpected_keys) - len(unexp_train)
        print(f"  trainable: missing={len(miss_train)} unexpected={len(unexp_train)}")
        if miss_train:
            print(f"    ! missing trainable keys (first 5): {miss_train[:5]}")
        if unexp_train:
            print(f"    ! unexpected trainable keys (first 5): {unexp_train[:5]}")
        print(f"  frozen backbones (expected missing): "
              f"{miss_frozen} missing, {unexp_frozen} unexpected")
    else:
        print(f"[run] WARNING: checkpoint not found at {args.ckpt_path}")
    model.cuda()
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers for per-point stores (mirrors slam_semantic/run.py)
# ─────────────────────────────────────────────────────────────────────────────

def _export_perpoint_semantic(store: PerPointFeatureStore,
                               text_feat, color_table, out_path,
                               transformation=None):
    pts, feats = store.get_all()
    if pts.shape[0] == 0:
        print(f"  [skip] empty per-point semantic store")
        return
    # Reuse export_semantic_per_point_ply by wrapping store as voxel-query target
    # Actually use direct path: compute label per point
    import torch.nn.functional as F
    feats_n = F.normalize(feats.float(), dim=-1)
    text_n  = F.normalize(text_feat.float(), dim=-1)
    sim     = feats_n @ text_n.t()
    labels  = sim.argmax(dim=-1).numpy()
    colors  = color_table[labels]

    pts_np = pts.numpy()
    if transformation is not None:
        pts_h  = np.concatenate([pts_np, np.ones((len(pts_np), 1), dtype=np.float32)], 1)
        pts_np = (transformation @ pts_h.T).T[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(out_path, pcd)


def _export_perpoint_instance_pca(store: PerPointFeatureStore,
                                   out_path, transformation=None):
    pts, feats = store.get_all()
    if pts.shape[0] == 0:
        print(f"  [skip] empty per-point instance store")
        return
    from sklearn.decomposition import PCA
    pca   = PCA(n_components=3)
    rgb   = pca.fit_transform(feats.numpy().astype(np.float32))
    lo, hi = rgb.min(0), rgb.max(0)
    colors = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)

    pts_np = pts.numpy()
    if transformation is not None:
        pts_h  = np.concatenate([pts_np, np.ones((len(pts_np), 1), dtype=np.float32)], 1)
        pts_np = (transformation @ pts_h.T).T[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    o3d.io.write_point_cloud(out_path, pcd)


# ─────────────────────────────────────────────────────────────────────────────
# Optional per-frame 2D instance-mask dump wiring
# ─────────────────────────────────────────────────────────────────────────────

def _set_2d_mask_dump(pipeline, args):
    """Attach 2D instance-mask dump config to a query-head pipeline.

    No-op-friendly: the pipeline only acts when `save_2d_masks` is True, so
    this is safe to call on any query pipeline.  Only query-head pipelines
    expose `_maybe_dump_2d_masks`; for others (dense head) this just sets
    unused attributes.
    """
    if not getattr(args, 'save_2d_inst_masks', False):
        return
    pipeline.save_2d_masks   = True
    pipeline.mask_2d_out_dir = os.path.join(args.results_path, '2d_inst_masks')
    pipeline.mask_2d_demo_name = args.demo_name
    pipeline.mask_2d_top_k   = int(args.save_2d_inst_topk)
    print(f"[run] --save_2d_inst_masks ON → per-frame top-"
          f"{args.save_2d_inst_topk} instance-mask grids → "
          f"{pipeline.mask_2d_out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    _apply_seed(int(args.seed))
    print(f"[run] seed={args.seed}  cudnn.deterministic=True  benchmark=False")
    os.makedirs(args.results_path, exist_ok=True)
    resolution = tuple(args.resolution)   # (W, H)

    # ── Persist run args so the output dir is self-describing ─────────────────
    # JSON-dump argparse.Namespace — easier to diff between runs / read months
    # later than re-deriving the bash config.  Skip non-serialisable values.
    import json as _json
    _args_path = os.path.join(args.results_path, f"{args.demo_name}_run_args.json")
    try:
        def _to_jsonable(v):
            if isinstance(v, (str, int, float, bool, type(None))):
                return v
            if isinstance(v, (list, tuple)):
                return [_to_jsonable(x) for x in v]
            if isinstance(v, dict):
                return {str(k): _to_jsonable(x) for k, x in v.items()}
            return repr(v)
        with open(_args_path, 'w') as _f:
            _json.dump({k: _to_jsonable(v) for k, v in vars(args).items()},
                       _f, indent=2, sort_keys=True)
        print(f"[run] Saved run args             → {_args_path}")
    except Exception as _e:
        print(f"[run] warn: failed to dump run args ({_e})")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.demo_type == 'scannet_gt':
        dataset = ScannetSLAMGTDataset(args.data_path, resolution=resolution)
        print(f"[run] Loaded ScanNet GT dataset: {args.data_path}")
    elif args.demo_type == 'scenenn_gt':
        from slam_opense3r.datasets.scenenn import SceneNNSLAMGTDataset
        dataset = SceneNNSLAMGTDataset(args.data_path, resolution=resolution)
        print(f"[run] Loaded SceneNN GT dataset: {args.data_path}")
    elif args.demo_type == 'in_the_wild':
        dataset = InTheWildSLAMDataset(
            args.data_path, resolution=resolution,
            max_frames=args.max_frames, frame_stride=args.frame_stride,
            depth_dir       = args.depth_folder,
            intrinsics_path = args.intrinsics_path,
            depth_scale     = args.depth_scale,
        )
        mode = ('RGBD' if args.depth_folder else 'RGB')
        print(f"[run] Loaded in-the-wild {mode} folder: {args.data_path}"
              + (f"  depth={args.depth_folder}  K={args.intrinsics_path}"
                 if args.depth_folder else ""))
    else:
        dataset = ScannetSLAMDataset(args.data_path, resolution=resolution)
        print(f"[run] Loaded ScanNet dataset: {args.data_path}")

    batch    = dataset[0]

    # ── Keyframe selection (LK disparity gate) ────────────────────────────────
    # Drop redundant low-parallax frames BEFORE SLAM.  Subsamples every per-frame
    # batch tensor by the same keyframe indices so images + GT depth/pts/poses stay
    # aligned.  Causal (first frame always kept) → matches streaming behaviour.
    if args.keyframe_disparity and args.keyframe_disparity > 0:
        from slam_opense3r.keyframe_flow import select_keyframes_from_tensor
        T0  = batch['images'].shape[0]
        kf  = select_keyframes_from_tensor(batch['images'], args.keyframe_disparity)
        idx = torch.as_tensor(kf, dtype=torch.long)
        for k, v in list(batch.items()):
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == T0:
                batch[k] = v.index_select(0, idx)
        print(f"[run] keyframe selection (disparity>{args.keyframe_disparity:g}px): "
              f"kept {len(kf)}/{T0} frames ({100.0*len(kf)/T0:.1f}%)")

    images   = batch['images'].unsqueeze(0)       # [1, T, 3, H, W]
    T        = images.shape[1]
    print(f"[run] {T} frames at {images.shape[-1]}×{images.shape[-2]}")

    if args.use_gt and args.demo_type not in ('scannet_gt', 'scenenn_gt'):
        raise ValueError("--use_gt requires --demo_type scannet_gt or scenenn_gt")
    if args.demo_type == 'in_the_wild' and args.use_gt:
        raise ValueError("--use_gt is not supported for in_the_wild data (no GT).")

    # ── Validate / normalize --gt_geo_input (mode B) ─────────────────────────
    if args.gt_geo_input:
        if not args.use_gt:
            raise ValueError("--gt_geo_input requires --use_gt.")
        if args.instance_head_type != 'query':
            raise ValueError("--gt_geo_input is currently implemented for the "
                             "query instance head only.")
        if 'depth_gt' not in batch or 'camera_intrinsics' not in batch:
            raise ValueError("--gt_geo_input needs GT depth + intrinsics from "
                             "the dataset (demo_type=scannet_gt provides them).")
        if args.global_scale_mode != 'gt_seq_median':
            print(f"[run] --gt_geo_input forces --global_scale_mode="
                  f"gt_seq_median (was '{args.global_scale_mode}').")
            args.global_scale_mode = 'gt_seq_median'

    # Resolve the 'auto' global_scale_mode sentinel per geometry backbone:
    #   lingbot → first_chunk  (anchor frames fix the scale; reuse the anchor
    #             median so cross-chunk tokens stay on one scale),
    #   da3/vggt → per_chunk   (legacy released behaviour).
    if args.global_scale_mode == 'auto':
        args.global_scale_mode = (
            'first_chunk' if args.geo_backend == 'lingbot' else 'per_chunk'
        )
        print(f"[run] --global_scale_mode=auto resolved to "
              f"'{args.global_scale_mode}' for geo_backend={args.geo_backend}")

    # ── Validate --use_depth_input ───────────────────────────────────────────
    if args.use_depth_input:
        if args.backend != 'mapanything':
            raise ValueError(
                "--use_depth_input requires --backend mapanything "
                f"(got --backend {args.backend!r})."
            )
        if args.demo_type == 'scannet_gt':
            pass   # dataset always emits depth_gt + camera_intrinsics
        elif args.demo_type == 'in_the_wild':
            if not args.depth_folder or not args.intrinsics_path:
                raise ValueError(
                    "--use_depth_input + --demo_type in_the_wild requires "
                    "BOTH --depth_folder and --intrinsics_path (MapAnything "
                    "multi-modal needs K alongside depth_z)."
                )
        else:
            raise ValueError(
                "--use_depth_input needs --demo_type scannet_gt or "
                "--demo_type in_the_wild (with --depth_folder + "
                f"--intrinsics_path); got demo_type={args.demo_type!r}."
            )
        if 'depth_gt' not in batch:
            raise ValueError(
                "Dataset did not produce a 'depth_gt' field — check the "
                "depth_folder / intrinsics_path wiring."
            )
        if 'camera_intrinsics' not in batch:
            raise ValueError(
                "Dataset did not produce 'camera_intrinsics' — MapAnything "
                "multi-modal needs K alongside depth_z."
            )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = load_model(args)

    # ── SLAM ──────────────────────────────────────────────────────────────────
    t0 = time.time()

    # Optional GT depth (metric, metres) plumbed to the pipeline.  Only the
    # pred-mode pipelines that support MapAnything multi-modal currently
    # consume this — for the rest it stays None and the pipeline ignores it.
    depth_gt   = batch.get('depth_gt') if args.use_depth_input else None
    intrins_gt = batch.get('camera_intrinsics') if args.use_depth_input else None

    # Push-time geometry conf filter: only when --conf_threshold is explicit.
    # Percentile mode keeps all points in the store (filter applied post-hoc
    # at export time, since the percentile depends on the full distribution).
    push_time_thr = args.conf_threshold if args.conf_threshold is not None else None

    # Build voxel-kv config once if requested (shared by pred + GT variants).
    voxel_kv_cfg = None
    if args.memory_mode == 'voxel_kv':
        voxel_kv_cfg = VoxelKVConfig(
            anchor_cap                       = args.vkv_anchor_cap,
            anchor_mode                      = args.vkv_anchor_mode,
            window_chunks                    = args.vkv_window_chunks,
            voxel_size                       = args.vkv_voxel_size,
            voxel_pivot_cap                  = args.vkv_pivot_cap,
            voxel_retrieval_radius           = args.vkv_retrieval_radius,
            voxel_layers                     = args.vkv_voxel_layers,
            pivot_signal                     = args.vkv_pivot_signal,
            merge_threshold                  = args.vkv_merge_threshold,
            update_p                         = args.vkv_update_p,
            update_positions_on_alignment    = args.vkv_update_positions_on_alignment,
            storage_location                 = args.vkv_storage,
        )

    # ── LingBot streaming pipeline ─────────────────────────────────────────
    # Mutually exclusive with --use_gt and voxel_kv memory; requires query head.
    if args.geo_backend == 'lingbot':
        if args.use_gt:
            raise ValueError(
                "--geo_backend lingbot does not support --use_gt "
                "(streaming pipeline is pred-only)."
            )
        if args.memory_mode != 'default':
            raise ValueError(
                "--geo_backend lingbot only supports --memory_mode default "
                f"(got {args.memory_mode}). voxel_kv is wired against the "
                "DA3/VGGT chunk geometry path."
            )
        if args.instance_head_type != 'query':
            raise ValueError(
                "--geo_backend lingbot only supports --instance_head_type query."
            )

        from slam_opense3r.pipeline_query_stream import (
            OpenSe3rStreamQuerySLAM, build_lingbot,
        )
        lingbot_model = build_lingbot(
            args.lingbot_ckpt,
            image_size              = args.lingbot_image_size,
            patch_size              = args.lingbot_patch_size,
            num_scale_frames        = args.lingbot_num_scale_frames,
            kv_cache_sliding_window = args.lingbot_kv_window,
            enable_3d_rope          = args.lingbot_enable_3d_rope,
            max_frame_num           = args.lingbot_max_frame_num,
            use_sdpa                = args.lingbot_use_sdpa,
            device                  = 'cuda',
        )
        inst_kwargs = dict(
            instance_voxel_size  = args.inst_voxel_size,
            knn_radius           = args.inst_knn_radius,
            max_candidates       = args.inst_max_candidates,
            match_threshold      = args.inst_match_threshold,
            w_desc               = args.inst_w_desc,
            w_sem                = args.inst_w_sem,
            w_vox                = args.inst_w_vox,
            w_geo                = args.inst_w_geo,
            sigma_dist           = args.inst_sigma_dist,
            obj_threshold        = args.inst_obj_threshold,
            obj_top_k            = args.inst_obj_top_k,
            min_points           = args.inst_min_points,
            nms_iou              = args.inst_nms_iou,
            score_threshold      = args.inst_score_threshold,
            outlier_iqr_mult     = args.inst_iqr_mult,
            vox_gate             = args.inst_vox_gate,
            desc_gate            = args.inst_desc_gate,
            score_combine        = args.inst_score_combine,
            box_iou_mode         = args.inst_box_iou_mode,
            sem_gate_thr         = args.inst_sem_gate_thr,
            vox_bonus_alpha      = args.inst_vox_bonus_alpha,
            desc_pow             = args.inst_desc_pow,
            vox_or_tau           = args.inst_vox_or_tau,
        )
        # Streaming pipeline scale: 'gt_seq_median' is invalid here (pred-only),
        # so fall back to the recommended 'first_chunk' if the user left the
        # global default (gt_seq_median is never the default, but guard anyway).
        stream_scale_mode = args.global_scale_mode
        if stream_scale_mode == 'gt_seq_median':
            print("[run] --global_scale_mode=gt_seq_median invalid for "
                  "streaming (no GT) — using 'first_chunk'.")
            stream_scale_mode = 'first_chunk'

        pipeline = OpenSe3rStreamQuerySLAM(
            lingbot_model        = lingbot_model,
            opense3r_model       = model,
            cfg_path             = args.cfg_path,
            sem_feat_mode        = args.sem_feat_mode,
            feat_storage         = args.feat_storage,
            num_scale_frames     = args.lingbot_num_scale_frames,
            chunk_size           = args.lingbot_chunk_size,
            geo_mode             = args.lingbot_geo_mode,
            attn_mode            = args.attn_mode,
            global_scale_mode    = stream_scale_mode,
            point_conf_threshold = push_time_thr,
            **inst_kwargs,
        )
        pipeline.embed_update   = args.inst_embed_update
        pipeline.embed_momentum = args.inst_embed_momentum
        pipeline.match_metric   = args.inst_match_metric
        pipeline.match_mode      = args.inst_match_mode
        pipeline.gate_overlap    = args.inst_gate_overlap
        pipeline.greedy_gate_use_observed = args.inst_greedy_gate_observed
        pipeline.merge_desc_high = args.inst_merge_desc_high
        pipeline.spawn_desc_low  = args.inst_spawn_desc_low
        pipeline.nms_mode          = args.inst_nms_mode
        pipeline.matrix_nms_kernel = args.inst_matrix_nms_kernel
        pipeline.panoptic          = (not args.inst_no_panoptic)
        pipeline.sp_score_thr      = args.inst_sp_score_thr
        pipeline.obj_normalize     = args.inst_obj_normalize
        pipeline.use_sem_conf      = args.inst_use_sem_conf
        pipeline.use_sem_conf_gate = args.inst_sem_conf_gate
        pipeline.use_sem_conf_select = args.inst_sem_conf_select
        memory = pipeline.run(images)

    elif args.use_gt:
        assert 'pts_gt' in batch, \
            "--use_gt requires demo_type=scannet_gt (GT depth must be available)"
        pts_gt   = batch['pts_gt']               # [T, H, W, 3]
        poses_gt = batch['camera_pose'].float()  # [T, 4, 4]

        if args.instance_head_type == 'query':
            # GT + query head.  Two memory modes (default flat KV / voxel_kv).
            inst_kwargs = dict(
                instance_voxel_size  = args.inst_voxel_size,
                knn_radius           = args.inst_knn_radius,
                max_candidates       = args.inst_max_candidates,
                match_threshold      = args.inst_match_threshold,
                w_desc               = args.inst_w_desc,
                w_sem                = args.inst_w_sem,
                w_vox                = args.inst_w_vox,
                w_geo                = args.inst_w_geo,
                sigma_dist           = args.inst_sigma_dist,
                obj_threshold        = args.inst_obj_threshold,
                obj_top_k            = args.inst_obj_top_k,
                min_points           = args.inst_min_points,
                nms_iou              = args.inst_nms_iou,
                score_threshold      = args.inst_score_threshold,
                outlier_iqr_mult     = args.inst_iqr_mult,
                vox_gate             = args.inst_vox_gate,
                desc_gate            = args.inst_desc_gate,
                score_combine        = args.inst_score_combine,
                box_iou_mode         = args.inst_box_iou_mode,
                sem_gate_thr         = args.inst_sem_gate_thr,
                vox_bonus_alpha      = args.inst_vox_bonus_alpha,
                desc_pow             = args.inst_desc_pow,
                vox_or_tau           = args.inst_vox_or_tau,
            )
            if args.memory_mode == 'voxel_kv':
                from slam_opense3r.pipeline_query_gt import OpenSe3rQuerySLAM_VoxelKV_GT
                pipeline = OpenSe3rQuerySLAM_VoxelKV_GT(
                    model                = model,
                    cfg_path             = args.cfg_path,
                    sem_feat_mode        = args.sem_feat_mode,
                    feat_storage         = args.feat_storage,
                    kv_window_size       = None,
                    kv_keyframe_stride   = None,
                    backend              = args.backend,
                    point_conf_threshold = push_time_thr,
                    attn_mode            = args.attn_mode,
                    global_scale_mode    = args.global_scale_mode,
                    gt_geo_input         = args.gt_geo_input,
                    voxel_kv_cfg         = voxel_kv_cfg,
                    **inst_kwargs,
                )
            else:
                from slam_opense3r.pipeline_query_gt import OpenSe3rQuerySLAM_GT
                pipeline = OpenSe3rQuerySLAM_GT(
                    model                = model,
                    cfg_path             = args.cfg_path,
                    sem_feat_mode        = args.sem_feat_mode,
                    feat_storage         = args.feat_storage,
                    kv_window_size       = args.kv_window_size,
                    kv_keyframe_stride   = args.kv_keyframe_stride,
                    backend              = args.backend,
                    point_conf_threshold = push_time_thr,
                    attn_mode            = args.attn_mode,
                    global_scale_mode    = args.global_scale_mode,
                    no_cross_chunk_kv    = args.no_cross_chunk_kv,
                    gt_geo_input         = args.gt_geo_input,
                    **inst_kwargs,
                )
        elif args.memory_mode == 'voxel_kv':
            pipeline = OpenSe3rSLAM_VoxelKV_GT(
                model                = model,
                cfg_path             = args.cfg_path,
                sem_feat_mode        = args.sem_feat_mode,
                feat_storage         = args.feat_storage,
                kv_window_size       = None,
                kv_keyframe_stride   = None,
                backend              = args.backend,
                point_conf_threshold = push_time_thr,
                attn_mode            = args.attn_mode,
                voxel_kv_cfg         = voxel_kv_cfg,
            )
        else:
            pipeline = OpenSe3rSLAM_GT(
                model                = model,
                cfg_path             = args.cfg_path,
                sem_feat_mode        = args.sem_feat_mode,
                feat_storage         = args.feat_storage,
                kv_window_size       = args.kv_window_size,
                kv_keyframe_stride   = args.kv_keyframe_stride,
                backend              = args.backend,
                point_conf_threshold = push_time_thr,
                attn_mode            = args.attn_mode,
            )
        run_kwargs = {}
        if args.instance_head_type == 'query' and args.gt_geo_input:
            run_kwargs = dict(depth_gt    = batch['depth_gt'],
                              intrinsics_gt = batch['camera_intrinsics'])
        pipeline.embed_update   = args.inst_embed_update
        pipeline.embed_momentum = args.inst_embed_momentum
        pipeline.match_metric   = args.inst_match_metric
        pipeline.match_mode      = args.inst_match_mode
        pipeline.gate_overlap    = args.inst_gate_overlap
        pipeline.greedy_gate_use_observed = args.inst_greedy_gate_observed
        pipeline.merge_desc_high = args.inst_merge_desc_high
        pipeline.spawn_desc_low  = args.inst_spawn_desc_low
        pipeline.nms_mode          = args.inst_nms_mode
        pipeline.matrix_nms_kernel = args.inst_matrix_nms_kernel
        pipeline.panoptic          = (not args.inst_no_panoptic)
        pipeline.sp_score_thr      = args.inst_sp_score_thr
        pipeline.obj_normalize     = args.inst_obj_normalize
        pipeline.use_sem_conf      = args.inst_use_sem_conf
        pipeline.use_sem_conf_gate = args.inst_sem_conf_gate
        pipeline.use_sem_conf_select = args.inst_sem_conf_select
        _set_2d_mask_dump(pipeline, args)
        memory = pipeline.run(images, pts_gt, poses_gt, **run_kwargs)
    elif args.instance_head_type == 'query':
        # ── Query-head SLAM path ────────────────────────────────────────────
        # InstanceTable + voxel-vote memory.  Dense ins_store is not built.
        # Supports both default (flat KV cache) and voxel_kv memory modes
        # via separate subclasses.
        inst_kwargs = dict(
            instance_voxel_size  = args.inst_voxel_size,
            knn_radius           = args.inst_knn_radius,
            max_candidates       = args.inst_max_candidates,
            match_threshold      = args.inst_match_threshold,
            w_desc               = args.inst_w_desc,
            w_sem                = args.inst_w_sem,
            w_vox                = args.inst_w_vox,
            w_geo                = args.inst_w_geo,
            sigma_dist           = args.inst_sigma_dist,
            obj_threshold        = args.inst_obj_threshold,
            obj_top_k            = args.inst_obj_top_k,
            min_points           = args.inst_min_points,
            nms_iou              = args.inst_nms_iou,
            score_threshold      = args.inst_score_threshold,
            outlier_iqr_mult     = args.inst_iqr_mult,
            vox_gate             = args.inst_vox_gate,
            desc_gate            = args.inst_desc_gate,
            score_combine        = args.inst_score_combine,
            box_iou_mode         = args.inst_box_iou_mode,
            sem_gate_thr         = args.inst_sem_gate_thr,
            vox_bonus_alpha      = args.inst_vox_bonus_alpha,
            desc_pow             = args.inst_desc_pow,
            vox_or_tau           = args.inst_vox_or_tau,
        )
        if args.memory_mode == 'voxel_kv':
            from slam_opense3r.pipeline_query_voxel_kv import OpenSe3rQuerySLAM_VoxelKV
            pipeline = OpenSe3rQuerySLAM_VoxelKV(
                model                = model,
                cfg_path             = args.cfg_path,
                sem_feat_mode        = args.sem_feat_mode,
                feat_storage         = args.feat_storage,
                kv_window_size       = None,
                kv_keyframe_stride   = None,
                backend              = args.backend,
                point_conf_threshold = push_time_thr,
                attn_mode            = args.attn_mode,
                global_scale_mode    = args.global_scale_mode,
                voxel_kv_cfg         = voxel_kv_cfg,
                **inst_kwargs,
            )
        else:
            from slam_opense3r.pipeline_query import OpenSe3rQuerySLAM
            pipeline = OpenSe3rQuerySLAM(
                model                = model,
                cfg_path             = args.cfg_path,
                sem_feat_mode        = args.sem_feat_mode,
                feat_storage         = args.feat_storage,
                kv_window_size       = args.kv_window_size,
                kv_keyframe_stride   = args.kv_keyframe_stride,
                backend              = args.backend,
                point_conf_threshold = push_time_thr,
                attn_mode            = args.attn_mode,
                global_scale_mode    = args.global_scale_mode,
                **inst_kwargs,
            )
        pipeline.embed_update   = args.inst_embed_update
        pipeline.embed_momentum = args.inst_embed_momentum
        pipeline.match_metric   = args.inst_match_metric
        pipeline.match_mode      = args.inst_match_mode
        pipeline.gate_overlap    = args.inst_gate_overlap
        pipeline.greedy_gate_use_observed = args.inst_greedy_gate_observed
        pipeline.merge_desc_high = args.inst_merge_desc_high
        pipeline.spawn_desc_low  = args.inst_spawn_desc_low
        pipeline.nms_mode          = args.inst_nms_mode
        pipeline.matrix_nms_kernel = args.inst_matrix_nms_kernel
        pipeline.panoptic          = (not args.inst_no_panoptic)
        pipeline.sp_score_thr      = args.inst_sp_score_thr
        pipeline.obj_normalize     = args.inst_obj_normalize
        pipeline.use_sem_conf      = args.inst_use_sem_conf
        pipeline.use_sem_conf_gate = args.inst_sem_conf_gate
        pipeline.use_sem_conf_select = args.inst_sem_conf_select
        _set_2d_mask_dump(pipeline, args)
        memory = pipeline.run(
            images,
            depth_z_input    = depth_gt,
            intrinsics_input = intrins_gt,
        )
    elif args.memory_mode == 'voxel_kv':
        pipeline = OpenSe3rSLAM_VoxelKV(
            model                = model,
            cfg_path             = args.cfg_path,
            sem_feat_mode        = args.sem_feat_mode,
            feat_storage         = args.feat_storage,
            kv_window_size       = None,        # voxel_kv ignores flat window
            kv_keyframe_stride   = None,        # and flat keyframe stride
            backend              = args.backend,
            point_conf_threshold = push_time_thr,
            attn_mode            = args.attn_mode,
            voxel_kv_cfg         = voxel_kv_cfg,
        )
        memory = pipeline.run(images)
    else:
        pipeline = OpenSe3rSLAM(
            model                = model,
            cfg_path             = args.cfg_path,
            sem_feat_mode        = args.sem_feat_mode,
            feat_storage         = args.feat_storage,
            kv_window_size       = args.kv_window_size,
            kv_keyframe_stride   = args.kv_keyframe_stride,
            backend              = args.backend,
            point_conf_threshold = push_time_thr,
            attn_mode            = args.attn_mode,
        )
        memory = pipeline.run(images)

    total_time = time.time() - t0
    fps        = T / total_time
    print(f"\n[run] Done — {fps:.1f} fps  ({total_time:.1f}s total)")
    print(f"  Semantic store : {memory.sem_count:,}")
    if args.instance_head_type == 'query':
        summ = memory.instance_summary()
        print(f"  Instance table : {summ['num_instances']} instances "
              f"({summ['total_voxels']:,} voxels, "
              f"{summ['total_merges']} merges)")
    else:
        print(f"  Instance store : {memory.ins_count:,}")

    # ── Optional: align pred map to GT first-frame world ─────────────────────
    # Pred-mode SLAM produces a map in "first-input-frame world" (i.e. the
    # output frame coincides with the first input camera's pose ≈ identity).
    # If the dataset's GT poses are NOT identity-anchored (e.g. ScanNet),
    # downstream eval is easier when the pred map is rotated/translated into
    # the GT world frame.  Apply ``T_align = GT_first_frame_c2w`` everywhere:
    #   • memory.pts   *=  T_align        (left-multiply per point)
    #   • memory.poses *=  T_align        (left-multiply each c2w)
    #   • voxel store stays in pred frame; we pass T_align to the per-voxel
    #     and per-point exporters so they apply the same transform on save.
    T_align = None
    if (args.align_to_gt_first_frame
            and not args.use_gt
            and 'camera_pose' in batch):
        T_align = batch['camera_pose'][0].cpu().float().numpy().astype(np.float32)
        print(f"[run] aligning pred map to GT first-frame c2w …")
        print(f"      T_align[:3, 3] = {T_align[:3, 3]}")

        # Transform memory.pts: pts_world = R @ pts + t
        pts_t = memory.pts                        # (T, H, W, 3)  torch
        T_t   = torch.from_numpy(T_align)
        pts_flat = pts_t.reshape(-1, 3)
        pts_flat = (T_t[:3, :3] @ pts_flat.T).T + T_t[:3, 3]
        memory.pts = pts_flat.reshape(pts_t.shape)

        # Transform memory.poses: c2w_global = T_align @ c2w_local
        memory.poses = T_t @ memory.poses
    elif args.align_to_gt_first_frame:
        # Quietly skip with a one-line note rather than fail
        if args.use_gt:
            print("[run] --align_to_gt_first_frame ignored (GT mode already in GT frame)")
        elif 'camera_pose' not in batch:
            print("[run] --align_to_gt_first_frame ignored (dataset has no camera_pose)")

    # Default-None so downstream blocks can guard `pts_np_full is None`
    # when --save_geo is False (per-point NPZ dump etc).
    pts_np = None
    pts_np_full = None

    # ── Geometry PLY ──────────────────────────────────────────────────────────
    if args.save_geo:
        imgs_for_color = images[0]   # [T, 3, H, W]
        pts_all  = memory.pts        # [T, H, W, 3]
        conf_all = memory.conf       # [T, H, W]

        # Resolve confidence threshold: BOTH the explicit value (when given)
        # and the percentile-derived value are computed; the final cut is
        # ``max(explicit, percentile)`` so the more restrictive one always
        # wins.  This means ``--conf_threshold`` is treated as a hard floor;
        # ``--conf_threshold_percentile`` continues to drop the bottom-N%
        # tail on top of that.  Set ``--conf_threshold_percentile 0`` to
        # disable the percentile component.
        thr_explicit = (
            float(args.conf_threshold) if args.conf_threshold is not None else None
        )

        thr_pct = None
        valid_conf = conf_all[conf_all > 0].float()
        if valid_conf.numel() == 0:
            print(f"  conf_threshold: no valid conf pixels — using 0.0")
        elif args.conf_threshold_percentile > 0:
            # torch.quantile caps at ~16M elements; subsample if exceeded
            # (16M is a very fine-grained sample of the distribution —
            # threshold drift vs full set is well below 0.001).
            MAX_QUANTILE_N = 15_000_000
            n_total = valid_conf.numel()
            if n_total > MAX_QUANTILE_N:
                g    = torch.Generator(device=valid_conf.device).manual_seed(0)
                pick = torch.randint(0, n_total, (MAX_QUANTILE_N,),
                                     generator=g, device=valid_conf.device)
                sample = valid_conf[pick]
            else:
                sample = valid_conf
            q = max(0.0, min(100.0, args.conf_threshold_percentile)) / 100.0
            thr_pct = float(torch.quantile(sample, q).item())
            print(f"  conf_threshold (p{args.conf_threshold_percentile:g}) "
                  f"= {thr_pct:.4f}  "
                  f"(valid conf range "
                  f"[{valid_conf.min().item():.4f}, "
                  f"{valid_conf.max().item():.4f}], "
                  f"n={n_total:,}"
                  f"{f', sampled {MAX_QUANTILE_N:,}' if n_total > MAX_QUANTILE_N else ''})")
        if thr_explicit is not None:
            print(f"  conf_threshold (explicit) = {thr_explicit:.4f}")

        # Combine: take the stricter of the two if both are present.
        candidates = [t for t in (thr_explicit, thr_pct) if t is not None]
        conf_thr = max(candidates) if candidates else 0.0
        if len(candidates) > 1:
            print(f"  conf_threshold (combined max) = {conf_thr:.4f}")

        if args.edge_mask:
            pts_mask, _ = get_pts_mask(
                pts_all, imgs_for_color, conf_all,
                conf_threshold=conf_thr,
            )
        else:
            # `>=` (not `>`) so a saturated conf distribution (e.g. DA3 high-
            # conf pixels are clamped to 1.0 — entire valid set ends up at 1.0,
            # making p_any threshold equal to max) doesn't filter every point.
            # Invalid-depth pixels (conf == 0, pts == origin) are excluded
            # below by the pts.norm check.
            pts_mask = conf_all >= conf_thr
        pts_mask = pts_mask & (pts_all.norm(dim=-1) > 1e-3)

        pts_np   = pts_all.numpy().reshape(-1, 3)
        mask_np  = np.asarray(pts_mask).reshape(-1)
        color_np = (imgs_for_color.numpy().transpose(0, 2, 3, 1).reshape(-1, 3) + 1.0) / 2.0
        pts_np   = pts_np [mask_np]
        color_np = color_np[mask_np]

        # Keep the FULL filtered point cloud (pre-downsample) for the post-hoc
        # voxel-store prune below.  Random downsampling that follows is only
        # for the geometry PLY's max_points budget, and would shrink the
        # surviving voxel set spuriously if used as the prune source.
        pts_np_full = pts_np.copy()

        if args.max_points > 0 and len(pts_np) > args.max_points:
            print(f"  Downsampling: {len(pts_np):,} → {args.max_points:,} (random)")
            # `np.random.choice(N, K, replace=False)` allocates an O(N)
            # int64 permutation (~544 MB at N=68M).  Under heavy CPU
            # memory pressure (e.g. 17 GB sem_store + pycolmap C++
            # statics from DA3's eagerly-imported colmap exporter) this
            # has been observed to SIGSEGV at process cleanup time.
            # Use argpartition on random keys: still O(N) memory but
            # only one float32 array (~272 MB), no full permutation.
            n = len(pts_np)
            keys = np.random.random(n).astype(np.float32, copy=False)
            idx  = np.argpartition(keys, args.max_points - 1)[:args.max_points]
            del keys
            idx.sort()
            pts_np   = pts_np[idx]
            color_np = color_np[idx]

        geo_ply = os.path.join(args.results_path, f"{args.demo_name}_geo.ply")
        # numpy writer (NOT open3d): immune to the o3d-0.18 / numpy-2.x ABI
        # SIGSEGV that was crashing this export on large clouds.
        _write_ply_xyzrgb(geo_ply, pts_np, color_np)
        print(f"Saved geometry PLY         → {geo_ply}")

        # ── Post-hoc voxel-store prune ───────────────────────────────────────
        # Drop any voxel that has NO surviving point in the filtered cloud.
        # Has to run BEFORE the semantic / instance export blocks, because
        # those blocks read from `memory.sem_store` and `memory.ins_store`.
        #
        # ⚠ Frame-of-reference detail:
        #   * voxel store keys were quantised when SLAM pushed pts in the
        #     LOCAL/pred frame (memory.pts BEFORE the align transform);
        #   * pts_np_full is taken from memory.pts AFTER the align transform,
        #     so when --align_to_gt_first_frame was applied it lives in
        #     GT-frame.
        # Quantising GT-frame pts would produce keys that mismatch the stored
        # ones → prune_to_keys drops EVERY voxel → all per-point queries
        # later return zero features → semantic PLY collapses to a single
        # class.  Inverse-transform live_pts back to LOCAL frame first.
        if pts_np_full.shape[0] > 0:
            if T_align is not None:
                T_inv = np.linalg.inv(T_align.astype(np.float64))
                ones  = np.ones((pts_np_full.shape[0], 1), dtype=np.float64)
                pts_h = np.concatenate([pts_np_full.astype(np.float64), ones], axis=1)
                pts_local_for_prune = (T_inv @ pts_h.T).T[:, :3].astype(np.float32)
                live_pts = torch.from_numpy(pts_local_for_prune)
            else:
                live_pts = torch.from_numpy(pts_np_full)   # (N_full, 3)

            for store_name, store in (('sem', memory.sem_store),
                                      ('ins', memory.ins_store)):
                if store is None or not isinstance(store, VoxelFeatureMap):
                    continue
                before = store.num_voxels
                live_keys = store.compute_keys(live_pts)
                dropped   = store.prune_to_keys(live_keys)
                after     = store.num_voxels
                print(f"  [{store_name}] voxel prune: "
                      f"{before:,} → {after:,}  (dropped {dropped:,})")

    # ── Semantic export ───────────────────────────────────────────────────────
    if args.save_semantic and memory.sem_store is not None:
        labels, _, color_table = get_scannet_label_and_color_map(args.label_set)
        device    = next(model.parameters()).device
        # Uses the 10-template indoor ensemble (DEFAULT_TEXT_TEMPLATES) —
        # mean-pool + L2-norm.  Significantly more robust on ambiguous classes
        # (picture/wall, door/cabinet) than the old single "a photo of a {}".
        #
        # Use the backbone's own text encoder so the text space matches the
        # space the dense semantic features were trained to land in:
        #   - lseg   → CLIP-B/32  (lives at model.lseg.clip_pretrained)
        #   - tipsv2 → SigLIP-style text head bundled with TIPSv2
        #   - clip-* → that CLIP variant's text encoder
        # `build_text_embeddings_via_backbone` calls `backbone.encode_text(...)`
        # which each backbone implements with the right tokenizer.
        if args.semantic_backbone == 'lseg':
            text_feat = build_text_embeddings(
                clip_model = model.lseg.clip_pretrained,
                tokenizer  = clip_lib.tokenize,
                labels     = labels,
                device     = device,
            ).cpu()
        else:
            text_feat = build_text_embeddings_via_backbone(
                backbone = model.semantic_backbone,
                labels   = labels,
                device   = device,
            ).cpu()

        if isinstance(memory.sem_store, VoxelFeatureMap):
            vox_ply = os.path.join(args.results_path, f"{args.demo_name}_semantic_voxels.ply")
            export_semantic_voxels_ply(
                memory.sem_store, text_feat, color_table, vox_ply,
                transformation=T_align,
            )
            print(f"Saved semantic voxel PLY   → {vox_ply}")

            # Per-point query (map voxel features to all geometry points)
            if args.save_geo and pts_np is not None:
                pp_ply = os.path.join(args.results_path,
                                      f"{args.demo_name}_semantic_per_point.ply")
                export_semantic_per_point_ply(
                    torch.from_numpy(pts_np), memory.sem_store,
                    text_feat, color_table, pp_ply, transformation=T_align,
                )
                print(f"Saved semantic per-pt PLY  → {pp_ply}")
        else:
            # per-point store
            pp_ply = os.path.join(args.results_path,
                                  f"{args.demo_name}_semantic_per_point.ply")
            _export_perpoint_semantic(memory.sem_store, text_feat, color_table,
                                      pp_ply, transformation=T_align)
            print(f"Saved semantic per-pt PLY  → {pp_ply}")

        legend = os.path.join(args.results_path, f"{args.demo_name}_semantic_legend.png")
        save_semantic_color_legend(labels, color_table, legend)
        print(f"Saved semantic legend      → {legend}")

        if args.save_npz:
            if isinstance(memory.sem_store, VoxelFeatureMap):
                centers, feats = memory.sem_store.get_all()
                centers_np = centers.numpy().astype(np.float32)
                if T_align is not None:
                    centers_h  = np.concatenate([centers_np,
                                                  np.ones((len(centers_np), 1), np.float32)], 1)
                    centers_np = (T_align @ centers_h.T).T[:, :3]
            else:
                centers_np, feats = memory.sem_store.get_all()
                centers_np = centers_np.numpy().astype(np.float32)
                feats      = feats
            npz_path = os.path.join(args.results_path, f"{args.demo_name}_semantic_feats.npz")
            np.savez_compressed(npz_path,
                centers  = centers_np,
                features = feats.numpy().astype(np.float32),
            )
            print(f"Saved semantic feats npz   → {npz_path}")

    # ── Instance export ───────────────────────────────────────────────────────
    if args.save_instance and args.instance_head_type == 'query':
        # Query-head export: voxel + per-point + wireframe-bbox PLY (all
        # from InstanceTable) plus a JSON dump of per-instance metadata.
        # Replaces the dense PCA / HDBSCAN exports entirely for this path.
        #
        # All three PLYs share ONE color_map computed once from the table:
        #   - mode='desc' (default): 1D PCA on descriptors → golden-angle
        #                            hue.  Visually similar objects get
        #                            maximally distinct colours.
        #   - mode='rank'          : sort by inst_id, golden-angle hue
        #                            (legacy behaviour).
        from amb3r.vis_query_instance_slam import (
            export_query_instance_voxels_ply,
            export_query_instance_per_point_ply,
            export_query_instance_boxes_ply,
            dump_instance_table_json,
            dump_instance_labels_npz,
            dump_instance_labels_per_point_npz,
            dump_instance_sem_queries_npz,
            color_map_for_table,
            filter_noise_instances,
        )
        T_align_t = torch.from_numpy(T_align) if T_align is not None else None
        color_map = color_map_for_table(memory.instance_table,
                                          mode=args.inst_color_mode)
        # Compute the noise-filter dropped-set ONCE and reuse across exports.
        dropped_ids = filter_noise_instances(
            memory.instance_table,
            total_frames        = int(memory.num_frames),
            min_count           = args.inst_filter_min_count,
            end_grace           = args.inst_filter_end_grace,
            max_size_m3         = args.inst_filter_max_size_m3,
            oversized_max_count = args.inst_filter_oversized_max_count,
            tiny_voxels         = args.inst_filter_tiny_voxels,
            tiny_max_count      = args.inst_filter_tiny_max_count,
            min_avg_objectness  = args.inst_filter_min_avg_objectness,
            max_extent_m        = args.inst_filter_max_extent_m,
            elongated_max_count = args.inst_filter_elongated_max_count,
        )
        print(f"[run] instance color_map: {len(color_map)} instances, "
              f"mode={args.inst_color_mode}  "
              f"filter dropped: {len(dropped_ids)}")

        vox_ply = os.path.join(args.results_path,
                                f"{args.demo_name}_query_instance_voxels.ply")
        export_query_instance_voxels_ply(
            memory, vox_ply, transformation=T_align_t,
            color_map=color_map, color_mode=args.inst_color_mode,
            dropped_ids=dropped_ids, verbose=True,
        )
        print(f"Saved query instance voxel PLY  → {vox_ply}")

        box_ply = os.path.join(args.results_path,
                                f"{args.demo_name}_query_instance_boxes.ply")
        export_query_instance_boxes_ply(
            memory, box_ply, transformation=T_align_t,
            color_map=color_map, color_mode=args.inst_color_mode,
            dropped_ids=dropped_ids, verbose=True,
        )
        print(f"Saved query instance box PLY    → {box_ply}")

        if args.save_geo:
            pp_ply = os.path.join(args.results_path,
                                    f"{args.demo_name}_query_instance_per_point.ply")
            export_query_instance_per_point_ply(
                memory, pp_ply, transformation=T_align_t,
                color_map=color_map, color_mode=args.inst_color_mode,
                dropped_ids=dropped_ids,
                point_conf_min=args.conf_threshold, verbose=True,
            )
            print(f"Saved query instance per-pt PLY → {pp_ply}")

        meta_path = os.path.join(args.results_path,
                                    f"{args.demo_name}_query_instance_table.json")
        dump_instance_table_json(
            memory, meta_path, transformation=T_align_t,
            dropped_ids=dropped_ids, verbose=True,
        )
        print(f"Saved instance table metadata    → {meta_path}")

        # NPZ in the same schema as instance_hdbscan_cluster.py's output —
        # makes eval_instance_npz_scannet.py work directly on query SLAM.
        if args.save_npz:
            npz_path = os.path.join(args.results_path,
                                      f"{args.demo_name}_instance_labels.npz")
            dump_instance_labels_npz(
                memory, npz_path, transformation=T_align_t,
                dropped_ids=dropped_ids, verbose=True,
                objectness_agg=getattr(args, 'inst_cluster_score_agg', 'mean'),
            )
            print(f"Saved instance labels NPZ        → {npz_path}")

            # Per-point version (centers = filtered geo pts in FINAL frame,
            # cluster_id looked up per-pt via voxel).  Closes the voxel-
            # quantisation IoU boundary loss vs per-mesh-vert dumps (ESAM).
            if args.save_npz_per_point and pts_np is not None:
                pp_npz_path = os.path.join(
                    args.results_path,
                    f"{args.demo_name}_instance_labels_per_point.npz",
                )
                dump_instance_labels_per_point_npz(
                    memory, pp_npz_path,
                    pts_world=pts_np_full if pts_np_full is not None else pts_np,
                    transformation=T_align_t,
                    dropped_ids=dropped_ids, verbose=True,
                )
                print(f"Saved instance labels NPZ (pp)   → {pp_npz_path}")

            # Per-instance aggregated sem_query (mask-region mean of dense
            # semantic features).  Used by `query_instance_semantic.py` for
            # offline broadcast → text-align comparison vs the dense path.
            sq_path = os.path.join(args.results_path,
                                      f"{args.demo_name}_instance_sem_queries.npz")
            dump_instance_sem_queries_npz(
                memory, sq_path,
                dropped_ids=dropped_ids, verbose=True,
            )
            print(f"Saved instance sem_query NPZ     → {sq_path}")
    elif args.save_instance and memory.ins_store is not None:
        if isinstance(memory.ins_store, VoxelFeatureMap):
            vox_ply = os.path.join(args.results_path, f"{args.demo_name}_instance_voxels.ply")
            export_instance_voxels_ply(memory.ins_store, vox_ply, transformation=T_align)
            print(f"Saved instance voxel PLY   → {vox_ply}")

            if args.save_geo and pts_np is not None:
                pp_ply = os.path.join(args.results_path,
                                      f"{args.demo_name}_instance_per_point.ply")
                export_instance_per_point_ply(
                    torch.from_numpy(pts_np), memory.ins_store, pp_ply,
                    transformation=T_align,
                )
                print(f"Saved instance per-pt PLY  → {pp_ply}")
        else:
            pp_ply = os.path.join(args.results_path,
                                  f"{args.demo_name}_instance_per_point.ply")
            _export_perpoint_instance_pca(memory.ins_store, pp_ply,
                                           transformation=T_align)
            print(f"Saved instance per-pt PLY  → {pp_ply}")

        if args.save_npz:
            if isinstance(memory.ins_store, VoxelFeatureMap):
                centers, feats = memory.ins_store.get_all()
                centers_np = centers.numpy().astype(np.float32)
            else:
                centers_np_t, feats = memory.ins_store.get_all()
                centers_np = centers_np_t.numpy().astype(np.float32)
            npz_path = os.path.join(args.results_path, f"{args.demo_name}_instance_feats.npz")
            np.savez_compressed(npz_path,
                centers  = centers_np,
                features = feats.numpy().astype(np.float32),
            )
            print(f"Saved instance feats npz   → {npz_path}")

    # ── Results npz ───────────────────────────────────────────────────────────
    # ``conf_threshold`` is the resolved value SLAM used for export-time
    # filtering (max of explicit + percentile).  Downstream clustering picks
    # this up as its default so the per-point cloud it reconstructs matches
    # the geometry PLY's filter.
    #
    # `memory.{pts,conf,poses}` are pre-allocated to a chunk-padded buffer
    # size.  If SLAM finishes before filling the whole buffer (short clip,
    # early termination, dataloader returning fewer frames than expected),
    # trailing entries are uninitialised zeros and would crash downstream
    # consumers (e.g. scipy.from_matrix on an all-zero rotation).  Detect the
    # written-prefix length via the homogeneous row of `poses` ([3,3] == 1 for
    # any valid c2w, == 0 for zero-init), then slice every array to N before
    # save so the npz reflects exactly the frames SLAM actually processed.
    poses_np = memory.poses.numpy()
    valid_mask = poses_np[:, 3, 3] != 0
    if not valid_mask.any():
        raise RuntimeError("No valid frames in memory.poses — SLAM produced "
                           "nothing to save.")
    N_valid = int(valid_mask.sum())
    # Defensive: detect non-contiguous gaps (shouldn't happen — SLAM writes
    # frame-by-frame — but if it does, surface it instead of silent slicing).
    if not np.all(valid_mask[:N_valid]):
        n_total = int(valid_mask.size)
        n_gap   = int(N_valid - valid_mask[:N_valid].sum())
        print(f"[save_npz] WARN: valid frames are non-contiguous "
              f"({n_gap} zero rows inside first {N_valid} frames, "
              f"total buffer={n_total}). Saving the full buffer un-sliced "
              f"so downstream can decide what to do.")
        N_valid = n_total
    else:
        n_total = int(valid_mask.size)
        if N_valid < n_total:
            print(f"[save_npz] truncating buffer {n_total} → {N_valid} "
                  f"actually-processed frames before save.")

    npz_path = os.path.join(args.results_path, f"{args.demo_name}_results.npz")
    saved_conf_thr = float(conf_thr) if 'conf_thr' in locals() else 0.0
    np.savez_compressed(
        npz_path,
        pts            = memory.pts.numpy()[:N_valid],
        conf           = memory.conf.numpy()[:N_valid],
        poses          = poses_np[:N_valid],
        images         = images[0].numpy()[:N_valid],
        fps            = fps,
        conf_threshold = np.float32(saved_conf_thr),
    )
    print(f"Saved results npz          → {npz_path}  "
          f"(conf_threshold={saved_conf_thr:.4f}, N={N_valid})")


if __name__ == '__main__':
    main()
