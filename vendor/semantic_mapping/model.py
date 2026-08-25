"""Checkpoint-driven model construction.

The architecture is read from the checkpoint's own `args` (41 of the 45 fields
`build_model` needs are stored there), so the caller only supplies a path.  This
is deliberate: every architecture flag that used to be hand-copied into a bash
script is now self-described by the weights.  Two checkpoints were silently
corrupted in Aug 2026 by exactly that hand-copying (see README §1).

Only the GT-geometry inference path is supported here.  With GT depth the DA3
geometry backbone is never executed, so it is not even built (~15 GB saved).
"""

from __future__ import annotations

import os
from argparse import Namespace
from typing import Dict, Optional

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Submodules that carry trained weights.  Anything else missing from the
# state_dict is a frozen backbone loaded inside __init__ (expected).
# NOTE `pq_head.` is top-level (NOT under instance_head) and was missing from
# the legacy guard in slam_opense3r/run.py — a dropped P/Q head would have been
# counted as a "frozen backbone" difference and passed silently.
TRAINABLE_PREFIXES = ('depth_encoder.', 'causal_branch.', 'semantic_head.',
                      'instance_head.', 'pq_head.', 'pos3d_encoder.',
                      'pose_encoder.', 'plucker_encoder.', 'region_pool.')

# Training-only head modules that the SLAM/mapping pipeline never builds or
# runs.  When a NEWER checkpoint carries them but the model here was built
# without them, their keys surface as "unexpected" under TRAINABLE_PREFIXES —
# drop them silently (they are not a real architecture mismatch).
#   - instance_head.sem_proj.*: the instance-aware semantic RESIDUAL head; its
#     per-instance semantics are a train-time supervision target, not used by
#     the streaming mapper (which reads mask_logits / descriptor / objectness).
OPTIONAL_DROP_PREFIXES = ('instance_head.sem_proj.',)


def read_ckpt_args(ckpt_path: str) -> Namespace:
    """Return the training-time args stored inside the checkpoint."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False, mmap=True)
    if 'args' not in state:
        raise KeyError(
            f'{ckpt_path} has no "args" — architecture cannot be self-described. '
            'Pass a checkpoint saved by train_opense3r.py.')
    return state['args']


def _resolve(path: Optional[str]) -> Optional[str]:
    """Make a checkpoint-relative backbone path absolute against the repo root."""
    if not path or os.path.isabs(path):
        return path
    return os.path.join(REPO_ROOT, path)


def _arch_kwargs(a: Namespace) -> Dict:
    """Translate stored training args into model constructor kwargs."""
    g = lambda k, d=None: getattr(a, k, d)
    kw = dict(
        sem_dim            = a.sem_dim,
        ins_dim            = a.ins_dim,
        n_blocks           = a.n_blocks,
        init_from_vggt     = True,
        dino_backbone      = a.dino_backbone,
        semantic_backbone  = a.semantic_backbone,
        prior_method       = a.prior_method,
        sem_supervision    = a.sem_supervision,
        depth_norm_mode    = a.depth_norm_mode,
        gt_anchor_mode     = g('gt_anchor_mode', 'se3'),
        depth_cond         = a.depth_cond,
        point_cond         = g('point_cond', False),
        rope_mode          = g('rope_mode', '2d'),
        rope_voxel_size    = g('rope_voxel_size', 0.1),
        pose_cond          = a.pose_cond,
        use_head_sem_cond  = (not a.no_head_sem_cond),
        use_sem_conf_head  = g('use_sem_conf_head', False),
        sem_channels       = tuple(int(x) for x in a.sem_channels.split(',')),
        ins_channels       = tuple(int(x) for x in a.ins_channels.split(',')),
        instance_head_type = a.instance_head_type,
        pos3d_cond         = a.pos3d_cond,
        use_depth_cross_attn = (not a.no_depth_cross_attn),
        query_init_type    = a.query_init_type,
        num_instance_queries = a.num_instance_queries,
        instance_query_block_indices = (
            None if g('instance_query_block_indices') is None
            else [int(x) for x in a.instance_query_block_indices.split(',') if x.strip()]),
        instance_queries_init_type = a.instance_queries_init_type,
        # Instance-aware semantic head (query token → per-instance semantic).
        # Old ckpts: g() → False → head not built (sem_proj stays an optional drop).
        inst_sem_head      = g('inst_sem_head', False),
        inst_sem_dim       = g('inst_sem_dim', 1024),
        no_dense_sem       = g('no_dense_sem', False),
        # Dense semantic residual (r + native): old ckpts → False → plain dense head.
        sem_residual       = g('sem_residual', False),
        seed_sem           = g('seed_sem', 'dense'),
        pq_sem_source      = g('pq_sem_source', 'dense'),
    )
    if a.instance_head_type != 'query':
        raise ValueError(
            f'semantic_mapping only supports the query instance head '
            f'(checkpoint says instance_head_type={a.instance_head_type!r})')
    kw.update(
        query_arch               = a.query_arch,
        query_num_queries        = a.query_num_queries,
        query_d_model            = a.query_d_model,
        query_c_mask             = a.query_c_mask,
        query_descriptor_dim     = a.query_descriptor_dim,
        query_n_heads            = a.query_n_heads,
        query_n_layers           = a.query_n_layers,
        query_num_feature_levels = a.query_num_feature_levels,
        query_descriptor_mlp_layers = g('query_descriptor_mlp_layers', 0),
        query_use_box_head       = g('query_use_box_head', False),
        query_box_mode           = g('query_box_mode', 'aabb'),
        query_box_size_anchor    = g('query_box_size_anchor', False),
        query_box_query_only     = g('query_box_query_only', False),
        query_fastsam_mask_prior = g('query_fastsam_mask_prior', False),
        query_box_size_extent_aware = g('query_box_size_extent_aware', False),
        use_pq_head              = g('use_pq_head', False),
        query_merge              = g('query_merge', False),
        query_merge_layer        = g('query_merge_layer', 5),
        query_merge_tau          = g('query_merge_tau', 0.9),
        query_merge_dist_veto    = g('query_merge_dist_veto', 1.5),
        query_merge_act_threshold = g('query_merge_act_threshold', 0.1),
        use_merge_head           = g('use_merge_head', False),
    )
    if g('fastsam_query', False):
        kw.update(fastsam_query       = True,
                  fastsam_ckpt        = _resolve(a.fastsam_ckpt),
                  fastsam_conf        = a.fastsam_conf,
                  fastsam_iou         = a.fastsam_iou,
                  fastsam_max_regions = a.fastsam_max_regions)
    return kw


def _infer_arch_from_weights(state: Dict, kw: Dict, verbose: bool = True) -> Dict:
    """Correct architecture fields using the WEIGHTS as ground truth.

    `ckpt['args']` records what the training script was *told*, which is not
    always what the checkpoint *contains*.  The canonical example ships with
    this repo: `checkpoint-last-descfix.pth` stores
    `query_descriptor_mlp_layers=0` because its training bash omitted the flag,
    yet the file carries 3 `instance_head.descriptor_proj.layers.*` tensors
    (spliced back in after the flag omission silently dropped them).  Building
    from the args alone would drop them a second time.

    So: derive the structural fields that are visible in the state_dict and
    override the args, reporting every correction.
    """
    import re

    fixes = {}
    desc_layers = {int(i) for k in state
                   for i in re.findall(r'instance_head\.descriptor_proj\.layers\.(\d+)\.weight', k)}
    if desc_layers:
        n = max(desc_layers) + 1
        if kw.get('query_descriptor_mlp_layers', 0) != n:
            fixes['query_descriptor_mlp_layers'] = n
    else:
        if kw.get('query_descriptor_mlp_layers', 0):
            fixes['query_descriptor_mlp_layers'] = 0

    has_pq = any(k.startswith('pq_head.') or k.startswith('instance_head.pq_head')
                 for k in state)
    if bool(kw.get('use_pq_head', False)) != has_pq:
        fixes['use_pq_head'] = has_pq

    blocks = {int(i) for k in state
              for i in re.findall(r'causal_branch\.global_blocks\.(\d+)\.', k)}
    if blocks and kw.get('n_blocks') != max(blocks) + 1:
        fixes['n_blocks'] = max(blocks) + 1

    if fixes and verbose:
        print('[model] architecture corrected from WEIGHTS (ckpt args were stale):')
        for k, v in fixes.items():
            print(f'          {k}: args said {kw.get(k)!r} → weights say {v!r}')
    kw.update(fixes)
    return kw


def build_model(ckpt_path: str, device: str = 'cuda', verbose: bool = True):
    """Construct the model described by `ckpt_path` and load its weights.

    Architecture comes from the checkpoint: `ckpt['args']` first, then corrected
    against the actual tensors (weights win — see `_infer_arch_from_weights`).
    Returns (model, ckpt_args).  The geometry backbone is skipped — this
    pipeline always feeds GT depth/pose through `forward_semantic`.
    """
    a = read_ckpt_args(ckpt_path)
    kw = _arch_kwargs(a)
    _state_peek = torch.load(ckpt_path, map_location='cpu',
                             weights_only=False, mmap=True)
    kw = _infer_arch_from_weights(_state_peek.get('model', _state_peek), kw, verbose)
    backend = getattr(a, 'backend', 'vggt')

    if backend == 'da3':
        from opense3r.model_da3 import OpenSe3rDA3
        model = OpenSe3rDA3(
            da3_ckpt    = _resolve(a.da3_ckpt),
            vggt_ckpt   = _resolve(a.vggt_ckpt),
            lbotv_ckpt  = _resolve(getattr(a, 'lbotv_ckpt', None)),
            dinov3_ckpt = _resolve(getattr(a, 'dinov3_ckpt', None)),
            lseg_ckpt   = _resolve(a.lseg_ckpt),
            _skip_geo_backbone = True,      # GT geometry → DA3 never runs
            **kw)
    elif backend == 'vggt':
        from opense3r.model import OpenSe3r
        model = OpenSe3r(vggt_ckpt = _resolve(a.vggt_ckpt),
                         lseg_ckpt = _resolve(a.lseg_ckpt), **kw)
    else:
        raise ValueError(f'unsupported backend {backend!r} '
                         '(semantic_mapping supports vggt | da3)')

    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = state.get('model', state)
    msg = model.load_state_dict(state, strict=False)
    miss = [k for k in msg.missing_keys
            if k.startswith(TRAINABLE_PREFIXES)
            and not k.startswith(OPTIONAL_DROP_PREFIXES)]
    unexp = [k for k in msg.unexpected_keys
             if k.startswith(TRAINABLE_PREFIXES)
             and not k.startswith(OPTIONAL_DROP_PREFIXES)]
    _dropped = [k for k in msg.unexpected_keys
                if k.startswith(OPTIONAL_DROP_PREFIXES)]
    if _dropped and verbose:
        print(f'[model] dropped {len(_dropped)} optional train-only head keys '
              f'(e.g. {_dropped[0]}) — not used by the mapping pipeline')
    if verbose:
        print(f'[model] backend={backend} class={type(model).__name__}')
        print(f'[model] trainable: missing={len(miss)} unexpected={len(unexp)}  '
              f'| frozen backbones: {len(msg.missing_keys)-len(miss)} missing, '
              f'{len(msg.unexpected_keys)-len(unexp)} unexpected (expected)')
    if miss or unexp:
        # A trainable key that did not come from the checkpoint is either a
        # randomly-initialised module (missing) or silently dropped weights
        # (unexpected).  Both produced multi-day debugging sessions before.
        raise RuntimeError(
            f'checkpoint/architecture mismatch: {len(miss)} missing and '
            f'{len(unexp)} unexpected TRAINABLE keys.\n'
            f'  missing (first 5):    {miss[:5]}\n'
            f'  unexpected (first 5): {unexp[:5]}')

    model.to(device).eval()
    if getattr(a, 'query_merge', False) and hasattr(model, 'set_merge_active'):
        model.set_merge_active(True)
    return model, a


@torch.no_grad()
def forward_chunk_gt(model, images, depth, intrinsics, c2w, world_points,
                     world_points_conf, kv_cache, attn_mode='chunk_bidir',
                     box_scale_override=None):
    """One streaming forward with GT geometry (mode B: DA3 is bypassed).

    Shapes: images [1,T,3,H,W]; depth [1,T,H,W]; intrinsics [1,T,3,3];
    c2w [1,T,4,4] (anchored); world_points [1,T,H,W,3]; conf [1,T,H,W,1].
    """
    return model.forward_semantic(
        {'images': images},
        depth             = depth,
        intrinsics        = intrinsics,
        c2w               = c2w,
        world_points      = world_points,
        world_points_conf = world_points_conf,
        box_scale_override = box_scale_override,
        kv_cache_list     = kv_cache,
        mode              = attn_mode,
        return_poses      = False,
    )
