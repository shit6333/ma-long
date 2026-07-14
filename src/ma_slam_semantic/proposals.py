"""Per-chunk proposal formation — mirror of ``memory_query.update_instances``'s
preprocessing around ``opense3r.instance_post.process_chunk_to_proposals``.

Proposals are born in GLOBAL coordinates (fed ``world_points_global``); the head-res
point maps used for the lift are returned so the caller can register them with the
provenance geometry resolver (re-lifts then reproduce the exact same code path).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ma_slam_semantic.config import SemConfig


def make_proposals(out: dict, chunk: dict, cfg: SemConfig
                   ) -> Tuple[List[Dict], torch.Tensor, torch.Tensor]:
    """Returns (proposals, pixel_to_proposal [T,H_f,W_f], pts_headres [T,H_f,W_f,3])."""
    from opense3r.instance_post import process_chunk_to_proposals

    mask_logits = out["mask_logits"][0].detach().cpu().float()      # [T,N,H_f,W_f]
    objectness = out["objectness"][0].detach().cpu().float()        # [T,N]
    descriptor = out["descriptor"][0].detach().cpu().float()        # [T,N,D]
    T, N, H_f, W_f = mask_logits.shape

    pts = chunk["world_points_global"].float()                      # [T,H,W,3] global
    conf = chunk["conf"].float()                                    # [T,H,W]
    if pts.shape[1:3] != (H_f, W_f):
        pts = F.interpolate(pts.permute(0, 3, 1, 2), size=(H_f, W_f),
                            mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    if conf.shape[1:3] != (H_f, W_f):
        conf = F.interpolate(conf.unsqueeze(1), size=(H_f, W_f),
                             mode="bilinear", align_corners=False).squeeze(1)
    keep = (pts.norm(dim=-1) > 1e-3) & (conf >= cfg.point_conf_threshold)

    sem_dense = None
    for k in ("semantic_feat_expanded", "semantic_feat"):
        if out.get(k) is not None:
            sem_dense = out[k][0].detach().cpu().float()            # [T,D,H_f,W_f]
            break

    proposals, pixel_to_proposal = process_chunk_to_proposals(
        mask_logits=mask_logits,
        objectness=objectness,
        descriptor=descriptor,
        world_points=pts,
        keep_mask=keep,
        obj_threshold=cfg.obj_threshold,
        obj_top_k=(int(cfg.obj_top_k) if cfg.obj_top_k and int(cfg.obj_top_k) > 0 else None),
        nms_iou=cfg.nms_iou,
        score_threshold=cfg.score_threshold,
        voxel_size=cfg.voxel_size,
        sem_feat=sem_dense,
        outlier_iqr_mult=cfg.outlier_iqr_mult,
        min_points=cfg.min_points,
        nms_mode=cfg.nms_mode,
        matrix_nms_kernel=cfg.matrix_nms_kernel,
        panoptic=cfg.panoptic,
        sp_score_thr=cfg.sp_score_thr,
        obj_normalize=False,
        sem_conf=None,
    )
    return proposals, pixel_to_proposal.cpu(), pts
