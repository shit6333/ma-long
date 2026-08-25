"""Per-chunk proposal formation — GPU implementation of the reference
``update_instances`` -> ``process_chunk_to_proposals`` path.

The reference runs entirely on CPU (426 ms/frame measured: 191 lift with python
tuple-set voxelization + 165 dense sem_feat GPU->CPU transfer). Here everything
stays on GPU until the *small* per-proposal results:

- NMS / per-pixel resolve: the reference's own ``mask_nms`` / ``resolve_per_pixel``
  (device-agnostic torch) fed GPU tensors directly.
- sem_query: one ``index_add_`` masked-mean over ALL kept slots -> transfer [K,D]
  instead of the dense [T,D,H,W] feature map (712 MB/frame -> ~70 KB).
- lift: quantile-IQR / median / bbox on GPU; voxelization via ``torch.unique`` on
  quantized coords, then a python set is built from the deduped (small) array so
  the reference scoring kernel (``len(A & B)``) is untouched.

Output contract is identical to the reference proposal dicts (CPU tensors +
python voxel sets), so InstanceTable / export need no changes.
``make_proposals_cpu`` keeps the original reference-path implementation for A/B
verification.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ma_slam_semantic.config import SemConfig


# --------------------------------------------------------------------------- sem_query sources
def _batched_sem_query(sem_t: torch.Tensor, owner: torch.Tensor, n_slots: int) -> torch.Tensor:
    """Mask-mean of a dense [D,H,W] map per owner slot in one index_add -> [K,D]
    (L2-normalized; slots owning no pixel get a zero row)."""
    D = sem_t.shape[0]
    flat = owner.reshape(-1)
    sel = flat >= 0
    idx = flat[sel]
    feats = sem_t.permute(1, 2, 0).reshape(-1, D)[sel].float()
    sums = torch.zeros(n_slots, D, device=sem_t.device).index_add_(0, idx, feats)
    cnt = torch.zeros(n_slots, device=sem_t.device).index_add_(
        0, idx, torch.ones_like(idx, dtype=torch.float32))
    out = F.normalize(sums / cnt.clamp(min=1).unsqueeze(1), dim=-1)
    out[cnt == 0] = 0.0
    return out


@torch.no_grad()
def _textregion_sem_query(sem_backbone, img_chw: torch.Tensor, owner_masked: torch.Tensor,
                          n_slots: int) -> torch.Tensor:
    """A1 route (port of semantic_mapping.stages.proposals._textregion_sem_query):
    frozen-teacher (TIPS) textregion pool over the PREDICTED masks. The proposal
    owner map is the instance mask for ``forward_supervision(sem_supervision=
    'textregion')`` -> per-proposal text-aligned feature (mask-restricted
    attention-V pool, no per-pixel context bleed), broadcast dense, then
    mask-mean-collapsed to [K,D]. Costs one extra teacher forward per frame.

    img_chw: [3,H,W] in [-1,1] (loader range) — converted to [0,1] for the teacher.
    """
    H, W = owner_masked.shape
    dev = next(sem_backbone.parameters()).device
    img01 = ((img_chw.to(dev).float() + 1.0) * 0.5).clamp(0, 1).unsqueeze(0)  # [1,3,H,W]
    inst = (owner_masked.to(dev) + 1).clamp(min=0).long().view(1, 1, H, W)    # 0=bg, k+1=prop k
    tr = sem_backbone.forward_supervision(img01, instance_mask=inst,
                                          sem_supervision="textregion")       # [1,1,D,H,W]
    semq = _batched_sem_query(tr[0, 0], owner_masked.to(dev), n_slots)        # [K,D]
    return semq.to(owner_masked.device)


def _sem_mode(cfg: SemConfig, sem_dense, sem_backbone) -> str:
    """Resolve cfg.sem_agg against what THIS forward produced."""
    mode = cfg.sem_agg
    if mode == "auto":
        mode = "dense" if sem_dense is not None else (
            "textregion" if sem_backbone is not None else "none")
    if mode == "dense" and sem_dense is None:
        raise RuntimeError("sem_agg='dense' but the model produced no dense semantic "
                           "map (A1-type checkpoint?) — use sem_agg auto/textregion")
    if mode == "textregion" and sem_backbone is None:
        raise RuntimeError("sem_agg='textregion' needs the frozen semantic backbone "
                           "(pass sem_backbone=model.semantic_backbone)")
    return mode


# --------------------------------------------------------------------------- GPU path
@torch.no_grad()
def make_proposals(out: dict, chunk: dict, cfg: SemConfig, sem_backbone=None
                   ) -> Tuple[List[Dict], torch.Tensor, torch.Tensor]:
    """Returns (proposals, pixel_to_proposal [T,H_f,W_f] CPU, pts_headres [T,H_f,W_f,3] CPU).

    ``sem_backbone`` (the model's frozen TIPS teacher) enables the textregion
    sem_query route for checkpoints without a dense semantic head (cfg.sem_agg).
    P/Q head outputs (``pq_p``/``pq_q``) are recorded on every proposal when
    present — export-side ranking only, they never touch matching."""
    from opense3r.instance_post import mask_nms, resolve_per_pixel

    if not cfg.panoptic:
        # per-query (non-panoptic) masks: sem_query pooling isn't batched here —
        # use the reference CPU path (not the frozen config; correctness first).
        return make_proposals_cpu(out, chunk, cfg)

    ml = out["mask_logits"][0].detach().float()                     # [T,N,H,W] GPU
    obj = out["objectness"][0].detach().float()                     # [T,N]
    dev = ml.device
    desc_cpu = out["descriptor"][0].detach().float().cpu()          # [T,N,D] small

    sem = None
    for k in ("semantic_feat_expanded", "semantic_feat"):
        if out.get(k) is not None:
            sem = out[k][0].detach().float()                        # [T,D,H,W] GPU
            break
    sem_mode = _sem_mode(cfg, sem, sem_backbone)
    images = chunk["images"][0] if sem_mode == "textregion" else None  # [T,3,H,W] [-1,1]
    pq_p = out["pq_p"][0].detach().float() if out.get("pq_p") is not None else None  # [T,N]
    pq_q = out["pq_q"][0].detach().float() if out.get("pq_q") is not None else None

    T, N, H, W = ml.shape
    pts = chunk["world_points_global"].float().to(dev)              # [T,H,W,3]
    conf = chunk["conf"].float().to(dev)
    if pts.shape[1:3] != (H, W):
        pts = F.interpolate(pts.permute(0, 3, 1, 2), size=(H, W),
                            mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    if conf.shape[1:3] != (H, W):
        conf = F.interpolate(conf.unsqueeze(1), size=(H, W),
                             mode="bilinear", align_corners=False).squeeze(1)
    keep = (pts.norm(dim=-1) > 1e-3) & (conf >= cfg.point_conf_threshold)

    top_k = int(cfg.obj_top_k) if cfg.obj_top_k and int(cfg.obj_top_k) > 0 else None
    proposals: List[Dict] = []
    pixel_to_proposal = torch.full((T, H, W), -1, dtype=torch.long, device=dev)

    for t in range(T):
        # 1) objectness gate (+ optional top-k), mirroring process_chunk_to_proposals
        base = obj[t].sigmoid()
        active = base > cfg.obj_threshold
        if top_k is not None and top_k < N:
            tm = torch.zeros_like(active)
            tm[base.topk(top_k).indices] = True
            active &= tm
        if not active.any():
            continue
        active_idx = active.nonzero(as_tuple=True)[0]

        # 2) NMS among active slots (reference impl, GPU tensors)
        if cfg.nms_mode == "matrix":
            from opense3r.instance_post import _matrix_nms_decay
            mbin = (ml[t, active_idx].sigmoid() > 0.5).float()
            decayed = _matrix_nms_decay(mbin, base[active_idx], kernel=cfg.matrix_nms_kernel)
            kept_local = (decayed > cfg.obj_threshold).nonzero(as_tuple=True)[0]
        else:
            mbin = ml[t, active_idx].sigmoid() > 0.5
            kept_local = mask_nms(mbin, base[active_idx], iou_threshold=cfg.nms_iou)
        if kept_local.numel() == 0:
            continue
        kept = active_idx[kept_local]                               # [K] global slot ids
        K = int(kept.numel())

        # 3) pixel ownership (panoptic per-pixel argmax)
        owner, _, act = resolve_per_pixel(ml[t, kept], obj[t, kept], keep[t],
                                          score_threshold=cfg.score_threshold)

        # 4) batched sem_query: masked mean per kept slot via index_add (dense
        #    head), or frozen-TIPS textregion pool over the owner map (A1 route)
        semq = None
        if sem_mode == "textregion":
            semq = _textregion_sem_query(sem_backbone, images[t],
                                         owner.masked_fill(~act, -1), K)
        elif sem_mode == "dense":
            flat_owner = owner.reshape(-1)
            sel = flat_owner >= 0
            idx = flat_owner[sel]
            feats = sem[t].permute(1, 2, 0).reshape(-1, sem.shape[1])[sel]
            sums = torch.zeros(K, sem.shape[1], device=dev).index_add_(0, idx, feats)
            cnt = torch.zeros(K, device=dev).index_add_(
                0, idx, torch.ones_like(idx, dtype=torch.float))
            semq = F.normalize(sums / cnt.clamp(min=1).unsqueeze(1), dim=-1)
            semq[cnt == 0] = 0.0

        # 5) per-slot 3D lift (GPU quantile/median/unique; python only sees the results)
        for k_local in range(K):
            mask_k = (owner == k_local) & act
            if not mask_k.any():
                continue
            p = pts[t][mask_k]
            p = p[torch.isfinite(p).all(dim=-1)]
            P = int(p.shape[0])
            if P == 0:
                continue
            if cfg.outlier_iqr_mult > 0 and P >= 50:               # reference min_pixels_for_filter
                q1 = p.quantile(0.25, dim=0)
                q3 = p.quantile(0.75, dim=0)
                lo = q1 - cfg.outlier_iqr_mult * (q3 - q1)
                hi = q3 + cfg.outlier_iqr_mult * (q3 - q1)
                p = p[((p >= lo) & (p <= hi)).all(dim=-1)]
                if p.shape[0] == 0:
                    continue
            if cfg.min_points > 0 and P < cfg.min_points:
                continue
            vox = torch.unique((p / cfg.voxel_size).floor().long(), dim=0)
            slot = int(kept[k_local].item())
            prop: Dict = {
                "center": p.median(dim=0).values.cpu(),
                "bbox_min": p.min(dim=0).values.cpu(),
                "bbox_max": p.max(dim=0).values.cpu(),
                "voxels": set(map(tuple, vox.cpu().tolist())),
                "num_voxels": int(vox.shape[0]),
                "num_pixels": P,
                "frame": t,
                "slot": slot,
                "descriptor": desc_cpu[t, slot],
                "objectness": float(obj[t, slot].sigmoid().item()),
            }
            if semq is not None:
                prop["sem_query"] = semq[k_local].cpu()
            if pq_p is not None and pq_q is not None:
                # recorded whether or not the ranker uses them (offline re-ranking)
                prop["pq_p"] = float(pq_p[t, slot].sigmoid().item())
                prop["pq_q"] = float(pq_q[t, slot].sigmoid().item())
            pixel_to_proposal[t][mask_k] = len(proposals)
            proposals.append(prop)

    return proposals, pixel_to_proposal.cpu(), pts.cpu()


# --------------------------------------------------------------------------- CPU reference path
def make_proposals_cpu(out: dict, chunk: dict, cfg: SemConfig
                       ) -> Tuple[List[Dict], torch.Tensor, torch.Tensor]:
    """Original reference-path implementation (CPU, mirrors update_instances).
    Kept for A/B verification of the GPU path."""
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
        pq_p=(out["pq_p"][0].detach().cpu().float() if out.get("pq_p") is not None else None),
        pq_q=(out["pq_q"][0].detach().cpu().float() if out.get("pq_q") is not None else None),
    )
    return proposals, pixel_to_proposal.cpu(), pts
