from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_thirdparty = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty')
sys.path.append(_thirdparty)
sys.path.append(os.path.join(_thirdparty, 'STream3R'))

from stream3r.loss.contrastive_loss import ContrastiveLoss


def _aggregate_lseg_by_instance(
    lseg_gt: torch.Tensor,        # [B, T, C, H, W]
    instance_mask: torch.Tensor,  # [B, T, H, W]  long, 0=background
) -> torch.Tensor:
    """
    Replace each pixel's LSeg feature with the per-instance mean.
    Background pixels (mask==0) keep their original per-pixel features.
    Fully vectorised via scatter_add — no Python loops over instances.
    """
    B, T, C, H, W = lseg_gt.shape
    BT, N = B * T, H * W

    gt_flat   = lseg_gt.reshape(BT, C, N)           # [BT, C, N]
    mask_flat = instance_mask.reshape(BT, N).long()  # [BT, N]

    max_id = int(mask_flat.max().item()) + 1

    idx_exp  = mask_flat.unsqueeze(1).expand(BT, C, N)   # [BT, C, N]
    sum_feat = torch.zeros(BT, C, max_id, device=lseg_gt.device, dtype=lseg_gt.dtype)
    count    = torch.zeros(BT, 1, max_id, device=lseg_gt.device, dtype=lseg_gt.dtype)
    sum_feat.scatter_add_(2, idx_exp, gt_flat)
    count.scatter_add_(2, mask_flat.unsqueeze(1),
                       torch.ones_like(mask_flat.unsqueeze(1), dtype=lseg_gt.dtype))

    mean_feat = sum_feat / count.clamp(min=1)         # [BT, C, max_id]
    agg_flat  = mean_feat.gather(2, idx_exp)          # [BT, C, N]

    # Background pixels → keep original
    bg = (mask_flat == 0).unsqueeze(1).expand_as(gt_flat)
    agg_flat = torch.where(bg, gt_flat, agg_flat)

    return agg_flat.reshape(B, T, C, H, W)


def _semantic_cosine_loss(
    feat:          torch.Tensor,              # [B, T, C, H, W]  model output
    gt:            torch.Tensor,              # [B, T, C, H, W] semantic GT (LSeg / PE / CLIP)
    conf:          torch.Tensor,              # [B, T, 1, H, W]
    instance_mask: torch.Tensor | None = None,  # [B, T, H, W]  long
    aggregate_gt_by_instance: bool = True,
    mask_unlabeled: bool = False,
) -> torch.Tensor:
    """
    Pull-only semantic loss: confidence-weighted cosine distillation.

    Args:
        aggregate_gt_by_instance: if True and instance_mask given, GT is
            replaced by per-instance prototypes (legacy LSeg behaviour).
            For TextRegion / mask-crops GT, the aggregation has already
            happened inside the backbone, so set this to False.
        mask_unlabeled: if True and instance_mask given, pixels with
            instance_mask == 0 are excluded from the loss (their conf
            is zeroed out before averaging).
    """
    if aggregate_gt_by_instance and instance_mask is not None:
        gt = _aggregate_lseg_by_instance(gt, instance_mask)

    feat_n = F.normalize(feat, dim=2)
    gt_n   = F.normalize(gt,   dim=2)
    cos    = (feat_n * gt_n).sum(dim=2, keepdim=True).clamp(-1, 1)   # [B,T,1,H,W]
    raw    = (1.0 - cos) * conf - 0.2 * torch.log(conf)               # [B,T,1,H,W]

    if mask_unlabeled and instance_mask is not None:
        valid = (instance_mask > 0).unsqueeze(2).to(raw.dtype)        # [B,T,1,H,W]
        denom = valid.sum().clamp(min=1)
        return (raw * valid).sum() / denom

    return raw.mean()


def _object_consistency_loss(
    feat: torch.Tensor,          # [B, T, C, H, W]
    instance_mask: torch.Tensor, # [B, T, H, W]  globally-consistent integer IDs
    inter_margin: float = 0.2,
) -> torch.Tensor:
    """
    Prototype-level cross-view consistency for instance features.
    For each instance ID, compute per-frame mean feature, then:
      - pull: all per-frame means close to the global mean
      - push: different instance prototypes apart (hinge 0.2)
    """
    B, T, C, H, W = feat.shape
    feat_n = F.normalize(feat, dim=2)
    mask_f = instance_mask.reshape(B, T, H * W)           # [B, T, N]
    feat_f = feat_n.permute(0, 1, 3, 4, 2).reshape(B, T, H * W, C)  # [B, T, N, C]

    total = feat.new_tensor(0.0)
    cnt   = 0

    for b in range(B):
        ids = mask_f[b].unique()
        ids = ids[ids > 0]

        protos = []
        for iid in ids.tolist():
            px = (mask_f[b] == iid)                        # [T, N]  bool
            per_frame = []
            for t in range(T):
                n = px[t].sum().item()
                if n == 0:
                    continue
                per_frame.append(feat_f[b, t][px[t]].mean(dim=0))  # [C]
            if len(per_frame) < 2:
                continue
            stacked = torch.stack(per_frame)               # [k, C]
            stacked = F.normalize(stacked, dim=-1)

            # pull: all frame-prototypes → global mean
            global_proto = F.normalize(stacked.mean(0), dim=0).detach()
            pull = (1.0 - (stacked * global_proto).sum(-1)).mean()
            total = total + pull
            cnt  += 1
            protos.append(F.normalize(stacked.mean(0), dim=0))

        # push: different instances
        if len(protos) >= 2:
            P = torch.stack(protos)                        # [M, C]
            sim = P @ P.t()
            eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
            push = F.relu(sim[~eye] - inter_margin).mean()
            total = total + push
            cnt  += 1

    return total / max(cnt, 1)


class OpenSe3rLoss(nn.Module):
    """
    Combined loss for OpenSe3r:
      - Semantic:     pull-only cosine distillation vs LSeg GT
      - Instance:     ContrastiveLoss (pull same ID, push different IDs)
      - Consistency:  cross-view prototype consistency for instances
    """

    def __init__(
        self,
        w_sem: float = 1.0,
        w_ins: float = 1.0,
        w_consistency: float = 0.5,
        inter_margin: float = 0.2,
        # Semantic GT handling:
        #   - aggregate_gt_by_instance=True  → legacy LSeg path: cosine vs
        #     per-instance-aggregated GT (LSeg dense feat aggregated in loss).
        #   - aggregate_gt_by_instance=False → TextRegion / mask-crops path:
        #     GT is already aggregated per-instance inside the backbone.
        aggregate_gt_by_instance: bool = True,
        mask_unlabeled: bool = False,
    ):
        super().__init__()
        self.w_sem = w_sem
        self.w_ins = w_ins
        self.w_consistency = w_consistency
        self.inter_margin = inter_margin
        self.contrastive = ContrastiveLoss(inter_margin=inter_margin)
        self.aggregate_gt_by_instance = aggregate_gt_by_instance
        self.mask_unlabeled = mask_unlabeled

    def forward(
        self,
        predictions: dict,
        targets: dict,
    ) -> dict:
        """
        predictions keys expected:
            semantic_feat_expanded  [B, T, 512, H, W]  (or semantic_feat if compact not used)
            semantic_feat           [B, T, sem_dim, H, W]
            semantic_conf           [B, T, 1, H, W]
            instance_feat           [B, T, ins_dim, H, W]
            instance_conf           [B, T, 1, H, W]
            _lseg_feat_gt           [B, T, 512, H, W]  (cached LSeg output)

        targets keys expected:
            instance_mask           [B, T, H, W]  long, 0=background
        """
        # Semantic GT — accept new key '_sem_feat_gt' (any backbone) or
        # legacy '_lseg_feat_gt' (LSeg-only models).
        sem_gt   = predictions.get('_sem_feat_gt', predictions.get('_lseg_feat_gt'))
        if sem_gt is None:
            raise KeyError(
                "Loss expected '_sem_feat_gt' or '_lseg_feat_gt' in predictions."
            )
        sem_conf = predictions['semantic_conf']

        # Use expanded features for loss if available (compact-dim + projector)
        sem_for_loss = predictions.get('semantic_feat_expanded', predictions['semantic_feat'])

        # Resize GT to match output spatial resolution if needed
        H_out, W_out = sem_for_loss.shape[-2:]
        if sem_gt.shape[-2:] != (H_out, W_out):
            B, T = sem_gt.shape[:2]
            sem_gt = F.interpolate(
                sem_gt.reshape(B * T, *sem_gt.shape[2:]),
                size=(H_out, W_out), mode='bilinear', align_corners=False
            ).reshape(B, T, *sem_gt.shape[2:])

        # Similarly for instance mask
        instance_mask = targets['instance_mask'].long()   # [B, T, H, W]
        if instance_mask.shape[-2:] != (H_out, W_out):
            B, T = instance_mask.shape[:2]
            instance_mask = F.interpolate(
                instance_mask.float().reshape(B * T, 1, *instance_mask.shape[2:]),
                size=(H_out, W_out), mode='nearest'
            ).long().reshape(B, T, H_out, W_out)

        losses = {}

        # ---- Semantic loss (cosine distillation vs backbone-specific GT) ----
        losses['semantic'] = _semantic_cosine_loss(
            sem_for_loss, sem_gt, sem_conf, instance_mask,
            aggregate_gt_by_instance=self.aggregate_gt_by_instance,
            mask_unlabeled=self.mask_unlabeled,
        )

        # ---- Instance contrastive loss + cross-view consistency ----
        # In 'query' instance-head mode the model returns instance_feat=None;
        # we skip the dense-instance terms entirely and let the query loss
        # (wired in train_opense3r.py CompositeQueryLoss) handle them.
        ins_feat = predictions.get('instance_feat', None)
        if ins_feat is not None:
            ins_intra = self.contrastive.scene_intra_loss(ins_feat, instance_mask)
            ins_inter = self.contrastive.scene_inter_loss(ins_feat, instance_mask)
            losses['instance_intra'] = ins_intra
            losses['instance_inter'] = ins_inter
            losses['consistency'] = _object_consistency_loss(
                ins_feat, instance_mask, inter_margin=self.inter_margin,
            )
            ins_term = (
                self.w_ins * (ins_intra + ins_inter)
                + self.w_consistency * losses['consistency']
            )
        else:
            ins_term = losses['semantic'].new_zeros(())

        # ---- Weighted total ----
        losses['objective'] = self.w_sem * losses['semantic'] + ins_term
        return losses
