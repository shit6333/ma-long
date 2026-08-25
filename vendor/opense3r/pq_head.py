"""P/Q head — decoupled objectness (P) / mask-quality (Q) scorer.

Mask-Scoring-R-CNN-inspired "inspect your own work" confidence head
(design + supervision: discussion 2026-07-22; diagnosis: 2D conf-rank-loss
22-25 AP50, 3D rank-only oracle +27 AP50 @ep69 strict).

Inputs per query (ALL DETACHED in phase A — pure scorer):
  1. q_norm  [K, d_model]        decoder-normed query (appearance prior)
  2. pooled  [K, c_mask]         mask-prob-weighted mean of F_mask (content)
  3. crop    [K, 17, 28, 28]     RoIAlign(σ(mask) 1ch ‖ 1x1conv(F_mask) 16ch)
                                  over the mask bbox (+15% margin) — shape +
                                  boundary/feature-alignment inspection
  4. scale   [K, 4]              log(bbox area frac), aspect, cx, cy

Outputs: (p_logit, q_logit) per query.
  P: is-a-thing.  BCE; matched→thing=1, matched→bg=0 (w1), ∅=0 (w eos).
  Q: mask quality. L2(σ(q), soft-IoU); matched→thing only. No binarize, no floor.
Inference: gate = σ(P); rank conf = σ(P)·σ(Q).
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

CROP = 28
FEAT_CH = 16          # F_mask channels after compression
MARGIN = 0.15         # bbox expansion (see boundary-context rationale)
PROJ_SEED = 1234      # fixed random projection seed (dump/train/infer MUST match)


def fixed_projection(c_in: int = 256, c_out: int = FEAT_CH,
                     seed: int = PROJ_SEED) -> torch.Tensor:
    """Deterministic random projection [c_out, c_in] for F_mask channel
    compression AT DUMP TIME (raw 256ch crops = ~120 GB / 3k frames — the
    disk blow-up).  JL-style distance preservation; the head's mask_cnn
    learns on top of the projected channels.  Same seed everywhere."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(c_out, c_in, generator=g) / (c_in ** 0.5)


def _mlp(i, o):
    return nn.Sequential(nn.Linear(i, o), nn.ReLU(inplace=True), nn.Linear(o, o))


class PQHead(nn.Module):
    def __init__(self, d_model: int = 256, c_mask: int = 256, hidden: int = 128,
                 sem_dim: int = 0):
        # sem_dim > 0 (v2): 4th input = mask-prob-pooled SEMANTIC feature
        # (text-space) — the A'-style ranking signal, learned into the fusion.
        super().__init__()
        self.sem_dim   = int(sem_dim)
        self.proj_q    = _mlp(d_model, hidden)
        self.proj_pool = _mlp(c_mask, hidden)
        self.proj_sem  = _mlp(sem_dim, hidden) if sem_dim else None
        self.feat_1x1  = nn.Conv2d(c_mask, FEAT_CH, 1)
        ch = FEAT_CH + 1
        self.mask_cnn = nn.Sequential(
            nn.Conv2d(ch, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # 28→14
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),   # 14→7
            nn.Conv2d(64, hidden, 3, stride=2, padding=1), nn.ReLU(inplace=True),  # 7→4
            nn.AdaptiveAvgPool2d(1),
        )
        n_in = hidden * (3 + (1 if sem_dim else 0)) + 4
        self.fusion = _mlp(n_in, 256)
        self.p_head = nn.Linear(256, 1)
        self.q_head = nn.Linear(256, 1)

    def forward(self, q_norm, pooled, crop, scale, sem_pooled=None):
        """All inputs [K, ...] (flat over valid queries). Returns ([K], [K]).

        crop is either [K, 1+c_mask, 28, 28] (raw dump — feat_1x1 applied here,
        keeping the compression trainable) or [K, 1+FEAT_CH, 28, 28] (already
        compressed at inference time, feat_1x1 applied pre-crop for speed).
        """
        if crop.shape[1] != 1 + FEAT_CH:
            crop = torch.cat([crop[:, :1], self.feat_1x1(crop[:, 1:])], dim=1)
        parts = [
            self.proj_q(q_norm),
            self.proj_pool(pooled),
            self.mask_cnn(crop).flatten(1),
        ]
        if self.proj_sem is not None:
            assert sem_pooled is not None, 'sem_dim>0 head needs sem_pooled input'
            parts.append(self.proj_sem(sem_pooled))
        parts.append(scale)
        z = torch.cat(parts, dim=-1)
        z = self.fusion(z)
        return self.p_head(z).squeeze(-1), self.q_head(z).squeeze(-1)


@torch.no_grad()
def build_pq_inputs(
    mask_logits: torch.Tensor,     # [N, H, W]  final-layer mask logits (one frame)
    F_mask:      torch.Tensor,     # [C, H, W]  pixel-decoder mask features
    q_norm:      torch.Tensor,     # [N, D]     final-layer decoder-normed queries
    feat_1x1:    Optional[nn.Conv2d] = None,   # PQHead.feat_1x1 (compress before crop);
                                               # None → compress with mean-pool to FEAT_CH? No:
                                               # caller MUST pass the head's conv for train/infer
                                               # consistency; dump stores RAW 256ch crop instead.
    raw_feat_crop: bool = False,   # True (dump): crop keeps 256ch F_mask (compress at train time)
) -> Tuple[dict, torch.Tensor]:
    """Build PQ inputs for ONE frame. Returns (inputs dict, valid [N] bool).

    invalid = mask empty after σ>0.5 (no work to inspect → Q untrained, P by
    identity; at inference score falls back to legacy objectness gate=fail).
    """
    N, H, W = mask_logits.shape
    dev = mask_logits.device
    prob = mask_logits.sigmoid()                                  # [N, H, W]
    hard = prob > 0.5
    valid = hard.flatten(1).any(1)                                # [N]

    # pooled content vector (soft weights, full frame — cheap einsum)
    w = prob.flatten(1)                                           # [N, HW]
    pooled = (w @ F_mask.flatten(1).t()) / w.sum(1, keepdim=True).clamp_min(1e-6)

    # bboxes from hard mask, +15% margin, clamped
    ys = hard.any(2).float()                                       # [N, H]
    xs = hard.any(1).float()                                       # [N, W]
    def _minmax(m, L):
        idx = torch.arange(L, device=dev).float()
        lo = torch.where(m.bool(), idx, torch.full_like(idx, 1e9)).min(1).values
        hi = torch.where(m.bool(), idx, torch.full_like(idx, -1e9)).max(1).values
        return lo, hi
    y1, y2 = _minmax(ys, H)
    x1, x2 = _minmax(xs, W)
    mh, mw = (y2 - y1).clamp_min(1) * MARGIN, (x2 - x1).clamp_min(1) * MARGIN
    y1, y2 = (y1 - mh).clamp(0, H - 1), (y2 + mh).clamp(0, H - 1)
    x1, x2 = (x1 - mw).clamp(0, W - 1), (x2 + mw).clamp(0, W - 1)
    # Empty masks (invalid rows) yield INVERTED boxes (minmax sentinels) —
    # the CUDA roi_align kernel hits illegal memory on them.  Force a dummy
    # valid box; those rows are filtered by `valid` downstream anyway.
    zero = torch.zeros_like(x1)
    x1 = torch.where(valid, x1, zero); y1 = torch.where(valid, y1, zero)
    x2 = torch.where(valid, x2, zero + 1); y2 = torch.where(valid, y2, zero + 1)
    x2 = torch.maximum(x2, x1 + 1).clamp(0, W - 1)
    y2 = torch.maximum(y2, y1 + 1).clamp(0, H - 1)

    # Crop via PURE-PYTORCH grid_sample (torchvision roi_align REPLACED: its
    # custom CUDA kernel already produced one illegal-memory-access on this
    # pipeline and is the prime suspect for the stochastic silent-corruption
    # NaNs in MV joint training).  Math-equivalent bilinear crop+resize.
    feat = F_mask if raw_feat_crop or feat_1x1 is None else feat_1x1(F_mask.unsqueeze(0))[0]
    ys_lin = torch.linspace(0.0, 1.0, CROP, device=dev)                # [28]
    xs_lin = torch.linspace(0.0, 1.0, CROP, device=dev)
    # per-query sample coords in ABSOLUTE pixels, then → [-1,1] grid space
    gy = y1.view(N, 1) + (y2 - y1).view(N, 1) * ys_lin                 # [N, 28]
    gx = x1.view(N, 1) + (x2 - x1).view(N, 1) * xs_lin                 # [N, 28]
    gy_n = gy / max(H - 1, 1) * 2 - 1
    gx_n = gx / max(W - 1, 1) * 2 - 1
    grid = torch.stack([gx_n.view(N, 1, CROP).expand(N, CROP, CROP),
                        gy_n.view(N, CROP, 1).expand(N, CROP, CROP)], dim=-1)  # [N,28,28,2]
    crop_p = torch.nn.functional.grid_sample(
        prob.unsqueeze(1), grid, mode='bilinear', align_corners=True)   # [N,1,28,28]
    # feat is ONE image for all N boxes → sample with batch=1 via a merged grid
    # (expanding feat N-fold would materialize ~17 GB on the raw-256ch path).
    Cf = feat.shape[0]
    grid_f = grid.reshape(1, N * CROP, CROP, 2)                          # [1, N*28, 28, 2]
    crop_f = torch.nn.functional.grid_sample(
        feat.unsqueeze(0), grid_f, mode='bilinear', align_corners=True)  # [1, C', N*28, 28]
    crop_f = crop_f.reshape(Cf, N, CROP, CROP).permute(1, 0, 2, 3)       # [N, C', 28, 28]
    crop = torch.cat([crop_p, crop_f], dim=1)                            # [N,1+C',28,28]

    scale = torch.stack([
        (((y2 - y1) * (x2 - x1)) / (H * W)).clamp_min(1e-6).log(),
        ((x2 - x1).clamp_min(1) / (y2 - y1).clamp_min(1)).log(),
        (x1 + x2) / (2 * W),
        (y1 + y2) / (2 * H),
    ], dim=1)                                                     # [N, 4]

    return dict(q_norm=q_norm, pooled=pooled, crop=crop, scale=scale), valid


@torch.no_grad()
def soft_iou_matrix(mask_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """[N,H,W] logits × [M,H,W] gt → [N,M] soft-IoU (v2: best-vs-any-GT labels)."""
    p = mask_logits.sigmoid().flatten(1)                       # [N, HW]
    g = (gt > 0.5).float().flatten(1)                          # [M, HW]
    inter = p @ g.t()                                          # [N, M]
    return inter / (p.sum(1, keepdim=True) + g.sum(1)[None] - inter + 1e-6)


@torch.no_grad()
def soft_iou(mask_logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Soft IoU per pair. mask_logits [M,H,W] (matched preds), gt [M,H,W] binary."""
    p = mask_logits.sigmoid().flatten(1)
    g = (gt > 0.5).float().flatten(1)
    inter = (p * g).sum(1)
    return inter / (p.sum(1) + g.sum(1) - inter + 1e-6)
