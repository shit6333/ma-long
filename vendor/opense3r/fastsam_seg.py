"""FastSAMSegmenter — frozen FastSAM-x → per-frame region-id maps.

Wraps ultralytics' native FastSAM (`from ultralytics import FastSAM`), NOT the
official FastSAM repo (whose vendored ultralytics 8.0.120 shadows / conflicts
with the installed 8.4.x).  Weights default to the local `FastSAM-x.pt`.

The segmenter is a PLAIN object (not nn.Module) and is held by the model under a
non-registered attribute, so its weights never enter the model state_dict —
existing checkpoints load unchanged (backward compatible).

Output is a single per-frame **region-id map** (a partition of the image):
    region_id  [B, T, H, W]  long   — pixel → region index (0..M_t-1), -1 = none
    conf       [B, T, Mmax]  float  — per-region FastSAM confidence (0 for pad)
    n_regions  [B, T]        long   — number of real regions in each frame

Overlapping FastSAM masks are resolved by painting in DESCENDING area order
(ESAM convention: smaller masks land on top).  When more than `max_regions`
masks are returned, the top-`max_regions` by confidence are kept.

Everything runs under no_grad — FastSAM is frozen and never part of autograd.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as _F


def _safe_adaptive_avg_pool2d(x: torch.Tensor, out_hw) -> torch.Tensor:
    """adaptive_avg_pool2d that avoids the CUDA int32 numel limit (2^31) by
    pooling per-batch-item when the whole tensor would overflow.  Needed once
    the semantic feature is wide (e.g. RADIO/SigLIP 1536-d) at full resolution."""
    if x.numel() < (1 << 31):
        return _F.adaptive_avg_pool2d(x, out_hw)
    return torch.cat([_F.adaptive_avg_pool2d(x[i:i + 1], out_hw)
                      for i in range(x.shape[0])], dim=0)


class FastSAMSegmenter:
    """Lazy-loaded FastSAM-x everything-mode segmenter."""

    def __init__(
        self,
        ckpt_path: str = "/mnt/HDD4/ricky/sam_test/FastSAM-x.pt",
        conf: float = 0.2,
        iou: float = 0.9,
        imgsz: int = 640,
        max_regions: int = 100,
        min_region_pixels: int = 0,
    ):
        self.ckpt_path = ckpt_path
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.max_regions = int(max_regions)
        self.min_region_pixels = int(min_region_pixels)
        self._model = None  # lazy

    def _ensure_model(self, device):
        if self._model is None:
            from ultralytics import FastSAM
            self._model = FastSAM(self.ckpt_path)
            # ultralytics moves the model to the inference device on call; we
            # only need it instantiated once.
        return self._model

    @torch.no_grad()
    def _segment_one(self, img01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Segment a single [3, H, W] RGB image in [0, 1].

        Returns:
            region_id [H, W] long (-1 = none), conf [Mmax] float, n_regions int
        """
        H, W = img01.shape[-2], img01.shape[-1]
        device = img01.device
        region_id = torch.full((H, W), -1, dtype=torch.long, device=device)
        conf_out = torch.zeros(self.max_regions, dtype=torch.float32, device=device)

        model = self._ensure_model(device)
        # ultralytics' tensor loader requires H, W divisible by stride 32.
        # Resize to the nearest /32 shape (RGB, [0,1]); masks return at that
        # shape and are interpolated back to (H, W) below.
        H32 = max(32, int(round(H / 32)) * 32)
        W32 = max(32, int(round(W / 32)) * 32)
        inp = img01.unsqueeze(0)
        if (H32, W32) != (H, W):
            inp = torch.nn.functional.interpolate(
                inp, size=(H32, W32), mode='bilinear', align_corners=False,
            )
        res = model(
            inp.clamp(0, 1), device=device, retina_masks=True,
            imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False,
        )[0]
        if res.masks is None or len(res.masks.data) == 0:
            return region_id, conf_out, 0

        masks = res.masks.data.to(device).bool()              # [N, h, w]
        conf = res.boxes.conf.to(device).float() if res.boxes is not None \
            else torch.ones(masks.shape[0], device=device)
        # FastSAM masks can come back at a slightly different (h, w); align.
        if masks.shape[-2:] != (H, W):
            masks = torch.nn.functional.interpolate(
                masks.unsqueeze(1).float(), size=(H, W), mode='nearest',
            ).squeeze(1).bool()

        areas = masks.flatten(1).sum(1)                        # [N]
        keep = areas >= max(1, self.min_region_pixels)
        masks, conf, areas = masks[keep], conf[keep], areas[keep]
        if masks.shape[0] == 0:
            return region_id, conf_out, 0

        # Cap to top-max_regions by confidence.
        if masks.shape[0] > self.max_regions:
            top = torch.topk(conf, self.max_regions).indices
            masks, conf, areas = masks[top], conf[top], areas[top]

        # Paint in DESCENDING area order so smaller masks land on top (ESAM).
        order = torch.argsort(areas, descending=True)
        for new_id, i in enumerate(order.tolist()):
            region_id[masks[i]] = new_id
            conf_out[new_id] = conf[i]
        n = int(order.numel())
        return region_id, conf_out, n

    @torch.no_grad()
    def segment(self, images01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Segment a batch of frames.

        Args:
            images01: [B, T, 3, H, W] RGB in [0, 1].
        Returns:
            region_id [B, T, H, W] long, conf [B, T, Mmax] float, n_regions [B, T] long
        """
        B, T, _, H, W = images01.shape
        region_id = torch.full((B, T, H, W), -1, dtype=torch.long, device=images01.device)
        conf = torch.zeros((B, T, self.max_regions), dtype=torch.float32, device=images01.device)
        n_regions = torch.zeros((B, T), dtype=torch.long, device=images01.device)
        for b in range(B):
            for t in range(T):
                rid, c, n = self._segment_one(images01[b, t])
                region_id[b, t] = rid
                conf[b, t] = c
                n_regions[b, t] = n
        return region_id, conf, n_regions


def build_seed_queries(
    fastsam_seg, region_pool, patch_size: int, max_regions: int,
    images_01: torch.Tensor,            # [B, T, 3, H, W] in [0, 1]
    ins_tokens_list,                    # causal patch tokens (instance slice)
    sem_feat: torch.Tensor,             # [B, T, sem_dim, H, W] (dense seeding)
    world_points: torch.Tensor,         # [B, T, H, W, 3]
    patch_start_idx: int,
    sem_backbone=None,                  # TextRegion seeding: the frozen teacher
    shared: dict = None,                # TextRegion seeding: cached shared_v_forward
):
    """FastSAM region → geo-aware-pooled seed queries (patch resolution).

    Shared by OpenSe3r and OpenSe3rDA3 (which do not share a base class).
    The semantic feature is DETACHED here so the seed path never trains the
    semantic head.

    Returns:
        seed        [B, T, Mmax, d_model]
        region_id   [B, T, H, W] long  (-1 = none)
        region_conf [B, T, Mmax] float
    """
    import torch.nn.functional as F
    B, T, _, H, W = images_01.shape
    BT = B * T
    ph, pw = H // patch_size, W // patch_size

    region_id, region_conf, _ = fastsam_seg.segment(images_01)             # [B,T,H,W],[B,T,Mmax]
    region_patch = region_id_to_patch(region_id.reshape(BT, H, W), ph, pw, max_regions)

    causal = ins_tokens_list[-1][:, :, patch_start_idx:]                    # [B,T,Npatch,Cf]
    causal = causal.reshape(BT, ph * pw, causal.shape[-1])

    sem_regions = None
    if sem_backbone is not None and shared is not None:
        # ── TextRegion seeding ──
        # The dense head's output is the STUDENT of exactly this teacher, so
        # pooling TextRegion straight off the cached V skips the student and
        # gives the seed the same mask-aware, bleed-free vector the instance
        # semantics are later read out with.  It also removes the last consumer
        # of the dense semantic head, which can then be dropped entirely.
        with torch.no_grad():
            onehot = F.one_hot(region_patch.clamp(min=0),
                               num_classes=max_regions).to(torch.float32)
            onehot = onehot * (region_patch >= 0).unsqueeze(-1).to(onehot.dtype)
            onehot = onehot.permute(0, 2, 1)                    # [BT, Mmax, Npatch]
            # The teacher pre-resizes its input, so its patch grid is NOT the
            # model's (e.g. 32x32 vs 24x37).  Resample the region masks onto the
            # teacher grid before pooling its V.
            th, tw = shared['patch_h'], shared['patch_w']
            if (th, tw) != (ph, pw):
                onehot = F.interpolate(onehot.reshape(BT, max_regions, ph, pw),
                                       (th, tw), mode='bilinear',
                                       align_corners=False).clamp(min=0)
                onehot = onehot.reshape(BT, max_regions, th * tw)
            sem_regions = torch.stack([
                sem_backbone.textregion_from_shared(shared, onehot[i], frame_idx=i)
                for i in range(BT)
            ])                                                  # [BT, Mmax, sem_dim]
        sem = causal.new_zeros((BT, ph * pw, sem_regions.shape[-1]))
    else:
        sem = sem_feat.detach().reshape(BT, sem_feat.shape[2], H, W)
        sem = _safe_adaptive_avg_pool2d(sem, (ph, pw)).flatten(2).permute(0, 2, 1)

    wp = world_points.reshape(BT, H, W, 3).permute(0, 3, 1, 2)
    wp = _safe_adaptive_avg_pool2d(wp, (ph, pw)).flatten(2).permute(0, 2, 1)    # [BT,Npatch,3]

    seed, _ = region_pool(causal, sem, wp, region_patch, max_regions,
                          sem_regions=sem_regions)                       # [BT,Mmax,d_model]
    seed = seed.reshape(B, T, max_regions, seed.shape[-1])
    return seed, region_id, region_conf


def region_id_to_patch(region_id: torch.Tensor, patch_h: int, patch_w: int,
                       max_regions: int) -> torch.Tensor:
    """Downsample a full-res region-id map to the patch grid by majority vote.

    Args:
        region_id: [BT, H, W] long (-1 = none)
        patch_h, patch_w: target patch grid.
        max_regions: region ids are in [0, max_regions); fixes the one-hot width
            so the op is sync-free (no `.item()` on a device tensor).
    Returns:
        [BT, patch_h * patch_w] long — per-patch region id (-1 if the patch's
        majority is background).  Region ids are unchanged (still 0-based dense
        per frame), so they index the same regions as the full-res map.
    """
    BT, H, W = region_id.shape
    device = region_id.device
    ph_size, pw_size = H // patch_h, W // patch_w
    out = torch.full((BT, patch_h * patch_w), -1, dtype=torch.long, device=device)
    for b in range(BT):
        rid = region_id[b]
        # Crop to an exact multiple of the patch size, then fold into patches.
        rid = rid[:patch_h * ph_size, :patch_w * pw_size]
        rid = rid.reshape(patch_h, ph_size, patch_w, pw_size).permute(0, 2, 1, 3)
        rid = rid.reshape(patch_h * patch_w, ph_size * pw_size)        # [Npatch, ppix]
        # Vectorized majority vote per patch (shift +1 so -1→0 is one-hot-able).
        # Fixed bin width (max_regions+1) keeps this sync-free.
        shifted = (rid + 1).clamp_(0, max_regions)
        onehot = torch.nn.functional.one_hot(shifted, max_regions + 1).sum(1)  # [Npatch, M+1]
        out[b] = onehot.argmax(dim=1) - 1                              # back to region id
    return out
