"""
slam_opense3r/memory_query.py
==============================
Memory class for query-head SLAM.

Subclasses `OpenSe3rMemory` to keep geometry + semantic-voxel-store logic
identical, but:

  - Drops the dense `ins_store` (VoxelFeatureMap).  Query-head predictions
    are not per-pixel features but per-slot mask/desc — they can't be voxel-
    averaged the same way.

  - Adds `instance_table: InstanceTable` — persistent dict of
    `InstanceRecord`s with cross-chunk Hungarian matching + voxel voting.

  - Adds `update_instances()` — called after `update_geo()` (which caches
    the globally-aligned pts) and before `update_sem()` (which consumes the
    cache).  Runs `process_chunk_to_proposals` then `instance_table.match_chunk`.

The export side reads `memory.instance_table.voxel_to_inst_map()` to turn
the running per-voxel votes into a final voxel → inst_id colouring.
"""

from __future__ import annotations
from typing import Dict, List, Optional

import torch
import numpy as np

from .memory             import OpenSe3rMemory
from .instance_table     import InstanceTable
from opense3r.instance_post import process_chunk_to_proposals


class OpenSe3rQueryMemory(OpenSe3rMemory):
    """OpenSe3rMemory variant whose instance store is an InstanceTable."""

    def __init__(
        self,
        cfg                = None,
        num_frames:        int   = 0,
        H:                 int   = 0,
        W:                 int   = 0,
        sem_feat_dim:      int   = 1024,
        feat_storage:      str   = 'voxel',
        sem_voxel_size:    float = 0.01,
        # Instance-table specifics (independent voxel grid from sem store).
        instance_voxel_size: float = 0.05,
        knn_radius:          float = 2.0,
        max_candidates:      int   = 32,
        match_threshold:     float = 0.35,
        # Descriptor accumulation + matching metric (CTVIS-aligned; defaults =
        # legacy behaviour).  embed_update: mean|momentum|sim_guided;
        # match_metric: hungarian|bisoftmax.
        embed_update:        str   = 'mean',
        embed_momentum:      float = 0.75,
        match_metric:        str   = 'hungarian',
        # Greedy + overlap-gate + desc dead-zone matching (see InstanceTable).
        match_mode:          str   = 'hungarian',
        gate_overlap:        float = 0.10,
        merge_desc_high:     float = 0.75,
        spawn_desc_low:      float = 0.40,
        greedy_gate_use_observed: bool = False,
        # Cue weights — defaults mirror training-time PLY pipeline.
        w_desc:              float = 0.35,
        w_sem:               float = 0.35,
        w_vox:               float = 0.15,
        w_geo:               float = 0.15,
        sigma_dist:          float = 0.5,
        # Proposal extraction thresholds (same defaults as training-time).
        obj_threshold:       float = 0.5,
        # Per-frame cap on proposals by top-K objectness (AND-ed with the
        # threshold above).  0 / None = disabled (threshold-only).  Useful for
        # focal-loss heads whose objectness range is compressed.
        obj_top_k:           Optional[int] = None,
        nms_iou:             float = 0.5,
        score_threshold:     float = 0.25,
        # Tukey-fence multiplier inside `lift_proposal_to_3d` (0 = filter off).
        outlier_iqr_mult:    float = 1.5,
        # Drop per-frame proposals with < this many 3D-lifted points (ESAM
        # npoint_thr analog).  0 = off.
        min_points:          int   = 0,
        # ESAM-style proposal post-processing (defaults = legacy).
        nms_mode:            str   = 'hard',      # 'hard' | 'matrix' (soft decay)
        matrix_nms_kernel:   str   = 'linear',
        panoptic:            bool  = True,        # False → each query's own mask
        sp_score_thr:        float = 0.5,
        obj_normalize:       bool  = False,
        use_sem_conf:        bool  = False,       # A': fold σ(sem_conf) into proposal confidence
        use_sem_conf_gate:   bool  = False,       # A': also fold σ(sem_conf) into gating/top-K
        use_sem_conf_select: bool  = False,       # A': fold σ(sem_conf) into SELECTION only (ownership/topK/NMS)

        # Multi-gate overrides for InstanceTable scoring (None = disabled).
        vox_gate:            Optional[float] = None,
        desc_gate:           Optional[float] = None,
        # Cue fusion + soft semantic gate (0 = off) + new-mode knobs.
        score_combine:       str   = 'add',
        box_iou_mode:        str   = 'min',
        merge_head           = None,
        sem_gate_thr:        float = 0.0,
        vox_bonus_alpha:     float = 1.0,
        desc_pow:            float = 1.0,
        vox_or_tau:          float = 0.5,
        backend:             str   = 'vggt',
        is_metric:           bool  = False,
        point_conf_threshold: Optional[float] = None,
    ):
        # Build the base with ins_feat_dim=0 → `ins_store = None`.  All
        # dense-instance code paths in OpenSe3rMemory are then auto-skipped.
        super().__init__(
            cfg          = cfg,
            num_frames   = num_frames,
            H            = H, W = W,
            sem_feat_dim = sem_feat_dim,
            ins_feat_dim = 0,
            feat_storage = feat_storage,
            voxel_size   = sem_voxel_size,
            backend      = backend,
            is_metric    = is_metric,
            point_conf_threshold = point_conf_threshold,
        )
        self.instance_table = InstanceTable(
            voxel_size      = instance_voxel_size,
            knn_radius      = knn_radius,
            max_candidates  = int(max_candidates),
            match_threshold = match_threshold,
            embed_update    = embed_update,
            embed_momentum  = embed_momentum,
            match_metric    = match_metric,
            match_mode      = match_mode,
            gate_overlap    = gate_overlap,
            merge_desc_high = merge_desc_high,
            spawn_desc_low  = spawn_desc_low,
            greedy_gate_use_observed = greedy_gate_use_observed,
            w_desc          = w_desc,
            w_sem           = w_sem,
            w_vox           = w_vox,
            w_geo           = w_geo,
            sigma_dist      = sigma_dist,
            vox_gate        = vox_gate,
            desc_gate       = desc_gate,
            score_combine   = score_combine,
            box_iou_mode    = box_iou_mode,
            merge_head      = merge_head,
            sem_gate_thr    = sem_gate_thr,
            vox_bonus_alpha = vox_bonus_alpha,
            desc_pow        = desc_pow,
            vox_or_tau      = vox_or_tau,
        )
        # Proposal extraction thresholds — stored on the memory so the
        # pipeline can override per-run via CLI without touching the table.
        self.obj_threshold    = float(obj_threshold)
        # Normalise 0 / negative → None (disabled).
        self.obj_top_k        = int(obj_top_k) if (obj_top_k and int(obj_top_k) > 0) else None
        self.nms_iou          = float(nms_iou)
        self.score_threshold  = float(score_threshold)
        self.outlier_iqr_mult = float(outlier_iqr_mult)
        self.min_points       = int(min_points) if (min_points and int(min_points) > 0) else 0
        self.nms_mode          = str(nms_mode)
        self.matrix_nms_kernel = str(matrix_nms_kernel)
        self.panoptic          = bool(panoptic)
        self.sp_score_thr      = float(sp_score_thr)
        self.obj_normalize     = bool(obj_normalize)
        self.use_sem_conf      = bool(use_sem_conf)
        self.use_sem_conf_gate = bool(use_sem_conf_gate)
        self.use_sem_conf_select = bool(use_sem_conf_select)

    # ------------------------------------------------------------------
    # Override the legacy _push_sem_ins so it doesn't try to read
    # `sem_res['instance_feat']` (which is None in query mode and would
    # crash with `.cpu()` on None).  Only sem voxels are pushed here;
    # query-instance handling lives in update_instances().
    # ------------------------------------------------------------------

    def _push_sem_ins(self, sem_res: dict, pts_world: torch.Tensor,
                      sem_feat_key: str, geom_conf: torch.Tensor = None):
        sem_feat = sem_res.get(sem_feat_key, sem_res['semantic_feat'])[0].cpu().float()
        sem_conf = sem_res['semantic_conf'][0].cpu().float()
        if self.sem_store is not None:
            self._push_features(self.sem_store, pts_world, sem_feat, sem_conf,
                                geom_conf=geom_conf)
        # No ins_store in query mode.

    # ------------------------------------------------------------------
    # Instance update (runs after update_geo, before update_sem)
    # ------------------------------------------------------------------

    def update_instances(
        self,
        sem_res:         dict,
        start_idx:       int,
        end_idx:         int,
        sem_feat_key:    Optional[str] = None,
    ):
        """Extract per-frame proposals from this chunk's query-head outputs
        and merge them into `self.instance_table`.

        Reads from `sem_res`:
            mask_logits  [1, T_new, N, H_f, W_f]
            objectness   [1, T_new, N]
            descriptor   [1, T_new, N, D]
            semantic_feat / semantic_feat_expanded  (optional, for sem_query)

        Uses `self._pending_pts_new` / `self._pending_conf_new` from the
        preceding `update_geo` for the 3D coords — does NOT consume them
        (so the subsequent `update_sem` can still use them).

        Args:
            start_idx, end_idx  inclusive global frame range of this chunk's
                                NEW frames (matches `update_geo` arguments).
        """
        if sem_res is None:
            return
        if 'mask_logits' not in sem_res:
            # Predictions don't carry query outputs → silently skip
            # (caller is probably running dense ins mode).
            return
        if self._pending_pts_new is None:
            raise RuntimeError(
                "update_instances called without a preceding update_geo "
                "(no cached pts_new_global available)."
            )

        n_new = end_idx - start_idx + 1
        device = 'cpu'

        # Head outputs for the new chunk frames.
        mask_logits = sem_res['mask_logits'][0].detach().to(device).float()    # [T,N,H,W]
        objectness  = sem_res['objectness'][0].detach().to(device).float()      # [T,N]
        descriptor  = sem_res['descriptor'][0].detach().to(device).float()      # [T,N,D]
        # A' semantic mask-quality head (if trained + enabled): folds into proposal
        # confidence as σ(obj)·σ(sem_conf).
        _want_sem_conf = (getattr(self, 'use_sem_conf', False)
                          or getattr(self, 'use_sem_conf_gate', False)
                          or getattr(self, 'use_sem_conf_select', False))       # extract if gate/rank/select wants it
        sem_conf = (sem_res['sem_conf'][0].detach().to(device).float()
                    if (_want_sem_conf and sem_res.get('sem_conf') is not None)
                    else None)                                                  # [T,N] or None
        T, N, H_f, W_f = mask_logits.shape
        if T != n_new:
            raise ValueError(
                f"update_instances: head output T={T} but chunk has "
                f"{n_new} new frames ({start_idx}..{end_idx})."
            )

        # 3D coords (already in SLAM global frame).
        pts_new  = self._pending_pts_new.to(device).float()                     # [T,H,W,3]
        conf_new = self._pending_conf_new.to(device).float()                    # [T,H,W]

        # Downsample pts to head resolution if needed.
        if pts_new.shape[1:3] != (H_f, W_f):
            pts_new = self._downsample_pts(pts_new, H_f, W_f)                  # [T,H_f,W_f,3]
        if conf_new.shape[1:3] != (H_f, W_f):
            cn = conf_new.unsqueeze(1)                                          # [T,1,H,W]
            import torch.nn.functional as F
            cn = F.interpolate(cn, size=(H_f, W_f),
                               mode='bilinear', align_corners=False)
            conf_new = cn.squeeze(1)                                            # [T,H_f,W_f]

        # `keep_mask` mirrors what training-time PLY does:
        # require a finite world point + geometric conf above threshold (if set).
        keep_mask = (pts_new.norm(dim=-1) > 1e-3)
        if self.point_conf_threshold is not None:
            keep_mask = keep_mask & (conf_new >= self.point_conf_threshold)

        # Optional sem feature (used as sem_query inside each proposal).
        sem_dense = None
        if sem_feat_key is not None and sem_feat_key in sem_res:
            sem_dense = sem_res[sem_feat_key][0].detach().to(device).float()    # [T,D,H_f,W_f]
        elif 'semantic_feat_expanded' in sem_res and sem_res['semantic_feat_expanded'] is not None:
            sem_dense = sem_res['semantic_feat_expanded'][0].detach().to(device).float()
        elif 'semantic_feat' in sem_res and sem_res['semantic_feat'] is not None:
            sem_dense = sem_res['semantic_feat'][0].detach().to(device).float()

        # Head-predicted box (camera-canonical) — only present when the
        # query head's box_head is enabled at training time.  When given,
        # `process_chunk_to_proposals` converts camera-canonical → world
        # AABB using the chunk's c2w (DA3 / VGGT predicted) and tags
        # proposals with bbox_type='predicted'.
        pred_box   = None
        box_scale_b = None
        poses_t     = None
        box_world_frame = False
        if 'box' in sem_res and sem_res['box'] is not None:
            pred_box = sem_res['box'][0].detach().to(device).float()         # [T, N, 6]
            bs = sem_res.get('box_scale', None)
            if bs is not None:
                box_scale_b = float(bs[0].item())
            # poses[start:end] are global frame c2w (already aligned).
            poses_t = self.poses[start_idx:end_idx + 1].to(device).float()    # [T, 4, 4]
            # `world_aabb` / `world_amodal` heads output a WORLD-frame AABB in
            # NORMALIZE scale (frame-0 anchored) — the correct recovery is just
            # ×box_scale (box_world_frame=True path).  `aabb` / `obb` heads output
            # a CAMERA-canonical box that must be transformed by c2w (path #2).
            # Without this flag the world-frame box was wrongly run through the
            # per-frame c2w transform → same object's box scattered every frame →
            # cross-frame box_IoU≈0 → zero merges (the 2217-fragment bug).
            box_world_frame = sem_res.get('box_mode', 'aabb') in (
                'world_aabb', 'world_amodal', 'world_amodal_grow')

        # Per-frame proposal extraction.
        proposals, _ = process_chunk_to_proposals(
            mask_logits     = mask_logits,
            objectness      = objectness,
            descriptor      = descriptor,
            world_points    = pts_new,
            keep_mask       = keep_mask,
            obj_threshold    = self.obj_threshold,
            obj_top_k        = self.obj_top_k,
            nms_iou          = self.nms_iou,
            score_threshold  = self.score_threshold,
            voxel_size       = self.instance_table.voxel_size,
            sem_feat         = sem_dense,
            box              = pred_box,
            box_scale        = box_scale_b,
            poses            = poses_t,
            box_world_frame  = box_world_frame,
            outlier_iqr_mult = self.outlier_iqr_mult,
            min_points       = self.min_points,
            nms_mode         = getattr(self, 'nms_mode', 'hard'),
            matrix_nms_kernel = getattr(self, 'matrix_nms_kernel', 'linear'),
            panoptic         = getattr(self, 'panoptic', True),
            sp_score_thr     = getattr(self, 'sp_score_thr', 0.5),
            obj_normalize    = getattr(self, 'obj_normalize', False),
            sem_conf         = sem_conf,
            sem_conf_gate    = getattr(self, 'use_sem_conf_gate', False),
            sem_conf_rank    = getattr(self, 'use_sem_conf', False),
            sem_conf_select  = getattr(self, 'use_sem_conf_select', False),
        )
        if not proposals:
            return

        # Each proposal carries a local frame index (0..T-1); map to global.
        frame_indices = [int(p['frame']) + start_idx for p in proposals]

        self.instance_table.match_chunk(proposals, frame_indices)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def instance_count(self) -> int:
        return len(self.instance_table)

    def instance_summary(self) -> Dict[str, int]:
        return self.instance_table.summary()
