"""
slam_opense3r/pipeline_query_gt.py
===================================
GT-mode SLAM pipelines for the query-based instance head.

Two classes:

  - `OpenSe3rQuerySLAM_GT`         — flat KV cache, GT depth + GT pose
  - `OpenSe3rQuerySLAM_VoxelKV_GT` — 3-tier voxel KV memory, GT depth + GT pose

Both subclass the corresponding legacy GT pipeline, override `run()`:
  - skip the `instance_feat` shape probe (it's None for query head),
  - build `OpenSe3rQueryMemory` instead of `OpenSe3rMemory`,
  - between the chunk forward and `memory.update_gt(...)` push, set
    `_pending_pts_new` / `_pending_conf_new` and call
    `memory.update_instances(...)` so the InstanceTable accumulates
    proposals before the sem voxel store update runs.

Existing GT pipelines are NOT modified; existing scripts continue to work.
"""

from __future__ import annotations
from typing import Optional
import time
import torch

from .pipeline_gt        import OpenSe3rSLAM_GT
from .pipeline_voxel_kv  import OpenSe3rSLAM_VoxelKV_GT, VoxelKVConfig
from .memory_query       import OpenSe3rQueryMemory
from .                   import mem_stats


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers for `--global_scale_mode` in GT pipelines.
# GT pipelines do not inherit from OpenSe3rSLAM (no coord alignment needed),
# so the helpers added there are not reachable from here — replicate them
# as free functions so both flat-KV and voxel-KV variants share one source.
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_SCALE_MODES = ('per_chunk', 'first_chunk', 'gt_seq_median')


def _validate_global_scale_mode(mode: str) -> str:
    if mode not in _GLOBAL_SCALE_MODES:
        raise ValueError(
            f"global_scale_mode must be one of {_GLOBAL_SCALE_MODES}, "
            f"got {mode!r}"
        )
    return mode


def _maybe_capture_seq_box_scale(pipeline, sem_res: dict) -> None:
    """Cache the model's reported `box_scale` from the first chunk on
    `pipeline._seq_box_scale` so subsequent chunks can pass it as
    `box_scale_override`.  No-op outside 'first_chunk' mode."""
    if pipeline.global_scale_mode != 'first_chunk':
        return
    if pipeline._seq_box_scale is not None:
        return
    bs = sem_res.get('box_scale', None)
    if bs is None:
        print("[global_scale_mode=first_chunk] WARN: model did not return "
              "'box_scale' from first chunk — staying in per-chunk mode.")
        return
    pipeline._seq_box_scale = bs.detach().float().cpu().clone()
    print(f"[global_scale_mode=first_chunk] captured "
          f"box_scale={pipeline._seq_box_scale.tolist()}")


def _current_box_scale_override(pipeline,
                                device: torch.device,
                                ) -> Optional[torch.Tensor]:
    bs = pipeline._seq_box_scale
    return None if bs is None else bs.to(device)


def _compute_gt_seq_box_scale(pts_gt: torch.Tensor,
                              poses_gt: torch.Tensor,
                              ) -> torch.Tensor:
    """Sequence-global GT depth median (filtered to valid pixels) wrapped
    in a [B=1] tensor matching the model's expected shape."""
    T = pts_gt.shape[0]
    w2c     = torch.inverse(poses_gt.float())                                # [T, 4, 4]
    pts_h   = torch.cat([pts_gt.float(),
                         torch.ones_like(pts_gt[..., :1])], dim=-1)          # [T, H, W, 4]
    pts_cam = torch.einsum('tij,thwj->thwi', w2c, pts_h)                      # [T, H, W, 4]
    depth   = pts_cam[..., 2]                                                 # [T, H, W]
    valid   = (pts_gt.float().norm(dim=-1) > 1e-3) & (depth > 0)              # [T, H, W]

    frame_meds = []
    for i in range(T):
        d_valid = depth[i][valid[i]]
        if d_valid.numel() > 0:
            frame_meds.append(d_valid.median())
    if not frame_meds:
        print("[global_scale_mode=gt_seq_median] WARN: no valid GT depths "
              "in the whole sequence — falling back to 1.0")
        return torch.tensor([1.0], dtype=torch.float32)
    frame_meds_t = torch.stack(frame_meds)
    return frame_meds_t.median().clamp(min=0.1).view(1).to(torch.float32)


def _anchor_pts_to_frame0(pts_chunk_raw: torch.Tensor,
                          pose_anchor_inv: torch.Tensor,
                          ) -> torch.Tensor:
    """Transform [Tc, H, W, 3] GT world-frame points to the
    poses_gt[0]-anchored SLAM-global frame.  Pixels that were originally
    invalid (zero) are re-zeroed after the rigid transform."""
    pts_h = torch.cat([pts_chunk_raw,
                       torch.ones_like(pts_chunk_raw[..., :1])], dim=-1)
    pts_anchored = torch.einsum(
        'ij,thwj->thwi', pose_anchor_inv, pts_h
    )[..., :3]
    invalid = pts_chunk_raw.norm(dim=-1) <= 1e-3
    pts_anchored[invalid] = 0.0
    return pts_anchored


def _forward_semantic_gt(model, chunk_imgs, kv_cache, *,
                         chunk_start: int, chunk_end: int,
                         pts_gt: torch.Tensor, poses_gt: torch.Tensor,
                         depth_gt: torch.Tensor, intrinsics_gt: torch.Tensor,
                         pose_anchor_inv: torch.Tensor,
                         box_scale_override: Optional[torch.Tensor],
                         attn_mode: str, device: torch.device):
    """mode-B forward: feed GT depth / world_points / pose / intrinsics into the
    model's geometry conditioning via `forward_semantic` (skips DA3 entirely).

    Geometry is anchored to poses_gt[0] (SLAM-global frame, == training's
    sequence-frame-0 anchor) and invalid world_points are mean-filled with the
    SAME helper as GT-geometry training so mode-B ≡ training preprocessing.
    """
    from opense3r.model import fill_invalid_world_points
    c2w_anchored = (pose_anchor_inv @ poses_gt[chunk_start:chunk_end].float()
                    ).unsqueeze(0).to(device)                                # [1,Tc,4,4]
    pts_raw  = pts_gt[chunk_start:chunk_end].float()                         # [Tc,H,W,3]
    valid    = (pts_raw.norm(dim=-1) > 1e-3)                                 # [Tc,H,W]
    pts_h    = torch.cat([pts_raw, torch.ones_like(pts_raw[..., :1])], dim=-1)
    pts_anch = torch.einsum('ij,thwj->thwi', pose_anchor_inv, pts_h)[..., :3]
    pts_fill = fill_invalid_world_points(pts_anch, valid).unsqueeze(0).to(device)   # [1,Tc,H,W,3]
    depth_ch = depth_gt[chunk_start:chunk_end].float().unsqueeze(0).to(device)       # [1,Tc,H,W]
    K_ch     = intrinsics_gt[chunk_start:chunk_end].float().unsqueeze(0).to(device)  # [1,Tc,3,3]
    conf_ch  = valid.float().unsqueeze(0).unsqueeze(-1).to(device)                   # [1,Tc,H,W,1]
    return model.forward_semantic(
        {'images': chunk_imgs},
        depth              = depth_ch,
        intrinsics         = K_ch,
        c2w                = c2w_anchored,
        world_points       = pts_fill,
        world_points_conf  = conf_ch,
        box_scale_override = box_scale_override,
        kv_cache_list      = kv_cache,
        mode               = attn_mode,
        return_poses       = False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: instance-table knob bundle (shared with non-GT counterpart).
# ─────────────────────────────────────────────────────────────────────────────

class _InstanceKnobMixin:
    """Holds the InstanceTable hyperparameters as plain attributes.
    Used by both GT pipelines below.  Not a true mixin (does not call
    super().__init__) — the caller copies these onto self by hand."""

    DEFAULT_KNOBS = dict(
        instance_voxel_size  = 0.05,
        knn_radius           = 2.0,
        max_candidates       = 32,
        match_threshold      = 0.35,
        w_desc               = 0.35,
        w_sem                = 0.35,
        w_vox                = 0.15,
        w_geo                = 0.15,
        sigma_dist           = 0.5,
        obj_threshold        = 0.5,
        obj_top_k            = 0,        # 0 = off (threshold-only)
        min_points           = 0,        # drop proposals < N 3D points (0 = off)
        nms_iou              = 0.5,
        score_threshold      = 0.25,
        outlier_iqr_mult     = 1.5,
        vox_gate             = None,
        desc_gate            = None,
        score_combine        = 'add',
        box_iou_mode         = 'min',
        sem_gate_thr         = 0.0,
        vox_bonus_alpha      = 1.0,
        desc_pow             = 1.0,
        vox_or_tau           = 0.5,
    )

    def _maybe_dump_2d_masks(self, imgs, sem_res, frame_start: int) -> None:
        """Optional per-frame 2D instance-mask grid dump (top-K objectness).
        No-op unless `save_2d_masks` is set (run.py --save_2d_inst_masks)."""
        if not getattr(self, 'save_2d_masks', False):
            return
        from .vis_2d_masks import dump_frame_instance_masks
        dump_frame_instance_masks(
            images      = imgs,
            sem_res     = sem_res,
            frame_start = frame_start,
            out_dir     = getattr(self, 'mask_2d_out_dir', './2d_inst_masks'),
            demo_name   = getattr(self, 'mask_2d_demo_name', 'scene'),
            top_k       = int(getattr(self, 'mask_2d_top_k', 15)),
        )

    def _store_inst_knobs(self, **overrides):
        for k, default in self.DEFAULT_KNOBS.items():
            v = overrides.get(k, default)
            # `score_combine` is a string knob — store verbatim.
            if k in ('score_combine', 'box_iou_mode'):
                setattr(self, k, str(v))
                continue
            # Some knobs (vox_gate / desc_gate) are nullable — None means OFF.
            setattr(self, k, None if v is None else float(v))

    def _build_query_memory(self, T: int, H: int, W: int,
                             sem_feat_dim: int) -> OpenSe3rQueryMemory:
        cfg = self.cfg
        _mh = getattr(self.model, 'merge_head', None)
        _sc = self.score_combine
        if _sc == 'learned' and _mh is None:
            print("[warn] inst_score_combine='learned' but model has no "
                  "merge_head \u2192 falling back to 'add'.")
            _sc = 'add'
        return OpenSe3rQueryMemory(
            cfg                  = cfg,
            num_frames           = T,
            H = H, W = W,
            sem_feat_dim         = sem_feat_dim,
            feat_storage         = self.feat_storage,
            sem_voxel_size       = float(cfg.semantic_voxel_size),
            instance_voxel_size  = self.instance_voxel_size,
            knn_radius           = self.knn_radius,
            max_candidates       = int(getattr(self, 'max_candidates', 32)),
            match_threshold      = self.match_threshold,
            embed_update         = getattr(self, 'embed_update', 'mean'),
            embed_momentum       = getattr(self, 'embed_momentum', 0.75),
            match_metric         = getattr(self, 'match_metric', 'hungarian'),
            match_mode           = getattr(self, 'match_mode', 'hungarian'),
            gate_overlap         = getattr(self, 'gate_overlap', 0.10),
            merge_desc_high      = getattr(self, 'merge_desc_high', 0.75),
            spawn_desc_low       = getattr(self, 'spawn_desc_low', 0.40),
            greedy_gate_use_observed = getattr(self, 'greedy_gate_use_observed', False),
            w_desc               = self.w_desc,
            w_sem                = self.w_sem,
            w_vox                = self.w_vox,
            w_geo                = self.w_geo,
            sigma_dist           = self.sigma_dist,
            obj_threshold        = self.obj_threshold,
            obj_top_k            = self.obj_top_k,
            min_points           = self.min_points,
            nms_iou              = self.nms_iou,
            score_threshold      = self.score_threshold,
            outlier_iqr_mult     = self.outlier_iqr_mult,
            nms_mode             = getattr(self, 'nms_mode', 'hard'),
            matrix_nms_kernel    = getattr(self, 'matrix_nms_kernel', 'linear'),
            panoptic             = getattr(self, 'panoptic', True),
            sp_score_thr         = getattr(self, 'sp_score_thr', 0.5),
            obj_normalize        = getattr(self, 'obj_normalize', False),
            use_sem_conf         = getattr(self, 'use_sem_conf', False),
            use_sem_conf_gate    = getattr(self, 'use_sem_conf_gate', False),
            use_sem_conf_select  = getattr(self, "use_sem_conf_select", False),
            vox_gate             = self.vox_gate,
            desc_gate            = self.desc_gate,
            score_combine        = _sc,
            box_iou_mode         = self.box_iou_mode,
            merge_head           = _mh,
            sem_gate_thr         = self.sem_gate_thr,
            vox_bonus_alpha      = self.vox_bonus_alpha,
            desc_pow             = self.desc_pow,
            vox_or_tau           = self.vox_or_tau,
            backend              = self.backend,
            is_metric            = False,           # GT mode: GT geometry directly
            point_conf_threshold = self.point_conf_threshold,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Flat KV cache + GT
# ─────────────────────────────────────────────────────────────────────────────

class OpenSe3rQuerySLAM_GT(OpenSe3rSLAM_GT, _InstanceKnobMixin):
    """GT-mode (no coordinate alignment) SLAM with query head."""

    def __init__(self, model, *,
                 cfg_path:             str = './slam_opense3r/config.yaml',
                 sem_feat_mode:        str = 'expanded',
                 feat_storage:         str = 'voxel',
                 kv_window_size:       Optional[int] = None,
                 kv_keyframe_stride:   Optional[int] = None,
                 backend:              str = 'vggt',
                 point_conf_threshold: Optional[float] = None,
                 attn_mode:            str = 'causal',
                 global_scale_mode:    str = 'per_chunk',
                 no_cross_chunk_kv:    bool = False,
                 gt_geo_input:         bool = False,
                 **inst_knobs):
        super().__init__(
            model                = model,
            cfg_path             = cfg_path,
            sem_feat_mode        = sem_feat_mode,
            feat_storage         = feat_storage,
            kv_window_size       = kv_window_size,
            kv_keyframe_stride   = kv_keyframe_stride,
            backend              = backend,
            point_conf_threshold = point_conf_threshold,
            attn_mode            = attn_mode,
        )
        self._store_inst_knobs(**inst_knobs)
        # When True: reset the KV cache to empty before EVERY chunk → each chunk
        # attends only within itself (no cross-chunk memory). Ablation to test
        # whether the persistent KV memory actually helps.
        self.no_cross_chunk_kv = bool(no_cross_chunk_kv)
        # mode B: feed GT depth/world_points/pose/intrinsics into the model's
        # conditioning via forward_semantic (skip DA3). Matches geo_source=gt.
        self.gt_geo_input = bool(gt_geo_input)

        self.global_scale_mode = _validate_global_scale_mode(global_scale_mode)
        # Cache slot for sequence-global scale.  Populated after the first
        # chunk by `_maybe_capture_seq_box_scale` ('first_chunk' mode) or
        # pre-populated by `run()` for 'gt_seq_median'.  None ⇒ no override.
        self._seq_box_scale: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    def run(self, images: torch.Tensor, pts_gt: torch.Tensor,
            poses_gt: torch.Tensor,
            depth_gt: Optional[torch.Tensor] = None,
            intrinsics_gt: Optional[torch.Tensor] = None) -> OpenSe3rQueryMemory:
        from .pipeline import _prune_kv_cache    # local import to avoid cycle
        assert images.shape[0] == 1
        assert pts_gt.shape[0] == images.shape[1]
        assert poses_gt.shape[0] == images.shape[1]
        if self.gt_geo_input:
            assert depth_gt is not None and intrinsics_gt is not None, \
                "gt_geo_input requires depth_gt + intrinsics_gt"

        cfg = self.cfg
        device = cfg.device
        chunk_size = int(cfg.chunk_size)
        _, T, _, H, W = images.shape

        patch_size = getattr(self.model.causal_branch, 'patch_size', 14)
        N_patch    = (H // patch_size) * (W // patch_size)

        pose_anchor_inv = torch.inverse(poses_gt[0].float())
        # Sequence-global box_scale from GT (gt_geo_input forces this mode).
        if self.global_scale_mode == 'gt_seq_median':
            self._seq_box_scale = _compute_gt_seq_box_scale(pts_gt, poses_gt)
            print(f"[global_scale_mode=gt_seq_median] "
                  f"box_scale={self._seq_box_scale.tolist()}")

        # Probe — query mode: no `instance_feat`, only sem dim is needed.
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if self.gt_geo_input:
                probe = _forward_semantic_gt(
                    self.model, images[:, :min(chunk_size, T)].to(device),
                    self.model.init_kv_cache(),
                    chunk_start=0, chunk_end=min(chunk_size, T),
                    pts_gt=pts_gt, poses_gt=poses_gt,
                    depth_gt=depth_gt, intrinsics_gt=intrinsics_gt,
                    pose_anchor_inv=pose_anchor_inv,
                    box_scale_override=_current_box_scale_override(self, device),
                    attn_mode=self.attn_mode, device=device)
            else:
                probe = self.model.forward_chunk(
                    {'images': images[:, :min(chunk_size, T)].to(device)},
                    kv_cache_list = self.model.init_kv_cache(),
                    mode          = self.attn_mode,
                )
        if 'mask_logits' not in probe:
            raise RuntimeError(
                "OpenSe3rQuerySLAM_GT needs a query-head model; probe "
                "returned no 'mask_logits'.  Use OpenSe3rSLAM_GT for "
                "dense-head models."
            )
        sem_key      = self._sem_feat_key()
        sem_feat_dim = probe.get(sem_key, probe['semantic_feat']).shape[2]

        memory = self._build_query_memory(T, H, W, sem_feat_dim)
        print(f"[QuerySLAM-GT] instance_table: "
              f"knn={self.knn_radius}m threshold={self.match_threshold} "
              f"weights desc/sem/vox/geo="
              f"{self.w_desc}/{self.w_sem}/{self.w_vox}/{self.w_geo}"
              + ("  [gt_geo_input]" if self.gt_geo_input else ""))

        kv_cache = self.model.init_kv_cache()
        t0 = time.time()

        for chunk_start in range(0, T, chunk_size):
            chunk_end  = min(chunk_start + chunk_size, T)
            chunk_imgs = images[:, chunk_start:chunk_end].to(device)
            frame_idx  = torch.arange(chunk_start, chunk_end)

            # mode A geometry prep (fp32, OUTSIDE autocast — preserves the exact
            # released numerics). Skipped for gt_geo_input (mode B builds its
            # geometry inside autocast, matching --geo_source=gt training).
            if not self.gt_geo_input:
                chunk_c2w_anchored = pose_anchor_inv @ poses_gt[chunk_start:chunk_end].float()
                pts_chunk_anchored = _anchor_pts_to_frame0(
                    pts_gt[chunk_start:chunk_end].float(), pose_anchor_inv)

            # Ablation: drop all cross-chunk memory → within-chunk attention only.
            if self.no_cross_chunk_kv:
                kv_cache = self.model.init_kv_cache()

            with mem_stats.GpuPeakTracker(self._mem_peaks, 'sem'), \
                    torch.no_grad(), \
                    torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if self.gt_geo_input:
                    # mode B: GT depth + world_points → forward_semantic (no DA3).
                    out = _forward_semantic_gt(
                        self.model, chunk_imgs, kv_cache,
                        chunk_start=chunk_start, chunk_end=chunk_end,
                        pts_gt=pts_gt, poses_gt=poses_gt,
                        depth_gt=depth_gt, intrinsics_gt=intrinsics_gt,
                        pose_anchor_inv=pose_anchor_inv,
                        box_scale_override=_current_box_scale_override(self, device),
                        attn_mode=self.attn_mode, device=device)
                else:
                    # mode A: run DA3, override world_points / poses with GT.
                    out = self.model.forward_chunk(
                        {'images': chunk_imgs},
                        kv_cache_list         = kv_cache,
                        return_poses          = False,
                        mode                  = self.attn_mode,
                        extr_rel_override     = chunk_c2w_anchored.unsqueeze(0),
                        world_points_override = pts_chunk_anchored.unsqueeze(0).to(device),
                        box_scale_override    = _current_box_scale_override(self, device),
                    )
            # Capture box_scale from the very first chunk (mode 'first_chunk').
            if chunk_start == 0:
                _maybe_capture_seq_box_scale(self, out)
            kv_cache = out['kv_cache']
            if self.kv_window_size is not None or self.kv_keyframe_stride is not None:
                kv_cache = _prune_kv_cache(
                    kv_cache, N_patch,
                    window_size     = self.kv_window_size,
                    keyframe_stride = self.kv_keyframe_stride,
                )
            self._last_kv_cache = kv_cache

            # Optional 2D instance-mask debug dump (top-K objectness per frame).
            self._maybe_dump_2d_masks(chunk_imgs, out, frame_start=chunk_start)

            # GT geometry for this chunk.
            pts_chunk   = pts_gt  [chunk_start:chunk_end].float()
            poses_chunk = poses_gt[chunk_start:chunk_end].float()
            conf_chunk  = (pts_chunk.norm(dim=-1) > 1e-3).float()

            # Stage pts/conf for update_instances (it reads _pending_pts_new).
            memory._pending_pts_new  = pts_chunk
            memory._pending_conf_new = conf_chunk
            memory.update_instances(out, chunk_start, chunk_end - 1,
                                      sem_feat_key=sem_key)
            memory._pending_pts_new  = None
            memory._pending_conf_new = None

            # Push GT geometry + sem (ins_feat / ins_conf = None → query memory
            # skips dense ins push automatically because ins_store is None).
            sem_feat = out.get(sem_key, out['semantic_feat'])[0].cpu().float()
            sem_conf = out['semantic_conf'][0].cpu().float()
            memory.update_gt(
                frame_indices = frame_idx,
                pts           = pts_chunk,
                poses         = poses_chunk,
                conf          = conf_chunk,
                sem_feat      = sem_feat,
                sem_conf      = sem_conf,
                ins_feat      = None,
                ins_conf      = None,
            )

            elapsed = time.time() - t0
            fps     = chunk_end / elapsed
            storage_label = 'voxels' if self.feat_storage == 'voxel' else 'pts'
            print(f"[QuerySLAM-GT] frames {chunk_end:4d}/{T}  "
                  f"sem {storage_label}: {memory.sem_count:,}  "
                  f"instances: {memory.instance_count}  "
                  f"{fps:.1f} fps")
            print(mem_stats.format_mem_log(
                memory,
                peaks         = self._mem_peaks,
                kv_cache_list = self._last_kv_cache,
            ))

        return memory


# ─────────────────────────────────────────────────────────────────────────────
# 3-tier voxel KV + GT
# ─────────────────────────────────────────────────────────────────────────────

class OpenSe3rQuerySLAM_VoxelKV_GT(OpenSe3rSLAM_VoxelKV_GT, _InstanceKnobMixin):
    """GT-mode SLAM with 3-tier voxel KV memory + query head."""

    def __init__(self, model, *,
                 cfg_path:              str = './slam_opense3r/config.yaml',
                 sem_feat_mode:         str = 'expanded',
                 feat_storage:          str = 'voxel',
                 kv_window_size:        Optional[int] = None,
                 kv_keyframe_stride:    Optional[int] = None,
                 backend:               str = 'vggt',
                 point_conf_threshold:  Optional[float] = None,
                 attn_mode:             str = 'causal',
                 global_scale_mode:     str = 'per_chunk',
                 gt_geo_input:          bool = False,
                 voxel_kv_cfg:          Optional[VoxelKVConfig] = None,
                 **inst_knobs):
        if voxel_kv_cfg is None:
            voxel_kv_cfg = VoxelKVConfig()
        if voxel_kv_cfg.pivot_signal == 'instance_feat':
            print("[VoxelKV-Query-GT] pivot_signal='instance_feat' invalid in "
                  "query-head mode (dense ins feat is None) — switching to 'V'")
            voxel_kv_cfg.pivot_signal = 'V'
        super().__init__(
            model                = model,
            cfg_path             = cfg_path,
            sem_feat_mode        = sem_feat_mode,
            feat_storage         = feat_storage,
            kv_window_size       = kv_window_size,
            kv_keyframe_stride   = kv_keyframe_stride,
            backend              = backend,
            point_conf_threshold = point_conf_threshold,
            attn_mode            = attn_mode,
            voxel_kv_cfg         = voxel_kv_cfg,
        )
        self._store_inst_knobs(**inst_knobs)

        self.global_scale_mode = _validate_global_scale_mode(global_scale_mode)
        self._seq_box_scale: Optional[torch.Tensor] = None
        self.gt_geo_input = bool(gt_geo_input)

    # ------------------------------------------------------------------
    def run(self, images: torch.Tensor, pts_gt: torch.Tensor,
            poses_gt: torch.Tensor,
            depth_gt: Optional[torch.Tensor] = None,
            intrinsics_gt: Optional[torch.Tensor] = None) -> OpenSe3rQueryMemory:
        assert images.shape[0] == 1
        assert pts_gt.shape[0] == images.shape[1]
        assert poses_gt.shape[0] == images.shape[1]
        if self.gt_geo_input:
            assert depth_gt is not None and intrinsics_gt is not None, \
                "gt_geo_input requires depth_gt + intrinsics_gt"

        cfg = self.cfg
        device = cfg.device
        chunk_size = int(cfg.chunk_size)
        _, T, _, H, W = images.shape
        patch_size = getattr(self.model.causal_branch, 'patch_size', 14)

        pose_anchor_inv = torch.inverse(poses_gt[0].float())
        # Sequence-global box_scale from GT (gt_geo_input forces this mode).
        if self.global_scale_mode == 'gt_seq_median':
            self._seq_box_scale = _compute_gt_seq_box_scale(pts_gt, poses_gt)
            print(f"[global_scale_mode=gt_seq_median] "
                  f"box_scale={self._seq_box_scale.tolist()}")

        # Probe (query mode: no instance_feat).
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if self.gt_geo_input:
                probe = _forward_semantic_gt(
                    self.model, images[:, :min(chunk_size, T)].to(device),
                    self.model.init_kv_cache(),
                    chunk_start=0, chunk_end=min(chunk_size, T),
                    pts_gt=pts_gt, poses_gt=poses_gt,
                    depth_gt=depth_gt, intrinsics_gt=intrinsics_gt,
                    pose_anchor_inv=pose_anchor_inv,
                    box_scale_override=_current_box_scale_override(self, device),
                    attn_mode=self.attn_mode, device=device)
            else:
                probe = self.model.forward_chunk(
                    {'images': images[:, :min(chunk_size, T)].to(device)},
                    kv_cache_list = self.model.init_kv_cache(),
                    mode          = self.attn_mode,
                )
        if 'mask_logits' not in probe:
            raise RuntimeError(
                "OpenSe3rQuerySLAM_VoxelKV_GT needs a query-head model."
            )
        sem_key      = self._sem_feat_key()
        sem_feat_dim = probe.get(sem_key, probe['semantic_feat']).shape[2]

        memory = self._build_query_memory(T, H, W, sem_feat_dim)
        print(f"[QuerySLAM-VoxelKV-GT] instance_table: "
              f"knn={self.knn_radius}m threshold={self.match_threshold} "
              f"weights desc/sem/vox/geo="
              f"{self.w_desc}/{self.w_sem}/{self.w_vox}/{self.w_geo}"
              + ("  [gt_geo_input]" if self.gt_geo_input else ""))

        # Manager init (lazy build from parent class)
        self._ensure_manager()
        mgr = self.voxel_kv_manager
        mgr.reset()
        self._kf_indices = []

        kf_threshold    = float(cfg.keyframe_threshold)
        t0 = time.time()

        for chunk_start in range(0, T, chunk_size):
            chunk_end  = min(chunk_start + chunk_size, T)
            chunk_imgs = images[:, chunk_start:chunk_end].to(device)
            frame_idx  = torch.arange(chunk_start, chunk_end)
            T_c        = chunk_end - chunk_start

            # Keyframe selection (inherits parent logic via inlined call).
            is_kf_mask = torch.zeros(T_c, dtype=torch.bool)
            for i, fi in enumerate(frame_idx.tolist()):
                if self._is_new_keyframe(poses_gt[fi].float(), poses_gt.float(),
                                          kf_threshold):
                    is_kf_mask[i] = True
                    self._kf_indices.append(int(fi))

            # Per-patch GT positions for voxel retrieval (parent helper).
            wp = pts_gt[chunk_start:chunk_end].unsqueeze(0).to(device).float()
            new_positions, N_patch = self._per_patch_world_points(wp, patch_size)

            # mode A geometry prep (fp32, OUTSIDE autocast — exact numerics).
            if not self.gt_geo_input:
                chunk_c2w_anchored = pose_anchor_inv @ poses_gt[chunk_start:chunk_end].float()
                pts_chunk_anchored = _anchor_pts_to_frame0(
                    pts_gt[chunk_start:chunk_end].float(), pose_anchor_inv)

            kv_cache_list, n_aw, n_ret = mgr.materialize_kv_cache(
                new_chunk_positions = new_positions,
                batch_size          = images.shape[0],
            )
            if all(k is None for k, _ in kv_cache_list):
                kv_cache_list = self.model.init_kv_cache()

            with mem_stats.GpuPeakTracker(self._mem_peaks, 'sem'), \
                    torch.no_grad(), \
                    torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if self.gt_geo_input:
                    # mode B: GT depth + world_points → forward_semantic (no DA3).
                    out = _forward_semantic_gt(
                        self.model, chunk_imgs, kv_cache_list,
                        chunk_start=chunk_start, chunk_end=chunk_end,
                        pts_gt=pts_gt, poses_gt=poses_gt,
                        depth_gt=depth_gt, intrinsics_gt=intrinsics_gt,
                        pose_anchor_inv=pose_anchor_inv,
                        box_scale_override=_current_box_scale_override(self, device),
                        attn_mode=self.attn_mode, device=device)
                else:
                    # mode A: run DA3, override world_points / poses with GT.
                    out = self.model.forward_chunk(
                        {'images': chunk_imgs},
                        kv_cache_list         = kv_cache_list,
                        return_poses          = False,
                        mode                  = self.attn_mode,
                        extr_rel_override     = chunk_c2w_anchored.unsqueeze(0),
                        world_points_override = pts_chunk_anchored.unsqueeze(0).to(device),
                        box_scale_override    = _current_box_scale_override(self, device),
                    )
            if chunk_start == 0:
                _maybe_capture_seq_box_scale(self, out)
            new_kv = out.pop('kv_cache', None)
            # KV is held by the manager; format_mem_log will read its breakdown.
            self._last_kv_cache = None

            # Query mode: no dense instance_feat → dummy tensor (manager
            # pivots on V instead).
            if out.get('instance_feat') is not None:
                new_instance_feat = self._instance_feat_to_patches(
                    out['instance_feat'], patch_size,
                )
            else:
                new_instance_feat = torch.zeros(
                    images.shape[0] * T_c * N_patch, 1,
                    device=device, dtype=torch.float32,
                )

            K_list = [layer[0] for layer in new_kv]
            V_list = [layer[1] for layer in new_kv]
            mgr.absorb_chunk(
                new_chunk_K_per_layer       = K_list,
                new_chunk_V_per_layer       = V_list,
                T_cache_anchor_window       = n_aw,
                n_retrieved_tokens          = n_ret,
                new_chunk_positions         = new_positions,
                new_chunk_instance_feat     = new_instance_feat,
                new_chunk_frame_indices     = frame_idx,
                new_chunk_is_keyframe_mask  = is_kf_mask,
                current_keyframes           = self._kf_indices,
                N_patch                     = N_patch,
                batch                       = 0,
            )
            mgr.sync_anchor_with_keyframes(self._kf_indices)

            # Optional 2D instance-mask debug dump (top-K objectness per frame).
            self._maybe_dump_2d_masks(chunk_imgs, out, frame_start=chunk_start)

            # GT geometry + sem push + instance update.
            pts_chunk   = pts_gt  [chunk_start:chunk_end].float()
            poses_chunk = poses_gt[chunk_start:chunk_end].float()
            conf_chunk  = (pts_chunk.norm(dim=-1) > 1e-3).float()

            memory._pending_pts_new  = pts_chunk
            memory._pending_conf_new = conf_chunk
            memory.update_instances(out, chunk_start, chunk_end - 1,
                                      sem_feat_key=sem_key)
            memory._pending_pts_new  = None
            memory._pending_conf_new = None

            sem_feat = out.get(sem_key, out['semantic_feat'])[0].cpu().float()
            sem_conf = out['semantic_conf'][0].cpu().float()
            memory.update_gt(
                frame_indices = frame_idx,
                pts           = pts_chunk,
                poses         = poses_chunk,
                conf          = conf_chunk,
                sem_feat      = sem_feat,
                sem_conf      = sem_conf,
                ins_feat      = None,
                ins_conf      = None,
            )

            elapsed = time.time() - t0
            fps     = chunk_end / elapsed
            diag    = mgr.diagnostics()
            storage_label = 'voxels' if self.feat_storage == 'voxel' else 'pts'
            print(f"[QuerySLAM-VoxelKV-GT] frames {chunk_end:4d}/{T}  "
                  f"sem {storage_label}: {memory.sem_count:,}  "
                  f"instances: {memory.instance_count}  "
                  f"kf={len(self._kf_indices)}  "
                  f"anchor={diag['anchor_pool_size']}+first={diag['first_frame_pinned']} "
                  f"voxels={diag['voxel_count']} pivots={diag['voxel_pivots']} "
                  f"ret={n_ret}  {fps:.1f} fps")
            print(mem_stats.format_mem_log(
                memory,
                peaks            = self._mem_peaks,
                voxel_kv_manager = mgr,
            ))

        return memory
