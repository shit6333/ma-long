"""
slam_opense3r/pipeline_query.py
================================
SLAM pipeline for the query-based instance head.

Subclasses `OpenSe3rSLAM` so geometry / sem-voxel / KV-cache logic stays
identical.  Three points of departure:

  - Probes only sem_feat dim from the test forward (no `instance_feat` —
    it's `None` in query mode).
  - Builds `OpenSe3rQueryMemory` (sem voxel store + InstanceTable).
  - In each mapping step, calls `memory.update_instances(...)` between
    `update_geo` and `update_sem` so:
        update_geo  → fills `_pending_pts_new`
        update_instances → reads `_pending_pts_new`, does NOT consume it
        update_sem  → reads + consumes `_pending_pts_new`

The legacy `OpenSe3rSLAM` is left fully intact; existing scripts that
import it continue to work unchanged.
"""

from __future__ import annotations
from typing import Optional, List
import time

import torch

from .pipeline      import OpenSe3rSLAM, _prune_kv_cache
from .memory_query  import OpenSe3rQueryMemory
from .              import mem_stats


class OpenSe3rQuerySLAM(OpenSe3rSLAM):
    """OpenSe3rSLAM variant whose instance store is an `InstanceTable`."""

    def __init__(
        self,
        model,
        cfg_path:              str   = './slam_opense3r/config.yaml',
        sem_feat_mode:         str   = 'expanded',
        feat_storage:          str   = 'voxel',
        kv_window_size:        Optional[int] = None,
        kv_keyframe_stride:    Optional[int] = None,
        backend:               str   = 'vggt',
        point_conf_threshold:  Optional[float] = None,
        attn_mode:             str   = 'causal',
        global_scale_mode:     str   = 'per_chunk',
        # InstanceTable knobs (CLI-overridable, sensible defaults)
        instance_voxel_size:   float = 0.05,
        knn_radius:            float = 2.0,
        max_candidates:        int   = 32,       # cap KNN candidate shortlist for matching
        match_threshold:       float = 0.35,
        w_desc:                float = 0.35,
        w_sem:                 float = 0.35,
        w_vox:                 float = 0.15,
        w_geo:                 float = 0.15,
        sigma_dist:            float = 0.5,
        obj_threshold:         float = 0.5,
        obj_top_k:             int   = 0,        # 0 = off (threshold-only)
        nms_iou:               float = 0.5,
        score_threshold:       float = 0.25,
        outlier_iqr_mult:      float = 1.5,
        min_points:            int   = 0,        # drop proposals < N 3D points (0 = off)
        # Multi-gate overrides (None = off → linear formula).
        vox_gate:              Optional[float] = None,
        desc_gate:             Optional[float] = None,
        # Cue fusion + soft semantic gate (0 = off) + new-mode knobs.
        score_combine:         str   = 'add',
        box_iou_mode:          str   = 'min',
        sem_gate_thr:          float = 0.0,
        vox_bonus_alpha:       float = 1.0,
        desc_pow:              float = 1.0,
        vox_or_tau:            float = 0.5,
    ):
        # 'gt_seq_median' only meaningful in the GT pipeline — silently
        # downgrade here so users mixing modes don't get a surprise crash.
        if global_scale_mode == 'gt_seq_median':
            print("[OpenSe3rQuerySLAM] WARN: --global_scale_mode=gt_seq_median "
                  "requires --use_gt; falling back to 'first_chunk' here.")
            global_scale_mode = 'first_chunk'

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
            global_scale_mode    = global_scale_mode,
        )
        self.instance_voxel_size = float(instance_voxel_size)
        self.knn_radius          = float(knn_radius)
        self.max_candidates      = int(max_candidates)
        self.match_threshold     = float(match_threshold)
        self.w_desc              = float(w_desc)
        self.w_sem               = float(w_sem)
        self.w_vox               = float(w_vox)
        self.w_geo               = float(w_geo)
        self.sigma_dist          = float(sigma_dist)
        self.obj_threshold       = float(obj_threshold)
        self.obj_top_k           = int(obj_top_k)
        self.nms_iou             = float(nms_iou)
        self.score_threshold     = float(score_threshold)
        self.outlier_iqr_mult    = float(outlier_iqr_mult)
        self.min_points          = int(min_points)
        self.vox_gate            = None if vox_gate  is None else float(vox_gate)
        self.desc_gate           = None if desc_gate is None else float(desc_gate)
        self.score_combine       = str(score_combine)
        self.box_iou_mode        = str(box_iou_mode)
        self.sem_gate_thr        = float(sem_gate_thr)
        self.vox_bonus_alpha     = float(vox_bonus_alpha)
        self.desc_pow            = float(desc_pow)
        self.vox_or_tau          = float(vox_or_tau)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _build_memory(self, T: int, H: int, W: int,
                      sem_feat_dim: int, is_metric: bool) -> OpenSe3rQueryMemory:
        """Factory for the per-run memory.  Replaces the legacy
        `OpenSe3rMemory(...)` call inside `run()`."""
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
            max_candidates       = self.max_candidates,
            match_threshold      = self.match_threshold,
            embed_update         = getattr(self, 'embed_update', 'mean'),
            embed_momentum       = getattr(self, 'embed_momentum', 0.75),
            match_metric         = getattr(self, 'match_metric', 'hungarian'),
            w_desc               = self.w_desc,
            w_sem                = self.w_sem,
            w_vox                = self.w_vox,
            w_geo                = self.w_geo,
            sigma_dist           = self.sigma_dist,
            obj_threshold        = self.obj_threshold,
            obj_top_k            = self.obj_top_k,
            nms_iou              = self.nms_iou,
            score_threshold      = self.score_threshold,
            outlier_iqr_mult     = self.outlier_iqr_mult,
            min_points           = self.min_points,
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
            is_metric            = is_metric,
            point_conf_threshold = self.point_conf_threshold,
            use_sem_conf         = getattr(self, 'use_sem_conf', False),
            use_sem_conf_gate    = getattr(self, 'use_sem_conf_gate', False),
            use_sem_conf_select  = getattr(self, "use_sem_conf_select", False),
        )

    # ------------------------------------------------------------------
    # Optional debug: per-frame 2D instance-mask grids (top-K objectness).
    # Controlled by `save_2d_masks` (default off), set by run.py from
    # `--save_2d_inst_masks`.  No-op unless explicitly enabled.
    # ------------------------------------------------------------------

    def _maybe_dump_2d_masks(self, imgs: torch.Tensor, sem_res: dict,
                             frame_start: int) -> None:
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

    # ------------------------------------------------------------------
    # Step functions — geo + sem path is identical, slot update_instances in
    # the middle (after geo's pts-cache, before sem consumes it).
    # ------------------------------------------------------------------

    def _initialize_map(self, images: torch.Tensor, num_init: int,
                        memory: OpenSe3rQueryMemory, sem_feat_key: str) -> list:
        """Init like base class but also push the first chunk's proposals
        into the instance_table.  We replicate base behaviour rather than
        super().__call__() so the order (geo → instances → sem) is correct."""
        device    = self.cfg.device
        init_imgs = images[:, :num_init].to(device)

        # Optional depth / K input for multi-modal backbones (e.g. MapAnything).
        init_depth   = self._slice_depth_z(0, num_init - 1, device)
        init_intrins = self._slice_intrinsics(0, num_init - 1, device)

        # Geometry on init frames (= SLAM-global frame by definition).
        geo_res  = self._geo_forward(init_imgs,
                                     depth_z=init_depth,
                                     intrinsics=init_intrins)

        # CausalBranch from scratch.  Init frames live in the SLAM-global
        # frame by definition, so geo_res values are already in that frame —
        # take the fast path (`forward_semantic`) which skips a redundant
        # DA3 / VGGT geometry pass.
        kv_cache   = self.model.init_kv_cache()
        sem_res, kv_cache = self._sem_forward(
            init_imgs, kv_cache,
            geo                 = {
                'depth':             geo_res['depth'],
                'intrinsics':        geo_res['intrinsics'],
                'world_points_conf': geo_res['world_points_conf'],
            },
            c2w_global          = geo_res['poses'],
            world_points_global = geo_res['world_points'],
            box_scale_override  = self._current_box_scale_override(device),
        )

        # Optional 2D instance-mask debug dump (top-K objectness per frame).
        self._maybe_dump_2d_masks(init_imgs, sem_res, frame_start=0)

        # Stash box_scale for sequence-global scale modes (no-op for
        # 'per_chunk').  Must run AFTER the first sem_forward.
        self._capture_seq_box_scale(sem_res)

        if self.kv_window_size is not None or self.kv_keyframe_stride is not None:
            patch_size = getattr(self.model.causal_branch, 'patch_size', 14)
            H = images.shape[-2]; W = images.shape[-1]
            N_patch = (H // patch_size) * (W // patch_size)
            kv_cache = _prune_kv_cache(
                kv_cache, N_patch,
                window_size     = self.kv_window_size,
                keyframe_stride = self.kv_keyframe_stride,
            )

        # `memory.initialize()` runs first-batch geo store + initial sem push.
        # Internally it calls `_push_sem_ins` which we have overridden to
        # only push sem (no ins_feat).  After that, pts/conf for the init
        # frames are still on `self._pending_pts_new`-equivalent (set in
        # initialize()), but `initialize()` does NOT actually populate
        # _pending_*; it calls _push_sem_ins directly.
        memory.initialize(geo_res, sem_res, sem_feat_key)

        # For instance update we need pts/conf in global frame.  Init frames
        # ARE the global frame by definition, so we can re-derive from
        # geo_res directly without going through coordinate_alignment.
        pts_new  = geo_res['world_points'][0].cpu().float()                  # [T,H,W,3]
        conf_new = memory._conf_sig(
            geo_res['world_points_conf'][0].cpu().squeeze(-1)
        )
        memory._pending_pts_new  = pts_new
        memory._pending_conf_new = conf_new
        memory.update_instances(sem_res, start_idx=0, end_idx=num_init - 1,
                                  sem_feat_key=sem_feat_key)
        memory._pending_pts_new  = None
        memory._pending_conf_new = None

        avg_conf = geo_res['world_points_conf'][0].squeeze(-1).mean().item()
        print(f"[init]  {num_init} frames  avg_conf={avg_conf:.3f}"
              f"  kf={memory.cur_kf_idx.tolist()}"
              f"  instances={memory.instance_count}")
        return kv_cache

    def _mapping_step(self, images: torch.Tensor,
                      start_idx: int, end_idx: int,
                      memory: OpenSe3rQueryMemory, sem_feat_key: str,
                      kv_cache: list) -> list:
        """One incremental map step. Adds `update_instances` between
        `update_geo` and `update_sem` (which consumes the pts-new cache)."""
        device    = self.cfg.device
        cur_kf    = memory.cur_kf_idx
        n_kf      = int(cur_kf.numel())                      # captured BEFORE update_geo

        # ── Geometry: keyframes + new frames ────────────────────────────────
        kf_imgs  = images[0, cur_kf].unsqueeze(0).to(device)
        new_imgs = images[:, start_idx:end_idx + 1].to(device)
        geo_imgs = torch.cat([kf_imgs, new_imgs], dim=1)

        # Multi-modal depth / K — concat the keyframe + new-frame slices in
        # the same order as `geo_imgs` so per-view inputs to MapAnything
        # line up with the per-view RGB.  When no depth source is wired,
        # both slices return None and `_geo_forward` runs image-only.
        kf_depth   = self._slice_depth_z_by_idx(cur_kf, device)
        new_depth  = self._slice_depth_z(start_idx, end_idx, device)
        geo_depth  = self._cat_optional(kf_depth, new_depth, dim=1)
        kf_intr    = self._slice_intrinsics_by_idx(cur_kf, device)
        new_intr   = self._slice_intrinsics(start_idx, end_idx, device)
        geo_intr   = self._cat_optional(kf_intr, new_intr, dim=1)
        geo_res  = self._geo_forward(geo_imgs,
                                     depth_z=geo_depth,
                                     intrinsics=geo_intr)

        # ── Geo-only memory update (alignment + cache for sem/inst push) ─────
        memory.update_geo(geo_res, start_idx, end_idx)

        # ── SLAM-global geo for the new frames (sem fast path) ───────────────
        # The DA3 / VGGT geometry has already been run once for chunk-level
        # alignment; feed the new-frames slice to `forward_semantic` to skip
        # the second pass.  memory.poses + memory._pending_pts_new are in
        # the SLAM-global frame after update_geo.
        new_c2w_global = memory.poses[start_idx:end_idx + 1].unsqueeze(0).float()
        new_wp_global  = memory._pending_pts_new.unsqueeze(0).float().to(device)  # [1, n_new, H, W, 3]
        new_geo = {
            'depth':             geo_res['depth'             ][:, n_kf:],
            'intrinsics':        geo_res['intrinsics'        ][:, n_kf:],
            'world_points_conf': geo_res['world_points_conf' ][:, n_kf:],
        }

        # ── Semantics forward (new frames only, fast path) ───────────────────
        sem_res, kv_cache = self._sem_forward(
            new_imgs, kv_cache,
            geo                 = new_geo,
            c2w_global          = new_c2w_global,
            world_points_global = new_wp_global,
            box_scale_override  = self._current_box_scale_override(device),
        )

        if self.kv_window_size is not None or self.kv_keyframe_stride is not None:
            patch_size = getattr(self.model.causal_branch, 'patch_size', 14)
            H = images.shape[-2]; W = images.shape[-1]
            N_patch = (H // patch_size) * (W // patch_size)
            kv_cache = _prune_kv_cache(
                kv_cache, N_patch,
                window_size     = self.kv_window_size,
                keyframe_stride = self.kv_keyframe_stride,
            )

        # Optional 2D instance-mask debug dump (top-K objectness per frame).
        self._maybe_dump_2d_masks(new_imgs, sem_res, frame_start=start_idx)

        # ── Instance update (must come BEFORE update_sem so the pts-new
        # cache is still alive) ───────────────────────────────────────────────
        memory.update_instances(sem_res, start_idx, end_idx,
                                  sem_feat_key=sem_feat_key)

        # ── Sem push (consumes the cache) ────────────────────────────────────
        memory.update_sem(sem_res, sem_feat_key)

        return kv_cache

    # ------------------------------------------------------------------
    # Optional depth / intrinsics helpers (multi-modal backbones, e.g. MapAnything).
    # ------------------------------------------------------------------

    def _slice_depth_z(self,
                       start_idx: int, end_idx: int,
                       device) -> Optional[torch.Tensor]:
        """[start_idx, end_idx] inclusive → depth_z slice [1, T, H, W] or None."""
        if self._depth_z_seq is None:
            return None
        return self._depth_z_seq[start_idx:end_idx + 1].unsqueeze(0).to(device)

    def _slice_depth_z_by_idx(self,
                              indices: torch.Tensor,
                              device) -> Optional[torch.Tensor]:
        if self._depth_z_seq is None:
            return None
        return self._depth_z_seq[indices].unsqueeze(0).to(device)

    def _slice_intrinsics(self,
                          start_idx: int, end_idx: int,
                          device) -> Optional[torch.Tensor]:
        if self._intrinsics_seq is None:
            return None
        return self._intrinsics_seq[start_idx:end_idx + 1].unsqueeze(0).to(device)

    def _slice_intrinsics_by_idx(self,
                                 indices: torch.Tensor,
                                 device) -> Optional[torch.Tensor]:
        if self._intrinsics_seq is None:
            return None
        return self._intrinsics_seq[indices].unsqueeze(0).to(device)

    @staticmethod
    def _cat_optional(a: Optional[torch.Tensor],
                      b: Optional[torch.Tensor],
                      dim: int) -> Optional[torch.Tensor]:
        if a is None and b is None:
            return None
        if a is None or b is None:
            raise RuntimeError(
                "_cat_optional: one side is None while the other is not "
                "— depth/K wiring inconsistent."
            )
        return torch.cat([a, b], dim=dim)

    # ------------------------------------------------------------------
    # run() — identical structure to base but builds the query memory and
    # skips `ins_feat_dim` probing.
    # ------------------------------------------------------------------

    def run(self,
            images: torch.Tensor,
            depth_z_input:    Optional[torch.Tensor] = None,
            intrinsics_input: Optional[torch.Tensor] = None,
            ) -> OpenSe3rQueryMemory:
        """SLAM run.  `depth_z_input` / `intrinsics_input` (when given) are
        per-frame inputs forwarded to a multi-modal backbone (currently
        MapAnything) via `_geo_forward`.  Shapes:
            depth_z_input:    [T, H, W]  metric metres (or None)
            intrinsics_input: [T, 3, 3]  pinhole K   (or None)
        Both must be provided together (MapAnything's multi-modal contract).
        """
        assert images.shape[0] == 1
        assert images.min() >= -1.0 and images.max() <= 1.0

        if (depth_z_input is None) != (intrinsics_input is None):
            raise ValueError(
                "depth_z_input and intrinsics_input must be provided "
                "together (or both omitted) — MapAnything depth fusion "
                "requires both."
            )
        self._depth_z_seq    = depth_z_input    # [T, H, W]   or None
        self._intrinsics_seq = intrinsics_input # [T, 3, 3]   or None
        if depth_z_input is not None:
            print(f"[OpenSe3rQuerySLAM] depth-input mode: "
                  f"depth_z {tuple(depth_z_input.shape)}  "
                  f"K {tuple(intrinsics_input.shape)}")

        cfg = self.cfg
        _, T, _, H, W = images.shape

        map_init_window = int(cfg.map_init_window)
        map_every       = int(cfg.map_every)
        sem_feat_key    = self._sem_feat_key()

        # Probe feature dims with a tiny forward.  Query mode →
        # `instance_feat` is None so we skip that read.
        probe_imgs = images[:, :min(map_init_window, T)].to(cfg.device)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            probe = self.model.forward_chunk(
                {'images': probe_imgs},
                kv_cache_list = self.model.init_kv_cache(),
                return_poses  = False,
                mode          = self.attn_mode,
            )
        if 'mask_logits' not in probe:
            raise RuntimeError(
                "OpenSe3rQuerySLAM requires a model with query-based instance "
                "head (instance_head_type='query'); probe returned no "
                "'mask_logits'.  Use OpenSe3rSLAM for dense-head models."
            )
        sem_feat_dim = probe.get(sem_feat_key, probe['semantic_feat']).shape[2]
        is_metric    = bool(probe.get('is_metric', False))
        if is_metric:
            print(f"[run] backend reports metric scale "
                  f"(scale_factor={probe.get('scale_factor', 1.0):.3f}) — "
                  f"alignment will seed scale=1.0")

        memory = self._build_memory(T, H, W, sem_feat_dim, is_metric)
        print(f"[run] semantic/instance attn_mode = '{self.attn_mode}'")
        print(f"[run] instance_table: knn_radius={self.knn_radius}m  "
              f"match_threshold={self.match_threshold}  "
              f"w_desc={self.w_desc} w_sem={self.w_sem} "
              f"w_vox={self.w_vox} w_geo={self.w_geo}")
        if self.point_conf_threshold is not None:
            print(f"[run] push-time geom conf threshold = "
                  f"{self.point_conf_threshold:.4f}")

        initialized     = False
        last_mapped_idx = -1
        kv_cache        = None
        t0              = time.time()

        for idx in range(T):
            if idx < map_init_window - 1:
                continue
            if not initialized:
                kv_cache = self._initialize_map(
                    images, idx + 1, memory, sem_feat_key
                )
                initialized     = True
                last_mapped_idx = idx
                self._log(idx, T, memory, t0)
                continue
            if (idx - last_mapped_idx < map_every) and (idx < T - 1):
                continue
            start_idx = last_mapped_idx + 1
            kv_cache  = self._mapping_step(
                images, start_idx, idx, memory, sem_feat_key, kv_cache
            )
            last_mapped_idx = idx
            self._log(idx, T, memory, t0)

        return memory

    def _log(self, idx, T, memory: OpenSe3rQueryMemory, t0):
        elapsed = time.time() - t0
        fps     = (idx + 1) / elapsed
        storage = 'voxels' if self.feat_storage == 'voxel' else 'pts'
        print(f"[OpenSe3rQuerySLAM] frame {idx+1:4d}/{T}"
              f"  sem {storage}: {memory.sem_count:,}"
              f"  instances: {memory.instance_count}"
              f"  kf={memory.cur_kf_idx.tolist()}"
              f"  {fps:.1f} fps")
        print(mem_stats.format_mem_log(
            memory,
            peaks            = self._mem_peaks,
            kv_cache_list    = self._last_kv_cache,
            voxel_kv_manager = getattr(self, 'voxel_kv_manager', None),
        ))
