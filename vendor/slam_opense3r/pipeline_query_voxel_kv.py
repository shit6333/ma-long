"""
slam_opense3r/pipeline_query_voxel_kv.py
=========================================
Voxel-KV SLAM pipeline for the query-based instance head.

Inherits the 3-tier KV memory machinery from `OpenSe3rSLAM_VoxelKV` and
swaps the dense `ins_store` for an `InstanceTable` (as in
`OpenSe3rQuerySLAM`).

Two design points worth highlighting:

  1. `_sem_forward_voxel_kv` in the dense pipeline pulls
     `out['instance_feat']` to feed the voxel manager's pivot signal.  In
     query mode that key is `None`, so we fall back to the V tensor of
     the last CausalBranch layer (which the manager already supports via
     `cfg.pivot_signal='V'`).  This subclass enforces that fallback when
     the user leaves `pivot_signal='instance_feat'`.

  2. `update_instances()` is called AFTER `_sem_forward_voxel_kv` and
     BEFORE `update_sem` (which consumes the pts-new cache), mirroring
     the order in `OpenSe3rQuerySLAM._mapping_step`.

The legacy `OpenSe3rSLAM_VoxelKV` is left fully intact.
"""

from __future__ import annotations
from typing import Optional
import torch

from .pipeline_voxel_kv import OpenSe3rSLAM_VoxelKV, VoxelKVConfig
from .pipeline_query    import OpenSe3rQuerySLAM
from .memory_query      import OpenSe3rQueryMemory
from .                  import mem_stats


class OpenSe3rQuerySLAM_VoxelKV(OpenSe3rSLAM_VoxelKV):
    """Query-head SLAM with 3-tier voxel KV memory."""

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
        voxel_kv_cfg:          Optional[VoxelKVConfig] = None,
        # InstanceTable knobs (same names + defaults as OpenSe3rQuerySLAM).
        instance_voxel_size:   float = 0.05,
        knn_radius:            float = 2.0,
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
        # Force pivot signal off 'instance_feat' for query mode — that key is
        # None.  Default to 'V' (last-layer Value, an existing supported
        # choice on the manager).  User can still pass 'K' via cfg if desired.
        if voxel_kv_cfg is None:
            voxel_kv_cfg = VoxelKVConfig()
        if voxel_kv_cfg.pivot_signal == 'instance_feat':
            print("[VoxelKV-Query] pivot_signal='instance_feat' not supported "
                  "in query-head mode (dense ins feat is None) — switching to 'V'")
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
        # InstanceTable knobs (held on self, used by _build_memory below).
        self.instance_voxel_size = float(instance_voxel_size)
        self.knn_radius          = float(knn_radius)
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
    # Memory factory — identical to OpenSe3rQuerySLAM._build_memory.  We
    # define it here as a free method (not via cooperative inheritance) so
    # the wiring is explicit.
    # ------------------------------------------------------------------

    def _build_memory(self, T: int, H: int, W: int,
                      sem_feat_dim: int, is_metric: bool) -> OpenSe3rQueryMemory:
        return OpenSe3rQuerySLAM._build_memory(self, T, H, W, sem_feat_dim, is_metric)

    # ------------------------------------------------------------------
    # _sem_forward_voxel_kv — must not crash when out['instance_feat'] is None
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sem_forward_voxel_kv(
        self,
        imgs: torch.Tensor,
        frame_indices: torch.Tensor,
        memory,
        extr_rel_override: Optional[torch.Tensor],
    ) -> dict:
        """Voxel-KV semantic forward adapted for query head.

        The dense pipeline pulls `instance_feat` to feed the manager's pivot
        signal.  In query mode that's None — we therefore tell the manager
        to fall back to K/V (via cfg.pivot_signal flip in __init__) and
        skip the per-patch instance-feat extraction.
        """
        self._ensure_manager()
        mgr = self.voxel_kv_manager

        T = frame_indices.numel()
        patch_size = self.model.causal_branch.patch_size

        wp = memory.pts[frame_indices].unsqueeze(0).to(self.cfg.device)
        new_positions, N_patch = self._per_patch_world_points(wp, patch_size)

        kv_cache_list, n_aw, n_ret = mgr.materialize_kv_cache(
            new_chunk_positions = new_positions,
            batch_size          = imgs.shape[0],
        )
        if all(k is None for k, _ in kv_cache_list):
            kv_cache_list = self.model.init_kv_cache()

        with mem_stats.GpuPeakTracker(self._mem_peaks, 'sem'), \
                torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = self.model.forward_chunk(
                {'images': imgs},
                kv_cache_list     = kv_cache_list,
                return_poses      = False,
                mode              = self.attn_mode,
                extr_rel_override = extr_rel_override,
            )
        new_kv = out.pop('kv_cache', None)
        # KV is managed by ThreeTierKVManager — let format_mem_log pull the
        # breakdown via `voxel_kv_manager=...` rather than the flat cache.
        self._last_kv_cache = None

        # In query mode, `out['instance_feat']` is None.  Pass a dummy
        # zero feature so absorb_chunk's interface is unchanged; the
        # manager itself uses pivot_signal='V' (forced in __init__) and
        # ignores the new_chunk_instance_feat argument when not needed.
        if out.get('instance_feat') is not None:
            new_instance_feat = self._instance_feat_to_patches(
                out['instance_feat'], patch_size,
            )
        else:
            new_instance_feat = torch.zeros(
                imgs.shape[0] * T * N_patch, 1,
                device=imgs.device, dtype=torch.float32,
            )

        is_kf_mask = self._is_keyframe_mask(memory, frame_indices)
        K_list = [layer_cache[0] for layer_cache in new_kv]
        V_list = [layer_cache[1] for layer_cache in new_kv]
        mgr.absorb_chunk(
            new_chunk_K_per_layer       = K_list,
            new_chunk_V_per_layer       = V_list,
            T_cache_anchor_window       = n_aw,
            n_retrieved_tokens          = n_ret,
            new_chunk_positions         = new_positions,
            new_chunk_instance_feat     = new_instance_feat,
            new_chunk_frame_indices     = frame_indices,
            new_chunk_is_keyframe_mask  = is_kf_mask,
            current_keyframes           = memory.cur_kf_idx.tolist(),
            N_patch                     = N_patch,
            batch                       = 0,
        )
        mgr.sync_anchor_with_keyframes(memory.cur_kf_idx.tolist())

        diag = mgr.diagnostics()
        print(f"  [VoxelKV-Q] anchor={diag['anchor_pool_size']}+first={diag['first_frame_pinned']} "
              f"window_chunks={diag['window_chunks']}({diag['window_frames']}f) "
              f"voxels={diag['voxel_count']} pivots={diag['voxel_pivots']} "
              f"retrieved={n_ret}  instances={memory.instance_count}")

        return out

    # ------------------------------------------------------------------
    # Init + mapping — extend parent to fire update_instances at the right
    # point.  We re-implement to make the ordering explicit (geo → sem
    # forward → instance update → sem push).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _initialize_map(self, images, num_init, memory: OpenSe3rQueryMemory,
                          sem_feat_key):
        device    = self.cfg.device
        init_imgs = images[:, :num_init].to(device)

        geo_res = self._geo_forward(init_imgs)
        init_c2w_g = geo_res['poses'].detach().cpu().float()

        # Inlined memory geometry init (mirrors parent class verbatim).
        conf_raw = geo_res['world_points_conf'][0].cpu().squeeze(-1)
        pts      = geo_res['world_points'][0].cpu().float()
        c2w      = geo_res['poses'][0].cpu().float()
        conf_sig = memory._conf_sig(conf_raw)
        T_init   = pts.shape[0]

        memory.pts  [:T_init] = pts
        memory.conf [:T_init] = conf_sig
        memory.poses[:T_init] = c2w
        memory.iter [:T_init] = 1

        # Keyframe selection (matches parent class — same inlined logic).
        from amb3r.tools.keyframes import select_keyframes_iteratively
        from amb3r.tools.pose_dist import extrinsic_distance_batch_query
        dists = extrinsic_distance_batch_query(c2w.numpy(), c2w.numpy())
        dists_t = torch.from_numpy(dists)
        is_kf = select_keyframes_iteratively(
            dists_t, conf_sig, memory.cfg.keyframe_threshold,
            keyframe_indices=[0],
        )
        kf_idx = torch.nonzero(is_kf, as_tuple=False).squeeze(1)
        memory.kf_idx     = kf_idx
        memory.cur_kf_idx = kf_idx.clone()

        # Voxel-KV semantic forward
        frame_indices = torch.arange(0, T_init)
        sem_res = self._sem_forward_voxel_kv(
            imgs              = init_imgs,
            frame_indices     = frame_indices,
            memory            = memory,
            extr_rel_override = init_c2w_g,
        )

        # Stage pts-new cache for instance & sem push (init frames = global by definition)
        memory._pending_pts_new  = pts
        memory._pending_conf_new = conf_sig

        memory.update_instances(sem_res, start_idx=0, end_idx=T_init - 1,
                                  sem_feat_key=sem_feat_key)
        if sem_res is not None:
            memory._push_sem_ins(sem_res, pts, sem_feat_key, geom_conf=conf_sig)
        memory._pending_pts_new  = None
        memory._pending_conf_new = None

        avg_conf = geo_res['world_points_conf'][0].squeeze(-1).mean().item()
        print(f"[init]  {T_init} frames  avg_conf={avg_conf:.3f}"
              f"  kf={memory.cur_kf_idx.tolist()}"
              f"  instances={memory.instance_count}")
        return None

    @torch.no_grad()
    def _mapping_step(self, images, start_idx, end_idx,
                       memory: OpenSe3rQueryMemory, sem_feat_key, kv_cache):
        device = self.cfg.device
        cur_kf = memory.cur_kf_idx

        kf_imgs  = images[0, cur_kf].unsqueeze(0).to(device)
        new_imgs = images[:, start_idx:end_idx + 1].to(device)
        geo_imgs = torch.cat([kf_imgs, new_imgs], dim=1)
        geo_res  = self._geo_forward(geo_imgs)

        # Snapshot poses for optional manager position-refinement (parent feat).
        map_idx = torch.cat([
            cur_kf, torch.arange(start_idx, end_idx + 1),
        ], dim=0)
        old_poses_for_delta: Optional[torch.Tensor] = None
        if self.voxel_kv_cfg.update_positions_on_alignment:
            old_poses_for_delta = memory.poses[map_idx].clone().float()
            old_map_idx_list = map_idx.tolist()

        memory.update_geo(geo_res, start_idx, end_idx)

        if (self.voxel_kv_manager is not None
                and self.voxel_kv_cfg.update_positions_on_alignment
                and old_poses_for_delta is not None):
            new_poses = memory.poses[map_idx].float()
            frame_to_delta: dict = {}
            for i, fidx in enumerate(old_map_idx_list):
                T_old = old_poses_for_delta[i]
                T_new = new_poses[i]
                T_delta = T_new @ torch.inverse(T_old)
                if (T_delta - torch.eye(4)).abs().max().item() > 1e-6:
                    frame_to_delta[int(fidx)] = T_delta
            if frame_to_delta:
                self.voxel_kv_manager.update_frame_positions(frame_to_delta)

        new_c2w_global = memory.poses[start_idx:end_idx + 1].unsqueeze(0).float()

        frame_indices = torch.arange(start_idx, end_idx + 1)
        sem_res = self._sem_forward_voxel_kv(
            imgs              = new_imgs,
            frame_indices     = frame_indices,
            memory            = memory,
            extr_rel_override = new_c2w_global,
        )

        # Instance update BEFORE update_sem consumes _pending_pts_new.
        memory.update_instances(sem_res, start_idx, end_idx,
                                  sem_feat_key=sem_feat_key)
        memory.update_sem(sem_res, sem_feat_key)

        return None

    # ------------------------------------------------------------------
    # run() — same shape as parent, but probe skips `instance_feat`
    # and the built memory is OpenSe3rQueryMemory.
    # ------------------------------------------------------------------

    def run(self, images: torch.Tensor) -> OpenSe3rQueryMemory:
        if self.voxel_kv_manager is not None:
            self.voxel_kv_manager.reset()
        return OpenSe3rQuerySLAM.run(self, images)

    def _log(self, idx, T, memory, t0):
        return OpenSe3rQuerySLAM._log(self, idx, T, memory, t0)
