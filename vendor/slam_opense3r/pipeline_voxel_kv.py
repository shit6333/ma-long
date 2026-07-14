"""
OpenSe3r SLAM with 3-tier voxel KV memory (experimental).

Two pipelines mirror the original pair:

  - `OpenSe3rSLAM_VoxelKV`     (subclass of `OpenSe3rSLAM`) — pred mode
  - `OpenSe3rSLAM_VoxelKV_GT`  (subclass of `OpenSe3rSLAM_GT`) — GT mode

Both swap the model's flat KV-cache for a spatial 3-tier manager:

    first_frame_kv  ⊕  anchor_pool  ⊕  window  ⊕  retrieved_voxel_pivots

Goals:
  * Bounded memory for long sequences (window + voxel cap).
  * Better instance separation across loop closures (anchor sticky mode
    keeps early instance sightings; voxel store provides spatial recall).

The geometry path is identical to the originals.  Only the semantic
forward's `kv_cache_list` is built from / written back to the
ThreeTierKVManager instead of the legacy flat list.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import torch

from slam_opense3r.pipeline import OpenSe3rSLAM, _prune_kv_cache
from slam_opense3r.pipeline_gt import OpenSe3rSLAM_GT
from slam_opense3r.memory import OpenSe3rMemory
from slam_opense3r.voxel_kv_memory import ThreeTierKVManager, VoxelKVConfig
from slam_opense3r import mem_stats


class OpenSe3rSLAM_VoxelKV(OpenSe3rSLAM):
    """OpenSe3r SLAM (pred mode) with voxel-aware 3-tier KV memory.

    Use exactly like `OpenSe3rSLAM`:

        pipeline = OpenSe3rSLAM_VoxelKV(
            model        = model,
            cfg_path     = './slam_opense3r/config.yaml',
            voxel_kv_cfg = VoxelKVConfig(...),
            backend      = 'da3',
            ...
        )
        memory = pipeline.run(images)

    The voxel KV manager is lazily constructed inside `run()` once we know
    the model's CausalBranch dims.  `kv_cache` argument passing through the
    `_initialize_map` / `_mapping_step` signature is kept for API parity
    but ignored — state lives in `self.voxel_kv_manager`.
    """

    def __init__(
        self,
        *args,
        voxel_kv_cfg: Optional[VoxelKVConfig] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.voxel_kv_cfg = voxel_kv_cfg or VoxelKVConfig()
        self.voxel_kv_manager: Optional[ThreeTierKVManager] = None

    # ── lazy manager init ───────────────────────────────────────────────────

    def _ensure_manager(self):
        """Build the manager on first use, reading dims from the live model."""
        if self.voxel_kv_manager is not None:
            return
        m = self.model
        cb = m.causal_branch
        # ins_dim — for the legacy dense head, read the first conv of
        # `instance_head.out_proj` Sequential.  For the query head
        # (Mask2FormerQueryInstanceHead) `out_proj` doesn't exist and
        # `out['instance_feat']` is None at forward time; the query
        # pipeline passes a `dim=1` dummy zero tensor instead, so the
        # manager just needs `ins_dim=1` to make its `.view()` happy.
        if hasattr(m.instance_head, 'out_proj'):
            try:
                ins_dim = int(m.instance_head.out_proj[0].out_channels)
            except (AttributeError, IndexError, TypeError) as e:
                raise RuntimeError(
                    "Could not read ins_dim from model.instance_head.out_proj — "
                    "set model.ins_dim manually or expose it on the head."
                ) from e
        else:
            ins_dim = 1   # query head: instance_feat is None / dummy dim=1
        # Num heads / head_dim — CausalBranch doesn't store these, but the
        # underlying Attention block does (see thirdparty/.../attention.py).
        attn = cb.global_blocks[0].attn
        num_heads = int(attn.num_heads)
        head_dim  = int(attn.head_dim)
        self.voxel_kv_manager = ThreeTierKVManager(
            n_blocks         = cb.n_blocks,
            num_heads        = num_heads,
            head_dim         = head_dim,
            ins_dim          = ins_dim,
            patch_start_idx  = cb.patch_start_idx,
            cfg              = self.voxel_kv_cfg,
            device           = str(self.cfg.device),
            dtype            = torch.bfloat16,
        )
        print(f"[VoxelKV] initialised manager  "
              f"n_blocks={cb.n_blocks} heads={num_heads} head_dim={head_dim} "
              f"ins_dim={ins_dim} patch_start_idx={cb.patch_start_idx} "
              f"voxel_size={self.voxel_kv_cfg.voxel_size}m "
              f"anchor_cap={self.voxel_kv_cfg.anchor_cap}({self.voxel_kv_cfg.anchor_mode}) "
              f"window={self.voxel_kv_cfg.window_chunks} "
              f"pivot_cap={self.voxel_kv_cfg.voxel_pivot_cap}")

    # ── helpers to extract per-patch metadata from forward output ───────────

    @staticmethod
    def _per_patch_world_points(world_points: torch.Tensor,
                                 patch_size: int) -> torch.Tensor:
        """world_points [B, T, H, W, 3] → per-patch mean position
        [B*T*N_patch, 3]."""
        B, T, H, W, _ = world_points.shape
        ph, pw = H // patch_size, W // patch_size
        wp = world_points[:, :, :ph * patch_size, :pw * patch_size, :]
        wp = wp.reshape(B, T, ph, patch_size, pw, patch_size, 3)
        wp = wp.mean(dim=(3, 5))                                      # [B, T, ph, pw, 3]
        return wp.reshape(B * T * ph * pw, 3), ph * pw

    @staticmethod
    def _instance_feat_to_patches(instance_feat: torch.Tensor,
                                  patch_size: int) -> torch.Tensor:
        """instance_feat [B, T, C, H, W] → per-patch mean [B*T*N_patch, C]."""
        B, T, C, H, W = instance_feat.shape
        ph, pw = H // patch_size, W // patch_size
        f = instance_feat[:, :, :, :ph * patch_size, :pw * patch_size]
        f = f.reshape(B, T, C, ph, patch_size, pw, patch_size)
        f = f.mean(dim=(4, 6))                                        # [B, T, C, ph, pw]
        f = f.permute(0, 1, 3, 4, 2).contiguous()                     # [B, T, ph, pw, C]
        return f.reshape(B * T * ph * pw, C)

    def _is_keyframe_mask(self, memory, frame_indices: torch.Tensor) -> torch.Tensor:
        kf_set = set(int(x) for x in memory.cur_kf_idx.tolist())
        return torch.tensor([int(f) in kf_set for f in frame_indices.tolist()],
                            dtype=torch.bool)

    # ── core overrides ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _sem_forward_voxel_kv(
        self,
        imgs: torch.Tensor,
        frame_indices: torch.Tensor,
        memory,
        extr_rel_override: Optional[torch.Tensor],
    ) -> dict:
        """Materialise 3-tier cache → forward_chunk → absorb new chunk back
        into the manager.  Returns the chunk's semantic / instance output."""
        self._ensure_manager()
        mgr = self.voxel_kv_manager

        # 1. Need per-patch world positions for retrieval.  Run a partial
        #    forward to get them — but we already have memory.pts for past
        #    frames; for new frames we'd need geometry first.  Pipeline
        #    flow guarantees geometry has run, and memory.pts[start:end+1]
        #    holds the globally-aligned world points for new frames.
        T = frame_indices.numel()
        patch_size = self.model.causal_branch.patch_size

        wp = memory.pts[frame_indices].unsqueeze(0).to(self.cfg.device)  # [1, T, H, W, 3]
        new_positions, N_patch = self._per_patch_world_points(wp, patch_size)

        # 2. Materialise the KV cache the model will see.
        kv_cache_list, n_aw, n_ret = mgr.materialize_kv_cache(
            new_chunk_positions = new_positions,
            batch_size          = imgs.shape[0],
        )
        # If everything is empty (first call), feed an empty cache.
        if all(k is None for k, _ in kv_cache_list):
            kv_cache_list = self.model.init_kv_cache()

        # 3. Forward.
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
        # KV cache for this pipeline is managed by ThreeTierKVManager; format_mem_log
        # will pull its breakdown via `voxel_kv_manager=...` instead of the flat list.
        self._last_kv_cache = None

        # 4. Pull per-patch instance feat for the new chunk (mean over patch).
        ins_dense = out['instance_feat']                  # [B, T, ins, H, W]
        new_instance_feat = self._instance_feat_to_patches(ins_dense, patch_size)

        # 5. Absorb the new chunk into memory.
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

        # 6. Rotate-mode anchor sync (no-op for sticky).
        mgr.sync_anchor_with_keyframes(memory.cur_kf_idx.tolist())

        # Diagnostics each step
        diag = mgr.diagnostics()
        print(f"  [VoxelKV] anchor={diag['anchor_pool_size']}+first={diag['first_frame_pinned']} "
              f"window_chunks={diag['window_chunks']}({diag['window_frames']}f) "
              f"voxels={diag['voxel_count']} pivots={diag['voxel_pivots']} "
              f"retrieved={n_ret}")

        return out

    @torch.no_grad()
    def _initialize_map(self, images, num_init, memory, sem_feat_key):
        """Initial chunk: run geometry (sets memory.pts/poses for init frames),
        then voxel-aware semantic forward."""
        device    = self.cfg.device
        init_imgs = images[:, :num_init].to(device)

        # Geometry (same as parent)
        geo_res = self._geo_forward(init_imgs)

        # Pose override = geo_res poses (init chunk's local frame = global frame)
        init_c2w_g = geo_res['poses'].detach().cpu().float()

        # First, populate memory.pts/poses for the init frames so we can read
        # world points for voxel retrieval.  This mirrors `memory.initialize`'s
        # geometry-side work.  Then run sem forward.  Then call full
        # `memory.initialize` to push features.
        # Easiest: just call memory.initialize with a stub sem_res, then
        # actually run sem forward AFTER memory.pts is set.
        # But memory.initialize also pushes sem_res → we want sem_res to
        # already exist.  Solution: do geometry-only init manually here, run
        # sem, then call _push_sem_ins.

        from slam_opense3r.memory import OpenSe3rMemory     # avoid circ-import at top

        conf_raw = geo_res['world_points_conf'][0].cpu().squeeze(-1)
        pts      = geo_res['world_points'][0].cpu().float()
        c2w      = geo_res['poses'][0].cpu().float()
        conf_sig = memory._conf_sig(conf_raw)
        T_init   = pts.shape[0]

        memory.pts  [:T_init] = pts
        memory.conf [:T_init] = conf_sig
        memory.poses[:T_init] = c2w
        memory.iter [:T_init] = 1

        # Keyframe selection (same as memory.initialize but inlined)
        from amb3r.tools.keyframes import select_keyframes_iteratively
        from amb3r.tools.pose_dist import extrinsic_distance_batch_query
        dists = extrinsic_distance_batch_query(c2w.numpy(), c2w.numpy())
        dists_t = torch.from_numpy(dists)
        is_kf = select_keyframes_iteratively(
            dists_t, conf_sig, memory.cfg.keyframe_threshold, keyframe_indices=[0],
        )
        kf_idx = torch.nonzero(is_kf, as_tuple=False).squeeze(1)
        memory.kf_idx     = kf_idx
        memory.cur_kf_idx = kf_idx.clone()

        # Semantic forward with voxel KV manager
        frame_indices = torch.arange(0, T_init)
        sem_res = self._sem_forward_voxel_kv(
            imgs              = init_imgs,
            frame_indices     = frame_indices,
            memory            = memory,
            extr_rel_override = init_c2w_g,
        )

        # Push semantic / instance features for init frames
        if sem_res is not None:
            memory._push_sem_ins(sem_res, pts, sem_feat_key, geom_conf=conf_sig)

        # Manager state holds KV; no flat kv_cache needed
        return None

    @torch.no_grad()
    def _mapping_step(self, images, start_idx, end_idx, memory, sem_feat_key,
                       kv_cache):
        """Pred-mode mapping step using the voxel KV manager."""
        device  = self.cfg.device
        cur_kf  = memory.cur_kf_idx

        kf_imgs  = images[0, cur_kf].unsqueeze(0).to(device)
        new_imgs = images[:, start_idx:end_idx + 1].to(device)
        geo_imgs = torch.cat([kf_imgs, new_imgs], dim=1)
        geo_res  = self._geo_forward(geo_imgs)

        # Snapshot the poses that `update_geo` will refine so we can compute
        # per-frame SE(3) deltas afterwards (used to update stored token
        # positions in the voxel manager).
        map_idx = torch.cat([
            memory.cur_kf_idx,
            torch.arange(start_idx, end_idx + 1),
        ], dim=0)
        old_poses_for_delta: Optional[torch.Tensor] = None
        if self.voxel_kv_cfg.update_positions_on_alignment:
            old_poses_for_delta = memory.poses[map_idx].clone().float()
            old_map_idx_list = map_idx.tolist()

        # Geometry alignment (sets memory.poses[start:end+1] to global)
        memory.update_geo(geo_res, start_idx, end_idx)

        # Propagate alignment refinement to manager (anchor / window / voxel
        # positions are updated in place; pivots are re-bucketed if positions
        # cross voxel boundaries).
        if (self.voxel_kv_manager is not None
                and self.voxel_kv_cfg.update_positions_on_alignment
                and old_poses_for_delta is not None):
            new_poses = memory.poses[map_idx].float()
            frame_to_delta: dict = {}
            for i, fidx in enumerate(old_map_idx_list):
                T_old = old_poses_for_delta[i]
                T_new = new_poses[i]
                T_delta = T_new @ torch.inverse(T_old)
                # Only include frames whose refinement is non-trivial.
                if (T_delta - torch.eye(4)).abs().max().item() > 1e-6:
                    frame_to_delta[int(fidx)] = T_delta
            if frame_to_delta:
                self.voxel_kv_manager.update_frame_positions(frame_to_delta)

        # Pose override = globally-aligned new c2w
        new_c2w_global = memory.poses[start_idx:end_idx + 1].unsqueeze(0).float()

        frame_indices = torch.arange(start_idx, end_idx + 1)
        sem_res = self._sem_forward_voxel_kv(
            imgs              = new_imgs,
            frame_indices     = frame_indices,
            memory            = memory,
            extr_rel_override = new_c2w_global,
        )

        # Push new chunk's features
        memory.update_sem(sem_res, sem_feat_key)

        return None   # kv_cache is internal to the manager

    # ── reset hook ──────────────────────────────────────────────────────────

    def run(self, images: torch.Tensor):
        # Reset manager between independent runs of the same pipeline instance
        if self.voxel_kv_manager is not None:
            self.voxel_kv_manager.reset()
        return super().run(images)


# ═════════════════════════════════════════════════════════════════════════════
# GT-mode variant
# ═════════════════════════════════════════════════════════════════════════════

class OpenSe3rSLAM_VoxelKV_GT(OpenSe3rSLAM_GT):
    """GT-mode SLAM using the 3-tier voxel KV manager.

    Mirrors `OpenSe3rSLAM_GT` (no coordinate_alignment, GT geometry pushed
    directly into memory) but the semantic forward goes through the KV
    manager.  Keyframes are selected inline via pose-distance thresholding
    (GT pipeline has no keyframe state by default) so anchor promotion works.
    """

    def __init__(
        self,
        *args,
        voxel_kv_cfg: Optional[VoxelKVConfig] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.voxel_kv_cfg = voxel_kv_cfg or VoxelKVConfig()
        self.voxel_kv_manager: Optional[ThreeTierKVManager] = None
        # GT-pipeline keyframe state (just a list of frame indices that
        # passed the pose-distance threshold).
        self._kf_indices: List[int] = []

    def _ensure_manager(self):
        if self.voxel_kv_manager is not None:
            return
        m = self.model
        cb = m.causal_branch
        # Same query-head guard as the non-GT _ensure_manager above.
        if hasattr(m.instance_head, 'out_proj'):
            try:
                ins_dim = int(m.instance_head.out_proj[0].out_channels)
            except (AttributeError, IndexError, TypeError) as e:
                raise RuntimeError(
                    "Could not read ins_dim from model.instance_head.out_proj"
                ) from e
        else:
            ins_dim = 1   # query head fallback
        attn = cb.global_blocks[0].attn
        num_heads = int(attn.num_heads)
        head_dim  = int(attn.head_dim)
        self.voxel_kv_manager = ThreeTierKVManager(
            n_blocks         = cb.n_blocks,
            num_heads        = num_heads,
            head_dim         = head_dim,
            ins_dim          = ins_dim,
            patch_start_idx  = cb.patch_start_idx,
            cfg              = self.voxel_kv_cfg,
            device           = str(self.cfg.device),
            dtype            = torch.bfloat16,
        )
        print(f"[VoxelKV-GT] manager init  storage={self.voxel_kv_cfg.storage_location} "
              f"heads={num_heads} head_dim={head_dim} ins_dim={ins_dim} "
              f"voxel_size={self.voxel_kv_cfg.voxel_size} "
              f"anchor_cap={self.voxel_kv_cfg.anchor_cap}({self.voxel_kv_cfg.anchor_mode}) "
              f"window={self.voxel_kv_cfg.window_chunks} "
              f"pivot_cap={self.voxel_kv_cfg.voxel_pivot_cap}")

    def _is_new_keyframe(self, candidate_pose: torch.Tensor,
                          all_poses_so_far: torch.Tensor,
                          threshold: float) -> bool:
        """Simple pose-distance keyframe rule: candidate is keyframe if min
        translation distance to any existing keyframe pose > threshold."""
        if not self._kf_indices:
            return True
        kf_poses = all_poses_so_far[self._kf_indices]            # [K, 4, 4]
        dists = (kf_poses[:, :3, 3] - candidate_pose[:3, 3]).norm(dim=-1)
        return dists.min().item() > threshold

    @staticmethod
    def _per_patch_world_points(world_points_5d: torch.Tensor,
                                 patch_size: int):
        """world_points_5d: [B, T, H, W, 3] → ([B*T*N_patch, 3], N_patch)."""
        B, T, H, W, _ = world_points_5d.shape
        ph, pw = H // patch_size, W // patch_size
        wp = world_points_5d[:, :, :ph * patch_size, :pw * patch_size, :]
        wp = wp.reshape(B, T, ph, patch_size, pw, patch_size, 3)
        wp = wp.mean(dim=(3, 5))                                 # [B, T, ph, pw, 3]
        return wp.reshape(B * T * ph * pw, 3), ph * pw

    @staticmethod
    def _instance_feat_to_patches(instance_feat: torch.Tensor,
                                  patch_size: int) -> torch.Tensor:
        B, T, C, H, W = instance_feat.shape
        ph, pw = H // patch_size, W // patch_size
        f = instance_feat[:, :, :, :ph * patch_size, :pw * patch_size]
        f = f.reshape(B, T, C, ph, patch_size, pw, patch_size)
        f = f.mean(dim=(4, 6))                                   # [B, T, C, ph, pw]
        f = f.permute(0, 1, 3, 4, 2).contiguous()                # [B, T, ph, pw, C]
        return f.reshape(B * T * ph * pw, C)

    # ── main run loop ──────────────────────────────────────────────────────

    def run(
        self,
        images:   torch.Tensor,
        pts_gt:   torch.Tensor,
        poses_gt: torch.Tensor,
    ) -> OpenSe3rMemory:
        assert images.shape[0] == 1
        assert pts_gt.shape[0] == images.shape[1]
        assert poses_gt.shape[0] == images.shape[1]

        cfg        = self.cfg
        device     = cfg.device
        chunk_size = int(cfg.chunk_size)
        _, T, _, H, W = images.shape

        patch_size = getattr(self.model.causal_branch, 'patch_size', 14)

        # Probe (under bf16 autocast — same as pipeline_gt)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            probe = self.model.forward_chunk(
                {'images': images[:, :min(chunk_size, T)].to(device)},
                kv_cache_list=self.model.init_kv_cache(),
                mode=self.attn_mode,
            )
        sem_key      = self._sem_feat_key()
        sem_feat_dim = probe.get(sem_key, probe['semantic_feat']).shape[2]
        ins_feat_dim = probe['instance_feat'].shape[2]

        memory = OpenSe3rMemory(
            cfg          = cfg,
            num_frames   = T,
            H = H, W = W,
            sem_feat_dim = sem_feat_dim,
            ins_feat_dim = ins_feat_dim,
            feat_storage = self.feat_storage,
            voxel_size   = float(cfg.semantic_voxel_size),
            backend      = self.backend,
            is_metric    = False,
            point_conf_threshold = self.point_conf_threshold,
        )

        # Reset / init manager state
        self._ensure_manager()
        mgr = self.voxel_kv_manager
        mgr.reset()
        self._kf_indices = []

        # Single pose anchor for the whole sequence — `extr_rel_override` for
        # every chunk is computed relative to GT frame 0.
        pose_anchor_inv = torch.inverse(poses_gt[0].float())
        kf_threshold    = float(cfg.keyframe_threshold)

        t0 = time.time()

        for chunk_start in range(0, T, chunk_size):
            chunk_end  = min(chunk_start + chunk_size, T)
            chunk_imgs = images[:, chunk_start:chunk_end].to(device)
            frame_idx  = torch.arange(chunk_start, chunk_end)
            T_c        = chunk_end - chunk_start

            # ── Keyframe selection for this chunk (pose-distance threshold)
            is_kf_mask = torch.zeros(T_c, dtype=torch.bool)
            for i, fi in enumerate(frame_idx.tolist()):
                if self._is_new_keyframe(poses_gt[fi].float(), poses_gt.float(),
                                          kf_threshold):
                    is_kf_mask[i] = True
                    self._kf_indices.append(int(fi))

            # ── pose-cond anchor: GT c2w anchored at GT frame 0
            chunk_c2w_anchored = pose_anchor_inv @ poses_gt[chunk_start:chunk_end].float()
            extr_rel_override  = chunk_c2w_anchored.unsqueeze(0)

            # ── per-patch GT positions for voxel retrieval
            wp = pts_gt[chunk_start:chunk_end].unsqueeze(0).to(device).float()   # [1, T_c, H, W, 3]
            new_positions, N_patch = self._per_patch_world_points(wp, patch_size)

            # ── materialise cache
            kv_cache_list, n_aw, n_ret = mgr.materialize_kv_cache(
                new_chunk_positions = new_positions,
                batch_size          = images.shape[0],
            )
            if all(k is None for k, _ in kv_cache_list):
                kv_cache_list = self.model.init_kv_cache()

            # ── forward
            with mem_stats.GpuPeakTracker(self._mem_peaks, 'sem'), \
                    torch.no_grad(), \
                    torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = self.model.forward_chunk(
                    {'images': chunk_imgs},
                    kv_cache_list     = kv_cache_list,
                    return_poses      = False,
                    mode              = self.attn_mode,
                    extr_rel_override = extr_rel_override,
                )
            new_kv = out.pop('kv_cache', None)
            # KV is managed by ThreeTierKVManager — let format_mem_log pull the
            # breakdown via `voxel_kv_manager=...` rather than the flat cache.
            self._last_kv_cache = None

            # ── per-patch instance feat for voxel store eviction signal
            new_instance_feat = self._instance_feat_to_patches(
                out['instance_feat'], patch_size,
            )

            # ── absorb chunk
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

            # ── push GT geometry + sem/ins features into SLAM memory (same
            # as pipeline_gt)
            pts_chunk   = pts_gt  [chunk_start:chunk_end].float()
            poses_chunk = poses_gt[chunk_start:chunk_end].float()
            pts_norm    = pts_chunk.norm(dim=-1)
            conf_chunk  = (pts_norm > 1e-3).float()

            sem_feat = out.get(sem_key, out['semantic_feat'])[0].cpu().float()
            sem_conf = out['semantic_conf'][0].cpu().float()
            ins_feat = out['instance_feat'][0].cpu().float()
            ins_conf = out['instance_conf'][0].cpu().float()

            memory.update_gt(
                frame_indices = frame_idx,
                pts   = pts_chunk,
                poses = poses_chunk,
                conf  = conf_chunk,
                sem_feat = sem_feat,
                sem_conf = sem_conf,
                ins_feat = ins_feat,
                ins_conf = ins_conf,
            )

            # ── logging
            elapsed = time.time() - t0
            fps     = chunk_end / elapsed
            diag    = mgr.diagnostics()
            storage_label = 'voxels' if self.feat_storage == 'voxel' else 'pts'
            print(f"[VoxelKV-GT] frames {chunk_end:4d}/{T}  "
                  f"sem {storage_label}: {memory.sem_count:,}  "
                  f"ins {storage_label}: {memory.ins_count:,}  "
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
