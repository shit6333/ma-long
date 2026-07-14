"""
OpenSe3rMemory
==============
AMB3R-style map memory for OpenSe3r SLAM.

Geometry handling mirrors slam/memory.py:
  - VGGT confidence (expp1 activation, values > 1) → converted to conf_sig ∈ (0,1)
  - Per-frame pts/poses/iter buffers with confidence-weighted running average
  - Keyframe selection and management (identical to AMB3R)

Feature storage backends (unchanged from original design):
  'voxel'    — VoxelFeatureMap: confidence-weighted voxel averaging
  'perpoint' — PerPointFeatureStore: accumulate every projected point
"""

import torch
import torch.nn.functional as F

from slam_semantic.semantic_voxel_map import VoxelFeatureMap
from amb3r.tools.pose_dist import extrinsic_distance_batch_query
from amb3r.tools.keyframes import select_keyframes_iteratively
from amb3r.tools.pts_align import coordinate_alignment
from amb3r.tools.pose_interp import interpolate_poses


# ─────────────────────────────────────────────────────────────────────────────
# Per-point store (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class PerPointFeatureStore:
    """
    Accumulates (pts, feat) pairs across SLAM steps without averaging.
    Memory grows as  n_steps × T_chunk × H_feat × W_feat × C × 4 bytes.
    """

    def __init__(self, feat_dim: int):
        self.feat_dim   = feat_dim
        self._pts_list  = []
        self._feat_list = []

    @property
    def num_points(self) -> int:
        return sum(p.shape[0] for p in self._pts_list)

    @torch.no_grad()
    def update(self, pts: torch.Tensor, feat: torch.Tensor,
               conf: torch.Tensor = None, conf_threshold: float = 0.0):
        pts  = pts.detach().float().cpu()
        feat = feat.detach().float().cpu()
        if conf is not None and conf_threshold > 0.0:
            mask = conf.detach().float().cpu() > conf_threshold
            pts  = pts[mask]
            feat = feat[mask]
        if pts.shape[0] > 0:
            self._pts_list.append(pts)
            self._feat_list.append(feat)

    def get_all(self):
        if not self._pts_list:
            return (torch.zeros(0, 3), torch.zeros(0, self.feat_dim))
        return torch.cat(self._pts_list), torch.cat(self._feat_list)

    def __repr__(self):
        return f"PerPointFeatureStore(feat_dim={self.feat_dim}, num_points={self.num_points})"


# ─────────────────────────────────────────────────────────────────────────────
# Main memory class
# ─────────────────────────────────────────────────────────────────────────────

class OpenSe3rMemory:
    """
    AMB3R-style map memory for OpenSe3r SLAM.

    Params
    ------
    cfg           OmegaConf config (keyframe thresholds, etc.).  Optional for
                  GT-mode pipelines that do not use keyframe management.
    num_frames    total frames in the sequence
    H, W          full image resolution
    sem_feat_dim  semantic feature channels
    ins_feat_dim  instance feature channels
    feat_storage  'voxel' or 'perpoint'
    voxel_size    voxel cell size in metres (only voxel mode)
    """

    def __init__(self, cfg=None, num_frames=0, H=0, W=0,
                 sem_feat_dim=512, ins_feat_dim=16,
                 feat_storage='voxel', voxel_size=0.01,
                 backend='vggt', is_metric=False,
                 point_conf_threshold=None):
        """
        backend              : 'vggt' or 'da3' (only kept for diagnostics
                               — both backends use the same expp1 conf head).
        is_metric            : if True, forwards scale=1.0 hint to
                               coordinate_alignment (DA3 nested-giant-large
                               produces metric depth).
        point_conf_threshold : if not None, pixels whose post-`_conf_sig`
                               geometry confidence is below this value are
                               filtered out BEFORE being pushed into the
                               semantic / instance voxel store.  Matches the
                               semantics of run.py's ``--conf_threshold``.
        """
        self.cfg          = cfg
        self.num_frames   = num_frames
        self.H, self.W    = H, W
        self.feat_storage = feat_storage
        self.sem_feat_dim = sem_feat_dim
        self.ins_feat_dim = ins_feat_dim
        self.backend      = backend
        self.is_metric    = is_metric
        self.point_conf_threshold = (
            float(point_conf_threshold) if point_conf_threshold is not None else None
        )

        # Per-frame geometry buffers
        self.pts   = torch.zeros(num_frames, H, W, 3)
        self.conf  = torch.zeros(num_frames, H, W)
        self.poses = torch.zeros(num_frames, 4, 4)
        self.iter  = torch.zeros(num_frames)          # update count per frame

        # Keyframe indices (global)
        self.kf_idx     = None   # all keyframes ever seen
        self.cur_kf_idx = None   # active keyframes fed to next geometry step

        # Cache shuttled from update_geo → update_sem so semantic push can
        # reuse the globally-aligned pts/conf without re-running alignment.
        self._pending_pts_new  = None
        self._pending_conf_new = None

        # Feature stores
        def _make(dim):
            if dim <= 0:
                return None
            if feat_storage == 'voxel':
                return VoxelFeatureMap(voxel_size, dim)
            return PerPointFeatureStore(dim)

        self.sem_store = _make(sem_feat_dim)
        self.ins_store = _make(ins_feat_dim)

    # ------------------------------------------------------------------
    # Confidence conversion
    # ------------------------------------------------------------------

    def _conf_sig(self, conf_raw: torch.Tensor) -> torch.Tensor:
        """
        Confidence → (0, 1) scalar via the standard expp1 → sigmoid map.

        Both VGGT and DA3 default DPT heads use ``conf_activation="expp1"``
        (= ``exp(x) + 1``), so raw conf is in ``[1, ∞)`` for both backends.
        We map it to (0, 1) the same way AMB3R / infer_opense3r.py do:

            cs = (c - 1) / c    →    asymptotes to 1 as c → ∞

        ``backend`` is kept on the class for diagnostics but no longer
        switches the formula.
        """
        cs = (conf_raw - 1.0) / conf_raw.clamp(min=1e-6)
        cs[cs <= 0] = 1e-6
        return cs

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, geo_res: dict, sem_res: dict, sem_feat_key: str):
        """
        Store first-batch geometry, select keyframes, push features.

        geo_res  — output of forward_geometry on the init frames
        sem_res  — output of forward_chunk on the init frames (or None)
        """
        conf_raw = geo_res['world_points_conf'][0].cpu().squeeze(-1)  # [T, H, W]
        pts      = geo_res['world_points'][0].cpu().float()            # [T, H, W, 3]
        c2w      = geo_res['poses'][0].cpu().float()                   # [T, 4, 4]
        conf_sig = self._conf_sig(conf_raw)

        T_init = pts.shape[0]

        self.pts  [:T_init] = pts
        self.conf [:T_init] = conf_sig
        self.poses[:T_init] = c2w
        self.iter [:T_init] = 1

        avg_conf = conf_sig.mean(dim=(-1, -2)).cpu().numpy()
        print("  [init] avg conf:", [f"{c:.3f}" for c in avg_conf])

        # Keyframe selection from init frames
        dists = extrinsic_distance_batch_query(c2w.numpy(), c2w.numpy())
        import torch as _torch
        dists_t = _torch.from_numpy(dists)
        is_kf   = select_keyframes_iteratively(
            dists_t, conf_sig, self.cfg.keyframe_threshold,
            keyframe_indices=[0],
        )
        kf_idx = _torch.nonzero(is_kf, as_tuple=False).squeeze(1)
        self.kf_idx     = kf_idx
        self.cur_kf_idx = kf_idx.clone()

        # Push features for all init frames (filter by geom conf if requested)
        if sem_res is not None:
            self._push_sem_ins(sem_res, pts, sem_feat_key, geom_conf=conf_sig)

    # ------------------------------------------------------------------
    # Incremental update  (mirrors AMB3R slam/memory.py::update)
    # ------------------------------------------------------------------

    def update(self, geo_res: dict, sem_res: dict,
               start_idx: int, end_idx: int, sem_feat_key: str):
        """
        Backwards-compat wrapper: runs `update_geo` then `update_sem`.

        For streaming inference with `pose_cond` you should call them
        separately so you can read `self.poses[start_idx:end_idx+1]` (now
        in the global frame) between the two and feed it as
        `extr_rel_override` to `model.forward_chunk(...)`.
        """
        self.update_geo(geo_res, start_idx, end_idx)
        self.update_sem(sem_res, sem_feat_key)

    # ------------------------------------------------------------------

    def update_geo(self, geo_res: dict, start_idx: int, end_idx: int):
        """
        Geometry-only half of `update()`: align new geometry to the global
        map, confidence-weighted merge of pts/poses, keyframe discovery.

        After this returns, `self.poses[start_idx:end_idx+1]` are valid
        global-frame c2w that can be used as `extr_rel_override` for the
        semantic forward (provided `self.poses[0]` = identity, which is
        the SLAM convention).

        Caches the per-new-frame globally-aligned pts and confidences on
        `self._pending_pts_new` / `self._pending_conf_new` so that
        `update_sem` can push them into the feature stores without
        recomputing alignment.

        geo_res contains [kf_frames + new_frames] geometry in their joint
        local frame (first kf frame's camera as origin).
        """
        n_kf  = len(self.cur_kf_idx)
        n_new = end_idx - start_idx + 1

        pts_local   = geo_res['world_points'][0].cpu().float()           # [n_kf+n_new, H, W, 3]
        conf_raw    = geo_res['world_points_conf'][0].cpu().squeeze(-1)  # [n_kf+n_new, H, W]
        c2w_local   = geo_res['poses'][0].cpu().float()                  # [n_kf+n_new, 4, 4]
        conf_sig_local = self._conf_sig(conf_raw)

        # Global map frame for [kf + new] frames
        map_idx    = torch.cat([self.cur_kf_idx,
                                torch.arange(start_idx, end_idx + 1)], dim=0)
        pts_global = self.pts  [map_idx].float()
        c2w_global = self.poses[map_idx].float()
        conf_global= self.conf [map_idx].float()
        iter_global= self.iter [map_idx]

        # ── Coordinate alignment ─────────────────────────────────────────────
        # The geometry forward returns pts/poses anchored to the *first input
        # frame* (= cur_kf_idx[0] in our concatenation order).  We therefore
        # always align local → global; transform=False is only safe when the
        # local frame already matches global (i.e. first kf is frame 0 AND
        # global poses[frame 0] is identity, which holds at init).
        need_transform = (self.cur_kf_idx[0].item() != 0)

        # If geometry is metric (DA3 nested), seed scale=1.0 so the alignment
        # has a metric candidate.  coordinate_alignment picks the best of all
        # candidates by z-error, so this is safe even if the actual best fit
        # is slightly different.
        scale_hint = 1.0 if self.is_metric else None
        pts_gfl, c2w_gfl = coordinate_alignment(
            pts_local,   c2w_local,   conf_sig_local,
            pts_global,  c2w_global,  conf_global,
            num_kf      = n_kf,
            transform   = need_transform,
            scale       = scale_hint,
            robust      = True,
        )

        # Debug: verify alignment quality
        kf_err  = (pts_gfl[:n_kf] - pts_global[:n_kf]).norm(dim=-1)
        valid_kf = (pts_global[:n_kf].norm(dim=-1) > 1e-3) & (conf_global[:n_kf] > 1e-4)
        if valid_kf.sum() > 0:
            mean_kf_err = kf_err[valid_kf].mean().item()
            t_gfl    = c2w_gfl   [:n_kf, :3, 3]
            t_global = c2w_global[:n_kf, :3, 3]
            pose_err = (t_gfl - t_global).norm(dim=-1).mean().item()
            # approximate kf_scale from norm ratio of aligned vs original local pts
            local_norms  = pts_local[:n_kf][valid_kf].norm(dim=-1)
            aligned_norms = pts_gfl[:n_kf][valid_kf].norm(dim=-1)
            scale_est = (aligned_norms / (local_norms + 1e-8)).median().item()
            print(f"  [align] transform={need_transform}"
                  f"  scale≈{scale_est:.3f}"
                  f"  kf_pts_err={mean_kf_err:.4f}"
                  f"  kf_pose_t_err={pose_err:.4f}"
                  f"  new_pts_range=[{pts_gfl[n_kf:].norm(dim=-1).min():.2f},"
                  f"{pts_gfl[n_kf:].norm(dim=-1).max():.2f}]")

        # ── Confidence-weighted pts merge (same as AMB3R) ────────────────────
        conf_global_sum = conf_global * iter_global[:, None, None]

        self.pts  [map_idx] = (
            conf_global_sum[..., None] * pts_global
            + conf_sig_local[..., None] * pts_gfl
        ) / (conf_global_sum[..., None] + conf_sig_local[..., None] + 1e-8)

        self.conf [map_idx] = (
            conf_global_sum + conf_sig_local
        ) / (iter_global[:, None, None] + 1).clamp(min=1e-8)

        self.iter [map_idx] = iter_global + 1

        # ── Pose interpolation ───────────────────────────────────────────────
        c2w_merged = interpolate_poses(
            c2w_global, c2w_gfl, conf_global, conf_sig_local, interpolate=True
        )
        self.poses[map_idx] = c2w_merged

        # ── Keyframe discovery from newly-processed frames ───────────────────
        all_poses_in_batch = c2w_merged
        dists = extrinsic_distance_batch_query(
            all_poses_in_batch.numpy(), all_poses_in_batch.numpy()
        )
        dists_t = torch.from_numpy(dists)
        is_kf   = select_keyframes_iteratively(
            dists_t,
            conf_sig_local,
            self.cfg.keyframe_threshold,
            keyframe_indices=list(range(n_kf)),
        )
        new_kf_mask = is_kf[n_kf:]   # only check new frames
        if new_kf_mask.sum() > 0:
            new_kf_local_idx  = torch.nonzero(new_kf_mask, as_tuple=False).squeeze(1)
            new_kf_global_idx = new_kf_local_idx + start_idx
            self.cur_kf_idx   = torch.cat([self.cur_kf_idx, new_kf_global_idx])
            self.kf_idx       = torch.cat([self.kf_idx,     new_kf_global_idx])
            self.keyframe_management()

        # Cache for `update_sem` (the next call), so we don't redo alignment.
        self._pending_pts_new  = pts_gfl[n_kf:].contiguous()       # [n_new, H, W, 3]
        self._pending_conf_new = conf_sig_local[n_kf:].contiguous()# [n_new, H, W]

    # ------------------------------------------------------------------

    def update_sem(self, sem_res: dict, sem_feat_key: str):
        """
        Push semantic / instance features for the new frames last visited
        by `update_geo` into the feature stores.

        Must be preceded by `update_geo` — uses the cached globally-aligned
        pts and confidences from that call.
        """
        if sem_res is None:
            self._pending_pts_new = None
            self._pending_conf_new = None
            return
        if self._pending_pts_new is None:
            raise RuntimeError(
                "update_sem called without a preceding update_geo "
                "(no cached pts_new_global available)."
            )
        self._push_sem_ins(
            sem_res, self._pending_pts_new, sem_feat_key,
            geom_conf=self._pending_conf_new,
        )
        # Consume the cache.
        self._pending_pts_new  = None
        self._pending_conf_new = None

    # ------------------------------------------------------------------
    # GT-mode update  (no coordinate alignment, no keyframe management)
    # ------------------------------------------------------------------

    def update_gt(
        self,
        frame_indices: torch.Tensor,   # [T_c] long
        pts:           torch.Tensor,   # [T_c, H, W, 3]  GT world pts
        poses:         torch.Tensor,   # [T_c, 4, 4]     GT c2w
        conf:          torch.Tensor,   # [T_c, H, W]     0/1 validity mask
        sem_feat:      torch.Tensor,   # [T_c, C_s, H_f, W_f]
        sem_conf:      torch.Tensor,   # [T_c, 1, H_f, W_f]
        ins_feat:      torch.Tensor,   # [T_c, C_i, H_f, W_f]
        ins_conf:      torch.Tensor,   # [T_c, 1, H_f, W_f]
    ):
        """Store GT geometry and push features — no coordinate alignment."""
        self.pts  [frame_indices] = pts.float()
        self.poses[frame_indices] = poses.float()
        self.conf [frame_indices] = conf.float()
        self.iter [frame_indices] = 1

        # GT mode: ``conf`` is a 0/1 validity mask, so use it as geom_conf.
        # Any threshold > 0 will then collapse to a "drop invalid pixels"
        # filter (same behaviour as the existing pts.norm > 1e-3 check).
        if self.sem_store is not None:
            self._push_features(self.sem_store, pts.float(), sem_feat, sem_conf,
                                geom_conf=conf.float())
        if self.ins_store is not None:
            self._push_features(self.ins_store, pts.float(), ins_feat, ins_conf,
                                geom_conf=conf.float())

    # ------------------------------------------------------------------
    # Keyframe management  (identical to AMB3R slam/memory.py)
    # ------------------------------------------------------------------

    def keyframe_management(self):
        last_kf_idx  = self.cur_kf_idx[-1]
        last_kf_pose = self.poses[last_kf_idx][None]
        other_kf_idx = self.cur_kf_idx[self.cur_kf_idx != last_kf_idx]
        if len(other_kf_idx) == 0:
            return
        other_kf_pose = self.poses[other_kf_idx]
        dists_to_others = extrinsic_distance_batch_query(
            last_kf_pose.numpy(), other_kf_pose.numpy()
        )
        max_dist = float(dists_to_others.max())

        cfg = self.cfg
        if (max_dist < cfg.keyframe_resample_threshold_max_f
                and len(self.cur_kf_idx) <= cfg.max_keyframes):
            return

        print(f"  [kf] Resampling ({len(self.cur_kf_idx)} → {cfg.min_keyframes}) …")
        self.cur_kf_idx = self._resample_keyframes(
            self.kf_idx,
            self.poses[self.kf_idx],
            cfg.min_keyframes,
            self.poses,
        )
        print(f"  [kf] Active kfs: {self.cur_kf_idx.tolist()}")

    def _resample_keyframes(self, current_kf_indices, current_kf_poses,
                            num_to_keep, all_mapped_poses):
        cfg = self.cfg
        if len(current_kf_indices) <= num_to_keep:
            return current_kf_indices

        dists = extrinsic_distance_batch_query(
            current_kf_poses.numpy(), current_kf_poses.numpy()
        )
        import torch
        dists = torch.from_numpy(dists)
        pos_map = {int(g): i for i, g in enumerate(current_kf_indices)}

        newest_g = int(current_kf_indices[-1])
        resampled = [newest_g]

        initial_cands = current_kf_indices[:-1]
        if len(initial_cands) > 0:
            newest_pos = pos_map[newest_g]
            cand_pos   = [pos_map[int(g)] for g in initial_cands]
            d_to_newest = dists[cand_pos, newest_pos]
            sorted_i    = torch.argsort(d_to_newest)
            sorted_cands = initial_cands[sorted_i]
            for cand in sorted_cands:
                if len(resampled) >= cfg.top_k_keyframes + 1:
                    break
                cg  = int(cand)
                cp  = pos_map[cg]
                if abs(cg - newest_g) > 200:   # loop closure candidate
                    resampled.append(cg)
                    continue
                diverse = all(
                    dists[cp, pos_map[s]] > cfg.keyframe_resample_threshold_min
                    for s in resampled
                )
                if diverse:
                    resampled.append(cg)

        pool = sorted(int(g) for g in current_kf_indices if int(g) not in resampled)
        num_fill = num_to_keep - cfg.bridge_keyframes
        while len(resampled) < num_fill and pool:
            best = None
            for cg in pool:
                cp = pos_map[cg]
                min_d = min(dists[cp, pos_map[s]].item() for s in resampled)
                max_d = max(dists[cp, pos_map[s]].item() for s in resampled)
                if (min_d <= cfg.keyframe_resample_threshold
                        and max_d <= cfg.keyframe_resample_threshold_max_b):
                    best = cg
                    break
            if best is None:
                break
            resampled.append(best)
            pool.remove(best)

        # Bridge frames: frames closest (in pose distance) to the resampled set
        bridge = []
        if cfg.bridge_keyframes > 0:
            resampled_set = set(resampled)
            mid_cands = [int(g) for g in current_kf_indices if int(g) not in resampled_set]
            if mid_cands:
                mid_poses  = all_mapped_poses[mid_cands]
                res_poses  = all_mapped_poses[resampled]
                d_mid = extrinsic_distance_batch_query(
                    mid_poses.numpy(), res_poses.numpy()
                )
                import numpy as np
                sum_d = d_mid.sum(axis=1)
                k     = min(cfg.bridge_keyframes, len(mid_cands))
                top_k = np.argsort(sum_d)[:k]
                bridge = [mid_cands[i] for i in top_k]

        final = sorted(set(bridge + resampled))
        return torch.tensor(final, dtype=torch.long)

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _downsample_pts(pts: torch.Tensor, H_f: int, W_f: int) -> torch.Tensor:
        """[T, H, W, 3] → [T, H_f, W_f, 3]"""
        p = pts.permute(0, 3, 1, 2).float()
        p = F.interpolate(p, size=(H_f, W_f), mode='bilinear', align_corners=False)
        return p.permute(0, 2, 3, 1)

    def _push_features(self, store, pts_world: torch.Tensor,
                       feat: torch.Tensor, conf: torch.Tensor,
                       geom_conf: torch.Tensor = None):
        """
        pts_world: [T, H, W, 3]
        feat:      [T, C, H_f, W_f]
        conf:      [T, 1, H_f, W_f]   — head conf (semantic / instance)
        geom_conf: [T, H, W] or None  — geometry conf in (0, 1) (post _conf_sig).
                    Used together with ``self.point_conf_threshold`` to drop
                    low-confidence pixels before the voxel store sees them.
        """
        T, C, H_f, W_f = feat.shape
        pts_dn    = self._downsample_pts(pts_world.float(), H_f, W_f)
        pts_flat  = pts_dn.reshape(-1, 3)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C).float()
        conf_flat = conf.squeeze(1).reshape(-1).float()

        valid = pts_flat.norm(dim=-1) > 1e-3

        if geom_conf is not None and self.point_conf_threshold is not None:
            # Downsample geom_conf to feature resolution and apply threshold.
            gc = geom_conf.float().unsqueeze(1)                          # [T,1,H,W]
            gc = F.interpolate(gc, size=(H_f, W_f),
                               mode='bilinear', align_corners=False)
            gc_flat = gc.reshape(-1)                                     # [T*H_f*W_f]
            valid = valid & (gc_flat >= self.point_conf_threshold)

        store.update(pts_flat[valid], feat_flat[valid], conf_flat[valid])

    def _push_sem_ins(self, sem_res: dict, pts_world: torch.Tensor,
                      sem_feat_key: str, geom_conf: torch.Tensor = None):
        """Push semantic + instance features for a batch of frames.

        ``geom_conf`` (if given) is the per-frame geometry conf in (0, 1) —
        forwarded to ``_push_features`` so the configured
        ``point_conf_threshold`` filter applies.
        """
        sem_feat = sem_res.get(sem_feat_key, sem_res['semantic_feat'])[0].cpu().float()
        sem_conf = sem_res['semantic_conf'][0].cpu().float()
        ins_feat = sem_res['instance_feat'][0].cpu().float()
        ins_conf = sem_res['instance_conf'][0].cpu().float()

        if self.sem_store is not None:
            self._push_features(self.sem_store, pts_world, sem_feat, sem_conf,
                                geom_conf=geom_conf)
        if self.ins_store is not None:
            self._push_features(self.ins_store, pts_world, ins_feat, ins_conf,
                                geom_conf=geom_conf)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def sem_count(self) -> int:
        if self.sem_store is None:
            return 0
        return (self.sem_store.num_voxels
                if isinstance(self.sem_store, VoxelFeatureMap)
                else self.sem_store.num_points)

    @property
    def ins_count(self) -> int:
        if self.ins_store is None:
            return 0
        return (self.ins_store.num_voxels
                if isinstance(self.ins_store, VoxelFeatureMap)
                else self.ins_store.num_points)
