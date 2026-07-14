"""
slam_opense3r/memory_stream.py
==============================
Map memory for the streaming-geometry SLAM pipeline (LingBot-Map as
geometry backbone).

Why a separate class instead of subclassing OpenSe3rQueryMemory?
----------------------------------------------------------------
The DA3 / VGGT pipeline has a per-chunk geometry pass that returns pts
in a chunk-local frame; OpenSe3rMemory then runs `coordinate_alignment`
and a confidence-weighted merge against the global map.  In streaming
mode (LingBot) the geometry backbone already maintains an anchor-frame
global coordinate system internally: every chunk's pts/poses come out
in the SAME global frame, so there is nothing to align and no need to
merge across iterations.

So this class strips everything that depends on that workflow:

  * No `coordinate_alignment` / `interpolate_poses` / scale hinting
  * No keyframe selection / `cur_kf_idx` / `kf_idx` management
  * No `_pending_pts_new` cache (each chunk is self-contained: caller
    passes pts/conf directly into the instance-table update)
  * No confidence-weighted pts/poses merge — just overwrite

What is kept:

  * Per-frame buffers (`pts`, `conf`, `poses`, `iter`) so the existing
    export / eval / vis code (`amb3r/vis_query_instance_slam.py`,
    `slam_opense3r/run.py` T_align block) reads from the same fields.
  * `sem_store` (VoxelFeatureMap / PerPointFeatureStore) shared with the
    legacy memory class — same feature schema, same downstream consumers.
  * `instance_table` (InstanceTable) — query-head-only instance store.
  * Confidence transform (`_conf_sig` = (c-1)/c, same as OpenSe3rMemory)
    so the conf scale matches downstream tools.

Compatibility-only fields (kept zero but present):
  * `cur_kf_idx`, `kf_idx` — empty tensors.  Some external tools call
    `.tolist()` on them; empty list is fine.
  * `ins_store` is always None (query-head only).
  * `ins_feat_dim = 0`.

Query mode only.  Dense `instance_feat` is rejected with an explicit
error if encountered — keeps the contract obvious.
"""

from __future__ import annotations
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

# `slam_opense3r.memory` (imported below) pulls in `amb3r.tools.pts_align`,
# which in turn imports `moge.moge.train.losses` from the bundled
# thirdparty namespace package.  The legacy pipeline does the same path
# munging at the top of pipeline.py — mirror it here so users can import
# memory_stream directly without going through pipeline first.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.append(_root)
_thirdparty = os.path.join(_root, 'thirdparty')
if _thirdparty not in sys.path:
    sys.path.append(_thirdparty)

from slam_semantic.semantic_voxel_map import VoxelFeatureMap
from .memory            import PerPointFeatureStore
from .instance_table    import InstanceTable
from opense3r.instance_post import process_chunk_to_proposals


class OpenSe3rStreamMemory:
    """Query-head SLAM memory for the streaming (LingBot) pipeline.

    Construction args mirror `OpenSe3rQueryMemory` so the same `run.py`
    knobs (CLI flags) work without translation.
    """

    def __init__(
        self,
        cfg                = None,
        num_frames:          int   = 0,
        H:                   int   = 0,
        W:                   int   = 0,
        sem_feat_dim:        int   = 1024,
        feat_storage:        str   = 'voxel',
        sem_voxel_size:      float = 0.01,
        instance_voxel_size: float = 0.05,
        knn_radius:          float = 2.0,
        match_threshold:     float = 0.35,
        w_desc:              float = 0.35,
        w_sem:               float = 0.35,
        w_vox:               float = 0.15,
        w_geo:               float = 0.15,
        sigma_dist:          float = 0.5,
        obj_threshold:       float = 0.5,
        obj_top_k:           Optional[int] = None,   # 0/None = off (threshold-only)
        nms_iou:             float = 0.5,
        score_threshold:     float = 0.25,
        outlier_iqr_mult:    float = 1.5,
        min_points:          int   = 0,
        vox_gate:            Optional[float] = None,
        desc_gate:           Optional[float] = None,
        score_combine:       str   = 'add',
        sem_gate_thr:        float = 0.0,
        vox_bonus_alpha:     float = 1.0,
        desc_pow:            float = 1.0,
        vox_or_tau:          float = 0.5,
        backend:             str   = 'lingbot',
        point_conf_threshold: Optional[float] = None,
    ):
        self.cfg          = cfg
        self.num_frames   = int(num_frames)
        self.H, self.W    = int(H), int(W)
        self.feat_storage = feat_storage
        self.sem_feat_dim = int(sem_feat_dim)
        self.ins_feat_dim = 0                                   # query-only
        self.backend      = backend
        # LingBot is canonical-scale; the model normalises depth/world_pts
        # internally so we never need a metric-scale alignment hint.
        self.is_metric    = False
        self.point_conf_threshold = (
            float(point_conf_threshold) if point_conf_threshold is not None else None
        )

        # Per-frame buffers (same schema as OpenSe3rMemory so downstream
        # code reads from the same fields).
        self.pts   = torch.zeros(num_frames, H, W, 3)
        self.conf  = torch.zeros(num_frames, H, W)
        self.poses = torch.zeros(num_frames, 4, 4)
        # `iter` is per-frame "has been written to" flag — kept as float so
        # eval/vis code that reads it (e.g. for masking unwritten frames)
        # behaves the same as in OpenSe3rMemory.
        self.iter  = torch.zeros(num_frames)

        # Keyframe-related fields: streaming pipeline does not pick keyframes
        # (LingBot handles its own anchor + sliding window internally).  Kept
        # as empty tensors so any external `.tolist()` / `.numel()` works.
        self.kf_idx     = torch.zeros(0, dtype=torch.long)
        self.cur_kf_idx = torch.zeros(0, dtype=torch.long)

        # Sem feature store (shared with legacy memory).
        if feat_storage == 'voxel':
            self.sem_store = VoxelFeatureMap(sem_voxel_size, self.sem_feat_dim)
        else:
            self.sem_store = PerPointFeatureStore(self.sem_feat_dim)
        # Dense instance store is unused in query-only mode.  Kept as
        # attribute = None so `getattr(memory, 'ins_store', None)` and
        # similar lookups in mem_stats / run.py work unchanged.
        self.ins_store = None

        # Persistent InstanceTable — same constructor as OpenSe3rQueryMemory.
        self.instance_table = InstanceTable(
            voxel_size      = instance_voxel_size,
            knn_radius      = knn_radius,
            match_threshold = match_threshold,
            w_desc          = w_desc,
            w_sem           = w_sem,
            w_vox           = w_vox,
            w_geo           = w_geo,
            sigma_dist      = sigma_dist,
            vox_gate        = vox_gate,
            desc_gate       = desc_gate,
            score_combine   = score_combine,
            sem_gate_thr    = sem_gate_thr,
            vox_bonus_alpha = vox_bonus_alpha,
            desc_pow        = desc_pow,
            vox_or_tau      = vox_or_tau,
        )
        self.obj_threshold    = float(obj_threshold)
        # Normalise 0 / negative → None (disabled).
        self.obj_top_k        = int(obj_top_k) if (obj_top_k and int(obj_top_k) > 0) else None
        self.nms_iou          = float(nms_iou)
        self.score_threshold  = float(score_threshold)
        self.outlier_iqr_mult = float(outlier_iqr_mult)
        self.min_points       = int(min_points) if (min_points and int(min_points) > 0) else 0

    # ------------------------------------------------------------------
    # Confidence transform — identical to OpenSe3rMemory so downstream
    # thresholds compare apples to apples across pipelines.
    # ------------------------------------------------------------------

    @staticmethod
    def _conf_sig(conf_raw: torch.Tensor) -> torch.Tensor:
        cs = (conf_raw - 1.0) / conf_raw.clamp(min=1e-6)
        cs[cs <= 0] = 1e-6
        return cs

    # ------------------------------------------------------------------
    # Writes: streaming pipeline drops a whole chunk in one call.
    # ------------------------------------------------------------------

    def initialize(self, geo_res: dict, sem_res: dict, sem_feat_key: str):
        """Store the scale-phase chunk's geometry and push its sem voxels.

        `geo_res` follows the schema of `OpenSe3rDA3.forward_geometry`
        plus a key `world_points_conf` already in [B,T,H,W,1] shape.  The
        pipeline derives this dict from LingBot outputs.
        """
        n_init = geo_res['world_points'].shape[1]
        frame_indices = torch.arange(0, n_init)

        self._write_geo_buffers(
            frame_indices,
            geo_res['world_points'][0].cpu().float(),
            geo_res['poses'       ][0].cpu().float(),
            geo_res['world_points_conf'][0].cpu().squeeze(-1).float(),
        )
        if sem_res is not None:
            self._push_sem(
                sem_res,
                pts_world = self.pts[:n_init],
                geom_conf = self.conf[:n_init],
                sem_feat_key = sem_feat_key,
            )

    def push_chunk(self,
                   geo_res:      dict,
                   sem_res:      dict,
                   start_idx:    int,
                   end_idx:      int,
                   sem_feat_key: str):
        """Streaming chunk update.

        Order matches the legacy `update_geo → update_instances →
        update_sem` cycle but without the alignment / KF management
        layer: write geo, update instances using the same pts, push
        sem voxels.

        `start_idx`, `end_idx` are global frame indices (inclusive).
        `geo_res` / `sem_res` cover the chunk's NEW frames only.
        """
        n_new = end_idx - start_idx + 1
        if geo_res['world_points'].shape[1] != n_new:
            raise ValueError(
                f"push_chunk: geo_res has {geo_res['world_points'].shape[1]} "
                f"frames but chunk spans {n_new} ({start_idx}..{end_idx})."
            )

        frame_indices = torch.arange(start_idx, end_idx + 1)
        pts_chunk  = geo_res['world_points'][0].cpu().float()                  # [T,H,W,3]
        c2w_chunk  = geo_res['poses'       ][0].cpu().float()                   # [T,4,4]
        conf_chunk = self._conf_sig(
            geo_res['world_points_conf'][0].cpu().squeeze(-1).float()
        )                                                                       # [T,H,W]

        self._write_geo_buffers(frame_indices, pts_chunk, c2w_chunk, conf_chunk)

        # Instance update (reads pts/conf directly; no _pending_* cache).
        self.update_instances(
            sem_res, start_idx, end_idx, sem_feat_key,
            pts_new  = pts_chunk,
            conf_new = conf_chunk,
        )

        # Sem voxel push.
        self._push_sem(
            sem_res,
            pts_world = pts_chunk,
            geom_conf = conf_chunk,
            sem_feat_key = sem_feat_key,
        )

    def _write_geo_buffers(self,
                           frame_indices: torch.Tensor,
                           pts:           torch.Tensor,
                           poses:         torch.Tensor,
                           conf:          torch.Tensor):
        """Overwrite (no merge) — LingBot output is final."""
        # `conf` may arrive as raw conf or as already-sigmoided.  The
        # caller passes the post-`_conf_sig` value (matches what
        # OpenSe3rMemory stores in `self.conf`).
        self.pts  [frame_indices] = pts
        self.poses[frame_indices] = poses
        self.conf [frame_indices] = conf
        self.iter [frame_indices] = 1

    # ------------------------------------------------------------------
    # Instance update — mirrors OpenSe3rQueryMemory.update_instances but
    # takes pts/conf directly instead of pulling from a `_pending_*` cache.
    # ------------------------------------------------------------------

    def update_instances(self,
                         sem_res:      dict,
                         start_idx:    int,
                         end_idx:      int,
                         sem_feat_key: Optional[str],
                         *,
                         pts_new:      torch.Tensor,
                         conf_new:     torch.Tensor):
        if sem_res is None:
            return
        if 'mask_logits' not in sem_res:
            # Caller is running dense ins mode — not supported here.
            raise ValueError(
                "OpenSe3rStreamMemory.update_instances requires query-head "
                "outputs ('mask_logits' missing).  Streaming pipeline is "
                "query-only."
            )

        n_new = end_idx - start_idx + 1
        device = 'cpu'

        mask_logits = sem_res['mask_logits'][0].detach().to(device).float()    # [T,N,Hf,Wf]
        objectness  = sem_res['objectness'][0].detach().to(device).float()      # [T,N]
        descriptor  = sem_res['descriptor'][0].detach().to(device).float()      # [T,N,D]
        T, N, H_f, W_f = mask_logits.shape
        if T != n_new:
            raise ValueError(
                f"update_instances: head T={T} but chunk has {n_new} frames"
            )

        pts_new  = pts_new.to(device).float()                                   # [T,H,W,3]
        conf_new = conf_new.to(device).float()                                  # [T,H,W]

        if pts_new.shape[1:3] != (H_f, W_f):
            pts_new = self._downsample_pts(pts_new, H_f, W_f)
        if conf_new.shape[1:3] != (H_f, W_f):
            cn = conf_new.unsqueeze(1)
            cn = F.interpolate(cn, size=(H_f, W_f),
                               mode='bilinear', align_corners=False)
            conf_new = cn.squeeze(1)

        keep_mask = (pts_new.norm(dim=-1) > 1e-3)
        if self.point_conf_threshold is not None:
            keep_mask = keep_mask & (conf_new >= self.point_conf_threshold)

        sem_dense = None
        if sem_feat_key is not None and sem_feat_key in sem_res:
            sem_dense = sem_res[sem_feat_key][0].detach().to(device).float()
        elif sem_res.get('semantic_feat_expanded') is not None:
            sem_dense = sem_res['semantic_feat_expanded'][0].detach().to(device).float()
        elif sem_res.get('semantic_feat') is not None:
            sem_dense = sem_res['semantic_feat'][0].detach().to(device).float()

        pred_box    = None
        box_scale_b = None
        poses_t     = None
        if sem_res.get('box') is not None:
            pred_box = sem_res['box'][0].detach().to(device).float()
            bs = sem_res.get('box_scale')
            if bs is not None:
                box_scale_b = float(bs[0].item())
            poses_t = self.poses[start_idx:end_idx + 1].to(device).float()

        proposals, _ = process_chunk_to_proposals(
            mask_logits      = mask_logits,
            objectness       = objectness,
            descriptor       = descriptor,
            world_points     = pts_new,
            keep_mask        = keep_mask,
            obj_threshold    = self.obj_threshold,
            obj_top_k        = self.obj_top_k,
            min_points       = self.min_points,
            nms_iou          = self.nms_iou,
            score_threshold  = self.score_threshold,
            voxel_size       = self.instance_table.voxel_size,
            sem_feat         = sem_dense,
            box              = pred_box,
            box_scale        = box_scale_b,
            poses            = poses_t,
            outlier_iqr_mult = self.outlier_iqr_mult,
        )
        if not proposals:
            return

        frame_indices = [int(p['frame']) + start_idx for p in proposals]
        self.instance_table.match_chunk(proposals, frame_indices)

    # ------------------------------------------------------------------
    # Feature push helpers — same logic as OpenSe3rMemory._push_features /
    # _push_sem_ins, but inlined here to keep the streaming memory
    # independent of the legacy class.
    # ------------------------------------------------------------------

    @staticmethod
    def _downsample_pts(pts: torch.Tensor, H_f: int, W_f: int) -> torch.Tensor:
        p = pts.permute(0, 3, 1, 2).float()
        p = F.interpolate(p, size=(H_f, W_f), mode='bilinear', align_corners=False)
        return p.permute(0, 2, 3, 1)

    def _push_features(self,
                       store,
                       pts_world: torch.Tensor,
                       feat:      torch.Tensor,
                       conf:      torch.Tensor,
                       geom_conf: torch.Tensor = None):
        T, C, H_f, W_f = feat.shape
        pts_dn   = self._downsample_pts(pts_world.float(), H_f, W_f)
        pts_flat = pts_dn.reshape(-1, 3)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, C).float()
        conf_flat = conf.squeeze(1).reshape(-1).float()

        valid = pts_flat.norm(dim=-1) > 1e-3
        if geom_conf is not None and self.point_conf_threshold is not None:
            gc = geom_conf.float().unsqueeze(1)
            gc = F.interpolate(gc, size=(H_f, W_f),
                               mode='bilinear', align_corners=False)
            gc_flat = gc.reshape(-1)
            valid = valid & (gc_flat >= self.point_conf_threshold)

        store.update(pts_flat[valid], feat_flat[valid], conf_flat[valid])

    def _push_sem(self,
                  sem_res:      dict,
                  pts_world:    torch.Tensor,
                  geom_conf:    torch.Tensor,
                  sem_feat_key: str):
        sem_feat = sem_res.get(sem_feat_key, sem_res['semantic_feat'])[0].cpu().float()
        sem_conf = sem_res['semantic_conf'][0].cpu().float()
        self._push_features(self.sem_store, pts_world, sem_feat, sem_conf,
                            geom_conf=geom_conf)

    # ------------------------------------------------------------------
    # Properties / summary (kept matching OpenSe3rQueryMemory)
    # ------------------------------------------------------------------

    @property
    def sem_count(self) -> int:
        return (self.sem_store.num_voxels
                if isinstance(self.sem_store, VoxelFeatureMap)
                else self.sem_store.num_points)

    @property
    def ins_count(self) -> int:
        # No dense ins store in query-only streaming pipeline.
        return 0

    @property
    def instance_count(self) -> int:
        return len(self.instance_table)

    def instance_summary(self) -> Dict[str, int]:
        return self.instance_table.summary()
