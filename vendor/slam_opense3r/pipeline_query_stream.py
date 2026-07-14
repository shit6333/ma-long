"""
slam_opense3r/pipeline_query_stream.py
======================================
Streaming-geometry SLAM pipeline that pairs a frozen LingBot-Map
(GCTStream) geometry backbone with our OpenSe3r semantic / query-instance
branch (`model.forward_semantic`).

How this differs from `pipeline_query.py`
-----------------------------------------
The DA3 / VGGT pipeline runs a one-shot geometry forward per chunk and
then aligns the result to a SLAM-global frame via `coordinate_alignment`
+ confidence-weighted merge.  LingBot, by contrast, IS a streaming model:

  * Its `aggregator` keeps an internal paged KV cache with anchor
    context + sliding window + trajectory-memory tokens.
  * After the scale-phase forward, every subsequent chunk's output is
    already expressed in the anchor-frame global coordinate system.

So this pipeline:

  * Skips `forward_geometry` + `memory.update_geo` + keyframe management
    entirely — LingBot handles all of that internally.
  * Runs ONE phase-1 forward over the first `N_scale` frames (bidirectional
    within phase-1), then chunk-by-chunk causal forwards thereafter.
  * Drives our OpenSe3r `forward_semantic` with the LingBot-supplied
    geometry as inputs — same exact contract as the fast path in
    `pipeline_query.py`, but the geometry source is the LingBot stream.
  * Each branch keeps its own KV cache (LingBot internal vs. OpenSe3r
    CausalBranch passed in as `kv_cache_list`).
  * Only query-head instance mode is supported (no dense `instance_feat`).

Scale note
----------
LingBot-Map outputs are CANONICAL (not metric) and anchored to the
scale-phase frames.  This is fine for us because OpenSe3r's
`forward_semantic` median-normalises depth / world_points / pose
translation internally (`box_scale = depth.median()` per chunk), so the
CausalBranch input distribution is identical to what it sees from
DA3-canonical training.  No `--global_scale_mode` is needed here.
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf

# Mirror the path mutation in slam_opense3r/pipeline.py so the bundled
# `thirdparty/moge` namespace package resolves regardless of which slam
# entry point gets imported first.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.append(_root)
_thirdparty = os.path.join(_root, 'thirdparty')
if _thirdparty not in sys.path:
    sys.path.append(_thirdparty)

from .memory_stream import OpenSe3rStreamMemory
from .              import mem_stats


# ---------------------------------------------------------------------------
# Make the bundled lingbot-map package importable without requiring
# `pip install -e thirdparty/lingbot-map`.
# ---------------------------------------------------------------------------
_THIRDPARTY_LINGBOT = Path(__file__).resolve().parent.parent / 'thirdparty' / 'lingbot-map'
if _THIRDPARTY_LINGBOT.exists() and str(_THIRDPARTY_LINGBOT) not in sys.path:
    sys.path.insert(0, str(_THIRDPARTY_LINGBOT))


def _import_lingbot():
    """Lazy import so the rest of the slam_opense3r package keeps working
    even when the lingbot-map dependency is missing on disk."""
    from lingbot_map.models.gct_stream import GCTStream
    from lingbot_map.utils.pose_enc    import pose_encoding_to_extri_intri
    # `_general` handles arbitrary leading batch dims [..., 4, 4]; the plain
    # `closed_form_inverse_se3` only takes [N, 4, 4] (uses torch.bmm).  We feed
    # [B, T, 4, 4], so use the general variant (same one demo.py uses).
    from lingbot_map.utils.geometry    import closed_form_inverse_se3_general
    return GCTStream, pose_encoding_to_extri_intri, closed_form_inverse_se3_general


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class OpenSe3rStreamQuerySLAM:
    """LingBot (geometry) + OpenSe3r forward_semantic (semantics) SLAM.

    Mirrors the public surface of `OpenSe3rQuerySLAM` so `run.py` can
    branch on a single flag:

        memory = pipeline.run(images)        # images [1, T, 3, H, W] in [-1, 1]
        memory.pts / poses / conf / sem_store / instance_table  # ← export reads same fields

    Construction args — broken down by group:
      * Geometry backbone (`lingbot_model`)
      * Semantic / query branch (`opense3r_model`)
      * Phase / chunking knobs (`num_scale_frames`, `chunk_size`)
      * InstanceTable hyperparams (`**inst_knobs`, same as OpenSe3rQuerySLAM)
    """

    def __init__(
        self,
        lingbot_model,                                 # GCTStream (frozen)
        opense3r_model,                                # OpenSe3rDA3 / OpenSe3r
        cfg_path:              str   = './slam_opense3r/config.yaml',
        sem_feat_mode:         str   = 'expanded',
        feat_storage:          str   = 'voxel',
        # Streaming knobs.
        num_scale_frames:      int   = 8,
        chunk_size:            int   = 8,
        attn_mode:             str   = 'causal',       # OpenSe3r CausalBranch within-chunk attn
        # Geometry-driving mode:
        #   'whole_seq' (default): run LingBot's proven `inference_streaming`
        #       over the WHOLE sequence once (frame-by-frame phase-2, exactly
        #       like the LingBot demo) → geometry is bit-identical to the
        #       reference pipeline.  Since LingBot geometry does NOT depend on
        #       OpenSe3r, the final map is identical to interleaving; only
        #       memory/latency differ (geo offloaded to CPU, sliced per chunk).
        #   'interleaved': drive LingBot per-chunk via `forward` (online, but
        #       re-implements the streaming loop — keep for experiments).
        geo_mode:              str   = 'whole_seq',
        # Sequence-global scale for OpenSe3r's depth / pos3d / pose-translation
        # normalisation.  'first_chunk' (default) captures the box_scale from
        # the scale-phase (anchor) frames and reuses it for the whole sequence,
        # which (a) keeps cross-chunk KV-cache token magnitudes consistent and
        # (b) preserves LingBot's anchor-relative depth (per_chunk would flatten
        # every chunk to median≈1, discarding the global geometry).  'per_chunk'
        # reverts to the legacy per-forward depth median.
        global_scale_mode:     str   = 'first_chunk',
        point_conf_threshold:  Optional[float] = None,
        # InstanceTable knobs (CLI-overridable, defaults mirror OpenSe3rQuerySLAM).
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
        vox_gate:              Optional[float] = None,
        desc_gate:             Optional[float] = None,
        score_combine:         str   = 'add',
        sem_gate_thr:          float = 0.0,
        vox_bonus_alpha:       float = 1.0,
        desc_pow:              float = 1.0,
        vox_or_tau:            float = 0.5,
    ):
        if sem_feat_mode not in ('expanded', 'compact'):
            raise ValueError(f"sem_feat_mode must be 'expanded' or 'compact'")
        if feat_storage not in ('voxel', 'perpoint'):
            raise ValueError(f"feat_storage must be 'voxel' or 'perpoint'")
        if attn_mode not in ('causal', 'chunk_bidir'):
            raise ValueError(
                f"attn_mode must be 'causal' or 'chunk_bidir', got {attn_mode!r}"
            )
        # gt_seq_median is meaningless without GT depths; streaming is pred-only.
        if global_scale_mode not in ('per_chunk', 'first_chunk'):
            raise ValueError(
                f"global_scale_mode for the streaming pipeline must be "
                f"'per_chunk' or 'first_chunk' (got {global_scale_mode!r}; "
                f"'gt_seq_median' is not available without GT)."
            )
        if geo_mode not in ('whole_seq', 'interleaved'):
            raise ValueError(
                f"geo_mode must be 'whole_seq' or 'interleaved', got {geo_mode!r}"
            )

        self.cfg                  = OmegaConf.load(cfg_path)
        device                    = self.cfg.device
        self.lingbot              = lingbot_model.to(device).eval()
        self.opense3r             = opense3r_model.to(device).eval()
        self.sem_feat_mode        = sem_feat_mode
        self.feat_storage         = feat_storage
        self.num_scale_frames     = int(num_scale_frames)
        self.chunk_size           = int(chunk_size)
        self.attn_mode            = attn_mode
        self.geo_mode             = geo_mode
        self.global_scale_mode    = global_scale_mode
        self.point_conf_threshold = point_conf_threshold
        # Captured after the scale phase when global_scale_mode='first_chunk'.
        # None ⇒ let the model use its per-forward depth median.
        self._seq_box_scale: Optional[torch.Tensor] = None

        # Instance knobs — stored on self so `_build_memory` reads them.
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
        self.sem_gate_thr        = float(sem_gate_thr)
        self.vox_bonus_alpha     = float(vox_bonus_alpha)
        self.desc_pow            = float(desc_pow)
        self.vox_or_tau          = float(vox_or_tau)

        # Per-step GPU peak / KV-cache trackers (read by `mem_stats.format_mem_log`).
        self._mem_peaks: dict = {}
        self._last_kv_cache   = None

        # Resolve LingBot helpers once.
        (_GCTStreamCls,
         self._pose_enc_to_extri_intri,
         self._w2c_to_c2w) = _import_lingbot()

    # ------------------------------------------------------------------

    def _sem_feat_key(self) -> str:
        return ('semantic_feat_expanded' if self.sem_feat_mode == 'expanded'
                else 'semantic_feat')

    # ------------------------------------------------------------------
    # LingBot forward → geometry dict in the format `forward_semantic` wants.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _lingbot_forward(self,
                          imgs_neg1: torch.Tensor,
                          num_frame_per_block: int,
                          ) -> Dict[str, torch.Tensor]:
        """Run a chunk through the LingBot streaming model.

        `imgs_neg1`: [B, T_chunk, 3, H, W] in [-1, 1] (SLAM convention).
        LingBot expects [0, 1], so the conversion happens here.

        Returns a dict with keys matching what `forward_semantic` needs:
            depth             [B, T, H, W]
            depth_conf        [B, T, H, W]
            intrinsics        [B, T, 3, 3]
            poses             [B, T, 4, 4]   c2w in LingBot's global frame
            world_points      [B, T, H, W, 3]   unprojected from (depth, K, c2w)
            world_points_conf [B, T, H, W, 1]   = depth_conf.unsqueeze(-1)
        """
        device = self.cfg.device
        imgs_01 = ((imgs_neg1 + 1.0) / 2.0).to(device)
        B, T, _, H, W = imgs_01.shape

        with mem_stats.GpuPeakTracker(self._mem_peaks, 'geo'), \
                torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = self.lingbot.forward(
                imgs_01,
                num_frame_for_scale  = self.num_scale_frames,
                num_frame_per_block  = num_frame_per_block,
                causal_inference     = True,
            )

        # LingBot's `pose_enc` is [B, T, 9]; helper returns world-to-camera
        # 3x4 extrinsics + 3x3 intrinsics.  Convert w2c → c2w to match the
        # `forward_semantic` contract (and OpenSe3rMemory.poses convention).
        pose_enc = out['pose_enc']                                          # [B, T, 9]
        extrinsics_w2c, intrinsics = self._pose_enc_to_extri_intri(
            pose_enc, image_size_hw=(H, W),
        )                                                                   # [B, T, 3, 4], [B, T, 3, 3]
        # Pad w2c to 4x4 then invert → c2w
        w2c_44 = torch.zeros(
            *extrinsics_w2c.shape[:-2], 4, 4,
            device=extrinsics_w2c.device, dtype=extrinsics_w2c.dtype,
        )
        w2c_44[..., :3, :4] = extrinsics_w2c
        w2c_44[..., 3, 3]   = 1.0
        c2w = self._w2c_to_c2w(w2c_44)                                       # [B, T, 4, 4]

        # `depth` from LingBot has shape [B, T, H, W, 1].
        depth_5d = out['depth']
        depth_4d = depth_5d.squeeze(-1)
        depth_conf = out['depth_conf']                                       # [B, T, H, W]
        # Per-pixel world points — LingBot's default config doesn't enable
        # the point head, so unproject here using its (depth, K, c2w).
        world_points = self._unproject_depth(depth_4d, intrinsics, c2w)      # [B, T, H, W, 3]

        return {
            'depth':             depth_4d,
            'depth_conf':        depth_conf,
            'intrinsics':        intrinsics,
            'poses':             c2w,
            'world_points':      world_points,
            'world_points_conf': depth_conf.unsqueeze(-1),
        }

    @torch.no_grad()
    def _lingbot_geometry_whole_seq(self,
                                    images_neg1: torch.Tensor,
                                    ) -> Dict[str, torch.Tensor]:
        """Run LingBot's proven `inference_streaming` over the WHOLE sequence
        and derive the full per-frame geometry (CPU-offloaded).

        This matches the LingBot demo exactly (phase-1 scale block + phase-2
        frame-by-frame with KV cache), so the geometry is bit-identical to the
        reference pipeline.  Returns CPU tensors; `run()` slices per chunk and
        moves the slice to GPU for `forward_semantic`.

        Returns dict (all on CPU):
            depth             [1, T, H, W]
            depth_conf        [1, T, H, W]
            intrinsics        [1, T, 3, 3]
            poses             [1, T, 4, 4]   c2w
            world_points      [1, T, H, W, 3]
            world_points_conf [1, T, H, W, 1]
        """
        device = self.cfg.device
        _, T, _, H, W = images_neg1.shape
        imgs_01 = ((images_neg1 + 1.0) / 2.0)[0]          # [T, 3, H, W] in [0,1]

        with mem_stats.GpuPeakTracker(self._mem_peaks, 'geo'), \
                torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            preds = self.lingbot.inference_streaming(
                imgs_01.to(device),
                num_scale_frames = self.num_scale_frames,
                keyframe_interval = 1,
                output_device    = torch.device('cpu'),
            )
        # preds: pose_enc [1,T,9], depth [1,T,H,W,1], depth_conf [1,T,H,W]  (CPU)
        pose_enc   = preds['pose_enc'].float()                          # [1, T, 9]
        depth_4d   = preds['depth'].float().squeeze(-1)                 # [1, T, H, W]
        depth_conf = preds['depth_conf'].float()                        # [1, T, H, W]

        extrinsics_w2c, intrinsics = self._pose_enc_to_extri_intri(
            pose_enc, image_size_hw=(H, W),
        )                                                               # [1,T,3,4], [1,T,3,3]
        w2c_44 = torch.zeros(*extrinsics_w2c.shape[:-2], 4, 4,
                             dtype=extrinsics_w2c.dtype)
        w2c_44[..., :3, :4] = extrinsics_w2c
        w2c_44[..., 3, 3]   = 1.0
        c2w = self._w2c_to_c2w(w2c_44)                                  # [1, T, 4, 4]

        # Unproject on CPU in float32 (cheap, one-shot).
        world_points = self._unproject_depth(depth_4d, intrinsics, c2w)  # [1, T, H, W, 3]

        return {
            'depth':             depth_4d,
            'depth_conf':        depth_conf,
            'intrinsics':        intrinsics,
            'poses':             c2w,
            'world_points':      world_points,
            'world_points_conf': depth_conf.unsqueeze(-1),
        }

    @staticmethod
    def _slice_geo(geo_all: Dict[str, torch.Tensor],
                   start_idx: int, end_idx: int,
                   device) -> Dict[str, torch.Tensor]:
        """Slice the whole-sequence geo dict to frames [start_idx, end_idx]
        (inclusive) and move to `device` for forward_semantic."""
        s = slice(start_idx, end_idx + 1)
        return {
            'depth':             geo_all['depth'            ][:, s].to(device),
            'depth_conf':        geo_all['depth_conf'       ][:, s].to(device),
            'intrinsics':        geo_all['intrinsics'       ][:, s].to(device),
            'poses':             geo_all['poses'            ][:, s].to(device),
            'world_points':      geo_all['world_points'     ][:, s].to(device),
            'world_points_conf': geo_all['world_points_conf'][:, s].to(device),
        }

    @staticmethod
    def _unproject_depth(depth: torch.Tensor,
                         intrinsics: torch.Tensor,
                         c2w: torch.Tensor) -> torch.Tensor:
        """[B,T,H,W] depth + [B,T,3,3] K + [B,T,4,4] c2w → [B,T,H,W,3] world pts.

        Mirrors `lingbot_map.models.gct_base._unproject_depth_to_world` but
        keeps the implementation here so we don't depend on a private method.
        """
        B, T, H, W = depth.shape
        device = depth.device
        # Use float32 for the unprojection to avoid bf16 numerical noise on
        # the K^-1 / matmul; output is cast back to depth's dtype.
        with torch.amp.autocast('cuda', enabled=False):
            depth_f = depth.float().reshape(B * T, H, W, 1)
            K_f     = intrinsics.float().reshape(B * T, 3, 3)
            c2w_f   = c2w.float().reshape(B * T, 4, 4)

            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij',
            )
            pix = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1)  # [H, W, 3]
            K_inv = torch.inverse(K_f)                                            # [BT, 3, 3]
            cam_rays = torch.einsum('bij,hwj->bhwi', K_inv, pix)                  # [BT, H, W, 3]
            cam_pts  = cam_rays * depth_f                                         # [BT, H, W, 3]
            ones = torch.ones_like(cam_pts[..., :1])
            cam_h = torch.cat([cam_pts, ones], dim=-1)                            # [BT, H, W, 4]
            world_h = torch.einsum('bij,bhwj->bhwi', c2w_f, cam_h)                # [BT, H, W, 4]
            world = world_h[..., :3].reshape(B, T, H, W, 3)
        return world.to(depth.dtype)

    # ------------------------------------------------------------------
    # OpenSe3r forward_semantic
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sem_forward(self,
                     imgs_neg1: torch.Tensor,
                     geo:       Dict[str, torch.Tensor],
                     kv_cache:  Optional[List]) -> Tuple[dict, list]:
        """Drive `opense3r.forward_semantic` with the LingBot-supplied geo.

        Pipeline contract:
          - `imgs_neg1` stays in [-1, 1] (`forward_semantic` adds +1 then /2 inside).
          - `geo` is the dict produced by `_lingbot_forward`.
          - `kv_cache` accumulates across chunks (OpenSe3r CausalBranch
            internal cache — separate from LingBot's geo KV).
          - `box_scale_override` is the cached sequence-global scale when
            `global_scale_mode='first_chunk'` (None ⇒ per-forward median).
        """
        with mem_stats.GpuPeakTracker(self._mem_peaks, 'sem'), \
                torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = self.opense3r.forward_semantic(
                {'images': imgs_neg1.to(self.cfg.device)},
                depth              = geo['depth'],
                intrinsics         = geo['intrinsics'],
                c2w                = geo['poses'],
                world_points       = geo['world_points'],
                world_points_conf  = geo['world_points_conf'],
                kv_cache_list      = kv_cache,
                mode               = self.attn_mode,
                box_scale_override = self._box_scale_override(),
                return_poses       = False,
            )
        new_kv = out.pop('kv_cache', kv_cache)
        self._last_kv_cache = new_kv
        return out, new_kv

    # ------------------------------------------------------------------
    # Sequence-global scale (first_chunk mode) — capture once, reuse.
    # ------------------------------------------------------------------

    def _box_scale_override(self) -> Optional[torch.Tensor]:
        """Override tensor to pass to forward_semantic for the current chunk,
        or None to let the model use its own per-forward depth median."""
        if self._seq_box_scale is None:
            return None
        return self._seq_box_scale.to(self.cfg.device)

    def _capture_seq_box_scale(self, sem_res: dict):
        """Stash the scale-phase box_scale so phase-2 chunks reuse it.
        No-op outside 'first_chunk' mode or if already captured."""
        if self.global_scale_mode != 'first_chunk':
            return
        if self._seq_box_scale is not None:
            return
        bs = sem_res.get('box_scale', None)
        if bs is None:
            print("[StreamQuerySLAM] WARN: scale phase returned no 'box_scale' "
                  "— staying in per-chunk scale for the rest of the run.")
            return
        self._seq_box_scale = bs.detach().float().cpu().clone()
        print(f"[StreamQuerySLAM] global_scale_mode=first_chunk → "
              f"box_scale={self._seq_box_scale.tolist()}")

    # ------------------------------------------------------------------
    # Memory builder
    # ------------------------------------------------------------------

    def _build_memory(self,
                      T:            int,
                      H:            int,
                      W:            int,
                      sem_feat_dim: int) -> OpenSe3rStreamMemory:
        return OpenSe3rStreamMemory(
            cfg                  = self.cfg,
            num_frames           = T,
            H = H, W = W,
            sem_feat_dim         = sem_feat_dim,
            feat_storage         = self.feat_storage,
            sem_voxel_size       = float(self.cfg.semantic_voxel_size),
            instance_voxel_size  = self.instance_voxel_size,
            knn_radius           = self.knn_radius,
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
            score_combine        = self.score_combine,
            sem_gate_thr         = self.sem_gate_thr,
            vox_bonus_alpha      = self.vox_bonus_alpha,
            desc_pow             = self.desc_pow,
            vox_or_tau           = self.vox_or_tau,
            backend              = 'lingbot',
            point_conf_threshold = self.point_conf_threshold,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, images: torch.Tensor) -> OpenSe3rStreamMemory:
        """images: [1, T, 3, H, W] in [-1, 1]."""
        assert images.shape[0] == 1
        assert images.min() >= -1.0 and images.max() <= 1.0
        _, T, _, H, W = images.shape
        device = self.cfg.device

        N_scale = max(1, min(self.num_scale_frames, T))
        chunk   = max(1, int(self.chunk_size))
        sem_feat_key = self._sem_feat_key()

        # Wipe LingBot's KV cache so a new sequence starts clean.
        if hasattr(self.lingbot, 'clean_kv_cache'):
            self.lingbot.clean_kv_cache()

        # ── Geometry ─────────────────────────────────────────────────────────
        # 'whole_seq' (default): one demo-exact inference_streaming over the
        # full sequence (geo on CPU, sliced per chunk).  LingBot geometry is
        # independent of OpenSe3r, so the final map equals interleaving.
        geo_all = None
        if self.geo_mode == 'whole_seq':
            print(f"[StreamQuerySLAM] geo_mode=whole_seq → running LingBot "
                  f"inference_streaming over {T} frames (N_scale={N_scale})…")
            geo_all = self._lingbot_geometry_whole_seq(images)

        def _geo_for(start_idx, end_idx, n_block):
            """Return the geo dict for frames [start_idx, end_idx] on GPU."""
            if geo_all is not None:
                return self._slice_geo(geo_all, start_idx, end_idx, device)
            return self._lingbot_forward(
                images[:, start_idx:end_idx + 1], num_frame_per_block=n_block,
            )

        # ── Probe + Phase 1 (scale frames) ────────────────────────────────────
        geo_scale = _geo_for(0, N_scale - 1, N_scale)
        kv_cache  = self.opense3r.init_kv_cache()
        sem_scale, kv_cache = self._sem_forward(images[:, :N_scale], geo_scale, kv_cache)
        if 'mask_logits' not in sem_scale:
            raise RuntimeError(
                "OpenSe3rStreamQuerySLAM requires a query-head OpenSe3r model "
                "(instance_head_type='query'); probe returned no 'mask_logits'."
            )
        sem_feat_dim = sem_scale.get(
            sem_feat_key, sem_scale['semantic_feat'],
        ).shape[2]

        # Capture the scale-phase (anchor) box_scale so every later chunk
        # normalises depth / pos3d / pose by the SAME divisor (first_chunk mode).
        self._capture_seq_box_scale(sem_scale)

        memory = self._build_memory(T, H, W, sem_feat_dim)
        print(f"[StreamQuerySLAM] init  T={T}  N_scale={N_scale}  chunk={chunk}  "
              f"attn_mode='{self.attn_mode}'  geo_mode='{self.geo_mode}'")
        print(f"[StreamQuerySLAM] instance_table: knn={self.knn_radius}m  "
              f"threshold={self.match_threshold}  "
              f"w desc/sem/vox/geo="
              f"{self.w_desc}/{self.w_sem}/{self.w_vox}/{self.w_geo}")
        if self.point_conf_threshold is not None:
            print(f"[StreamQuerySLAM] push-time geom conf threshold = "
                  f"{self.point_conf_threshold:.4f}")

        memory.initialize(geo_scale, sem_scale, sem_feat_key)
        memory.update_instances(
            sem_scale, start_idx=0, end_idx=N_scale - 1,
            sem_feat_key=sem_feat_key,
            pts_new  = geo_scale['world_points'][0].cpu().float(),
            conf_new = memory._conf_sig(
                geo_scale['world_points_conf'][0].cpu().squeeze(-1).float()
            ),
        )
        del geo_scale, sem_scale
        t0 = time.time()
        self._log(N_scale - 1, T, memory, t0)

        # ── Phase 2: streaming chunks (semantic).  OpenSe3r's CausalBranch KV
        # carries over; geometry is either sliced from geo_all or re-forwarded.
        for start_idx in range(N_scale, T, chunk):
            end_idx   = min(start_idx + chunk, T) - 1
            T_chunk   = end_idx - start_idx + 1
            chunk_imgs = images[:, start_idx:end_idx + 1]

            geo_chunk = _geo_for(start_idx, end_idx, T_chunk)
            sem_chunk, kv_cache = self._sem_forward(
                chunk_imgs, geo_chunk, kv_cache,
            )

            memory.push_chunk(
                geo_chunk, sem_chunk,
                start_idx=start_idx, end_idx=end_idx,
                sem_feat_key=sem_feat_key,
            )
            del geo_chunk, sem_chunk
            self._log(end_idx, T, memory, t0)

        return memory

    # ------------------------------------------------------------------
    # Status / logging
    # ------------------------------------------------------------------

    def _log(self, idx: int, T: int, memory: OpenSe3rStreamMemory, t0: float):
        elapsed = time.time() - t0
        fps     = max(1, idx + 1) / max(elapsed, 1e-6)
        storage = 'voxels' if self.feat_storage == 'voxel' else 'pts'
        print(f"[StreamQuerySLAM] frame {idx+1:4d}/{T}"
              f"  sem {storage}: {memory.sem_count:,}"
              f"  instances: {memory.instance_count}"
              f"  {fps:.1f} fps")
        print(mem_stats.format_mem_log(
            memory,
            peaks         = self._mem_peaks,
            kv_cache_list = self._last_kv_cache,
        ))


# ---------------------------------------------------------------------------
# Top-level LingBot loader.  Kept here so run.py just imports one symbol.
# ---------------------------------------------------------------------------

def build_lingbot(
    ckpt_path:               str,
    *,
    image_size:              int   = 518,
    patch_size:              int   = 14,
    num_scale_frames:        int   = 3,
    kv_cache_sliding_window: int   = 64,
    enable_3d_rope:          bool  = True,
    max_frame_num:           int   = 4096,
    use_sdpa:                bool  = True,
    device:                  str   = 'cuda',
):
    """Build a `GCTStream` model and load the LingBot-Map checkpoint.

    SDPA backend is the default here (vs. FlashInfer) so the pipeline runs
    without that extra dependency.  Set `use_sdpa=False` to opt into
    FlashInfer if installed.
    """
    GCTStream, _, _ = _import_lingbot()
    model = GCTStream(
        img_size                       = image_size,
        patch_size                     = patch_size,
        enable_3d_rope                 = enable_3d_rope,
        max_frame_num                  = max_frame_num,
        kv_cache_sliding_window        = kv_cache_sliding_window,
        kv_cache_scale_frames          = num_scale_frames,
        kv_cache_cross_frame_special   = True,
        kv_cache_include_scale_frames  = True,
        use_sdpa                       = use_sdpa,
        # Default head config: depth + camera (no point head — we unproject in pipeline).
    )
    print(f"[LingBot] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[LingBot]   missing keys: {len(missing)}")
    if unexpected:
        print(f"[LingBot]   unexpected keys: {len(unexpected)}")
    return model.to(device).eval()
