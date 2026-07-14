"""
pipeline.py — OpenSe3r SLAM with predicted geometry.
=====================================================
Architecture follows AMB3R's slam/pipeline.py closely:

  Geometry step  → forward_geometry([keyframes + new_frames])
                   — VGGT bidirectional, no KV cache, arbitrary context
                   — coordinate_alignment brings local → global frame
                   — confidence-weighted merge of pts/poses in memory

  Semantic step  → forward_chunk([new_frames only], kv_cache)
                   — CausalBranch with growing KV cache
                   — KV cache implicitly encodes all past frames: no need to
                     re-feed keyframe images for semantic features

Key differences from AMB3R:
  ✓ Geometry forward is VGGT-only (forward_geometry), separated from semantics
  ✓ Semantic forward uses KV cache (stateful), AMB3R is stateless
  ✓ Semantic features are only computed for new frames each step
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from typing import Optional

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_root)
sys.path.append(os.path.join(_root, 'thirdparty'))

from slam_opense3r.memory import OpenSe3rMemory
from slam_opense3r import mem_stats


# ─────────────────────────────────────────────────────────────────────────────
# KV-cache helpers  (exported so pipeline_gt.py can reuse)
# ─────────────────────────────────────────────────────────────────────────────

def _prune_kv_cache(
    kv_cache_list,
    N_patch: int,
    window_size: Optional[int] = None,
    keyframe_stride: Optional[int] = None,
):
    """
    Prune each global block's KV cache to a union of three subsets:

        anchor    : the very first frame (absolute index 0)
        keyframes : frames {S, 2S, 3S, ...} from absolute index 0, only inside
                    the region BEFORE the sliding window (so the window region
                    is always kept verbatim, never re-sampled).  Disabled when
                    ``keyframe_stride`` is None.
        window    : the last ``window_size`` frames (sliding tail).  Disabled
                    when ``window_size`` is None.

    Either or both of ``window_size`` and ``keyframe_stride`` may be set:
      - both None     → no-op (caller should normally skip the call entirely)
      - window only   → legacy anchor + window behaviour
      - stride only   → anchor + every-S-th past frame retained, no recency tail
      - both set      → anchor + sparse keyframes in mid-region + sliding tail
    """
    if window_size is None and keyframe_stride is None:
        return kv_cache_list

    for layer_cache in kv_cache_list:
        for ki in range(2):
            tensor = layer_cache[ki]
            if tensor is None:
                continue
            total_tokens = tensor.size(2)
            total_frames = total_tokens // N_patch
            if total_frames <= 1:
                continue

            keep = {0}  # anchor (absolute frame 0)

            # Sliding-window tail.
            if window_size is not None:
                win_start = max(1, total_frames - window_size)
                for f in range(win_start, total_frames):
                    keep.add(f)
            else:
                # No window → keyframe sampling can extend to the end.
                win_start = total_frames

            # Keyframe pool: frames {S, 2S, ...} strictly before the window
            # region (window region is already covered above).
            if keyframe_stride is not None and keyframe_stride > 0:
                f = keyframe_stride
                while f < win_start:
                    keep.add(f)
                    f += keyframe_stride

            if len(keep) == total_frames:
                continue

            keep_sorted = sorted(keep)
            slices = []
            cs = ce = keep_sorted[0]
            for f in keep_sorted[1:]:
                if f == ce + 1:
                    ce = f
                else:
                    slices.append((cs, ce + 1))
                    cs = ce = f
            slices.append((cs, ce + 1))

            parts = [tensor[:, :, s * N_patch : e * N_patch] for s, e in slices]
            layer_cache[ki] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)

    return kv_cache_list


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class OpenSe3rSLAM:
    """
    Online semantic SLAM using OpenSe3r with predicted geometry.

    Usage:
        pipeline = OpenSe3rSLAM(model, cfg_path='slam_opense3r/config.yaml')
        memory   = pipeline.run(images)   # images: [1, T, 3, H, W] in [-1, 1]
    """

    def __init__(
        self,
        model,
        cfg_path:       str  = './slam_opense3r/config.yaml',
        sem_feat_mode:  str  = 'expanded',    # 'expanded' (512-dim) or 'compact'
        feat_storage:   str  = 'voxel',       # 'voxel' or 'perpoint'
        kv_window_size: Optional[int] = None,  # None = no sliding-window pruning
        kv_keyframe_stride: Optional[int] = None,  # None = no sparse-keyframe KV
        backend:        str  = 'vggt',        # 'vggt' | 'da3' | 'mapanything'
        point_conf_threshold: Optional[float] = None,
        attn_mode:      str  = 'causal',
        global_scale_mode: str = 'per_chunk',  # 'per_chunk' | 'first_chunk' | 'gt_seq_median'
    ):
        # Validate before any model.to() — gives clearer errors on misuse
        if sem_feat_mode not in ('expanded', 'compact'):
            raise ValueError(f"sem_feat_mode must be 'expanded' or 'compact'")
        if feat_storage not in ('voxel', 'perpoint'):
            raise ValueError(f"feat_storage must be 'voxel' or 'perpoint'")
        if backend not in ('vggt', 'da3', 'mapanything'):
            raise ValueError(
                f"backend must be 'vggt', 'da3', or 'mapanything', got {backend!r}"
            )
        if attn_mode not in ('causal', 'chunk_bidir'):
            raise ValueError(
                f"attn_mode must be 'causal' or 'chunk_bidir', got {attn_mode!r}"
            )
        if global_scale_mode not in ('per_chunk', 'first_chunk', 'gt_seq_median'):
            raise ValueError(
                f"global_scale_mode must be one of "
                f"('per_chunk','first_chunk','gt_seq_median'), "
                f"got {global_scale_mode!r}"
            )

        self.cfg                  = OmegaConf.load(cfg_path)
        self.model                = model.to(self.cfg.device)
        self.sem_feat_mode        = sem_feat_mode
        self.feat_storage         = feat_storage
        self.kv_window_size       = kv_window_size
        self.kv_keyframe_stride   = kv_keyframe_stride
        self.backend              = backend
        self.point_conf_threshold = point_conf_threshold
        self.attn_mode            = attn_mode
        self.global_scale_mode    = global_scale_mode

        # Cached sequence-global scale [B] tensor, set after the first chunk
        # for 'first_chunk' mode (or pre-computed for 'gt_seq_median' by
        # subclasses that have access to GT depths).  None ⇒ no override
        # (per-chunk behaviour inside the model).
        self._seq_box_scale: Optional[torch.Tensor] = None

        # Per-step GPU peak memory (MB) captured by _geo_forward / _sem_forward
        # (and the voxel-kv pipelines' _sem_forward_voxel_kv).  Read by _log /
        # mem_stats.format_mem_log when emitting the [MEM] line.
        self._mem_peaks: dict = {}
        # Tracks the most recent KV cache passed through _sem_forward so _log
        # can report its size without the pipeline having to thread it through.
        self._last_kv_cache = None

    def _sem_feat_key(self):
        return 'semantic_feat_expanded' if self.sem_feat_mode == 'expanded' else 'semantic_feat'

    # ------------------------------------------------------------------
    # Sub-steps
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _geo_forward(self,
                     imgs:           torch.Tensor,
                     depth_z:        torch.Tensor = None,
                     intrinsics:     torch.Tensor = None) -> dict:
        """Run the model's geometry-only pass.

        Args:
            imgs:        [1, T, 3, H, W] in [-1, 1].
            depth_z:     [1, T, H, W] metric (optional).  Forwarded to the
                         backbone for multi-modal fusion when supported
                         (currently MapAnything).  Ignored by VGGT / DA3.
            intrinsics:  [1, T, 3, 3] (optional).  Required by MapAnything
                         when `depth_z` is supplied; ignored otherwise.
        """
        frames: dict = {'images': imgs}
        if depth_z is not None:
            frames['depth_z'] = depth_z
        if intrinsics is not None:
            frames['intrinsics'] = intrinsics
        with mem_stats.GpuPeakTracker(self._mem_peaks, 'geo'), \
                torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            return self.model.forward_geometry(frames)

    @torch.no_grad()
    def _sem_forward(self, imgs: torch.Tensor, kv_cache: list,
                     return_poses: bool = False,
                     extr_rel_override:     torch.Tensor = None,
                     world_points_override: torch.Tensor = None,
                     box_scale_override:    torch.Tensor = None,
                     # ── Fast path (pred-mode SLAM): caller passes geometry
                     # already computed by `forward_geometry` (+ aligned to the
                     # SLAM-global frame).  Routes to `model.forward_semantic`
                     # which skips the redundant DA3 / VGGT depth+pose pass.
                     geo:                   dict          = None,
                     c2w_global:            torch.Tensor  = None,
                     world_points_global:   torch.Tensor  = None) -> tuple:
        """
        Run the trainable semantic / instance branch on the given frames.

        ``self.attn_mode`` selects within-chunk attention (see class docstring).

        Two paths, selected by whether ``geo`` is supplied:

        - **Legacy path** (``geo is None``, e.g. GT pipelines): calls
          ``model.forward_chunk`` which re-computes geometry from images.
          ``extr_rel_override`` / ``world_points_override`` /
          ``box_scale_override`` patch the chunk-local geometry into the
          SLAM-global frame inside the model — required for pose_cond /
          pos3d_cond streaming.

        - **Fast path** (``geo`` supplied, pred pipelines): calls
          ``model.forward_semantic`` which skips the DA3 / VGGT geo branch
          entirely and uses the caller-supplied SLAM-global geometry:
            * ``geo['depth']``, ``geo['intrinsics']``,
              ``geo.get('world_points_conf', None)``  — from
              `forward_geometry` on the current chunk's frames
            * ``c2w_global`` — SLAM-global c2w (e.g. ``memory.poses[...]``)
            * ``world_points_global`` — SLAM-global per-pixel pts
              (e.g. ``memory._pending_pts_new`` after ``update_geo``)

          On this path ``extr_rel_override`` / ``world_points_override``
          are ignored — caller-supplied values already are SLAM-global.

        Returns (sem_res dict, updated kv_cache).
        """
        with mem_stats.GpuPeakTracker(self._mem_peaks, 'sem'), \
                torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if geo is not None:
                # Fast path — no second DA3/VGGT geometry pass.
                assert c2w_global is not None and world_points_global is not None, (
                    "_sem_forward fast path requires c2w_global and "
                    "world_points_global when geo is provided"
                )
                out = self.model.forward_semantic(
                    {'images': imgs},
                    depth              = geo['depth'],
                    intrinsics         = geo['intrinsics'],
                    c2w                = c2w_global,
                    world_points       = world_points_global,
                    world_points_conf  = geo.get('world_points_conf', None),
                    kv_cache_list      = kv_cache,
                    return_poses       = return_poses,
                    mode               = self.attn_mode,
                    box_scale_override = box_scale_override,
                )
            else:
                out = self.model.forward_chunk(
                    {'images': imgs},
                    kv_cache_list         = kv_cache,
                    return_poses          = return_poses,
                    mode                  = self.attn_mode,
                    extr_rel_override     = extr_rel_override,
                    world_points_override = world_points_override,
                    box_scale_override    = box_scale_override,
                )
        new_kv = out.pop('kv_cache', kv_cache)
        self._last_kv_cache = new_kv
        return out, new_kv

    # ------------------------------------------------------------------
    # Sequence-global scale helpers (used by `--global_scale_mode`)
    # ------------------------------------------------------------------

    def _capture_seq_box_scale(self, sem_res: dict):
        """Stash the model's reported `box_scale` from the first chunk so
        subsequent chunks can pass it as `box_scale_override` (mode
        'first_chunk').  No-op for other modes."""
        if self.global_scale_mode != 'first_chunk':
            return
        if self._seq_box_scale is not None:
            return
        bs = sem_res.get('box_scale', None)
        if bs is None:
            print("[global_scale_mode=first_chunk] WARN: model did not return "
                  "'box_scale' from first chunk — staying in per-chunk mode "
                  "for the rest of this run.")
            return
        # Detach + move to CPU so we hold no GPU graph reference; we re-send
        # to device on every chunk.  Keep dtype consistent with what the
        # model would have computed (float32 from the median path).
        self._seq_box_scale = bs.detach().float().cpu().clone()
        print(f"[global_scale_mode=first_chunk] captured "
              f"box_scale={self._seq_box_scale.tolist()}")

    def _current_box_scale_override(self,
                                    device: torch.device,
                                    ) -> Optional[torch.Tensor]:
        """Return the [B] override tensor to pass to the model for the
        current chunk, or None to let the model use its chunk-level
        depth median."""
        if self._seq_box_scale is None:
            return None
        return self._seq_box_scale.to(device)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_map(self, images: torch.Tensor, num_init: int,
                        memory: OpenSe3rMemory, sem_feat_key: str) -> list:
        """
        Process the first num_init frames.
        Geometry and semantics are run once together on all init frames.
        Returns initialized kv_cache.
        """
        device    = self.cfg.device
        init_imgs = images[:, :num_init].to(device)

        # Geometry: VGGT bidirectional on init frames (= global frame by definition)
        geo_res  = self._geo_forward(init_imgs)

        # Semantics: CausalBranch processes all init frames from scratch.
        # Init chunk's frame 0 IS the global origin (geometry forces c2w[0] = I),
        # so geo_res['poses'] / geo_res['world_points'] are already in the
        # SLAM-global frame.  Pass them through as overrides so init follows
        # the same code path as subsequent mapping steps (required for
        # KV-cache consistency under pose_cond / pos3d_cond).
        kv_cache    = self.model.init_kv_cache()
        init_c2w_g  = geo_res['poses'].detach().cpu().float()        # [1, T_init, 4, 4]
        init_wp_g   = geo_res['world_points'].detach().float()        # [1, T_init, H, W, 3]
        sem_res, kv_cache = self._sem_forward(
            init_imgs, kv_cache,
            extr_rel_override     = init_c2w_g,
            world_points_override = init_wp_g,
            box_scale_override    = self._current_box_scale_override(device),
        )

        # Stash box_scale for sequence-global scale modes (no-op for
        # 'per_chunk').  Must run AFTER the first sem_forward.
        self._capture_seq_box_scale(sem_res)

        if self.kv_window_size is not None or self.kv_keyframe_stride is not None:
            patch_size = getattr(self.model.causal_branch, 'patch_size', 14)
            H = images.shape[-2];  W = images.shape[-1]
            N_patch = (H // patch_size) * (W // patch_size)
            kv_cache = _prune_kv_cache(
                kv_cache, N_patch,
                window_size     = self.kv_window_size,
                keyframe_stride = self.kv_keyframe_stride,
            )

        memory.initialize(geo_res, sem_res, sem_feat_key)

        avg_conf = geo_res['world_points_conf'][0].squeeze(-1).mean().item()
        print(f"[init]  {num_init} frames  avg_conf={avg_conf:.3f}"
              f"  kf={memory.cur_kf_idx.tolist()}")
        return kv_cache

    # ------------------------------------------------------------------
    # Incremental mapping step
    # ------------------------------------------------------------------

    def _mapping_step(self, images: torch.Tensor,
                      start_idx: int, end_idx: int,
                      memory: OpenSe3rMemory, sem_feat_key: str,
                      kv_cache: list) -> list:
        """
        One incremental map step for frames [start_idx, end_idx] (inclusive).

        Order:
          1. Geometry forward on [kf_imgs + new_imgs]
          2. memory.update_geo  → coordinate_alignment, sets memory.poses
                                   for new frames in the SLAM global frame
          3. Read those global poses → use as `extr_rel_override` for the
             semantic forward (essential for pose_cond + KV-cache to stay
             coherent across chunks)
          4. Semantic forward on [new_imgs] only, with the override
          5. memory.update_sem  → push sem/ins features (reuses the cached
                                   globally-aligned pts from step 2)
        """
        device    = self.cfg.device
        cur_kf    = memory.cur_kf_idx

        # ── Geometry: keyframes + new frames ────────────────────────────────
        kf_imgs  = images[0, cur_kf].unsqueeze(0).to(device)          # [1, n_kf, 3, H, W]
        new_imgs = images[:, start_idx:end_idx + 1].to(device)        # [1, n_new, 3, H, W]
        geo_imgs = torch.cat([kf_imgs, new_imgs], dim=1)              # [1, n_kf+n_new, ...]
        geo_res  = self._geo_forward(geo_imgs)

        # ── Geo-only memory update (alignment runs here) ─────────────────────
        memory.update_geo(geo_res, start_idx, end_idx)

        # ── Compose overrides from globally-aligned geo state ────────────────
        # memory.poses[0] is identity (set at init), so the SLAM-global c2w
        # of new frames doubles as the relative-to-frame-0 pose the model
        # expects.  Build the [1, n_new, 4, 4] tensor directly.
        # memory._pending_pts_new was just populated by update_geo with
        # pts_gfl[n_kf:] — already in the SLAM-global frame — so feed it
        # directly as the world_points_override (the model will use it for
        # pos3d_cond / box-head / output).
        new_c2w_global = memory.poses[start_idx:end_idx + 1].unsqueeze(0).float()
        new_wp_global  = memory._pending_pts_new.unsqueeze(0).float().to(device)  # [1, n_new, H, W, 3]

        # ── Semantics: new frames only, with global pose / pts overrides ─────
        sem_res, kv_cache = self._sem_forward(
            new_imgs, kv_cache,
            extr_rel_override     = new_c2w_global,
            world_points_override = new_wp_global,
            box_scale_override    = self._current_box_scale_override(device),
        )

        if self.kv_window_size is not None or self.kv_keyframe_stride is not None:
            patch_size = getattr(self.model.causal_branch, 'patch_size', 14)
            H = images.shape[-2];  W = images.shape[-1]
            N_patch = (H // patch_size) * (W // patch_size)
            kv_cache = _prune_kv_cache(
                kv_cache, N_patch,
                window_size     = self.kv_window_size,
                keyframe_stride = self.kv_keyframe_stride,
            )

        # ── Sem-only memory update (push features into stores) ───────────────
        memory.update_sem(sem_res, sem_feat_key)

        return kv_cache

    # ------------------------------------------------------------------
    # Main run loop  (mirrors AMB3R slam/pipeline.py)
    # ------------------------------------------------------------------

    def run(self, images: torch.Tensor) -> OpenSe3rMemory:
        """
        Args:
            images: [1, T, 3, H, W]  in [-1, 1]

        Returns:
            OpenSe3rMemory with globally-consistent geometry + feature stores.
        """
        assert images.shape[0] == 1
        assert images.min() >= -1.0 and images.max() <= 1.0

        cfg        = self.cfg
        _, T, _, H, W = images.shape

        map_init_window = int(cfg.map_init_window)
        map_every       = int(cfg.map_every)
        sem_feat_key    = self._sem_feat_key()

        # Probe feature dims with a tiny forward
        probe_imgs = images[:, :min(map_init_window, T)].to(cfg.device)
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            probe = self.model.forward_chunk(
                {'images': probe_imgs},
                kv_cache_list = self.model.init_kv_cache(),
                return_poses  = False,
                mode          = self.attn_mode,
            )
        sem_key      = sem_feat_key
        sem_feat_dim = probe.get(sem_key, probe['semantic_feat']).shape[2]
        ins_feat_dim = probe['instance_feat'].shape[2]
        is_metric    = bool(probe.get('is_metric', False))
        if is_metric:
            print(f"[run] backend reports metric scale (scale_factor="
                  f"{probe.get('scale_factor', 1.0):.3f}) — "
                  f"alignment will seed scale=1.0")

        memory = OpenSe3rMemory(
            cfg          = cfg,
            num_frames   = T,
            H = H, W = W,
            sem_feat_dim = sem_feat_dim,
            ins_feat_dim = ins_feat_dim,
            feat_storage = self.feat_storage,
            voxel_size   = float(cfg.semantic_voxel_size),
            backend      = self.backend,
            is_metric    = is_metric,
            point_conf_threshold = self.point_conf_threshold,
        )
        print(f"[run] semantic/instance attn_mode = '{self.attn_mode}'")
        if self.point_conf_threshold is not None:
            print(f"[run] push-time geom conf threshold = "
                  f"{self.point_conf_threshold:.4f}")

        initialized     = False
        last_mapped_idx = -1
        kv_cache        = None
        t0              = time.time()

        for idx in range(T):
            # ── Accumulate until init window ─────────────────────────────────
            if idx < map_init_window - 1:
                continue

            # ── Map initialization ────────────────────────────────────────────
            if not initialized:
                kv_cache = self._initialize_map(
                    images, idx + 1, memory, sem_feat_key
                )
                initialized     = True
                last_mapped_idx = idx
                self._log(idx, T, memory, t0)
                continue

            # ── Incremental mapping ───────────────────────────────────────────
            if (idx - last_mapped_idx < map_every) and (idx < T - 1):
                continue

            start_idx = last_mapped_idx + 1
            kv_cache  = self._mapping_step(
                images, start_idx, idx, memory, sem_feat_key, kv_cache
            )
            last_mapped_idx = idx
            self._log(idx, T, memory, t0)

        return memory

    def _log(self, idx, T, memory, t0):
        elapsed = time.time() - t0
        fps     = (idx + 1) / elapsed
        storage = 'voxels' if self.feat_storage == 'voxel' else 'pts'
        print(f"[OpenSe3rSLAM] frame {idx+1:4d}/{T}"
              f"  sem {storage}: {memory.sem_count:,}"
              f"  ins {storage}: {memory.ins_count:,}"
              f"  kf={memory.cur_kf_idx.tolist()}"
              f"  {fps:.1f} fps")
        print(mem_stats.format_mem_log(
            memory,
            peaks            = self._mem_peaks,
            kv_cache_list    = self._last_kv_cache,
            voxel_kv_manager = getattr(self, 'voxel_kv_manager', None),
        ))
