"""
pipeline_gt.py — OpenSe3r SLAM with GT point cloud + GT poses.
==============================================================
The model still runs its full forward pass (VGGT geometry + LSeg + CausalBranch)
for semantic and instance features.  The predicted world_points and poses are
DISCARDED and replaced with provided GT pts_gt and poses_gt.

This gives geometrically accurate semantic maps for evaluation on ScanNet etc.

KV-cache window pruning is supported identically to pipeline.py.
"""

import os
import sys
import time
import torch
from omegaconf import OmegaConf
from typing import Optional

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(_root, 'thirdparty'))

from slam_opense3r.memory import OpenSe3rMemory
from slam_opense3r.pipeline import _prune_kv_cache
from slam_opense3r import mem_stats


class OpenSe3rSLAM_GT:
    """
    Online semantic SLAM using GT point cloud and GT poses.

    The CausalBranch runs causally with KV-cache for semantic/instance features.
    World points and camera poses are taken directly from GT — no scale estimation
    or coordinate alignment needed.

    Usage:
        pipeline = OpenSe3rSLAM_GT(model, ...)
        memory   = pipeline.run(images, pts_gt, poses_gt)

    Args for run():
        images  : [1, T, 3, H, W]  in [-1, 1]
        pts_gt  : [T, H, W, 3]     GT world-space point cloud (float32, cpu)
        poses_gt: [T, 4, 4]        GT camera-to-world (float32, cpu)
    """

    def __init__(
        self,
        model,
        cfg_path:       str  = './slam_opense3r/config.yaml',
        sem_feat_mode:  str  = 'expanded',    # 'expanded' (512-dim) or 'compact'
        feat_storage:   str  = 'voxel',       # 'voxel' or 'perpoint'
        kv_window_size: Optional[int] = None,  # None = no sliding-window pruning
        kv_keyframe_stride: Optional[int] = None,  # None = no sparse-keyframe KV
        backend:        str  = 'vggt',        # 'vggt' | 'da3' | 'mapanything' (only matters for memory ctor)
        point_conf_threshold: Optional[float] = None,
        attn_mode:      str  = 'causal',
    ):
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

        self.cfg                  = OmegaConf.load(cfg_path)
        self.model                = model.to(self.cfg.device)
        self.sem_feat_mode        = sem_feat_mode
        self.feat_storage         = feat_storage
        self.kv_window_size       = kv_window_size
        self.kv_keyframe_stride   = kv_keyframe_stride
        self.backend              = backend
        self.point_conf_threshold = point_conf_threshold
        self.attn_mode            = attn_mode

        # GPU peak / kv-cache tracking for the [MEM] log line.
        self._mem_peaks: dict = {}
        self._last_kv_cache = None

    # ------------------------------------------------------------------

    def _sem_feat_key(self):
        return 'semantic_feat_expanded' if self.sem_feat_mode == 'expanded' else 'semantic_feat'

    # ------------------------------------------------------------------

    def run(
        self,
        images:   torch.Tensor,   # [1, T, 3, H, W]  in [-1, 1]
        pts_gt:   torch.Tensor,   # [T, H, W, 3]  GT world pts  (cpu float32)
        poses_gt: torch.Tensor,   # [T, 4, 4]     GT c2w poses  (cpu float32)
    ) -> OpenSe3rMemory:
        """
        Process the full sequence with GT geometry.

        pts_gt and poses_gt are used for storage only — they do not affect
        the model's internal computations (VGGT geometry, DepthEncoder, etc.).
        """
        assert images.shape[0] == 1
        assert images.min() >= -1.0 and images.max() <= 1.0
        assert pts_gt.shape[0] == images.shape[1], \
            f"pts_gt frames ({pts_gt.shape[0]}) != images frames ({images.shape[1]})"
        assert poses_gt.shape[0] == images.shape[1], \
            f"poses_gt frames ({poses_gt.shape[0]}) != images frames ({images.shape[1]})"

        cfg        = self.cfg
        device     = cfg.device
        chunk_size = int(cfg.chunk_size)
        _, T, _, H, W = images.shape

        patch_size = getattr(self.model.causal_branch, 'patch_size', 14)
        N_patch    = (H // patch_size) * (W // patch_size)

        # Probe to get feature dims.  Must run under bf16 autocast because some
        # frozen 2D backbones (e.g. TIPSv2's xformers MemEffAttention) reject
        # fp32 on CUDA — same wrapping as the main loop below.
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
            is_metric    = False,    # GT mode uses GT geometry directly
            point_conf_threshold = self.point_conf_threshold,
        )

        kv_cache = self.model.init_kv_cache()
        t0 = time.time()

        # Anchor for pose-cond: GT frame 0 in the GT world frame.  All chunks'
        # pose tokens are computed relative to this single anchor so that the
        # KV-cached past tokens stay consistent with new tokens.  Without
        # this override, every chunk would re-anchor to its own first frame,
        # which is the same streaming bug as in the pred-mode pipeline.
        pose_anchor_inv = torch.inverse(poses_gt[0].float())              # [4, 4]

        for chunk_start in range(0, T, chunk_size):
            chunk_end  = min(chunk_start + chunk_size, T)
            chunk_imgs = images[:, chunk_start:chunk_end].to(device)
            frame_idx  = torch.arange(chunk_start, chunk_end)

            # Build pose override for this chunk: GT c2w expressed in the
            # frame where GT frame 0 = identity.
            chunk_c2w_anchored = pose_anchor_inv @ poses_gt[chunk_start:chunk_end].float()
            extr_rel_override  = chunk_c2w_anchored.unsqueeze(0)         # [1, T_c, 4, 4]

            with mem_stats.GpuPeakTracker(self._mem_peaks, 'sem'), \
                    torch.no_grad(), \
                    torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = self.model.forward_chunk(
                    {'images': chunk_imgs},
                    kv_cache_list=kv_cache,
                    return_poses=False,     # GT poses used — skip VGGT camera head
                    mode=self.attn_mode,
                    extr_rel_override=extr_rel_override,
                )

            kv_cache = out['kv_cache']
            self._last_kv_cache = kv_cache

            if self.kv_window_size is not None or self.kv_keyframe_stride is not None:
                kv_cache = _prune_kv_cache(
                    kv_cache, N_patch,
                    window_size     = self.kv_window_size,
                    keyframe_stride = self.kv_keyframe_stride,
                )

            # --- GT geometry (replaces predicted world_points + poses) ---
            pts   = pts_gt  [chunk_start:chunk_end].float()   # [T_c, H, W, 3]
            poses = poses_gt[chunk_start:chunk_end].float()   # [T_c, 4, 4]
            # Confidence: valid where depth was non-zero (pts norm > small threshold).
            # depth=0 pixels unproject to (0,0,0) and must be masked out.
            pts_norm = pts.norm(dim=-1)                        # [T_c, H, W]
            conf = (pts_norm > 1e-3).float()

            # --- semantic feat (mode-selected) ---
            sem_feat = out.get(sem_key, out['semantic_feat'])[0].cpu().float()
            sem_conf = out['semantic_conf'][0].cpu().float()

            # --- instance feat ---
            ins_feat = out['instance_feat'][0].cpu().float()
            ins_conf = out['instance_conf'][0].cpu().float()

            memory.update_gt(
                frame_indices = frame_idx,
                pts   = pts,
                poses = poses,
                conf  = conf,
                sem_feat = sem_feat,
                sem_conf = sem_conf,
                ins_feat = ins_feat,
                ins_conf = ins_conf,
            )

            elapsed = time.time() - t0
            fps     = chunk_end / elapsed
            storage_label = 'voxels' if self.feat_storage == 'voxel' else 'pts'
            print(f"[OpenSe3rSLAM_GT] frames {chunk_end:4d}/{T}"
                  f"  sem {storage_label}: {memory.sem_count:,}"
                  f"  ins {storage_label}: {memory.ins_count:,}"
                  f"  {fps:.1f} fps")
            print(mem_stats.format_mem_log(
                memory,
                peaks            = self._mem_peaks,
                kv_cache_list    = self._last_kv_cache,
                voxel_kv_manager = getattr(self, 'voxel_kv_manager', None),
            ))

        return memory
