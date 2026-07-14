"""OpenSe3r query-instance semantic model behind a per-chunk streaming API.

Wraps ``slam_opense3r.run.load_model`` (arch guaranteed to match the checkpoint —
same builder as the reference run) and threads the causal KV cache across chunks.
FastSAM query seeding happens INSIDE the model (``fastsam_query=True`` builds
``_fastsam_seg`` + ``region_pool``; ``_build_fastsam_seed`` feeds ``init_queries``),
so no external segmenter is needed here.

Chunk contract (CPU float32 torch tensors, from ``bridge``):
    images                 [1,T,3,H,W]  in [-1,1]
    depth                  [1,T,H,W]    metric z
    intrinsics             [1,T,3,3]    K at (H,W)
    c2w                    [1,T,4,4]    cam->world, sequence-frame-0 anchored (gauge)
    world_points_anchored  [1,T,H,W,3]  same anchored frame; invalid pixels zeroed
    valid                  [1,T,H,W]    bool geometry validity

Output: the model's prediction dict — ``mask_logits`` [1,T,N,H,W], ``objectness``
[1,T,N], ``descriptor`` [1,T,N,256], ``semantic_feat_expanded``, ``box_scale``,
``kv_cache`` (consumed here), ...
"""

from __future__ import annotations

from typing import Optional

import torch

from ma_slam_semantic.config import SemConfig, ensure_opense3r
from ma_slam_semantic.kv_policies import make_kv_policy


class SemanticModel:
    def __init__(self, cfg: SemConfig, device: Optional[str] = None):
        ensure_opense3r(cfg.opense3r_root)
        from slam_opense3r.run import load_model
        self.cfg = cfg
        self.device = device or cfg.device
        self.model = load_model(cfg.model_args())      # builds + loads ckpt + cuda + eval
        self.kv = self.model.init_kv_cache()
        self.policy = make_kv_policy(cfg)
        self._box_scale: Optional[torch.Tensor] = None  # [1] sequence-global scale

    # ------------------------------------------------------------------ scale
    def set_box_scale(self, scale) -> None:
        """Lock the sequence-global box_scale (golden gt mode: gt_seq_median)."""
        self._box_scale = torch.as_tensor(scale, dtype=torch.float32).reshape(1)
        print(f"[sem_model] box_scale locked = {float(self._box_scale):.4f}")

    def _maybe_capture_box_scale(self, out: dict) -> None:
        if self._box_scale is not None or self.cfg.box_scale_mode != "first_chunk":
            return
        bs = out.get("box_scale")
        if bs is None:
            print("[sem_model] WARN: box_scale_mode=first_chunk but model returned no "
                  "'box_scale' — staying in per-chunk mode.")
            return
        self._box_scale = bs.detach().float().cpu().reshape(1)
        print(f"[sem_model] box_scale captured from first chunk = {float(self._box_scale):.4f}")

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward_chunk(self, chunk: dict) -> dict:
        from opense3r.model import fill_invalid_world_points
        dev = self.device
        valid = chunk["valid"][0]                                        # [T,H,W] bool
        pts = fill_invalid_world_points(
            chunk["world_points_anchored"][0].float(), valid).unsqueeze(0).to(dev)
        bs = None if self._box_scale is None else self._box_scale.to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.model.forward_semantic(
                {"images": chunk["images"].to(dev)},
                depth=chunk["depth"].float().to(dev),
                intrinsics=chunk["intrinsics"].float().to(dev),
                c2w=chunk["c2w"].float().to(dev),
                world_points=pts,
                world_points_conf=valid.float().unsqueeze(0).unsqueeze(-1).to(dev),
                box_scale_override=bs,
                kv_cache_list=self.kv,
                mode=self.cfg.attn_mode,
                return_poses=False,
            )
        self._maybe_capture_box_scale(out)
        self.kv = self.policy.after_chunk(out["kv_cache"])
        return out

    def reset_kv(self) -> None:
        self.kv = self.model.init_kv_cache()
        print("[sem_model] KV cache reset")
