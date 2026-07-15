"""KV-cache retention policies for the causal semantic branch.

The frozen reference config keeps the FULL flat KV (no pruning) — ``NonePolicy``.
``WindowPolicy`` reuses OpenSe3r's ``_prune_kv_cache`` (sliding window / keyframe
stride over per-frame token blocks). ``H2OPolicy`` is the STAC-style temporal
compression (plan S5): pinned frame-0 tokens + recent window + per-layer/per-head
heavy-hitter tokens ranked by decayed attention column-sums. The score problem
("SDPA doesn't expose attention") is solved by the vendored STAC triton kernel
(``opense3r/stac_kv/flash_attn_triton.py``) which returns per-key column sums
alongside the attention output.
"""

from __future__ import annotations

from typing import Optional


class NonePolicy:
    """Flat KV, no pruning — the frozen-config default."""

    def after_chunk(self, kv, meta=None):
        return kv


class WindowPolicy:
    """Sliding-window (+ optional keyframe stride) pruning of per-frame KV blocks."""

    def __init__(self, n_patches: int, window_size: Optional[int],
                 keyframe_stride: Optional[int] = None):
        self.n_patches = n_patches
        self.window_size = window_size
        self.keyframe_stride = keyframe_stride

    def after_chunk(self, kv, meta=None):
        from slam_opense3r.pipeline import _prune_kv_cache
        return _prune_kv_cache(kv, self.n_patches, window_size=self.window_size,
                               keyframe_stride=self.keyframe_stride)


class H2OPolicy:
    """STAC/H2O temporal KV compression for the CausalBranch global blocks.

    Registers a ``HeavyHittersKV`` manager on ``model.causal_branch`` (the
    manager owns the managed layers' KV in a pre-allocated buffer; the flat
    ``kv_cache`` list slots for those layers stay ``[None, None]`` and pass
    through untouched).  ``after_chunk`` runs the H2O prune:

        keep = pinned(frame 0) ∪ recent ``h2o_recent_frames`` ∪
               per-(layer, head) top-k by decayed attention colsum
               (budget ``h2o_hh_frames`` frame-equivalents)

    The manager resets automatically on ``model.init_kv_cache()`` (new
    sequence / ``reset_kv``), so no extra lifecycle handling is needed here.
    """

    def __init__(self, cfg, model):
        import torch
        from opense3r.stac_kv import HeavyHittersKV
        self._branch = model.causal_branch
        self._branch.register_kv_mgr(
            HeavyHittersKV,
            chunk_size      = int(cfg.chunk_size),
            recent_size     = int(cfg.h2o_recent_frames),
            hh_size         = int(cfg.h2o_hh_frames),
            pinned_idx      = [0],
            temperature     = float(cfg.h2o_decay),
            subsample_ratio = float(cfg.h2o_subsample),
            attn_backend    = cfg.h2o_attn_backend,
            dtype           = torch.bfloat16,
            device          = torch.device(cfg.device),
        )
        self.prune_ms = 0.0
        print(f"[kv_policy] h2o: window={cfg.h2o_recent_frames}f "
              f"hh={cfg.h2o_hh_frames}f pinned=[0] decay={cfg.h2o_decay} "
              f"subsample={cfg.h2o_subsample} backend={cfg.h2o_attn_backend}")

    def after_chunk(self, kv, meta=None):
        self.prune_ms = self._branch.prune_kv_mgr(timing=True)
        return kv

    def stats(self) -> Optional[dict]:
        """Manager cache stats (tokens/layer, MB) for logging; None pre-build."""
        return self._branch.get_kv_mgr_info()


def make_kv_policy(cfg, model=None):
    if cfg.kv_policy == "none":
        return NonePolicy()
    if cfg.kv_policy == "window":
        if cfg.kv_window_size is None and cfg.kv_keyframe_stride is None:
            raise ValueError("kv_policy='window' needs kv_window_size and/or kv_keyframe_stride")
        return WindowPolicy(cfg.n_patches, cfg.kv_window_size, cfg.kv_keyframe_stride)
    if cfg.kv_policy == "h2o":
        if model is None:
            raise ValueError("kv_policy='h2o' needs the model (make_kv_policy(cfg, model=...))")
        return H2OPolicy(cfg, model)
    raise ValueError(f"unknown kv_policy {cfg.kv_policy!r} (expected 'none' | 'window' | 'h2o')")
