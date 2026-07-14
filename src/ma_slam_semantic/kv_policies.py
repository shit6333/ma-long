"""KV-cache retention policies for the causal semantic branch.

The frozen reference config keeps the FULL flat KV (no pruning) — ``NonePolicy``.
``WindowPolicy`` reuses OpenSe3r's ``_prune_kv_cache`` (sliding window / keyframe
stride over per-frame token blocks). An H2O (attention-score) policy is deferred:
SDPA may not expose attention scores — evaluate eager fallback / proxy scores when
implementing (plan S5).
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


def make_kv_policy(cfg):
    if cfg.kv_policy == "none":
        return NonePolicy()
    if cfg.kv_policy == "window":
        if cfg.kv_window_size is None and cfg.kv_keyframe_stride is None:
            raise ValueError("kv_policy='window' needs kv_window_size and/or kv_keyframe_stride")
        return WindowPolicy(cfg.n_patches, cfg.kv_window_size, cfg.kv_keyframe_stride)
    raise ValueError(f"unknown kv_policy {cfg.kv_policy!r} (expected 'none' | 'window')")
