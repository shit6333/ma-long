"""STAC-style temporal KV-cache compression for CausalBranch streaming.

Vendored from thirdparty/STAC (`stac/kv_manager.py`, `stac/h2o.py`,
`stac/flash_attn_triton.py`) — Phase 1 uses ONLY the temporal memory
(pinned first-frame tokens + recent window + H2O heavy-hitter anchors);
the voxel/spatial tier is intentionally not vendored.

Integration surface:
  - `HeavyHittersKV`      — the KV manager (token-level, per-layer/per-head H2O).
  - `SparseStreamAttention` — drop-in replacement for STream3R `Attention`
    that routes streaming KV through the manager (see sparse_attention.py).
  - `CausalBranch.register_kv_mgr()` / `prune_kv_mgr()` etc. (causal_branch.py).
"""

from .kv_manager import KVManager
from .h2o import HeavyHittersKV
from .sparse_attention import SparseStreamAttention, convert_global_attn

__all__ = [
    "KVManager",
    "HeavyHittersKV",
    "SparseStreamAttention",
    "convert_global_attn",
]
