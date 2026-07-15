"""Score-aware attention for CausalBranch global blocks (STAC temporal KV).

`SparseStreamAttention` subclasses STream3R's `Attention` and, when a
`HeavyHittersKV` manager is attached AND the call is a streaming one
(`kv_cache is not None`), replaces the `torch.cat` + SDPA path with the
manager's `append_kv` + `decode_sparse_attn` (which also produces the
per-key attention column-sums used for H2O importance scores).

Everything else (training, non-streaming eval, blocks without a manager)
falls through to the original STream3R forward unchanged.

The swap is zero-copy: `from_attention()` re-points the original module's
submodules (qkv / q_norm / k_norm / proj / rope ...) onto the new instance,
so parameter identity, device, dtype and state_dict names are all preserved.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
from torch import Tensor, nn

_thirdparty = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'thirdparty',
)
if _thirdparty not in sys.path:
    sys.path.append(_thirdparty)
if os.path.join(_thirdparty, 'STream3R') not in sys.path:
    sys.path.append(os.path.join(_thirdparty, 'STream3R'))

from stream3r.models.components.layers.attention import Attention


class SparseStreamAttention(Attention):
    """STream3R `Attention` + optional STAC KV-manager routing.

    Extra (non-parameter) attributes:
      layer_idx  : int  — global-block index, used as the manager's layer key.
      kv_manager : HeavyHittersKV | None — attached by
                   `CausalBranch.register_kv_mgr()`; None ⇒ vanilla behaviour.
    """

    # ------------------------------------------------------------------
    # Construction: wrap an existing Attention without copying weights.
    # ------------------------------------------------------------------
    @classmethod
    def from_attention(cls, attn: Attention, layer_idx: int) -> "SparseStreamAttention":
        obj = cls.__new__(cls)
        nn.Module.__init__(obj)
        # Plain attributes.
        obj.num_heads  = attn.num_heads
        obj.head_dim   = attn.head_dim
        obj.scale      = attn.scale
        obj.fused_attn = attn.fused_attn
        obj.rope       = attn.rope
        # Submodules — shared with the original (no copy, same params).
        obj.qkv       = attn.qkv
        obj.q_norm    = attn.q_norm
        obj.k_norm    = attn.k_norm
        obj.attn_drop = attn.attn_drop
        obj.proj      = attn.proj
        obj.proj_drop = attn.proj_drop
        # Manager hookup.
        obj.layer_idx  = int(layer_idx)
        obj.kv_manager = None
        return obj

    def set_layer_idx(self, layer_idx: int):
        self.layer_idx = int(layer_idx)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: Tensor, pos=None, attn_mask=None, kv_cache=None):
        mgr = getattr(self, 'kv_manager', None)
        if mgr is None or kv_cache is None:
            # Training / non-streaming / manager not registered → original path.
            return super().forward(x, pos=pos, attn_mask=attn_mask, kv_cache=kv_cache)

        assert attn_mask is None, (
            "SparseStreamAttention manager path expects attn_mask=None "
            "(streaming causality is enforced by the cache itself)."
        )
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)          # [B, H, N, D]

        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # The triton/cuda colsum kernels require fp16/bf16; under bf16 autocast
        # LayerNorm/rope may have upcast q/k to fp32 — align to manager dtype
        # (matches what SDPA-under-autocast would do anyway).
        dt = mgr.dtype
        q = q.to(dt)
        k = k.to(dt)
        v = v.to(dt)

        mgr.append_kv(k, v, layer_idx=self.layer_idx)
        # decode expects query_states [B, Tq, H, D].
        out = mgr.decode_sparse_attn(q.transpose(1, 2), layer_idx=self.layer_idx)
        x = out.reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        # Keep the Block-level contract (`x, kv_cache = self.attn(...)`).
        # The list entry stays [None, None]; real state lives in the manager.
        return x, kv_cache


def convert_global_attn(global_blocks: nn.ModuleList, managed_idx) -> None:
    """Swap `global_blocks[i].attn` → SparseStreamAttention for i in managed_idx.

    Idempotent; leaves non-managed blocks untouched.
    """
    for i in managed_idx:
        blk = global_blocks[i]
        if isinstance(blk.attn, SparseStreamAttention):
            blk.attn.set_layer_idx(i)
            continue
        blk.attn = SparseStreamAttention.from_attention(blk.attn, layer_idx=i)
