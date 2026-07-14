"""
opense3r/instance_head_query.py
================================
Query-based instance head (MaskFormer-style) for OpenSe3r.

Replaces the legacy dense `FeatureDPTHead` instance pathway with a set of
class-agnostic object proposals.  Each predicted proposal carries:

  * a per-pixel mask logit
  * an objectness score (probability that this slot represents a real object)
  * a descriptor vector (for cross-frame instance re-identification by SLAM)

Architecture (per-frame, parallel across the chunk's T frames):

    multi-scale tokens          last-layer tokens
    output_tokens_list[2,5,8,11]   output_tokens_list[-1]
            │                          │
            ▼                          ▼
       FeatureDPTHead (pixel decoder)  Query Transformer Decoder
       → F_mask [B*T, C_mask, H, W]   N learnable query tokens
                                      cross-attend to last-layer tokens
                                      → Q [B*T, N, C_q]
                                            │
                                            ├─ mask_embed (MLP)  → [B*T, N, C_mask]
                                            ├─ objectness (Linear) → [B*T, N]
                                            └─ descriptor (MLP + L2-norm) → [B*T, N, D]

    mask_logits = einsum('btnc,btchw->btnhw', mask_embed, F_mask)
                  → [B, T, N, H, W]

The transformer decoder is a minimal post-norm DETR-style stack
(self-attn over queries → cross-attn queries-to-memory → FFN, ×L layers).
Sinusoidal 2D positional encoding is added to memory tokens; queries get
a learnable positional embedding.

Backbone-agnostic:  only the input channel `causal_dim` matters.  When
`--ins_channels` slices the CausalBranch token (e.g. (512,1024)), this
head operates on the 512-channel sub-space; both the pixel decoder
(causal_dim=512) and the transformer decoder's input projection
(512→C_q) handle the dim mismatch transparently.

Streaming:  this head has NO state.  It is a pure stateless reader of
CausalBranch's output_tokens_list per frame.  KV-cache lives entirely
in CausalBranch.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_heads import FeatureDPTHead


# ─────────────────────────────────────────────────────────────────────────────
# Small building blocks
# ─────────────────────────────────────────────────────────────────────────────

class _MLP(nn.Module):
    """MaskFormer-style MLP: num_layers linear layers with ReLU activations
    on all but the last layer."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        layers = []
        for n, k in zip([in_dim] + h, h + [out_dim]):
            layers.append(nn.Linear(n, k))
        self.layers = nn.ModuleList(layers)
        self.num_layers = num_layers

    def forward(self, x):
        for i, lin in enumerate(self.layers):
            x = F.relu(lin(x)) if i < self.num_layers - 1 else lin(x)
        return x


class _PositionEmbeddingSine2D(nn.Module):
    """2D sinusoidal positional encoding (ported from DETR / MaskFormer)."""

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000,
                 normalize: bool = True, scale: float = 2 * math.pi):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature   = temperature
        self.normalize     = normalize
        self.scale         = scale

    def forward(self, B: int, H: int, W: int, device, dtype) -> torch.Tensor:
        """Returns [B, 2*num_pos_feats, H, W]."""
        # not_mask = True everywhere — no padding mask.
        y_embed = torch.arange(1, H + 1, dtype=dtype, device=device).reshape(1, H, 1).expand(B, H, W)
        x_embed = torch.arange(1, W + 1, dtype=dtype, device=device).reshape(1, 1, W).expand(B, H, W)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=dtype, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t                          # [B, H, W, C]
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1).permute(0, 3, 1, 2)  # [B, 2C, H, W]
        return pos


# ─────────────────────────────────────────────────────────────────────────────
# Transformer decoder layer
# ─────────────────────────────────────────────────────────────────────────────

class _QueryDecoderLayer(nn.Module):
    """One DETR-style decoder layer.

    Post-norm.  Self-attention over queries (queries see each other) followed
    by cross-attention (queries attend to memory tokens) and an FFN.
    Positional encodings are added to Q,K of self-attn and to Q of cross-attn
    via `query_pos`; `memory_pos` is added to memory K of cross-attn.
    """

    def __init__(self, d_model: int, n_heads: int = 8,
                 dim_feedforward: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                  batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                  batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                queries:    torch.Tensor,                         # [B, N, d_model]
                memory:     torch.Tensor,                         # [B, S, d_model]
                query_pos:  Optional[torch.Tensor] = None,
                memory_pos: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        # Self-attention over queries
        q = k = (queries + query_pos) if query_pos is not None else queries
        sa_out, _ = self.self_attn(q, k, queries, need_weights=False)
        queries = self.norm1(queries + self.dropout(sa_out))

        # Cross-attention queries → memory
        q = (queries + query_pos) if query_pos is not None else queries
        k = (memory  + memory_pos) if memory_pos is not None else memory
        ca_out, _ = self.cross_attn(q, k, memory, need_weights=False)
        queries = self.norm2(queries + self.dropout(ca_out))

        # FFN
        ffn = self.linear2(self.dropout(F.relu(self.linear1(queries))))
        queries = self.norm3(queries + self.dropout(ffn))
        return queries


class _QueryDecoder(nn.Module):
    """Stack of `n_layers` decoder layers + final LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            _QueryDecoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, queries, memory, query_pos=None, memory_pos=None):
        x = queries
        for layer in self.layers:
            x = layer(x, memory, query_pos=query_pos, memory_pos=memory_pos)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# QueryInstanceHead
# ─────────────────────────────────────────────────────────────────────────────

class QueryInstanceHead(nn.Module):
    """MaskFormer-style class-agnostic instance head with descriptor output.

    Inputs (from `CausalBranch.forward`):
        output_tokens_list  list of 12 layer outputs, each [B, T, N+ps, in_channels]
                            (Stage 2 already slices channels if --ins_channels active.)
        images              [B, T, 3, H, W]  (used only for shape / device info)
        patch_start_idx     leading non-patch tokens (1 if pose_cond else 0)

    Outputs (single dict):
        mask_logits   [B, T, N, H, W]   per-pixel mask logits
        objectness    [B, T, N]         "is this slot a real object" logit
        descriptor    [B, T, N, D]      L2-normalized object descriptor
    """

    def __init__(
        self,
        causal_dim:    int = 1024,             # input dim (= --ins_channels span)
        num_queries:   int = 100,
        d_model:       int = 256,              # transformer + mask_embed dim
        descriptor_dim: int = 256,
        c_mask:        int = 256,              # pixel-decoder output channels
        n_heads:       int = 8,
        n_layers:      int = 6,
        dim_feedforward: int = 2048,
        dropout:       float = 0.0,
        intermediate_layer_idx: Optional[List[int]] = None,
        patch_size:    int = 14,
        mask_mlp_layers: int = 3,
        desc_mlp_layers: int = 3,
        use_pixel_shuffle_head: bool = False,
    ):
        super().__init__()
        if intermediate_layer_idx is None:
            intermediate_layer_idx = [2, 5, 8, 11]

        self.causal_dim        = causal_dim
        self.num_queries       = num_queries
        self.d_model           = d_model
        self.descriptor_dim    = descriptor_dim
        self.c_mask            = c_mask
        self.intermediate_layer_idx = intermediate_layer_idx
        self.patch_size        = patch_size

        # Pixel decoder — reuse FeatureDPTHead (without semantic conditioning,
        # without dim expansion).  Produces [B,T,C_mask,H,W].
        self.pixel_decoder = FeatureDPTHead(
            causal_dim   = causal_dim,
            out_channels = c_mask,
            expand_to    = None,
            intermediate_layer_idx = intermediate_layer_idx,
            patch_size   = patch_size,
            lseg_dim     = None,             # no semantic conditioning on this branch
            lseg_proj_dim = 128,
            use_pixel_shuffle_head = use_pixel_shuffle_head,
        )

        # Project last-layer patch tokens (in_dim=causal_dim) to d_model for
        # cross-attn memory.  Conv1x1 for spatial format compatibility.
        if causal_dim != d_model:
            self.memory_proj = nn.Linear(causal_dim, d_model)
        else:
            self.memory_proj = nn.Identity()

        # Learnable query embeddings (act as both query token init AND
        # positional embedding — DETR style).
        self.query_embed = nn.Embedding(num_queries, d_model)
        nn.init.normal_(self.query_embed.weight, std=0.02)

        # 2D sinusoidal positional encoding for memory tokens.
        # d_model must be even (we use d_model/2 features per axis).
        assert d_model % 2 == 0, f"d_model ({d_model}) must be even"
        self.pe_layer = _PositionEmbeddingSine2D(num_pos_feats=d_model // 2)

        # Transformer decoder.
        self.transformer = _QueryDecoder(
            d_model         = d_model,
            n_heads         = n_heads,
            n_layers        = n_layers,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
        )

        # Three small output heads on Q.
        self.mask_embed = _MLP(d_model, d_model, c_mask,         num_layers=mask_mlp_layers)
        self.objectness = nn.Linear(d_model, 1)
        self.descriptor = _MLP(d_model, d_model, descriptor_dim, num_layers=desc_mlp_layers)

    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        output_tokens_list: List[torch.Tensor],   # 12 × [B, T, N+ps, causal_dim]
        images:             torch.Tensor,         # [B, T, 3, H, W]
        patch_start_idx:    int = 0,
    ) -> Dict[str, torch.Tensor]:
        B, T = images.shape[:2]
        H, W = images.shape[-2:]
        ph = H // self.patch_size
        pw = W // self.patch_size
        BT = B * T

        # ── Pixel decoder: dense mask feature ──
        F_mask, _, _ = self.pixel_decoder(
            output_tokens_list, images, patch_start_idx=patch_start_idx,
            semantic_cond=None,
        )
        # F_mask: [B, T, c_mask, H, W]

        # ── Memory for cross-attn: last-layer patch tokens (per frame) ──
        # Drop the pose token (or other leading non-patch tokens) before the head.
        last_tokens = output_tokens_list[-1][:, :, patch_start_idx:]      # [B, T, N_patch, causal_dim]
        N_patch = last_tokens.shape[2]
        assert N_patch == ph * pw, (
            f"Last-layer patch token count {N_patch} != ph*pw = {ph*pw} "
            f"(image {H}x{W}, patch_size {self.patch_size})"
        )
        memory = last_tokens.reshape(BT, N_patch, self.causal_dim)
        memory = self.memory_proj(memory)                                  # [BT, N_patch, d_model]

        # ── Memory positional encoding (2D sinusoidal, per patch grid) ──
        pos2d = self.pe_layer(BT, ph, pw, device=memory.device, dtype=memory.dtype)   # [BT, d_model, ph, pw]
        memory_pos = pos2d.flatten(2).transpose(1, 2)                                  # [BT, N_patch, d_model]

        # ── Initial queries (zero) + learnable positional embedding ──
        # MaskFormer / DETR convention: query content starts at zero, only the
        # learnable positional embedding distinguishes slots.
        query_pos = self.query_embed.weight.unsqueeze(0).expand(BT, -1, -1)  # [BT, N, d_model]
        queries   = torch.zeros_like(query_pos)

        # ── Transformer decoder ──
        Q = self.transformer(queries, memory,
                              query_pos=query_pos, memory_pos=memory_pos)
        # Q: [BT, N, d_model]

        # ── Three small heads ──
        mask_embed = self.mask_embed(Q)                                    # [BT, N, c_mask]
        objectness = self.objectness(Q).squeeze(-1)                        # [BT, N]
        descriptor = self.descriptor(Q)                                    # [BT, N, D]
        descriptor = F.normalize(descriptor, dim=-1)

        # ── Mask prediction by dot product ──
        F_mask_flat = F_mask.reshape(BT, self.c_mask, H, W)
        mask_logits = torch.einsum('bnc,bchw->bnhw', mask_embed, F_mask_flat)   # [BT, N, H, W]

        # Reshape to [B, T, ...]
        mask_logits = mask_logits.reshape(B, T, self.num_queries, H, W)
        objectness  = objectness.reshape(B, T, self.num_queries)
        descriptor  = descriptor.reshape(B, T, self.num_queries, self.descriptor_dim)

        return {
            'mask_logits': mask_logits,
            'objectness':  objectness,
            'descriptor':  descriptor,
        }
