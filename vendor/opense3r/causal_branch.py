import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch.utils.checkpoint import checkpoint

_thirdparty = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'thirdparty')
sys.path.append(_thirdparty)
sys.path.append(os.path.join(_thirdparty, 'STream3R'))

from stream3r.models.components.layers.block import Block
from stream3r.models.components.layers.rope import RotaryPositionEmbedding2D, PositionGetter


class _QueryMemoryCrossAttn(nn.Module):
    """Per-frame instance queries (Q) cross-attend to patch-token memory (K, V).

    Unlike the previous `_QueryCrossAttn` (sequence-level queries that read the
    whole chunk's patches at once), this module is CAUSAL and STREAMING-aware:

      - Training (kv_cache=None): queries are per-frame `[B*T, N_q, D]` and
        attend to patch tokens of frames `<= own frame` via a block-causal
        mask.  Query frame tq sees key frames tk for all tk <= tq.
      - Streaming (kv_cache given): the module keeps its OWN patch K,V cache
        across chunks.  Each call projects only the CURRENT chunk's patches to
        K,V, appends to the cache, and queries attend to the full accumulated
        memory (past + current).  Past K,V are reused, never recomputed →
        compute overhead ~zero; the cost is the extra cache VRAM.  (Within the
        current chunk the queries see all current-chunk frames, i.e.
        chunk-bidirectional; chunk-to-chunk causality is preserved because the
        cache only ever holds past frames.)

    Explicit q/k/v projections (instead of nn.MultiheadAttention) so K,V can be
    cached.  Output projection is zero-init → path starts as a no-op.
    """
    def __init__(
        self,
        dim:         int,
        num_heads:   int,
        mlp_ratio:   float = 4.0,
        init_values: float = 0.01,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj  = nn.Linear(dim, dim)
        self.k_proj  = nn.Linear(dim, dim)
        self.v_proj  = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.norm_ffn = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))

    def _heads(self, x: torch.Tensor, B: int, L: int) -> torch.Tensor:
        # [B, L, D] → [B, n_heads, L, head_dim]
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _block_causal_mask(T, N_q, N_patch, dtype, device) -> torch.Tensor:
        """Additive mask [T*N_q, T*N_patch]: 0 where query-frame tq may attend
        to key-frame tk (tk <= tq), -inf otherwise."""
        qt = torch.arange(T, device=device).repeat_interleave(N_q)       # [T*N_q]
        kt = torch.arange(T, device=device).repeat_interleave(N_patch)   # [T*N_patch]
        allowed = kt.unsqueeze(0) <= qt.unsqueeze(1)                     # [T*N_q, T*N_patch]
        mask = torch.zeros(T * N_q, T * N_patch, dtype=dtype, device=device)
        mask.masked_fill_(~allowed, float('-inf'))
        return mask

    def forward(
        self,
        queries:  torch.Tensor,                 # [B*T, N_q, dim]   per-frame
        patches:  torch.Tensor,                 # [B*T, N_patch, dim] current frames' patches
        B: int, T: int, N_q: int, N_patch: int,
        kv_cache: Optional[list] = None,        # [k_cache, v_cache] or None (training)
    ):
        """Returns (queries, new_kv_cache).  new_kv_cache is None during training."""
        D = queries.shape[-1]
        q = self.q_proj(self.norm_q(queries))                            # [B*T, N_q, D]
        kv_in = self.norm_kv(patches)
        k = self.k_proj(kv_in)                                           # [B*T, N_patch, D]
        v = self.v_proj(kv_in)

        if kv_cache is None:
            # ── Training / full-sequence: block-causal cross-attn ──
            qh = self._heads(q.reshape(B, T * N_q, D),     B, T * N_q)        # [B,nh,T*N_q,hd]
            kh = self._heads(k.reshape(B, T * N_patch, D), B, T * N_patch)
            vh = self._heads(v.reshape(B, T * N_patch, D), B, T * N_patch)
            mask = self._block_causal_mask(T, N_q, N_patch, qh.dtype, qh.device)
            out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)  # [B,nh,T*N_q,hd]
            out = out.transpose(1, 2).reshape(B, T * N_q, D).reshape(B * T, N_q, D)
            new_cache = None
        else:
            # ── Streaming: append current K,V to this module's own cache ──
            k_new = self._heads(k.reshape(B, T * N_patch, D), B, T * N_patch)  # [B,nh,T*N_patch,hd]
            v_new = self._heads(v.reshape(B, T * N_patch, D), B, T * N_patch)
            if kv_cache[0] is not None:
                k_all = torch.cat([kv_cache[0], k_new], dim=2)
                v_all = torch.cat([kv_cache[1], v_new], dim=2)
            else:
                k_all, v_all = k_new, v_new
            new_cache = [k_all, v_all]
            qh = self._heads(q.reshape(B, T * N_q, D), B, T * N_q)
            # No mask: queries see all accumulated memory (past + current chunk).
            out = F.scaled_dot_product_attention(qh, k_all, v_all)
            out = out.transpose(1, 2).reshape(B, T * N_q, D).reshape(B * T, N_q, D)

        queries = queries + self.ls1 * self.out_proj(out)
        queries = queries + self.ls2 * self.ffn(self.norm_ffn(queries))
        return queries, new_cache


class DepthCrossAttn(nn.Module):
    """
    Cross-attention where semantic tokens (Q) attend to depth-derived tokens (K, V).
    Output projection is zero-initialized so depth fusion starts as a no-op.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor, depth_tokens: torch.Tensor) -> torch.Tensor:
        """
        x:            [B*T, N_patch, dim]
        depth_tokens: [B*T, N_patch, dim]
        """
        q = self.norm_q(x)
        kv = self.norm_kv(depth_tokens)
        attn_out, _ = self.attn(q, kv, kv)
        return self.norm_out(x + attn_out)


class CausalBranch(nn.Module):
    """
    Causal Transformer for Semantic / Instance feature extraction.

    Input tokens are formed by fusing:
      - DINOv2 per-frame patch tokens  [B, T, N_patch, dino_token_dim]
        (from VGGT aggregator.encode_patch_tokens — no cross-frame attention,
         safe for online/streaming inference)
      - LSeg dense features            [B*T, lseg_dim, H, W]

    Architecture: n_blocks pairs of (frame_block + global_block).
      - Frame blocks: per-frame self-attention.
      - Global blocks: causal cross-frame attention during training;
                       KV-cache incremental attention during inference.
      - Every depth_cross_every global blocks, a DepthCrossAttn injects geometry.

    Initialization: frame/global blocks are loaded from VGGT's first n_blocks pairs.
    """

    def __init__(
        self,
        dino_token_dim: int = 1024,   # DINOv2 ViT-L embed_dim
        lseg_dim: int = 512,
        causal_dim: int = 1024,
        n_blocks: int = 12,
        n_heads: int = 16,
        mlp_ratio: float = 4.0,
        depth_cross_every: int = 3,
        patch_size: int = 14,
        rope_freq: int = 100,
        # 3D RoPE injection of per-patch world position:
        #   '2d'       (default) — legacy 2D image-grid RoPE on all blocks.
        #   '3d_full'  — 3D world-voxel RoPE on BOTH frame & global blocks.
        #   '3d_cross' — frame blocks keep 2D (intra-frame image grid), global
        #                (cross-frame) blocks use 3D world-voxel RoPE.  Recommended:
        #                fixes the fact that 2D RoPE gives every frame's (r,c) the
        #                SAME position in cross-frame attention.
        rope_mode: str = '2d',
        rope_voxel_size: float = 0.1,   # metric voxel size for 3D RoPE indices
        qk_norm: bool = True,
        init_values: float = 0.01,
        # Optional conditioning streams (default off → behaviour identical to
        # pre-conditioning architecture; ckpts load cleanly with strict=False).
        depth_cond:   bool = False,
        pose_cond:    bool = False,
        plucker_cond: bool = False,
        # VGGT-Ω-style register attention.  When num_register_tokens > 0:
        #   - `register_tokens` is a learnable [num_register_tokens, dim] param
        #     replicated per (b, t) → prepended after the pose token (if any).
        #   - At blocks listed in `register_block_indices`, the cross-frame
        #     global attention runs ONLY on the special tokens (pose + register),
        #     not on patches → KV cache for those blocks is ~(num_special / P)
        #     times smaller.  Patches still get cross-frame info INDIRECTLY via
        #     frame self-attention with the (just-updated) special tokens.
        #   - Default behaviour (num_register_tokens=0) is identical to the
        #     previous full-cross-frame design; legacy ckpts load cleanly.
        num_register_tokens:     int = 0,
        register_block_indices:  Optional[List[int]] = None,
        # ── 3DPPE-style 3D positional encoding (per-patch residual) ──
        # When pos3d_cond=True, caller provides `pos3d_tokens` to forward();
        # they are projected and added to the input residual stream alongside
        # depth / plucker tokens.  Default off → no extra params built.
        pos3d_cond: bool = False,
        # ── Middle-layer depth cross-attn gate ──
        # When False, DepthCrossAttn at every depth_cross_every blocks is
        # SKIPPED (but blocks are still built for state_dict back-compat).
        # Default True = current behaviour.
        use_depth_cross_attn: bool = True,
        # ── Instance queries fed through CausalBranch (Idea 2) ──
        # When num_instance_queries > 0, learnable instance query slots
        # are processed alongside patch tokens.  At blocks listed in
        # `instance_query_block_indices`, queries do self-attn and cross-attn
        # against current chunk's patch tokens (intra-chunk, no extra KV cache).
        # Final processed queries are returned (alongside output_tokens_list)
        # so Mask2Former head can use them as init queries (skip its own
        # learnable embedding).  Default 0 = off, full backward-compat.
        num_instance_queries:        int = 0,
        instance_query_block_indices: Optional[List[int]] = None,
        # ── How instance_queries are INITIALIZED inside CausalBranch ──
        #   'learnable' (default): nn.Parameter([1, N_q, D]) random init.
        #   '3d_anchor':            learnable 3D anchors [N_q, 3] in
        #                           NORMALIZE space (~unit cube), Fourier
        #                           PE → MLP → init queries.  At forward
        #                           time the queries are RECOMPUTED from
        #                           anchors (so they remain 3D-aware
        #                           throughout training).
        # Both flow through the same q_self_attn / q_cross_patch blocks.
        instance_queries_init_type:  str = 'learnable',
    ):
        super().__init__()
        self.causal_dim = causal_dim
        self.n_blocks = n_blocks
        self.depth_cross_every = depth_cross_every
        self.patch_size = patch_size

        # Conditioning flags
        self.depth_cond   = depth_cond
        self.pose_cond    = pose_cond
        self.plucker_cond = plucker_cond
        # patch_start_idx counts ALL non-patch prefix tokens.
        #   pose_cond ⇒ +1 (pose token at position 0)
        #   register_tokens ⇒ +num_register_tokens (right after pose token)
        n_pose_tok = 1 if pose_cond else 0
        self.num_register_tokens = int(num_register_tokens)
        self.patch_start_idx = n_pose_tok + self.num_register_tokens
        # Default register block positions: 3 evenly-spaced blocks (~25% of 12).
        if self.num_register_tokens > 0:
            if register_block_indices is None:
                register_block_indices = [n_blocks // 4,
                                           n_blocks // 2,
                                           3 * n_blocks // 4]
            self.register_block_indices = set(int(i) for i in register_block_indices)
            for idx in self.register_block_indices:
                if not (0 <= idx < n_blocks):
                    raise ValueError(
                        f"register_block_indices contains invalid block {idx} "
                        f"(must be in [0, {n_blocks}))"
                    )
            # Learnable register tokens.  Small-norm init (matches VGGT-Ω).
            self.register_tokens = nn.Parameter(
                torch.empty(1, self.num_register_tokens, causal_dim)
            )
            nn.init.normal_(self.register_tokens, std=1e-3)
        else:
            self.register_block_indices = set()
            self.register_tokens = None

        # --- Input fusion: DINOv2 tokens + LSeg → causal_dim ---
        # lseg_dim=None/0 → single-source input (e.g. RADIO raw patch feat only, no
        # separate semantic-prior concat); otherwise concat DINO ⊕ semantic prior.
        self.use_lseg_input = lseg_dim is not None and lseg_dim > 0
        _in_dim = dino_token_dim + (lseg_dim if self.use_lseg_input else 0)
        self.input_proj = nn.Sequential(
            nn.Linear(_in_dim, causal_dim),
            nn.GELU(),
            nn.Linear(causal_dim, causal_dim),
        )

        # --- Optional residual fusion paths (default kaiming init) ---
        # depth_cond: depth_tokens [B*T, N_patch, causal_dim] → residual add.
        # plucker_cond: plucker_tokens [B*T, N_patch, causal_dim] → residual add.
        # pose_cond: pose_token [B, T, causal_dim] → prepend (no projection here).
        if depth_cond:
            self.depth_input_proj = nn.Linear(causal_dim, causal_dim)
        if plucker_cond:
            self.plucker_input_proj = nn.Linear(causal_dim, causal_dim)
        # 3D-PE residual projection (Idea 1): pos3d_tokens [B*T, N_patch, D] → residual add
        self.pos3d_cond = pos3d_cond
        if pos3d_cond:
            self.pos3d_input_proj = nn.Linear(causal_dim, causal_dim)
        # Depth cross-attn gate (default True = current behaviour)
        self.use_depth_cross_attn = use_depth_cross_attn

        # --- Rotary position embeddings (shared, stateless) ---
        if rope_mode not in ('2d', '3d_full', '3d_cross'):
            raise ValueError(f"rope_mode must be 2d/3d_full/3d_cross, got {rope_mode!r}")
        self.rope_mode = rope_mode
        self.rope_voxel_size = float(rope_voxel_size)
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        # 3D RoPE (head_dim = causal_dim // n_heads; (24,24,16) for head_dim 64).
        self.rope_3d = None
        if rope_mode in ('3d_full', '3d_cross') and rope_freq > 0:
            from .rope3d import RotaryPositionEmbedding3D
            hd = causal_dim // n_heads
            base = (hd // 2 // 3) * 2                       # even per-axis base
            dz = hd - 2 * base                              # remainder → z axis (even)
            self.rope_3d = RotaryPositionEmbedding3D(frequency=rope_freq,
                                                     dims_split=(base, base, dz))
        # Per-block rope assignment: frame vs global may differ (3d_cross).
        rope_frame  = self.rope_3d if rope_mode == '3d_full' else self.rope
        rope_global = self.rope_3d if rope_mode in ('3d_full', '3d_cross') else self.rope

        # --- Frame blocks (per-frame self-attention) ---
        self.frame_blocks = nn.ModuleList([
            Block(
                dim=causal_dim, num_heads=n_heads, mlp_ratio=mlp_ratio,
                qk_norm=qk_norm, init_values=init_values, rope=rope_frame,
            )
            for _ in range(n_blocks)
        ])

        # --- Global blocks (causal cross-frame attention) ---
        self.global_blocks = nn.ModuleList([
            Block(
                dim=causal_dim, num_heads=n_heads, mlp_ratio=mlp_ratio,
                qk_norm=qk_norm, init_values=init_values, rope=rope_global,
            )
            for _ in range(n_blocks)
        ])

        # --- Depth cross-attention blocks ---
        n_depth = n_blocks // depth_cross_every
        self.depth_cross_blocks = nn.ModuleList([
            DepthCrossAttn(causal_dim, n_heads)
            for _ in range(n_depth)
        ])

        # --- (Idea 2) Instance query injection blocks ---
        # Default off (num_instance_queries=0) → no extra params built.
        # When >0:
        #   - learnable `instance_queries` [1, N_q, D]  (chunk-shared)
        #   - per-injection-layer: q_self_attn + q_cross_attn (queries × patches)
        self.num_instance_queries = int(num_instance_queries)
        if self.num_instance_queries > 0:
            if instance_query_block_indices is None:
                # Spread over the model: 4 evenly-distributed injection points
                # at 1/4, 2/4, 3/4, 4/4 boundaries.
                # n_blocks=12 → [2, 5, 8, 11]
                instance_query_block_indices = [
                    (k + 1) * n_blocks // 4 - 1
                    for k in range(4)
                ]
                # Clamp + dedupe + sort
                instance_query_block_indices = sorted({
                    max(0, min(n_blocks - 1, i)) for i in instance_query_block_indices
                })
            self.instance_query_block_indices = list(instance_query_block_indices)
            for idx in self.instance_query_block_indices:
                if not (0 <= idx < n_blocks):
                    raise ValueError(
                        f"instance_query_block_indices contains invalid block "
                        f"{idx} (must be in [0, {n_blocks}))"
                    )
            if instance_queries_init_type not in ('learnable', '3d_anchor'):
                raise ValueError(
                    f"instance_queries_init_type must be 'learnable' or "
                    f"'3d_anchor', got {instance_queries_init_type!r}"
                )
            self.instance_queries_init_type = instance_queries_init_type
            if instance_queries_init_type == 'learnable':
                # Original: direct learnable embedding [1, N_q, D].
                self.instance_queries = nn.Parameter(
                    torch.empty(1, self.num_instance_queries, causal_dim)
                )
                nn.init.normal_(self.instance_queries, std=0.02)
                self.instance_query_anchor    = None
                self.instance_query_pos_encoder = None
            else:   # '3d_anchor'
                # 3DPPE-style: learnable [1, N_q, 3] anchor in NORMALIZE space
                # → Fourier PE → MLP → init queries.  Re-computed each forward
                # so anchors learn jointly via gradient.
                from .pos_encoders import PosEncoder3D
                self.instance_query_anchor = nn.Parameter(
                    torch.empty(1, self.num_instance_queries, 3)
                )
                nn.init.uniform_(self.instance_query_anchor, -1.0, 1.0)
                self.instance_query_pos_encoder = PosEncoder3D(out_dim=causal_dim)
                # Not a learnable Parameter (computed every forward), but we
                # still expose `instance_queries` as None so downstream
                # checks `is not None` works via the anchor path.
                self.instance_queries = None
            # Per-injection self-attn (queries × queries) and cross-attn
            # (queries × patch_tokens of current chunk).
            n_inj = len(self.instance_query_block_indices)
            self.q_self_attn_blocks = nn.ModuleList([
                Block(
                    dim=causal_dim, num_heads=n_heads, mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm, init_values=init_values, rope=None,
                )
                for _ in range(n_inj)
            ])
            self.q_mem_cross_blocks = nn.ModuleList([
                _QueryMemoryCrossAttn(dim=causal_dim, num_heads=n_heads,
                                      mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(n_inj)
            ])
        else:
            self.instance_query_block_indices = []
            self.instance_queries = None

    # ------------------------------------------------------------------
    # Attention mask
    # ------------------------------------------------------------------

    def _build_causal_mask(
        self, T: int, P: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Causal mask: frame i cannot attend to frame j > i."""
        N = T * P
        mask = torch.zeros((N, N), dtype=dtype, device=device)
        for i in range(T):
            s, e = i * P, (i + 1) * P
            mask[s:e, e:] = float('-inf')
        return mask

    # ------------------------------------------------------------------
    # VGGT initialization
    # ------------------------------------------------------------------

    def init_from_vggt(self, vggt_model: nn.Module):
        """Load first n_blocks frame/global blocks from VGGT aggregator."""
        agg = vggt_model.aggregator
        for i in range(self.n_blocks):
            if i < len(agg.frame_blocks):
                msg = self.frame_blocks[i].load_state_dict(
                    agg.frame_blocks[i].state_dict(), strict=False
                )
                print(f"  frame_block[{i}]: missing={msg.missing_keys[:3]}")
            if i < len(agg.global_blocks):
                msg = self.global_blocks[i].load_state_dict(
                    agg.global_blocks[i].state_dict(), strict=False
                )
                print(f"  global_block[{i}]: missing={msg.missing_keys[:3]}")

    def init_from_omega(self, omega_state_dict: dict, verbose: bool = True):
        """Load CausalBranch weights from a VGGT-Omega state_dict.

        Source layout (under the 'aggregator.' prefix in the omega ckpt):
            aggregator.frame_blocks.{0..23}.*        → frame_blocks[i]
            aggregator.inter_frame_blocks.{0..23}.*  → global_blocks[i]
                                                       (omega uses bidirectional
                                                        inter-frame attention,
                                                        but the weights are
                                                        structurally identical
                                                        to our causal global
                                                        attention — only the
                                                        attention mask differs
                                                        at runtime)
            aggregator.register_token  [1, 2, R, D]  → register_tokens (the
                                                       'other frames' slot at
                                                       index 1; the index-0
                                                       'first frame' slot is
                                                       dropped since we use a
                                                       shared register token
                                                       across all frames)

        Loads up to `min(n_blocks, 24)` blocks (omega ships 24, we use 12).
        Uses strict=False per-block so micro-attribute name differences
        (q_norm, k_norm, etc. between vggt-omega's SelfAttentionBlock and
        stream3r's Block) don't error — they just become missing keys you
        can inspect in the printout.
        """
        # Build per-prefix sub-dicts so we can call load_state_dict per block.
        def _strip_prefix(prefix: str) -> dict:
            n = len(prefix)
            return {k[n:]: v for k, v in omega_state_dict.items() if k.startswith(prefix)}

        n_loaded_f, n_loaded_g = 0, 0
        for i in range(self.n_blocks):
            fb_sd = _strip_prefix(f'aggregator.frame_blocks.{i}.')
            if fb_sd:
                msg = self.frame_blocks[i].load_state_dict(fb_sd, strict=False)
                if verbose and i == 0:
                    print(f"  frame_block[0]: missing={msg.missing_keys[:3]}, "
                          f"unexpected={msg.unexpected_keys[:3]}")
                n_loaded_f += 1

            gb_sd = _strip_prefix(f'aggregator.inter_frame_blocks.{i}.')
            if gb_sd:
                msg = self.global_blocks[i].load_state_dict(gb_sd, strict=False)
                if verbose and i == 0:
                    print(f"  global_block[0]: missing={msg.missing_keys[:3]}, "
                          f"unexpected={msg.unexpected_keys[:3]}")
                n_loaded_g += 1

        # Register tokens: omega stores [1, 2, R_omega, D] (first frame / others).
        # We share registers across frames → take the index-1 'others' slice.
        if self.register_tokens is not None:
            key = 'aggregator.register_token'
            if key in omega_state_dict:
                src = omega_state_dict[key]                      # [1, 2, R_omega, D]
                R_omega = src.shape[2]
                R_self  = self.register_tokens.shape[1]
                if R_omega >= R_self:
                    with torch.no_grad():
                        self.register_tokens.copy_(src[:, 1, :R_self, :])
                    if verbose:
                        print(f"  register_tokens: loaded {R_self}/{R_omega} "
                              f"tokens from 'other frames' slot")
                else:
                    print(f"  register_tokens: omega ckpt has only "
                          f"{R_omega} < {R_self} — skipped (kept random init)")

        print(f"[CausalBranch] init_from_omega: "
              f"frame_blocks={n_loaded_f}/{self.n_blocks}, "
              f"global_blocks={n_loaded_g}/{self.n_blocks}")

    # ------------------------------------------------------------------
    # KV-cache helpers (inference only)
    # ------------------------------------------------------------------

    def init_kv_cache(self) -> List:
        """Create empty KV-cache list.

        Entries 0..n_blocks-1  → global-block cross-frame KV (one per block).
        Entries n_blocks..      → per-injection-layer instance-query memory KV
                                  (only appended when num_instance_queries>0,
                                  one per injection layer).  Appending here keeps
                                  the return type a flat list and leaves the
                                  global-block indexing untouched (backward
                                  compatible when num_instance_queries==0).
        """
        cache = [[None, None] for _ in range(self.n_blocks)]
        if self.num_instance_queries > 0:
            cache += [[None, None] for _ in range(len(self.instance_query_block_indices))]
        return cache

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        dino_tokens: torch.Tensor,                       # [B, T, N_patch, dino_token_dim]
        lseg_feat: torch.Tensor,                         # [B*T, lseg_dim, H, W]
        depth_tokens: torch.Tensor,                      # [B*T, N_patch, causal_dim]
        patch_h: int,
        patch_w: int,
        mode: str = "causal",
        kv_cache_list: Optional[List] = None,
        pose_token: Optional[torch.Tensor] = None,       # [B, T, causal_dim]      pose_cond
        plucker_tokens: Optional[torch.Tensor] = None,   # [B, T, N_patch, dim]    plucker_cond
        pos3d_tokens: Optional[torch.Tensor] = None,     # [B*T, N_patch, dim]     pos3d_cond
        world_pts_patch: Optional[torch.Tensor] = None,  # [B, T, N_patch, 3]      per-patch world coords (3D RoPE)
    ) -> Tuple[List[torch.Tensor], Optional[List], Optional[torch.Tensor]]:
        """
        Attention modes (`mode` argument):
          - 'causal'      training default. Within the current chunk's queries,
                          frame i cannot attend to frame j>i (mask applied only
                          when kv_cache_list is None — in streaming, past tokens
                          live in an immutable cache so the mask is unnecessary).
          - 'full'        no mask; bidirectional across all current-chunk tokens.
                          Intended for non-streaming evaluation (kv_cache_list=None).
          - 'chunk_bidir' inference-only. Bidirectional within the current chunk
                          AND compatible with kv_cache_list — frames in the same
                          chunk see each other; chunk-to-chunk causality is
                          preserved by the cache (which only ever contains past
                          frames). Use this for streaming inference when you want
                          intra-chunk frames to refine each other before being
                          committed to the cache.

        Optional conditioning (controlled by flags set in __init__):
          - depth_cond=True:    add depth_input_proj(depth_tokens) to base input.
                                The existing DepthCrossAttn path also still
                                runs (independent).
          - pose_cond=True:     prepend pose_token to each frame at position 0;
                                tokens shape becomes [B*T, 1+N_patch, dim].
                                DPT heads must use patch_start_idx=1.
          - plucker_cond=True:  add plucker_input_proj(plucker_tokens) to base
                                input (per-patch, residual).

        Returns:
            output_tokens_list: list of n_blocks tensors, each
                [B, T, P, causal_dim]   where P = N_patch + patch_start_idx.
            kv_cache_list:      updated cache (only when input kv_cache_list is not None)
        """
        assert mode in ("causal", "full", "chunk_bidir"), \
            f"Unknown mode: {mode}. Expected 'causal', 'full', or 'chunk_bidir'."

        B, T = dino_tokens.shape[:2]
        N_patch = patch_h * patch_w

        # --- Input fusion: DINO tokens [⊕ semantic prior] → causal_dim ---
        dino_flat  = dino_tokens.reshape(B * T, N_patch, dino_tokens.shape[-1])  # [B*T, N, dino_dim]
        if self.use_lseg_input:
            lseg_patch = F.adaptive_avg_pool2d(lseg_feat, (patch_h, patch_w))    # [B*T, C, ph, pw]
            lseg_patch = lseg_patch.flatten(2).permute(0, 2, 1)                  # [B*T, N, C]
            x = self.input_proj(torch.cat([dino_flat, lseg_patch], dim=-1))      # [B*T, N, causal_dim]
        else:
            x = self.input_proj(dino_flat)                                       # single-source (RADIO raw)

        # --- Optional: depth as residual at input (depth_cond) ---
        if self.depth_cond:
            assert depth_tokens is not None, "depth_cond=True requires depth_tokens"
            x = x + self.depth_input_proj(depth_tokens)

        # --- Optional: Plücker rays as residual at input (plucker_cond) ---
        if self.plucker_cond:
            assert plucker_tokens is not None, "plucker_cond=True requires plucker_tokens"
            plucker_flat = plucker_tokens.reshape(B * T, N_patch, self.causal_dim)
            x = x + self.plucker_input_proj(plucker_flat)

        # --- (Idea 1) Optional: 3D position encoding as residual at input ---
        if self.pos3d_cond:
            assert pos3d_tokens is not None, "pos3d_cond=True requires pos3d_tokens"
            x = x + self.pos3d_input_proj(pos3d_tokens)

        # --- Optional: prepend per-frame pose token (pose_cond) ---
        # After this, sequence length is P = patch_start_idx + N_patch.
        if self.pose_cond:
            assert pose_token is not None, "pose_cond=True requires pose_token"
            pose_tok = pose_token.reshape(B * T, 1, self.causal_dim)              # [B*T, 1, dim]
            x = torch.cat([pose_tok, x], dim=1)                                   # [B*T, 1+N, dim]

        # --- Optional: prepend learnable register tokens (VGGT-Ω) ---
        # Inserted AFTER pose_token, BEFORE patches.  Layout:
        #   [pose_tok? | register_tokens | patches]
        # patch_start_idx already accounts for both.
        if self.register_tokens is not None:
            reg = self.register_tokens.expand(B * T, -1, -1)                       # [B*T, R, dim]
            if self.pose_cond:
                # x is currently [B*T, 1+N, dim]; pose at position 0
                x = torch.cat([x[:, :1], reg, x[:, 1:]], dim=1)
            else:
                x = torch.cat([reg, x], dim=1)                                     # [B*T, R+N, dim]
        P = N_patch + self.patch_start_idx

        # --- Position embeddings ---
        # When pose_cond=True, the pose token gets pos=(0,0); all patch positions
        # are shifted by patch_start_idx so they don't clash.  Mirrors VGGT's
        # convention for the camera/register special tokens.
        pos = None
        if self.rope is not None:
            pos_patches = self.position_getter(B * T, patch_h, patch_w, device=x.device)
            if self.patch_start_idx > 0:
                pos_patches = pos_patches + self.patch_start_idx
                pos_special = torch.zeros(
                    B * T, self.patch_start_idx, 2,
                    dtype=pos_patches.dtype, device=x.device,
                )
                pos = torch.cat([pos_special, pos_patches], dim=1)                # [B*T, P, 2]
            else:
                pos = pos_patches                                                  # [B*T, N, 2]

        # --- 3D world-voxel positions (for 3D-RoPE blocks) ---
        pos3d = None
        if self.rope_3d is not None:
            if world_pts_patch is None:
                raise ValueError(
                    f"rope_mode={self.rope_mode!r} requires world_pts_patch "
                    "([B,T,N_patch,3]); none was passed (SLAM streaming must "
                    "thread global-frame per-patch world points)."
                )
            from .rope3d import voxelize_world_points
            vox = voxelize_world_points(world_pts_patch, self.rope_voxel_size)      # [B,T,N_patch,3]
            vox = vox.reshape(B * T, N_patch, 3)
            if self.patch_start_idx > 0:
                vox = vox + self.patch_start_idx
                spec3 = torch.zeros(B * T, self.patch_start_idx, 3,
                                    dtype=vox.dtype, device=x.device)
                pos3d = torch.cat([spec3, vox], dim=1)                             # [B*T, P, 3]
            else:
                pos3d = vox

        # Per-block-type position (frame vs global may use different RoPE).
        pos_frame  = pos3d if self.rope_mode == '3d_full' else pos
        pos_global = pos3d if self.rope_mode in ('3d_full', '3d_cross') else pos

        # --- Causal mask (training only; not needed when kv_cache_list given) ---
        attn_mask = None
        if kv_cache_list is None and mode == "causal":
            attn_mask = self._build_causal_mask(T, P, x.dtype, x.device)

        output_tokens_list: List[torch.Tensor] = []
        depth_cross_idx = 0

        # (Idea 2) Expand instance queries to a PER-FRAME batch dim if enabled.
        # All frames start from the SAME learnable init (DETR-style); the
        # per-frame differentiation comes entirely from the causal memory each
        # frame reads in `_QueryMemoryCrossAttn`.  Shape: [B*T, N_q, D].
        # Two init paths:
        #   'learnable': self.instance_queries [1, N_q, D] directly
        #   '3d_anchor': self.instance_query_anchor [1, N_q, 3] → Fourier PE → MLP
        instance_q = None
        if self.num_instance_queries > 0:
            if self.instance_queries is not None:
                # 'learnable' path
                instance_q = self.instance_queries.expand(B * T, -1, -1).contiguous()
            else:
                # '3d_anchor' path: compute init queries from anchor each forward
                # so anchors get gradient (re-computed graph).
                anchor = self.instance_query_anchor.to(x.dtype)
                pe_init = self.instance_query_pos_encoder(anchor)            # [1, N_q, D]
                instance_q = pe_init.expand(B * T, -1, -1).contiguous()
        # Build a set for O(1) lookup
        inj_set = set(self.instance_query_block_indices)

        for i in range(self.n_blocks):
            # ---- Frame block: per-frame self-attention [B*T, P, dim] ----
            if self.training:
                x = checkpoint(
                    self.frame_blocks[i], x, pos_frame, None, None,
                    use_reentrant=False,
                )
            else:
                x = self.frame_blocks[i](x, pos=pos_frame)

            # ---- Global block: cross-frame attention ----
            # Two routes:
            #   (a) "register" block (i in self.register_block_indices):
            #         Only the prefix special tokens (pose+register) participate
            #         in cross-frame attention; patches passthrough.  KV cache
            #         for this block is (patch_start_idx/P)× the size of (b).
            #   (b) "full" block (default): all P tokens cross-frame as before.
            n_special = self.patch_start_idx
            is_register_block = (n_special > 0
                                   and i in self.register_block_indices)
            if is_register_block:
                # Split off the special prefix.
                x_special = x[:, :n_special, :].reshape(B, T * n_special, self.causal_dim)
                x_patches = x[:,  n_special:, :]                              # passthrough
                pos_special_g = (pos[:, :n_special, :].reshape(B, T * n_special, 2)
                                  if pos is not None else None)
                # Causal mask only over the special tokens.
                attn_mask_sp = None
                if kv_cache_list is None and mode == "causal":
                    attn_mask_sp = self._build_causal_mask(
                        T, n_special, x.dtype, x.device,
                    )

                if kv_cache_list is not None:
                    x_special, kv_cache_list[i] = self.global_blocks[i](
                        x_special, pos=pos_special_g, attn_mask=None,
                        kv_cache=kv_cache_list[i],
                    )
                elif self.training:
                    x_special = checkpoint(
                        self.global_blocks[i], x_special, pos_special_g,
                        attn_mask_sp, None, use_reentrant=False,
                    )
                else:
                    x_special = self.global_blocks[i](
                        x_special, pos=pos_special_g, attn_mask=attn_mask_sp,
                    )
                x_special = x_special.reshape(B * T, n_special, self.causal_dim)
                x = torch.cat([x_special, x_patches], dim=1)
            else:
                x_g = x.reshape(B, T * P, self.causal_dim)
                pos_g = (pos_global.reshape(B, T * P, pos_global.shape[-1])
                         if pos_global is not None else None)

                if kv_cache_list is not None:
                    # Inference: incremental with KV-cache
                    x_g, kv_cache_list[i] = self.global_blocks[i](
                        x_g, pos=pos_g, attn_mask=None, kv_cache=kv_cache_list[i]
                    )
                elif self.training:
                    x_g = checkpoint(
                        self.global_blocks[i], x_g, pos_g, attn_mask, None,
                        use_reentrant=False,
                    )
                else:
                    x_g = self.global_blocks[i](x_g, pos=pos_g, attn_mask=attn_mask)

                x = x_g.reshape(B * T, P, self.causal_dim)

            # ---- Depth cross-attention (every depth_cross_every blocks) ----
            # Cross-attn queries are the patch tokens (skip pose token if any),
            # then we splice the pose-token slice back unchanged.
            # Gated by `use_depth_cross_attn` (default True = legacy behavior).
            if (self.use_depth_cross_attn
                    and (i + 1) % self.depth_cross_every == 0
                    and depth_cross_idx < len(self.depth_cross_blocks)):
                if self.patch_start_idx > 0:
                    x_special = x[:, :self.patch_start_idx]                       # [B*T, K, dim]
                    x_patches = x[:, self.patch_start_idx:]                       # [B*T, N, dim]
                    x_patches = self.depth_cross_blocks[depth_cross_idx](
                        x_patches, depth_tokens
                    )
                    x = torch.cat([x_special, x_patches], dim=1)
                else:
                    x = self.depth_cross_blocks[depth_cross_idx](x, depth_tokens)
            # Always increment counter so the SAME depth block is used at the
            # SAME position regardless of whether we skipped — keeps state_dict
            # indexing aligned across gated / ungated runs.
            if (i + 1) % self.depth_cross_every == 0 and depth_cross_idx < len(self.depth_cross_blocks):
                depth_cross_idx += 1

            # ---- (Idea 2) Instance-query injection at specified layers ----
            # Per-frame queries self-attend (within their own frame) then read
            # causal patch memory: frame t attends to patch K,V of frames <= t
            # (block-causal mask at train; this module's own KV cache at infer).
            if instance_q is not None and i in inj_set:
                idx = self.instance_query_block_indices.index(i)
                instance_q = self.q_self_attn_blocks[idx](instance_q, pos=None)   # [B*T, N_q, D]
                # patch tokens of the CURRENT frames (skip special prefix)
                patches_t = x[:, self.patch_start_idx:] if self.patch_start_idx > 0 else x
                Np = patches_t.shape[1]                                            # == N_patch
                q_mem_cache = (kv_cache_list[self.n_blocks + idx]
                                if kv_cache_list is not None else None)
                instance_q, new_qc = self.q_mem_cross_blocks[idx](
                    instance_q, patches_t, B, T, self.num_instance_queries, Np,
                    kv_cache=q_mem_cache,
                )
                if kv_cache_list is not None:
                    kv_cache_list[self.n_blocks + idx] = new_qc

            output_tokens_list.append(x.reshape(B, T, P, self.causal_dim))

        # Per-frame, memory-aware instance queries → [B, T, N_q, D].
        instance_q_out = (instance_q.reshape(B, T, self.num_instance_queries, self.causal_dim)
                          if instance_q is not None else None)
        return output_tokens_list, kv_cache_list, instance_q_out
