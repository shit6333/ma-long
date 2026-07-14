"""
opense3r/stage2.py
==================
Stage 2 semantic refinement on top of a frozen Stage 1 OpenSe3r model.

Design (see conversation notes for full reasoning):

  Stage 1 is fully frozen and produces:
    sem_feat_expanded  [B, T, output_dim, H, W]   text-aligned dense features
    instance_feat      [B, T, ins_dim,    H, W]
    dino_patches       [B, T, dino_dim,   ph, pw]

  Stage 2 (this module) produces a refined `D_refined` in the SAME
  text-aligned space, via:

    1. Pool sem/ins to patch grid (ph, pw) for tractable attention.
    2. Three cross-attention paths at patch grid:
       - Path 1 (instance-aware):  Q = ins_low,  KV = sem_low
       - Path 2 (DINO-aware):       Q = dino,     KV = sem_low
       - Path 3 (self):             linear projection of sem_low
    3. Concat + linear fuse the three paths.
    4. Cross-view global attention (bidirectional within chunk) over all
       T·N_patch tokens.  4 transformer blocks.
    5. Project back to output_dim, bilinear upsample to (H, W).
    6. Residual merge with the FULL-RES Stage 1 sem_feat via a zero-init
       1x1 conv — so initial output == Stage 1 exactly.  Dense per-pixel
       detail from Stage 1's DPT decoding is never thrown away; Stage 2
       only adds a low-frequency class-disambiguation correction.

  Backbone-agnostic:  the module takes dims (output_dim, ins_dim, dino_dim)
  at construction; the text encoder for the loss is whatever Stage 1's
  semantic_backbone provides.  Works with LSeg / CLIP / SigLIP / TIPSv2 /
  PE-Core.

  Streaming:  Stage 2 has NO state, NO cache, NO recurrence.  It just
  processes the current chunk's Stage 1 outputs.  Training and inference
  are identical (no truncation needed).
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionPath(nn.Module):
    """One cross-attention retrieval pass.  Q from one source, KV from another.

    Shape conventions: inputs are [B*T, N, dim].  Single multi-head attention
    layer + residual MLP + LayerNorms.  No positional encoding (added later
    at the cross-view level).
    """

    def __init__(self, q_dim: int, kv_dim: int, d_attn: int, num_heads: int = 8,
                 mlp_ratio: float = 2.0):
        super().__init__()
        assert d_attn % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_attn // num_heads

        self.q_proj = nn.Linear(q_dim,  d_attn, bias=True)
        self.k_proj = nn.Linear(kv_dim, d_attn, bias=True)
        self.v_proj = nn.Linear(kv_dim, d_attn, bias=True)
        self.attn_out = nn.Linear(d_attn, d_attn, bias=True)

        self.ln_q = nn.LayerNorm(q_dim)
        self.ln_kv = nn.LayerNorm(kv_dim)
        self.ln_post = nn.LayerNorm(d_attn)

        hidden = int(d_attn * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_attn, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_attn),
        )
        self.ln_mlp = nn.LayerNorm(d_attn)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        q:  [B*T, N, q_dim]
        kv: [B*T, N, kv_dim]
        returns: [B*T, N, d_attn]
        """
        BT, N, _ = q.shape

        q_n = self.ln_q(q)
        kv_n = self.ln_kv(kv)

        Q = self.q_proj(q_n).reshape(BT, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv_n).reshape(BT, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv_n).reshape(BT, N, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(Q, K, V)              # [BT, h, N, d]
        out = out.transpose(1, 2).reshape(BT, N, -1)
        out = self.attn_out(out)
        out = self.ln_post(out)

        out = out + self.mlp(self.ln_mlp(out))                     # MLP residual
        return out


class CrossViewBlock(nn.Module):
    """Pre-norm transformer block for bidirectional cross-view self-attention."""

    def __init__(self, d_attn: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        assert d_attn % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_attn // num_heads

        self.ln1 = nn.LayerNorm(d_attn)
        self.qkv = nn.Linear(d_attn, 3 * d_attn, bias=True)
        self.proj = nn.Linear(d_attn, d_attn, bias=True)
        self.drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_attn)
        hidden = int(d_attn * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_attn, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_attn),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, d_attn]   S = T * N_patch
        B, S, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv.unbind(0)
        attn_out = F.scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        x = x + self.drop(self.proj(attn_out))
        x = x + self.mlp(self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 refiner
# ─────────────────────────────────────────────────────────────────────────────

class Stage2Refiner(nn.Module):
    """Refines Stage 1's semantic feature using DINO + instance signals,
    cross-view bidirectional attention, and a zero-init residual merge with
    the full-resolution Stage 1 output.
    """

    def __init__(
        self,
        output_dim: int,                       # backbone's text-aligned dim (768 / 1152 / 512 …)
        ins_dim:    int = 16,                  # Stage 1 instance feature dim
        dino_dim:   int = 1024,                # VGGT/DA3 patch token dim
        d_attn:     int = 512,                 # internal attention dim
        num_heads:  int = 8,
        n_cross_view_layers: int = 4,
        mlp_ratio_path: float = 2.0,
        mlp_ratio_cv:   float = 4.0,
        patch_grid_h:   int = 24,              # default for OpenSe3r 336x518 + patch=14
        patch_grid_w:   int = 37,
        use_pos_embed:  bool = True,
    ):
        super().__init__()
        self.output_dim     = output_dim
        self.ins_dim        = ins_dim
        self.dino_dim       = dino_dim
        self.d_attn         = d_attn
        self.patch_grid_h   = patch_grid_h
        self.patch_grid_w   = patch_grid_w

        N_patch = patch_grid_h * patch_grid_w

        # ── Three paths ──
        self.path1 = CrossAttentionPath(
            q_dim=ins_dim,  kv_dim=output_dim, d_attn=d_attn,
            num_heads=num_heads, mlp_ratio=mlp_ratio_path,
        )
        self.path2 = CrossAttentionPath(
            q_dim=dino_dim, kv_dim=output_dim, d_attn=d_attn,
            num_heads=num_heads, mlp_ratio=mlp_ratio_path,
        )
        # Path 3 (self): just project sem_low into d_attn space
        self.path3_ln = nn.LayerNorm(output_dim)
        self.path3_proj = nn.Linear(output_dim, d_attn)

        # ── Fuse three paths ──
        self.fuse = nn.Sequential(
            nn.LayerNorm(3 * d_attn),
            nn.Linear(3 * d_attn, d_attn),
        )

        # ── Optional spatial positional embedding for cross-view attention ──
        # 2D learnable, shared across frames (no temporal positional — frames
        # are exchangeable within a chunk).
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, N_patch, d_attn))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None

        # ── Cross-view bidirectional self-attention ──
        self.cross_view_blocks = nn.ModuleList([
            CrossViewBlock(d_attn=d_attn, num_heads=num_heads,
                           mlp_ratio=mlp_ratio_cv)
            for _ in range(n_cross_view_layers)
        ])
        self.cv_norm = nn.LayerNorm(d_attn)

        # ── Output projection back to text-aligned dim ──
        self.proj_out = nn.Linear(d_attn, output_dim)

        # ── Zero-init residual merge (init: output == stage 1 exactly) ──
        # 1x1 conv at full-res: per-channel learned gating of the correction
        self.merge_conv = nn.Conv2d(output_dim, output_dim, kernel_size=1, bias=False)
        nn.init.zeros_(self.merge_conv.weight)

    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        sem_feat:      torch.Tensor,           # [B, T, output_dim, H, W]   (Stage 1 sem_feat_expanded)
        instance_feat: torch.Tensor,           # [B, T, ins_dim,    H, W]
        dino_patches:  torch.Tensor,           # [B, T, dino_dim, ph, pw]  OR  [B, T, N_patch, dino_dim]
    ) -> torch.Tensor:
        """Returns D_refined of shape [B, T, output_dim, H, W]."""
        B, T, C_sem, H, W = sem_feat.shape
        assert C_sem == self.output_dim, (
            f"sem_feat channel {C_sem} != output_dim {self.output_dim}; "
            f"check Stage 1's sem_expand setting."
        )
        ph, pw = self.patch_grid_h, self.patch_grid_w
        N_patch = ph * pw
        BT = B * T

        # Stage 1 sem_feat at full res — kept aside, will be residual base.
        sem_full = sem_feat.reshape(BT, C_sem, H, W)

        # ── 1. Pool to patch grid ─────────────────────────────────────────
        sem_low = F.adaptive_avg_pool2d(sem_full, (ph, pw))                         # [BT, output_dim, ph, pw]
        ins_full = instance_feat.reshape(BT, self.ins_dim, H, W)
        ins_low = F.adaptive_avg_pool2d(ins_full, (ph, pw))                         # [BT, ins_dim, ph, pw]

        # DINO accepts either flat sequence [B,T,N,D] or 4D [B,T,D,ph',pw'].
        if dino_patches.dim() == 4:
            # [B, T, N_patch, dino_dim]  — Stage 1's native shape
            assert dino_patches.shape[-1] == self.dino_dim, (
                f"dino_patches last-dim {dino_patches.shape[-1]} != "
                f"dino_dim {self.dino_dim}"
            )
            assert dino_patches.shape[2] == N_patch, (
                f"dino_patches has {dino_patches.shape[2]} tokens, "
                f"expected ph·pw = {N_patch} ({ph}×{pw})"
            )
            dino_low = (dino_patches.reshape(BT, N_patch, self.dino_dim)
                        .transpose(1, 2).reshape(BT, self.dino_dim, ph, pw))
        else:
            # [B, T, dino_dim, ph', pw']
            dino_low = dino_patches.reshape(BT, self.dino_dim, *dino_patches.shape[-2:])
            if dino_low.shape[-2:] != (ph, pw):
                dino_low = F.interpolate(dino_low, size=(ph, pw),
                                         mode='bilinear', align_corners=False)

        # Reshape to token sequences [BT, N_patch, dim]
        sem_seq  = sem_low.flatten(2).transpose(1, 2)                               # [BT, N, output_dim]
        ins_seq  = ins_low.flatten(2).transpose(1, 2)                               # [BT, N, ins_dim]
        dino_seq = dino_low.flatten(2).transpose(1, 2)                              # [BT, N, dino_dim]

        # ── 2. Three paths ────────────────────────────────────────────────
        out1 = self.path1(q=ins_seq,  kv=sem_seq)                                    # [BT, N, d_attn]
        out2 = self.path2(q=dino_seq, kv=sem_seq)                                    # [BT, N, d_attn]
        out3 = self.path3_proj(self.path3_ln(sem_seq))                                # [BT, N, d_attn]

        # ── 3. Fuse ───────────────────────────────────────────────────────
        fused = self.fuse(torch.cat([out1, out2, out3], dim=-1))                     # [BT, N, d_attn]

        # ── 4. Cross-view global attention ────────────────────────────────
        # [BT, N, d_attn] → [B, T*N, d_attn]
        x = fused.reshape(B, T, N_patch, self.d_attn)
        if self.pos_embed is not None:
            x = x + self.pos_embed.unsqueeze(0)                                      # broadcast over B, T
        x = x.reshape(B, T * N_patch, self.d_attn)

        for blk in self.cross_view_blocks:
            x = blk(x)
        x = self.cv_norm(x)

        # ── 5. Project back to output_dim + upsample ──────────────────────
        x = x.reshape(B * T, N_patch, self.d_attn)
        x = self.proj_out(x)                                                         # [BT, N, output_dim]
        correction_low = x.transpose(1, 2).reshape(BT, self.output_dim, ph, pw)
        correction_up = F.interpolate(correction_low, size=(H, W),
                                       mode='bilinear', align_corners=False)

        # ── 6. Residual merge (zero-init: D_refined starts == sem_full) ──
        D_refined_flat = sem_full + self.merge_conv(correction_up)

        return D_refined_flat.reshape(B, T, self.output_dim, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 loss
# ─────────────────────────────────────────────────────────────────────────────

class Stage2Loss(nn.Module):
    """SigLIP-style per-pixel sigmoid contrastive loss + small cosine
    regularizer towards Stage 1's sem_feat_expanded.

    The text embeddings are precomputed once per dataset (class names are
    fixed) and passed in.  Module is backbone-agnostic.

    GT format:
        class_label: [B, T, H, W] long.  Convention: 0 = unlabeled (skipped);
                     1..N_class = class indices into the text embedding table
                     (NOTE: text_emb expects classes 0..N_class-1, so this
                     loss internally subtracts 1).
    """

    def __init__(
        self,
        w_sem: float = 1.0,
        w_reg: float = 0.1,
        k_init: float = 10.0,
        b_init: float = -10.0,
        mask_unlabeled: bool = True,
        # When per-pixel × all-classes BCE is too big, you can downsample classes
        # at training time.  None = use all classes.
        num_neg_classes: Optional[int] = None,
    ):
        super().__init__()
        self.w_sem = w_sem
        self.w_reg = w_reg
        self.mask_unlabeled = mask_unlabeled
        self.num_neg_classes = num_neg_classes

        # Learnable SigLIP scalars (parameterized as log_k to keep k > 0).
        self.log_k = nn.Parameter(torch.log(torch.tensor(float(k_init))))
        self.b     = nn.Parameter(torch.tensor(float(b_init)))

    def forward(
        self,
        D_refined:      torch.Tensor,            # [B, T, output_dim, H, W]
        sem_feat_stage1: torch.Tensor,           # [B, T, output_dim, H, W] (frozen, no grad)
        text_emb:       Optional[torch.Tensor] = None,   # [N_class, output_dim] L2-normalized
        class_label:    Optional[torch.Tensor] = None,   # [B, T, H, W] long, 0 = unlabeled
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with keys: 'objective', 'L_sem', 'L_reg', and meta.

        If text_emb / class_label is None → L_sem is skipped (returns 0).
        Useful for dry-running before dataset has class labels.
        """
        out = {}

        # ── L_reg: cosine pull to stage 1 ────────────────────────────────
        D_n = F.normalize(D_refined, dim=2)
        S1_n = F.normalize(sem_feat_stage1, dim=2)
        cos_to_s1 = (D_n * S1_n).sum(dim=2)                       # [B, T, H, W]
        L_reg = (1.0 - cos_to_s1).mean()
        out['L_reg'] = L_reg.detach()

        # ── L_sem: per-pixel sigmoid contrastive vs class text embeddings ─
        if text_emb is None or class_label is None:
            L_sem = D_refined.new_zeros(())
            valid_count = D_refined.new_zeros(())
        else:
            B, T, _, H, W = D_refined.shape
            # Flatten to [N_px, D]
            D_flat = D_n.permute(0, 1, 3, 4, 2).reshape(-1, D_refined.shape[2])
            gt_flat = class_label.reshape(-1)                     # [N_px] long
            valid_mask = gt_flat > 0
            valid_count = valid_mask.sum().float()

            if valid_count.item() == 0:
                L_sem = D_refined.new_zeros(())
            else:
                Dv = D_flat[valid_mask]                            # [N_valid, D]
                gt_v = gt_flat[valid_mask] - 1                     # 0-indexed: 0..N_class-1
                T_emb = F.normalize(text_emb, dim=-1)              # [N_class, D]

                logits = Dv @ T_emb.T                              # cosine sim
                scores = self.log_k.exp() * logits + self.b        # [N_valid, N_class]

                N_class = T_emb.shape[0]

                if self.num_neg_classes is not None and self.num_neg_classes < N_class - 1:
                    # Negative sampling — keep positive logit, sample K negatives
                    # uniformly per pixel.
                    K = self.num_neg_classes
                    Nv = scores.shape[0]
                    rand_neg = torch.randint(
                        0, N_class, (Nv, K), device=scores.device,
                    )
                    # Pull positive separately
                    pos_logit = scores.gather(1, gt_v.unsqueeze(1))   # [Nv, 1]
                    neg_logit = scores.gather(1, rand_neg)             # [Nv, K]
                    # Build labels for [pos_logit + neg_logit] concat
                    cat_logits = torch.cat([pos_logit, neg_logit], dim=1)   # [Nv, 1+K]
                    cat_targets = torch.zeros_like(cat_logits)
                    cat_targets[:, 0] = 1.0
                    L_sem = F.binary_cross_entropy_with_logits(
                        cat_logits, cat_targets, reduction='mean',
                    )
                else:
                    # Full BCE: positive at gt, negative for all other classes.
                    targets = torch.zeros_like(scores)
                    targets.scatter_(1, gt_v.unsqueeze(1), 1.0)
                    L_sem = F.binary_cross_entropy_with_logits(
                        scores, targets, reduction='mean',
                    )

        out['L_sem'] = L_sem.detach()
        out['valid_pixels'] = valid_count.detach()
        out['k'] = self.log_k.exp().detach()
        out['b'] = self.b.detach()

        objective = self.w_sem * L_sem + self.w_reg * L_reg
        out['objective'] = objective
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: shape inspection for instantiation
# ─────────────────────────────────────────────────────────────────────────────

def stage2_dims_from_stage1(stage1_model, dino_dim: int = 1024) -> Dict[str, int]:
    """Inspect a built Stage 1 model and return the dimensions needed to
    construct Stage2Refiner.  Backbone-agnostic — pulls output_dim from the
    semantic backbone itself.
    """
    backbone = getattr(stage1_model, 'semantic_backbone', None)
    if backbone is None:
        raise ValueError("Stage 1 model has no `semantic_backbone` attribute.")
    output_dim = int(getattr(backbone, 'output_dim'))

    ins_head = getattr(stage1_model, 'instance_head', None)
    if ins_head is None:
        raise ValueError("Stage 1 model has no `instance_head` — cannot infer ins_dim.")
    ins_dim = int(ins_head.out_proj[0].out_channels)

    sem_expand = int(getattr(stage1_model, 'sem_expand', output_dim))

    return {
        'output_dim': output_dim,        # what Stage 2 should output
        'sem_expand': sem_expand,        # actual dim of stage1's semantic_feat_expanded
        'ins_dim':    ins_dim,
        'dino_dim':   dino_dim,
    }
