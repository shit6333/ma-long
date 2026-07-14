"""
SigLIP / SigLIP-v2 backbone via open_clip (Google, 2024-2025).

Unlike CLIP / PE-Core, SigLIP's ViT does NOT have a separate image-text
projection: ``embed_dim`` equals the shared image-text dim, so raw patch
tokens (post-final-LN) already lie in the text-aligned space.  The global
embedding is produced by a **Multihead Attention Pooling (MAP) head** — a
single learnable latent query attends to all patches via cross-attention,
followed by Proj + MLP residual.  No CLS / register tokens are used
(``num_prefix_tokens=0``).

This makes SigLIP's raw patches a *reasonable* dense semantic teacher even
without explicit dense-loss training (TIPSv2 paper Fig. 2 vs. CLIP/PE).

**Minimal wrapper**: this version supports

  ✅ ``prior_method='native'``           (raw post-LN patches → shared space)
  ✅ ``sem_supervision='patch_mean'``    (hard-mask mean of native patches)
  ✅ ``sem_supervision='mask_crops'``    (per-instance crop forward + MAP head)

  ❌ ``prior_method='proxy_clip'``       — would need ``_v_forward``
  ❌ ``sem_supervision='textregion'``    — would need ``_v_forward``

Calling those raises ``NotImplementedError``.  Add ``_v_forward`` later if
the textregion / proxy_clip paths are wanted for SigLIP.

Image normalisation is Inception-style ``(0.5, 0.5, 0.5)`` → maps to [-1, 1],
NOT ImageNet.

Reference (open_clip configs):
  - ViT-SO400M-14-SigLIP-384  → SigLIP-v1, 1152-dim, patch=14, 384 input
  - ViT-SO400M-14-SigLIP2-378 → SigLIP-v2, 1152-dim, patch=14, 378 input
  - ViT-SO400M-16-SigLIP2-384 → SigLIP-v2, 1152-dim, patch=16, 384 input
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import VLMBackboneBase


# Short name → open_clip model name + pretrained tag + spec.
# Patch grid must divide input_size cleanly (verified in __init__).
_SIGLIP_VARIANTS = {
    # ---------- SigLIP-v1 ----------
    'siglip-so400m-14-384':   dict(oc_name='ViT-SO400M-14-SigLIP-384',   pretrained='webli',
                                   input_size=384, patch_size=14, output_dim=1152),
    'siglip-so400m-14-378':   dict(oc_name='ViT-SO400M-14-SigLIP-378',   pretrained='webli',
                                   input_size=378, patch_size=14, output_dim=1152),

    # ---------- SigLIP-v2 (Gemma tokenizer, 256k vocab) ----------
    'siglip2-so400m-14-378':  dict(oc_name='ViT-SO400M-14-SigLIP2-378',  pretrained='webli',
                                   input_size=378, patch_size=14, output_dim=1152),
    'siglip2-so400m-16-256':  dict(oc_name='ViT-SO400M-16-SigLIP2-256',  pretrained='webli',
                                   input_size=256, patch_size=16, output_dim=1152),
    'siglip2-so400m-16-384':  dict(oc_name='ViT-SO400M-16-SigLIP2-384',  pretrained='webli',
                                   input_size=384, patch_size=16, output_dim=1152),
    'siglip2-so400m-16-512':  dict(oc_name='ViT-SO400M-16-SigLIP2-512',  pretrained='webli',
                                   input_size=512, patch_size=16, output_dim=1152),
    'siglip2-b16-256':        dict(oc_name='ViT-B-16-SigLIP2-256',       pretrained='webli',
                                   input_size=256, patch_size=16, output_dim=768),
    'siglip2-b16-384':        dict(oc_name='ViT-B-16-SigLIP2-384',       pretrained='webli',
                                   input_size=384, patch_size=16, output_dim=768),
    'siglip2-l16-256':        dict(oc_name='ViT-L-16-SigLIP2-256',       pretrained='webli',
                                   input_size=256, patch_size=16, output_dim=1024),
    'siglip2-l16-384':        dict(oc_name='ViT-L-16-SigLIP2-384',       pretrained='webli',
                                   input_size=384, patch_size=16, output_dim=1024),
}


class SigLIPBackbone(VLMBackboneBase):
    """Frozen SigLIP / SigLIP-v2 wrapper for OpenSe3r (minimal)."""

    def __init__(
        self,
        backbone_name: str = 'siglip2-so400m-14-378',
        proxy_beta:    float = 1.2,
        proxy_gamma:   float = 3.0,
        **_unused,
    ):
        super().__init__()
        if backbone_name not in _SIGLIP_VARIANTS:
            raise ValueError(
                f"Unknown SigLIP variant: '{backbone_name}'. "
                f"Supported: {list(_SIGLIP_VARIANTS)}"
            )
        cfg = _SIGLIP_VARIANTS[backbone_name]
        self.short_name = backbone_name

        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "SigLIPBackbone requires open_clip. Install with:\n"
                "  pip install open-clip-torch"
            ) from e

        # Build model + tokenizer.  Auto-downloads weights from HF Hub on first
        # use (~1.2 GB for SO400M).  We discard the transforms — `_prep` does
        # its own resize + normalise.
        self.model, _, _ = open_clip.create_model_and_transforms(
            cfg['oc_name'], pretrained=cfg['pretrained'],
        )
        self._tokenizer = open_clip.get_tokenizer(cfg['oc_name'])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Vision spec.
        self.output_dim = cfg['output_dim']
        self.input_size = cfg['input_size']
        self.patch_size = cfg['patch_size']
        if self.input_size % self.patch_size != 0:
            raise ValueError(
                f"input_size ({self.input_size}) must be divisible by "
                f"patch_size ({self.patch_size}) for {backbone_name}."
            )
        # SigLIP has no CLS / register tokens.
        self.n_special  = 0

        # Inception-style normalisation (NOT ImageNet stats).
        self.register_buffer('img_mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('img_std',  torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        self.proxy_beta  = proxy_beta
        self.proxy_gamma = proxy_gamma

    # ------------------------------------------------------------------
    # Native dense prior — raw post-LN patches from the timm ViT trunk.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_prior_native(self, images_01: torch.Tensor) -> torch.Tensor:
        x = self._prep(images_01)
        B, _, H, W = x.shape
        ph = H // self.patch_size
        pw = W // self.patch_size

        # timm VisionTransformer.forward_features → [B, N_patch (+ prefix), D]
        # already passed through self.norm (final LN).  SigLIP has 0 prefix
        # tokens but defensively strip if present.
        tokens = self.model.visual.trunk.forward_features(x)
        n_prefix = getattr(self.model.visual.trunk, 'num_prefix_tokens', 0)
        if n_prefix > 0:
            tokens = tokens[:, n_prefix:]

        expected = ph * pw
        if tokens.shape[1] != expected:
            raise RuntimeError(
                f"SigLIP forward_features produced {tokens.shape[1]} tokens, "
                f"expected {expected} ({ph}×{pw}). Check input_size / patch_size."
            )

        return tokens.reshape(B, ph, pw, self.output_dim).permute(0, 3, 1, 2).contiguous()

    # ------------------------------------------------------------------
    # Global image embedding (MAP head + L2-norm).  Used by mask_crops.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_image_global(self, crops_01: torch.Tensor) -> torch.Tensor:
        x = self._prep(crops_01)
        emb = self.model.encode_image(x)
        return F.normalize(emb, dim=-1)

    # ------------------------------------------------------------------
    # Text encoder.
    # ------------------------------------------------------------------
    def encode_text(self, prompts) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        device = next(self.model.parameters()).device
        tokens = self._tokenizer(prompts).to(device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
        return F.normalize(emb, dim=-1)

    # ------------------------------------------------------------------
    # Not implemented in minimal version.
    # ------------------------------------------------------------------
    def _v_forward(self, images_prepped: torch.Tensor) -> dict:
        raise NotImplementedError(
            "SigLIPBackbone (minimal) does not implement _v_forward. "
            "Use prior_method='native' and "
            "sem_supervision ∈ {'patch_mean', 'mask_crops'}. "
            "ProxyCLIP / TextRegion paths would need extracting V from the "
            "last block — TODO."
        )
