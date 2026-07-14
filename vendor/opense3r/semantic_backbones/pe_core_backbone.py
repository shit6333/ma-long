"""
PE-Core backbone: ViT-B/16 / L/14 / G/14 (Perception Encoder, FAIR).

Subclasses VLMBackboneBase, providing only the backbone-specific hooks:
  - _v_forward           : PE forward to last-1 layer + ln_1 + qkv → V
  - encode_image_global  : standard PE.encode_image (used by Method 2)
  - encode_text          : standard PE.encode_text (used at inference / vis)

All other logic (ProxyCLIP prior, TextRegion supervision, mask-crop
supervision, multi-resolution V) lives in VLMBackboneBase.

PE forward is split so prior + TextRegion supervision share the forward up to
V (last block's qkv linear); only the post-attention path differs.
"""

from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make perception_models importable.
_PE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'thirdparty', 'perception_models',
)
if _PE_PATH not in sys.path:
    sys.path.insert(0, _PE_PATH)

from core.vision_encoder.pe import CLIP                                    # noqa: E402

from ._base import VLMBackboneBase


_SHORT_TO_PE_NAME = {
    'pe-core-b16-224': 'PE-Core-B16-224',
    'pe-core-l14-336': 'PE-Core-L14-336',
    'pe-core-g14-448': 'PE-Core-G14-448',
    'pe-core-s16-384': 'PE-Core-S16-384',
    'pe-core-t16-384': 'PE-Core-T16-384',
}


def short_to_pe_name(short: str) -> str:
    if short not in _SHORT_TO_PE_NAME:
        raise ValueError(
            f"Unknown PE-Core variant: '{short}'. "
            f"Supported: {list(_SHORT_TO_PE_NAME)}"
        )
    return _SHORT_TO_PE_NAME[short]


class PECoreBackbone(VLMBackboneBase):
    """Frozen PE-Core (vision + text) wrapper for OpenSe3r."""

    def __init__(
        self,
        backbone_name: str = 'pe-core-b16-224',
        proxy_beta:    float = 1.2,
        proxy_gamma:   float = 3.0,
        pretrained:    bool = True,
        **_unused,
    ):
        super().__init__()
        pe_name = short_to_pe_name(backbone_name)
        self.short_name = backbone_name
        self.pe_name    = pe_name

        self.pe = CLIP.from_config(pe_name, pretrained=pretrained)
        self.pe.eval()
        for p in self.pe.parameters():
            p.requires_grad = False

        v = self.pe.visual
        self.embed_dim   = v.width
        self.output_dim  = v.proj_dim if v.proj_dim is not None else v.width
        self.patch_size  = v.patch_size
        self.input_size  = v.image_size
        # PE-Core has CLS token only (no register tokens in vision backbone).
        self.n_special   = 1 if v.use_cls_token else 0

        self.proxy_beta  = proxy_beta
        self.proxy_gamma = proxy_gamma

        # PE expects mean=0.5, std=0.5 normalisation.
        self.register_buffer('img_mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('img_std',  torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    # ------------------------------------------------------------------
    # Backbone-specific hooks
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _v_forward(self, images_prepped: torch.Tensor) -> dict:
        v_module = self.pe.visual
        N, _, S, _ = images_prepped.shape
        ph = pw = S // self.patch_size

        # Mirror VisionTransformer.forward_features but stop one layer early.
        x = v_module.conv1(images_prepped)                              # [N, width, ph, pw]
        x = x.permute(0, 2, 3, 1).reshape(N, -1, v_module.width)        # [N, ph*pw, width]
        if v_module.use_cls_token:
            x = torch.cat([
                v_module.class_embedding.view(1, 1, -1).expand(N, -1, -1),
                x,
            ], dim=1)
        if v_module.use_abs_posemb:
            x = x + v_module._sample_abs_posemb(ph, pw)
        if v_module.use_rope2d:
            v_module.rope.update_grid(x.device, ph, pw)
        x = v_module.ln_pre(x)

        last_idx = len(v_module.transformer.resblocks) - 1
        for blk in v_module.transformer.resblocks[:last_idx]:
            x = blk(x)
        last_block = v_module.transformer.resblocks[last_idx]

        # Manually do ln_1 + qkv linear of the last block.
        x_ln = last_block.ln_1(x)
        attn = last_block.attn
        proj = F.linear(x_ln, attn.in_proj_weight, attn.in_proj_bias)
        proj = proj.unflatten(-1, (3, attn.embed_dim))
        v = proj[..., 2, :]

        S_seq = v.shape[1]
        v = v.reshape(N, S_seq, attn.num_heads, attn.head_dim).permute(0, 2, 1, 3).contiguous()

        proj_w = v_module.proj if v_module.proj_dim is not None else None

        return {
            'v':             v,
            'patch_h':       ph,
            'patch_w':       pw,
            'last_block':    last_block,
            'attn_out_proj': last_block.attn.out_proj,
            'ls1':           last_block.ls_1,             # Identity if ls_init_value is None
            'mlp_norm':      last_block.ln_2,
            'mlp':           last_block.mlp,
            'ls2':           last_block.ls_2,             # Identity if ls_init_value is None
            'final_norm':    v_module.ln_post,
            'final_proj':    proj_w,
        }

    @torch.no_grad()
    def forward_prior_native(self, images_01: torch.Tensor) -> torch.Tensor:
        """
        PE-Core native dense prior: standard forward + take all patch tokens
        through ln_post + proj.  Note PE-Core was trained CLIP-style so dense
        patch features may carry CLS-bias — prefer 'proxy_clip' for cleaner
        spatial structure (this option is here for ablation).
        """
        x = self._prep(images_01)
        # forward_features: norm=True, strip_cls_token=True → [B, N_patch, width]
        feat = self.pe.visual.forward_features(x, norm=True, strip_cls_token=True)
        if self.pe.visual.proj_dim is not None:
            feat = feat @ self.pe.visual.proj                 # [B, N_patch, output_dim]
        B = feat.shape[0]
        S = self.input_size
        ph = pw = S // self.patch_size
        return feat.reshape(B, ph, pw, -1).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def encode_image_global(self, crops_01: torch.Tensor) -> torch.Tensor:
        x = self._prep(crops_01)
        emb = self.pe.encode_image(x, normalize=True)
        return emb

    @torch.no_grad()
    def encode_text(self, prompts) -> torch.Tensor:
        if not hasattr(self, '_tokenizer'):
            from core.vision_encoder.tokenizer import SimpleTokenizer
            self._tokenizer = SimpleTokenizer(context_length=32)

        if isinstance(prompts, str):
            prompts = [prompts]
        toks = self._tokenizer(prompts).to(next(self.pe.parameters()).device)
        return self.pe.encode_text(toks, normalize=True)
