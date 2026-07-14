"""
OpenAI CLIP backbone (ViT-B/16, ViT-L/14, ViT-L/14@336px).

Subclasses VLMBackboneBase, providing only the backbone-specific hooks
(_v_forward, encode_image_global, encode_text). All other logic lives
in the base class.

Uses TextRegion's vendored ``custom_clip`` (a fork of OpenAI CLIP that
supports value-only / SCLIP-style attention, but we don't actually need
those features here — we only use the standard model + state_dict loader).
"""

from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add TextRegion's custom_clip to import path.
_CC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'thirdparty', 'TextRegion',
)
if _CC_PATH not in sys.path:
    sys.path.insert(0, _CC_PATH)

from custom_clip import clip as _clip                                       # noqa: E402

from ._base import VLMBackboneBase


# Map short names to OpenAI CLIP's _MODELS keys + (input_size, patch_size, output_dim).
# output_dim and width follow OpenAI CLIP's standard config; we read them off
# the loaded state_dict at runtime so we don't hardcode anything fragile.
_SHORT_TO_CLIP_NAME = {
    'clip-vit-b-32':     'ViT-B/32',
    'clip-vit-b-16':     'ViT-B/16',
    'clip-vit-l-14':     'ViT-L/14',
    'clip-vit-l-14-336': 'ViT-L/14@336px',
}


def short_to_clip_name(short: str) -> str:
    if short not in _SHORT_TO_CLIP_NAME:
        raise ValueError(
            f"Unknown CLIP variant: '{short}'. "
            f"Supported: {list(_SHORT_TO_CLIP_NAME)}"
        )
    return _SHORT_TO_CLIP_NAME[short]


# OpenAI CLIP's ImageNet normalisation values.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


class CLIPBackbone(VLMBackboneBase):
    """Frozen OpenAI CLIP (vision + text) wrapper for OpenSe3r."""

    def __init__(
        self,
        backbone_name: str = 'clip-vit-b-16',
        proxy_beta:    float = 1.2,
        proxy_gamma:   float = 3.0,
        download_root: str = None,
        **_unused,
    ):
        super().__init__()
        clip_name = short_to_clip_name(backbone_name)
        self.short_name = backbone_name
        self.clip_name  = clip_name

        # _clip.load returns (model, preprocess) — we only want the model.
        # jit=False forces the python module path (we want to introspect).
        device = 'cpu'
        model, _ = _clip.load(clip_name, device=device, jit=False, download_root=download_root)
        self.clip = model.float()         # cast fp16 weights → fp32 for trainability/grad-free safety
        self.clip.eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        v = self.clip.visual
        self.embed_dim   = v.width
        self.output_dim  = v.proj.shape[1]              # [width, output_dim]
        self.patch_size  = v.patch_size
        self.input_size  = v.input_resolution
        self.n_special   = 1                             # CLIP-ViT: CLS token only

        self.proxy_beta  = proxy_beta
        self.proxy_gamma = proxy_gamma

        self.register_buffer('img_mean', torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer('img_std',  torch.tensor(_CLIP_STD).view(1, 3, 1, 1))

    # ------------------------------------------------------------------
    # Backbone-specific hooks
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _v_forward(self, images_prepped: torch.Tensor) -> dict:
        v_module = self.clip.visual
        N, _, S, _ = images_prepped.shape
        ph = pw = S // self.patch_size

        # Mirror VisionTransformer.forward but stop one layer early and only
        # extract V from the last block.
        x = v_module.conv1(images_prepped)                              # [N, width, ph, pw]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)      # [N, ph*pw, width]
        x = torch.cat([
            v_module.class_embedding.to(x.dtype)
                + torch.zeros(N, 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ], dim=1)                                                        # [N, ph*pw + 1, width]

        if x.shape[1] != v_module.positional_embedding.shape[0]:
            # Non-native resolution → bicubic-interp the positional embedding.
            x = x + v_module.interpolate_pos_encoding(x, S, S).to(x.dtype)
        else:
            x = x + v_module.positional_embedding.to(x.dtype)
        x = v_module.ln_pre(x)

        # Custom CLIP transformer expects LND.
        x = x.permute(1, 0, 2)                                           # [seq, N, width]

        last_idx = len(v_module.transformer.resblocks) - 1
        for blk in v_module.transformer.resblocks[:last_idx]:
            x = blk(x)
        last_block = v_module.transformer.resblocks[last_idx]

        # Manually do ln_1 + qkv linear on the last block.  x is still LND.
        x_ln = last_block.ln_1(x)
        attn = last_block.attn                                           # nn.MultiheadAttention
        proj = F.linear(x_ln, attn.in_proj_weight, attn.in_proj_bias)
        proj = proj.unflatten(-1, (3, attn.embed_dim))                   # [seq, N, 3, width]
        v = proj[..., 2, :]                                              # [seq, N, width]

        # Permute back to NLD and reshape to [N, num_heads, S_seq, head_dim].
        v = v.permute(1, 0, 2).contiguous()                              # [N, seq, width]
        S_seq = v.shape[1]
        head_dim = attn.embed_dim // attn.num_heads
        v = v.reshape(N, S_seq, attn.num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

        # OpenAI CLIP has no LayerScale → use Identity placeholders so the
        # canonical post-attention pipeline is uniform across backbones.
        identity = nn.Identity()

        return {
            'v':             v,
            'patch_h':       ph,
            'patch_w':       pw,
            'last_block':    last_block,
            'attn_out_proj': last_block.attn.out_proj,
            'ls1':           identity,
            'mlp_norm':      last_block.ln_2,
            'mlp':           last_block.mlp,
            'ls2':           identity,
            'final_norm':    v_module.ln_post,
            'final_proj':    v_module.proj,                 # CLIP always has proj
        }

    @torch.no_grad()
    def forward_prior_native(self, images_01: torch.Tensor) -> torch.Tensor:
        """
        OpenAI CLIP native dense prior: standard forward + take all patch
        tokens through ln_post + proj.  Same caveat as PE-Core: CLIP wasn't
        trained for patch-text alignment, so dense features are CLS-biased.
        """
        x = self._prep(images_01)
        # custom_clip's VisionTransformer.forward(return_all=True, csa=False)
        # returns [B, N+1, output_dim] using standard QK attention.
        out = self.clip.visual(x, return_all=True, csa=False)        # [B, N+1, output_dim]
        out = out[:, 1:, :]                                          # drop CLS
        B = out.shape[0]
        S = self.input_size
        ph = pw = S // self.patch_size
        return out.reshape(B, ph, pw, -1).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def encode_image_global(self, crops_01: torch.Tensor) -> torch.Tensor:
        x = self._prep(crops_01)
        emb = self.clip.encode_image(x)                  # [N, output_dim]
        emb = F.normalize(emb, dim=-1)
        return emb

    @torch.no_grad()
    def encode_text(self, prompts) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        toks = _clip.tokenize(prompts).to(next(self.clip.parameters()).device)
        emb = self.clip.encode_text(toks)                # [N, output_dim]
        emb = F.normalize(emb, dim=-1)
        return emb
