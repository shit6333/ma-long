"""C-RADIOv4 semantic backbone (agglomerative VFM + SigLIP2 adaptor).

Replaces the TIPS/LSeg semantic backbone with a single frozen C-RADIOv4 encoder
whose `siglip2-g` adaptor `head_mlp` projects each RADIO patch token into the
SigLIP2 text-aligned space — the RADSeg dense-OVSS trick.  Optionally denoised
with SCGA (self-correlating global aggregation).

Exposed to OpenSe3r via the standard semantic-backbone interface:
  - forward_prior_native(images_01) -> [BT, output_dim, ph, pw]   (dense prior + patch_mean supervision)
  - encode_text(prompts)            -> [N, output_dim]            (open-vocab text)
  - output_dim / patch_size / input_size attributes

Use with `--prior_method native --sem_supervision patch_mean`.  output_dim is
the SigLIP2 text dim (1536 for siglip2-giant), NOT the RADIO backbone embed_dim.

C-RADIOv4 is patch-16 (vs TIPS/DINO patch-14); the caller (model_da3) interpolates
the returned dense map to the working (H, W), so the grid change is transparent to
downstream heads.  RADIO ingests [0,1] RGB and applies OpenAI-CLIP norm internally.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import VLMBackboneBase

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_RADIO_DIR = os.path.join(_REPO_ROOT, 'thirdparty', 'RADIO')

# short name -> (torchhub version, adaptor key)
# v4 ships only the siglip2-g adaptor; the CLIP(DFN) adaptor exists in v3 —
# 'radio-v3l-clip' = C-RADIOv3-L (patch 16) + DFN-CLIP text space (1024-d,
# cheaper than siglip2's 1536).  Adaptor interface (head_mlp / tokenizer /
# encode_text) is identical across versions, so everything below is shared.
_VARIANTS = {
    'radio-v4-so400m': ('c-radio_v4-so400m', 'siglip2-g'),
    'radio-v4-h':      ('c-radio_v4-h',      'siglip2-g'),
    'radio-v3l-clip':  ('c-radio_v3-l',      'clip'),
    # RADIO v2.5-L + SigLIP-so400m adaptor: 1152-d text space (vs siglip2's
    # 1536) — probed as a cheaper dense teacher candidate.
    'radio-v25l-siglip': ('radio_v2.5-l',    'siglip'),
    # C-RADIOv3-B + SigLIP2 adaptor (SigLIP2Adaptor class, so400m text space
    # 1152-d): SigLIP2-generation dense alignment at ViT-B cost.
    'radio-v3b-siglip2': ('c-radio_v3-b',    'siglip2'),
}


def _last_linear_out(module: nn.Module) -> int:
    out = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            out = m.out_features
    if out is None:
        raise RuntimeError("head_mlp has no nn.Linear to read output dim from")
    return out


def _scga(feat_chw: torch.Tensor, scaling: float = 10.0) -> torch.Tensor:
    """Self-Correlating Global Aggregation (RADSeg).  feat_chw: [B, C, h, w]."""
    b, c, h, w = feat_chw.shape
    fm = feat_chw.flatten(2, 3).transpose(1, 2)               # [B, hw, C]
    sim = F.normalize(fm, dim=-1)
    sm = torch.bmm(sim, sim.transpose(1, 2))
    sm = (sm - sm.mean()) * scaling
    sm = sm.masked_fill(sm < 0, float('-inf'))
    sm = F.softmax(sm, dim=-1)
    out = torch.bmm(sm, fm).transpose(1, 2).view(b, c, h, w)
    return out


class RadioBackbone(VLMBackboneBase):
    """Frozen C-RADIOv4 + siglip2-g dense semantic teacher/prior.

    Only `forward_prior_native`, `encode_text` and the dim attributes are
    provided; supervision runs through the base-class `patch_mean` path (which
    calls `forward_prior_native` then does per-instance hard-mask mean).
    """

    def __init__(self,
                 backbone_name: str = 'radio-v4-so400m',
                 use_scga: bool = True,
                 scga_scaling: float = 10.0,
                 text_templates: tuple = ("a photo of a {}.",
                                          "a {} in a room.",
                                          "there is a {} in the scene."),
                 **_unused):
        super().__init__()
        if backbone_name not in _VARIANTS:
            raise ValueError(f"Unknown RADIO backbone '{backbone_name}'. "
                             f"Choose from {list(_VARIANTS)}.")
        version, adaptor_key = _VARIANTS[backbone_name]
        self.backbone_name = backbone_name
        self.use_scga = bool(use_scga)
        self.scga_scaling = float(scga_scaling)
        self.text_templates = tuple(text_templates)

        radio = torch.hub.load(_RADIO_DIR, 'radio_model', source='local',
                               version=version, adaptor_names=[adaptor_key],
                               progress=True, skip_validation=True)
        # Grab the siglip adaptor, then disable RADIO's auto-adaptor pass so the
        # backbone forward returns a plain (summary, features) tuple.
        self.adaptor = radio.adaptors[adaptor_key]
        radio.adaptors = None
        self.radio = radio

        self.patch_size = int(radio.patch_size)
        self.output_dim = _last_linear_out(self.adaptor.head_mlp)   # SigLIP2 text dim (1536 giant)
        self.input_size = 512                                       # nominal; forward snaps to mult-of-patch
        self.n_special = 0                                          # RADIO forward already strips CLS/registers
        # RADIO backbone spatial-feature width — used when RADIO also replaces the
        # DINO appearance tokens (dino_backbone='radio'), probed with a tiny forward.
        with torch.no_grad():
            _, _probe = self.radio(torch.zeros(1, 3, self.patch_size * 2, self.patch_size * 2))
        self.appearance_dim = int(_probe.shape[-1])                 # RADIO embed_dim (1152 so400m / 1280 h)

        # Identity norm buffers (RADIO applies OpenAI-CLIP norm internally); kept so
        # any accidental base-class _prep() call is a no-op.
        self.register_buffer('img_mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('img_std',  torch.ones(1, 3, 1, 1))

        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    def _snap(self, images_01: torch.Tensor) -> torch.Tensor:
        """Resize so H, W are multiples of patch_size (RADIO requirement)."""
        _, _, H, W = images_01.shape
        p = self.patch_size
        Hs, Ws = max(p, round(H / p) * p), max(p, round(W / p) * p)
        if (Hs, Ws) != (H, W):
            images_01 = F.interpolate(images_01, (Hs, Ws), mode='bilinear',
                                      align_corners=False)
        return images_01

    @torch.no_grad()
    def forward_features(self, images_01: torch.Tensor, want_dense: bool = True):
        """ONE RADIO forward → (raw appearance, dense SigLIP semantic).

        raw:   [BT, appearance_dim, ph, pw]  RADIO backbone spatial features (for
               the CausalBranch DINO-token slot when dino_backbone='radio').
        dense: [BT, output_dim,     ph, pw]  (optional SCGA) → siglip head_mlp per
               patch = SigLIP-text-aligned semantic (prior + patch_mean supervision).
        Computing both from a single RADIO forward avoids a second encoder pass.
        """
        x = self._snap(images_01.float())
        BT, _, H, W = x.shape
        ph, pw = H // self.patch_size, W // self.patch_size
        summary, feats = self.radio(x)                     # [BT, L, embed_dim] (NLC, stripped)
        feats = feats.float()
        raw = feats.transpose(1, 2).reshape(BT, -1, ph, pw).contiguous()   # appearance (no SCGA)
        dense = None
        if want_dense:
            f = feats
            if self.use_scga:
                fc = f.transpose(1, 2).reshape(BT, -1, ph, pw)
                fc = _scga(fc, self.scga_scaling)
                f = fc.flatten(2, 3).transpose(1, 2)
            dense = self.adaptor.head_mlp(f)                                # [BT, L, output_dim]
            dense = dense.transpose(1, 2).reshape(BT, self.output_dim, ph, pw).contiguous()
        return raw, dense

    @torch.no_grad()
    def forward_prior_native(self, images_01: torch.Tensor) -> torch.Tensor:
        """[BT,3,H,W] in [0,1] -> [BT, output_dim, ph, pw] SigLIP-text-aligned dense."""
        _raw, dense = self.forward_features(images_01, want_dense=True)
        return dense

    @torch.no_grad()
    def encode_image_global(self, crops_01: torch.Tensor) -> torch.Tensor:
        """Image-level SigLIP-aligned embedding (adaptor summary). [N, output_dim]."""
        x = self._snap(crops_01.float())
        summary, feats = self.radio(x)
        emb = self.adaptor.head_mlp(feats.float().mean(dim=1))  # pooled patch → text space
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_text(self, prompts) -> torch.Tensor:
        """Open-vocab text → [N, output_dim] L2-normalised, template-ensembled."""
        if isinstance(prompts, str):
            prompts = [prompts]
        device = next(self.radio.parameters()).device
        embs = []
        for c in prompts:
            toks = self.adaptor.tokenizer([t.format(c) for t in self.text_templates])
            toks = ({k: v.to(device) for k, v in toks.items()}
                    if isinstance(toks, dict) else toks.to(device))
            te = self.adaptor.encode_text(toks, normalize=True).float()   # [n_templ, D]
            embs.append(F.normalize(te.mean(0), dim=-1))
        return torch.stack(embs)
