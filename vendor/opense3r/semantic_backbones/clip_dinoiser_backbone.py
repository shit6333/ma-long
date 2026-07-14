"""
CLIP-DINOiser semantic backbone (no mmseg / no models.builder dependency).

CLIP-DINOiser (Wysoczanska et al., 2024) refines MaskCLIP dense features by a
learned 3x3 conv (`obj_proj`) that produces DINO-like patch self-similarity;
the similarity is then used as weights to spatially smooth MaskCLIP's
text-aligned dense features.  At inference time no DINOv2 is needed — the
conv internalises a DINO-affinity prior at training.

This wrapper:
  * Loads CLIP ViT-B/16 with `laion2b_s34b_b88k` weights via `open_clip`
    (the dinoiser ckpt was trained against these weights — OpenAI weights
    will not produce sensible affinity).
  * Reproduces MaskCLIP-style V extraction at the last block, projected to
    the 512-dim CLIP image-text space (via `visual.proj`).
  * Captures the 3rd-from-last block's ln_2 output for the trainable
    `obj_proj` conv → 256-dim affinity space → cosine self-sim → corrs.
  * Weighted-pools the MaskCLIP V map by corrs to produce refined dense
    features (the actual "DINOiser output").

Use as:
  - prior:       forward_prior(images_01) → [B, 512, H, W]
  - supervision: forward_supervision(images_01, ...) → same dense feat
                 (per-instance aggregation is done inside the loss, just
                 like LSegBackbone — we don't pre-aggregate here).
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# OpenAI CLIP / open_clip ImageNet stats (both use the same).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def _try_import_open_clip():
    try:
        import open_clip                                    # noqa: F401
        return open_clip
    except ImportError as e:
        raise ImportError(
            "CLIPDinoiserBackbone requires `open-clip-torch` because the "
            "released CLIP-DINOiser checkpoint was trained against "
            "`laion2b_s34b_b88k` weights (open_clip), not OpenAI CLIP.  "
            "Install with: pip install open-clip-torch"
        ) from e


class CLIPDinoiserBackbone(nn.Module):
    """
    CLIP-DINOiser wrapper exposing the OpenSe3r semantic-backbone interface.

    Trainable buffers (loaded from `last.pt`): `obj_proj`, `bkg_decoder`.
    Everything else (CLIP ViT-B/16) is frozen.
    """

    name = 'clip-dinoiser'

    def __init__(
        self,
        clip_dinoiser_ckpt: str = './thirdparty/clip_dinoiser/checkpoints/last.pt',
        clip_arch:          str = 'ViT-B-16',
        clip_pretrained:    str = 'laion2b_s34b_b88k',
        gamma:              float = 0.2,
        feats_idx:          int = -3,
        in_dim:             int = 256,
        conv_kernel:        int = 3,
        **_unused,
    ):
        super().__init__()
        open_clip = _try_import_open_clip()

        self.gamma     = gamma
        self.feats_idx = feats_idx

        # 1. CLIP ViT-B/16 (open_clip, laion2b weights).
        model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained=clip_pretrained)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self.clip = model
        self.tokenizer = open_clip.get_tokenizer(clip_arch)

        v = self.clip.visual
        # open_clip ViT-B/16: width=768, output_dim=512, patch=16, input=224
        self.embed_dim  = v.conv1.out_channels                 # 768
        self.output_dim = v.proj.shape[1]                      # 512 (CLIP image-text dim)
        self.patch_size = v.patch_size[0] if isinstance(v.patch_size, (tuple, list)) else v.patch_size
        self.input_size = v.image_size[0] if isinstance(v.image_size, (tuple, list)) else v.image_size
        self.n_special  = 1                                    # CLS

        # 2. Trainable convs (loaded from dinoiser ckpt).
        self.obj_proj = nn.Conv2d(
            self.embed_dim, in_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            padding_mode='replicate',
        )
        self.bkg_decoder = nn.Conv2d(self.embed_dim, 1, kernel_size=1)
        self.in_dim = in_dim

        # 3. Hooks for V extraction (resblocks[-2] output) and obj_proj input
        #    (resblocks[feats_idx].ln_2 output).
        self._hook_features: dict = {}
        blocks = self.clip.visual.transformer.resblocks
        blocks[-2].register_forward_hook(self._make_hook('clip_v_in'))
        blocks[feats_idx].ln_2.register_forward_hook(self._make_hook('clip_inter'))

        # 4. Image normalisation (CLIP stats).
        self.register_buffer('img_mean', torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer('img_std',  torch.tensor(_CLIP_STD).view(1, 3, 1, 1))

        # 5. Load dinoiser ckpt (only obj_proj / bkg_decoder).
        if os.path.isfile(clip_dinoiser_ckpt):
            ckpt = torch.load(clip_dinoiser_ckpt, map_location='cpu', weights_only=False)
            sd   = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            mine = {k: v for k, v in sd.items()
                    if k.startswith('obj_proj.') or k.startswith('bkg_decoder.')}
            missing, unexpected = self.load_state_dict(mine, strict=False)
            # We expect ALL CLIP keys to be missing (we don't load them here, they're already in self.clip).
            real_missing = [k for k in missing
                            if not (k.startswith('clip.') or k.startswith('img_mean') or k.startswith('img_std'))]
            print(f'[CLIPDinoiserBackbone] loaded dinoiser ckpt from {clip_dinoiser_ckpt}  '
                  f'(real-missing={len(real_missing)}, unexpected={len(unexpected)})')
        else:
            print(f'[CLIPDinoiserBackbone] WARNING ckpt not found at {clip_dinoiser_ckpt}; '
                  f'obj_proj / bkg_decoder are RANDOM — affinity will be meaningless.')

        # Freeze the trainable parts too — at inference time everything is frozen.
        for p in self.obj_proj.parameters():
            p.requires_grad = False
        for p in self.bkg_decoder.parameters():
            p.requires_grad = False
        self.eval()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            self._hook_features[name] = out
        return hook

    def _prep(self, images_01: torch.Tensor) -> torch.Tensor:
        """Pad to patch multiple, then normalise.  Keeps native aspect ratio
        (unlike `_prep` in VLMBackboneBase, which resizes to (S, S)).
        Patch grid is (H/patch, W/patch) after padding."""
        H, W = images_01.shape[-2:]
        ps = self.patch_size
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h or pad_w:
            images_01 = F.pad(images_01, (0, pad_w, 0, pad_h), value=0.0)
        return (images_01 - self.img_mean) / self.img_std

    def _interp_pos_embed(self, x_seq_len: int, hw_shape: tuple) -> torch.Tensor:
        """Bicubic-interp positional embedding to match the current grid.
        Returns posemb on x.device with shape [seq, embed_dim]."""
        v = self.clip.visual
        pos_embed = v.positional_embedding                                  # [pos_len, C]
        pos_len, C = pos_embed.shape
        if x_seq_len == pos_len:
            return pos_embed
        # CLS + grid layout (CLS first, then row-major patches).
        cls_pe   = pos_embed[:1]
        grid_pe  = pos_embed[1:]
        side = int(round((pos_len - 1) ** 0.5))
        grid_pe = grid_pe.reshape(1, side, side, C).permute(0, 3, 1, 2)
        grid_pe = F.interpolate(grid_pe, size=hw_shape, mode='bicubic', align_corners=False)
        grid_pe = grid_pe.permute(0, 2, 3, 1).reshape(hw_shape[0] * hw_shape[1], C)
        return torch.cat([cls_pe, grid_pe], dim=0)

    @torch.no_grad()
    def _maskclip_v_and_inter(self, images_norm: torch.Tensor) -> tuple:
        """
        Run CLIP forward, capture:
          * V at the last block (MaskCLIP-style, projected to image-text dim).
          * `feats_idx` block's ln_2 output (input to obj_proj).

        Returns:
          v_clip:     [B, output_dim=512, h, w]   refined-target features
          clip_inter: [B, embed_dim=768, h, w]    obj_proj input
          (h, w):     patch grid
        """
        v_module = self.clip.visual
        B, _, H, W = images_norm.shape
        ps = self.patch_size
        h, w = H // ps, W // ps

        # Patch embed.
        x = v_module.conv1(images_norm)                                     # [B, C, h, w]
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)                   # [B, h*w, C]
        cls = v_module.class_embedding.to(x.dtype) + torch.zeros(
            B, 1, x.shape[-1], dtype=x.dtype, device=x.device,
        )
        x = torch.cat([cls, x], dim=1)                                      # [B, h*w+1, C]

        # Pos embed (interp if needed).  Don't mutate `positional_embedding`
        # in place — open_clip's might be a Parameter we want to keep.
        pos_embed = self._interp_pos_embed(x.shape[1], (h, w)).to(x.dtype)
        x = x + pos_embed
        x = v_module.ln_pre(x)

        # open_clip transformer takes [B, N, C] (not LND).  Run all blocks
        # except the last; hooks will fire along the way.
        n = len(v_module.transformer.resblocks)
        for blk in v_module.transformer.resblocks[:n - 1]:
            x = blk(x)

        # Manual last-block: extract V + run residual+MLP path (MaskCLIP).
        last = v_module.transformer.resblocks[-1]
        v_clip = self._run_maskclip_last_block(x, last)                     # [B, h*w+1, C=embed_dim]
        v_clip = v_module.ln_post(v_clip)
        v_clip = v_clip[:, 1:]                                              # drop CLS
        v_clip = v_clip @ v_module.proj                                     # [B, h*w, output_dim=512]
        v_clip = v_clip.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()

        # Pull the `clip_inter` hook capture (-3 block ln_2 output).
        clip_inter = self._hook_features['clip_inter']                      # [B, h*w+1, C]
        clip_inter = clip_inter[:, 1:].permute(0, 2, 1).reshape(B, self.embed_dim, h, w).contiguous()

        return v_clip, clip_inter, (h, w)

    @staticmethod
    def _run_maskclip_last_block(x: torch.Tensor, block) -> torch.Tensor:
        """
        Replicate MaskCLIP's V-replacing last block:
          y = ln_1(x) → in_proj (Q,K,V) → take V → out_proj
          residual: v + x
          mlp:      v + mlp(ln_2(v))
        """
        y = block.ln_1(x)
        y = F.linear(y, block.attn.in_proj_weight, block.attn.in_proj_bias)
        B, N, C3 = y.shape
        # Split into Q, K, V (each [B, N, C]).
        y = y.reshape(B, N, 3, C3 // 3).permute(2, 0, 1, 3)
        v = y[2]
        v = F.linear(v, block.attn.out_proj.weight, block.attn.out_proj.bias)
        v = v + x
        v = v + block.mlp(block.ln_2(v))
        return v

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_prior_native(self, images_01: torch.Tensor) -> torch.Tensor:
        """Native MaskCLIP dense feature, no DINOiser refinement.
        Useful as ablation against the refined version."""
        H, W = images_01.shape[-2:]
        x = self._prep(images_01)
        v_clip, _, _ = self._maskclip_v_and_inter(x)
        return F.interpolate(v_clip, size=(H, W), mode='bilinear', align_corners=False)

    @torch.no_grad()
    def forward_prior(
        self,
        images_01: torch.Tensor,
        dino_tokens: torch.Tensor = None,    # ignored — dinoiser predicts its own affinity
        **_unused,
    ) -> torch.Tensor:
        """
        CLIP-DINOiser refined dense feature.
        Returns:
            [B, 512, H, W] in CLIP image-text space.
        """
        H, W = images_01.shape[-2:]
        x = self._prep(images_01)
        v_clip, clip_inter, (h, w) = self._maskclip_v_and_inter(x)

        # 1. obj_proj → 256-dim per pixel → cosine sim → corrs [B, hw, h, w].
        proj = self.obj_proj(clip_inter)                                    # [B, in_dim, h, w]
        proj = proj.reshape(proj.shape[0], self.in_dim, -1)
        proj = F.normalize(proj, dim=1, eps=1e-6)                           # [B, in_dim, hw]
        corrs = torch.einsum('bcm,bcn->bmn', proj, proj)                    # [B, hw, hw]
        corrs = corrs.reshape(corrs.shape[0], h * w, h, w)                  # [B, hw, h, w]
        if self.gamma is not None:
            corrs = torch.where(corrs < self.gamma, torch.zeros_like(corrs), corrs)

        # 2. Weighted pool MaskCLIP V by corrs.
        refined = torch.einsum('bnij,bcij->bcn', corrs, v_clip)             # [B, C, hw]
        norm    = corrs.flatten(-2, -1).sum(dim=-1).clamp_min(1e-6)         # [B, hw]
        refined = refined / norm.unsqueeze(1)                               # [B, C, hw]
        refined = refined.reshape(refined.shape[0], -1, h, w)               # [B, C, h, w]

        # 3. Upsample to (H, W) for the rest of the pipeline.
        return F.interpolate(refined, size=(H, W), mode='bilinear', align_corners=False)

    @torch.no_grad()
    def forward_supervision(
        self,
        images_01: torch.Tensor,
        instance_mask: torch.Tensor = None,    # ignored — loss does aggregation
        sem_supervision: str = 'lseg',
        cached_prior: torch.Tensor = None,
        **_unused,
    ) -> torch.Tensor:
        """
        Dense GT for supervision.  Mirrors LSegBackbone: loss aggregates
        per-instance (no pre-pooling here).  Reuses cached_prior to avoid
        recomputation when the prior was already run upstream.
        """
        if cached_prior is not None:
            return cached_prior
        return self.forward_prior(images_01)

    @torch.no_grad()
    def encode_image_global(self, crops_01: torch.Tensor) -> torch.Tensor:
        """Standard CLIP image embedding for crops (used by `mask_crops` path)."""
        # Resize crops to native input_size first.
        S = self.input_size
        x = F.interpolate(crops_01, size=(S, S), mode='bilinear', align_corners=False)
        x = (x - self.img_mean) / self.img_std
        emb = self.clip.encode_image(x)                                     # [N, output_dim]
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_text(self, prompts) -> torch.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        device = next(self.clip.parameters()).device
        toks = self.tokenizer(prompts).to(device)
        emb = self.clip.encode_text(toks).float()
        return F.normalize(emb, dim=-1)
