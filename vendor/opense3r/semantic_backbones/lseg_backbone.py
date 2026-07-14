"""
LSeg backbone — wraps the existing amb3r LSegFeatureExtractor as a unified
SemanticBackbone with the same interface as PE-Core.

Behaviour identical to OpenSe3r's pre-PE LSeg flow: dense feature is used as
both prior and supervision GT (instance aggregation done in loss).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSegBackbone(nn.Module):
    """
    LSeg variant of unified semantic backbone.

    output_dim is 512 (CLIP-B/32 distill). Same feature is used for both
    prior and supervision GT — this matches the legacy flow.
    """

    name = 'lseg'
    output_dim = 512

    def __init__(
        self,
        lseg_ckpt: str = './checkpoints/lseg_minimal_e200.ckpt',
        **_unused,
    ):
        super().__init__()
        from amb3r.lseg import LSegFeatureExtractor

        if os.path.isfile(lseg_ckpt):
            self.lseg = LSegFeatureExtractor.from_pretrained(
                pretrained_model_name_or_path=lseg_ckpt,
                half_res=True,
            )
            print(f"[LSegBackbone] loaded from {lseg_ckpt}")
        else:
            self.lseg = LSegFeatureExtractor(half_res=True)
            print(f"[LSegBackbone] checkpoint not found, using random weights")

        for p in self.lseg.parameters():
            p.requires_grad = False
        self.lseg.eval()

    @torch.no_grad()
    def extract_dense_features(self, images_01: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images_01: [B*T, 3, H, W] in [0, 1]
        Returns:
            [B*T, 512, H, W]
        """
        BT, _, H, W = images_01.shape
        target = 384
        scale  = target / max(H, W)
        H_lseg = int(round(H * scale / 32)) * 32
        W_lseg = int(round(W * scale / 32)) * 32
        x = F.interpolate(
            images_01, size=(H_lseg, W_lseg),
            mode='bilinear', align_corners=False,
        )
        feat = self.lseg.extract_features(x)
        feat = F.interpolate(
            feat, size=(H, W),
            mode='bilinear', align_corners=False,
        )
        return feat

    def forward_prior(
        self,
        images_01: torch.Tensor,
        dino_tokens: torch.Tensor = None,    # ignored (LSeg has its own backbone)
        **_unused,
    ) -> torch.Tensor:
        """
        Returns prior dense feature [B*T, 512, H, W]. Same shape as input H, W.
        """
        return self.extract_dense_features(images_01)

    def forward_supervision(
        self,
        images_01: torch.Tensor,
        instance_mask: torch.Tensor = None,    # ignored (loss does aggregation)
        sem_supervision: str = 'lseg',
        cached_prior: torch.Tensor = None,
        **_unused,
    ) -> torch.Tensor:
        """
        For LSeg: GT is just the dense feature (loss aggregates per-instance).
        Reuses cached_prior if provided to avoid recomputation.
        """
        if cached_prior is not None:
            return cached_prior
        return self.extract_dense_features(images_01)

    @torch.no_grad()
    def encode_text(self, prompts):
        """
        Encode text via LSeg's bundled CLIP-B/32 text encoder.  LSeg distills
        from CLIP-B/32, so its dense image features land in the same 512-dim
        text-aligned space → using CLIP-B/32 text head here is correct.

        Returns [N, 512] L2-normalised.
        """
        import clip
        if isinstance(prompts, str):
            prompts = [prompts]
        device = next(self.lseg.parameters()).device
        toks = clip.tokenize(prompts).to(device)
        emb = self.lseg.clip_pretrained.encode_text(toks).float()
        import torch.nn.functional as F
        return F.normalize(emb, dim=-1)
