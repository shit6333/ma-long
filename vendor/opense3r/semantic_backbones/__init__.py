"""
Semantic backbones for OpenSe3r.

Three components per backbone:
  - forward_prior:        dense feature for input fusion + DPT cond
                          (always run; LSeg uses its own DPT, PE-Core uses ProxyCLIP)
  - forward_supervision:  dense GT feature for cosine distillation (training only)
                          ('textregion', 'mask_crops', or backbone's default ='lseg')
  - encode_text:          text query → [N, output_dim] for visualization

Available backbones:
  - 'lseg'             LSeg ViT-B/32 distill (legacy default, output_dim=512)
  - 'pe-core-b16-224'  PE-Core ViT-B/16, 224×224 input  (output_dim=1024)
  - 'pe-core-l14-336'  PE-Core ViT-L/14, 336×336 input  (output_dim=1024)
  - 'pe-core-g14-448'  PE-Core ViT-G/14, 448×448 input  (output_dim=1280)

Usage::

    backbone = build_semantic_backbone('pe-core-b16-224', proxy_beta=1.2, proxy_gamma=3.0)
    prior_feat = backbone.forward_prior(images_01, dino_tokens=..., dino_grid=(ph,pw))
    gt_feat    = backbone.forward_supervision(
        images_01, instance_mask, sem_supervision='textregion'
    )
"""

from .lseg_backbone import LSegBackbone


def build_semantic_backbone(name: str, **kwargs):
    """
    Build a semantic backbone by short name.

    PE-Core / OpenAI CLIP are imported lazily so that environments without
    those dependencies (timm.layers, custom_clip) can still use the default
    'lseg' backbone unchanged.

    Args:
        name: one of:
              - 'lseg'             (output_dim=512, ViT-B/32 distill)
              - 'pe-core-b16-224'  (output_dim=1024)
              - 'pe-core-l14-336'  (output_dim=1024)
              - 'pe-core-g14-448'  (output_dim=1280)
              - 'clip-vit-b-32'    (output_dim=512)
              - 'clip-vit-b-16'    (output_dim=512)
              - 'clip-vit-l-14'    (output_dim=768)
              - 'clip-vit-l-14-336'(output_dim=768)
              - 'tips-v2-b14'      (output_dim=768,  patch=14, 448 input)
              - 'tips-v2-l14'      (output_dim=1024, patch=14, 448 input)
              - 'tips-v2-so14'     (output_dim=1152, patch=14, 448 input)
              - 'tips-v2-g14'      (output_dim=1536, patch=14, 448 input)
              - 'clip-dinoiser'    (output_dim=512, ViT-B/16 + learned DINO-affinity refinement)
        kwargs: forwarded to the backbone constructor (extras are ignored).
    """
    if name == 'lseg':
        return LSegBackbone(**kwargs)
    if name.startswith('pe-core-'):
        from .pe_core_backbone import PECoreBackbone
        return PECoreBackbone(backbone_name=name, **kwargs)
    if name.startswith('clip-vit-'):
        from .clip_backbone import CLIPBackbone
        return CLIPBackbone(backbone_name=name, **kwargs)
    if name.startswith('tips-'):
        from .tips_backbone import TIPSBackbone
        return TIPSBackbone(backbone_name=name, **kwargs)
    if name.startswith('siglip-') or name.startswith('siglip2-'):
        from .siglip_backbone import SigLIPBackbone
        return SigLIPBackbone(backbone_name=name, **kwargs)
    if name == 'clip-dinoiser':
        from .clip_dinoiser_backbone import CLIPDinoiserBackbone
        return CLIPDinoiserBackbone(**kwargs)
    if name.startswith('radio-'):
        from .radio_backbone import RadioBackbone
        return RadioBackbone(backbone_name=name, **kwargs)
    raise ValueError(
        f"Unknown semantic backbone: '{name}'. Supported: "
        f"'lseg', 'pe-core-b16-224', 'pe-core-l14-336', 'pe-core-g14-448', "
        f"'pe-core-s16-384', 'pe-core-t16-384', "
        f"'clip-vit-b-32', 'clip-vit-b-16', 'clip-vit-l-14', 'clip-vit-l-14-336', "
        f"'tips-v2-b14', 'tips-v2-l14', 'tips-v2-so14', 'tips-v2-g14', "
        f"'siglip-so400m-14-384', 'siglip-so400m-14-378', "
        f"'siglip2-so400m-14-378', 'siglip2-so400m-16-256/-384/-512', "
        f"'siglip2-b16-256/-384', 'siglip2-l16-256/-384', "
        f"'clip-dinoiser'."
    )


__all__ = ['LSegBackbone', 'build_semantic_backbone']
