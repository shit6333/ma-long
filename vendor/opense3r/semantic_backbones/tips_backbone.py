"""
TIPS / TIPSv2 backbone (Google DeepMind, CVPR 2026).

TIPSv2 = Text-Image Pretraining with Spatial Awareness, with explicit
patch-text alignment.  Vision tower is a DINOv2-style ViT with 1 CLS + 1
register token; text tower is a standard transformer with global avg pool.

Image-text shared dim = embed_dim per variant size (no projection layer):
  - tips-v2-b14   : 768   (12-layer, 12-heads,  patch=14)
  - tips-v2-l14   : 1024  (24-layer, 16-heads,  patch=14)
  - tips-v2-so14  : 1152  (27-layer, 16-heads,  patch=14, mlp_ratio≈3.74)
  - tips-v2-g14   : 1536  (40-layer, 24-heads,  patch=14, swiglu FFN)

All TIPSv2 vision towers default to 448×448 input (32×32 patch grid).

Frozen.  Subclasses VLMBackboneBase, providing only the backbone-specific
hooks (_v_forward, encode_image_global, encode_text).

Checkpoints are NumPy .npz files; download via
`thirdparty/tips/pytorch/checkpoints/download_checkpoints.sh`.
The text encoder + tokenizer are loaded lazily on first encode_text() call
(tokenizer requires `tensorflow_text` which isn't needed at training time).
"""

from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make tips importable.
_TIPS_PARENT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'thirdparty',
)
if _TIPS_PARENT not in sys.path:
    sys.path.insert(0, _TIPS_PARENT)

from tips.pytorch import image_encoder                            # noqa: E402

from ._base import VLMBackboneBase


_VARIANT_BUILDER = {
    'b14':  image_encoder.vit_base,
    'l14':  image_encoder.vit_large,
    'so14': image_encoder.vit_so400m,
    'g14':  image_encoder.vit_giant2,
}

# Short name → config.
_TIPS_V2 = {
    'tips-v2-b14':  dict(builder='b14',  ffn_layer='mlp',    img_size=448, ckpt_stem='tips_v2_oss_b14'),
    'tips-v2-l14':  dict(builder='l14',  ffn_layer='mlp',    img_size=448, ckpt_stem='tips_v2_oss_l14'),
    'tips-v2-so14': dict(builder='so14', ffn_layer='mlp',    img_size=448, ckpt_stem='tips_v2_oss_so14'),
    'tips-v2-g14':  dict(builder='g14',  ffn_layer='swiglu', img_size=448, ckpt_stem='tips_v2_oss_g14'),
}

_VARIANT_LETTER = {
    'b14':  'B',
    'l14':  'L',
    'so14': 'So400m',
    'g14':  'g',
}

_TEXT_VOCAB_SIZE = 32000
_TEXT_MAX_LEN    = 64

_TEXT_HIDDEN_BY_LETTER = {'B': 768, 'L': 1024, 'So400m': 1152, 'g': 1536, 'S': 384}
_TEXT_MLP_BY_LETTER    = {'B': 3072, 'L': 4096, 'So400m': 4304, 'g': 6144, 'S': 1536}
_TEXT_HEADS_BY_LETTER  = {'B': 12, 'L': 16, 'So400m': 16, 'g': 24, 'S': 6}
_TEXT_LAYERS_BY_LETTER = {'B': 12, 'L': 12, 'So400m': 27, 'g': 12, 'S': 12}


def _text_config(letter: str) -> dict:
    return {
        'hidden_size': _TEXT_HIDDEN_BY_LETTER[letter],
        'mlp_dim':     _TEXT_MLP_BY_LETTER[letter],
        'num_heads':   _TEXT_HEADS_BY_LETTER[letter],
        'num_layers':  _TEXT_LAYERS_BY_LETTER[letter],
    }


class TIPSBackbone(VLMBackboneBase):
    """Frozen TIPS / TIPSv2 wrapper for OpenSe3r."""

    def __init__(
        self,
        backbone_name: str = 'tips-v2-b14',
        proxy_beta:    float = 1.2,
        proxy_gamma:   float = 3.0,
        ckpt_dir:      str = None,
        tokenizer_path: str = None,
        input_size:    int = None,
        **_unused,
    ):
        """
        Args:
            input_size:  override the input resolution.  None (default) uses the
                         variant's pretrained res (448 for v2).  Smaller values
                         (e.g. 224) save compute/VRAM at the cost of a coarser
                         patch grid; the positional embedding is interpolated
                         on-the-fly to match (any multiple of patch_size works).
        """
        super().__init__()
        if backbone_name not in _TIPS_V2:
            raise ValueError(
                f"Unknown TIPS variant: '{backbone_name}'. "
                f"Supported: {list(_TIPS_V2)}"
            )
        cfg = _TIPS_V2[backbone_name]
        self.short_name = backbone_name

        # Build vision tower at its pretrained resolution (so pretrained
        # weights load cleanly), then override input_size below if requested.
        # `interpolate_pos_encoding` in TIPS handles arbitrary input sizes.
        builder = _VARIANT_BUILDER[cfg['builder']]
        self.tips = builder(
            img_size              = cfg['img_size'],
            patch_size            = 14,
            ffn_layer             = cfg['ffn_layer'],
            block_chunks          = 0,
            init_values           = 1.0,
            interpolate_antialias = True,
            interpolate_offset    = 0.0,
        )

        # Load weights from .npz if available.
        ckpt_dir = ckpt_dir or os.path.join(
            _TIPS_PARENT, 'tips', 'pytorch', 'checkpoints',
        )
        ckpt_path = os.path.join(ckpt_dir, f"{cfg['ckpt_stem']}_vision.npz")
        if os.path.isfile(ckpt_path):
            try:
                import numpy as np
                ckpt = dict(np.load(ckpt_path, allow_pickle=False))
                ckpt = {k: torch.tensor(v) for k, v in ckpt.items()}
                msg = self.tips.load_state_dict(ckpt, strict=False)
                print(f"[TIPSBackbone] vision loaded: {ckpt_path} "
                      f"(missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)})")
            except Exception as e:
                print(f"[TIPSBackbone] vision ckpt load failed ({e!r}); using random weights")
        else:
            print(f"[TIPSBackbone] vision ckpt not found at {ckpt_path}; using random weights")

        self.tips.eval()
        for p in self.tips.parameters():
            p.requires_grad = False

        # Vision config.
        self.embed_dim   = self.tips.embed_dim
        self.output_dim  = self.tips.embed_dim                  # no separate projection
        self.patch_size  = self.tips.patch_size
        # input_size overridable: the model loads at native res (cfg['img_size'])
        # but forward can run at any size via interpolate_pos_encoding.
        self.input_size  = input_size or cfg['img_size']
        if self.input_size % self.patch_size != 0:
            raise ValueError(
                f"input_size ({self.input_size}) must be divisible by "
                f"patch_size ({self.patch_size})."
            )
        # CLS + register tokens both come before patch tokens in V.
        self.n_special   = 1 + self.tips.num_register_tokens

        self.proxy_beta  = proxy_beta
        self.proxy_gamma = proxy_gamma

        # TIPS' image preprocessor uses no normalisation (mean=0, std=1).
        self.register_buffer('img_mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('img_std',  torch.ones(1, 3, 1, 1))

        # Text encoder + tokenizer are loaded lazily on first encode_text().
        # Tokenizer needs `tensorflow_text` which is not required at training.
        self._text_letter    = _VARIANT_LETTER[cfg['builder']]
        self._text_ckpt_path = os.path.join(ckpt_dir, f"{cfg['ckpt_stem']}_text.npz")
        self._tokenizer_path = tokenizer_path or os.path.join(ckpt_dir, 'tokenizer.model')
        self._text_loaded    = False

    # ------------------------------------------------------------------
    # Backbone-specific hooks
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _v_forward(self, images_prepped: torch.Tensor) -> dict:
        N, _, H, W = images_prepped.shape
        ph = H // self.patch_size
        pw = W // self.patch_size

        # Patch embed + CLS + register + posembed (handled by prepare_tokens).
        x = self.tips.prepare_tokens_with_masks(images_prepped, masks=None)
        # x: [N, 1 + num_register + ph*pw, embed_dim]

        # Forward through blocks[:-1].
        last_idx = len(self.tips.blocks) - 1
        for blk in self.tips.blocks[:last_idx]:
            x = blk(x)
        last_block = self.tips.blocks[last_idx]

        # Manually do norm1 + qkv linear of the last block.
        # TIPS uses fused qkv = Linear(dim, 3*dim, bias=qkv_bias).
        x_norm = last_block.norm1(x)
        attn = last_block.attn
        qkv = attn.qkv(x_norm)                                  # [N, S_seq, 3*dim]

        S_seq = qkv.shape[1]
        head_dim = self.embed_dim // attn.num_heads
        qkv = qkv.reshape(N, S_seq, 3, attn.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        # qkv: [3, N, num_heads, S_seq, head_dim]
        v = qkv[2].contiguous()

        return {
            'v':             v,
            'patch_h':       ph,
            'patch_w':       pw,
            'last_block':    last_block,
            'attn_out_proj': last_block.attn.proj,            # TIPS uses .proj (DINOv2 naming)
            'ls1':           last_block.ls1,                   # active LayerScale (init_values=1.0)
            'mlp_norm':      last_block.norm2,
            'mlp':           last_block.mlp,
            'ls2':           last_block.ls2,
            'final_norm':    self.tips.norm,                   # top-level final LayerNorm
            'final_proj':    None,                             # no projection (output_dim = embed_dim)
        }

    @torch.no_grad()
    def forward_prior_native(self, images_01: torch.Tensor) -> torch.Tensor:
        """
        TIPSv2 dense prior: value-only attention at the last block (the
        TIPS-official zero-shot segmentation path).  In the LAST block we
        skip Q·K^T attention mixing and use V directly — same idea as
        MaskCLIP.  Without this trick, raw `x_norm_patchtokens` from
        forward_features have spatially-coherent structure but are NOT
        well-aligned to the text embedding space (CLS-token mixing in the
        last attention pollutes patches).  With this trick, patches retain
        text alignment and stay locally distinct.

        Reference: thirdparty/tips/pytorch/TIPS_zeroshot_segmentation.ipynb
        ``encode_image_value_attention``.
        """
        x = self._prep(images_01)
        B, _, h, w = x.shape
        x = self.tips.prepare_tokens_with_masks(x, masks=None)
        # x: [B, 1 + num_register + ph*pw, embed_dim]

        last_idx = len(self.tips.blocks) - 1
        for blk in self.tips.blocks[:last_idx]:
            x = blk(x)
        last_block = self.tips.blocks[last_idx]

        # Last block — replace `attn(norm1(x))` with `proj(V_only(norm1(x)))`.
        x_normed = last_block.norm1(x)
        attn = last_block.attn
        n_dim = x_normed.shape[1]
        head_dim = self.embed_dim // attn.num_heads
        qkv = (
            attn.qkv(x_normed)
                .reshape(B, n_dim, 3, attn.num_heads, head_dim)
                .permute(2, 0, 3, 1, 4)
        )
        v = qkv[2]                                              # [B, h, n, d]
        v_out = v.transpose(1, 2).reshape(B, n_dim, self.embed_dim)
        v_out = attn.proj(v_out)
        v_out = last_block.ls1(v_out)
        x_val = v_out + x                                        # residual to pre-norm1 x

        # Standard MLP residual.
        y_val = last_block.norm2(x_val)
        y_val = last_block.ls2(last_block.mlp(y_val))
        x_val = x_val + y_val

        # Top-level final LayerNorm.
        x_val = self.tips.norm(x_val)

        # Drop CLS + register tokens.
        patches = x_val[:, 1 + self.tips.num_register_tokens:, :]   # [B, ph*pw, D]
        ph = h // self.patch_size
        pw = w // self.patch_size
        return patches.reshape(B, ph, pw, -1).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def encode_image_global(self, crops_01: torch.Tensor) -> torch.Tensor:
        """
        Returns the first CLS token, which is the image-text-aligned
        global embedding in TIPS.
        """
        x = self._prep(crops_01)
        first_cls, _second_cls, _patches = self.tips(x)
        emb = first_cls.squeeze(1) if first_cls.dim() == 3 else first_cls
        return F.normalize(emb, dim=-1)

    def encode_text(self, prompts) -> torch.Tensor:
        if not self._text_loaded:
            self._load_text_encoder()
        if isinstance(prompts, str):
            prompts = [prompts]
        device = next(self.tips.parameters()).device
        ids_np, paddings_np = self._tokenizer.tokenize(prompts, max_len=_TEXT_MAX_LEN)
        ids = torch.from_numpy(ids_np).to(device)
        paddings = torch.from_numpy(paddings_np).to(device)
        with torch.no_grad():
            emb = self._text_model(ids, paddings)
        return F.normalize(emb, dim=-1)

    # ------------------------------------------------------------------
    # Lazy text-encoder loading (defers tensorflow_text import)
    # ------------------------------------------------------------------

    def _load_text_encoder(self):
        if not os.path.isfile(self._text_ckpt_path):
            raise FileNotFoundError(
                f"TIPS text checkpoint not found: {self._text_ckpt_path}\n"
                f"Run thirdparty/tips/pytorch/checkpoints/download_checkpoints.sh."
            )
        if not os.path.isfile(self._tokenizer_path):
            raise FileNotFoundError(
                f"TIPS tokenizer not found: {self._tokenizer_path}\n"
                f"Run thirdparty/tips/pytorch/checkpoints/download_checkpoints.sh."
            )

        # Loading text_encoder.py would pull in `tensorflow_text`.  Bypass that
        # by importing only the TextEncoder class via a manual import path
        # (which doesn't require the tensorflow_text-using Tokenizer class).
        # We then build a lightweight SentencePiece-only tokenizer ourselves.
        import importlib.util
        import numpy as np

        te_path = os.path.join(_TIPS_PARENT, 'tips', 'pytorch', 'text_encoder.py')
        spec = importlib.util.spec_from_file_location('_tips_text_encoder', te_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)                          # may pull in tf
            TextEncoder = module.TextEncoder
        except (ImportError, ModuleNotFoundError):
            # tensorflow not installed — fall back to extracting just the
            # TextEncoder class definition by parsing the file (it doesn't
            # actually use tf at runtime, only at import time at the top).
            TextEncoder = _import_tips_text_encoder_class_only(te_path)

        with open(self._text_ckpt_path, 'rb') as f:
            np_weights = dict(np.load(f, allow_pickle=False))
        np_weights.pop('temperature', None)
        np_weights.pop('temperature_contrastive', None)
        torch_weights = {k: torch.from_numpy(v) for k, v in np_weights.items()}

        cfg = _text_config(self._text_letter)
        self._text_model = TextEncoder(cfg, vocab_size=_TEXT_VOCAB_SIZE)
        msg = self._text_model.load_state_dict(torch_weights, strict=False)
        print(f"[TIPSBackbone] text loaded: missing={len(msg.missing_keys)} "
              f"unexpected={len(msg.unexpected_keys)}")
        self._text_model.eval()
        for p in self._text_model.parameters():
            p.requires_grad = False
        self._text_model.to(next(self.tips.parameters()).device)

        # SentencePiece-only tokenizer (pure Python, no tensorflow).
        self._tokenizer = _SentencePieceTokenizer(self._tokenizer_path)
        self._text_loaded = True


# ----------------------------------------------------------------------
# Lightweight SentencePiece tokenizer (no tensorflow_text dependency)
# ----------------------------------------------------------------------

class _SentencePieceTokenizer:
    """
    Drop-in replacement for tips.pytorch.text_encoder.Tokenizer using the
    pure-Python sentencepiece package (no tensorflow / tensorflow_text).

    Output matches the original: numpy (ids, paddings) of shape (B, max_len),
    where paddings == 1 marks pad positions.
    """

    def __init__(self, tokenizer_path: str):
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError(
                "TIPS text encoder requires either `tensorflow_text` or "
                "`sentencepiece`.  Install with:  pip install sentencepiece"
            ) from e
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)

    def tokenize(self, input_text, max_len: int = 64):
        import numpy as np
        if isinstance(input_text, str):
            input_text = [input_text]
        ids_list = [self.sp.EncodeAsIds(s.lower()) for s in input_text]
        B = len(ids_list)
        ids = np.zeros((B, max_len), dtype=np.int32)
        paddings = np.ones((B, max_len), dtype=np.int32)             # 1 = pad
        for i, t in enumerate(ids_list):
            n = min(len(t), max_len)
            ids[i, :n] = t[:n]
            paddings[i, :n] = 0                                       # 0 = real
        return ids, paddings


def _import_tips_text_encoder_class_only(te_path: str):
    """
    Extract just the `TextEncoder` (and its module-level deps) from
    text_encoder.py without executing the top-level `import tensorflow as tf`
    line (so users without tensorflow installed can still run text retrieval).

    We parse the file, drop any `import tensorflow*` lines, and exec the rest.
    Only the Tokenizer class uses tensorflow at runtime, and we replace it
    with our own SentencePiece-only version anyway.
    """
    import re
    with open(te_path) as f:
        src = f.read()
    # Strip `import tensorflow ...` and `import tensorflow_text ...` lines.
    cleaned = re.sub(
        r'^\s*import\s+tensorflow(?:_text)?\b.*$', '',
        src, flags=re.MULTILINE,
    )
    cleaned = re.sub(
        r'^\s*from\s+tensorflow.*$', '', cleaned, flags=re.MULTILINE,
    )
    ns = {'__name__': '_tips_text_encoder_stripped'}
    exec(compile(cleaned, te_path, 'exec'), ns)
    return ns['TextEncoder']
