# OpenSe3r

Online causal semantic / instance SLAM backbone built on top of VGGT (or DA3)
geometry + a frozen vision-language semantic backbone. Trainable on top:
DepthEncoder, CausalBranch (causal transformer with KV-cache), Semantic /
Instance DPT heads, and optional pose / Plücker conditioning encoders.

This doc covers the **design intent + flag surface**. For full details read
the code (it's annotated). Keep this doc in sync with future redesigns.

---

## Architecture overview

```
                ┌─────────────────────────────┐
images [B,T,3,H,W] →┤ VGGT (frozen)                         ├→ depth, world_points,
                │   ├─ encode_patch_tokens (DINOv2)        │  extrinsics, intrinsics
                │   └─ aggregator + heads                  │  + DINOv2 patch tokens
                └─────────────────────────────┘
                                ▼
                ┌─────────────────────────────┐
                │ Semantic backbone (frozen)  │  → prior_feat (dense)
                │ {LSeg | PE-Core | CLIP |    │  → GT (training only)
                │  TIPSv2}                    │
                └─────────────────────────────┘
                                ▼
                ┌─────────────────────────────┐
                │ DepthEncoder (trainable)    │  per-frame median-normalised
                │ depth → patch tokens        │  depth → geometry tokens
                └─────────────────────────────┘
                                ▼
                ┌─────────────────────────────┐
                │ CausalBranch (trainable)    │  DINOv2 + prior fused via
                │ 12 × (frame_block,          │  input_proj; depth tokens
                │       global_block) +       │  injected at input (optional)
                │ DepthCrossAttn (every 3)    │  + per-frame pose / Plücker
                │ KV-cache for streaming      │  tokens (optional)
                └─────────────────────────────┘
                                ▼
                ┌──────────────┬──────────────┐
                │ SemanticHead │ InstanceHead │  Two instance head modes:
                │ (DPT)        │ (DPT │ Query)│   - DPT: per-pixel feature
                │              │              │   - Query: Mask2Former-style
                │              │              │     discrete proposals
                └──────────────┴──────────────┘
                                ▼
              semantic_feat [B,T,sem_dim,H,W]    ← compact feature for SLAM voxel map
              semantic_feat_expanded             ← MLP-expanded to backbone dim,
                  [B,T,sem_expand,H,W]              used for cosine loss + text retrieval
              # Dense instance head:
              instance_feat [B,T,ins_dim,H,W] + conf
              # Query instance head:
              mask_logits [B,T,N_q,H_f,W_f]   ← per-slot mask logits
              objectness  [B,T,N_q]           ← per-slot logits
              descriptor  [B,T,N_q,D]         ← per-slot L2-normed descriptor
```

### Key design points

- **Frontend / Backend split**: VGGT is the frozen geometry backbone (depth,
  world_points, camera). Semantic backbone is independent and frozen too.
  All learning happens in CausalBranch + DepthEncoder + DPT heads + optional
  conditioning encoders.

- **Causal branch with KV cache**: 12 pairs of (frame, global) blocks. Frame
  blocks do per-frame self-attention; global blocks do **causal cross-frame
  attention with KV cache** for incremental streaming inference. Initialized
  from VGGT's first 12 frame/global block pairs.

- **Semantic backbone ↔ prior + supervision**: the semantic backbone serves
  three roles, all swappable at the constructor level:
  1. **Prior** at CausalBranch input (fused with DINOv2 tokens)
  2. **Conditioning** at each DPT layer of the semantic head
  3. **Supervision GT** for cosine distillation in loss

- **`sem_dim` vs `sem_expand`**:
  - `sem_dim` = compact dim that the model actually outputs and SLAM voxel map
    stores (e.g. 64).
  - `sem_expand` = expanded dim via a learnable MLP (`expand_projector`).
    Aligned to the backbone's image-text shared dim so cosine loss + text
    query retrieval at inference both work in that space. Auto-set to
    `backbone.output_dim` when not specified.

- **Backward compatibility**: defaults reproduce legacy LSeg behaviour exactly
  (existing checkpoints load and run unchanged).

---

## Semantic backbones

All non-LSeg backbones expose three forward paths:
- `forward_prior(images, dino_tokens, dino_grid)` — ProxyCLIP-style:
  DINOv2 self-similarity drives the last attention layer.
- `forward_prior_native(images)` — backbone's standard dense forward
  (best for **TIPSv2**, which has explicit patch-text alignment).
- `forward_supervision(images, instance_mask, ...)` — TextRegion (mask-attn
  pool over backbone V) or mask_crops (per-instance crop forward + broadcast).

| Backbone (`--semantic_backbone`) | output_dim | patch | input_size | Text-aligned? | Notes |
|---|---|---|---|---|---|
| `lseg` | 512 | (DPT) | ~384 max-dim | ✅ (CLIP-B/32 distill) | Legacy default; full LSeg DPT |
| `pe-core-b16-224` | 1024 | 16 | 224 | ✅ | PE-Core, ViT-B/16 |
| `pe-core-l14-336` | 1024 | 14 | 336 | ✅ | PE-Core, ViT-L/14 — patch=14 aligns with DINOv2 |
| `pe-core-g14-448` | 1280 | 14 | 448 | ✅ | PE-Core, ViT-G/14 — strongest PE |
| `pe-core-s16-384` / `pe-core-t16-384` | 512 | 16 | 384 | ✅ | Lightweight PE |
| `clip-vit-b-32` | 512 | 32 | 224 | ✅ | OpenAI CLIP — standard |
| `clip-vit-b-16` | 512 | 16 | 224 | ✅ | OpenAI CLIP, ViT-B/16 |
| `clip-vit-l-14` | 768 | 14 | 224 | ✅ | OpenAI CLIP, ViT-L/14 — patch=14 |
| `clip-vit-l-14-336` | 768 | 14 | 336 | ✅ | OpenAI CLIP, ViT-L/14@336px |
| `tips-v2-b14` | 768 | 14 | 448 | ✅ + **patch-text** | TIPSv2-Base; 86M vision params |
| `tips-v2-l14` | 1024 | 14 | 448 | ✅ + **patch-text** | TIPSv2-Large; 303M |
| `tips-v2-so14` | 1152 | 14 | 448 | ✅ + **patch-text** | TIPSv2-So400m; 412M |
| `tips-v2-g14` | 1536 | 14 | 448 | ✅ + **patch-text** | TIPSv2-Giant; 1.1B |
| `siglip-so400m-14-384` | 1152 | 14 | 384 | ✅ (MAP head over patches) | SigLIP-v1 SO400M; via `open_clip` (webli). No CLS/register tokens. |
| `siglip2-so400m-14-378` | 1152 | 14 | 378 | ✅ (MAP head over patches) | SigLIP-v2 SO400M (Gemma tokenizer); via `open_clip`. **Recommended SigLIP variant.** |
| `siglip2-so400m-16-{256,384,512}` | 1152 | 16 | {256,384,512} | ✅ | SigLIP-v2 SO400M patch-16 family |
| `siglip2-{b16,l16}-{256,384}` | {768,1024} | 16 | {256,384} | ✅ | SigLIP-v2 smaller variants |
| `clip-dinoiser` | 512 | 16 | 224 | ✅ (laion2b CLIP-B/16) | MaskCLIP V + learned 3×3 conv that mimics DINO patch self-similarity → DINO-affinity-weighted refined feat. Self-contained (no DINOv2 at inference). Requires `pip install open-clip-torch`. |

### SigLIP / SigLIP-v2 specifics

`siglip-*` / `siglip2-*` are loaded via `open_clip` (webli weights from HF Hub).
Architectural quirks worth knowing:

- **No CLS / register tokens** (`num_prefix_tokens=0`) — the trunk emits only
  patch tokens.  All `_supervision_*` paths handle this correctly because
  ``n_special=0``.
- **No image-text projection**: ``embed_dim`` already equals the shared dim
  (1152 for SO400M).  Raw post-LN patches are directly in text-aligned space.
- **Global emb via MAP head**, not pooling: a single learnable latent query
  attends to all patches → Proj + MLP residual → 1152-dim.  So
  `encode_image_global` and `model.encode_image` produce a different vector
  than `patches.mean(dim=1)` — they're a learned attention-weighted pool.
- **Inception-style normalisation** `(0.5, 0.5, 0.5)` mean/std → maps pixels
  to `[-1, 1]`, NOT ImageNet.  The wrapper handles this via internal buffers.

**Minimal wrapper status** (`siglip_backbone.py`):
- ✅ `prior_method='native'`
- ✅ `sem_supervision='patch_mean'`
- ✅ `sem_supervision='mask_crops'`
- ❌ `prior_method='proxy_clip'` — raises `NotImplementedError`
- ❌ `sem_supervision='textregion'` — raises `NotImplementedError`

Add `_v_forward` (extract last-block V from the timm trunk) to enable the
proxy_clip / textregion paths.

### Why TIPSv2 is special

- **Explicit patch-text alignment**: TIPSv2 trains the patch tokens to be
  text-aligned (CLIP/PE only train CLS-text alignment). So TIPSv2's native
  patch tokens are the cleanest dense GT — no ProxyCLIP / CSA tricks needed.
- **DINOv2-style backbone with register tokens** (1 per variant): better
  spatial structure than CLIP/PE.
- **`prior_method='native'` is the natural choice for TIPSv2**.

### CLIP-DINOiser specifics

`clip-dinoiser` is structurally separate from the `VLMBackboneBase` family —
it doesn't fit the `_v_forward` / ProxyCLIP / TextRegion abstraction because
its dense feature pipeline (MaskCLIP V → predicted DINO affinity → weighted
pool) is its own thing.  As a result it only supports
`prior_method='clip_dinoiser'` (= `forward_prior` returns the refined feat)
and supervision falls back to LSeg-style pass-through (dense feat directly,
per-instance aggregation happens in the loss).  It does NOT participate in
the `proxy_clip` / `textregion` / `mask_crops` matrix the other backbones do.

Inference and training both need `pip install open-clip-torch` because the
released dinoiser checkpoint (`obj_proj` / `bkg_decoder`) was trained against
laion2b CLIP weights, not OpenAI CLIP.

### ProxyCLIP NaN guard

`proxy_attn.py:compute_proxy_attn_weights` clamps its threshold to
`min(sim.mean()*beta, sim.amax()-1e-4)` so that softmax never sees an
all-`-inf` row.  Without the clamp, very uniform DINOv2 features (close-up
wall / floor / ceiling shots → off-diagonal cosine mean > 0.83) push
`sim.mean() * beta` above `sim.amax()=1`, every entry gets masked, softmax
returns NaN, and the dense feature collapses to zero everywhere downstream
(all-black PCA, every voxel predicted as the class with text embedding
closest to the zero vector).  The clamp falls back to a near-identity
attention on degenerate inputs.

---

## Instance head: dense vs query

Two architectures share the same CausalBranch outputs but emit
fundamentally different instance representations.  Pick at training
time via `--instance_head_type {dense, query}`; the choice locks the
checkpoint — the SLAM-side `run.py` must be invoked with the same
flag.

### Dense (`--instance_head_type dense`, legacy default)

- DPT head, same shape as the semantic head.
- Per-pixel `instance_feat [B, T, ins_dim, H, W]` + confidence.
- Loss: `ContrastiveLoss` (pull same-ID pixels, push different-ID).
- SLAM persistence: per-voxel running average in `ins_store`
  (a `VoxelFeatureMap`); clustering at export time via PCA →
  optional HDBSCAN.
- No persistent instance identity across chunks.

### Query (`--instance_head_type query`)

- Mask2Former-style: a fixed pool of `N_queries` learnable query
  tokens cross-attends to per-pixel features through `n_layers`
  transformer layers.  Each query emits one (mask, objectness,
  descriptor) triplet per frame.
- Outputs: `mask_logits [B,T,N,H_f,W_f]`, `objectness [B,T,N]`,
  `descriptor [B,T,N,D]`.
- Loss (`opense3r/loss_query.py`): Hungarian-matched per-query
  segmentation loss (BCE + DICE for mask, focal for objectness) +
  contrastive on descriptors.  Bipartite matcher in
  [opense3r/query_matcher.py](query_matcher.py).
- SLAM persistence: discrete proposals lifted to 3D and merged into a
  persistent `InstanceTable` (see
  [slam_opense3r/README.md](../slam_opense3r/README.md) → "Query-head
  pipelines").
- Carries instance identity across chunks; per-instance score
  (`mean_objectness`) propagates into the AP evaluator's
  `cluster_score` field.

### Flag surface (query head)

Architecture flags must match the trained checkpoint at SLAM time.

| Flag | Default | Effect |
|---|---|---|
| `--instance_head_type` | `dense` | `dense` (legacy) or `query` (M2F) |
| `--query_arch` | `maskformer` (train) / `mask2former` (run.py) | Head arch flavour |
| `--query_num_queries` | 100 | # slot tokens |
| `--query_d_model` | 256 | Transformer hidden dim |
| `--query_c_mask` | 256 | Mask-decoder hidden dim |
| `--query_descriptor_dim` | 256 | Output descriptor dim (L2-normed for contrastive + cosine matching at SLAM time) |
| `--query_n_heads` | 8 | Attention heads |
| `--query_n_layers` | 6 (train) / 9 (run.py) | Decoder layers |
| `--query_num_feature_levels` | 3 | # of DPT feature levels piped to head |

Training-only:

| Flag | Default | Effect |
|---|---|---|
| `--no_query_deep_supervision` | False | Disable aux losses at intermediate decoder layers |
| `--query_aux_loss_layer_indices` | '' (= last only) | Comma-list of decoder layer indices to add aux loss on |
| `--query_use_box_head` | False | Add a per-query camera-canonical 3D bbox regression head (experimental — currently parked) |

### When to use which

- **Dense** — simpler pipeline, no Hungarian matching, no persistence
  bookkeeping.  Best for "just colour the voxels" demos.  Instance AP
  requires HDBSCAN post-clustering whose confidence is just cluster
  size (poor AP correlate).
- **Query** — better instance AP (per-instance scores ScanNet-protocol
  compliant), persistent identity, smaller export footprint (one
  record per instance, not one feature per voxel).  Recommended for
  ScanNet benchmarking.

---

## Flag surface (training-time)

### Backbone & paths

| Flag | Default | Choices | Effect |
|---|---|---|---|
| `--semantic_backbone` | `lseg` | see table above | Frozen backbone for prior + GT supervision |
| `--prior_method` | `lseg` | `lseg` / `proxy_clip` / `native` | How the prior dense feature is computed |
| `--sem_supervision` | `lseg` | `lseg` / `textregion` / `mask_crops` / `patch_mean` | What's used as cosine GT |
| `--sem_dim` | 64 | int | Compact dim model outputs (SLAM voxel map dim) |
| `--ins_dim` | 16 | int | Instance feature dim |
| `--sem_expand` | None | int | Expanded dim (None → auto = backbone.output_dim) |

### Sanity rules
- `semantic_backbone='lseg'` requires `prior_method='lseg'` and `sem_supervision='lseg'`.
- `semantic_backbone='clip-dinoiser'` is incompatible with the unified
  prior / supervision API — use only `prior_method='clip_dinoiser'` and run
  it as a standalone evaluation (see `test_semantic_2d.py`); not currently
  wired into `train_opense3r.py` because its dense feature isn't textregion /
  mask_crops compatible.
- All other backbones (`pe-core-*`, `clip-vit-*`, `tips-v2-*`) require
  `prior_method ∈ {proxy_clip, native}`.
- Method 1 (textregion), Method 2 (mask_crops), and Method 3 (patch_mean) are mutually exclusive.
- `patch_mean` requires a backbone that implements `forward_prior_native` AND
  has patch-text-aligned dense tokens (TIPSv2, SigLIP-v2 dense variants).
  Running it on CLIP-OpenAI / PE-Core technically works (their `forward_prior_native`
  exists) but the GT will be CLS-biased and worse than `textregion`.

### Prior / supervision tunables

| Flag | Default | Effect |
|---|---|---|
| `--proxy_beta` | 1.2 | ProxyCLIP attention re-centring factor |
| `--proxy_gamma` | 3.0 | ProxyCLIP attention scaling factor |
| `--multi_res_crops` | False | TextRegion: enable multi-resolution V (1 global + N tiles, each tile resized to backbone.input_size) |
| `--mr_grid_h` / `--mr_grid_w` | None (auto) | Override tile count (H / W).  None → auto: `max(H // S, 1)` / `max(W // S, 1)` (only fires when image ≥ backbone input).  Set explicitly to force tiling on smaller images (zoom-in for finer V grid) — e.g. `--mr_grid_h 2 --mr_grid_w 3` on 336×518 ⇒ V at 64×96 with backbone still seeing native S×S. |
| `--mask_crop_size` | 224 | Method 2: per-instance crop input size |
| `--mask_crop_margin` | 0.1 | Method 2: bbox margin (fraction of side length) |
| `--mask_crop_mode` | `bbox` | `bbox` (with context) or `masked` (zero outside instance) |
| `--mask_unlabeled` | False | Skip semantic loss on `instance_mask == 0` pixels |

### Conditioning streams (at CausalBranch input)

All default off — flagged combinations are valid and orthogonal.

| Flag | Default | Effect |
|---|---|---|
| `--depth_cond` | False | Add depth tokens at CausalBranch input (residual). Existing DepthCrossAttn path also stays. |
| `--pose_cond` | False | Prepend per-frame pose token (rot6d + tanh(t / scene_scale) + normalised K). Sets `patch_start_idx=1`. |
| `--plucker_cond` | False | Add per-patch Plücker ray tokens at CausalBranch input (residual). |
| `--plucker_num_freqs` | 4 | Fourier frequencies for Plücker encoder (0 → no Fourier features). |
| `--pos3d_cond` | False | Add per-patch 3D positional encoding (Fourier on NORMALIZED `world_pts / box_scale`) as residual at CausalBranch input.  3DPPE-style, scale-agnostic.  ([`opense3r/pos_encoders.py`](pos_encoders.py)) |
| `--no_depth_cross_attn` | False (flag off → cross-attn ON) | Gate the middle-layer `DepthCrossAttn` (every `depth_cross_every` blocks).  Use together with `--pos3d_cond` and no `--depth_cond` to fully disable the depth path so 3D info comes ONLY from 3D PE. |

### Descriptor-strengthening: query path & init (3D PE + streaming queries)

Three knobs, all backward-compatible (defaults preserve current behavior).  See [CLAUDE.md → Descriptor-strengthening designs](../CLAUDE.md#descriptor-strengthening-designs-3d-pe--instance-queries-in-causalbranch) for the recommended combinations table.

| Flag | Default | Effect |
|---|---|---|
| `--query_init_type` | `learnable` | Mask2Former query feat init source. `learnable` = `nn.Embedding` (current).  `3d_anchor` = learnable `[N_q, 3]` anchor in NORMALIZE space → Fourier PE → MLP.  `external` = caller (CausalBranch) provides the init queries — combine with `--num_instance_queries > 0`. |
| `--num_instance_queries` | 0 | **Idea 2**: when > 0, learnable instance queries flow through CausalBranch alongside patches.  Final queries are returned and (when `--query_init_type external`) fed to Mask2Former.  Suggested with `--query_n_layers 3` (shrunken decoder). |
| `--instance_query_block_indices` | `None` (auto-spread) | Comma-separated block indices where queries do self-attn + cross-attn to patches.  Default for `n_blocks=12` → `[2, 5, 8, 11]`. |
| `--instance_queries_init_type` | `learnable` | How the CausalBranch instance queries are INITIALIZED.  `learnable` = direct `nn.Parameter` random init.  `3d_anchor` = learnable `[N_q, 3]` anchor (NORMALIZE) → Fourier PE → MLP (re-computed every forward, anchors learn via gradient).  Orthogonal to `--query_init_type` — you can have CausalBranch init from 3D anchors AND Mask2Former receive via `external`. |

### Geometry backbone

| Flag | Default | Effect |
|---|---|---|
| `--backend` | `vggt` | `vggt` (default) or `da3` (Depth Anything 3) |
| `--vggt_ckpt` | `./checkpoints/VGGT.pt` | **VGGT weights — always required.** Both backends call `vggt.aggregator.encode_patch_tokens` to produce DINOv2 patch tokens, which form the CausalBranch's primary input token sequence (concat'd with the semantic prior, projected through `input_proj`).  Used at every block as Q/K/V of self-attn.  Without VGGT, the CausalBranch has no input.  See [Module map → DINOv2 is the input](#module-map). |
| `--da3_ckpt` | `./checkpoints/da3_base` | DA3 weights (only used when `--backend=da3`) |

#### DINOv2's role

The user often asks where DINOv2 sits in the model — short version: **it IS
the CausalBranch's main input sequence**, not a cross-attention modality.

```python
# opense3r/causal_branch.py:231-238 (simplified)
dino_flat  = dino_tokens.reshape(B*T, N_patch, 1024)         # VGGT.encode_patch_tokens
lseg_patch = F.adaptive_avg_pool2d(lseg_feat, ...)            # semantic prior (LSeg/TIPS/CLIP/PE)
x = self.input_proj(torch.cat([dino_flat, lseg_patch], -1))   # concat→MLP→causal_dim
# x then becomes Q + K + V of every block's self-attention
```

So `prior_method` (the semantic backbone's role) only controls **one slice
of the input channels**; the DINOv2 1024-dim slice is constant.  Depth /
pose / Plücker conditioning streams are either residual-added to `x` or
prepended as special tokens — they do NOT replace DINOv2.

### Loss weights

| Flag | Default | Effect |
|---|---|---|
| `--w_sem` | 1.0 | Semantic cosine loss weight |
| `--w_ins` | 1.0 | Instance contrastive loss weight |
| `--w_consistency` | 0.5 | Cross-view instance prototype consistency |
| `--inter_margin` | 0.2 | Hinge margin for instance push |

---

## Example commands

### Legacy LSeg baseline (current default; no behaviour change)
```bash
torchrun --nproc_per_node=1 train_opense3r.py \
    --backend vggt \
    --batch_size 1 --accum_iter 4 --epochs 30 \
    --output_dir ./outputs/opense3r_lseg
```

### TIPSv2-B/14 + native prior + TextRegion supervision  (recommended first try)
TIPSv2 has patch-text alignment built in, so `native` is the natural prior path.

```bash
torchrun --nproc_per_node=1 train_opense3r.py \
    --backend vggt \
    --semantic_backbone tips-v2-b14 \
    --prior_method native \
    --sem_supervision textregion \
    --mask_unlabeled \
    --batch_size 1 --accum_iter 4 --epochs 30 \
    --output_dir ./outputs/opense3r_tips_native_textregion
```

### TIPSv2-B/14 + ProxyCLIP + TextRegion (ablation vs native)
```bash
torchrun --nproc_per_node=1 train_opense3r.py \
    --backend vggt \
    --semantic_backbone tips-v2-b14 \
    --prior_method proxy_clip \
    --sem_supervision textregion \
    --mask_unlabeled \
    --output_dir ./outputs/opense3r_tips_proxy_textregion
```

### TIPSv2-B/14 + native prior + TextRegion + multi-res (zoom-in for finer GT)
For 336×518 training images (smaller than TIPS native 448), the auto formula
gives `crop_num = 1×1` (no tiling). Force tiling explicitly with `--mr_grid_h/w`
— each tile is upsampled to TIPS native 448×448 (no posemb interp), V grid
becomes 64 × 96 instead of 32 × 32. Compute cost ≈ 7× single-pass.

```bash
torchrun --nproc_per_node=1 train_opense3r.py \
    --backend vggt \
    --semantic_backbone tips-v2-b14 \
    --prior_method native \
    --sem_supervision textregion \
    --multi_res_crops \
    --mr_grid_h 2 --mr_grid_w 3 \
    --mask_unlabeled \
    --output_dir ./outputs/opense3r_tips_native_textregion_mr2x3
```

### PE-Core-B/16 + ProxyCLIP + TextRegion (strongest standard VLM baseline)
```bash
torchrun --nproc_per_node=1 train_opense3r.py \
    --backend vggt \
    --semantic_backbone pe-core-b16-224 \
    --prior_method proxy_clip \
    --sem_supervision textregion \
    --mask_unlabeled \
    --output_dir ./outputs/opense3r_pe_proxy_textregion
```

### Method 2 (per-instance crop forward) supervision ablation
```bash
torchrun --nproc_per_node=1 train_opense3r.py \
    --backend vggt \
    --semantic_backbone tips-v2-b14 \
    --prior_method native \
    --sem_supervision mask_crops \
    --mask_crop_mode bbox --mask_crop_margin 0.1 \
    --mask_unlabeled \
    --output_dir ./outputs/opense3r_tips_native_maskcrops
```

### SigLIP-v2 SO400M + native + patch_mean
SigLIP-v2 SO400M is the strongest patch-text-aligned VLM that doesn't need
TIPS-specific training. Cleanest non-LSeg dense teacher.

```bash
torchrun --nproc_per_node=1 train_opense3r.py \
    --backend vggt \
    --semantic_backbone siglip2-so400m-14-378 \
    --prior_method native \
    --sem_supervision patch_mean \
    --mask_unlabeled \
    --batch_size 1 --accum_iter 4 --epochs 30 \
    --output_dir ./outputs/opense3r_siglip2_so400m_patchmean
```

### Method 3 (patch_mean) — hard-mask mean of backbone-native dense patches
Lightest pipeline: reuses the same `forward_prior_native` output that the
prior already computes (one backbone forward per frame, no last-attn
re-pool, no per-instance crop forwards). Instance mask is downsampled to
the patch grid via **nearest-neighbour** (hard 0/1), patch features inside
each instance are averaged, then broadcast back to the full-res mask.
Best on backbones where patch tokens are text-aligned by training
(TIPSv2, SigLIP-v2 dense variants).

```bash
torchrun --nproc_per_node=1 train_opense3r.py \
    --backend vggt \
    --semantic_backbone tips-v2-b14 \
    --prior_method native \
    --sem_supervision patch_mean \
    --mask_unlabeled \
    --output_dir ./outputs/opense3r_tips_native_patchmean
```

### Query (Mask2Former-style) instance head with TIPSv2
```bash
torchrun --nproc_per_node=1 train_opense3r.py \
    --backend vggt \
    --semantic_backbone tips-v2-l14 \
    --prior_method native --sem_supervision textregion \
    --mask_unlabeled --depth_cond \
    --instance_head_type query \
    --query_arch maskformer --query_num_queries 100 \
    --query_d_model 256 --query_c_mask 256 \
    --query_descriptor_dim 256 \
    --query_n_heads 8 --query_n_layers 9 \
    --query_num_feature_levels 3 \
    --output_dir ./outputs/opense3r_tips_query_m2f
```

The trained ckpt's instance head signature is locked — the matching
SLAM `run.py` invocation must mirror every `--query_*` flag plus
`--instance_head_type query`.

### DA3 (metric) backend with PE-Core
```bash
torchrun --nproc_per_node=1 train_opense3r.py \
    --backend da3 --da3_ckpt /path/to/da3_giant_metric \
    --semantic_backbone pe-core-b16-224 \
    --prior_method proxy_clip --sem_supervision textregion \
    --mask_unlabeled \
    --output_dir ./outputs/opense3r_da3_pe_textregion
```

### Adding pose / Plücker conditioning
Orthogonal to backbone choice — combine freely:
```bash
torchrun --nproc_per_node=1 train_opense3r.py \
    --backend vggt \
    --semantic_backbone tips-v2-b14 --prior_method native --sem_supervision textregion \
    --mask_unlabeled \
    --pose_cond --depth_cond \
    --pretrained ./outputs/opense3r_tips_native_textregion/checkpoint-last.pth \
    --output_dir ./outputs/opense3r_tips_native_pose
```
`--pretrained` loads with `strict=False`; shape-mismatched layers (e.g.
`expand_projector` when `sem_expand` changes, or `input_proj` when backbone
changes) are auto-dropped from the load and re-initialised from scratch.

#### `--pose_cond` and streaming inference

Training does one forward over the entire sequence so the model's
`make_relative_pose(c2w)` always anchors to sequence frame 0.  Streaming
inference processes one chunk at a time and each new chunk's DA3/VGGT
output re-anchors `c2w[0] = identity` to the **chunk's** first frame, not
the SLAM global origin — which would silently make the new pose tokens
incompatible with the past pose tokens already in the KV-cache.

`forward()` / `forward_chunk()` therefore expose an `extr_rel_override`
argument: pass per-chunk c2w expressed in the SLAM global frame (frame 0
= identity) and the model skips `make_relative_pose`.  See
`slam_opense3r/README.md` → "Pose conditioning under streaming" for how
the SLAM pipeline computes and feeds this override.  This matters for
checkpoints trained with `--pose_cond`; checkpoints without pose_cond are
unaffected.

---

## Checkpoint downloads

Required:
- **VGGT**: `./checkpoints/VGGT.pt` (or wherever `--vggt_ckpt` points to)
- **LSeg**: `./checkpoints/lseg_minimal_e200.ckpt` (only needed for `lseg` backbone)

Optional (for non-LSeg backbones):
- **PE-Core / OpenAI CLIP**: auto-downloaded from HF / OpenAI on first use
  → `~/.cache/huggingface` and `~/.cache/clip`.
- **TIPSv2**: NumPy `.npz` files. Run:
  ```bash
  cd thirdparty/tips/pytorch/checkpoints
  bash download_checkpoints.sh
  ```
  or grab a single variant manually (e.g. B/14):
  ```bash
  cd thirdparty/tips/pytorch/checkpoints
  wget https://storage.googleapis.com/tips_data/v1_0/checkpoints/tokenizer.model
  wget https://storage.googleapis.com/tips_data/v2_0/checkpoints/pytorch/tips_v2_oss_b14_vision.npz
  wget https://storage.googleapis.com/tips_data/v2_0/checkpoints/pytorch/tips_v2_oss_b14_text.npz
  ```

---

## Inference

`infer_opense3r.py` and `query_openvocab.py` automatically read
`semantic_backbone` / `prior_method` / `sem_supervision` / `sem_expand` /
`proxy_*` from the saved `args` dict in the checkpoint, so trained models
"just work" with the same backbone they were trained with — no need to
re-specify on the inference command line.

---

## Module map

| File | Purpose |
|---|---|
| `model.py` | `OpenSe3r` (VGGT geometry backend) |
| `model_da3.py` | `OpenSe3rDA3` (DA3 geometry backend) — same trainable submodules |
| `causal_branch.py` | Causal transformer with KV cache + DepthCrossAttn |
| `depth_encoder.py` | Depth map → patch-level geometry tokens |
| `dpt_heads.py` / `dpt_heads_multiscale.py` | DPT upsampling head (single / multi-scale) with optional prior conditioning |
| `cond_encoders.py` | Pose / Plücker / depth conditioning encoders |
| `loss.py` | `OpenSe3rLoss`: dense semantic cosine + dense instance contrastive + cross-view consistency |
| `loss_query.py` | Query-head loss: Hungarian-matched BCE/DICE/focal + descriptor contrastive |
| `instance_head_query.py` / `instance_head_query_m2f.py` | Query (Mask2Former-style) instance heads |
| `query_matcher.py` | Bipartite matcher (Hungarian on mask+class cost) for query loss |
| `instance_post.py` | `process_chunk_to_proposals`, `lift_proposal_to_3d`, `_proposal_pair_score_dict` — SLAM-side proposal extraction + cross-chunk scoring (reused by `slam_opense3r/instance_table.py`) |
| `box_gt.py` | Cross-view 3D bbox GT generation (experimental, query-head box-head only) |
| `stage2.py` | Stage-2 refinement experiments (separate from main path) |
| `semantic_backbones/` | LSeg / PE-Core / OpenAI CLIP / TIPSv2 wrappers + ProxyCLIP / TextRegion / MaskCrops shared helpers |

### `semantic_backbones/` internals

| File | Purpose |
|---|---|
| `_base.py` | `VLMBackboneBase` — common forward_prior / forward_supervision / multi_res_v_forward |
| `lseg_backbone.py` | LSeg wrapper (legacy) |
| `pe_core_backbone.py` | PE-Core wrapper |
| `clip_backbone.py` | OpenAI CLIP wrapper (via vendored `custom_clip`) |
| `tips_backbone.py` | TIPSv2 wrapper (Google DeepMind, CVPR 2026) |
| `clip_dinoiser_backbone.py` | CLIP-DINOiser wrapper — MaskCLIP V + learned 3×3 conv for DINO-affinity refinement (uses `open_clip` + laion2b CLIP-B/16 weights, no mmseg dependency) |
| `proxy_attn.py` | DINOv2-driven last-attention (ProxyCLIP) with NaN guard |
| `text_region_attn.py` | Mask-weighted last-attention pooling (TextRegion) |
| `mask_crop_extractor.py` | Per-instance bbox/masked crop forward (Method 2) |
