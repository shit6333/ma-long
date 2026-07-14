# slam_opense3r

Online semantic + instance SLAM that wraps the trained OpenSe3r model
(`opense3r/model.py` or `opense3r/model_da3.py`) into an incremental
pipeline with **per-chunk streaming inference**, KV-cache, persistent
geometry alignment, and per-voxel feature aggregation.

Same conceptual structure as `slam/pipeline.py` (geometry-only SLAM) and
`slam_semantic/pipeline.py` (the older AMB3R-based semantic SLAM), but:

- Geometry forward is **separated** from semantic forward.  Geometry runs
  bidirectionally on `[keyframes + new frames]` (VGGT/DA3 produce
  consistent poses for the joint set); semantic forward only runs on new
  frames + grows the KV-cache.
- KV-cache is the explicit streaming state — `CausalBranch.global_blocks`
  cache K and V per layer, so each new frame attends to all past frames
  without re-running them through the network.
- The semantic store can be voxel-based (confidence-weighted average) or
  per-point (every projected point retained).

---

## Pipeline overview

The pipeline matrix has three axes:

1. **Geometry source** — `pred` (model predicts depth + poses) vs `gt` (use
   ScanNet GT depth + poses; model still runs for semantic only).
2. **Instance head type** — `dense` (legacy: per-pixel `instance_feat`
   pushed into a voxel feature map + optional HDBSCAN at export) vs
   `query` (Mask2Former-style: head emits discrete proposals with mask
   + descriptor + objectness, persisted in an `InstanceTable`).
3. **KV cache memory mode** — `default` (flat KV cache, optionally
   anchor + sliding-window pruned) vs `voxel_kv` (3-tier anchor / window
   / spatial voxel pool).

Eight pipeline classes total — `run.py` picks one based on
`--use_gt` / `--instance_head_type` / `--memory_mode`:

| Geo | Head  | KV memory | Class |
|---|---|---|---|
| pred | dense | flat       | `OpenSe3rSLAM` (`pipeline.py`) |
| pred | dense | voxel_kv   | `OpenSe3rSLAM_VoxelKV` (`pipeline_voxel_kv.py`) |
| pred | query | flat       | `OpenSe3rQuerySLAM` (`pipeline_query.py`) |
| pred | query | voxel_kv   | `OpenSe3rQuerySLAM_VoxelKV` (`pipeline_query_voxel_kv.py`) |
| gt   | dense | flat       | `OpenSe3rSLAM_GT` (`pipeline_gt.py`) |
| gt   | dense | voxel_kv   | `OpenSe3rSLAM_VoxelKV_GT` (`pipeline_voxel_kv.py`) |
| gt   | query | flat       | `OpenSe3rQuerySLAM_GT` (`pipeline_query_gt.py`) |
| gt   | query | voxel_kv   | `OpenSe3rQuerySLAM_VoxelKV_GT` (`pipeline_query_gt.py`) |

All variants share `OpenSe3rMemory` for geometry + semantic; query
variants additionally hold an `InstanceTable` via `OpenSe3rQueryMemory`
(see "Query-head pipelines" below).

### Pred mode (`OpenSe3rSLAM`)

```
for each chunk:
  1. Geometry forward on [keyframes + new frames]
       VGGT/DA3 bidirectional, returns world_points + c2w in a chunk-local frame
       (frame 0 of the joint batch = identity).
  2. memory.update_geo
       coordinate_alignment aligns chunk-local → SLAM global frame using
       the keyframe overlap (Procrustes / similarity transform), then
       confidence-weighted merges pts/poses into self.pts / self.poses,
       runs keyframe discovery.  Caches per-new-frame globally-aligned pts
       on `_pending_pts_new` for the upcoming sem step.
  3. Build extr_rel_override = memory.poses[start:end+1]
       (memory.poses[0] = identity, so these are c2w anchored to SLAM
       sequence frame 0 — exactly the convention training uses.)
  4. Semantic forward on [new frames only], with kv_cache and
     extr_rel_override → semantic_feat + instance_feat + updated kv_cache.
  5. memory.update_sem
       Push the new frames' semantic / instance features into the feature
       stores, using the cached globally-aligned pts from step 2.
```

### GT mode (`OpenSe3rSLAM_GT`)

```
for each chunk:
  1. Build extr_rel_override = inv(poses_gt[0]) @ poses_gt[chunk]
       (single anchor across the whole run, computed once.)
  2. Semantic forward on chunk frames, kv_cache + override.
  3. memory.update_gt
       Skip alignment entirely — GT pts and poses are written directly
       into memory.pts / memory.poses; semantic features are pushed at
       the GT point locations.
```

GT mode doesn't run `coordinate_alignment`, doesn't manage keyframes,
and doesn't call the geometry head — it exists as a "what-if SLAM had
perfect geometry" baseline for semantic-only evaluation.

---

## Query-head pipelines

Activated by `--instance_head_type query`.  The model's instance head is
Mask2Former-style — each of `N_queries` slots outputs a mask logit map,
an objectness logit, and a descriptor.  Slots are filtered → NMS →
per-pixel argmax → lifted to 3D as discrete proposals, then merged
across chunks into a persistent **InstanceTable**.

### Mapping step (per chunk)

```
1. memory.update_geo   (or update_gt)
2. Semantic + query head forward on new frames → sem_res with:
      mask_logits  [1, T_new, N, H_f, W_f]
      objectness   [1, T_new, N]
      descriptor   [1, T_new, N, D]
      semantic_feat / semantic_feat_expanded (per-pixel, for sem store)
3. memory.update_instances(sem_res, start_idx, end_idx)
      → process_chunk_to_proposals (per-frame extraction):
          objectness filter (obj_threshold)
          mask NMS (nms_iou)
          per-pixel argmax with score_threshold
          lift_proposal_to_3d:
              - mask pixels → 3D points (from _pending_pts_new)
              - per-axis Tukey-fence IQR filter (outlier_iqr_mult) for
                depth-bleed / mask-edge outliers
              - voxel keys (floor(pts / instance_voxel_size))
              - center = median, bbox = AABB of inliers
              - sem_query = mean of dense sem_feat inside mask (L2-norm)
      → InstanceTable.match_chunk:
          KNN gating (point-to-AABB ≤ knn_radius)
          Sequential Hungarian on score = w_desc·sim_desc
                                          + w_sem ·sim_sem
                                          + w_vox ·sim_vox
                                          + w_geo ·sim_geo
          (exp-decay on point-to-AABB distance, sigma_dist)
          merge if best ≥ match_threshold, else spawn new inst_id
4. memory.update_sem   (pushes dense sem feat into sem voxel store as usual)
```

### `InstanceRecord` (one per inst_id)

Each record runs a streaming average across all merged proposals:

| Field | Source | Update |
|---|---|---|
| `descriptor`  | head | L2-norm running avg |
| `sem_query`   | per-mask mean of dense sem feat | L2-norm running avg |
| `center`      | proposal median 3D | running avg |
| `bbox_min/max`| AABB of all observed voxels (= union) | min/max accumulate |
| `voxels`      | set of `(vx,vy,vz)` | union |
| `count`       | # of proposals merged | increment |
| `mean_objectness` | per-frame sigmoid(objectness) | running avg |
| `first_seen`/`last_seen` | frame index | update |

### `voxel_to_inst_map` (winner-take-all)

When two instances claim the same voxel, the one with higher
`mean_objectness` (or larger count as tiebreak) wins.  This map is what
the export PLYs / NPZ / eval all read from.

### Noise filter (export-time)

A multi-rule filter (`filter_noise_instances` in
[amb3r/vis_query_instance_slam.py](../amb3r/vis_query_instance_slam.py))
runs after SLAM finishes.  Filtered instances stay in JSON (with
`filtered: true`) and render grey in the PLYs but are excluded from the
`dropped_ids` set passed to eval:

| Rule | Knob | Default | Effect |
|---|---|---|---|
| min observation count | `--inst_filter_min_count` | 1 | Drop `count <= N` mid-sequence (spare last `end_grace` frames) |
| oversized bbox | `--inst_filter_max_size_m3` | 5.0 | Drop if `bbox_volume > N m³` AND `count <= max_count` |
| tiny voxel set | `--inst_filter_tiny_voxels` | 0 (off) | Drop if `num_voxels <= N` AND `count <= max_count` |
| low avg objectness | `--inst_filter_min_avg_objectness` | 0 (off) | Drop ghosts |

### IQR outlier filter (proposal-lift time)

`--inst_iqr_mult m` (default 1.5) — Tukey fence on per-axis 3D point
distribution when lifting a 2D mask to 3D.  Drops `m·IQR`-outside
outliers (mask-edge depth bleed from DA3 / VGGT).  Smaller `m` →
tighter bbox & voxel set.  See
[opense3r/instance_post.py](../opense3r/instance_post.py)
`lift_proposal_to_3d` for the math.

### Why this beats dense instance features

- **Persistent identity** across chunks — instance IDs don't drift; new
  proposals are matched to existing IDs via Hungarian.
- **Bounded memory** — one record per instance vs one feature per voxel.
- **Discrete instance scores** (`mean_objectness`) propagate to the AP
  evaluator (`instance_labels.npz:cluster_score`) — ScanNet-style AP
  expects per-instance scores, not size proxies.
- **Reuses the same `_pending_pts_new` / `_pending_conf_new` cache** that
  `update_sem` uses, so the proposal voxel keys live in the same SLAM
  global frame as `memory.pts`.

---

## Voxel KV memory mode

`--memory_mode voxel_kv` swaps the flat KV cache for the 3-tier
manager in [voxel_kv_memory.py](voxel_kv_memory.py):

- **Anchor tier** — sticky first-N keyframes (always retrieved)
- **Window tier** — last-W chunks (sliding)
- **Spatial voxel pool** — every other token bucketed by 3D voxel of
  its corresponding world point; retrieved by overlap with the new
  chunk's voxel footprint

This gives long-horizon spatial context without unbounded growth.
Config knobs (`VKV_*` env vars in the bash):

| Knob | Default | What |
|---|---|---|
| `vkv_voxel_size` | 0.1 m | Spatial bucket size |
| `vkv_anchor_cap` | 10 | # of anchor keyframes |
| `vkv_anchor_mode` | sticky | `sticky` (first-N) or `rotate` |
| `vkv_window_chunks` | 6 | Sliding window length (chunks) |
| `vkv_pivot_cap` | 4 | Max # of pivot tokens per voxel |
| `vkv_retrieval_radius` | 1 | Voxel radius around new chunk's footprint |
| `vkv_pivot_signal` | V (forced to V for query mode — `instance_feat` is None) | What feature represents each pivot |
| `vkv_merge_threshold` | 0.85 | Cosine threshold for absorbing dup pivots |

In query mode `instance_feat` is None, so `pivot_signal` is
auto-flipped to `V` (last-layer Value) in
`OpenSe3rQuerySLAM_VoxelKV.__init__`.

---

## Pose conditioning under streaming

This was the most subtle bug in the streaming pipeline.  If a model is
trained with `--pose_cond`, naively running streaming inference produces
inconsistent pose tokens across chunks:

1. Training does one forward over the entire sequence.  Inside
   `model.forward`, `make_relative_pose(c2w)` makes everything
   chunk[0]-relative.  Chunk[0] = sequence[0] = global origin → all pose
   tokens are anchored to the global frame.

2. Streaming inference: each new chunk's `_da3_forward` /
   `_vggt_forward` re-anchors `c2w[0] = identity` to the **current**
   chunk's first frame.  `make_relative_pose` again anchors to chunk[0],
   which is some random pose mid-trajectory.  The new chunk's pose tokens
   are in a different coordinate system than the past pose tokens stored
   in the KV-cache.  Global-block cross-attention sees inconsistent pose
   encodings → spatial reasoning across chunks breaks.

**Fix** (already applied to `opense3r/model.py` + `opense3r/model_da3.py`):
`forward()` and `forward_chunk()` accept `extr_rel_override`.  When
provided, the model uses it directly for `pose_to_features` /
`compute_plucker_rays` and skips `make_relative_pose`.

`pipeline.py:_mapping_step` and `pipeline_gt.py:run` compute the override
using either the pose-aligned `memory.poses` (pred mode) or the GT
sequence anchored to its first frame.  Models trained without
`--pose_cond` are unaffected: the override is computed but never used.

`scene_scale` (used to normalise pose translations) is still computed
per-chunk from the chunk's depth median — same as training — so it's
not part of this bug.

---

## Memory

`OpenSe3rMemory` (`memory.py`) holds:

| Buffer | Shape | What |
|---|---|---|
| `self.pts` | `[N_frames, H, W, 3]` | Per-pixel world-frame xyz (confidence-merged across visits) |
| `self.conf` | `[N_frames, H, W]` | Sigmoid-style depth conf |
| `self.poses` | `[N_frames, 4, 4]` | c2w in SLAM global frame |
| `self.iter` | `[N_frames]` | Number of times each frame was visited (re-included via keyframes) |
| `self.kf_idx`, `self.cur_kf_idx` | int tensor | All keyframes ever seen / active keyframes fed to the next geometry step |
| `self.sem_store` | voxel map or per-point store | Semantic features (compact or expanded dim) |
| `self.ins_store` | voxel map or per-point store | Instance features |

### `update_geo` / `update_sem` split

`update()` was split (commit-level change, kept as thin wrapper for
back-compat) so the pipeline can read globally-aligned poses **between**
the geo and sem steps and feed them as `extr_rel_override`:

```python
memory.update_geo(geo_res, start_idx, end_idx)     # alignment + keyframe mgmt
# ... now memory.poses[start_idx:end_idx+1] are global
extr_rel = memory.poses[start_idx:end_idx+1].unsqueeze(0).float()
sem_res, kv_cache = pipeline._sem_forward(new_imgs, kv_cache,
                                           extr_rel_override=extr_rel)
memory.update_sem(sem_res, sem_feat_key)           # push features at aligned pts
```

`update_geo` caches `_pending_pts_new` / `_pending_conf_new` on the
memory instance; `update_sem` consumes and clears them.

### Feature storage

Two options via `--feat_storage`:

- **`voxel`** (default) — `slam_semantic.semantic_voxel_map.VoxelFeatureMap`.
  Hash table keyed by quantised `(x, y, z)`; new features are merged with
  existing ones via confidence-weighted average.  Compact (one feat per
  voxel, voxel size set by `semantic_voxel_size` in `config.yaml`,
  default 5 cm).
- **`perpoint`** — `PerPointFeatureStore` (`memory.py:PerPointFeatureStore`).
  Every projected 3D point is appended to a growing tensor (no merging,
  no quantisation).  Larger but lossless.

### Compact vs expanded semantic feature

`--sem_feat_mode {expanded, compact}` selects which of the model's two
semantic outputs goes into the store:

| Mode | Dim | Space | Use case |
|---|---|---|---|
| `compact` | `sem_dim` (model output, e.g. 64 / 512 / 768) | Student model's compact space | Smaller voxel memory, no text query unless `sem_dim == text_emb_dim` |
| `expanded` | `sem_expand` (typically `backbone.output_dim`) | Same space as the backbone's text encoder | Direct cosine vs text queries — required for text-aligned PLY export |

For ScanNet eval we always use `expanded`.

---

## KV-cache pruning

`CausalBranch.global_blocks` cache grows by `N_patch × T_chunk` tokens
per chunk.  For long sequences this is unbounded.

`--kv_window_size W` enables anchor-plus-sliding-window pruning:

```
Keep:  first N_patch tokens (anchor — first frame of the sequence)
     + last W × N_patch tokens (sliding window of W most recent frames)
Drop:  everything in between
```

`--kv_keyframe_stride S` additionally retains a sparse pool of past
frames at uniform stride.  Stride is in **frames**, measured from
absolute index 0:

```
Keep:  anchor (frame 0)
     + keyframes {S, 2S, 3S, ...} (only in the region BEFORE the window)
     + sliding window (if --kv_window_size set)
Drop:  everything else
```

The two flags compose freely:
- `--kv_window_size W` alone → legacy anchor + window
- `--kv_keyframe_stride S` alone → anchor + sparse-uniform keyframes,
  no recency tail
- both set → anchor + sparse mid-region keyframes + sliding tail
  (recommended for long sequences)
- neither → no pruning (may OOM)

E.g. with `map_every=4`, `--kv_keyframe_stride 12` keeps one frame
per ~3 chunks.  Frame 0 is always the anchor; the first keyframe is
at `S`.  Window region is never re-sampled — it's always kept verbatim.

`_prune_kv_cache` in `pipeline.py` does this in-place after each
mapping step.  Pruning happens **after** the current chunk has been
processed, so the chunk just past still attended to the full cache.

Trade-off: anchor preserves global reference, window keeps recent context,
keyframes fill the mid-region gap → fewer dropped tokens than pure
window-only at the cost of slightly larger KV.  Both flags are unused
by `--memory_mode voxel_kv` (which manages its own 3-tier cache).

---

## Run flags

`run.py` mirrors the trained model's architecture flags — they MUST match
the checkpoint, otherwise weights silently load with mismatched shapes
(`strict=False`).

### Common

| Flag | Default | Effect |
|---|---|---|
| `--seed` | 0 | Seeds Python / NumPy / Torch + sets `cudnn.deterministic=True, benchmark=False`. Roughly bit-exact reruns at 5-30% perf cost. |
| `--backend` | `vggt` | Geometry backbone (must match ckpt) |
| `--semantic_backbone` | `lseg` | Frozen 2D backbone (must match ckpt) |
| `--prior_method` | `lseg` | `lseg` / `native` / `proxy_clip` (must match ckpt) |
| `--sem_supervision` | `lseg` | `lseg` / `textregion` / `mask_crops` / `patch_mean` (used at model build, not at inference) |
| `--depth_cond` / `--pose_cond` | False | Must match ckpt |
| `--plucker_cond` / `--pos3d_cond` | False | Must match ckpt (3D PE residual, [`opense3r/pos_encoders.py`](../opense3r/pos_encoders.py)) |
| `--no_depth_cross_attn` | False | Must match ckpt |
| `--query_init_type` | `learnable` | Must match ckpt — `learnable` / `3d_anchor` / `external` |
| `--num_instance_queries` | 0 | Must match ckpt — CausalBranch streaming queries (Idea 2) |
| `--instance_query_block_indices` / `--instance_queries_init_type` | None / `learnable` | Must match ckpt |
| `--sem_dim` / `--ins_dim` | 64 / 16 | Must match ckpt |
| `--sem_feat_mode` | `expanded` | `expanded` for text query, `compact` for storage-only |
| `--feat_storage` | `voxel` | `voxel` (5 cm by default) or `perpoint` |
| `--use_gt` | False | Use GT geometry (calls a `*_GT` pipeline) |
| `--demo_type` | `scannet` | `scannet` (RGB + pred poses), `scannet_gt` (RGB + GT depth/pose), `in_the_wild` (RGB folder only) |
| `--align_to_gt_first_frame` | False | Pred-only — rotate/translate the SLAM output into the GT world frame using GT first-frame c2w (for ScanNet eval) |
| `--kv_window_size` | None | Prune flat KV cache to anchor + sliding window |
| `--kv_keyframe_stride` | None | Additionally retain frames {S, 2S, 3S, ...} (in frames) of the past KV. Composes with `--kv_window_size`. |
| `--memory_mode` | `default` | `default` (flat KV cache) or `voxel_kv` (3-tier manager) |
| `--attn_mode` | `causal` | `causal` (matches training) or `chunk_bidir` (bidirectional within chunk, causal across chunks via KV cache) |
| `--conf_threshold` | None | Geometry conf cutoff for export. None → derive from `--conf_threshold_percentile` (default drop bottom 10%) |
| `--max_points` | 6M | Cap on exported geometry / per-point PLY size |
| `--save_npz` | False | Also dump raw voxel-centre features as `.npz` (used by `eval_*_npz_scannet.py`) |

### Query head (`--instance_head_type query`)

Must match the trained model's head type.  Architecture flags must match
the checkpoint exactly.

| Flag | Default | Effect |
|---|---|---|
| `--instance_head_type` | `dense` | `dense` (legacy per-pixel) or `query` (Mask2Former) |
| `--query_arch` | `mask2former` | Head arch (must match ckpt) |
| `--query_num_queries` | 100 | # slots |
| `--query_d_model` / `--query_c_mask` / `--query_descriptor_dim` | 256 | Internal dims |
| `--query_n_heads` / `--query_n_layers` | 8 / 9 | Head transformer geometry |
| `--query_num_feature_levels` | 3 | # of DPT feature levels fed to the head |

### InstanceTable knobs (query mode only)

These don't have to match the ckpt — they're SLAM-side decisions for
cross-chunk merging.  See "Query-head pipelines" above for what each does.

| Flag | Default | Effect |
|---|---|---|
| `--inst_voxel_size` | 0.05 | InstanceTable voxel grid (independent from sem store) |
| `--inst_knn_radius` | 2.0 m | Candidate gating radius (point-to-AABB) |
| `--inst_match_threshold` | 0.35 | Hungarian score floor to merge; below → spawn new id |
| `--inst_w_desc` / `--inst_w_sem` / `--inst_w_vox` / `--inst_w_geo` | 0.35/0.35/0.15/0.15 | Cue weights in match score |
| `--inst_sigma_dist` | 0.5 m | Decay scale for geo cue: `sim_geo = exp(-dist / sigma)` |
| `--inst_obj_threshold` | 0.5 | Per-frame objectness floor for accepting a query slot |
| `--inst_nms_iou` | 0.5 | Mask NMS IoU threshold |
| `--inst_score_threshold` | 0.25 | Per-pixel argmax score floor |
| `--inst_iqr_mult` | 1.5 | Tukey-fence multiplier in `lift_proposal_to_3d` (0 = off) |
| `--inst_filter_min_count` | 0 | Export: drop instances seen ≤ N times |
| `--inst_filter_end_grace` | 0 | Export: spare last-N-frame observations from min_count rule |
| `--inst_filter_max_size_m3` | 0 (off) | Export: drop bbox > N m³ AND `count <= max_count` |
| `--inst_filter_oversized_max_count` | 3 | "small count" threshold for the oversized rule |
| `--inst_filter_tiny_voxels` | 0 (off) | Export: drop instances with `num_voxels <= N` AND `count <= max_count` |
| `--inst_filter_tiny_max_count` | 2 | "small count" threshold for the tiny rule |
| `--inst_filter_min_avg_objectness` | 0 (off) | Export: drop instances with `mean_objectness < threshold` |

---

## Output schema

For scene `<demo_name>` in `--results_path`:

### Always produced

| File | What |
|---|---|
| `<demo>_geo.ply` | Confidence-filtered geometry point cloud |
| `<demo>_semantic_voxels.ply` | Per-voxel argmax label colour (text-matched) |
| `<demo>_semantic_per_point.ply` | Same labels broadcast to all geometry points |
| `<demo>_semantic_legend.png` | Class → colour key |
| `<demo>_semantic_feats.npz` (with `--save_npz`) | `{'centers': [N,3], 'features': [N, D]}` — voxel features in the backbone's text space |
| `<demo>_results.npz` | Full `pts`, `conf`, `poses`, etc. for replay |

### Dense instance head only

| File | What |
|---|---|
| `<demo>_instance_voxels.ply` | Per-voxel instance feature PCA→RGB |
| `<demo>_instance_per_point.ply` | Per-point instance feature PCA→RGB |
| `<demo>_instance_feats.npz` (with `--save_npz`) | `{'centers', 'features'}` for instance feature voxels |

### Query instance head only

| File | What |
|---|---|
| `<demo>_query_instance_voxels.ply`     | Per-voxel coloured by `inst_id` (descriptor-PCA palette by default) |
| `<demo>_query_instance_per_point.ply`  | Per-point version. **Inverts `transformation` before voxel lookup** so frame-of-reference matches the table — required when `--align_to_gt_first_frame` is on (memory.pts is post-T_align, voxel keys are pre-T_align). |
| `<demo>_query_instance_boxes.ply`      | Per-instance AABB wireframe |
| `<demo>_query_instance_table.json`     | Per-instance metadata (center, bbox, count, mean_objectness, member_frames, …; `filtered: true` for noise-filtered) |
| `<demo>_instance_labels.npz` (with `--save_npz`) | `{centers [V,3], cluster_id [V] int32, score_ids [K], cluster_score [K]}` — voxel-centre cluster map + per-instance score (mean_objectness, ScanNet-style confidence). Eats `eval_instance_npz_scannet.py` directly. |
| `<demo>_instance_sem_queries.npz` (with `--save_npz`) | `{inst_ids [K], sem_queries [K, D], voxel_size}` — per-instance aggregated sem feature for offline open-vocab analysis (used by `query_instance_semantic.py`) |

`semantic_voxels.ply` and `semantic_per_point.ply` already have ScanNet
class IDs in the `label` field (PLY pre-labeling at SLAM export time, uses
the single template `"a photo of a {}"`).

---

## Text encoder dispatch

`run.py` picks the text encoder based on `--semantic_backbone`:

| Backbone | Text encoder |
|---|---|
| `lseg` | LSeg's bundled CLIP-B/32 (via `model.lseg.clip_pretrained.encode_text`) |
| anything else (`tips-v2-*`, `clip-vit-*`, `pe-core-*`) | The backbone's own `encode_text` (TIPSv2 has its SigLIP-style text head, OpenAI CLIP / PE-Core have their own) |

The text encoder MUST match the space the dense semantic features were
trained to land in.  If you trained against TIPSv2's dense feat, you must
do retrieval with TIPSv2's text head, etc.  `run.py` handles this
automatically based on the `--semantic_backbone` flag.

For voxel ↔ text matching at export time we use a 10-template indoor
ensemble (see `amb3r/tools/semantic_vis_utils.py:DEFAULT_TEXT_TEMPLATES`,
which currently has 9 templates commented out → effectively single
`"a photo of a {}"` to match the legacy LSeg eval).

---

## Evaluation

Three offline eval scripts read the `--save_npz` outputs.  Bash wrappers
(`run_semantic_opense3r_slam{,_query}.bash`) auto-invoke them at the
end of a SLAM run.

### Semantic mIoU / mAcc (`eval_semantic_npz_scannet.py`)

```bash
python eval_semantic_npz_scannet.py \
    --semantic_npz <OUT_DIR>/<scene>_semantic_feats.npz \
    --gt_ply       <GT_LABEL_PLY> \
    --benchmark    {20, 200} \
    --backbone     <SEMANTIC_BACKBONE> \
    --save_csv     <OUT_DIR>/eval_semantic_scannet{20,200}.csv
```

Cosine-matches NPZ features against the backbone's text encoder, then
kNN majority vote (k=5) maps voxel labels onto GT vertices.

### Instance AP (`eval_instance_npz_scannet.py`) — query head only

Class-agnostic AP / AP50 / AP25 against ScanNet's per-segment instance
GT (aggregation.json + segs.json).

```bash
python eval_instance_npz_scannet.py \
    --hdbscan_labels_npz <OUT_DIR>/<scene>_instance_labels.npz \
    --gt_ply             <GT_LABEL_PLY> \
    --aggregation_json   <SCENE>.aggregation.json \
    --segs_json          <SCENE>_vh_clean_2.0.010000.segs.json \
    --benchmark          {20, 200} \
    --filter_pred_stuff  --semantic_npz <OUT_DIR>/<scene>_semantic_feats.npz \
    --backbone           <SEMANTIC_BACKBONE> \
    --save_csv           <OUT_DIR>/eval_instance_scannet{20,200}.csv
```

Key behaviours:

- **Class-agnostic** main metric — matches any GT instance regardless of
  class; only the STUFF filter cares about classes.
- **STUFF filter on GT**: drops GT instances whose dominant class is in
  `{wall, floor, ceiling}` (`STUFF_CLASS_NAMES` in the script).  ScanNet20
  drops only `{wall, floor}` since ScanNet20 has no `ceiling` class.
- **Confidence source**: prefers `cluster_score` from the NPZ
  (= `mean_objectness` from the head — ScanNet-protocol-compliant); falls
  back to `cluster_size / max_size` for legacy NPZs without scores
  (prints `confidence source: ...` so you can tell).
- **`--filter_pred_stuff`** (optional): also drops pred clusters whose
  predicted class (sem feat ⋅ text argmax) is in
  `{wall, floor, ceiling}`.  Closes the gap with closed-vocab evaluators
  (e.g. ESAM) where the head architecture itself can't emit stuff masks.
- **AP formula**: greedy match per IoU threshold, Pascal-VOC envelope
  integration, AP averaged over `[0.5..0.95 step 0.05]`; AP50, AP25 at
  fixed thresholds.  Mathematically identical to ESAM via
  `mmdet3d.scannet_eval`.

### Instance-broadcast semantic (`query_instance_semantic.py`)

Alternative to the dense semantic path that enforces "one class per
instance".  Reads the per-instance aggregated `sem_query` (from the
`_instance_sem_queries.npz` dump), broadcasts it to every voxel/point
owned by that instance, then text-aligns.

```bash
python query_instance_semantic.py \
    --instance_labels_npz       <OUT_DIR>/<scene>_instance_labels.npz \
    --instance_sem_queries_npz  <OUT_DIR>/<scene>_instance_sem_queries.npz \
    --geo_ply                   <OUT_DIR>/<scene>_geo.ply \
    --benchmark   {20, 200} \
    --backbone    <SEMANTIC_BACKBONE> \
    --output_dir  <OUT_DIR>
```

Outputs:
- `<scene>_instance_semantic_voxels.ply` / `_per_point.ply` — voxel/per-point label PLY
- `<scene>_instance_semantic_feats.npz` — `{centers, features}` in the same schema as
  `_semantic_feats.npz`, so you can immediately pipe it to
  `eval_semantic_npz_scannet.py` to get an mIoU number directly
  comparable with the dense path.

When this "one label per instance" constraint matches reality (most
indoor scenes), it usually beats the dense path by a few mIoU points.

### mIoU vs mAcc convention

- `mIoU` includes `class_iou=0` when GT=0 but pred>0 (penalises
  hallucination).
- `mAcc` excludes classes with GT=0 (`nan`, not included in the mean).
  This is the standard convention used by all major segmentation
  benchmarks including ScanNet's own reference eval at
  `scannet_200/reference_scannet200_eval/`.

The older `eval_semantic_slam.py` (still in the repo as a reference) has
a non-standard quirk where mAcc=0 was counted for GT=0 classes — this
drags scores down and isn't comparable across scenes with different
class coverage.  Use the new eval for all numbers going forward.

---

## File map

### Core

| File | Purpose |
|---|---|
| `run.py` | CLI + model load + dataset routing + export. Seeds determinism at startup (`_apply_seed`). |
| `config.yaml` | Chunk size, keyframe thresholds, voxel size, conf threshold |
| `memory.py` | `OpenSe3rMemory` (`initialize`, `update_geo`, `update_sem`, `update_gt`), `PerPointFeatureStore` |
| `mem_stats.py` | GPU peak-mem tracking helper used by all pipelines |

### Pipelines

| File | Class(es) |
|---|---|
| `pipeline.py` | `OpenSe3rSLAM` — pred / dense head / flat KV |
| `pipeline_gt.py` | `OpenSe3rSLAM_GT` — gt / dense head / flat KV |
| `pipeline_voxel_kv.py` | `OpenSe3rSLAM_VoxelKV`, `OpenSe3rSLAM_VoxelKV_GT` — dense head, 3-tier KV |
| `pipeline_query.py` | `OpenSe3rQuerySLAM` — pred / query head / flat KV |
| `pipeline_query_voxel_kv.py` | `OpenSe3rQuerySLAM_VoxelKV` — pred / query head / 3-tier KV |
| `pipeline_query_gt.py` | `OpenSe3rQuerySLAM_GT`, `OpenSe3rQuerySLAM_VoxelKV_GT` — gt / query head |

### Query-head infrastructure

| File | Purpose |
|---|---|
| `memory_query.py` | `OpenSe3rQueryMemory` — extends `OpenSe3rMemory` with `instance_table` and `update_instances()` |
| `instance_table.py` | `InstanceTable` + `InstanceRecord`: KNN gating, Sequential Hungarian matching, running averages, `voxel_to_inst_map` |

### Voxel KV infrastructure

| File | Purpose |
|---|---|
| `voxel_kv_memory.py` | `ThreeTierKVManager` + `VoxelKVConfig` — anchor / window / spatial-voxel KV cache |

### External dependencies (within OpenSe3r repo)

| Path | Used for |
|---|---|
| `opense3r/model.py` / `model_da3.py` | The trained model (`forward_chunk`, `init_kv_cache`) |
| `opense3r/instance_post.py` | `process_chunk_to_proposals`, `lift_proposal_to_3d` (query mode) |
| `opense3r/instance_head_query*.py` | Mask2Former-style query head (model side) |
| `amb3r/tools/pts_align.py` | `coordinate_alignment` for chunk-local → SLAM global pose transform |
| `amb3r/tools/semantic_vis_utils.py` | `build_text_embeddings*`, `export_semantic_*_ply`, palette / legend helpers |
| `amb3r/vis_query_instance_slam.py` | Query-head PLY/JSON/NPZ exports, `filter_noise_instances`, `dump_instance_labels_npz`, `dump_instance_sem_queries_npz` |
| `amb3r/vis_instance_hdbscan.py` | Optional HDBSCAN clustering on dense instance features for cleaner PLY |
| `slam_semantic/semantic_voxel_map.py` | `VoxelFeatureMap` — the actual voxel store for sem features |
