# ma_slam Semantic Branch — Implementation Plan

Port OpenSe3r's query-instance semantic mapping (the `pipeline_query_gt.py` path) onto
`ma_slam` as an independent package `src/ma_slam_semantic/`. ma_slam itself gets only
3 hooks + 2 accessors (zero behavior change); everything else is new code.

**Reference design (frozen config)**: the settings of
`/mnt/HDD4/ricky/OpenSe3r/outputs/slam_opense3r_query/scannet_gt8_fastsam/005_da3_tipsv2l14_query_fastsam_semconf_vox0.05_scenenn_geo_default_conf032_gt/005_run_args.json`
— DINOv2+TIPSv2-L14 standalone encoder → 12-block causal transformer (KV-cache, per-frame
causal; "chunk" is just batching) → Mask2Former query head, FastSAM external query seeding,
matching score `0.3·desc + 0.3·vox + 0.25·geo` @ threshold 0.32, KNN gate 1.0 m,
voxel 0.05 m, resolution 518×336, flat KV (no pruning; H2O later).

**Key decisions** (rationale in conversation 2026-07-13):
- Geometry comes from ma_slam; the semantic side follows the old `update_gt` external-geometry
  contract: `c2w [T,4,4]` global (frame0=I), `world_points [T,H,W,3]` global, validity/conf,
  depth, intrinsics.
- **Instance-level storage only** — the dense semantic voxel store (`update_sem`/`VoxelFeatureMap`)
  is NOT ported. Per-instance: descriptor [256], sem_query [1024], provenance observations.
- **LC-safe design**: provenance (immutable, pose-free: mask RLE + submap-LOCAL centroid/bbox)
  vs derived geometry (global center/AABB always recomputed from the current graph; per-instance
  global voxel set as a versioned cache keyed on owning frames' `M_i = G_i @ inv(L_i)` signature,
  re-keyed lazily when moved > voxel_size/2). Loop closure then *heals* the instance map
  (vox/geo score terms snap back) instead of corrupting it.
- KV-cache staleness (pos3d conditioning of old tokens) is accepted; a reset valve
  (`max pose delta > threshold → drop KV`) is wired but default-off; measure first.
- Overlap frame dedup: ma_slam submaps share 1 frame; feed frames [1:] of every submap
  after the first so the causal stream has no duplicates.

---

## S1 — ma_slam interfaces (~40 lines, only edit of existing code)

### S1.1 `src/ma_slam/solver.py` — hooks

```python
def __init__(self, config=None, model=None, device="cuda", hooks=None):
    ...
    self.hooks = hooks            # duck-typed: on_submap / on_loop / on_finish

# process_submap(), after self.graph.optimize():
if self.hooks is not None:
    self.hooks.on_submap(sm, self.map, self.graph)     # post-optimize poses

# _add_loop_constraint(), after self.stats["n_loops"] += 1:
if self.hooks is not None:
    self.hooks.on_loop(sm.key(cand.query_frame), match_sm.key(cand.match_frame))

# run(), right before self.map.write_poses(...):
if self.hooks is not None:
    self.hooks.on_finish(self.map, self.graph)         # final graph, pre-export
```

### S1.2 `src/ma_slam/submap.py` — global-geometry accessors

```python
def frame_pose_global(self, graph, i):
    """Optimized global cam->world of frame i (current graph state)."""
    return graph.get_pose(self.key(i))

def frame_points_global(self, graph, i):
    """Unfiltered (H,W,3) point map of frame i in the CURRENT global frame.
    points are submap-LOCAL -> must go through M = G_i @ inv(L_i) (fog gotcha)."""
    M = graph.get_pose(self.key(i)) @ np.linalg.inv(self.poses_local[i])
    return self.points[i] @ M[:3, :3].T + M[:3, 3]
```

Regression gate: rerun one existing eval (s0011 rgb+depth+intr) — numbers must be identical.

---

## S2 — `ma_slam_semantic/` skeleton: config, bridge, semantic model

### S2.1 `config.py`
`FROZEN` = the 005-geo run-args verbatim (arch flags MUST match ckpt; loaded strict=False so
mismatches fail silently). `TUNABLE` ≈ 10 knobs: conf gate, knn_radius, match_threshold,
w_desc/w_vox/w_geo, sigma_dist, voxel_size, kv_policy, kv_reset_delta, opense3r_root, ckpt paths.

### S2.2 `bridge.py` — ma_slam → OpenSe3r tensor contract

```python
class SubmapBridge:
    def __init__(self, res=(518, 336)):
        self.res, self.T0_inv = res, None

    def chunk(self, sm, gmap, graph):
        i0 = 0 if self.T0_inv is None else 1          # drop shared overlap frame
        idx = list(range(i0, sm.n))
        c2w, wp = [], []
        for i in idx:
            G = sm.frame_pose_global(graph, i)
            if self.T0_inv is None:
                self.T0_inv = np.linalg.inv(G)        # lock gauge: frame0 == identity
            c2w.append(self.T0_inv @ G)
            P = sm.frame_points_global(graph, i)      # (H,W,3) at MA resolution
            wp.append(apply_T(self.T0_inv, resize_nn(P, self.res)))
        images = load_images([sm.image_paths[i] for i in idx], self.res)   # [-1,1]
        conf  = np.stack([resize_nn(sm.conf[i],  self.res) for i in idx])
        depth = np.stack([resize_nn(sm.depth[i], self.res) for i in idx])
        K     = np.stack([rescale_K(sm.intrinsics[i], sm.depth[i].shape, self.res) for i in idx])
        valid = conf > conf.mean()                    # MA conf -> validity (update_gt-style)
        return dict(images=images, c2w=np.stack(c2w), world_points=np.stack(wp),
                    depth=depth, intrinsics=K, valid=valid, conf=conf,
                    frame_keys=[sm.key(i) for i in idx],
                    submap_base=sm.base_id, frame_idx=idx)

class PoseTracker:
    """Snapshot M_i per frame key each submap; report max translation delta since last
    call (drives the KV reset valve; voxel cache uses its own per-instance signature)."""
```

Gotchas handled here: resolution mismatch (nearest for points/depth/conf, K rescaled),
conf remap (MA conf is not (0,1) — `_conf_sig` was VGGT-specific), gauge normalization,
overlap dedup.

### S2.3 `sem_model.py` + `kv/policies.py`

```python
class SemanticModel:
    def __init__(self, cfg, device="cuda"):
        sys.path.insert(0, cfg.opense3r_root)          # phase-1: import, don't vendor
        self.model = build_from_run_args(cfg.FROZEN)   # DINOv2+TIPSv2 + causal + query head
        load_ckpt(self.model, cfg.FROZEN["ckpt_path"]) # strict=False (frozen backbones missing)
        self.fastsam = FastSAMSegmenter(cfg.FROZEN["fastsam_ckpt"], ...)
        self.kv = self.model.init_kv_cache()
        self.policy = make_kv_policy(cfg.kv_policy)    # none | window | h2o (later)

    @torch.no_grad()
    def forward_chunk(self, chunk):                    # per submap == batched causal frames
        regions = self.fastsam.segment(to01(chunk["images"]))     # query seeding
        out = self.model.forward_semantic(             # signatures fixed during port
            chunk["images"], depth=chunk["depth"], intrinsics=chunk["intrinsics"],
            c2w=chunk["c2w"], world_points=chunk["world_points"], valid=chunk["valid"],
            seed_regions=regions, kv_cache=self.kv)    # c2w/world_points feed pos3d cond
        self.kv = self.policy.after_chunk(self.kv)
        return out    # mask_logits (T,N,Hf,Wf), objectness (T,N), descriptor (T,N,256), sem feat

    def reset_kv(self): self.kv = self.model.init_kv_cache()

class KVPolicy:                                        # kv/policies.py
    def after_chunk(self, kv, meta=None): return kv    # NonePolicy = frozen config default
# WindowPolicy(existing prune port) / H2OPolicy(later; needs attention scores — SDPA may
# not expose them: eager fallback or proxy score, evaluate when implementing)
```

**S2 verification**: port `vis_2d_masks` dump; eyeball masks/objectness on scenenn 005.

### S2.4 Golden test mode — `--geometry gt`
Bridge can also be fed GT poses+depth (bypassing ma_slam entirely) → must reproduce the old
repo's 005 GT run (instance count, mIoU within noise). This isolates port bugs from
SLAM-geometry differences. Implement early; it is the main correctness harness for S2/S3.

---

## S3 — `instance_map.py`: provenance-based InstanceTable

```python
@dataclass
class Obs:                                   # immutable, pose-free
    submap_base: int; frame_idx: int; slot: int
    mask_rle: bytes                          # head-res bool mask, ~1-3 KB
    local_centroid: np.ndarray               # (3,) submap-LOCAL
    local_bbox: np.ndarray                   # (2,3) submap-LOCAL

class InstanceRecord:
    descriptor: np.ndarray                   # [256] running mean (momentum 0.75)
    sem_query: np.ndarray                    # [1024] running mean (TIPSv2 space)
    obs: List[Obs]; count: int; objectness: RunningStat
    _vox: set | None; _vox_sig: bytes | None # versioned global-voxel cache

class GeoResolver:                           # holds live refs to gmap+graph
    def M(self, base, i): ...                # graph.get_pose(base+i) @ inv(L_i)
    def sig(self, rec): ...                  # hash of owning frames' current M_i (quantized)
    def lift(self, obs): ...                 # mask (upsampled to MA res) -> Submap.points -> @ M

class InstanceTable:
    def center_global(self, rec):            # per-obs local_centroid @ M — always fresh
    def aabb_global(self, rec):              # union of transformed per-obs boxes
    def voxels_global(self, rec):            # cache hit iff sig unchanged (< voxel_size/2 move);
                                             # else re-lift all obs, quantize @ 0.05 m
    def match_chunk(self, proposals, submap):
        # per frame: gate = point-to-AABB(center-fresh) <= knn_radius(1.0), max 32 cands
        # cost = -(0.3*cos(desc) + 0.3*vox_iou + 0.25*exp(-d/0.25))   [w_sem=0]
        # hungarian; score >= 0.32 -> merge(update means, append Obs) else spawn
```

Notes:
- Proposal formation reuses `opense3r.instance_post.process_chunk_to_proposals` on the
  bridge's *global* world points → proposals are born in global coords.
- Lazy geometry is matching-critical here (not just the gate): 55% of the score is
  geometric and sigma_dist is 0.25 m. Cache re-key cost: only gated candidates (≤32)
  whose owners moved — ~ms each, only after LC events.
- `on_loop` hook just logs; dirtiness is fully handled by the M-signature mechanism.

---

## S4 — export, text query, eval

### `export.py` (driven by on_finish, final graph)
1. `rekey_all(force=True)` — final re-lift of every instance from masks.
2. Final merge pass: union-find over pairs with AABB overlap ∧ fused score ≥ merge_desc_high
   (0.75) — cleans duplicates spawned during pre-LC drift.
3. Noise filter (port subset of `filter_noise_instances`: inst_filter_min_count /
   max_extent_m 4.0 / max_size_m3 5.0 / elongated / tiny caps from FROZEN).
4. Write (formats identical to old repo so its eval scripts run unmodified):
   - `<name>_query_instance_table.json`, `<name>_instance_sem_queries.npz`
     `{inst_ids, sem_queries [K,1024], voxel_size}`  ← the instance-level features (the
     only semantic features persisted; ~MB instead of GB)
   - `<name>_query_instance_voxels.ply` / `_boxes.ply`
   - labeled `combined_pcd`: ma_slam cloud → per-instance voxel lookup → instance/semantic colors
   - `<name>_instance_labels.npz` for `eval_instance_npz_scannet.py`

### `query_text.py`
TIPSv2 SigLIP text head, 10-template indoor ensemble (port `build_text_embeddings_via_backbone`
+ `DEFAULT_TEXT_TEMPLATES`) → cosine vs `sem_queries [K,1024]` → per-instance label /
top-k retrieval; broadcast label to instance voxels for the semantic PLY
(the `query_instance_semantic.py` "one label per instance" route).

### `run.py`

```python
hooks = SemanticHooks(sem_cfg)     # bridge + model + table + exporter + PoseTracker
slam = MaSlam(config=cfg, model=backend, hooks=hooks)
slam.run(image_paths, out, mode=..., depth_paths=..., intrinsics=...)

class SemanticHooks:
    def on_submap(self, sm, gmap, graph):
        if self.tracker.update(gmap, graph) > self.cfg.kv_reset_delta and self.cfg.kv_reset:
            self.model.reset_kv()                     # default OFF; log deltas regardless
        chunk = self.bridge.chunk(sm, gmap, graph)
        props = make_proposals(self.model.forward_chunk(chunk), chunk)
        self.table.match_chunk(props, sm)
    def on_loop(self, k_new, k_old): self.events.append((k_new, k_old))
    def on_finish(self, gmap, graph): self.exporter.run(gmap, graph, self.table)
```

Eval wiring: reuse OpenSe3r's ScanNet/SceneNN instance + semantic eval scripts as-is.

---

## S5 — later / deferred
- H2O KV policy (attention-score exposure vs SDPA; proxy scores) behind `KVPolicy`.
- KV-reset default decision from S4 measurements (log max-pose-delta + post-LC mask quality).
- Perf profiling: FastSAM per-frame cost is the suspected new bottleneck; DINOv2+TIPSv2+causal
  is far lighter than DA3-giant, so >15fps combined is plausible but unverified.
- Broader datasets (TUM, real-sensor with --depth_max).

## Dependencies / env actions
- `amb3r_bw` env: check `ultralytics` (FastSAM) + `scipy` present; **ask before installing**
  anything into this shared env.
- Weights: OpenSe3r ckpt (`opense3r_fastsam_query_semconf/checkpoint-last.pth`), `VGGT.pt`
  (DINO weight source only), TIPSv2-L14 (HF), `FastSAM-x.pt`.
- Phase-1 imports OpenSe3r code via `--opense3r_root` sys.path (no vendoring); the dependency
  dissolves when ma_long is eventually merged into the OpenSe3r repo.

## Order & effort
| Step | Deliverable | New/changed LOC | Gate |
|---|---|---|---|
| S1 | hooks + accessors | ~40 (edit) | existing eval identical |
| S2 | config/bridge/sem_model + 2D mask dump + `--geometry gt` | ~500 | golden test reproduces old 005 GT run qualitatively |
| S3 | InstanceTable + proposals | ~500 | golden test instance metrics match old repo |
| S4 | export/query/eval + full ma_slam-geometry run | ~350 | scenenn 005 + s0011 end-to-end; LC scene (s0231) shows merge-not-duplicate |
| S5 | KV policies, perf | — | — |
