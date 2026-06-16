# AMB3R SLAM — design ideas to borrow for ma_long

Notes from reading `thirdparty/amb3r/slam/` and `amb3r/model.py`. AMB3R's SLAM is a
**sliding-window + keyframe-graph** system (init on first `map_init_window=20` frames, then
map every `map_every=8` frames, always re-including the previous window's keyframes). ma_long
is **chunk-based** (fixed overlapping windows, no adaptive keyframes yet). Several AMB3R ideas
are directly worth adopting; the model-specific parts are not.

Key files: `slam/pipeline.py`, `slam/memory.py` (`SLAMemory`), `amb3r/tools/keyframes.py`,
`tools/pose_dist.py`, `tools/pts_align.py`, `tools/pose_interp.py`, `amb3r/model.py:288`
(`run_amb3r_vo`), `slam/slam_config.yaml`.

## Worth adopting

### 1. Keyframe selection by pose-distance + confidence (`keyframes.py:select_keyframes_iteratively`)
- **Pose distance** = `rotation_angle/180 + lambda_t · translation_distance` (`pose_dist.py:54`).
- Greedy: a frame becomes a keyframe if it is **> `keyframe_threshold` (0.15)** from *all*
  existing keyframes; ties broken by **highest mean confidence**, then **earliest index**.
- *For ma_long:* replace today's fixed every-N-frames chunking with **content-adaptive chunk
  boundaries / reference frames** — pick chunk anchors by this pose-distance criterion so chunks
  cover motion evenly (helps the cs40-on-0011 kind of anomaly where a fixed split aligns badly).

### 2. Confidence-weighted iterative map fusion (`memory.py:~200`)
```
pts   = (conf_old·iter_old·pts_old + conf_new·pts_new) / (conf_old·iter_old + conf_new)
conf  = (conf_old·iter_old + conf_new) / (iter_old + 1);   iter += 1
```
- High-confidence pixels dominate; `iter` counter gives older, repeatedly-observed points
  inertia. Uses `conf_sig = (conf-1)/conf` to normalize DUSt3R-family confidence.
- *For ma_long:* in overlap regions we currently just pick one chunk's points. This weighted
  fusion would give cleaner merged geometry (less ghosting) in the overlap bands.

### 3. Periodic keyframe resampling for bounded memory (`memory.py:resample_keyframes`)
- Trigger when window grows (`max_dist < 1.2` and `len > max_keyframes=10`). Keep a diverse
  set of `min_keyframes=7`: always the newest, + `top_k=4` closest *diverse* (>0.1 apart, or
  >200-frame loop candidates), + fill with "transition" frames (0.1–0.4 apart), + `bridge=1`.
- *For ma_long:* relevant if we move to a streaming/online mode (robot use) where we can't keep
  all chunks — gives a principled bounded-memory keyframe set instead of all-frames-on-disk.

### 4. Confidence-adaptive compute gating (`model.py:288 run_amb3r_vo`)
- `if conf.mean() > conf_threshold_front (2.7): use frontend only; else run heavier backend`.
- Then blend frontend/backend only if cross-consistent (`|p0-p1|.mean() < 0.04`), else pick the
  variant with lowest keyframe reprojection error.
- *For ma_long:* a cheap confidence gate could skip the expensive loop-window re-inference when a
  chunk is already high-confidence, or decide when to spend extra passes — directly relevant to
  the 2× LC cost we measured. Our verification's "co-location + scale" check is the same spirit
  as AMB3R's cross-consistency gate.

### 5. Pose blending / interpolation (`pose_interp.py` SLERP + linear) and scale-invariant
coordinate alignment (`pts_align.py:coordinate_alignment`) are self-contained and reusable if we
ever blend two pose estimates for the same frame (e.g. overlap frames owned by two chunks).

## Caveats — recalibrate, don't copy thresholds

- **Confidence ranges differ per model.** AMB3R conf is not 0–1 (sigmoid up to ~10); the 2.7
  gate and 0.04 consistency are AMB3R-specific. MapAnything conf rises sharply with depth input
  (≈1 rgb → ≈37 rgb+depth+intr), so any gate must be measured on MapAnything outputs per mode.
- **Backend voxel refinement + feature fusion** (`backend.py`, PointTransformerV3, hash voxels)
  is tied to AMB3R's two-stage architecture — not reusable as-is; the hash-voxel *aggregation*
  idea is general if we ever add a voxel map.
- **Cross-chunk consistency is ma_long's job** (our loop closure / global Sim3 opt). Per the
  multi-backend plan, when adapting AMB3R as a backend we keep its front→conf-gate→backend trunk
  but **drop its keyframe-memory / blend-vs-memory** part. See `[[ma-long-multibackend-plan]]`.

## Implementation status (2026-06-15)

Ported as model-agnostic numpy utilities in `ma_long/tools/` (CPU unit-tested):
- ✅ `pose_dist.py` — `pairwise_extrinsic_distance` (normalized rot/180 + λ·trans).
- ✅ `keyframes.py::select_keyframes` — greedy pose-distance + confidence selection (idea #1 core).
- ✅ `keyframes.py::resample_keyframes` — bounded diverse keyframe set (idea #3).
- ✅ `keyframes.py::adaptive_chunk_indices` — content-adaptive chunk boundaries (idea #1).

Integrated into the pipeline (CPU-tested; quality benefit pending GPU validation):
- ✅ **Confidence-weighted overlap fusion** (idea #2) in `pipeline._save_pointclouds`: writes
  each chunk's *owned* frames only (no duplicated overlap → no ghosting) and conf-blends the
  overlap band with the previous chunk. Toggle `Model.fuse_overlap` (default True).

Not yet wired (need GPU validation + design choices):
- **Adaptive chunking (idea #1) full integration.** The primitive works, but: (a) it needs a
  provisional pose pass to drive boundaries, and (b) with a *fixed* overlap, fast-motion regions
  degenerate to step≈1 (too many small chunks). Efficient use requires the pipeline to support
  **per-chunk proportional overlap** (today `overlap` is a global constant used by alignment and
  frame-ownership). Decide: second-pass re-chunk vs. online keyframe-anchored windows.
- **Bounded keyframe memory (idea #3)** only matters once ma_long has a streaming/online mode.

## Concrete next steps these suggest for ma_long
1. **Content-adaptive chunk anchors** via pose-distance (idea #1) — likely fixes fixed-split
   alignment anomalies (0011 cs40).
2. **Confidence-weighted overlap fusion** (idea #2) for cleaner merged clouds.
3. **Confidence gate** (idea #4) to cut LC cost when chunks are already confident.
4. (Online mode only) **bounded keyframe memory** (idea #3) instead of caching every chunk.
