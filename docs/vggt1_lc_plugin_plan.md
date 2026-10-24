# Plan — plugging ma-slam's instance LC verifier (S1+S2) into VGGT-SLAM 1.0

Goal: show the S1 (Bi-Desc) + S2 (layout graph) loop verifier is a backbone-agnostic
plug-in. Host = VGGT-SLAM 1.0 (`thirdparty/vggt-slam-1/VGGT-SLAM`, env `vggt_slam1`),
which has NO verification between SALAD retrieval and the loop factor. 2.0 is treated
as concurrent work (its attention verifier stays untouched).

## 1. How 1.0 closes loops (what we hook into)

`solver.py::run_predictions`
1. SALAD embeddings for the new submap; `find_loop_closures(map, submap, max_similarity_thres=0.80, max_loop_closures=max_loops)`
   — score is an **L2 distance** (`best_score < thres` = match); best-N queue (`--max_loops`, default 1).
2. The retrieved frame(s) are **appended to the new submap's image batch** and VGGT runs
   once on `[32 new frames + N loop frames]` → per-frame depth / K / extrinsic for the
   query frame AND the retrieved frame in ONE consistent frame+scale.
`solver.py::add_points` (~L341)
3. For each `LoopMatch`: Sim3 mode → `Pose3.between(pose_detected, pose_query)`;
   SL4 mode → `ransac_projective(points_query, points_detected)`; then
   `graph.add_between_factor(detected_submap_id, query_submap_id, H, noise)`.
No verification anywhere → our gate goes between 1 and 3.

Why this host is ideal for S2: both frames of a candidate are re-inferred jointly (step 2),
so their camera-frame instance centres share a scale — E_d / theta_R / E_R work unchanged;
only the absolute `||t*|| > 1.0 m` gate must become relative (`tnorm / d_ref`, VGGT scale is
not metric). That is the single S2 change, and it is one we already planned.

## 2. Single-process design (verified 2026-08-28: 1.0 runs inside amb3r_bw)

The env clash turned out to be confined to SL(4) (gtsam-develop, cp311-only, same module
name as gtsam 4.3a0). The Sim3 path needs only `gtsam.Pose3`, the facebookresearch `vggt`
we already vendor (`vendor/thirdparty/vggt`) and SALAD. `scripts/vggt1_lc/run_vggt1_in_amb3r.py`
puts those on `sys.path`, stubs the viewer-only deps (viser/gradio/trimesh) and SALAD's
training-only bases (`pytorch_lightning.LightningModule` shim with no-op
`save_hyperparameters/log`; `pytorch_metric_learning`, `prettytable` as MagicMock) and
execs their untouched `main.py`. Result on fr1_xyz: 4 submaps / 2 loops / **Sim3-ATE 0.0151
== their own env** (`outputs/vggt_slam_tum/xyz_v1_sim3_amb3r`).

So the plug-in is an in-process gate, no dumps / allowlists:

- `scripts/vggt1_lc/main_gated.py` = the launcher above + two monkeypatches applied before
  `main.py` runs:
  1. `ImageRetrieval.find_loop_closures` → forward `max_similarity_thres=args.lc_thres`
     (relaxed retrieval, recall-first).
  2. `Solver.add_points` → pre-hook: for each `LoopMatch` build a 2-frame chunk from
     `pred_dict` (query frame index `loop.query_submap_frame`, loop frame index
     `last_non_loop_frame_index + index + 1`; images/depth/K/extrinsic/conf are all there,
     same scale), run `SemanticModel` (a1_pq, fresh KV) + `make_proposals` → per-frame raw
     descriptors + camera-frame centres → `bidesc_lc.verify` (raw vs raw) and
     `verify_layout` (relative translation gate `s2_tau_t_rel`) → drop rejected loops from
     `pred_dict["detected_loops"]` before the factor is added. Optional `--lc_dump` writes
     the usual per-candidate JSON for `render_semlc_graph.py`.
- Two-pass (dump → verify → allowlist) stays as the fallback only if SL(4) mode is ever wanted.
- Cost per candidate: one 2-frame semantic forward (~0.2 s), negligible next to their VGGT pass.

## 3. Arms and scenes

| arm | retrieval thres | gate |
|---|---|---|
| A. 1.0 default | 0.80 (their hardcoded) | none |
| B. 1.0 relaxed | 0.95 | none |
| C. 1.0 relaxed + our LC | 0.95 | S1+S2 |
(SL4 variants would need the two-pass fallback; Sim3 is the stable mode at 25 px anyway.)

Scenes: **scene0231_00** (GT; the SALAD-aliasing scene: 157↔28 / 170↔53 at sim 0.80+ are
the known false positives), **office_loop** (no GT; exactly one true loop 469↔25 — count
false accepts), TUM **360 / floor / teddy / room** (GT; sanity that the gate never removes a
true loop: loops ≈ identical to ours today). Keyframer 25 px, submap 32, raw distorted frames.

Metrics: per-arm Sim3-ATE (our evaluator), loops accepted, and for C the gate's
precision/recall against our GT audit (camera distance < 0.5 m ∧ view angle < 60° = real).

## 4. Work items (≈ half a day + ~1 h GPU)

1. `bidesc_lc.BiDescConfig.s2_tau_t_rel` + relative gate in `verify_layout` (use when > 0;
   keep the absolute 1.0 m as the metric default) — validate on our own TUM run = no change.
2. `scripts/vggt1_lc/verifier.py` (2-frame chunk builder from their pred_dict + map-free S1/S2 + dump).
3. `scripts/vggt1_lc/main_gated.py` (launcher + 2 monkeypatches, `--lc_thres --sem_lc --lc_dump`).
4. `scripts/vggt1_lc/run_arms.sh` (A/B/C × scenes, eval, summary table).
5. Results → RESULTS.md ("1.0 + our LC" row) + galleries.

Risks: VGGT depth quality on the 2-frame joint inference is what S2 sees (worse than DA3's
metric depth) — S2's M / E_d may be noisier than in ma-slam; keep S1 as the primary
signal and treat S2 thresholds as-is first, tune only with the GT audit. Frame naming:
paths are reconstructed from frame ids, so only index-named datasets are supported.

## 6. Implementation status (2026-08-28) — DONE, single process

- `src/ma_slam_semantic/bidesc_lc.py`: `BiDescConfig.s2_tau_t_rel` (default 0 = off; when set,
  T_FAR iff `t_norm > rel * d_ref`, replacing the absolute metre gate — VGGT scale is arbitrary).
- `scripts/vggt1_lc/verifier.py` — `Vggt1SemVerifier`: 2-frame chunk from their `pred_dict`
  (VGGT depth/depth_conf/extrinsic/intrinsic + the submap's joint image batch), anchor = query
  frame, validity = depth>0 & conf ≥ their 25-percentile, crop/resize to 518×336 with the bridge
  helpers; fresh KV + fresh box_scale per candidate; raw proposal descriptors (top-16 by objectness,
  = FrameIndex convention) → S1 `verify`; camera-frame centres of the unique pairs → S2
  `verify_layout` (`s2_tau_t_rel=0.5`). Dumps the usual per-candidate JSON (`salad` = their L2).
- `scripts/vggt1_lc/main_gated.py` — env shims + `--lc_thres` (patches
  `ImageRetrieval.find_loop_closures`) + `--sem_lc` (pre-hook on `Solver.add_points`: verifies
  each `LoopMatch`, drops rejected ones before the loop factor). Gotchas: their `main.py` runs
  with cwd=repo (hub ckpt) → our path args are absolutized; `Submap.frame_ids` are NUMBERS parsed
  from filenames → mapped back to paths via the image folder; dropping a loop shifts their
  `loop_index = last_non_loop + index + 1` bookkeeping → only `--max_loops 1` (their default)
  is supported (asserted).
- `scripts/vggt1_lc/run_arms.sh` — arms A/B/C × {TUM xyz,360,floor,teddy,room; s0231; office_loop}
  → `outputs/vggt1_lc/<scene>_<arm>/{poses.txt,run.log,semlc_log.txt,semlc_dump/}` + `summary.csv`.
- Smoke (fr1_xyz, arm C): 2 candidates, both ACCEPT (S1 nuniq 14/16, S2 Ed≈0.01), 0.2–0.6 s each,
  Sim3-ATE 0.0151 == ungated.
