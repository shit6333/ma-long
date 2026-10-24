# ⚠️ MAIN ASSOCIATION CONFIG CHANGED (2026-09-04) — all map-side numbers below are the RE-RUN

`docs/config_change_2026-09-04.md`: the voxel-overlap cue is dropped and the descriptor carries most of the
weight, `w_desc .30/w_vox .30/w_geo .25 @ thr .32` -> **`w_desc .70 / w_vox 0 / w_geo .30 @ thr .65`** (same
checkpoint `a1_pq`). Implemented as the new `SemConfig` default plus `--assoc {new,old}` and individual
`--w_* / --match_threshold` flags. Runs are in `outputs/ma_sem/<dataset>_v2/`; the pre-change runs
(`_final/`) are kept for the ablation table.

**A second, local divergence was fixed at the same time.** Our `final_merge` used the FUSED score
(`w_desc*cos + w_vox*iou + w_geo*geo >= merge_desc_high 0.75`), whose scale depends on the cue weights — the
reference implementation (`semantic_mapping/stages/merging.py:116`) uses the PURE accumulated-descriptor cosine
(`>= merge_desc 0.85`), which is cue-invariant. New default `merge_mode="desc"` (0.85) removes the divergence;
`--merge_mode fused --merge_desc_high 0.75` reproduces every pre-2026-09-04 run.

## Instance results under the new config (AP / AP50 / AP25)

| dataset | **v2 SE3 (main)** | v2 Sim(3) | old config (SE3) | delta AP50 | GT-geo (new config) |
|---|---|---|---|---|---|
| ScanNet val-50 (full) | **27.69 / 53.23 / 72.82** | 30.05 / 54.69 / 74.38 | 27.12 / 51.86 / 72.68 | **+1.37** | 42.16 / 68.36 / 82.78 |
| ScanNet val-50 (no-stuff) | **28.11 / 54.15 / 74.14** | — | 27.48 / 52.70 / 74.07 | **+1.45** | 42.81 / 69.66 / 84.28 |
| SceneNN-12 | **24.20 / 45.71 / 63.63** | 25.11 / 44.94 / 62.35 | 22.72 / 44.00 / 62.98 | **+1.71** | 33.47 / 55.63 / 69.81 |
| 3RScan-47 | **13.58 / 29.59 / 55.37** | 16.46 / 32.88 / 57.87 | 12.21 / 26.84 / 52.91 | **+2.75** | 25.18 / 46.72 / 70.52 |
| ScanNet++ 50, all (excl. 2 div.) | **13.16 / 29.21 / 48.25** | 14.63 / 31.03 / 50.33 | 13.55 / 29.71 / 49.79 | −0.50 | 19.51 / 39.09 / 58.00 |
| ScanNet++ 50, top100 (excl.) | **20.09 / 41.30 / 62.26** | — | 20.72 / 43.37 / 64.52 | −2.07 | 28.06 / 52.89 / 70.98 |

## Semantic results under the new config (instsem mIoU / mAcc)

| benchmark | **v2 SE3** | v2 Sim(3) | old config (SE3 · Sim3) | GT-geo |
|---|---|---|---|---|
| **ScanNet val-50 SN20** (50 scenes, 21 classes) | **36.39 / 68.39** | 36.93 / 68.94 | 35.12 · 35.72 | **41.95** (new cfg) / 41.16 (old) |
| **ScanNet val-50 SN200** (50 scenes, 200 classes) | **15.29 / 25.82** (gtcls-mIoU 25.34) | 15.79 / 26.39 (gtcls 26.17) | 14.46 · 14.86 (gtcls 23.96 · 24.77) | 29.60 gtcls (new) / 29.75 (old) |
| ScanNet-5 SN20 (5 scenes) | — | **39.64 / 58.95** | 38.26 / 57.65 (Sim3) | 44.26 / 62.99 (old cfg; new-cfg 5-scene not reported) |
| ScanNet-5 SN200 (5 scenes, 200 classes) | — | **15.96 / 22.75** (gtcls-mIoU 33.72) | 15.14 / 22.31 (gtcls 31.43) | 17.79 / 26.17 (old cfg) |
| Replica-8 (stride 20) | **15.11 / 21.36** (f-mIoU 44.23) | **19.42 / 27.36** (49.87) | 14.70 / 21.16 · 19.02 / 27.12 | 28.92 |
| Replica-8 (full 2000-frame) | **15.51 / 22.46** (46.26) | **18.40 / 27.06** (50.85) | 15.40 / 22.30 · 18.27 / 26.82 | — |

⚠️ Scene sets: **val-50** is the set the OpenSe3r new-config semantic number (41.95) is on; **ScanNet-5**
(scene0011/0050/0231/0378/0518) is the older 5-scene semantic set. **SN200 GT also exists for all 50 val-50 scenes**
(`OpenSe3r/scannet_200/<sc>/gt_scannet200_rawid_<sc>.ply`, see `model_eval/semantic_eval/VAL50_SN200_GT.md`) —
an earlier revision of this note said otherwise (copied from the eval script's stale `--scannet_root` help text);
val-50 SN200 is now evaluated (row above; `--benches sn200`, default GT root). Earlier revisions of this table compared our ScanNet-5 SN20
against their val-50 41.95 and our SN200 mIoU against their gtcls-mIoU — both mismatched, corrected here.
val-50 command: `tools/ovo_style_semantic_eval.py --scannet_root /mnt/HDD4/ricky/data/scannet_eval/scans
--benches sn20 --tasks instsem` on `outputs/ma_sem/scannet_val50_{v2,final}/`.

## Verification: the loop verifier is unaffected (as predicted)

S1 reads per-frame proposal bags (`FrameIndex.get_lc`), never the map's association weights, so the LC verdicts
and hence every ATE table are structurally independent of this change. Confirmed by re-running:

| check | new config | 2026-08-30 |
|---|---|---|
| TUM 50 px, ours w/o inst-LC (default / relaxed) | 0.0356 / 0.0538 | 0.0357 / 0.0538 |
| TUM 50 px, ours + inst-LC (default / relaxed) | **0.0329 / 0.0300** | 0.0338 / 0.0300 |
| TUM false loops admitted (w/o -> w/) | 1 -> 0 (default), 8 -> 0 (relaxed) | identical |
| VGGT-SLAM 1.0 SL(4) + our gate, TUM 9 seqs | 11 true kept / 4 false rejected | identical, scene by scene |
| VGGT-SLAM 1.0 SL(4) + our gate, Clio 3 scenes | 7 true kept / 9 false rejected | identical, scene by scene |

35/35 verdicts unchanged, so the plug-in table keeps its 5-run means (re-running them would only resample
VGGT-SLAM's own run-to-run noise, which exceeds the effect being measured).

## The one regression: ScanNet++ over-segments under the new config

ScanNet++ is the only dataset that loses (AP50 −0.50 all / −2.07 top100) while the GT-geometry side *gains*
(+2.01 top100). It is not a systematic drop — 26/48 scenes improve, 21 lose, |delta| > 3 in 24 — but the losers
share a signature: **instances per scene rise 115 -> 145** and `corr(delta AP50, delta n_inst) = −0.51`. The worst
cases are pure over-segmentation (bde1e479ad 49.6 -> 28.9 with 167 -> 227 instances; c4c04e6d6c 39.8 -> 26.7 with
134 -> 212; 21d970d8de 28.5 -> 20.5 with 152 -> 263). ScanNet++ has by far the longest sequences (~430 keyframes
vs 135 val-50 / 257 3RScan) and the most loop closures (54/scene): once the voxel-overlap cue is gone, an object
re-observed after a long gap whose descriptor similarity misses the stricter 0.65 bar has nothing left to rescue
the association, and spawns a second instance. Reported as-is; no local re-tuning was applied.


## E4 re-run under the new config — same conclusion

`docs/e4_lc_map_healing.md`, LC-OFF arm re-run with `--assoc new` (`outputs/ma_sem/{scannetpp,scannet_val50}_noLC_v2/`).

| metric (mean/scene) | ScanNet++ 48 ON | OFF | delta | val-50 ON | OFF | delta |
|---|---|---|---|---|---|---|
| loops accepted | 54.19 | 0 | +54.19 | 12.18 | 0 | +12.18 |
| SE3-ATE = align RMSE (m) | 0.089 | 0.133 | **−0.044** | 0.142 | 0.155 | −0.014 |
| AP | 13.16 | 10.32 | **+2.83** | 27.69 | 26.18 | +1.51 |
| AP50 | 29.21 | 24.09 | **+5.12** | 53.23 | 51.68 | +1.55 |
| AP25 | 48.25 | 43.64 | +4.61 | 72.82 | 74.15 | −1.33 |
| n_pred | 95.3 | 93.5 | +1.9 | 38.8 | 38.5 | +0.3 |
| GT matched (IoU>=0.25) | 46.5 | 42.1 | **+4.4** | 17.8 | 18.3 | −0.4 |
| **duplicate instances** | **1.13** | **1.17** | **−0.04** | **0.64** | **0.64** | **0.00** |

Per-scene duplicates: ScanNet++ lower in 14/48, higher in 11, tied 23 (54 vs 56); val-50 lower 8/50, higher 8,
tied 34 (32 vs 32). Identical conclusion to the 2026-08-31 run under the old config: **the duplicate-instance
mechanism is a measured null** (54 loop closures/scene move it by 0.04) and the map gain is geometric —
align RMSE 0.133 -> 0.089 m with +4.4 more GT instances matched at all. The new config lowers duplicates slightly
in both arms (1.31/1.35 -> 1.13/1.17), consistent with a descriptor-led association keeping an object together.

# FINAL METHOD — Instance-Descriptor Loop Verification (as of 2026-08-31; retrieval default updated 2026-10-22)

One line: SALAD says the two images *look* alike; the gate asks whether they contain the *same set of objects* —
at least four one-to-one (mutual nearest-neighbour) instance correspondences, and no fallback if the evidence is
absent. Code: `src/ma_slam_semantic/{hooks.py,bidesc_lc.py}` (`--sem_lc veto --sem_lc_mode rigid --rigid_geom off`).

**1. Instance bag per keyframe** (`SemanticHooks.on_submap`). The OpenSe3r `a1_pq` model runs on the submap in
4-frame chunks (input = images + the SLAM geometry). Per frame we keep the top **k = 16** proposals by objectness,
each carrying a 256-d L2-normalised instance descriptor and its 3D centre in that frame's camera coordinates
(pixel ownership in `prefer` mode, i.e. not gated by the map's confidence mask). The bag lives only in the
`FrameIndex` — it is **never merged into the instance table** (doing so cost map AP 16.7 -> 14.4).

**2. Retrieval** (unchanged): SALAD/DINOv2 global descriptors, cos >= **0.40** with **top-5 per submap** (the
relaxed setting — the main setting of the latest tables and the code default since 2026-10-22; 0.50 / top-1 is
kept as the strict-retrieval comparison), at least 2 submaps of temporal separation.

**3. S1 — instance-correspondence gate** (`verify_rigid(..., rigid_geom=off)`), run BEFORE the expensive geometry:
  * **Mutual nearest neighbour** on the descriptor cosine matrix: (i, j) counts only if j is i's best match AND i
    is j's best match — one-to-one, so repeated furniture cannot produce a fan of matches.
  * A correspondence also needs cos >= **tau = 0.85**.
  * Verdict: fewer than m objects in either bag -> **REJECT**; fewer than **m = 4** correspondences -> **REJECT**;
    otherwise PASS. No third state, no handing the candidate back to geometry.
  * The three constants (k = 16, tau = 0.85, m = 4) are fixed across every dataset in this file; none was tuned.
    m = 4 because three correspondences can be accidentally congruent in 3D, four is the smallest over-determined
    set for a rigid correspondence.
  * Cost: one 16x16 matrix. Rejected candidates never pay for the joint re-inference.

**4. Geometric verification** (a standard pipeline step, NOT a contribution, present in both "ours" rows of every
table): a PASS goes to the loop window's joint DA3 re-inference, which yields the relative SE(3) and a
co-location ratio; <= **0.70** admits the loop factor (Huber kernel), then incremental re-optimisation of the pose
graph propagates to all frames.

**Plugged into a foreign SLAM** (VGGT-SLAM 1.0 SL(4)): only step 3 is used. `scripts/vggt1_lc/verify_server.py`
holds the model in `amb3r_bw` behind a Unix socket; `scripts/vggt1_lc/main_gated_rpc.py` runs their untouched
`main.py` in their own env and ships each candidate's two frames plus THEIR VGGT depth/pose/K over the socket.
Their code and their SL(4) factors are unmodified. Verdicts were correct 15/15 on TUM and 20/20 on Clio.

**Deliberately dropped** (ablations are in this file): the S2 rigid layout check (only Clio-office ever needed it,
and geometric verification catches the same loops); joint re-embedding (`--sem_lc_mode joint`, 0.0338 -> 0.0321 on
TUM 50 px for one extra forward per candidate); and the old Bi-Desc rules, whose FAIL verdict merely raised the
coloc bar to 0.45 and therefore leaked two false loops on fr1_room under relaxed retrieval (0.149 vs 0.033).

# PAPER TABLES (2026-08-30, FINAL framing) — inst-LC = S1 instance-correspondence gate; geometric verification
# (coloc) is a standard pipeline step present in both "Ours" rows and is NOT a contribution.

> **LATEST (2026-10-22): relaxed retrieval (SALAD 0.40, top-5/submap) is the main setting and the code
> default.** Main rows = Ident3R-SLAM w/o IDLV 0.054 → w/ IDLV **0.030**. The 0.50/top-1 rows below are kept as
> the stricter-retrieval comparison. Code defaults (`ma_slam_semantic.run`, `ma_slam.run`, live stream) now
> reproduce the w/ IDLV row with no flags: DA3 rgb, w32, 50 px keyframer, `--sem_lc veto --sem_lc_mode rigid
> --rigid_geom off --lc_proposal_conf prefer --lc_history bag`, h2o 8/8/0.25. Archived-run scripts pin their
> old flags explicitly.

## Main: TUM RGB-D fr1, RGB-only (Sim3-ATE RMSE m; w=32, 50 px keyframer, evo_ape -as)

| method | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | avg |
|---|---|---|---|---|---|---|---|---|---|---|
| DROID-SLAM* [VGGT-SLAM 2.0 Tab. I] | 0.202 | 0.032 | 0.091 | 0.064 | 0.045 | 0.918 | 0.056 | 0.045 | 0.012 | 0.163 |
| MASt3R-SLAM* [idem] | 0.070 | 0.035 | 0.055 | 0.056 | 0.035 | 0.118 | 0.041 | 0.114 | 0.020 | 0.060 |
| ViSTA-SLAM [idem] | 0.104 | 0.030 | 0.030 | 0.070 | 0.052 | 0.067 | 0.023 | 0.080 | 0.015 | 0.052 |
| VGGT-SLAM 1.0 SL(4), lc 0.95 (5-run mean) [outputs/vggt1_lc_sl4_d50{,_rep2..5} sl4_relax95] | 0.106 | 0.027 | 0.038 | 0.154 | 0.023 | 0.301 | 0.033 | 0.041 | 0.023 | 0.083 |
| VGGT-SLAM 1.0 SL(4) + our inst-LC (5-run mean) [sl4_relax95_gate] | 0.077 | 0.029 | 0.054 | 0.158 | 0.022 | 0.200 | 0.036 | 0.042 | 0.016 | 0.071 |
| VGGT-SLAM 2.0 [outputs/vggt_slam_official_d50] | 0.050 | 0.027 | 0.029 | 0.100 | 0.025 | 0.064 | 0.026 | 0.035 | 0.014 | 0.041 |
| Ident3R-SLAM w/o IDLV (SALAD 0.40 top-5 + geom. verify) [ma_slam/tum_kf50_relax] | 0.069 | 0.027 | 0.021 | 0.051 | 0.026 | 0.189 | 0.020 | 0.070 | 0.011 | 0.054 |
| **Ident3R-SLAM w/ IDLV** [sem_lc_shadow/a1pq_w32_kf50_relax_bs1] | 0.043 | 0.027 | 0.021 | 0.055 | 0.026 | 0.033 | 0.020 | 0.033 | 0.011 | **0.030** |
| *strict retrieval (0.50, top-1):* w/o IDLV [ma_slam/tum_kf50/da3_rgb] | 0.046 | 0.027 | 0.021 | 0.056 | 0.025 | 0.031 | 0.021 | 0.083 | 0.011 | 0.036 |
| *strict retrieval (0.50, top-1):* w/ IDLV [sem_lc_shadow/a1pq_w32_kf50_bs1] | 0.050 | 0.027 | 0.021 | 0.065 | 0.027 | 0.031 | 0.021 | 0.051 | 0.011 | 0.034 |

Relaxed retrieval (ours SALAD 0.40, 5/submap): Ours (SALAD + geom. verify) [ma_slam/tum_kf50_relax] 0.069/0.027/0.021/
0.051/0.026/0.189/0.020/0.070/0.011 = 0.054 (8 false loops) -> Ours + inst-LC [a1pq_w32_kf50_relax_bs1] 0.043/0.027/
0.021/0.055/0.026/0.033/0.020/0.033/0.011 = **0.030** (0 false).
Supp (no geometric verification): SALAD only 0.058 (default, 4 false) / 0.112 (relaxed, 22 false); SALAD + inst-LC
without geom. verify 0.033 / 0.030 (0 false).
False-loop counts (GT co-visibility): geom.verify 1 -> inst-LC 0 (default), 8 -> 0 (relaxed); 1.0 SL(4)@0.95 4 -> 0 in
every one of 5 runs. 1.0 SL(4) and +inst-LC rows are same-session pairs (their run variance is large).

## Clio (50 px; loops = true / FALSE)

| | apartment | office | cubicle |
|---|---|---|---|
| default: Ours (SALAD + geom. verify) | 0.065 (2/0) | 0.100 (2/0) | 0.165 (1/1) |
| default: Ours + inst-LC | 0.066 (1/0) | 0.113 (0) | **0.048** (1/0) |
| relaxed: Ours (SALAD + geom. verify) | 0.434 (7/1) | 0.106 (9/0) | 0.164 (6/2) |
| relaxed: Ours + inst-LC | **0.066** (4/0) | 0.106 (3/0) | **0.050** (2/0) |
| SL(4) their 0.80 / +inst-LC | 0.188 (0) / 0.277 (0 cand.) | 7.43 (2/1) / 3.78 (2/0) | 0.054 (1/0) / 0.043 (1/0) |
| SL(4) relaxed 1.10 / +inst-LC | 0.248 (3/2) / **0.121** (3/0) | 6.62 (3/5) / 5.38 (3/0) | 0.060 (1/1+1w) / 0.089 (1/0) |
| supp: SALAD only (default / relaxed) | 0.261 / 0.423 | 1.39 / 3.52 | 0.148 / 0.545 |
| supp: SALAD + inst-LC, no geom. verify (default / relaxed) | 0.066 / 0.066 | 0.113 / 1.32 (3 false, 5 pairs each — identical cubicles) | 0.048 / 0.050 |

Note on office relaxed without geometric verification: S1 alone passes 3 aliasing loops with 5 MNN pairs each
(same object classes, different arrangement); geometric verification removes them (coloc 0.73-4.42; one within 0.03
of the 0.70 bar) and so does the rigid layout check S2 (inliers 0-2, 0.106). Kept as a limitation / discussion item.

# ma_long / ma_slam — Results & Findings

Long-sequence reconstruction on the **MapAnything** backbone (with an optional **DA3** backbone).
Three front-ends share one chunk model + one eval: the **offline chunk pipeline** (`ma_long`), the
AMB3R-style **fused online SLAM** (`ma_long/slam.py`), and **`ma_slam`** — the VGGT-SLAM-style
online system (submaps → SE3 gtsam factor graph → incremental global re-opt → robust loop closure),
which is the most developed and recommended.

- **Hardware:** 1× RTX PRO 6000 Blackwell (~97 GB), CUDA 12.8, torch 2.8+cu128.
- **Data:** ScanNet-style scenes `scene0011_00` (238f), `scene0378_00` (190f), `scene0231_00` (444f),
  RGB + depth (16-bit mm) + `intrinsic.txt` + `gt_pose.txt` (TUM). Plus an in-the-wild RealSense scene.
- **Metric:** ATE RMSE (m) after aligning the trajectory to GT.
  - **Sim3-ATE** — scale-invariant (classic monocular ATE; *hides* metric-scale error).
  - **SE3-ATE (metric)** — rotation+translation only; reveals absolute/scale accuracy.
  - `scale` = best-fit Sim3 scale est→GT (1.0 = perfectly metric).
  - **Judge multi-modal / metric modes with SE3-ATE, not Sim3-ATE.**

---

## TUM RGB-D benchmark

ma_slam (LK keyframe gate 25 px + `submap_size 32`, SE3) on the 9 `freiburg1` sequences; baselines from
the VGGT-SLAM 2.0 paper Table I (Sim3-aligned ATE RMSE, m). Split **Uncalibrated (RGB-only)** vs
**Calibrated** as in that table — our `+intr` / `+depth+intr` configs use more input, so they sit in the
Calibrated group.

| method | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | **avg** |
|---|---|---|---|---|---|---|---|---|---|---|
| *— Uncalibrated (RGB-only) —* | | | | | | | | | | |
| MASt3R-SLAM\* | 0.070 | 0.035 | 0.055 | 0.056 | 0.035 | 0.118 | 0.041 | 0.114 | 0.020 | 0.060 |
| DROID-SLAM\* | 0.202 | 0.032 | 0.091 | 0.064 | 0.045 | 0.918 | 0.056 | 0.045 | 0.012 | 0.163 |
| ViSTA-SLAM | 0.104 | 0.030 | 0.030 | 0.070 | 0.052 | 0.067 | 0.023 | 0.080 | 0.015 | 0.052 |
| VGGT-SLAM Sim(3) | 0.123 | 0.040 | 0.055 | 0.254 | 0.022 | 0.088 | 0.041 | 0.032 | 0.016 | 0.075 |
| VGGT-SLAM SL(4) | 0.071 | 0.025 | 0.040 | 0.141 | 0.023 | 0.102 | 0.030 | 0.034 | 0.014 | 0.053 |
| VGGT-SLAM 2.0 | 0.050 | 0.025 | 0.029 | 0.102 | 0.026 | 0.063 | 0.026 | 0.038 | 0.014 | **0.041** |
| **ours DA3-rgb** | 0.118 | 0.033 | 0.028 | 0.115 | 0.044 | 0.045 | 0.025 | 0.047 | 0.011 | **0.052** |
| **ours MA-rgb** | 0.091 | 0.054 | 0.060 | 0.075 | 0.054 | 0.097 | 0.042 | 0.078 | 0.030 | 0.064 |
| *— Calibrated (intrinsics / depth — more input) —* | | | | | | | | | | |
| MASt3R-SLAM | 0.049 | 0.016 | 0.024 | 0.025 | 0.020 | 0.061 | 0.027 | 0.041 | 0.009 | **0.030** |
| GO-SLAM | 0.089 | 0.016 | 0.028 | 0.025 | 0.026 | 0.052 | 0.019 | 0.048 | 0.010 | 0.035 |
| DROID-SLAM | 0.111 | 0.018 | 0.042 | 0.021 | 0.016 | 0.049 | 0.026 | 0.048 | 0.012 | 0.038 |
| **ours MA-rgb+intr** | 0.074 | 0.029 | 0.040 | 0.058 | 0.067 | 0.084 | 0.029 | 0.064 | 0.016 | 0.051 |
| **ours MA-rgb+depth+intr** | 0.082 | 0.026 | 0.032 | 0.036 | 0.030 | 0.063 | 0.024 | 0.039 | 0.014 | **0.038** |

**Setup:** distorted (raw, no undistort) input; keyframe gate 25 px → `submap_size 32` counts 32
keyframes (= VGGT-SLAM's `w`); Sim3-ATE (their protocol). SE3-avg (metric): MA-rgb 0.220 · DA3-rgb
0.103 · MA-rgb+intr 0.305 · **MA-rgb+depth+intr 0.042**.

**Findings:**
- **RGB-only:** `DA3-rgb` **0.052 ≈ VGGT-SLAM SL(4) (0.053)**, and beats VGGT-SLAM Sim(3), ViSTA, MASt3R\* —
  a metric backbone + plain **SE(3)** matches the 15-DoF SL(4) projective alignment. Behind VGGT-SLAM 2.0 (0.041).
- **With depth:** `MA-rgb+depth+intr` **0.038 < VGGT-SLAM 2.0 (0.041)** and is metrically consistent
  (SE3 ≈ Sim3, scale ≈ 1.0). Competitive among *calibrated* methods (ties DROID-SLAM 0.038, ≈ GO-SLAM 0.035; behind MASt3R-SLAM 0.030).
- **Depth — not intrinsics — anchors scale:** `+intr` alone improves trajectory shape (Sim3 0.051) but
  *worsens* the metric (SE3 0.305, scale 0.6–0.9); only depth fixes it (SE3 0.042). Judge these with SE3.
- **Weak spot = `360`** (pure rotation → no triangulation baseline) across all configs — an inherent backbone limit.

## Semantic loop-closure verification (Bi-Desc S1 + Layout-Graph S2)

Two-stage instance-level verification of SALAD loop candidates (branch `semantic`;
`--sem_lc veto`), sitting between retrieval and the coloc geometric check — the same slot
where VGGT-SLAM 2.0 adds its attention-based verifier, but using the semantic-mapping
branch's instance descriptors + camera-frame instance layouts (zero extra model forwards).
S1 = bidirectional descriptor verification (a FAIL raises the coloc bar 0.70→0.45);
S2 = layout-graph comparison (E_d distance consistency, theta_R common rotation via
Wahba, ||t*|| implied camera displacement) with hard rejects for LARGE_VIEW_CHANGE /
T_FAR / M_LOW (<=2 co-visible instances). Also includes the submap-0 retrieval fix
(back-to-start loops were unretrievable since inception — improves the base rows too).

Sim3-ATE @ submap 32 (same protocol as the table above; base = post-fix DA3-rgb):

| method | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | **avg** |
|---|---|---|---|---|---|---|---|---|---|---|
| VGGT-SLAM 2.0 | 0.050 | 0.025 | 0.029 | 0.102 | 0.026 | 0.063 | 0.026 | 0.038 | 0.014 | 0.041 |
| ours base (post-fix) | 0.095 | 0.030 | 0.023 | 0.183 | 0.040 | 0.043 | 0.025 | 0.052 | 0.010 | 0.056 |
| ours + S1 | 0.095 | 0.030 | 0.023 | 0.145 | 0.043 | 0.043 | 0.025 | 0.052 | 0.010 | 0.052 |
| **ours + S1+S2** | **0.049** | 0.030 | 0.023 | **0.069** | 0.046 | 0.043 | 0.025 | 0.049 | 0.010 | **0.038** |

At submap 20: base 0.053 → +S1 0.044 → +S1+S2 **0.040**. Both submap sizes beat
VGGT-SLAM 2.0's 0.041 from RGB only; per-scene 6W/1T/2L at w32. GT-audited hard rejects:
20 confirmed-false edges removed for 2 thin-evidence real ones (M<=2, redundant loops
covered the same revisits). Full run logs under `outputs/sem_lc_shadow/`.

**Takeaway:** building on VGGT-SLAM's submap + factor-graph recipe, ma_slam shows a *flexible* SLAM that
works directly on **metric** scale (plain SE(3), no projective SL(4)) and is **backbone-agnostic**
(MapAnything / DA3, RGB → RGB+depth+intr), reaching results **competitive** with the strongest
learning-based SLAM methods on TUM RGB-D.

---

## TL;DR

1. **`ma_slam` (online) matches or beats the offline pipeline** in depth modes, with **stable**
   loop closure — at ~15.5 fps / ~15 GB. On the loopy 444-frame scene it gets the best result of
   any method (0.099), where the older AMB3R online LC *degraded* the trajectory.
2. **Depth is the biggest accuracy lever for MapAnything** (fixes metric scale, ~halves ATE).
   **`submap_size 20`** is the sweet spot (VRAM grows ~linearly with submap size, fps stays flat).
3. **DA3 is a far better RGB backbone**: it predicts *metric* geometry from RGB, so `--backend da3
   --mode rgb` gives SE3-ATE ~0.10 (scale ≈1.0) vs MapAnything-rgb ~0.4–0.5 — **3–5× better**, and
   rivals MapAnything's *depth* modes from RGB alone.
4. **Loop closure, tuned:** geometric **co-location verification discriminates real loops** (even
   from look-alike aliasing); `coloc_ratio 0.7` + a **Huber robust kernel on loop factors** (free
   insurance against the ~16 % of aliasing that slips through) is the validated default. A
   re-inference *window* was tested and **dropped** (no ATE gain, slower, worsens aliasing).
5. **Real-sensor depth:** MapAnything **keeps the depth you give it** (no denoising) → noisy depth
   in = noisy geometry out. For RealSense, **`--depth_max 6`** (or DA3 rgb) recovers clean results.

---

## 1. `ma_slam` — headline (submap_size 20, current defaults)

Defaults: `coloc_ratio 0.7`, `half_window 0`, Huber loop kernel. Sim3-ATE (m), loops = accepted in
the `+depth+intr` run.

| scene (frames) | rgb | +depth | **+depth+intr** | loops |
|---|---|---|---|---|
| s0011 (238) | 0.150 | 0.069 | **0.059** | 4 |
| s0378 (190) | 0.098 | 0.051 | 0.055 | 3 |
| s0231 (444) | 0.140 | 0.106 | **0.099** | 10 |

**Performance:** ~**15.5 fps**, ~**15 GB VRAM** (RTX PRO 6000, submap_size 20). VRAM grows ~linearly
with submap_size; fps stays roughly flat (see §5).

---

## 2. `ma_slam` vs offline pipeline vs AMB3R online

`rgb+depth+intr`, Sim3-ATE (m):

| scene | **ma_slam** (online) | offline chunk pipeline | AMB3R online (`slam.py`) |
|---|---|---|---|
| s0011 | **0.059** | 0.081 | ~0.068 |
| s0378 | 0.055 | **0.054** | ~0.067 |
| s0231 | **0.099** | 0.131 (best prior) | 0.131 no-loop → **0.30 with LC (hurts)** |

- In depth modes **ma_slam matches or beats the offline pipeline**, while being online + streaming.
- **The s0231 story (why ma_slam's design wins):** on this loopy 444-frame scene the AMB3R online
  LC and the chunk pipeline's batch LC both *degrade* with the metric model (a late loop's
  re-inference noise exceeds the small accumulated drift, so an ad-hoc closure adds error).
  ma_slam's **global factor graph re-optimized every submap + Huber robust loop kernel** stays
  stable and gives **0.099** — the best of any method here.

---

## 3. Backbone: DA3 vs MapAnything (RGB mode)

`--backend da3` wraps **DA3NESTED-GIANT-LARGE-1.1**, which predicts **metric** depth+poses from RGB
(`is_metric=1`); MapAnything-rgb is scale-ambiguous. `rgb` mode, ma_slam, submap_size 20:

| scene | MapAnything-rgb (Sim3 / SE3 / scale) | **DA3-rgb** (Sim3 / SE3 / scale) |
|---|---|---|
| s0011 | 0.150 / 0.543 / 0.78 | **0.068 / 0.104 / 0.96** |
| s0378 | 0.098 / 0.370 / 0.60 | **0.097 / 0.108 / 0.92** |
| s0231 | 0.140 / 0.389 / 0.84 | **0.093 / 0.104 / 1.02** |

- **Metric SE3-ATE drops 3–5×** (≈0.4–0.5 → ≈0.10) because DA3-rgb is metric (scale ≈1.0).
- **DA3 rgb-only rivals MapAnything's depth modes** (s0231 DA3-rgb 0.093 beats MA `+d+i` 0.099).
- `--backend da3 --mode rgb+intr` ≡ `rgb` (DA3's self-predicted intrinsics already accurate).
- **Perf (submap_size 20):** DA3 **~12.5 fps, ~12 GB VRAM** (DA3-nested is 1.4 B) vs MapAnything
  **~15.5 fps, ~15 GB**. DA3 ingests **no depth** → `rgb`/`rgb+intr` only.
- **Use DA3-rgb when you only have RGB or unreliable depth; use MapAnything `+depth+intr` for clean depth.**

---

## 4. Loop closure — verification, robustness, tuning

ma_slam loop closure = SALAD retrieval → **geometric co-location verification** (re-infer the
candidate pair, accept if `‖cam-center dist‖ / median-depth < coloc_ratio`) → SE3 `BetweenFactorPose3`.

**Does co-location actually discriminate?** Built candidates the realistic way (SALAD sim ≥ 0.5,
temporally far), split by GT distance:

| group | coloc median | pass @0.5 |
|---|---|---|
| true revisits (GT < 0.4 m) | **0.14** | 100 % |
| perceptual aliasing (SALAD-similar, GT > 2.5 m) | **0.89** | 16 % |

→ coloc **does** separate real loops from look-alike aliasing; the model does *not* force similar
places together. But ~**16 %** of aliasing still slips under 0.5 (the residual risk).

**Geo gate on/off (coloc 0.5 was too strict):** turning the gate off *improved* ATE
(s0011 0.102→0.055, s0231 0.097→0.091) — at 0.5 it over-rejected **real** large-viewpoint loops.
→ **`coloc_ratio` default 0.5 → 0.70.**

**Verification window (`half_window`) — tested, dropped (default 0).** On s0231 a window gave
≈0 ATE gain but cost fps (17→11.5 at hw6) and **worsened aliasing FPR 16 %→32 %** (more context lets
the model force look-alikes together). `half_window × coloc_ratio` sweep (s0231):

| hw \ coloc | 0.4 | 0.6 | 0.8 |
|---|---|---|---|
| 0 (17 fps) | 0.120 | 0.095 | **0.091** |
| 3 (14 fps) | 0.117 | 0.093 | 0.091 |
| 6 (11.5 fps) | 0.119 | 0.092 | **0.090** |

`coloc_ratio` is the dominant knob; the window isn't worth it.

**Huber robust kernel on loop factors — free insurance (default on).** Synthetic test (one
deliberately-wrong loop forcing two points 10 m apart together): plain Gaussian warps the whole
trajectory (mean err **3.57**); Huber neutralizes it (**0.17**). On *clean* loops it's exactly
neutral (s0231 identical 0.0985 with/without). So it costs nothing when loops are good and prevents
the slipped-aliasing loops from blowing up the map. Only loop factors are robustified; odometry
stays a plain Gaussian.

---

## 5. `submap_size` sweep (accuracy / fps / VRAM)

`rgb+depth+intr`, ma_slam. ATE = Sim3-ATE (m); loops = accepted.

| submap_size | s0231 ATE / loops | s0011 ATE / loops | VRAM | fps |
|---|---|---|---|---|
| 10 | 0.119 / 23 | 0.168 / 6 | 10.4 GB | 16.1 |
| **20** | **0.097 / 8** | 0.102 / 1 | 14.9 GB | 17.1 |
| 30 | 0.118 / 6 | 0.162 / 0 | 19.5 GB | 17.5 |
| 40 | 0.106 / 6 | **0.096 / 0** | 24.0 GB | 16.9 |
| 60 | 0.180 / 2 | 0.110 / 0 | 33.2 GB | 16.7 |

- **VRAM is ~linear** in submap_size (~0.46 GB/frame); **fps is ~flat** (~16–17.5) — submap_size
  trades VRAM, not speed.
- **More loops ≠ better:** cs10 gets the most loops (23) but mediocre ATE (over-segmentation +
  noisy small loops). **cs60 degrades on loopy s0231** (too few submaps → only 2 loops → big loops
  uncorrected). **cs20 is the robust sweet spot.**

---

## 6. Real-sensor depth (RealSense) — `--depth_max`

**MapAnything keeps the depth you give it — it does *not* denoise.** Measured: on pixels where you
provide depth, output ≈ input (ratio median 1.005, 90 % within ±2 %, ~1.5 cm median diff); only
`depth==0` **holes** get the network's own prediction. So **garbage in → garbage out**.

RealSense depth is unreliable beyond ~5–6 m (raw far values reach 30 m+); MapAnything trusts these
as valid → the cloud sprays to ±60 m. **`--depth_max 6`** zeros depth past 6 m, turning the far
noise into holes the network re-predicts cleanly → recovers a clean reconstruction. Verified on an
in-the-wild RealSense scene: raw unfilled depth → broken; `--depth_max 6` → clean, matching a
hole-filled (range-limited) version of the same scene. **`--backend da3 --mode rgb`** sidesteps depth
entirely → clean metric reconstruction from RGB on the same data, no `--depth_max` tuning needed.

---

## 7. Inter-submap constraint — a negative result

Hypothesis: the single-overlap-frame tie between submaps is the weak link. Tested replacing it with
a robust **dense point-cloud alignment** of the shared overlap frame → **no improvement** (s0011
0.059→0.064, s0231 0.099→0.100). At `overlap=1` a single frame's point-align is no better than
MapAnything's already-consistent predicted pose (overlap geometry already agrees to ~1 cm). So the
residual drift comes from **accumulation across submaps**, not a weak inter-submap estimator —
pointing future work toward a global dense bundle adjustment rather than better adjacent ties. (Kept
as a config option, off by default.)

---

## 8. Pose-graph manifold: SE3 vs Sim3 — a negative result

`ma_slam` exposes `--manifold {se3,sim3,sl4}` (`se3` default; `sim3` places submaps with a scaled
overlap alignment and optimizes a Sim(3) pose graph via the vendored pypose `Sim3LoopOptimizer`;
`sl4` is reserved, not wired). Motivation: MapAnything/DA3 are uncalibrated in RGB mode, so a
scale/projective gauge between submaps could in principle be absorbed by a higher-DOF manifold.

Tested end-to-end (rgb, submap_size 20, Sim3-ATE / SE3-ATE / scale):

| backbone · scene | **SE3** (default) | **Sim3** |
|---|---|---|
| DA3 · s0011 | **0.068 / 0.104 / 0.96** | 0.090 / 0.232 / 0.90 |
| DA3 · s0231 | **0.093 / 0.104 / 1.02** | 0.101 / 0.117 / 1.03 |
| MA · s0011  | **0.150 / 0.543 / 0.78** | 0.174 / 0.639 / 0.75 |
| MA · s0231  | **0.140 / 0.389 / 0.84** | 0.142 / 0.446 / 0.82 |

**SE3 wins or ties everywhere — Sim3 never improves ATE**, and it clearly worsens metric SE3-ATE
(DA3 s0011 0.104→0.232) because the freed inter-submap scale drifts away from 1.0 and compounds
across the chain. A per-pair scale correction can fit the overlap better locally yet still degrade
the *global* trajectory once accumulated. Conclusion: for a metric backbone (DA3 always; MA with
depth) keep `--manifold se3`. The flag is retained for experimentation, but Sim3 is a loss here.

**Update — TUM fr1 (keyframe gate 25 px + `submap_size 32`, distorted, avg over 9 seq).** With the
keyframe config the effect splits by backbone:

| backbone (rgb) | SE3 — Sim3-ATE / SE3-ATE | Sim3 — Sim3-ATE / SE3-ATE |
|---|---|---|
| DA3-rgb | 0.052 / 0.103 | 0.052 / **0.087** |
| MA-rgb  | **0.064 / 0.220** | 0.083 / 0.284 |

**DA3-rgb improves** on metric SE3-ATE (0.103 → 0.087, trajectory shape unchanged); **MA-rgb degrades**
on both. `--manifold se3` stays the default; Sim3 only helps the metric of a reliable RGB backbone (DA3).

> Implementation note: the loop edge fed to `Sim3LoopOptimizer` must be `inv(T_ab)` — its edge
> `(i,j)` constraint is the reverse of gtsam's `BetweenFactor` convention `inv(P_a)@P_b`. Storing it
> un-inverted makes every loop pull the wrong way and explodes the trajectory (Sim3-ATE → ~1.1–1.4).

---

## Appendix — offline pipeline ablations (component analysis)

Per-component sweeps on the offline chunk pipeline (`facebook/map-anything`), cs=20 unless noted.

### A) Input modes (loop closure ON, Sim3-ATE / SE3-ATE / scale)

| scene | rgb | rgb+depth | **rgb+depth+intr** |
|---|---|---|---|
| s0011 | 0.296 / 0.60 / 0.78 | 0.084 / 0.084 / 1.00 | **0.081 / 0.081 / 0.99** |
| s0378 | 0.206 / 0.49 / 0.54 | **0.052 / 0.058 / 0.95** | 0.054 / 0.062 / 0.95 |

Monotonic `rgb < rgb+intr < rgb+depth < rgb+depth+intr`. **Depth is the big jump** (scale → ~1.0,
ATE ~halved); intrinsics add a modest further gain (sometimes marginal over rgb+depth).

### B) `coloc_ratio` (offline pipeline)

`0.6` was the chunk-pipeline sweet spot (s0011 admits its one true loop at ≥0.6: 0.169→0.126); for
**ma_slam** the looser **0.7** + robust kernel is better (§4). Below the gate, the true loop at
coloc ≈ 0.52 is rejected.

### C) Chunk size (offline pipeline)

No universal best: **s0378 prefers small chunks** (more loop benefit), **s0011 prefers large**
(less accumulated drift). Without loop closure, smaller chunks drift more (s0011 rgb:
cs60 0.169 → cs30 0.199 → cs20 0.259 → cs12 0.340 → cs8 0.477), which motivates loop closure and the
online front-ends.

### Why SALAD alone is insufficient

True revisits with large viewpoint change score only ~0.16–0.28 SALAD cos-sim, while visually
similar but *non-co-located* pairs score ~0.68. So **appearance retrieval needs the geometric
co-location gate** (§4) to keep precision — and a geometric candidate proposal to keep recall.

## TUM fr1 — VGGT-SLAM 1.0/2.0 re-run at a 25 px keyframer (2026-08-28) — NOT their official protocol, see audit below

**CORRECTION (audit, same day):** their official protocol is `evals/eval_tum.sh` = raw TUM
folders, `--min_disparity 50` (the paper TEXT says 25 px, the script and the 2.0 paper say 50),
`--submap_size 32 --max_loops 1 --conf_threshold 25`, `evo_ape tum -as`. The table below is a
**25 px** re-run (ours + all baselines), which is a legitimate matched comparison but NOT the paper
setting; the official-protocol reproduction is in the next section. Both baselines were
re-run here (`thirdparty/VGGT-SLAM` @ env `vggt_slam2`, `thirdparty/vggt-slam-1` @
`vggt_slam1`) on the SAME raw distorted frames (`data/tum/fr1_*_dist/rgb`), their own
keyframer at `--min_disparity 25`, `--submap_size 32`, loop closure on, and evaluated with
our `src/eval.py` Sim3-ATE via `scripts/eval_vggt_slam_log.py` (frame-index association).
Keyframe counts match ours within 3% (e.g. 360: 309 vs 300). Outputs: `outputs/vggt_slam_tum/`.

| method | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | **avg** |
|---|---|---|---|---|---|---|---|---|---|---|
| VGGT-SLAM 1.0 Sim(3) [paper] | 0.123 | 0.040 | 0.055 | 0.254 | 0.022 | 0.088 | 0.041 | 0.032 | 0.016 | 0.0746 |
| VGGT-SLAM 1.0 SL(4) [paper] | 0.071 | 0.025 | 0.040 | 0.141 | 0.023 | 0.102 | 0.030 | 0.034 | 0.014 | 0.0533 |
| VGGT-SLAM 2.0 [paper] | 0.050 | 0.025 | 0.029 | 0.102 | 0.026 | 0.063 | 0.026 | 0.038 | 0.014 | 0.0414 |
| VGGT-SLAM 1.0 Sim(3) [re-run, same setting] | 0.054 | 0.023 | 0.029 | 0.034 | 0.049 | 0.060 | 0.026 | 0.042 | 0.015 | 0.0369 |
| VGGT-SLAM 1.0 SL(4) [re-run] | 0.128 | 0.033 | 0.058 | 0.169 | 0.039 | 0.197 | 0.034 | 0.040 | 0.017 | 0.0795 |
| **VGGT-SLAM 2.0 [re-run]** | 0.053 | 0.025 | 0.029 | 0.035 | 0.028 | 0.051 | 0.027 | 0.057 | 0.015 | **0.0355** |
| ours base (DA3-rgb, no inst-LC) | 0.095 | 0.030 | 0.023 | 0.183 | 0.040 | 0.043 | 0.025 | 0.052 | 0.010 | 0.0557 |
| ours +S1 only [a1_pq] | 0.094 | 0.030 | 0.023 | 0.145 | 0.040 | 0.043 | 0.025 | 0.050 | 0.010 | 0.0512 |
| **ours +S1+S2 [a1_pq]** | 0.050 | 0.030 | 0.023 | 0.069 | 0.042 | 0.043 | 0.025 | 0.050 | 0.010 | **0.0380** |

Reading: at a 25 px keyframer VGGT-SLAM 2.0 is 0.0355 (0.041 at their official 50 px) — the
denser keyframer helps everyone (floor 0.102→0.035) except SL(4). 1.0-Sim(3) ≈ 2.0 here for a
different reason (release-code Sim3 == 2.0's scale-only alignment; see audit). Head-to-head,
ours (S1+S2) wins **6/9 scenes vs 2.0** (360/desk2/room/rpy/teddy/xyz) but loses the mean by
0.0025 because of **floor (0.069 vs 0.035; DA3 textureless scale collapse, scale 0.68)** plus
plant/desk. SL(4) is unstable at 25 px (360/floor/room blow up). The earlier "beats 2.0" claim
holds only against the paper number; the honest statement is "on par (6W/3L), floor-limited".

## Our semantic LC plugged into VGGT-SLAM 1.0 (Sim3) — 2026-08-28

Same-process plug-in (`scripts/vggt1_lc/main_gated.py`, plan in `docs/vggt1_lc_plugin_plan.md`):
their untouched pipeline, their keyframer 25 px, submap 32, `--max_loops 1`, Sim3 factor graph.
Only two things change: the SALAD gate (their hardcoded L2 < 0.80) and, in arm C, a map-free
S1 Bi-Desc + S2 layout-graph verdict on the two candidate frames (built from THEIR joint VGGT
pass — no depth input, no instance map, scale-free S2 via `s2_tau_t_rel=0.5`). Sim3-ATE (m):

| scene | A: theirs (L2<0.80) | B: relaxed 0.95, no gate | C: relaxed 0.95 + our S1+S2 |
|---|---|---|---|
| fr1_xyz | 0.0151 (2 loops) | 0.0151 (2) | 0.0151 (2, acc 2/2) |
| fr1_360 | 0.0544 (2) | 0.0877 (6) | **0.0533** (3, acc 3/6) |
| fr1_floor | 0.0337 (1) | 0.3446 (4) | **0.0332** (1, acc 1/4) |
| fr1_teddy | 0.0415 (5) | 0.0404 (6) | **0.0404** (6, acc 6/6) |
| fr1_room | 0.0600 (4) | 0.2216 (8) | **0.0570** (5, acc 5/8) |
| **TUM-5 avg** | 0.0409 | 0.1419 | **0.0398** |
| scene0231_00 | 0.1226 (6) | 0.1269 (9) | 0.1224 (6, acc 6/9) |
| office_loop (no GT) | 1 loop | 5 loops | 1 loop (acc 1/5) |

Reading:
- **Relaxing retrieval alone is catastrophic** (B: floor 10×, room 3.7×, 360 1.6× worse) — VGGT-SLAM 1.0
  has no verifier, so every SALAD false positive becomes a Sim3 factor.
- **Our gate removes every one of those false positives**: all 15 rejections across the 7 scenes have
  L2 ≥ 0.81, i.e. C never rejected a loop their default would have taken (C ⊇ A on loops) — zero
  recall cost on their own operating point, judged by their own trajectories.
- **And it adds recall**: 3 accepted loops at L2 0.85–0.92 are unreachable at their threshold — 360
  sub9↔sub1 (S1 nuniq 4, S2 Ed 0.14), teddy sub6↔sub0 (nuniq 6, Ed 0.18), room sub3↔sub1 (M=3
  rot-abstain) — which is where C beats A (360 −2%, teddy −3%, room −5%; avg 0.0409 → 0.0398).
- Stage 2 fires inside VGGT-SLAM too: floor sub6:2↔sub0:1 passes S1 (a_bi 0.93, nuniq 3) and is
  killed by layout (Ed 0.40, θ_R 79°, |t*|/d_ref 1.05) — a Type-2 catch on a non-metric backbone.
- Cost: 0.2–0.6 s per candidate (one 2-frame semantic forward), a few candidates per scene.
- Caveats: a1_pq model; s0231 shows the gate is neutral where VGGT's own scale drift (2.3×) dominates;
  `--max_loops 1` only (their default); SL(4) mode not covered (env-bound).
Outputs: `outputs/vggt1_lc/<scene>_<A|B|C>/` (poses, run.log, `semlc_log.txt`, `semlc_dump/`,
`semlc_gallery/index.html` for arm C), `outputs/vggt1_lc/summary.csv`.

## Audit: VGGT-SLAM reproduction with THEIR official protocol (2026-08-28)

`scripts/repro_vggt_slam_official.sh`: raw `data/tum/_raw/rgbd_dataset_freiburg1_*/rgb`
(timestamp filenames), `--max_loops 1 --min_disparity 50 --conf_threshold 25 --submap_size 32`,
their envs (`vggt_slam1` for 1.0 SL4/Sim3, `vggt_slam2` for 2.0), `evo_ape tum groundtruth.txt
est -as` (single run; paper averages 5). Outputs `outputs/vggt_slam_official_d50/`.
Also verified: our `src/eval.py` Sim3-ATE == `evo_ape -as` RMSE to 4 decimals on all 27 logs.

| seq | SL(4) paper / official / 25px | Sim(3) paper / official / 25px | 2.0 paper / official / 25px |
|---|---|---|---|
| 360 | 0.071 / 0.074 / 0.128 | 0.123 / **0.047** / 0.054 | 0.050 / 0.050 / 0.053 |
| desk | 0.025 / 0.037 / 0.033 | 0.040 / 0.023 / 0.023 | 0.025 / 0.027 / 0.025 |
| desk2 | 0.040 / 0.037 / 0.058 | 0.055 / 0.031 / 0.029 | 0.029 / 0.029 / 0.029 |
| floor | 0.141 / 0.159 / 0.169 | 0.254 / **0.101** / 0.034 | 0.102 / 0.100 / 0.035 |
| plant | 0.023 / 0.022 / 0.039 | 0.022 / 0.024 / 0.049 | 0.026 / 0.025 / 0.028 |
| room | 0.102 / 0.174 / 0.197 | 0.088 / 0.063 / 0.060 | 0.063 / 0.064 / 0.051 |
| rpy | 0.030 / 0.034 / 0.034 | 0.041 / 0.026 / 0.026 | 0.026 / 0.026 / 0.027 |
| teddy | 0.034 / 0.037 / 0.040 | 0.032 / 0.034 / 0.042 | 0.038 / 0.035 / 0.057 |
| xyz | 0.014 / 0.013 / 0.017 | 0.016 / 0.014 / 0.015 | 0.014 / 0.014 / 0.015 |
| **avg** | 0.0533 / 0.0653 / 0.0795 | 0.0746 / **0.0403** / 0.0369 | 0.0414 / **0.0410** / 0.0355 |

- **2.0 reproduces exactly** (avg 0.0410 vs 0.0414; every scene within 0.002).
- **SL(4) reproduces roughly** (single run; RANSAC randomness; room 0.174 vs 0.102 is the outlier).
- **The released 1.0 code's `--use_sim3` does NOT reproduce the paper's Sim(3) column** — it lands
  at 0.0403 ≈ 2.0 (floor 0.101 vs 2.0's 0.100). Cause (code, not protocol): `solver.py` Sim3 path
  = copy R,t from the overlapping frame's VGGT pose + scalar scale from the median distance ratio,
  i.e. VGGT-SLAM 2.0's "constrained scale-only" alignment, present since the first release commit
  (2025-06-29). The paper's Sim(3) baseline was something else (pre-release). => label our
  baseline "VGGT-SLAM 1.0 release code (--use_sim3)"; it is a STRONGER baseline than the paper's.
- The 25 px keyframer (our earlier "matched" runs) makes the benchmark easier for everything but
  SL(4); it is a fine matched comparison but must not be presented as their setting.

### Loop-level truth of the plug-in arms (GT co-visibility)

`scripts/vggt1_lc/loop_precision.py`: for every SALAD candidate of arms A/B/C, project one
frame's GT depth into the other with GT poses (occlusion-aware) → co-visibility ratio; true if
≥0.20, FALSE if ≈0 (TUM-5 + s0231, 6 scenes with GT).

| | A: their default (L2<0.80, top-1/submap) | B: extra candidates from relaxing to 0.95 | C: our S1+S2 verdict on them |
|---|---|---|---|
| loops | **20, all true** (covis 0.56–1.00) | +15: **7 FALSE** (covis 0.00, view angle 94–179°) + 8 true | 7/7 FALSE rejected; 3/8 true accepted, **5 true rejected** (covis 0.35–0.83, 1.3–1.9 m apart, S1 nuniq 0–1) |

Honest reading: on TUM fr1 / s0231 their conservative retrieval has **zero false positives**, so at
their operating point our verifier is redundant; its only demonstrable value here is enabling
relaxed retrieval at 100% precision (+3 true loops, avg 0.0409→0.0398), with limited recall on
large-viewpoint true loops. Evidence for *needing* instance verification requires scenes where
SALAD top-1@0.80 itself produces false loops (perceptual aliasing: repeated furniture/corridors —
cf. VGGT-SLAM 2.0 Table II, Clio Office). Not available with GT in this repo yet.

## TUM fr1 at VGGT-SLAM's OFFICIAL keyframer (50 px), w32 — ours vs reproduced baselines (2026-08-28)

Ours: `scripts/run_tum_fr1_d50_base.sh` (ma_slam.run, `--keyframe_disparity 50`) and
`scripts/run_tum_fr1_d50_sem.sh` (ma_slam_semantic.run, a1_pq, veto, h2o KV), on
`data/tum/fr1_*_dist` (raw distorted frames; triplet-associated so 0–2% fewer frames than the
raw folders the baselines ran on), evaluated with `src/eval.py` (== evo_ape -as, verified).
Baselines: official protocol reproduction above. Sim3-ATE RMSE (m):

| method | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | avg |
|---|---|---|---|---|---|---|---|---|---|---|
| VGGT-SLAM 1.0 SL(4) [paper] | 0.071 | 0.025 | 0.040 | 0.141 | 0.023 | 0.102 | 0.030 | 0.034 | 0.014 | 0.0533 |
| VGGT-SLAM 1.0 SL(4) [repro] | 0.074 | 0.037 | 0.037 | 0.159 | 0.022 | 0.174 | 0.034 | 0.037 | 0.013 | 0.0653 |
| VGGT-SLAM 1.0 release Sim3 [repro] | 0.047 | 0.023 | 0.031 | 0.101 | 0.024 | 0.063 | 0.026 | 0.034 | 0.014 | 0.0403 |
| VGGT-SLAM 2.0 [paper] | 0.050 | 0.025 | 0.029 | 0.102 | 0.026 | 0.063 | 0.026 | 0.038 | 0.014 | 0.0414 |
| VGGT-SLAM 2.0 [repro] | 0.050 | 0.027 | 0.029 | 0.100 | 0.025 | 0.064 | 0.026 | 0.035 | 0.014 | 0.0410 |
| ours DA3-rgb, no inst-LC | 0.046 | 0.027 | 0.021 | 0.056 | 0.025 | 0.031 | 0.021 | 0.083 | 0.011 | 0.0357 |
| ours DA3-rgb +S1 | 0.046 | 0.027 | 0.021 | 0.065 | 0.025 | 0.031 | 0.021 | 0.051 | 0.011 | **0.0332** |
| ours DA3-rgb +S1+S2 | 0.052 | 0.027 | 0.021 | 0.065 | 0.025 | 0.031 | 0.021 | 0.051 | 0.011 | 0.0338 |
| ours MA-rgb+depth+intr | 0.058 | 0.027 | 0.030 | 0.037 | 0.029 | 0.050 | 0.029 | 0.040 | 0.013 | 0.0348 |

- RGB-only, same keyframer as the paper: **ours without any semantic LC (0.0357) already beats
  VGGT-SLAM 2.0 (0.0410) and the release-code 1.0 Sim3 (0.0403)** — the metric DA3 backbone +
  SE3 graph + coloc-verified LC is the win; floor (0.056 vs their 0.100) is the biggest gap.
- Semantic LC at 50 px changes exactly 3 loop decisions (GT co-visibility checked):
  - teddy f780↔f127: **false loop** (covis 0.00, cameras 168° apart) that coloc (0.68) let
    through → S1/S2 veto it → 0.083→0.051. This is a real catch inside our own pipeline.
  - floor f1090↔f135: **true loop** (covis 0.88) vetoed UNKNOWN (no instances on a floor
    frame) → 0.056→0.065. Recall cost of unknown_fail.
  - 360 f692↔f91: **true loop** (covis 0.81) vetoed by S2 T_FAR → 0.046→0.052. S2 false alarm.
  Net: +S1 0.0332 (best), +S1+S2 0.0338; the S2 stage costs a true loop here and catches nothing
  S1 didn't (the teddy false loop was already M_LOW/S1-visible).
- 25 px (previous section) vs 50 px: our base 0.0557→0.0357, i.e. the denser keyframer HURT our
  base (more low-parallax frames per submap → 360/floor drift) while helping the VGGT baselines;
  so the earlier "S1+S2 fixes 360/floor" gains at 25 px were partly fixing a keyframer-induced
  weakness. The official 50 px table is the one to report.

## Sem-LC gate rules revisited with GT loop truth (2026-08-29) — V6

Tools: `scripts/vggt1_lc/loop_precision.py` (GT co-visibility per candidate) and
`scripts/replay_semlc_rules.py` (offline replay of rule variants over every recorded candidate,
joined with the base run's coloc value). Findings on 59 GT-labelled candidates (25 px 45, 50 px 14):
- every false loop that reached the gate failed S1 already (a_bi <= 0.77, nuniq <= 2); S2's
  catches were M_LOW / theta_R >= 90 deg proxies — Ed/E_R ("same objects, different layout")
  never fired on TUM;
- the old rules killed true loops via UNKNOWN (object-poor frames) and T_FAR (t* inherits the
  rotation residual: noise ~ d_ref*sin(E_R)); S2 metrics vary run-to-run (theta_R 161->90 deg
  for the same pair), so razor-thin gates do not hold.
New rules (flags in `ma_slam_semantic.run`): `--sem_lc_unknown coloc` (abstain -> geometric
verifier), `--s2_t_noise on` (T_FAR gate widened by d_ref*sin(E_R)), `--s2_lvc_first on`
(theta_R > 50 deg hard-rejects before the t/Ed/E_R gates). M_LOW stays hard (abstaining re-admits
floor 463<->79). Replay: 0 false loops added in either regime; true rejections 2->0 (50 px), 8->5 (25 px,
the rest are coloc/M_LOW).

| 50 px | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | avg |
|---|---|---|---|---|---|---|---|---|---|---|
| base DA3-rgb (no inst-LC) | 0.046 | 0.027 | 0.021 | 0.056 | 0.025 | 0.031 | 0.021 | 0.083 | 0.011 | 0.0357 |
| +S1+S2 old rules | 0.052 | 0.027 | 0.021 | 0.065 | 0.025 | 0.031 | 0.021 | 0.051 | 0.011 | 0.0338 |
| **+S1+S2 V6** | 0.046 | 0.027 | 0.021 | 0.056 | 0.025 | 0.031 | 0.021 | 0.051 | 0.011 | **0.0322** |

| 25 px | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | avg |
|---|---|---|---|---|---|---|---|---|---|---|
| base (no inst-LC) | 0.095 | 0.030 | 0.023 | 0.183 | 0.040 | 0.043 | 0.025 | 0.052 | 0.010 | 0.0557 |
| +S1+S2 old rules | 0.050 | 0.030 | 0.023 | 0.069 | 0.046 | 0.043 | 0.025 | 0.049 | 0.010 | 0.0384 |
| **+S1+S2 V6** | 0.049 | 0.030 | 0.023 | 0.069 | 0.042 | 0.043 | 0.025 | 0.050 | 0.010 | **0.0380** |

V6 keeps every catch of the old rules (all 5 coloc-missed false loops) and stops the true-loop
kills (50 px floor/360 back to base; 25 px 360/plant/teddy recover 1/1/2 true loops). Loop counts
25 px: 3/2/3/1/2/4/5/5/2 (old 2/2/3/1/1/4/5/3/2). Outputs `outputs/sem_lc_shadow/a1pq_w32_kf{50,25}_v6/`.
Design analysis (why veto-only + hard gates on noisy scalars is the ceiling; three-state evidence
+ instance relative-pose cross-check as the next design) is in the session notes / plan.
- 25 px, +S1 only under V6 rules: 0.095/0.030/0.023/0.145/0.040/0.043/0.025/0.052/0.010 → **0.0515**
  (loops 4/2/3/2/4/4/5/6/2) — identical to the old-rule S1-only: at 25 px the two false loops
  that survive S1 (360 416<->6, floor 463<->79) are only caught by Stage 2 (theta_R / M_LOW).

**Decision (2026-08-29, user):** V6 is NOT adopted. Rationale: (1) UNKNOWN->coloc hands the
verdict back to the geometric check exactly where instance verification exists to distrust it
(similar-but-different views force-aligned by the backbone); the 4 GT-true UNKNOWN cases are too
few to justify it in aliasing scenes. (2) Gains are noise-level (25 px 0.0384->0.0380, 50 px
0.0338->0.0322, 7-Scenes flat/worse) for three extra flags. The method is the OLD rule set:
"a loop edge needs positive instance evidence — UNKNOWN / M_LOW / T_FAR / LVC reject, S1/S2 FAIL
raises the coloc bar". V6 flags stay opt-in for ablation only. Defaults unchanged.

## 7-Scenes (seq-01 x7, 25 px keyframer, w32) — 2026-08-29

Data: Microsoft 7-Scenes seq-01 (`scripts/download_7scenes.sh`, `prep_scannetpp`-style
`prep_7scenes.py`; Kinect 640x480, K 585/585/320/240, KinectFusion poses as GT). Baselines
re-run on the raw folders with their code (25 px, same as our TUM table); ours on
`data/7scenes/<scene>`. Sim3-ATE RMSE (m):

| method | chess | fire | heads | office | pumpkin | redkitchen | stairs | avg |
|---|---|---|---|---|---|---|---|---|
| VGGT-SLAM 1.0 SL(4) | 0.047 | 0.030 | 0.043 | 0.093 | 0.134 | 0.056 | 0.032 | 0.0621 |
| VGGT-SLAM 1.0 release Sim3 | 0.041 | 0.028 | 0.022 | 0.102 | 0.136 | 0.050 | 0.024 | 0.0574 |
| ours DA3-rgb, no inst-LC | 0.031 | 0.027 | 0.021 | 0.093 | 0.128 | 0.053 | 0.128 | 0.0687 |
| ours +S1 | 0.031 | 0.027 | 0.021 | 0.093 | 0.128 | 0.053 | 0.128 | 0.0687 |
| **ours +S1+S2** (method = old rules) | 0.031 | 0.027 | 0.023 | 0.093 | 0.128 | 0.053 | 0.128 | 0.0690 |
| ours +S1+S2 (V6, ablation) | 0.031 | 0.027 | 0.023 | 0.093 | 0.128 | 0.053 | 0.128 | 0.0690 |

Loops (base / +S1 / +S1+S2): chess 1/1/1; fire 1/1/1; heads 2/2/1; office 2/2/1; pumpkin 1/1/1; redkitchen 1/1/1; stairs 0/0/0.
Reading: 7-Scenes sequences scan one small area (a desk, a corner, a bust) so the instance set
never changes: 8 loops total, **0 false**, nothing for the verifier to catch; S2's only actions are
two removed TRUE loops (heads 741<->205: history frame has 2 instances -> M_LOW; office 752<->92:
E_R 27 deg soft-FAIL, coloc 0.61 > 0.45). Per scene ours base wins 5/7 vs 1.0 but loses the mean
because of **stairs (0.128 vs 0.024–0.032; repeated steps, DA3 relative poses drift, 42 keyframes /
2 submaps / 0 loops)** — same failure family as TUM floor. Verdict (user): supplementary at most;
not a verifier showcase. Outputs `outputs/7scenes_kf25/`.

## New verifier: object-level rigid consistency (`--sem_lc_mode rigid`) — 2026-08-29

One rule instead of five gates: MNN instance pairs with cos >= tau (0.85) -> exhaustive triplet
Kabsch on the matched camera-frame centres -> a pair is an inlier if |R c_r + t - c_q| < 0.15 x
median matched depth -> PASS iff >= 4 inliers (3 points define the pose, the 4th verifies it;
exactly 3 matched pairs form a congruent triangle by chance — that is the 25 px 360 416<->6
false loop, now rejected as "inliers 3 < 4"). Frames with < 4 instances: `--rigid_lowobj reject`
(positive evidence required) or `coloc` (hand to the geometric check). Old bidesc path untouched.

Offline, on 126 GT-labelled candidates (104 true / 18 false): old rules keep 62% true / 0 false;
rigid m=4 keeps 62% / 0 false; m=3 keeps 72% / 1 false (the degenerate triangle). Conditional on
>= 3 instances on both sides: rigid 81% vs old 69%. Remaining losses are upstream: 11 candidates
have < 3 instances on a side (proposal density), 13 have >= 3 instances but < 3 descriptor
matches (descriptor viewpoint invariance), only 5 fail the geometry.

TUM fr1, 25 px, w32 (Sim3-ATE):

| method | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | avg |
|---|---|---|---|---|---|---|---|---|---|---|
| base (no inst-LC) | 0.095 | 0.030 | 0.023 | 0.183 | 0.040 | 0.043 | 0.025 | 0.052 | 0.010 | 0.0557 |
| old rules S1+S2 | 0.050 | 0.030 | 0.023 | 0.069 | 0.046 | 0.043 | 0.025 | 0.049 | 0.010 | 0.0384 |
| rigid, lowobj=reject | 0.050 | 0.030 | 0.023 | 0.113 | 0.046 | 0.043 | 0.025 | 0.050 | 0.010 | 0.0434 |
| rigid, lowobj=coloc | 0.049 | 0.030 | 0.023 | 0.188 | 0.043 | 0.043 | 0.025 | 0.052 | 0.010 | 0.0514 |

Rigid == old rules on 8/9 scenes (avg excl. floor 0.0346 vs 0.0345), zero false loops, and it
catches 416<->6 by construction. floor is the whole difference and it is exactly the low-object
regime: 5 candidates with 1-4 instances per side; reject -> 0 loops (0.113), coloc -> the two
false loops (463<->79 coloc 0.36, 730<->167 coloc 0.67) get in (0.188, worse than base); the old
rules' 0.069 came from S1 FAIL raising the coloc bar to 0.45 (kills 730<->167) plus M_LOW (kills
463<->79) while keeping 1101<->134 — i.e. thin instance evidence backed by strict geometry. Rigid
rejections over TUM: 10 lowobj, 10 pairs<4, 1 inliers<4. The GitHub torch.hub hang (SALAD DINOv2
validation blocking on github.com) was fixed by loading from the local hub cache.

### Rigid verifier at 50 px (official keyframer), TUM fr1

| method | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | avg | loops |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VGGT-SLAM 1.0 SL(4) [official repro] | 0.074 | 0.037 | 0.037 | 0.159 | 0.022 | 0.174 | 0.034 | 0.037 | 0.013 | 0.0653 | |
| VGGT-SLAM 2.0 [official repro] | 0.050 | 0.027 | 0.029 | 0.100 | 0.025 | 0.064 | 0.026 | 0.035 | 0.014 | 0.0410 | |
| ours base (no inst-LC) | 0.046 | 0.027 | 0.021 | 0.056 | 0.025 | 0.031 | 0.021 | 0.083 | 0.011 | 0.0357 | 2/0/1/1/2/2/1/3/0 |
| ours +S1+S2 (old rules) | 0.052 | 0.027 | 0.021 | 0.065 | 0.025 | 0.031 | 0.021 | 0.051 | 0.011 | 0.0338 | 1/0/1/0/2/2/1/2/0 |
| ours + rigid (lowobj=reject) | 0.055 | 0.027 | 0.021 | 0.065 | 0.027 | 0.031 | 0.021 | 0.047 | 0.011 | 0.0338 | 0/0/1/0/1/2/1/0/0 |
| ours + rigid (lowobj=coloc) | 0.055 | 0.027 | 0.021 | 0.056 | 0.027 | 0.031 | 0.021 | 0.047 | 0.011 | **0.0328** | 0/0/1/1/1/2/1/0/0 |

At 50 px the rigid verifier is stricter than at 25 px: 10/14 candidates rejected (5 pairs<4,
3 inliers<4, 2 lowobj) — it drops the teddy false loop (0.083->0.047, best of all) but also both
360 true loops (4 matched pairs, 3 inliers: centre noise at d_ref ~2 m exceeds eps 0.15*d_ref) and
plant's second loop. Net avg 0.0338 (= old rules); with lowobj->coloc 0.0328 (best row) because
floor's true low-object loop comes back and teddy's false one still fails the inlier test.

### Upstream fixes for the semantic branch (2026-08-29): proposal survival + trailing chunk

Funnel diagnosis (`MA_PROP_STATS=1`): proposals that survive objectness + NMS lose ~50% at pixel
ownership because a proposal may only own geometry-valid pixels (`conf > per-submap mean conf`,
the ma_slam point-cloud mask reused by the bridge). Fix `--proposal_conf prefer`: ownership on all
finite pixels, 3D footprint from valid pixels when any (a proposal always keeps >= 1 point); the
SLAM point cloud is untouched. Also `split_chunk` folds a 1-frame trailing chunk into the previous
one (33 = 4x7 + 5).

| final proposals / frame | TUM room (84 kf) | ScanNet++ 13c3e046d7 (295 kf) |
|---|---|---|
| gate (shipped) | med 10, 10% frames with 0, 15% <= 2 | med 4, 2% with 0, 22% <= 2, 69% >= 4 |
| gate + tail-merge | med 10 (same) | med 4 (same); frame 32 mean 3.1 vs 4.8 others |
| **prefer + tail-merge** | **med 16 = NMS count, 0% with 0** | **med 9 = NMS count, 0% with 0, 98% >= 4** |

The trailing-chunk merge alone changes nothing measurable (the submap's last frame stays a bit
weaker even in a 5-frame chunk); the ownership mask was the loss. Map-quality (instance AP) impact
of `prefer` still to be validated on the SceneNN golden set.

### `prefer` must stay out of the map (2026-08-29)

ScanNet-5 instance AP with OUR SLAM geometry (DA3 rgb, submap 20, h2o KV, a1_pq), gate vs prefer
as the MAP proposal rule:

| scene | gate AP/AP50/AP25 | prefer AP/AP50/AP25 | dAP |
|---|---|---|---|
| scene0011 | 24.7/53.1/80.6 | 17.1/49.3/70.5 | -7.7 |
| scene0050 | 21.6/52.0/69.5 | 22.9/51.0/78.0 | +1.3 |
| scene0231 | 10.3/32.7/59.4 | 6.1/22.1/66.0 | -4.2 |
| scene0378 | 14.1/32.3/56.3 | 12.4/30.7/50.2 | -1.7 |
| scene0518 | 12.5/27.4/59.3 | 13.3/34.1/58.4 | +0.8 |
| mean | 16.7/39.5/65.0 | 14.4/37.4/64.6 | **-2.3** |

The rescued low-confidence proposals become poorly placed instances (AP/AP50 drop, AP25 flat).
Decision (user): `prefer` proposals are only the CURRENT frame's transient loop-verification
evidence — `--lc_proposal_conf prefer` builds a per-frame bag {descriptor, camera-frame centre}
from the same model output and stores it in FrameIndex; the InstanceTable / map keeps
`--proposal_conf gate`. History side of the rigid check = map instances of the history frame.

TUM 25 px, rigid (lowobj=reject) with prefer applied to BOTH map and LC (the earlier `_rigid_up`
run): 360 0.050 (2) · desk 0.030 (2) · desk2 0.023 (3) · floor 0.069 (1) · plant 0.042 (2) ·
room 0.045 (3) · rpy 0.025 (5) · teddy 0.053 (4) · xyz 0.010 (2) → **0.0386** (old rules 0.0384,
rigid without fixes 0.0434); lowobj rejections 10 -> 0, floor's five candidates all judged on
evidence (true 1101<->134: 6 pairs / 6 inliers PASS; the two coloc-passing false loops: 1 and 0
matched pairs). The transient-bag variant (`_rigid_lcbag`) is the one to report; running.

### Rigid verifier with transient evidence bags on both sides (`--lc_proposal_conf prefer --lc_history bag`, map=gate) — TUM 25 px

360 0.050 (2) · desk 0.030 (2) · desk2 0.023 (3) · floor **0.069** (1) · plant 0.042 (2) · room 0.045 (3) ·
rpy 0.025 (5) · teddy 0.053 (4) · xyz 0.010 (2) → **avg 0.0387** (shipped S1+S2 0.0384; rigid without
bags 0.0434). The map is untouched (bags never enter the InstanceTable); floor's true loop comes back
(6 pairs / 6 inliers) and both coloc-passing false loops are still rejected on pair count. Query-side-only
bag (`--lc_history map`) does NOT recover floor (history capped at 4 gate instances -> 3 pairs): 0.0433.
GT-labelled rejection breakdown (no-fix rigid): 50 px — 4 false loops all die at pair count (<4), rigid
step removes 2 TRUE loops (4 pairs / 3 inliers); 25 px — 9 false die at pair count, rigid removes the single
false loop with 4 pairs (416<->6, 3 inliers) and no true one. B-S1 (pair count only, `--rigid_geom off`)
queued at 50/25 px.

## Clio apartment (multi-room, RealSense D455, COLMAP-metric GT) — 2026-08-29

Data: `scripts/prep_clio.py` (COLMAP sparse rescaled to metres with the depth pngs: 0.637 m/unit;
1056 frames, 4.6 x 6.1 m). Keyframer 50 px, w32 -> 236 keyframes / 7-8 submaps.

**Retrieval audit** (`scripts/retrieval_recall.py`, 16,005 cross-submap keyframe pairs, GT co-visibility):
542 TRUE revisit pairs exist, but their SALAD cosine is low (median 0.29, p90 0.41, max 0.60) and overlaps
the false pairs (p99 0.45). Above our default 0.50: 12 true / 16 false pairs; above 0.60: none. VGGT-SLAM's
0.80 gate (cos 0.68) finds 0 loops; 0.95 (cos 0.55) finds 3. => on Clio the bottleneck is SALAD recall; a
verifier only matters once retrieval is relaxed.

**Default retrieval (cos 0.50, 1 candidate/submap):** 1.0 SL4 0.260 (0 loops) · 1.0 Sim3 0.080 (0) ·
plug-in relax-only (L2<0.95) 0.061 (3, all true) · plug-in relax + our map-free gate 0.087 (1; killed 2 true) ·
ours base 0.065 (2 true) · +S1 0.066 (1) · +S1+S2 0.066 (0) · rigid 0.066 (1).

**Relaxed retrieval (cos 0.40, 5 candidates/submap; `--sim_threshold 0.40 --max_per_submap 5`)** — 27 candidates,
13 true / 14 false (GT):

| verifier | accepted true | accepted FALSE | ATE |
|---|---|---|---|
| coloc only (base) | 7 | **1** (989<->635, coloc 0.69) | **0.434** (trajectory destroyed) |
| shipped S1+S2 | 3 | 0 | 0.066 |
| rigid (transient bags) | 2 | 0 | 0.066 |
| VGGT-SLAM 1.0 relax-only (L2<1.10, 1/submap) | 3 | **2** | 0.850 |

The false candidates are genuine perceptual aliasing (SALAD 0.40-0.53, indistinguishable from the true ones;
one passes coloc); instance verification rejects all 14 on evidence (S1 FAIL / 0-2 matched pairs). Recall is
the cost: 10-11 of 13 true loops are also rejected (few matched pairs), but ATE is unaffected here because
the scene is small. This is the Table-II-style evidence: relaxed retrieval + geometry alone breaks; relaxed
retrieval + instance verification stays clean.
| VGGT-SLAM 1.0 relax (L2<1.10) + our map-free S1+S2 (pC) | 1 | 0 | 0.085 |

pC detail: the two false loops that broke relax-only (828<->583, 998<->599; SALAD L2 0.98-0.99, 2.5-2.6 m
apart) fail S1 outright (a_bi 0.34-0.36, 0 matched pairs — 13-16 instances on each side, none in common);
two true loops are lost to S2 T_FAR (1.03 m / 1.25 m apart, M=9, Ed 0.10, E_R 15 deg) — the known T_FAR
false alarm; the surviving loop (467<->321) is true. Note: their SALAD (serizba hub) also blocks on GitHub
validation; `scripts/vggt1_lc/run_v1_nohub.py` disables torch.hub validation for their code paths.

## Clio office (17 m open office, the scene where VGGT-SLAM 2.0 reports 1.0 producing false positives)

Default retrieval (50 px, 303 kf / 10 submaps): 1.0 SL4 **7.67** (3 loops, 1 FALSE 4.1 m apart) · 1.0 Sim3
**3.03** (same 3 loops) · ours base **0.100** (2 true loops, coloc rejects the 4 false candidates) · ours
+S1 / +S1+S2 / rigid 0.113 (0 loops: the two true loops are killed — no instance record (0-2 gate
instances on those frames) and, with bags, best descriptor match only 0.37-0.40).
Retrieval audit: 1218 true revisit pairs, SALAD true median 0.24 / max 0.75, false p99 0.52 / max 0.70;
submaps 3 and 4 have a FALSE top-1 (sim 0.70 / 0.66, covis 0) — textbook aliasing.

**VGGT-SLAM 1.0 plug-in arms (their pipeline, their loops):**

| arm | loops | true / FALSE | ATE |
|---|---|---|---|
| A: their default (L2<0.80) | 3 | 2 / 1 | 3.03 |
| B: relax to 0.95, no gate | 5 | 2 / **3** | 2.82 |
| C: relax to 0.95 + our map-free S1+S2 | 2 | **2 / 0** | 1.39 |

Arm C is perfect on office: both true loops pass (S1 0.97/0.98, nuniq 10/7, S2 PASS), all three false ones
fail S1 (a_bi 0.40-0.57, 0 matched pairs). ATE stays large because 1.0's own drift on office is metres
(SE3 3.7-4.6 m); loop closure cannot repair VGGT's scale collapse there.

Key observation: the SAME true loops that our own pipeline's verifier rejects (descriptor match 0.37-0.40)
are matched at 0.97-0.98 by the map-free plug-in verifier, which encodes the two frames JOINTLY in one
2-frame semantic forward. The a1_pq instance descriptor is context-dependent (chunk attention / KV):
descriptors of the same object embedded in different chunks, minutes apart, do not match; embedded
together they do. => verification-time joint re-embedding of the candidate pair is the recall fix.

## Clio cubicle (default retrieval, 50 px, 156 kf / 5 submaps)

1.0 SL4 0.083 (1 loop) · 1.0 Sim3 0.030 (1) · plug-in relax-only 0.027 (2: 1 true + 1 FALSE, 1.7 m / 97 deg) ·
plug-in relax + our gate 0.030 (1 true, false rejected) · **ours base 0.165** (2 loops: 1 true + **1 FALSE**
426<->180 with coloc 0.47!) · ours +S1 0.048 (false removed) · +S1+S2 0.039 (0 loops) · rigid 0.048 (1 true).
Third Clio scene in a row where coloc alone admits an aliasing loop and the instance check removes it.

## Joint verification (`--sem_lc_mode joint`) on TUM fr1 50 px, w32 (2026-08-29)

Verify on coloc's joint re-inference window: embed query+match frames TOGETHER (fresh KV), transient
`prefer` bags, rigid verdict (tau 0.85, eps 0.15 d_ref, m=4). Map instances untouched.

| 50 px | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | avg |
|---|---|---|---|---|---|---|---|---|---|---|
| base DA3-rgb (no inst-LC) | 0.046 | 0.027 | 0.021 | 0.056 | 0.025 | 0.031 | 0.021 | 0.083 | 0.011 | 0.0357 |
| +S1 | 0.046 | 0.027 | 0.021 | 0.065 | 0.025 | 0.031 | 0.021 | 0.051 | 0.011 | 0.0332 |
| A: +S1+S2 (old rules) | 0.052 | 0.027 | 0.021 | 0.065 | 0.025 | 0.031 | 0.021 | 0.051 | 0.011 | 0.0338 |
| B: rigid (bags) | 0.055 | 0.027 | 0.021 | 0.065 | 0.027 | 0.031 | 0.021 | 0.047 | 0.011 | 0.0338 |
| B lowobj->coloc | 0.055 | 0.027 | 0.021 | 0.056 | 0.027 | 0.031 | 0.021 | 0.047 | 0.011 | 0.0328 |
| **joint (rigid on joint embedding)** | 0.046 | 0.027 | 0.021 | 0.056 | 0.025 | 0.031 | 0.021 | 0.051 | 0.011 | **0.0321** |

GT-labelled verdicts (co-visibility): joint **11/11 true PASS** (6–14 MNN pairs, 4–13 inliers; the same
loops had 0–6 pairs against map/bag history), **1/1 false REJECT** (teddy 780<->127, coloc 0.68 — coloc
alone had accepted it, hence base teddy 0.083; 0 pairs jointly). First variant with no true-loop kill
and no false pass on this set; floor/360 back to base geometry, teddy false loop removed.
`outputs/sem_lc_shadow/a1pq_w32_kf50_joint/`.

## Clio office / cubicle, RELAXED retrieval (SALAD cos 0.40, 5/submap; 1.0 L2 1.10), 50 px

| scene | 1.0 relax no-gate | 1.0 relax + our gate | ours base | +S1+S2 | rigid |
|---|---|---|---|---|---|
| office  | 4.29 (8: 3 true 3 FALSE 2 weak) | 1.26 (2 true, 0 false; 6 rej of which 1 true) | 0.106 (9 true, 0 false; coloc rejects 19 false) | 0.113 (0; kills 3 true) | 0.106 (2 true) |
| cubicle | 0.030 (3: 1 true 1 FALSE 1 weak) | 0.031 (1 true, false rejected) | **0.164 (6 true + 2 FALSE**: 426<->180 coloc 0.47, 457<->180 coloc 0.53) | 0.047 (1 true; 2 true killed) | 0.051 (1 true; 1 true killed) |

Office: coloc suffices (all 19 false candidates coloc 0.73–8.6); the instance gate is only a recall
cost there because office descriptors across submaps match at 0.37–0.40 (joint mode pending).
Cubicle: coloc admits 2 aliasing loops -> 0.164; every instance variant removes both (0.047/0.051).
Same pattern as apartment relaxed (base 0.434 with 1 false; S1+S2/rigid 0.066).

### Joint mode on Clio relaxed (apartment / office)

| scene | base | +S1+S2 | rigid | **joint** |
|---|---|---|---|---|
| apartment | 0.434 (7 true + 1 FALSE) | 0.066 (3 true) | 0.066 (2 true) | **0.066 (5 true, 0 false)** — false 989<->635 (coloc 0.69) rejected with 0 joint pairs; 2 true rejected at the m=4 boundary (4 pairs/3 inliers, 3 pairs) |
| office | 0.106 (9 true) | 0.113 (0) | 0.106 (2 true) | **0.107 (8 true, 0 false; 1 true rejected at 4 pairs/1 inlier)** — joint pairs 7–14 where the map/bag history gave 0–2 |

# FINAL instance-LC tables (2026-08-29, generated by `scripts/final_lc_tables.py`)

Rows: base = DA3-rgb ma_slam, SALAD + coloc only · +S1 = Bi-Desc only · A = S1+S2 old layout rules (shipped) ·
B = rigid (MNN pairs cos>=0.85 -> >=4 pairs -> triplet-Kabsch rigid layout check, >=4 inliers; lowobj reject) ·
B-S1 = B without the layout check (pairs>=4 only) · B lowobj->coloc = B but frames with <4 objects fall back to
coloc · joint = B on a joint re-embedding (appendix only). "loops" = accepted loops per sequence labelled by GT
co-visibility: true[+weak][+FALSE]. All instance gates run BEFORE coloc (a PASS still has to pass coloc<=0.70).

### TUM fr1, 50 px keyframer, w32 (Sim3-ATE; loops = accepted true[+weak][+FALSE])

| method | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | avg | loops |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base (no inst-LC) | 0.046 | 0.027 | 0.021 | 0.056 | 0.025 | 0.031 | 0.021 | 0.083 | 0.011 | 0.0357 | 2/0/1/1/2/2/1/1+**1F**/0 |
| +S1 | 0.046 | 0.027 | 0.021 | 0.065 | 0.025 | 0.031 | 0.021 | 0.051 | 0.011 | 0.0332 | 2/0/1/0/2/2/1/1/0 |
| A: S1+S2 old rules | 0.052 | 0.027 | 0.021 | 0.065 | 0.025 | 0.031 | 0.021 | 0.051 | 0.011 | 0.0338 | 1/0/1/0/2/2/1/1/0 |
| B: rigid | 0.055 | 0.027 | 0.021 | 0.065 | 0.027 | 0.031 | 0.021 | 0.047 | 0.011 | 0.0338 | 0/0/1/0/1/2/1/0/0 |
| B-S1: pairs only | 0.050 | 0.027 | 0.021 | 0.065 | 0.027 | 0.031 | 0.021 | 0.051 | 0.011 | 0.0338 | 1/0/1/0/1/2/1/1/0 |
| B lowobj->coloc | 0.055 | 0.027 | 0.021 | 0.056 | 0.027 | 0.031 | 0.021 | 0.047 | 0.011 | 0.0328 | 0/0/1/1/1/2/1/0/0 |
| joint | 0.046 | 0.027 | 0.021 | 0.056 | 0.025 | 0.031 | 0.021 | 0.051 | 0.011 | 0.0321 | 2/0/1/1/2/2/1/1/0 |

### TUM fr1, 25 px keyframer, w32 (Sim3-ATE; loops = accepted true[+weak][+FALSE])

| method | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | avg | loops |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base (no inst-LC) | 0.095 | 0.030 | 0.023 | 0.183 | 0.040 | 0.043 | 0.025 | 0.052 | 0.010 | 0.0557 | 3+**1F**/2/3/1+**2F**/2+2w/4/5/6/2 |
| +S1 | 0.094 | 0.030 | 0.023 | 0.145 | 0.043 | 0.043 | 0.025 | 0.050 | 0.010 | 0.0515 | 2+**1F**/2/3/1+**1F**/2+1w/4/5/5/2 |
| A: S1+S2 old rules | 0.050 | 0.030 | 0.023 | 0.069 | 0.046 | 0.043 | 0.025 | 0.049 | 0.010 | 0.0384 | 2/2/3/1/1/4/5/3/2 |
| B: rigid | 0.050 | 0.030 | 0.023 | 0.113 | 0.046 | 0.043 | 0.025 | 0.050 | 0.010 | 0.0434 | 2/2/3/0/1/4/5/4/2 |
| B-S1: pairs only | 0.092 | 0.030 | 0.023 | 0.113 | 0.046 | 0.043 | 0.025 | 0.050 | 0.010 | 0.0481 | 1+**1F**/2/3/0/1/4/5/4/2 |
| B lowobj->coloc | 0.049 | 0.030 | 0.023 | 0.188 | 0.043 | 0.043 | 0.025 | 0.052 | 0.010 | 0.0514 | 3/2/3/1+**2F**/2+1w/4/5/6/2 |

### Clio, 50 px, default retrieval

| method | apartment | office | cubicle |
|---|---|---|---|
| 1.0 SL(4) | 0.260 (0) | 7.671 (3) | 0.083 (1) |
| 1.0 Sim3 | 0.080 (0) | 3.030 (3) | 0.029 (1) |
| 1.0 Sim3 plug-in L2 0.80 | 0.080 (0) | 3.030 (3) | 0.029 (1) |
| 1.0 relaxed, no gate | 0.061 (3) | 2.822 (5) | 0.027 (2) |
| 1.0 relaxed + our gate | 0.087 (1) | 1.390 (2) | 0.030 (1) |
| ours base | 0.065 (2) | 0.100 (2) | 0.165 (1+**1F**) |
| +S1 | 0.066 (1) | 0.113 (0) | 0.048 (1) |
| A: S1+S2 | 0.066 (0) | 0.113 (0) | 0.039 (0) |
| B: rigid | 0.066 (1) | 0.113 (0) | 0.048 (1) |
| B-S1: pairs only | 0.066 (1) | 0.113 (0) | 0.048 (1) |
| B lowobj->coloc | 0.066 (1) | 0.114 (0) | 0.048 (1) |
| joint | — | — | 0.048 (1) |

### Clio, 50 px, RELAXED retrieval (cos 0.40, 5/submap; 1.0 L2 1.10)

| method | apartment | office | cubicle |
|---|---|---|---|
| 1.0 relaxed, no gate | 0.850 (5) | 4.293 (8) | 0.029 (3) |
| 1.0 relaxed + our gate | 0.085 (1) | 1.259 (2) | 0.030 (1) |
| ours base | 0.434 (7+**1F**) | 0.106 (9) | 0.164 (6+**2F**) |
| +S1 | 0.066 (5) | 0.113 (0) | 0.051 (6) |
| A: S1+S2 | 0.066 (3) | 0.113 (0) | 0.047 (1) |
| B: rigid | 0.066 (2) | 0.106 (2) | 0.051 (1) |
| B-S1: pairs only | 0.066 (4) | 0.106 (3) | 0.050 (2) |
| B lowobj->coloc | 0.066 (2) | 0.107 (2) | 0.051 (1) |
| joint | 0.066 (5) | 0.107 (8) | 0.051 (6) |

### Where the layout (rigid) stage actually decides — every candidate with >=4 matched pairs that it rejected

| run | candidate | pairs/inliers | GT | did coloc also reject it? |
|---|---|---|---|---|
| TUM 25 px 360, B | 416<->6 | 4/3 | **FALSE** | no (coloc passed; B-S1 admits it -> 360 0.092 vs 0.050) |
| TUM 50 px 360 / teddy, B | 692<->91, 1312<->86 | 4/3 | true | — (cost: 0.050->0.055, 0.051->0.047) |
| Clio office relaxed, B | 1385<->1029, 1428<->930 (+1394/1388<->1040) | 4-5 / 0-1 | **FALSE** ×4 | yes (coloc 4.42 / **0.73**, i.e. one within 0.03 of the bar) |
| Clio office relaxed, B | 344<->45, 1116<->460 | 5/3, 4/1 | true / weak | — |
| Clio apartment relaxed, B | 338<->183, 346<->183 | 4/3, 5/3 | true | — |
| Clio cubicle relaxed, B | 558<->24 | 4/3 | true | — |
| Clio cubicle, joint | 426<->180 | 5/3 | **FALSE** | no (coloc 0.47) |

Reading: 112 candidates reached the layout stage over all runs. Pair count is the main filter (every TUM 50 px false
loop and the Clio aliasing loops at default retrieval die there); the layout check is an independent safety net for
"same object classes, wrong arrangement" aliasing that pair count cannot see — decisive on TUM 25 px (0.0481 -> 0.0434),
redundant with coloc on Clio (both reject), and its cost is the m=4-boundary true loops (4 pairs / 3 inliers).
Instance gates admitted **0 false loops in any run**; base admitted 1 (TUM 50 teddy), 3 (TUM 25), 1 / 1+2 (Clio cubicle
default / relaxed), 1 (apartment relaxed) — every one removed by S1 / A / B.

## TUM fr1 50 px with RELAXED retrieval (SALAD cos 0.40, 5 candidates/submap — the Clio setting) (2026-08-30)



- base: coloc admits **8 false loops** (360 ×2, room ×4, teddy ×2), avg 0.0357 -> 0.0538; room 0.031 -> 0.189.
  Two room false loops have coloc **0.15 / 0.20** — geometrically indistinguishable from a true revisit.
- **A (old rules) leaks 2 false loops in room (0.149)**: S1 said FAIL (a_bi 0.29 / 0.17, 0 pairs) but the old
  rules treat FAIL as "raise the coloc bar to 0.45", and 0.15 / 0.20 clear that bar. Structural hole: instance
  disagreement handed back to geometry fails exactly when the aliasing geometry is perfect. avg 0.0440.
- **B (rigid) admits 0 false loops**, keeps 38 true, avg **0.0316** (best row on TUM 50 px, below default B 0.0338):
  relaxed retrieval recovers true loops (plant 1->9, teddy 0->8) while the gate removes every aliasing candidate.
  All 8 base false loops died at the pair-count stage (0-1 matched pairs); the layout stage fired only on 4 true
  loops at the m=4 boundary (4-5 pairs / 2-3 inliers: 360 ×2, floor, teddy) -> B-S1 (pairs only) 0.0300, also 0 false.
- Conclusion for the design choice: B's "positive evidence required" is what makes relaxed retrieval safe; A's
  fallback-to-geometry is not. Layout stage on TUM: safety net with a small recall cost; decisive only at 25 px.

## Ablation ladder: who decides? (TUM fr1 50 px; 2026-08-30)

SALAD only = every retrieved candidate becomes a loop edge (`ma_slam.run --coloc_ratio 1e9`) · base = + coloc gate
(joint re-inference co-location <= 0.70) · S1 = instance-correspondence count (MNN pairs cos>=0.85, >=4, hard reject)
· S2 = rigid layout check (Kabsch, >=4 inliers). "NO coloc" rows (`--coloc_ratio 1e9` in ma_slam_semantic.run) still
run the joint re-inference (the loop edge needs it) but never reject on it. B-S1 = S1 + coloc; B = S1 + S2 + coloc.

### default retrieval (SALAD cos 0.50, 1/submap)

| method | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | avg | loops |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SALAD only (no coloc, no inst) | 0.103 | 0.027 | 0.021 | 0.055 | 0.025 | 0.175 | 0.021 | 0.083 | 0.011 | 0.0580 | 2+**2F**/0/1/1/2/2+**1F**/1/1+**1F**/0 |
| base (no inst-LC) | 0.046 | 0.027 | 0.021 | 0.056 | 0.025 | 0.031 | 0.021 | 0.083 | 0.011 | 0.0357 | 2/0/1/1/2/2/1/1+**1F**/0 |
| B: rigid | 0.055 | 0.027 | 0.021 | 0.065 | 0.027 | 0.031 | 0.021 | 0.047 | 0.011 | 0.0338 | 0/0/1/0/1/2/1/0/0 |
| B-S1: pairs only | 0.050 | 0.027 | 0.021 | 0.065 | 0.027 | 0.031 | 0.021 | 0.051 | 0.011 | 0.0338 | 1/0/1/0/1/2/1/1/0 |
| S1 only, NO coloc | 0.050 | 0.027 | 0.021 | 0.055 | 0.025 | 0.034 | 0.021 | 0.051 | 0.011 | 0.0329 | 1/0/1/1/2/1/1/1/0 |
| S1+S2, NO coloc | 0.050 | 0.027 | 0.021 | 0.064 | 0.025 | 0.034 | 0.021 | 0.051 | 0.011 | 0.0339 | 1/0/1/0/2/1/1/1/0 |

### relaxed retrieval (cos 0.40, 5/submap)

| method | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | avg | loops |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SALAD only (no coloc, no inst) | 0.175 | 0.027 | 0.021 | 0.051 | 0.026 | 0.627 | 0.020 | 0.047 | 0.011 | 0.1117 | 10+**10F**/0/5/5/9/10+**10F**/5/9+3w+**2F**/0 |
| base (no inst-LC) | 0.069 | 0.027 | 0.021 | 0.051 | 0.026 | 0.189 | 0.020 | 0.070 | 0.011 | 0.0538 | 10+**2F**/0/5/5/9/10+**4F**/5/9+**2F**/0 |
| B: rigid | 0.047 | 0.027 | 0.021 | 0.065 | 0.026 | 0.033 | 0.020 | 0.035 | 0.011 | 0.0316 | 5/0/5/0/9/6/5/8/0 |
| B-S1: pairs only | 0.043 | 0.027 | 0.021 | 0.055 | 0.026 | 0.033 | 0.020 | 0.033 | 0.011 | 0.0300 | 7/0/5/1/9/6/5/9/0 |
| S1 only, NO coloc | 0.043 | 0.027 | 0.021 | 0.056 | 0.026 | 0.033 | 0.020 | 0.033 | 0.011 | 0.0299 | 7/0/5/1/9/6/5/9/0 |
| S1+S2, NO coloc | 0.047 | 0.027 | 0.021 | 0.065 | 0.026 | 0.033 | 0.020 | 0.035 | 0.011 | 0.0316 | 5/0/5/0/9/6/5/8/0 |

Ladder (default / relaxed avg): SALAD only 0.0580 / 0.1117 (4 / 22 false loops) -> +coloc 0.0357 / 0.0538 (1 / 8 false)
-> S1 only 0.0329 / 0.0299 (0 false) -> S1+S2 0.0339 / 0.0316 (0) -> S1+coloc 0.0338 / 0.0300 (0) -> S1+S2+coloc
0.0338 / 0.0316 (0). Geometric co-location halves the damage but cannot close it (room relaxed 0.627 -> 0.189, still 4
aliasing loops with coloc 0.15-0.62); the instance gate alone closes it (room 0.033) and everything stacked on top of
S1 is neutral or a small recall cost. S2 never removes a false loop on TUM 50 px (those die at the pair count).

## VGGT-SLAM 1.0 **SL(4)** + our S1 gate via RPC (2026-08-30) — TUM fr1, official protocol (raw folders, 50 px, evo -as)

`scripts/vggt1_lc/{verify_server.py (amb3r_bw), main_gated_rpc.py (their env), run_sl4_gate.sh}`; gate = final S1
(MNN pairs cos>=0.85, >=4, hard reject) on the two candidate frames with THEIR VGGT geometry; their code untouched.
Same-session runs (VGGT-SLAM 1.0 has large run-to-run variance: identical loop sets give e.g. desk2 0.043-0.107,
floor 0.106-0.202 — compare loop-level truth, not single-run ATE).

| method | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | avg | loops |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SL4, their 0.80 | 0.067 | 0.025 | 0.107 | 0.106 | 0.022 | 0.169 | 0.026 | 0.040 | 0.014 | 0.0639 | 1/0/2/1/2/2/1/2/0 (all true) |
| SL4, 0.80 + S1 | 0.052 | 0.026 | 0.043 | 0.154 | 0.023 | 0.132 | 0.028 | 0.042 | 0.017 | 0.0573 | same (11/11 ACCEPT) |
| SL4, relaxed 0.95 | 0.083 | 0.023 | 0.045 | 0.167 | 0.025 | **0.276** | 0.042 | 0.046 | 0.019 | 0.0807 | +**4 FALSE** (360 ×3 @164-179 deg, room ×1 @179 deg) |
| SL4, 0.95 + S1 | 0.080 | 0.031 | 0.036 | 0.202 | 0.023 | 0.188 | 0.027 | 0.047 | 0.028 | 0.0734 | 4/4 FALSE rejected (0-1 pairs), 11/11 true kept |

Reading: at their operating point TUM retrieval has no false loops, so the gate is neutral (both rows have identical
loop sets; the avg difference is run variance). Relaxing retrieval on TUM adds only false loops (their max_loops=1 /
top-1-per-submap means no extra true loops appear), the gate removes all of them; ATE is then bounded by their
variance. The gate transfers cleanly (15/15 verdicts correct) but TUM gives it nothing to gain — Clio SL4 run next.

## VGGT-SLAM 1.0 **SL(4)** + our S1 gate via RPC — Clio (50 px, `scripts/vggt1_lc/run_sl4_gate_clio.sh`, 2026-08-30)

| SL(4) | apartment | office | cubicle |
|---|---|---|---|
| their 0.80 | 0.188 (0 loops) | 7.43 (3: 2 true + 1 FALSE 4 m) | 0.054 (1 true) |
| 0.80 + S1 | 0.277 (0 candidates) | 3.78 (2 true kept, FALSE rejected) | 0.043 (1 true) |
| relaxed L2 1.10, no gate | 0.248 (5: 3 true + 2 FALSE) | 6.62 (8: 3 true + 5 false/weak) | 0.060 (3: 1 true + 1 weak + 1 FALSE) |
| **relaxed 1.10 + S1** | **0.121 (3 true, 0 false)** | 5.38 (3 true, 0 false) | 0.089 (1 true, 0 false) |

Gate verdicts on their SL(4) pipeline: 10/10 accepted loops true, 10/10 rejected loops false/weak (0 pairs each) —
20/20 correct, as on TUM (15/15). Apartment is the showcase: their 0.80 finds no loop; relaxing finds 3 true but
also 2 false (0.248 ~ no better than no loops); with the gate the 3 true loops are pure gain, 0.188 -> 0.121.
Office is degenerate for SL(4) (scale 0.19-2.4, 3-7 m whatever the loops) and cubicle's 0.043 vs 0.089 with the
identical single true loop is their run variance. Combined with TUM: the gate transfers to a foreign pipeline
without touching its code, is never wrong on 35 candidates, and turns relaxed retrieval from harmful into useful
wherever the relaxed candidates include true loops (apartment, Clio-Sim3 office earlier).

## Clio in the paper framing (no coloc anywhere) — S1 alone has a failure mode in repetitive offices (2026-08-30)

| Clio, 50 px | apartment | office | cubicle |
|---|---|---|---|
| default: SALAD only | 0.261 (5) | 1.39 (6) | 0.148 (3) |
| default: + S1 | 0.066 (1 true) | 0.113 (0) | 0.048 (1 true) |
| relaxed: SALAD only | 0.423 (27) | 3.52 (40) | 0.545 (15) |
| relaxed: + S1 | 0.066 (4 true) | **1.32 (3 true + 3 FALSE)** | 0.050 (3) |
| relaxed: + S1 + S2 (rigid layout) | (running) | **0.106 (2 true, 0 false)** | (running) |

Office relaxed: three false candidates (1385<->1029, 1435<->988, 1428<->930 — identical cubicles) have 5 MNN pairs
each, so the pair count alone passes them (ATE 1.32); the rigid layout check rejects all of them (inliers 0-2) and
keeps 2 true loops (0.106). This is the "same object classes, different arrangement" aliasing that S2 was designed
for; it never occurs on TUM (no repeated furniture). Consequence for the method: S1 needs S2 in repetitive scenes.

## Semantic / instance eval with PREDICTED depth + PREDICTED pose (DA3-rgb SLAM + final inst-LC) — setup (2026-08-30)

`scripts/prep_sem_eval_data.py` (frame sets = OpenSe3r GT-geometry loaders) + `scripts/run_sem_eval.sh <dataset>`
(SLAM: submap 20, sem chunk 4, keyframe 25 px [TUM geometry benchmark uses 50 px — different protocol, on purpose],
conf_gate_coef 1.0, ranker p_top25q, H2O KV, inst-LC S1 relaxed + coloc; export aligned to GT by SE3-Umeyama on
camera centres, scale fixed 1). Eval flags verbatim from OpenSe3r's _eval_*.bash. 3RScan: frames rotated upright,
semantic model at 336x518 (`--resolution auto`), as the GT-geo runs.

Smoke (one scene each; GT-geo per-scene AP/AP50/AP25 in brackets):
| scene | ours pred-pose | GT-geo | align RMSE | Sim3 scale | map->GT NN median / frac<0.1 m |
|---|---|---|---|---|---|
| ScanNet scene0221_01 | 23.4 / 44.8 / 69.7 | 42.5 / 76.6 / 86.9 | 0.157 | 0.956 | 0.031 / 0.95 |
| SceneNN 015 | 6.0 / 21.6 / 61.6 | 27.3 / 39.6 / 57.5 (pred-depth) | 0.152 | 0.916 | 0.099 / 0.51 |
| 3RScan fcf66d7b | 2.2 / 5.8 / 46.6 | 19.5 / 44.5 / 71.2 | 0.052 | 1.056 | 0.060 / 0.84 |
| ScanNet++ 09c1414f1b | 0.6 / 2.8 / 14.5 (top100 0.8 / 3.4 / 16.5) | (avg 27.7 / 51.2 / 69.0) | 0.120 | 1.042 | 0.057 / 0.73 |

Diagnostics: an orientation-aware alignment (`--align pose`, R from GT camera axes) differs from the centre-only one by
<1 deg on 3RScan (cam-rot err 1.0 deg) -> the AP gap is not an alignment tilt. Maps sit on the GT geometry
(NN median 3-10 cm) so the loss is at IoU>=0.5 granularity: predicted geometry noise + coverage (GT->map frac<0.25 m
0.66-0.77) rather than a frame/scale bug. ScanNet++ is the outlier (48 loops on the smoke scene; check LC there).

### ScanNet val-50 — pred depth + pred pose (DA3-rgb SLAM + inst-LC), 2026-08-30

| protocol | AP / AP50 / AP25 | GT depth + GT pose (a1_pq, OpenSe3r `scannet_val50_a1ep28pq`) |
|---|---|---|
| full (stuff filter) | **19.89 / 42.06 / 66.81** | 41.55 / 66.96 / 82.66 |
| no-stuff | 19.93 / 42.37 / 67.98 | 42.05 / 67.94 / 83.95 |

Trajectory decomposition: align RMSE median 0.095 m; 4/50 scenes with RMSE > 0.25 m (0025_02, 0221_00, 0665_01,
0488_00-type; cam-rot err 10-19 deg, 1-2 loops) average AP50 13.6 vs 44.5 on the other 46 (corr(RMSE, AP50) = -0.39).
`outputs/ma_sem/scannet_val50_s1lc/SUMMARY.csv` has per-scene ATE / align RMSE / loops / n_inst next to the APs.

# SEMANTIC / INSTANCE EVAL with PREDICTED depth + PREDICTED pose (DA3-rgb ma_slam + final inst-LC) — 2026-08-30

Protocol A ("original"): SE3-Umeyama align (scale 1), 25 px keyframer, submap 20, sem chunk 4, H2O KV,
`--conf_gate_coef 1.0`, inst-LC = S1 gate (rigid pairs>=4, relaxed SALAD 0.40/5) + coloc 0.70. Frame sets = the
OpenSe3r GT-geometry loaders (`scripts/prep_sem_eval_data.py`); eval flags verbatim from OpenSe3r's `_eval_*.bash`.
Runs: `outputs/ma_sem/<dataset>_s1lc/` (launched 11:21–11:24 by a child session of this one from
scratchpad/sem_chain{A,B}.sh; harness `scripts/run_sem_eval.sh`). Protocol B (conf 0.5, paired sim3map supp
export) is running into `outputs/ma_sem/<dataset>_final/`.

| dataset | scenes | AP / AP50 / AP25 (ours, pred depth + pred pose) | GT-geo reference (GT depth + GT pose) | align RMSE | loops/scene |
|---|---|---|---|---|---|
| ScanNet val-50 (full, stuff filter) | 50 | 19.89 / 42.06 / 66.81 | 41.55 / 66.96 / 82.66 | 0.148 m | 9.5 |
| ScanNet val-50 (no-stuff) | 50 | 19.93 / 42.37 / 67.98 | 42.05 / 67.94 / 83.95 | | |
| SceneNN-12 | 12 | 16.20 / 34.47 / 58.91 | 32.35 / 53.71 / 68.30 | 0.112 m | 3.3 |
| 3RScan-47 | 47 | 9.89 / 20.35 / 52.82 | 25.82 / 47.73 / 69.67 | 0.192 m | 8.7 |
| ScanNet++ 50 (all objects) | 49 (1 running) | 7.20 / 18.94 / 36.25 | 19.44 / 37.95 / 55.85 | 78.8 m* | 42.6 |
| ScanNet++ 50 (top100) | 49 | 11.96 / 29.40 / 49.79 | 27.67 / 51.17 / 69.00 | | |
| ScanNet++ excl. 2 diverged scenes (align > 1 m) | 47 | 7.50 / 19.73 / 37.74 · top100 12.47 / 30.63 / 51.86 | | 0.09 m | |

| semantic (OVO instsem) | mIoU / mAcc | GT-geo |
|---|---|---|
| ScanNet-5 SN20 (4/5 scenes; 0518 OOM, rerun pending) | 37.15 / 54.99 | 44.26 / 62.99 |
| ScanNet-5 SN200 | 15.95 / 21.64 | 17.79 / 26.17 |
| Replica-8 | 11.20 / 17.35 | 29.26 / 40.16 |

*ScanNet++ d755b3d9d8 diverged (ATE 3856 m, 41 accepted loops; graph blow-up), 3db0a1c8f3 2.5 m. Replica's DA3
metric scale is off by 2–15 % per scene (traj Sim3 scale 1.02–1.15) which, under scale-1 alignment, misplaces the
instsem voxels — the paired sim3map export in protocol B is the diagnostic for this.
Findings from the per-dataset smoke/diagnostics (one scene each): geometry is fine (map->GT-mesh median 4–6 cm,
instance voxels on the mesh); the AP gap to GT-geo is (i) DA3 conf gating removing map coverage (mesh->map p90
0.2–0.7 m; `conf_gate_coef 0.5` recovers +6–14 AP50, hence protocol B), (ii) per-scene metric-scale bias 2–10 %
(kills AP50, not AP25), (iii) trajectory residual 0.05–0.2 m. A single global scale fitted to the GT surface
(`--align sim3map`) helps ScanNet/ScanNet++ but hurts SceneNN/3RScan on the same map -> kept as supp only.

## Protocol B (FINAL) semantic/instance results — SE3 align + conf_gate_coef 0.5 + 25 px keyframer (2026-08-31)

Runs: `outputs/ma_sem/<dataset>_final/` (ScanNet-5: `scannet5_final`). Same SLAM as protocol A, only
`--conf_gate_coef 0.5` (reporting knob: the per-pixel validity gate is halved, recovering map coverage) and the
paired `align_sim3map/` secondary export (one global scale fitted to the GT surface — supp only).
NOTE keyframer: semantic runs use **25 px** (TUM geometry benchmark uses 50 px, VGGT-SLAM protocol).

### Instance (class-agnostic AP / AP50 / AP25)

| dataset | A: conf 1.0 | **B: conf 0.5 (final)** | B + sim3map (supp) | GT-geo (GT depth+pose) |
|---|---|---|---|---|
| ScanNet val-50 (stuff filter) | 19.89 / 42.06 / 66.81 | **27.12 / 51.86 / 72.68** | 29.42 / 54.18 / 74.61 | 41.55 / 66.96 / 82.66 |
| ScanNet val-50 (no-stuff) | 19.93 / 42.37 / 67.98 | **27.48 / 52.70 / 74.07** | — | 42.05 / 67.94 / 83.95 |
| SceneNN-12 | 16.20 / 34.47 / 58.91 | **22.72 / 44.00 / 62.98** | 24.88 / 45.83 / 61.98 | 32.35 / 53.71 / 68.30 |
| 3RScan-47 | 9.89 / 20.35 / 52.82 | **12.21 / 26.84 / 52.91** | 14.92 / 30.55 / 56.07 | 25.82 / 47.73 / 69.67 |
| ScanNet++ 50, all (excl. 2 diverged) | 7.68 / 20.13 / 37.95 | **13.43 / 29.58 / 49.41** | 14.54 / 31.56 / 51.60 | 19.44 / 37.95 / 55.85 |
| ScanNet++ 50, top100 (excl. 2 diverged) | 12.55 / 30.87 / 51.79 | **20.64 / 43.44 / 64.68** | — | 27.67 / 51.17 / 69.00 |

(ScanNet++ B was 46/50 scenes at the time of writing; 3db0a1c8f3 and d755b3d9d8 diverge in SLAM — align RMSE
2.5 m / 3856 m — and are excluded above, with-them means are ~1 AP50 lower.)

### Semantic (OVO-style instsem, mIoU / mAcc)

| benchmark | A: conf 1.0 | **B: conf 0.5 (final)** | B + sim3map (supp) | GT-geo |
|---|---|---|---|---|
| ScanNet-5 SN20 | 37.15 / 54.99 (4 scenes) | **37.34 / 56.84** (5) | — | 44.26 / 62.99 |
| ScanNet-5 SN200 | 15.95 / 21.64 (4) | **14.47 / 21.29** (5) | — | 17.79 / 26.17 |
| Replica-8 | 11.20 / 17.35 | **14.70 / 21.16** | 19.02 / 27.12 | 29.26 / 40.16 |

### Run statistics (protocol B)

| dataset | keyframes / frames | loops per scene | SE3 align residual | sim3map global scale |
|---|---|---|---|---|
| ScanNet val-50 | 135 / 168 | 12.2 | 0.142 m | 1.00 |
| SceneNN-12 | 147 / 154 | 5.2 | 0.104 m | 0.96 |
| 3RScan-47 | 257 / 291 | 13.1 | 0.184 m | 1.01 (per-scene \|s-1\| mean 7 %) |
| ScanNet++ 50 | 433 / 438 | 54.2 | 0.085 m (excl. diverged) | 1.00 |
| Replica-8 | 95 / 100 | 5.1 | 0.101 m | 1.02–1.15 |

Reading: conf 0.5 is worth +6 to +13 AP50 everywhere; ours reaches 77 % of the GT-geometry AP50 on ScanNet val-50,
82 % on SceneNN, 78 % on ScanNet++ (top100 85 %), 56 % on 3RScan (the dataset where DA3 geometry is weakest:
7 % scale bias, 7 cm surface error, 0.18 m trajectory residual). AP25 is consistently closer to GT-geo than AP50 —
instances are found, their boundaries are a few cm off, which is the price of predicted depth.

### Replica protocol verification vs the GT-geometry rows (2026-08-31)

Checked because the Replica gap (14.7 vs 29.26 mIoU) is much larger than ScanNet's (37.3 vs 44.3). The protocol is
**identical** — same script (`tools/ovo_style_semantic_eval.py`), same flags (`--benches replica --replica_gt
/mnt/HDD4/ricky/data/replica/_gt --backbone tips-v2-l14 --tasks instsem`), same packed GT, same fixed 51-class
vocabulary for the argmax (all 8 scenes share it; per-scene GT only contains 16–28), same 5-NN-mode pred->GT
transfer with no distance limit, same `voxel_size` 0.05, same `<sc>_instance_semantic_feats.npz` contract.
Re-ran their dirs myself: `replica8_a1ep28pq` = **29.26 / 40.16** (GT depth+pose, the reference row) and
`replica8_a1ep28pq_pd` = 29.41 / 40.72 (predicted depth + GT pose) -> **predicted depth alone costs nothing**.

The gap is geometric, not semantic. Fraction of GT surface within 0.10 m of a predicted voxel:

| | office0 | office1 | office2 | office3 | office4 | room0 | room1 | room2 | mean |
|---|---|---|---|---|---|---|---|---|---|
| GT-geo | 79.6 | 80.7 | 80.7 | 89.6 | 88.5 | 91.1 | 89.6 | 83.3 | **85.4 %** |
| ours (SE3, 25 px) | 39.0 | 22.4 | 52.0 | 80.1 | 57.9 | 85.4 | 31.7 | 44.1 | **51.6 %** |
| ours + sim3map | 73.2 | 22.2 | 56.5 | 78.1 | 81.4 | 86.5 | 83.8 | 77.1 | **69.9 %** |
| per-scene DA3 scale | 1.15 | 1.15 | 1.11 | 1.02 | 1.06 | 1.04 | 1.12 | 1.07 | |

Coverage tracks the per-scene metric-scale error exactly (office3 scale 1.02 -> 80 % covered, mIoU close to GT-geo;
office1/room1 scale 1.12–1.15 -> 22–32 %). Our maps have MORE voxels than theirs; they simply land 10–40 cm off, so
OVO's distance-free 5-NN transfer hands GT vertices the neighbouring surface's label. Restricting the metric to
covered regions (`--max_nn_dist 0.10`): GT-geo 31.90, ours+sim3map 26.28, ours SE3 19.06/21.55 (25/50 px) — i.e. the
semantics are close once the geometry is comparable. Replica is the worst dataset for DA3's metric scale (2–15 %,
vs ~3–4 % on ScanNet), which is why its gap is the largest of the six.

### Replica keyframe-density ablation (protocol B)

| Replica-8 instsem (51 classes) | mIoU / mAcc | f-mIoU / f-mAcc | kf/100 | loops | align RMSE |
|---|---|---|---|---|---|
| 25 px (as reported) | 14.70 / 21.16 | 43.02 / 55.36 | 95 | 7.6 | 0.096 m |
| 50 px | 15.87 / 23.33 | 42.35 / 54.93 | 74 | 4.4 | 0.100 m |
| 25 px + sim3map | 19.02 / 27.12 | 48.56 / 59.76 | | | |
| 50 px + sim3map | 19.83 / 29.67 | 47.46 / 59.19 | | | |

Halving the keyframe density costs nothing on Replica (+1.2 mIoU, within noise); the reported rows stay 25 px for
consistency with the other datasets. `outputs/ma_sem/replica_final_kf50/`.

#### Where the remaining Replica gap actually is (common-mask decomposition)

Evaluating GT-geo and our runs on **exactly the same GT vertices** (those covered within 0.10 m by all three runs
= 44.2 % of GT, so coverage is equalised by construction):

| Replica-8 instsem | full GT (protocol) | covered by that run (<=0.10 m) | **common mask (identical vertices)** |
|---|---|---|---|
| GT-geo (GT depth + GT pose) | 29.26 | 31.90 (86 % of GT) | **34.74** |
| ours + sim3map, 50 px | 19.83 | 26.28 (70 %) | **28.82** |
| ours SE3, 50 px | 15.87 | 21.55 (48 %) | **22.80** |

So the 9.4 mIoU that survives the global-scale correction splits into
**~5.9 perception** (28.82 vs 34.74 on identical, well-covered surface — predicted depth gives noisier surfaces and
different instance splits) and **~3.5 coverage** (our map still misses ~30 % of the GT surface within 10 cm vs their
15 %: a single global scale cannot fix the residual 0.1 m trajectory error, per-submap depth-scale variation, the
conf gate, or office1's genuine tracking failure — Sim3-ATE 0.143, trajectory scale 1.42 vs map scale 1.15).
Note the common-mask perception gap (5.9) matches the ScanNet-5 SN20 gap (37.34 vs 44.26 = 6.9): once coverage is
equalised, the cost of predicted depth+pose is a consistent ~6-7 mIoU on both datasets, and Replica only looks
worse because its metric-scale bias (2-15 %) costs it coverage on top.

### Replica: frame-set protocol (stride-20 vs the full 2000-frame sequence)

The GT-geometry rows use `frame_stride 20` (100 of 2000 frames) — an offline-model convention; OVO-SLAM/ORB-SLAM
run the full sequences. Converted the full sequences (`scripts/prep_sem_eval_data.py replica --stride 1
--out_name replica_full`) and re-ran protocol B on them (`outputs/ma_sem/replica_full_final/`, 25 px gate ->
264 keyframes/scene on average, 32 loops/scene).

| Replica-8 | frames used | keyframes | map voxels | GT-surface coverage (<=0.10 m) | instsem mIoU / mAcc |
|---|---|---|---|---|---|
| GT-geo (GT depth+pose) | 100 (stride 20) | 100 | 50.7 k | 85.4 % | 29.26 / 40.16 |
| ours, stride 20, SE3 | 100 | 95 | 79.0 k | 51.6 % | 14.70 / 21.16 |
| ours, stride 20, +sim3map | 100 | 95 | 79.0 k | 71.9 % | 19.02 / 27.12 |
| ours, all frames (KF 0) | 100 | 100 | — | 47.5 % | 13.89 / 20.35 |
| **ours, full 2000, SE3** | 2000 | 264 | 109.9 k | **61.1 %** | **15.40 / 22.30** |
| **ours, full 2000, +sim3map** | 2000 | 264 | 109.9 k | 72.8 % | 18.27 / 26.82 |

Running the full sequence buys +9.5 points of coverage (51.6 -> 61.1 %) but only +0.7 mIoU; the metric-scale
correction is still worth 3–4x more (+20 points of coverage, +3 to +4 mIoU). Mean per-scene scale 1.07 (vs 1.09 at
stride 20), Sim3-ATE 0.052 m, SE3-ATE 0.101 m — i.e. more frames slightly improve the trajectory but do not remove
the depth backbone's metric bias, which is what the coverage (and therefore the mIoU) is limited by.

#### Replica frame-set: stride-20 (the GT-geometry convention) vs the FULL 2000-frame sequences

The GT-geometry rows use `frame_stride 20` = 100 frames/scene (`office0_mapping_config.json`), while OVO-SLAM /
ORB-SLAM run the full 2000-frame video. Ours re-run on the full sequences (`data/replica_full/`, protocol B,
25 px gate -> 264 keyframes/scene): `outputs/ma_sem/replica_full_final/`.

| Replica-8 | frames used | keyframes | instsem mIoU / mAcc | + sim3map | GT-surface coverage (<=0.1 m) |
|---|---|---|---|---|---|
| GT-geo (GT depth+pose) | 100 (stride 20) | 100 | 29.26 / 40.16 | — | 85.4 % |
| ours, stride 20 | 100 | 95 | 14.70 / 21.16 | 19.02 / 27.12 | 51.6 % |
| ours, 50 px gate | 100 | 74 | 15.87 / 23.33 | 19.83 / 29.67 | 46.3 % |
| ours, every frame (KF 0) | 100 | 100 | 13.89 / 20.35 | 19.79 / 28.44 | 47.5 % |
| **ours, FULL sequence** | **2000** | **264** | **15.40 / 22.30** | **18.27 / 26.82** | **61.1 %** (sim3map 72.8 %) |

The full sequence adds ~10 points of coverage (51.6 -> 61.1 %) and 32 loop closures/scene (vs 7.6), but mIoU moves
by +0.7 only: the limiting factor is the metric-scale bias (scale 1.07-1.09), which costs ~25 points of coverage on
its own (61.1 -> 72.8 % when corrected). Denser sampling inside the 100-frame set does nothing (KF 0 = 47.5 %).

## FINAL semantic/instance tables — Sim(3)-aligned (matches the IGGT baseline's protocol), 2026-08-31

Alignment audit: our `umeyama_sim3` (src/eval.py) is numerically identical to IGGT's (max |dR| = 0, |dt| ~1e-16,
same scale). The difference was only which correspondences are fed to it — IGGT uses dense per-pixel
pred<->GT-depth-backprojection (7 DoF); we use camera centres for R,t plus ONE global scale fitted to the GT surface
(`--align sim3map`, the paired export written by every protocol-B run). Both are "pred geometry, Sim(3)-aligned to
GT for scoring" — declare that in the caption; the SE3 (scale=1) column is the stricter, no-scale variant.

### Instance (class-agnostic AP / AP50 / AP25)

| dataset | SE3 (scale = 1) | **Sim(3)** | GT-geo (GT depth+pose) | Sim3 / GT-geo AP50 |
|---|---|---|---|---|
| ScanNet val-50 (full) | 27.12 / 51.86 / 72.68 | **29.42 / 54.18 / 74.61** | 41.55 / 66.96 / 82.66 | 81 % |
| ScanNet val-50 (no-stuff) | 27.48 / 52.70 / 74.07 | **29.78 / 55.01 / 75.88** | 42.05 / 67.94 / 83.95 | 81 % |
| SceneNN-12 | 22.72 / 44.00 / 62.98 | **24.88 / 45.83 / 61.98** | 32.35 / 53.71 / 68.30 | 85 % |
| 3RScan-47 | 12.21 / 26.84 / 52.91 | **14.92 / 30.55 / 56.07** | 25.82 / 47.73 / 69.67 | 64 % |
| ScanNet++ 50, all (excl. 2 diverged) | 13.55 / 29.71 / 49.79 | **15.14 / 32.23 / 52.11** | 19.44 / 37.95 / 55.85 | 85 % |
| ScanNet++ 50, top100 (excl. 2 diverged) | 20.72 / 43.37 / 64.52 | **22.99 / 46.58 / 67.36** | 27.67 / 51.17 / 69.00 | 91 % |

### Semantic (instsem mIoU / mAcc)

| benchmark | SE3 | **Sim(3)** | GT-geo | Sim3 / GT-geo |
|---|---|---|---|---|
| ScanNet-5 SN20 | 37.49 / 56.85 | **38.26 / 57.65** | 44.26 / 62.99 | 86 % |
| ScanNet-5 SN200 | 14.91 / 22.07 | **15.14 / 22.31** | 17.79 / 26.17 | 85 % |
| Replica-8 (stride 20, GT-geo frame set) | 14.70 / 21.16 | **19.02 / 27.12** | 29.26 / 40.16 | 65 % |
| Replica-8 (full 2000-frame sequences) | 15.40 / 22.30 | **18.27 / 26.82** | — | — |
| Replica-8 f-mIoU (voxel-weighted, full seq.) | 45.51 | **50.05** | 55.44 | **90 %** |

Sim(3) is worth +0.8 mIoU on ScanNet (scale error 3-4 %) but +4.3 on Replica (2-15 %) — the size of the correction
tracks the backbone's per-scene metric bias, as expected. `outputs/ma_sem/scannet5_final_s3/` (both alignments).

### Why Replica's unweighted mIoU still trails by 10 (per-class analysis, common mask)

On identical GT vertices, split by object size (39 of the 51 classes occupy < 1 % of voxels each):

| | classes | GT-geo IoU | ours IoU | delta |
|---|---|---|---|---|
| large (>= 1 % of voxels: wall/floor/ceiling/table/chair/bed/rug/...) | 12 | 64.5 | 59.0 | −5.5 |
| small (< 1 %: monitor, vase, bin, plant-stand, desk-organizer, ...) | 39 | 23.3 | 11.8 | −11.5 |
| all 51 (the reported mIoU) | 51 | 33.0 | 22.9 | −10.1 |
| voxel-weighted (f-mIoU) | | 63.9 | 59.8 | −4.1 |

Worst per class (GT-geo -> ours): plant-stand 85.5 -> 0.0, monitor 98.7 -> 18.9, indoor-plant 59.7 -> 4.9,
bin 88.5 -> 35.7, shelf 53.0 -> 15.3, vase 28.2 -> 0.0. Large flat classes are equal or better (rug, window,
tv-screen). A 40 cm object at 2-3 m displaced by a 5-10 % depth error loses essentially all its IoU while a wall
does not, and an unweighted mean over 51 classes charges ~2 points per lost class. Report both mIoU and f-mIoU.

## E4 — does inst-LC heal the instance MAP, not just the trajectory? (2026-09-03)

Brief: `docs/e4_lc_map_healing.md`. Only the **LC-OFF** arm is new (`--no_loop`); everything else is byte-identical
to protocol B (`a1_pq`, `conf_gate_coef 0.5`, 25 px keyframer, SE3 align). The ATE evidence in this file was not
re-run. Duplicate instances = for each GT instance, the number of predicted clusters whose best match is that GT
with IoU >= 0.25, summed as `sum(max(0, k-1))`; both arms are scored by the SAME matcher, read from the
evaluator's own `--save_json` `pred_details` (`scripts/e4_dup_count.py`).
Runs: `outputs/ma_sem/{scannetpp,scannet_val50}_noLC/` vs `..._final/`.

| metric (mean per scene) | ScanNet++ 48* — LC ON | LC OFF | delta | val-50 — LC ON | LC OFF | delta |
|---|---|---|---|---|---|---|
| loops accepted | 54.25 | 0 | +54.25 | 12.18 | 0 | +12.18 |
| Sim3-ATE (m) | 0.063 | 0.114 | **−0.051** | 0.112 | 0.124 | −0.011 |
| SE3-ATE = align RMSE (m) | 0.089 | 0.133 | **−0.044** | 0.142 | 0.154 | −0.012 |
| AP | 13.55 | 10.40 | **+3.15** | 27.12 | 25.91 | +1.21 |
| AP50 | 29.71 | 24.17 | **+5.54** | 51.86 | 51.17 | +0.69 |
| AP25 | 49.79 | 44.71 | **+5.08** | 72.68 | 73.25 | −0.57 |
| n_pred | 94.1 | 92.0 | +2.1 | 39.6 | 39.4 | +0.2 |
| GT instances matched (IoU>=0.25) | 47.0 | 42.0 | **+4.9** | 17.96 | 18.18 | −0.22 |
| **duplicate instances** | **1.31** | **1.35** | **−0.04** | **0.84** | **0.60** | +0.24 |

*the two diverged ScanNet++ scenes (3db0a1c8f3, d755b3d9d8) are excluded, as everywhere else in this file.
Per-scene duplicates: ScanNet++ LC-ON lower in 17/48, higher in 12, tied in 19 (totals 63 vs 65); val-50 lower in
2/50, higher in 11, tied in 37 (42 vs 30).

**Answer: the duplicate-instance mechanism is a measured null; the map gain is real but comes from geometry.**
Duplicates are essentially absent in both arms (~1.3 per scene on ScanNet++ against 47 matched GT instances, ~0.8
on val-50 against 18) and loop closure does not move them — 54 loop closures per scene change the mean by 0.04.
The identity descriptor re-associates an object across a 0.13 m drift *without* the loop closure, so there is
nothing for LC to merge. What LC does buy on ScanNet++ is +5.5 AP50, and the mechanism is visible in the same
table: align RMSE 0.133 -> 0.089 m and **+4.9 more GT instances matched at all** — i.e. instances that were
displaced out of the IoU window, not split in two. On val-50 (12 loops/scene, drift already 0.14 m) the effect is
within noise (+0.7 AP50).

Per docs/e4_lc_map_healing.md §2 this is the outcome that supports the paper's central claim: identity is carried
by the descriptor, not by the trajectory. The `trajectory -> map` arrow should be stated as "loop closure improves
the map by re-registering displaced instances (+5.5 AP50 where loops are frequent)", NOT as "it removes duplicates".
