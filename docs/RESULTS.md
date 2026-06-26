# ma_long / ma_slam — Results & Findings

Long-sequence reconstruction on the **MapAnything** backbone (with an optional **DA3** backbone).
Three front-ends share one chunk model + one eval: the **offline chunk pipeline** (`ma_long`), the
AMB3R-style **fused online SLAM** (`ma_long/slam.py`), and **`ma_slam`** — the VGGT-SLAM-2.0-style
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

## TUM RGB-D (freiburg1) — head-to-head vs VGGT-SLAM & others

Most direct external comparison. ma_slam (LK keyframe gate 25 px + `submap_size 32`, SE3) on the 9
`freiburg1` sequences. Baselines from the VGGT-SLAM 2.0 paper Table I (Sim3-aligned ATE RMSE, m). Rows
split **Uncalibrated (RGB-only)** vs **Calibrated** exactly as that table does — our `+intr` /
`+depth+intr` configs feed intrinsics (and depth), i.e. **more input than the RGB-only methods**, so
they sit in the Calibrated group and are *not* a like-for-like comparison with VGGT-SLAM.

| method | 360 | desk | desk2 | floor | plant | room | rpy | teddy | xyz | **avg** |
|---|---|---|---|---|---|---|---|---|---|---|
| *— Uncalibrated (RGB-only) —* | | | | | | | | | | |
| MASt3R-SLAM\* | 0.070 | 0.035 | 0.055 | 0.056 | 0.035 | 0.118 | 0.041 | 0.114 | 0.020 | 0.060 |
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
  (SE3 ≈ Sim3, scale ≈ 1.0). Among *calibrated* methods it's competitive (ties DROID 0.038; behind MASt3R-SLAM 0.030).
- **Depth — not intrinsics — anchors scale:** `+intr` alone improves trajectory shape (Sim3 0.051) but
  *worsens* the metric (SE3 0.305, scale 0.6–0.9); only depth fixes it (SE3 0.042). Judge these with SE3.
- **Weak spot = `360`** (pure rotation → no triangulation baseline) across all configs — an inherent backbone limit.

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
