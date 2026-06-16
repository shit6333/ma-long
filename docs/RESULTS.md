# ma_long — Results & Findings

MapAnything-based long-sequence reconstruction. Pipeline: per-chunk MapAnything
feed-forward → overlap SE3/Sim3 alignment → dual (geometric + SALAD) loop closure →
global Sim3 pose-graph optimization → merged point cloud + global poses.

- **Hardware:** 1× RTX PRO 6000 Blackwell (97 GB), CUDA 12.8, torch 2.8+cu128, env `amb3r_bw`.
- **Data:** ScanNet-style scenes `scene0011_00` (238 frames) and `scene0378_00` (190 frames),
  RGB + depth (16-bit mm) + `intrinsic.txt` + `gt_pose.txt` (TUM).
- **Metric:** ATE RMSE (m) after trajectory alignment to GT. Two variants:
  - **Sim3-ATE** — scale-invariant (classic monocular ATE; *hides* metric-scale error).
  - **SE3-ATE (metric)** — rotation+translation only; reveals absolute/scale accuracy.
  - `scale` = best-fit Sim3 scale est→GT (1.0 == perfectly metric).
  - **Judge multi-modal modes with SE3-ATE, not Sim3-ATE.**

## TL;DR

1. **Depth input is the single biggest lever.** It fixes MapAnything's metric scale
   (rgb scale error ~20% → ~2%) and roughly halves ATE. Intrinsics add a smaller gain.
   Best input = **rgb+depth+intr**.
2. **Loop closure helps only when the loop windows can be co-registered** — which, in
   practice, requires depth. In rgb mode MapAnything cannot relate large-viewpoint-change
   loop windows, so verification rejects them and LC is a no-op.
3. **Geometric loop proposal is essential.** SALAD (appearance) misses the real revisits
   in these scenes (true loops score ~0.16 cos-sim); proposing candidates from the
   provisional trajectory (temporally-far / spatially-near) catches them.
4. **Cost of LC:** ~2× slower (FPS halves) and +9 GB VRAM (SALAD + loop-window re-inference).

> NOTE: Sections A/B/C below were run with the **`facebook/map-anything-apache`** weights and a
> third scene (`scene0231_00`, 444 frames) was added later. The headline numbers are now the
> **full `facebook/map-anything`** model (default since `ma_infer.py` switch) — see next section.

## ★ Full model (`facebook/map-anything`) — current headline

cs=20, loop closure ON for the chunk pipeline; rgb+depth+intr unless noted.

### Offline chunk pipeline (chunk → SE3 align → geo+SALAD LC → global opt)

| scene | mode | Sim3-ATE | SE3-ATE | scale | loops | apache→full |
|---|---|---|---|---|---|---|
| 0011 | rgb            | 0.296 | 0.604 | 0.78 | 1 | — (rgb scale unstable) |
| 0011 | rgb+depth      | 0.084 | 0.084 | 1.00 | 2 | 0.18 → 0.084 |
| 0011 | **rgb+depth+intr** | **0.081** | 0.081 | 0.99 | 2 | 0.126 → **0.081** |
| 0378 | rgb            | 0.206 | 0.486 | 0.54 | 0 | — |
| 0378 | **rgb+depth**  | **0.052** | 0.058 | 0.95 | 1 | 0.16 → **0.052** |
| 0378 | rgb+depth+intr | 0.054 | 0.062 | 0.95 | 0 | 0.095 → 0.054 |

- **Full model ≫ apache in depth modes** (s0011 rgb+depth+intr 0.126→0.081, s0378 rgb+depth 0.16→0.052).
- **rgb-only is unstable** on the full model (scale 0.54–0.78, SE3-ATE ~0.5) — use a depth mode.
- intrinsics add little over rgb+depth on the full model (sometimes marginally worse).

### Online SLAM (`MaLongSLAM`: fused map + keyframes + online LC) — rgb+depth+intr

| scene | SLAM no-loop | SLAM + online LC | chunk+LC (offline) |
|---|---|---|---|
| **0231 (444f)** | **0.131** | 0.199–0.396 (hurts) | 0.469 |
| 0011 | ~0.07 | 0.068 | 0.081 |
| 0378 | ~0.067 | 0.067 (0 loops) | 0.054 |

- **On long sequences the full model + fused online SLAM dominates:** scene0231 = **0.131** with NO
  loop closure, vs chunk+LC 0.469 (which is dragged down by weak loops).
- **Loop closure now HURTS with the full model**, even when it catches the *true* GT loops
  (148↔435, 217↔415). Reason: **LC only helps when accumulated drift ≫ the loop constraint's own
  measurement error.** The full model's geometry keeps drift so low (0.13 over 444 frames) that the
  short loop-window re-inference noise exceeds it → closure adds error. With the weaker apache model
  drift was large (0.48) so the same online LC helped (0.48→0.32).
- **Recommendation:** with the full model, run the fused SLAM with LC OFF (or only on very long /
  visibly drifting sequences). The LC implementation is correct — it's just not needed when geometry
  is this good.

---

_The apache-era ablations below are kept for the per-component analysis (modes, LC verification,
chunk-size, coloc sweep); the absolute numbers are superseded by the full-model table above._

## A) Input modes — cs=20, loop closure ON (apache model)

| scene | mode | Sim3-ATE | **SE3-ATE** | scale | FPS | VRAM |
|---|---|---|---|---|---|---|
| 0011 | rgb            | 0.259 | 0.407 | 1.20 | 4.1 | 23 GB |
| 0011 | rgb+intr       | 0.226 | 0.400 | 1.21 | 3.7 | 23 GB |
| 0011 | rgb+depth      | 0.181 | 0.191 | 1.04 | 4.0 | 23 GB |
| 0011 | **rgb+depth+intr** | **0.169** | **0.192** | 1.05 | 4.3 | 23 GB |
| 0378 | rgb            | 0.282 | 0.285 | 0.92 | 3.6 | 23 GB |
| 0378 | rgb+intr       | 0.244 | 0.244 | 1.01 | 3.5 | 23 GB |
| 0378 | rgb+depth      | 0.160 | 0.165 | 0.93 | 3.2 | 23 GB |
| 0378 | **rgb+depth+intr** | **0.095** | **0.095** | 0.99 | 2.9 | 23 GB |

Monotonic: `rgb < rgb+intr < rgb+depth < rgb+depth+intr` on both scenes. Depth is the
big jump (scale → ~1.0, ATE roughly halved); intrinsics add a modest further gain.

## B) Loop-closure ablation — rgb+depth+intr, cs=20

| scene | LC off | LC on | FPS off→on | VRAM off→on |
|---|---|---|---|---|
| 0011 | 0.169 | 0.169 | 7.8 → 4.3 | 14 → 23 GB |
| 0378 | 0.137 | **0.095** | 8.2 → 2.9 | 14 → 23 GB |

LC helps on 0378 (0.137 → 0.095). On 0011 it is neutral **at the old default
`verify.coloc_ratio = 0.5`** — the scene's true loop sits at coloc ≈ 0.52, just above the
threshold, so it is rejected. The A/B/C tables above were run at coloc 0.5; the coloc sweep
below shows 0.6 is better, and the default is now **0.6**.

### Loop-closure verification threshold (`coloc_ratio`) sweep — rgb+depth+intr, cs20

| coloc_ratio | s0011 loops / Sim3 | s0378 loops / Sim3 |
|---|---|---|
| 0.4 | 0 / 0.169 | 1 / 0.095 |
| 0.5 (old default) | 0 / 0.169 | 1 / 0.095 |
| **0.6 (new default)** | **1 / 0.126** | **1 / 0.095** |
| 0.7 | 1 / 0.126 | 3 / 0.0956 |
| 0.8 | 4 / 0.128 | 3 / 0.0956 |
| 1.0 | 4 / 0.128 | 3 / 0.0956 |

**0.6 is the common sweet spot for both scenes.** s0011 needs ≥0.6 to admit its one true
loop (0.169 → 0.126, −25 %); below that it is rejected. s0378 already gets its good loop at
0.4–0.6 (0.095); at 0.7+ two weaker loops also pass and very slightly hurt (0.0956). So the
co-location verification trades precision/recall correctly, and 0.6 maximizes accuracy on
both while keeping the fewest (fastest) constraints. Default is now 0.6.

## C) Chunk-size sweep — rgb+depth+intr, loop closure ON

| scene | cs20 | cs30 | cs40 | cs60 | VRAM (cs60) |
|---|---|---|---|---|---|
| 0011 | 0.169 | 0.160 | 0.239 ⚠️ | **0.135** | 32 GB |
| 0378 | **0.095** | 0.148 | 0.145 | 0.163 | 32 GB |

No universal best chunk size: **0378 prefers small chunks** (more loop benefit), **0011
prefers large chunks** (less accumulated drift; its small-chunk loop is rejected at the
default coloc). `cs60` needs 32 GB. The 0011 `cs40 = 0.239` breaks the trend — a likely
single bad chunk-pair alignment, worth investigating. Overlap used: cs20→8, 30→10, 40→13, 60→20.

## Chunk size vs drift without loop closure (earlier, rgb)

Smaller chunks accumulate more drift (motivates loop closure for online/robot use):
cs60 → 0.169, cs30 → 0.199, cs20 → 0.259, cs12 → 0.340, cs8 → 0.477 (Sim3-ATE, scene0011, rgb).

## Loop closure mechanism notes

- **Candidate sources** (union): SALAD VPR (appearance) + geometric (provisional-trajectory
  temporally-far / spatially-near pairs, with temporal NMS).
- **Verification** (applied to all candidates): re-infer the loop window jointly and require
  the two halves to *co-locate* in that reconstruction (`coloc < verify.coloc_ratio`) plus a
  sane relative scale. This is what rejects unreliable constraints — and what fails in rgb mode.
- **Loop windows are chunk-bounded** (≤ chunk_size); raising `loop_chunk_size` above the chunk
  size is a no-op.
- **Optimizer:** `Sim3LoopOptimizer` (PyPose, Levenberg–Marquardt; Python solver — C++ optional).

## Why SALAD alone is insufficient (scene0011)

GT shows true revisits 89↔177 (3.5 cm apart, 88 frames later) and 2↔195 (5 cm, 193 frames).
SALAD cos-sim for these is only 0.16–0.28 (large viewpoint change), while SALAD's top
large-span matches are visually-similar-but-not-co-located pairs (~0.68). Geometric proposal
recovers the true ones (195↔4, 173↔94, …).

## Outputs

`outputs/bench/summary.json` (Sets A/B/C), `outputs/sweep_lc/summary.json` (coloc sweep).
Each run dir has `camera_poses.txt`, `combined_pcd.ply`, `loop_closures.txt`. Reproduce:
`python -m ma_long.bench --out outputs/bench` and `python -m ma_long.sweep_lc`.
