# Evaluating ma_slam_semantic with predicted depth + predicted pose

**Written 2026-08-29 from the OpenSe3r side**, after reading this repo. Aimed at
extending what already works (ScanNet-5) to the other four datasets, and at
making the numbers land in the **same table** as the OpenSe3r GT-geometry rows.

---

## 0. What you already have — do not rebuild it

`scripts/run_scannet5_ma_sem.sh` is the complete chain and it works:

```
--geometry ma --backend da3 --mode rgb      ← predicted depth AND predicted pose
  → <sc>_instance_labels.npz + <sc>_instance_sem_queries.npz
  → query_instance_semantic.py               → <sc>_instance_semantic_feats.npz
  → eval_instance_npz_scannet.py             → instance AP
  → tools/ovo_style_semantic_eval.py         → instsem mIoU / mAcc
```

Measured (`outputs/ma_sem/scannet5_gate_a1pq/SUMMARY.txt`): SN20 instsem
35.39 / 52.58, SN200 14.45 / 20.39.

Everything below is (a) how to reach the other datasets and (b) three protocol
details that currently make your numbers **not** comparable to ours.

---

## 1. ⚠️ Three things that make the numbers incomparable today

### 1.1 ScanNet instance uses a different benchmark AND a different scene set

| | yours (`run_scannet5_ma_sem.sh` step 2) | ours (the table row) |
|---|---|---|
| scenes | ScanNet **5** (0011/0050/0231/0378/0518) | ScanNet **val-50** (first 50 of `scannetv2_val.txt`) |
| flags | `--benchmark 200`, `--gt_ply gt_scannet200_rawid_<sc>.ply` | `--benchmark 20`, `--gt_ply <sc>_vh_clean_2.labels.ply` |
| stuff | none | `--filter_pred_stuff --semantic_npz <..._instance_semantic_feats.npz> --backbone tips-v2-l14` |

Those 5 scenes are our **semantic** set, not our instance set. To land in the
instance table, run val-50 with our flags:

```bash
# ours, verbatim (OpenSe3r/_eval_val50_full.bash)
python eval_instance_npz_scannet.py \
  --hdbscan_labels_npz <odir>/<sc>_instance_labels.npz \
  --gt_ply             $SCANS/$SC/${SC}_vh_clean_2.labels.ply \
  --aggregation_json   $SCANS/$SC/${SC}.aggregation.json \
  --segs_json          $SCANS/$SC/${SC}_vh_clean_2.0.010000.segs.json \
  --benchmark 20 --constants_path scannet_200/scannet200_constants.py \
  --benchmark_void --segment_snap \
  --filter_pred_stuff --semantic_npz <odir>/<sc>_instance_semantic_feats.npz \
  --backbone tips-v2-l14
# SCANS=/mnt/HDD4/ricky/data/scannet_eval/scans
```

Note the ordering constraint this creates: `--filter_pred_stuff` needs
`_instance_semantic_feats.npz`, so **`query_instance_semantic.py` must run
BEFORE the instance eval**, not after it as in the current script.

There is also a no-stuff variant (`_eval_val50_nostuff.bash`, same minus the last
two lines) — we report both.

### 1.2 `--conf_gate_coef` is a reporting knob and must be declared

Your README says 1.0 for real inference, 0.5 for benchmark reporting, and the
5-scene results differ by ~6 AP50 between them (DA3-rgb: 41.9 at 1.0 vs 47.5 at
0.5). `run_scannet5_ma_sem.sh` hardcodes 1.0 while `scannet5_gate_a1pq` used
gate. **Pick one, put it in the table caption, use it for every dataset.** Our
GT-geometry rows have no equivalent knob, so whichever you pick is a difference
from us that has to be stated.

### 1.3 The alignment is rigid, and its residual is charged to your AP

`hooks.py::on_finish` runs `eval.umeyama_sim3(est, gt)` but **keeps only R and
forces scale = 1**. Good choice — a Sim3 fit would let a scale error be absorbed
by the alignment and quietly flatter the map. But it means any residual metric
error stays in the map and costs IoU directly. You already log it:

```
[sem_hooks] SE3-Umeyama align to GT (N frames): residual RMSE 0.079–0.125 m
```

At a 0.05 m voxel that residual is 1.5–2.5 voxels of systematic offset, which at
IoU 0.5 is not nothing. **Two things worth doing:**

1. **Report the residual RMSE next to every AP number.** It separates "the
   perception is worse" from "the trajectory drifted".
2. **Also run the Sim3 variant as a diagnostic** (not as the headline). If
   Sim3-aligned AP is much higher than SE3-aligned AP, the gap is scale drift in
   the geometry backbone, not segmentation quality — and that is a different
   thing to fix. `src/eval.py::umeyama_sim3` already returns the scale.

You mentioned aligning to first frame with metric scale. Note that is a
**different, stronger** claim than what the code does today: first-frame
alignment is zero-DoF (`T = c2w_gt[0]`, closed form) and does not consult the
rest of the trajectory, so drift accumulates into the tail of the sequence
instead of being spread. The current whole-trajectory SE3-Umeyama is the more
forgiving of the two. Whichever you use, say which — they are not interchangeable.

---

## 2. Reaching the other four datasets

`--geometry ma` does not use a dataset class at all — `run_ma_slam()` globs a
directory. So every dataset only needs a converter to this layout:

```
<scene>/rgb/*.png            (or jpg/jpeg/tiff)
<scene>/intrinsic.txt        4x4 or 3x3
<scene>/gt_pose.txt          TUM rows: timestamp tx ty tz qx qy qz qw, one per frame
                             (needed for --gt: without it T_align is None and the
                              exports stay in the SLAM gauge → instance eval is
                              meaningless against a GT mesh)
```

| dataset | converter | status |
|---|---|---|
| ScanNet val-50 | reuse the ScanNet-5 path, just widen the scene list | ✅ ready |
| ScanNet++ 50 | `scripts/prep_scannetpp.py` | ✅ exists (geometry-only so far) |
| SceneNN 12 | — | ❌ write one; source `/mnt/HDD4/ricky/data/instance_eval_data/scenenn/oni/<id>/{image,depth}`, poses in `oni/scenenn_mv_infos_val.pkl`, fixed K fx=fy=544.47 @640×480 |
| 3RScan 47 | — | ❌ write one. **Frames are stored SIDEWAYS — rotate 90° CW, and rotate colour + depth + K + pose together.** See `semantic_mapping/data.py::ThreeRScanGTScene` for the exact convention we settled on |
| Replica 8 | — | ❌ write one; `/mnt/HDD4/ricky/data/replica/Replica/<scene>/`, we use `--frame_stride 20` |

Our per-dataset eval invocations (copy the flags exactly):

| dataset | GT | flags |
|---|---|---|
| ScanNet val-50 | `--gt_ply <sc>_vh_clean_2.labels.ply` + `--aggregation_json` + `--segs_json` | `--benchmark 20 --benchmark_void --segment_snap` (+ `--filter_pred_stuff` for the full protocol) |
| SceneNN-12 | `--gt_scenenn_npz model_eval/outputs/scenenn_gt/<sc>_scenenn_gt.npz` | `--benchmark 20 --benchmark_void --segment_snap --segs_json <mesh_segs>/<sc>.segs.json --snap_mesh_ply <oni>/<sc>/<sc>.ply` |
| 3RScan-47 | `--gt_scenenn_npz /mnt/HDD4/ricky/data/3RScan_data/_gt/<sc>_3rscan_gt.npz` | `--benchmark 20 --benchmark_void --segment_snap --segs_json <sc>/mesh.refined.0.010000.segs.v2.json` |
| ScanNet++ 50 | `--gt_scenenn_npz .../_gt/<sc>_scannetpp_gt[_top100].npz` | `--benchmark 20 --benchmark_void --segment_snap --segs_json .../_segs/<sc>.segs.json` — **run both the `_top100` and the plain GT, they are separate columns** |

⚠️ `--filter_pred_stuff` **crashes on the `--gt_scenenn_npz` path** (empty class
list → reshape of 0 elements), so SceneNN / 3RScan / ScanNet++ are evaluated
without it. That is what we do too — it is a protocol difference between the
ScanNet column and the other three, not a bug in your run.

Semantic side: `tools/ovo_style_semantic_eval.py --tasks instsem --benches sn20 sn200`.
For Replica use `--benches replica --replica_gt /mnt/HDD4/ricky/data/replica/_gt`
(the replica branch was added 2026-08-28; note `/tmp/replica_gt` gets wiped, pass
the persistent path).

---

## 3. Our GT-geometry reference numbers

This is the row your pred-depth+pred-pose numbers should be compared against —
the gap **is the contribution of doing it RGB-only**, which is the point of the
experiment.

Checkpoint `outputs/opense3r_pq_only_on_a1ep28/checkpoint-pq-ep14.pth`
(= your `--ckpt a1_pq`), GT depth + GT pose, `chunk_size 6`, `frame_stride 1`:

| protocol | AP / AP50 / AP25 |
|---|---|
| ScanNet val-50 (full, with stuff filter) | 41.55 / 66.96 / 82.66 |
| ScanNet val-50 (no-stuff) | 42.05 / 67.94 / 83.95 |
| SceneNN-12 | 32.35 / 53.71 / 68.30 |
| 3RScan-47 | 25.82 / 47.73 / 69.67 |
| ScanNet++ 50 (top100) | 27.67 / 51.17 / 69.00 |
| ScanNet++ 50 (all objects) | 19.44 / 37.95 / 55.85 |

| semantic (OVO protocol, instsem) | mIoU / mAcc |
|---|---|
| ScanNet-5 SN20 | 44.26 / 62.99 |
| ScanNet-5 SN200 | 17.79 / 26.17 |
| Replica-8 | 29.26 / 40.16 |

We also have a **depth-only** ablation on the older checkpoint (DA3 predicted
depth as model input, GT pose, GT geometry for the map and the eval): the loss is
**−1.33 to +0.39 AP50** across four datasets and **−0.65 to +1.91 mIoU** on
semantics. So on our side depth substitution is nearly free — which means
whatever gap you see is dominated by **pose**, not depth. That is the useful
decomposition to report.

---

## 4. Two silent failure modes to guard against

Both cost real numbers and neither raises:

**`python tools/check_run_health.py <out_dir>`** (OpenSe3r repo) scans for:

1. **obs/frame collapse** — a starved FastSAM or frames lost upstream produce a
   plausible-looking map with a chunk of the scene missing. Healthy runs are
   unimodal; reference bands from our GT runs: SceneNN 7.1–14.8, ScanNet val-50
   5.4–18.9, 3RScan 3.5–13.5, ScanNet++ 8.2–22.0, Replica 11.6–20.3.
2. **non-finite `cluster_score`** — AP is rank-sensitive, so a NaN score
   randomises the ranking for that scene. `export.py` on our side now raises
   rather than writing one; your vendored copy predates that.

And a third that matters for your ranking specifically: `--ranker p_top25q`
**falls back to objectness silently** when the checkpoint has no P/Q head. With
`--ckpt a1_pq` it does have one, but if you ever compare against `a1` (no P/Q)
the two are not on the same ranking. Cheap check: is `cluster_score` equal to
`mean_objectness` in the instance table? Then P/Q is not being used.

---

## 5. Suggested order

1. **ScanNet val-50** first — same converter as ScanNet-5, just 50 scenes, and it
   is the only dataset where you can compare directly against a 50-scene GT-geo
   row rather than a 5-scene one.
2. **ScanNet++** second — the converter already exists, and it is the dataset
   where our GT-geo model is strongest relative to the baselines, so it is where
   an RGB-only result would be most interesting.
3. SceneNN / 3RScan / Replica need converters; 3RScan needs the 90° rotation
   handled or every number will be wrong in a way that still looks plausible.
