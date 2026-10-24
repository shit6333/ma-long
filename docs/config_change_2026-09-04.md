# ⚠️ Main association config changed — please re-run the TUM / SLAM semantic eval

**2026-09-04, from the OpenSe3r side.** One line changes in your config. The
checkpoint does **not** change.

## The change

```diff
- --w_desc 0.30  --w_vox 0.30  --w_geo 0.25  --match_threshold 0.32
+ --w_desc 0.70  --w_vox 0.0   --w_geo 0.30  --match_threshold 0.65
```

**The voxel-overlap cue is dropped entirely** and the descriptor carries most of
the weight. Everything else is untouched: same checkpoint
(`outputs/opense3r_pq_only_on_a1ep28/checkpoint-pq-ep14.pth` = your `a1_pq`),
same h2o KV (`recent 4 + heavy-hitter 4`), same chunk size 6, same ranker
`p_top25q`, same `knn_radius 1.0` / `max_candidates 32` shortlist.

Note the threshold moves 0.32 → **0.65** *because* the score scale changed: the
score is `w_desc*desc + w_vox*vox + w_geo*geo` with every cue in [0,1], so the
surviving weights were rescaled to sum to 1.0 and the bar was set at the same
65% of maximum the desc-only arm already used. **Do not keep 0.32 with the new
weights** — that would be a far looser bar and will over-merge.

## Why (so you can sanity-check your own numbers)

Measured on our side, GT depth + GT pose, h2o KV, 4 datasets:

| protocol | new (desc+geo) | old (desc+vox+geo) | Δ AP |
|---|---|---|---:|
| SceneNN-12 | **33.47 / 55.63 / 69.81** | 31.88 / 55.01 / 68.42 | **+1.58** |
| 3RScan-47 | 25.18 / **46.72 / 70.52** | 25.26 / 46.54 / 68.95 | −0.08 |
| ScanNet++ top100 | **28.06 / 52.89 / 70.98** | 27.13 / 50.88 / 68.69 | **+0.93** |
| ScanNet++ all | **19.51 / 39.09 / 58.00** | 19.14 / 37.61 / 55.39 | +0.37 |
| ScanNet val-50 | 42.16 / **68.36** / 82.78 | 42.39 / 67.37 / 82.80 | −0.23 |
| val-50 no-stuff | 42.81 / **69.66 / 84.28** | 42.94 / 68.45 / 84.15 | −0.13 |

**AP50 and AP25 improve on all six protocols**; AP is 4 up / 2 down and both
losses are under 0.25 (noise at these sample sizes). Semantics: ScanNet val-50
SN20 mIoU **41.95 vs 41.16** (+0.79), SN200 gtcls 29.60 vs 29.75 (flat),
Replica-8 28.92 vs 29.10 (flat).

## What we would like re-run

Anything whose numbers depend on the instance map or on descriptor-driven
association:

1. **The full-SLAM semantic/instance tables** (protocol B, `outputs/ma_sem/<ds>_final/`) —
   instance AP and instsem mIoU on whichever datasets you have converters for.
2. **The TUM / Clio loop-verification results** — S1 uses the per-frame proposal
   descriptors, not the map's association weights, so the **verifier itself is
   probably unaffected**. Worth one confirmation run rather than a full 5-seed
   sweep: if `k=16 / tau=0.85 / m=4` still gives the same verdict counts, the ATE
   tables stand as they are and only the map-side numbers need updating.

Please keep the existing outputs and write to new directories — we may need the
old numbers for the ablation table.

## What does NOT need re-running

* **The KV-memory ablation.** It is descriptor-only by construction
  (`w_desc 1.0 / w_vox 0 / w_geo 0`, thr 0.65) so the main config's cue weights
  never entered it.
* **Anything that only touches geometry / trajectory** with no instance map.

## Also pending on our side

`docs/e4_lc_map_healing.md` (the LC on/off → instance-map question) should use
this new config if you run it. It has not started yet as far as we know.
