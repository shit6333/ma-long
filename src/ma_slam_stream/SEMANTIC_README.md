# Live Semantic SLAM (streaming) + Robot Deploy Export

This document covers **running the semantic layer on the live stream** and **post-processing
a finished run into robot-ready artifacts**. For the offline (dataset-folder) pipeline,
architecture, benchmark & eval protocol, see
[ma_slam_semantic/README.md](../ma_slam_semantic/README.md). For camera/transport/viewer
setup details, see [README.md](README.md) — everything there applies unchanged; the semantic
layer is a strict add-on.

```
producer: RealSense ─ blur-select ─ keyframe gate ─► transport (zmq/local/folder)
server:   ma_slam submap reconstruction ─ pose graph ─ loop closure
          └─ SemanticHooks (per submap): causal semantic model (KV cache) ─ FastSAM-seeded
             Mask2Former proposals ─ instance table (LC self-healing)
viewer:   live point cloud + instance-colored voxels/bboxes
stop:     instance npz/table/PLY export  ─►  [deploy.py]  robot artifact set
```

## 1. Quickstart — laptop RealSense → server, with semantics

**① Server** (repo root; add `--semantic` to the standard streaming command — `h2o` KV
policy is the default and the recommendation for live runs):

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m ma_slam_stream.run_stream \
    --source zmq --bind 'tcp://*:5599' --mode rgb+depth+intr --depth_max 6 \
    --out outputs/maslam_live --submap_size 20 --viz web \
    --semantic --kv_policy h2o
```

**② Client** (laptop, unchanged from [README.md](README.md) §2):

```bash
cd src/ma_slam_stream
python client.py --transport zmq --addr tcp://SERVER_IP:5599 --mode rgb+depth+intr
```

**③** Open the viewer URL the server prints → **④** press ENTER in the client to start.
The viewer shows the RGB cloud plus an instance-colored semantic layer (voxels + bboxes)
that updates per submap and *snaps back into place* after loop closures.

**Stop** with `Ctrl+C` in the client. The server flushes to `--out`:
`camera_poses.txt`, `combined_pcd.ply`, `run_stats.*`, `loops.txt` (geometry) plus
`<name>_instance_labels.npz`, `<name>_instance_sem_queries.npz`,
`<name>_query_instance_table.json`, `_voxels.ply`, `_boxes.ply` (semantics).

Single-machine variant: same as README §2 `--source local`, plus `--semantic`.

## 2. Semantic streaming arguments

| arg | default | meaning |
|---|---|---|
| `--semantic` | off | run the OpenSe3r query-instance semantic branch on every submap. v0 is synchronous (~0.6 s/frame GPU), so it bounds throughput on fast feeds — a hand-held camera feed is usually slower than that, so in practice it streams live. |
| `--sem_chunk_size` | 4 | frames per causal semantic forward. **Pure VRAM knob** (KV-causal ⇒ results are batching-invariant): 4 ≈ 15 GB peak, 8 ≈ 26 GB. Keep 4 when sharing the GPU with MapAnything (total ~30 GB). |
| `--conf_gate_coef` | 1.0 | scales the geometry-validity gate (per-submap mean confidence). **Keep 1.0 live** — it filters unreliable predicted geometry before it enters the map. Lower (e.g. 0.5) only for benchmark tables where GT-mesh coverage is scored. |
| `--kv_policy` | **`h2o`** | KV-cache retention. A live stream is unbounded — the flat `none` cache grows ~62 MB/frame (61 GB @ 1000 frames) and will eventually OOM; **`h2o`** (STAC temporal compression) keeps it constant (~0.7 GB) with AP on par. `window` = plain sliding window. |
| `--h2o_recent` | 8 | [h2o] recent window: the last N frames' tokens are always kept. |
| `--h2o_hh` | 8 | [h2o] heavy-hitter budget (frame-equivalents): per-layer/per-head top tokens ranked by decayed attention column-sums. |
| `--h2o_decay` | 0.9 | [h2o] score decay γ (higher = longer memory of past attention). |
| `--h2o_subsample` | 0.25 | [h2o] query subsample ratio when accumulating attention scores. 0.25 is validated: AP on par with full scores at ~zero extra cost. |

Frame-0 tokens are always pinned under `h2o` (the sequence anchor for the pos3d gauge).
All geometry args (`--mode`, `--backend`, `--submap_size`, `--depth_max`,
`--keyframe_disparity`, loop-closure knobs, …) are unchanged from [README.md](README.md) §3.

## 3. Deploy post-processing (robot artifacts)

Turn any finished run directory (streaming or offline) into the robot-ready artifact set.
Runs entirely from files — no SLAM state needed, so it can be done any time after the run:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m ma_slam_semantic.deploy \
    --dir outputs/maslam_live --name maslam_live
```

(Offline runs can instead pass `--deploy` to `ma_slam_semantic.run` to do this in one go.)

| arg | default | meaning |
|---|---|---|
| `--dir` | (required) | finished run directory (needs `camera_poses.txt`, `combined_pcd.ply` + the instance exports). |
| `--name` | basename of `--dir` | artifact prefix; must match the run's export prefix. |
| `--out` | `<dir>/deploy` | output directory. |
| `--max_points` | 2,000,000 | sample cap for the per-point clouds. |
| `--levels` | `coarse,medium,fine` | multi-level label sets (built-in: ~10 / ~29 / ~57 indoor classes). |
| `--backbone` | `tips-v2-l14` | text encoder for instance classification (must match the ckpt). |
| `--fps` | 30 | trajectory timestamp rate. |

Outputs in `<dir>/deploy/` (same schema as the OpenSe3r `outputs/deploy/` reference):

| artifact | content |
|---|---|
| `robot_deploy_<name>_camera_trajectory.json` | per-frame c2w: `timestamp, basename, tx/ty/tz, qx/qy/qz/qw, transform_matrix` (ROS bridge schema). |
| `robot_deploy_<name>_instance_semantic_table.json` | **the file the robot reads**: one entry per instance — `inst_id, class_id/class_name, cosine_similarity, center, bbox_min/max, num_voxels, mean_objectness, filtered`. |
| `..._instance_semantic_table_{coarse,medium,fine}.json` + `_all_levels.json` | the same instances classified at three label granularities; `all_levels` carries all three per entry. |
| `robot_deploy_<name>_{inst,rgb,sem}_color.ply` | per-point clouds (`--max_points`-sampled) with an integer `inst_id` vertex property indexing the table; colored by instance palette / original RGB / class palette. |
| `..._sem_color_<level>.ply` + `..._semantic_legend_<level>.png` | per-level semantic coloring + legends. |

**Coordinate-frame rule** — every artifact ends up in ONE frame:
- a **robot run** (no `--gt`) stays in the ma_slam world frame end to end — consistent by
  construction;
- a **benchmark run** (`--gt` given) exports the instance map in the GT-mesh frame; the run
  records `alignment.json`, and `deploy.py` re-applies the same transform to the trajectory
  and point cloud so everything matches.
