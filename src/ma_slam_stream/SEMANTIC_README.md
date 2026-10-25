# Ident3R-SLAM — Deployment Handoff (live streaming + robot artifacts)

**Version:** branch `semantic` @ `2f24e38`+ (2026-10-25). Supersedes the 2026-07-16 handoff
(see [§9](#9-changes-since-the-2026-07-16-handoff) for what changed).

This document is for whoever deploys the system on a robot or wraps it in a ROS package. It covers
the environment and weights, running the system (live stream or recorded image folder), the
robot-ready output files, coordinate frames, and the runtime limits you need to design around.
Architecture, benchmarks and evaluation live in
[ma_slam_semantic/README.md](../ma_slam_semantic/README.md) and [docs/RESULTS.md](../../docs/RESULTS.md);
camera / transport / viewer details in [README.md](README.md).

```
producer (laptop/robot): camera ─ blur filter (--k) ─ 50 px LK keyframe gate ─► zmq / local / folder
server (GPU):  DA3 submap reconstruction (32 keyframes, RGB only, metric) ─ SE(3) pose graph (gtsam)
               ├─ loop closure: SALAD retrieval (cos ≥ 0.40, top-5/submap)
               │    → IDLV instance check (≥ 4 mutual-NN instance matches, cos ≥ 0.85, else reject)
               │    → geometric check (joint re-inference, co-location ≤ 0.70) → Huber loop factor
               └─ semantic layer per submap: Ident3R causal model (Temporal Token Cache KV)
                    → instance proposals → persistent instance map
stop:          poses / point cloud / instance exports ─► deploy.py ─► robot artifact set
```

**All defaults are the shipped configuration.** Nothing below needs extra flags unless stated.

---

## 1. Environment

Tested: Ubuntu, Python 3.10, **torch 2.8.0 + CUDA 12.8**, NVIDIA RTX PRO 6000 Blackwell (97 GB).

1. Follow [INSTALL.md](../../INSTALL.md) §1 (PyTorch), §3 (gtsam), §4 (geometry deps), §6 (DA3 source
   clone into `thirdparty/Depth-Anything-3`). §2 (MapAnything) is only needed for `--backend ma`.
2. Semantic layer extras (versions as tested):

```bash
pip install ultralytics==8.4.75 sentencepiece==0.2.1 xformers==0.0.32.post2 \
            plyfile==1.1.3 addict==2.4.0 omegaconf==2.3.0 einops==0.8.0 timm==1.0.27 \
            safetensors scikit-learn trimesh moviepy pyzmq
```

Pitfalls:
- **gtsam must be 4.3a0** (`pip install --pre gtsam`); 4.2 segfaults under numpy 2.x.
- Blackwell (sm_120) needs torch ≥ 2.7 with cu128. On older GPUs pick the torch/CUDA build for
  your driver and keep the rest; `xformers` / `triton` versions must match that torch.
- TIPS does **not** need tensorflow (the backbone loads its text encoder without it).
- First run downloads DA3 from HuggingFace and DINOv2 from torch hub (GitHub). On an offline
  robot/server, ship the caches instead (§2) and set `HF_HOME` / `TORCH_HOME`.

## 2. Weights (~19 GB)

| file (path relative to repo root) | size | what | how |
|---|---|---|---|
| `vendor/weights/opense3r_a1ep28_pq_ep14.pth` | 4.8 GB | **Ident3R checkpoint** (`--ckpt a1_pq`, default) | bundle (not public) |
| `vendor/weights/vggt_1b.pt` | 5.0 GB | frozen patch-token encoder used by Ident3R | bundle |
| `vendor/weights/tips_v2_oss_l14_vision.npz`, `..._text.npz` | 1.2 + 0.7 GB | frozen TIPS-v2 vision/text backbone | bundle (`vendor/thirdparty/tips/pytorch/checkpoints/` holds relative symlinks to these) |
| `vendor/weights/fastsam_x.pt` | 145 MB | FastSAM query seeding inside Ident3R | bundle |
| `src/weights/dino_salad.ckpt` | 352 MB | SALAD place recognition | bundle, or INSTALL.md §5 download |
| HF `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | 6.3 GB | SLAM geometry backbone | auto-download, or ship `$HF_HOME/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1` |
| torch hub `dinov2_vitb14_pretrain.pth` + `facebookresearch_dinov2_main/` | 350 MB | SALAD's DINOv2 | auto-download, or ship `$TORCH_HOME/hub/` |

Not needed for this configuration: `opense3r_fastsam_query_semconf.pth` (old checkpoint),
MapAnything weights (`--backend ma` only). The older `weights.zip` (2026-07-16) does **not**
contain the current checkpoint.

## 3. Quickstart A — live: camera client → GPU server, with semantics

**① Server** (repo root):

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m ma_slam_stream.run_stream \
    --source zmq --bind 'tcp://*:5599' --out outputs/maslam_live --viz web --semantic
```

**② Client** (laptop or robot computer with the camera; needs only `realsense.py` + `client.py`):

```bash
cd src/ma_slam_stream
python client.py --transport zmq --addr tcp://SERVER_IP:5599      # --mode rgb (default)
```

**③** Open the full viewer URL the server prints and wait for `===== READY =====`.
**④** Press ENTER in the client to start streaming. **Stop** with `Ctrl+C` in the client: the
server optimizes the final graph and writes everything to `--out` (§6).

Single machine: `--source local` on the server, no client (camera attached to the server).

## 4. Quickstart B — offline: recorded image folder → robot artifacts in one command

Input folder layout: `<scene>/rgb/*.png|jpg` (sorted by filename; the first number in the file
name is used as the frame id). `gt_pose.txt` (TUM format) is optional and only used for evaluation.

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m ma_slam_semantic.run \
    --data <scene> --out outputs/<scene> --deploy
```

`--deploy` runs the post-processing of §6 right after SLAM. Without `--gt` everything stays in
the SLAM world frame (§7).

## 5. Smoke test (verify an install)

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m ma_slam_semantic.run \
    --data example_data/scene0378_00 --gt example_data/scene0378_00/gt_pose.txt \
    --out outputs/_smoke --deploy
```

Expected on the reference machine (measured 2026-10-25, exit 0, ~6 min wall clock incl. model load):

- `keyframe gate: kept 66/190 frames`, 3 submaps, `loops=2`
- `Sim3-ATE=0.0832 | SE3-ATE=0.0984` (±0.01 run to run is normal)
- peak VRAM 22.3 GB, peak host RAM 15.7 GB
- `outputs/_smoke/deploy/` holds the 15 `robot_deploy_scene0378_00_*` files of §6 (96 instances);
  the first row of `camera_poses.txt` is the identity (world = first keyframe's camera).

## 6. Outputs

**Raw run directory (`--out`)**

| file | content |
|---|---|
| `camera_poses.txt` | one flattened 4×4 cam→world matrix per keyframe, ordered by frame id |
| `combined_pcd.ply` | merged confidence-filtered RGB point cloud |
| `<name>_query_instance_table.json`, `_instance_labels.npz`, `_instance_sem_queries.npz`, `_voxels.ply`, `_boxes.ply` | instance map |
| `loops.txt`, `run_stats.json` | every loop-candidate decision; fps / VRAM / loop counts |

**Robot artifact set (`<out>/deploy/`, written by `--deploy` or by `deploy.py`).** Same schema as
the 2026-07-16 handoff, so existing readers keep working:

| artifact | content |
|---|---|
| `robot_deploy_<name>_camera_trajectory.json` | per-keyframe c2w: `timestamp, basename, tx/ty/tz, qx/qy/qz/qw, transform_matrix` |
| `robot_deploy_<name>_instance_semantic_table.json` | **the file the robot reads**: one entry per instance — `inst_id, class_id/class_name, cosine_similarity, center, bbox_min/max, num_voxels, mean_objectness, filtered` |
| `..._instance_semantic_table_{coarse,medium,fine}.json`, `_all_levels.json` | the same instances classified at ~10 / ~29 / ~57 indoor classes |
| `robot_deploy_<name>_{inst,rgb,sem}_color.ply` | per-point clouds with an integer `inst_id` vertex property that indexes the table |
| `..._sem_color_<level>.ply`, `..._semantic_legend_<level>.png` | per-level semantic colouring + legends |

Post-process a finished run (streaming or offline) at any time:

```bash
PYTHONPATH=src python -m ma_slam_semantic.deploy --dir outputs/maslam_live --name maslam_live
```

| arg | default | meaning |
|---|---|---|
| `--dir` | required | finished run directory |
| `--name` | basename of `--dir` | artifact prefix (must match the run's export prefix) |
| `--out` | `<dir>/deploy` | output directory |
| `--max_points` | 2,000,000 | sample cap for the per-point clouds |
| `--levels` | `coarse,medium,fine` | label granularities |
| `--backbone` | `tips-v2-l14` | text encoder for classification (must match the checkpoint) |
| `--fps` | 30 | **synthetic** trajectory timestamp rate — timestamps are `index / fps`, not capture time |

## 7. Coordinate frames (read before writing the ROS bridge)

- Poses are **cam→world**, camera axes **OpenCV / RDF** (x right, y down, z forward).
- The world frame is the **first keyframe's camera frame** (origin at its centre, same axes).
  It is not gravity-aligned.
- Units are metres. DA3 predicts metric scale from RGB; expect a few percent scale error.
- ROS conventions differ (REP-103 body frame x-forward / z-up; REP-105 `map`/`odom`). Publish the
  SLAM world as its own frame (e.g. `ma_slam_map`) and the camera as an `*_optical_frame`, then
  connect them to the robot's TF tree with a fixed transform; do not reinterpret the axes in place.
- `--gt` runs export the instance map in the ground-truth frame (`alignment.json` records the
  transform). A robot run has no `--gt` and stays in the SLAM frame end to end.

## 8. Runtime characteristics and known limitations

| topic | value / behaviour | consequence for integration |
|---|---|---|
| compute | with `--semantic`: ~22 GB VRAM, ~16 GB host RAM, **~1 keyframe/s** end to end (smoke test, shared GPU, incl. warm-up); geometry only (no `--semantic`): 7–12 keyframes/s | run on a GPU server, not on the robot's onboard computer. Keyframe arrival rate depends on motion: TUM fr1_room (hand-held, 45 s) yields 186 keyframes at 50 px ≈ 4/s, faster than the semantic layer processes them — budget for lag, slower motion, or a faster GPU |
| pose latency | poses are produced **per submap** (32 keyframes at 50 px parallax) and re-optimized after every submap; loop edges land one submap later | not a high-rate odometry source — fuse with wheel odometry / VIO for control and use this as the mapping + loop-closure layer |
| memory growth | host RAM ~7 MB per keyframe (dense point maps kept for export) plus instance observations; the semantic KV cache is bounded (Temporal Token Cache) | long missions grow RAM linearly; plan restarts or a map-saving policy |
| optimization cost | full Levenberg–Marquardt over the whole graph after every submap | per-submap time grows with trajectory length |
| instance map after loop closure | an object whose estimated position moves > 1 m during a loop correction may be duplicated instead of re-associated | expect occasional duplicate instances after large loop corrections |
| instance table frame ids | `first_seen` / `last_seen` / `member_frames` are internal graph keys (frame index + submap count), not raw frame numbers | do not join them to image indices without correction |
| `--sem_chunk_size` | affects results slightly (frames inside a chunk attend to each other), not only VRAM | keep the default 4 |
| scale | metric from RGB, a few percent error, no IMU | if the robot has odometry, estimate a scale correction online |

## 9. Changes since the 2026-07-16 handoff

| item | 2026-07-16 | now |
|---|---|---|
| geometry | MapAnything `rgb+depth+intr`, RealSense depth, `--depth_max 6` | **DA3 RGB only** (`--backend da3 --mode rgb`, default); client sends RGB only |
| submap / keyframe gate | 20 keyframes / 25 px | **32 / 50 px** |
| loop closure | SALAD 0.50, 1 candidate per submap + geometric check | SALAD **0.40, top-5** + **IDLV instance check** + geometric check |
| semantic checkpoint | `opense3r_fastsam_query_semconf.pth` | **`opense3r_a1ep28_pq_ep14.pth`** (`a1_pq`) — new weights bundle needed |
| semantic KV | `h2o` recommended for live | `h2o` is the default everywhere (offline too) |
| output schema | robot artifact set | **unchanged** |
| old config | — | `--backend ma --mode rgb+depth+intr --depth_max 6` on server and client still works |

## 10. Semantic streaming arguments (reference)

| arg | default | meaning |
|---|---|---|
| `--semantic` | off | run the Ident3R semantic layer on every submap (enables the instance map and IDLV) |
| `--sem_lc` | `veto` | loop-candidate instance check: `veto` = IDLV (shipped), `off` = SALAD + geometric check only |
| `--sem_chunk_size` | 4 | frames per causal semantic forward (VRAM: 4 ≈ 15 GB, 8 ≈ 26 GB) |
| `--conf_gate_coef` | 1.0 | geometry-validity gate for map points; keep 1.0 live |
| `--kv_policy` | `h2o` | bounded KV cache (frame-0 anchor + 8 recent + 8 heavy-hitter frames, ~0.7 GB); `none` grows ~62 MB/frame |
| `--h2o_recent` / `--h2o_hh` / `--h2o_decay` / `--h2o_subsample` | 8 / 8 / 0.9 / 0.25 | Temporal Token Cache budgets |

Geometry and loop args (`--backend`, `--mode`, `--submap_size`, `--sim_threshold`,
`--max_per_submap`, `--coloc_ratio`, `--no_loop`, viewer, transport) are in [README.md](README.md) §3–4.
