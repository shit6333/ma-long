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

**All of the above ship in one `weights.zip` (19.4 GB, 2026-10-25, stored uncompressed).** Unzip it
at the repo root, then point the caches at the bundled copies:

```bash
unzip weights.zip                       # creates vendor/weights/, src/weights/, cache/, WEIGHTS_README.txt
export HF_HOME=$PWD/cache/huggingface TORCH_HOME=$PWD/cache/torch
export HF_HUB_OFFLINE=1                 # optional: no network calls to HuggingFace
```

Not needed for this configuration (and not in the bundle): `opense3r_fastsam_query_semconf.pth`
(old checkpoint), MapAnything weights (`--backend ma` only). The 2026-07-16 `weights.zip` has been
replaced by this one.

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

**③** Open the full viewer URL the server prints and wait for `===== READY =====`. The viewer
shows each keyframe immediately in the *live keyframes* panel; the 3D view (points, camera frustum,
trajectory, instances) is revealed frame by frame, one submap (32 keyframes) behind live, and the
`frame` timeline can be scrubbed. On a loop closure every frame is redrawn with corrected poses.
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
| compute | with `--semantic`: ~21 GB VRAM, ~16 GB host RAM, **1.1–2.3 keyframes/s** of reconstruction in streaming tests (TUM fr1_room 2.3, ScanNet++ 1.1–1.9; the GPU was shared with another job, so expect the upper end on a dedicated GPU); the `voxels` viewer style costs ~40 % of that; geometry only (no `--semantic`): 7–12 keyframes/s | run on a GPU server, not on the robot's onboard computer. Keyframe arrival rate depends on motion: TUM fr1_room (hand-held, 45 s) yields 186 keyframes at 50 px ≈ 4/s, faster than the semantic layer processes them — budget for lag, slower motion, or a faster GPU |
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
| `--filter_min_count` | **2** (streaming) | export noise filter: drop instances observed ≤ N times. Streaming/robot default 2 keeps only objects seen in ≥ 3 keyframes (ScanNet++ test: 32 of 169 dropped vs 17 at 1); the offline runner keeps the paper setting 1 |

**Viewer arguments**

| arg | default | meaning |
|---|---|---|
| `--viz` | `none` | `web` (browser URL), `spawn` (native window), `connect` (remote rerun viewer) |
| `--viz_reveal` | `paced` | `paced` = reveal each submap frame by frame over its compute time; `instant` = no pacing |
| `--viz_inst_style` | `points` | instance layer: `points` (cheap) or `voxels` (solid translucent 5 cm cubes, nicer but ~40 % slower overall) |
| `--viz_inst_radius` / `--viz_inst_alpha` | 0.25 / 0.5 | point radius as a fraction of the voxel / cube opacity |
| `--viz_max_points` | 500,000 | viewer point budget (per revealed frame: /64) |

On the robot server prefer `--viz none` (or `points`); the viewer is for debugging and demos.

Geometry and loop args (`--backend`, `--mode`, `--submap_size`, `--sim_threshold`,
`--max_per_submap`, `--coloc_ratio`, `--no_loop`, viewer, transport) are in [README.md](README.md) §3–4.

## 11. Code map for integrators (e.g. building a ROS bridge)

The system is **not a ROS node**: it reads frames from one of three transports and writes
results to files at stop (plus the live rerun viewer). This section lists where things are in the
code so a bridge can be written by hand or with an AI assistant. Paths are relative to `src/`.

### 11.1 Process and threads

```
run_stream.py  main() ── builds MaSlam (+ SemanticHooks with --semantic, + RerunViz with --viz)
               stream_loop()                                   run_stream.py:124
                 source thread    (sources.py)   receives frames, writes them to <out>/live/rgb/,
                                                  queues (rgb_path, depth_path)
                 live tap thread  (_live_tap)     shows each frame in the viewer       run_stream.py:92
                 main thread      buffers submap_size keyframes (+1 overlap frame) ->
                                  _process(window) -> MaSlam.process_submap()   run_stream.py:157
                 viewer thread    (viz.py)        paced frame-by-frame reveal
               at STOP: final optimize -> hooks.on_finish (instance export) -> _finalize (poses, cloud)
```

All SLAM state (pose graph, submaps, instance table) is mutated **only in the main thread inside
`process_submap`**. Read it from the main thread right after `process_submap` returns, or copy it.
The pipeline is path-based: every received frame is written once to disk on the server.

### 11.2 Input transports (producer → server)

Keyframe selection is **producer-side** in `zmq` / `folder` mode: the server processes every frame
it receives, so a bridge must apply the blur filter and the 50 px keyframe gate itself. Reuse
`ma_slam/keyframe_flow.py::DisparityKeyframer` (`step(frame_bgr) -> bool`, causal, OpenCV only).

**zmq** (recommended). Server: `PULL`, binds `--bind tcp://*:5599` (`sources.py:118`). Producer:
`PUSH` connect, reference implementation `ma_slam_stream/client.py::ZmqSender` (`client.py:42`).
Each message is a multipart list:

| message | parts | notes |
|---|---|---|
| META (optional in `rgb` mode) | `[b"META", json {"intrinsics": 3×3 list, "depth_scale": float}]` | needed only for `rgb+intr` / depth modes |
| FRAME | `[b"FRAME", json {"idx": int, "rgb_ext": "jpg"}, rgb_bytes, depth_bytes_or_b""]` | `rgb_bytes` = encoded JPEG/PNG; depth = 16-bit PNG in mm (depth modes only). Unknown header keys are ignored, so a capture timestamp can be added without breaking the server (the server does not use it yet) |
| STOP | `[b"STOP"]` | server finishes the queue, optimizes, exports, exits |

Back-pressure: `RCVHWM`/`SNDHWM` 1000 messages and a 256-frame server queue — when the semantic
layer is slower than keyframe arrival, frames wait in these buffers (latency grows); once they are
full the producer's send blocks. Nothing is dropped silently.

**folder** (`--source folder --watch_dir D`, `sources.py:182`): write `D/rgb/<name>.jpg` atomically
(write `<name>.jpg.tmp`, then rename); frames are consumed in sorted file-name order; create the
file `D/STOP` to finish. Reference: `client.py::FolderSender` (`client.py:63`).

### 11.3 Where results become available (the place to publish from)

After each submap, `run_stream.py:173` (where `viz.push_submap` is called) has:

| data | how to read it | notes |
|---|---|---|
| keyframe poses | `for sm in slam.map.ordered(): T = slam.graph.get_pose(sm.key(i))` for `i in range(sm.n)` | 4×4 cam→world, OpenCV axes, world = first keyframe (§7). The overlap frame appears in two submaps — dedupe by `sm.image_paths[i]`. All poses can change after every submap (global re-optimization) and jump on loop closure |
| per-frame point cloud | `sm.frame_points_global(slam.graph, i)` (`ma_slam/submap.py:127`) → (H,W,3) world points; confidence: `sm.conf[i] > sm.conf_threshold`; colours `sm.colors[i]` | `RerunViz._frame_payload` (`viz.py:270`) is a ready filtered + subsampled version |
| instance map | `hooks.table.records` → `InstanceRecord` (`ma_slam_semantic/instance_map.py:119`): `inst_id, center, bbox_min, bbox_max, voxels` (5 cm grid indices), `count, descriptor` | `RerunViz._instance_payload` (`viz.py:189`) builds display arrays. Record geometry refreshes lazily after pose changes; exact positions come from the export at stop. Calling `table.rekey_all()` mid-stream would refresh them but also changes later association (not the evaluated behaviour) |
| class labels | not available live | assigned at export by text classification (`ma_slam_semantic/deploy.py:79`) |
| loop closures | `slam.stats["n_loops"]`, `slam.loop_log` (one line per candidate decision) | |

At stop: `hooks.on_finish` → `ma_slam_semantic/export.py:32::export_all` (final re-lift, merge,
noise filter, instance files) and `_finalize` (`run_stream.py:49`: `camera_poses.txt`,
`combined_pcd.ply`, stats). The robot artifact set (§6) is then produced by `deploy.py`.

### 11.4 Core classes (reference)

| class / function | file | role |
|---|---|---|
| `MaSlam.process_submap` | `ma_slam/solver.py:547` | inference → placement → loop closure → `graph.optimize()` → `hooks.on_submap` → IDLV pass |
| `PoseGraph.get_pose` / `optimize` | `ma_slam/graph.py:117` / `:109` | gtsam SE(3) graph, node key = submap base id + frame index |
| `Submap` | `ma_slam/submap.py` | per-submap points (submap-local), poses, conf, image paths |
| `SemanticHooks.on_submap` / `on_finish` | `ma_slam_semantic/hooks.py:65` / `:368` | Ident3R forward per 4-frame chunk, proposals, instance association; export |
| `InstanceTable.match_chunk` | `ma_slam_semantic/instance_map.py:353` | association into the persistent instance map |
| `RerunViz` | `ma_slam_stream/viz.py:32` | viewer; shows how to read every output above |

### 11.5 Notes for a ROS bridge

- Keep the SLAM server in its own process and Python environment (torch 2.8 + cu128, gtsam 4.3a0,
  Python 3.10) and connect the ROS side over zmq: ROS1 Noetic (Python 3.8) cannot import this
  code, and mixing ROS2 and this environment is fragile.
- Input side needs no server change: subscribe the image topic, keyframe-gate, JPEG-encode, send
  FRAME messages as in `ZmqSender`.
- Output side needs a small server addition at `run_stream.py:173` (e.g. a zmq `PUB` of poses /
  instance boxes / a subsampled cloud per submap); the ROS side republishes as TF, `nav_msgs/Path`,
  `sensor_msgs/PointCloud2`, `visualization_msgs/MarkerArray`.
- Trajectory timestamps are currently synthetic (`index / fps`); carry the image `header.stamp`
  through the FRAME header if the robot needs real time alignment.
