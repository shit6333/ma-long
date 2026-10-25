# ma_slam_stream — real-time RGBD streaming reconstruction

Live front-end for `ma_slam`: stream RGBD from a RealSense (on a laptop **or** on the server),
blur-select the sharp frames, accumulate them into submaps, and run the same `ma_slam`
reconstruction (chunk → SE3/Sim3 pose-graph → incremental global optimize → loop closure)
**until you stop** — with optional real-time point-cloud visualisation.

Three transports (`--source`): **`local`** (camera on the server), **`zmq`** (laptop pushes over
a socket, recommended), **`folder`** (laptop drops files into a shared dir). The offline solver is
not modified — this package only drives it.

## 1. Install

**Server** (the GPU box) — already covered by the `amb3r_bw` env (`mapanything`/`da3`, `torch`,
`gtsam`, `opencv`, `rerun-sdk`). Extra:
```bash
pip install pyzmq            # for --source zmq
pip install pyrealsense2     # only for --source local (camera on the server)
```

**Laptop** (the camera host, for `zmq`/`folder`) — no repo/GPU/model needed; copy `realsense.py`
+ `client.py` over, then:
```bash
pip install pyrealsense2 opencv-python numpy   # official Intel RealSense wrapper
pip install pyzmq                              # for --transport zmq
```

## 2. How to run

**Where to run:** the **server** runs from the **repo root** with `PYTHONPATH=src` (so `--out`
paths land at the repo root, not under `src/`). The **client** runs on the laptop from wherever
you copied `realsense.py` + `client.py`.

### Laptop → server, watch in a browser (`zmq` + `--viz web`)  ← recommended

**① Server** (repo root) — binds the socket, serves the viewer, loads the model, prints `READY`:
```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m ma_slam_stream.run_stream \
    --source zmq --bind 'tcp://*:5599' --out outputs/maslam_live --viz web
```
Defaults = the Ident3R-SLAM config: `--backend da3 --mode rgb` (RGB only, metric), `--submap_size 32`,
50 px keyframe gate. MapAnything with RealSense depth instead: add
`--backend ma --mode rgb+depth+intr --depth_max 6` (and `--mode rgb+depth+intr` on the client).

**② Client** (laptop) — sends intrinsics, then **waits at a `press ENTER` prompt** (it does NOT
stream yet):
```bash
cd src/ma_slam_stream
python client.py --transport zmq --addr tcp://SERVER_IP:5599        # --mode rgb (default) must match the server
```

**③ Open the viewer** — the server prints a full URL like
`http://<server-ip>:9090/?url=rerun%2Bhttp...`. Open **that whole URL** in a browser (the bare
`:9090` only shows rerun's welcome screen). Wait for the server's `===== READY =====` line.

**④ Start** — press **ENTER** in the client terminal → it begins streaming, reconstruction +
point cloud appear live. **Stop** with `Ctrl+C` in the client → the server flushes outputs to
`--out`. (Add `--auto_start` to the client to skip the ENTER handshake.)

### Everything on one machine (`local`)
```bash
cd /path/to/ma_long
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m ma_slam_stream.run_stream \
    --source local --out outputs/maslam_live --k 3 --viz spawn
```
Camera, reconstruction, and viewer all here; intrinsics come from the camera (no `--intrinsic`,
no client). `--viz spawn` opens a native window; use `--viz web` + the printed URL if headless.

## 3. Important args (reconstruction)

| arg | meaning |
|---|---|
| `--mode {rgb, rgb+intr, rgb+depth, rgb+depth+intr}` | which inputs the backbone uses. **Default `rgb`** (DA3). Depth modes need `--backend ma` (MapAnything); with a RealSense use `rgb+depth+intr --depth_max 6`. |
| `--backend {ma, da3}` | backbone: **`da3` (default)** DA3-nested (rgb / rgb+intr only, metric from RGB) · `ma` MapAnything (all modes). |
| `--submap_size` | **keyframes** per submap/chunk (default **32**; counts raw frames only if `--keyframe_disparity 0`). |
| `--keyframe_disparity` | LK optical-flow keyframe gate (px): drop low-parallax frames before accumulation so `submap_size` counts keyframes (causal, same as offline). **Default 50** (on); `0` = off (every frame). Complements the client's `--k` blur filter (sharpness) with a parallax filter. |
| `--sim_threshold` / `--max_per_submap` | loop retrieval: SALAD cos threshold / candidates per submap (default **0.40 / 5**). |
| `--depth_max` | zero out depth beyond N metres before the model — **set ~6 for RealSense** (far depth is noisy). |
| `--manifold {se3, sim3}` | pose-graph group; `se3` (metric, default) is right for RealSense depth. |
| `--no_loop` / `--coloc_ratio` | disable / tune loop closure. |

## 4. Streaming & viz args

| arg | meaning |
|---|---|
| `--source {local, zmq, folder}` | frame transport (see top). |
| `--bind` | `[zmq]` server PULL address (default `tcp://*:5599`). |
| `--watch_dir` | `[folder]` dir to watch (default `<out>/live/incoming`). |
| `--k` | blur window (producer) — keep the sharpest of every `k` frames (`1` = every frame). Start at 2–4. |
| `--keyframe_disparity` | LK keyframe gate (producer) — see §3; default 25 px on, `0` = every frame. |
| `--viz {none, spawn, web, connect}` | real-time rerun viewer: `spawn` native window here · `web` browser · `connect` a remote viewer. |
| `--viz_addr` | `[viz connect]` running viewer url, e.g. `rr+http://LAPTOP_IP:9876`. |
| `--dump_every N` | write intermediate poses every N submaps (live preview file). |

**RealSense capture** (`local` source / the client) — defaults match a validated D4xx config:
capture **848×480**, depth `visual_preset` **Medium Density**, **spatial + temporal** filters on
(decimation + hole-filling off), range **0.2–4.0 m**, then **center-crop to 640×480** for SLAM
(crop keeps fx/fy, only shifts cx/cy; `K` is adjusted automatically). Override with `--preset`,
`--no_spatial` / `--no_temporal` / `--hole` / `--decimation`, `--spatial_alpha/…`, `--rs_depth_max`,
`--out_width/--out_height` (0 = keep native), `--enable_ir`, `--no_emitter`, `--laser_power`, `--k`.

Stream to a rerun viewer already running on your laptop (laptop: run `rerun` first):
```bash
PYTHONPATH=src python -m ma_slam_stream.run_stream --source zmq \
    --out outputs/maslam_live --viz connect --viz_addr rr+http://LAPTOP_IP:9876
```
For `--viz connect` the laptop needs `pip install rerun-sdk==0.26.2` (version must match the
server); `--viz web` needs only a browser.

## 5. Streaming pipeline

```
producer: RealSense ─► blur-select (sharpest of k) ─► LK keyframe gate (disparity) ─► transport
server:   accumulate submap_size keyframes ─► ma_slam.process_submap
          (reconstruct chunk → place via shared overlap frame → global PGO → loop closure)
          ─► [--viz] log new points + trajectory to rerun
```

- **Frame selection runs on the producer** (laptop for `zmq`/`folder`, in-process for `local`),
  in two stages: **(1) blur-select** (`--k`) — Laplacian-variance sharpest of every `k`; then
  **(2) LK keyframe gate** (`--keyframe_disparity`, default 25 px, on — mirrors offline
  `ma_slam`) drops low-parallax frames so `submap_size` counts genuine keyframes. Gating on the
  producer means a near-static camera sends almost nothing → less bandwidth, and the server skips
  disk-writing / LK on dropped frames (the LK runs on the idle laptop CPU, no extra packages).
  Only keyframes go on the wire, as JPEG (rgb) + 16-bit PNG (depth in mm → server
  `depth_scale=1000`). `--keyframe_disparity 0` = every frame. Larger `k` removes blur but widens
  the baseline — keep it small.
- **Two FPS numbers** are reported at stop (and in `run_stats`). *End-to-end fps* = keyframes /
  wall-clock from the first frame — bounded by how fast keyframes arrive (camera motion, `k`,
  gate, network), so it's low by design and is **not** the model speed. *Reconstruction fps* =
  keyframes / time actually inside `process_submap` (GPU only) — the real backbone throughput;
  the GPU mostly idles waiting for live input.
- **Submap accumulation** carries each submap's last frame as the next submap's first (shared
  overlap), exactly like the offline windowing; this anchors each new submap to the global map.
- **Reconstruction / loop closure** is unchanged `ma_slam` — global pose-graph re-optimized after
  every submap; a verified loop adds an edge and corrects drift.
- **Visualisation** is a pure sink (never affects the result): new submap points are logged once,
  the trajectory re-logs each submap, and on a loop closure all clouds re-log to show the
  correction. Outputs (`camera_poses.txt`, `combined_pcd.ply`, `run_stats.txt`, `loops.txt`) are
  written to `--out` at stop.

Files: `realsense.py` (capture, shared/copy to laptop) · `client.py` (laptop CLI) ·
`sources.py` (server frame sources) · `viz.py` (rerun) · `run_stream.py` (server CLI).
