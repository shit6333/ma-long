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

### Laptop → server, watch in a browser (`zmq` + `--viz web`)
Server (listens, serves the viewer):
```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m ma_slam_stream.run_stream \
    --source zmq --bind 'tcp://*:5599' --mode rgb+depth+intr --depth_max 6 \
    --out outputs/maslam_live --submap_size 20 --viz web
```
Laptop (push frames; replace `SERVER_IP`):
```bash
python client.py --transport zmq --addr tcp://SERVER_IP:5599 --mode rgb+depth+intr --k 3
```
Open the URL the server prints in any browser. Stop the laptop with Ctrl+C — outputs flush
automatically. (The laptop sends intrinsics, so the server needs no `--intrinsic`.)

### Everything on one machine (`local`)
```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m ma_slam_stream.run_stream \
    --source local --mode rgb+depth+intr --depth_max 6 \
    --out outputs/maslam_live --submap_size 20 --k 3 --viz spawn
```
Camera, reconstruction, and viewer all here; intrinsics come from the camera.

## 3. Important args (reconstruction)

| arg | meaning |
|---|---|
| `--mode {rgb, rgb+intr, rgb+depth, rgb+depth+intr}` | which inputs MapAnything uses. Depth modes are far more accurate; **use `rgb+depth+intr` with a RealSense**. |
| `--backend {ma, da3}` | backbone: `ma` MapAnything (all modes) · `da3` DA3-nested (rgb / rgb+intr only, best rgb-only). |
| `--submap_size` | frames per submap/chunk (default 20 ≈ 15 fps, ~15 GB VRAM). |
| `--depth_max` | zero out depth beyond N metres before the model — **set ~6 for RealSense** (far depth is noisy). |
| `--manifold {se3, sim3}` | pose-graph group; `se3` (metric, default) is right for RealSense depth. |
| `--no_loop` / `--coloc_ratio` | disable / tune loop closure. |

## 4. Streaming & viz args

| arg | meaning |
|---|---|
| `--source {local, zmq, folder}` | frame transport (see top). |
| `--bind` | `[zmq]` server PULL address (default `tcp://*:5599`). |
| `--watch_dir` | `[folder]` dir to watch (default `<out>/live/incoming`). |
| `--k` | blur window — keep the sharpest of every `k` frames (`1` = every frame). Start at 2–4. |
| `--viz {none, spawn, web, connect}` | real-time rerun viewer: `spawn` native window here · `web` browser · `connect` a remote viewer. |
| `--viz_addr` | `[viz connect]` running viewer url, e.g. `rr+http://LAPTOP_IP:9876`. |
| `--dump_every N` | write intermediate poses every N submaps (live preview file). |

**RealSense capture** (`local` source / the client): `--enable_ir`, `--no_emitter`,
`--laser_power`, `--no_postprocess` (the official threshold→spatial→temporal→hole-filling depth
filter chain is on by default), `--rs_depth_max` (sensor-side range clip). `--width/--height/--fps`.

Stream to a rerun viewer already running on your laptop (laptop: run `rerun` first):
```bash
PYTHONPATH=src python -m ma_slam_stream.run_stream --source zmq --mode rgb+depth+intr \
    --out outputs/maslam_live --viz connect --viz_addr rr+http://LAPTOP_IP:9876
```
For `--viz connect` the laptop needs `pip install rerun-sdk==0.26.2` (version must match the
server); `--viz web` needs only a browser.

## 5. Streaming pipeline

```
RealSense ─► blur-select (sharpest of k) ─► transport (zmq / folder / in-process)
          ─► server: accumulate submap_size frames ─► ma_slam.process_submap
             (reconstruct chunk → place via shared overlap frame → global PGO → loop closure)
          ─► [--viz] log new points + trajectory to rerun
```

- **Blur selection** runs on the producer (laptop for `zmq`/`folder`, in-process for `local`):
  Laplacian-variance sharpness, sent as JPEG (rgb) + 16-bit PNG (depth in mm → server
  `depth_scale=1000`). Larger `k` removes blur but widens the baseline within a submap and can
  hurt reconstruction — keep it small.
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
