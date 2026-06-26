"""Server CLI for ma_slam streaming — real-time reconstruction from a live RGBD stream.

Standalone driver: it does NOT modify the offline solver. It builds a normal ``MaSlam`` and
drives it frame-by-frame through the SAME public ``process_submap`` the offline ``run`` uses,
accumulating live frames into submaps (last frame carried as the shared overlap, ``overlap=1``)
and writing the same artefacts (poses / merged PLY / stats / loops) when the stream stops.

Three transports (``--source``):

    # LOCAL — RealSense attached to THIS machine (no network)
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m ma_slam_stream.run_stream \
        --source local --mode rgb+depth+intr --depth_max 6 \
        --out outputs/maslam_live --submap_size 20 --k 3

    # ZMQ — a laptop pushes to this host:5599
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m ma_slam_stream.run_stream \
        --source zmq --bind 'tcp://*:5599' --mode rgb+depth+intr --depth_max 6 \
        --out outputs/maslam_live --submap_size 20

    # FOLDER — a laptop rsyncs into <out>/live/incoming/{rgb,depth}
    PYTHONPATH=src python -m ma_slam_stream.run_stream --source folder --mode rgb \
        --backend da3 --out outputs/maslam_live --submap_size 20

The laptop side is ``ma_slam_stream/client.py``. See ``ma_slam_stream/README.md``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

# make the src/ sibling packages (model, ma_slam, align, ...) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch

from model.inputs import MODES, load_intrinsics
from ma_slam.solver import MaSlam, DEFAULT_CONFIG
from ma_slam.keyframe_flow import DisparityKeyframer
from ma_slam_stream.sources import make_source
from ma_slam_stream.realsense import add_capture_args, capture_from_args
from ma_slam_stream.viz import make_viz


def _finalize(slam, out, n, elapsed, fps, vram, mode):
    """Write poses / merged PLY / stats / loops — mirrors MaSlam.run's tail, externally."""
    print(f"[ma_slam] timing: {elapsed:.1f}s for {n} frames = {fps:.2f} fps "
          f"(streaming, model-load excluded) | peak VRAM {vram:.1f} GB | "
          f"submap_size={slam.cfg['submap_size']}")
    poses_txt = os.path.join(out, "camera_poses.txt")
    slam.map.write_poses(slam.graph, poses_txt)
    combined_ply = os.path.join(out, "combined_pcd.ply")
    pc = slam.cfg["Pointcloud"]
    slam.map.write_points(slam.graph, combined_ply, voxel_size=pc["voxel_size"],
                          max_points=pc["max_points"], conf_coef=pc["conf_coef"])
    print(f"[ma_slam] done. {n} frames | submaps={slam.map.num_submaps()} "
          f"loops={slam.graph.get_num_loops()} | poses -> {poses_txt}  pcd -> {combined_ply}")

    n_accept = sum(1 for ln in slam.loop_log if "ACCEPT" in ln)
    stats = {
        "backend": type(slam.model).__name__, "mode": mode,
        "submap_size": slam.cfg["submap_size"], "overlap": slam.cfg["overlap"],
        "n_frames": n, "n_submaps": slam.map.num_submaps(),
        "loops_accepted": n_accept, "loops_candidates": len(slam.loop_log),
        "fps": round(fps, 2), "seconds": round(elapsed, 1), "peak_vram_gb": round(vram, 2),
        "coloc_ratio": slam.cfg["Loop"]["coloc_ratio"], "loop_enabled": slam.cfg["Loop"]["enable"],
        "depth_max": slam._depth_max, "streaming": True,
    }
    with open(os.path.join(out, "run_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(out, "run_stats.txt"), "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    with open(os.path.join(out, "loops.txt"), "w") as f:
        f.write(f"# {n_accept} accepted / {len(slam.loop_log)} candidates\n")
        f.write("\n".join(slam.loop_log) + ("\n" if slam.loop_log else ""))
    return {"n_submaps": slam.map.num_submaps(), "n_loops": slam.graph.get_num_loops(),
            "poses_txt": poses_txt, "combined_ply": combined_ply, "fps": fps}


def stream_loop(slam, source, out, *, mode, intrinsics, depth_scale, depth_max, dump_every,
                viz=None, keyframe_disparity=0.0):
    """Accumulate live frames into submaps and feed them to slam.process_submap.

    With ``keyframe_disparity > 0`` a causal LK optical-flow gate runs per arriving frame
    (same ``DisparityKeyframer`` as the offline path): low-parallax frames are dropped before
    accumulation, so ``submap_size`` counts keyframes. Non-keyframes are still fed to the gate
    (to track displacement) but never enter a submap."""
    # the solver caches these on the instance (offline run() sets them the same way).
    slam._mode, slam._input_K, slam._depth_scale, slam._depth_max = (
        mode, intrinsics, depth_scale, depth_max)
    want_depth = "depth" in mode
    size = slam.cfg["submap_size"]
    overlap = slam.cfg["overlap"]
    keyframer = DisparityKeyframer(keyframe_disparity) if keyframe_disparity > 0 else None
    if keyframer is not None:
        print(f"[ma_slam] keyframe gate ON: disparity > {keyframe_disparity:g}px "
              f"(submap_size counts keyframes)")

    _ = slam.model                                # force model load before timing
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_frames = 0
    carry: list = []                              # overlap frame(s) from the previous submap
    buf: list = []                                # frames accumulating toward the next submap

    loops_seen = [0]

    def _process(window):
        rgb = [f[0] for f in window]
        dep = [f[1] for f in window] if want_depth else None
        if want_depth and any(d is None for d in dep):
            raise ValueError("depth mode but a frame arrived without a depth path")
        slam.process_submap(rgb, dep)
        print(f"[ma_slam] submap {slam.stats['n_submaps']} (+{len(window)} frames, "
              f"{n_frames} total) | nodes={len(slam.graph.initialized)} "
              f"loops={slam.stats['n_loops']}")
        if viz is not None:
            loop_now = slam.stats["n_loops"]
            viz.update(slam, n_frames, loop_happened=loop_now > loops_seen[0])
            loops_seen[0] = loop_now

    try:
        for frame in source:                      # blocks until next frame / stop sentinel
            if frame is None:
                break
            if keyframer is not None:              # causal LK gate: drop low-parallax frames
                img = cv2.imread(frame[0], cv2.IMREAD_COLOR)
                if img is None or not keyframer.step(img):
                    continue
            buf.append(frame)
            n_frames += 1
            if len(buf) >= size:
                window = carry + buf
                _process(window)
                carry = window[-overlap:]
                buf = []
                if dump_every and slam.stats["n_submaps"] % dump_every == 0:
                    slam.map.write_poses(slam.graph, os.path.join(out, "camera_poses.txt"))
    except KeyboardInterrupt:
        print("\n[ma_slam] interrupted; flushing what we have")
    finally:
        source.close()

    if buf:                                       # trailing partial submap at stop
        window = carry + buf
        if len(window) >= 2:
            _process(window)

    elapsed = time.time() - t0
    fps = n_frames / elapsed if elapsed > 0 else 0.0
    vram = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
    return _finalize(slam, out, n_frames, elapsed, fps, vram, mode)


def main():
    ap = argparse.ArgumentParser(description="ma_slam streaming (live RGBD -> incremental SLAM)")
    ap.add_argument("--source", default="local", choices=["local", "zmq", "folder"],
                    help="frame transport: 'local' camera here | 'zmq' socket push | 'folder' watch a dir")
    ap.add_argument("--bind", default="tcp://*:5599", help="[zmq] PULL bind address")
    ap.add_argument("--watch_dir", help="[folder] dir to watch (default <out>/live/incoming)")
    ap.add_argument("--poll", type=float, default=0.05, help="[folder] poll interval (s)")

    ap.add_argument("--mode", default="rgb", choices=list(MODES))
    ap.add_argument("--backend", default="ma", choices=["ma", "da3"],
                    help="backbone: 'ma' MapAnything | 'da3' DA3-nested (rgb/rgb+intr only)")
    ap.add_argument("--manifold", default="se3", choices=["se3", "sim3", "sl4"])
    ap.add_argument("--intrinsic",
                    help="K file; omit for 'local' (taken from the camera) or 'zmq' (META)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--submap_size", type=int, default=20)
    ap.add_argument("--keyframe_disparity", type=float, default=25.0,
                    help="LK optical-flow keyframe gate (px): drop low-parallax frames before "
                         "accumulation; submap_size then counts keyframes. Default 25; 0 = off (every frame)")
    ap.add_argument("--no_loop", action="store_true")
    ap.add_argument("--sim_threshold", type=float, default=DEFAULT_CONFIG["Loop"]["sim_threshold"])
    ap.add_argument("--coloc_ratio", type=float, default=DEFAULT_CONFIG["Loop"]["coloc_ratio"])
    ap.add_argument("--loop_half_window", type=int, default=DEFAULT_CONFIG["Loop"]["half_window"])
    ap.add_argument("--voxel_size", type=float, default=DEFAULT_CONFIG["Pointcloud"]["voxel_size"])
    ap.add_argument("--max_points", type=int, default=DEFAULT_CONFIG["Pointcloud"]["max_points"])
    ap.add_argument("--conf_coef", type=float, default=DEFAULT_CONFIG["Pointcloud"]["conf_coef"])
    ap.add_argument("--depth_scale", type=float, default=1000.0)
    ap.add_argument("--depth_max", type=float, default=0.0,
                    help="model-side: zero out depth beyond this many metres (RealSense: ~6); 0 = off")
    ap.add_argument("--dump_every", type=int, default=0,
                    help="write intermediate poses every N submaps for live preview (0 = only at stop)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--viz", default="none", choices=["none", "spawn", "web", "connect"],
                    help="real-time rerun visualisation: 'spawn' native viewer here | "
                         "'web' serve web viewer | 'connect' stream to a remote viewer (--viz_addr)")
    ap.add_argument("--viz_addr", default=None,
                    help="[viz connect] running rerun viewer url, e.g. rr+http://LAPTOP_IP:9876")
    ap.add_argument("--viz_max_points", type=int, default=500_000,
                    help="cap on points logged to rerun (split across submaps)")
    add_capture_args(ap)                          # --width/--height/--fps/--k/--enable_ir/... (local only)
    a = ap.parse_args()

    if a.backend == "da3" and "depth" in a.mode:
        ap.error("backend da3 supports rgb / rgb+intr only (DA3 does not ingest depth)")

    os.makedirs(a.out, exist_ok=True)
    need_depth = "depth" in a.mode
    need_intr = ("intr" in a.mode) or need_depth

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["submap_size"] = a.submap_size
    cfg["Graph"]["manifold"] = a.manifold
    cfg["Loop"].update(enable=not a.no_loop, sim_threshold=a.sim_threshold,
                       coloc_ratio=a.coloc_ratio, half_window=a.loop_half_window)
    cfg["Pointcloud"].update(voxel_size=a.voxel_size, max_points=a.max_points, conf_coef=a.conf_coef)

    model = None
    if a.backend == "da3":
        from model.da3_infer import Da3ChunkModel
        model = Da3ChunkModel(device=a.device)

    live_dir = os.path.join(a.out, "live")
    capture = capture_from_args(a, need_depth) if a.source == "local" else None
    source = make_source(a.source, live_dir=live_dir, want_depth=need_depth, bind=a.bind,
                         watch_dir=a.watch_dir, poll=a.poll, capture=capture, k=a.k)

    # intrinsics: CLI file wins; otherwise (intr mode) take from the source (local camera / zmq META).
    intrinsics = load_intrinsics(a.intrinsic) if a.intrinsic else None
    depth_scale = a.depth_scale
    if need_intr and intrinsics is None:
        print("[ma_slam] no --intrinsic; waiting up to 30s for intrinsics from the stream...")
        for _ in range(300):
            if getattr(source, "intrinsics", None) is not None:
                intrinsics = source.intrinsics
                if getattr(source, "depth_scale", None):
                    depth_scale = source.depth_scale
                print(f"[ma_slam] got intrinsics from stream; depth_scale={depth_scale}")
                break
            time.sleep(0.1)
        if intrinsics is None:
            source.close()
            ap.error(f"mode {a.mode} needs intrinsics but none given and none arrived on the stream")

    viz = make_viz(a.viz, a.viz_addr, conf_coef=a.conf_coef, max_points=a.viz_max_points)

    slam = MaSlam(config=cfg, model=model, device=a.device)
    res = stream_loop(slam, source, a.out, mode=a.mode, intrinsics=intrinsics,
                      depth_scale=depth_scale,
                      depth_max=(a.depth_max if a.depth_max > 0 else None),
                      dump_every=a.dump_every, viz=viz, keyframe_disparity=a.keyframe_disparity)
    print(f"[ma_slam] stream done: submaps={res['n_submaps']} loops={res['n_loops']} "
          f"fps={res['fps']:.2f} -> {res['poses_txt']}")


if __name__ == "__main__":
    main()
