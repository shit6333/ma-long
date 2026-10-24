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

import numpy as np
import torch

from model.inputs import MODES, load_intrinsics
from ma_slam.solver import MaSlam, DEFAULT_CONFIG
from ma_slam_stream.sources import make_source
from ma_slam_stream.realsense import add_capture_args, capture_from_args
from ma_slam_stream.viz import make_viz


def _finalize(slam, out, n, elapsed, fps, vram, mode, compute_fps=None, compute_s=None):
    """Write poses / merged PLY / stats / loops — mirrors MaSlam.run's tail, externally."""
    extra = ""
    if compute_fps:
        extra = (f" | reconstruction {compute_fps:.2f} fps (GPU only, {compute_s:.1f}s) "
                 f"— end-to-end is bounded by frame arrival, not the model")
    print(f"[ma_slam] {n} keyframes in {elapsed:.1f}s = {fps:.2f} fps end-to-end"
          f"{extra} | peak VRAM {vram:.1f} GB | submap_size={slam.cfg['submap_size']}")
    poses_txt = os.path.join(out, "camera_poses.txt")
    slam.map.write_poses(slam.graph, poses_txt)
    combined_ply = os.path.join(out, "combined_pcd.ply")
    pc = slam.cfg["Pointcloud"]
    slam.map.write_points(slam.graph, combined_ply, voxel_size=pc["voxel_size"],
                          max_points=pc["max_points"], conf_coef=pc["conf_coef"])
    print(f"[ma_slam] done. {n} frames | submaps={slam.map.num_submaps()} "
          f"loops={slam.graph.get_num_loops()} | poses -> {poses_txt}  pcd -> {combined_ply}")

    n_accept = sum(1 for ln in slam.loop_log
                   if ln.startswith("LOOPCAND ACCEPT") and "-DRY" not in ln)
    stats = {
        "backend": type(slam.model).__name__, "mode": mode,
        "submap_size": slam.cfg["submap_size"], "overlap": slam.cfg["overlap"],
        "n_frames": n, "n_submaps": slam.map.num_submaps(),
        "loops_accepted": n_accept, "loops_candidates": len(slam.loop_log),
        "fps_end_to_end": round(fps, 2),
        "fps_reconstruction": (round(compute_fps, 2) if compute_fps else None),
        "seconds": round(elapsed, 1), "compute_seconds": (round(compute_s, 1) if compute_s else None),
        "peak_vram_gb": round(vram, 2),
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
                viz=None, hooks=None):
    """Accumulate live keyframes into submaps and feed them to slam.process_submap.

    The producer (client / local source) already blur-selects AND runs the LK keyframe gate, so
    frames arriving here are keyframes; we just window them (``submap_size`` counts keyframes).

    Timing reports TWO numbers so a slow live feed doesn't look like a slow model:
      * end-to-end fps  = keyframes / wall-clock from the FIRST frame to the last — bounded by how
        fast keyframes arrive (camera motion, k, gate, network), NOT by the model.
      * reconstruction fps = keyframes / time actually spent inside process_submap (GPU only).
    """
    # the solver caches these on the instance (offline run() sets them the same way).
    slam._mode, slam._input_K, slam._depth_scale, slam._depth_max = (
        mode, intrinsics, depth_scale, depth_max)
    want_depth = "depth" in mode
    size = slam.cfg["submap_size"]
    overlap = slam.cfg["overlap"]

    print("[ma_slam] loading model...")
    _ = slam.model                                # force model load before timing
    print("\n[ma_slam] ===== READY — model loaded, viewer served. "
          "Start/allow the client to stream now. =====\n")
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    n_frames = 0
    carry: list = []                              # overlap frame(s) from the previous submap
    buf: list = []                                # frames accumulating toward the next submap
    loops_seen = [0]
    t_first = [None]                              # wall time of the 1st frame (excludes startup idle)
    compute_s = [0.0]                             # cumulative process_submap time (reconstruction only)

    def _process(window):
        rgb = [f[0] for f in window]
        dep = [f[1] for f in window] if want_depth else None
        if want_depth and any(d is None for d in dep):
            raise ValueError("depth mode but a frame arrived without a depth path")
        tc = time.time()
        slam.process_submap(rgb, dep)
        compute_s[0] += time.time() - tc          # count only the GPU/reconstruction work
        print(f"[ma_slam] submap {slam.stats['n_submaps']} (+{len(window)} frames, "
              f"{n_frames} total) | nodes={len(slam.graph.initialized)} "
              f"loops={slam.stats['n_loops']}")
        if viz is not None:
            loop_now = slam.stats["n_loops"]
            viz.update(slam, n_frames, loop_happened=loop_now > loops_seen[0])
            loops_seen[0] = loop_now
            # Instance-colored semantic layer (hooks ran inside process_submap).
            if hooks is not None and getattr(hooks, "table", None) is not None:
                viz.update_instances(hooks.table, n_frames)

    try:
        for frame in source:                      # blocks until next frame / stop sentinel
            if frame is None:
                break
            if t_first[0] is None:
                t_first[0] = time.time()          # start the clock at the FIRST real frame
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

    wall = (time.time() - t_first[0]) if t_first[0] else 0.0
    wall_fps = n_frames / wall if wall > 0 else 0.0
    compute_fps = n_frames / compute_s[0] if compute_s[0] > 0 else 0.0
    vram = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
    # Offline run() fires on_finish itself; the stream loop drives process_submap
    # directly, so fire it here (final re-lift + merge + reference-format export).
    if getattr(slam, "_unopt_loop_edges", 0) > 0:  # tier-B edge after last optimize
        slam.graph.optimize()
        slam._unopt_loop_edges = 0
    if hooks is not None:
        hooks.on_finish(slam.map, slam.graph)
    return _finalize(slam, out, n_frames, wall, wall_fps, vram, mode,
                     compute_fps=compute_fps, compute_s=compute_s[0])


def main():
    ap = argparse.ArgumentParser(description="ma_slam streaming (live RGBD -> incremental SLAM)")
    ap.add_argument("--source", default="local", choices=["local", "zmq", "folder"],
                    help="frame transport: 'local' camera here | 'zmq' socket push | 'folder' watch a dir")
    ap.add_argument("--bind", default="tcp://*:5599", help="[zmq] PULL bind address")
    ap.add_argument("--watch_dir", help="[folder] dir to watch (default <out>/live/incoming)")
    ap.add_argument("--poll", type=float, default=0.05, help="[folder] poll interval (s)")

    ap.add_argument("--mode", default="rgb", choices=list(MODES))
    ap.add_argument("--backend", default="da3", choices=["ma", "da3"],
                    help="backbone: 'ma' MapAnything | 'da3' DA3-nested (rgb/rgb+intr only)")
    ap.add_argument("--manifold", default="se3", choices=["se3", "sim3", "sl4"])
    ap.add_argument("--intrinsic",
                    help="K file; omit for 'local' (taken from the camera) or 'zmq' (META)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--submap_size", type=int, default=32)
    # --keyframe_disparity is a producer-side selection flag (with --k), defined in add_capture_args.
    ap.add_argument("--no_loop", action="store_true")
    ap.add_argument("--sim_threshold", type=float, default=DEFAULT_CONFIG["Loop"]["sim_threshold"])
    ap.add_argument("--max_per_submap", type=int, default=DEFAULT_CONFIG["Loop"]["max_per_submap"])
    ap.add_argument("--coloc_ratio", type=float, default=DEFAULT_CONFIG["Loop"]["coloc_ratio"])
    ap.add_argument("--loop_half_window", type=int, default=DEFAULT_CONFIG["Loop"]["half_window"])
    ap.add_argument("--rot_max_deg", type=float, default=DEFAULT_CONFIG["Loop"]["rot_max_deg"])
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

    # ── semantic mapping (ma_slam_semantic hooks; v0 = synchronous per submap) ──
    sp = ap.add_argument_group("semantic mapping")
    sp.add_argument("--semantic", action="store_true",
                    help="run the OpenSe3r query-instance semantic branch on every submap "
                         "(SemanticHooks) — instance-colored voxels/bboxes in the viewer + "
                         "reference-format instance export at stop. Adds ~0.6 s/frame of "
                         "GPU work, so it bounds throughput on fast feeds (v0 is synchronous).")
    sp.add_argument("--sem_chunk_size", type=int, default=4,
                    help="frames per causal semantic forward (VRAM knob; results invariant)")
    sp.add_argument("--conf_gate_coef", type=float, default=1.0,
                    help="scale on the per-submap mean-conf validity gate (bridge)")
    sp.add_argument("--kv_policy", default="h2o", choices=["none", "window", "h2o"],
                    help="CausalBranch KV retention. Default 'h2o' — a live stream is "
                         "unbounded, so the flat 'none' cache grows without limit "
                         "(~60 MB/frame); h2o keeps it constant (~0.7 GB).")
    sp.add_argument("--kv_window_size", type=int, default=None, help="[window] frames")
    sp.add_argument("--kv_keyframe_stride", type=int, default=None, help="[window] stride")
    sp.add_argument("--h2o_recent", type=int, default=8, help="[h2o] recent-window frames")
    sp.add_argument("--h2o_hh", type=int, default=8, help="[h2o] heavy-hitter frame-equivalents")
    sp.add_argument("--h2o_decay", type=float, default=0.9, help="[h2o] score decay γ")
    sp.add_argument("--h2o_subsample", type=float, default=0.25,
                    help="[h2o] query subsample for colsum scores (0.25 validated: "
                         "AP par with full scores, colsum cost ~gone)")
    # ── instance-LC (semantic-assisted loop candidates; needs --semantic) ──
    sp.add_argument("--sem_lc", default="veto", choices=["off", "shadow", "enforce", "veto"],
                    help="instance-bag gating of SALAD tier-B candidates: shadow = "
                         "log SEMLC verdicts only (default), enforce = gated tier-B "
                         "loops become real edges")
    sp.add_argument("--sem_lc_floor", type=float, default=0.45,
                    help="tier-B SALAD floor: candidates in [floor, sim_threshold) "
                         "go through the instance gate")
    sp.add_argument("--sem_lc_max", type=int, default=2,
                    help="max tier-B candidates per submap (coloc budget)")
    sp.add_argument("--bd_nmin", type=int, default=2,
                    help="bidesc: instances per side below this -> UNKNOWN")
    sp.add_argument("--bd_pair", type=float, default=0.85,
                    help="bidesc: MNN unique-match cosine (tau_pair)")
    sp.add_argument("--bd_dir", type=float, default=0.85,
                    help="bidesc: per-direction MeanTopK threshold (tau_dir)")
    sp.add_argument("--bd_bi", type=float, default=0.85,
                    help="bidesc: harmonic-mean threshold (tau_bi)")
    sp.add_argument("--bd_nuniq", type=int, default=2,
                    help="bidesc: min unique MNN matches for PASS")
    sp.add_argument("--bd_kbag", type=int, default=16,
                    help="bidesc: per-frame instance-id cap (by objectness)")
    sp.add_argument("--bd_desc_src", default="proposal", choices=["map", "proposal"],
                     help="bidesc current-side descriptors: map running-mean (id-based) or the frame's raw proposal descs (association-independent)")
    sp.add_argument("--sem_lc_coloc", type=float, default=0.50,
                    help="stricter coloc gate for tier-B edges; <=0 uses coloc_ratio")

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
                       max_per_submap=a.max_per_submap, coloc_ratio=a.coloc_ratio, half_window=a.loop_half_window,
                       rot_max_deg=a.rot_max_deg)
    cfg["Loop"]["Instance"].update(
        enable=(a.semantic and a.sem_lc != "off"), shadow=(a.sem_lc not in ("enforce", "veto")),
        veto=(a.sem_lc == "veto"),
        tier_b_floor=a.sem_lc_floor, max_tier_b=a.sem_lc_max,
        coloc_ratio_b=(a.sem_lc_coloc if a.sem_lc_coloc > 0 else None))
    cfg["Pointcloud"].update(voxel_size=a.voxel_size, max_points=a.max_points, conf_coef=a.conf_coef)

    model = None
    if a.backend == "da3":
        from model.da3_infer import Da3ChunkModel
        model = Da3ChunkModel(device=a.device)

    live_dir = os.path.join(a.out, "live")
    capture = capture_from_args(a, need_depth) if a.source == "local" else None
    source = make_source(a.source, live_dir=live_dir, want_depth=need_depth, bind=a.bind,
                         watch_dir=a.watch_dir, poll=a.poll, capture=capture, k=a.k,
                         keyframe_disparity=a.keyframe_disparity)

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

    # Semantic hooks (optional). Built BEFORE stream_loop so the semantic model's
    # checkpoint load happens during startup, not after the READY banner.
    hooks = None
    if a.semantic:
        from ma_slam_semantic.config import SemConfig
        from ma_slam_semantic.hooks import SemanticHooks
        sem_cfg = SemConfig(
            box_scale_mode     = "first_chunk",
            chunk_size         = a.sem_chunk_size,
            conf_gate_coef     = a.conf_gate_coef,
            kv_policy          = a.kv_policy,
            kv_window_size     = a.kv_window_size,
            kv_keyframe_stride = a.kv_keyframe_stride,
            h2o_recent_frames  = a.h2o_recent,
            h2o_hh_frames      = a.h2o_hh,
            h2o_decay          = a.h2o_decay,
            h2o_subsample      = a.h2o_subsample,
        )
        name = os.path.basename(os.path.normpath(a.out))
        from ma_slam_semantic.bidesc_lc import BiDescConfig
        lc_cfg = BiDescConfig(n_min=a.bd_nmin, tau_pair=a.bd_pair, tau_dir=a.bd_dir,
                              tau_bi=a.bd_bi, n_uniq=a.bd_nuniq, k_bag=a.bd_kbag,
                              desc_src=a.bd_desc_src)
        hooks = SemanticHooks(sem_cfg, a.out, name=name, lc_cfg=lc_cfg)
        print(f"[ma_slam] semantic hooks armed (kv_policy={a.kv_policy}, "
              f"chunk={a.sem_chunk_size}, sem_lc={a.sem_lc}) — instance layer in "
              f"viewer, export at stop")

    slam = MaSlam(config=cfg, model=model, device=a.device, hooks=hooks)
    res = stream_loop(slam, source, a.out, mode=a.mode, intrinsics=intrinsics,
                      depth_scale=depth_scale,
                      depth_max=(a.depth_max if a.depth_max > 0 else None),
                      dump_every=a.dump_every, viz=viz, hooks=hooks)
    print(f"[ma_slam] stream done: submaps={res['n_submaps']} loops={res['n_loops']} "
          f"fps={res['fps']:.2f} -> {res['poses_txt']}")


if __name__ == "__main__":
    main()
