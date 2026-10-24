"""Per-component timing + peak-VRAM profiler for the full ma_slam_semantic pipeline (geo prediction,
Rememb3r semantic mapping, loop closure), measured INSIDE the SLAM loop without touching core code.

It monkeypatches the component entry points with synchronised, hierarchical timers and runs the normal
`ma_slam_semantic.run` CLI (all its flags pass through). Attribution is by call stack, so the DA3 joint
re-inference done by loop-closure verification is charged to loop closure, not to geo prediction.

  sections (inclusive, nested):
    geo.infer                 DA3 inference of the new submap
    geo.place                 submap build + overlap SE(3) placement (CPU)
    lc.retrieval              SALAD describe + DB query
    lc.inst_verify            Instance-Descriptor Loop Verification (bag MNN count gate)
    lc.coloc                  geometric verification — includes lc.coloc.infer (DA3 joint window)
    lc.optimize               pose-graph optimisation (gtsam)
    rememb3r.forward          semantic model forward on the chunk (GPU)
    rememb3r.proposals        mask NMS / per-pixel ownership / 3D footprints (map + transient LC bag)
    rememb3r.associate        instance-table matching (Hungarian on the fused score)
    rememb3r.other            bridge / tracker / frame-index bookkeeping inside on_submap
    export                    final merge + export (once, excluded from streaming fps)

Peak VRAM per section is torch.cuda.max_memory_allocated() within that section (children's peaks are
propagated to parents). Weight VRAM is reported per model (params + buffers).

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python -m ma_slam_semantic.profile_pipeline --prof_out prof.json \\
        -- --geometry ma --data data/scannet_val50/scene0568_00 --backend da3 --mode rgb ... --out /tmp/x
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict

import torch

TOT = defaultdict(float)         # inclusive seconds
CNT = defaultdict(int)
PEAK = defaultdict(float)        # max peak bytes observed within the section
STACK: list = []                 # [(name, t0, running_peak_bytes)]
CUDA = torch.cuda.is_available()


def _sync():
    if CUDA:
        torch.cuda.synchronize()


def section(name):
    def deco(fn):
        def wrapped(*a, **k):
            full = f"{STACK[-1][0]}/{name}" if STACK else name
            _sync()
            if CUDA:
                if STACK:                                   # fold the parent's peak so far before resetting
                    STACK[-1][2] = max(STACK[-1][2], torch.cuda.max_memory_allocated())
                torch.cuda.reset_peak_memory_stats()
            STACK.append([full, time.perf_counter(), torch.cuda.memory_allocated() if CUDA else 0])
            try:
                return fn(*a, **k)
            finally:
                _sync()
                nm, t0, pk = STACK.pop()
                pk = max(pk, torch.cuda.max_memory_allocated()) if CUDA else 0
                TOT[nm] += time.perf_counter() - t0; CNT[nm] += 1; PEAK[nm] = max(PEAK[nm], pk)
                if STACK:
                    STACK[-1][2] = max(STACK[-1][2], pk)
        wrapped.__wrapped__ = fn
        return wrapped
    return deco


def install():
    import ma_slam.solver as S
    import ma_slam.graph as G
    import ma_slam.loop as L
    import ma_slam_semantic.hooks as H
    import ma_slam_semantic.sem_model as M
    import ma_slam_semantic.instance_map as I

    S.MaSlam._infer = section("infer")(S.MaSlam._infer)
    orig_process = S.MaSlam.process_submap

    def process_submap(self, image_paths, depth_paths):
        return section("submap")(orig_process)(self, image_paths, depth_paths)
    S.MaSlam.process_submap = process_submap
    S.MaSlam._build_submap = section("place")(S.MaSlam._build_submap)
    S.MaSlam._add_submap = section("place")(S.MaSlam._add_submap)
    S.MaSlam._loop_closure = section("lc")(S.MaSlam._loop_closure)
    S.MaSlam._instance_lc_pass = section("lc")(S.MaSlam._instance_lc_pass)
    S.MaSlam._add_loop_constraint = section("coloc")(S.MaSlam._add_loop_constraint)
    for cls in (G.PoseGraph, G.Sim3PoseGraph):
        cls.optimize = section("optimize")(cls.optimize)
    L.ImageRetrieval.describe = section("retrieval")(L.ImageRetrieval.describe)
    L.ImageRetrieval.query = section("retrieval")(L.ImageRetrieval.query)
    H.SemanticHooks.on_submap = section("rememb3r")(H.SemanticHooks.on_submap)
    H.SemanticHooks.verify_loop_candidate = section("inst_verify")(H.SemanticHooks.verify_loop_candidate)
    H.SemanticHooks.on_finish = section("export")(H.SemanticHooks.on_finish)
    M.SemanticModel.forward_chunk = section("forward")(M.SemanticModel.forward_chunk)
    H.make_proposals = section("proposals")(H.make_proposals)
    I.InstanceTable.match_chunk = section("associate")(I.InstanceTable.match_chunk)


def weights_gb(module) -> float:
    try:
        return sum(p.numel() * p.element_size() for p in module.parameters()) / 1e9 + \
               sum(b.numel() * b.element_size() for b in module.buffers()) / 1e9
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--prof_out", required=True)
    ours, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    install()

    import ma_slam.solver as S
    captured = {}
    orig_init = S.MaSlam.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        captured["slam"] = self
    S.MaSlam.__init__ = init

    import ma_slam_semantic.run as R
    sys.argv = ["ma_slam_semantic.run"] + rest
    t_wall = time.perf_counter()
    R.main()
    wall = time.perf_counter() - t_wall

    slam = captured.get("slam")
    rep = {"sections": {}, "weights_gb": {}, "wall_s": wall}
    if slam is not None:
        m = getattr(slam, "_model", None)                  # the backend object (Da3ChunkModel / MaChunkModel)
        net = getattr(m, "model", m)                        # its torch nn.Module
        rep["weights_gb"]["geo (DA3)"] = weights_gb(net) if isinstance(net, torch.nn.Module) else float("nan")
        hk = getattr(slam, "hooks", None)
        sm = getattr(getattr(hk, "model", None), "model", None)
        rep["weights_gb"]["rememb3r"] = weights_gb(sm) if isinstance(sm, torch.nn.Module) else float("nan")
        rt = getattr(slam, "retrieval", None)
        det = getattr(rt, "model", None) or getattr(rt, "detector", None)
        rep["weights_gb"]["SALAD"] = weights_gb(det) if isinstance(det, torch.nn.Module) else float("nan")
        rep["n_keyframes"] = int(slam.stats.get("n_frames", 0)) if hasattr(slam, "stats") else None
    for k in TOT:
        rep["sections"][k] = dict(seconds=TOT[k], calls=CNT[k], peak_vram_gb=PEAK[k] / 1e9)
    json.dump(rep, open(ours.prof_out, "w"), indent=1)
    print(f"[profile] -> {ours.prof_out}")


if __name__ == "__main__":
    main()
