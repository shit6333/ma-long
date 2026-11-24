"""Per-component timing + peak-VRAM profiler for the full ma_slam_semantic pipeline (geo prediction,
Ident3R semantic mapping, loop closure), measured INSIDE the SLAM loop without touching core code.

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
    ident3r.forward          semantic model forward on the chunk (GPU)
    ident3r.proposals        mask NMS / per-pixel ownership / 3D footprints (map + transient LC bag)
    ident3r.associate        instance-table matching (Hungarian on the fused score)
    ident3r.other            bridge / tracker / frame-index bookkeeping inside on_submap
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
DELTA = defaultdict(float)       # max (peak - allocated at section entry): the section's own transient need
GLOB = dict(alloc=0.0, reserved=0.0, device=0.0, device_base=0.0)   # run-wide peaks (bytes)
TIMELINE: list = []              # per submap: allocated / reserved / persistent state by owner (bytes)
STACK: list = []                 # [(name, t0, running_peak_bytes)]
CUDA = torch.cuda.is_available()


def _sync():
    if CUDA:
        torch.cuda.synchronize()


def _glob():
    if CUDA:
        GLOB["alloc"] = max(GLOB["alloc"], torch.cuda.max_memory_allocated())
        GLOB["reserved"] = max(GLOB["reserved"], torch.cuda.max_memory_reserved())


def _device_sampler(stop):
    """device-wide used memory (what nvidia-smi shows), sampled; valid when the GPU is otherwise idle"""
    while not stop.is_set():
        try:
            free, total = torch.cuda.mem_get_info()
            GLOB["device"] = max(GLOB["device"], total - free)
        except Exception:
            pass
        stop.wait(0.05)


def _walk(obj, seen_obj, seen_st, skip=(), depth=0):
    """bytes of CUDA tensor storage reachable from obj (each storage counted once across the whole snapshot)"""
    import numpy as np
    if obj is None or depth > 12 or id(obj) in seen_obj or isinstance(obj, (str, bytes, int, float, bool, np.ndarray)):
        return 0
    if len(seen_obj) > 400000:
        return 0
    seen_obj.add(id(obj))
    if torch.is_tensor(obj):
        if not obj.is_cuda:
            return 0
        try:
            st = obj.untyped_storage(); key = st.data_ptr()
        except Exception:
            return 0
        if key in seen_st:
            return 0
        seen_st.add(key)
        return st.nbytes()
    n = 0
    if isinstance(obj, torch.nn.Module):
        for m in obj.modules():
            for v in list(m._parameters.values()) + list(m._buffers.values()):
                n += _walk(v, seen_obj, seen_st, skip, depth + 1)
            for k, v in m.__dict__.items():
                if k not in ("_parameters", "_buffers", "_modules"):
                    n += _walk(v, seen_obj, seen_st, skip, depth + 1)
        return n
    if isinstance(obj, dict):
        for v in obj.values():
            n += _walk(v, seen_obj, seen_st, skip, depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)) or type(obj).__name__ == "deque":
        for v in obj:
            n += _walk(v, seen_obj, seen_st, skip, depth + 1)
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if id(v) not in skip:
                n += _walk(v, seen_obj, seen_st, skip, depth + 1)
    return n


def _params(mod, seen_obj, seen_st):
    n = 0
    if isinstance(mod, torch.nn.Module):
        for v in list(mod.parameters()) + list(mod.buffers()):
            n += _walk(v, seen_obj, seen_st)
    return n


def _nets(slam):
    m = getattr(slam, "_model", None); net = getattr(m, "model", m)
    hk = getattr(slam, "hooks", None); sm = getattr(getattr(hk, "model", None), "model", None)
    rt = getattr(slam, "retrieval", None); det = getattr(rt, "model", None) or getattr(rt, "detector", None)
    return m, net, hk, sm, rt, det


def snapshot(slam):
    """who holds the persistent CUDA memory right now (after a submap): weights first, then state by owner"""
    if not CUDA or slam is None:
        return {}
    so, ss = set(), set()
    m, net, hk, sm, rt, det = _nets(slam)
    r = {"w.DA3": _params(net, so, ss), "w.Ident3R": _params(sm, so, ss), "w.SALAD": _params(det, so, ss)}
    if hk is not None:                                   # Ident3R state, by hooks attribute
        r["s.Ident3R.model (KV memory etc.)"] = _walk(getattr(hk, "model", None), so, ss)
        for k, v in vars(hk).items():
            if k != "model":
                b = _walk(v, so, ss)
                if b:
                    r[f"s.Ident3R.{k}"] = b
    r["s.retrieval (SALAD db)"] = _walk(rt, so, ss)
    r["s.geo backend (non-weight)"] = _walk(m, so, ss)
    r["s.slam (map / graph / submaps)"] = _walk(slam, so, ss)
    r["allocated"] = torch.cuda.memory_allocated(); r["reserved"] = torch.cuda.memory_reserved()
    r["unattributed"] = r["allocated"] - sum(v for k, v in r.items() if k[:2] in ("w.", "s."))
    return r


def section(name):
    def deco(fn):
        def wrapped(*a, **k):
            full = f"{STACK[-1][0]}/{name}" if STACK else name
            _sync()
            if CUDA:
                if STACK:                                   # fold the parent's peak so far before resetting
                    STACK[-1][2] = max(STACK[-1][2], torch.cuda.max_memory_allocated())
                _glob()
                torch.cuda.reset_peak_memory_stats()
            a0 = torch.cuda.memory_allocated() if CUDA else 0
            STACK.append([full, time.perf_counter(), a0, a0])
            try:
                return fn(*a, **k)
            finally:
                _sync()
                nm, t0, pk, a0 = STACK.pop()
                pk = max(pk, torch.cuda.max_memory_allocated()) if CUDA else 0
                _glob()
                TOT[nm] += time.perf_counter() - t0; CNT[nm] += 1; PEAK[nm] = max(PEAK[nm], pk)
                DELTA[nm] = max(DELTA[nm], pk - a0)
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
        out = section("submap")(orig_process)(self, image_paths, depth_paths)
        try:                                               # outside the timed section
            TIMELINE.append(snapshot(self))
        except Exception as e:
            print(f"[profile] snapshot failed: {e}")
        return out
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
    import ma_slam.keyframe_flow as KF
    KF.select_keyframes_by_disparity = section("keyframe_select")(KF.select_keyframes_by_disparity)
    H.SemanticHooks.on_submap = section("ident3r")(H.SemanticHooks.on_submap)
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
    import threading
    sys.argv = ["ma_slam_semantic.run"] + rest
    stop = threading.Event()
    if CUDA:
        free, total = torch.cuda.mem_get_info(); GLOB["device_base"] = total - free   # other processes + CUDA context
        threading.Thread(target=_device_sampler, args=(stop,), daemon=True).start()
    t_wall = time.perf_counter()
    try:
        R.main()
    finally:
        stop.set()
    wall = time.perf_counter() - t_wall
    _glob()

    slam = captured.get("slam")
    rep = {"sections": {}, "weights_gb": {}, "wall_s": wall}
    if slam is not None:
        m = getattr(slam, "_model", None)                  # the backend object (Da3ChunkModel / MaChunkModel)
        net = getattr(m, "model", m)                        # its torch nn.Module
        rep["weights_gb"]["geo (DA3)"] = weights_gb(net) if isinstance(net, torch.nn.Module) else float("nan")
        hk = getattr(slam, "hooks", None)
        sm = getattr(getattr(hk, "model", None), "model", None)
        rep["weights_gb"]["Ident3R"] = weights_gb(sm) if isinstance(sm, torch.nn.Module) else float("nan")
        rt = getattr(slam, "retrieval", None)
        det = getattr(rt, "model", None) or getattr(rt, "detector", None)
        rep["weights_gb"]["SALAD"] = weights_gb(det) if isinstance(det, torch.nn.Module) else float("nan")
        rep["n_keyframes"] = int(slam.stats.get("n_frames", 0)) if hasattr(slam, "stats") else None
    for k in TOT:
        rep["sections"][k] = dict(seconds=TOT[k], calls=CNT[k], peak_vram_gb=PEAK[k] / 1e9, delta_vram_gb=DELTA[k] / 1e9)
    rep["peak_gb"] = dict(allocated=GLOB["alloc"] / 1e9, reserved=GLOB["reserved"] / 1e9, device_used=GLOB["device"] / 1e9,
                          device_used_before_start=GLOB["device_base"] / 1e9)
    rep["timeline_gb"] = [{k: v / 1e9 for k, v in s.items()} for s in TIMELINE]
    json.dump(rep, open(ours.prof_out, "w"), indent=1)
    print(f"[profile] -> {ours.prof_out}")


if __name__ == "__main__":
    main()
