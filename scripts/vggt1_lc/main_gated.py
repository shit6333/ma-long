"""VGGT-SLAM 1.0 (Sim3) with OUR semantic loop verification plugged in — single process.

Runs their untouched `main.py` inside amb3r_bw (see run_vggt1_in_amb3r.py for the env
shims) with two monkeypatches:
  1. `ImageRetrieval.find_loop_closures` gets `--lc_thres` (their SALAD L2-distance
     gate is hardcoded 0.80; 0.95 = relaxed / recall-first, as VGGT-SLAM 2.0 does).
  2. `Solver.add_points` pre-hook: every SALAD candidate is verified by
     `Vggt1SemVerifier` (map-free S1 Bi-Desc + S2 layout graph on the two frames of
     the candidate, straight from their joint VGGT pass); rejected loops are removed
     from `pred_dict["detected_loops"]` BEFORE their loop factor is added.

Arms:  A = --lc_thres 0.80 (their default)   B = --lc_thres 0.95 (relaxed, no gate)
       C = --lc_thres 0.95 --sem_lc          (relaxed + our gate)
Everything after our flags is passed to their main.py verbatim.

    PYTHONPATH=src python scripts/vggt1_lc/main_gated.py --lc_thres 0.95 --sem_lc \\
        --lc_dump out/semlc_dump --semlc_log out/semlc_log.txt -- \\
        --image_folder ... --use_sim3 --submap_size 32 --min_disparity 25 --log_results --log_path out/poses.txt
"""
import argparse
import os
import runpy
import sys
import time
import types
from unittest import mock

import torch
import torch.hub as _hub
if hasattr(_hub, "_validate_not_a_forked_repo"):      # GitHub validation hangs when rate-limited
    _hub._validate_not_a_forked_repo = lambda *a, **k: None

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = os.path.join(ROOT, "thirdparty", "vggt-slam-1", "VGGT-SLAM")

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--lc_thres", type=float, default=0.80, help="SALAD L2 gate (their default 0.80)")
ap.add_argument("--sem_lc", action="store_true", help="enable our S1+S2 verifier")
ap.add_argument("--bd_s2", choices=["on", "off"], default="on")
ap.add_argument("--sem_ckpt", default="a1_pq")
ap.add_argument("--tau_t_rel", type=float, default=0.5, help="S2 relative t gate (x median inst depth)")
ap.add_argument("--lc_dump", default="", help="per-candidate JSON dir (render_semlc_graph.py)")
ap.add_argument("--semlc_log", default="", help="append per-candidate verdict lines here")
ours, rest = ap.parse_known_args()
if rest and rest[0] == "--":
    rest = rest[1:]
# their main.py runs with cwd=V1 (hub ckpt paths) -> absolutize our path args first
for i, a in enumerate(rest):
    if a in ("--image_folder", "--log_path") and i + 1 < len(rest):
        rest[i + 1] = os.path.abspath(rest[i + 1])
sys.argv = [os.path.join(V1, "main.py")] + rest

# ------------------------------------------------------------------ env shims (== run_vggt1_in_amb3r.py)
for p in (os.path.join(ROOT, "vendor", "thirdparty"), os.path.join(V1, "third_party", "salad"), V1,
          os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
for m in ("viser", "viser.transforms", "gradio", "trimesh",
          "pytorch_metric_learning", "pytorch_metric_learning.losses",
          "pytorch_metric_learning.miners", "pytorch_metric_learning.distances",
          "pytorch_metric_learning.reducers", "prettytable"):
    sys.modules.setdefault(m, mock.MagicMock(name=m))


class _LightningShim(torch.nn.Module):
    def save_hyperparameters(self, *a, **k): pass
    def log(self, *a, **k): pass


pl = types.ModuleType("pytorch_lightning")
pl.LightningModule = _LightningShim
sys.modules.setdefault("pytorch_lightning", pl)

# ------------------------------------------------------------------ patch 1: retrieval threshold
import vggt_slam.loop_closure as _lc            # noqa: E402
import vggt_slam.solver as _sv                  # noqa: E402

_orig_find = _lc.ImageRetrieval.find_loop_closures


def _find_loop_closures(self, map, submap, max_similarity_thres=0.80, max_loop_closures=0):
    return _orig_find(self, map, submap, max_similarity_thres=ours.lc_thres,
                      max_loop_closures=max_loop_closures)


_lc.ImageRetrieval.find_loop_closures = _find_loop_closures

# ------------------------------------------------------------------ patch 2: verifier gate
_verifier = None
if ours.sem_lc:
    from vggt1_lc.verifier import Vggt1SemVerifier   # scripts/vggt1_lc on path via scripts/
    _verifier = Vggt1SemVerifier(ckpt=ours.sem_ckpt, dump_dir=ours.lc_dump,
                                 s2_enable=(ours.bd_s2 == "on"), tau_t_rel=ours.tau_t_rel)

# their Submap.frame_ids are the numbers parsed from filenames -> map back to paths
_id2path = {}
if "--image_folder" in sys.argv:
    import glob as _glob, re as _re
    for f in _glob.glob(os.path.join(sys.argv[sys.argv.index("--image_folder") + 1], "*")):
        m = _re.search(r"\d+(?:\.\d+)?", os.path.basename(f))
        if m:
            _id2path[float(m.group())] = f


def _path_of(fid):
    return _id2path.get(float(fid), str(fid))


_orig_add_points = _sv.Solver.add_points
_log_f = open(ours.semlc_log, "a") if ours.semlc_log else None


def _log(line):
    print(line, flush=True)
    if _log_f is not None:
        _log_f.write(line + "\n"); _log_f.flush()


def _add_points(self, pred_dict):
    loops = list(pred_dict.get("detected_loops", []))
    if _verifier is not None and loops:
        sm = self.current_working_submap
        frames = sm.get_all_frames()                   # (S+n,3,H,W) [0,1] torch, joint batch
        base = sm.get_last_non_loop_frame_index()
        # Dropping a loop shifts the index bookkeeping of the ones after it
        # (loop_index = base + index + 1) -> only the max_loops=1 regime is supported.
        assert len(loops) == 1, "sem gate supports --max_loops 1 (VGGT-SLAM's default)"
        kept = []
        for index, loop in enumerate(loops):
            iq = int(loop.query_submap_frame)
            ir = base + index + 1
            q_path = _path_of(sm.get_frame_ids()[iq])
            r_path = _path_of(self.map.get_submap(loop.detected_submap_id).get_frame_ids()[loop.detected_submap_frame])
            t0 = time.time()
            ok, info = _verifier.verify(frames, pred_dict["depth"], pred_dict["depth_conf"],
                                        pred_dict["extrinsic"], pred_dict["intrinsic"],
                                        iq, ir, q_path=q_path, r_path=r_path,
                                        salad=float(loop.similarity_score))
            _log(f"[semlc] {'ACCEPT' if ok else 'REJECT'} sub{loop.query_submap_id}:{iq}"
                 f"({os.path.basename(q_path)}) <-> sub{loop.detected_submap_id}:"
                 f"{loop.detected_submap_frame}({os.path.basename(r_path)}) salad_l2={loop.similarity_score:.3f}"
                 f" | S1 {info['s1']} | S2 {info['s2']} | {time.time()-t0:.2f}s")
            if ok:
                kept.append(loop)
        pred_dict["detected_loops"] = kept
    return _orig_add_points(self, pred_dict)


_sv.Solver.add_points = _add_points

# ------------------------------------------------------------------ run their main
os.chdir(V1)
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    if _verifier is not None:
        _log(f"[semlc] summary: {_verifier.stats}")
    if _log_f is not None:
        _log_f.close()
