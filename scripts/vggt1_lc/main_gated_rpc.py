"""VGGT-SLAM 1.0 (SL(4) or Sim3) with OUR instance-LC gate via RPC — runs in THEIR env (vggt_slam1).

Same two monkeypatches as main_gated.py (SALAD threshold + Solver.add_points pre-hook), but the
verdict is obtained from scripts/vggt1_lc/verify_server.py (amb3r_bw) over a Unix socket, so
gtsam-develop's SL(4) and our torch-2.8/py3.10 semantic stack never share a process.

    CUDA_VISIBLE_DEVICES=0 <vggt_slam1 python> scripts/vggt1_lc/main_gated_rpc.py --lc_thres 0.95 --sem_lc \\
        --socket /tmp/semlc.sock --semlc_log out/semlc_log.txt -- --image_folder ... --submap_size 32 ...
"""
import argparse
import glob
import os
import re
import runpy
import sys
import time
from multiprocessing.connection import Client

import numpy as np
import torch
import torch.hub as _hub
if hasattr(_hub, "_validate_not_a_forked_repo"):
    _hub._validate_not_a_forked_repo = lambda *a, **k: None

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V1 = os.path.join(ROOT, "thirdparty", "vggt-slam-1", "VGGT-SLAM")

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--lc_thres", type=float, default=0.80)
ap.add_argument("--sem_lc", action="store_true")
ap.add_argument("--socket", default="/tmp/semlc.sock")
ap.add_argument("--semlc_log", default="")
ours, rest = ap.parse_known_args()
if rest and rest[0] == "--":
    rest = rest[1:]
for i, a in enumerate(rest):
    if a in ("--image_folder", "--log_path") and i + 1 < len(rest):
        rest[i + 1] = os.path.abspath(rest[i + 1])
sys.argv = [os.path.join(V1, "main.py")] + rest
if V1 not in sys.path:
    sys.path.insert(0, V1)

import vggt_slam.loop_closure as _lc            # noqa: E402
import vggt_slam.solver as _sv                  # noqa: E402

_orig_find = _lc.ImageRetrieval.find_loop_closures


def _find_loop_closures(self, map, submap, max_similarity_thres=0.80, max_loop_closures=0):
    return _orig_find(self, map, submap, max_similarity_thres=ours.lc_thres, max_loop_closures=max_loop_closures)


_lc.ImageRetrieval.find_loop_closures = _find_loop_closures

_conn = Client(ours.socket, "AF_UNIX", authkey=b"semlc") if ours.sem_lc else None
if _conn is not None:
    _conn.send({"cmd": "reset_stats"}); _conn.recv()

_id2path = {}
if "--image_folder" in sys.argv:
    for f in glob.glob(os.path.join(sys.argv[sys.argv.index("--image_folder") + 1], "*")):
        m = re.search(r"\d+(?:\.\d+)?", os.path.basename(f))
        if m:
            _id2path[float(m.group())] = f


def _path_of(fid):
    return _id2path.get(float(fid), str(fid))


_log_f = open(ours.semlc_log, "a") if ours.semlc_log else None


def _log(line):
    print(line, flush=True)
    if _log_f is not None:
        _log_f.write(line + "\n"); _log_f.flush()


def _pack(x):
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    x = np.ascontiguousarray(x)
    return (x.tobytes(), x.shape, x.dtype.str)


_orig_add_points = _sv.Solver.add_points


def _add_points(self, pred_dict):
    loops = list(pred_dict.get("detected_loops", []))
    if _conn is not None and loops:
        sm = self.current_working_submap
        frames = sm.get_all_frames()
        base = sm.get_last_non_loop_frame_index()
        assert len(loops) == 1, "sem gate supports --max_loops 1"
        kept = []
        for index, loop in enumerate(loops):
            iq = int(loop.query_submap_frame)
            ir = base + index + 1
            q_path = _path_of(sm.get_frame_ids()[iq])
            r_path = _path_of(self.map.get_submap(loop.detected_submap_id).get_frame_ids()[loop.detected_submap_frame])
            t0 = time.time()
            _conn.send(dict(frames=_pack(frames), depth=_pack(pred_dict["depth"]), conf=_pack(pred_dict["depth_conf"]),
                            extr=_pack(pred_dict["extrinsic"]), K=_pack(pred_dict["intrinsic"]),
                            iq=iq, ir=ir, q_path=q_path, r_path=r_path, salad=float(loop.similarity_score)))
            rep = _conn.recv()
            ok, info = rep["accept"], rep["info"]
            _log(f"[semlc] {'ACCEPT' if ok else 'REJECT'} sub{loop.query_submap_id}:{iq}"
                 f"({os.path.basename(q_path)}) <-> sub{loop.detected_submap_id}:"
                 f"{loop.detected_submap_frame}({os.path.basename(r_path)}) salad_l2={loop.similarity_score:.3f}"
                 f" | S1 {info['s1']} | S2 {info['s2']} | {time.time()-t0:.2f}s")
            if ok:
                kept.append(loop)
        pred_dict["detected_loops"] = kept
    return _orig_add_points(self, pred_dict)


_sv.Solver.add_points = _add_points

os.chdir(V1)
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    if _conn is not None:
        _conn.send({"cmd": "stats"}); _log(f"[semlc] summary: {_conn.recv()}")
        _conn.close()
    if _log_f is not None:
        _log_f.close()
