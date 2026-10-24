"""Verification server: runs Vggt1SemVerifier inside amb3r_bw and answers requests over a Unix socket,
so VGGT-SLAM 1.0's SL(4) mode (gtsam-develop, py3.11 env) can use our instance-LC gate.

    PYTHONPATH=src:scripts CUDA_VISIBLE_DEVICES=0 python scripts/vggt1_lc/verify_server.py --socket /tmp/semlc.sock --mode pairs

Protocol (multiprocessing.connection, pickle): request = dict(frames, depth, conf, extr, K: numpy arrays
(sent as (bytes, shape, dtype) triples), iq, ir, q_path, r_path, salad) -> reply = dict(accept, info).
`{"cmd": "stats"}` returns the running stats; `{"cmd": "quit"}` stops the server.
"""
import argparse
import os
import sys
import numpy as np
from multiprocessing.connection import Listener

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from vggt1_lc.verifier import Vggt1SemVerifier  # noqa: E402


def arr(t):
    b, shape, dtype = t
    return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/tmp/semlc.sock")
    ap.add_argument("--mode", default="pairs", choices=["bidesc", "pairs", "rigid"])
    ap.add_argument("--ckpt", default="a1_pq")
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    ver = Vggt1SemVerifier(ckpt=a.ckpt, dump_dir=a.dump, mode=a.mode)
    if os.path.exists(a.socket):
        os.unlink(a.socket)
    with Listener(a.socket, "AF_UNIX", authkey=b"semlc") as srv:
        print(f"[verify_server] ready on {a.socket} mode={a.mode}", flush=True)
        while True:
            with srv.accept() as conn:
                while True:
                    try:
                        req = conn.recv()
                    except EOFError:
                        break
                    if req.get("cmd") == "quit":
                        print("[verify_server] quit", flush=True); return
                    if req.get("cmd") == "stats":
                        conn.send(dict(ver.stats)); continue
                    if req.get("cmd") == "reset_stats":
                        ver.stats = dict(n=0, accept=0, reject=0); conn.send(True); continue
                    frames = arr(req["frames"]); depth = arr(req["depth"]); conf = arr(req["conf"])
                    extr = arr(req["extr"]); K = arr(req["K"])
                    ok, info = ver.verify(frames, depth, conf, extr, K, int(req["iq"]), int(req["ir"]),
                                          q_path=req.get("q_path", ""), r_path=req.get("r_path", ""),
                                          salad=float(req.get("salad", float("nan"))))
                    conn.send(dict(accept=bool(ok), info=info))


if __name__ == "__main__":
    main()
