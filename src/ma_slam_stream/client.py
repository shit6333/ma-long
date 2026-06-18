"""Laptop-side capture client for ma_slam streaming.

Runs on the machine with the RealSense camera (the laptop). Captures aligned RGBD via the
official RealSense API (``realsense.RealSenseCapture`` — IR / emitter / post-processing
filters), does Laplacian blur selection locally (keep the sharpest of every ``k`` frames),
and sends the selected frames to the server's ``ma_slam_stream.run_stream``.

Standalone on the laptop: copy ``realsense.py`` + ``client.py`` over. Depends only on
``pyrealsense2``, ``opencv-python``, ``numpy`` and — for the zmq transport — ``pyzmq``.
No repo / GPU / model needed on the laptop.

    # ZeroMQ: push to the server (server runs run_stream --source zmq)
    python client.py --transport zmq --addr tcp://SERVER_IP:5599 --mode rgb+depth+intr --k 3

    # Folder: write into a dir the server watches (rsync/NFS target; server --source folder)
    python client.py --transport folder --out_dir /mnt/share/incoming --mode rgb+depth+intr --k 3

``--mode`` only decides whether depth/intrinsics are sent; the server's ``--mode`` must match.
Depth is normalised to uint16 mm, so the server keeps ``--depth_scale 1000`` (its default).
Stop with Ctrl+C — an end-of-stream marker is sent so the server flushes and writes outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys

import cv2
import numpy as np

# allow running both as `python client.py` (copied next to realsense.py) and `-m ma_slam_stream.client`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from realsense import add_capture_args, capture_from_args, blur_select
except ImportError:
    from ma_slam_stream.realsense import add_capture_args, capture_from_args, blur_select


class ZmqSender:
    def __init__(self, addr: str):
        import zmq
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, 1000)
        self.sock.connect(addr)
        print(f"[client] ZMQ PUSH connected to {addr}")

    def send_meta(self, K, depth_scale):
        meta = {"intrinsics": np.asarray(K).tolist(), "depth_scale": depth_scale}
        self.sock.send_multipart([b"META", json.dumps(meta).encode()])

    def send_frame(self, idx, rgb_bytes, depth_bytes):
        header = json.dumps({"idx": idx, "rgb_ext": "jpg"}).encode()
        self.sock.send_multipart([b"FRAME", header, rgb_bytes, depth_bytes or b""])

    def send_stop(self):
        self.sock.send_multipart([b"STOP"])


class FolderSender:
    """Write frames atomically (``*.tmp`` then rename) into ``out_dir/{rgb,depth}``."""

    def __init__(self, out_dir: str, want_depth: bool):
        self.out_dir = out_dir
        self.rgb_dir = os.path.join(out_dir, "rgb")
        self.depth_dir = os.path.join(out_dir, "depth")
        self.stop_file = os.path.join(out_dir, "STOP")
        self.want_depth = want_depth
        os.makedirs(self.rgb_dir, exist_ok=True)
        if want_depth:
            os.makedirs(self.depth_dir, exist_ok=True)
        if os.path.exists(self.stop_file):       # clear a stale STOP from a previous run
            os.remove(self.stop_file)
        print(f"[client] writing frames into {out_dir}")

    @staticmethod
    def _atomic_write(path, data):
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.rename(tmp, path)

    def send_meta(self, K, depth_scale):
        meta = {"intrinsics": np.asarray(K).tolist(), "depth_scale": depth_scale}
        self._atomic_write(os.path.join(self.out_dir, "meta.json"),
                           json.dumps(meta, indent=2).encode())

    def send_frame(self, idx, rgb_bytes, depth_bytes):
        # write depth FIRST so the server (which waits for depth) never sees rgb without it.
        if self.want_depth and depth_bytes:
            self._atomic_write(os.path.join(self.depth_dir, f"frame_{idx:06d}.png"), depth_bytes)
        self._atomic_write(os.path.join(self.rgb_dir, f"frame_{idx:06d}.jpg"), rgb_bytes)

    def send_stop(self):
        open(self.stop_file, "w").close()


def main():
    ap = argparse.ArgumentParser(description="RealSense capture + blur-select + stream to ma_slam")
    ap.add_argument("--transport", default="zmq", choices=["zmq", "folder"])
    ap.add_argument("--addr", default="tcp://127.0.0.1:5599", help="[zmq] server PULL address")
    ap.add_argument("--out_dir", help="[folder] dir the server watches")
    ap.add_argument("--mode", default="rgb+depth+intr",
                    help="must match the server; decides whether depth/intrinsics are sent")
    ap.add_argument("--jpeg_quality", type=int, default=90)
    ap.add_argument("--max_frames", type=int, default=0,
                    help="stop after sending this many selected frames (0 = until Ctrl+C)")
    add_capture_args(ap)                          # --width/--height/--fps/--k/--enable_ir/...
    a = ap.parse_args()

    want_depth = "depth" in a.mode
    want_intr = ("intr" in a.mode) or want_depth

    if a.transport == "zmq":
        sender = ZmqSender(a.addr)
    else:
        if not a.out_dir:
            ap.error("--out_dir required for folder transport")
        sender = FolderSender(a.out_dir, want_depth)

    cap = capture_from_args(a, want_depth)
    cap.start()
    if want_intr:
        ci = cap.K
        sender.send_meta(cap.K, cap.depth_scale)
        print(f"[client] intrinsics fx={ci[0,0]:.1f} fy={ci[1,1]:.1f} cx={ci[0,2]:.1f} "
              f"cy={ci[1,2]:.1f} | depth sent as uint16 mm (server depth_scale={cap.depth_scale})")

    running = {"go": True}
    signal.signal(signal.SIGINT, lambda *_: running.__setitem__("go", False))

    sent = 0
    print(f"[client] capturing (k={a.k}, ir={a.enable_ir}, postprocess={not a.no_postprocess}); "
          f"Ctrl+C to stop")
    try:
        for f in blur_select(cap.frames(), a.k):
            if not running["go"]:
                break
            ok, rgb_buf = cv2.imencode(".jpg", f["bgr"], [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])
            if not ok:
                continue
            depth_buf = b""
            if want_depth and f["depth_mm"] is not None:
                okd, dbuf = cv2.imencode(".png", f["depth_mm"])
                depth_buf = dbuf.tobytes() if okd else b""
            sender.send_frame(sent, rgb_buf.tobytes(), depth_buf)
            sent += 1
            if sent % 10 == 0:
                print(f"[client] sent {sent} frames (last sharpness {f.get('sharpness', 0):.0f})")
            if a.max_frames and sent >= a.max_frames:
                break
    finally:
        sender.send_stop()
        cap.stop()
        print(f"[client] stopped. sent {sent} selected frames; STOP signalled.")


if __name__ == "__main__":
    main()
