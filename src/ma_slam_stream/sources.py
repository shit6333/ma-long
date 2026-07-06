"""Server-side frame sources for ma_slam streaming.

A *frame source* is an iterable yielding ``(rgb_path, depth_path | None)`` tuples and stopping
(``None`` sentinel) when the producer signals end-of-stream. ``run_stream`` consumes one and
feeds the frames into the normal submap pipeline. Three transports:

* ``LocalRealSenseSource`` — camera attached to THIS machine; captures + blur-selects in a
  background thread and writes frames to ``<out>/live/``. No network.
* ``ZmqFrameSource`` — a laptop pushes encoded frames over ZeroMQ PUSH; a background thread
  ``recv``s and writes the bytes straight to ``<out>/live/`` (no decode here). Wire format::

      [b"FRAME", header_json, rgb_bytes, depth_bytes]   # depth_bytes empty in rgb-only modes
      [b"META",  header_json]                            # optional: intrinsics 3x3 + depth_scale
      [b"STOP"]

* ``FolderFrameSource`` — a laptop drops files (rsync/scp/NFS) into ``<watch>/rgb`` and
  ``<watch>/depth``; a polling thread picks up each new complete pair in order. The producer
  writes atomically (``*.tmp`` then rename) and creates a ``STOP`` file when done.

All expose the same ``__iter__`` contract plus ``.intrinsics`` / ``.depth_scale`` (populated by
``local`` from the camera and by ``zmq`` from a META message), so the solver is transport-agnostic.
"""

from __future__ import annotations

import glob
import json
import os
import queue
import threading
import time
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

Frame = Tuple[str, Optional[str]]


class _QueueSource:
    """Base: a background producer thread fills ``self._q``; ``__iter__`` drains it until None."""

    def __init__(self, maxsize: int = 256):
        self._q: "queue.Queue[Optional[Frame]]" = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.intrinsics: Optional[np.ndarray] = None
        self.depth_scale: Optional[float] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def __iter__(self) -> Iterator[Frame]:
        while True:
            item = self._q.get()
            if item is None:
                break
            yield item

    def close(self):
        self._stop.set()


# ---------------------------------------------------------------------- local (camera here)
class LocalRealSenseSource(_QueueSource):
    """Capture from a RealSense attached to this machine; blur-select; write frames to disk."""

    def __init__(self, live_dir: str, want_depth: bool, capture, k: int, maxsize: int = 64):
        super().__init__(maxsize=maxsize)
        self.capture = capture
        self.k = k
        self.want_depth = want_depth
        self.rgb_dir = os.path.join(live_dir, "rgb")
        self.depth_dir = os.path.join(live_dir, "depth")
        self.ir_dir = os.path.join(live_dir, "ir")
        os.makedirs(self.rgb_dir, exist_ok=True)
        if want_depth:
            os.makedirs(self.depth_dir, exist_ok=True)

    def start(self):
        self.capture.start()
        self.intrinsics = self.capture.K
        self.depth_scale = self.capture.depth_scale       # 1000.0 (mm)
        return super().start()

    def _run(self):
        from ma_slam_stream.realsense import blur_select
        try:
            idx = 0
            for f in blur_select(self.capture.frames(), self.k):
                if self._stop.is_set():
                    break
                rgb_path = os.path.join(self.rgb_dir, f"frame_{idx:06d}.jpg")
                cv2.imwrite(rgb_path, f["bgr"])
                depth_path = None
                if self.want_depth and f["depth_mm"] is not None:
                    depth_path = os.path.join(self.depth_dir, f"frame_{idx:06d}.png")
                    cv2.imwrite(depth_path, f["depth_mm"])
                if f.get("ir"):
                    os.makedirs(self.ir_dir, exist_ok=True)
                    for cam, img in f["ir"].items():
                        cv2.imwrite(os.path.join(self.ir_dir, f"frame_{idx:06d}_ir{cam}.png"), img)
                self._q.put((rgb_path, depth_path))         # blocks if full -> caps capture rate
                idx += 1
        finally:
            self.capture.stop()
            self._q.put(None)


# ---------------------------------------------------------------------- zmq (laptop pushes)
class ZmqFrameSource(_QueueSource):
    def __init__(self, bind: str, live_dir: str, want_depth: bool, maxsize: int = 256):
        super().__init__(maxsize=maxsize)
        self.bind = bind
        self.want_depth = want_depth
        self.rgb_dir = os.path.join(live_dir, "rgb")
        self.depth_dir = os.path.join(live_dir, "depth")
        os.makedirs(self.rgb_dir, exist_ok=True)
        if want_depth:
            os.makedirs(self.depth_dir, exist_ok=True)

    def _run(self):
        try:
            import zmq
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("ZmqFrameSource needs pyzmq: pip install pyzmq") from e
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PULL)
        sock.setsockopt(zmq.RCVHWM, 1000)     # buffer up to 1000 frames -> backpressure to laptop
        sock.bind(self.bind)
        print(f"[stream] ZMQ PULL bound at {self.bind}, waiting for frames...")
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        idx = 0
        while not self._stop.is_set():
            if not dict(poller.poll(timeout=200)):
                continue
            parts = sock.recv_multipart()
            tag = parts[0]
            if tag == b"STOP":
                print("[stream] STOP received")
                break
            if tag == b"META":
                meta = json.loads(parts[1].decode())
                if "intrinsics" in meta:
                    self.intrinsics = np.asarray(meta["intrinsics"], dtype=np.float32)
                if "depth_scale" in meta:
                    self.depth_scale = float(meta["depth_scale"])
                print(f"[stream] META: K set={self.intrinsics is not None} "
                      f"depth_scale={self.depth_scale}")
                continue
            if tag != b"FRAME":
                continue
            header = json.loads(parts[1].decode())
            ext = header.get("rgb_ext", "jpg")
            rgb_bytes = parts[2]
            depth_bytes = parts[3] if len(parts) > 3 else b""
            rgb_path = os.path.join(self.rgb_dir, f"frame_{idx:06d}.{ext}")
            with open(rgb_path, "wb") as f:
                f.write(rgb_bytes)
            depth_path = None
            if self.want_depth:
                if not depth_bytes:
                    print(f"[stream] frame {idx} missing depth in a depth mode; skipping")
                    continue
                depth_path = os.path.join(self.depth_dir, f"frame_{idx:06d}.png")
                with open(depth_path, "wb") as f:
                    f.write(depth_bytes)
            self._q.put((rgb_path, depth_path))
            idx += 1
        self._q.put(None)


# ---------------------------------------------------------------------- folder (laptop drops)
class FolderFrameSource(_QueueSource):
    """Poll ``<watch>/rgb`` (and ``<watch>/depth``) for new complete pairs, in sorted order."""

    def __init__(self, watch_dir: str, want_depth: bool, poll: float = 0.05, maxsize: int = 256):
        super().__init__(maxsize=maxsize)
        self.rgb_dir = os.path.join(watch_dir, "rgb")
        self.depth_dir = os.path.join(watch_dir, "depth")
        self.stop_file = os.path.join(watch_dir, "STOP")
        self.want_depth = want_depth
        self.poll = poll
        os.makedirs(self.rgb_dir, exist_ok=True)
        if want_depth:
            os.makedirs(self.depth_dir, exist_ok=True)

    @staticmethod
    def _stem(p: str) -> str:
        return os.path.splitext(os.path.basename(p))[0]

    def _run(self):
        emitted = set()
        print(f"[stream] watching {self.rgb_dir} (depth={self.want_depth}) ...")
        while not self._stop.is_set():
            rgbs = sorted(p for p in glob.glob(os.path.join(self.rgb_dir, "*"))
                          if not p.endswith(".tmp"))
            depth_map = {}
            if self.want_depth:
                depth_map = {self._stem(p): p
                             for p in glob.glob(os.path.join(self.depth_dir, "*"))
                             if not p.endswith(".tmp")}
            progressed = False
            for rgb in rgbs:
                stem = self._stem(rgb)
                if stem in emitted:
                    continue
                depth_path = None
                if self.want_depth:
                    depth_path = depth_map.get(stem)
                    if depth_path is None:
                        break                  # wait for this frame's depth before advancing
                self._q.put((rgb, depth_path))
                emitted.add(stem)
                progressed = True
            if os.path.exists(self.stop_file):
                print("[stream] STOP file seen")
                break
            if not progressed:
                time.sleep(self.poll)
        self._q.put(None)


def make_source(kind: str, *, live_dir: str, want_depth: bool, bind: str = "tcp://*:5599",
                watch_dir: Optional[str] = None, poll: float = 0.05,
                capture=None, k: int = 1) -> _QueueSource:
    if kind == "local":
        if capture is None:
            raise ValueError("source 'local' needs a RealSenseCapture (capture=...)")
        return LocalRealSenseSource(live_dir, want_depth, capture, k).start()
    if kind == "zmq":
        return ZmqFrameSource(bind=bind, live_dir=live_dir, want_depth=want_depth).start()
    if kind == "folder":
        return FolderFrameSource(watch_dir=watch_dir or os.path.join(live_dir, "incoming"),
                                 want_depth=want_depth, poll=poll).start()
    raise ValueError(f"unknown source kind: {kind!r} (expected 'local', 'zmq' or 'folder')")
