"""Incremental reconstruction snapshots — one record per submap step, replayable
offline into a video (scripts/render_snapshots.py).

Storage is split so the log stays small and loop-closure healing is visible:

    <dir>/submap_<base>.npz   written ONCE when a submap first appears:
        pts   (P,3) float32  submap-LOCAL points (conf-filtered, subsampled)
        cols  (P,3) uint8
        fidx  (P,)  int16    which frame each point came from
        L     (n,4,4)        local cam->world per frame
    <dir>/step_<k>.npz       written after EVERY submap (post-optimize):
        bases (S,)   int32   submaps alive at this step (non-LC), in order
        G_<base> (n,4,4)     optimized global cam->world per frame  -> global point
                             of frame i = (G_i @ inv(L_i)) @ local point
        inst_id/inst_center/inst_bbox_min/inst_bbox_max/inst_count   instance state
        vox (V,3) int16, vox_id (V,) int32   instance voxels (grid units, global frame)
        loops (E,2) int32    loop events so far (key_new, key_old)
        meta: step, n_frames, voxel_size, elapsed_s

Because poses are stored per step and points once, replaying step t with step
t's poses shows the map exactly as the system believed it at that moment —
including the jump when a loop closes.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Set

import numpy as np


class SnapshotWriter:
    def __init__(self, out_dir: str, max_pts_per_submap: int = 50000,
                 conf_coef: float = 1.0, fresh_instances: bool = True, seed: int = 0):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.max_pts = max_pts_per_submap
        self.conf_coef = conf_coef
        self.fresh = fresh_instances
        self.rng = np.random.default_rng(seed)
        self._saved: Set[int] = set()
        self.step = 0
        self.t0 = time.time()

    # ---------------------------------------------------------------- submap (once)
    def _save_submap(self, sm) -> None:
        thr = sm.conf_threshold * self.conf_coef
        pts, cols, fidx = [], [], []
        for i in range(sm.n):
            m = (sm.conf[i] > thr).reshape(-1)
            if sm.mask is not None:
                m &= sm.mask[i].reshape(-1)
            if not m.any():
                continue
            P = sm.points[i].reshape(-1, 3)[m]
            C = sm.colors[i].reshape(-1, 3)[m]
            pts.append(P); cols.append(C); fidx.append(np.full(len(P), i, np.int16))
        if pts:
            P = np.concatenate(pts).astype(np.float32)
            C = np.concatenate(cols).astype(np.uint8)
            F = np.concatenate(fidx)
            if len(P) > self.max_pts:
                sel = self.rng.choice(len(P), self.max_pts, replace=False)
                P, C, F = P[sel], C[sel], F[sel]
        else:
            P = np.zeros((0, 3), np.float32); C = np.zeros((0, 3), np.uint8)
            F = np.zeros((0,), np.int16)
        np.savez_compressed(os.path.join(self.dir, f"submap_{sm.base_id:06d}.npz"),
                            pts=P, cols=C, fidx=F,
                            L=np.asarray(sm.poses_local, np.float32),
                            frame_ids=np.asarray(sm.frame_ids, np.float64))
        self._saved.add(sm.base_id)

    # ---------------------------------------------------------------- step
    def write_step(self, gmap, graph, table=None, loop_events=None,
                   voxel_size: float = 0.05, n_frames: int = 0) -> str:
        mode = os.environ.get("MA_SNAP_MODE", "full")   # bisect aid: full|geo|poses
        rec: Dict[str, np.ndarray] = {}
        bases = []
        for sm in gmap.ordered():
            if sm.is_lc:
                continue
            if sm.base_id not in self._saved and mode != "poses":
                self._save_submap(sm)
            bases.append(sm.base_id)
            rec[f"G_{sm.base_id:06d}"] = np.stack(
                [graph.get_pose(sm.key(i)) for i in range(sm.n)]).astype(np.float32)
        rec["bases"] = np.asarray(bases, np.int32)

        if table is not None and len(table) and mode == "full":
            ids, cen, bmin, bmax, cnt, vox, vid = [], [], [], [], [], [], []
            for iid, r in table.records.items():
                if self.fresh:
                    table._ensure_fresh(r)
                ids.append(iid); cen.append(r.center.numpy()); cnt.append(r.count)
                bmin.append(r.bbox_min.numpy()); bmax.append(r.bbox_max.numpy())
                if r.voxels:
                    v = np.asarray(list(r.voxels), np.int16)
                    vox.append(v); vid.append(np.full(len(v), iid, np.int32))
            rec.update(inst_id=np.asarray(ids, np.int32),
                       inst_center=np.asarray(cen, np.float32),
                       inst_bbox_min=np.asarray(bmin, np.float32),
                       inst_bbox_max=np.asarray(bmax, np.float32),
                       inst_count=np.asarray(cnt, np.int32),
                       vox=(np.concatenate(vox) if vox else np.zeros((0, 3), np.int16)),
                       vox_id=(np.concatenate(vid) if vid else np.zeros((0,), np.int32)))
        rec["loops"] = (np.asarray(loop_events, np.int32).reshape(-1, 2)
                        if loop_events else np.zeros((0, 2), np.int32))
        rec["meta"] = np.array(json.dumps(dict(step=self.step, n_frames=int(n_frames),
                                               voxel_size=float(voxel_size),
                                               elapsed_s=time.time() - self.t0)))
        path = os.path.join(self.dir, f"step_{self.step:04d}.npz")
        np.savez_compressed(path, **rec)
        self.step += 1
        return path
