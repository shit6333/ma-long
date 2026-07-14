"""Provenance-based instance map — an LC-safe port of OpenSe3r's ``InstanceTable``.

Reference behaviour (``slam_opense3r/instance_table.py``) is preserved exactly on the
hot path: records keep the SAME incremental caches (running-mean descriptor/sem_query/
center, bbox min/max union, voxel-set union with votes) and matching reuses the SAME
scoring kernel (``_proposal_pair_score_dict``) + per-frame Hungarian assignment.

The difference is *provenance*: every merge also appends an immutable, pose-free
``Obs`` (frame key + head-res mask + slot + objectness). The incremental geometry
caches are tagged with a signature of the owning frames' current placements
``M_key`` (quantized at voxel_size/2). While geometry is static (golden GT mode, or
ma_slam between loop closures) the signature never changes and behaviour is
identical to the reference. When a loop closure moves owning frames, the record
re-lifts ALL its observations through the healed geometry (same
``lift_proposal_to_3d`` code path proposals were born from) — center/bbox/voxels
snap to the corrected map instead of staying corrupted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from ma_slam_semantic.config import SemConfig


# --------------------------------------------------------------------------- provenance
@dataclass
class Obs:
    """One observation of an instance: immutable and pose-free."""
    key: int                      # frame key (graph node id / global frame ordinal)
    slot: int                     # query-head slot that produced the mask
    objectness: float
    mask_bits: np.ndarray         # np.packbits of the head-res bool mask
    mask_shape: Tuple[int, int]

    @classmethod
    def from_mask(cls, key: int, slot: int, objectness: float, mask: torch.Tensor) -> "Obs":
        m = mask.detach().cpu().numpy().astype(bool)
        return cls(key=key, slot=slot, objectness=objectness,
                   mask_bits=np.packbits(m), mask_shape=tuple(m.shape))

    def mask(self) -> torch.Tensor:
        h, w = self.mask_shape
        m = np.unpackbits(self.mask_bits, count=h * w).reshape(h, w).astype(bool)
        return torch.from_numpy(m)


# --------------------------------------------------------------------------- geometry resolvers
class StaticGeoResolver:
    """Golden GT mode: reference geometry IS global and never moves (M == I)."""

    def __init__(self):
        self._pts: Dict[int, torch.Tensor] = {}       # key -> [H_f,W_f,3] float16 global

    def register(self, key: int, pts_global: torch.Tensor) -> None:
        self._pts[key] = pts_global.detach().cpu().to(torch.float16)

    def points_global(self, key: int, hw=None) -> torch.Tensor:
        return self._pts[key].float()

    def sig(self, keys: Sequence[int]) -> bytes:
        return b"static"


class SlamGeoResolver:
    """ma_slam mode: reference geometry is submap-LOCAL; placement is read off the
    CURRENT pose graph as ``M_key = G_key @ inv(L_key)`` (the 'fog' gotcha)."""

    def __init__(self, gmap, graph, res: Tuple[int, int]):
        self.gmap, self.graph, self.res = gmap, graph, res
        self._bases: List[int] = []

    def _submap_frame(self, key: int):
        import bisect
        bases = sorted(self.gmap.submaps)
        i = bisect.bisect_right(bases, key) - 1
        sm = self.gmap.submaps[bases[i]]
        return sm, key - sm.base_id

    def M(self, key: int) -> np.ndarray:
        sm, i = self._submap_frame(key)
        return self.graph.get_pose(key) @ np.linalg.inv(sm.poses_local[i])

    def points_global(self, key: int, hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Frame ``key``'s point map at head resolution, through the CURRENT graph.

        Mirrors the production proposal path exactly: crop/nearest-resize the
        submap-LOCAL points to the bridge resolution, bilinear-downsample to the
        head resolution if it differs, then apply M. (Rigid transforms commute
        with interpolation, so M(resize(P)) == resize(M(P)) — bit-identical to
        what the proposals were born from.)"""
        from ma_slam_semantic.bridge import _crop_resize_map, _apply_T
        sm, i = self._submap_frame(key)
        P = torch.from_numpy(_crop_resize_map(sm.points[i].astype(np.float32), self.res))
        if hw is not None and tuple(P.shape[:2]) != tuple(hw):
            P = torch.nn.functional.interpolate(
                P.permute(2, 0, 1).unsqueeze(0), size=tuple(hw),
                mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        return torch.from_numpy(_apply_T(self.M(key), P.numpy()))

    def sig(self, keys: Sequence[int]) -> bytes:
        # translation quantized at half the instance voxel (re-key only when an
        # owning frame moved enough to shift its voxels); rotation at ~0.01 rad.
        parts = []
        for k in sorted(set(keys)):
            M = self.M(k)
            parts.append(np.round(M[:3, 3] / 0.025).astype(np.int64).tobytes())
            parts.append(np.round(M[:3, :3] * 100).astype(np.int64).tobytes())
        import hashlib
        return hashlib.blake2b(b"".join(parts), digest_size=16).digest()


# --------------------------------------------------------------------------- record
class InstanceRecord:
    def __init__(self, inst_id: int, prop: Dict, key: int, obs: Obs):
        self.inst_id = inst_id
        self.descriptor: torch.Tensor = prop["descriptor"].detach().cpu().clone()
        self.sem_query: Optional[torch.Tensor] = (
            prop["sem_query"].detach().cpu().clone() if "sem_query" in prop else None)
        self.count = 1
        self.first_seen = key
        self.last_seen = key
        self.member_objectness: List[float] = [float(prop.get("objectness", 0.0))]
        self.obs: List[Obs] = [obs]
        # incremental geometry caches (identical math to the reference table)
        self.center: torch.Tensor = prop["center"].detach().cpu().clone()
        self.bbox_min: torch.Tensor = prop["bbox_min"].detach().cpu().clone()
        self.bbox_max: torch.Tensor = prop["bbox_max"].detach().cpu().clone()
        self.voxels: Set[Tuple[int, int, int]] = set(prop["voxels"])
        self.voxel_votes: Dict[Tuple[int, int, int], int] = {v: 1 for v in prop["voxels"]}
        self._geo_sig: Optional[bytes] = None         # set by the table after updates

    # ---------------------------------------------------------------- features
    def merge(self, prop: Dict, key: int, obs: Obs,
              embed_update: str = "mean", embed_momentum: float = 0.75) -> None:
        n = self.count
        a, b = n / (n + 1), 1.0 / (n + 1)
        new_desc = prop["descriptor"].detach().cpu()
        if embed_update == "sim_guided":
            beta = max(0.0, float((self.descriptor * new_desc).sum()))
            da, db = 1.0 - beta, beta
        elif embed_update == "momentum":
            da, db = 1.0 - embed_momentum, embed_momentum
        else:  # 'mean'
            da, db = a, b
        self.descriptor = torch.nn.functional.normalize(
            da * self.descriptor + db * new_desc, dim=-1)
        if self.sem_query is not None and "sem_query" in prop:
            self.sem_query = torch.nn.functional.normalize(
                a * self.sem_query + b * prop["sem_query"].detach().cpu(), dim=-1)
        # incremental geometry (valid while the sig is unchanged)
        self.center = a * self.center + b * prop["center"].detach().cpu()
        self.bbox_min = torch.minimum(self.bbox_min, prop["bbox_min"].detach().cpu())
        self.bbox_max = torch.maximum(self.bbox_max, prop["bbox_max"].detach().cpu())
        for v in prop["voxels"]:
            self.voxels.add(v)
            self.voxel_votes[v] = self.voxel_votes.get(v, 0) + 1
        self.count += 1
        self.last_seen = key
        self.member_objectness.append(float(prop.get("objectness", 0.0)))
        self.obs.append(obs)

    # ---------------------------------------------------------------- derived
    @property
    def num_voxels(self) -> int:
        return len(self.voxels)

    @property
    def mean_objectness(self) -> float:
        return float(sum(self.member_objectness) / max(len(self.member_objectness), 1))

    @property
    def max_objectness(self) -> float:
        return float(max(self.member_objectness)) if self.member_objectness else 0.0

    @property
    def bbox_volume(self) -> float:
        return float((self.bbox_max - self.bbox_min).clamp(min=0).prod().item())

    def owning_keys(self) -> List[int]:
        return [o.key for o in self.obs]

    # duck-type compatibility with the reference export writers
    @property
    def member_frames(self) -> List[int]:
        return [o.key for o in self.obs]

    @property
    def member_slots(self) -> List[int]:
        return [o.slot for o in self.obs]

    def relift(self, resolver, cfg: SemConfig) -> None:
        """Rebuild center/bbox/voxels from provenance through CURRENT geometry
        (same lift code path the proposals were born from)."""
        from opense3r.instance_post import lift_proposal_to_3d
        centers, votes = [], {}
        bmin = torch.full((3,), float("inf"))
        bmax = torch.full((3,), float("-inf"))
        for o in self.obs:
            lifted = lift_proposal_to_3d(
                o.mask(), resolver.points_global(o.key, o.mask_shape),
                voxel_size=cfg.voxel_size, outlier_iqr_mult=cfg.outlier_iqr_mult)
            if lifted is None:
                continue
            centers.append(lifted["center"])
            bmin = torch.minimum(bmin, lifted["bbox_min"])
            bmax = torch.maximum(bmax, lifted["bbox_max"])
            for v in lifted["voxels"]:
                votes[v] = votes.get(v, 0) + 1
        if centers:
            self.center = torch.stack(centers).mean(dim=0)
            self.bbox_min, self.bbox_max = bmin, bmax
            self.voxels = set(votes)
            self.voxel_votes = votes
        self._geo_sig = resolver.sig(self.owning_keys())


# --------------------------------------------------------------------------- table
class InstanceTable:
    """Persistent inst_id -> record map with reference-identical matching."""

    def __init__(self, cfg: SemConfig, resolver):
        self.cfg = cfg
        self.resolver = resolver
        self.records: Dict[int, InstanceRecord] = {}
        self._next_id = 0
        self._dirty = True
        self._cache_ids: Optional[np.ndarray] = None
        self._cache_bmin: Optional[np.ndarray] = None
        self._cache_bmax: Optional[np.ndarray] = None
        self.voxel_size = cfg.voxel_size

    def __len__(self) -> int:
        return len(self.records)

    # ---------------------------------------------------------------- freshness
    def _ensure_fresh(self, rec: InstanceRecord) -> None:
        sig = self.resolver.sig(rec.owning_keys())
        if rec._geo_sig != sig:
            rec.relift(self.resolver, self.cfg)
            self._dirty = True

    def rekey_all(self, force: bool = False) -> int:
        """Re-lift every stale (or all, when ``force``) record; returns #relifted."""
        n = 0
        for rec in self.records.values():
            if force:
                rec.relift(self.resolver, self.cfg)
                n += 1
            else:
                sig = self.resolver.sig(rec.owning_keys())
                if rec._geo_sig != sig:
                    rec.relift(self.resolver, self.cfg)
                    n += 1
        if n:
            self._dirty = True
        return n

    # ---------------------------------------------------------------- KNN gate
    def _rebuild_cache(self) -> None:
        if not self.records:
            self._cache_ids = None
            self._dirty = False
            return
        ids = np.fromiter(self.records.keys(), dtype=np.int64)
        self._cache_ids = ids
        self._cache_bmin = np.stack([self.records[int(i)].bbox_min.numpy() for i in ids])
        self._cache_bmax = np.stack([self.records[int(i)].bbox_max.numpy() for i in ids])
        self._dirty = False

    def query_candidates(self, center: torch.Tensor) -> List[int]:
        """Inst ids whose AABB is within ``knn_radius`` (point-to-AABB distance),
        capped at the ``max_candidates`` nearest."""
        if self._dirty:
            self._rebuild_cache()
        if self._cache_ids is None:
            return []
        pt = center.detach().cpu().numpy().astype(np.float32).reshape(3)
        delta = (np.maximum(self._cache_bmin - pt, 0.0)
                 + np.maximum(pt - self._cache_bmax, 0.0))
        dists = np.linalg.norm(delta, axis=-1)
        within = np.nonzero(dists <= self.cfg.knn_radius)[0]
        if within.size == 0:
            return []
        if within.size > self.cfg.max_candidates:
            within = within[np.argsort(dists[within])[: self.cfg.max_candidates]]
        return [int(self._cache_ids[i]) for i in within]

    # ---------------------------------------------------------------- scoring
    def _score(self, prop: Dict, rec: InstanceRecord) -> float:
        from opense3r.instance_post import _proposal_pair_score_dict
        rec_view = {
            "descriptor": rec.descriptor,
            "voxels": rec.voxels,
            "num_voxels": rec.num_voxels,
            "center": rec.center,
            "bbox_min": rec.bbox_min,
            "bbox_max": rec.bbox_max,
        }
        if rec.sem_query is not None:
            rec_view["sem_query"] = rec.sem_query
        c = self.cfg
        return _proposal_pair_score_dict(
            prop, rec_view,
            w_desc=c.w_desc, w_vox=c.w_vox, w_geo=c.w_geo, w_sem=c.w_sem,
            w_box=0.0, vox_gate=None, desc_gate=None,
            sigma_dist=c.sigma_dist, sem_gate_thr=0.0, combine="add",
            box_iou_mode="min", merge_head=None,
            vox_bonus_alpha=1.0, desc_pow=1.0, vox_or_tau=0.5,
        )

    # ---------------------------------------------------------------- spawn/merge
    def _spawn(self, prop: Dict, key: int, obs: Obs) -> int:
        iid = self._next_id
        self._next_id += 1
        rec = InstanceRecord(iid, prop, key, obs)
        rec._geo_sig = self.resolver.sig([key])
        self.records[iid] = rec
        self._dirty = True
        return iid

    def _merge(self, iid: int, prop: Dict, key: int, obs: Obs) -> None:
        rec = self.records[iid]
        rec.merge(prop, key, obs,
                  embed_update=self.cfg.embed_update,
                  embed_momentum=self.cfg.embed_momentum)
        rec._geo_sig = self.resolver.sig(rec.owning_keys())
        self._dirty = True

    # ---------------------------------------------------------------- matching
    def match_chunk(self, proposals: List[Dict], frame_keys: List[int],
                    pixel_to_proposal: torch.Tensor) -> List[int]:
        """Reference algorithm: per frame, KNN-shortlist -> Hungarian on -score ->
        merge at ``score >= match_threshold`` else spawn.

        Args:
            proposals: output of ``process_chunk_to_proposals`` (born in GLOBAL coords).
            frame_keys: per chunk-frame graph key (len == chunk T).
            pixel_to_proposal: [T,H_f,W_f] long — provenance masks per proposal.
        """
        from scipy.optimize import linear_sum_assignment

        per_frame: Dict[int, List[int]] = {}
        for k, p in enumerate(proposals):
            per_frame.setdefault(int(p["frame"]), []).append(k)

        out = [-1] * len(proposals)
        for t in sorted(per_frame):
            key = frame_keys[t]
            idxs = per_frame[t]
            props = [proposals[k] for k in idxs]

            def _obs(k: int, p: Dict) -> Obs:
                return Obs.from_mask(key, int(p.get("slot", -1)),
                                     float(p.get("objectness", 0.0)),
                                     pixel_to_proposal[t] == k)

            cand_set: Set[int] = set()
            for p in props:
                cand_set.update(self.query_candidates(p["center"]))
            cands = sorted(cand_set)
            for c in cands:                            # heal stale candidates pre-score
                self._ensure_fresh(self.records[c])

            if not cands:
                for k, p in zip(idxs, props):
                    out[k] = self._spawn(p, key, _obs(k, p))
                continue

            P, C = len(props), len(cands)
            scores = np.zeros((P, C), dtype=np.float64)
            for i, p in enumerate(props):
                for j, cid in enumerate(cands):
                    scores[i, j] = self._score(p, self.records[cid])
            row, col = linear_sum_assignment(-scores)
            matched: Set[int] = set()
            for r, c in zip(row, col):
                if scores[r, c] >= self.cfg.match_threshold:
                    k = idxs[r]
                    self._merge(cands[c], props[r], key, _obs(k, props[r]))
                    out[k] = cands[c]
                    matched.add(r)
            for r in range(P):
                if r not in matched:
                    k = idxs[r]
                    out[k] = self._spawn(props[r], key, _obs(k, props[r]))
        return out

    # ---------------------------------------------------------------- export view
    def to_metadata_list(self) -> List[Dict]:
        """Lightweight per-instance metadata for the JSON dump (reference schema)."""
        out = []
        for inst_id in sorted(self.records.keys()):
            r = self.records[inst_id]
            out.append({
                "inst_id": inst_id,
                "center": r.center.tolist(),
                "bbox_min": r.bbox_min.tolist(),
                "bbox_max": r.bbox_max.tolist(),
                "bbox_volume_m3": r.bbox_volume,
                "count": r.count,
                "first_seen": r.first_seen,
                "last_seen": r.last_seen,
                "num_voxels": r.num_voxels,
                "mean_objectness": r.mean_objectness,
                "member_frames": r.member_frames,
                "member_slots": r.member_slots,
                "member_objectness": r.member_objectness,
            })
        return out

    def voxel_to_inst_map(self) -> Dict[Tuple[int, int, int], int]:
        """Winner-take-all voxel -> inst_id (most votes; ties -> lower id).
        Verbatim port of the reference implementation."""
        best: Dict[Tuple[int, int, int], Tuple[int, int]] = {}
        for inst_id, rec in self.records.items():
            for v, n in rec.voxel_votes.items():
                cur = best.get(v)
                if cur is None or n > cur[0] or (n == cur[0] and inst_id < cur[1]):
                    best[v] = (n, inst_id)
        return {v: iid for v, (_, iid) in best.items()}

    # ---------------------------------------------------------------- final merge (export)
    def final_merge(self) -> int:
        """Union-find over pairs with AABB overlap AND fused score >= merge_desc_high —
        cleans duplicates spawned during pre-LC drift. Returns #merged-away."""
        ids = sorted(self.records)
        parent = {i: i for i in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for rec in self.records.values():
            self._ensure_fresh(rec)
        for a_i in range(len(ids)):
            for b_i in range(a_i + 1, len(ids)):
                a, b = self.records[ids[a_i]], self.records[ids[b_i]]
                if (a.bbox_min > b.bbox_max).any() or (b.bbox_min > a.bbox_max).any():
                    continue
                prop_view = {
                    "descriptor": a.descriptor, "voxels": a.voxels,
                    "num_voxels": a.num_voxels, "center": a.center,
                    "bbox_min": a.bbox_min, "bbox_max": a.bbox_max,
                }
                if a.sem_query is not None:
                    prop_view["sem_query"] = a.sem_query
                if self._score(prop_view, b) >= self.cfg.merge_desc_high:
                    ra, rb = find(ids[a_i]), find(ids[b_i])
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)

        merged = 0
        for i in ids:
            r = find(i)
            if r == i:
                continue
            root, rec = self.records[r], self.records[i]
            for o, obj in zip(rec.obs, rec.member_objectness):
                root.obs.append(o)
                root.member_objectness.append(obj)
            root.count += rec.count
            n_r, n_i = root.count - rec.count, rec.count
            w_r, w_i = n_r / (n_r + n_i), n_i / (n_r + n_i)
            root.descriptor = torch.nn.functional.normalize(
                w_r * root.descriptor + w_i * rec.descriptor, dim=-1)
            if root.sem_query is not None and rec.sem_query is not None:
                root.sem_query = torch.nn.functional.normalize(
                    w_r * root.sem_query + w_i * rec.sem_query, dim=-1)
            root.first_seen = min(root.first_seen, rec.first_seen)
            root.last_seen = max(root.last_seen, rec.last_seen)
            root.relift(self.resolver, self.cfg)
            del self.records[i]
            merged += 1
        if merged:
            self._dirty = True
        return merged
