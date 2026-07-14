"""
slam_opense3r/instance_table.py
================================
Persistent instance memory for query-head SLAM.

Replaces the dense per-pixel `ins_store` (VoxelFeatureMap) of the legacy
pipeline.  Each tracked instance lives as one `InstanceRecord` containing
its running-average descriptor / sem_query / center, its accumulated
voxel occupancy with per-voxel "vote count", and trace metadata
(member frames + slots).

Cross-chunk matching is **bipartite Hungarian** against a spatial
shortlist (KNN on `center`, default radius 2 m) so the cost matrix
stays bounded even on very long SLAM sequences.

Q4 (voxel-ownership conflicts) is handled by winner-take-all voting:
each instance records `voxel_votes[v] += 1` whenever a merged proposal
contained voxel `v`.  At export time, `voxel_to_inst_map()` resolves
each voxel to whichever instance has the most votes for it.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import math
import torch
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    linear_sum_assignment = None
    _HAS_SCIPY = False

from opense3r.instance_post import (
    _aabb_iou,
    _proposal_pair_score_dict,
    aabb_iou_min,
    _spatial_distance_for_pair,
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-instance record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InstanceRecord:
    """Persistent representation of one tracked instance.

    All tensors live on CPU.  Running-average updates keep `descriptor` /
    `sem_query` / `center` numerically aligned to all members merged so far.
    `voxel_votes` is the per-voxel hit count, used by `voxel_to_inst_map()`
    to resolve cross-instance conflicts winner-take-all.
    """
    inst_id:        int
    descriptor:     torch.Tensor                     # [D]   L2-normed
    sem_query:      Optional[torch.Tensor]           # [D_s] L2-normed or None
    center:         torch.Tensor                     # [3]
    bbox_min:       torch.Tensor                     # [3]
    bbox_max:       torch.Tensor                     # [3]
    voxels:         Set[Tuple[int, int, int]]        = field(default_factory=set)
    voxel_votes:    Dict[Tuple[int, int, int], int]  = field(default_factory=dict)
    count:          int                              = 1
    first_seen:     int                              = -1
    last_seen:      int                              = -1
    member_frames:  List[int]                        = field(default_factory=list)
    member_slots:   List[int]                        = field(default_factory=list)
    # Per-merge objectness (head confidence).  Provides a cheap noise signal:
    # ghosty proposals tend to come from low-confidence slots, so a low
    # running average suggests the cluster is not a real object.
    member_objectness: List[float]                   = field(default_factory=list)
    # Optional head-predicted bbox (running-average across merges).
    # Kept separate from `bbox_min` / `bbox_max` which always remain the
    # mask-derived observed extent (min/max union of all merged proposals).
    # Populated only when the head outputs a box.  Used by the matching
    # cue as the "predicted full extent" signal — accumulated bbox stays
    # tight for storage / vis.
    pred_bbox_min:  Optional[torch.Tensor]           = None
    pred_bbox_max:  Optional[torch.Tensor]           = None

    @classmethod
    def from_proposal(cls, inst_id: int, prop: Dict, frame_idx: int):
        pred_bmin = (prop['pred_bbox_min'].detach().cpu().clone()
                      if 'pred_bbox_min' in prop else None)
        pred_bmax = (prop['pred_bbox_max'].detach().cpu().clone()
                      if 'pred_bbox_max' in prop else None)
        rec = cls(
            inst_id    = inst_id,
            descriptor = prop['descriptor'].detach().cpu().clone(),
            sem_query  = (prop['sem_query'].detach().cpu().clone()
                          if 'sem_query' in prop else None),
            center     = prop['center'].detach().cpu().clone(),
            bbox_min   = prop['bbox_min'].detach().cpu().clone(),
            bbox_max   = prop['bbox_max'].detach().cpu().clone(),
            count      = 1,
            first_seen = frame_idx,
            last_seen  = frame_idx,
            member_frames = [frame_idx],
            member_slots  = [int(prop.get('slot', -1))],
            member_objectness = [float(prop.get('objectness', 0.0))],
            pred_bbox_min = pred_bmin,
            pred_bbox_max = pred_bmax,
        )
        for v in prop['voxels']:
            rec.voxels.add(v)
            rec.voxel_votes[v] = 1
        return rec

    def merge(self, prop: Dict, frame_idx: int,
              embed_update: str = 'mean', embed_momentum: float = 0.75):
        """Running-average merge of a new proposal into this instance.

        `embed_update` controls ONLY the descriptor (instance embedding) update —
        to match CTVIS-style training (--desc_contrastive_mode momentum):
          'mean'       (default) cumulative running mean (legacy).
          'momentum'   EMA: (1-m)*old + m*new.
          'sim_guided' beta = max(0, cos(old,new)); (1-beta)*old + beta*new
                       (outlier-robust — new obs unlike the accumulated ref are
                       down-weighted).  Mirrors the training EMA reference.
        center / sem_query keep the cumulative running mean regardless.
        """
        n   = self.count
        a   = n / (n + 1)
        b   = 1.0 / (n + 1)
        # descriptor: update mode (mean | momentum | sim_guided), then renorm
        new_desc = prop['descriptor'].detach().cpu()
        if embed_update == 'sim_guided':
            beta = max(0.0, float((self.descriptor * new_desc).sum()))
            da, db = 1.0 - beta, beta
        elif embed_update == 'momentum':
            da, db = 1.0 - embed_momentum, embed_momentum
        else:  # 'mean'
            da, db = a, b
        self.descriptor = torch.nn.functional.normalize(
            da * self.descriptor + db * new_desc, dim=-1,
        )
        if self.sem_query is not None and 'sem_query' in prop:
            self.sem_query = torch.nn.functional.normalize(
                a * self.sem_query + b * prop['sem_query'].detach().cpu(), dim=-1,
            )
        # center: linear running avg
        self.center = a * self.center + b * prop['center'].detach().cpu()

        # Observed bbox (mask × world_points): min/max union — accumulates
        # the actual observed extent.  Always present.
        pmin = prop['bbox_min'].detach().cpu()
        pmax = prop['bbox_max'].detach().cpu()
        self.bbox_min = torch.minimum(self.bbox_min, pmin)
        self.bbox_max = torch.maximum(self.bbox_max, pmax)

        # Predicted bbox (head full-extent): running-average — each merge
        # is an estimate of the same full extent, so averaging smooths
        # per-frame noise rather than compounding it.  Optional.
        if 'pred_bbox_min' in prop and 'pred_bbox_max' in prop:
            pbmin = prop['pred_bbox_min'].detach().cpu()
            pbmax = prop['pred_bbox_max'].detach().cpu()
            if self.pred_bbox_min is None:
                self.pred_bbox_min = pbmin.clone()
                self.pred_bbox_max = pbmax.clone()
            else:
                self.pred_bbox_min = a * self.pred_bbox_min + b * pbmin
                self.pred_bbox_max = a * self.pred_bbox_max + b * pbmax
        # voxels: union + vote increment
        for v in prop['voxels']:
            self.voxels.add(v)
            self.voxel_votes[v] = self.voxel_votes.get(v, 0) + 1
        # metadata
        self.count    += 1
        self.last_seen = frame_idx
        self.member_frames.append(frame_idx)
        self.member_slots .append(int(prop.get('slot', -1)))
        self.member_objectness.append(float(prop.get('objectness', 0.0)))

    @property
    def num_voxels(self) -> int:
        return len(self.voxels)

    @property
    def mean_objectness(self) -> float:
        if not self.member_objectness:
            return 0.0
        return float(sum(self.member_objectness) / len(self.member_objectness))

    @property
    def max_objectness(self) -> float:
        # Best-view confidence.  For A' obj×sem_conf, avoids mean diluting the
        # per-frame signal (a real object needs only one good-view observation).
        if not self.member_objectness:
            return 0.0
        return float(max(self.member_objectness))

    @property
    def bbox_volume(self) -> float:
        """Axis-aligned bbox volume in m³."""
        size = (self.bbox_max - self.bbox_min).clamp(min=0)
        return float(size.prod().item())


# ─────────────────────────────────────────────────────────────────────────────
# Instance table (the SLAM-side "ins_store" replacement)
# ─────────────────────────────────────────────────────────────────────────────

class InstanceTable:
    """Persistent map from inst_id to `InstanceRecord`.

    Two operations dominate hot paths:
      - `match_chunk()`     : new proposals → existing instances (point-to-bbox
                              gating + Hungarian).
      - `voxel_to_inst_map`: voxel grid → inst_id (winner-take-all over votes).

    Spatial gating uses **point-to-AABB distance** (proposal centre → instance
    bbox) rather than centre-to-centre.  This makes the KNN radius scale
    with each instance's accumulated spatial extent, so partial views of a
    long object still pass the filter while distant same-class objects do
    not.  Bbox arrays are cached and rebuilt lazily on table mutation.
    """

    def __init__(
        self,
        voxel_size:        float = 0.05,
        knn_radius:        float = 2.0,
        match_threshold:   float = 0.35,
        # ── NEW matching path (match_mode='greedy_deadzone') ──────────────────
        # Greedy many-to-one (handles FastSAM-seeded duplicate/partial proposals)
        # with a 2-stage decision: amodal-box ∩/min GEO gate → desc-cosine
        # 3-band (merge / discard / spawn).  Thresholds grounded in the FastSAM
        # desc analysis (self-sim ~0.85-0.93, diff-inst ~0.15-0.28, AUROC ~0.98):
        #   desc > merge_desc_high  → merge (same instance)
        #   desc < spawn_desc_low   → spawn (clearly different, overlapping object)
        #   in-between              → DISCARD (ambiguous odd-view obs; temporal
        #                             redundancy re-captures it later)
        # Semantic cue is NOT used here (prepared via sem gate elsewhere, off).
        match_mode:        str   = 'hungarian',   # 'hungarian' (legacy) | 'greedy_deadzone'
        gate_overlap:      float = 0.10,           # τ_geo: amodal-box ∩/min gate (loose prune)
        merge_desc_high:   float = 0.75,           # τ_high
        spawn_desc_low:    float = 0.40,           # τ_low
        greedy_gate_use_observed: bool = False,    # greedy GEO gate box: observed (voxel, reliable)
                                                   # instead of amodal pred box (view-invariant but noisy)
        # Descriptor (instance embedding) accumulation across merges:
        #   'mean' (default) | 'momentum' | 'sim_guided'.  Use 'sim_guided' to
        #   mirror CTVIS-style training (--desc_contrastive_mode momentum).
        embed_update:      str   = 'mean',
        embed_momentum:    float = 0.75,
        # Matching score transform: 'hungarian' (default; raw fused score) or
        #   'bisoftmax' (CTVIS mutual-softmax (d2t+t2d)/2 before thresholding).
        match_metric:      str   = 'hungarian',
        # cue weights — match training-time PLY defaults
        w_desc:            float = 0.35,
        w_sem:             float = 0.35,
        w_vox:             float = 0.15,
        w_geo:             float = 0.15,
        sigma_dist:        float = 0.5,
        max_candidates:    int   = 32,
        # Multi-gate overrides (None = off → use linear formula only).
        vox_gate:          Optional[float] = None,
        desc_gate:         Optional[float] = None,
        # Cue fusion: add | mul | mul_voxbonus | geo_gated | mul_descpow | or_dual
        score_combine:     str   = 'add',
        box_iou_mode:      str   = 'min',
        merge_head         = None,
        # Soft semantic gate: pairs with sem_query cosine < thr are vetoed.
        # 0 = off.  Applies to all combine modes.
        sem_gate_thr:      float = 0.0,
        # Extra knobs for the new combine modes (ignored by add/mul):
        vox_bonus_alpha:   float = 1.0,   # A (mul_voxbonus)
        desc_pow:          float = 1.0,   # D (mul_descpow)
        vox_or_tau:        float = 0.5,   # E (or_dual)
    ):
        if not _HAS_SCIPY:
            raise ImportError(
                "scipy is required for InstanceTable (Hungarian assignment). "
                "Install with `pip install scipy`."
            )
        self.voxel_size      = float(voxel_size)
        self.knn_radius      = float(knn_radius)
        self.match_threshold = float(match_threshold)
        self.match_mode      = str(match_mode)
        self.gate_overlap    = float(gate_overlap)
        self.merge_desc_high = float(merge_desc_high)
        self.spawn_desc_low  = float(spawn_desc_low)
        self.greedy_gate_use_observed = bool(greedy_gate_use_observed)
        if embed_update not in ('mean', 'momentum', 'sim_guided'):
            raise ValueError(f"embed_update must be 'mean'|'momentum'|'sim_guided', got {embed_update!r}")
        self.embed_update    = str(embed_update)
        self.embed_momentum  = float(embed_momentum)
        # NOTE: 'bisoftmax' was removed — the old impl wrongly applied the mutual
        # softmax to the FUSED (desc+vox+geo) score, which loses the absolute
        # scale the merge-or-spawn-new threshold needs → AP collapsed.  CTVIS's
        # bisoftmax is on the DESCRIPTOR-similarity matrix only and assumes a
        # tracking-style new-track mechanism (a "dustbin" column).  Re-add only
        # with that design.  For now it falls back to hungarian.
        if match_metric == 'bisoftmax':
            print("[InstanceTable] match_metric='bisoftmax' is not correctly "
                  "implemented (needs desc-only + dustbin) → using 'hungarian'.")
            match_metric = 'hungarian'
        if match_metric != 'hungarian':
            raise ValueError(f"match_metric must be 'hungarian', got {match_metric!r}")
        self.match_metric    = str(match_metric)
        self.w_desc          = float(w_desc)
        self.w_sem           = float(w_sem)
        self.w_vox           = float(w_vox)
        self.w_geo           = float(w_geo)
        self.sigma_dist      = float(sigma_dist)
        self.max_candidates  = int(max_candidates)
        self.vox_gate        = None if vox_gate  is None else float(vox_gate)
        self.desc_gate       = None if desc_gate is None else float(desc_gate)
        _valid_combine = ('add', 'mul', 'mul_voxbonus', 'geo_gated', 'learned',
                          'mul_descpow', 'or_dual')
        if score_combine not in _valid_combine:
            raise ValueError(f"score_combine must be one of {_valid_combine}, "
                             f"got {score_combine!r}")
        self.score_combine    = str(score_combine)
        self.box_iou_mode     = str(box_iou_mode)
        self.merge_head       = merge_head
        self.sem_gate_thr     = float(sem_gate_thr)
        self.vox_bonus_alpha  = float(vox_bonus_alpha)
        self.desc_pow         = float(desc_pow)
        self.vox_or_tau       = float(vox_or_tau)

        self.records:    Dict[int, InstanceRecord] = {}
        self._next_id:   int = 0
        # Cached arrays for vectorised point-to-AABB query; rebuilt lazily.
        self._cache_ids:    Optional[np.ndarray] = None   # [N] sorted inst_ids
        self._cache_bmin:   Optional[np.ndarray] = None   # [N, 3]
        self._cache_bmax:   Optional[np.ndarray] = None   # [N, 3]
        self._dirty:        bool = True

        # ── Optional per-pair score-component logging (analysis only) ──────
        # Enabled by env OPENSE3R_SCORE_LOG=<path.npz>.  When on, every
        # (proposal, candidate) pair evaluated in match_chunk records its
        # individual cue similarities + final score + merge decision + the
        # two 3D centers (for offline GT-label assignment).  No effect on the
        # matching itself; default OFF → zero overhead.
        import os as _os
        self._score_log_path = _os.environ.get('OPENSE3R_SCORE_LOG', None)
        self._score_log: Optional[list] = [] if self._score_log_path else None
        if self._score_log is not None:
            import atexit
            atexit.register(self._flush_score_log)
            print(f"[InstanceTable] score-component logging ON → "
                  f"{self._score_log_path}")

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_proposal(self, prop: Dict, frame_idx: int) -> int:
        """Create a new instance from this proposal.  Returns inst_id."""
        inst_id = self._next_id
        self._next_id += 1
        self.records[inst_id] = InstanceRecord.from_proposal(inst_id, prop, frame_idx)
        self._dirty = True
        return inst_id

    def merge_proposal(self, inst_id: int, prop: Dict, frame_idx: int):
        """Merge a proposal into an existing instance."""
        self.records[inst_id].merge(prop, frame_idx,
                                    embed_update=self.embed_update,
                                    embed_momentum=self.embed_momentum)
        self._dirty = True

    # ------------------------------------------------------------------
    # Spatial shortlist
    # ------------------------------------------------------------------

    def _rebuild_cache(self):
        """Refresh the cached bbox arrays used by `query_candidates`."""
        if not self.records:
            self._cache_ids  = None
            self._cache_bmin = None
            self._cache_bmax = None
            self._dirty      = False
            return
        ids = np.array(sorted(self.records.keys()), dtype=np.int64)
        bmins = np.stack(
            [self.records[int(i)].bbox_min.detach().cpu().numpy() for i in ids],
            axis=0,
        ).astype(np.float32)
        bmaxs = np.stack(
            [self.records[int(i)].bbox_max.detach().cpu().numpy() for i in ids],
            axis=0,
        ).astype(np.float32)
        self._cache_ids  = ids
        self._cache_bmin = bmins
        self._cache_bmax = bmaxs
        self._dirty      = False

    def query_candidates(
        self,
        center: torch.Tensor,
        radius: Optional[float] = None,
        k: Optional[int]        = None,
    ) -> List[int]:
        """Inst_ids whose AABB is within `radius` of `center`.

        Distance metric is point-to-AABB:
            d(point, bbox) = ||max(bmin - point, 0) + max(point - bmax, 0)||
        which is 0 when `point` lies inside the bbox and grows linearly
        with the gap when outside.  This naturally scales the candidate
        shortlist to each instance's accumulated spatial extent — long
        objects keep accepting partial-view proposals near their edges,
        compact objects only match proposals close to them.

        Capped at `k = max_candidates` nearest if more match.
        """
        if self._dirty:
            self._rebuild_cache()
        if self._cache_ids is None:
            return []
        r = float(radius if radius is not None else self.knn_radius)

        pt = center.detach().cpu().numpy().astype(np.float32).reshape(3)
        # Vectorised: [N, 3]  =  max(bmin - pt, 0) + max(pt - bmax, 0)
        delta = (np.maximum(self._cache_bmin - pt, 0.0)
               + np.maximum(pt - self._cache_bmax, 0.0))               # [N, 3]
        dists = np.linalg.norm(delta, axis=-1)                          # [N]

        within = np.nonzero(dists <= r)[0]
        if within.size == 0:
            return []
        cap = k if k is not None else self.max_candidates
        if cap is not None and within.size > cap:
            # Keep the `cap` nearest (by point-to-AABB distance).
            order = np.argsort(dists[within])[:cap]
            within = within[order]
        return [int(self._cache_ids[i]) for i in within]

    # ------------------------------------------------------------------
    # Cross-chunk Hungarian matching
    # ------------------------------------------------------------------

    def _score(self, prop: Dict, rec: InstanceRecord) -> float:
        """Pair-score between a proposal and an instance record.

        Reuses the same scoring kernel as the training-time PLY pipeline
        so SLAM and eval behave identically.  `bbox_min`/`bbox_max` are
        exposed in the rec view so the geo cue computes point-to-AABB
        distance (consistent with the KNN gating).
        """
        rec_view = {
            'descriptor': rec.descriptor,
            'voxels':     rec.voxels,
            'num_voxels': rec.num_voxels,
            'center':     rec.center,
            'bbox_min':   rec.bbox_min,
            'bbox_max':   rec.bbox_max,
        }
        if rec.sem_query is not None:
            rec_view['sem_query'] = rec.sem_query
        if rec.pred_bbox_min is not None:
            rec_view['pred_bbox_min'] = rec.pred_bbox_min
            rec_view['pred_bbox_max'] = rec.pred_bbox_max
        return _proposal_pair_score_dict(
            prop, rec_view,
            w_desc = self.w_desc, w_vox = self.w_vox,
            w_geo  = self.w_geo,  w_sem = self.w_sem,
            w_box  = 0.0,
            vox_gate  = self.vox_gate,
            desc_gate = self.desc_gate,
            sigma_dist = self.sigma_dist,
            sem_gate_thr = self.sem_gate_thr,
            combine    = self.score_combine,
            box_iou_mode = self.box_iou_mode,
            merge_head = self.merge_head,
            vox_bonus_alpha = self.vox_bonus_alpha,
            desc_pow        = self.desc_pow,
            vox_or_tau      = self.vox_or_tau,
        )

    def _score_components(self, prop: Dict, rec: InstanceRecord) -> dict:
        """Recompute the individual cue similarities for analysis logging.
        Mirrors `_proposal_pair_score_dict` math but returns the breakdown.
        Cheap; only called when score-component logging is enabled."""
        sim_desc = float((prop['descriptor'] * rec.descriptor).sum().item())
        if prop['num_voxels'] > 0 and rec.num_voxels > 0:
            inter = len(prop['voxels'] & rec.voxels)
            sim_vox = inter / max(min(prop['num_voxels'], rec.num_voxels), 1)
        else:
            sim_vox = 0.0
        use_pred = ('pred_bbox_min' in prop and rec.pred_bbox_min is not None)
        if use_pred:
            sim_geo = _aabb_iou(self.box_iou_mode, prop['pred_bbox_min'], prop['pred_bbox_max'],
                                   rec.pred_bbox_min, rec.pred_bbox_max)
        else:
            rec_view = {'center': rec.center,
                        'bbox_min': rec.bbox_min, 'bbox_max': rec.bbox_max}
            dist = _spatial_distance_for_pair(prop, rec_view)
            sim_geo = math.exp(-dist / max(self.sigma_dist, 1e-6))
        sim_sem = 0.0
        if rec.sem_query is not None and 'sem_query' in prop:
            sim_sem = float((prop['sem_query'] * rec.sem_query).sum().item())
        return dict(sim_desc=sim_desc, sim_vox=float(sim_vox),
                    sim_geo=float(sim_geo), sim_sem=sim_sem)

    def _flush_score_log(self):
        if not self._score_log:
            return
        import numpy as _np
        rows = self._score_log
        arr = {k: _np.array([r[k] for r in rows], dtype=_np.float32)
               for k in ('sim_desc', 'sim_vox', 'sim_geo', 'sim_sem',
                         'score', 'merged', 'selected', 'frame')}
        arr['prop_center'] = _np.array([r['prop_center'] for r in rows],
                                       dtype=_np.float32)
        arr['cand_center'] = _np.array([r['cand_center'] for r in rows],
                                       dtype=_np.float32)
        _np.savez(self._score_log_path, **arr)
        print(f"[InstanceTable] flushed {len(rows)} score-pair rows → "
              f"{self._score_log_path}")
        self._score_log = []

    def match_chunk(
        self,
        proposals:     List[Dict],         # output of process_chunk_to_proposals
        frame_indices: List[int],          # per-proposal absolute frame index
    ) -> List[int]:
        """Assign each proposal to an existing instance or spawn a new one.

        Algorithm (frame-by-frame Sequential Hungarian against the table):

          for each frame's proposals P_t:
            1. Build a candidate set C = ⋃ KNN(p.center) for p in P_t.
            2. Build cost[|P_t|, |C|] = -score(p, table[c]).
            3. Solve linear_sum_assignment.
            4. (p, c) with score >= match_threshold → merge into c.
                Else → spawn new instance.

        Returns: list of inst_id per input proposal (same length).
        """
        if len(proposals) != len(frame_indices):
            raise ValueError(
                f"proposals/frame_indices length mismatch: "
                f"{len(proposals)} vs {len(frame_indices)}"
            )

        if self.match_mode == 'greedy_deadzone':
            return self.match_chunk_greedy(proposals, frame_indices)

        # Group proposals by frame (preserves original index for the return list).
        per_frame: Dict[int, List[int]] = {}
        for k, f in enumerate(frame_indices):
            per_frame.setdefault(int(f), []).append(k)

        inst_id_out: List[int] = [-1] * len(proposals)

        for f in sorted(per_frame.keys()):
            prop_idxs = per_frame[f]
            P = len(prop_idxs)
            props = [proposals[k] for k in prop_idxs]

            # Step 1: spatial shortlist (union over P).
            cand_set: Set[int] = set()
            for p in props:
                for c in self.query_candidates(p['center']):
                    cand_set.add(c)
            cands = sorted(cand_set)
            C = len(cands)

            if C == 0:
                # No existing instances near any proposal: all spawn new.
                for k, p in zip(prop_idxs, props):
                    inst_id_out[k] = self.add_proposal(p, f)
                continue

            # Step 2: cost matrix.
            cost = np.full((P, C), 1.0, dtype=np.float64)   # 1.0 = "no match"
            scores = np.zeros((P, C), dtype=np.float64)
            comps = None if self._score_log is None else {}
            for i, p in enumerate(props):
                for j, c_id in enumerate(cands):
                    s = self._score(p, self.records[c_id])
                    scores[i, j] = s
                    cost[i, j]   = -s    # Hungarian minimises
                    if comps is not None:
                        comps[(i, j)] = self._score_components(p, self.records[c_id])

            # Step 3: solve assignment over min(P, C) pairs.
            row, col = linear_sum_assignment(cost)
            matched_p: Set[int] = set()
            selected_pairs: Set[tuple] = set()
            for r, c in zip(row, col):
                selected_pairs.add((int(r), int(c)))
                if scores[r, c] >= self.match_threshold:
                    self.merge_proposal(cands[c], props[r], f)
                    inst_id_out[prop_idxs[r]] = cands[c]
                    matched_p.add(r)

            # Analysis logging: every evaluated pair + decision + centers.
            if comps is not None:
                for (i, j), cc in comps.items():
                    merged = ((i, j) in selected_pairs
                              and scores[i, j] >= self.match_threshold)
                    self._score_log.append(dict(
                        frame=int(f),
                        score=float(scores[i, j]),
                        selected=float((i, j) in selected_pairs),
                        merged=float(merged),
                        prop_center=props[i]['center'].detach().cpu().numpy().reshape(3),
                        cand_center=self.records[cands[j]].center.detach().cpu().numpy().reshape(3),
                        **cc,
                    ))
            # Step 4: unmatched proposals spawn new instances.
            for r in range(P):
                if r not in matched_p:
                    inst_id_out[prop_idxs[r]] = self.add_proposal(props[r], f)

        return inst_id_out

    # ------------------------------------------------------------------
    @staticmethod
    def _amodal_box(obj):
        """(bmin, bmax) of obj's matching box — prefer the predicted AMODAL box
        (view-invariant, covers occluded parts), fall back to the observed box.
        obj is a proposal dict or an InstanceRecord."""
        if isinstance(obj, dict):
            if obj.get('pred_bbox_min') is not None:
                return obj['pred_bbox_min'], obj['pred_bbox_max']
            return obj.get('bbox_min'), obj.get('bbox_max')
        if getattr(obj, 'pred_bbox_min', None) is not None:
            return obj.pred_bbox_min, obj.pred_bbox_max
        return obj.bbox_min, obj.bbox_max

    def _gate_box(self, obj):
        """Box used for the greedy GEO gate.  Observed (reliable voxel-derived
        extent) when `greedy_gate_use_observed`, else the amodal pred box
        (view-invariant but noisier).  obj is a proposal dict or InstanceRecord."""
        if self.greedy_gate_use_observed:
            if isinstance(obj, dict):
                return obj.get('bbox_min'), obj.get('bbox_max')
            return obj.bbox_min, obj.bbox_max
        return self._amodal_box(obj)

    def match_chunk_greedy(self, proposals: List[Dict],
                           frame_indices: List[int]) -> List[int]:
        """Greedy many-to-one matching with amodal-box GEO gate + desc 3-band.

        Proposals are processed by objectness (high→low) against the LIVE table,
        so a duplicate/partial proposal can merge into an instance that an
        earlier proposal in the SAME chunk just spawned (handles FastSAM-seeded
        over-segmentation that 1-to-1 Hungarian would split).

        Per proposal p:
          cands = { inst : aabb_iou_min(p.box, inst.box) > gate_overlap }   # ∩/min
          if no cands:                       spawn (spatially isolated → new)
          else, s = max desc-cosine over cands:
            s > merge_desc_high :            merge into best
            s < spawn_desc_low  :            spawn (overlaps but clearly different)
            else                :            DISCARD (ambiguous; re-captured later)
        """
        inst_id_out: List[int] = [-1] * len(proposals)
        order = sorted(range(len(proposals)),
                       key=lambda k: -float(proposals[k].get('objectness', 0.0)))
        for k in order:
            p = proposals[k]
            f = int(frame_indices[k])
            pbmin, pbmax = self._gate_box(p)
            p_desc = p['descriptor'].detach().cpu()
            best_id, best_s = -1, -2.0
            for c_id, rec in self.records.items():
                cbmin, cbmax = self._gate_box(rec)
                if cbmin is None:
                    continue
                if aabb_iou_min(pbmin, pbmax, cbmin, cbmax) <= self.gate_overlap:
                    continue                                   # GEO gate
                s = float((p_desc * rec.descriptor).sum())     # desc cosine (both L2-normed)
                if s > best_s:
                    best_s, best_id = s, c_id
            if best_id < 0:                                    # no geo candidate
                inst_id_out[k] = self.add_proposal(p, f)
            elif best_s > self.merge_desc_high:                # confident same instance
                self.merge_proposal(best_id, p, f)
                inst_id_out[k] = best_id
            elif best_s < self.spawn_desc_low:                 # overlapping but different
                inst_id_out[k] = self.add_proposal(p, f)
            # else: ambiguous → discard (inst_id stays -1; not added/merged)
        return inst_id_out

    # ------------------------------------------------------------------
    # Voxel ↔ instance map (Q4: winner-take-all over votes)
    # ------------------------------------------------------------------

    def voxel_to_inst_map(self) -> Dict[Tuple[int, int, int], int]:
        """Resolve the global voxel → inst_id mapping by winner-take-all.

        If multiple instances "claim" the same voxel, the one with the
        most votes for that voxel wins.  Ties are broken by lower inst_id
        (deterministic).
        """
        best:  Dict[Tuple[int, int, int], Tuple[int, int]] = {}  # voxel → (votes, inst_id)
        for inst_id, rec in self.records.items():
            for v, n in rec.voxel_votes.items():
                cur = best.get(v)
                if cur is None or n > cur[0] or (n == cur[0] and inst_id < cur[1]):
                    best[v] = (n, inst_id)
        return {v: iid for v, (_, iid) in best.items()}

    # ------------------------------------------------------------------
    # Stats / debug
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def summary(self) -> Dict[str, int]:
        return {
            'num_instances': len(self.records),
            'total_voxels':  sum(r.num_voxels for r in self.records.values()),
            'total_merges':  sum(r.count       for r in self.records.values()),
        }

    def to_metadata_list(self) -> List[Dict]:
        """Lightweight dict-of-metadata for JSON/NPZ dump (no big tensors)."""
        out = []
        for inst_id in sorted(self.records.keys()):
            r = self.records[inst_id]
            entry = {
                'inst_id':         inst_id,
                'center':          r.center.tolist(),
                'bbox_min':        r.bbox_min.tolist(),
                'bbox_max':        r.bbox_max.tolist(),
                'bbox_volume_m3':  r.bbox_volume,
                'count':           r.count,
                'first_seen':      r.first_seen,
                'last_seen':       r.last_seen,
                'num_voxels':      r.num_voxels,
                'mean_objectness': r.mean_objectness,
                'member_frames':   r.member_frames,
                'member_slots':    r.member_slots,
                'member_objectness': r.member_objectness,
            }
            if r.pred_bbox_min is not None:
                entry['pred_bbox_min'] = r.pred_bbox_min.tolist()
                entry['pred_bbox_max'] = r.pred_bbox_max.tolist()
            out.append(entry)
        return out
