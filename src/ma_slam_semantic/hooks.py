"""SemanticHooks — the glue between ma_slam's S1 hooks and the semantic pipeline.

    hooks = SemanticHooks(SemConfig(), out_dir, name)
    MaSlam(config=cfg, model=backend, hooks=hooks).run(...)

on_submap (post-optimize): bridge the submap -> forward the causal semantic model ->
proposals -> provenance instance table. on_loop: record the event (healing itself is
signature-driven — stale candidates re-lift lazily during matching). on_finish:
final re-lift through the final graph + final merge + reference-format export.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from ma_slam_semantic.config import SemConfig
from ma_slam_semantic.bridge import SubmapBridge, PoseTracker
from ma_slam_semantic.sem_model import SemanticModel
from ma_slam_semantic.proposals import make_proposals
from ma_slam_semantic.instance_map import InstanceTable, SlamGeoResolver


class SemanticHooks:
    def __init__(self, cfg: SemConfig, out_dir: str, name: str = "scene",
                 device: Optional[str] = None):
        self.cfg = cfg
        self.out_dir = out_dir
        self.name = name
        self.model = SemanticModel(cfg, device=device)
        self.bridge = SubmapBridge(cfg.resolution)
        self.tracker = PoseTracker()
        self.resolver: Optional[SlamGeoResolver] = None
        self.table: Optional[InstanceTable] = None
        self.loop_events: List[Tuple[int, int]] = []
        self.n_frames = 0

    # ------------------------------------------------------------------ hooks
    def on_submap(self, sm, gmap, graph) -> None:
        if self.table is None:
            self.resolver = SlamGeoResolver(gmap, graph, self.cfg.resolution)
            self.table = InstanceTable(self.cfg, self.resolver)
        delta = self.tracker.update(gmap, graph)
        if delta > self.cfg.kv_reset_delta:
            print(f"[sem_hooks] max pose delta {delta:.3f} m > {self.cfg.kv_reset_delta}"
                  f" ({'KV reset' if self.cfg.kv_reset else 'logged; kv_reset off'})")
            if self.cfg.kv_reset:
                self.model.reset_kv()
        from ma_slam_semantic.bridge import split_chunk
        chunk = self.bridge.chunk(sm, graph)
        n_props = 0
        for sub in split_chunk(chunk, self.cfg.chunk_size):
            out = self.model.forward_chunk(sub)
            props, pix_map, _ = make_proposals(out, sub, self.cfg)
            self.table.match_chunk(props, sub["frame_keys"], pix_map)
            n_props += len(props)
        self.n_frames += len(chunk["frame_keys"])
        print(f"[sem_hooks] submap {sm.base_id}: +{len(chunk['frame_keys'])} frames, "
              f"{n_props} proposals -> {len(self.table)} instances")

    def on_loop(self, key_new: int, key_old: int) -> None:
        # Healing is handled by the M-signature mechanism (stale records re-lift
        # lazily on their next match / at export); just record the event.
        self.loop_events.append((key_new, key_old))
        print(f"[sem_hooks] loop event {key_new} <-> {key_old} (instances will re-key lazily)")

    def on_finish(self, gmap, graph) -> None:
        if self.table is None:
            return
        from ma_slam_semantic.export import export_all
        res = export_all(self.table, num_frames=self.n_frames,
                         out_dir=self.out_dir, name=self.name, cfg=self.cfg)
        print(f"[sem_hooks] exported {res['n_instances']} instances "
              f"({res['n_dropped']} noise-filtered, {res['n_merged']} final-merged, "
              f"{len(self.loop_events)} loop events) -> {self.out_dir}")
