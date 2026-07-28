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
from ma_slam_semantic.bidesc_lc import (BiDescConfig, BiDescVerdict, FrameIndex,
                                        lookup_descs, verify as bidesc_verify)


class SemanticHooks:
    def __init__(self, cfg: SemConfig, out_dir: str, name: str = "scene",
                 device: Optional[str] = None, gt_positions=None,
                 lc_cfg: Optional[BiDescConfig] = None):
        self.cfg = cfg
        self.out_dir = out_dir
        self.name = name
        # GT camera positions [N,3] (row i = frame i). When set, exports are rigidly
        # mapped into the GT-mesh frame via SE3 Umeyama over the WHOLE trajectory —
        # more robust than first-frame anchoring, whose frame-0 estimation error
        # would shift the entire map (visible as depressed AP on drifty scenes).
        self.gt_positions = gt_positions
        self.model = SemanticModel(cfg, device=device)
        self.bridge = SubmapBridge(cfg.resolution, conf_gate_coef=cfg.conf_gate_coef)
        self.tracker = PoseTracker()
        self.resolver: Optional[SlamGeoResolver] = None
        self.table: Optional[InstanceTable] = None
        self.loop_events: List[Tuple[int, int]] = []
        self.n_frames = 0
        # Bi-Desc LC (v2): per-frame storage is just the associated instance IDS;
        # descriptors are looked up from the table's running-mean memory at verify
        # time. Queried by the solver via verify_loop_candidate.
        self.lc_cfg = lc_cfg or BiDescConfig()
        self.frame_index = FrameIndex(k_bag=self.lc_cfg.k_bag,
                                      keep_desc=(self.lc_cfg.desc_src == "proposal"))

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
            assign = self.table.match_chunk(props, sub["frame_keys"], pix_map)
            paths = [sm.image_paths[i] for i in sub["frame_idx"]]
            per_frame: dict = {}
            for k, p in enumerate(props):
                if assign[k] >= 0:
                    d = p["descriptor"]
                    per_frame.setdefault(int(p["frame"]), []).append(
                        ((assign[k], float(p.get("objectness", 0.0))),
                         d.detach().cpu().numpy() if hasattr(d, "detach") else d))
            for t, rows in per_frame.items():
                self.frame_index.add(paths[t], [r[0] for r in rows],
                                     descs=[r[1] for r in rows])
            n_props += len(props)
        self.n_frames += len(chunk["frame_keys"])
        print(f"[sem_hooks] submap {sm.base_id}: +{len(chunk['frame_keys'])} frames, "
              f"{n_props} proposals -> {len(self.table)} instances "
              f"| lc-index {len(self.frame_index)}")
        if hasattr(self.model.policy, "stats"):
            info = self.model.policy.stats()
            if info is not None:
                print(f"[sem_hooks]   h2o-kv: tokens/layer={info['kvcache_size'][0]}"
                      f"  used={info['kvcache_used_mb']:.0f}MB"
                      f"  prune={self.model.policy.prune_ms:.1f}ms")

    def verify_loop_candidate(self, q_path: str, m_path: str):
        """Bi-Desc verdict for a SALAD loop candidate (solver-initiated; read-only).
        UNKNOWN when either frame has no recorded ids / too few instances."""
        ids_q = self.frame_index.get(q_path)
        ids_r = self.frame_index.get(m_path)
        if not ids_q or not ids_r or self.table is None:
            return BiDescVerdict("UNKNOWN",
                                 n_q=0 if not ids_q else len(ids_q),
                                 n_r=0 if not ids_r else len(ids_r),
                                 reason="no frame index")
        # history side is always the map's running mean (it IS the memory); the
        # current side is either the same lookup ('map') or the frame's raw
        # proposal descriptors ('proposal' — independent of the association layer).
        if self.lc_cfg.desc_src == "proposal":
            D_q = self.frame_index.get_descs(q_path)
            if D_q is None or len(D_q) != len(ids_q):
                D_q = lookup_descs(ids_q, self.table.records)   # pre-switch frames
        else:
            D_q = lookup_descs(ids_q, self.table.records)
        D_r = lookup_descs(ids_r, self.table.records)
        return bidesc_verify(D_q, D_r, self.lc_cfg)

    def on_loop(self, key_new: int, key_old: int) -> None:
        # Healing is handled by the M-signature mechanism (stale records re-lift
        # lazily on their next match / at export); just record the event.
        self.loop_events.append((key_new, key_old))
        print(f"[sem_hooks] loop event {key_new} <-> {key_old} (instances will re-key lazily)")

    def on_finish(self, gmap, graph) -> None:
        if self.table is None:
            return
        import numpy as np
        if len(self.frame_index):
            ns = np.array([len(v) for v in self.frame_index._ids.values()])
            print(f"[sem_hooks] lc-index: {len(self.frame_index)} frames, ids/frame "
                  f"p10={np.percentile(ns,10):.0f} p50={np.percentile(ns,50):.0f} "
                  f"p90={np.percentile(ns,90):.0f} | frames<{self.lc_cfg.n_min}: "
                  f"{(ns < self.lc_cfg.n_min).mean()*100:.0f}%")
        T_align = None
        if self.gt_positions is not None:
            from eval import umeyama_sim3
            # per-frame optimized camera centres, deduped by FRAME ID (filename
            # ordinal, last-wins like map.write_poses) — graph keys are NOT frame
            # ordinals: overlap frames own two nodes (base+i per submap).
            centres = {}
            for sm in gmap.ordered():
                if sm.is_lc:
                    continue
                for i in range(sm.n):
                    centres[int(sm.frame_ids[i])] = graph.get_pose(sm.key(i))[:3, 3]
            gt_all = np.asarray(self.gt_positions)
            fids = sorted(f for f in centres if 0 <= f < len(gt_all))
            est = np.stack([centres[f] for f in fids])
            gt = gt_all[fids]
            _, R, _ = umeyama_sim3(est, gt)          # rigid: rotation + centroid shift
            t = gt.mean(0) - R @ est.mean(0)
            T_align = np.eye(4)
            T_align[:3, :3], T_align[:3, 3] = R, t
            rmse = float(np.sqrt(np.mean(np.sum(((est @ R.T + t) - gt) ** 2, axis=1))))
            print(f"[sem_hooks] SE3-Umeyama align to GT ({len(fids)} frames): "
                  f"residual RMSE {rmse:.3f} m")
            # persist so deploy.py can bring camera_poses.txt / combined_pcd.ply
            # (SLAM frame) into the same frame as the instance npz/table.
            import json
            with open(os.path.join(self.out_dir, "alignment.json"), "w") as f:
                json.dump({"T_align": T_align.tolist(),
                           "note": "instance npz/table are in the GT-mesh frame; "
                                   "apply T_align to SLAM-frame poses/points"}, f, indent=2)
        from ma_slam_semantic.export import export_all
        res = export_all(self.table, num_frames=self.n_frames,
                         out_dir=self.out_dir, name=self.name, cfg=self.cfg,
                         transformation=T_align)
        print(f"[sem_hooks] exported {res['n_instances']} instances "
              f"({res['n_dropped']} noise-filtered, {res['n_merged']} final-merged, "
              f"{len(self.loop_events)} loop events) -> {self.out_dir}")
