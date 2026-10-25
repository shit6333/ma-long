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

import numpy as np

from ma_slam_semantic.config import SemConfig
from ma_slam_semantic.bridge import SubmapBridge, PoseTracker
from ma_slam_semantic.sem_model import SemanticModel
from ma_slam_semantic.proposals import make_proposals
from ma_slam_semantic.instance_map import InstanceTable, SlamGeoResolver
from ma_slam_semantic.bidesc_lc import (BiDescConfig, BiDescVerdict, FrameIndex,
                                        verify as bidesc_verify, verify_layout)


class SemanticHooks:
    def __init__(self, cfg: SemConfig, out_dir: str, name: str = "scene",
                 device: Optional[str] = None, gt_positions=None, gt_poses=None, align_mode: str = "pos",
                 lc_cfg: Optional[BiDescConfig] = None, gt_mesh: Optional[str] = None):
        self.cfg = cfg
        self.gt_mesh = gt_mesh                  # GT surface (ply) for align_mode 'sim3map' (1-DoF global scale fit)
        self.out_dir = out_dir
        self.name = name
        # GT camera positions [N,3] (row i = frame i). When set, exports are rigidly
        # mapped into the GT-mesh frame via SE3 Umeyama over the WHOLE trajectory —
        # more robust than first-frame anchoring, whose frame-0 estimation error
        # would shift the entire map (visible as depressed AP on drifty scenes).
        self.gt_positions = gt_positions
        self.gt_poses = gt_poses                  # (N,4,4) c2w, optional (orientation-aware alignment)
        self.align_mode = align_mode
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
        self._dump_n = 0        # candidate counter for lc_cfg.dump_dir file naming
        # incremental reconstruction snapshots (cfg.snapshot_dir) -> video offline
        self.snap = None
        if getattr(cfg, "snapshot_dir", ""):
            from ma_slam_semantic.snapshots import SnapshotWriter
            self.snap = SnapshotWriter(cfg.snapshot_dir, conf_coef=cfg.conf_gate_coef,
                                       max_pts_per_submap=getattr(cfg, "snapshot_pts", 50000))

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
            props, pix_map, _ = make_proposals(out, sub, self.cfg,
                                               sem_backbone=self.model.semantic_backbone)
            assign = self.table.match_chunk(props, sub["frame_keys"], pix_map)
            paths = [sm.image_paths[i] for i in sub["frame_idx"]]
            cb = getattr(self, "chunk_callback", None)    # e.g. the stream viewer
            if cb is not None:
                cb(self.table, paths)
            # Camera-frame 3D centres for Stage-2 layout graphs: undo the frame's
            # global pose so the stored position is a pure per-frame observation
            # (drift-free, healing-independent). Per-axis median doesn't strictly
            # commute with rotation, but at voxel-centroid noise level it's fine.
            inv_G = {i: np.linalg.inv(sm.frame_pose_global(graph, i))
                     for i in sub["frame_idx"]}
            per_frame: dict = {}
            for k, p in enumerate(props):
                if assign[k] >= 0:
                    d = p["descriptor"]
                    t = int(p["frame"])
                    c = p["center"].detach().cpu().numpy().astype(np.float64)
                    T = inv_G[sub["frame_idx"][t]]
                    c_cam = (T[:3, :3] @ c + T[:3, 3]).astype(np.float32)
                    per_frame.setdefault(t, []).append(
                        ((assign[k], float(p.get("objectness", 0.0))),
                         d.detach().cpu().numpy() if hasattr(d, "detach") else d,
                         c_cam))
            K_sub = sub["intrinsics"][0].cpu().numpy()   # (T,3,3) at semantic res
            for t, rows in per_frame.items():
                self.frame_index.add(paths[t], [r[0] for r in rows],
                                     descs=[r[1] for r in rows],
                                     centers=[r[2] for r in rows],
                                     K=K_sub[t])
            n_props += len(props)
            # Transient LC evidence bag (user decision 2026-08-29): the current frame's
            # verification evidence may use a looser proposal mode than the map, but those
            # proposals are never matched into the InstanceTable.
            if getattr(self.cfg, "lc_proposal_conf_mode", "gate") != self.cfg.proposal_conf_mode:
                import copy as _copy
                cfg_lc = _copy.copy(self.cfg)
                cfg_lc.proposal_conf_mode = self.cfg.lc_proposal_conf_mode
                props_lc, _, _ = make_proposals(out, sub, cfg_lc,
                                                sem_backbone=self.model.semantic_backbone)
                bag: dict = {t: [] for t in range(len(paths))}
                for p in props_lc:
                    t = int(p["frame"])
                    d = p["descriptor"]
                    c = p["center"].detach().cpu().numpy().astype(np.float64)
                    T = inv_G[sub["frame_idx"][t]]
                    c_cam = (T[:3, :3] @ c + T[:3, 3]).astype(np.float32)
                    bag[t].append((float(p.get("objectness", 0.0)),
                                   d.detach().cpu().numpy() if hasattr(d, "detach") else d, c_cam))
                for t, rows in bag.items():
                    rows.sort(key=lambda r: -r[0])
                    rows = rows[:self.lc_cfg.k_bag]
                    self.frame_index.add_lc(paths[t], [r[1] for r in rows], [r[2] for r in rows])
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
        if self.snap is not None:
            self.snap.write_step(gmap, graph, table=self.table,
                                 loop_events=self.loop_events,
                                 voxel_size=self.cfg.voxel_size, n_frames=self.n_frames)

    def verify_loop_candidate(self, q_path: str, m_path: str):
        """Bi-Desc verdict for a SALAD loop candidate (solver-initiated; read-only).
        UNKNOWN when either frame has no recorded ids / too few instances."""
        ids_q = self.frame_index.get(q_path)
        ids_r = self.frame_index.get(m_path)
        lc_q = self.frame_index.get_lc(q_path) if self.lc_cfg.mode == "rigid" else None
        lc_r = self.frame_index.get_lc(m_path) if (lc_q is not None and
                                                    getattr(self.cfg, "lc_history", "map") == "bag") else None
        if lc_q is not None and lc_r is not None:
            from ma_slam_semantic.bidesc_lc import verify_rigid
            D_q, C_q = lc_q
            D_r, C_r = lc_r
            v = verify_rigid(D_q, D_r, C_q, C_r, self.lc_cfg)
            if self.lc_cfg.dump_dir:
                self._dump_candidate(q_path, m_path, v, ids_q or [], ids_r or [],
                                     list(range(len(D_q))), list(range(len(D_r))),
                                     nodes_q=C_q, nodes_r=C_r)
            return v
        if lc_q is not None and ids_r and self.table is not None:
            # rigid mode with a transient query bag: history side = map instances of the
            # history frame (aggregated descriptors + that frame's camera-frame centres)
            from ma_slam_semantic.bidesc_lc import verify_rigid
            D_q, C_q = lc_q
            ds, rows = [], []
            for k, (iid, _) in enumerate(ids_r):
                rec = self.table.records.get(iid)
                if rec is None:
                    continue
                d = rec.descriptor.numpy().astype(np.float32)
                ds.append(d / (np.linalg.norm(d) + 1e-12)); rows.append(k)
            D_r = np.stack(ds) if ds else np.zeros((0, 256), np.float32)
            C_all = self.frame_index.get_centers(m_path)
            C_r = (C_all[rows] if C_all is not None and rows and max(rows) < len(C_all)
                   else np.zeros((len(D_r), 3), np.float32))
            v = verify_rigid(D_q, D_r, C_q, C_r, self.lc_cfg)
            if self.lc_cfg.dump_dir:
                self._dump_candidate(q_path, m_path, v, ids_q or [], ids_r, list(range(len(D_q))), rows)
            return v
        if not ids_q or not ids_r or self.table is None:
            if self.lc_cfg.mode == "rigid":
                from ma_slam_semantic.bidesc_lc import RigidVerdict
                st = "UNKNOWN" if self.lc_cfg.rigid_lowobj == "coloc" else "REJECT"
                v = RigidVerdict(st, n_q=0 if not ids_q else len(ids_q),
                                 n_r=0 if not ids_r else len(ids_r), reason="no frame index")
                if self.lc_cfg.dump_dir:
                    self._dump_candidate(q_path, m_path, v, ids_q or [], ids_r or [], [], [])
                return v
            v = BiDescVerdict("UNKNOWN",
                              n_q=0 if not ids_q else len(ids_q),
                              n_r=0 if not ids_r else len(ids_r),
                              reason="no frame index")
            if self.lc_cfg.dump_dir:
                # dump these too — the UNKNOWN-accept path is exactly the case a
                # reviewer wants to eyeball (frame with no recorded instances).
                self._dump_candidate(q_path, m_path, v, ids_q or [], ids_r or [],
                                     [], [])
            return v
        # history side is always the map's running mean (it IS the memory); the
        # current side is either the same lookup ('map') or the frame's raw
        # proposal descriptors ('proposal' — independent of the association layer).
        # ``rows`` maps each descriptor row back to its FrameIndex row so the
        # Stage-2 centres stay aligned even when a lookup skips a missing id.
        def _lookup(ids):
            ds, rows = [], []
            for k, (iid, _) in enumerate(ids):
                rec = self.table.records.get(iid)
                if rec is None:
                    continue
                d = rec.descriptor.numpy().astype(np.float32)
                ds.append(d / (np.linalg.norm(d) + 1e-12))
                rows.append(k)
            return (np.stack(ds) if ds else np.zeros((0, 256), np.float32)), rows

        if self.lc_cfg.desc_src == "proposal":
            D_q = self.frame_index.get_descs(q_path)
            q_rows = list(range(len(D_q))) if D_q is not None else []
            if D_q is None or len(D_q) != len(ids_q):
                D_q, q_rows = _lookup(ids_q)                    # pre-switch frames
        else:
            D_q, q_rows = _lookup(ids_q)
        D_r, r_rows = _lookup(ids_r)
        if self.lc_cfg.mode == "rigid":
            from ma_slam_semantic.bidesc_lc import verify_rigid
            C_q = self.frame_index.get_centers(q_path)
            C_r = self.frame_index.get_centers(m_path)
            Cq = (C_q[q_rows] if C_q is not None and len(q_rows) and max(q_rows) < len(C_q)
                  else np.zeros((len(D_q), 3), np.float32))
            Cr = (C_r[r_rows] if C_r is not None and len(r_rows) and max(r_rows) < len(C_r)
                  else np.zeros((len(D_r), 3), np.float32))
            v = verify_rigid(D_q, D_r, Cq, Cr, self.lc_cfg)
            if self.lc_cfg.dump_dir:
                self._dump_candidate(q_path, m_path, v, ids_q, ids_r, q_rows, r_rows)
            return v
        v = bidesc_verify(D_q, D_r, self.lc_cfg)

        # Stage 2 (shadow): layout graphs over the Stage-1 unique matches.
        C_q = self.frame_index.get_centers(q_path)
        C_r = self.frame_index.get_centers(m_path)
        if self.lc_cfg.s2_enable and C_q is not None and C_r is not None and v.pairs:
            uniq = [(q_rows[p["q"]], r_rows[p["r"]]) for p in v.pairs if p["uniq"]]
            if uniq:
                qi, ri = zip(*uniq)
                v.layout = verify_layout(C_q[list(qi)], C_r[list(ri)], self.lc_cfg)
        if self.lc_cfg.dump_dir:
            self._dump_candidate(q_path, m_path, v, ids_q, ids_r, q_rows, r_rows)
        return v

    def verify_joint(self, out: dict, iq: int, ir: int, q_path: str, m_path: str):
        """Joint verification (cfg.lc mode 'joint'): the solver already re-inferred the loop
        window jointly (`out`: images/depth/conf/poses/intrinsics of query+history frames in
        one local frame). Embed the two candidate frames TOGETHER with a fresh KV (the a1_pq
        descriptor is context-dependent: same object embedded in different chunks does not
        match, embedded together it does), build per-frame bags, and apply the rigid verdict.
        The map's KV cache / box scale are saved and restored around the extra forward."""
        import torch
        from ma_slam_semantic.bidesc_lc import verify_rigid
        from ma_slam_semantic.bridge import _center_crop_to_aspect, _crop_resize_K, _crop_resize_map
        import cv2, copy as _copy
        res = self.cfg.resolution
        imgs = out["images"]; depth = out["depth"]; conf = out.get("world_points_conf"); poses = out["poses"]; Ks = out["intrinsics"]
        anchor = np.linalg.inv(poses[iq])
        f_imgs, f_dep, f_K, f_c2w, f_wp, f_val = [], [], [], [], [], []
        for t in (iq, ir):
            im = np.asarray(imgs[t])
            if im.ndim == 3 and im.shape[0] == 3: im = im.transpose(1, 2, 0)
            im = im.astype(np.float32); im = im / 255.0 if im.max() > 1.5 else im
            tw, th = res; tt, ll, ch, cw = _center_crop_to_aspect(im.shape[0], im.shape[1], tw, th)
            im = cv2.resize(np.ascontiguousarray(im[tt:tt + ch, ll:ll + cw]), (tw, th), interpolation=cv2.INTER_AREA)
            f_imgs.append((im.transpose(2, 0, 1) * 2.0 - 1.0).astype(np.float32))
            d = np.asarray(depth[t], np.float32); d = d[..., 0] if d.ndim == 3 else d
            c = np.asarray(conf[t], np.float32) if conf is not None else np.ones_like(d)
            hw = d.shape; d_r = _crop_resize_map(d, res); c_r = _crop_resize_map(c, res)
            K_r = _crop_resize_K(np.asarray(Ks[t], np.float32), hw, res)
            thr = float(np.percentile(c_r, 25.0)); valid = (d_r > 1e-6) & (c_r >= thr)
            c2w = (anchor @ poses[t]).astype(np.float32)
            u, v = np.meshgrid(np.arange(res[0], dtype=np.float32), np.arange(res[1], dtype=np.float32))
            P = np.stack([(u - K_r[0, 2]) / K_r[0, 0] * d_r, (v - K_r[1, 2]) / K_r[1, 1] * d_r, d_r], -1)
            f_dep.append(d_r); f_K.append(K_r); f_c2w.append(c2w); f_wp.append((P @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32)); f_val.append(valid)
        f32 = lambda a: torch.from_numpy(np.stack(a)).float()
        chunk = dict(images=f32(f_imgs).unsqueeze(0), depth=f32(f_dep).unsqueeze(0), intrinsics=f32(f_K).unsqueeze(0),
                     c2w=f32(f_c2w).unsqueeze(0), world_points_anchored=f32(f_wp).unsqueeze(0),
                     valid=torch.from_numpy(np.stack(f_val)).unsqueeze(0), world_points_global=f32(f_wp),
                     conf=f32([v.astype(np.float32) for v in f_val]), frame_idx=[0, 1], frame_keys=[q_path, m_path])
        kv_backup, bs_backup = self.model.kv, self.model._box_scale
        try:
            self.model.reset_kv(); self.model._box_scale = None
            o = self.model.forward_chunk(chunk)
            cfg_lc = _copy.copy(self.cfg); cfg_lc.proposal_conf_mode = getattr(self.cfg, "lc_proposal_conf_mode", "prefer")
            props, _, _ = make_proposals(o, chunk, cfg_lc, sem_backbone=self.model.semantic_backbone)
        finally:
            self.model.kv, self.model._box_scale = kv_backup, bs_backup
        c2w_np = chunk["c2w"][0].numpy(); per = {0: [], 1: []}
        for pr in props:
            t = int(pr["frame"]); d = pr["descriptor"]; d = d.detach().cpu().numpy() if hasattr(d, "detach") else np.asarray(d)
            d = d.astype(np.float32); d /= (np.linalg.norm(d) + 1e-12)
            c = pr["center"].detach().cpu().numpy().astype(np.float64); T = np.linalg.inv(c2w_np[t].astype(np.float64))
            per[t].append((float(pr.get("objectness", 0.0)), d, (T[:3, :3] @ c + T[:3, 3]).astype(np.float32)))
        for t in per: per[t].sort(key=lambda r: -r[0]); per[t] = per[t][:self.lc_cfg.k_bag]
        D = {t: (np.stack([r[1] for r in per[t]]) if per[t] else np.zeros((0, 256), np.float32)) for t in per}
        C = {t: (np.stack([r[2] for r in per[t]]) if per[t] else np.zeros((0, 3), np.float32)) for t in per}
        v = verify_rigid(D[0], D[1], C[0], C[1], self.lc_cfg)
        if self.lc_cfg.dump_dir:
            self._dump_candidate(q_path, m_path, v, [], [], list(range(len(D[0]))), list(range(len(D[1]))), nodes_q=C[0], nodes_r=C[1])
        return v

    def _dump_candidate(self, q_path, m_path, v, ids_q, ids_r, q_rows, r_rows, nodes_q=None, nodes_r=None):
        """One JSON per verified candidate for scripts/render_semlc_graph.py:
        full node lists (also unmatched — the M_LOW story needs them), MNN pairs
        mapped back to FrameIndex rows, per-frame K, and both verdicts."""
        import json
        os.makedirs(self.lc_cfg.dump_dir, exist_ok=True)

        def nodes(path, ids, explicit=None):
            if explicit is not None:                    # transient bag rows (rigid/bag mode)
                return [dict(row=k, inst_id=-1, obj=0.0, xyz=[float(x) for x in c]) for k, c in enumerate(explicit)]
            C = self.frame_index.get_centers(path)
            return [dict(row=k, inst_id=int(iid), obj=float(ob),
                         xyz=(C[k].tolist() if C is not None and k < len(C) else None))
                    for k, (iid, ob) in enumerate(ids)]

        def kmat(path):
            K = self.frame_index.get_K(path)
            return K.tolist() if K is not None else None

        lay = getattr(v, "layout", None)
        rig = (dict(status=v.status, n_pairs=v.n_pairs, inliers=v.inliers, eps=v.eps,
                    d_ref=v.d_ref, reason=v.reason) if getattr(v, "rigid", False) else None)
        rec = dict(rigid=rig,
            q_path=q_path, r_path=m_path,
            s1=dict(status=v.status, a_qr=v.a_qr, a_rq=v.a_rq, a_bi=v.a_bi,
                    n_uniq=v.n_uniq, n_q=v.n_q, n_r=v.n_r, reason=v.reason),
            s2=(dict(status=lay.status, m=lay.m, e_d=lay.e_d, theta_r=lay.theta_r,
                     e_r=lay.e_r, e_scale=lay.e_scale, t_norm=lay.t_norm,
                     s2=lay.s2, d_ref=lay.d_ref, reason=lay.reason)
                if lay is not None else None),
            pairs=[dict(q_row=q_rows[p["q"]], r_row=r_rows[p["r"]],
                        sim=p["sim"], uniq=p["uniq"]) for p in v.pairs],
            q_nodes=nodes(q_path, ids_q, nodes_q), r_nodes=nodes(m_path, ids_r, nodes_r),
            K_q=kmat(q_path), K_r=kmat(m_path),
            cfg=dict(s2_tau_d=self.lc_cfg.s2_tau_d,
                     s2_tau_r_deg=self.lc_cfg.s2_tau_r_deg,
                     s2_tau_theta_deg=self.lc_cfg.s2_tau_theta_deg,
                     s2_tau_t=self.lc_cfg.s2_tau_t,
                     s2_min_m=self.lc_cfg.s2_min_m,
                     s2_min_m_rot=self.lc_cfg.s2_min_m_rot,
                     resolution=list(self.cfg.resolution)))
        self._dump_n += 1
        qn = os.path.splitext(os.path.basename(q_path))[0]
        rn = os.path.splitext(os.path.basename(m_path))[0]
        out = os.path.join(self.lc_cfg.dump_dir, f"{self._dump_n:04d}_{qn}__{rn}.json")
        with open(out, "w") as f:
            json.dump(rec, f)

    def on_loop(self, key_new: int, key_old: int) -> None:
        # Healing is handled by the M-signature mechanism (stale records re-lift
        # lazily on their next match / at export); just record the event.
        self.loop_events.append((key_new, key_old))
        print(f"[sem_hooks] loop event {key_new} <-> {key_old} (instances will re-key lazily)")

    def on_finish(self, gmap, graph) -> None:
        if self.table is None:
            return
        if self.snap is not None:          # final optimized map as the last frame
            self.snap.write_step(gmap, graph, table=self.table,
                                 loop_events=self.loop_events,
                                 voxel_size=self.cfg.voxel_size, n_frames=self.n_frames)
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
            centres = {}; rots = {}
            for sm in gmap.ordered():
                if sm.is_lc:
                    continue
                for i in range(sm.n):
                    P = graph.get_pose(sm.key(i))
                    centres[int(sm.frame_ids[i])] = P[:3, 3]; rots[int(sm.frame_ids[i])] = P[:3, :3]
            gt_all = np.asarray(self.gt_positions)
            fids = sorted(f for f in centres if 0 <= f < len(gt_all))
            est = np.stack([centres[f] for f in fids])
            gt = gt_all[fids]
            s_al, R, _ = umeyama_sim3(est, gt)       # rigid: rotation + centroid shift (camera centres only)
            R_pos, t_pos = R.copy(), gt.mean(0) - R @ est.mean(0)          # SE3 (scale 1) — always computed
            T_map = None                                                  # SE3 + one global scale fitted to the GT surface
            if self.gt_mesh and os.path.exists(self.gt_mesh):
                import open3d as o3d
                from scipy.spatial import cKDTree
                msh = o3d.io.read_triangle_mesh(self.gt_mesh)
                G = np.asarray(msh.vertices) if len(msh.vertices) else np.asarray(o3d.io.read_point_cloud(self.gt_mesh).points)
                tree = cKDTree(G)
                nonlc = [sm for sm in gmap.ordered() if not sm.is_lc]
                rng = np.random.default_rng(0); pts = []
                for sm in nonlc:
                    p_, _ = sm.world_points_and_colors(graph, conf_coef=1.0, max_points=max(1, 200_000 // max(1, len(nonlc))), rng=rng)
                    if len(p_):
                        pts.append(p_)
                P = np.concatenate(pts, 0)
                best = None
                for s_try in np.arange(0.85, 1.1501, 0.01):
                    _, R_s, _ = umeyama_sim3(est * s_try, gt)
                    t_s = gt.mean(0) - R_s @ (est * s_try).mean(0)
                    d, _ = tree.query((P * s_try) @ R_s.T + t_s)
                    med = float(np.median(d))
                    if best is None or med < best[1]:
                        best = (float(s_try), med, R_s)
                d1, _ = tree.query(P @ R_pos.T + t_pos)
                print(f"[sem_hooks] sim3map align: global scale {best[0]:.2f} (map->GT-surface median {best[1]:.3f} m; "
                      f"s=1.00 gives {float(np.median(d1)):.3f} m; trajectory-Umeyama scale {float(s_al):.3f})")
                R_m = best[0] * best[2]; T_map = np.eye(4); T_map[:3, :3], T_map[:3, 3] = R_m, gt.mean(0) - R_m @ est.mean(0)
            elif self.align_mode == "sim3map":
                raise SystemExit(f"--align sim3map needs --gt_mesh <ply> (got {self.gt_mesh!r})")
            if self.align_mode == "sim3":             # similarity from the trajectory: T = [sR | t] (logged; unreliable on short/drifted trajectories)
                R = float(s_al) * R
                print(f"[sem_hooks] Sim3 align: scale est->gt = {float(s_al):.4f}")
            elif self.align_mode == "sim3map":
                R = T_map[:3, :3]
            t = gt.mean(0) - R @ est.mean(0)
            rmse = float(np.sqrt(np.mean(np.sum(((est @ R.T + t) - gt) ** 2, axis=1))))
            msg = f"[sem_hooks] SE3-Umeyama align to GT ({len(fids)} frames): residual RMSE {rmse:.3f} m"
            if self.gt_poses is not None:
                # orientation-aware rigid alignment: R from the camera axes (Kabsch over 3 axis vectors per
                # frame), t from centroids. Camera centres alone leave the tilt of a planar trajectory
                # ill-conditioned; the orientations pin it down.
                Rg = np.asarray(self.gt_poses)[fids, :3, :3]; Re = np.stack([rots[f] for f in fids])
                A = np.concatenate([Re[:, :, k] for k in range(3)], 0); B = np.concatenate([Rg[:, :, k] for k in range(3)], 0)
                U, _, Vt = np.linalg.svd(A.T @ B); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
                Ro = Vt.T @ D @ U.T; to = gt.mean(0) - Ro @ est.mean(0)
                rmse_o = float(np.sqrt(np.mean(np.sum(((est @ Ro.T + to) - gt) ** 2, axis=1))))
                def _rerr(Ra):
                    return float(np.mean([np.degrees(np.arccos(np.clip((np.trace((Ra @ Re[i]).T @ Rg[i]) - 1) / 2, -1, 1))) for i in range(len(fids))]))
                ang = float(np.degrees(np.arccos(np.clip((np.trace(R.T @ Ro) - 1) / 2, -1, 1))))
                msg += (f" | cam-rot err {_rerr(R):.1f} deg || orientation-aware: residual {rmse_o:.3f} m, "
                        f"cam-rot err {_rerr(Ro):.1f} deg, differs by {ang:.1f} deg -> using '{self.align_mode}'")
                if self.align_mode == "pose":
                    R, t, rmse = Ro, to, rmse_o
            T_align = np.eye(4)
            T_align[:3, :3], T_align[:3, 3] = R, t
            print(msg)
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
        # paired secondary export of the SAME map under the other alignment (SE3 <-> sim3map), so the
        # alignment choice can be compared without run-to-run SLAM noise.
        sec = None
        if T_align is not None and self.gt_positions is not None:
            if self.align_mode == "sim3map":
                T2 = np.eye(4); T2[:3, :3], T2[:3, 3] = R_pos, t_pos; sec = ("pos", T2)
            elif self.align_mode == "pos" and T_map is not None:
                sec = ("sim3map", T_map)
        if sec is not None:
            sub = os.path.join(self.out_dir, f"align_{sec[0]}")
            res2 = export_all(self.table, num_frames=self.n_frames, out_dir=sub, name=self.name, cfg=self.cfg,
                              transformation=sec[1], do_final_merge=False)
            print(f"[sem_hooks] secondary export (align={sec[0]}): {res2['n_instances']} instances -> {sub}")
