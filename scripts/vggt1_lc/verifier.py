"""Map-free semantic loop verifier for VGGT-SLAM 1.0 (Sim3 mode).

Takes the two frames of a SALAD candidate straight out of VGGT-SLAM's own submap
inference (`pred_dict`: images / depth / depth_conf / extrinsic / intrinsic — one
joint VGGT pass, so both frames share one arbitrary scale), builds a 2-frame chunk
in the OpenSe3r bridge contract, runs our semantic model with a fresh KV cache, and
applies Stage-1 Bi-Desc + Stage-2 layout-graph verification on the raw per-frame
proposal descriptors / camera-frame centres. No instance map, no history: every
candidate is judged from its own two frames, so the verdict is independent of
VGGT-SLAM's drift and of scale (Stage-2 uses the relative t gate `s2_tau_t_rel`).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from ma_slam_semantic.bidesc_lc import BiDescConfig, BiDescVerdict, verify as bidesc_verify, \
    verify_layout, verify_rigid
from ma_slam_semantic.bridge import _center_crop_to_aspect, _crop_resize_K, _crop_resize_map
from ma_slam_semantic.config import SemConfig
from ma_slam_semantic.proposals import make_proposals
from ma_slam_semantic.sem_model import SemanticModel


def _crop_resize_img01(img: np.ndarray, res: Tuple[int, int]) -> np.ndarray:
    """(3,H,W) [0,1] -> center-crop to aspect + AREA resize -> (3,h,w) in [-1,1]."""
    tw, th = res
    hwc = np.ascontiguousarray(img.transpose(1, 2, 0))
    t, l, ch, cw = _center_crop_to_aspect(hwc.shape[0], hwc.shape[1], tw, th)
    hwc = hwc[t:t + ch, l:l + cw]
    out = cv2.resize(hwc, (tw, th), interpolation=cv2.INTER_AREA)
    return (out.transpose(2, 0, 1) * 2.0 - 1.0).astype(np.float32)


def _unproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], axis=-1).astype(np.float32)


class Vggt1SemVerifier:
    def __init__(self, ckpt: str = "a1_pq", device: str = "cuda", dump_dir: str = "",
                 s2_enable: bool = True, tau_t_rel: float = 0.5,
                 conf_percentile: float = 25.0, k_bag: int = 16, mode: str = "bidesc",
                 rigid_tau: float = 0.85, rigid_min_inliers: int = 4):
        # mode: "bidesc" = old S1 Bi-Desc (+S2 layout rules) | "pairs" = S1 instance-correspondence gate
        #       (MNN pairs cos>=rigid_tau, >= rigid_min_inliers, hard reject) | "rigid" = pairs + Kabsch layout
        self.mode = mode
        self.cfg = SemConfig(ckpt=ckpt, device=device, kv_policy="none",
                             box_scale_mode="first_chunk")
        self.model = SemanticModel(self.cfg, device=device)
        # Raw-vs-raw descriptors (no map), absolute t gate off, relative gate on.
        self.lc_cfg = BiDescConfig(desc_src="proposal", s2_enable=s2_enable,
                                   s2_tau_t=0.0, s2_tau_t_rel=tau_t_rel,
                                   dump_dir=dump_dir)
        self.rigid_cfg = BiDescConfig(desc_src="proposal", mode="rigid", rigid_tau=rigid_tau,
                                      rigid_min_inliers=rigid_min_inliers, rigid_lowobj="reject",
                                      rigid_geom=(mode == "rigid"), dump_dir=dump_dir)
        self.conf_percentile = conf_percentile
        self.k_bag = k_bag
        self._dump_n = 0
        self.stats = dict(n=0, accept=0, reject=0)

    # ------------------------------------------------------------------ chunk
    def _build_chunk(self, frames, depth, conf, extr, K, idx: List[int]) -> Dict:
        res = self.cfg.resolution                     # (W, H)
        imgs, deps, Ks, c2ws, wps, vals = [], [], [], [], [], []
        c2w_raw = []
        for t in idx:
            E = np.eye(4, dtype=np.float64)
            E[:3, :4] = np.asarray(extr[t], dtype=np.float64)     # world->cam
            c2w_raw.append(np.linalg.inv(E))
        A = np.linalg.inv(c2w_raw[0])                             # anchor = query frame
        for k, t in enumerate(idx):
            img = frames[t]
            img = img.detach().cpu().numpy() if torch.is_tensor(img) else np.asarray(img)
            imgs.append(_crop_resize_img01(img.astype(np.float32), res))
            d = np.asarray(depth[t], dtype=np.float32)
            if d.ndim == 3:
                d = d[..., 0]
            c = np.asarray(conf[t], dtype=np.float32)
            hw = d.shape
            d_r = _crop_resize_map(d, res)
            c_r = _crop_resize_map(c, res)
            K_r = _crop_resize_K(np.asarray(K[t], dtype=np.float32), hw, res)
            thr = float(np.percentile(c_r, self.conf_percentile))
            valid = (d_r > 1e-6) & (c_r >= thr)
            c2w = (A @ c2w_raw[k]).astype(np.float32)
            P = _unproject(d_r, K_r)
            Pw = P @ c2w[:3, :3].T + c2w[:3, 3]
            deps.append(d_r); Ks.append(K_r); c2ws.append(c2w); wps.append(Pw.astype(np.float32))
            vals.append(valid)
        f32 = lambda a: torch.from_numpy(np.stack(a)).float()
        return dict(images=f32(imgs).unsqueeze(0), depth=f32(deps).unsqueeze(0),
                    intrinsics=f32(Ks).unsqueeze(0), c2w=f32(c2ws).unsqueeze(0),
                    world_points_anchored=f32(wps).unsqueeze(0),
                    valid=torch.from_numpy(np.stack(vals)).unsqueeze(0),
                    world_points_global=f32(wps),
                    conf=f32([v.astype(np.float32) for v in vals]),
                    frame_idx=list(range(len(idx))), frame_keys=[str(t) for t in idx])

    # ----------------------------------------------------------------- verify
    @torch.no_grad()
    def verify(self, frames, depth, conf, extr, K, iq: int, ir: int,
               q_path: str = "", r_path: str = "", salad: float = float("nan")):
        """Returns (accept: bool, info: dict). Frame iq/ir index the joint batch."""
        chunk = self._build_chunk(frames, depth, conf, extr, K, [iq, ir])
        self.model.reset_kv()
        self.model._box_scale = None                  # per-candidate scale (VGGT scale is per-submap)
        out = self.model.forward_chunk(chunk)
        props, _, _ = make_proposals(out, chunk, self.cfg,
                                     sem_backbone=self.model.semantic_backbone)
        c2w = chunk["c2w"][0].numpy()
        per: Dict[int, list] = {0: [], 1: []}
        for p in props:
            t = int(p["frame"])
            d = p["descriptor"]
            d = d.detach().cpu().numpy() if hasattr(d, "detach") else np.asarray(d)
            d = d.astype(np.float32); d /= (np.linalg.norm(d) + 1e-12)
            c = p["center"].detach().cpu().numpy().astype(np.float64)
            T = np.linalg.inv(c2w[t].astype(np.float64))
            c_cam = (T[:3, :3] @ c + T[:3, 3]).astype(np.float32)
            per[t].append((float(p.get("objectness", 0.0)), d, c_cam))
        for t in per:                                  # FrameIndex convention: top-k by objectness
            per[t].sort(key=lambda r: -r[0])
            per[t] = per[t][:self.k_bag]
        D = {t: (np.stack([r[1] for r in per[t]]) if per[t] else np.zeros((0, 256), np.float32))
             for t in per}
        C = {t: (np.stack([r[2] for r in per[t]]) if per[t] else np.zeros((0, 3), np.float32))
             for t in per}
        if self.mode in ("pairs", "rigid"):
            v = verify_rigid(D[0], D[1], C[0], C[1], self.rigid_cfg)
            accept = v.status == "PASS"
            self.stats["n"] += 1
            self.stats["accept" if accept else "reject"] += 1
            info = dict(s1=v.summary(), s2="-", n_q=len(D[0]), n_r=len(D[1]), accept=accept)
            if self.lc_cfg.dump_dir:
                try:
                    self._dump(q_path, r_path, v, per, chunk, salad)
                except Exception as e:                  # RigidVerdict lacks some BiDesc fields
                    print(f"[verifier] dump skipped: {e}")
            return accept, info
        v = bidesc_verify(D[0], D[1], self.lc_cfg)
        v.layout = None
        if self.lc_cfg.s2_enable and v.status == "PASS" and v.pairs:
            uniq = [(p["q"], p["r"]) for p in v.pairs if p["uniq"]]
            if uniq:
                qi, ri = zip(*uniq)
                v.layout = verify_layout(C[0][list(qi)], C[1][list(ri)], self.lc_cfg)
        accept = v.status == "PASS" and (not self.lc_cfg.s2_enable or
                                         (v.layout is not None and v.layout.status == "PASS"))
        self.stats["n"] += 1
        self.stats["accept" if accept else "reject"] += 1
        s1 = (f"{v.status} a_qr={v.a_qr:.2f} a_rq={v.a_rq:.2f} a_bi={v.a_bi:.2f} "
              f"nuniq={v.n_uniq} nq/nr={v.n_q}/{v.n_r}" + (f" [{v.reason}]" if v.reason else ""))
        lay = v.layout
        s2 = ("-" if lay is None else
              f"{lay.status} M={lay.m} Ed={lay.e_d:.3f} thR={lay.theta_r:.0f} ER={lay.e_r:.1f} "
              f"tnorm={lay.t_norm:.2f} dref={lay.d_ref:.2f}" + (f" [{lay.reason}]" if lay.reason else ""))
        info = dict(s1=s1, s2=s2,
                    n_q=len(D[0]), n_r=len(D[1]), accept=accept)
        if self.lc_cfg.dump_dir:
            self._dump(q_path, r_path, v, per, chunk, salad)
        return accept, info

    # ------------------------------------------------------------------- dump
    def _dump(self, q_path, r_path, v, per, chunk, salad):
        os.makedirs(self.lc_cfg.dump_dir, exist_ok=True)
        K = chunk["intrinsics"][0].numpy()

        def nodes(t):
            return [dict(row=k, inst_id=-1, obj=float(ob), xyz=c.tolist())
                    for k, (ob, _, c) in enumerate(per[t])]
        lay = v.layout
        rec = dict(
            q_path=q_path, r_path=r_path, salad=salad,
            s1=dict(status=v.status, a_qr=v.a_qr, a_rq=v.a_rq, a_bi=v.a_bi,
                    n_uniq=v.n_uniq, n_q=v.n_q, n_r=v.n_r, reason=v.reason),
            s2=(dict(status=lay.status, m=lay.m, e_d=lay.e_d, theta_r=lay.theta_r,
                     e_r=lay.e_r, e_scale=lay.e_scale, t_norm=lay.t_norm,
                     s2=lay.s2, d_ref=lay.d_ref, reason=lay.reason)
                if lay is not None else None),
            pairs=[dict(q_row=p["q"], r_row=p["r"], sim=p["sim"], uniq=p["uniq"])
                   for p in v.pairs],
            q_nodes=nodes(0), r_nodes=nodes(1), K_q=K[0].tolist(), K_r=K[1].tolist(),
            cfg=dict(s2_tau_d=self.lc_cfg.s2_tau_d, s2_tau_r_deg=self.lc_cfg.s2_tau_r_deg,
                     s2_tau_theta_deg=self.lc_cfg.s2_tau_theta_deg,
                     s2_tau_t=self.lc_cfg.s2_tau_t, s2_tau_t_rel=self.lc_cfg.s2_tau_t_rel,
                     s2_min_m=self.lc_cfg.s2_min_m, s2_min_m_rot=self.lc_cfg.s2_min_m_rot,
                     resolution=list(self.cfg.resolution)))
        self._dump_n += 1
        qn = os.path.splitext(os.path.basename(q_path))[0]
        rn = os.path.splitext(os.path.basename(r_path))[0]
        with open(os.path.join(self.lc_cfg.dump_dir, f"{self._dump_n:04d}_{qn}__{rn}.json"), "w") as f:
            json.dump(rec, f)
