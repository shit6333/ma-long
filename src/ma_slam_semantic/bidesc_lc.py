"""Bi-Desc LC (v2) — bidirectional instance-descriptor verification of SALAD loop
candidates, VGGT-SLAM-2.0-inspired redesign of instance_lc (which is retired).

Per-frame storage is the instance IDs a frame's proposals were associated to plus
(by default) their raw descriptors — no extra model forward, no area filtering
(what the model proposes is what the gate sees, capped at ``k_bag`` by objectness).
The CURRENT side uses those raw proposal descriptors and the HISTORY side the
InstanceTable's running mean, so every score is a genuine cross-frame descriptor
similarity. Looking BOTH sides up by id (``desc_src='map'``) is kept as an
ablation, but measured 92% of its counted pairs at exactly 1.00 — both ids resolve
to the same vector, degenerating the test into an id intersection that inherits
the association layer's decisions (55% of whose weight is geometry).

Verdict (three-state, shadow-first):
  S = D_q D_r^T (cosine); directional scores a_qr / a_rq = mean of the top-K row/
  column maxima; a_bi = harmonic mean (punishes one-sided support); N_uniq =
  mutual-nearest-neighbour pairs above tau_pair (kills many-to-one inflation).
  UNKNOWN if either side has < n_min instances (abstain, never veto);
  PASS if N_uniq >= n_uniq and a_qr > tau_dir and a_rq > tau_dir and a_bi > tau_bi;
  FAIL otherwise.
All MNN pairs (also sub-threshold ones) are kept on the verdict for BIDESCPAIRS
logging — the "what sim did the rejected assignment have" debugging blind spot
of v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- config
@dataclass
class BiDescConfig:
    n_min: int = 2          # < n_min instances on either side -> UNKNOWN
    tau_pair: float = 0.85  # MNN unique-match cos (calibrated 2026-07: PASS must sit in the id-consistent ~1.0 cluster; 0.7-0.85 is the same-type aliasing band)
    tau_dir: float = 0.85   # per-direction MeanTopK threshold (calibrated, see tau_pair)
    tau_bi: float = 0.85    # harmonic-mean threshold (calibrated, see tau_pair)
    n_uniq: int = 2         # min unique MNN matches for PASS
    k_lo: int = 2           # adaptive top-K = clip(ceil(N*k_frac), k_lo, k_hi)
    k_hi: int = 4
    k_frac: float = 0.5
    k_bag: int = 16         # per-frame id cap (by objectness); no area filtering
    eps: float = 1e-6
    # Descriptor source for the CURRENT side.
    #   'map'      — look the frame's instance ids up in InstanceTable (running mean).
    #                Measured: 92% of counted pairs then score exactly 1.00 because both
    #                sides resolve to the SAME id, i.e. the test degenerates into an
    #                id-intersection that inherits match_chunk's decisions (which are
    #                55% geometry: w_vox .3 + w_geo .25 vs w_desc .3).
    #   'proposal' — use the frame's RAW proposal descriptors. Every score is then a
    #                genuine cross-frame descriptor similarity, independent of the
    #                association layer — so the gate still works when large drift made
    #                match_chunk spawn new ids instead of merging (exactly the loops
    #                that matter most). Costs ~16 KB/frame of storage.
    # The history side always uses the map's running mean (it IS the memory).
    desc_src: str = "proposal"

    # ---- Stage 2: instance layout graph (stage2_instance_graph_lc_design.md) ----
    # Camera-frame 3D positions of the Stage-1-matched instances form two graphs;
    # corresponding edges are compared via three metrics (all closed-form):
    #   E_d     = median |log(d_q/d_r)|          distance consistency
    #   theta_R = angle of R* = Wahba argmin     common rotation magnitude
    #   E_R     = median angle(u_q, R* u_r)      rotation consistency residual
    # Decision (doc section 7): Ed<=tau_d and ER<=tau_R -> (thetaR<=tau_theta ?
    # PASS : LARGE_VIEW_CHANGE) else FAIL. Shadow-only for now (logged, not acted on).
    # Calibrated on the 6-scene shadow + GT audit (2026-07-29 rescore: 13/13 real
    # edges kept, 8/8 coloc-mis-accepted false edges removed):
    #   - each metric votes only where OBSERVABLE: distance needs 1 edge (M>=2);
    #     rotation needs >=2 non-parallel edges and translation depends on R*, so
    #     both abstain when M < s2_min_m_rot (Wahba is rank-deficient / wild there);
    #   - ||t*|| (t* = mean(X_q - R* X_r), i.e. the instance-evidence estimate of
    #     the camera-to-camera distance) gates translation-type false loops that
    #     all edge-based metrics are blind to by construction;
    #   - tau_d at the doc range's lower edge recovers rotation-type catches that
    #     the M-rule would otherwise hand back (e.g. plant 730<->233, M=3).
    s2_tau_d: float = 0.3        # doc range 0.3-0.5 (log ratio, ~±35%)
    s2_tau_r_deg: float = 25.0   # doc range 20-30 deg
    s2_tau_theta_deg: float = 50.0  # doc range 45-60 deg
    s2_tau_t: float = 1.0        # m; 0.75 sat inside the estimator's ~0.3 m noise
                                 # band (w32 killed a real edge at tnorm 0.97 vs GT
                                 # 0.41); 1.0 dominates on the merged w20+w32 audit.
                                 # TODO general form: tnorm / median instance depth
                                 # (d_ref, now logged) — scale-free like coloc.
    s2_min_m: int = 3            # M<=2 -> M_LOW hard reject: 1-2 co-visible objects
                                 # cannot support ANY verdict (single-edge chance
                                 # consistency let w32 730<->233 through; M=1 has no
                                 # edges at all — floor 463<->79, coloc-veto-proof).
                                 # Real loops come in clusters, so losing a thin-
                                 # evidence member is cheap (redundancy).
    s2_min_m_rot: int = 4        # below this, rotation/translation metrics abstain

    # Debug: when set, verify_loop_candidate dumps one JSON per candidate here
    # (both frames' full node lists with camera-frame XYZ, the MNN pairs, per-frame
    # K, and the S1+S2 verdicts) for scripts/render_semlc_graph.py.
    dump_dir: str = ""


def adaptive_k(n: int, cfg: BiDescConfig) -> int:
    return max(1, min(n, max(cfg.k_lo, min(cfg.k_hi, int(np.ceil(cfg.k_frac * n))))))


# --------------------------------------------------------------------------- storage
class FrameIndex:
    """path -> [(inst_id, objectness)] recorded at mapping time (Obs-equivalent
    provenance, inverted) — bytes per frame.

    With ``keep_desc`` it additionally retains each frame's RAW proposal
    descriptors (~16 KB/frame at k_bag=16), which ``desc_src='proposal'`` needs:
    the raw vectors are otherwise dropped right after ``match_chunk`` (only the
    per-instance running mean survives, inside InstanceRecord)."""

    def __init__(self, k_bag: int = 16, keep_desc: bool = False):
        self.k_bag = k_bag
        self.keep_desc = keep_desc
        self._ids: Dict[str, List[Tuple[int, float]]] = {}
        self._descs: Dict[str, np.ndarray] = {}          # path -> (K,256) L2-normed
        self._cents: Dict[str, np.ndarray] = {}          # path -> (K,3) CAMERA-frame 3D
        self._K: Dict[str, np.ndarray] = {}              # path -> (3,3) K at semantic res

    def add(self, path: str, items: Sequence[Tuple[int, float]],
            descs: Optional[Sequence[np.ndarray]] = None,
            centers: Optional[Sequence[np.ndarray]] = None,
            K: Optional[np.ndarray] = None) -> None:
        """``items[i]`` = (inst_id, objectness); ``descs[i]`` / ``centers[i]`` = that
        proposal's raw descriptor / CAMERA-frame 3D centre (same order). All are
        truncated to the same top-k_bag subset, so indices stay aligned."""
        order = sorted(range(len(items)), key=lambda i: -items[i][1])[: self.k_bag]
        self._ids[path] = [(int(items[i][0]), float(items[i][1])) for i in order]
        if self.keep_desc and descs is not None:
            D = np.stack([np.asarray(descs[i], np.float32) for i in order]) \
                if order else np.zeros((0, 256), np.float32)
            D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
            self._descs[path] = D
        if centers is not None:
            self._cents[path] = (np.stack([np.asarray(centers[i], np.float32)
                                           for i in order])
                                 if order else np.zeros((0, 3), np.float32))
        if K is not None:
            self._K[path] = np.asarray(K, np.float32)

    def get(self, path: str) -> Optional[List[Tuple[int, float]]]:
        return self._ids.get(path)

    def get_descs(self, path: str) -> Optional[np.ndarray]:
        return self._descs.get(path)

    def get_centers(self, path: str) -> Optional[np.ndarray]:
        return self._cents.get(path)

    def get_K(self, path: str) -> Optional[np.ndarray]:
        return self._K.get(path)

    def __len__(self):
        return len(self._ids)


def lookup_descs(ids: Sequence[Tuple[int, float]], records) -> np.ndarray:
    """(inst_id, obj) list -> (N,256) L2-normalized running-mean descriptors.
    Ids missing from the table (shouldn't happen mid-run) are skipped."""
    ds = []
    for iid, _ in ids:
        rec = records.get(iid)
        if rec is None:
            continue
        d = rec.descriptor.numpy().astype(np.float32)
        ds.append(d / (np.linalg.norm(d) + 1e-12))
    return np.stack(ds) if ds else np.zeros((0, 256), np.float32)


# --------------------------------------------------------------------------- verdict
@dataclass
class BiDescVerdict:
    status: str             # 'PASS' | 'FAIL' | 'UNKNOWN'
    n_q: int = 0
    n_r: int = 0
    a_qr: float = 0.0
    a_rq: float = 0.0
    a_bi: float = 0.0
    n_uniq: int = 0
    reason: str = ""
    # every mutual-NN pair: dict(q, r, sim, uniq) — uniq means sim > tau_pair
    pairs: list = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.status} a_qr={self.a_qr:.2f} a_rq={self.a_rq:.2f} "
                f"a_bi={self.a_bi:.2f} nuniq={self.n_uniq} nq/nr={self.n_q}/{self.n_r}"
                + (f" ({self.reason})" if self.reason else ""))

    def pairs_str(self) -> str:
        return " ".join(f"q{p['q']}~r{p['r']}:s{p['sim']:.2f}"
                        f":{'IN' if p['uniq'] else 'out'}" for p in self.pairs)


# --------------------------------------------------------------------------- stage 2
@dataclass
class LayoutVerdict:
    """Instance-layout-graph comparison (stage2_instance_graph_lc_design.md).
    ``status`` follows the doc's rule verbatim; ``e_scale``/``t_norm``/``s2`` are
    PASSIVE diagnostics (logged for offline analysis, never part of the decision):
    e_scale = median log distance ratio (common scale offset), t_norm = ||mean(X_q
    - R* X_r)|| (implied camera displacement), s2 = 2nd singular value of the Wahba
    B matrix (rotation observability; ~0 = collinear/degenerate)."""
    status: str              # 'PASS' | 'LARGE_VIEW_CHANGE' | 'T_FAR' | 'M_LOW' | 'FAIL' | 'UNKNOWN'
    m: int = 0               # matched instances used (uniq pairs)
    e_d: float = 0.0
    theta_r: float = 0.0     # deg
    e_r: float = 0.0         # deg
    e_scale: float = 0.0
    t_norm: float = 0.0
    s2: float = 0.0
    d_ref: float = 0.0       # median history-side instance depth (for the future
                             # scale-free tnorm/d_ref ratio gate)
    reason: str = ""

    def summary(self) -> str:
        return (f"{self.status} Ed={self.e_d:.3f} thR={self.theta_r:.0f} "
                f"ER={self.e_r:.0f} M={self.m} | Escale={self.e_scale:+.3f} "
                f"tnorm={self.t_norm:.2f} s2={self.s2:.2f} dref={self.d_ref:.2f}"
                + (f" ({self.reason})" if self.reason else ""))


def verify_layout(P_q: np.ndarray, P_r: np.ndarray, cfg: BiDescConfig) -> LayoutVerdict:
    """Doc-faithful Stage 2: nodes = Stage-1-matched instances' CAMERA-frame 3D
    positions (both sides observed in their own frame -> drift-free); edges =
    (distance, direction) between every pair; compare corresponding edges."""
    m = len(P_q)
    d_ref = float(np.median(np.linalg.norm(P_r, axis=1))) if m else 0.0
    if m < cfg.s2_min_m:
        # 1-2 co-visible objects cannot support any verdict (M=1: zero edges;
        # M=2: a single edge is chance-consistent too easily) -> reject outright.
        return LayoutVerdict("M_LOW", m=m, d_ref=d_ref,
                             reason=f"M={m}<{cfg.s2_min_m}: insufficient evidence")

    ii, jj = np.triu_indices(m, k=1)
    dq_v = P_q[jj] - P_q[ii]                    # (E,3) edge vectors
    dr_v = P_r[jj] - P_r[ii]
    dq = np.linalg.norm(dq_v, axis=1)
    dr = np.linalg.norm(dr_v, axis=1)
    ok = (dq > 1e-6) & (dr > 1e-6)
    if not ok.any():
        return LayoutVerdict("UNKNOWN", m=m, reason="degenerate edges")
    dq, dr = dq[ok], dr[ok]
    uq = dq_v[ok] / dq[:, None]
    ur = dr_v[ok] / dr[:, None]

    # 5.1 distance consistency (doc: median |log ratio|)
    logr = np.log(dq / dr)
    e_d = float(np.median(np.abs(logr)))

    # 5.2 common rotation via Wahba/Kabsch on edge directions. With a single edge
    # the problem is rank-1 (SVD picks an arbitrary minimizer), so use the natural
    # single-pair reading: R = minimal rotation taking u_r to u_q, theta = their
    # angle, residual exactly 0.
    if len(uq) == 1:
        c = float(np.clip(uq[0] @ ur[0], -1, 1))
        theta_r = float(np.degrees(np.arccos(c)))
        ax = np.cross(ur[0], uq[0])
        n = np.linalg.norm(ax)
        if n > 1e-9:
            ax /= n
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            s = np.sqrt(max(0.0, 1 - c * c))
            R = np.eye(3) + s * K + (1 - c) * (K @ K)
        else:
            R = np.eye(3)
        e_r = 0.0
        S = np.array([1.0, 0.0, 0.0])
    else:
        B = uq.T @ ur                           # (3,3)
        U, S, Vt = np.linalg.svd(B)
        D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
        R = U @ D @ Vt                          # u_q ~= R u_r
        theta_r = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        # 5.3 rotation consistency residual
        cosr = np.clip(np.sum(uq * (ur @ R.T), axis=1), -1, 1)
        e_r = float(np.degrees(np.median(np.arccos(cosr))))

    v = LayoutVerdict("FAIL", m=m, e_d=e_d, theta_r=theta_r, e_r=e_r,
                      e_scale=float(np.median(logr)),
                      t_norm=float(np.linalg.norm((P_q - P_r @ R.T).mean(0))),
                      s2=float(S[1]), d_ref=d_ref)
    # Calibrated decision (see config comment): metrics vote only where observable.
    if m < cfg.s2_min_m_rot:
        # rotation/translation unobservable -> distance-only ruling
        v.status = "PASS" if e_d <= cfg.s2_tau_d else "FAIL"
        v.reason = "rot-abstain" if v.status == "PASS" else "Ed(rot-abstain)"
    elif cfg.s2_tau_t and v.t_norm > cfg.s2_tau_t:
        v.status = "T_FAR"        # cameras too far apart per instance evidence
    elif e_d <= cfg.s2_tau_d and e_r <= cfg.s2_tau_r_deg:
        v.status = "PASS" if theta_r <= cfg.s2_tau_theta_deg else "LARGE_VIEW_CHANGE"
    else:
        why = []
        if e_d > cfg.s2_tau_d:
            why.append("Ed")
        if e_r > cfg.s2_tau_r_deg:
            why.append("ER")
        v.reason = "+".join(why)
    return v


def verify(D_q: np.ndarray, D_r: np.ndarray, cfg: BiDescConfig) -> BiDescVerdict:
    n_q, n_r = len(D_q), len(D_r)
    if n_q < cfg.n_min or n_r < cfg.n_min:
        return BiDescVerdict("UNKNOWN", n_q=n_q, n_r=n_r,
                             reason=f"insufficient instances (n_min={cfg.n_min})")

    S = D_q @ D_r.T                                        # (n_q, n_r) cosine

    row_max = S.max(axis=1)                                # current -> history
    col_max = S.max(axis=0)                                # history -> current
    kq, kr = adaptive_k(n_q, cfg), adaptive_k(n_r, cfg)
    a_qr = float(np.sort(row_max)[::-1][:kq].mean())
    a_rq = float(np.sort(col_max)[::-1][:kr].mean())
    a_bi = float(2.0 * a_qr * a_rq / (a_qr + a_rq + cfg.eps))

    # mutual nearest neighbours (all logged; unique iff above tau_pair)
    row_arg = S.argmax(axis=1)
    col_arg = S.argmax(axis=0)
    pairs = []
    n_uniq = 0
    for i in range(n_q):
        j = int(row_arg[i])
        if int(col_arg[j]) == i:
            uniq = bool(S[i, j] > cfg.tau_pair)
            n_uniq += int(uniq)
            pairs.append(dict(q=i, r=j, sim=float(S[i, j]), uniq=uniq))

    v = BiDescVerdict("FAIL", n_q=n_q, n_r=n_r, a_qr=a_qr, a_rq=a_rq,
                      a_bi=a_bi, n_uniq=n_uniq, pairs=pairs)
    if (n_uniq >= cfg.n_uniq and a_qr > cfg.tau_dir
            and a_rq > cfg.tau_dir and a_bi > cfg.tau_bi):
        v.status = "PASS"
    else:
        why = []
        if n_uniq < cfg.n_uniq:
            why.append(f"nuniq<{cfg.n_uniq}")
        if a_qr <= cfg.tau_dir or a_rq <= cfg.tau_dir:
            why.append("dir")
        if a_bi <= cfg.tau_bi:
            why.append("bi")
        v.reason = "+".join(why)
    return v
