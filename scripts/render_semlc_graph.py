"""Render sem-LC scene-graph visualizations from --sem_lc_dump JSONs.

For every candidate the run dumped (<out>/semlc_dump/NNNN_<q>__<r>.json) this
draws the two frames side by side with the instance-layout graph on top:

  - gray hollow circle  = instance the frame observed but Stage 1 did NOT match
                          (makes nq/nr vs M visible — the M_LOW story)
  - colored filled dot  = Stage-1 unique match (same color = same object on both
                          sides; cosine sim printed next to the query-side node)
  - dashed hollow dot   = sub-threshold MNN pair (the "same-type aliasing" band)
  - graph edges         = between matched nodes; green if THAT edge is length- and
                          rotation-consistent (|log dq/dr| <= tau_d and
                          angle(u_q, R* u_r) <= tau_R), red otherwise. R* is
                          recomputed from the dumped camera-frame centers with the
                          same Wahba as bidesc_lc.verify_layout.

Nodes are the CAMERA-frAME 3D instance centers projected through the per-frame K
(both stored at semantic resolution, so the image is re-cropped/resized exactly
like the semantic bridge does). A 3D centroid projects to the instance's mass
center, not its visual mask center — partially-visible objects can look offset;
that is expected, not a bug.

Run:  PYTHONPATH=src python scripts/render_semlc_graph.py --run outputs/.../<run>
      (or --dump <dir> directly)  -> <dump>/render/*.png + index.html
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
from PIL import Image

try:
    import PIL.Image
    _LANCZOS = PIL.Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    _LANCZOS = PIL.Image.LANCZOS

# 12 well-separated colors, cycled per matched pair
_COLORS = plt.get_cmap("tab10").colors + ((0.8, 0.5, 0.0), (0.0, 0.6, 0.6))


# ------------------------------------------------------------------ geometry
def center_crop_to_aspect(h, w, tw, th):
    """Mirror of ma_slam_semantic.bridge._center_crop_to_aspect."""
    ta = tw / th
    if w / h > ta:
        cw = int(round(h * ta))
        return 0, (w - cw) // 2, h, cw
    if w / h < ta:
        ch = int(round(w / ta))
        return (h - ch) // 2, 0, ch, w
    return 0, 0, h, w


def load_image(path, res):
    """RGB, center-cropped + resized exactly like the semantic bridge (so the
    dumped K projects onto THIS pixel grid)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    tw, th = res
    t, l, ch, cw = center_crop_to_aspect(h, w, tw, th)
    img = img.crop((l, t, l + cw, t + ch)).resize((tw, th), resample=_LANCZOS)
    return np.asarray(img)


def project(K, X):
    """(3,3),(N,3) camera-frame -> (N,2) pixels; z<=eps -> NaN row."""
    X = np.asarray(X, np.float64)
    z = X[:, 2]
    uv = np.full((len(X), 2), np.nan)
    ok = z > 0.05
    uv[ok, 0] = K[0, 0] * X[ok, 0] / z[ok] + K[0, 2]
    uv[ok, 1] = K[1, 1] * X[ok, 1] / z[ok] + K[1, 2]
    return uv


def wahba(uq, ur):
    """R* minimizing sum ||u_q - R u_r||^2 (mirror of verify_layout, incl. the
    single-edge minimal-rotation special case)."""
    if len(uq) == 1:
        c = float(np.clip(uq[0] @ ur[0], -1, 1))
        ax = np.cross(ur[0], uq[0])
        n = np.linalg.norm(ax)
        if n < 1e-9:
            return np.eye(3)
        ax /= n
        Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        s = np.sqrt(max(0.0, 1 - c * c))
        return np.eye(3) + s * Kx + (1 - c) * (Kx @ Kx)
    B = uq.T @ ur
    U, _, Vt = np.linalg.svd(B)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    return U @ D @ Vt


def edge_verdicts(Xq, Xr, tau_d, tau_r_deg):
    """Per-edge consistency over matched centers: [(i, j, ok, dlog, ang)]."""
    m = len(Xq)
    if m < 2:
        return []
    ii, jj = np.triu_indices(m, k=1)
    dq_v, dr_v = Xq[jj] - Xq[ii], Xr[jj] - Xr[ii]
    dq, dr = np.linalg.norm(dq_v, axis=1), np.linalg.norm(dr_v, axis=1)
    ok = (dq > 1e-6) & (dr > 1e-6)
    uq, ur = dq_v[ok] / dq[ok, None], dr_v[ok] / dr[ok, None]
    if not len(uq):
        return []
    R = wahba(uq, ur)
    out = []
    k = 0
    for e in range(len(ii)):
        if not ok[e]:
            continue
        dlog = abs(float(np.log(dq[e] / dr[e])))
        ang = float(np.degrees(np.arccos(np.clip(uq[k] @ (R @ ur[k]), -1, 1))))
        out.append((int(ii[e]), int(jj[e]), dlog <= tau_d and ang <= tau_r_deg,
                    dlog, ang))
        k += 1
    return out


# ------------------------------------------------------------------ decision mirror
def final_gate(s1, s2):
    """Mirror of solver._instance_lc_pass (veto mode)."""
    if s2 and s2["status"] in ("LARGE_VIEW_CHANGE", "T_FAR", "M_LOW"):
        return f"S2GATE REJECT ({s2['status']})", "reject"
    if s1["status"] == "FAIL" or (s2 and s2["status"] == "FAIL"):
        return "FAIL -> coloc bar 0.45 (veto)", "veto"
    if s1["status"] == "PASS" and (not s2 or s2["status"] == "PASS"):
        return "PASS -> coloc bar 0.70", "pass"
    return f"{s1['status']} -> coloc bar 0.70", "unknown"


# ------------------------------------------------------------------ drawing
def draw_side(ax, img, rec_nodes, K, matched, sub_mnn, edges, side):
    H, W = img.shape[:2]
    ax.imshow(img)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    xyz = np.array([n["xyz"] if n["xyz"] else [0, 0, -1] for n in rec_nodes], float)
    uv = project(K, xyz) if len(xyz) else np.zeros((0, 2))
    matched_rows = {row: ci for ci, row in enumerate(matched)}
    sub_rows = {row: s for row, s in sub_mnn}
    off = 0

    # graph edges first (under the nodes); matched[i] -> row index
    n_e = len(edges)
    lw = 2.2 if n_e <= 6 else 1.2
    alpha = 0.85 if n_e <= 6 else 0.55
    for (i, j, good, dlog, ang) in edges:
        ra, rb = matched[i], matched[j]
        a, b = uv[ra], uv[rb]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        ax.plot([a[0], b[0]], [a[1], b[1]], color=("#22bb44" if good else "#ee3333"),
                lw=lw, alpha=alpha, zorder=2)

    for row, n in enumerate(rec_nodes):
        u, v = uv[row] if row < len(uv) else (np.nan, np.nan)
        if np.isnan(u) or u < -20 or u > W + 20 or v < -20 or v > H + 20:
            off += 1
            continue
        if row in matched_rows:
            c = _COLORS[matched_rows[row] % len(_COLORS)]
            ax.scatter([u], [v], s=180, color=c, edgecolor="white", lw=1.6, zorder=4)
            ax.annotate(str(matched_rows[row]), (u, v), color="white", fontsize=8,
                        ha="center", va="center", zorder=5, fontweight="bold")
        elif row in sub_rows:
            ci, sim = sub_rows[row]
            c = _COLORS[ci % len(_COLORS)]
            ax.scatter([u], [v], s=120, facecolor="none", edgecolor=c, lw=1.8,
                       linestyle="--", zorder=3)
            ax.annotate(f"{sim:.2f}", (u + 8, v - 8), color=c, fontsize=7,
                        zorder=5, path_effects=[patheffects.withStroke(
                            linewidth=2, foreground="black")])
        else:
            ax.scatter([u], [v], s=70, facecolor="none", edgecolor="#bbbbbb",
                       lw=1.5, zorder=3)
    if off:
        ax.text(6, H - 8, f"off-view: {off}", color="#ffdd55", fontsize=8,
                bbox=dict(fc="black", alpha=0.5, pad=1.5))
    ax.set_title(side, fontsize=10)


def load_salad_sims(dump_dir):
    """(q_basename, r_basename) -> SALAD sim, parsed from the run's loops.txt
    (the dump JSON predates the solver's BIDESC log line, which carries sim)."""
    lt = os.path.join(os.path.dirname(os.path.abspath(dump_dir)), "loops.txt")
    sims = {}
    if os.path.isfile(lt):
        for line in open(lt):
            if line.startswith("BIDESC A"):
                m = re.search(r"sim=([\d.]+) .*qimg=(\S+) mimg=(\S+)", line)
                if m:
                    sims[(os.path.basename(m.group(2)),
                          os.path.basename(m.group(3)))] = float(m.group(1))
    return sims


def render_one(rec, out_png, res, salad=None):
    s1, s2 = rec["s1"], rec["s2"]
    K_q = np.array(rec["K_q"]) if rec["K_q"] else None
    K_r = np.array(rec["K_r"]) if rec["K_r"] else None
    if K_q is None or K_r is None:
        return False

    uniq = [p for p in rec["pairs"] if p["uniq"]]
    subp = [p for p in rec["pairs"] if not p["uniq"]]
    mq = [p["q_row"] for p in uniq]           # pair index -> node row, per side
    mr = [p["r_row"] for p in uniq]
    # sub-threshold MNN: same color index continues after the uniq ones
    sub_q = [(p["q_row"], (len(uniq) + k, p["sim"])) for k, p in enumerate(subp)]
    sub_r = [(p["r_row"], (len(uniq) + k, p["sim"])) for k, p in enumerate(subp)]

    Xq = np.array([rec["q_nodes"][r]["xyz"] for r in mq], float) if mq else np.zeros((0, 3))
    Xr = np.array([rec["r_nodes"][r]["xyz"] for r in mr], float) if mr else np.zeros((0, 3))
    cfg = rec.get("cfg", {})
    edges = edge_verdicts(Xq, Xr, cfg.get("s2_tau_d", 0.3), cfg.get("s2_tau_r_deg", 25.0))

    img_q = load_image(rec["q_path"], res)
    img_r = load_image(rec["r_path"], res)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    qn = os.path.basename(rec["q_path"])
    rn = os.path.basename(rec["r_path"])
    draw_side(axes[0], img_q, rec["q_nodes"], K_q, mq, sub_q, edges,
              f"current  {qn}")
    draw_side(axes[1], img_r, rec["r_nodes"], K_r, mr, sub_r, edges,
              f"history  {rn}")

    gate, _ = final_gate(s1, s2)
    if salad is not None:
        gate = f"SALAD {salad:.2f}  |  {gate}"
    sims = "  ".join(f"{k}:{p['sim']:.2f}" for k, p in enumerate(uniq))
    l1 = (f"S1 {s1['status']}  a_qr={s1['a_qr']:.2f} a_rq={s1['a_rq']:.2f} "
          f"a_bi={s1['a_bi']:.2f}  nuniq={s1['n_uniq']}  nq/nr={s1['n_q']}/{s1['n_r']}"
          + (f"  [{s1['reason']}]" if s1.get("reason") else ""))
    if s2:
        l2 = (f"S2 {s2['status']}  M={s2['m']}  Ed={s2['e_d']:.3f}  "
              f"thR={s2['theta_r']:.0f}deg  ER={s2['e_r']:.0f}deg  "
              f"|t*|={s2['t_norm']:.2f}m (dref={s2['d_ref']:.1f})"
              + (f"  [{s2['reason']}]" if s2.get("reason") else ""))
    else:
        l2 = "S2 -- (no unique pairs)"
    fig.suptitle(f"{gate}", fontsize=12,
                 color={"reject": "#cc2222", "veto": "#bb6600",
                        "pass": "#117722"}.get(final_gate(s1, s2)[1], "#444444"))
    legend = ("filled+num = uniq match (counted)   dashed+sim = MNN below tau 0.85 "
              "(NOT counted; same color = tentative pair)   gray = unmatched")
    fig.text(0.5, 0.045, l1 + "\n" + l2
             + ("\n" + "match sims:  " + sims if sims else "") + "\n" + legend,
             ha="center", va="bottom", fontsize=8, family="monospace")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return True


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", help="run output dir (uses <run>/semlc_dump)")
    ap.add_argument("--dump", help="dump dir directly")
    ap.add_argument("--out", help="output dir (default <dump>/render)")
    ap.add_argument("--status", default="",
                    help="comma list of S2 statuses to keep (default all); "
                         "e.g. PASS,T_FAR,M_LOW,LARGE_VIEW_CHANGE,FAIL")
    ap.add_argument("--min_m", type=int, default=0, help="min S2 M to render")
    ap.add_argument("--limit", type=int, default=0, help="max candidates (0=all)")
    a = ap.parse_args()

    dump = a.dump or (os.path.join(a.run, "semlc_dump") if a.run else None)
    if not dump or not os.path.isdir(dump):
        raise SystemExit(f"no dump dir: {dump}")
    out = a.out or os.path.join(dump, "render")
    os.makedirs(out, exist_ok=True)
    keep = set(s.strip() for s in a.status.split(",") if s.strip())

    files = sorted(glob.glob(os.path.join(dump, "*.json")))
    salad_sims = load_salad_sims(dump)
    rows, n_done = [], 0
    for f in files:
        rec = json.load(open(f))
        s2 = rec["s2"]
        if keep and (not s2 or s2["status"] not in keep):
            continue
        if a.min_m and (not s2 or s2["m"] < a.min_m):
            continue
        png = os.path.join(out, os.path.basename(f).replace(".json", ".png"))
        try:
            ok = render_one(rec, png, tuple(rec.get("cfg", {}).get("resolution",
                                                                   (518, 336))),
                            salad=salad_sims.get((os.path.basename(rec["q_path"]),
                                                  os.path.basename(rec["r_path"]))))
        except FileNotFoundError as e:
            print(f"  skip {os.path.basename(f)}: {e}")
            continue
        if not ok:
            continue
        gate, cls = final_gate(rec["s1"], rec["s2"])
        rows.append((os.path.basename(png), rec, gate, cls))
        n_done += 1
        if a.limit and n_done >= a.limit:
            break

    # gallery
    order = {"pass": 0, "reject": 1, "veto": 2, "unknown": 3}
    rows.sort(key=lambda r: (order.get(r[3], 9),
                             -(r[1]["s2"]["m"] if r[1]["s2"] else 0)))
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write("<html><body style='font-family:monospace;background:#111;color:#ddd'>\n")
        f.write(f"<h2>sem-LC scene graphs — {dump} ({len(rows)})</h2>\n")
        for png, rec, gate, cls in rows:
            s1, s2 = rec["s1"], rec["s2"]
            col = {"reject": "#f66", "veto": "#fa4", "pass": "#6d6"}.get(cls, "#aaa")
            s2s = (f"S2 {s2['status']} M={s2['m']} Ed={s2['e_d']:.2f} "
                   f"thR={s2['theta_r']:.0f} tnorm={s2['t_norm']:.2f}") if s2 else "S2 --"
            f.write(f"<div style='margin:18px 0'><b style='color:{col}'>{gate}</b> "
                    f"| S1 {s1['status']} a_bi={s1['a_bi']:.2f} | {s2s} "
                    f"| {os.path.basename(rec['q_path'])} vs "
                    f"{os.path.basename(rec['r_path'])}<br>"
                    f"<img src='{png}' style='max-width:1100px'></div>\n")
        f.write("</body></html>\n")
    print(f"rendered {len(rows)} -> {out}/index.html")


if __name__ == "__main__":
    main()
