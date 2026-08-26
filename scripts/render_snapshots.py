"""Replay ma_slam_semantic reconstruction snapshots (--snapshot) into a video.

    PYTHONPATH=src python scripts/render_snapshots.py --run outputs/.../<run> [--fps 2]
    -> <run>/snapshots/frames/step_XXXX.png + <run>/snapshots/recon.mp4

Each frame = the map as the system believed it at that submap step: every
submap's points placed with THAT step's optimized poses (so loop-closure
healing shows as a snap), instance bounding boxes colored by id, the camera
trajectory, and an annotation bar (step / frames / instances / loops). The view
is fixed from the FINAL map's extent (drops the thinnest world axis -> top-down).
"""
import argparse
import glob
import json
import os
import subprocess

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def id_color(i):
    h = (i * 0.618033988749895) % 1.0
    import colorsys
    return colorsys.hsv_to_rgb(h, 0.85, 0.95)


def load_step(path):
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    return d, meta


def compose(d, cache, snapdir, max_pts):
    """Global points/colors + camera centres (ordered) + camera up-vectors for one step."""
    pts, cols, cams, cam_up = [], [], [], []
    for b in d["bases"]:
        key = int(b)
        if key not in cache:
            cache[key] = np.load(os.path.join(snapdir, f"submap_{key:06d}.npz"))
        s = cache[key]
        G = d[f"G_{key:06d}"]                              # (n,4,4)
        L = s["L"]
        M = np.einsum("nij,njk->nik", G, np.linalg.inv(L))  # per-frame local->global
        P, F = s["pts"], s["fidx"].astype(int)
        R = M[F, :3, :3]; t = M[F, :3, 3]
        pts.append(np.einsum("nij,nj->ni", R, P) + t)
        cols.append(s["cols"])
        cams.append(G[:, :3, 3])
        cam_up.append(-G[:, :3, 1])          # OpenCV cam: -y = up
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    C = np.concatenate(cols) if cols else np.zeros((0, 3), np.uint8)
    if len(P) > max_pts:
        sel = np.random.default_rng(0).choice(len(P), max_pts, replace=False)
        P, C = P[sel], C[sel]
    return P, C, (np.concatenate(cams) if cams else np.zeros((0, 3))), \
        (np.concatenate(cam_up) if cam_up else np.zeros((0, 3)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max_pts", type=int, default=600000)
    ap.add_argument("--no_inst", action="store_true", help="geometry + trajectory only")
    ap.add_argument("--boxes", action="store_true",
                    help="draw instance bounding boxes instead of colored voxels")
    ap.add_argument("--pt_size", type=float, default=0.6)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--min_count", type=int, default=3,
                    help="draw only instances seen in >= this many frames")
    ap.add_argument("--max_extent", type=float, default=1.5,
                    help="skip instances with any bbox side longer than this (m; "
                         "table tops / walls / floor dominate the top-down view otherwise)")
    a = ap.parse_args()
    snapdir = os.path.join(a.run, "snapshots")
    steps = sorted(glob.glob(os.path.join(snapdir, "step_*.npz")))
    if not steps:
        raise SystemExit(f"no snapshots in {snapdir}")
    frames_dir = os.path.join(snapdir, "frames"); os.makedirs(frames_dir, exist_ok=True)
    cache = {}

    # fixed view from the final map: drop the axis with the smallest extent
    d_last, _ = load_step(steps[-1])
    P_last, _, cams_last, ups = compose(d_last, cache, snapdir, a.max_pts)
    # top-down: world-up = mean camera up (-y of OpenCV cam) over the whole trajectory;
    # in-plane basis = the two principal axes of the point cloud within that plane.
    up = ups.mean(0); up /= np.linalg.norm(up) + 1e-9
    Pc = P_last - P_last.mean(0)
    Pp = Pc - np.outer(Pc @ up, up)
    _, _, Vt = np.linalg.svd(Pp[np.random.default_rng(0).choice(len(Pp), min(len(Pp), 50000), replace=False)], full_matrices=False)
    e1 = Vt[0] - (Vt[0] @ up) * up; e1 /= np.linalg.norm(e1) + 1e-9
    e2 = np.cross(up, e1)
    basis = np.stack([e1, e2])                                    # (2,3): xy = P @ basis.T
    proj = lambda X: X @ basis.T
    Q = proj(P_last)
    lo, hi = np.percentile(Q, 1, axis=0), np.percentile(Q, 99, axis=0)
    if len(cams_last):                       # keep the trajectory inside the view
        qc = proj(cams_last); lo, hi = np.minimum(lo, qc.min(0)), np.maximum(hi, qc.max(0))
    ext = hi - lo
    pad = 0.05 * ext.max()
    xlim = (lo[0] - pad, hi[0] + pad); ylim = (lo[1] - pad, hi[1] + pad)
    ax2 = [0, 1]
    W = xlim[1] - xlim[0]; H = ylim[1] - ylim[0]
    figw = 12.0; figh = max(5.0, figw * H / max(W, 1e-6) + 1.2)

    for k, sp in enumerate(steps):
        d, meta = load_step(sp)
        P3, C, cams3, _ = compose(d, cache, snapdir, a.max_pts)
        P = proj(P3); cams = proj(cams3)
        fig, ax = plt.subplots(figsize=(figw, figh))
        ax.set_facecolor("#111")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        if len(P):
            order = np.argsort(P3 @ up)        # low first so high points draw on top
            ax.scatter(P[order, ax2[0]], P[order, ax2[1]], c=C[order] / 255.0, s=a.pt_size,
                       linewidths=0, alpha=0.9, rasterized=True, zorder=1)
        if "inst_id" in d and not a.no_inst:
            vs = meta["voxel_size"]
            if not a.boxes and len(d["vox"]):
                keep_ids = set(int(i) for i, c, bmin, bmax in
                               zip(d["inst_id"], d["inst_count"], d["inst_bbox_min"], d["inst_bbox_max"])
                               if int(c) >= a.min_count and float((bmax - bmin).max()) <= a.max_extent)
                sel = np.array([int(i) in keep_ids for i in d["vox_id"]], bool)
                V = proj((d["vox"][sel].astype(np.float32) + 0.5) * vs)
                cols = np.array([id_color(int(i)) for i in d["vox_id"][sel]]) if sel.any() else np.zeros((0, 3))
                if len(V):
                    # marker area so one voxel tiles ~one cell at this zoom (pt^2):
                    # axes width in points (after layout) / view width in metres
                    fig.canvas.draw()
                    ax_w_pt = ax.get_window_extent().width / fig.dpi * 72.0
                    pt_per_m = ax_w_pt / (xlim[1] - xlim[0])
                    s_vox = max(3.0, (pt_per_m * vs) ** 2 * 0.8)
                    ax.scatter(V[:, 0], V[:, 1], c=cols, s=s_vox, linewidths=0,
                               alpha=0.45, marker="s", zorder=3)
            elif a.boxes:
                for iid, bmin, bmax, cnt in zip(d["inst_id"], d["inst_bbox_min"],
                                                d["inst_bbox_max"], d["inst_count"]):
                    if int(cnt) < a.min_count or float((bmax - bmin).max()) > a.max_extent:
                        continue
                    q = proj(np.stack([bmin, bmax])); x0, y0 = q.min(0); x1, y1 = q.max(0)
                    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                               ec=id_color(int(iid)), lw=1.0, alpha=0.9))
        if len(cams) > 1:
            segs = np.stack([cams[:-1][:, ax2], cams[1:][:, ax2]], axis=1)
            ax.add_collection(LineCollection(segs, colors="#ffdd55", lw=1.5, alpha=0.95, zorder=5))
            ax.plot(cams[-1, ax2[0]], cams[-1, ax2[1]], "o", ms=8, color="#ff3333", mec="white", zorder=6)
        if len(d["loops"]):
            allc = {}
            for b in d["bases"]:
                G = d[f"G_{int(b):06d}"]
                for i in range(len(G)):
                    allc[int(b) + i] = proj(G[i, :3, 3][None])[0]
            for kn, ko in d["loops"]:
                if int(kn) in allc and int(ko) in allc:
                    p, q = allc[int(kn)], allc[int(ko)]
                    ax.plot([p[ax2[0]], q[ax2[0]]], [p[ax2[1]], q[ax2[1]]], "-", color="#33ddff", lw=2.2, zorder=6)
        n_inst = int(len(d["inst_id"])) if "inst_id" in d else 0
        ax.set_title(f"step {meta['step']:3d}   frames {meta['n_frames']:4d}   submaps {len(d['bases']):3d}   "
                     f"instances {n_inst:3d}   loops {len(d['loops'])}   t={meta['elapsed_s']:.0f}s",
                     fontsize=11, family="monospace")
        fig.savefig(os.path.join(frames_dir, f"step_{k:04d}.png"), dpi=a.dpi, facecolor="#111")
        plt.close(fig)
        print(f"\r  frame {k + 1}/{len(steps)}", end="", flush=True)
    print()
    mp4 = os.path.join(snapdir, "recon.mp4")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
           "-i", os.path.join(frames_dir, "step_%04d.png"),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", mp4]
    subprocess.run(cmd, check=True)
    print(f"[render_snapshots] {len(steps)} steps -> {mp4}")


if __name__ == "__main__":
    main()
