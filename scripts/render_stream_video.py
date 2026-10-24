"""Per-frame streaming-reconstruction videos from ma_slam_semantic --snapshot records.

    PYTHONPATH=src python scripts/render_stream_video.py --run outputs/.../<run> --images data/<scene>/rgb
    -> <run>/snapshots/video/{bev,chase}_{rgb,inst,mix}.mp4

Unlike render_snapshots.py (one image per submap step), every snapshot point carries the
frame it came from (submap npz `fidx`), so the map is revealed ONE KEYFRAME AT A TIME:
keyframe i of the submap added at step s is drawn with step s's optimized poses, together
with every earlier frame. A step that only adds loop edges (no new submap) is played as a
short pose interpolation (old -> new poses) so the loop-closure correction is visible.

Colour modes: rgb  = predicted geometry with image colours
              inst = instance colours, unassigned grey. --color_source final (default): each point keeps the
                     colour of the FINAL exported instance it belongs to (after final merge); online: the
                     instance table at that step (pre-merge, fragments visible)
              mix  = instance colours blended over rgb
              --nostruct_dir: extra inst/mix where wall/floor/ceiling instances keep the image colour
--ceiling auto drops a detected ceiling plane (with a margin = its own surface thickness) in all views.
Views:        bev   = fixed top-down orthographic view of the final map extent
              chase = perspective camera trailing the current keyframe camera (smoothed)
Both views share one timeline (--sub video frames per keyframe), so they play in sync.
"""
import argparse
import colorsys
import glob
import json
import os
import subprocess

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp


def id_color(i):
    return np.array(colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.85, 0.95)) * 255


def load_run(snapdir):
    subs = {}
    for f in sorted(glob.glob(os.path.join(snapdir, "submap_*.npz"))):
        s = np.load(f)
        subs[int(os.path.basename(f)[7:13])] = dict(
            pts=s["pts"].astype(np.float32), cols=s["cols"], fidx=s["fidx"].astype(int),
            L=s["L"].astype(np.float64), fids=s["frame_ids"].astype(int))
    steps = []
    for f in sorted(glob.glob(os.path.join(snapdir, "step_*.npz"))):
        d = np.load(f)
        st = dict(bases=[int(b) for b in d["bases"]], loops=d["loops"].tolist(),
                  meta=json.loads(str(d["meta"])),
                  G={int(b): d[f"G_{int(b):06d}"].astype(np.float64) for b in d["bases"]})
        if "vox" in d.files:
            cnt = dict(zip(d["inst_id"].tolist(), d["inst_count"].tolist()))
            st.update(vox=d["vox"].astype(np.int64), vox_id=d["vox_id"].astype(int), inst_count=cnt)
        steps.append(st)
    return subs, steps


def world_points(sub, G):
    M = np.einsum("nij,njk->nik", G, np.linalg.inv(sub["L"]))[sub["fidx"]]
    return np.einsum("nij,nj->ni", M[:, :3, :3], sub["pts"]) + M[:, :3, 3]


def vox_key(v):
    v = v + 40000
    return (v[:, 0] * 80000 + v[:, 1]) * 80000 + v[:, 2]


def label_points(P, st, vs, min_count):
    """Instance id per point (-1 = none): the smallest instance owning the point's voxel."""
    if "vox" not in st or not len(st["vox"]):
        return np.full(len(P), -1)
    ids = st["vox_id"]; cnt = np.array([st["inst_count"].get(int(i), 0) for i in ids])
    ok = cnt >= min_count
    keys, ids = vox_key(st["vox"][ok]), ids[ok]
    out = np.full(len(P), -1)
    if not len(keys):
        return out
    u, inv, n = np.unique(ids, return_inverse=True, return_counts=True)
    o = np.lexsort((n[inv], keys)); keys, ids = keys[o], ids[o]
    keys, first = np.unique(keys, return_index=True); ids = ids[first]
    q = vox_key(np.floor(P / vs).astype(np.int64))
    j = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
    hit = keys[j] == q
    out[hit] = ids[j[hit]]
    return out


def mode_colors(rgb, lab, struct=None):
    """rgb / inst / mix; with `struct` (bool per point) also inst_ns / mix_ns, where wall/floor/ceiling
    instances fall back to the image colour so only objects are coloured."""
    rgb = rgb.astype(np.float32)
    has = lab >= 0
    ic = np.full((len(lab), 3), 70.0, np.float32)
    if has.any():
        u, inv = np.unique(lab[has], return_inverse=True)
        ic[has] = np.stack([id_color(int(i)) for i in u])[inv]
    mix = rgb * 0.85
    mix[has] = 0.4 * rgb[has] + 0.6 * ic[has]
    out = dict(rgb=rgb, inst=ic, mix=mix)
    if struct is not None:
        ins = ic.copy(); ins[struct] = rgb[struct] * 0.8
        mxs = mix.copy(); mxs[struct] = rgb[struct] * 0.85
        out.update(inst_ns=ins, mix_ns=mxs)
    return out


def final_labels(P_slam, run, T_align):
    """Final exported instance id per point (-1 = unassigned): the point's position under the FINAL
    poses, mapped into the export frame, takes the label of the export voxel it falls in."""
    from scipy.spatial import cKDTree
    f = glob.glob(os.path.join(run, "*_instance_labels.npz"))[0]
    z = np.load(f); C = z["centers"]; L = z["cluster_id"]
    tj = glob.glob(os.path.join(run, "*_query_instance_table.json"))
    vsz = float(json.load(open(tj[0]))["voxel_size"]) if tj else 0.05
    X = P_slam @ T_align[:3, :3].T + T_align[:3, 3]
    d, i = cKDTree(C).query(X, distance_upper_bound=0.9 * vsz)
    out = np.full(len(X), -1); ok = np.isfinite(d)
    out[ok] = L[i[ok]]
    return out


def fit_hplane(P, sel, up, rng, max_tilt=10.0, min_frac=0.003):
    """Dominant plane among P[sel] whose normal is within max_tilt deg of `up` (RANSAC + SVD refine).
    Returns (n oriented with up, d, residual std, n_inliers) or None."""
    cand = P[sel]
    if len(cand) < min_frac * len(P):
        return None
    sub = cand[rng.choice(len(cand), min(len(cand), 60000), replace=False)]
    best, bn, bd = 0, None, None
    for _ in range(600):
        p = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0]); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if abs(n @ up) < np.cos(np.radians(max_tilt)):
            continue
        n = n * np.sign(n @ up); d0 = n @ p[0]
        c = int((np.abs(sub @ n - d0) < 0.02).sum())
        if c > best:
            best, bn, bd = c, n, d0
    if bn is None:
        return None
    inl = cand[np.abs(cand @ bn - bd) < 0.04]
    if len(inl) < min_frac * len(P):
        return None
    mu = inl.mean(0); _, _, Vt = np.linalg.svd(inl - mu, full_matrices=False)
    n = Vt[2] * np.sign(Vt[2] @ up); d0 = n @ mu
    return n, d0, float((inl @ n - d0).std()), len(inl)


def gravity_and_ceiling(P, cams, up0, rng):
    """Gravity 'up' from the floor plane (fallback: ceiling, then mean camera up) and the ceiling plane.
    Mean camera up is biased by the cameras' pitch (they look down), which tilts the BEV by 5-10 deg."""
    up, src = up0, "camera"
    h = P @ up0; ch = cams @ up0
    fl = fit_hplane(P, (h < ch.min() - 0.3) & (h < np.percentile(h, 0.5) + 0.5), up0, rng, max_tilt=30.0)
    if fl is not None:
        up, src = fl[0], "floor"
    h = P @ up; ch = cams @ up
    ce = fit_hplane(P, (h > ch.max() + 0.05) & (h > np.percentile(h, 99.5) - 0.5), up, rng, max_tilt=10.0)
    if ce is not None:
        n, d0, sd, ninl = ce
        above = int(((P @ n - d0) > max(0.03, 3 * sd) + 0.05).sum())
        # a ceiling is the top of the map: reject planes with real structure above them (shelf / cabinet tops)
        if d0 - (cams @ n).max() < 0.15 or above > 0.1 * ninl:
            ce = None
    if ce is not None and fl is None:
        up, src = ce[0], "ceiling"
    return up, src, ce


def splat(uv, depth, W, H, r):
    """Winner point index per pixel (nearest depth wins), splatted to r x r pixels."""
    us, vs_, ds, src = [], [], [], []
    n = np.arange(len(uv))
    for dx in range(r):
        for dy in range(r):
            us.append(uv[:, 0] + dx); vs_.append(uv[:, 1] + dy); ds.append(depth); src.append(n)
    u = np.concatenate(us); v = np.concatenate(vs_); d = np.concatenate(ds); s = np.concatenate(src)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pix = v[ok] * W + u[ok]; d = d[ok]; s = s[ok]
    o = np.lexsort((d, pix)); pix, s = pix[o], s[o]
    pix, first = np.unique(pix, return_index=True)
    return pix, s[first]


def look_at(eye, target, up):
    z = target - eye; z /= np.linalg.norm(z) + 1e-9
    x = np.cross(-up, z); x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    T = np.eye(4); T[:3, :3] = np.stack([x, y, z], 1); T[:3, 3] = eye
    return T


def interp_poses(Ga, Gb, t):
    out = Ga.copy()
    for i in range(len(Ga)):
        sl = Slerp([0, 1], Rot.from_matrix(np.stack([Ga[i, :3, :3], Gb[i, :3, :3]])))
        out[i, :3, :3] = sl(t).as_matrix(); out[i, :3, 3] = (1 - t) * Ga[i, :3, 3] + t * Gb[i, :3, 3]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--outdir", default="", help="default <run>/snapshots/video")
    ap.add_argument("--images", default="", help="rgb dir for the input-frame inset (frame_XXXXX.*)")
    ap.add_argument("--views", nargs="+", default=["bev", "chase"])
    ap.add_argument("--modes", nargs="+", default=["rgb", "inst", "mix"])
    ap.add_argument("--W", type=int, default=1280); ap.add_argument("--H", type=int, default=720)
    ap.add_argument("--fps", type=int, default=15); ap.add_argument("--sub", type=int, default=3,
                    help="video frames per keyframe")
    ap.add_argument("--pt", type=int, default=2, help="splat size (px)")
    ap.add_argument("--fov", type=float, default=75.0)
    ap.add_argument("--back", type=float, default=1.2); ap.add_argument("--height", type=float, default=0.7)
    ap.add_argument("--ahead", type=float, default=1.0); ap.add_argument("--smooth", type=float, default=0.35)
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--loop_frames", type=int, default=30, help="video frames for a loop-closure correction")
    ap.add_argument("--loop_min_disp", type=float, default=0.03,
                    help="animate a loop correction only if some camera moves more than this (m)")
    ap.add_argument("--outro", type=int, default=45, help="chase: frames pulling up to an overview at the end")
    ap.add_argument("--color_source", default="final", choices=["final", "online"],
                    help="final = every point coloured by the FINAL exported instance it ends up in (stable "
                         "colours, after final merge); online = the instance table at that step (pre-merge)")
    ap.add_argument("--ceiling", default="auto", choices=["auto", "off"],
                    help="auto = detect a ceiling plane above the cameras and drop it (all views)")
    ap.add_argument("--nostruct_dir", default="",
                    help="also write inst/mix with wall/floor/ceiling instances in image colour to this dir "
                         "(needs <run>/*_instance_classes.json from scripts/classify_instances.py)")
    a = ap.parse_args()

    snapdir = os.path.join(a.run, "snapshots"); outdir = a.outdir or os.path.join(snapdir, "video"); os.makedirs(outdir, exist_ok=True)
    subs, steps = load_run(snapdir)
    vs = steps[-1]["meta"]["voxel_size"]
    imgs = {}
    if a.images:
        for f in glob.glob(os.path.join(a.images, "frame_*")):
            imgs[int(os.path.basename(f)[6:11])] = f

    # ---------------- timeline: (step, poses{base:G}, revealed{base: max fidx}, current (base,i), event)
    TL = []; seen = set(); prev = None
    for si, st in enumerate(steps):
        new = [b for b in st["bases"] if b not in seen]
        # loop closure: animate the already-built map from the previous step's poses to this step's
        if prev is not None and len(st["loops"]) > len(prev["loops"]) and TL:
            old = [b for b in prev["bases"] if b in st["G"]]
            disp = max(np.linalg.norm(st["G"][b][:, :3, 3] - prev["G"][b][:, :3, 3], axis=1).max() for b in old)
            added = st["loops"][len(prev["loops"]):]
            txt = "  ".join(f"{p_}<->{q_}" for p_, q_ in added[:3]) + (" ..." if len(added) > 3 else "")
            ev = f"loop closure ({len(added)} new: {txt}) accepted -> pose graph re-optimized"
            nf = a.loop_frames if disp > a.loop_min_disp else 0
            rev = {bb: 10 ** 6 for bb in old}; cur = TL[-1]["cur"]
            for k in range(nf):
                t = min(1.0, max(0.0, (k - nf * 0.3) / (nf * 0.5)))
                t = 0.5 - 0.5 * np.cos(np.pi * t)
                G = {b: interp_poses(prev["G"][b], st["G"][b], t) for b in old}
                TL.append(dict(step=si if t >= 0.5 else si - 1, G=G, rev=rev, cur=cur, loops=st["loops"],
                               ev=ev, sub=1, loop=True, interp=True))
        if new:
            for b in new:
                n = len(st["G"][b])
                for i in range(1 if b > 0 else 0, n):
                    rev = {bb: 10 ** 6 for bb in st["bases"] if bb in seen}; rev[b] = i
                    TL.append(dict(step=si, G=st["G"], rev=rev, cur=(b, i), loops=st["loops"], ev=""))
                seen.add(b)
        prev = st
    print(f"[video] {len(TL)} timeline entries, {len(subs)} submaps, {sum(len(s['pts']) for s in subs.values()):,} points")

    fin = steps[-1]; Gall = np.concatenate([fin["G"][b] for b in fin["bases"]])
    up = (-Gall[:, :3, 1]).mean(0); up /= np.linalg.norm(up)
    Pfin = {b: world_points(subs[b], fin["G"][b]) for b in fin["bases"]}

    # ---------------- gravity (floor plane) + ceiling: per-point keep mask from the FINAL map (fixed over time)
    keep = {b: np.ones(len(subs[b]["pts"]), bool) for b in subs}
    Pall = np.concatenate([Pfin[b] for b in fin["bases"]])
    up0 = up
    up, src, ce = gravity_and_ceiling(Pall, Gall[:, :3, 3], up0, np.random.default_rng(0))
    print(f"[video] up from {src} (tilt vs mean camera up {np.degrees(np.arccos(min(1, up @ up0))):.1f} deg)")
    if a.ceiling == "auto" and ce is not None:
        n, d0, sd, ninl = ce
        delta = float(np.clip(3 * sd, 0.03, 0.10))
        for b in fin["bases"]:
            keep[b] = (Pfin[b] @ n - d0) < -delta
        nrm = sum((~keep[b]).sum() for b in fin["bases"])
        print(f"[video] ceiling: {d0 - (Gall[:, :3, 3] @ n).max():.2f} m above the highest camera, cut margin "
              f"{delta * 100:.1f} cm (3x surface std), inliers {ninl:,}, removed {nrm:,} pts ({100 * nrm / len(Pall):.1f}%)")
    else:
        print("[video] ceiling: " + ("none detected" if a.ceiling == "auto" else "off"))

    # ---------------- final-instance colours (fixed per point) or per-step online colours
    struct_ids = set()
    if a.nostruct_dir:
        cj = glob.glob(os.path.join(a.run, "*_instance_classes.json"))[0]
        struct_ids = {int(k) for k, v in json.load(open(cj)).items() if v in ("wall", "floor", "ceiling")}
    modes_out = list(a.modes) + (["inst_ns", "mix_ns"] if a.nostruct_dir else [])
    fixed = {}
    if a.color_source == "final":
        T_al = np.array(json.load(open(os.path.join(a.run, "alignment.json")))["T_align"])
        for b in fin["bases"]:
            lab = final_labels(Pfin[b], a.run, T_al)
            st_ = np.isin(lab, list(struct_ids)) if a.nostruct_dir else None
            fixed[b] = ({k: v.astype(np.uint8) for k, v in mode_colors(subs[b]["cols"], lab, st_).items()},
                        lab, st_ if st_ is not None else np.zeros(len(lab), bool))
        allab = np.concatenate([fixed[b][1][keep[b]] for b in fin["bases"]])
        print(f"[video] final-instance colours: {len(np.unique(allab[allab >= 0]))} instances, "
              f"{100 * (allab >= 0).mean():.0f}% of points labelled, {len(struct_ids)} structural ids hidden in *_ns")

    cache = {}
    def pts_for(entry, b):
        """Positions with the entry's poses; colours fixed (final) or from that step's own poses (online)."""
        si = entry["step"]
        for k in [k for k in cache if k[0] < si - 1]:
            del cache[k]
        key = (si, b)
        if key not in cache:
            P = world_points(subs[b], steps[si]["G"][b])
            if a.color_source == "final":
                cols, lab, st_ = fixed[b]
            else:
                lab = label_points(P, steps[si], vs, a.min_count); st_ = np.zeros(len(lab), bool)
                cols = {k: v.astype(np.uint8) for k, v in mode_colors(subs[b]["cols"], lab).items()}
            cache[key] = (P, cols, lab, st_)
        P, cols, lab, st_ = cache[key]
        if entry.get("interp"):
            P = world_points(subs[b], entry["G"][b])
        return P, cols, lab, st_

    # BEV basis from the final map (ceiling removed)
    Pf = np.concatenate([Pfin[b][keep[b]] for b in fin["bases"]])
    Pc = Pf - Pf.mean(0); Pp = Pc - np.outer(Pc @ up, up)
    _, _, Vt = np.linalg.svd(Pp[np.random.default_rng(0).choice(len(Pp), min(len(Pp), 50000), replace=False)], full_matrices=False)
    e1 = Vt[0] - (Vt[0] @ up) * up; e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
    Q = np.stack([Pf @ e1, -(Pf @ e2)], 1); qc = np.stack([Gall[:, :3, 3] @ e1, -(Gall[:, :3, 3] @ e2)], 1)
    lo = np.minimum(np.percentile(Q, 1, 0), qc.min(0)); hi = np.maximum(np.percentile(Q, 99, 0), qc.max(0))
    pad = 0.06 * (hi - lo).max(); lo -= pad; hi += pad
    scale = min((a.W - 20) / (hi[0] - lo[0]), (a.H - 60) / (hi[1] - lo[1]))
    off = np.array([(a.W - scale * (hi[0] - lo[0])) / 2, 40 + (a.H - 40 - scale * (hi[1] - lo[1])) / 2])
    bev_proj = lambda X: np.stack([X @ e1, -(X @ e2)], 1) - lo
    centroid = Pf.mean(0)

    # chase camera path (causal EMA over keyframes, then per-frame interpolation)
    eyes, tgts = [], []
    e_s = t_s = None
    for en in TL:
        b, i = en["cur"]; C = en["G"][b][i]; c = C[:3, 3]
        f = C[:3, 2] - (C[:3, 2] @ up) * up; f /= np.linalg.norm(f) + 1e-9
        e = c - a.back * f + a.height * up; t = c + a.ahead * f
        if e_s is None: e_s, t_s = e, t
        elif not en.get("loop"):  # do not move the camera during loop playback
            e_s = (1 - a.smooth) * e_s + a.smooth * e; t_s = (1 - a.smooth) * t_s + a.smooth * t
        eyes.append(e_s.copy()); tgts.append(t_s.copy())
    fpx = 0.5 * a.W / np.tan(np.radians(a.fov) / 2)

    writers = {}
    if a.nostruct_dir:
        os.makedirs(a.nostruct_dir, exist_ok=True)
    for v in a.views:
        for m in modes_out:
            p = (os.path.join(a.nostruct_dir, f"{v}_{m[:-3]}.mp4") if m.endswith("_ns")
                 else os.path.join(outdir, f"{v}_{m}.mp4"))
            writers[(v, m)] = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{a.W}x{a.H}",
                 "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", p],
                stdin=subprocess.PIPE)

    def gather(en):
        P, C, cams, keys, labs, sts = [], {m: [] for m in modes_out}, [], [], [], []
        for b in sorted(en["rev"]):
            Pb, cols, lab, st_ = pts_for(en, b); m = (subs[b]["fidx"] <= en["rev"][b]) & keep[b]
            P.append(Pb[m]); [C[k].append(cols[k][m]) for k in modes_out]; labs.append(lab[m]); sts.append(st_[m])
            n = min(en["rev"][b], len(en["G"][b]) - 1)
            for i in range(1 if b > 0 else 0, n + 1):
                cams.append(en["G"][b][i]); keys.append(b + i)
        lab = np.concatenate(labs); st_ = np.concatenate(sts)
        en["_ninst"] = len(np.unique(lab[lab >= 0])); en["_nobj"] = len(np.unique(lab[(lab >= 0) & ~st_]))
        return np.concatenate(P), {k: np.concatenate(v) for k, v in C.items()}, np.stack(cams), keys

    def overlay(img, en, title, n_inst):
        b, i = en["cur"]; fid = subs[b]["fids"][i]
        if a.color_source == "final":
            n_inst = en["_nobj"] if title.endswith("_ns") else en["_ninst"]
        title = title.replace("inst_ns", "inst (objects)").replace("mix_ns", "mix (objects)")
        cv2.rectangle(img, (0, 0), (a.W, 34), (20, 20, 20), -1)
        cv2.putText(img, f"{title}   frame {fid:05d}   keyframe {len(en['_keys']):3d}   instances {n_inst:3d}   loops {len(en['loops'])}",
                    (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
        if en["ev"]:
            cv2.putText(img, en["ev"], (12, a.H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 60), 2, cv2.LINE_AA)
        if fid in imgs:
            th = cv2.imread(imgs[fid]); w = a.W // 5; h = int(th.shape[0] * w / th.shape[1])
            img[44:44 + h, a.W - w - 10:a.W - 10] = cv2.resize(th, (w, h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(img, (a.W - w - 11, 43), (a.W - 9, 44 + h), (200, 200, 200), 1)

    def draw_path(img, uv, ok, keys, loops, cur_px):
        for k in range(1, len(uv)):
            if ok[k] and ok[k - 1]:
                cv2.line(img, tuple(uv[k - 1]), tuple(uv[k]), (85, 221, 255), 2, cv2.LINE_AA)
        kp = {kk: k for k, kk in enumerate(keys)}
        for kn, ko in loops:
            if kn in kp and ko in kp and ok[kp[kn]] and ok[kp[ko]]:
                cv2.line(img, tuple(uv[kp[kn]]), tuple(uv[kp[ko]]), (255, 221, 51), 3, cv2.LINE_AA)
        if cur_px is not None:
            cv2.circle(img, cur_px, 7, (51, 51, 255), -1, cv2.LINE_AA); cv2.circle(img, cur_px, 7, (255, 255, 255), 1, cv2.LINE_AA)

    def render(en, eye_T, n_rep):
        P, C, cams, keys = gather(en); en["_keys"] = keys
        n_inst = sum(1 for c in steps[en["step"]].get("inst_count", {}).values() if c >= a.min_count)
        frames = {}
        if "bev" in a.views:
            q = bev_proj(P) * scale + off; uv = np.round(q).astype(int)
            pix, win = splat(uv, -(P @ up), a.W, a.H, a.pt)
            qc = (np.round(bev_proj(cams[:, :3, 3]) * scale + off)).astype(int)
            for m in modes_out:
                img = np.full((a.H * a.W, 3), 17, np.uint8); img[pix] = C[m][win][:, ::-1].astype(np.uint8)
                img = img.reshape(a.H, a.W, 3)
                draw_path(img, qc, np.ones(len(qc), bool), keys, en["loops"], tuple(qc[-1]))
                overlay(img, en, f"BEV | {m}", n_inst); frames[("bev", m)] = img
        if "chase" in a.views:
            for r, T in enumerate(eye_T):
                Wc = np.linalg.inv(T); X = P @ Wc[:3, :3].T + Wc[:3, 3]; z = X[:, 2]; ok = z > 0.05
                uv = np.round(X[ok, :2] / z[ok, None] * fpx + [a.W / 2, a.H / 2]).astype(int)
                pix, win = splat(uv, z[ok], a.W, a.H, a.pt + (1 if a.pt < 3 else 0))
                idx = np.nonzero(ok)[0][win]
                Xc = cams[:, :3, 3] @ Wc[:3, :3].T + Wc[:3, 3]; okc = Xc[:, 2] > 0.05
                uvc = np.round(Xc[:, :2] / np.maximum(Xc[:, 2:], 1e-3) * fpx + [a.W / 2, a.H / 2]).astype(int)
                for m in modes_out:
                    img = np.full((a.H * a.W, 3), 17, np.uint8); img[pix] = C[m][idx][:, ::-1].astype(np.uint8)
                    img = img.reshape(a.H, a.W, 3)
                    draw_path(img, uvc, okc, keys, en["loops"], tuple(uvc[-1]) if okc[-1] else None)
                    overlay(img, en, f"chase | {m}", n_inst); frames.setdefault(("chase", m), []).append(img)
        for (v, m), w in writers.items():
            fr = frames[(v, m)]
            seq = fr if isinstance(fr, list) else [fr] * n_rep
            for f in seq:
                w.stdin.write(np.ascontiguousarray(f).tobytes())

    def interp_eye(k, r, n):
        t = (r + 1) / n
        e0 = eyes[k - 1] if k else eyes[0]; t0 = tgts[k - 1] if k else tgts[0]
        return look_at((1 - t) * e0 + t * eyes[k], (1 - t) * t0 + t * tgts[k], up)

    for k, en in enumerate(TL):
        n = en.get("sub", a.sub)
        eye_T = [interp_eye(k, r, n) for r in range(n)] if "chase" in a.views else []
        render(en, eye_T, n)
        print(f"\r  [{k + 1}/{len(TL)}]", end="", flush=True)
    # outro: hold BEV, chase pulls up to an overview of the whole map
    if a.outro:
        en = TL[-1]; ext = np.percentile(np.linalg.norm(Pf - centroid, axis=1), 90)
        e_end = centroid + up * ext * 1.6 - (tgts[-1] - eyes[-1]) / (np.linalg.norm(tgts[-1] - eyes[-1]) + 1e-9) * ext * 0.6
        eye_T = []
        for r in range(a.outro):
            t = 0.5 - 0.5 * np.cos(np.pi * (r + 1) / a.outro)
            eye_T.append(look_at((1 - t) * eyes[-1] + t * e_end, (1 - t) * tgts[-1] + t * centroid, up))
        en = dict(en, ev="")
        render(en, eye_T, a.outro)
    print()
    for (v, m), w in writers.items():
        w.stdin.close(); w.wait()
    if a.nostruct_dir and "rgb" in a.modes:          # the rgb stream is identical -> copy so each dir is complete
        import shutil
        for v in a.views:
            shutil.copy(os.path.join(outdir, f"{v}_rgb.mp4"), os.path.join(a.nostruct_dir, f"{v}_rgb.mp4"))
    print(f"[video] -> {outdir}" + (f" + {a.nostruct_dir}" if a.nostruct_dir else ""))


if __name__ == "__main__":
    main()
