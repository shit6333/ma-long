"""Visualise the S1 instance-correspondence gate on a loop candidate pair: both frames are jointly
re-inferred (DA3, 2-frame chunk -> geometry) and embedded by the instance model; MNN-matched instances
(cos >= tau) get the SAME colour on both images, unmatched proposals a thin grey outline.

    PYTHONPATH=src:scripts python scripts/vis_lc_pair.py --data data/tum/fr1_room_dist --pairs 747:394 1227:235 --out outputs/figs/lc_examples
"""
import argparse, os, sys
import numpy as np, torch, cv2
sys.path.insert(0, "scripts")
from vggt1_lc.verifier import Vggt1SemVerifier
from ma_slam_semantic.proposals import make_proposals
from ma_slam_semantic.bidesc_lc import verify_rigid
from model.da3_infer import Da3ChunkModel

# unmatched instances: muted colours, unique across BOTH frames (never reused between the two images)
PALETTE_UNMATCHED = [(160, 160, 160), (190, 150, 120), (120, 150, 190), (150, 190, 150), (200, 180, 120),
                     (170, 130, 170), (130, 170, 170), (200, 140, 140), (140, 140, 200), (180, 200, 140),
                     (120, 120, 120), (200, 160, 100), (100, 160, 200), (160, 200, 160), (210, 200, 150),
                     (150, 110, 150), (110, 150, 150), (210, 150, 150), (150, 150, 210), (190, 210, 150),
                     (140, 100, 80), (80, 100, 140), (100, 140, 100), (180, 160, 90), (140, 90, 140),
                     (90, 140, 140), (180, 110, 110), (110, 110, 180), (160, 180, 110), (200, 200, 200),
                     (100, 100, 100), (170, 140, 110)]
# --colour_unmatched: unmatched instances get muted colours, unique across BOTH frames (never shared between images)
PALETTE_UNMATCHED = [(160, 160, 160), (190, 150, 120), (120, 150, 190), (150, 190, 150), (200, 180, 120),
                     (170, 130, 170), (130, 170, 170), (200, 140, 140), (140, 140, 200), (180, 200, 140),
                     (120, 120, 120), (200, 160, 100), (100, 160, 200), (160, 200, 160), (210, 200, 150),
                     (150, 110, 150), (110, 150, 150), (210, 150, 150), (150, 150, 210), (190, 210, 150),
                     (140, 100, 80), (80, 100, 140), (100, 140, 100), (180, 160, 90), (140, 90, 140),
                     (90, 140, 140), (180, 110, 110), (110, 110, 180), (160, 180, 110), (200, 200, 200),
                     (100, 100, 100), (170, 140, 110)]
PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128), (170, 110, 40),
           (128, 0, 0), (170, 255, 195), (128, 128, 0), (0, 0, 128)]


def overlay(img, own, colour_of, unmatched_colour=None, alpha=0.55, alpha_un=0.45):
    """img (H,W,3) uint8 RGB; own (H,W) proposal index map (-1 = none); colour_of: MATCHED proposal idx -> RGB.
    Unmatched bag members: thin grey outline, or a muted fill when unmatched_colour maps them to a colour."""
    out = img.astype(np.float32).copy()
    for pid in np.unique(own):
        if pid < 0:
            continue
        m = own == pid
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        col = colour_of.get(int(pid))
        if col is not None:
            out[m] = (1 - alpha) * out[m] + alpha * np.array(col, np.float32)
            cv2.drawContours(out, cnts, -1, tuple(int(c) for c in col), 2)
            continue
        uc = unmatched_colour.get(int(pid)) if unmatched_colour else None
        if uc is None:
            cv2.drawContours(out, cnts, -1, (200, 200, 200), 1)
        else:
            out[m] = (1 - alpha_un) * out[m] + alpha_un * np.array(uc, np.float32)
            cv2.drawContours(out, cnts, -1, tuple(int(c) for c in uc), 1)
    return out.clip(0, 255).astype(np.uint8)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/figs/lc_examples"); ap.add_argument("--tau", type=float, default=0.85)
    ap.add_argument("--k_bag", type=int, default=16); ap.add_argument("--tag", default="")
    ap.add_argument("--colour_unmatched", action="store_true", help="also write a *_allcol variant: unmatched instances get distinct muted colours")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ext = "jpg" if os.path.exists(os.path.join(a.data, "rgb", "frame_00000.jpg")) else "png"
    da3 = Da3ChunkModel(); V = Vggt1SemVerifier(dump_dir="", mode="pairs", rigid_tau=a.tau, k_bag=a.k_bag)
    scene = os.path.basename(a.data.rstrip("/"))
    for pr in a.pairs:
        q, r = [int(x) for x in pr.split(":")]
        paths = [os.path.join(a.data, "rgb", f"frame_{i:05d}.{ext}") for i in (q, r)]
        out = da3.infer_chunk(paths, mode="rgb")
        imgs = out["images"].numpy().astype(np.float32) / 255.0
        frames = [torch.from_numpy(im.transpose(2, 0, 1)) for im in imgs]
        depth = out["depth"].numpy()[..., None]; conf = out["world_points_conf"].numpy()
        c2w = out["poses"].numpy(); extr = np.linalg.inv(c2w)[:, :3, :4]; K = out["intrinsics"].numpy()
        chunk = V._build_chunk(frames, depth, conf, extr, K, [0, 1])
        V.model.reset_kv(); V.model._box_scale = None
        so = V.model.forward_chunk(chunk)
        props, p2p, _ = make_proposals(so, chunk, V.cfg, sem_backbone=V.model.semantic_backbone)
        c2w_c = chunk["c2w"][0].numpy()
        per = {0: [], 1: []}
        for pi, p in enumerate(props):
            t = int(p["frame"]); d = p["descriptor"]
            d = d.detach().cpu().numpy() if hasattr(d, "detach") else np.asarray(d)
            d = d.astype(np.float32); d /= (np.linalg.norm(d) + 1e-12)
            c = p["center"].detach().cpu().numpy().astype(np.float64); T = np.linalg.inv(c2w_c[t].astype(np.float64))
            per[t].append((float(p.get("objectness", 0.0)), d, (T[:3, :3] @ c + T[:3, 3]).astype(np.float32), pi))
        for t in per:
            per[t].sort(key=lambda x: -x[0]); per[t] = per[t][:a.k_bag]
        D = {t: (np.stack([x[1] for x in per[t]]) if per[t] else np.zeros((0, 256), np.float32)) for t in per}
        C = {t: (np.stack([x[2] for x in per[t]]) if per[t] else np.zeros((0, 3), np.float32)) for t in per}
        v = verify_rigid(D[0], D[1], C[0], C[1], V.rigid_cfg)
        matched = [(pp["q"], pp["r"], pp["sim"]) for pp in v.pairs if pp.get("uniq")]
        # colours: matched pairs share a colour (col); unmatched bag members get a unique muted colour (uncol),
        # used only for the *_allcol variant; the default images draw them as grey outlines.
        col = {0: {}, 1: {}}
        for k, (qi, ri, s_) in enumerate(matched):
            col[0][per[0][qi][3]] = PALETTE[k % len(PALETTE)]; col[1][per[1][ri][3]] = PALETTE[k % len(PALETTE)]
        uncol = {0: {}, 1: {}}; u = 0
        for t in per:
            for x in per[t]:
                if x[3] not in col[t]:
                    uncol[t][x[3]] = PALETTE_UNMATCHED[u % len(PALETTE_UNMATCHED)]; u += 1
        W, H = V.cfg.resolution; panels = []; panels_all = []
        for t in (0, 1):
            im = ((chunk["images"][0, t].numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
            own = p2p[t].numpy(); own = cv2.resize(own.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
            keep = np.full(own.shape, -1, np.int32)               # restrict to bag members
            for pid in list(col[t]) + list(uncol[t]):
                keep[own == pid] = pid
            panels.append(overlay(im, keep, col[t]))
            if a.colour_unmatched:
                panels_all.append(overlay(im, keep, col[t], unmatched_colour=uncol[t]))
        label = f"{scene} f{q:05d} <-> f{r:05d}   matched pairs = {len(matched)}  ->  {'ACCEPT' if v.status == 'PASS' else 'REJECT'}"
        canvas = np.concatenate([panels[0], np.full((H, 12, 3), 255, np.uint8), panels[1]], axis=1)
        bar = np.full((34, canvas.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        canvas = np.concatenate([bar, canvas], axis=0)
        base = os.path.join(a.out, f"{scene}_{q:05d}_{r:05d}{a.tag}")
        cv2.imwrite(base + ".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        cv2.imwrite(base + "_q.png", cv2.cvtColor(panels[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite(base + "_r.png", cv2.cvtColor(panels[1], cv2.COLOR_RGB2BGR))
        if a.colour_unmatched:
            c2 = np.concatenate([panels_all[0], np.full((H, 12, 3), 255, np.uint8), panels_all[1]], axis=1)
            c2 = np.concatenate([bar, c2], axis=0)
            cv2.imwrite(base + "_allcol.png", cv2.cvtColor(c2, cv2.COLOR_RGB2BGR))
            cv2.imwrite(base + "_allcol_q.png", cv2.cvtColor(panels_all[0], cv2.COLOR_RGB2BGR))
            cv2.imwrite(base + "_allcol_r.png", cv2.cvtColor(panels_all[1], cv2.COLOR_RGB2BGR))
        print(f"{label}  | bag sizes {len(per[0])}/{len(per[1])} | sims {[round(s, 2) for _, _, s in matched]} -> {base}.png")


if __name__ == "__main__":
    main()
