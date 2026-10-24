"""Hypothesis check: instance descriptors are context-dependent. Embed a candidate pair JOINTLY (one 2-frame
semantic forward, fresh KV) using RGB + the dataset's depth/K, and compare the MNN descriptor similarities
with what the pipeline saw when the two frames were embedded in different submaps.

    python scripts/joint_embed_check.py --data data/clio/office --pairs 897:492 928:474 545:84 590:115 1380:1040
"""
import argparse, os, sys, numpy as np, cv2, torch
sys.path.insert(0, "scripts")
from vggt1_lc.verifier import Vggt1SemVerifier
from ma_slam_semantic.bidesc_lc import BiDescConfig

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--depth_scale", type=float, default=1000.0); a = ap.parse_args()
    K = np.loadtxt(os.path.join(a.data, "intrinsic.txt"))[:3, :3]
    V = Vggt1SemVerifier(dump_dir="", s2_enable=True)
    ext = "jpg" if os.path.exists(os.path.join(a.data, "rgb", "frame_00000.jpg")) else "png"
    for pr in a.pairs:
        q, r = [int(x) for x in pr.split(":")]
        frames, depth, conf, extr, Ks = [], [], [], [], []
        for i in (q, r):
            im = cv2.cvtColor(cv2.imread(os.path.join(a.data, "rgb", f"frame_{i:05d}.{ext}")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            d = cv2.imread(os.path.join(a.data, "depth", f"frame_{i:05d}.png"), -1).astype(np.float32) / a.depth_scale
            if d.shape != im.shape[:2]: d = cv2.resize(d, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
            frames.append(torch.from_numpy(im.transpose(2, 0, 1))); depth.append(d[..., None]); conf.append((d > 0).astype(np.float32))
            extr.append(np.eye(4)[:3]); Ks.append(K)          # both frames in their own camera frame (identity)
        ok, info = V.verify(frames, np.stack(depth), np.stack(conf), np.stack(extr), np.stack(Ks), 0, 1, q_path=str(q), r_path=str(r))
        print(f"f{q:05d}<->f{r:05d}: JOINT embed -> {info['s1']} | S2 {info['s2']}")
if __name__ == "__main__":
    main()
