"""Verification-time joint re-inference: run DA3 on the candidate pair (2-frame chunk -> relative pose,
depth, K), then our semantic model on that joint geometry (fresh KV) and the map-free S1+S2 verdict.
This mirrors the VGGT-SLAM plug-in arm C (which was perfect on Clio office) inside our own stack.

    python scripts/joint_verify_da3.py --data data/clio/office --pairs 897:492 928:474 545:84 ...
"""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, "scripts")
from vggt1_lc.verifier import Vggt1SemVerifier
from model.da3_infer import Da3ChunkModel

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--pairs", nargs="+", required=True)
    a = ap.parse_args()
    ext = "jpg" if os.path.exists(os.path.join(a.data, "rgb", "frame_00000.jpg")) else "png"
    da3 = Da3ChunkModel(); V = Vggt1SemVerifier(dump_dir="", s2_enable=True)
    for pr in a.pairs:
        q, r = [int(x) for x in pr.split(":")]
        paths = [os.path.join(a.data, "rgb", f"frame_{i:05d}.{ext}") for i in (q, r)]
        out = da3.infer_chunk(paths, mode="rgb")
        imgs = out["images"].numpy().astype(np.float32) / 255.0          # (N,H,W,3)
        frames = [torch.from_numpy(im.transpose(2, 0, 1)) for im in imgs]
        depth = out["depth"].numpy()[..., None]; conf = out["world_points_conf"].numpy()
        c2w = out["poses"].numpy(); extr = np.linalg.inv(c2w)[:, :3, :4]; K = out["intrinsics"].numpy()
        rel_t = float(np.linalg.norm(c2w[1, :3, 3] - c2w[0, :3, 3]))
        ok, info = V.verify(frames, depth, conf, extr, K, 0, 1, q_path=str(q), r_path=str(r))
        print(f"f{q:05d}<->f{r:05d}: DA3 joint |t|={rel_t:.2f}m -> {info['s1']} | S2 {info['s2']} -> {'ACCEPT' if ok else 'REJECT'}")
if __name__ == "__main__":
    main()
