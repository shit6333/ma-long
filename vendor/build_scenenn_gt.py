"""Build a per-scene SceneNN instance/semantic GT point cloud for eval.

Aggregates the ESAM per-frame GT (point/ins/label .npy, 20k pts/frame — globally
consistent instance ids from the mesh annotation) across all val frames of a
scene, voxel-dedups, and saves:
    <out>/<scene>_scenenn_gt.npz   centers [Nv,3] world, instance [Nv] int,
                                   semantic [Nv] nyu40-id
Consumed by eval_instance_npz_scannet.py --gt_scenenn_npz for class-agnostic AP.

Usage:
    python build_scenenn_gt.py --scene 015 \
        --oni /mnt/HDD4/ricky/data/instance_eval_data/scenenn/oni \
        --voxel 0.02 --out /tmp/scenenn_gt
"""
import argparse, os, os.path as osp, pickle
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True)
    ap.add_argument('--oni', default='/mnt/HDD4/ricky/data/instance_eval_data/scenenn/oni')
    ap.add_argument('--voxel', type=float, default=0.02)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    infos = pickle.load(open(osp.join(a.oni, 'scenenn_mv_infos_val.pkl'), 'rb'))
    e = next((x for x in infos if str(x['point_cloud']['scene_idx']) == str(a.scene)), None)
    if e is None:
        raise ValueError(f"scene {a.scene} not in mv_infos")

    xyz_l, ins_l, sem_l = [], [], []
    for pp, ip, sp in zip(e['pts_paths'], e['instance_paths'], e['semantic_paths']):
        pts = np.load(osp.join(a.oni, pp))          # [20000, 7] world xyz + rgb + pix_idx
        ins = np.load(osp.join(a.oni, ip)).astype(np.int64)   # [20000] instance
        sem = np.load(osp.join(a.oni, sp)).astype(np.int64)   # [20000] nyu40
        xyz_l.append(pts[:, :3].astype(np.float32))
        ins_l.append(ins); sem_l.append(sem)
    xyz = np.concatenate(xyz_l); ins = np.concatenate(ins_l); sem = np.concatenate(sem_l)
    print(f"[gt] scene {a.scene}: {len(e['pts_paths'])} frames → {xyz.shape[0]:,} raw pts")

    # voxel-dedup: majority instance + semantic per voxel
    vs = a.voxel
    vox = np.floor(xyz / vs).astype(np.int64)
    OFF = 1 << 20
    key = ((vox[:, 0] + OFF) << 42) + ((vox[:, 1] + OFF) << 21) + (vox[:, 2] + OFF)
    uk, inv = np.unique(key, return_inverse=True)
    Nv = len(uk)
    centers = np.zeros((Nv, 3), np.float32)
    np.add.at(centers, inv, xyz)
    cnt = np.bincount(inv, minlength=Nv).astype(np.float32)
    centers /= cnt[:, None]

    def majority(vals):
        out = np.full(Nv, -1, np.int64)
        # per-voxel mode via sort-group
        order = np.argsort(inv, kind='stable')
        inv_s, vals_s = inv[order], vals[order]
        bnd = np.concatenate(([0], np.where(np.diff(inv_s) != 0)[0] + 1, [len(inv_s)]))
        for b0, b1 in zip(bnd[:-1], bnd[1:]):
            vv = vals_s[b0:b1]
            u, c = np.unique(vv, return_counts=True)
            out[inv_s[b0]] = u[c.argmax()]
        return out

    ins_v = majority(ins)
    sem_v = majority(sem)
    print(f"[gt] voxelized @{vs}m → {Nv:,} voxels;  instances={len(np.unique(ins_v[ins_v>=0]))}  "
          f"nyu40 classes={len(np.unique(sem_v[sem_v>=0]))}")
    print(f"[gt] world range x[{centers[:,0].min():.2f},{centers[:,0].max():.2f}] "
          f"y[{centers[:,1].min():.2f},{centers[:,1].max():.2f}] "
          f"z[{centers[:,2].min():.2f},{centers[:,2].max():.2f}]")

    out = osp.join(a.out, f'{a.scene}_scenenn_gt.npz')
    np.savez(out, centers=centers, instance=ins_v, semantic=sem_v)
    print(f"[gt] saved → {out}")


if __name__ == '__main__':
    main()
