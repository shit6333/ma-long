"""Convert the semantic/instance eval datasets to the repo scene layout used by `ma_slam_semantic.run --geometry ma`:

    data/<set>/<scene>/rgb/frame_XXXXX.{png,jpg}   (hard link, or re-encoded when the frame must be rotated)
    data/<set>/<scene>/intrinsic.txt               4x4 pinhole K of the rgb frames
    data/<set>/<scene>/gt_pose.txt                 TUM rows (timestamp = frame index), one per rgb frame, c2w

Frame sets follow the OpenSe3r GT-geometry loaders (semantic_mapping/data.py) exactly:
  scannet_val50 : scannet_eval/extracted/<sc>/color (the stride-10 .sens extraction), K = intrinsic_depth (frames are 640x480)
  scenenn       : frames + poses from oni/scenenn_mv_infos_val.pkl (ESAM val convention), K fixed 544.47 @ 640x480
  replica       : results/frame%06d.jpg every 20th frame (100/scene), traj.txt c2w, K from cam_params.json
  3rscan        : sequence/frame-*.color.jpg rotated 90 deg CW (upright), K swapped as in ThreeRScanGTScene,
                  c2w_new = c2w_old @ R_CAM; frames with non-finite pose dropped

    python scripts/prep_sem_eval_data.py scannet_val50 | scenenn | replica | 3rscan   [--scenes ...]
"""
import argparse, glob, json, os, pickle, sys
import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data")


def link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(os.path.abspath(src), dst)


def write_scene(out, K, c2ws, note=""):
    os.makedirs(out, exist_ok=True)
    K4 = np.eye(4); K4[:3, :3] = K[:3, :3]
    np.savetxt(os.path.join(out, "intrinsic.txt"), K4, fmt="%.6f")
    rows = []
    for i, T in enumerate(c2ws):
        q = R.from_matrix(T[:3, :3]).as_quat()
        rows.append(f"{i} {T[0,3]:.6f} {T[1,3]:.6f} {T[2,3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    with open(os.path.join(out, "gt_pose.txt"), "w") as f:
        f.write(f"# timestamp tx ty tz qx qy qz qw  {note}\n" + "\n".join(rows) + "\n")


def scannet_val50(scenes):
    E = "/mnt/HDD4/ricky/data/scannet_eval/extracted"
    lst = [l.strip() for l in open("/mnt/HDD4/ricky/data/scannet_eval/splits/scannetv2_val.txt") if l.strip()][:50]
    for sc in (scenes or lst):
        src = os.path.join(E, sc); out = os.path.join(OUT, "scannet_val50", sc)
        color = sorted(f for f in os.listdir(os.path.join(src, "color")) if f.endswith((".png", ".jpg")))
        poses = sorted(f for f in os.listdir(os.path.join(src, "pose_txt")) if f.endswith(".txt"))
        assert len(color) == len(poses), (sc, len(color), len(poses))
        os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
        c2ws = []
        for i, (c, pz) in enumerate(zip(color, poses)):
            link(os.path.join(src, "color", c), os.path.join(out, "rgb", f"frame_{i:05d}{os.path.splitext(c)[1]}"))
            c2ws.append(np.loadtxt(os.path.join(src, "pose_txt", pz)))
        K = np.loadtxt(os.path.join(src, "intrinsic", "intrinsic_depth.txt"))
        write_scene(out, K, c2ws, "(ScanNet extracted stride-10, K=intrinsic_depth, 640x480)")
        print(f"[scannet_val50] {sc}: {len(color)} frames -> {out}")


def scenenn(scenes):
    ONI = "/mnt/HDD4/ricky/data/instance_eval_data/scenenn/oni"
    infos = pickle.load(open(os.path.join(ONI, "scenenn_mv_infos_val.pkl"), "rb"))
    K = np.array([[544.47329, 0, 320.0], [0, 544.47329, 240.0], [0, 0, 1]])
    for e in infos:
        sc = str(e["point_cloud"]["scene_idx"])
        if scenes and sc not in scenes:
            continue
        out = os.path.join(OUT, "scenenn", sc); os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
        for i, rel in enumerate(e["img_paths"]):
            link(os.path.join(ONI, rel), os.path.join(out, "rgb", f"frame_{i:05d}.png"))
        write_scene(out, K, np.asarray(e["poses"], dtype=np.float64), "(SceneNN ESAM val frames from scenenn_mv_infos_val.pkl)")
        print(f"[scenenn] {sc}: {len(e['img_paths'])} frames -> {out}")


def replica(scenes, stride=20, out_name=None):
    RP = "/mnt/HDD4/ricky/data/replica/Replica"
    cam = json.load(open(os.path.join(RP, "cam_params.json")))["camera"]
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]])
    for sc in (scenes or ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]):
        src = os.path.join(RP, sc); out = os.path.join(OUT, out_name or "replica", sc); os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
        color = sorted(f for f in os.listdir(os.path.join(src, "results")) if f.startswith("frame"))
        poses = np.loadtxt(os.path.join(src, "traj.txt")).reshape(-1, 4, 4)
        sel = list(range(0, len(color), stride))
        for i, j in enumerate(sel):
            link(os.path.join(src, "results", color[j]), os.path.join(out, "rgb", f"frame_{i:05d}{os.path.splitext(color[j])[1]}"))
        write_scene(out, K, poses[sel], f"(Replica frame_stride {stride}: frames {sel[0]},{sel[1]},... of results/)")
        print(f"[replica] {sc}: {len(sel)} frames -> {out}")


def threerscan(scenes):
    import PIL.Image
    RS = "/mnt/HDD4/ricky/data/3RScan_data"
    R_CAM = np.eye(4); R_CAM[:3, :3] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]
    ids = scenes or sorted(f[:-len("_3rscan_gt.npz")] for f in os.listdir(os.path.join(RS, "_gt")) if f.endswith("_3rscan_gt.npz"))
    for sc in ids:
        seq = os.path.join(RS, sc, "sequence"); out = os.path.join(OUT, "3rscan", sc); os.makedirs(os.path.join(out, "rgb"), exist_ok=True)
        info = {}
        for line in open(os.path.join(seq, "_info.txt")):
            if "=" in line:
                k, v = line.split("=", 1); info[k.strip()] = v.strip()
        W, H = int(info["m_colorWidth"]), int(info["m_colorHeight"])
        K4 = np.asarray([float(x) for x in info["m_calibrationColorIntrinsic"].split()]).reshape(4, 4)
        fx, fy, cx, cy = K4[0, 0], K4[1, 1], K4[0, 2], K4[1, 2]
        K = np.array([[fy, 0, (H - 1) - cy], [0, fx, cx], [0, 0, 1]])          # upright (90 deg CW) frame
        stems = sorted(f[:-len(".color.jpg")] for f in os.listdir(seq) if f.endswith(".color.jpg"))
        c2ws, n = [], 0
        for st in stems:
            T = np.loadtxt(os.path.join(seq, st + ".pose.txt"))
            if T.shape != (4, 4) or not np.isfinite(T).all():
                continue
            dst = os.path.join(out, "rgb", f"frame_{n:05d}.jpg")
            if not os.path.exists(dst):
                PIL.Image.open(os.path.join(seq, st + ".color.jpg")).convert("RGB").transpose(PIL.Image.ROTATE_270).save(dst, quality=95)
            c2ws.append(T @ R_CAM); n += 1
        write_scene(out, K, c2ws, f"(3RScan upright: 90 deg CW rotation, c2w @ R_CAM; {len(stems)-n} non-finite-pose frames dropped)")
        print(f"[3rscan] {sc}: {n}/{len(stems)} frames -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dataset", choices=["scannet_val50", "scenenn", "replica", "3rscan"])
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=20, help="replica only: frame stride (20 = the OpenSe3r GT-geometry convention, 1 = the full 2000-frame sequence as OVO-SLAM/ORB-SLAM use)")
    ap.add_argument("--out_name", default=None, help="replica only: output dir name under data/ (default 'replica')")
    a = ap.parse_args()
    if a.dataset == "replica":
        replica(a.scenes, stride=a.stride, out_name=a.out_name)
    else:
        {"scannet_val50": scannet_val50, "scenenn": scenenn, "3rscan": threerscan}[a.dataset](a.scenes)
