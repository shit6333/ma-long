"""SceneNN SLAM dataset (GT geometry) — mirrors ScannetSLAMGTDataset.

Layout (ESAM-prepared, `/mnt/HDD4/ricky/data/instance_eval_data/scenenn/oni/`):
    oni/
      scenenn_mv_infos_val.pkl        list[dict]; one entry per val scene, keys:
        point_cloud={'scene_idx': '015'}, poses=[T×(4×4 c2w)],
        img_paths=['015/image/image00001.png', ...],
        pts_paths / semantic_paths / instance_paths  (per-frame 20k-pt GT npy)
      015/image/imageXXXXX.png        RGB 640×480
      015/depth/depthXXXXX.png        uint16, millimetres
      015/{point,label,ins}/XXXXX.npy per-frame GT point cloud (20000 pts)

Fixed intrinsics (data_process.py): fx=fy=544.47329, cx=320, cy=240, 640×480.
Poses are camera-to-world (c2w).  Returns the SAME dict as ScannetSLAMGTDataset:
    images / camera_pose / camera_intrinsics / pts_gt / depth_gt.
"""
import os
import os.path as osp
import pickle

import numpy as np
import cv2
import PIL.Image
import torch
from torch.utils.data import Dataset

from .scannet import _crop_resize_pair, _unproject_depth_np, lanczos

# SceneNN native intrinsics (Asus Xtion, from data_process.py).
_FX = _FY = 544.47329
_CX, _CY = 320.0, 240.0
_SCENENN_K = np.array([[_FX, 0.0, _CX],
                       [0.0, _FY, _CY],
                       [0.0, 0.0, 1.0]], dtype=np.float32)
_MV_INFOS = 'scenenn_mv_infos_val.pkl'


class SceneNNSLAMGTDataset(Dataset):
    """One SceneNN scene, GT depth+poses.  `data_path` = <oni>/<scene_id>
    (e.g. .../scenenn/oni/015); the val-info pkl is read from the parent dir so
    the exact eval frame set (and its GT npy) stay aligned."""

    DEPTH_SCALE_DEFAULT = 1000.0   # SceneNN depth png is uint16 millimetres

    def __init__(self, ROOT: str, resolution=(518, 336),
                 depth_scale: float = DEPTH_SCALE_DEFAULT):
        self.ROOT = ROOT.rstrip('/')
        self.oni_dir = osp.dirname(self.ROOT)
        self.scene_id = osp.basename(self.ROOT)
        self.resolution = resolution
        self.depth_scale = depth_scale

        pkl_path = osp.join(self.oni_dir, _MV_INFOS)
        if not osp.isfile(pkl_path):
            raise FileNotFoundError(f"SceneNN val-info pkl not found: {pkl_path}")
        infos = pickle.load(open(pkl_path, 'rb'))
        entry = next((e for e in infos
                      if str(e['point_cloud']['scene_idx']) == str(self.scene_id)), None)
        if entry is None:
            raise ValueError(f"scene '{self.scene_id}' not in {pkl_path}")

        # img_paths are relative to oni_dir ('015/image/image00001.png').
        self.img_files = list(entry['img_paths'])
        self.poses_all = np.asarray(entry['poses'], dtype=np.float32)          # [T,4,4] c2w
        assert len(self.img_files) == len(self.poses_all), \
            f"img/pose count mismatch: {len(self.img_files)} vs {len(self.poses_all)}"
        self.intri = _SCENENN_K.copy()

    def __len__(self):
        return 1

    def _depth_path(self, img_rel: str) -> str:
        # '015/image/image00001.png' → '015/depth/depth00001.png'
        base = osp.basename(img_rel)                       # image00001.png
        idx = base.replace('image', '').replace('.png', '')  # 00001
        return osp.join(self.oni_dir, self.scene_id, 'depth', f'depth{idx}.png')

    def __getitem__(self, idx):
        images, poses, intris, pts_list, depth_list = [], [], [], [], []
        for i, img_rel in enumerate(self.img_files):
            img = PIL.Image.open(osp.join(self.oni_dir, img_rel)).convert('RGB')
            depth_raw = cv2.imread(self._depth_path(img_rel), cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise FileNotFoundError(f"Cannot read depth for {img_rel}")
            dH, dW = depth_raw.shape
            img = img.resize((dW, dH), resample=lanczos)             # align to depth-native res
            depth_m = depth_raw.astype(np.float32) / self.depth_scale

            img, depth_m, K = _crop_resize_pair(img, depth_m, self.intri.copy(), self.resolution)
            c2w = self.poses_all[i].astype(np.float32)
            pts = _unproject_depth_np(depth_m, K, c2w)

            images.append(torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 127.5 - 1.0)
            poses.append(torch.from_numpy(c2w))
            intris.append(torch.from_numpy(K))
            pts_list.append(torch.from_numpy(pts))
            depth_list.append(torch.from_numpy(depth_m))

        return {
            'images':            torch.stack(images),
            'camera_pose':       torch.stack(poses),
            'camera_intrinsics': torch.stack(intris),
            'pts_gt':            torch.stack(pts_list),
            'depth_gt':          torch.stack(depth_list),   # [T, H, W] metres
        }
