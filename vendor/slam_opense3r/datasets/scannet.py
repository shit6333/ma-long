"""
ScanNet dataset loaders for OpenSe3r SLAM.

ScannetSLAMDataset        — RGB + GT poses + intrinsics (no depth)
ScannetSLAMGTDataset      — above + GT depth → precomputed world-space pts_gt

Both datasets follow slam_semantic/datasets/scannet_slam_gt_depth.py:
    1. Color image is first resized to the *depth* native resolution so color
       and depth share the same spatial grid and the same K (intrinsic_depth.txt
       is calibrated for that grid).
    2. The pair is then center-cropped to match target aspect ratio.
    3. Both are resized to the target resolution.
    4. K is updated in lock-step for the crop + resize.

This avoids the previous bug where color was crop+resized using a K calibrated
for the depth grid (different resolution → wrong fx/cx) and depth was resized
without the matching crop, so color and depth pixels did not correspond.
"""

import os
import os.path as osp
from typing import Dict

import numpy as np
import torch
import cv2
import PIL.Image

from torch.utils.data import Dataset

try:
    lanczos = PIL.Image.Resampling.LANCZOS
except AttributeError:
    lanczos = PIL.Image.LANCZOS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _crop_resize_pair(color_img: PIL.Image.Image,
                      depth_m: np.ndarray,
                      intrinsics: np.ndarray,
                      resolution):
    """
    Center-crop color + depth to target aspect, resize both to `resolution`,
    update K accordingly.  Color and depth must already share the same grid.

    Args:
        color_img : PIL Image (W_d, H_d)
        depth_m   : np.ndarray (H_d, W_d) float32 metres (or None)
        intrinsics: (3, 3) calibrated for (W_d, H_d)
        resolution: (target_W, target_H)

    Returns:
        color_img, depth_m (or None), intrinsics — all consistent at target res
    """
    W, H = color_img.size
    target_W, target_H = resolution
    intri = intrinsics.copy().astype(np.float32)

    # Handle portrait orientation (rotate target if image is portrait)
    if target_W >= target_H and H > 1.1 * W:
        target_W, target_H = target_H, target_W

    target_aspect = target_W / target_H
    image_aspect  = W / H

    if image_aspect > target_aspect:
        # Crop width
        new_W = int(round(H * target_aspect))
        left  = (W - new_W) // 2
        color_img = color_img.crop((left, 0, left + new_W, H))
        if depth_m is not None:
            depth_m = depth_m[:, left: left + new_W]
        intri[0, 2] -= left
        crop_W, crop_H = new_W, H
    elif image_aspect < target_aspect:
        # Crop height
        new_H = int(round(W / target_aspect))
        top   = (H - new_H) // 2
        color_img = color_img.crop((0, top, W, top + new_H))
        if depth_m is not None:
            depth_m = depth_m[top: top + new_H, :]
        intri[1, 2] -= top
        crop_W, crop_H = W, new_H
    else:
        crop_W, crop_H = W, H

    # Resize
    color_img = color_img.resize((target_W, target_H), resample=lanczos)
    if depth_m is not None:
        depth_m = cv2.resize(
            depth_m, (target_W, target_H), interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)

    sx = target_W / float(crop_W)
    sy = target_H / float(crop_H)
    intri[0, 0] *= sx;  intri[0, 2] *= sx
    intri[1, 1] *= sy;  intri[1, 2] *= sy

    return color_img, depth_m, intri.astype(np.float32)


def _unproject_depth_np(depth_m: np.ndarray,
                        intrinsics: np.ndarray,
                        c2w: np.ndarray) -> np.ndarray:
    """
    depth_m   : [H, W]  float32, metres (0 = invalid)
    intrinsics: [3, 3]  K at depth_m's resolution
    c2w       : [4, 4]  camera-to-world

    Returns [H, W, 3] float32 world-space pts (zero where depth==0).
    """
    H, W = depth_m.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    us, vs = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )
    X = (us - cx) / fx * depth_m
    Y = (vs - cy) / fy * depth_m
    Z = depth_m

    pts_cam = np.stack([X, Y, Z, np.ones_like(Z)], axis=-1)        # [H, W, 4]
    pts_w   = (c2w @ pts_cam.reshape(-1, 4).T).T[:, :3]            # [H*W, 3]
    pts_w   = pts_w.reshape(H, W, 3).astype(np.float32)
    pts_w[depth_m <= 0] = 0.0
    return pts_w


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: RGB + poses (no depth)
# ─────────────────────────────────────────────────────────────────────────────

class ScannetSLAMDataset(Dataset):
    """
    Loads a ScanNet scene for OpenSe3r SLAM (no GT depth).

    Expected directory layout:
        ROOT/
          color/                          *.jpg or *.png  (sorted by name)
          poses.npy   or pose/*.txt       [T, 4, 4] float32 c2w
          intrinsic/intrinsic_depth.txt   4×4 → take [:3,:3]

    To match the GT-depth dataset's spatial alignment (so color pixels here
    correspond to the same pixel grid as the GT-mode dataset), color images
    are first resized to the depth-native resolution (read from one depth
    image if available, otherwise we keep color native and scale K from
    color native instead).

    Returns a single item with:
        images            [T, 3, H, W]  in [-1, 1]
        camera_pose       [T, 4, 4]     c2w
        camera_intrinsics [T, 3, 3]     K at target (W, H)
    """

    def __init__(self, ROOT: str, resolution=(518, 336)):
        self.ROOT       = ROOT
        self.resolution = resolution
        self.color_dir  = osp.join(ROOT, 'color')
        self.depth_dir  = osp.join(ROOT, 'depth')
        self.intri_path = osp.join(ROOT, 'intrinsic', 'intrinsic_depth.txt')

        self.intri = np.loadtxt(self.intri_path).astype(np.float32)[:3, :3]

        poses_npy = osp.join(ROOT, 'poses.npy')
        if osp.isfile(poses_npy):
            self.poses_all = np.load(poses_npy).astype(np.float32)
        else:
            self.poses_all = self._load_pose_txts()

        self.image_files = sorted(
            f for f in os.listdir(self.color_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        )

        # Probe depth-native resolution if available (for color pre-resize)
        self._depth_native = None
        if osp.isdir(self.depth_dir):
            depth_files = sorted(
                f for f in os.listdir(self.depth_dir) if f.lower().endswith('.png')
            )
            if depth_files:
                _d0 = PIL.Image.open(osp.join(self.depth_dir, depth_files[0]))
                self._depth_native = _d0.size   # (W, H)

    def _load_pose_txts(self):
        pose_dir = osp.join(self.ROOT, 'pose')
        files    = sorted(f for f in os.listdir(pose_dir) if f.endswith('.txt'))
        return np.stack([np.loadtxt(osp.join(pose_dir, f)).astype(np.float32)
                         for f in files])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        images, poses, intris = [], [], []

        for i, fname in enumerate(self.image_files):
            img = PIL.Image.open(osp.join(self.color_dir, fname)).convert('RGB')

            # Resize color to depth-native res (so K applies cleanly)
            if self._depth_native is not None:
                img = img.resize(self._depth_native, resample=lanczos)

            img, _, K = _crop_resize_pair(img, None, self.intri.copy(), self.resolution)
            images.append(
                torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 127.5 - 1.0
            )
            poses.append(torch.from_numpy(self.poses_all[i]))
            intris.append(torch.from_numpy(K))

        return {
            'images':            torch.stack(images),
            'camera_pose':       torch.stack(poses),
            'camera_intrinsics': torch.stack(intris),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset: RGB + poses + GT depth → pts_gt
# ─────────────────────────────────────────────────────────────────────────────

class ScannetSLAMGTDataset(ScannetSLAMDataset):
    """
    Extends ScannetSLAMDataset to also load GT depth maps and compute
    world-space pts_gt [T, H, W, 3] for use in pipeline_gt.py.

    Additional layout:
        ROOT/depth/    *.png  uint16, depth in millimetres (ScanNet default)

    Color and depth are aligned by:
        1. Resize color to depth-native resolution.
        2. Center-crop the pair to target aspect ratio.
        3. Resize both to target resolution.
        4. Update K to track crop + resize.

    pts_gt is unprojected with the *same* K applied to the same grid as the
    color image, so pts and color pixels correspond exactly.
    """

    DEPTH_SCALE_DEFAULT = 1000.0

    def __init__(self, ROOT: str, resolution=(518, 336),
                 depth_scale: float = DEPTH_SCALE_DEFAULT):
        super().__init__(ROOT, resolution)
        self.depth_dir   = osp.join(ROOT, 'depth')
        self.depth_scale = depth_scale

        if not osp.isdir(self.depth_dir):
            raise FileNotFoundError(
                f"GT-mode requires depth/: {self.depth_dir} not found"
            )

        depth_files = sorted(
            f for f in os.listdir(self.depth_dir) if f.lower().endswith('.png')
        )
        assert len(depth_files) == len(self.image_files), (
            f"depth count ({len(depth_files)}) != color count ({len(self.image_files)})"
        )
        self.depth_files = depth_files

    def __getitem__(self, idx):
        images, poses, intris, pts_list, depth_list = [], [], [], [], []

        for i, (cname, dname) in enumerate(zip(self.image_files, self.depth_files)):
            # Load color and resize to depth-native res
            img = PIL.Image.open(osp.join(self.color_dir, cname)).convert('RGB')
            depth_raw = cv2.imread(
                osp.join(self.depth_dir, dname), cv2.IMREAD_UNCHANGED,
            )
            if depth_raw is None:
                raise FileNotFoundError(f"Cannot read depth: {dname}")

            depth_native_H, depth_native_W = depth_raw.shape
            img = img.resize((depth_native_W, depth_native_H), resample=lanczos)
            depth_m = depth_raw.astype(np.float32) / self.depth_scale

            # Joint center-crop + resize for color, depth, K
            img, depth_m, K = _crop_resize_pair(
                img, depth_m, self.intri.copy(), self.resolution,
            )

            c2w = self.poses_all[i].astype(np.float32)
            pts = _unproject_depth_np(depth_m, K, c2w)

            images.append(
                torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 127.5 - 1.0
            )
            poses.append(torch.from_numpy(c2w))
            intris.append(torch.from_numpy(K))
            pts_list.append(torch.from_numpy(pts))
            # Camera-frame Z-depth (metres) — fed into MapAnything as
            # `depth_z` when `--use_depth_input` is set.  Invalid pixels
            # (depth==0 from ScanNet) stay zero; downstream models treat
            # them as no-signal.
            depth_list.append(torch.from_numpy(depth_m))

        return {
            'images':            torch.stack(images),
            'camera_pose':       torch.stack(poses),
            'camera_intrinsics': torch.stack(intris),
            'pts_gt':            torch.stack(pts_list),
            'depth_gt':          torch.stack(depth_list),   # [T, H, W]  metres
        }


# ─────────────────────────────────────────────────────────────────────────────
# In-the-wild RGB folder (no poses, no depth, no K)
# ─────────────────────────────────────────────────────────────────────────────

class InTheWildSLAMDataset(Dataset):
    """
    RGB(+D) dataset for in-the-wild inference.

    Expected layout (RGB-only mode):
        ROOT/                  *.jpg or *.png  (sorted by filename), OR
        ROOT/color/            same

    Optional RGBD mode (set `depth_dir` + `intrinsics_path` in __init__):
        ROOT/color/   color frames (one per timestamp)
        depth_dir/    same number of depth files, sorted to match (uint16 PNG
                      = depth in millimetres by default, or float32 metres
                      when `depth_scale=1.0`).  Color and depth must already
                      share the SAME native resolution (no per-frame K
                      tracking yet — supply one global K via
                      `intrinsics_path`).
        intrinsics_path:  text file with one of
                          - 9 floats (3x3 K, row-major), OR
                          - 4 floats: fx fy cx cy (zero skew, K[2,2]=1)
                          intrinsics are expressed at the (color = depth)
                          native resolution and rescaled to `resolution`
                          inside `_crop_resize_pair` (joint crop + resize).

    Returns:
        images:           [T, 3, H, W]  in [-1, 1]
        depth_gt:         [T, H, W]     metres (only when depth_dir provided)
        camera_intrinsics: [T, 3, 3]    (only when intrinsics provided —
                                         repeated per frame; in RGBD mode K
                                         is shared across the sequence)
    """

    IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    DEPTH_EXTS = ('.png', '.tif', '.tiff', '.npy', '.npz')

    def __init__(self, ROOT: str, resolution=(518, 336),
                 max_frames: int = None, frame_stride: int = 1,
                 depth_dir:       str = None,
                 intrinsics_path: str = None,
                 depth_scale:     float = 1000.0,
                 ):
        self.ROOT       = ROOT
        self.resolution = resolution
        self.depth_scale = float(depth_scale)

        sub = osp.join(ROOT, 'color')
        self.image_dir = sub if osp.isdir(sub) else ROOT

        files = sorted(
            f for f in os.listdir(self.image_dir)
            if osp.splitext(f)[1].lower() in self.IMG_EXTS
        )
        if not files:
            raise FileNotFoundError(f"No images in {self.image_dir}")

        # ── Depth handling (optional RGBD mode) ────────────────────────────
        self.depth_dir = depth_dir
        if self.depth_dir is not None:
            if not osp.isdir(self.depth_dir):
                raise FileNotFoundError(f"depth_dir not found: {self.depth_dir}")
            depth_files = sorted(
                f for f in os.listdir(self.depth_dir)
                if osp.splitext(f)[1].lower() in self.DEPTH_EXTS
            )
            if len(depth_files) != len(files):
                raise ValueError(
                    f"In-the-wild RGBD: depth file count "
                    f"({len(depth_files)}) != color file count "
                    f"({len(files)}) — sequences must align frame-by-frame."
                )
            self.depth_files = depth_files
        else:
            self.depth_files = None

        # ── Intrinsics (optional, but required when depth_dir is set) ──────
        self.intrinsics_path = intrinsics_path
        if intrinsics_path is not None:
            self._K_native = self._load_intrinsics(intrinsics_path)
        elif depth_dir is not None:
            raise ValueError(
                "in-the-wild RGBD mode requires intrinsics_path "
                "(MapAnything multi-modal needs K alongside depth_z)."
            )
        else:
            self._K_native = None

        if frame_stride > 1:
            files = files[::frame_stride]
            if self.depth_files is not None:
                self.depth_files = self.depth_files[::frame_stride]
        if max_frames is not None:
            files = files[:max_frames]
            if self.depth_files is not None:
                self.depth_files = self.depth_files[:max_frames]
        self.image_files = files

    @staticmethod
    def _load_intrinsics(path: str) -> np.ndarray:
        """Parse a text file with one of:
          * 4 floats   →  fx fy cx cy   (zero skew, K[2,2]=1)
          * 9 floats   →  3x3 K row-major
          * 16 floats  →  4x4 matrix (e.g. ScanNet-style); the upper-left
                          3x3 is treated as K and the homogeneous padding
                          is dropped.
        """
        with open(path, 'r') as f:
            vals = [float(v) for v in f.read().split() if v.strip()]
        if len(vals) == 4:
            fx, fy, cx, cy = vals
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        elif len(vals) == 9:
            K = np.array(vals, dtype=np.float32).reshape(3, 3)
        elif len(vals) == 16:
            K4 = np.array(vals, dtype=np.float32).reshape(4, 4)
            K = K4[:3, :3].copy()
        else:
            raise ValueError(
                f"intrinsics_path={path!r}: expected 4 (fx fy cx cy), 9 "
                f"(3x3 K), or 16 (4x4 matrix) floats — got {len(vals)} values."
            )
        return K

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        images   = []
        depths   = [] if self.depth_files is not None else None
        intris   = [] if self._K_native is not None else None

        for i, fname in enumerate(self.image_files):
            img = PIL.Image.open(osp.join(self.image_dir, fname)).convert('RGB')

            depth_m = None
            if self.depth_files is not None:
                dname = self.depth_files[i]
                dpath = osp.join(self.depth_dir, dname)
                ext = osp.splitext(dname)[1].lower()
                if ext in ('.npy',):
                    depth_raw = np.load(dpath)
                elif ext in ('.npz',):
                    depth_raw = np.load(dpath)['depth']
                else:
                    depth_raw = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
                if depth_raw is None:
                    raise FileNotFoundError(f"Cannot read depth: {dpath}")
                # Align color to depth's native grid so K applies cleanly
                # (mirrors the ScanNet-GT loader convention).
                if depth_raw.ndim != 2:
                    raise ValueError(
                        f"Depth must be HxW; got {depth_raw.shape} for {dname}"
                    )
                depth_H, depth_W = depth_raw.shape
                img = img.resize((depth_W, depth_H), resample=lanczos)
                depth_m = depth_raw.astype(np.float32) / self.depth_scale

            # Joint center-crop + resize for color (+ depth + K)
            K_in = (self._K_native.copy() if self._K_native is not None
                    else np.eye(3, dtype=np.float32))
            img, depth_m, K = _crop_resize_pair(img, depth_m, K_in, self.resolution)

            images.append(
                torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 127.5 - 1.0
            )
            if depths is not None:
                depths.append(torch.from_numpy(depth_m))
            if intris is not None:
                intris.append(torch.from_numpy(K))

        out: Dict[str, torch.Tensor] = {
            'images': torch.stack(images),   # [T, 3, H, W]
        }
        if depths is not None:
            out['depth_gt'] = torch.stack(depths)         # [T, H, W]  metres
        if intris is not None:
            out['camera_intrinsics'] = torch.stack(intris)  # [T, 3, 3]
        return out
