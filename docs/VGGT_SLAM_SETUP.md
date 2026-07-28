# Running VGGT-SLAM 1.0 / 2.0 on this box

Both baselines are installed and verified end-to-end (2026-07-27).
Reproduce with `bash scripts/setup_vggt_slam.sh [2|1|both]`.

## Envs

| | VGGT-SLAM 2.0 | VGGT-SLAM 1.0 |
|---|---|---|
| repo | `thirdparty/VGGT-SLAM` (`main`) | `thirdparty/vggt-slam-1/VGGT-SLAM` (`version1.0`) |
| env | `/mnt/HDD4/ricky/envs/vggt_slam2` | `/mnt/HDD4/ricky/envs/vggt_slam1` |
| python | 3.11.15 | 3.11.15 |
| torch | 2.8.0+cu128 | 2.8.0+cu128 |
| VGGT | `MIT-SPARK/VGGT_SPARK` (fork) | `facebookresearch/vggt` |
| gtsam | `gtsam-develop 4.3a1.dev202607020747` | same |
| numpy | 2.4.4 | 2.4.4 |

## Why the upstream `./setup.sh` cannot be used as-is

1. **`torch==2.3.1` pin (cu121) is fatal here.** The GPUs are sm_120
   (RTX PRO 6000 Blackwell, compute cap 12.0) and need cu128 → torch ≥ 2.7.
   Install `torch 2.8.0+cu128 / torchvision 0.23.0` first, then `requirements.txt`
   with the two torch lines stripped. Same for the VGGT fork's own
   `numpy<2` / `torch==2.3.1` pins → install third-party repos with `--no-deps`.
2. **Python must be 3.11.** Both versions need `from gtsam import SL4, PriorFactorSL4,
   BetweenFactorSL4`; only the `gtsam-develop` nightly exports those, and it ships
   **cp311/312/313/314 wheels — no cp310**. So `amb3r_bw` (py3.10) is out.
   `amb3r_bw`'s regular `gtsam 4.3a0` does **not** have SL4, and `gtsam` /
   `gtsam-develop` both own the `gtsam` module → they cannot coexist.
3. **1.0 and 2.0 need different VGGT repos**, both named `vggt` → one env each.
   The fork is not cosmetic (adds image-retrieval verification, match check, layer viz).
4. **SAM 3 + Perception Encoder are skipped** (setup.sh steps 4/5). They are used only
   by `main.py --run_os` (open-set 3D object detection); nothing in `vggt_slam/` imports
   them. Add later if the open-set path is wanted.

## Weights

- `dino_salad.ckpt` — already in this repo at `src/weights/dino_salad.ckpt` (352 MB),
  symlinked to `~/.cache/torch/hub/checkpoints/dino_salad.ckpt`. No re-download.
- `facebook/VGGT-1B` `model.pt` (5.0 GB) — `torch.hub`'s plain HTTP download **truncated
  at ~8%** and left a corrupt file (`PytorchStreamReader failed reading zip archive`).
  Fetched with `huggingface_hub.hf_hub_download` (resumable) and symlinked into the hub
  cache. If that error reappears, delete `~/.cache/torch/hub/checkpoints/model.pt` and
  re-run the download block at the end of `scripts/setup_vggt_slam.sh`.
- DINOv2 ViT-B/14 (330 MB) downloads automatically for SALAD.

## Smoke test (both verified)

```bash
cd thirdparty/VGGT-SLAM                 # 2.0
CUDA_VISIBLE_DEVICES=0 /mnt/HDD4/ricky/envs/vggt_slam2/bin/python main.py \
    --image_folder /mnt/HDD4/ricky/slam/ma_long/data/office_loop \
    --max_loops 1 --log_results --skip_dense_log --log_path poses.txt
```

`office_loop` (473 frames), both found the same single loop closure (SALAD similarity 0.621):

| | keyframes | submaps | loops | wall time | notes |
|---|---|---|---|---|---|
| 2.0 | 208 | 14 | 1 (submap 204→0) | 69.9 s (~3.0 FPS) | VGGT 0.065 s/frame, LC 0.215 s/frame |
| 1.0 | — | 13 | 1 (submap 12→0) | — | |

Add `--vis_map` for the viser visualizer (`--vis_voxel_size 0.005` on big scenes).

## Gotchas

- **GPU memory.** VGGT-SLAM needs ~10 GB. Both GPUs on this box are often nearly full
  from other jobs — check `nvidia-smi` and pick the free one; 1.0 OOM'd on GPU 0 at
  87/98 GB used and ran fine on GPU 1. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  helps at the margin.
- **`xFormers is not available` warnings** from DINOv2 are harmless (falls back to SDPA).
- **Pose log format.** `write_poses_to_file` writes `frame_id x y z qx qy qz qw`, where
  `frame_id` is parsed from the *image filename*, not a real timestamp. Their
  `evals/eval_tum.sh` works because MASt3R-SLAM's TUM dump names files by timestamp.
  Our `data/tum/fr1_*/rgb/frame_XXXXX.png` is index-named, so `evo_ape tum` against
  `gt_pose.txt` (real timestamps) will not associate — a head-to-head vs `ma_slam` needs
  an index→timestamp remap of the estimate (or symlinking the rgb frames to
  timestamp names) before evaluation.
- Loop closure logs are only emitted for keyframes; frame count in the pose file is
  keyframes, not input frames (208 of 473 for office_loop).
