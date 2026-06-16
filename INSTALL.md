# Install — ma_long / ma_slam

Dependency list for this repo. Developed with **PyTorch 2.8.0 + CUDA 12.8**, Python 3.10 —
adjust the PyTorch CUDA build to match your own GPU / driver / CUDA version.

```bash
# create + activate an env (name it whatever you like; `ma_long` used here)
conda create -p /your/env/path/ma_long python=3.10 -y   # or: conda create -n ma_long python=3.10 -y
conda activate ma_long
```

---

## 1. PyTorch (CUDA 12.8)

```bash
# we used CUDA 12.8 (cu128); pick the index/wheel matching your CUDA — see https://pytorch.org/get-started/
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 2. MapAnything backbone (default `--backend ma`)

```bash
pip install mapanything==1.1.2     # pulls uniception==0.1.7, hydra-core, huggingface-hub, einops, opencv-python-headless
pip install timm==1.0.27           # DINOv2 backbone (keep 1.0.x)
```
Model weights auto-download from HuggingFace on first run (`facebook/map-anything`).

## 3. ma_slam factor-graph backend (gtsam)

```bash
pip install --pre gtsam            # installs 4.3a0
```
⚠️ **gtsam 4.2 segfaults** under numpy 2.x (`gtsam.Pose3(np.eye(4))` crashes) — the `--pre`
4.3a0 wheel is required. Only `ma_slam` needs gtsam.

## 4. Geometry / alignment / loop-closure / eval deps

```bash
pip install numpy scipy pillow matplotlib tqdm pyyaml \
            open3d \                  # point-cloud IO / merge
            pypose \                  # Sim3 pose-graph optimizer (align/)
            faiss-cpu \               # loop retrieval (or faiss-gpu)
            numba trimesh scikit-learn rich triton   # imported (unguarded) by align/
```
(`triton`/`numba` are the alignment accelerators; the default `align_lib=torch` doesn't use them
at runtime, but `align/` imports them at load, so they must be installed.)

## 5. Model weights

| weight | source | size | how |
|---|---|---|---|
| MapAnything (`--backend ma`) | HF `facebook/map-anything` | ~1 GB | **auto** on first run → HF cache |
| DA3 (`--backend da3`) | HF `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | ~5.6 GB | **auto** on first run → HF cache |
| DINOv2 backbone (SALAD) | `torch.hub` `facebookresearch/dinov2` | small | **auto** → torch hub cache |
| **SALAD checkpoint** | `github.com/serizba/salad` v1.0.0 | ~352 MB | **MANUAL** → `src/weights/dino_salad.ckpt` |

Only the **SALAD checkpoint** needs a manual step (it's too large to ship in the repo):

```bash
mkdir -p src/weights
curl -L https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt \
     -o src/weights/dino_salad.ckpt
```
The loop-closure code looks for it at `src/weights/dino_salad.ckpt` by default. HuggingFace
downloads land in the default HF cache (`~/.cache/huggingface`, or set `HF_HOME`).

## 6. DA3 backbone — optional (`--backend da3`, metric rgb)

DA3 has **no pip package** — the adapter (`src/model/da3_infer.py`) imports it from a source
clone under `thirdparty/` (and stubs the export-only `moviepy.editor`, unused for inference).
Clone the DA3 source (not redistributed in this repo):

```bash
mkdir -p thirdparty
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 thirdparty/Depth-Anything-3
```
DA3 weights auto-download on first run (see table above); `gsplat` (3DGS export) is not used here.

> `thirdparty/` is git-ignored — it holds external repos cloned separately. MapAnything is used
> via its **pip package** (step 2), so `thirdparty/map-anything` is not required. `VGGT-SLAM` /
> `VGGT-Long` are reference blueprints only (not imported at runtime).

---

## 7. Verify

```bash
conda activate ma_long
python -c "import torch, gtsam, mapanything, pypose, open3d, faiss; print('core OK')"
# tiny end-to-end (writes poses + run_stats.txt + loops.txt); expect Sim3-ATE ~0.016, scale ~1.0
python src/ma_slam/run.py --scene data/scene0011_00 --mode rgb+depth+intr \
    --out outputs/_install_test --gt data/scene0011_00/gt_pose.txt --submap_size 20 --max_frames 30
# DA3 backbone (metric rgb)
python src/ma_slam/run.py --scene data/scene0011_00 --mode rgb --backend da3 \
    --out outputs/_install_test_da3 --max_frames 30
```
