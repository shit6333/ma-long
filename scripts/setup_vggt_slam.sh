#!/usr/bin/env bash
# Reproduce the two VGGT-SLAM conda envs on this Blackwell box.
#
# Why not just run the upstream ./setup.sh:
#   1. requirements.txt pins torch==2.3.1/torchvision==0.18.1 (cu121). This GPU is
#      sm_120 (RTX PRO 6000 Blackwell) and needs cu128 -> torch >= 2.7. We install
#      torch 2.8.0+cu128 first and strip the pins from requirements.
#   2. gtsam-develop (the only gtsam with SL4/PriorFactorSL4/BetweenFactorSL4) ships
#      cp311+ wheels only -> python 3.11, and it collides with plain `gtsam` (both
#      provide the `gtsam` module), so this cannot live in amb3r_bw (py3.10 + gtsam 4.3a0).
#   3. VGGT-SLAM 2.0 needs the MIT-SPARK/VGGT_SPARK fork, 1.0 needs facebookresearch/vggt.
#      Both install as package `vggt` -> one env each.
#   4. third_party deps are installed --no-deps so their own torch==2.3.1 / numpy<2
#      pins don't undo step 1.
#   5. SAM 3 + Perception Encoder (setup.sh steps 4/5) are skipped: they are only used
#      by main.py --run_os (open-set 3D detection); vggt_slam/ itself never imports them.
#
# Usage:  bash scripts/setup_vggt_slam.sh [2|1|both]     (default: both)

set -euo pipefail

WHICH="${1:-both}"
ENV_ROOT=/mnt/HDD4/ricky/envs
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TORCH_ARGS=(torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128)

setup_one() {
    local env_path="$1" repo="$2" vggt_url="$3"
    local py="$env_path/bin/python"

    if [ ! -x "$py" ]; then
        conda create -p "$env_path" python=3.11 -y
    fi
    "$py" -m pip install --upgrade pip setuptools wheel
    "$py" -m pip install "${TORCH_ARGS[@]}"

    # requirements minus the torch pins
    local req; req="$(mktemp)"
    grep -v -E '^(torch==|torchvision==)' "$repo/requirements.txt" > "$req"
    "$py" -m pip install -r "$req"
    rm -f "$req"

    mkdir -p "$repo/third_party"
    [ -d "$repo/third_party/salad" ] || \
        git clone https://github.com/Dominic101/salad.git "$repo/third_party/salad"
    [ -d "$repo/third_party/vggt" ] || \
        git clone "$vggt_url" "$repo/third_party/vggt"

    "$py" -m pip install --no-deps -e "$repo/third_party/salad" \
                                   -e "$repo/third_party/vggt" \
                                   -e "$repo"
}

# SALAD checkpoint is already in this repo (352 MB) -- link it instead of re-downloading.
mkdir -p "$HOME/.cache/torch/hub/checkpoints"
ln -sfn "$REPO_ROOT/src/weights/dino_salad.ckpt" \
        "$HOME/.cache/torch/hub/checkpoints/dino_salad.ckpt"

if [ "$WHICH" = "2" ] || [ "$WHICH" = "both" ]; then
    setup_one "$ENV_ROOT/vggt_slam2" \
              "$REPO_ROOT/thirdparty/VGGT-SLAM" \
              "https://github.com/MIT-SPARK/VGGT_SPARK.git"
fi

if [ "$WHICH" = "1" ] || [ "$WHICH" = "both" ]; then
    setup_one "$ENV_ROOT/vggt_slam1" \
              "$REPO_ROOT/thirdparty/vggt-slam-1/VGGT-SLAM" \
              "https://github.com/facebookresearch/vggt.git"
fi

# VGGT-1B weights: main.py pulls them through torch.hub, whose plain-HTTP download
# truncated at ~8% here. Fetch via huggingface_hub (resumable) and link into the hub cache.
PY_ANY="$ENV_ROOT/vggt_slam2/bin/python"
[ -x "$PY_ANY" ] || PY_ANY="$ENV_ROOT/vggt_slam1/bin/python"
"$PY_ANY" - <<'EOF'
import os
from huggingface_hub import hf_hub_download
p = hf_hub_download("facebook/VGGT-1B", "model.pt")
dst = os.path.expanduser("~/.cache/torch/hub/checkpoints/model.pt")
if os.path.realpath(dst) != p:
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(p, dst)
print("VGGT-1B ->", dst, "->", p)
EOF

echo "Done. Smoke test:"
echo "  cd thirdparty/VGGT-SLAM && CUDA_VISIBLE_DEVICES=0 $ENV_ROOT/vggt_slam2/bin/python main.py \\"
echo "      --image_folder $REPO_ROOT/data/office_loop --max_loops 1 --log_results --skip_dense_log"
