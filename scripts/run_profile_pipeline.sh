#!/bin/bash
# Per-component timing + peak VRAM of the full pipeline (geo prediction / Rememb3r / loop closure).
# Run ONLY on an otherwise idle GPU (no training, no Isaac Sim, no other eval) or the fps is meaningless.
#   GPU=0 bash scripts/run_profile_pipeline.sh
# Output: outputs/profile/<scene>.json (+ run logs) and outputs/profile/SUMMARY.md
set -u
cd "$(dirname "$0")/.."
PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python
GPU=${GPU:-0}
OUT=outputs/profile; mkdir -p $OUT
KV="--kv_policy h2o --h2o_recent 8 --h2o_hh 8 --h2o_decay 0.9 --h2o_subsample 0.25"
LC="--sem_lc veto --sem_lc_mode rigid --rigid_geom off --rigid_lowobj reject --proposal_conf gate --lc_proposal_conf prefer --lc_history bag --sim_threshold 0.40 --max_per_submap 5"
BASE="--geometry ma --backend da3 --mode rgb --submap_size 20 --chunk_size 4 --keyframe_disparity 25 --ckpt a1_pq --conf_gate_coef 0.5 --assoc new $KV $LC"
# warmup (cuDNN autotune / allocator), result discarded; then a short / medium / long scene
SCENES=${SCENES:-"scannet_val50:scene0221_01 scannet_val50:scene0568_00 scannet_val50:scene0050_00 3rscan:0988ea72-eb32-2e61-8344-99e2283c2728 scannetpp:09c1414f1b"}
echo "== nvidia-smi before run"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "== warmup $(date +%H:%M)"
PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -m ma_slam_semantic.profile_pipeline --prof_out $OUT/_warmup.json -- \
    $BASE --data data/scannet_val50/scene0221_01 --gt data/scannet_val50/scene0221_01/gt_pose.txt --out $OUT/_warmup > $OUT/_warmup.log 2>&1
for spec in $SCENES; do
  DS=${spec%%:*}; SC=${spec#*:}; D=data/$DS/$SC
  [ -d $D ] || { echo "missing $D"; continue; }
  echo "== $DS $SC $(date +%H:%M)"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -m ma_slam_semantic.profile_pipeline --prof_out $OUT/${DS}_${SC}.json -- \
      $BASE --data $D --gt $D/gt_pose.txt --out $OUT/${DS}_${SC} > $OUT/${DS}_${SC}.log 2>&1
  echo "   exit=$?"
done
PYTHONPATH=src $PY scripts/summarize_profile.py $OUT > $OUT/SUMMARY.md && cat $OUT/SUMMARY.md
echo PROFILE_DONE
