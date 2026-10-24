#!/bin/bash
# Ours WITHOUT semantic LC on TUM fr1 x9 at VGGT-SLAM's OFFICIAL keyframer (50 px), w32:
#   da3_rgb  = DA3 rgb-only (same input as VGGT-SLAM)      ma_rgbdi = MA rgb+depth+intr
set -u
cd /mnt/HDD4/ricky/slam/ma_long
PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python
GPU=${GPU:-1}
KF=${KF:-50}
ROOT=outputs/ma_slam/tum_kf${KF}; mkdir -p $ROOT
SUM=$ROOT/summary.csv; [ -f $SUM ] || echo "seq,config,sim3_ate,se3_ate,scale,submaps,loops" > $SUM
for SEQ in ${SEQS:-360 desk desk2 floor plant room rpy teddy xyz}; do
  D=data/tum/fr1_${SEQ}_dist
  for CFG in ${CFGS:-da3_rgb ma_rgbdi}; do
    case $CFG in
      da3_rgb)  X="--backend da3 --mode rgb";;
      ma_rgbdi) X="--backend ma --mode rgb+depth+intr --depth_scale 5000";;
    esac
    od=$ROOT/${CFG}/fr1_${SEQ}; mkdir -p $od
    echo "== $SEQ $CFG $(date +%H:%M)"
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -m ma_slam.run --scene $D $X --manifold se3 \
        --submap_size 32 --keyframe_disparity $KF --sim_threshold 0.50 --max_per_submap 1 --gt $D/gt_pose.txt --out $od > $od/run.log 2>&1
    ex=$?; L=$(grep "Sim3-ATE" $od/run.log | tail -1); echo "  exit=$ex $L"
    s3=$(echo "$L" | grep -oE "Sim3-ATE=[0-9.]+" | cut -d= -f2)
    se=$(echo "$L" | grep -oE "SE3-ATE=[0-9.]+" | cut -d= -f2)
    sc=$(echo "$L" | grep -oE "scale=[0-9.]+" | cut -d= -f2)
    sm=$(echo "$L" | grep -oE "submaps=[0-9]+" | cut -d= -f2)
    lp=$(echo "$L" | grep -oE "loops=[0-9]+" | cut -d= -f2)
    echo "$SEQ,$CFG,$s3,$se,$sc,$sm,$lp" >> $SUM
  done
done
echo BASE_D50_DONE; cat $SUM
