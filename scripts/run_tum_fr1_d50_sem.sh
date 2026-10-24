#!/bin/bash
# Ours (DA3-rgb + sem-LC, a1_pq) on TUM fr1 x9 at VGGT-SLAM's OFFICIAL keyframer (50 px), w32.
# Same flags as the 25 px batch (outputs/sem_lc_shadow/*) except --keyframe_disparity 50 and
# no dump/snapshots. CFG s1 = S1-only (bd_s2 off), s1s2 = S1+S2.
set -u
cd /mnt/HDD4/ricky/slam/ma_long
PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python
GPU=${GPU:-0}
KF=${KF:-50}
EXTRA=${EXTRA:-}
TAG=${TAG:-}
ROOT=outputs/sem_lc_shadow/a1pq_w32_kf${KF}${TAG}
KV="--kv_policy h2o --h2o_recent 8 --h2o_hh 8 --h2o_decay 0.9 --h2o_subsample 0.25"
mkdir -p $ROOT
SUM=$ROOT/summary.csv; [ -f $SUM ] || echo "seq,config,sim3_ate,se3_ate,scale,loops" > $SUM
for SEQ in ${SEQS:-360 desk desk2 floor plant room rpy teddy xyz}; do
  D=data/tum/fr1_${SEQ}_dist
  for CFG in ${CFGS:-s1s2 s1}; do
    od=$ROOT/${CFG}/fr1_${SEQ}; mkdir -p $od
    S2=$([ $CFG = s1 ] && echo off || echo on)
    echo "== $SEQ $CFG $(date +%H:%M)"
    PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -u -m ma_slam_semantic.run \
        --geometry ma --data $D --backend da3 --mode rgb --submap_size 32 --chunk_size 4 \
        --keyframe_disparity $KF --ckpt a1_pq --sem_lc veto --bd_s2 $S2 --sim_threshold 0.50 --max_per_submap 1 --sem_lc_mode bidesc --lc_proposal_conf gate --lc_history map $KV $EXTRA \
        --gt $D/gt_pose.txt --out $od > $od/run.log 2>&1
    ex=$?; L=$(grep "Sim3-ATE" $od/run.log | tail -1); echo "  exit=$ex $L"
    s3=$(echo "$L" | grep -oE "Sim3-ATE=[0-9.]+" | cut -d= -f2)
    se=$(echo "$L" | grep -oE "SE3-ATE=[0-9.]+" | cut -d= -f2)
    sc=$(echo "$L" | grep -oE "scale=[0-9.]+" | cut -d= -f2)
    lp=$(echo "$L" | grep -oE "loops=[0-9]+" | cut -d= -f2)
    echo "$SEQ,$CFG,$s3,$se,$sc,$lp" >> $SUM
  done
done
echo SEM_D50_DONE; cat $SUM
