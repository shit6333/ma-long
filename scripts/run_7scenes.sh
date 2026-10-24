#!/bin/bash
# 7-Scenes seq-01 x7: VGGT-SLAM 1.0 (SL4, Sim3) baselines + ours (base / +S1 / +S1+S2 V6),
# same recipe as the TUM 25 px table (KF=25) or their official keyframer (KF=50).
#   METHODS: v1_sl4 v1_sim3 base s1 s1s2      GPU=N  KF=25|50  SCENES="chess fire ..."
set -u
ABS=/mnt/HDD4/ricky/slam/ma_long; cd $ABS
GPU=${GPU:-0}; KF=${KF:-25}
METHODS=${METHODS:-"v1_sl4 v1_sim3 base s1 s1s2"}
OUT=$ABS/outputs/7scenes_kf${KF}; mkdir -p $OUT
SUM=$OUT/summary.csv; [ -f $SUM ] || echo "scene,method,sim3_ate,se3_ate,scale,n_kf,n_loops" > $SUM
PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python; E1=/mnt/HDD4/ricky/envs/vggt_slam1/bin/python
R1=$ABS/thirdparty/vggt-slam-1/VGGT-SLAM
KV="--kv_policy h2o --h2o_recent 8 --h2o_hh 8 --h2o_decay 0.9 --h2o_subsample 0.25"
V6="--sem_lc_unknown coloc --s2_t_noise on --s2_lvc_first on"
run() { # $1=scene $2=method
  local SC=$1 M=$2 od=$OUT/${1}_$2; mkdir -p $od
  local D=$ABS/data/7scenes/$SC RAW=$ABS/data/7scenes/_raw/$SC/seq-01
  echo "== $SC $M $(date +%H:%M)"
  local L=""
  case $M in
    v1_sl4|v1_sim3)
      local common="--image_folder $RAW --max_loops 1 --min_disparity $KF --conf_threshold 25 --submap_size 32 --log_results --skip_dense_log --log_path $od/poses.txt"
      [ $M = v1_sim3 ] && common="$common --use_sim3"
      (cd $R1 && CUDA_VISIBLE_DEVICES=$GPU $E1 main.py $common > $od/run.log 2>&1)
      # their frame ids = number parsed from 'frame-000123.color.png' = our frame index
      L=$(PYTHONPATH=src $PY scripts/eval_vggt_slam_log.py --log $od/poses.txt --gt $D/gt_pose.txt --out $od/eval 2>&1 | grep "Sim3-ATE")
      local nl=$(grep -oE "loop closures in map [0-9]+" $od/run.log | grep -oE "[0-9]+$")
      echo "  $L loops=$nl"
      echo "$SC,$M,$(echo "$L"|grep -oE 'Sim3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'SE3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'scale=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'n=[0-9]+'|cut -d= -f2),$nl" >> $SUM;;
    base)
      PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -m ma_slam.run --scene $D --backend da3 --mode rgb --manifold se3 \
        --submap_size 32 --keyframe_disparity $KF --sim_threshold 0.50 --max_per_submap 1 --gt $D/gt_pose.txt --out $od > $od/run.log 2>&1
      L=$(grep "Sim3-ATE" $od/run.log | tail -1); echo "  $L"
      echo "$SC,$M,$(echo "$L"|grep -oE 'Sim3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'SE3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'scale=[0-9.]+'|cut -d= -f2),,$(echo "$L"|grep -oE 'loops=[0-9]+'|cut -d= -f2)" >> $SUM;;
    s1|s1s2|s1s2_old)
      local S2=$([ $M = s1 ] && echo off || echo on)
      local RULES=$V6; [ $M = s1s2_old ] && RULES=""      # s1s2_old = pre-V6 gate rules (unknown fail, plain T_FAR)
      PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -u -m ma_slam_semantic.run --geometry ma --data $D --backend da3 --mode rgb \
        --submap_size 32 --chunk_size 4 --keyframe_disparity $KF --ckpt a1_pq --sem_lc veto --bd_s2 $S2 --sim_threshold 0.50 --max_per_submap 1 --sem_lc_mode bidesc --lc_proposal_conf gate --lc_history map $KV $RULES \
        --gt $D/gt_pose.txt --out $od > $od/run.log 2>&1
      L=$(grep "Sim3-ATE" $od/run.log | tail -1); echo "  $L"
      echo "$SC,$M,$(echo "$L"|grep -oE 'Sim3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'SE3-ATE=[0-9.]+'|cut -d= -f2),,,$(echo "$L"|grep -oE 'loops=[0-9]+'|cut -d= -f2)" >> $SUM;;
  esac
}
for SC in ${SCENES:-chess fire heads office pumpkin redkitchen stairs}; do for M in $METHODS; do run $SC $M; done; done
echo SEVEN_SCENES_RUNS_DONE; cat $SUM
