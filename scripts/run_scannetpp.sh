#!/bin/bash
# Ours on the ScanNet++ eval set (50 scenes, iPhone stride-20 frames, COLMAP GT):
#   base = DA3-rgb, no inst-LC | s1 = +S1 only | s1s2 = +S1+S2 (OLD gate rules = the method)
#   GPU=N KF=25 METHODS="base s1 s1s2" SCENES="id1 id2 ..." (default: all of my_arrow_val_50.txt)
set -u
ABS=/mnt/HDD4/ricky/slam/ma_long; cd $ABS
GPU=${GPU:-0}; KF=${KF:-25}
METHODS=${METHODS:-"base s1 s1s2"}
LIST=/mnt/HDD4/ricky/data/scannetpp_eval/my_arrow_val_50.txt
SCENES=${SCENES:-$(cat $LIST | tr '\n' ' ')}
OUT=$ABS/outputs/scannetpp_kf${KF}; mkdir -p $OUT
SUM=$OUT/summary.csv; [ -f $SUM ] || echo "scene,method,sim3_ate,se3_ate,scale,n_loops" > $SUM
PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python
KV="--kv_policy h2o --h2o_recent 8 --h2o_hh 8 --h2o_decay 0.9 --h2o_subsample 0.25"
run() { # $1=scene $2=method
  local SC=$1 M=$2 od=$OUT/${1}_$2; mkdir -p $od
  local D=$ABS/data/scannetpp/$SC
  if grep -q "^$SC,$M," $SUM; then echo "== $SC $M (done, skip)"; return; fi
  echo "== $SC $M $(date +%H:%M)"
  local L=""
  case $M in
    base)
      PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -m ma_slam.run --scene $D --backend da3 --mode rgb --manifold se3 \
        --submap_size 32 --keyframe_disparity $KF --sim_threshold 0.50 --max_per_submap 1 --gt $D/gt_pose.txt --out $od > $od/run.log 2>&1
      L=$(grep "Sim3-ATE" $od/run.log | tail -1); echo "  $L"
      echo "$SC,$M,$(echo "$L"|grep -oE 'Sim3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'SE3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'scale=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'loops=[0-9]+'|cut -d= -f2)" >> $SUM;;
    s1|s1s2)
      local S2=$([ $M = s1 ] && echo off || echo on)
      PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -u -m ma_slam_semantic.run --geometry ma --data $D --backend da3 --mode rgb \
        --submap_size 32 --chunk_size 4 --keyframe_disparity $KF --ckpt a1_pq --sem_lc veto --bd_s2 $S2 --sim_threshold 0.50 --max_per_submap 1 --sem_lc_mode bidesc --lc_proposal_conf gate --lc_history map $KV \
        --gt $D/gt_pose.txt --out $od > $od/run.log 2>&1
      L=$(grep "Sim3-ATE" $od/run.log | tail -1); echo "  $L"
      echo "$SC,$M,$(echo "$L"|grep -oE 'Sim3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'SE3-ATE=[0-9.]+'|cut -d= -f2),,$(echo "$L"|grep -oE 'loops=[0-9]+'|cut -d= -f2)" >> $SUM;;
  esac
}
for SC in $SCENES; do for M in $METHODS; do run $SC $M; done; done
echo SCANNETPP_RUNS_DONE
