#!/bin/bash
# Our semantic LC plugged into VGGT-SLAM 1.0 (Sim3): three arms per scene.
#   A = their default (SALAD L2 < 0.80, no verifier)
#   B = relaxed retrieval (0.95), no verifier   -> shows what relaxing alone does
#   C = relaxed retrieval (0.95) + our S1+S2 gate
# Same settings as the RESULTS.md re-run: their keyframer 25 px, submap 32, max_loops 1.
set -u
ABS=/mnt/HDD4/ricky/slam/ma_long; cd $ABS
GPU=${GPU:-0}
PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python
OUT=$ABS/outputs/vggt1_lc; mkdir -p $OUT
SUM=$OUT/summary.csv; [ -f $SUM ] || echo "scene,arm,sim3_ate,se3_ate,scale,n_kf,n_submaps,n_loops,sem_n,sem_acc" > $SUM
ARMS=${ARMS:-"A B C"}
SCENES=${SCENES:-"tum:xyz tum:360 tum:floor tum:teddy tum:room scannet:scene0231_00 office:office_loop"}
run() { # $1=scene-spec $2=arm
  local spec=$1 ARM=$2 kind=${1%%:*} name=${1##*:}
  local IMG GT
  case $kind in
    tum)     IMG=$ABS/data/tum/fr1_${name}_dist/rgb; GT=$ABS/data/tum/fr1_${name}_dist/gt_pose.txt;;
    scannet) IMG=$ABS/data/$name/rgb; GT=$ABS/data/$name/gt_pose.txt;;
    office)  IMG=$ABS/data/$name; GT="";;
  esac
  local od=$OUT/${name}_$ARM; mkdir -p $od; rm -f $od/semlc_log.txt
  local flags
  case $ARM in
    A) flags="--lc_thres 0.80";;
    B) flags="--lc_thres 0.95";;
    C) flags="--lc_thres 0.95 --sem_lc --lc_dump $od/semlc_dump --semlc_log $od/semlc_log.txt";;
  esac
  echo "== $name $ARM $(date +%H:%M)"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY scripts/vggt1_lc/main_gated.py $flags -- \
    --image_folder $IMG --use_sim3 --submap_size 32 --min_disparity 25 --max_loops 1 \
    --log_results --skip_dense_log --log_path $od/poses.txt > $od/run.log 2>&1
  local ex=$?
  local nsub=$(grep -oE "submaps in map [0-9]+" $od/run.log | grep -oE "[0-9]+$")
  local nloop=$(grep -oE "loop closures in map [0-9]+" $od/run.log | grep -oE "[0-9]+$")
  local semn="" sema=""
  if [ -f $od/semlc_log.txt ]; then semn=$(grep -c "^\[semlc\] [AR]" $od/semlc_log.txt); sema=$(grep -c "^\[semlc\] ACCEPT" $od/semlc_log.txt); fi
  local L=""
  if [ -n "$GT" ] && [ -s $od/poses.txt ]; then
    L=$(PYTHONPATH=src $PY scripts/eval_vggt_slam_log.py --log $od/poses.txt --gt $GT --out $od/eval 2>&1 | grep "Sim3-ATE")
  fi
  echo "  exit=$ex submaps=$nsub loops=$nloop sem=$sema/$semn $L"
  echo "$name,$ARM,$(echo "$L"|grep -oE 'Sim3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'SE3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'scale=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'n=[0-9]+'|cut -d= -f2),$nsub,$nloop,$semn,$sema" >> $SUM
}
for spec in $SCENES; do for ARM in $ARMS; do run $spec $ARM; done; done
echo VGGT1_LC_ARMS_DONE; cat $SUM
