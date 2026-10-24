#!/bin/bash
# VGGT-SLAM 1.0 (SL4 / Sim3) and 2.0 on the SAME input as ours: raw distorted TUM fr1
# frames, their own keyframer at 25 px (paper value; code default is 50), submap 32,
# evaluated with ma_slam's eval.py (Sim3 + SE3 ATE) via scripts/eval_vggt_slam_log.py.
set -u
ABS=/mnt/HDD4/ricky/slam/ma_long; cd $ABS
GPU=${GPU:-0}
OUT=$ABS/outputs/vggt_slam_tum; mkdir -p $OUT
SUM=$OUT/summary.csv; echo "seq,method,sim3_ate,se3_ate,scale,n_kf" > $SUM
E1=/mnt/HDD4/ricky/envs/vggt_slam1/bin/python; E2=/mnt/HDD4/ricky/envs/vggt_slam2/bin/python
R1=$ABS/thirdparty/vggt-slam-1/VGGT-SLAM; R2=$ABS/thirdparty/VGGT-SLAM
run() { # $1=seq $2=method(v2|v1_sl4|v1_sim3)
  local SEQ=$1 M=$2 od=$OUT/${1}_$2; mkdir -p $od
  local IMG=$ABS/data/tum/fr1_${SEQ}_dist/rgb
  local common="--image_folder $IMG --submap_size 32 --min_disparity 25 --log_results --skip_dense_log --log_path $od/poses.txt"
  echo "== $SEQ $M $(date +%H:%M)"
  case $M in
    v2)      (cd $R2 && CUDA_VISIBLE_DEVICES=$GPU $E2 main.py $common > $od/run.log 2>&1);;
    v1_sl4)  (cd $R1 && CUDA_VISIBLE_DEVICES=$GPU $E1 main.py $common > $od/run.log 2>&1);;
    v1_sim3) (cd $R1 && CUDA_VISIBLE_DEVICES=$GPU $E1 main.py $common --use_sim3 > $od/run.log 2>&1);;
  esac
  local ex=$?
  if [ -s $od/poses.txt ]; then
    L=$(PYTHONPATH=src /mnt/HDD4/ricky/envs/amb3r_bw/bin/python scripts/eval_vggt_slam_log.py --log $od/poses.txt --gt data/tum/fr1_${SEQ}_dist/gt_pose.txt --out $od/eval 2>&1 | grep "Sim3-ATE")
    echo "  exit=$ex $L"
    echo "$SEQ,$M,$(echo "$L"|grep -oE 'Sim3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'SE3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'scale=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'n=[0-9]+'|cut -d= -f2)" >> $SUM
  else
    echo "  exit=$ex NO POSES ($(grep -iE 'error|Traceback' $od/run.log | tail -1 | cut -c1-80))"; echo "$SEQ,$M,,,," >> $SUM
  fi
}
for SEQ in 360 desk desk2 floor plant room rpy teddy xyz; do
  for M in v2 v1_sl4 v1_sim3; do run $SEQ $M; done
done
echo VGGT_SLAM_TUM_DONE; cat $SUM
