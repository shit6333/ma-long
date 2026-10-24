#!/bin/bash
# Reproduce VGGT-SLAM 1.0 (SL4 + Sim3) and 2.0 with THEIR official TUM protocol
# (evals/eval_tum.sh): raw TUM rgb folders (timestamp filenames), --max_loops 1
# --min_disparity 50 --conf_threshold 25 --submap_size 32, evaluated with
# `evo_ape tum groundtruth.txt est -as` (Sim3 alignment incl. scale, RMSE).
# Paper (1.0 Table 2, w=32): Sim3 avg 0.075, SL4 avg 0.053. n=1 run here (paper: 5).
set -u
ABS=/mnt/HDD4/ricky/slam/ma_long; cd $ABS
GPU=${GPU:-0}
METHODS=${METHODS:-"v1_sl4 v1_sim3 v2"}
DISP=${DISP:-50}
OUT=$ABS/outputs/vggt_slam_official_d${DISP}; mkdir -p $OUT
SUM=$OUT/summary.csv; [ -f $SUM ] || echo "seq,method,evo_rmse,n_submaps,n_loops" > $SUM
E1=/mnt/HDD4/ricky/envs/vggt_slam1/bin/python; E2=/mnt/HDD4/ricky/envs/vggt_slam2/bin/python
EVO=/mnt/HDD4/ricky/envs/amb3r_bw/bin/evo_ape
R1=$ABS/thirdparty/vggt-slam-1/VGGT-SLAM; R2=$ABS/thirdparty/VGGT-SLAM
run() { # $1=seq $2=method
  local SEQ=$1 M=$2 od=$OUT/${1}_$2; mkdir -p $od
  local D=$ABS/data/tum/_raw/rgbd_dataset_freiburg1_$SEQ
  [ -d $D/rgb ] || { echo "  $SEQ raw missing"; return; }
  local common="--image_folder $D/rgb --max_loops 1 --min_disparity $DISP --conf_threshold 25 --submap_size 32 --log_results --skip_dense_log --log_path $od/poses.txt"
  echo "== $SEQ $M $(date +%H:%M)"
  case $M in
    v2)      (cd $R2 && CUDA_VISIBLE_DEVICES=$GPU $E2 main.py $common > $od/run.log 2>&1);;
    v1_sl4)  (cd $R1 && CUDA_VISIBLE_DEVICES=$GPU $E1 main.py $common > $od/run.log 2>&1);;
    v1_sim3) (cd $R1 && CUDA_VISIBLE_DEVICES=$GPU $E1 main.py $common --use_sim3 > $od/run.log 2>&1);;
  esac
  local ex=$?
  local nsub=$(grep -oE "submaps in map [0-9]+" $od/run.log | grep -oE "[0-9]+$")
  local nloop=$(grep -oE "loop closures in map [0-9]+" $od/run.log | grep -oE "[0-9]+$")
  local r=""
  if [ -s $od/poses.txt ]; then
    r=$($EVO tum $D/groundtruth.txt $od/poses.txt -as 2>&1 | tee $od/evo.txt | grep -E "^\s+rmse" | awk '{print $2}')
  fi
  echo "  exit=$ex submaps=$nsub loops=$nloop evo_rmse=$r"
  echo "$SEQ,$M,$r,$nsub,$nloop" >> $SUM
}
for SEQ in ${SEQS:-360 desk desk2 floor plant room rpy teddy xyz}; do for M in $METHODS; do run $SEQ $M; done; done
echo OFFICIAL_REPRO_DONE; cat $SUM
