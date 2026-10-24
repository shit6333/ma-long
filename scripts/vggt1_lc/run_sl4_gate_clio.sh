#!/bin/bash
# VGGT-SLAM 1.0 SL(4) (their env) + OUR S1 gate via RPC on Clio (data/clio/<scene>, 50 px, COLMAP-metric GT).
#   METHODS: sl4_base80 | sl4_gate80 | sl4_relax110 (L2 1.10, no gate) | sl4_relax110_gate
set -u
ABS=/mnt/HDD4/ricky/slam/ma_long; cd $ABS
GPU=${GPU:-0}; KF=${KF:-50}; MODE=${MODE:-pairs}; RELAX=${RELAX:-1.10}
METHODS=${METHODS:-"sl4_base80 sl4_gate80 sl4_relax110 sl4_relax110_gate"}
SCENES=${SCENES:-"apartment office cubicle"}
OUT=${OUT:-$ABS/outputs/vggt1_lc_sl4_clio_kf${KF}}; mkdir -p $OUT
SUM=$OUT/summary.csv; [ -f $SUM ] || echo "scene,method,sim3_ate,se3_ate,scale,n_submaps,n_loops,sem_n,sem_acc" > $SUM
E1=/mnt/HDD4/ricky/envs/vggt_slam1/bin/python; PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python
SOCK=/tmp/semlc_clio_$$.sock
PYTHONPATH=src:scripts CUDA_VISIBLE_DEVICES=$GPU $PY scripts/vggt1_lc/verify_server.py --socket $SOCK --mode $MODE > $OUT/server.log 2>&1 &
SRV=$!; trap "kill $SRV 2>/dev/null" EXIT
until grep -q "ready on" $OUT/server.log; do sleep 2; kill -0 $SRV 2>/dev/null || { echo "server died"; tail -20 $OUT/server.log; exit 1; }; done
NOGIT="https_proxy=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9"
run() { # $1=scene $2=method
  local SC=$1 M=$2 od=$OUT/${1}_$2; mkdir -p $od
  if grep -q "^$SC,$M," $SUM; then echo "== $SC $M (done, skip)"; return; fi
  local D=$ABS/data/clio/$SC
  local common="--image_folder $D/rgb --max_loops 1 --min_disparity $KF --conf_threshold 25 --submap_size 32 --log_results --skip_dense_log --log_path $od/poses.txt"
  local flags="--lc_thres 0.80 --sem_lc"; [ $M = sl4_base80 ] && flags="--lc_thres 0.80"; [ $M = sl4_relax110 ] && flags="--lc_thres $RELAX"; [ $M = sl4_relax110_gate ] && flags="--lc_thres $RELAX --sem_lc"
  rm -f $od/semlc_log.txt
  echo "== $SC $M $(date +%H:%M)"
  env $NOGIT CUDA_VISIBLE_DEVICES=$GPU $E1 scripts/vggt1_lc/main_gated_rpc.py $flags --socket $SOCK --semlc_log $od/semlc_log.txt -- $common > $od/run.log 2>&1
  local ex=$?
  local nsub=$(grep -oE "submaps in map [0-9]+" $od/run.log | grep -oE "[0-9]+$")
  local nloop=$(grep -oE "loop closures in map [0-9]+" $od/run.log | grep -oE "[0-9]+$")
  local L=$(PYTHONPATH=src $PY scripts/eval_vggt_slam_log.py --log $od/poses.txt --gt $D/gt_pose.txt --out $od/eval 2>&1 | grep "Sim3-ATE")
  local sem=","; [ -f $od/semlc_log.txt ] && sem="$(grep -c '^\[semlc\] [AR]' $od/semlc_log.txt),$(grep -c '^\[semlc\] ACCEPT' $od/semlc_log.txt)"
  echo "  exit=$ex submaps=$nsub loops=$nloop $L sem=$sem"
  echo "$SC,$M,$(echo "$L"|grep -oE 'Sim3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'SE3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'scale=[0-9.]+'|cut -d= -f2),$nsub,$nloop,$sem" >> $SUM
}
for SC in $SCENES; do for M in $METHODS; do run $SC $M; done; done
echo SL4_CLIO_DONE; cat $SUM
