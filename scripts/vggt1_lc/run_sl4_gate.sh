#!/bin/bash
# VGGT-SLAM 1.0 SL(4) (their env, official TUM protocol: raw folders, 50 px, evo -as) with OUR S1 gate via RPC.
#   METHODS: sl4_base80 (their SL4, same session) | sl4_gate80 (their 0.80 + S1) | sl4_relax95 (0.95, no gate) | sl4_relax95_gate (0.95 + S1)
set -u
ABS=/mnt/HDD4/ricky/slam/ma_long; cd $ABS
GPU=${GPU:-0}; DISP=${DISP:-50}; MODE=${MODE:-pairs}
METHODS=${METHODS:-"sl4_gate80 sl4_relax95 sl4_relax95_gate"}
OUT=${OUT:-$ABS/outputs/vggt1_lc_sl4_d${DISP}}; mkdir -p $OUT
SUM=$OUT/summary.csv; [ -f $SUM ] || echo "seq,method,evo_rmse,n_submaps,n_loops,sem_n,sem_acc" > $SUM
E1=/mnt/HDD4/ricky/envs/vggt_slam1/bin/python; PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python
EVO=/mnt/HDD4/ricky/envs/amb3r_bw/bin/evo_ape
SOCK=/tmp/semlc_$$.sock
PYTHONPATH=src:scripts CUDA_VISIBLE_DEVICES=$GPU $PY scripts/vggt1_lc/verify_server.py --socket $SOCK --mode $MODE > $OUT/server.log 2>&1 &
SRV=$!; trap "kill $SRV 2>/dev/null" EXIT
until grep -q "ready on" $OUT/server.log; do sleep 2; kill -0 $SRV 2>/dev/null || { echo "server died"; cat $OUT/server.log | tail -20; exit 1; }; done
NOGIT="https_proxy=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9"
run() { # $1=seq $2=method
  local SEQ=$1 M=$2 od=$OUT/${1}_$2; mkdir -p $od
  if grep -q "^$SEQ,$M," $SUM; then echo "== $SEQ $M (done, skip)"; return; fi
  local D=$ABS/data/tum/_raw/rgbd_dataset_freiburg1_$SEQ
  local common="--image_folder $D/rgb --max_loops 1 --min_disparity $DISP --conf_threshold 25 --submap_size 32 --log_results --skip_dense_log --log_path $od/poses.txt"
  local flags="--lc_thres 0.80 --sem_lc"; [ $M = sl4_base80 ] && flags="--lc_thres 0.80"; [ $M = sl4_relax95 ] && flags="--lc_thres 0.95"; [ $M = sl4_relax95_gate ] && flags="--lc_thres 0.95 --sem_lc"
  rm -f $od/semlc_log.txt
  echo "== $SEQ $M $(date +%H:%M)"
  env $NOGIT CUDA_VISIBLE_DEVICES=$GPU $E1 scripts/vggt1_lc/main_gated_rpc.py $flags --socket $SOCK --semlc_log $od/semlc_log.txt -- $common > $od/run.log 2>&1
  local ex=$?
  local nsub=$(grep -oE "submaps in map [0-9]+" $od/run.log | grep -oE "[0-9]+$")
  local nloop=$(grep -oE "loop closures in map [0-9]+" $od/run.log | grep -oE "[0-9]+$")
  local r=""; [ -s $od/poses.txt ] && r=$($EVO tum $D/groundtruth.txt $od/poses.txt -as 2>&1 | tee $od/evo.txt | grep -E "^\s+rmse" | awk '{print $2}')
  local sem=","; [ -f $od/semlc_log.txt ] && sem="$(grep -c '^\[semlc\] [AR]' $od/semlc_log.txt),$(grep -c '^\[semlc\] ACCEPT' $od/semlc_log.txt)"
  echo "  exit=$ex submaps=$nsub loops=$nloop evo_rmse=$r sem=$sem"
  echo "$SEQ,$M,$r,$nsub,$nloop,$sem" >> $SUM
}
for SEQ in ${SEQS:-360 desk desk2 floor plant room rpy teddy xyz}; do for M in $METHODS; do run $SEQ $M; done; done
echo SL4_GATE_DONE; cat $SUM
