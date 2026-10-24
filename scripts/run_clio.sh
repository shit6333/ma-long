#!/bin/bash
# Clio scenes (multi-room, RealSense D455, COLMAP-metric GT via scripts/prep_clio.py):
# replicate VGGT-SLAM 2.0 Table II + our pipeline. Keyframer 50 px (their protocol), w32.
#   METHODS: v1_sl4 v1_sim3            VGGT-SLAM 1.0 (their code / env), SALAD 0.80, no verification
#            pA pB pC                   1.0 Sim3 plug-in arms: 0.80 | 0.95 no gate | 0.95 + our S1+S2 (map-free rigid)
#            base s1 s1s2 rigid         ours: no inst-LC | +S1 | +S1+S2 (shipped rules) | rigid w/ bags (map=gate)
set -u
ABS=/mnt/HDD4/ricky/slam/ma_long; cd $ABS
GPU=${GPU:-0}; KF=${KF:-50}
METHODS=${METHODS:-"v1_sl4 v1_sim3 base s1 s1s2 rigid pA pB pC"}
SCENES=${SCENES:-"apartment office cubicle building"}
OUT=$ABS/outputs/clio_kf${KF}; mkdir -p $OUT
SUM=$OUT/summary.csv; [ -f $SUM ] || echo "scene,method,sim3_ate,se3_ate,scale,n_kf,n_loops,sem_n,sem_acc" > $SUM
PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python; E1=/mnt/HDD4/ricky/envs/vggt_slam1/bin/python
R1=$ABS/thirdparty/vggt-slam-1/VGGT-SLAM
KV="--kv_policy h2o --h2o_recent 8 --h2o_hh 8 --h2o_decay 0.9 --h2o_subsample 0.25"
evalrow() { # $1=scene $2=method $3=od $4=log-line $5=extra(n_loops,sem_n,sem_acc)
  echo "$1,$2,$(echo "$4"|grep -oE 'Sim3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$4"|grep -oE 'SE3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$4"|grep -oE 'scale=[0-9.]+'|cut -d= -f2),$(echo "$4"|grep -oE 'n=[0-9]+'|cut -d= -f2),$5" >> $SUM
}
run() {
  local SC=$1 M=$2 od=$OUT/${1}_$2; mkdir -p $od
  local D=$ABS/data/clio/$SC RAW=$ABS/data/clio/$SC/rgb    # frame_XXXXX.jpg: index == gt_pose row
  if grep -q "^$SC,$M," $SUM; then echo "== $SC $M (done, skip)"; return; fi
  echo "== $SC $M $(date +%H:%M)"
  local L="" nl="" sem=""
  case $M in
    v1_sl4|v1_sim3)
      local common="--image_folder $RAW --max_loops 1 --min_disparity $KF --conf_threshold 25 --submap_size 32 --log_results --skip_dense_log --log_path $od/poses.txt"
      [ $M = v1_sim3 ] && common="$common --use_sim3"
      # GitHub unreachable on purpose: torch.hub's validation of serizba/salad + dinov2 otherwise blocks for hours
      https_proxy=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 \
        CUDA_VISIBLE_DEVICES=$GPU $E1 $ABS/scripts/vggt1_lc/run_v1_nohub.py $R1/main.py $common > $od/run.log 2>&1
      L=$(PYTHONPATH=src $PY scripts/eval_vggt_slam_log.py --log $od/poses.txt --gt $D/gt_pose.txt --out $od/eval 2>&1 | grep "Sim3-ATE")
      nl=$(grep -oE "loop closures in map [0-9]+" $od/run.log | grep -oE "[0-9]+$"); echo "  $L loops=$nl"; evalrow $SC $M $od "$L" "$nl,,";;
    pA|pB|pC)
      local flags="--lc_thres 0.80"; [ $M = pB ] && flags="--lc_thres 0.95"; [ $M = pC ] && flags="--lc_thres 0.95 --sem_lc --lc_dump $od/semlc_dump --semlc_log $od/semlc_log.txt"
      rm -f $od/semlc_log.txt
      https_proxy=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY scripts/vggt1_lc/main_gated.py $flags -- --image_folder $RAW --use_sim3 --submap_size 32 --min_disparity $KF --max_loops 1 --log_results --skip_dense_log --log_path $od/poses.txt > $od/run.log 2>&1
      L=$(PYTHONPATH=src $PY scripts/eval_vggt_slam_log.py --log $od/poses.txt --gt $D/gt_pose.txt --out $od/eval 2>&1 | grep "Sim3-ATE")
      nl=$(grep -oE "loop closures in map [0-9]+" $od/run.log | grep -oE "[0-9]+$")
      [ -f $od/semlc_log.txt ] && sem="$(grep -c '^\[semlc\] [AR]' $od/semlc_log.txt),$(grep -c '^\[semlc\] ACCEPT' $od/semlc_log.txt)" || sem=","
      echo "  $L loops=$nl sem=$sem"; evalrow $SC $M $od "$L" "$nl,$sem";;
    base)
      PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -m ma_slam.run --scene $D --backend da3 --mode rgb --manifold se3 --submap_size 32 --keyframe_disparity $KF --sim_threshold 0.50 --max_per_submap 1 --gt $D/gt_pose.txt --out $od > $od/run.log 2>&1
      L=$(grep "Sim3-ATE" $od/run.log | tail -1); echo "  $L"; evalrow $SC $M $od "$L" "$(echo "$L"|grep -oE 'loops=[0-9]+'|cut -d= -f2),,";;
    s1|s1s2|rigid)
      local X="--sem_lc veto --bd_s2 on"; [ $M = s1 ] && X="--sem_lc veto --bd_s2 off"
      [ $M = rigid ] && X="--sem_lc veto --sem_lc_mode rigid --rigid_lowobj reject --proposal_conf gate --lc_proposal_conf prefer --lc_history bag"
      PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -u -m ma_slam_semantic.run --geometry ma --data $D --backend da3 --mode rgb --submap_size 32 --chunk_size 4 --keyframe_disparity $KF --ckpt a1_pq --sim_threshold 0.50 --max_per_submap 1 --sem_lc_mode bidesc --lc_proposal_conf gate --lc_history map --rigid_geom on $X --sem_lc_dump $KV --gt $D/gt_pose.txt --out $od > $od/run.log 2>&1
      L=$(grep "Sim3-ATE" $od/run.log | tail -1); echo "  $L"; evalrow $SC $M $od "$L" "$(echo "$L"|grep -oE 'loops=[0-9]+'|cut -d= -f2),,";;
  esac
}
for SC in $SCENES; do for M in $METHODS; do run $SC $M; done; done
echo CLIO_RUNS_DONE
