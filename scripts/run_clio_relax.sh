#!/bin/bash
# Clio, recall-first retrieval: SALAD cos >= 0.40 and up to 5 candidates per submap (defaults: 0.50 / 1),
# so the verifier actually sees the ~65 true / ~700 false pairs that exist at that level.
#   base  = coloc only | s1s2 = shipped S1+S2 rules | rigid = rigid w/ transient bags (map=gate)
#   pB/pC = VGGT-SLAM 1.0 Sim3 plug-in at L2<1.10 (cos>0.395), no gate / our map-free gate (max_loops 1)
set -u
ABS=/mnt/HDD4/ricky/slam/ma_long; cd $ABS
GPU=${GPU:-0}; KF=${KF:-50}; SIM=${SIM:-0.40}; MPS=${MPS:-5}
METHODS=${METHODS:-"base s1s2 rigid pB pC"}; SCENES=${SCENES:-"apartment"}
OUT=$ABS/outputs/clio_kf${KF}_relax; mkdir -p $OUT
SUM=$OUT/summary.csv; [ -f $SUM ] || echo "scene,method,sim3_ate,se3_ate,n_loops,sem_n,sem_acc" > $SUM
PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python
KV="--kv_policy h2o --h2o_recent 8 --h2o_hh 8 --h2o_decay 0.9 --h2o_subsample 0.25"
for SC in $SCENES; do D=$ABS/data/clio/$SC; for M in $METHODS; do
  od=$OUT/${SC}_$M; mkdir -p $od; grep -q "^$SC,$M," $SUM && { echo "== $SC $M (done)"; continue; }
  echo "== $SC $M $(date +%H:%M)"; L=""; nl=""; sem=","
  case $M in
    base) PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -m ma_slam.run --scene $D --backend da3 --mode rgb --manifold se3 --submap_size 32 --keyframe_disparity $KF --sim_threshold $SIM --max_per_submap $MPS --gt $D/gt_pose.txt --out $od > $od/run.log 2>&1
          L=$(grep "Sim3-ATE" $od/run.log | tail -1); nl=$(echo "$L"|grep -oE 'loops=[0-9]+'|cut -d= -f2);;
    s1s2|rigid)
          X="--sem_lc veto --bd_s2 on"; [ $M = rigid ] && X="--sem_lc veto --sem_lc_mode rigid --rigid_lowobj reject --proposal_conf gate --lc_proposal_conf prefer --lc_history bag"
          PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -u -m ma_slam_semantic.run --geometry ma --data $D --backend da3 --mode rgb --submap_size 32 --chunk_size 4 --keyframe_disparity $KF --sim_threshold $SIM --max_per_submap $MPS --ckpt a1_pq --sem_lc_mode bidesc --lc_proposal_conf gate --lc_history map --rigid_geom on $X --sem_lc_dump $KV --gt $D/gt_pose.txt --out $od > $od/run.log 2>&1
          L=$(grep "Sim3-ATE" $od/run.log | tail -1); nl=$(echo "$L"|grep -oE 'loops=[0-9]+'|cut -d= -f2);;
    pB|pC) flags="--lc_thres 1.10"; [ $M = pC ] && flags="--lc_thres 1.10 --sem_lc --lc_dump $od/semlc_dump --semlc_log $od/semlc_log.txt"; rm -f $od/semlc_log.txt
          https_proxy=http://127.0.0.1:9 http_proxy=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY scripts/vggt1_lc/main_gated.py $flags -- --image_folder $D/rgb --use_sim3 --submap_size 32 --min_disparity $KF --max_loops 1 --log_results --skip_dense_log --log_path $od/poses.txt > $od/run.log 2>&1
          L=$(PYTHONPATH=src $PY scripts/eval_vggt_slam_log.py --log $od/poses.txt --gt $D/gt_pose.txt --out $od/eval 2>&1 | grep "Sim3-ATE"); nl=$(grep -oE "loop closures in map [0-9]+" $od/run.log | grep -oE "[0-9]+$")
          [ -f $od/semlc_log.txt ] && sem="$(grep -c '^\[semlc\] [AR]' $od/semlc_log.txt),$(grep -c '^\[semlc\] ACCEPT' $od/semlc_log.txt)";;
  esac
  echo "  $L loops=$nl sem=$sem"
  echo "$SC,$M,$(echo "$L"|grep -oE 'Sim3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'SE3-ATE=[0-9.]+'|cut -d= -f2),$nl,$sem" >> $SUM
done; done
echo CLIO_RELAX_DONE
