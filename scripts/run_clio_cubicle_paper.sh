#!/bin/bash
# Paper example on Clio cubicle (default SALAD retrieval, KF 50, submap 32, new 2026-09-04 assoc config):
#   ARM=idlv   final method (S1 instance-correspondence gate + standard coloc)
#   ARM=noidlv --sem_lc off (SALAD + coloc only) but WITH the instance map, for side-by-side seg figures
# Both write --snapshot for the streaming reconstruction video (scripts/render_snapshots.py).
set -u
ABS=/mnt/HDD4/ricky/slam/ma_long; cd $ABS
GPU=${GPU:-0}; ARM=${ARM:?idlv|noidlv}; KF=50
PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python; D=$ABS/data/clio/cubicle
OUT=$ABS/outputs/clio_kf50_v2/cubicle_$ARM${SUFFIX:-}; mkdir -p $OUT
KV="--kv_policy h2o --h2o_recent 8 --h2o_hh 8 --h2o_decay 0.9 --h2o_subsample 0.25"
case $ARM in
  idlv)   LC="--sem_lc veto --sem_lc_mode rigid --rigid_geom off --rigid_lowobj reject --proposal_conf gate --lc_proposal_conf prefer --lc_history bag --sem_lc_dump";;
  noidlv) LC="--sem_lc off --proposal_conf gate";;
esac
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -u -m ma_slam_semantic.run \
  --geometry ma --data $D --backend da3 --mode rgb --submap_size 32 --chunk_size 4 --keyframe_disparity $KF --sim_threshold 0.50 --max_per_submap 1 \
  --ckpt a1_pq $LC $KV --snapshot --snapshot_pts ${SNAP_PTS:-50000} --gt $D/gt_pose.txt --out $OUT > $OUT/run.log 2>&1
grep "Sim3-ATE" $OUT/run.log | tail -1
echo CUBICLE_${ARM}_DONE
