#!/bin/bash
# Paper/demo videos: re-run selected scenes with dense --snapshot records, then render per-frame
# BEV + chase videos (rgb / inst / mix) with scripts/render_stream_video.py.
#   QUEUE="name name ..." GPU=0 bash scripts/run_example_videos.sh
# Job = name|data|keyframe_px|submap|arm(idlv|noidlv)|snapshot_pts|retrieval-extra
#   semantic datasets: run_sem_eval.sh settings (25 px, submap 20, conf_gate_coef 1.0, relaxed SALAD)
#   TUM / Clio:         geometry-benchmark settings (50 px, submap 32, relaxed SALAD)
set -u
ABS=/mnt/HDD4/ricky/slam/ma_long; cd $ABS
PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python; GPU=${GPU:-0}
ROOT=$ABS/outputs/videos; mkdir -p $ROOT; SUM=$ROOT/summary.csv
[ -f $SUM ] || echo "name,arm,sim3_ate,se3_ate,loops,n_kf" > $SUM
KV="--kv_policy h2o --h2o_recent 8 --h2o_hh 8 --h2o_decay 0.9 --h2o_subsample 0.25"
REL="--sim_threshold 0.40 --max_per_submap 5"
SEM="--conf_gate_coef 1.0"
declare -A J=(
  [spp_45b0dac5e3]="data/scannetpp/45b0dac5e3|25|20|idlv|140000|$SEM"
  [spp_f9f95681fd]="data/scannetpp/f9f95681fd|25|20|idlv|125000|$SEM"
  [scenenn_093]="data/scenenn/093|25|20|idlv|260000|$SEM"
  [replica_room0]="data/replica_full/room0|25|20|idlv|220000|$SEM"
  [scannet_0426_00]="data/scannet_val50/scene0426_00|25|20|idlv|220000|$SEM"
  [tum_room_idlv]="data/tum/fr1_room_dist|50|32|idlv|400000|"
  [tum_room_noidlv]="data/tum/fr1_room_dist|50|32|noidlv|400000|"
  [clio_apartment_idlv]="data/clio/apartment|50|32|idlv|300000|"
  [clio_apartment_noidlv]="data/clio/apartment|50|32|noidlv|300000|"
)
for N in $QUEUE; do
  IFS='|' read -r D KF SM ARM PTS X <<< "${J[$N]}"
  od=$ROOT/$N; mkdir -p $od
  case $ARM in
    idlv)   LC="--sem_lc veto --sem_lc_mode rigid --rigid_geom off --rigid_lowobj reject --proposal_conf gate --lc_proposal_conf prefer --lc_history bag";;
    noidlv) LC="--sem_lc off --proposal_conf gate";;
  esac
  echo "== $N ($ARM) GPU$GPU $(date +%H:%M)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -u -m ma_slam_semantic.run \
    --geometry ma --data $D --backend da3 --mode rgb --submap_size $SM --chunk_size 4 --keyframe_disparity $KF \
    --ckpt a1_pq $LC $REL $X $KV --snapshot --snapshot_pts $PTS --gt $D/gt_pose.txt --out $od > $od/run.log 2>&1
  L=$(grep "Sim3-ATE" $od/run.log | tail -1); nk=$(grep -m1 -oE "kept [0-9]+" $od/run.log | grep -oE "[0-9]+")
  echo "   $L"
  echo "$N,$ARM,$(echo "$L"|grep -oE 'Sim3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'SE3-ATE=[0-9.]+'|cut -d= -f2),$(echo "$L"|grep -oE 'loops=[0-9]+'|cut -d= -f2),$nk" >> $SUM
  # render on CPU in the background so the GPU queue moves on
  nohup env PYTHONPATH=src $PY scripts/render_stream_video.py --run $od --images $D/rgb > $od/render.log 2>&1 &
done
wait
echo "QUEUE_DONE $QUEUE"
