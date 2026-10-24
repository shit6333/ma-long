#!/bin/bash
# Re-render all example videos with final-instance colours + ceiling removal (video_final/) and the
# objects-only variant with wall/floor/ceiling in image colour (video_final_objects/). CPU only.
set -u
cd /mnt/HDD4/ricky/slam/ma_long
PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python
declare -A IMG=(
  [outputs/videos/spp_45b0dac5e3]=data/scannetpp/45b0dac5e3/rgb
  [outputs/videos/spp_f9f95681fd]=data/scannetpp/f9f95681fd/rgb
  [outputs/videos/scenenn_093]=data/scenenn/093/rgb
  [outputs/videos/replica_room0]=data/replica_full/room0/rgb
  [outputs/videos/scannet_0426_00]=data/scannet_val50/scene0426_00/rgb
  [outputs/videos/tum_room_idlv]=data/tum/fr1_room_dist/rgb
  [outputs/videos/tum_room_noidlv]=data/tum/fr1_room_dist/rgb
  [outputs/videos/clio_apartment_idlv]=data/clio/apartment/rgb
  [outputs/videos/clio_apartment_noidlv]=data/clio/apartment/rgb
  [outputs/clio_kf50_v2/cubicle_idlv_video]=data/clio/cubicle/rgb
)
for R in ${RUNS:-${!IMG[@]}}; do
  CEIL=auto; case $R in *clio_apartment*) CEIL=off;; esac   # no ceiling scanned; the broken w/o map fakes one
  nohup env PYTHONPATH=src $PY scripts/render_stream_video.py --run $R --images ${IMG[$R]} --ceiling $CEIL \
      --outdir $R/snapshots/video_final --nostruct_dir $R/snapshots/video_final_objects > $R/render_final.log 2>&1 &
done
wait; echo RENDER_FINAL_DONE
