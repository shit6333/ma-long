#!/bin/bash
# 5-scene ScanNet benchmark of ma_slam_semantic (--geometry ma, DA3 rgb geometry,
# submap 20, sem chunk 4) + the OpenSe3r eval chain, per scene:
#   1. SLAM + semantic hooks -> reference-format exports
#   2. instance AP  (SN200, benchmark_void + segment_snap)   -> eval_inst_sn200.{log,csv}
#   3. instsem broadcast (per-instance sem_query -> voxels)   -> broadcast.log
#   4. instance-semantic eval (SN20 + SN200, per scene)       -> eval_instsem_sn{20,200}.{log,csv}
# then an OVO-style instsem aggregate over the 5 scenes       -> ovo_instsem.log
#
# Usage:
#   OUT=outputs/ma_sem/<tag> KV_ARGS="--kv_policy h2o --h2o_recent 8 --h2o_hh 8 \
#       --h2o_decay 0.9 --h2o_subsample 0.25" GPU=0 bash scripts/run_scannet5_ma_sem.sh
#   KV_ARGS=""                              -> flat KV (kv_policy none baseline)
#   KV_ARGS="--kv_policy window --kv_window_size 16"  -> window baseline
set -u
cd "$(dirname "$0")/.."
PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python
GPU=${GPU:-0}
OUT=${OUT:?set OUT=outputs/ma_sem/<tag>}
KV_ARGS=${KV_ARGS:-}
EXTRA_ARGS=${EXTRA_ARGS:-}      # e.g. --proposal_conf prefer
DATA=/mnt/HDD4/ricky/data/slam_data/data/ScanNet
OSR=/mnt/HDD4/ricky/OpenSe3r
S200=$OSR/scannet_200
SUM=$OUT/SUMMARY.txt
ROOT=$(pwd)
SCENES=${SCENES:-"scene0011_00 scene0050_00 scene0231_00 scene0378_00 scene0518_00"}

mkdir -p $OUT
[ -f $SUM ] || echo "ma_slam_semantic scannet-5 (--geometry ma, submap 20, chunk 4, DA3 rgb; KV_ARGS='${KV_ARGS}')  $(date)" > $SUM

for SC in $SCENES; do
  echo "======== $SC  $(date) ========" | tee -a $SUM
  odir=$OUT/$SC; mkdir -p $odir
  # 1. SLAM + semantics
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -u -m ma_slam_semantic.run \
      --geometry ma --data $DATA/$SC --backend da3 --mode rgb --submap_size 20 --keyframe_disparity 25 --sim_threshold 0.50 --max_per_submap 1 \
      --sem_lc shadow --bd_s2 on --sem_lc_mode bidesc --lc_proposal_conf gate --lc_history map --kv_policy none --h2o_subsample 1.0 \
      --out $odir --gt $DATA/$SC/gt_pose.txt --conf_gate_coef 1.0 --gt_mesh $DATA/$SC/${SC}_vh_clean_2.labels.ply ${KV_ARGS} ${EXTRA_ARGS} > $odir/run.log 2>&1
  grep -E "Sim3-ATE|exported" $odir/run.log | tail -2 | tee -a $SUM
  # 2. instance AP (SN200; class-agnostic SN20 is identical under this protocol)
  (cd $OSR && $PY eval_instance_npz_scannet.py \
      --hdbscan_labels_npz $ROOT/$odir/${SC}_instance_labels.npz \
      --gt_ply $S200/$SC/gt_scannet200_rawid_${SC}.ply \
      --aggregation_json $S200/$SC/${SC}.aggregation.json \
      --segs_json $S200/$SC/${SC}_vh_clean_2.0.010000.segs.json \
      --benchmark 200 --constants_path $S200/scannet200_constants.py \
      --benchmark_void --segment_snap \
      --save_csv $ROOT/$odir/eval_inst_sn200.csv \
      > $ROOT/$odir/eval_inst_sn200.log 2>&1)
  row=$(grep "^class_agnostic," $odir/eval_inst_sn200.csv 2>/dev/null | tail -1)
  echo "  inst SN200:  AP=$(echo $row|cut -d, -f2) AP50=$(echo $row|cut -d, -f3) AP25=$(echo $row|cut -d, -f4) (n_gt=$(echo $row|cut -d, -f5) n_pred=$(echo $row|cut -d, -f6))" | tee -a $SUM
  # 3. instsem broadcast (per-instance sem_query -> voxels)
  (cd $OSR && CUDA_VISIBLE_DEVICES=$GPU $PY query_instance_semantic.py \
      --instance_labels_npz      $ROOT/$odir/${SC}_instance_labels.npz \
      --instance_sem_queries_npz $ROOT/$odir/${SC}_instance_sem_queries.npz \
      --benchmark 20 --backbone tips-v2-l14 \
      --constants_path $S200/scannet200_constants.py \
      --output_dir $ROOT/$odir --demo_name $SC \
      > $ROOT/$odir/broadcast.log 2>&1) \
    && echo "  broadcast ok" | tee -a $SUM || echo "  broadcast FAILED" | tee -a $SUM
  # 4. per-scene instance-semantic eval (SN20 + SN200)
  for BM in 20 200; do
    if [ "$BM" = "20" ]; then GT_PLY=$DATA/$SC/${SC}_vh_clean_2.labels.ply
    else GT_PLY=$S200/$SC/gt_scannet200_rawid_${SC}.ply; fi
    (cd $OSR && CUDA_VISIBLE_DEVICES=$GPU $PY eval_semantic_npz_scannet.py \
        --semantic_npz $ROOT/$odir/${SC}_instance_semantic_feats.npz \
        --gt_ply $GT_PLY \
        --benchmark $BM --backbone tips-v2-l14 \
        --constants_path $S200/scannet200_constants.py \
        --save_csv $ROOT/$odir/eval_instsem_sn${BM}.csv \
        > $ROOT/$odir/eval_instsem_sn${BM}.log 2>&1)
    miou=$(grep -ioE "miou[^0-9]*[0-9.]+" $odir/eval_instsem_sn${BM}.log | head -1)
    echo "  instsem SN${BM}: ${miou:-see eval_instsem_sn${BM}.log}" | tee -a $SUM
  done
done

# 5. OVO-style instsem aggregate over the 5 scenes (SN20 + SN200)
echo "" | tee -a $SUM
echo "======== OVO-style instsem aggregate (5 scenes) ========" | tee -a $SUM
(cd $OSR && CUDA_VISIBLE_DEVICES=$GPU $PY tools/ovo_style_semantic_eval.py \
    --dir_pattern "$ROOT/$OUT/{sc}" \
    --backbone tips-v2-l14 --tasks instsem --benches sn20 sn200 \
    > $ROOT/$OUT/ovo_instsem.log 2>&1)
grep -E "instsem|mIoU|sn20|sn200" $OUT/ovo_instsem.log | tail -20 | tee -a $SUM
echo "DONE $(date)" | tee -a $SUM
