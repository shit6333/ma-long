#!/bin/bash
# Semantic / instance evaluation of ma_slam_semantic with PREDICTED depth + PREDICTED pose (DA3-rgb) and the
# final inst-LC (S1 instance-correspondence gate, relaxed SALAD, standard coloc), per dataset:
#   bash scripts/run_sem_eval.sh <scannet_val50|scannetpp|scenenn|3rscan|replica> [scene ...]
# env: GPU (default 1), OUT (default outputs/ma_sem/<dataset>_s1lc), KF (keyframe disparity, default 25 — NOTE: the
#      TUM geometry benchmark uses 50 px (VGGT-SLAM protocol); the semantic runs use 25 px like the ScanNet-5 harness)
# Frame sets = OpenSe3r GT-geometry loaders (scripts/prep_sem_eval_data.py). Eval flags copied verbatim from OpenSe3r's
# _eval_val50_{full,nostuff}.bash / _eval_scannetpp50.bash / _eval_scenenn12.bash / _eval_3rscan47.bash / _eval_replica_pairs.bash.
set -u
cd "$(dirname "$0")/.."
DS=${1:?dataset}; shift || true
PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python
GPU=${GPU:-1}; KF=${KF:-25}
OUT=${OUT:-outputs/ma_sem/${DS}_s1lc}; mkdir -p $OUT
OSR=/mnt/HDD4/ricky/OpenSe3r; S200=$OSR/scannet_200; ROOT=$(pwd)
KV="--kv_policy h2o --h2o_recent 8 --h2o_hh 8 --h2o_decay 0.9 --h2o_subsample 0.25"
EXTRA=${EXTRA:-}
ALIGN=${ALIGN:-pos}
LC="--align $ALIGN --sem_lc veto --sem_lc_mode rigid --rigid_geom off --rigid_lowobj reject --proposal_conf gate --lc_proposal_conf prefer --lc_history bag --sim_threshold 0.40 --max_per_submap 5"
SUM=$OUT/SUMMARY.csv; [ -f $SUM ] || echo "scene,n_frames,n_kf,sim3_ate,se3_ate,align_rmse,scale,loops,n_inst,AP,AP50,AP25,AP_ns,AP50_ns,AP25_ns,AP_top100,AP50_top100,AP25_top100,AP_alt,AP50_alt,AP25_alt,alt" > $SUM
case $DS in
  scannet_val50) DATA=data/scannet_val50; SCENES=${@:-$(ls $DATA)};;
  scannetpp)     DATA=data/scannetpp;     SCENES=${@:-$(cat /mnt/HDD4/ricky/data/scannetpp_eval/my_arrow_val_50.txt)};;
  scenenn)       DATA=data/scenenn;       SCENES=${@:-$(ls $DATA)};;
  3rscan)        DATA=${DATA:-data/3rscan};        SCENES=${@:-$(ls $DATA)};;
  replica)       DATA=${DATA:-data/replica};       SCENES=${@:-office0 office1 office2 office3 office4 room0 room1 room2};;
  *) echo "unknown dataset $DS"; exit 1;;
esac
ap3() { # parse "AP (avg ...): x", "AP50: y", "AP25: z" from an eval log -> "x,y,z"
  local f=$1; local a=$(grep -iE "^\s*AP \(avg" $f | tail -1 | grep -oE "[0-9]+\.[0-9]+" | tail -1)
  local b=$(grep -iE "^\s*AP50" $f | tail -1 | grep -oE "[0-9]+\.[0-9]+" | tail -1); local c=$(grep -iE "^\s*AP25" $f | tail -1 | grep -oE "[0-9]+\.[0-9]+" | tail -1)
  echo "${a:-},${b:-},${c:-}"
}
for SC in $SCENES; do
  od=$OUT/$SC; mkdir -p $od
  if grep -q "^$SC," $SUM; then echo "== $DS $SC (done, skip)"; continue; fi
  echo "== $DS $SC $(date +%H:%M)"
  MESH=""; case $DS in
    scannet_val50) MESH=/mnt/HDD4/ricky/data/scannet_eval/scans/$SC/${SC}_vh_clean_2.ply;;
    scannetpp)     MESH=/mnt/HDD4/ricky/data/scannetpp_eval/data/$SC/scans/mesh_aligned_0.05.ply;;
    scenenn)       MESH=/mnt/HDD4/ricky/data/instance_eval_data/scenenn/oni/$SC/$SC.ply;;
    3rscan)        MESH=/mnt/HDD4/ricky/data/3RScan_data/$SC/labels.instances.annotated.v2.ply;;
    replica)       MESH=/mnt/HDD4/ricky/data/replica/Replica/${SC}_mesh.ply;;
  esac
  AL=""; [ -n "${ALIGN:-}" ] && AL="--align $ALIGN --gt_mesh $MESH"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=$GPU $PY -u -m ma_slam_semantic.run --geometry ma --data $DATA/$SC --backend da3 --mode rgb \
      --submap_size 20 --chunk_size 4 --keyframe_disparity $KF --ckpt a1_pq --conf_gate_coef 1.0 $KV $LC \
      --gt $DATA/$SC/gt_pose.txt --out $od $AL $EXTRA > $od/run.log 2>&1
  L=$(grep "Sim3-ATE" $od/run.log | tail -1)
  nf=$(ls $DATA/$SC/rgb | wc -l); nk=$(grep -oE "kept [0-9]+/" $od/run.log | grep -oE "[0-9]+" | head -1)
  s3=$(echo "$L"|grep -oE 'Sim3-ATE=[0-9.]+'|cut -d= -f2); se=$(echo "$L"|grep -oE 'SE3-ATE=[0-9.]+'|cut -d= -f2); lp=$(echo "$L"|grep -oE 'loops=[0-9]+'|cut -d= -f2)
  al=$(grep -oE "residual RMSE [0-9.]+" $od/run.log | tail -1 | grep -oE "[0-9.]+$"); gs=$(grep -oE "global scale [0-9.]+" $od/run.log | tail -1 | grep -oE "[0-9.]+$"); ni=$(grep -oE "exported [0-9]+ instances" $od/run.log | grep -oE "[0-9]+" | head -1)
  echo "   $L align_rmse=$al scale=${gs:-1} inst=$ni"
  evalone() { # $1=dir (abs) $2=tag -> sets AP/APNS/APT for that export
    local D1=$1 TAG=$2 NPZ=$1/${SC}_instance_labels.npz SEM=$1/${SC}_instance_semantic_feats.npz
    AP=",,"; APNS=",,"; APT=",,"
    [ -f $NPZ ] || { echo "   !! no instance npz in $D1"; return; }
    case $DS in
      scannet_val50)
        local S=/mnt/HDD4/ricky/data/scannet_eval/scans/$SC
        (cd $OSR && CUDA_VISIBLE_DEVICES=$GPU $PY query_instance_semantic.py --instance_labels_npz $NPZ --instance_sem_queries_npz $D1/${SC}_instance_sem_queries.npz \
            --benchmark 20 --backbone tips-v2-l14 --constants_path $S200/scannet200_constants.py --output_dir $D1 --demo_name $SC > $D1/broadcast.log 2>&1)
        (cd $OSR && CUDA_VISIBLE_DEVICES=$GPU $PY eval_instance_npz_scannet.py --hdbscan_labels_npz $NPZ --gt_ply $S/${SC}_vh_clean_2.labels.ply --aggregation_json $S/${SC}.aggregation.json \
            --segs_json $S/${SC}_vh_clean_2.0.010000.segs.json --benchmark 20 --constants_path $S200/scannet200_constants.py --benchmark_void --segment_snap \
            --filter_pred_stuff --semantic_npz $SEM --backbone tips-v2-l14 --save_csv $D1/eval_instance_scannet20.csv --save_json $D1/eval_full_details.json > $D1/eval_full.log 2>&1)
        (cd $OSR && CUDA_VISIBLE_DEVICES=$GPU $PY eval_instance_npz_scannet.py --hdbscan_labels_npz $NPZ --gt_ply $S/${SC}_vh_clean_2.labels.ply --aggregation_json $S/${SC}.aggregation.json \
            --segs_json $S/${SC}_vh_clean_2.0.010000.segs.json --benchmark 20 --constants_path $S200/scannet200_constants.py --benchmark_void --segment_snap \
            --save_csv $D1/eval_instance_scannet20_nostuff.csv --save_json $D1/eval_nostuff_details.json > $D1/eval_nostuff.log 2>&1)
        AP=$(ap3 $D1/eval_full.log); APNS=$(ap3 $D1/eval_nostuff.log);;
      scannetpp)
        local R2=/mnt/HDD4/ricky/data/scannetpp_eval
        for V in "" _top100; do
          (cd $OSR && CUDA_VISIBLE_DEVICES=$GPU $PY eval_instance_npz_scannet.py --hdbscan_labels_npz $NPZ --gt_scenenn_npz $R2/_gt/${SC}_scannetpp_gt${V}.npz \
              --benchmark 20 --constants_path $S200/scannet200_constants.py --benchmark_void --segment_snap --segs_json $R2/_segs/$SC.segs.json \
              --save_csv $D1/eval_instance_scannetpp${V}_snap.csv --save_json $D1/eval${V}_details.json > $D1/eval${V}.log 2>&1)
        done
        AP=$(ap3 $D1/eval.log); APT=$(ap3 $D1/eval_top100.log);;
      scenenn)
        local ONI=/mnt/HDD4/ricky/data/instance_eval_data/scenenn/oni SEGS=/mnt/HDD4/ricky/data/instance_eval_data/scenenn/mesh_segs
        (cd $OSR && CUDA_VISIBLE_DEVICES=$GPU $PY eval_instance_npz_scannet.py --hdbscan_labels_npz $NPZ --gt_scenenn_npz $OSR/model_eval/outputs/scenenn_gt/${SC}_scenenn_gt.npz \
            --benchmark 20 --constants_path $S200/scannet200_constants.py --benchmark_void --segment_snap --segs_json $SEGS/${SC}.segs.json --snap_mesh_ply $ONI/$SC/${SC}.ply \
            --save_csv $D1/eval_instance_scenenn.csv --save_json $D1/eval_details.json > $D1/eval.log 2>&1)
        AP=$(ap3 $D1/eval.log);;
      3rscan)
        local R3=/mnt/HDD4/ricky/data/3RScan_data
        (cd $OSR && CUDA_VISIBLE_DEVICES=$GPU $PY eval_instance_npz_scannet.py --hdbscan_labels_npz $NPZ --gt_scenenn_npz $R3/_gt/${SC}_3rscan_gt.npz \
            --benchmark 20 --constants_path $S200/scannet200_constants.py --benchmark_void --segment_snap --segs_json $R3/$SC/mesh.refined.0.010000.segs.v2.json \
            --save_csv $D1/eval_instance_3rscan.csv --save_json $D1/eval_details.json > $D1/eval.log 2>&1)
        AP=$(ap3 $D1/eval.log);;
      replica)
        (cd $OSR && CUDA_VISIBLE_DEVICES=$GPU $PY query_instance_semantic.py --instance_labels_npz $NPZ --instance_sem_queries_npz $D1/${SC}_instance_sem_queries.npz \
            --benchmark 20 --backbone tips-v2-l14 --constants_path $S200/scannet200_constants.py --output_dir $D1 --demo_name $SC > $D1/broadcast.log 2>&1);;
    esac
  }
  evalone $ROOT/$od primary; AP1=$AP; APNS1=$APNS; APT1=$APT
  AP2=",,"; ALT=$(ls -d $ROOT/$od/align_* 2>/dev/null | head -1)
  if [ -n "$ALT" ]; then evalone $ALT alt; AP2=$AP; fi
  AP=$AP1; APNS=$APNS1; APT=$APT1
  echo "   AP=$AP nostuff=$APNS top100=$APT | alt($(basename ${ALT:-none}))=$AP2"
  echo "$SC,$nf,${nk:-},${s3:-},${se:-},${al:-},${gs:-1},${lp:-},${ni:-},$AP,$APNS,$APT,$AP2,$(basename ${ALT:-none})" >> $SUM
done
if [ $DS = replica ]; then
  (cd $OSR && CUDA_VISIBLE_DEVICES=$GPU $PY tools/ovo_style_semantic_eval.py --dir_pattern "$ROOT/$OUT/{sc}" --benches replica \
      --replica_gt /mnt/HDD4/ricky/data/replica/_gt --backbone tips-v2-l14 --scenes office0 office1 office2 office3 office4 room0 room1 room2 --tasks instsem > $ROOT/$OUT/ovo_instsem.log 2>&1)
  grep -iE "^replica|instsem|mIoU" $OUT/ovo_instsem.log | tail -5
fi
(cd $OSR && $PY tools/check_run_health.py $ROOT/$OUT > $ROOT/$OUT/health.log 2>&1); tail -3 $OUT/health.log
echo "SEM_EVAL_DONE $DS"
