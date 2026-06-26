#!/usr/bin/env bash
# Head-to-head vs VGGT-SLAM on TUM RGB-D fr1.
# 9 sequences x 3 configs x 2 submap sizes = 54 runs. Sim3-ATE is the comparable metric
# (matches their RMSE-APE / Sim3-aligned protocol).
#
#   DO NOT run until GPU + state confirmed.  GPU is parameterized below.
#   Prereq: scripts/prep_tum_fr1.py already produced data/tum/fr1_<seq>/.
#
# Usage:  GPU=0 bash scripts/run_tum_fr1.sh        (optionally SEQS="xyz desk" SIZES="32")
set -euo pipefail

cd "$(dirname "$0")/.."
PY=/mnt/HDD4/ricky/envs/amb3r_bw/bin/python
GPU="${GPU:-0}"
SEQS="${SEQS:-360 desk desk2 floor plant room rpy teddy xyz}"
SIZES="${SIZES:-20 32}"
# VARIANT="" uses undistorted scenes (data/tum/fr1_<seq>); VARIANT="_dist" uses the raw
# distorted scenes (data/tum/fr1_<seq>_dist). Output dirs are tagged the same way.
VARIANT="${VARIANT:-}"
OUTROOT=outputs/ma_slam/tum
SUMMARY="$OUTROOT/summary.csv"
mkdir -p "$OUTROOT"
echo "config,seq,submap_size,sim3_ate,se3_ate,scale,submaps,loops" > "$SUMMARY"

# config name | backend | mode | extra flags
CONFIGS=(
  "da3_rgb|da3|rgb|"
  "da3_rgbintr|da3|rgb+intr|"
  "ma_rgbdi|ma|rgb+depth+intr|--depth_scale 5000"
)

run_one() {
  local name="$1" backend="$2" mode="$3" extra="$4" seq="$5" sz="$6"
  local scene="data/tum/fr1_${seq}${VARIANT}"
  local out="$OUTROOT/fr1_${seq}${VARIANT}_${name}_w${sz}"
  echo "=== [$name] fr1_${seq} w=${sz} -> $out ==="
  local log="$out.log"; mkdir -p "$out"
  PYTHONPATH=src CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m ma_slam.run \
      --scene "$scene" --mode "$mode" --backend "$backend" --manifold se3 \
      --submap_size "$sz" $extra \
      --out "$out" --gt "$scene/gt_pose.txt" 2>&1 | tee "$log"
  # scrape the eval line: "[ma_slam] Sim3-ATE=.. | SE3-ATE=.. | scale=.. | submaps=.. | loops=.."
  local line; line=$(grep -E "Sim3-ATE=" "$log" | tail -1 || true)
  local s3 se sc sm lp
  s3=$(sed -nE 's/.*Sim3-ATE=([0-9.]+).*/\1/p' <<<"$line")
  se=$(sed -nE 's/.*SE3-ATE=([0-9.]+).*/\1/p' <<<"$line")
  sc=$(sed -nE 's/.*scale=([0-9.]+).*/\1/p' <<<"$line")
  sm=$(sed -nE 's/.*submaps=([0-9]+).*/\1/p' <<<"$line")
  lp=$(sed -nE 's/.*loops=([0-9]+).*/\1/p' <<<"$line")
  echo "${name},${seq},${sz},${s3},${se},${sc},${sm},${lp}" >> "$SUMMARY"
}

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name backend mode extra <<<"$cfg"
  for sz in $SIZES; do
    for seq in $SEQS; do
      run_one "$name" "$backend" "$mode" "$extra" "$seq" "$sz"
    done
  done
done

echo "=== DONE. summary -> $SUMMARY ==="
column -t -s, "$SUMMARY"
