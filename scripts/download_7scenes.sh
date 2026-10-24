#!/bin/bash
# Download Microsoft 7-Scenes (all 7 scenes, ~21.6 GB of zips) and extract ONLY seq-01 of each
# (the VGGT-SLAM / MASt3R-SLAM evaluation protocol). Layout after extraction:
#   data/7scenes/_raw/<scene>/seq-01/frame-XXXXXX.{color.png,depth.png,pose.txt}
# Re-runnable: wget -c resumes, extracted scenes are skipped.
set -u
cd /mnt/HDD4/ricky/slam/ma_long
RAW=data/7scenes/_raw; mkdir -p $RAW
B=https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8
for s in ${SCENES:-chess fire heads office pumpkin redkitchen stairs}; do
  if [ -d $RAW/$s/seq-01 ] && [ "$(ls $RAW/$s/seq-01/*.color.png 2>/dev/null | wc -l)" -gt 100 ]; then
    echo "== $s already extracted ($(ls $RAW/$s/seq-01/*.color.png | wc -l) frames)"; continue
  fi
  echo "== $s download $(date +%H:%M)"
  wget -c -q --show-progress --progress=dot:giga -O $RAW/$s.zip $B/$s.zip 2>&1 | tail -1
  echo "== $s extract $(date +%H:%M)"
  # scene zip contains <scene>/seq-01.zip (plus other seqs + split files); pull seq-01 only
  unzip -q -o $RAW/$s.zip "$s/seq-01.zip" -d $RAW/ && unzip -q -o $RAW/$s/seq-01.zip -d $RAW/$s/ && rm -f $RAW/$s/seq-01.zip
  n=$(ls $RAW/$s/seq-01/*.color.png 2>/dev/null | wc -l)
  echo "== $s DONE frames=$n $(date +%H:%M)"
  [ "$n" -gt 100 ] && rm -f $RAW/$s.zip   # keep disk tidy; re-download if ever needed
done
echo SEVEN_SCENES_DOWNLOAD_DONE
