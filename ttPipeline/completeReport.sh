#!/usr/bin/env bash
#
# completeReport.sh — re-run every method with prediction saving, then write
# the full report. Run from the directory holding tt/, Data/ and ds/.
#
#   ./completeReport.sh
#
# ds/lidar, ds/p1 and ds/rgb are the datasets your earlier run built; they are
# rebuilt only if missing. The sweeps are re-run only if sweepLidar.json /
# sweepP1.json are missing. On an RTX 4090 the three Mask R-CNN runs take about
# 20 minutes each; everything else is minutes.

set -euo pipefail

LIDAR=Data/chm_lidar_l2_2025_clipped_modified_last.tif
P1=Data/CHM_P1_2026_clipped_last.tif
ORTHO=Data/ortho_p1_2026_clipped_last.tif
CROWNS=Data/shp/annotation_new_0925.shp
AREA=Data/shp/area_p1.shp
DEVICE=0

mkdir -p logs runs

# Everything, step headers included, also goes to one timestamped file; the
# per-step logs below are kept as well, since they are easier to read singly.
RUNLOG="logs/completeReport_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUNLOG") 2>&1
echo "logging to $RUNLOG"

step() { printf '\n=== %s  (%s) ===\n' "$1" "$(date +%H:%M:%S)"; }

for pair in "lidar:chm:$LIDAR" "p1:chm:$P1" "rgb:rgb:$ORTHO"; do
    name=${pair%%:*}; source=${pair#*:}
    if [ ! -f "ds/$name/dataset.json" ]; then
        step "prepare $name"
        python -m tt.dl prepare --source "$source" --crowns "$CROWNS" \
            --boundary "$AREA" --output "ds/$name" --name "$name" \
            2>&1 | tee "logs/prepare_$name.log"
    fi
done

[ -f sweepLidar.json ] || { step "sweep LiDAR"; python -m tt.cli sweep --chm "$LIDAR" \
    --crowns "$CROWNS" --boundary "$AREA" --output sweepLidar.json 2>&1 | tee logs/sweepLidar.log; }
[ -f sweepP1.json ] || { step "sweep P1"; python -m tt.cli sweep --chm "$P1" \
    --crowns "$CROWNS" --boundary "$AREA" --output sweepP1.json 2>&1 | tee logs/sweepP1.log; }

step "connected components, LiDAR"
python -m tt.dl concomp --dataset ds/lidar --output runs/ccLidar \
    --percentiles 20,30 --minTopAreas 0.12,0.06 --topSteps 0.12,0.25 \
    --erosions 1,2 --saddleDrops 0.2,0.3,0.5 2>&1 | tee logs/ccLidar.log

step "connected components, P1"
python -m tt.dl concomp --dataset ds/p1 --output runs/ccP1 \
    --percentiles 10,20 --minTopAreas 0.25,0.12 --topSteps 0.12,0.25 \
    --erosions 1,2 --saddleDrops 0.2,0.3,0.5 2>&1 | tee logs/ccP1.log

step "Mask R-CNN, LiDAR CHM (about an hour)"
python -m tt.dl maskrcnn --dataset ds/lidar --output runs/mrcnnLidar \
    --epochs 40 --batchSize 4 --device $DEVICE 2>&1 | tee logs/mrcnnLidar.log

step "Mask R-CNN, P1 CHM"
python -m tt.dl maskrcnn --dataset ds/p1 --output runs/mrcnnP1 \
    --epochs 40 --batchSize 4 --device $DEVICE 2>&1 | tee logs/mrcnnP1.log

step "Mask R-CNN, RGB mosaic (about an hour)"
python -m tt.dl maskrcnn --dataset ds/rgb --output runs/mrcnnRgb \
    --epochs 40 --batchSize 4 --device $DEVICE 2>&1 | tee logs/mrcnnRgb.log

step "report"
python -m tt.report --lidarChm "$LIDAR" --p1Chm "$P1" --crowns "$CROWNS" \
    --boundary "$AREA" --dataset ds/lidar \
    --cv ccLidar=runs/ccLidar --cv ccP1=runs/ccP1 \
    --cv mrcnnLidar=runs/mrcnnLidar --cv mrcnnP1=runs/mrcnnP1 \
    --cv mrcnnRgb=runs/mrcnnRgb \
    --sweep ccLidar=sweepLidar.json --sweep ccP1=sweepP1.json \
    --output report 2>&1 | tee logs/report.log

echo; echo "done: report/report.odt, figures in report/figures/"
echo "full log: $RUNLOG"
