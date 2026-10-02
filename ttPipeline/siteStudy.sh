#!/usr/bin/env bash
#
# siteStudy.sh — the Terelj study on one prepared site: connected components
# tuned by leave-one-block-out, Mask R-CNN on the height model and on the RGB,
# every combination of the two, and connected components calibrated from the
# RGB Mask R-CNN.
#
#   ./siteStudy.sh SITE_DIR OUT_DIR
#
# SITE_DIR is a folder written by tt.qpPrepare or tt.sergiPrepare: chm.tif,
# rgb.tif, crowns.shp (the crowns to score) and scoredArea.shp (the area to
# score in, don't-care parts already cut out). OUT_DIR gets ds/, runs/,
# report/, calibration/, logs/ and STATUS.txt.
#
# Stages, in this order, each skipped when its result already exists:
#   datasets     ds/chm and ds/rgb, tiled, 4 x 2 spatial blocks
#   cc           CC tuned on the training blocks, scored on the held-out one,
#                over a grid that includes the minimum height and minimum tree
#                area; also the whole-area best, the optimistic ceiling
#   mrcnnRgb     Mask R-CNN on the RGB
#   calibration  CC calibrated from the RGB Mask R-CNN's confident boxes
#   pseudoTuning CC's grid searched against the RGB Mask R-CNN's boxes instead
#                of the crowns (tt.pseudoTuning)
#   report       every RGB x height combination, all strategies
#   mrcnnHeight  Mask R-CNN on the height model
#   report       again, now with the second height detector
# The calibration comes before the height Mask R-CNN because it needs only CC
# and the RGB model, and it is the question that matters most.
#
# Settings come from the environment, defaulting to the Quebec plantations:
#   HEIGHT    name of the height model in run names: Lidar (Quebec) or P1
#             (photogrammetric height, Sergi); the report pairs runs by name
#   RES       tile resolution of the datasets, m (0.03: small crowns ~ 55 px)
#   CCRES     resolution connected components works at, m
#   MINH      minimum height, m, and MINTREE minimum tree area, m2: the fixed
#             values the calibration uses (it may not tune on the crowns);
#             CC tunes its own over --minHeights and --minTreeAreas
#   CCGRID    the connected-component grid, as tt.dl concomp options
#   CCJOBS    settings detected in parallel
#   EPOCHS, BATCH, DEVICE, BLOCKCOLS, BLOCKROWS
#   OBJECTIVE f1 (default) or weighted: what every tuning step maximises
#   DRYRUN=1  everything but the training, to test the chain quickly
#
# The calibration's fixed MINH and MINTREE for Quebec were chosen while
# exploring afcamoisan: state that, or leave afcamoisan out of the final tables.

set -u
cd "$(dirname "$0")"

SITE=${1:?usage: siteStudy.sh SITE_DIR OUT_DIR}
OUT=${2:?usage: siteStudy.sh SITE_DIR OUT_DIR}
NAME=$(basename "$SITE")

HEIGHT=${HEIGHT:-Lidar}
RES=${RES:-0.03}
CCRES=${CCRES:-0.05}
MINH=${MINH:-1.0}
MINTREE=${MINTREE:-0.1}
CCGRID=${CCGRID:---percentiles 0,10 --minTopAreas 0.06,0.12 --topSteps 0.12,0.25 --erosions 0,1 --saddleDrops 0.3,0.5 --minHeights 0.75,1.0,1.5 --minTreeAreas 0.05,0.1,0.25,0.5}
CCJOBS=${CCJOBS:-16}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-4}
DEVICE=${DEVICE:-0}
BLOCKCOLS=${BLOCKCOLS:-4}
BLOCKROWS=${BLOCKROWS:-2}
DRYRUN=${DRYRUN:-0}
# what every tuning step maximises: f1, or weighted (0.6 recall + 0.4
# precision); every result reports both
export TT_OBJECTIVE=${OBJECTIVE:-${TT_OBJECTIVE:-f1}}

CHM="$SITE/chm.tif"; RGB="$SITE/rgb.tif"
CROWNS="$SITE/crowns.shp"; AREA="$SITE/scoredArea.shp"
CC="cc$HEIGHT"; MH="mrcnn$HEIGHT"

mkdir -p "$OUT/logs" "$OUT/runs"
STATUS="$OUT/STATUS.txt"
export PYTHONUNBUFFERED=1
status() { printf '%s  %s  %s\n' "$(date '+%F %T')" "$NAME" "$*" | tee -a "$STATUS"; }
pooled() { grep -h "pooled over" "$1" 2>/dev/null | tail -1 | sed 's/^ *//'; }

stage() {   # stage NAME DONE_FILE COMMAND...: run unless DONE_FILE exists
    local name=$1 done=$2; shift 2
    local log="$OUT/logs/$name.log"
    if [ -e "$done" ]; then status "$name: already done"; return 0; fi
    status "$name: started"
    if "$@" > "$log" 2>&1; then
        status "$name: done  $(pooled "$log")"
    else
        status "$name: FAILED (exit $?), see logs/$name.log"
        return 1
    fi
}

train() {   # train NAME DATASET: one Mask R-CNN, unless a dry run
    if [ "$DRYRUN" = 1 ]; then status "$1: dry run, not trained"; return 0; fi
    stage "$1" "$OUT/runs/$1/results.json" python -m tt.dl maskrcnn \
        --dataset "$2" --output "$OUT/runs/$1" --modelType maskrcnn \
        --epochs "$EPOCHS" --batchSize "$BATCH" --device "$DEVICE"
}

report() {   # report LABEL: combinations of whatever runs exist
    local cv=() p1=() sweep=()
    for dir in "$OUT"/runs/*/; do
        [ -f "$dir/results.json" ] && cv+=(--cv "$(basename "$dir")=$dir")
    done
    [ "$HEIGHT" = P1 ] && p1=(--p1Chm "$CHM")
    [ -f "$OUT/sweep.json" ] && sweep=(--sweep "$CC=$OUT/sweep.json")
    rm -rf "$OUT/report"
    stage "report_$1" "/nonexistent" python -m tt.report --lidarChm "$CHM" \
        "${p1[@]}" --crowns "$CROWNS" --boundary "$AREA" \
        --dataset "$OUT/ds/chm" "${cv[@]}" "${sweep[@]}" \
        --output "$OUT/report"
}

for f in "$CHM" "$RGB" "$CROWNS" "$AREA"; do
    [ -e "$f" ] || { status "ERROR: $f missing; site skipped"; exit 1; }
done
status "start (HEIGHT=$HEIGHT RES=$RES CCRES=$CCRES MINH=$MINH MINTREE=$MINTREE OBJECTIVE=$TT_OBJECTIVE)"

for kind in chm rgb; do
    src="$CHM"; [ $kind = rgb ] && src="$RGB"
    stage "dataset_$kind" "$OUT/ds/$kind/dataset.json" python -m tt.dl prepare \
        --source "$kind:$src" --crowns "$CROWNS" --boundary "$AREA" \
        --output "$OUT/ds/$kind" --name "$NAME" --resolution "$RES" \
        --minHeight "$MINH" --blockCols "$BLOCKCOLS" --blockRows "$BLOCKROWS" \
        || { status "no dataset, site stopped"; exit 1; }
done

# connected components tuned to the site: the grid includes the minimum height
# and minimum tree area, chosen per fold on the training blocks; the run also
# reports the whole-area best, the optimistic ceiling (no separate sweep)
# shellcheck disable=SC2086  (CCGRID is a list of options)
stage "$CC" "$OUT/runs/$CC/results.json" python -m tt.dl concomp \
    --dataset "$OUT/ds/chm" --output "$OUT/runs/$CC" --resolution "$CCRES" \
    --minHeight "$MINH" --minTreeArea "$MINTREE" --jobs "$CCJOBS" \
    --saveDetections $CCGRID

train mrcnnRgb "$OUT/ds/rgb"
if [ -f "$OUT/runs/mrcnnRgb/results.json" ] && [ -f "$OUT/runs/$CC/results.json" ]; then
    stage calibration "$OUT/calibration/calibration.json" python -m tt.calibration \
        --chm "$CHM" --crowns "$CROWNS" --boundary "$AREA" \
        --dataset "$OUT/ds/chm" --rgbRun "$OUT/runs/mrcnnRgb" \
        --ccRun "$OUT/runs/$CC" --resolution "$CCRES" --minHeight "$MINH" \
        --minTreeArea "$MINTREE" --output "$OUT/calibration"
fi
# CC tuned on the RGB Mask R-CNN's boxes instead of the crowns (no crown of
# the held-out block is used), against CC tuned on the real crowns
if [ -f "$OUT/runs/mrcnnRgb/results.json" ] && [ -f "$OUT/runs/$CC/results.json" ]; then
    stage pseudoTuning "$OUT/pseudoTuning/pseudoTuning.json" python -m tt.pseudoTuning \
        --ccRun "$OUT/runs/$CC" --rgbRun "$OUT/runs/mrcnnRgb" \
        --dataset "$OUT/ds/chm" --jobs "$CCJOBS" --output "$OUT/pseudoTuning"
fi
report rgbAndCc

train "$MH" "$OUT/ds/chm"
report final

status "finished"
