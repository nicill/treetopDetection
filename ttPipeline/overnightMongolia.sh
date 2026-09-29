#!/usr/bin/env bash
#
# overnightMongolia.sh — an unattended run on the Mongolian site: more networks
# on the RGB mosaic and the CHMs, every combination of a mosaic detector with a
# height detector, a report after every stage, and an image of every held-out
# block of every result.
#
#   cd ttPipeline && ./overnightMongolia.sh
#
# Reuses the runs already in runs/ (connected components on both CHMs, Mask
# R-CNN on both CHMs and on the mosaic) and the datasets in ds/. Trains, in
# order: YOLO on the mosaic; Faster R-CNN, RetinaNet and FCOS on the mosaic;
# YOLO on the LiDAR and on the P1 CHM. About 4-5 hours on the RTX 4090.
#
# Before those, the Sergi site: connected components at settings fixed on
# Mongolia against Sergi's own tuning (runs/ccSergi), then Mask R-CNN on the
# Sergi height model (ds/sergi), each block drawn — under sergi/.
#
# A stage that fails is recorded and skipped; the night goes on. STATUS.txt in
# the output folder says, stage by stage, what finished and with what score,
# and is the file to open first from elsewhere.

set -u
cd "$(dirname "$0")"

ROOT="${ONGOING_EXPS:-/home/yago/Yago Lab Dropbox/Meetings/ongoingExps}"
DRYRUN=${DRYRUN:-0}       # DRYRUN=1: everything but the training, to test
WEIGHTS_TIMEOUT=${WEIGHTS_TIMEOUT:-600}   # seconds to fetch YOLO weights
OUT="$ROOT/mongolia_$(date +%Y%m%d_%H%M)"
LIDAR=Data/chm_lidar_l2_2025_clipped_modified_last.tif
P1=Data/CHM_P1_2026_clipped_last.tif
CROWNS=Data/shp/annotation_new_0925.shp
AREA=Data/shp/area_p1.shp
DEVICE=0

mkdir -p "$OUT/runs" "$OUT/logs" "$OUT/report/snapshots" "$OUT/images"
STATUS="$OUT/STATUS.txt"
export PYTHONUNBUFFERED=1
exec > >(tee -a "$OUT/logs/overnight.log") 2>&1

status() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS"; }

# ---------------------------------------------------------------------- #
# checks before anything long starts
# ---------------------------------------------------------------------- #

status "start; output in $OUT"
cap=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq | sort -u | tr '\n' ' ')
if [ "$cap" != "4000000 " ]; then
    status "WARNING: CPU frequency cap is not 4 GHz (max: $cap); the machine may be unstable"
fi
for d in ds/lidar ds/p1 ds/rgb; do
    [ -f "$d/dataset.json" ] || { status "ERROR: $d missing; stopping"; exit 1; }
done
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader \
    | sed 's/^/GPU: /' | tee -a "$STATUS"

YOLO_OK=1
# with a time limit: a download that stalls at night must not stall the night
if ! timeout "$WEIGHTS_TIMEOUT" python -c \
        "from ultralytics import YOLO; YOLO('yolo11s-seg.pt')" \
        > "$OUT/logs/yoloWeights.log" 2>&1; then
    YOLO_OK=0
    status "WARNING: YOLO weights could not be loaded; YOLO stages will be skipped"
fi

# the runs already made, copied so every result of the night sits together
for name in ccLidar ccP1 mrcnnLidar mrcnnP1 mrcnnRgb; do
    if ls runs/$name/predictions_*.json > /dev/null 2>&1; then
        cp -r "runs/$name" "$OUT/runs/"
    else
        status "WARNING: runs/$name has no predictions; left out"
    fi
done
cp sweepLidar.json sweepP1.json "$OUT/" 2>/dev/null

# ---------------------------------------------------------------------- #
# stages
# ---------------------------------------------------------------------- #

STAGE=0

pooled() { grep -h "pooled over" "$1" 2>/dev/null | tail -1 | sed 's/^ *//'; }

train() {   # train NAME COMMAND...: a learned model into $OUT/runs/NAME
    trainTo "$1" "$OUT/runs/$1" "${@:2}"
}

trainTo() {   # trainTo NAME OUTDIR COMMAND...: one learned model, recorded
    local name=$1 outdir=$2; shift 2
    local log="$OUT/logs/$name.log"
    if [ -f "$outdir/results.json" ]; then
        status "$name: already done, skipped"; return
    fi
    if [ "$DRYRUN" = 1 ]; then status "$name: dry run, not trained"; return; fi
    status "$name: started"
    if "$@" --output "$outdir" > "$log" 2>&1; then
        status "$name: done  $(pooled "$log")"
    else
        status "$name: FAILED (exit $?), see logs/$name.log"
    fi
}

reportAndImages() {   # reportAndImages LABEL: report, snapshot, block images
    STAGE=$((STAGE + 1))
    local label=$1 cv=() runs=()
    for dir in "$OUT"/runs/*/; do
        local name; name=$(basename "$dir")
        [ -f "$dir/results.json" ] && cv+=(--cv "$name=$dir")
        ls "$dir"predictions_*.json > /dev/null 2>&1 && runs+=(--run "$name=$dir")
    done
    status "report $STAGE ($label): $(( ${#runs[@]} / 2 )) runs with predictions"
    if python -m tt.report --lidarChm "$LIDAR" --p1Chm "$P1" \
            --crowns "$CROWNS" --boundary "$AREA" --dataset ds/lidar \
            "${cv[@]}" --sweep "ccLidar=$OUT/sweepLidar.json" \
            --sweep "ccP1=$OUT/sweepP1.json" --output "$OUT/report" \
            > "$OUT/logs/report_$STAGE.log" 2>&1; then
        cp "$OUT/report/report.odt" \
            "$OUT/report/snapshots/report_${STAGE}_$label.odt"
        status "report $STAGE: written (snapshots/report_${STAGE}_$label.odt)"
    else
        status "report $STAGE: FAILED, see logs/report_$STAGE.log"
    fi
    if python -m tt.blockImages --chm "$LIDAR" --crowns "$CROWNS" \
            --boundary "$AREA" --dataset ds/lidar "${runs[@]}" \
            --fused "$OUT/report/fused" --output "$OUT/images" \
            > "$OUT/logs/images_$STAGE.log" 2>&1; then
        status "images $STAGE: $(tail -1 "$OUT/logs/images_$STAGE.log")"
    else
        status "images $STAGE: FAILED, see logs/images_$STAGE.log"
    fi
}

tv() {   # tv NAME MODELTYPE DATASET: a torchvision detector
    train "$1" python -m tt.dl maskrcnn --dataset "$3" --modelType "$2" \
        --epochs 40 --batchSize 4 --device $DEVICE
}

yolo() {   # yolo NAME DATASET
    if [ $YOLO_OK -eq 1 ]; then
        train "$1" python -m tt.dl yolo --dataset "$2" --epochs 150 \
            --batchSize 8 --device $DEVICE
    else
        status "$1: skipped, no YOLO weights"
    fi
}

# ---------------------------------------------------------------------- #
# Sergi: does the site need its own tuning, and how does Mask R-CNN do there
# ---------------------------------------------------------------------- #

SERGI="$OUT/sergi"
SCHM=Data/sergi/chm.tif
SCROWNS=Data/sergi/crowns.shp
SAREA=Data/sergi/area.shp

sergiReady() {
    for f in "$SCHM" "$SCROWNS" "$SAREA" ds/sergi/dataset.json \
            runs/ccSergi/results.json; do
        [ -e "$f" ] || { status "sergi: $f missing; Sergi stages skipped"; \
                         return 1; }
    done
}

sergiCompare() {   # sergiCompare LABEL: fixed settings, learned runs, images
    local label=$1 compare=() runs=()
    [ -f "$SERGI/runs/mrcnnSergi/results.json" ] && \
        compare+=(--compare "mrcnnSergi=$SERGI/runs/mrcnnSergi")
    if python -m tt.transfer --chm "$SCHM" --crowns "$SCROWNS" \
            --boundary "$SAREA" --dataset ds/sergi \
            --tuned "$SERGI/runs/ccSergi" \
            --modalFrom "ccLidar=$OUT/runs/ccLidar" \
            --modalFrom "ccP1=$OUT/runs/ccP1" "${compare[@]}" \
            --output "$SERGI/transfer" \
            > "$OUT/logs/sergiTransfer_$label.log" 2>&1; then
        status "sergi comparison ($label), F1 pooled over blocks:"
        sed -n '/^setting/,/^$/p' "$OUT/logs/sergiTransfer_$label.log" \
            | sed 's/^/        /' | tee -a "$STATUS"
    else
        status "sergi comparison ($label): FAILED, see logs/sergiTransfer_$label.log"
    fi
    for dir in "$SERGI"/runs/*/ "$SERGI"/transfer/runs/*/; do
        [ -d "$dir" ] || continue
        ls "$dir"predictions_*.json > /dev/null 2>&1 && \
            runs+=(--run "$(basename "$dir")=$dir")
    done
    if python -m tt.blockImages --chm "$SCHM" --crowns "$SCROWNS" \
            --boundary "$SAREA" --dataset ds/sergi "${runs[@]}" \
            --output "$SERGI/images" \
            > "$OUT/logs/sergiImages_$label.log" 2>&1; then
        status "sergi images ($label): $(tail -1 "$OUT/logs/sergiImages_$label.log")"
    else
        status "sergi images ($label): FAILED, see logs/sergiImages_$label.log"
    fi
}

reportAndImages existingRuns          # within minutes: what is already there

if sergiReady; then
    mkdir -p "$SERGI/runs"
    cp -r runs/ccSergi "$SERGI/runs/"
    sergiCompare fixedSettings        # minutes: no training
    trainTo mrcnnSergi "$SERGI/runs/mrcnnSergi" python -m tt.dl maskrcnn \
        --dataset ds/sergi --modelType maskrcnn --epochs 40 --batchSize 4 \
        --device $DEVICE
    sergiCompare withMaskRcnn
fi

yolo yoloRgb ds/rgb
reportAndImages yoloRgb

tv fasterrcnnRgb fasterrcnn ds/rgb
reportAndImages fasterrcnnRgb

tv retinanetRgb retinanet ds/rgb
tv fcosRgb fcos ds/rgb
reportAndImages oneStageRgb

yolo yoloLidar ds/lidar
yolo yoloP1 ds/p1
reportAndImages yoloChm               # the final report

du -sh "$OUT" | sed 's/^/size: /' | tee -a "$STATUS"
status "finished"
