#!/usr/bin/env bash
#
# tereljRoiRun.sh — the three Terelj areas with Burmaa's ROIs, on the P1 CHM:
# the datasets (CHM and RGB), CC and the two baselines, and Mask R-CNN on the
# RGB and on the CHM, each run replaying the first area's old run; then
# tt.runCompare per area. It waits for the edge-only rerun (edgeRerun.sh) to
# free the machine, so it can be started at any time:
#
#   nohup ./tereljRoiRun.sh > tereljRoiRun.log 2>&1 &
#
#   first:    the datasets of every area (CHM and RGB, a few minutes, light)
#   GPU lane: waits for "GPU lane finished" in study_edge/STATUS.txt (or for
#             edgeRerun.sh to be gone), trains the RGB and the CHM network of
#             every area
#   CPU lane: waits for "CPU lane finished" likewise, then runs
#             tereljMoreStudy.sh per area (CC, local maxima, watershed on P1),
#             with JOBS workers; its CHM dataset is already there (built the
#             same way) and is reused
#
# Everything inside each ROI is scored: no don't-care cut. Output in
# 2026TreeDet/study_<area>/<area>/ (tereljFirst, tereljSecond, tereljThird),
# one line per step in study_tereljROI_STATUS.txt, comparisons in
# study_tereljROI_compare.txt. A finished step is skipped when run again.

set -u
cd "$(dirname "$0")" || exit 1
DROP=${DROP:-"$HOME/Yago Lab Dropbox/Meetings/2026TreeDet"}
PROC=${PROC:-"$DROP/study_procedure"}
EDGE_STATUS=${EDGE_STATUS:-"$DROP/study_edge/STATUS.txt"}
JOBS=${JOBS:-24}
REF=${REF:-"$PROC/terelj_p1"}
STATUS="$DROP/study_tereljROI_STATUS.txt"
COMPARE="$DROP/study_tereljROI_compare.txt"
export TT_OBJECTIVE=${TT_OBJECTIVE:-weighted} PYTHONUNBUFFERED=1
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS"; }
pooled() { grep -h "pooled over" "$1" 2>/dev/null | tail -1 | sed 's/^ *//'; }
value() { python -c "import json,sys; print(json.load(open(sys.argv[1]+'/dataset.json'))[sys.argv[2]])" "$1" "$2"; }
datasetOf() { python -c "import json,sys; print(json.load(open(sys.argv[1]+'/results.json'))['settings']['dataset'])" "$1" 2>/dev/null; }

# site, P1 CHM, crowns, ROI, RGB
AREAS=${AREAS:-"tereljFirst|Data/CHM_P1_2026_clipped_last.tif|Data/tereljMore/annotations_new09_25/annotationnew_0925.shp|Data/roiNew/area_first.shp|Data/ortho_p1_2026_clipped_last.tif
tereljSecond|Data/tereljMore/CHM_p1_2026_2.tif|Data/tereljMore/annotation_2nd.shp|Data/roiNew/area_second.shp|Data/tereljMore/RGB_P1_2026_2.tif
tereljThird|Data/tereljMore/chm_fromP1_Clip3.tif|Data/tereljMore/annotation3.shp|Data/roiNew/area_third.shp|Data/result_Clip313.tif"}

waitFor() {   # waitFor TEXT: until the edge rerun has logged TEXT, or is gone
    log "waiting for the edge-only rerun: '$1'"
    while ! grep -q "$1" "$EDGE_STATUS" 2>/dev/null && pgrep -f edgeRerun.sh > /dev/null; do
        sleep 300
    done
    log "go: '$1' (or the edge-only rerun is no longer running)"
}

step() {   # step LABEL LOG DONE_FILE COMMAND...
    local label=$1 logFile=$2 done=$3; shift 3
    if [ -e "$done" ]; then log "$label: already done"; return 0; fi
    log "$label: started"
    if "$@" > "$logFile" 2>&1; then log "$label: done  $(pooled "$logFile")"
    else log "$label: FAILED, see $logFile"; return 1; fi
}

refRun() {   # refRun NAME: the first area's old run of that name (P1, else LiDAR folder)
    for d in "$REF" "$PROC/terelj_lidar"; do
        [ -f "$d/runs/$1/results.json" ] && { echo "$d/runs/$1"; return 0; }
    done
    return 1
}

datasets() {   # datasets SITE CHM CROWNS ROI RGB: ds/P1 (as tereljMoreStudy) and ds/rgb
    local site=$1 chm=$2 crowns=$3 roi=$4 rgb=$5 out="$DROP/study_$1/$1"
    local refChm="$REF/ds/chm" refRgb
    mkdir -p "$out/logs" "$out/runs"
    step "$site/ds_P1" "$out/logs/datasetP1.log" "$out/ds/P1/dataset.json" \
        python -m tt.dl prepare --source "chm:$chm" --crowns "$crowns" --boundary "$roi" \
        --output "$out/ds/P1" --name "$site" --resolution "$(value "$refChm" resolution)" \
        --tileSize "$(value "$refChm" tileSize)" --overlap "$(value "$refChm" overlap)" \
        --minHeight "$(value "$refChm" minHeight)"
    refRgb=$(datasetOf "$(refRun mrcnnRgb)")
    [ -f "$refRgb/dataset.json" ] || refRgb="$REF/ds/rgb"
    if [ ! -f "$refRgb/dataset.json" ]; then log "$site/ds_rgb: no reference RGB dataset; skipped"; return 0; fi
    step "$site/ds_rgb" "$out/logs/datasetRgb.log" "$out/ds/rgb/dataset.json" \
        python -m tt.dl prepare --source "rgb:$rgb" --crowns "$crowns" --boundary "$roi" \
        --output "$out/ds/rgb" --name "$site" --resolution "$(value "$refRgb" resolution)" \
        --tileSize "$(value "$refRgb" tileSize)" --overlap "$(value "$refRgb" overlap)" \
        --minHeight "$(value "$refRgb" minHeight)"
}

network() {   # network SITE RUN DATASET
    local site=$1 run=$2 ds=$3 out="$DROP/study_$1/$1" ref args
    ref=$(refRun "$run") || { log "$site/$run: no reference run; skipped"; return 0; }
    [ -f "$ds/dataset.json" ] || { log "$site/$run: no dataset $ds; skipped"; return 0; }
    args=$(python -m tt.replayRun "$ref/results.json" tt.dl maskrcnn 2>> "$out/logs/$run.replay") \
        || { log "$site/$run: FAILED to replay $ref"; return 1; }
    eval "set -- $args"
    step "$site/$run" "$out/logs/$run.log" "$out/runs/$run/results.json" \
        python -m tt.dl maskrcnn --dataset "$ds" --output "$out/runs/$run" "$@"
}

gpuLane() {
    waitFor "GPU lane finished"
    while IFS='|' read -r site chm crowns roi rgb; do
        [ -n "$site" ] || continue
        network "$site" mrcnnRgb "$DROP/study_$site/$site/ds/rgb"
        network "$site" mrcnnP1 "$DROP/study_$site/$site/ds/P1"
    done <<< "$AREAS"
    log "Terelj GPU lane finished"
}

cpuLane() {
    waitFor "CPU lane finished"
    while IFS='|' read -r site chm crowns roi rgb; do
        [ -n "$site" ] || continue
        log "$site: height methods (tereljMoreStudy.sh, $JOBS workers)"
        JOBS=$JOBS SITE=$site SOURCES=P1 P1CHM=$chm CROWNS=$crowns AREA=$roi \
            ./tereljMoreStudy.sh >> "$DROP/study_$site/tereljMoreStudy.log" 2>&1 \
            || log "$site: height methods FAILED, see study_$site/tereljMoreStudy.log"
    done <<< "$AREAS"
    log "Terelj CPU lane finished"
}

log "Terelj ROI run started"
while IFS='|' read -r site chm crowns roi rgb; do
    [ -n "$site" ] || continue
    for f in "$chm" "$crowns" "$roi" "$rgb"; do
        [ -e "$f" ] || { log "ERROR: $site: $f missing; stopping"; exit 1; }
    done
    mkdir -p "$DROP/study_$site"
done <<< "$AREAS"

# the datasets first, once, before either lane (a few minutes, light): the
# CPU lane's tereljMoreStudy.sh then finds them done and reuses them
while IFS='|' read -r site chm crowns roi rgb; do
    [ -n "$site" ] || continue
    datasets "$site" "$chm" "$crowns" "$roi" "$rgb"
done <<< "$AREAS"

gpuLane &
G=$!
cpuLane &
C=$!
wait $G $C

: > "$COMPARE"
while IFS='|' read -r site chm crowns roi rgb; do
    [ -n "$site" ] || continue
    R="$DROP/study_$site/$site/runs"
    others=()
    for o in lmvwP1 watershedP1 mrcnnP1 mrcnnRgb; do [ -f "$R/$o/results.json" ] && others+=("$R/$o"); done
    if [ -f "$R/ccP1/results.json" ] && [ ${#others[@]} -gt 0 ]; then
        { echo; echo "== $site, CC on P1 against the rest"; } >> "$COMPARE"
        python -m tt.runCompare --reference "$R/ccP1" --others "${others[@]}" \
            --output "$DROP/study_$site/compare_ccP1.json" >> "$COMPARE" 2>/dev/null
    fi
done <<< "$AREAS"
log "Terelj ROI run finished; comparisons in study_tereljROI_compare.txt"
