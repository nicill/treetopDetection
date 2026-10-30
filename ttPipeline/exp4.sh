#!/usr/bin/env bash
#
# exp4.sh — experiment 4, unattended: what is left with little or no annotation.
#
#   nohup ./exp4.sh > exp4.log 2>&1 &
#   DRYRUN=1 ./exp4.sh                        # what it would do
#   STAGES="hybrid deepforest" ./exp4.sh      # some stages only
#   ONLY="koiwai enshurinV2" ./exp4.sh        # some sites only
#
# Stages, at every site (P1 only):
#   hybrid      CC tuned on the RGB Mask R-CNN's held-out boxes instead of
#               crowns (tt.pseudoTuning, scorer pseudoF1, fixed in advance):
#               <site>/exp4/runs/ccAutoRgbP1. The tuning computes every CC
#               setting's detections once (<site>/exp4/pseudoRgb/detections.npz)
#               and links them as runs/ccP1/detections.npz for the curve.
#   deepforest  DeepForest as downloaded (tt.deepForestDetect, its fixed
#               threshold): <site>/exp4/runs/deepforestRgb; then CC tuned on
#               its boxes the same way: <site>/exp4/runs/ccAutoDfP1. Nothing
#               of the site's annotation is used.
#   ccCurve     CC's learning curve (tt.learningCurve cc: fractions
#               0.05-1, seeds 1-3, drawn on the RGB tiles so the networks share
#               the draws): DROP/exp4/curves/<site>/ccCurve_ccP1.json
#   netCurve    RGB Mask R-CNN's learning curve: the RGB run's own settings,
#               replayed with --trainFraction F --subsetSeed S, for F in
#               NET_FRACTIONS and S in NET_SEEDS: curves/<site>/mrcnn_fF_sS;
#               the full run is linked as mrcnn_f1_s1. The 0% point is the same
#               network with --epochs 0 (COCO weights, untrained tree head),
#               one run per seed (curves/<site>/mrcnn_f0_sS, a minute each;
#               the head is not seeded, so each draw differs). The expensive stage:
#               with the defaults about 2.7 full trainings per site.
# Two lanes run at once: the CPU lane (hybrid, ccCurve, then the CC tuned on
# DeepForest, once DeepForest has run) and the GPU lane (DeepForest, then the
# network curve). At the end: the curve summary and experiment4.py, values in
# DROP/exp4/experiment4.txt. One line per step in DROP/exp4/STATUS.txt; a
# finished step is skipped when run again; the machine is kept from sleeping.
#
# Settings from the environment: DROP, JOBS (16), STAGES (all), ONLY,
# DF_PYTHON (the python with DeepForest installed; default python),
# NET_FRACTIONS ("0.05 0.1 0.25 0.5"), NET_SEEDS ("1 2 3"), DRYRUN.

set -u
cd "$(dirname "$0")" || exit 1
if [ -z "${EXP4_INHIBITED:-}" ] && systemd-inhibit --list > /dev/null 2>&1; then
    EXP4_INHIBITED=1 exec systemd-inhibit --what=idle:sleep --who=exp4 \
        --why="experiment 4" "$0" "$@"
fi
export DROP=${DROP:-"$HOME/Yago Lab Dropbox/Meetings/2026TreeDet"}
JOBS=${JOBS:-16}
STAGES=${STAGES:-"hybrid deepforest ccCurve netCurve"}
ONLY=${ONLY:-}
DF_PYTHON=${DF_PYTHON:-python}
NET_FRACTIONS=${NET_FRACTIONS:-"0.05 0.1 0.25 0.5"}
NET_SEEDS=${NET_SEEDS:-"1 2 3"}
DRYRUN=${DRYRUN:-0}
OUT="$DROP/exp4"
CURVES="$OUT/curves"
STATUS="$OUT/STATUS.txt"
export TT_OBJECTIVE=${TT_OBJECTIVE:-weighted} PYTHONUNBUFFERED=1
mkdir -p "$CURVES"
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS"; }
pooled() { grep -h "pooled over\|\[deepforest\]" "$1" 2>/dev/null | tail -1 | sed 's/^ *//'; }
wants() { [[ " $STAGES " == *" $1 "* ]]; }

# name | folder holding ds/ and runs/   (the sites of experiments 2 and 3)
SITES="afcamoisan|$DROP/study_edge/20230712_afcamoisan_itrf20
afcahoule|$DROP/study_edge/20230712_afcahoule_itrf20
cbblackburn5|$DROP/study_edge/20230606_cbblackburn5
cbbernard4|$DROP/study_edge/20230608_cbbernard4
cbblackburn6|$DROP/study_edge/20230606_cbblackburn6
koiwai|$DROP/study_edge/koiwainoujo220616
tereljFirst|$DROP/study_tereljFirst/tereljFirst
tereljSecond|$DROP/study_tereljSecond/tereljSecond
tereljThird|$DROP/study_tereljThird/tereljThird
enshurinV2|$DROP/study_enshurinVlad2"

step() {   # step LABEL LOG DONE_FILE COMMAND...
    local label=$1 logFile=$2 done=$3; shift 3
    if [ -e "$done" ]; then log "$label: already done"; return 0; fi
    if [ "$DRYRUN" = 1 ]; then log "$label: would run: $*"; return 0; fi
    mkdir -p "$(dirname "$logFile")"
    log "$label: started"
    if "$@" > "$logFile" 2>&1; then log "$label: done  $(pooled "$logFile")"
    else log "$label: FAILED, see $logFile"; return 1; fi
}

chmDataset() {   # the site's CHM dataset: ds/P1, else ds/chm
    local d
    for d in "$1/ds/P1" "$1/ds/chm"; do
        [ -f "$d/dataset.json" ] && { echo "$d"; return 0; }
    done
    return 1
}

rgbDataset() {   # the dataset the RGB run trained on
    python -c "import json,sys; print(json.load(open(sys.argv[1]))['settings']['dataset'])" \
        "$1/runs/mrcnnRgb/results.json" 2>/dev/null
}

eachSite() {   # eachSite FUNCTION: FUNCTION NAME FOLDER for every site with its runs
    local name folder
    while IFS='|' read -r name folder; do
        [ -n "$name" ] || continue
        if [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
        if [ ! -f "$folder/runs/mrcnnRgb/results.json" ] || [ ! -f "$folder/runs/ccP1/results.json" ]; then
            log "$name: no runs/mrcnnRgb or runs/ccP1 in $folder; skipped"; continue
        fi
        "$1" "$name" "$folder"
    done <<< "$SITES"
}

# ---------------------------------------------------------------- CPU lane

hybrid() {
    local name=$1 S=$2 ds
    ds=$(chmDataset "$S") || { log "$name: no CHM dataset; skipped"; return 0; }
    step "$name/ccAutoRgbP1" "$S/exp4/logs/pseudoRgb.log" "$S/exp4/runs/ccAutoRgbP1/results.json" \
        python -m tt.pseudoTuning --ccRun "$S/runs/ccP1" --rgbRun "$S/runs/mrcnnRgb" \
        --dataset "$ds" --jobs "$JOBS" --output "$S/exp4/pseudoRgb" \
        --autoRun "$S/exp4/runs/ccAutoRgbP1" --autoScorer pseudoF1
    if [ ! -e "$S/runs/ccP1/detections.npz" ] && [ -f "$S/exp4/pseudoRgb/detections.npz" ]; then
        ln -s "$S/exp4/pseudoRgb/detections.npz" "$S/runs/ccP1/detections.npz" \
            && log "$name: CC detections linked into runs/ccP1 for the curve"
    fi
}

ccCurve() {
    local name=$1 S=$2 ds rgb
    ds=$(chmDataset "$S") || { log "$name: no CHM dataset; skipped"; return 0; }
    rgb=$(rgbDataset "$S")
    if [ "$DRYRUN" != 1 ] && [ ! -e "$S/runs/ccP1/detections.npz" ]; then
        log "$name: no CC detections (run the hybrid stage first); CC curve skipped"; return 0
    fi
    mkdir -p "$CURVES/$name"
    step "$name/ccCurve" "$S/exp4/logs/ccCurve.log" "$CURVES/$name/ccCurve_ccP1.json" \
        python -m tt.learningCurve cc --ccRun "$S/runs/ccP1" --dataset "$ds" --tiles "$rgb" \
        --jobs "$JOBS" --output "$CURVES/$name/ccCurve_ccP1.json"
}

ccOnDeepforest() {
    local name=$1 S=$2 ds waited=0
    ds=$(chmDataset "$S") || return 0
    until [ -f "$S/exp4/runs/deepforestRgb/results.json" ]; do   # the GPU lane makes it
        if [ "$DRYRUN" = 1 ] || [ -n "${dfUnavailable:-}" ] || ! wants deepforest; then break; fi
        [ -f "$OUT/.deepforestFinished" ] && break
        [ $waited -ge 14400 ] && break
        sleep 120; waited=$((waited + 120))
    done
    if [ "$DRYRUN" != 1 ] && [ ! -f "$S/exp4/runs/deepforestRgb/results.json" ]; then
        log "$name: no DeepForest run; CC on DeepForest skipped"; return 0
    fi
    step "$name/ccAutoDfP1" "$S/exp4/logs/pseudoDf.log" "$S/exp4/runs/ccAutoDfP1/results.json" \
        python -m tt.pseudoTuning --ccRun "$S/runs/ccP1" --rgbRun "$S/exp4/runs/deepforestRgb" \
        --dataset "$ds" --jobs "$JOBS" --output "$S/exp4/pseudoDf" \
        --autoRun "$S/exp4/runs/ccAutoDfP1" --autoScorer pseudoF1
}

cpuLane() {
    wants hybrid && eachSite hybrid
    wants ccCurve && eachSite ccCurve
    wants deepforest && eachSite ccOnDeepforest
    log "experiment 4 CPU lane finished"
}

# ---------------------------------------------------------------- GPU lane

deepforest() {
    local name=$1 S=$2 rgb
    rgb=$(rgbDataset "$S")
    step "$name/deepforestRgb" "$S/exp4/logs/deepforest.log" "$S/exp4/runs/deepforestRgb/results.json" \
        "$DF_PYTHON" -m tt.deepForestDetect --dataset "$rgb" --output "$S/exp4/runs/deepforestRgb"
}

netCurve() {
    local name=$1 S=$2 rgb args f s
    rgb=$(rgbDataset "$S")
    mkdir -p "$CURVES/$name"
    [ -e "$CURVES/$name/mrcnn_f1_s1" ] || ln -s "$S/runs/mrcnnRgb" "$CURVES/$name/mrcnn_f1_s1"
    if [ "$DRYRUN" = 1 ]; then
        args="(the RGB run's settings)"
    else
        args=$(python -m tt.replayRun "$S/runs/mrcnnRgb/results.json" tt.dl maskrcnn 2>> "$S/exp4/logs/netCurve.replay") \
            || { log "$name: FAILED to replay runs/mrcnnRgb; network curve skipped"; return 0; }
    fi
    for s in $NET_SEEDS; do      # the 0% point: no training, COCO weights and a random tree head
        if [ "$DRYRUN" = 1 ]; then
            step "$name/mrcnn_f0_s${s}" /dev/null "$CURVES/$name/mrcnn_f0_s${s}/results.json" \
                python -m tt.dl maskrcnn --dataset "$rgb" "$args" --epochs 0
            continue
        fi
        eval "set -- $args"
        step "$name/mrcnn_f0_s${s}" "$S/exp4/logs/mrcnn_f0_s${s}.log" \
            "$CURVES/$name/mrcnn_f0_s${s}/results.json" \
            python -m tt.dl maskrcnn --dataset "$rgb" --output "$CURVES/$name/mrcnn_f0_s${s}" \
            "$@" --epochs 0
    done
    for f in $NET_FRACTIONS; do
        for s in $NET_SEEDS; do
            if [ "$DRYRUN" = 1 ]; then
                step "$name/mrcnn_f${f}_s${s}" /dev/null "$CURVES/$name/mrcnn_f${f}_s${s}/results.json" \
                    python -m tt.dl maskrcnn --dataset "$rgb" "$args" --trainFraction "$f" --subsetSeed "$s"
                continue
            fi
            eval "set -- $args"
            step "$name/mrcnn_f${f}_s${s}" "$S/exp4/logs/mrcnn_f${f}_s${s}.log" \
                "$CURVES/$name/mrcnn_f${f}_s${s}/results.json" \
                python -m tt.dl maskrcnn --dataset "$rgb" --output "$CURVES/$name/mrcnn_f${f}_s${s}" \
                "$@" --trainFraction "$f" --subsetSeed "$s"
        done
    done
}

gpuLane() {
    if wants deepforest; then
        if [ "$DRYRUN" = 1 ] || "$DF_PYTHON" -c "import deepforest" > /dev/null 2>&1; then
            eachSite deepforest
        else
            log "DeepForest is not installed for $DF_PYTHON (set DF_PYTHON); DeepForest skipped"
        fi
        touch "$OUT/.deepforestFinished"
    fi
    wants netCurve && eachSite netCurve
    log "experiment 4 GPU lane finished"
}

log "experiment 4 started (stages: $STAGES; dry run: $DRYRUN)"
rm -f "$OUT/.deepforestFinished"
gpuLane &
G=$!
cpuLane &
C=$!
wait $G $C

if [ "$DRYRUN" != 1 ]; then
    python -m tt.learningCurve summary --root "$CURVES" --output "$CURVES/summary.txt" \
        > "$OUT/curveSummary.log" 2>&1 && log "curve summary: $CURVES/summary.txt"
    if python experiment4.py --output "$OUT/experiment4.json" > "$OUT/experiment4.txt" 2>&1; then
        log "values: $OUT/experiment4.txt"
    else
        log "experiment4.py FAILED, see $OUT/experiment4.txt"
    fi
fi
log "experiment 4 finished"
