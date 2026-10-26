#!/usr/bin/env bash
#
# nightP1.sh — the remaining photogrammetric work, unattended, in order:
#
#   1. cbbernard4's photogrammetric CHM, from its local point cloud
#   2. waits until cbblackburn6's point cloud has finished arriving (its size
#      unchanged for STABLE_MIN minutes), then builds its CHM
#   3. CC, local maxima and watershed on every photogrammetric Quebec CHM
#      (quebecP1Study.sh: only what is missing is run)
#   4. Mask R-CNN on the new CHMs only: each Quebec site's photogrammetric CHM
#      (replaying that site's LiDAR CHM network, runs/mrcnnLidar) and the
#      second Terelj area's P1 CHM (replaying the first area's runs/mrcnnP1);
#      nothing already run is repeated
#   5. the paired comparison again, now with Mask R-CNN on the CHM
#
#   nohup ./nightP1.sh > nightP1.log 2>&1 &
#
# A step that fails is logged and the night goes on. NIGHT_STATUS.txt in
# study_procedureP1 has one line per step.

set -u
cd "$(dirname "$0")" || exit 1

DROP=${DROP:-"$HOME/Yago Lab Dropbox/Meetings/2026TreeDet"}
PROC=${PROC:-"$DROP/study_procedure"}
QP1=${QP1:-"$DROP/Quebec2P1"}
OUTQ=${OUTQ:-"$DROP/study_procedureP1"}
TERELJ=${TERELJ:-"$DROP/study_tereljMore/tereljMore"}
CLOUDS=${CLOUDS:-"$HOME/qpClouds"}
PDAL=${PDAL:-"$HOME/anaconda3/envs/pdal/bin/pdal"}
STABLE_MIN=${STABLE_MIN:-10}
export TT_OBJECTIVE=${TT_OBJECTIVE:-weighted} PYTHONUNBUFFERED=1

mkdir -p "$OUTQ"
STATUS="$OUTQ/NIGHT_STATUS.txt"
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS"; }
pooled() { grep -h "pooled over" "$1" 2>/dev/null | tail -1 | sed 's/^ *//'; }

buildChm() {   # buildChm SITE: the photogrammetric CHM from the local cloud
    if [ -f "$QP1/$1/check.json" ]; then log "$1: CHM already built"; return 0; fi
    log "$1: CHM started"
    if python -m tt.prepare.quebecPhotogrammetry --prepared "$DROP/Quebec2" --output "$QP1" \
            --pdal "$PDAL" --cloudDir "$CLOUDS" --sites "$1" --jobs 8 \
            --tileTimeoutMin 120 > "$OUTQ/chm_$1.log" 2>&1; then
        log "$1: CHM done  $(grep -h "floor" "$OUTQ/chm_$1.log" | tail -1 | cut -c1-120)"
    else
        log "$1: CHM FAILED, see chm_$1.log"
    fi
}

waitForCloud() {   # waitForCloud SITE: until its cloud exists and stops growing
    local path size last="" still=0
    log "$1: waiting for its point cloud to finish arriving in $CLOUDS"
    while [ "$still" -lt "$STABLE_MIN" ]; do
        path=$(python -c "from tt.prepare.quebecPhotogrammetry import localCloud; print(localCloud('$1', '$CLOUDS'))" 2>/dev/null)
        size=$( [ -n "$path" ] && stat -c %s "$path" 2>/dev/null || echo none)
        if [ "$size" != none ] && [ "$size" = "$last" ]; then still=$((still + 1)); else still=0; fi
        last=$size
        sleep 60
    done
    log "$1: point cloud complete, $(numfmt --to=iec "$last") ($path)"
}

network() {   # network NAME REFERENCE_RUN DATASET OUTPUT: the CHM Mask R-CNN
    local name=$1 ref=$2 ds=$3 out=$4 refDs args
    if [ -f "$out/results.json" ]; then log "$name: already done"; return 0; fi
    if [ ! -f "$ref/results.json" ]; then log "$name: no reference $ref; skipped"; return 0; fi
    if [ ! -f "$ds/dataset.json" ]; then log "$name: no dataset $ds; skipped"; return 0; fi
    refDs=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['settings']['dataset'])" "$ref/results.json")
    if [ "$(basename "$refDs")" != chm ]; then
        log "$name: the reference trained on $refDs, not on the CC dataset; skipped"; return 0
    fi
    args=$(python -m tt.replayRun "$ref/results.json" tt.dl maskrcnn 2>> "$STATUS") \
        || { log "$name: FAILED to read the reference's settings"; return 1; }
    eval "set -- $args"
    mkdir -p "$(dirname "$out")"
    log "$name: started"
    if python -m tt.dl maskrcnn --dataset "$ds" --output "$out" "$@" > "$out.log" 2>&1; then
        log "$name: done  $(pooled "$out.log")"
    else
        log "$name: FAILED, see $out.log"
    fi
}

log "night started"

# 1-2. the two remaining CHMs
buildChm 20230608_cbbernard4
[ -f "$QP1/20230606_cbblackburn6/check.json" ] || waitForCloud 20230606_cbblackburn6
buildChm 20230606_cbblackburn6

# 3. CC and the baselines on every photogrammetric CHM ready
log "CC and baselines: started"
JOBS=16 ./quebecP1Study.sh > "$OUTQ/quebecP1Study.log" 2>&1 \
    && log "CC and baselines: done (study_procedureP1/compare.txt)" \
    || log "CC and baselines: FAILED, see quebecP1Study.log"

# 4. Mask R-CNN on the new CHMs only
for dir in "$OUTQ"/*/; do
    site=$(basename "$dir")
    [ -f "$dir/ds/chm/dataset.json" ] || continue
    network "$site/mrcnnP1" "$PROC/$site/runs/mrcnnLidar" "$dir/ds/chm" "$dir/runs/mrcnnP1"
done
network "tereljMore/mrcnnP1" "$PROC/terelj_p1/runs/mrcnnP1" "$TERELJ/ds/P1" "$TERELJ/runs/mrcnnP1"

# 5. the comparison, now with the CHM networks
python -m tt.heightSourceCompare --first "$PROC" --second "$OUTQ" \
    --output "$OUTQ/compare.json" > "$OUTQ/compare.txt" 2>&1 \
    && log "comparison with Mask R-CNN: study_procedureP1/compare.txt" \
    || log "comparison FAILED"
log "night finished"
