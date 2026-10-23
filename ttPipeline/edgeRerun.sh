#!/usr/bin/env bash
#
# edgeRerun.sh — Quebec (five sites, LiDAR and photogrammetry) and Koiwai
# (current and DSM CHM) again, scored in the edge-only area: the interior gaps
# between crowns count, only the canopy at the area's border stays don't-care.
#
#   nohup ./edgeRerun.sh > edgeRerun.log 2>&1 &
#
#  0. scoredAreaEdge.shp for every site (tt.edgeOnlyArea), once
#  1. per site the datasets on that area: the CHM of each height source and
#     the RGB, each at its old dataset's resolution, tile size, overlap and
#     minimum height, with the old dataset's name
#  2. two lanes at once, smaller sites first and cbblackburn6 last:
#       CPU: CC, local maxima, watershed on each height source (JOBS workers)
#       GPU: Mask R-CNN on each CHM and on the RGB
#     every run replays its old run's own settings (tt.replayRun): Quebec from
#     the LiDAR runs (ccLidar, lmvwLidar, watershedLidar, mrcnnLidar, mrcnnRgb),
#     Koiwai from the current CHM's (ccP1, ..., mrcnnP1, mrcnnRgb)
#  3. tt.runCompare per site: CC against the rest on each height source, and
#     each method's two height sources against each other
#
# Output in OUT (default 2026TreeDet/study_edge), STATUS.txt there with one
# line per step. A finished step is skipped when the script is run again. The
# computer is left on.
#
# Settings from the environment: DROP, PROC, Q2, Q2P1, KSITE, KDSM, OUT, JOBS
# (default 24), QUEBEC (the sites, in order), KOIWAI (1 to include it).

set -u
cd "$(dirname "$0")" || exit 1
DROP=${DROP:-"$HOME/Yago Lab Dropbox/Meetings/2026TreeDet"}
PROC=${PROC:-"$DROP/study_procedure"}
Q2=${Q2:-"$DROP/Quebec2"}
Q2P1=${Q2P1:-"$DROP/Quebec2P1"}
KSITE=${KSITE:-Data/sergiOut/koiwainoujo220616}
KDSM=${KDSM:-Data/sergiDsm/koiwainoujo220616_dsm0.25}
OUT=${OUT:-"$DROP/study_edge"}
JOBS=${JOBS:-24}
QUEBEC=${QUEBEC:-"20230712_afcamoisan_itrf20 20230712_afcahoule_itrf20 20230606_cbblackburn5 20230608_cbbernard4"}
BIG=${BIG:-20230606_cbblackburn6}
KOIWAI=${KOIWAI:-1}
export TT_OBJECTIVE=${TT_OBJECTIVE:-weighted} PYTHONUNBUFFERED=1
mkdir -p "$OUT"
STATUS="$OUT/STATUS.txt"
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS"; }
pooled() { grep -h "pooled over" "$1" 2>/dev/null | tail -1 | sed 's/^ *//'; }
value() { python -c "import json,sys; print(json.load(open(sys.argv[1]+'/dataset.json'))[sys.argv[2]])" "$1" "$2"; }
datasetOf() { python -c "import json,sys; print(json.load(open(sys.argv[1]+'/results.json'))['settings']['dataset'])" "$1"; }

ORDER=""
for s in $QUEBEC; do ORDER="$ORDER $s"; done
[ "$KOIWAI" = 1 ] && ORDER="$ORDER koiwai"
[ -n "$BIG" ] && ORDER="$ORDER $BIG"

setSite() {   # setSite NAME: SITE_DIR REF REFSRC SOURCES CHM_<source> NAME_OUT
    if [ "$1" = koiwai ]; then
        SITE_DIR=$KSITE; REF="$PROC/koiwainoujo220616"; REFSRC=P1
        SOURCES="P1 Dsm"; CHM_P1="$KSITE/chm.tif"; CHM_Dsm="$KDSM/chm.tif"
        NAME_OUT=koiwainoujo220616
    else
        SITE_DIR="$Q2/$1"; REF="$PROC/$1"; REFSRC=Lidar
        SOURCES="Lidar P1"; CHM_Lidar="$Q2/$1/chm.tif"; CHM_P1="$Q2P1/$1/chm.tif"
        NAME_OUT=$1
    fi
    S_OUT="$OUT/$NAME_OUT"
    mkdir -p "$S_OUT/logs" "$S_OUT/runs" "$S_OUT/ds"
}

step() {   # step SITE NAME DONE_FILE COMMAND...: run unless DONE_FILE exists
    local site=$1 name=$2 done=$3; shift 3
    local logFile="$S_OUT/logs/$name.log"
    if [ -e "$done" ]; then log "$site/$name: already done"; return 0; fi
    log "$site/$name: started"
    if "$@" > "$logFile" 2>&1; then
        log "$site/$name: done  $(pooled "$logFile")"
    else
        log "$site/$name: FAILED, see $NAME_OUT/logs/$name.log"; return 1
    fi
}

dataset() {   # dataset SITE NAME SOURCE_SPEC REFERENCE_DATASET
    local site=$1 name=$2 source=$3 refDs=$4
    if [ ! -f "$refDs/dataset.json" ]; then log "$site/ds_$name: no reference dataset $refDs; skipped"; return 1; fi
    step "$site" "ds_$name" "$S_OUT/ds/$name/dataset.json" python -m tt.dl prepare \
        --source "$source" --crowns "$SITE_DIR/crowns.shp" \
        --boundary "$SITE_DIR/scoredAreaEdge.shp" --output "$S_OUT/ds/$name" \
        --name "$(value "$refDs" name)" --resolution "$(value "$refDs" resolution)" \
        --tileSize "$(value "$refDs" tileSize)" --overlap "$(value "$refDs" overlap)" \
        --minHeight "$(value "$refDs" minHeight)"
}

replay() {   # replay SITE RUN REFERENCE_RUN DATASET JOBS? MODULE...
    local site=$1 run=$2 ref=$3 ds=$4 jobs=$5; shift 5
    local args extra=()
    if [ ! -f "$ref/results.json" ]; then log "$site/$run: no reference $ref; skipped"; return 0; fi
    if [ ! -f "$ds/dataset.json" ]; then log "$site/$run: no dataset $ds; skipped"; return 0; fi
    args=$(python -m tt.replayRun "$ref/results.json" "$@" 2>> "$S_OUT/logs/$run.replay") \
        || { log "$site/$run: FAILED to replay $ref (see logs/$run.replay)"; return 1; }
    [ -n "$jobs" ] && extra=(--jobs "$jobs")
    eval "set -- \"\$@\" $args"
    step "$site" "$run" "$S_OUT/runs/$run/results.json" python -m "$@" \
        --dataset "$ds" --output "$S_OUT/runs/$run" "${extra[@]}"
}

cpuLane() {
    for site in $ORDER; do
        setSite "$site"
        for src in $SOURCES; do
            replay "$site" "cc$src" "$REF/runs/cc$REFSRC" "$S_OUT/ds/$src" "$JOBS" tt.dl concomp
            replay "$site" "lmvw$src" "$REF/runs/lmvw$REFSRC" "$S_OUT/ds/$src" "$JOBS" tt.dl.dlBaselines
            replay "$site" "watershed$src" "$REF/runs/watershed$REFSRC" "$S_OUT/ds/$src" "$JOBS" tt.dl.dlBaselines
        done
    done
    log "CPU lane finished"
}

gpuLane() {
    for site in $ORDER; do
        setSite "$site"
        replay "$site" mrcnnRgb "$REF/runs/mrcnnRgb" "$S_OUT/ds/rgb" "" tt.dl maskrcnn
        for src in $SOURCES; do
            replay "$site" "mrcnn$src" "$REF/runs/mrcnn$REFSRC" "$S_OUT/ds/$src" "" tt.dl maskrcnn
        done
    done
    log "GPU lane finished"
}

log "edge-only rerun started (JOBS=$JOBS; order:$ORDER)"

# 0. the edge-only areas
for site in $ORDER; do
    setSite "$site"
    if [ ! -f "$SITE_DIR/scoredAreaEdge.shp" ]; then
        python -m tt.edgeOnlyArea "$SITE_DIR" | tee -a "$STATUS"
    fi
done

# 1. the datasets
for site in $ORDER; do
    setSite "$site"
    refChm=$(datasetOf "$REF/runs/cc$REFSRC" 2>/dev/null); [ -f "$refChm/dataset.json" ] || refChm="$REF/ds/chm"
    for src in $SOURCES; do
        eval "chm=\$CHM_$src"
        dataset "$site" "$src" "chm:$chm" "$refChm"
    done
    refRgb=$(datasetOf "$REF/runs/mrcnnRgb" 2>/dev/null); [ -f "$refRgb/dataset.json" ] || refRgb="$REF/ds/rgb"
    dataset "$site" rgb "rgb:$SITE_DIR/rgb.tif" "$refRgb"
done

# 2. both lanes at once
cpuLane &
CPU=$!
gpuLane &
GPU=$!
wait $CPU $GPU

# 3. comparisons
: > "$OUT/compare.txt"
for site in $ORDER; do
    setSite "$site"
    R="$S_OUT/runs"
    for src in $SOURCES; do
        others=()
        for o in "lmvw$src" "watershed$src" "mrcnn$src" mrcnnRgb; do [ -f "$R/$o/results.json" ] && others+=("$R/$o"); done
        if [ -f "$R/cc$src/results.json" ] && [ ${#others[@]} -gt 0 ]; then
            { echo; echo "== $NAME_OUT, CC on $src against the rest"; } >> "$OUT/compare.txt"
            python -m tt.runCompare --reference "$R/cc$src" --others "${others[@]}" \
                --output "$S_OUT/compare_cc$src.json" >> "$OUT/compare.txt" 2>/dev/null
        fi
    done
    set -- $SOURCES; a=$1; b=$2
    for m in cc lmvw watershed mrcnn; do
        if [ -f "$R/$m$b/results.json" ] && [ -f "$R/$m$a/results.json" ]; then
            { echo; echo "== $NAME_OUT, $m: $b against $a"; } >> "$OUT/compare.txt"
            python -m tt.runCompare --reference "$R/$m$b" --others "$R/$m$a" \
                --output "$S_OUT/compare_$m$b$a.json" >> "$OUT/compare.txt" 2>/dev/null
        fi
    done
done
log "finished; comparisons in compare.txt"
