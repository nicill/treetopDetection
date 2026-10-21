#!/usr/bin/env bash
#
# tereljMoreStudy.sh — CC and the two height baselines (local maxima,
# watershed) on another Terelj area, on its LiDAR CHM and/or its P1 CHM,
# with the settings of the first Terelj area's runs.
#
#   ./tereljMoreStudy.sh                      # second area, LiDAR and P1
#   SITE=tereljThird SOURCES=P1 D=Data/terelj3 P1CHM=$D/chm_fromP1_Clip3.tif \
#       CROWNS=$D/annotation3.shp ./tereljMoreStudy.sh     # third area, P1 only
#
# For each height source (Lidar, P1):
#   ds/<source>        the dataset, at the first Terelj area's resolution, tile
#                      size, overlap and minimum height; both sources use the
#                      same crowns and the same area, so their blocks are the
#                      same (checked): the LiDAR/P1 comparison is paired
#   runs/cc<S>         CC with the first area's own settings (grid, resolution,
#                      minimum height and tree area, smoothing, ...), replayed
#                      from PROC/terelj_<source>/runs/cc<S>/results.json
#   runs/lmvw<S>, runs/watershed<S>   the baselines, likewise
# and, with both sources, at the end tt.heightSourceCompare: each method on P1
# against LiDAR, and CC against the better baseline on each, paired Wilcoxon
# over blocks; with P1 only, the P1 scores.
# A finished step is skipped when the script is run again.
#
# The area to score in: AREA (a shapefile of the area polygon). If there is
# none, one is derived from the crowns the way qpPrepare does for Quebec
# (crowns buffered AREA_BUFFER_M, holes filled), written to OUT/area.shp and
# said so in the status file: replace it with the real polygon when there is
# one (delete OUT and rerun). The area decides which tops count as mistakes,
# so it must not be changed after results have been seen.
#
# Settings from the environment: DROP, PROC, D (the data folder), SITE (names
# the runs' folder, default tereljMore), SOURCES (Lidar, P1 or "Lidar P1",
# default both), LIDAR (the LiDAR CHM, the shifted one), P1CHM, CROWNS, AREA,
# OUT (default study_<SITE>), JOBS (default 16), AREA_BUFFER_M (default 2).

set -u
cd "$(dirname "$0")" || exit 1

DROP=${DROP:-"$HOME/Yago Lab Dropbox/Meetings/2026TreeDet"}
PROC=${PROC:-"$DROP/study_procedure"}
D=${D:-Data/tereljMore}
LIDAR=${LIDAR:-"$D/shifted2/CHM_idar_l2_2025_2.tif.vrt"}
P1CHM=${P1CHM:-"$D/CHM_p1_2026_2.tif"}
CROWNS=${CROWNS:-"$D/annotation_2nd.shp"}
AREA=${AREA:-}
SITE=${SITE:-tereljMore}
SOURCES=${SOURCES:-"Lidar P1"}
OUT=${OUT:-"$DROP/study_$SITE"}
JOBS=${JOBS:-16}
AREA_BUFFER_M=${AREA_BUFFER_M:-2}
export TT_OBJECTIVE=${TT_OBJECTIVE:-weighted} PYTHONUNBUFFERED=1

SITE_OUT="$OUT/$SITE"
mkdir -p "$SITE_OUT/logs" "$SITE_OUT/runs"
STATUS="$OUT/STATUS.txt"
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS"; }
pooled() { grep -h "pooled over" "$1" 2>/dev/null | tail -1 | sed 's/^ *//'; }

needed=("$CROWNS")
for source in $SOURCES; do
    if [ "$source" = Lidar ]; then needed+=("$LIDAR"); else needed+=("$P1CHM"); fi
done
for f in "${needed[@]}"; do
    [ -e "$f" ] || { log "ERROR: $f missing; stopping"; exit 1; }
done

if [ -z "$AREA" ]; then
    AREA="$OUT/area.shp"
    if [ ! -f "$AREA" ]; then
        python - "$CROWNS" "$AREA" "$AREA_BUFFER_M" <<'PY' || { log "ERROR: could not derive the area"; exit 1; }
import sys
import geopandas as gpd
from tt.qpPrepare import annotatedArea
crowns = gpd.read_file(sys.argv[1])
area = annotatedArea(crowns, float(sys.argv[3]))
area.to_file(sys.argv[2])
print("[area] %d crowns, %.0f m2" % (len(crowns), area.area.iloc[0]))
PY
        log "AREA derived from the crowns (buffered ${AREA_BUFFER_M} m, holes filled): $AREA; replace with the real polygon when there is one"
    fi
else
    [ -e "$AREA" ] || { log "ERROR: AREA $AREA missing; stopping"; exit 1; }
    log "AREA given: $AREA"
fi

replayArgs() {   # replayArgs RESULTS MODULE...: a run's own options, quoted
    python - "$@" <<'PY'
import json, shlex, subprocess, sys
settings = json.load(open(sys.argv[1]))["settings"]
helpText = subprocess.run([sys.executable, "-m"] + sys.argv[2:] + ["--help"],
                          capture_output=True, text=True).stdout
skip = {"dataset", "output", "log", "jobs", "progress"}
words = []
for key, value in settings.items():
    if key in skip or value is None or value is False:
        continue
    if "--" + key not in helpText:
        sys.exit("replay: --%s is not an option of %s" % (key, sys.argv[2:]))
    words += ["--" + key] if value is True else ["--" + key, str(value)]
print(" ".join(shlex.quote(w) for w in words))
PY
}

datasetValue() {   # datasetValue DATASET_DIR KEY
    python -c "import json,sys; print(json.load(open(sys.argv[1]+'/dataset.json'))[sys.argv[2]])" "$1" "$2"
}

sameBlocks() {   # sameBlocks DATASET_A DATASET_B: exit 0 when the blocks match
    python - "$1" "$2" <<'PY'
import json, sys
from shapely import wkt
a, b = ([(x["name"], wkt.loads(x["wkt"])) for x in
         json.load(open(d + "/dataset.json"))["blocks"]] for d in sys.argv[1:])
same = len(a) == len(b) and all(
    na == nb and ga.symmetric_difference(gb).area < 1e-6 * max(ga.area, 1)
    for (na, ga), (nb, gb) in zip(a, b))
sys.exit(0 if same else 1)
PY
}

step() {   # step NAME DONE_FILE COMMAND...: run unless DONE_FILE exists
    local name=$1 done=$2; shift 2
    local logFile="$SITE_OUT/logs/$name.log"
    if [ -e "$done" ]; then log "$name: already done"; return 0; fi
    log "$name: started"
    if "$@" > "$logFile" 2>&1; then
        log "$name: done  $(pooled "$logFile")"
    else
        log "$name: FAILED, see logs/$name.log"
        return 1
    fi
}

replay() {   # replay NAME REFERENCE_RUN DATASET MODULE...: a run again here
    local name=$1 reference=$2 ds=$3; shift 3
    local args
    [ -f "$reference/results.json" ] || { log "$name: no $reference, skipped"; return 0; }
    args=$(replayArgs "$reference/results.json" "$@") || { log "$name: FAILED to read $reference"; return 1; }
    eval "set -- \"\$@\" $args"
    step "$name" "$SITE_OUT/runs/$name/results.json" python -m "$@" \
        --dataset "$ds" --output "$SITE_OUT/runs/$name" --jobs "$JOBS"
}

for source in $SOURCES; do
    if [ "$source" = Lidar ]; then chm=$LIDAR; ref="$PROC/terelj_lidar"; else chm=$P1CHM; ref="$PROC/terelj_p1"; fi
    refDs="$ref/ds/chm"
    ds="$SITE_OUT/ds/$source"
    if [ ! -f "$refDs/dataset.json" ]; then log "$source: no first-area dataset $refDs; skipped"; continue; fi
    step "dataset$source" "$ds/dataset.json" python -m tt.dl prepare \
        --source "chm:$chm" --crowns "$CROWNS" --boundary "$AREA" \
        --output "$ds" --name "$SITE" \
        --resolution "$(datasetValue "$refDs" resolution)" \
        --tileSize "$(datasetValue "$refDs" tileSize)" \
        --overlap "$(datasetValue "$refDs" overlap)" \
        --minHeight "$(datasetValue "$refDs" minHeight)" || continue
    replay "cc$source" "$ref/runs/cc$source" "$ds" tt.dl concomp
    replay "lmvw$source" "$ref/runs/lmvw$source" "$ds" tt.dl.dlBaselines
    replay "watershed$source" "$ref/runs/watershed$source" "$ds" tt.dl.dlBaselines
done

if [ -f "$SITE_OUT/ds/Lidar/dataset.json" ] && [ -f "$SITE_OUT/ds/P1/dataset.json" ]; then
    if ! sameBlocks "$SITE_OUT/ds/Lidar" "$SITE_OUT/ds/P1"; then
        log "STOPPED: the LiDAR and P1 datasets have different blocks; no paired comparison"
        exit 1
    fi
    python -m tt.heightSourceCompare --first "$OUT" --second "$OUT" --sites "$SITE" \
        --output "$OUT/compare.json" | tee "$OUT/compare.txt"
    log "finished; comparison in compare.txt"
else
    for r in "$SITE_OUT"/runs/*/results.json; do
        [ -f "$r" ] && printf '%-14s %s\n' "$(basename "$(dirname "$r")")" \
            "$(python -c "import json,sys; p=json.load(open(sys.argv[1]))['pooled']; print('R %.3f  P %.3f  F1 %.3f' % (p['recall'], p['precision'], p['f1']))" "$r")"
    done | tee "$OUT/scores.txt"
    log "finished; scores in scores.txt"
fi
