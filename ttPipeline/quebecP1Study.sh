#!/usr/bin/env bash
#
# quebecP1Study.sh — CC and the two height baselines (local maxima, watershed)
# on the Quebec photogrammetric CHMs (tt.qpPhotogrammetry), exactly as they
# were run on the LiDAR CHMs, so the two can be compared site by site.
#
#   ./quebecP1Study.sh
#
# For each site with a photogrammetric CHM (QP1/<site>/check.json) and a
# finished LiDAR run (PROC/<site>/runs/ccLidar/results.json):
#   ds/chm             the dataset at the LiDAR dataset's resolution, tile size,
#                      overlap and minimum height, on the same crowns and area;
#                      its blocks must equal the LiDAR dataset's, or the site
#                      stops (a paired comparison needs the same blocks)
#   runs/ccP1          CC with the LiDAR run's own settings (grid, resolution,
#                      minimum height and tree area, smoothing, ...), replayed
#                      from its results.json: only dataset and output change
#   runs/lmvwP1        local maxima, likewise replayed from runs/lmvwLidar
#   runs/watershedP1   watershed, from runs/watershedLidar
# Output in OUT/<site>/ (a folder of its own: the LiDAR study is untouched),
# STATUS.txt there, and a table of F1 per site and method, LiDAR beside
# photogrammetry, at the end. A finished step is skipped when run again.
#
# Settings from the environment: DROP, PROC, QP1, OUT, JOBS (default 16).

set -u
cd "$(dirname "$0")" || exit 1

DROP=${DROP:-"$HOME/Yago Lab Dropbox/Meetings/2026TreeDet"}
PROC=${PROC:-"$DROP/study_procedure"}
QP1=${QP1:-"$DROP/Quebec2P1"}
OUT=${OUT:-"$DROP/study_procedureP1"}
JOBS=${JOBS:-16}
export TT_OBJECTIVE=${TT_OBJECTIVE:-weighted} PYTHONUNBUFFERED=1

mkdir -p "$OUT"
STATUS="$OUT/STATUS.txt"
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS"; }
pooled() { grep -h "pooled over" "$1" 2>/dev/null | tail -1 | sed 's/^ *//'; }

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
    local logFile="$site_out/logs/$name.log"
    if [ -e "$done" ]; then log "$site/$name: already done"; return 0; fi
    log "$site/$name: started"
    if "$@" > "$logFile" 2>&1; then
        log "$site/$name: done  $(pooled "$logFile")"
    else
        log "$site/$name: FAILED, see logs/$name.log"
        return 1
    fi
}

prepareDataset() {
    local lidarDs=$1 ds=$2
    step dataset "$ds/dataset.json" python -m tt.dl prepare \
        --source "chm:$QP1/$site/chm.tif" --crowns "$QP1/$site/crowns.shp" \
        --boundary "$QP1/$site/scoredArea.shp" --output "$ds" --name "$site" \
        --resolution "$(datasetValue "$lidarDs" resolution)" \
        --tileSize "$(datasetValue "$lidarDs" tileSize)" \
        --overlap "$(datasetValue "$lidarDs" overlap)" \
        --minHeight "$(datasetValue "$lidarDs" minHeight)" || return 1
    if ! sameBlocks "$lidarDs" "$ds"; then
        log "$site: STOPPED, blocks differ from the LiDAR dataset's"
        return 1
    fi
}

replay() {   # replay NAME LIDAR_RUN MODULE...: the LiDAR run again on ds/chm
    local name=$1 lidarRun=$2; shift 2
    local args
    [ -f "$lidarRun/results.json" ] || { log "$site/$name: no $lidarRun, skipped"; return 0; }
    args=$(replayArgs "$lidarRun/results.json" "$@") || { log "$site/$name: FAILED to read $lidarRun"; return 1; }
    eval "set -- \"\$@\" $args"
    step "$name" "$site_out/runs/$name/results.json" python -m "$@" \
        --dataset "$site_out/ds/chm" --output "$site_out/runs/$name" --jobs "$JOBS"
}

f1() {   # f1 RESULTS: pooled F1, or - when missing
    python -c "import json,sys; print('%.3f' % json.load(open(sys.argv[1]))['pooled']['f1'])" "$1" 2>/dev/null || echo "-"
}

log "start (PROC=$PROC QP1=$QP1 OUT=$OUT OBJECTIVE=$TT_OBJECTIVE)"
for dir in "$QP1"/*/; do
    site=$(basename "$dir")
    lidar="$PROC/$site"
    site_out="$OUT/$site"
    if [ ! -f "$dir/check.json" ]; then log "$site: photogrammetric CHM not ready, skipped"; continue; fi
    if [ ! -f "$lidar/runs/ccLidar/results.json" ]; then log "$site: no LiDAR CC run in $PROC, skipped"; continue; fi
    mkdir -p "$site_out/logs" "$site_out/runs"
    prepareDataset "$lidar/ds/chm" "$site_out/ds/chm" || continue
    replay ccP1 "$lidar/runs/ccLidar" tt.dl concomp
    replay lmvwP1 "$lidar/runs/lmvwLidar" tt.dl.dlBaselines
    replay watershedP1 "$lidar/runs/watershedLidar" tt.dl.dlBaselines
done

{
    printf '\n%-28s %9s %9s %9s %9s %9s %9s\n' site ccLidar ccP1 lmvwLidar lmvwP1 wsLidar wsP1
    for dir in "$OUT"/*/; do
        site=$(basename "$dir")
        [ -d "$dir/runs" ] || continue
        printf '%-28s %9s %9s %9s %9s %9s %9s\n' "$site" \
            "$(f1 "$PROC/$site/runs/ccLidar/results.json")" "$(f1 "$dir/runs/ccP1/results.json")" \
            "$(f1 "$PROC/$site/runs/lmvwLidar/results.json")" "$(f1 "$dir/runs/lmvwP1/results.json")" \
            "$(f1 "$PROC/$site/runs/watershedLidar/results.json")" "$(f1 "$dir/runs/watershedP1/results.json")"
    done
} | tee "$OUT/f1.txt"
log "finished; F1 table in f1.txt"
