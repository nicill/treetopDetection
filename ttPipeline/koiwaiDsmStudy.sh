#!/usr/bin/env bash
#
# koiwaiDsmStudy.sh — CC and the two height baselines (local maxima,
# watershed) on Koiwai's CHM built from Sergi's finer surface model
# (tt.prepare.dsmChm), with the settings of the current CHM's runs, and the two CHMs
# compared block by block.
#
#   N=Data/sergiDsm/koiwainoujo220616_dsm0.25 ./koiwaiDsmStudy.sh
#
# N is a site folder tt.prepare.dsmChm wrote (chm.tif and the site's layers, copied
# unchanged from the current site: the same crowns, the same scored area). The
# current runs are PROC/<REF>/runs/{cc,lmvw,watershed}P1.
#   ds/chm             the dataset at the current dataset's resolution, tile
#                      size, overlap and minimum height, with its name, so the
#                      blocks and their names are the same (checked): the
#                      comparison is paired
#   runs/ccP1, runs/lmvwP1, runs/watershedP1
#                      each with its current run's own settings, replayed from
#                      its results.json: only the dataset and output change,
#                      so only the surface differs from the current runs
# and at the end tt.heightSourceCompare, new CHM against current for each
# method (paired Wilcoxon over blocks), and CC against the better baseline on
# each. With the 0.25 m CHM the resolution matches the current runs; a finer
# CHM would need the settings' resolution changed too, which this script does
# not do (it would be resampled to the dataset's resolution). A finished step
# is skipped when the script is run again.
#
# Settings from the environment: DROP, PROC, REF (default koiwainoujo220616),
# N (required), OUT (default study_koiwaiDsm<res>), JOBS (default 16).

set -u
cd "$(dirname "$0")" || exit 1

DROP=${DROP:-"$HOME/Yago Lab Dropbox/Meetings/2026TreeDet"}
PROC=${PROC:-"$DROP/study_procedure"}
REF=${REF:-koiwainoujo220616}
N=${N:?"set N to the tt.prepare.dsmChm site folder, e.g. Data/sergiDsm/koiwainoujo220616_dsm0.25"}
OUT=${OUT:-"$DROP/study_koiwaiDsm_$(basename "$N")"}
JOBS=${JOBS:-16}
export TT_OBJECTIVE=${TT_OBJECTIVE:-weighted} PYTHONUNBUFFERED=1

SITE_OUT="$OUT/$REF"
mkdir -p "$SITE_OUT/logs" "$SITE_OUT/runs"
STATUS="$OUT/STATUS.txt"
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS"; }
pooled() { grep -h "pooled over" "$1" 2>/dev/null | tail -1 | sed 's/^ *//'; }

for f in "$N/chm.tif" "$N/crowns.shp" "$N/scoredArea.shp" "$PROC/$REF/runs/ccP1/results.json"; do
    [ -e "$f" ] || { log "ERROR: $f missing; stopping"; exit 1; }
done

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

# the current dataset, from the current CC run's own record
refDs=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['settings']['dataset'])" "$PROC/$REF/runs/ccP1/results.json")
[ -f "$refDs/dataset.json" ] || refDs="$PROC/$REF/ds/chm"
[ -f "$refDs/dataset.json" ] || { log "ERROR: the current dataset not found ($refDs)"; exit 1; }
log "current dataset: $refDs; new CHM: $N"

step dataset "$SITE_OUT/ds/chm/dataset.json" python -m tt.dl prepare \
    --source "chm:$N/chm.tif" --crowns "$N/crowns.shp" \
    --boundary "$N/scoredArea.shp" --output "$SITE_OUT/ds/chm" \
    --name "$(datasetValue "$refDs" name)" \
    --resolution "$(datasetValue "$refDs" resolution)" \
    --tileSize "$(datasetValue "$refDs" tileSize)" \
    --overlap "$(datasetValue "$refDs" overlap)" \
    --minHeight "$(datasetValue "$refDs" minHeight)" || exit 1
if ! sameBlocks "$refDs" "$SITE_OUT/ds/chm"; then
    log "STOPPED: the new dataset's blocks differ from the current one's; no paired comparison"
    exit 1
fi

replay() {   # replay NAME MODULE...: the current run again on the new dataset
    local name=$1; shift
    local ref="$PROC/$REF/runs/$name" args
    [ -f "$ref/results.json" ] || { log "$name: no $ref; skipped"; return 0; }
    args=$(replayArgs "$ref/results.json" "$@") || { log "$name: FAILED to read $ref"; return 1; }
    eval "set -- \"\$@\" $args"
    step "$name" "$SITE_OUT/runs/$name/results.json" python -m "$@" \
        --dataset "$SITE_OUT/ds/chm" --output "$SITE_OUT/runs/$name" --jobs "$JOBS"
}

replay ccP1 tt.dl concomp
replay lmvwP1 tt.dl.dlBaselines
replay watershedP1 tt.dl.dlBaselines

python -m tt.heightSourceCompare --first "$PROC" --second "$OUT" --sites "$REF" \
    --firstHeight P1 --secondHeight P1 --output "$OUT/compare.json" | tee "$OUT/compare.txt"
log "finished; new against current CHM in compare.txt (diff is new minus current)"
