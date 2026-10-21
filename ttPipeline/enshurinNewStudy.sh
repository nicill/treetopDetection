#!/usr/bin/env bash
#
# enshurinNewStudy.sh — CC and the two height baselines (local maxima,
# watershed) on Enshurin with Vlad's new annotation, with the settings of the
# old Enshurin runs, and the old and new scores side by side.
#
#   ./enshurinNewStudy.sh
#
# N is the site tt.enshurinPrepare --crowns wrote (chm.tif, crowns.shp,
# scoredArea.shp). The old runs are PROC/enshurin/runs/{cc,lmvw,watershed}P1.
#   ds/chm             the dataset at the old dataset's resolution, tile size,
#                      overlap and minimum height, on the new crowns and the
#                      new scored area (4 x 2 blocks, as tt.dl prepare makes
#                      them by default; the old dataset's number of blocks is
#                      compared and a difference reported)
#   runs/ccP1, runs/lmvwP1, runs/watershedP1
#                      each with its old run's own settings (grid, resolution,
#                      minimum heights, smoothing, ...), replayed from that
#                      run's results.json: only dataset and output change
# The old and new scores are on different crowns and areas, so they are
# printed side by side (R, P, F1) and not tested against each other. A
# finished step is skipped when the script is run again.
#
# Settings from the environment: DROP, PROC, N, OUT, JOBS (default 16).

set -u
cd "$(dirname "$0")" || exit 1

DROP=${DROP:-"$HOME/Yago Lab Dropbox/Meetings/2026TreeDet"}
PROC=${PROC:-"$DROP/study_procedure"}
N=${N:-Data/enshurinNewOut/enshurin}
OUT=${OUT:-"$DROP/study_enshurinNew"}
JOBS=${JOBS:-16}
export TT_OBJECTIVE=${TT_OBJECTIVE:-weighted} PYTHONUNBUFFERED=1

mkdir -p "$OUT/logs" "$OUT/runs"
STATUS="$OUT/STATUS.txt"
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS"; }
pooled() { grep -h "pooled over" "$1" 2>/dev/null | tail -1 | sed 's/^ *//'; }

for f in "$N/chm.tif" "$N/crowns.shp" "$N/scoredArea.shp" "$PROC/enshurin/runs/ccP1/results.json"; do
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
    python -c "import json,sys; d=json.load(open(sys.argv[1]+'/dataset.json')); print(len(d[sys.argv[2]]) if sys.argv[2]=='blocks' else d[sys.argv[2]])" "$1" "$2"
}

step() {   # step NAME DONE_FILE COMMAND...: run unless DONE_FILE exists
    local name=$1 done=$2; shift 2
    local logFile="$OUT/logs/$name.log"
    if [ -e "$done" ]; then log "$name: already done"; return 0; fi
    log "$name: started"
    if "$@" > "$logFile" 2>&1; then
        log "$name: done  $(pooled "$logFile")"
    else
        log "$name: FAILED, see logs/$name.log"
        return 1
    fi
}

# the old dataset, from the old CC run's own record
refDs=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['settings']['dataset'])" "$PROC/enshurin/runs/ccP1/results.json")
[ -f "$refDs/dataset.json" ] || refDs="$PROC/enshurin/ds/chm"
[ -f "$refDs/dataset.json" ] || { log "ERROR: the old Enshurin dataset not found ($refDs)"; exit 1; }
log "old dataset: $refDs"

step dataset "$OUT/ds/chm/dataset.json" python -m tt.dl prepare \
    --source "chm:$N/chm.tif" --crowns "$N/crowns.shp" \
    --boundary "$N/scoredArea.shp" --output "$OUT/ds/chm" --name enshurin \
    --resolution "$(datasetValue "$refDs" resolution)" \
    --tileSize "$(datasetValue "$refDs" tileSize)" \
    --overlap "$(datasetValue "$refDs" overlap)" \
    --minHeight "$(datasetValue "$refDs" minHeight)" || exit 1
oldBlocks=$(datasetValue "$refDs" blocks); newBlocks=$(datasetValue "$OUT/ds/chm" blocks)
[ "$oldBlocks" = "$newBlocks" ] || log "NOTE: the old dataset had $oldBlocks blocks, the new one $newBlocks"

replay() {   # replay NAME MODULE...: the old run again on the new dataset
    local name=$1; shift
    local ref="$PROC/enshurin/runs/$name" args
    [ -f "$ref/results.json" ] || { log "$name: no $ref; skipped"; return 0; }
    args=$(replayArgs "$ref/results.json" "$@") || { log "$name: FAILED to read $ref"; return 1; }
    eval "set -- \"\$@\" $args"
    step "$name" "$OUT/runs/$name/results.json" python -m "$@" \
        --dataset "$OUT/ds/chm" --output "$OUT/runs/$name" --jobs "$JOBS"
}

replay ccP1 tt.dl concomp
replay lmvwP1 tt.dl.dlBaselines
replay watershedP1 tt.dl.dlBaselines

python - "$PROC/enshurin/runs" "$OUT/runs" <<'PY' | tee "$OUT/old_vs_new.txt"
import json, os, sys
old, new = sys.argv[1:]
def pooled(root, name):
    path = os.path.join(root, name, "results.json")
    if not os.path.exists(path):
        return None
    p = json.load(open(path))["pooled"]
    return p["recall"], p["precision"], p["f1"]
print("%-14s %-20s %-20s" % ("", "old annotation", "new annotation"))
print("%-14s %6s %6s %6s   %6s %6s %6s" % ("method", "R", "P", "F1", "R", "P", "F1"))
for name in ("ccP1", "lmvwP1", "watershedP1"):
    a, b = pooled(old, name), pooled(new, name)
    fmt = lambda v: "%6.3f %6.3f %6.3f" % v if v else "%20s" % "-"
    print("%-14s %s   %s" % (name, fmt(a), fmt(b)))
PY
log "finished; old vs new in old_vs_new.txt"
