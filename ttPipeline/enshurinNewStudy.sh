#!/usr/bin/env bash
#
# enshurinNewStudy.sh — CC and the two height baselines (local maxima,
# watershed) on Enshurin with Vlad's new annotation, with the settings of the
# old Enshurin runs, and the old and new scores side by side.
#
#   ./enshurinNewStudy.sh
#
# N is the site tt.prepare.enshurin --crowns wrote (chm.tif, crowns.shp,
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
# OVERRIDES="minHeights=4.0 minHeight=4.0" replaces those options of every
# replayed run that has them (and only those): Vlad's rule that a tree is at
# least 4 m tall, for the height methods, whose minimum height is otherwise a
# tuned grid.
#
# PART chooses what runs:
#   height     (the default) the CHM dataset, CC and the two baselines, on CPU
#   networks   the RGB dataset and every network run of the old study
#              (PROC/enshurin/runs/mrcnn*, yolo*), each with its own settings,
#              on the RGB or the CHM dataset as before; on the GPU. It waits
#              for the CHM dataset the height part makes, so the two can run
#              at the same time into the same OUT.
#   all        both, one after the other
# Runs of the old study made for other experiments (a transfer, or a subset
# of the annotation) are not replayed.
#
# Settings from the environment: DROP, PROC, N, OUT, JOBS (default 16),
# OVERRIDES (default none), PART (default height).

set -u
cd "$(dirname "$0")" || exit 1

DROP=${DROP:-"$HOME/Yago Lab Dropbox/Meetings/2026TreeDet"}
PROC=${PROC:-"$DROP/study_procedure"}
N=${N:-Data/enshurinNewOut/enshurin}
OUT=${OUT:-"$DROP/study_enshurinNew"}
JOBS=${JOBS:-16}
PART=${PART:-height}
export TT_OBJECTIVE=${TT_OBJECTIVE:-weighted} PYTHONUNBUFFERED=1

mkdir -p "$OUT/logs" "$OUT/runs"
STATUS="$OUT/STATUS.txt"
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS"; }
pooled() { grep -h "pooled over" "$1" 2>/dev/null | tail -1 | sed 's/^ *//'; }

for f in "$N/chm.tif" "$N/rgb.tif" "$N/crowns.shp" "$N/scoredArea.shp" "$PROC/enshurin/runs/ccP1/results.json"; do
    [ -e "$f" ] || { log "ERROR: $f missing; stopping"; exit 1; }
done

replayArgs() {   # replayArgs RESULTS MODULE...: a run's own options, quoted
    python - "$@" <<'PY'
import json, shlex, subprocess, sys
import os
settings = json.load(open(sys.argv[1]))["settings"]
for pair in os.environ.get("OVERRIDES", "").split():
    key, value = pair.split("=", 1)
    if key in settings:          # only the options this run has
        settings[key] = value
helpText = subprocess.run([sys.executable, "-m"] + sys.argv[2:] + ["--help"],
                          capture_output=True, text=True).stdout
skip = {"dataset", "output", "log", "jobs", "progress", "folds"}
negated = {"pretrained": "noPretrained"}       # store_false options
words = []
for key, value in settings.items():
    if key in negated:
        if value is False:
            words.append("--" + negated[key])
        continue
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

replay() {   # replay NAME DATASET EXTRA MODULE...: an old run on a new dataset
    local name=$1 dataset=$2 extra=$3; shift 3
    local ref="$PROC/enshurin/runs/$name" args
    [ -f "$ref/results.json" ] || { log "$name: no $ref; skipped"; return 0; }
    args=$(replayArgs "$ref/results.json" "$@") || { log "$name: FAILED to read $ref"; return 1; }
    eval "set -- \"\$@\" $args $extra"
    step "$name" "$OUT/runs/$name/results.json" python -m "$@" \
        --dataset "$dataset" --output "$OUT/runs/$name"
}

oldDataset() {   # oldDataset KIND: the old study's dataset of that kind: the
                 # one its main run recorded (ccP1, mrcnnRgb), else any run's,
                 # else PROC/enshurin/ds/KIND
    local d
    while IFS= read -r d; do
        [ -f "$d/dataset.json" ] && { echo "$d"; return 0; }
    done < <(python - "$PROC/enshurin" "$1" <<'PY'
import glob, json, os, sys
root, kind = sys.argv[1:]
main = {"chm": "ccP1", "rgb": "mrcnnRgb"}[kind]
paths = sorted(glob.glob(root + "/runs/*/results.json"),
               key=lambda p: os.path.basename(os.path.dirname(p)) != main)
for path in paths:
    d = json.load(open(path)).get("settings", {}).get("dataset") or ""
    if os.path.basename(os.path.normpath(d)) == kind:
        print(d)
print(os.path.join(root, "ds", kind))
PY
)
    return 1
}

newDataset() {   # newDataset KIND SOURCE: the dataset on the new crowns, with
                 # the old one's resolution, tile size, overlap, minimum height
    local kind=$1 source=$2 ref
    ref=$(oldDataset "$kind") || { log "ERROR: the old Enshurin $kind dataset not found"; return 1; }
    log "old $kind dataset: $ref"
    step "dataset_$kind" "$OUT/ds/$kind/dataset.json" python -m tt.dl prepare \
        --source "$kind:$source" --crowns "$N/crowns.shp" \
        --boundary "$N/scoredArea.shp" --output "$OUT/ds/$kind" --name enshurin \
        --resolution "$(datasetValue "$ref" resolution)" \
        --tileSize "$(datasetValue "$ref" tileSize)" \
        --overlap "$(datasetValue "$ref" overlap)" \
        --minHeight "$(datasetValue "$ref" minHeight)" || return 1
    local old new
    old=$(datasetValue "$ref" blocks); new=$(datasetValue "$OUT/ds/$kind" blocks)
    [ "$old" = "$new" ] || log "NOTE: the old $kind dataset had $old blocks, the new one $new"
}

heightPart() {
    newDataset chm "$N/chm.tif" || exit 1
    replay ccP1 "$OUT/ds/chm" "--jobs $JOBS" tt.dl concomp
    replay lmvwP1 "$OUT/ds/chm" "--jobs $JOBS" tt.dl.dlBaselines
    replay watershedP1 "$OUT/ds/chm" "--jobs $JOBS" tt.dl.dlBaselines
}

networkRuns() {   # the old study's network runs: NAME KIND MODULE per line
    python - "$PROC/enshurin/runs" <<'PY'
import glob, json, os, re, sys
for path in sorted(glob.glob(sys.argv[1] + "/*/results.json")):
    name = os.path.basename(os.path.dirname(path))
    if not re.match(r"(mrcnn|yolo)", name):
        continue
    settings = json.load(open(path)).get("settings", {})
    if settings.get("transferFrom") or float(settings.get("trainFraction", 1)) < 1:
        continue                          # another experiment's run
    kind = os.path.basename(os.path.normpath(settings.get("dataset", "")))
    if kind not in ("rgb", "chm"):
        continue
    print(name, kind, "yolo" if name.startswith("yolo") else "maskrcnn")
PY
}

networkPart() {
    newDataset rgb "$N/rgb.tif" || exit 1
    local waited=0
    until [ -f "$OUT/ds/chm/dataset.json" ]; do      # the height part makes it
        [ "$PART" = networks ] || break
        [ $waited -ge 7200 ] && { log "ERROR: no CHM dataset after 2 h; is the height part running?"; exit 1; }
        sleep 60; waited=$((waited + 60))
    done
    [ -f "$OUT/ds/chm/dataset.json" ] || newDataset chm "$N/chm.tif" || exit 1
    local name kind module
    while read -r name kind module; do
        replay "$name" "$OUT/ds/$kind" "" tt.dl "$module"
    done < <(networkRuns)
}

case "$PART" in
    height) heightPart ;;
    networks) networkPart ;;
    all) heightPart; networkPart ;;
    *) log "ERROR: PART must be height, networks or all"; exit 1 ;;
esac

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
names = sorted(n for n in os.listdir(new)
               if os.path.exists(os.path.join(new, n, "results.json")))
for name in names:
    a, b = pooled(old, name), pooled(new, name)
    fmt = lambda v: "%6.3f %6.3f %6.3f" % v if v else "%20s" % "-"
    print("%-14s %s   %s" % (name, fmt(a), fmt(b)))
PY
log "finished; old vs new in old_vs_new.txt"
