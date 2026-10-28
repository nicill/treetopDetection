#!/usr/bin/env bash
#
# exp3Run.sh — experiment 3's early fusion: Mask R-CNN on four channels, the
# RGB mosaic and the P1 CHM stacked, at every site. The late fusions (RGB Mask
# R-CNN with CC, and with the CHM network) need no training: experiment3.py
# makes them from the saved runs.
#
#   DRYRUN=1 ./exp3Run.sh                          # what it would do
#   nohup ./exp3Run.sh > exp3Run.log 2>&1 &        # everything
#   ONLY="koiwai enshurinV2" ./exp3Run.sh          # some sites only
#
# Per site, beside its runs:
#   ds/rgbP1           the four-channel dataset: the RGB dataset's mosaic,
#                      crowns, area, resolution, tile size, overlap, minimum
#                      height and block grid, with the P1 CHM as the 4th
#                      channel; its blocks are checked to be the RGB dataset's
#   runs/mrcnnRgbP1    Mask R-CNN on it, with the RGB run's own settings
#                      (replayed from runs/mrcnnRgb/results.json), so the only
#                      difference from RGB alone is the fourth channel
# One line per step in exp3_STATUS.txt (in the Dropbox folder); a finished
# step is skipped when run again. One site after another, on the GPU.

set -u
cd "$(dirname "$0")" || exit 1
DROP=${DROP:-"$HOME/Yago Lab Dropbox/Meetings/2026TreeDet"}
DRYRUN=${DRYRUN:-0}
ONLY=${ONLY:-}
STATUS="$DROP/exp3_STATUS.txt"
export TT_OBJECTIVE=${TT_OBJECTIVE:-weighted} PYTHONUNBUFFERED=1
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATUS"; }
pooled() { grep -h "pooled over" "$1" 2>/dev/null | tail -1 | sed 's/^ *//'; }

# name | folder holding ds/ and runs/
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
    log "$label: started"
    if "$@" > "$logFile" 2>&1; then log "$label: done  $(pooled "$logFile")"
    else log "$label: FAILED, see $logFile"; return 1; fi
}

siteInfo() {   # shell assignments: the RGB dataset's settings, the P1 CHM
    python - "$1" <<'PY'
import json, os, shlex, sys
folder = sys.argv[1]
rgbDs = json.load(open(os.path.join(folder, "runs", "mrcnnRgb", "results.json")))["settings"]["dataset"]
chmDs = next(os.path.join(folder, "ds", d) for d in ("P1", "chm")
             if os.path.exists(os.path.join(folder, "ds", d, "dataset.json")))
r = json.load(open(os.path.join(rgbDs, "dataset.json")))
c = json.load(open(os.path.join(chmDs, "dataset.json")))
names = [b["name"] for b in r["blocks"]]
rows = 1 + max(int(n[n.rindex("_b") + 2]) for n in names)
cols = 1 + max(int(n[n.rindex("_b") + 3]) for n in names)
values = dict(rgbDs=rgbDs, chmDs=chmDs,
              rgbTif=next(ch["path"] for ch in r["channels"] if ch["kind"] == "rgb"),
              chmTif=next(ch["path"] for ch in c["channels"] if ch["kind"] == "chm"),
              crowns=r["crownsPath"], boundary=r["boundaryPath"], res=r["resolution"],
              tile=r["tileSize"], overlap=r["overlap"], minHeight=r["minHeight"],
              dsName=r["name"], cols=cols, rows=rows)
for key, value in values.items():
    print("%s=%s" % (key, shlex.quote(str(value))))
PY
}

sameBlocks() {   # sameBlocks DATASET_A DATASET_B: 0 when their blocks are the same
    python - "$1" "$2" <<'PY'
import json, sys
from shapely import wkt
def blocks(path):
    out = {}
    for b in json.load(open(path + "/dataset.json"))["blocks"]:
        out[b["name"][b["name"].rindex("_b"):]] = wkt.loads(b["wkt"])
    return out
a, b = blocks(sys.argv[1]), blocks(sys.argv[2])
same = set(a) == set(b) and all(a[k].symmetric_difference(b[k]).area <= 1e-6 * max(a[k].area, 1.0) for k in a)
print("same blocks (%d)" % len(a) if same else "DIFFERENT blocks: %s against %s" % (sorted(a), sorted(b)))
sys.exit(0 if same else 1)
PY
}

runSite() {
    local name=$1 folder=$2 info out run logs args
    logs="$folder/logs"; out="$folder/ds/rgbP1"; run="$folder/runs/mrcnnRgbP1"
    if [ ! -f "$folder/runs/mrcnnRgb/results.json" ]; then
        log "$name: no runs/mrcnnRgb in $folder; skipped"; return 0
    fi
    info=$(siteInfo "$folder") || { log "$name: could not read its datasets; skipped"; return 0; }
    eval "$info"
    mkdir -p "$logs"
    log "$name: RGB $rgbTif + CHM $chmTif, ${res} m, tiles $tile, blocks ${rows}x${cols}"
    step "$name/ds_rgbP1" "$logs/dataset_rgbP1.log" "$out/dataset.json" \
        python -m tt.dl prepare --source "rgb:$rgbTif+chm:$chmTif" --crowns "$crowns" \
        --boundary "$boundary" --output "$out" --name "$dsName" --resolution "$res" \
        --tileSize "$tile" --overlap "$overlap" --minHeight "$minHeight" \
        --blockCols "$cols" --blockRows "$rows" || return 0
    if [ "$DRYRUN" != 1 ]; then
        local check
        check=$(sameBlocks "$rgbDs" "$out") || { log "ERROR: $name: $check; not trained"; return 0; }
        log "$name: four-channel dataset has the RGB dataset's $check"
    fi
    args=$(python -m tt.replayRun "$folder/runs/mrcnnRgb/results.json" tt.dl maskrcnn 2>> "$logs/mrcnnRgbP1.replay") \
        || { log "$name: FAILED to replay runs/mrcnnRgb, see $logs/mrcnnRgbP1.replay"; return 0; }
    eval "set -- $args"
    step "$name/mrcnnRgbP1" "$logs/mrcnnRgbP1.log" "$run/results.json" \
        python -m tt.dl maskrcnn --dataset "$out" --output "$run" "$@"
}

log "experiment 3, four-channel runs started (dry run: $DRYRUN)"
while IFS='|' read -r name folder; do
    [ -n "$name" ] || continue
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]]; then continue; fi
    runSite "$name" "$folder"
done <<< "$SITES"
log "experiment 3, four-channel runs finished"
