#!/usr/bin/env bash
#
# checksAndShutdown.sh — the comparisons and checks that need no decision,
# then the computer is switched off.
#
#   nohup ./checksAndShutdown.sh > checks.log 2>&1 &
#
#  1. tt.runCompare, CC against each other run at every site where they share
#     blocks: pooled F1 difference with a 95% interval over blocks, and the
#     per-block test with a Holm adjustment:
#       Enshurin (new annotation): local maxima, watershed, both networks
#       Quebec, each site (photogrammetric CHM): local maxima, watershed, the
#         CHM network, and the RGB network (unchanged, from study_procedure)
#       Koiwai (the DSM CHM): local maxima, watershed, the RGB network
#  2. tt.candidateTrees --validate: the proposal method inside Vlad's own
#     crowns, where the answer is known, at prominences 0.5, 0.75 and 1 m
#  3. a summary of both, then waits for Dropbox to finish syncing (at most
#     30 minutes) and switches the computer off (NO_SHUTDOWN=1 to keep it on)
#
# Everything goes to Dropbox, OUT (default 2026TreeDet/checks_<date>), so it
# can be read from another computer once this one is off.

set -u
cd "$(dirname "$0")" || exit 1
DB=${DB:-"$HOME/Yago Lab Dropbox/Meetings/2026TreeDet"}
OUT=${OUT:-"$DB/checks_$(date +%m%d)"}
ENS=${ENS:-"$DB/study_enshurinNew/runs"}
PROC=${PROC:-"$DB/study_procedure"}
QP1=${QP1:-"$DB/study_procedureP1"}
KOIWAI=${KOIWAI:-"$(ls -d "$DB"/study_koiwaiDsm_*/koiwainoujo220616 2>/dev/null | head -1)"}
CHM=${CHM:-Data/vlad/enshurin_chm025.tif}
POLYGONS=${POLYGONS:-Data/enshurinNewOut/enshurin_all_shifted.shp}
MOSAICS=${MOSAICS:-"Data/enshurinNewOut/enshurin/shifted/*.vrt"}
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$SUMMARY"; }

compareSite() {   # compareSite LABEL REFERENCE OTHER...: skips missing runs
    local label=$1 ref=$2; shift 2
    local others=()
    for o in "$@"; do [ -f "$o/results.json" ] && others+=("$o"); done
    if [ ! -f "$ref/results.json" ] || [ ${#others[@]} -eq 0 ]; then
        log "compare $label: runs missing, skipped"; return 0
    fi
    { echo; echo "== $label"; } >> "$OUT/compare.txt"
    python -m tt.runCompare --reference "$ref" --others "${others[@]}" \
        --output "$OUT/compare_$label.json" >> "$OUT/compare.txt" 2>&1 \
        && log "compare $label: done" || log "compare $label: FAILED"
}

log "checks started"
: > "$OUT/compare.txt"
compareSite enshurin "$ENS/ccP1" "$ENS/lmvwP1" "$ENS/watershedP1" "$ENS/mrcnnRgb" "$ENS/mrcnnP1"
for dir in "$QP1"/*/; do
    site=$(basename "$dir")
    [ -d "$dir/runs" ] || continue
    compareSite "$site" "$dir/runs/ccP1" "$dir/runs/lmvwP1" "$dir/runs/watershedP1" \
        "$dir/runs/mrcnnP1" "$PROC/$site/runs/mrcnnRgb"
done
[ -n "$KOIWAI" ] && compareSite koiwaiDsm "$KOIWAI/runs/ccP1" "$KOIWAI/runs/lmvwP1" \
    "$KOIWAI/runs/watershedP1" "$PROC/koiwainoujo220616/runs/mrcnnRgb"
sed -i 's#'"$DB"'/##g' "$OUT/compare.txt"   # shorter paths in the table

for p in 0.5 0.75 1.0; do
    if python -m tt.candidateTrees --validate --chm "$CHM" --polygons "$POLYGONS" \
            --mosaics "$MOSAICS" --minProminenceM "$p" \
            --output "$OUT/validate_p$p" > "$OUT/validate_p$p.log" 2>&1; then
        log "validation, prominence $p m: $(grep -h '\[validate\]' "$OUT/validate_p$p.log" | cut -c12-)"
    else
        log "validation, prominence $p m: FAILED, see validate_p$p.log"
    fi
done
log "checks finished; comparisons in compare.txt"

if [ "${NO_SHUTDOWN:-0}" = 1 ]; then log "NO_SHUTDOWN set: staying on"; exit 0; fi
if command -v dropbox > /dev/null; then
    for i in $(seq 1 30); do
        dropbox status 2>/dev/null | grep -qi "up to date" && break
        sleep 60
    done
    log "Dropbox: $(dropbox status 2>/dev/null | head -1)"
else
    log "no dropbox command; waiting 10 minutes for the sync"; sleep 600
fi
log "switching off"
sync
systemctl poweroff || sudo -n poweroff || log "could not switch off (no permission)"
