#!/usr/bin/env bash
#
# exp23.sh — experiments 2 and 3 in one go, unattended:
#   1. experiment 2: RGB against CHM Mask R-CNN, per site and across sites
#      (experiment2.py)                         -> OUT/experiment2.txt, .json
#   2. experiment 3 now, the late fusions from the saved runs
#      (experiment3.py)                         -> OUT/experiment3_late.txt, .json
#   3. waits until no other network is training (the Terelj LiDAR networks),
#      then trains the 4-channel networks at every site (exp3Run.sh)
#   4. experiment 3 again, with the 4-channel rows -> OUT/experiment3.txt, .json
# One line per step in OUT/STATUS.txt; the machine is kept from sleeping
# while it runs. Steps 1 and 2 take minutes; step 3 about a day.
#
#   nohup ./exp23.sh > exp23.log 2>&1 &
#   SKIP_TRAINING=1 ./exp23.sh       # steps 1, 2 and 4 only
#
# Settings from the environment: DROP (the Dropbox folder), OUT (default
# DROP/exp23), SKIP_TRAINING.

set -u
cd "$(dirname "$0")" || exit 1
if [ -z "${EXP23_INHIBITED:-}" ] && systemd-inhibit --list > /dev/null 2>&1; then
    EXP23_INHIBITED=1 exec systemd-inhibit --what=idle:sleep --who=exp23 \
        --why="experiments 2 and 3" "$0" "$@"
fi
export DROP=${DROP:-"$HOME/Yago Lab Dropbox/Meetings/2026TreeDet"}
OUT=${OUT:-"$DROP/exp23"}
mkdir -p "$OUT"
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$OUT/STATUS.txt"; }

analysis() {   # analysis LABEL SCRIPT NAME: the script's output in OUT/NAME.txt and .json
    log "$1: started"
    if python "$2" --output "$OUT/$3.json" > "$OUT/$3.txt" 2>&1; then
        log "$1: done, $OUT/$3.txt"
    else
        log "$1: FAILED, see $OUT/$3.txt"
    fi
}

log "experiments 2 and 3 started"
analysis "experiment 2" experiment2.py experiment2
analysis "experiment 3, late fusions" experiment3.py experiment3_late

if [ -n "${SKIP_TRAINING:-}" ]; then
    log "4-channel training skipped (SKIP_TRAINING)"
else
    while pgrep -f "tt.dl maskrcnn|tt.dl yolo|tereljLidarRun.sh" > /dev/null; do
        [ -n "${waiting:-}" ] || log "waiting for the networks already training to finish"
        waiting=1
        sleep 300
    done
    log "4-channel networks: started (exp3Run.sh, progress in DROP/exp3_STATUS.txt)"
    ./exp3Run.sh >> "$OUT/exp3Run.log" 2>&1
    log "4-channel networks: finished"
fi
analysis "experiment 3, with the 4-channel networks" experiment3.py experiment3
log "experiments 2 and 3 finished; results in $OUT"
