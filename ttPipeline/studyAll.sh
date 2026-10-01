#!/usr/bin/env bash
#
# studyAll.sh — siteStudy.sh on Sergi's site and on every usable Quebec site,
# one after the other, with the cross-site summary refreshed after each.
#
#   ./studyAll.sh                  # everything
#   ONLY=sergi ./studyAll.sh       # Sergi only;  ONLY=quebec for Quebec only
#   DRYRUN=1 ./studyAll.sh         # the whole chain without training
#
# Output, one folder per site, under $STUDY (in Dropbox, so it can be followed
# and continued from another computer: a finished stage is never redone):
#   $STUDY/<site>/STATUS.txt       stage by stage, the file to open first
#   $STUDY/STATUS.txt              all sites together
#   $STUDY/summary/                the cross-site tables (tt.studySummary)
#
# Quebec sites left out: the two serpentin plots (incomplete annotation) and
# afcagauthier (no tree heights in half of its LiDAR). Sites not yet prepared
# are skipped and taken up by the next run of this script.

set -u
cd "$(dirname "$0")"

DROP=${DROP:-"$HOME/Yago Lab Dropbox/Meetings/2026TreeDet"}
QP=${QP:-"$DROP/Quebec2"}
STUDY=${STUDY:-"$DROP/study"}
SERGI_SITE=${SERGI_SITE:-Data/sergiOut/koiwainoujo220616}
PDAL=${PDAL:-"$HOME/anaconda3/envs/pdal/bin/pdal"}
ONLY=${ONLY:-all}

QUEBEC_SITES="20230712_afcamoisan_itrf20 20230712_afcahoule_itrf20
  20230606_cbblackburn4 20230606_cbblackburn5 20230608_cbbernard4
  20230608_cbbernard3 20230712_afcagauthmelpin_itrf20 20230605_cbblackburn1
  20230607_cbblackburn2 20230608_cbpapinas 20230606_cbblackburn6
  20230608_cbbernard1 20230606_cbblackburn3 20230608_cbbernard2"

mkdir -p "$STUDY"
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$STUDY/STATUS.txt"; }

summary() {
    python -m tt.studySummary --root "$STUDY" --output "$STUDY/summary" \
        > "$STUDY/summary/summary.txt" 2>&1 || true
}

runSite() {   # runSite SITE_DIR: the study, its status lines copied up
    local name; name=$(basename "$1")
    log "$name: study started"
    ./siteStudy.sh "$1" "$STUDY/$name"
    log "$name: study ended (exit $?); $(tail -1 "$STUDY/$name/STATUS.txt" | cut -c22-)"
    mkdir -p "$STUDY/summary"; summary
}

log "start (ONLY=$ONLY DRYRUN=${DRYRUN:-0})"

if [ "$ONLY" != quebec ]; then
    if [ -f "$SERGI_SITE/check.json" ]; then
        # Sergi: larch, photogrammetric height, crowns like Terelj's: the
        # Terelj P1 settings and grid
        HEIGHT=P1 RES=0.05 CCRES=0.25 MINH=2.0 MINTREE=0.5 \
        CCGRID="--percentiles 10,20 --minTopAreas 0.25,0.12 --topSteps 0.12,0.25 --erosions 1,2 --saddleDrops 0.2,0.3,0.5" \
            runSite "$SERGI_SITE"
    else
        log "sergi: $SERGI_SITE not prepared (no check.json); skipped"
    fi
fi

if [ "$ONLY" != sergi ]; then
    # the scored crowns and areas must include the invisible-crown rule, which
    # sites prepared before it lack; a recheck takes seconds and reads nothing
    # from the server
    python -m tt.qpPrepare --vectors "$QP/Vector_Data" --pdal "$PDAL" \
        --output "$QP" --recheck > "$STUDY/recheck.log" 2>&1 \
        && log "quebec: prepared sites rechecked" \
        || log "quebec: recheck FAILED, see recheck.log"
    for site in $QUEBEC_SITES; do
        if [ -f "$QP/$site/check.json" ]; then
            runSite "$QP/$site"
        else
            log "$site: not prepared yet; skipped"
        fi
    done
fi

log "finished; tables in $STUDY/summary/summary.txt"
