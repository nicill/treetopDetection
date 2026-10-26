"""
One report over every study: each site's runs under the current rules next
to the earlier ones, the comparisons of experiments 1 and 2, and every place
where a conclusion changed between the earlier runs and the current ones.

    python -m tt.report.experiments --db "$DB" --output "$DB/report.txt"

Sites and where their runs are (missing runs are skipped):
  Quebec (five sites)  now study_edge (edge-only cut), height maps Lidar and P1;
                       before study_procedure (LiDAR) and study_procedureP1
  Koiwai               now study_edge, height maps P1 (current CHM) and Dsm;
                       before study_procedure and study_koiwaiDsm_*
  Terelj areas 1-3     now study_terelj{First,Second,Third} (Burmaa's ROIs),
                       P1; before study_procedure/terelj_p1 (area 1) and
                       study_tereljMore (area 2, derived area)
  Enshurin             now study_enshurinFinal if it exists (whole mosaics),
                       else study_enshurinNew (new annotation, usual rule);
                       before study_procedure/enshurin

Experiment 1: each method on one height map against the other (P1 minus
LiDAR; Dsm minus the current CHM at Koiwai), and CC against the better of the
two baselines on each map. Experiment 2: the RGB network against the CHM
network on each map, and CC against the better network. Every comparison is
tt.runCompare's: the pooled F1 difference (trees weighted) with a 95%
bootstrap interval over blocks, and how many blocks the first run wins. A
comparison is marked CHANGED when its conclusion differs between the earlier
and the current runs: level (the interval includes 0) against one run clearly
ahead, or the other run ahead. The better baseline and the better network are
chosen on the current runs, and the earlier runs are compared on the same
pair.
Experiment 3 (the combinations) has no runs under the current rules yet.
"""

import argparse
import glob
import os
import sys

import numpy as np

from ..runCompare import compare, loadRun

QUEBEC = ("20230712_afcamoisan_itrf20", "20230712_afcahoule_itrf20",
          "20230606_cbblackburn5", "20230608_cbbernard4",
          "20230606_cbblackburn6")
METHODS = ("cc", "lmvw", "watershed", "mrcnn")


def sites(db):
    """(name, height maps, now(run) -> folder, before(run) -> folder)."""
    def at(*parts):
        return os.path.join(db, *parts)
    out = []
    for s in QUEBEC:
        def before(r, s=s):
            if r == "mrcnnRgb" or not r.endswith("P1"):
                return at("study_procedure", s, "runs", r)
            return at("study_procedureP1", s, "runs", r)
        out.append((s, ("Lidar", "P1"),
                    lambda r, s=s: at("study_edge", s, "runs", r), before))
    k = "koiwainoujo220616"
    dsm = (glob.glob(at("study_koiwaiDsm_*", k, "runs")) or [""])[0]
    out.append(("koiwai", ("P1", "Dsm"),
                lambda r: at("study_edge", k, "runs", r),
                lambda r: os.path.join(dsm, r[:-3] + "P1") if r.endswith("Dsm")
                else at("study_procedure", k, "runs", r)))
    for area, old in (("tereljFirst", at("study_procedure", "terelj_p1", "runs")),
                      ("tereljSecond", at("study_tereljMore", "tereljMore", "runs")),
                      ("tereljThird", None)):
        out.append((area, ("P1",),
                    lambda r, a=area: at("study_" + a, a, "runs", r),
                    lambda r, o=old: os.path.join(o, r) if o else ""))
    final = at("study_enshurinFinal", "runs")
    now = final if os.path.isdir(final) else at("study_enshurinNew", "runs")
    out.append(("enshurin" + (" (final)" if now == final else " (new annotation)"),
                ("P1",), lambda r, n=now: os.path.join(n, r),
                lambda r: at("study_procedure", "enshurin", "runs", r)))
    return out


def load(path):
    if not path or not os.path.exists(os.path.join(path, "results.json")):
        return None
    pooled, folds = loadRun(None, path)
    return {"pooled": pooled, "folds": folds}


def scores(run):
    if run is None:
        return "  -  "
    p = run["pooled"]
    return "%.3f (R %.3f P %.3f)" % (p["f1"], p.get("recall", np.nan),
                                    p.get("precision", np.nan))


def pair(a, b, rng, boot):
    if a is None or b is None or not set(a["folds"]) & set(b["folds"]):
        return None
    return compare(a["folds"], b["folds"], boot, rng)


def verdict(c):
    """The conclusion: 'level' (interval includes 0), else which run is ahead;
    a change of sign between two level results is not a change."""
    if c is None:
        return None
    lo, hi = c["interval95"]
    return "level" if lo <= 0 <= hi else ("first" if lo > 0 else "second")


def describe(c):
    if c is None:
        return "-"
    lo, hi = c["interval95"]
    clear = "clear" if (lo > 0 or hi < 0) else "level"
    return "%+.3f [%+.3f, %+.3f] %d/%d blocks, %s" % (
        c["pooledDifference"], lo, hi, c["referenceWins"], c["blocks"], clear)


def comparisons(runs, maps):
    """(experiment, label, first run name, second run name)."""
    out = []
    if len(maps) == 2:
        a, b = maps
        for m in METHODS:
            out.append(("1", "%s: %s minus %s" % (m, b, a), m + b, m + a))
    for s in maps:
        base = [m + s for m in ("lmvw", "watershed") if runs.get(m + s)]
        if base:
            better = max(base, key=lambda r: runs[r]["pooled"]["f1"])
            out.append(("1", "CC vs better baseline (%s) on %s" % (better, s),
                        "cc" + s, better))
        out.append(("2", "RGB network vs CHM network on %s" % s, "mrcnnRgb",
                    "mrcnn" + s))
        nets = [r for r in ("mrcnnRgb", "mrcnn" + s) if runs.get(r)]
        if nets:
            better = max(nets, key=lambda r: runs[r]["pooled"]["f1"])
            out.append(("2", "CC on %s vs better network (%s)" % (s, better),
                        "cc" + s, better))
    return out


def report(db, rng, boot):
    lines, changed = [], []
    for name, maps, now, before in sites(db):
        names = [m + s for s in maps for m in METHODS] + ["mrcnnRgb"]
        new = {r: load(now(r)) for r in names}
        old = {r: load(before(r)) for r in names}
        if not any(new.values()) and not any(old.values()):
            continue
        lines += ["", "=" * 78, name, "=" * 78,
                  "%-14s %-32s %-32s" % ("run", "before", "now")]
        for r in names:
            if new[r] or old[r]:
                lines.append("%-14s %-32s %-32s" % (r, scores(old[r]),
                                                    scores(new[r])))
        for exp, label, a, b in comparisons(new, maps):
            cNew = pair(new.get(a), new.get(b), rng, boot)
            cOld = pair(old.get(a), old.get(b), rng, boot)
            if cNew is None and cOld is None:
                continue
            flag = ""
            vN, vO = verdict(cNew), verdict(cOld)
            if vN and vO and vN != vO:
                flag = "   <-- CHANGED"
                changed.append("%s, experiment %s, %s: before %s; now %s"
                               % (name, exp, label, describe(cOld),
                                  describe(cNew)))
            lines.append("exp %s  %-46s now %s%s" % (exp, label,
                                                     describe(cNew), flag))
            lines.append("       %-46s before %s" % ("", describe(cOld)))
    head = ["Changed conclusions (level against clearly ahead, or the other "
            "run ahead):"]
    head += ["  " + c for c in changed] or ["  none"]
    return head + lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    text = "\n".join(report(args.db, np.random.default_rng(0), args.bootstrap))
    print(text)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
