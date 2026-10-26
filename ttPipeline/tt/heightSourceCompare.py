"""
The height methods (CC, local maxima, watershed, and Mask R-CNN trained on
the CHM when it has been run) on two CHMs of the same
sites, block by block: photogrammetry against LiDAR for each method, and CC
against the better baseline on each CHM.

    python -m tt.heightSourceCompare --first "$PROC" --second "$OUT" \\
        --output "$OUT/compare.json"

--first and --second are study folders holding <site>/runs/<method><height>/
results.json, as quebecP1Study.sh writes them (LiDAR in study_procedure,
photogrammetry in study_procedureP1). Each test is a two-sided paired
Wilcoxon on the per-block scores the runs report (the held-out folds), on the
blocks both runs scored, for F1 and for the weighted score (0.6 R + 0.4 P).
The better baseline is the one with the higher pooled F1 on that CHM, as in
the paper's tables. With 6 to 8 blocks the smallest possible p is about 0.03
(6 blocks) to 0.008 (8), so a difference can be real and still not reach 0.05.
"""

import argparse
import json
import os
import sys


from .analysis.statistics import pairedTest

METHODS = ("cc", "lmvw", "watershed", "mrcnn")   # mrcnn: on the CHM, when run
KEYS = ("f1", "weighted")


def loadRun(path):
    """(pooled, {block: fold}) of a run, or None when it has not been run."""
    path = os.path.join(path, "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        result = json.load(handle)
    return result["pooled"], {f["block"]: f for f in result["folds"]}


def compareRuns(a, b):
    """Pooled F1 of both, and the paired tests (b against a)."""
    row = {"first": a[0]["f1"], "second": b[0]["f1"]}
    for key in KEYS:
        mean, p, blocks = pairedTest(a[1], b[1], key)
        row.update({key + "MeanDiff": mean, key + "P": p, "blocks": blocks})
    return row


def siteRuns(root, site, height):
    return {m: loadRun(os.path.join(root, site, "runs", m + height))
            for m in METHODS}


def ccAgainstBaseline(runs):
    """CC against the better baseline (by pooled F1) on one CHM."""
    baselines = [m for m in ("lmvw", "watershed") if runs[m]]
    if not runs["cc"] or not baselines:
        return None
    better = max(baselines, key=lambda m: runs[m][0]["f1"])
    return dict(compareRuns(runs[better], runs["cc"]), baseline=better)


def compareSite(site, args):
    first = siteRuns(args.first, site, args.firstHeight)
    second = siteRuns(args.second, site, args.secondHeight)
    sources = {m: compareRuns(first[m], second[m]) for m in METHODS
               if first[m] and second[m]}
    return {"site": site, "sources": sources,
            "ccVsBaseline": {args.firstHeight: ccAgainstBaseline(first),
                             args.secondHeight: ccAgainstBaseline(second)}}


def star(p):
    return "*" if p < 0.05 else " "


def printSite(result, args):
    print("\n%s" % result["site"])
    for method, r in result["sources"].items():
        print("  %-10s %s %.3f  %s %.3f  diff %+.3f  p(F1) %.3f%s  "
              "p(weighted) %.3f  (%d blocks)"
              % (method, args.firstHeight, r["first"], args.secondHeight,
                 r["second"], r["second"] - r["first"], r["f1P"],
                 star(r["f1P"]), r["weightedP"], r["blocks"]))
    for height, r in result["ccVsBaseline"].items():
        if r:
            print("  CC vs %-9s on %-6s %+.3f F1  p(F1) %.3f%s  p(weighted) "
                  "%.3f" % (r["baseline"], height, r["second"] - r["first"],
                            r["f1P"], star(r["f1P"]), r["weightedP"]))


def sitesIn(args):
    names = sorted(os.listdir(args.second)) if not args.sites else args.sites
    return [s for s in names
            if os.path.isdir(os.path.join(args.second, s, "runs"))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--first", required=True,
                        help="Study folder of the reference CHM (LiDAR)")
    parser.add_argument("--second", required=True,
                        help="Study folder of the other CHM (photogrammetry)")
    parser.add_argument("--firstHeight", default="Lidar")
    parser.add_argument("--secondHeight", default="P1")
    parser.add_argument("--sites", nargs="*", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    results = [compareSite(s, args) for s in sitesIn(args)]
    print("pooled F1, and paired Wilcoxon over blocks (* p < 0.05); diff is "
          "%s minus %s" % (args.secondHeight, args.firstHeight))
    for result in results:
        printSite(result, args)
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(results, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
