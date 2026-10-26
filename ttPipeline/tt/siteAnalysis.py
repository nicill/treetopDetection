"""
Why a method loses precision at a site: diagnostics, never a result.

    python -m tt.siteAnalysis --root "<study folder>"          # every site
    python -m tt.siteAnalysis --site "<study folder>/<site>"   # one site

For every single method of a site (runs/*), on the blocks all its runs share:

  breakdown   hits, repeats (a second detection in a crown already found)
              and outside (in no crown), as shares of the detections; for
              connected components also the heights of hits and of outside
              tops, and how many outside tops lie within NEAR_M of a crown
  offset      the crowns shifted by up to SHIFT_M east and north in steps of
              SHIFT_STEP_M, F1 at each shift: a misalignment between the
              height model and the imagery the crowns were drawn on shows as
              a clear gain, for the height-based methods only, at one shift
  tolerance   the crowns grown by each of TOLERANCES_M: a method whose
              detections sit at crown edges gains far more than one whose
              detections sit at crown centres

The measures are tt.analysis.Analysis's. Written to <site>/analysis/analysis.json, with a table printed per site and,
with --root, one line per site and method at the end.
"""

import argparse
import glob
import os
import sys

from shapely.ops import unary_union

from .analysis import Analysis
from .analysis.analysis import NEAR_M
from .data.site import TDData
from .dl import dlCommon as dc
from .experiments import RunSet
from .experiments.runset import sharedBlocks

MIN_GAIN = 0.005           # a shift gaining less than this is no evidence


def siteData(siteDir):
    """(site, region, {name: RunSet}) on the blocks every run shares."""
    site = TDData.fromDataset(os.path.join(siteDir, "ds", "chm"))
    blocks = {b.name: b.geometry for b in site.blocks}
    runs = {os.path.basename(os.path.dirname(p)):
            RunSet.load(os.path.dirname(p))
            for p in sorted(glob.glob(os.path.join(siteDir, "runs", "*",
                                                   "results.json")))}
    shared = sharedBlocks(blocks, list(runs.values()), verbose=False)
    return site, blocks, unary_union(list(shared.values())), runs


def breakdown(analysis, detections, region, heights):
    result = analysis.evaluator.score(detections, region)
    n = max(result["predictions"], 1)
    row = {"recall": result["recall"], "precision": result["precision"],
           "f1": result["f1"], "detections": result["predictions"],
           "hits": result["hits"] / float(n),
           "repeats": result["repeats"] / float(n),
           "outside": result["falsePositives"] / float(n)}
    if heights:
        row.update(analysis.outsideDetail(detections, region))
    return row


def analyseSite(siteDir, shifts=True):
    site, blocks, region, runs = siteData(siteDir)
    analysis = Analysis(site, surface=False)
    result = {}
    for name, run in runs.items():
        detections = run.heldOut(blocks)
        row = {"breakdown": breakdown(analysis, detections, region,
                                      name.startswith("cc")),
               "tolerance": analysis.tolerance(detections, region)}
        if shifts:
            row["offset"] = analysis.offsetScan(detections, region)
        result[name] = row
    os.makedirs(os.path.join(siteDir, "analysis"), exist_ok=True)
    dc.saveJson(result, os.path.join(siteDir, "analysis", "analysis.json"))
    return result


def printSite(name, result):
    print("\n== %s" % name)
    for method, row in result.items():
        b, t = row["breakdown"], row["tolerance"]
        line = ("  %-12s R %.3f P %.3f F1 %.3f | hits %4.1f%% repeats %4.1f%% "
                "outside %4.1f%% | F1 grown 0.25 %.3f 0.5 %.3f"
                % (method, b["recall"], b["precision"], b["f1"],
                   100 * b["hits"], 100 * b["repeats"], 100 * b["outside"],
                   t["0.25"]["f1"], t["0.50"]["f1"]))
        if "offset" in row:
            o = row["offset"]
            line += (" | no gain from shifting" if o["gain"] < MIN_GAIN else
                     " | best shift (%+.1f, %+.1f) m: F1 %+.3f"
                     % (o["bestShiftM"][0], o["bestShiftM"][1], o["gain"]))
        print(line)
        if "outsideNear" in b:
            print("  %12s outside tops within %.1f m of a crown: %.0f%%; "
                  "heights hit %s, outside %s"
                  % ("", NEAR_M, 100 * b["outsideNear"],
                     "/".join("%.2f" % v for v in b["hitHeights"]),
                     "/".join("%.2f" % v for v in b["outsideHeights"])))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Breakdown, offset scan and crown tolerance per site.")
    parser.add_argument("--root", help="A study folder: every site in it")
    parser.add_argument("--site", help="One site's study folder")
    parser.add_argument("--noShifts", action="store_true",
                        help="Skip the offset scan (the slow part)")
    args = parser.parse_args(argv)
    sites = [args.site] if args.site else sorted(
        d for d in glob.glob(os.path.join(args.root or ".", "*"))
        if os.path.exists(os.path.join(d, "ds", "chm", "dataset.json")))
    for site in sites:
        try:
            printSite(os.path.basename(site.rstrip("/")),
                      analyseSite(site, shifts=not args.noShifts))
        except Exception as error:          # noqa: BLE001 — one site, not all
            print("\n== %s: %s: %s" % (os.path.basename(site.rstrip("/")),
                                       type(error).__name__, error))
    return 0


if __name__ == "__main__":
    sys.exit(main())
