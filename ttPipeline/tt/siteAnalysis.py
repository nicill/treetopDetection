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

Written to <site>/analysis/analysis.json, with a table printed per site and,
with --root, one line per site and method at the end.
"""

import argparse
import glob
import json
import os
import sys

import geopandas as gpd
import numpy as np
from shapely.affinity import translate
from shapely.geometry import Point
from shapely.ops import unary_union

from .comparison import MethodRun, sharedBlocks
from .dl import dlCommon as dc
from .dl import dlPrepare as dp
from .evaluation import assignToCrowns

NEAR_M = 0.5
SHIFT_M = 1.0
SHIFT_STEP_M = 0.1
TOLERANCES_M = (0.25, 0.5)
MIN_GAIN = 0.005           # a shift gaining less than this is no evidence


def siteData(siteDir):
    """(crowns, region, {name: MethodRun}) on the blocks every run shares."""
    meta = dp.loadDataset(os.path.join(siteDir, "ds", "chm"))
    blocks = dict(meta["blockGeometries"])
    runs = {os.path.basename(os.path.dirname(p)):
            MethodRun(os.path.basename(os.path.dirname(p)),
                      os.path.dirname(p), blocks)
            for p in sorted(glob.glob(os.path.join(siteDir, "runs", "*",
                                                   "results.json")))}
    shared = sharedBlocks(blocks, list(runs.values()), verbose=False)
    crowns = gpd.read_file(meta["crownsPath"])
    return crowns, unary_union(list(shared.values())), runs


def breakdown(predictions, crowns, region, heights):
    result = dc.evaluateDetections(predictions, crowns, region)
    n = max(result["predictions"], 1)
    row = {"recall": result["recall"], "precision": result["precision"],
           "f1": result["f1"], "detections": result["predictions"],
           "hits": result["hits"] / float(n),
           "repeats": result["repeats"] / float(n),
           "outside": result["falsePositives"] / float(n)}
    if heights:
        row.update(heightDetail(predictions, crowns, region))
    return row


def heightDetail(predictions, crowns, region):
    """Heights of hits and outside tops (score = height), nearness of the
    outside ones to a drawn crown."""
    inBlock = dc.crownsInRegion(crowns, region)
    kept = dc.predictionsInRegion(predictions, region)
    xs = np.array([p["centreX"] for p in kept])
    ys = np.array([p["centreY"] for p in kept])
    h = np.array([p["score"] for p in kept])
    kinds, _, _ = assignToCrowns(xs, ys, h, inBlock)
    kinds = np.array(kinds)
    out = kinds == "background"
    drawn = unary_union(list(inBlock.geometry))
    near = [drawn.distance(Point(x, y)) <= NEAR_M
            for x, y in zip(xs[out], ys[out])]
    quantiles = lambda v: [float(q) for q in np.percentile(v, [10, 50, 90])] \
        if len(v) else []
    return {"hitHeights": quantiles(h[kinds == "hit"]),
            "outsideHeights": quantiles(h[out]),
            "outsideNear": float(np.mean(near)) if near else 0.0}


def offsetScan(predictions, crowns, region):
    """F1 for every shift of the crowns; the best and the unshifted."""
    steps = np.round(np.arange(-SHIFT_M, SHIFT_M + 1e-9, SHIFT_STEP_M), 3)
    scores = {}
    for dx in steps:
        for dy in steps:
            moved = crowns.copy()
            moved["geometry"] = [translate(g, dx, dy) for g in crowns.geometry]
            scores[(float(dx), float(dy))] = dc.evaluateDetections(
                predictions, moved, region)["f1"]
    top = max(scores.values())
    # a whole patch of shifts can reach the best score: its centre is the
    # estimate (and (0, 0) when every shift scores the same)
    reaching = np.array([k for k, v in scores.items() if v >= top - 1e-9])
    centre = [round(float(v), 2) + 0.0 for v in reaching.mean(axis=0)]
    return {"bestShiftM": centre, "bestF1": top,
            "unshiftedF1": scores[(0.0, 0.0)],
            "gain": top - scores[(0.0, 0.0)],
            "shiftsAtBest": int(len(reaching))}


def tolerance(predictions, crowns, region):
    out = {}
    for grow in TOLERANCES_M:
        grown = crowns.copy()
        grown["geometry"] = crowns.geometry.buffer(grow)
        r = dc.evaluateDetections(predictions, grown, region)
        out["%.2f" % grow] = {"recall": r["recall"],
                              "precision": r["precision"], "f1": r["f1"]}
    return out


def analyseSite(siteDir, shifts=True):
    crowns, region, runs = siteData(siteDir)
    result = {}
    for name, run in runs.items():
        predictions = run.predictions
        isCc = name.startswith("cc")
        row = {"breakdown": breakdown(predictions, crowns, region, isCc),
               "tolerance": tolerance(predictions, crowns, region)}
        if shifts:
            row["offset"] = offsetScan(predictions, crowns, region)
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
