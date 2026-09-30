"""
Connected components tuned on NEON with leave-one-site-out cross-validation.

    python -m tt.neonCv --manifest Data/neon/eval/manifest.csv \\
        --chmFolder CHM050 --jobs 8 --output neonCv050

Every setting of the grid is run once on every tile and its counts (crowns,
detections, hits, repeats, false positives) are kept. Then, for each site in
turn, the setting with the best pooled F1 over all the other sites is chosen
and scored on the held-out site. The site's own crowns never enter the choice,
so each site's score is a held-out test and the pooled score says how well a
setting tuned on other forests does on a new one.

The detector's defaults are always part of the grid and scored on every site
too, for the paired comparison (Wilcoxon over sites, as tt.transfer does over
blocks). The whole-area best, tuned and scored on all sites at once, is
reported as the optimistic reference it is.

The per-tile counts are cached in <output>/counts.json: rerunning with the
same grid, tiles and CHM folder reuses them.
"""

import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .dl import dlCommon as dc
from .neonRun import chmFor, readManifest, siteOf
from .scene import Scene
from .transfer import DEFAULTS, SETTING_KEYS, detect, paired

COUNT_KEYS = ("crowns", "predictions", "hits", "repeats", "falsePositives")


def makeGrid(percentiles, minTopAreas, topSteps, erosions, saddleDrops):
    grid = [dict(zip(SETTING_KEYS, values)) for values in itertools.product(
        percentiles, minTopAreas, topSteps, erosions, saddleDrops)]
    if DEFAULTS not in grid:
        grid.append(dict(DEFAULTS))
    return grid


def tileCounts(task):
    """Counts of every setting on one tile, as a (settings, counts) array."""
    row, chmPath, grid, minHeight = task
    scene = Scene(chmPath, crownsPath=row["crownPath"],
                  boundaryPath=row["boundaryPath"], resolution=0.0,
                  minHeight=minHeight, verbose=False)
    counts = []
    for setting in grid:
        result = dc.evaluateDetections(detect(scene, setting), scene.crowns,
                                       scene.boundary)
        counts.append([result[k] for k in COUNT_KEYS])
    return row["tile"], counts


def f1Of(total):
    """F1 of summed counts, along the last axis in COUNT_KEYS order."""
    crowns, predictions, hits = total[..., 0], total[..., 1], total[..., 2]
    recall = np.divide(hits, crowns, out=np.zeros_like(hits), where=crowns > 0)
    precision = np.divide(hits, predictions, out=np.zeros_like(hits),
                          where=predictions > 0)
    return np.divide(2 * recall * precision, recall + precision,
                     out=np.zeros_like(hits), where=(recall + precision) > 0)


def fold(site, counts, setting):
    """One site's result as a fold record averageFolds can pool."""
    record = dict(zip(COUNT_KEYS, (int(v) for v in counts)))
    record["recall"] = record["hits"] / float(max(record["crowns"], 1))
    record["precision"] = record["hits"] / float(max(record["predictions"], 1))
    record["f1"] = float(f1Of(np.asarray(counts, dtype=float)))
    record.update({"block": site, "settings": setting})
    return record


def crossValidate(tiles, counts, grid):
    """Leave-one-site-out: per site, the best setting on the other sites."""
    sites = np.array([siteOf(t) for t in tiles])
    defaultIndex = grid.index(dict(DEFAULTS))
    tuned, default = [], []
    for site in sorted(set(sites)):
        held = sites == site
        best = int(np.argmax(f1Of(counts[~held].sum(axis=0))))
        heldTotal = counts[held].sum(axis=0)
        tuned.append(fold(site, heldTotal[best], grid[best]))
        default.append(fold(site, heldTotal[defaultIndex], grid[defaultIndex]))
    return tuned, default


def computeCounts(manifestPath, chmFolder, grid, minHeight, jobs):
    tasks, missing = [], []
    for row in readManifest(manifestPath):
        chmPath = chmFor(row["rgb"], row["tile"], chmFolder)
        if chmPath is None:
            missing.append(row["tile"])
        else:
            tasks.append((row, chmPath, grid, minHeight))
    print("[cv] %d tiles x %d settings, %d job(s)" % (len(tasks), len(grid),
                                                     jobs))
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(tileCounts, tasks, chunksize=1))
    return dict(results), missing


def loadOrCompute(args, grid):
    cachePath = os.path.join(args.output, "counts.json")
    key = {"grid": grid, "chmFolder": args.chmFolder,
           "minHeight": args.minHeight, "manifest": args.manifest}
    if os.path.exists(cachePath):
        cached = dc.loadJson(cachePath)
        if cached["key"] == key:
            print("[cv] reusing %s" % cachePath)
            return cached["counts"], cached["missing"]
    counts, missing = computeCounts(args.manifest, args.chmFolder, grid,
                                    args.minHeight, args.jobs)
    dc.saveJson({"key": key, "counts": counts, "missing": missing}, cachePath)
    return counts, missing


def printResults(tuned, default, whole, comparison):
    t, d = dc.averageFolds(tuned), dc.averageFolds(default)
    print("\n%-26s %7s %7s %7s" % ("", "recall", "prec", "F1"))
    for name, p in (("tuned, leave-one-site-out", t), ("defaults", d),
                    ("whole-area best (optim.)", whole)):
        print("%-26s %7.3f %7.3f %7.3f" % (name, p["recall"], p["precision"],
                                           p["f1"]))
    print("tuned minus defaults %+.3f, tuned better on %d of %d sites, p=%.3f"
          % (comparison["tunedMinusFixed"], comparison["tunedBetter"],
             comparison["blocks"], comparison["p"]))
    print("\n%-6s %7s %8s %8s   chosen setting" % ("site", "crowns", "tuned",
                                                  "default"))
    for a, b in zip(tuned, default):
        s = a["settings"]
        print("%-6s %7d %8.3f %8.3f   pct %d, top %.2f m2, step %.2f m, "
              "erosion %d, drop %.2f m"
              % (a["block"], a["crowns"], a["f1"], b["f1"],
                 s["lowerPercentile"], s["minTopAreaM2"], s["topStepM"],
                 s["erosionIterations"], s["saddleDropM"]))


def wholeAreaBest(counts, grid):
    total = counts.sum(axis=0)
    best = int(np.argmax(f1Of(total)))
    return dict(fold("all", total[best], grid[best]))


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Connected components tuned on NEON, leave one site out.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--chmFolder", default="CHM050")
    parser.add_argument("--minHeight", type=float, default=3.0)
    parser.add_argument("--percentiles", default="10,20,30")
    parser.add_argument("--minTopAreas", default="0.12,0.25,0.5")
    parser.add_argument("--topSteps", default="0.12,0.25")
    parser.add_argument("--erosions", default="0,1")
    parser.add_argument("--saddleDrops", default="0.2,0.3,0.5,1.0")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--output", default="neonCv")
    return parser.parse_args(argv)


def numbers(text, cast):
    return [cast(v) for v in text.split(",") if v.strip()]


def main(argv=None):
    args = parseArguments(argv)
    os.makedirs(args.output, exist_ok=True)
    grid = makeGrid(numbers(args.percentiles, int),
                    numbers(args.minTopAreas, float),
                    numbers(args.topSteps, float),
                    numbers(args.erosions, int),
                    numbers(args.saddleDrops, float))
    perTile, missing = loadOrCompute(args, grid)
    if missing:
        print("[cv] %d tile(s) without a CHM skipped" % len(missing))
    tiles = sorted(perTile)
    counts = np.array([perTile[t] for t in tiles], dtype=float)
    tuned, default = crossValidate(tiles, counts, grid)
    whole = wholeAreaBest(counts, grid)
    comparison = paired(tuned, default)
    dc.saveJson({"grid": grid, "folds": tuned, "defaultFolds": default,
                 "pooled": dc.averageFolds(tuned),
                 "defaultPooled": dc.averageFolds(default),
                 "wholeAreaBest": whole, "vsDefaults": comparison},
                os.path.join(args.output, "results.json"))
    printResults(tuned, default, whole, comparison)
    return 0


if __name__ == "__main__":
    sys.exit(main())
