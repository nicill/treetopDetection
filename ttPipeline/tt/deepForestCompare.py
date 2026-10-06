"""
The annotation-free hybrid: DeepForest's prebuilt boxes, connected components
tuned on them (tt.pseudoTuning --rgbRun <deepforest run>), and their union,
next to the procedure's networks for reference.

    python -m tt.deepForestCompare --dataset <site>/ds/chm \\
        --deepforest <out>/runs/deepforestRgb --ccAuto <out>/runs/ccAutoDfP1 \\
        --reference mrcnnRgb=<site>/runs/mrcnnRgb \\
        --reference ccAutoP1=<site>/runs/ccAutoP1 --output <out>/compare.json

The union keeps DeepForest's boxes at its fixed operating point
(tt.deepForestDetect.THRESHOLD, recorded in its run) and adds every
connected-component top no kept box contains: fusion.union, with nothing
chosen on the crowns. Every row is pooled over the blocks all runs share;
the union is compared block by block with each of its parts and with each
reference (paired Wilcoxon on per-block F1 and weighted score).
"""

import argparse
import sys

import numpy as np
from scipy.stats import wilcoxon

from .comparison import MethodRun, sharedBlocks
from .dl import dlCommon as dc
from .dl import dlPrepare as dp
from .fusion import union
from .scene import readCrowns


def blockScores(run, blocks, crowns):
    """{block: result} of a run's held-out predictions, at its own threshold."""
    return {b: dc.evaluateDetections(
        [p for p in run.predictions if p["block"] == b], crowns, g)
        for b, g in blocks.items()}


def unionScores(boxRun, pointRun, blocks, crowns):
    """{block: result} of the union at the box run's recorded threshold."""
    scores = {}
    for block, geometry in blocks.items():
        boxes = dc.predictionsInRegion(boxRun.record(block)["predictions"],
                                       geometry)
        points = dc.predictionsInRegion(pointRun.record(block)["predictions"],
                                        geometry)
        fused = union(boxes, points, boxRun.threshold(block))
        scores[block] = dc.evaluateDetections(fused, crowns, geometry)
    return scores


def paired(first, second, key):
    shared = sorted(set(first) & set(second))
    difference = np.array([first[b][key] - second[b][key] for b in shared])
    return float(wilcoxon(difference).pvalue) if np.any(difference) else 1.0


def row(name, scores):
    pooled = dc.averageFolds(list(scores.values()))
    return {"name": name, "recall": pooled["recall"],
            "precision": pooled["precision"], "f1": pooled["f1"],
            "weighted": pooled["weighted"], "blocks": len(scores)}


def compare(args):
    meta = dp.loadDataset(args.dataset)
    crowns, _ = readCrowns(meta["crownsPath"], meta["crs"])
    allBlocks = dict(meta["blockGeometries"])
    named = [("deepforest", args.deepforest), ("ccOnDeepforest", args.ccAuto)]
    named += [tuple(r.split("=", 1)) for r in args.reference]
    runs = {n: MethodRun(n, d, allBlocks) for n, d in named}
    blocks = sharedBlocks(allBlocks, list(runs.values()))
    scores = {n: blockScores(r, blocks, crowns) for n, r in runs.items()}
    scores["union"] = unionScores(runs["deepforest"], runs["ccOnDeepforest"],
                                  blocks, crowns)
    rows = [row(n, s) for n, s in scores.items()]
    tests = {other: {k: paired(scores["union"], scores[other], k)
                     for k in ("f1", "weighted")}
             for other in scores if other != "union"}
    return rows, tests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--deepforest", required=True)
    parser.add_argument("--ccAuto", required=True)
    parser.add_argument("--reference", action="append", default=[],
                        help="NAME=RUN_DIR, shown for comparison")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    rows, tests = compare(args)
    print("%-16s %6s  %5s  %5s  %5s  %5s" % ("method", "blocks", "R", "P",
                                             "F1", "W"))
    for r in rows:
        print("%-16s %6d  %.3f  %.3f  %.3f  %.3f"
              % (r["name"], r["blocks"], r["recall"], r["precision"], r["f1"],
                 r["weighted"]))
    print("\nunion against (paired Wilcoxon over blocks):")
    for other, p in tests.items():
        print("  %-16s F1 p=%.3f  W p=%.3f" % (other, p["f1"], p["weighted"]))
    if args.output:
        dc.saveJson({"rows": rows, "wilcoxon": tests}, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
