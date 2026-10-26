#!/usr/bin/env python
"""
One command with subcommands, replacing fifteen scripts whose mains were mostly
duplicated argument parsing.

    tt detect      run the detector, write tops and images
    tt analyse     what it gets wrong, and why
    tt sweep       search detector parameters
    tt merge       search merge metrics and thresholds
    tt align       measure offsets between CHM, photo and crowns

Every subcommand takes the same site arguments, declared once. The work is
done by the classes (TDData, CCDetector, Analysis, GridSearch, PseudoCrowns,
Alignment); this module only parses arguments.
"""

import argparse
import os
import sys

from .analysis import Analysis
from .data.alignment import Alignment
from .data.detections import writeList, writeMask
from .data.site import TDData
from .detectors.height import CCDetector
from .detectors.merging import METRICS
from .experiments import ParameterGrid
from .experiments.fixed import GridSearch
from .scoring.pseudoCrowns import PseudoCrowns


def addSiteArguments(parser):
    parser.add_argument("--chm", required=True)
    parser.add_argument("--crowns", default=None)
    parser.add_argument("--boundary", default=None)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--minHeight", type=float, default=2.0)
    parser.add_argument("--crownShift", type=float, nargs=2, default=(0.0, 0.0),
                        metavar=("EAST", "SOUTH"),
                        help="Where the crowns sit relative to the CHM, as "
                             "'tt align' reports it. They are moved by the "
                             "negative of this")


def addDetectorArguments(parser):
    parser.add_argument("--windowSize", type=float, default=40.0)
    parser.add_argument("--windowOverlap", type=float, default=0.2)
    parser.add_argument("--lowerPercentile", type=int, default=10)
    parser.add_argument("--minTreeArea", type=float, default=0.5)
    parser.add_argument("--minTopArea", type=float, default=0.12)
    parser.add_argument("--minTopAreaSlope", type=float, default=0.0,
                        help="m2 added to --minTopArea per metre of tree "
                             "height; 0 keeps it fixed")
    parser.add_argument("--topStep", type=float, default=0.12)
    parser.add_argument("--erosionIterations", type=int, default=1)
    parser.add_argument("--erosionKernel", type=int, default=3)


def addMergerArguments(parser):
    parser.add_argument("--metric", default="saddle", choices=METRICS)
    parser.add_argument("--eps", type=float, default=8.0,
                        help="Merge threshold in metres. Saddle needs a large "
                             "one, 5 to 12; the others want under 2")
    parser.add_argument("--heightWeight", type=float, default=0.7)
    parser.add_argument("--saddleDrop", type=float, default=0.5)
    parser.add_argument("--saddleDropSlope", type=float, default=0.0,
                        help="metres added to --saddleDrop per metre of tree "
                             "height; 0 keeps it fixed")


def checkPaths(args):
    """
    Fail on a missing input with one line, not a GDAL traceback. Also
    catches the commonest paste error: a documentation placeholder left in.
    """
    for name in ("chm", "crowns", "boundary"):
        path = getattr(args, name, None)
        if path is None:
            continue
        if path.strip(". ") == "":
            raise SystemExit("--%s is '%s' — that is the placeholder from the "
                             "documentation, not a path." % (name, path))
        if not os.path.exists(path):
            raise SystemExit("--%s not found: %s" % (name, path))


def buildSite(args):
    checkPaths(args)
    return TDData.fromFiles("site", chm=args.chm, crowns=args.crowns,
                            area=args.boundary,
                            crownShiftEastM=args.crownShift[0],
                            crownShiftSouthM=args.crownShift[1])


def heightParams(args):
    return {"resolution": args.resolution, "minHeight": args.minHeight}


def detectorParams(args):
    return {"windowSizeM": args.windowSize, "windowOverlap": args.windowOverlap,
            "lowerPercentile": args.lowerPercentile,
            "minTreeAreaM2": args.minTreeArea, "minTopAreaM2": args.minTopArea,
            "topStepM": args.topStep, "minTopAreaSlope": args.minTopAreaSlope,
            "erosionIterations": args.erosionIterations,
            "erosionKernelSize": args.erosionKernel}


def mergerParams(args):
    return {"mergeMetric": args.metric, "saddleEpsM": args.eps,
            "heightWeight": args.heightWeight, "saddleDropM": args.saddleDrop,
            "saddleDropSlope": args.saddleDropSlope}


def buildDetector(args):
    return CCDetector(**dict(heightParams(args), **detectorParams(args),
                             **mergerParams(args)))


def analysisOf(site, detector):
    return Analysis(site, detector.surface(site))


# ---------------------------------------------------------------------- #

def commandDetect(args):
    site = buildSite(args)
    detector = buildDetector(args)
    detections = detector.detect(site)
    surface = detector.surface(site)
    os.makedirs(args.output, exist_ok=True)
    writeList(detections, os.path.join(args.output, "tops.txt"), surface)
    writeMask(detections, os.path.join(args.output, "tops_mask.png"), surface)
    if args.crowns:
        result = analysisOf(site, detector).errors(detections)
        print("R %.3f  P %.3f  F1 %.3f  (%d detections, %d repeats, "
              "%d background)"
              % (result["recall"], result["precision"], result["f1"],
                 result["predictions"], result["repeats"],
                 result["falsePositives"]))
    print("wrote %s" % args.output)
    return 0


def commandAnalyse(args):
    site = buildSite(args)
    detector = buildDetector(args)
    summary = analysisOf(site, detector).writeErrors(
        args.output, detector.detect(site), detector,
        images=not args.noImages, neighbourhood=args.neighbourhood,
        canopyFraction=args.canopyFraction)
    printErrors(summary)
    return 0


def printErrors(s):
    total = max(s["predictions"], 1)
    print("=" * 70)
    print("%d crowns, %d detections | recall %.1f%% | precision %.1f%%"
          % (s["crowns"], s["predictions"], 100 * s["recall"],
             100 * s["precision"]))
    print("=" * 70)
    print("WHERE THE DETECTIONS GO")
    for label, key, note in (
            ("in a crown, primary", "hits", ""),
            ("in a crown, repeat", "repeats", "over-segmented trees"),
            ("outside, canopy-level", "backgroundCanopy",
             "probably unannotated trees"),
            ("outside, below canopy", "backgroundLow",
             "the real false positives"),
            ("outside, no neighbours", "backgroundUnknown", "")):
        if s[key] or key in ("hits", "repeats"):
            print("  %-24s %5d  %5.1f%%   %s"
                  % (label, s[key], 100.0 * s[key] / total, note))
    print("\nMISSED CROWNS: %d of %d (%.1f%%)"
          % (s["missed"], s["crowns"], 100.0 * s["missed"]
             / max(s["crowns"], 1)))
    for name, key, unit in (("height", "peak", "m"), ("area", "area", "m2"),
                            ("nn distance", "nearest", "m")):
        found, missed = s["foundCrowns"].get(key), s["missedCrowns"].get(key)
        print("  %-14s found %7.2f %-3s   missed %7.2f %s"
              % (name, found if found is not None else float("nan"), unit,
                 missed if missed is not None else float("nan"), unit))
    if "missedBelowWindowCut" in s:
        print("  %d of the missed have their apex below their window's "
              "percentile cut" % s["missedBelowWindowCut"])
    print("\n  precision as scored                          %.1f%%"
          % (100 * s["precision"]))
    print("  precision if canopy-level ones are real      %.1f%%"
          % (100 * s["precisionIfCanopyAreTrees"]))


def runSearch(args, detector, grid):
    search = GridSearch(buildSite(args), detector, ParameterGrid(grid))
    search.run()
    search.table(sortBy=args.sortBy, limit=args.limit)
    best = search.best(minimumRecall=args.minRecall)
    if best:
        print("\nbest: %s -> R %.1f%% P %.1f%% F1 %.1f%%"
              % (best["label"], 100 * best["recall"],
                 100 * best["precision"], 100 * best["f1"]))
    search.save(args.output)
    print("wrote %s" % args.output)
    return 0


def commandSweep(args):
    detector = CCDetector(windowSizeM=args.windowSize,
                          minTreeAreaM2=args.minTreeArea,
                          **heightParams(args))
    # the merge is swept with the detector, not left at a default: a base
    # that over-detects less needs a gentler merge
    grid = {"lowerPercentile": _numbers(args.percentiles, int),
            "minTopAreaM2": _numbers(args.minTopAreas, float),
            "topStepM": _numbers(args.topSteps, float),
            "erosionIterations": _numbers(args.erosions, int)}
    _addSlopes(grid, "minTopAreaSlope", args.minTopAreaSlopes)
    grid.update({"mergeMetric": [args.metric], "saddleEpsM": [args.eps]})
    if args.metric == "saddle":
        grid["saddleDropM"] = _numbers(args.saddleDrops, float)
        _addSlopes(grid, "saddleDropSlope", args.saddleDropSlopes)
    elif args.metric == "composite":
        grid["heightWeight"] = [args.heightWeight]
    return runSearch(args, detector, grid)


def commandMerge(args):
    detector = CCDetector(**dict(heightParams(args), **detectorParams(args)))
    grid = {"mergeMetric": [m.strip() for m in args.metrics.split(",")],
            "saddleEpsM": _numbers(args.epsilons, float)}
    if "saddle" in grid["mergeMetric"]:
        grid["saddleDropM"] = _numbers(args.saddleDrops, float)
        _addSlopes(grid, "saddleDropSlope", args.saddleDropSlopes)
    if "composite" in grid["mergeMetric"]:
        grid["heightWeight"] = _numbers(args.weights, float)
    return runSearch(args, detector, grid)


def commandCrowns(args):
    site = buildSite(args)
    detector = buildDetector(args)
    crowns = PseudoCrowns(detector.surface(site), detector.detect(site),
                          percentile=args.crownPercentile,
                          windowSizeM=args.windowSize, many=args.many,
                          lots=args.lots, voronoiWeight=args.voronoiWeight)
    crowns.build()
    crowns.write(args.output)
    if args.crowns:
        comparison = crowns.compareWith(site.crowns)
        if comparison:
            print("\nagainst the real crowns: %d of %d matched, "
                  "median IoU %.3f, %.1f%% above 0.5, %.1f%% above 0.25"
                  % (comparison["matched"], comparison["pseudoCrowns"],
                     comparison["medianIou"], 100 * comparison["aboveHalf"],
                     100 * comparison["aboveQuarter"]))
            print("median area: pseudo %.1f m2, real %.1f m2"
                  % (comparison["pseudoAreaMedian"],
                     comparison["trueAreaMedian"]))
    return 0


def commandAlign(args):
    checkPaths(args)
    for spec in args.layer:
        if "=" not in spec:
            raise SystemExit("--layer expects name=path, got %r" % spec)
        path = spec.split("=", 1)[1].strip()
        if not os.path.exists(path):
            raise SystemExit("--layer path not found: %s" % path)
    alignment = Alignment(args.chm, resolution=args.resolution,
                          boundaryPath=args.boundary)
    for spec in args.layer:
        name, path = spec.split("=", 1)
        alignment.add(name.strip(), path.strip())
    alignment.measure(tileM=args.tile, maxShiftM=args.maxShift)
    alignment.report()
    if args.output:
        alignment.save(args.output)
    return 0


def _numbers(text, cast):
    return [cast(v) for v in str(text).split(",") if str(v).strip()]


def _addSlopes(grid, key, text):
    """
    Sweep a height slope only when asked: a slope of 0 alone is the fixed
    parameter, and leaving the key out keeps the labels of existing results.
    """
    values = _numbers(text, float)
    if values and values != [0.0]:
        grid[key] = values


# ---------------------------------------------------------------------- #

def addDetectCommand(subparsers):
    detect = subparsers.add_parser("detect", help="run the detector")
    addSiteArguments(detect)
    addDetectorArguments(detect)
    addMergerArguments(detect)
    detect.add_argument("--output", default="output/detect")
    detect.set_defaults(handler=commandDetect)
    return detect

def addAnalyseCommand(subparsers):
    analyse = subparsers.add_parser("analyse", help="what it gets wrong")
    addSiteArguments(analyse)
    addDetectorArguments(analyse)
    addMergerArguments(analyse)
    analyse.add_argument("--output", default="output/analysis")
    analyse.add_argument("--neighbourhood", type=float, default=12.0)
    analyse.add_argument("--canopyFraction", type=float, default=0.7)
    analyse.add_argument("--noImages", action="store_true")
    analyse.set_defaults(handler=commandAnalyse)
    return analyse

def addSweepCommand(subparsers):
    sweep = subparsers.add_parser("sweep", help="search detector parameters")
    addSiteArguments(sweep)
    sweep.add_argument("--windowSize", type=float, default=40.0)
    sweep.add_argument("--minTreeArea", type=float, default=0.5)
    sweep.add_argument("--percentiles", default="10,20,30")
    sweep.add_argument("--minTopAreas", default="0.25,0.12,0.06")
    sweep.add_argument("--minTopAreaSlopes", default="0",
                       help="m2 per metre of tree height, e.g. 0,0.02,0.05")
    sweep.add_argument("--topSteps", default="0.25,0.12")
    sweep.add_argument("--erosions", default="1,2")
    sweep.add_argument("--metric", default="saddle", choices=METRICS)
    sweep.add_argument("--eps", type=float, default=8.0)
    sweep.add_argument("--heightWeight", type=float, default=0.7)
    sweep.add_argument("--saddleDrops", default="0.2,0.3,0.5",
                       help="Merge thresholds swept alongside the detector "
                            "parameters, since the two interact")
    sweep.add_argument("--saddleDropSlopes", default="0",
                       help="metres per metre of tree height, "
                            "e.g. 0,0.02,0.05")
    sweep.add_argument("--limit", type=int, default=20)
    sweep.add_argument("--sortBy", default="f1", choices=["f1", "recall",
                                                          "precision"])
    sweep.add_argument("--minRecall", type=float, default=None)
    sweep.add_argument("--output", default="sweep.json")
    sweep.set_defaults(handler=commandSweep)
    return sweep

def addMergeCommand(subparsers):
    merge = subparsers.add_parser("merge", help="search merge settings")
    addSiteArguments(merge)
    addDetectorArguments(merge)
    merge.add_argument("--metrics", default="saddle")
    merge.add_argument("--epsilons", default="5,8,12")
    merge.add_argument("--saddleDrops", default="0.25,0.5,0.75,1.0")
    merge.add_argument("--saddleDropSlopes", default="0")
    merge.add_argument("--weights", default="0.3,0.5,0.7")
    merge.add_argument("--sortBy", default="f1", choices=["f1", "recall",
                                                          "precision"])
    merge.add_argument("--minRecall", type=float, default=None)
    merge.add_argument("--limit", type=int, default=25)
    merge.add_argument("--output", default="mergeSweep.json")
    merge.set_defaults(handler=commandMerge)
    return merge

def addCrownsCommand(subparsers):
    crowns = subparsers.add_parser(
        "crowns", help="pseudo-crowns and boxes from the detected tops")
    addSiteArguments(crowns)
    addDetectorArguments(crowns)
    addMergerArguments(crowns)
    crowns.add_argument("--output", default="output/crowns")
    crowns.add_argument("--crownPercentile", type=int, default=20,
                        help="Local percentile the canopy is thresholded at "
                             "(default: 20, matching the detector's own cut)")
    crowns.add_argument("--many", type=int, default=2,
                        help="Tops per canopy blob above which a box is "
                             "centred on its top rather than taken whole")
    crowns.add_argument("--lots", type=int, default=6,
                        help="Tops per canopy blob above which the distance "
                             "transform is blended in")
    crowns.add_argument("--voronoiWeight", type=float, default=0.65,
                        help="How far from the distance box toward the "
                             "Voronoi box for dense blobs (default: 0.65)")
    crowns.set_defaults(handler=commandCrowns)
    return crowns

def addAlignCommand(subparsers):
    align = subparsers.add_parser("align", help="measure layer offsets")
    align.add_argument("--chm", required=True)
    align.add_argument("--layer", action="append", default=[],
                       metavar="NAME=PATH")
    align.add_argument("--boundary", default=None)
    align.add_argument("--resolution", type=float, default=0.10)
    align.add_argument("--tile", type=float, default=40.0)
    align.add_argument("--maxShift", type=float, default=4.0)
    align.add_argument("--output", default=None)
    align.set_defaults(handler=commandAlign)
    return align


COMMANDS = (addDetectCommand, addAnalyseCommand, addSweepCommand, addMergeCommand, addCrownsCommand, addAlignCommand)


def build():
    parser = argparse.ArgumentParser(prog="tt",
                                     description=__doc__.split("\n")[1])
    subparsers = parser.add_subparsers(dest="command", required=True)
    for addCommand in COMMANDS:
        addCommand(subparsers)
    return parser


def main(argv=None):
    args = build().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
