#!/usr/bin/env python
"""
One command with subcommands, replacing fifteen scripts whose mains were mostly
duplicated argument parsing.

    tt detect      run the detector, write tops and images
    tt analyse     what it gets wrong, and why
    tt sweep       search detector parameters
    tt merge       search merge metrics and thresholds
    tt align       measure offsets between CHM, photo and crowns

Every subcommand takes the same scene arguments, declared once.
"""

import argparse
import os
import sys

from .alignment import Alignment
from .analysis import MissAnalysis
from .detector import ConCompDetector
from .evaluation import CrownEvaluator
from .merging import TopMerger, METRICS
from .pseudoCrowns import PseudoCrowns
from .scene import Scene
from .sweeps import Sweep


def addSceneArguments(parser):
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
    Fail on a missing input with one line, not a GDAL traceback.

    Also catches the commonest paste error: copying a command out of
    documentation with the placeholder left in.
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


def buildScene(args, restrict=True):
    checkPaths(args)
    return Scene(args.chm, crownsPath=args.crowns,
                 boundaryPath=args.boundary, resolution=args.resolution,
                 minHeight=args.minHeight,
                 crownShiftEast=args.crownShift[0],
                 crownShiftSouth=args.crownShift[1], restrict=restrict)


def buildDetector(args):
    return ConCompDetector(
        windowSizeM=args.windowSize, windowOverlap=args.windowOverlap,
        lowerPercentile=args.lowerPercentile, minTreeAreaM2=args.minTreeArea,
        minTopAreaM2=args.minTopArea, topStepM=args.topStep,
        minTopAreaSlope=args.minTopAreaSlope,
        erosionIterations=args.erosionIterations,
        erosionKernelSize=args.erosionKernel, merger=buildMerger(args))


def buildMerger(args):
    return TopMerger(metric=args.metric, epsM=args.eps,
                     heightWeight=args.heightWeight,
                     saddleDropM=args.saddleDrop,
                     saddleDropSlope=args.saddleDropSlope)


# ---------------------------------------------------------------------- #

def commandDetect(args):
    scene = buildScene(args)
    detector = buildDetector(args)
    tops = detector.detect(scene)

    os.makedirs(args.output, exist_ok=True)
    tops.writeList(os.path.join(args.output, "tops.txt"), scene)
    tops.writeMask(os.path.join(args.output, "tops_mask.png"), scene.shape)
    if scene.crowns is not None:
        result = CrownEvaluator(scene).score(tops)
        print("R %.3f  P %.3f  F1 %.3f  (%d detections, %d repeats, "
              "%d background)"
              % (result["recall"], result["precision"], result["f1"],
                 result["detections"], result["repeats"],
                 result["background"]))
    print("wrote %s" % args.output)
    return 0


def commandAnalyse(args):

    scene = buildScene(args)
    detector = buildDetector(args)
    tops = detector.detect(scene)
    evaluator = CrownEvaluator(scene)
    analysis = MissAnalysis(scene, tops, evaluator,
                            neighbourhood=args.neighbourhood,
                            canopyFraction=args.canopyFraction)
    analysis.report(detector)
    analysis.writeOutputs(args.output, detector, images=not args.noImages)
    return 0


def commandSweep(args):

    scene = buildScene(args)
    sweep = Sweep(scene, base=dict(windowSizeM=args.windowSize,
                                   minTreeAreaM2=args.minTreeArea))
    # The merge is swept alongside the detector, not left at a default. The
    # two interact: a base that over-detects less needs a gentler merge, so
    # tuning them separately finds the best of neither.
    mergerGrid = {"metric": [args.metric],
                  "epsM": [args.eps]}
    if args.metric == "saddle":
        mergerGrid["saddleDropM"] = _numbers(args.saddleDrops, float)
        _addSlopes(mergerGrid, "saddleDropSlope", args.saddleDropSlopes)
    elif args.metric == "composite":
        mergerGrid["heightWeight"] = [args.heightWeight]

    grid = {
        "lowerPercentile": _numbers(args.percentiles, int),
        "minTopAreaM2": _numbers(args.minTopAreas, float),
        "topStepM": _numbers(args.topSteps, float),
        "erosionIterations": _numbers(args.erosions, int),
    }
    _addSlopes(grid, "minTopAreaSlope", args.minTopAreaSlopes)
    sweep.run(grid, mergerGrid=mergerGrid)
    sweep.table(sortBy=args.sortBy, limit=args.limit)
    best = sweep.best(minimumRecall=args.minRecall)
    if best:
        print("\nbest: %s -> R %.1f%% P %.1f%% F1 %.1f%%"
              % (best["label"], 100 * best["recall"],
                 100 * best["precision"], 100 * best["f1"]))
    sweep.save(args.output)
    print("wrote %s" % args.output)
    return 0


def commandMerge(args):

    scene = buildScene(args)
    sweep = Sweep(scene, base=dict(
        windowSizeM=args.windowSize, lowerPercentile=args.lowerPercentile,
        minTopAreaM2=args.minTopArea, topStepM=args.topStep,
        erosionIterations=args.erosionIterations,
        minTreeAreaM2=args.minTreeArea))
    grid = {"metric": [m.strip() for m in args.metrics.split(",")],
            "epsM": _numbers(args.epsilons, float)}
    if "saddle" in grid["metric"]:
        grid["saddleDropM"] = _numbers(args.saddleDrops, float)
        _addSlopes(grid, "saddleDropSlope", args.saddleDropSlopes)
    if "composite" in grid["metric"]:
        grid["heightWeight"] = _numbers(args.weights, float)
    sweep.run({}, mergerGrid=grid)
    sweep.table(sortBy=args.sortBy, limit=args.limit)
    best = sweep.best(minimumRecall=args.minRecall)
    if best:
        print("\nbest: %s -> R %.1f%% P %.1f%% F1 %.1f%%"
              % (best["label"], 100 * best["recall"],
                 100 * best["precision"], 100 * best["f1"]))
    sweep.save(args.output)
    print("wrote %s" % args.output)
    return 0


def commandCrowns(args):

    scene = buildScene(args)
    tops = buildDetector(args).detect(scene)

    crowns = PseudoCrowns(scene, tops, percentile=args.crownPercentile,
                          windowSizeM=args.windowSize, many=args.many,
                          lots=args.lots, voronoiWeight=args.voronoiWeight)
    crowns.build()
    crowns.write(args.output)

    if scene.crowns is not None:
        comparison = crowns.compareWith(scene.crowns)
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
    addSceneArguments(detect)
    addDetectorArguments(detect)
    addMergerArguments(detect)
    detect.add_argument("--output", default="output/detect")
    detect.set_defaults(handler=commandDetect)
    return detect

def addAnalyseCommand(subparsers):
    analyse = subparsers.add_parser("analyse", help="what it gets wrong")
    addSceneArguments(analyse)
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
    addSceneArguments(sweep)
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
    addSceneArguments(merge)
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
    addSceneArguments(crowns)
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
