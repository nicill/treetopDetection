"""
Classical tree-top detectors on the CHM, as baselines, under exactly the
leave-one-block-out tuning and scoring of connected components
(dlConComp.ConCompCrossValidation): every setting of a small grid detects once
over the whole area, each fold keeps the setting best on the other blocks, and
it is scored on the held-out block. Same results.json and predictions files,
so the summary and the reports read them like any run.

    python -m tt.dl.dlBaselines --method lmvw --dataset <site>/ds/chm \\
        --output <site>/runs/lmvwLidar --resolution 0.25 --minHeight 2.0 \\
        --minHeights 0.5,1.0,1.5 --smooths 0,0.25 --jobs 20

Methods
  lmvw       local maxima in a window whose size grows with height
             (Popescu and Wynne 2004): a pixel is a top when it is the highest
             in the square window of side base + slope x its own height.
             slope 0 is the fixed-window local maximum.
  watershed  marker-controlled watershed: markers are the local maxima in a
             fixed window; the CHM is flooded from them (scipy watershed_ift)
             and a marker is kept as a tree when its region covers at least
             the minimum crown area. The top is the marker.

Minimum height and smoothing are the scene's (SCENE_KEY, SMOOTH_KEY), tuned
when listed, as for CC. A plateau of equal maxima gives one top.
"""

import argparse
import os
import sys

from ..detectors.height import (LocalMaximaDetector, WatershedDetector,
                               localMaxima, markerWatershed)
from ..data.site import TDData
from ..experiments import CrossValidation, ParameterGrid
from . import dlCommon as dc
from .dlConComp import (ConCompCrossValidation, SCENE_KEY, SMOOTH_KEY,
                        saveDetections)

GRIDS = {"lmvw": LocalMaximaDetector.defaultGrid,
         "watershed": WatershedDetector.defaultGrid}


class Tops(object):
    """What the cross-validation reads from a detector: (x, y, height)."""

    def __init__(self, points):
        self.points = points


class LocalMaximaVariableWindow(object):

    def __init__(self, windowBaseM, windowSlope):
        self.base, self.slope = windowBaseM, windowSlope

    def detect(self, scene):
        return Tops(localMaxima(scene.chm, scene.pixelSize, self.base,
                                self.slope))


class MarkerWatershed(object):

    def __init__(self, markerWindowM, minCrownAreaM2):
        self.window, self.minArea = markerWindowM, minCrownAreaM2

    def detect(self, scene):
        return Tops(markerWatershed(scene.chm, scene.pixelSize, self.window,
                                    self.minArea))


BASELINES = {"lmvw": LocalMaximaDetector, "watershed": WatershedDetector}

DETECTORS = {
    "lmvw": lambda s: LocalMaximaVariableWindow(s["windowBaseM"],
                                                s["windowSlope"]),
    "watershed": lambda s: MarkerWatershed(s["markerWindowM"],
                                           s["minCrownAreaM2"]),
}


class BaselineCrossValidation(ConCompCrossValidation):
    """CC's cross-validation with a classical detector in its place."""

    def __init__(self, method, *args, **keywords):
        self.method = method
        super().__init__(*args, **keywords)

    def _detector(self, setting):
        return DETECTORS[self.method](setting)

    @staticmethod
    def _printFold(fold):
        print("  fold %s: tuned %.3f -> held-out R %.3f P %.3f F1 %.3f  %s"
              % (fold["block"], fold["tuningScore"], fold["recall"],
                 fold["precision"], fold["f1"], fold["settings"]))


OVERRIDES = {"windowBaseM": "windowBases", "windowSlope": "windowSlopes",
             "markerWindowM": "markerWindows", "minCrownAreaM2": "minCrownAreas"}


def gridFor(args):
    """The method's grid, a dimension replaced when given on the command line
    (a site whose trees are larger than the default grid allows for)."""
    grid = {k: list(v) for k, v in GRIDS[args.method].items()}
    for key in grid:
        given = getattr(args, OVERRIDES[key], None)
        if given:
            grid[key] = [float(v) for v in given.split(",")]
    if args.minHeights:
        grid[SCENE_KEY] = [float(v) for v in args.minHeights.split(",")]
    if args.smooths:
        grid[SMOOTH_KEY] = [float(v) for v in args.smooths.split(",")]
    return grid


def crossValidate(args):
    """The baseline command, run through the experiment classes."""
    os.makedirs(args.output, exist_ok=True)
    dc.startLogging(os.path.join(args.output, "run.log"))
    data = TDData.fromDataset(args.dataset)
    detector = BASELINES[args.method](resolution=args.resolution,
                                      minHeight=args.minHeight)
    validation = CrossValidation(data, detector, ParameterGrid(gridFor(args)),
                                 jobs=args.jobs, output=args.output)
    runSet = validation.run()
    runSet.settings = vars(args)
    runSet.save(args.output)
    validation.writeSettingsTable(os.path.join(args.output, "settings.tsv"))
    if args.saveDetections:
        validation.saveAllDetections(os.path.join(args.output,
                                                  "detections.npz"))
    pooled = runSet.pooled
    print("\n[%s] pooled over %d folds: R %.3f P %.3f F1 %.3f weighted %.3f "
          "(objective %s)" % (args.method, pooled["folds"], pooled["recall"],
                              pooled["precision"], pooled["f1"],
                              pooled["weighted"], pooled["objective"]))
    ceiling = pooled["wholeAreaBest"]
    print("[%s] whole-area best (optimistic ceiling): F1 %.3f at %s"
          % (args.method, ceiling["f1"], ceiling["settings"]))
    return pooled


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Classical CHM tree-top baselines, leave-one-block-out.")
    parser.add_argument("--method", choices=sorted(DETECTORS), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--minHeight", type=float, default=2.0)
    parser.add_argument("--minHeights", default=None,
                        help="minimum heights to tune over, as for CC")
    parser.add_argument("--smooths", default=None,
                        help="Gaussian smoothing (m) to tune over, as for CC")
    parser.add_argument("--windowBases", default=None,
                        help="lmvw: window side at height 0, m (comma list)")
    parser.add_argument("--windowSlopes", default=None,
                        help="lmvw: window growth per metre of height")
    parser.add_argument("--markerWindows", default=None,
                        help="watershed: marker window sides, m")
    parser.add_argument("--minCrownAreas", default=None,
                        help="watershed: minimum crown areas, m2")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--saveDetections", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(0 if crossValidate(parseArguments()) else 1)
