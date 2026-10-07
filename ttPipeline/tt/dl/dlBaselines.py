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

import numpy as np
from scipy.ndimage import label, maximum_filter, watershed_ift

from . import dlCommon as dc
from .dlConComp import (ConCompCrossValidation, SCENE_KEY, SMOOTH_KEY,
                        saveDetections)

GRIDS = {
    "lmvw": {"windowBaseM": [0.5, 1.0, 1.5, 2.0],
             "windowSlope": [0.0, 0.05, 0.1, 0.15, 0.2]},
    "watershed": {"markerWindowM": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
                  "minCrownAreaM2": [0.0, 0.5, 1.0, 2.0]},
}
MAX_WINDOW_M = 12.0     # no window larger than this, whatever the height


class Tops(object):
    """What the cross-validation reads from a detector: (x, y, height)."""

    def __init__(self, points):
        self.points = points


def oddPixels(metres, pixel):
    """A window side in pixels, odd, at least 3."""
    side = int(round(metres / pixel))
    return max(3, side + (side + 1) % 2)


def plateauTops(chm, candidates):
    """One top per 8-connected group of candidate pixels: its highest pixel."""
    labels, count = label(candidates, structure=np.ones((3, 3), bool))
    if count == 0:
        return []
    rows, cols = np.nonzero(labels)
    order = np.lexsort((-chm[rows, cols], labels[rows, cols]))
    rows, cols = rows[order], cols[order]
    first = np.r_[True, np.diff(labels[rows, cols]) != 0]
    return [(int(c), int(r), float(chm[r, c]))
            for r, c in zip(rows[first], cols[first])]


class LocalMaximaVariableWindow(object):

    def __init__(self, windowBaseM, windowSlope):
        self.base, self.slope = windowBaseM, windowSlope

    def detect(self, scene):
        chm, pixel = scene.chm, scene.pixelSize
        valid = chm > 0
        window = np.minimum(self.base + self.slope * chm, MAX_WINDOW_M)
        # each pixel's window side in pixels, odd and at least 3
        sides = np.zeros(chm.shape, int)
        sides[valid] = np.maximum(3, np.rint(window[valid] / pixel).astype(int))
        sides[valid] += (sides[valid] + 1) % 2
        candidates = np.zeros(chm.shape, bool)
        for side in np.unique(sides[valid]):
            here = valid & (sides == side)
            candidates |= here & (chm >= maximum_filter(chm, size=int(side)))
        return Tops(plateauTops(chm, candidates))


class MarkerWatershed(object):

    def __init__(self, markerWindowM, minCrownAreaM2):
        self.window, self.minArea = markerWindowM, minCrownAreaM2

    def detect(self, scene):
        chm, pixel = scene.chm, scene.pixelSize
        valid = chm > 0
        side = oddPixels(self.window, pixel)
        peaks = valid & (chm >= maximum_filter(chm, size=side))
        tops = plateauTops(chm, peaks)
        if not tops:
            return Tops([])
        markers = np.zeros(chm.shape, np.int32)
        for index, (x, y, _) in enumerate(tops, 1):
            markers[y, x] = index
        markers[~valid] = -1                     # ground floods as background
        top = float(chm.max())
        depth = np.round((top - chm) / max(top, 1e-9) * 65000).astype(np.uint16)
        regions = watershed_ift(depth, markers)
        areas = np.bincount(regions[regions > 0].ravel(),
                            minlength=len(tops) + 1) * pixel * pixel
        return Tops([t for i, t in enumerate(tops, 1)
                     if areas[i] >= self.minArea])


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


def gridFor(args):
    grid = {k: list(v) for k, v in GRIDS[args.method].items()}
    if args.minHeights:
        grid[SCENE_KEY] = [float(v) for v in args.minHeights.split(",")]
    if args.smooths:
        grid[SMOOTH_KEY] = [float(v) for v in args.smooths.split(",")]
    return grid


def crossValidate(args):
    os.makedirs(args.output, exist_ok=True)
    dc.startLogging(os.path.join(args.output, "run.log"))
    validation = BaselineCrossValidation(
        args.method, args.dataset, grid=gridFor(args),
        resolution=args.resolution, minHeight=args.minHeight,
        output=args.output, jobs=args.jobs)
    folds, pooled = validation.run()
    if args.saveDetections:
        saveDetections(os.path.join(args.output, "detections.npz"),
                       validation.detections)
    print("\n[%s] pooled over %d folds: R %.3f P %.3f F1 %.3f weighted %.3f "
          "(objective %s)" % (args.method, pooled["folds"], pooled["recall"],
                              pooled["precision"], pooled["f1"],
                              pooled["weighted"], dc.tuningObjective()))
    ceiling = pooled["wholeAreaBest"]
    print("[%s] whole-area best (optimistic ceiling): F1 %.3f at %s"
          % (args.method, ceiling["f1"], ceiling["settings"]))
    dc.saveJson({"method": "baseline-" + args.method,
                 "dataset": validation.meta["name"],
                 "source": validation.meta["source"], "settings": vars(args),
                 "folds": folds, "pooled": pooled},
                os.path.join(args.output, "results.json"))
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
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--saveDetections", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(0 if crossValidate(parseArguments()) else 1)
