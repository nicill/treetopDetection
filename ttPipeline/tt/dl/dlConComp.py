#!/usr/bin/env python
"""
dlConComp.py

The connected-component detector run through the same leave-one-block-out
scheme as the learned models, so all three can share one table.

    python -m tt.dl concomp --dataset datasets/lidarChm --output runs/cc

Each fold tunes on the training blocks and is scored on the held-out block it
never saw — exactly what the learned models get. Without that the comparison
would be rigged in favour of the hand-tuned method.

How the work is organised
-------------------------
Every setting in the tuning grid is run once, over the whole area, and the
folds are then scored from those detections by selecting what falls in each
block. An earlier version re-ran every setting inside every fold on that fold's
training blocks; since each block is a training block in seven of eight folds,
every detection was computed seven times over — 200 detector runs where 24
suffice.

Running over the whole area is not leakage. The CHM is the input, not the
label: tuning still scores against training-block crowns only, and the held-out
block's crowns are never consulted until its own fold. What changes is that the
detector no longer sees an artificial cliff where the held-out block was zeroed
out, so trees near block borders are detected as they would be in practice.

The detector has no notion of confidence, so a detection's height stands in as
its score. That only decides which detection is primary inside a crown when
several land in the same one, the rule the detector already uses internally.
"""

import argparse
import itertools
import os
import sys

import numpy as np
from shapely.ops import unary_union

from ..detector import ConCompDetector
from ..merging import TopMerger
from ..scene import Scene
from . import dlCommon as dc
from . import dlPrepare as dp

# The merge threshold is tuned alongside the detector, because the two interact:
# a base that over-detects less needs a gentler merge, so tuning them separately
# finds the best of neither.
DEFAULT_GRID = {
    "lowerPercentile": [20, 30],
    "minTopAreaM2": [0.12, 0.06],
    "topStepM": [0.25],
    "erosionIterations": [1, 2],
    "saddleDropM": [0.2, 0.3, 0.5],
}
GRID_CASTS = {"lowerPercentile": int, "erosionIterations": int}

# the half-width of the box drawn round each point, only so the shared scoring
# and NMS code, written for boxes, has something to work with
POINT_BOX_M = 0.5


class ConCompCrossValidation(object):

    def __init__(self, datasetPath, grid=None, resolution=0.25, minHeight=2.0,
                 minTreeAreaM2=0.5, windowSizeM=40.0, saddleEpsM=8.0,
                 output=None, verbose=True):
        self.output = output
        self.meta = dp.loadDataset(datasetPath)
        self.grid = grid or dict(DEFAULT_GRID)
        self.resolution = resolution
        self.minHeight = minHeight
        self.minTreeAreaM2 = minTreeAreaM2
        self.windowSizeM = windowSizeM
        self.saddleEpsM = saddleEpsM
        self.verbose = verbose

        self.scene = self._loadScene()
        self.blocks = self.meta["blockGeometries"]
        self.settings = self._expandGrid()
        self.detections = None

    # ------------------------------------------------------------------ #

    def _loadScene(self):
        chmPath = next((c["path"] for c in self.meta["channels"]
                        if c["kind"] == "chm"), None)
        if chmPath is None:
            raise ValueError("this dataset has no height model, so the "
                             "connected-component detector cannot run on it")
        return Scene(chmPath, crownsPath=self.meta["crownsPath"],
                     boundaryPath=self.meta["boundaryPath"],
                     resolution=self.resolution, minHeight=self.minHeight,
                     verbose=False)

    def _expandGrid(self):
        keys = list(self.grid)
        return [dict(zip(keys, values))
                for values in itertools.product(*[self.grid[k] for k in keys])]

    def _detector(self, setting):
        return ConCompDetector(
            windowSizeM=self.windowSizeM, minTreeAreaM2=self.minTreeAreaM2,
            lowerPercentile=setting["lowerPercentile"],
            minTopAreaM2=setting["minTopAreaM2"],
            topStepM=setting["topStepM"],
            erosionIterations=setting["erosionIterations"],
            merger=TopMerger("saddle", epsM=self.saddleEpsM,
                             saddleDropM=setting["saddleDropM"]),
            verbose=False)

    def _asPredictions(self, tops):
        predictions = []
        for x, y, height in tops.points:
            east, north = self.scene.toWorld(x, y)
            predictions.append({
                "box": [east - POINT_BOX_M, north - POINT_BOX_M,
                        east + POINT_BOX_M, north + POINT_BOX_M],
                "centreX": float(east), "centreY": float(north),
                "score": float(height)})
        return predictions

    # ------------------------------------------------------------------ #

    def detectAll(self):
        """One detector run per setting, over the whole area."""
        self.detections = []
        for index, setting in enumerate(self.settings):
            tops = self._detector(setting).detect(self.scene)
            self.detections.append(self._asPredictions(tops))
            if self.verbose:
                print("  detected setting %d/%d: %d tops"
                      % (index + 1, len(self.settings), len(tops)))
        return self.detections

    def foldResult(self, name, geometry):
        """Tune on every other block, then score the chosen setting here."""
        training = unary_union([g for n, g in self.blocks if n != name])
        crowns = self.scene.crowns

        tuning = [dc.evaluateDetections(p, crowns, training)["f1"]
                  for p in self.detections]
        best = int(np.argmax(tuning))

        result = dc.evaluateDetections(self.detections[best], crowns,
                                       geometry)
        result.update(block=name, settings=self.settings[best],
                      tuningF1=tuning[best])
        if self.output:
            # the whole area, not just this block: fusion needs this fold's
            # detections on its validation block as well as its test block
            dc.saveJson({"block": name, "settings": self.settings[best],
                         "region": "area", "predictions": self.detections[best]},
                        os.path.join(self.output, "predictions_%s.json" % name))
        return result

    def run(self):
        if self.verbose:
            print("[concomp] %s, %d blocks, %d settings, one run each"
                  % (self.meta["name"], len(self.blocks), len(self.settings)))
        self.detectAll()
        folds = [self.foldResult(name, geometry)
                 for name, geometry in self.blocks]
        if self.verbose:
            for fold in folds:
                self._printFold(fold)
        pooled = dc.averageFolds(folds)
        pooled["tuningOptimism"] = float(np.mean(
            [f["tuningF1"] - f["f1"] for f in folds]))
        return folds, pooled

    @staticmethod
    def _printFold(fold):
        s = fold["settings"]
        print("  fold %s: tuned F1 %.3f -> held-out R %.3f P %.3f F1 %.3f  "
              "(%dth, minTop %.2f, step %.2f, erode %d, drop %.2f)"
              % (fold["block"], fold["tuningF1"], fold["recall"],
                 fold["precision"], fold["f1"], s["lowerPercentile"],
                 s["minTopAreaM2"], s["topStepM"], s["erosionIterations"],
                 s["saddleDropM"]))


# ---------------------------------------------------------------------- #

def gridFromArguments(args):
    grid = dict(DEFAULT_GRID)
    for key, option in (("lowerPercentile", args.percentiles),
                        ("minTopAreaM2", args.minTopAreas),
                        ("topStepM", args.topSteps),
                        ("erosionIterations", args.erosions),
                        ("saddleDropM", args.saddleDrops)):
        if option:
            cast = GRID_CASTS.get(key, float)
            grid[key] = [cast(v) for v in option.split(",") if v.strip()]
    return grid


def crossValidate(args):
    os.makedirs(args.output, exist_ok=True)
    dc.startLogging(args.log or os.path.join(args.output, "run.log"))

    validation = ConCompCrossValidation(
        args.dataset, grid=gridFromArguments(args),
        resolution=args.resolution, minHeight=args.minHeight,
        minTreeAreaM2=args.minTreeArea, output=args.output)
    folds, pooled = validation.run()

    print("\n[concomp] pooled over %d folds: R %.3f  P %.3f  F1 %.3f "
          "(per-fold F1 %.3f +- %.3f)"
          % (pooled["folds"], pooled["recall"], pooled["precision"],
             pooled["f1"], pooled["perFoldF1Mean"], pooled["perFoldF1Std"]))
    print("[concomp] tuning optimism: %+.3f F1 — how much the figure would "
          "have been overstated by reporting the tuning score"
          % pooled["tuningOptimism"])

    dc.saveJson({"method": "concomp", "dataset": validation.meta["name"],
                 "source": validation.meta["source"], "settings": vars(args),
                 "folds": folds, "pooled": pooled},
                os.path.join(args.output, "results.json"))
    print("[concomp] wrote %s" % os.path.join(args.output, "results.json"))
    return pooled


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Connected-component detection under leave-one-block-out.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", default=None,
                        help="Also write everything printed to this file. "
                             "Defaults to <output>/run.log")
    parser.add_argument("--resolution", type=float, default=0.25,
                        help="Working resolution for the detector (default: "
                             "0.25). Independent of the tile resolution the "
                             "learned models use; each method gets its own")
    parser.add_argument("--minHeight", type=float, default=2.0)
    parser.add_argument("--minTreeArea", type=float, default=0.5,
                        help="Smallest blob searched for tops, m2 (default: "
                             "0.5, matching the sweeps and the CLI)")
    parser.add_argument("--percentiles", default=None,
                        help="Override the tuning grid, comma separated")
    parser.add_argument("--minTopAreas", default=None)
    parser.add_argument("--topSteps", default=None)
    parser.add_argument("--erosions", default=None)
    parser.add_argument("--saddleDrops", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(0 if crossValidate(parseArguments()) else 1)
