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
import multiprocessing
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
# the minimum height and minimum tree area are fixed unless given as lists
# (--minHeights, --minTreeAreas); then they are tuned like the rest. The crowns
# scored never depend on them: only the detector's input does.
SCENE_KEY = "minHeight"
TREE_AREA_KEY = "minTreeAreaM2"

_WORKER = None      # the cross-validation, shared with forked workers


def _detectInWorker(index):
    return _WORKER.detectOne(index)


def _scoreInWorker(task):
    region, setting = task
    return dc.evaluateDetections(_WORKER.detections[setting],
                                 _WORKER.scene.crowns, _WORKER.regions[region])


def saveDetections(path, detections):
    """Every setting's tops (x, y, height), compactly, in setting order."""
    arrays = {"s%05d" % i: np.array([[p["centreX"], p["centreY"], p["score"]]
                                     for p in d], float).reshape(-1, 3)
              for i, d in enumerate(detections)}
    np.savez_compressed(path, **arrays)
    return path


def loadDetections(path):
    """saveDetections read back as prediction dicts."""
    with np.load(path) as data:
        return [[{"box": [x - POINT_BOX_M, y - POINT_BOX_M, x + POINT_BOX_M,
                          y + POINT_BOX_M],
                  "centreX": float(x), "centreY": float(y), "score": float(h)}
                 for x, y, h in data["s%05d" % i]]
                for i in range(len(data.files))]

# the half-width of the box drawn round each point, only so the shared scoring
# and NMS code, written for boxes, has something to work with
POINT_BOX_M = 0.5


class ConCompCrossValidation(object):

    def __init__(self, datasetPath, grid=None, resolution=0.25, minHeight=2.0,
                 minTreeAreaM2=0.5, windowSizeM=40.0, saddleEpsM=8.0,
                 output=None, verbose=True, jobs=1):
        self.output = output
        self.meta = dp.loadDataset(datasetPath)
        self.grid = grid or dict(DEFAULT_GRID)
        self.resolution = resolution
        self.minHeight = minHeight
        self.minTreeAreaM2 = minTreeAreaM2
        self.windowSizeM = windowSizeM
        self.saddleEpsM = saddleEpsM
        self.verbose = verbose
        self.jobs = max(1, int(jobs))

        self.scenes = {}
        self.scene = self._loadScene()
        self.blocks = self.meta["blockGeometries"]
        self.settings = self._expandGrid()
        self.detections = None

    # ------------------------------------------------------------------ #

    def _loadScene(self, minHeight=None):
        """The scene at one minimum height, built once and kept."""
        minHeight = self.minHeight if minHeight is None else minHeight
        if minHeight in self.scenes:
            return self.scenes[minHeight]
        chmPath = next((c["path"] for c in self.meta["channels"]
                        if c["kind"] == "chm"), None)
        if chmPath is None:
            raise ValueError("this dataset has no height model, so the "
                             "connected-component detector cannot run on it")
        scene = Scene(chmPath, crownsPath=self.meta["crownsPath"],
                      boundaryPath=self.meta["boundaryPath"],
                      resolution=self.resolution, minHeight=minHeight,
                      verbose=False)
        self.scenes[minHeight] = scene
        return scene

    def sceneFor(self, setting):
        return self._loadScene(setting.get(SCENE_KEY, self.minHeight))

    def _expandGrid(self):
        keys = list(self.grid)
        return [dict(zip(keys, values))
                for values in itertools.product(*[self.grid[k] for k in keys])]

    def _detector(self, setting):
        return ConCompDetector(
            windowSizeM=self.windowSizeM,
            minTreeAreaM2=setting.get(TREE_AREA_KEY, self.minTreeAreaM2),
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

    def detectOne(self, index):
        setting = self.settings[index]
        scene = self.sceneFor(setting)
        return self._asPredictions(self._detector(setting).detect(scene))

    def detectAll(self):
        """
        One detector run per setting, over the whole area; --jobs settings at
        a time in forked processes (the scenes are built first, so the
        workers share them). The detections are the same in any case.
        """
        global _WORKER
        for setting in self.settings:
            self.sceneFor(setting)
        indices = range(len(self.settings))
        if self.jobs == 1:
            self.detections = [self.detectOne(i) for i in indices]
        else:
            _WORKER = self
            context = multiprocessing.get_context("fork")
            with context.Pool(self.jobs) as pool:
                self.detections = pool.map(_detectInWorker, indices,
                                           chunksize=1)
            _WORKER = None
        if self.verbose:
            print("  detected %d settings, %d jobs" % (len(self.settings),
                                                     self.jobs))
        return self.detections

    def scoreAll(self):
        """
        Every setting on every fold's training blocks and on the whole area,
        --jobs at a time: {region index: [result per setting]}. Regions 0..n-1
        are the folds' training blocks, region n the whole area. Each score is
        computed exactly as one at a time would be.
        """
        global _WORKER
        names = [n for n, _ in self.blocks]
        self.regions = [unary_union([g for m, g in self.blocks if m != n])
                        for n in names] + [unary_union([g for _, g in
                                                         self.blocks])]
        tasks = [(r, s) for r in range(len(self.regions))
                 for s in range(len(self.settings))]
        if self.jobs == 1:
            _WORKER = self
            results = [_scoreInWorker(t) for t in tasks]
        else:
            _WORKER = self
            context = multiprocessing.get_context("fork")
            with context.Pool(self.jobs) as pool:
                results = pool.map(_scoreInWorker, tasks,
                                   chunksize=max(1, len(tasks) //
                                                 (self.jobs * 8)))
        _WORKER = None
        self.scores = {r: [] for r in range(len(self.regions))}
        for (r, _), result in zip(tasks, results):
            self.scores[r].append(result)
        return self.scores

    def foldResult(self, name, geometry):
        """Tune on every other block, then score the chosen setting here."""
        crowns = self.scene.crowns
        index = [n for n, _ in self.blocks].index(name)
        tuning = [r["f1"] for r in self.scores[index]]
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
        self.scoreAll()
        folds = [self.foldResult(name, geometry)
                 for name, geometry in self.blocks]
        if self.verbose:
            for fold in folds:
                self._printFold(fold)
        pooled = dc.averageFolds(folds)
        pooled["tuningOptimism"] = float(np.mean(
            [f["tuningF1"] - f["f1"] for f in folds]))
        pooled["wholeAreaBest"] = self.wholeAreaBest()
        return folds, pooled

    def wholeAreaBest(self):
        """
        The best setting tuned and scored on all blocks at once: the ceiling
        connected components can reach here, optimistic by construction since
        the crowns it is scored on also chose it.
        """
        scores = self.scores[len(self.blocks)]
        best = int(np.argmax([r["f1"] for r in scores]))
        return dict(scores[best], settings=self.settings[best])

    @staticmethod
    def _printFold(fold):
        s = fold["settings"]
        extra = "".join(", %s %s" % (k, s[k]) for k in (SCENE_KEY,
                                                         TREE_AREA_KEY)
                        if k in s)
        print("  fold %s: tuned F1 %.3f -> held-out R %.3f P %.3f F1 %.3f  "
              "(%dth, minTop %.2f, step %.2f, erode %d, drop %.2f%s)"
              % (fold["block"], fold["tuningF1"], fold["recall"],
                 fold["precision"], fold["f1"], s["lowerPercentile"],
                 s["minTopAreaM2"], s["topStepM"], s["erosionIterations"],
                 s["saddleDropM"], extra))


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
    for key, option in ((SCENE_KEY, args.minHeights),
                        (TREE_AREA_KEY, args.minTreeAreas)):
        if option:
            grid[key] = [float(v) for v in option.split(",") if v.strip()]
    return grid


def crossValidate(args):
    os.makedirs(args.output, exist_ok=True)
    dc.startLogging(args.log or os.path.join(args.output, "run.log"))

    validation = ConCompCrossValidation(
        args.dataset, grid=gridFromArguments(args),
        resolution=args.resolution, minHeight=args.minHeight,
        minTreeAreaM2=args.minTreeArea, output=args.output, jobs=args.jobs)
    folds, pooled = validation.run()
    if args.saveDetections:
        saveDetections(os.path.join(args.output, "detections.npz"),
                       validation.detections)

    print("\n[concomp] pooled over %d folds: R %.3f  P %.3f  F1 %.3f "
          "(per-fold F1 %.3f +- %.3f)"
          % (pooled["folds"], pooled["recall"], pooled["precision"],
             pooled["f1"], pooled["perFoldF1Mean"], pooled["perFoldF1Std"]))
    print("[concomp] tuning optimism: %+.3f F1 — how much the figure would "
          "have been overstated by reporting the tuning score"
          % pooled["tuningOptimism"])
    ceiling = pooled["wholeAreaBest"]
    print("[concomp] whole-area best (optimistic ceiling): R %.3f  P %.3f  "
          "F1 %.3f at %s" % (ceiling["recall"], ceiling["precision"],
                             ceiling["f1"], ceiling["settings"]))

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
    parser.add_argument("--minHeights", default=None,
                        help="Minimum heights to tune over (else --minHeight "
                             "is fixed)")
    parser.add_argument("--minTreeAreas", default=None,
                        help="Minimum tree areas to tune over (else "
                             "--minTreeArea is fixed)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Settings detected at once, in parallel")
    parser.add_argument("--saveDetections", action="store_true",
                        help="Keep every setting's detections in "
                             "detections.npz (for tt.pseudoTuning)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(0 if crossValidate(parseArguments()) else 1)
