"""
Experiment — a detector tried over a parameter grid on a site and scored by an
evaluator — and CrossValidation, the leave-one-block-out protocol every method
is reported under.

Every setting is detected once over the whole site; each fold then keeps the
setting best on the other blocks and is scored on the held-out block it never
chose with. Running over the whole site is not leakage: the height model is the
input, not the label, and the held-out crowns are consulted only in their own
fold. It avoids re-detecting every setting in every fold and leaves no
artificial cliff where the held-out block would have been cut out.
"""

import multiprocessing
import os

import numpy as np

from ..data.detections import Detections
from ..scoring.evaluator import Evaluator
from ..scoring.reference import AnnotatedCrowns
from .runset import RunSet, poolFolds

_WORKER = None      # the experiment, shared with forked workers


def _detectInWorker(index):
    return _WORKER.detectOne(index)


def _detectAndScoreInWorker(index):
    detections = _WORKER.detectOne(index)
    return index, detections, _WORKER.evaluator.score(
        detections, _WORKER.data.blockArea)


def _scoreInWorker(task):
    region, setting = task
    return _WORKER.evaluator.score(_WORKER.detections[setting],
                                   _WORKER.regions[region])


class Experiment(object):

    def __init__(self, data, detector, grid=None, evaluator=None, jobs=1,
                 verbose=True, output=None):
        self.data = data
        self.detector = detector
        self.grid = grid if grid is not None else detector.parameterGrid()
        self.evaluator = evaluator or Evaluator(
            AnnotatedCrowns(data.annotations))
        self.jobs = max(1, int(jobs))
        self.verbose = verbose
        self.output = output
        self.settings = self.grid.expand()
        self.detections = None

    def detectorFor(self, setting):
        return self.detector.withParams(**setting)

    def detectOne(self, index):
        return self.detectorFor(self.settings[index]).detect(self.data)

    def run(self):
        raise NotImplementedError


class CrossValidation(Experiment):

    # grid axes whose values are compared as separately tuned runs, under
    # the pooled keys runs have always used
    comparedAxes = (("byMergeMetric", "mergeMetric", "saddle"),
                    ("bySmoothing", "smoothM", 0.0))
    progress = False        # print and log each setting as it is detected

    def detectAll(self):
        """One run per setting over the whole site; --jobs at a time."""
        global _WORKER
        for setting in self.settings:
            self.detectorFor(setting).prepare(self.data)
        indices = range(len(self.settings))
        _WORKER = self
        try:
            if self.progress:
                self.detections = self._detectWithProgress(indices)
            elif self.jobs == 1:
                self.detections = [self.detectOne(i) for i in indices]
            else:
                context = multiprocessing.get_context("fork")
                with context.Pool(self.jobs) as pool:
                    self.detections = pool.map(_detectInWorker, indices,
                                               chunksize=1)
        finally:
            _WORKER = None
        if self.verbose:
            print("  detected %d settings, %d jobs" % (len(self.settings),
                                                     self.jobs))
        return self.detections

    def _detectWithProgress(self, indices):
        """detectAll, each setting's whole-area score printed and logged."""
        keys = list(self.settings[0])
        path = (os.path.join(self.output, "progress.tsv")
                if self.output else None)
        if path:
            with open(path, "w") as handle:
                handle.write("\t".join(["done", "setting"] + keys + [
                    "tops", "recall", "precision", "f1", "weighted"]) + "\n")
        detections = [None] * len(indices)
        pool = None
        if self.jobs == 1:
            stream = (_detectAndScoreInWorker(i) for i in indices)
        else:
            pool = multiprocessing.get_context("fork").Pool(self.jobs)
            stream = pool.imap_unordered(_detectAndScoreInWorker, indices)
        try:
            for done, (index, found, score) in enumerate(stream, 1):
                detections[index] = found
                setting = self.settings[index]
                row = ([str(done), str(index)]
                       + [str(setting[k]) for k in keys] + [str(len(found))]
                       + ["%.3f" % score[k] for k in
                          ("recall", "precision", "f1", "weighted")])
                if path:
                    with open(path, "a") as handle:
                        handle.write("\t".join(row) + "\n")
                print("  [%d/%d] setting %d: R %.3f P %.3f F1 %.3f W %.3f  %s"
                      % (done, len(indices), index, score["recall"],
                         score["precision"], score["f1"], score["weighted"],
                         " ".join("%s=%s" % (k, setting[k]) for k in keys)),
                      flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
        return detections

    def scoreAll(self):
        """
        Every setting on every fold's training blocks (regions 0..n-1) and on
        all blocks (region n): {region: [scores per setting]}.
        """
        global _WORKER
        self.regions = ([self.data.trainingRegion(b.name)
                         for b in self.data.blocks] + [self.data.blockArea])
        tasks = [(r, s) for r in range(len(self.regions))
                 for s in range(len(self.settings))]
        _WORKER = self
        try:
            if self.jobs == 1:
                results = [_scoreInWorker(t) for t in tasks]
            else:
                context = multiprocessing.get_context("fork")
                with context.Pool(self.jobs) as pool:
                    results = pool.map(_scoreInWorker, tasks, chunksize=max(
                        1, len(tasks) // (self.jobs * 8)))
        finally:
            _WORKER = None
        self.scores = {r: [] for r in range(len(self.regions))}
        for (r, _), result in zip(tasks, results):
            self.scores[r].append(result)
        return self.scores

    def _tuning(self, region, indices=None):
        indices = range(len(self.settings)) if indices is None else indices
        return [(i, self.evaluator.objectiveOf(self.scores[region][i]))
                for i in indices]

    def foldResult(self, position, block, indices=None):
        """Tune on every other block, then score the chosen setting here."""
        tuning = self._tuning(position, indices)
        best, tuned = max(tuning, key=lambda item: item[1])
        result = self.evaluator.score(self.detections[best], block.geometry)
        # tuningF1 is the objective on the training blocks; the name is kept
        # for older readers
        result.update(block=block.name, settings=self.settings[best],
                      tuningF1=tuned, tuningScore=tuned)
        return result, best

    def run(self):
        """Detect, score, choose per fold. Returns a RunSet."""
        if self.verbose:
            print("[%s] %s, %d blocks, %d settings, one run each"
                  % (self.detector.method, self.data.name,
                     len(self.data.blocks), len(self.settings)))
        self.detectAll()
        self.scoreAll()
        folds, predictions = [], {}
        for position, block in enumerate(self.data.blocks):
            fold, best = self.foldResult(position, block)
            folds.append(fold)
            # the whole site, not just this block: combinations need this
            # fold's detections on its validation block as well
            chosen = self.detections[best].subset(
                np.arange(len(self.detections[best])))
            chosen.block, chosen.params = block.name, self.settings[best]
            predictions[block.name] = chosen
        if self.verbose:
            for fold in folds:
                printFold(self.detector.method, fold)
        pooled = poolFolds(folds, self.evaluator.objective)
        pooled["tuningOptimism"] = float(np.mean(
            [f["tuningScore"] - self.evaluator.objectiveOf(f) for f in folds]))
        pooled["wholeAreaBest"] = self.wholeAreaBest()
        for name, axis, default in self.comparedAxes:
            pooled[name] = self.byAxis(axis, default)
        return RunSet(self.detector.method, self.data.name, folds, pooled,
                      source=self.data.source, predictions=predictions)

    def restricted(self, indices):
        """The same cross-validation, choosing only among these settings."""
        folds = [self.foldResult(position, block, indices)[0]
                 for position, block in enumerate(self.data.blocks)]
        for fold in folds:
            del fold["tuningF1"], fold["tuningScore"]
        return poolFolds(folds, self.evaluator.objective)

    def byAxis(self, axis, default):
        """
        {value: pooled score choosing only among settings with that value},
        when the grid holds more than one value of the axis.
        """
        values = sorted({s.get(axis, default) for s in self.settings})
        if len(values) < 2:
            return {}
        return {str(v): self.restricted(
                    [i for i, s in enumerate(self.settings)
                     if s.get(axis, default) == v]) for v in values}

    def wholeAreaBest(self):
        """
        The best setting tuned and scored on all blocks at once: an optimistic
        ceiling, since the crowns it is scored on also chose it.
        """
        scores = self.scores[len(self.data.blocks)]
        best = int(np.argmax([self.evaluator.objectiveOf(r) for r in scores]))
        return dict(scores[best], settings=self.settings[best])

    def writeSettingsTable(self, path):
        """Every setting's score on all blocks, as a TSV."""
        whole = len(self.regions) - 1
        keys = list(self.settings[0])
        with open(path, "w") as handle:
            handle.write("\t".join(["setting"] + keys + [
                "tops", "recall", "precision", "f1", "weighted"]) + "\n")
            for i, setting in enumerate(self.settings):
                score = self.scores[whole][i]
                handle.write("\t".join(
                    [str(i)] + [str(setting[k]) for k in keys]
                    + [str(len(self.detections[i]))]
                    + ["%.3f" % score[k] for k in
                       ("recall", "precision", "f1", "weighted")]) + "\n")
        return path

    def saveAllDetections(self, path):
        """Every setting's points (x, y, score), in setting order (.npz)."""
        np.savez_compressed(path, **{
            "s%05d" % i: np.column_stack([d.x, d.y, d.scores]).reshape(-1, 3)
            for i, d in enumerate(self.detections)})
        return path


def printFold(method, fold):
    print("  fold %s: tuned %.3f -> held-out R %.3f P %.3f F1 %.3f  %s"
          % (fold["block"], fold["tuningScore"], fold["recall"],
             fold["precision"], fold["f1"], fold["settings"]))


def loadAllDetections(path, crs=None):
    """saveAllDetections read back: one points Detections per setting."""
    with np.load(path) as data:
        out = []
        for i in range(len(data.files)):
            rows = data["s%05d" % i]
            out.append(Detections(rows[:, 0], rows[:, 1], rows[:, 2],
                                  heights=rows[:, 2], crs=crs))
        return out
