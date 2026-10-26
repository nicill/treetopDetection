"""
Experiments that choose nothing on the site, or choose with less of it.

FixedSettings — a detector at settings fixed elsewhere (its defaults, or the
setting another site's cross-validation chose most often), detected once over
the site and scored block by block and over the whole area.

LearningCurve — how much annotation a tuned detector needs: for every held-out
block, the annotation is cut down to a fraction of each training block's tiles
(PatchSet.draw: whole tiles, at random, the same fraction from every block),
drawn with fixed seeds; the setting is chosen on the crowns of the drawn tiles
only (those inside the scored area: a drawn tile may reach past it) and
scored on the whole held-out block. Every setting's detections are
given (a cross-validation's saved detections), so only the choice changes.
"""

import json
import multiprocessing
import time

import numpy as np

from ..data.annotations import crownsInRegion
from .crossValidation import Experiment
from .runset import RunSet, matchBlock, poolFolds

_CURVE = None      # the curve being scored, shared with forked workers


class FixedSettings(Experiment):

    def __init__(self, data, detector, setting, evaluator=None,
                 verbose=False):
        super().__init__(data, detector.withParams(**setting),
                         evaluator=evaluator, verbose=verbose)
        self.setting = dict(setting)

    def run(self):
        """A RunSet: every block scored, the whole site in each fold file."""
        detections = self.detector.detect(self.data)
        folds, predictions = [], {}
        for block in self.data.blocks:
            folds.append(dict(self.evaluator.score(detections, block.geometry),
                              block=block.name))
            copy = detections.subset(np.arange(len(detections)))
            copy.block = block.name
            copy.fileFields = {"settings": self.setting, "region": "area"}
            predictions[block.name] = copy
        return RunSet(self.detector.method, self.data.name, folds,
                      poolFolds(folds, self.evaluator.objective),
                      settings=self.setting, predictions=predictions,
                      extra={"wholeArea": self.evaluator.score(
                          detections, self.data.area)})


class GridSearch(Experiment):
    """
    Every setting of a grid scored on all of a site's crowns: what to try,
    not a figure to quote (the crowns that score a setting also chose it;
    use CrossValidation for that).
    """

    def run(self):
        """One row per setting: its scores, its setting and a label."""
        rows = []
        started = time.time()
        if self.verbose:
            print("%d crowns, %d settings" % (len(self.evaluator.reference
                                                  .crowns()),
                                              len(self.settings)))
        for index, setting in enumerate(self.settings):
            detections = self.detectOne(index)
            matching = self.evaluator.match(detections)
            total = float(max(len(detections), 1))
            row = dict(matching.scores(), detections=len(detections),
                       background=matching.count("background"),
                       repeatShare=matching.count("repeat") / total,
                       backgroundShare=matching.count("background") / total)
            row.update(setting)
            row["label"] = label(setting)
            rows.append(row)
            if self.verbose and (index + 1) % 10 == 0:
                elapsed = time.time() - started
                print("  %d/%d (%.0fs elapsed, ~%.0fs left)"
                      % (index + 1, len(self.settings), elapsed,
                         elapsed / (index + 1)
                         * (len(self.settings) - index - 1)))
        self.rows = rows
        return rows

    def best(self, minimumRecall=None):
        candidates = [r for r in self.rows
                      if minimumRecall is None or r["recall"] >= minimumRecall]
        return max(candidates, key=lambda r: r["f1"]) if candidates else None

    def table(self, sortBy="f1", limit=20):
        """The best rows, showing only the axes that vary."""
        keys = self.grid.varying()
        rows = sorted(self.rows, key=lambda r: -r[sortBy])[:limit]
        width = max(24, min(56, sum(len(shortName(k)) + 8 for k in keys)))
        print("%-*s %6s %7s %7s %7s %8s"
              % (width, "setting", "dets", "recall", "prec", "F1", "repeat%"))
        print("-" * (width + 42))
        for row in rows:
            label = " ".join("%s=%s" % (shortName(k), row[k]) for k in keys)
            print("%-*s %6d %6.1f%% %6.1f%% %6.1f%% %7.1f%%"
                  % (width, label[:width], row["detections"],
                     100 * row["recall"], 100 * row["precision"],
                     100 * row["f1"], 100 * row["repeatShare"]))
        return rows

    def save(self, path):
        with open(path, "w") as handle:
            json.dump(self.rows, handle, indent=1, default=float)
        return path


SHORT = {"lowerPercentile": "pct", "minTopAreaM2": "minTop",
         "topStepM": "step", "erosionIterations": "erode",
         "saddleDropM": "drop", "heightWeight": "w", "saddleEpsM": "eps",
         "mergeMetric": "metric", "windowSizeM": "window",
         "minTreeAreaM2": "minTree", "minTopAreaSlope": "minTopSlope",
         "saddleDropSlope": "dropSlope"}


# the merge parameters, ordered as labels have always listed them
MERGE_ORDER = {"saddleEpsM": "epsM", "heightWeight": "heightWeight",
               "mergeMetric": "metric", "saddleDropM": "saddleDropM",
               "saddleDropSlope": "saddleDropSlope"}


def shortName(key):
    return SHORT.get(key, key)


def label(setting):
    """
    A setting in short form: detector parameters first, then the merge
    ones, each group alphabetical (the labels of existing sweep files).
    """
    detector = sorted(k for k in setting if k not in MERGE_ORDER)
    merge = sorted((k for k in setting if k in MERGE_ORDER),
                   key=lambda k: MERGE_ORDER[k])
    return " ".join("%s=%s" % (shortName(k), setting[k])
                    for k in detector + merge)


def _scoreInWorker(task):
    region, setting = task
    return _CURVE.evaluator.objectiveOf(_CURVE.evaluator.score(
        _CURVE.detections[setting], _CURVE.regions[region]))


class LearningCurve(object):
    """
    data: the site (its blocks are scored); settings and detections: every
    setting and its Detections over the site; patchSet: the tiles drawn from
    (the RGB patch set, so the networks draw the same ones); evaluator: what
    every region is scored against.
    """

    def __init__(self, data, settings, detections, patchSet, evaluator,
                 fractions, seeds, bufferM=10.0, jobs=1):
        if len(detections) != len(settings):
            raise ValueError("%d saved settings, the grid has %d"
                             % (len(detections), len(settings)))
        self.data, self.settings, self.detections = data, settings, detections
        self.patchSet, self.evaluator = patchSet, evaluator
        self.fractions, self.seeds = list(fractions), list(seeds)
        self.bufferM = bufferM
        self.jobs = max(1, int(jobs))

    def draws(self):
        """(fold, fraction, seed, drawn patches) for every point."""
        names = dict(self.patchSet.blocks)
        draws = []
        for block in self.data.blocks:
            held = matchBlock(block.name, names)
            train, _ = self.patchSet.foldSplit(held, self.bufferM)
            for fraction in self.fractions:
                for seed in (self.seeds[:1] if fraction >= 1 else self.seeds):
                    draws.append((block.name, fraction, seed,
                                  self.patchSet.draw(train, fraction, seed,
                                                     held)))
        return draws

    def scoreRegions(self):
        """Every setting's tuning score on every draw, jobs at a time."""
        global _CURVE
        tasks = [(r, s) for r in range(len(self.regions))
                 for s in range(len(self.settings))]
        _CURVE = self
        try:
            if self.jobs == 1:
                results = [_scoreInWorker(t) for t in tasks]
            else:
                with multiprocessing.get_context("fork").Pool(
                        self.jobs) as pool:
                    results = pool.map(_scoreInWorker, tasks, chunksize=max(
                        1, len(tasks) // (self.jobs * 8)))
        finally:
            _CURVE = None
        return np.array(results).reshape(len(self.regions),
                                         len(self.settings))

    def run(self):
        draws = self.draws()
        self.regions = [self.patchSet.region(p) for _, _, _, p in draws]
        print("[curve] %s: %d settings, %d draws over %d blocks"
              % (self.data.name, len(self.settings), len(draws),
                 len(self.data.blocks)))
        scores = self.scoreRegions()
        crowns = self.evaluator.reference.crowns()
        points = {}
        for (name, fraction, seed, patches), region, row in zip(
                draws, self.regions, scores):
            best = int(np.argmax(row))
            fold = self.evaluator.score(self.detections[best],
                                        self.data.block(name).geometry)
            fold.update(block=name, chosen=best, tuningScore=float(row[best]),
                        tiles=len(patches),
                        crownsUsed=len(crownsInRegion(crowns, region)))
            points.setdefault((fraction, seed), []).append(fold)
        return [{"fraction": f, "seed": s, "folds": folds,
                 "pooled": poolPoint(folds, self.evaluator.objective)}
                for (f, s), folds in sorted(points.items())]

    def transfer(self, setting):
        """Another site's setting, applied here with no tuning."""
        if setting not in self.settings:
            raise ValueError("the other site's best setting %s is not in "
                             "this run's grid" % setting)
        index = self.settings.index(setting)
        folds = [dict(self.evaluator.score(self.detections[index],
                                           b.geometry),
                      block=b.name, chosen=index, tiles=0, crownsUsed=0)
                 for b in self.data.blocks]
        return {"settings": setting, "folds": folds,
                "pooled": poolPoint(folds, self.evaluator.objective)}


def poolPoint(folds, objective=None):
    pooled = poolFolds(folds, objective)
    pooled["crownsUsedPerFold"] = float(np.mean([f["crownsUsed"]
                                                 for f in folds]))
    return pooled

