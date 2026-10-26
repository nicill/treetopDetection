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

import multiprocessing

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

