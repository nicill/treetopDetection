"""
Tuning without crowns: every setting of a detector's grid scored, block by
held-out block, against scorers that never look at the block's crowns, and
the setting each picks then scored against the crowns.

    real        the block's own crowns: what tuning on the block itself would
                pick (an optimistic bound, never a method)
    pseudoF1    another detector's held-out boxes for the block, above the
                operating point its own cross-validation chose, as crowns
                (scoring.BoxCrowns)
    zones@c     that detector's boxes at confidence c or more: the share
                holding exactly one top. Nothing outside a box counts, so
                finding trees it missed is free.

The box detector made those predictions without seeing the block. For each
scorer the setting it picks is scored against the real crowns, pooled over
blocks like any fold, and its ranking of all settings compared with the real
ranking (Spearman).
"""

import multiprocessing

import numpy as np
from scipy.stats import spearmanr

from ..scoring.evaluator import Evaluator
from ..scoring.reference import AnnotatedCrowns, BoxCrowns
from .runset import RunSet, poolFolds, sharedBlocks

ZONE_LEVELS = (0.5, 0.7, 0.9)

_BLOCK = None      # the block being scored, shared with forked workers


def _scoreReal(index):
    return _BLOCK.real.score(_BLOCK.inBlock[index], _BLOCK.region)


def _scorePseudo(index):
    return _BLOCK.pseudo.objectiveOf(_BLOCK.pseudo.score(
        _BLOCK.inBlock[index], _BLOCK.region))


def exactlyOneShare(points, boxes):
    """Share of boxes holding exactly one point."""
    if not len(boxes) or not len(points):
        return 0.0
    x, y = points.x[None, :], points.y[None, :]
    inside = ((x >= boxes[:, 0:1]) & (x <= boxes[:, 2:3]) &
              (y >= boxes[:, 1:2]) & (y <= boxes[:, 3:4]))
    return float((inside.sum(axis=1) == 1).mean())


class BlockScores(object):
    """One held-out block: every setting under every scorer."""

    def __init__(self, block, region, real, boxes, threshold, detections,
                 objective=None, jobs=1):
        self.block, self.region = block, region
        self.real = Evaluator(real, objective)
        self.jobs = max(1, int(jobs))
        self.inBlock = [d.inRegion(region) for d in detections]
        held = boxes.inRegion(region)
        self.pseudo = Evaluator(BoxCrowns(held.aboveScore(threshold),
                                          real.crowns().crs), objective)
        self.zones = {c: held.aboveScore(c).boxesOrPoints()
                      for c in ZONE_LEVELS}

    def _map(self, function):
        """function(i) for every setting, jobs at a time; same results."""
        global _BLOCK
        _BLOCK = self
        try:
            indices = range(len(self.inBlock))
            if self.jobs == 1:
                return [function(i) for i in indices]
            with multiprocessing.get_context("fork").Pool(self.jobs) as pool:
                return pool.map(function, indices, chunksize=max(
                    1, len(self.inBlock) // (self.jobs * 8)))
        finally:
            _BLOCK = None

    def realScores(self):
        if not hasattr(self, "_real"):
            self._real = self._map(_scoreReal)
        return self._real

    def scorers(self):
        """{scorer: score per setting}, none of them using the crowns."""
        out = {"pseudoF1": self._map(_scorePseudo)}
        for c, boxes in self.zones.items():
            out["zones@%.1f" % c] = [exactlyOneShare(d, boxes)
                                     for d in self.inBlock]
        return out

    def tune(self):
        """
        Per scorer: the chosen setting's real result and the rank agreement;
        None for a block without crowns (it has nothing to score).
        """
        real = self.realScores()
        if not real or not real[0]["crowns"]:
            return None
        realScore = np.array([self.real.objectiveOf(r) for r in real])
        best = int(np.argmax(realScore))
        rows = {"real": dict(real[best], spearman=1.0, chosen=best)}
        for name, values in self.scorers().items():
            values = np.array(values)
            chosen = int(np.argmax(values))
            rho = (spearmanr(values, realScore).correlation
                   if np.ptp(values) > 0 and np.ptp(realScore) > 0
                   else float("nan"))
            rows[name] = dict(real[chosen], spearman=float(rho),
                              chosen=chosen)
        return rows


class PseudoTuning(object):
    """
    data: the site (its crowns score); detections: every setting's
    Detections over the whole site, in grid order; boxRun: the box detector's
    RunSet.
    """

    def __init__(self, data, settings, detections, boxRun, objective=None,
                 jobs=1):
        self.data = data
        self.settings = settings
        self.detections = detections
        self.boxRun = boxRun
        self.objective = objective
        self.jobs = jobs
        self.reference = AnnotatedCrowns(data.annotations)

    def run(self):
        """({block: {scorer: row}}, {scorer: pooled})"""
        blocks = {b.name: b.geometry for b in self.data.blocks}
        perBlock = {}
        for name, region in sorted(sharedBlocks(blocks,
                                                (self.boxRun,)).items()):
            rows = BlockScores(name, region, self.reference,
                               self.boxRun.fold(name),
                               self.boxRun.threshold(name), self.detections,
                               self.objective, self.jobs).tune()
            if rows is not None:
                for row in rows.values():
                    row["settings"] = self.settings[row["chosen"]]
                perBlock[name] = rows
        names = list(next(iter(perBlock.values())))
        return perBlock, {n: pool(perBlock, n, self.objective) for n in names}

    def asRun(self, perBlock, pooled, scorer):
        """
        The scorer's chosen setting per block as a run like any other: per
        block the whole-site detections of that setting (as the height
        detectors' runs store them), and the pooled results.
        """
        predictions, folds = {}, []
        for block, rows in sorted(perBlock.items()):
            row = rows[scorer]
            chosen = self.detections[row["chosen"]].subset(
                np.arange(len(self.detections[row["chosen"]])))
            chosen.block, chosen.fileFields = block, {}
            predictions[block] = chosen
            folds.append(dict(row, block=block))
        return RunSet("connected components tuned on the RGB Mask R-CNN's "
                      "held-out boxes (%s)" % scorer, self.data.name, folds,
                      pooled, predictions=predictions)


def pool(perBlock, name, objective=None):
    folds = [rows[name] for rows in perBlock.values()]
    pooled = poolFolds(folds, objective)
    rhos = [f["spearman"] for f in folds if np.isfinite(f["spearman"])]
    pooled["meanSpearman"] = float(np.mean(rhos)) if rhos else float("nan")
    return pooled
