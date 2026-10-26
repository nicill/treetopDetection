"""
FusionCrossValidation — every combination strategy under leave-one-block-out,
on two saved runs: a box run (an RGB network) and a point run (connected
components, or a network on the CHM entering through its centres).

Each part enters at the operating point its own cross-validation chose for the
fold, so the "alone" rows repeat each model's own fold results. Every other
parameter, thresholds included, is chosen per fold on that fold's validation
block and applied unchanged to its test block: a combination tuned and scored
on the same block would look better than it is by exactly the margin under
test.
"""

import json
import os

import numpy as np

from ..detectors.fusion import (BY_NAME, HEIGHT_STRATEGIES, REDUCED,
                                STRATEGIES, TRIPLE)
from ..scoring.evaluator import Evaluator
from .runset import poolFolds, sharedBlocks


class FusionCrossValidation(object):
    """
    Strategies combining the members' saved runs, tuned per fold on the
    validation block. runs: the members' RunSets in the order the strategies
    take them; flooredMembers: the members kept at their own operating point
    before combining (the point detector: CC's runs record none, so for CC it
    is everything; a CHM network entering as points is cut at its own).
    """

    def __init__(self, reference, blocks, runs, strategies, flooredMembers=(-1,),
                 objective=None, verbose=True):
        self.evaluator = Evaluator(reference, objective)
        self.allBlocks = dict(blocks)
        self.runs = list(runs)
        self.blocks = sharedBlocks(blocks, self.runs, verbose)
        self.strategies = {s.name: s for s in strategies}
        self.floored = {i % len(self.runs) for i in flooredMembers}
        self.verbose = verbose
        self.outputs = {}

    @classmethod
    def pair(cls, reference, blocks, boxRun, pointRun, surface=None,
             heightSurface=None, strategies=None, objective=None,
             verbose=True):
        """
        A box run with a point run: every two-member strategy the given
        surfaces allow (the saddle ones need `surface`, the height filter
        `heightSurface`), or the ones named.
        """
        classes = [s for s in STRATEGIES
                   if surface is not None or not s.needsSurface]
        if heightSurface is not None:
            classes += HEIGHT_STRATEGIES
        if strategies:
            classes = [BY_NAME[name] for name in strategies]
        return cls(reference, blocks, (boxRun, pointRun),
                   [c(surface, heightSurface) for c in classes],
                   objective=objective, verbose=verbose)

    @classmethod
    def threeWay(cls, reference, blocks, rgbRun, chmRun, ccRun, surface,
                 reduced=False, objective=None, verbose=True):
        """The RGB network, the CHM network and CC (Triple, or the reduced
        versions: the union, unionClean and agreeClean)."""
        classes = REDUCED if reduced else TRIPLE
        return cls(reference, blocks, (rgbRun, chmRun, ccRun),
                   [c(surface) for c in classes], objective=objective,
                   verbose=verbose)

    def _foldInputs(self, block):
        """Every member on this fold's validation and test blocks."""
        validationBlock = None
        validation, test = [], []
        for index, run in enumerate(self.runs):
            named, detections = run.validation(block, self.allBlocks)
            validationBlock = validationBlock or named
            held = run.fold(block)
            if index in self.floored:
                floor = run.threshold(block)
                detections = detections.aboveScore(floor)
                held = held.aboveScore(floor)
            validation.append(detections)
            test.append(held)
        # geometry from every block: a block a model skipped as a test fold
        # can still be another fold's validation block
        region, area = self.allBlocks[validationBlock], self.allBlocks[block]
        inputs = {"validationBlock": validationBlock,
                  "thresholds": [run.threshold(block) for run in self.runs],
                  "validation": [d.inRegion(region) for d in validation],
                  "test": [d.inRegion(area) for d in test]}
        surfaces = {s.surface for s in self.strategies.values()
                    if s.needsSurface and s.members == 2} - {None}
        for surface in surfaces:
            # once per fold, before any strategy takes a subset
            for split in ("validation", "test"):
                for detections in inputs[split]:
                    surface.apexes(detections)
        return inputs

    def _score(self, detections, block):
        return self.evaluator.score(detections, self.allBlocks[block])

    def tune(self, strategy, inputs):
        """The strategy's setting that scores best on the validation block."""
        settings = strategy.settings()
        if not strategy.tuned:
            return settings[0], None
        best, bestScore = {}, -1.0
        for setting in settings:
            score = self.evaluator.objectiveOf(self._score(
                strategy.combine(*inputs["validation"], **setting),
                inputs["validationBlock"]))
            if score > bestScore:
                best, bestScore = setting, score
        return best, bestScore

    def runFold(self, block):
        """Every strategy on one fold: its score, setting and output."""
        inputs = self._foldInputs(block)
        results = {}
        for name, strategy in self.strategies.items():
            strategy.memberThresholds = inputs["thresholds"]
            setting, validationScore = self.tune(strategy, inputs)
            output = strategy.combine(*inputs["test"], **setting)
            result = self._score(output, block)
            if strategy.infiniteAsNone:
                setting = {k: (None if not np.isfinite(v) else v)
                           for k, v in setting.items()}
            result.update(block=block, setting=setting,
                          validationF1=validationScore, output=output)
            results[name] = result
        return results

    def run(self):
        """
        {strategy: {"folds", "pooled"}}; each strategy's held-out output per
        block is kept in self.outputs, apart, so the summary stays small.
        """
        perStrategy = {name: [] for name in self.strategies}
        self.outputs = {name: {} for name in self.strategies}
        for block in sorted(self.blocks):
            for name, result in self.runFold(block).items():
                self.outputs[name][block] = result.pop("output")
                perStrategy[name].append(result)
        summary = {name: {"folds": folds,
                          "pooled": poolFolds(folds,
                                              self.evaluator.objective)}
                   for name, folds in perStrategy.items()}
        if self.verbose:
            printSummary(summary)
        return summary

    def saveOutputs(self, directory):
        """
        One folder per strategy, one predictions_<block>.json per fold, each
        detection as where, how sure, and its box.
        """
        for strategy, perBlock in self.outputs.items():
            folder = os.path.join(directory, strategy)
            os.makedirs(folder, exist_ok=True)
            for block, detections in perBlock.items():
                with open(os.path.join(folder, "predictions_%s.json" % block),
                          "w") as handle:
                    json.dump({"block": block,
                               "predictions": detections.slim()},
                              handle, indent=2, default=float)


def printSummary(summary, baseline="boxes"):
    reference = ([f["f1"] for f in summary[baseline]["folds"]]
                 if baseline in summary else None)
    print("%-16s %7s %7s %7s %8s %7s" % ("strategy", "recall", "prec", "F1",
                                         "vs boxes", "wins"))
    print("-" * 58)
    for name, entry in sorted(summary.items(),
                              key=lambda item: -item[1]["pooled"]["f1"]):
        pooled = entry["pooled"]
        line = "%-16s %7.3f %7.3f %7.3f" % (name, pooled["recall"],
                                            pooled["precision"], pooled["f1"])
        if reference is not None and name != baseline:
            difference = (np.array([f["f1"] for f in entry["folds"]])
                          - np.array(reference))
            line += " %+8.3f %4d/%d" % (difference.mean(),
                                        int((difference > 0).sum()),
                                        len(difference))
        print(line)
