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

import itertools
import json
import os

import numpy as np

from ..detectors.fusion import BY_NAME, HEIGHT_STRATEGIES, STRATEGIES
from ..scoring.evaluator import Evaluator
from .runset import poolFolds, sharedBlocks


def expand(parameters):
    keys = list(parameters)
    return [dict(zip(keys, values))
            for values in itertools.product(*[parameters[k] for k in keys])]


class FusionCrossValidation(object):

    def __init__(self, reference, blocks, boxRun, pointRun, surface=None,
                 heightSurface=None, strategies=None, objective=None,
                 verbose=True):
        """
        reference: what is scored against (a scoring.Reference); blocks:
        {name: geometry}; boxRun, pointRun: RunSets; surface: the saddle
        rule's SaddleSurface; heightSurface: a SaddleSurface over the height
        model with no minimum height, for the height filter.
        """
        self.evaluator = Evaluator(reference, objective)
        self.allBlocks = dict(blocks)
        self.boxRun, self.pointRun = boxRun, pointRun
        self.blocks = sharedBlocks(blocks, (boxRun, pointRun), verbose)
        self.surface, self.heightSurface = surface, heightSurface
        classes = [s for s in STRATEGIES
                   if surface is not None or not s.needsSurface]
        if heightSurface is not None:
            classes += HEIGHT_STRATEGIES
        if strategies:
            classes = [BY_NAME[name] for name in strategies]
        self.strategies = {c.name: c(surface, heightSurface) for c in classes}
        self.verbose = verbose
        self.outputs = {}

    def _foldInputs(self, block):
        """Boxes and points on this fold's validation and test blocks."""
        validationBlock, boxValidation = self.boxRun.validation(
            block, self.allBlocks)
        _, pointValidation = self.pointRun.validation(block, self.allBlocks)
        floor = self.pointRun.threshold(block)
        pointValidation = pointValidation.aboveScore(floor)
        pointTest = self.pointRun.fold(block).aboveScore(floor)
        boxTest = self.boxRun.fold(block)
        # geometry from every block: a block a model skipped as a test fold
        # can still be another fold's validation block
        region, test = self.allBlocks[validationBlock], self.allBlocks[block]
        inputs = {"validationBlock": validationBlock,
                  "boxThreshold": self.boxRun.threshold(block),
                  "validation": (boxValidation.inRegion(region),
                                 pointValidation.inRegion(region)),
                  "test": (boxTest.inRegion(test), pointTest.inRegion(test))}
        if self.surface is not None:
            # once per fold, before any strategy takes a subset
            for split in ("validation", "test"):
                for detections in inputs[split]:
                    self.surface.apexes(detections)
        return inputs

    def _score(self, detections, block):
        return self.evaluator.score(detections, self.allBlocks[block])

    def tune(self, name, inputs):
        """The strategy's setting that scores best on the validation block."""
        strategy = self.strategies[name]
        parameters = strategy.parameters
        if name == "boxes":
            # the box model alone, at the threshold it chose for itself
            return {"threshold": inputs["boxThreshold"]}, None
        if name == "boxesHigh":
            # the same threshold: only the height filter is tuned
            parameters = dict(parameters, threshold=[inputs["boxThreshold"]])
        boxes, points = inputs["validation"]
        best, bestScore = {}, -1.0
        for setting in expand(parameters):
            score = self.evaluator.objectiveOf(self._score(
                strategy.combine(boxes, points, **setting),
                inputs["validationBlock"]))
            if score > bestScore:
                best, bestScore = setting, score
        return best, bestScore

    def runFold(self, block):
        """Every strategy on one fold: its score, setting and output."""
        inputs = self._foldInputs(block)
        boxes, points = inputs["test"]
        results = {}
        for name, strategy in self.strategies.items():
            setting, validationScore = self.tune(name, inputs)
            output = strategy.combine(boxes, points, **setting)
            result = self._score(output, block)
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
