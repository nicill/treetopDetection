"""
Experiments for detectors that are trained rather than tuned over a grid.

TrainedCrossValidation — leave one block out, as for the height detectors, but
each fold trains a fresh network: on the training tiles a buffer away from the
held-out block, minus one training block kept to choose the operating point
(the score threshold) on. The held-out block is predicted and scored at that
threshold. Every prediction down to the score floor is saved, on the test block
and on the validation block, so methods can be compared tree by tree at any
operating point and combinations tuned without touching the test block.

TrainedTransfer — one network trained on every block of another site and
applied unchanged to every block here: the learning curve's point with no
annotation of this site.

FixedRun — a detector with nothing to train or tune (DeepForest as
downloaded), run once and scored on every block at its fixed threshold.
"""

import os

from ..scoring.evaluator import Evaluator
from ..scoring.reference import AnnotatedCrowns
from .crossValidation import Experiment
from .runset import RunSet, poolFolds


class TrainedCrossValidation(Experiment):

    def __init__(self, data, detector, evaluator=None, bufferM=10.0,
                 folds=None, fixedThreshold=False, trainFraction=1.0,
                 subsetSeed=1, drawFrom=None, saveModels=False, output=None,
                 verbose=True):
        super().__init__(data, detector, grid=None, evaluator=evaluator,
                         verbose=verbose, output=output)
        self.patchSet = data.patchSet
        self.bufferM = float(bufferM)
        self.wanted = set(folds) if folds else None
        self.fixedThreshold = fixedThreshold
        self.trainFraction = float(trainFraction)
        self.subsetSeed = int(subsetSeed)
        self.drawFrom = drawFrom            # a PatchSet to draw the subset on
        self.saveModels = saveModels

    # ------------------------------------------------------------------ #
    # splits
    # ------------------------------------------------------------------ #

    def splitFold(self, name):
        """(fit patches, validation patches, validation block, test patches)"""
        train, test = self.patchSet.foldSplit(name, self.bufferM)
        if not train or not test:
            return None
        train = self.drawSubset(train, name)
        validationBlock = self.patchSet.validationBlock(train, name)
        validation = [p for p in train if p.block == validationBlock]
        fit = [p for p in train if p.block != validationBlock]
        if not validation or not fit:
            fit, validation = train, train[:1]
        return fit, validation, validationBlock, test

    def drawSubset(self, train, name):
        """
        The fold's annotated subset (all of it by default). With drawFrom the
        draw is made on that patch set's tiles and these keep the drawn tiles
        they have, so two patch sets of one site whose tile lists differ by a
        few tiles still get the same annotation.
        """
        if self.trainFraction >= 1:
            return train
        if self.drawFrom is None:
            return self.patchSet.draw(train, self.trainFraction,
                                      self.subsetSeed, name)
        other, _ = self.drawFrom.foldSplit(name, self.bufferM)
        drawn = {(p.block, p.record["r0"], p.record["c0"]) for p in
                 self.drawFrom.draw(other, self.trainFraction,
                                    self.subsetSeed, name)}
        return [p for p in train
                if (p.block, p.record["r0"], p.record["c0"]) in drawn]

    def validationRegion(self, validationBlock, validation):
        """The validation block, or only what its drawn tiles cover."""
        geometry = self.data.block(validationBlock).geometry
        if self.trainFraction >= 1:
            return geometry
        return geometry.intersection(self.patchSet.region(validation))

    # ------------------------------------------------------------------ #

    def chooseThreshold(self, patches, region, evaluator=None,
                        patchSet=None):
        """The operating point, and the validation predictions it came from."""
        _, validation = self.detector.predict(patchSet or self.patchSet,
                                              patches, self.data)
        if self.fixedThreshold:
            return self.detector.params["threshold"], validation
        threshold, _ = (evaluator or self.evaluator).chooseThreshold(
            validation, region)
        return threshold, validation

    def runFold(self, block):
        split = self.splitFold(block.name)
        if split is None:
            print("  fold %s: skipped, empty split" % block.name)
            return None, None
        fit, validation, validationBlock, test = split
        print("  fold %s: fit %d tiles, validate on %s (%d), test %d"
              % (block.name, len(fit), validationBlock, len(validation),
                 len(test)))
        detector = self.detector
        detector.reset(self.patchSet)
        detector.fit(self.patchSet, fit, validation)
        threshold, validationDetections = self.chooseThreshold(
            validation, self.validationRegion(validationBlock, validation))
        raw, merged = detector.predict(self.patchSet, test, self.data)
        result = self.evaluator.score(merged.aboveScore(threshold),
                                      block.geometry)
        printScores(result)
        result.update(block=block.name, validationBlock=validationBlock,
                      threshold=threshold, rawPredictions=raw,
                      fitTiles=len(fit), validationTiles=len(validation),
                      trainFraction=self.trainFraction,
                      subsetSeed=self.subsetSeed)
        merged.block = block.name
        merged.fileFields = {"threshold": threshold,
                             "scoreFloor": detector.params["scoreFloor"],
                             "validationBlock": validationBlock,
                             "validationPredictions": validationDetections}
        if self.saveModels and self.output:
            detector.saveWeights(os.path.join(
                self.output, "%s_%s.pt" % (detector.method, block.name)))
        detector.release()
        return result, merged

    def run(self):
        print("[%s] %s, %d tiles, %d blocks"
              % (self.detector.method, self.data.name, len(self.patchSet),
                 len(self.data.blocks)))
        folds, predictions = [], {}
        for block in self.data.blocks:
            if self.wanted and block.name not in self.wanted:
                continue
            result, merged = self.runFold(block)
            if result is not None:
                folds.append(result)
                predictions[block.name] = merged
        return RunSet(self.detector.method, self.data.name, folds,
                      poolFolds(folds, self.evaluator.objective),
                      source=self.data.source, predictions=predictions,
                      extra={"channelCount": self.patchSet.channelCount})


class TrainedTransfer(TrainedCrossValidation):
    """Trained on every block of `source` (a TDData), applied to every block here."""

    def __init__(self, data, detector, source, **keywords):
        super().__init__(data, detector, **keywords)
        self.source = source

    def trainOnSource(self):
        source = self.source.patchSet
        if source.channelCount != self.patchSet.channelCount:
            raise ValueError("the source has %d channels, this site %d"
                             % (source.channelCount,
                                self.patchSet.channelCount))
        patches = [p for p in source if p.block]
        validationBlock = source.validationBlock(patches, "")
        fit = [p for p in patches if p.block != validationBlock]
        validation = [p for p in patches if p.block == validationBlock]
        print("  source %s: fit %d tiles, validate on %s (%d)"
              % (self.source.name, len(fit), validationBlock,
                 len(validation)))
        self.detector.reset(source)
        self.detector.fit(source, fit, validation)
        evaluator = Evaluator(AnnotatedCrowns(self.source.annotations),
                              self.evaluator.objective)
        threshold, _ = self.chooseThreshold(
            validation, self.source.block(validationBlock).geometry,
            evaluator, source)
        return threshold

    def run(self):
        threshold = self.trainOnSource()
        print("[%s] trained on %s, threshold %.2f, applied to %s"
              % (self.detector.method, self.source.name, threshold,
                 self.data.name))
        folds, predictions = [], {}
        for block in self.data.blocks:
            patches = self.patchSet.inBlocks([block.name])
            if not patches:
                continue
            raw, merged = self.detector.predict(self.patchSet, patches,
                                                self.data)
            result = self.evaluator.score(merged.aboveScore(threshold),
                                          block.geometry)
            printScores(result)
            result.update(block=block.name, threshold=threshold,
                          rawPredictions=raw,
                          transferredFrom=self.source.name)
            merged.block = block.name
            merged.fileFields = {"threshold": threshold,
                                 "scoreFloor":
                                     self.detector.params["scoreFloor"],
                                 "transferredFrom": self.source.name}
            folds.append(result)
            predictions[block.name] = merged
        self.detector.release()
        return RunSet(self.detector.method, self.data.name, folds,
                      poolFolds(folds, self.evaluator.objective),
                      source=self.data.source, predictions=predictions,
                      extra={"channelCount": self.patchSet.channelCount})


class FixedRun(Experiment):
    """Detect once over the site; score every block at the fixed threshold."""

    def run(self):
        everything = self.detector.detect(self.data)    # down to the floor
        threshold = self.detector.params["threshold"]
        detections = everything.aboveScore(threshold)
        folds, predictions = [], {}
        for block in self.data.blocks:
            result = self.evaluator.score(detections, block.geometry)
            folds.append(dict(result, block=block.name, threshold=threshold))
            copy = everything.subset(range(len(everything)))
            copy.block = block.name
            copy.fileFields = {"threshold": threshold,
                               "scoreFloor":
                                   self.detector.params["scoreFloor"]}
            predictions[block.name] = copy
        return RunSet(self.detector.method, self.data.name, folds,
                      poolFolds(folds, self.evaluator.objective),
                      source=self.data.source, predictions=predictions,
                      extra={"threshold": threshold})


def printScores(result):
    print("    crowns %d, predictions %d -> hits %d, repeats %d, fp %d  "
          "| R %.3f P %.3f F1 %.3f"
          % (result["crowns"], result["predictions"], result["hits"],
             result["repeats"], result["falsePositives"], result["recall"],
             result["precision"], result["f1"]))

