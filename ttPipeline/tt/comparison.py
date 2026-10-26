"""
TRANSITIONAL: MethodRun, kept for blockImages and tt.legacy.calibration only.
The maintained equivalents are tt.experiments.RunSet (folds by block,
held-out detections) and tt.analysis.TreeTable (the tree-by-tree comparison).

Tree-by-tree comparison of detectors from their cross-validated predictions.

Pooled F1 says how well each method does; it does not say whether two methods
miss the same trees. That matters for two decisions: which method to use, and
whether combining them can help, since a combination can only recover trees at
least one of its parts found.

Every method is represented by its held-out predictions: for each fold, what
the model trained without that block predicted inside it, at the operating
point chosen on its validation block. Stitched across the eight folds that is
one prediction set covering the whole area in which no prediction was made by
a model that had seen the crowns it lands on.

Run directories are those written by `python -m tt.dl concomp|maskrcnn|yolo`,
each holding one predictions_<block>.json per fold.
"""

import glob
import os

import numpy as np

from .dl import dlCommon as dc
from .experiments.runset import (matchBlock, nextBlock,  # noqa: F401
                                sharedBlocks)

class MethodRun(object):
    """One method's held-out predictions, stitched across folds."""

    def __init__(self, name, runDir, blocks):
        self.name = name
        self.runDir = runDir
        self.blocks = dict(blocks)
        self.predictions = self._load()

    def __repr__(self):
        return "MethodRun(%s, %d held-out predictions)" % (self.name,
                                                          len(self.predictions))

    def _load(self):
        paths = sorted(glob.glob(os.path.join(self.runDir,
                                              "predictions_*.json")))
        if not paths:
            raise FileNotFoundError("no predictions_<block>.json in %s; runs "
                                    "from before predictions were saved need "
                                    "repeating" % self.runDir)
        held = []
        for path in paths:
            record = dc.loadJson(path)
            region = self.blocks[matchBlock(record["block"], self.blocks)]
            threshold = record.get("threshold", -np.inf)
            inBlock = dc.predictionsInRegion(record["predictions"], region)
            for prediction in inBlock:
                if prediction["score"] >= threshold:
                    held.append(dict(prediction, block=matchBlock(
                        record["block"], self.blocks)))
        return held

    def validation(self, block):
        """
        (validation block, predictions there) for one fold.

        A learned model records both. A connected-component run saves the
        whole area per fold and records no validation block, since it tunes on
        all seven training blocks; for it the block is chosen by the rule the
        learned models use, so the two can be paired fold for fold.
        """
        record = self.record(block)
        if "validationPredictions" in record:
            # the run names blocks after its own dataset (rgb_b01); translate
            # to the names the caller's blocks use (lidar_b01)
            return (matchBlock(record["validationBlock"], self.blocks),
                    record["validationPredictions"])
        return nextBlock(block, self.blocks), record["predictions"]

    def threshold(self, block):
        """The operating point this run chose for the fold, if it chose one."""
        return self.record(block).get("threshold", -np.inf)

    def has(self, block):
        """Whether this run has a fold for the block (exact or by _bNN)."""
        if os.path.exists(os.path.join(self.runDir,
                                       "predictions_%s.json" % block)):
            return True
        suffix = block[block.rindex("_b"):] if "_b" in block else block
        return len(glob.glob(os.path.join(self.runDir, "predictions_*%s.json"
                                          % suffix))) == 1

    def record(self, block):
        """
        One fold's saved predictions. Datasets prefix their blocks with their
        own name (lidar_b00, p1_b00) over identical geometry, so a block is
        matched by its _bNN suffix when the exact name is not found.
        """
        path = os.path.join(self.runDir, "predictions_%s.json" % block)
        if not os.path.exists(path):
            suffix = block[block.rindex("_b"):]
            matches = glob.glob(os.path.join(self.runDir,
                                             "predictions_*%s.json" % suffix))
            if len(matches) != 1:
                raise FileNotFoundError("no fold %s in %s" % (block,
                                                              self.runDir))
            path = matches[0]
        return dc.loadJson(path)

    @property
    def isBoxDetector(self):
        return any(p["box"][2] - p["box"][0] > 1.01 for p in self.predictions)


def loadRuns(specs, blocks):
    """{name: MethodRun} from "name=runDir" strings, skipping missing ones."""
    runs = {}
    for spec in specs:
        name, runDir = spec.split("=", 1)
        try:
            runs[name] = MethodRun(name, runDir, blocks)
        except FileNotFoundError as error:
            print("[compare] %s skipped: %s" % (name, error))
    return runs
