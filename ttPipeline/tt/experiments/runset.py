"""
RunSet — the result of an experiment: per-fold scores, the pooled score, and
each fold's detections. Saved as results.json plus predictions_<block>.json,
the files every reader of runs already uses.
"""

import glob
import json
import os

import numpy as np

from ..data.detections import Detections
from ..scoring.objective import Objective, f1Of, weightedOf

COUNTS = ("crowns", "predictions", "hits", "repeats", "falsePositives")


def poolFolds(folds, objective=None):
    """
    Pool the folds. Counts are summed before the ratios are formed, so a block
    with few trees does not weigh the same as a block with many.
    """
    objective = objective or Objective.fromEnvironment()
    total = {key: sum(f[key] for f in folds) for key in COUNTS}
    recall = total["hits"] / float(total["crowns"]) if total["crowns"] else 0.0
    precision = (total["hits"] / float(total["predictions"])
                 if total["predictions"] else 0.0)
    perFoldF1 = [f["f1"] for f in folds if f["crowns"] > 0]
    total.update({"recall": recall, "precision": precision,
                  "f1": f1Of(recall, precision),
                  "weighted": weightedOf(recall, precision),
                  "objective": objective.name, "folds": len(folds),
                  "perFoldF1Mean": float(np.mean(perFoldF1))
                  if perFoldF1 else 0.0,
                  "perFoldF1Std": float(np.std(perFoldF1))
                  if perFoldF1 else 0.0})
    return total


class RunSet(object):

    def __init__(self, method, dataset, folds, pooled, source=None,
                 settings=None, predictions=None, extra=None):
        self.method = method
        self.dataset = dataset
        self.source = source
        self.settings = settings or {}     # how the run was asked for
        self.folds = folds
        self.pooled = pooled
        self.predictions = dict(predictions or {})   # block -> Detections
        self.extra = dict(extra or {})     # more top-level results.json keys

    def __repr__(self):
        return "RunSet(%s on %s: F1 %.3f over %d folds)" % (
            self.method, self.dataset, self.pooled.get("f1", 0.0),
            len(self.folds))

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        for block, detections in self.predictions.items():
            detections.save(os.path.join(directory,
                                         "predictions_%s.json" % block))
        path = os.path.join(directory, "results.json")
        with open(path, "w") as handle:
            content = {"method": self.method, "dataset": self.dataset,
                       "source": self.source}
            content.update(self.extra)
            content.update({"settings": self.settings, "folds": self.folds,
                            "pooled": self.pooled})
            json.dump(content, handle, indent=2, default=float)
        return path

    @classmethod
    def load(cls, directory, withPredictions=True):
        with open(os.path.join(directory, "results.json")) as handle:
            content = json.load(handle)
        predictions = {}
        if withPredictions:
            for path in sorted(glob.glob(os.path.join(directory,
                                                      "predictions_*.json"))):
                detections = Detections.load(path)
                block = detections.block or os.path.basename(path)[12:-5]
                predictions[block] = detections
        known = ("method", "dataset", "source", "settings", "folds", "pooled")
        return cls(content.get("method"), content.get("dataset"),
                   content.get("folds", []), content.get("pooled", {}),
                   content.get("source"), content.get("settings"),
                   predictions, {k: v for k, v in content.items()
                                 if k not in known})

    # ------------------------------------------------------------------ #
    # folds, by block name
    # ------------------------------------------------------------------ #

    def foldName(self, block):
        """
        This run's name for the fold of `block`. Datasets prefix their blocks
        with their own name (rgb_b01, lidar_b01) over identical geometry, so a
        fold is matched by its _bNN suffix when the exact name is not found.
        """
        if block in self.predictions:
            return block
        suffix = block[block.rindex("_b"):] if "_b" in block else block
        matches = [b for b in self.predictions if b.endswith(suffix)]
        if len(matches) != 1:
            raise KeyError("no fold %s in this run" % block)
        return matches[0]

    def has(self, block):
        try:
            self.foldName(block)
            return True
        except KeyError:
            return False

    def fold(self, block):
        """Everything the fold saved (above the score floor)."""
        return self.predictions[self.foldName(block)]

    def threshold(self, block):
        """The operating point the fold chose, if it chose one."""
        fields = self.fold(block).fileFields or {}
        return fields.get("threshold", -np.inf)

    def validation(self, block, blocks):
        """
        (validation block, detections there) for one fold, the block named as
        `blocks` name it. A network records both. A height detector's run
        saves the whole area per fold and records no validation block, since
        it tunes on every training block; for it the block is chosen by the
        networks' rule, so the two can be paired fold for fold.
        """
        detections = self.fold(block)
        fields = detections.fileFields or {}
        if "validationPredictions" in fields:
            return (matchBlock(fields["validationBlock"], blocks),
                    Detections.fromPredictions(fields["validationPredictions"],
                                               crs=detections.crs))
        return nextBlock(block, blocks), detections

    def heldOut(self, blocks):
        """
        Every fold's detections inside its own block at its own operating
        point, stitched into one set, with the block each came from.
        """
        parts, names = [], []
        for name in sorted(self.predictions):
            block = matchBlock(name, blocks)
            inBlock = self.predictions[name].inRegion(blocks[block])
            kept = inBlock.aboveScore(self.threshold(name))
            parts.append(kept)
            names += [block] * len(kept)
        stitched = Detections.concat(parts)
        stitched.cache["block"] = np.array(names, dtype=object)
        return stitched


def matchBlock(name, blocks):
    """The block called `name`, or the one sharing its _bNN suffix."""
    if name in blocks:
        return name
    suffix = name[name.rindex("_b"):]
    matches = [b for b in blocks if b.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError("no block matching %s" % name)
    return matches[0]


def nextBlock(block, blocks):
    """The networks' validation-block rule (Patches.chooseValidationBlock)."""
    others = sorted(name for name in blocks if name != block)
    later = [name for name in others if name > block]
    return later[0] if later else others[0]


def sharedBlocks(blocks, runs, verbose=True):
    """
    The blocks every run has a fold for: a network skips a block it cannot
    train or test on, so a comparison or combination is made on the blocks
    they share, and the ones left out are named.
    """
    blocks = dict(blocks)
    kept = {b: g for b, g in blocks.items() if all(r.has(b) for r in runs)}
    if verbose and len(kept) < len(blocks):
        print("[blocks] left out, not in every run: %s"
              % ", ".join(sorted(set(blocks) - set(kept))))
    return kept
