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
