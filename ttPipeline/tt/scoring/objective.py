"""
Objectives: the one number tuning maximises. Every choice made away from the
held-out data (a setting per fold, a network's threshold, a combination's
settings) maximises one of these.
"""

import os

RECALL_WEIGHT = 0.6


def f1Of(recall, precision):
    return (2 * recall * precision / (recall + precision)
            if recall + precision else 0.0)


def weightedOf(recall, precision, recallWeight=RECALL_WEIGHT):
    return recallWeight * recall + (1.0 - recallWeight) * precision


class Objective(object):
    name = "objective"

    def value(self, scores):
        raise NotImplementedError

    def __repr__(self):
        return type(self).__name__ + "()"

    @staticmethod
    def named(name):
        if name == "f1":
            return F1()
        if name == "weighted":
            return Weighted()
        raise ValueError("objective must be f1 or weighted, not %r" % name)

    @staticmethod
    def fromEnvironment():
        """TT_OBJECTIVE (f1 by default), so a whole run uses one objective."""
        return Objective.named(os.environ.get("TT_OBJECTIVE", "f1"))


class F1(Objective):
    name = "f1"

    def value(self, scores):
        return scores["f1"]


class Weighted(Objective):
    """recallWeight x recall + (1 - recallWeight) x precision."""
    name = "weighted"

    def __init__(self, recallWeight=RECALL_WEIGHT):
        self.recallWeight = float(recallWeight)

    def value(self, scores):
        if "weighted" in scores and self.recallWeight == RECALL_WEIGHT:
            return scores["weighted"]
        return weightedOf(scores["recall"], scores["precision"],
                          self.recallWeight)
