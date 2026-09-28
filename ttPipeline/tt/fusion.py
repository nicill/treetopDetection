"""
Combining a box detector (Mask R-CNN, YOLO) with the point detector
(connected components), evaluated under the same cross-validation as each part.

The two fail differently. The box detector sees crown shape and texture but
its score is uncalibrated and its repeats are split crowns; the point detector
knows exactly where the apexes are but not where one crown ends. Each strategy
below lets one correct a specific failure of the other:

    confirmed        a box containing a connected-component top is more likely
                     a tree. Two thresholds, one for boxes a top confirms and
                     one for boxes none does. This is "extra score for boxes
                     with a point inside", in its general form: a flat bonus is
                     the special case of the two thresholds differing by it.
    agreement        confirmed with the unconfirmed threshold at infinity: only
                     boxes a top confirms survive. The precision extreme.
    boxMergedPoints  connected-component tops that fall inside the same kept
                     box are fused into the highest of them. The box acts as
                     the merge prior the saddle rule approximates, which
                     targets exactly the repeats the point detector makes.
    union            kept boxes plus every top that no kept box covers. The
                     recall extreme: a tree either method found is kept.

Every parameter, the thresholds included, is chosen per fold on that fold's
validation block and applied unchanged to its test block. A combination tuned
and scored on the same block would look better than it is by exactly the
margin under test.

Inputs are MethodRun objects from comparison.py. The box run must have saved
validation predictions (runs made by this version of the code do).
"""

import itertools

import numpy as np

from .dl import dlCommon as dc

THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
POINT_SCORE = 0.0  # a top brought in by union never outranks a box in a crown


# ---------------------------------------------------------------------- #
# geometry
# ---------------------------------------------------------------------- #

def containment(points, boxes):
    """Boolean (points x boxes): is each point inside each box."""
    if not points or not boxes:
        return np.zeros((len(points), len(boxes)), bool)
    px = np.array([p["centreX"] for p in points])[:, None]
    py = np.array([p["centreY"] for p in points])[:, None]
    b = np.array([q["box"] for q in boxes])
    return ((px >= b[None, :, 0]) & (px <= b[None, :, 2])
            & (py >= b[None, :, 1]) & (py <= b[None, :, 3]))


def inflate(predictions, halfWidthM):
    """Give point predictions a square box, so they can play the box role."""
    out = []
    for p in predictions:
        x, y = p["centreX"], p["centreY"]
        out.append(dict(p, box=[x - halfWidthM, y - halfWidthM,
                                x + halfWidthM, y + halfWidthM]))
    return out


# ---------------------------------------------------------------------- #
# strategies: (boxes, points, parameters) -> predictions
# ---------------------------------------------------------------------- #

def boxesAlone(boxes, points, threshold):
    return [b for b in boxes if b["score"] >= threshold]


def pointsAlone(boxes, points):
    return list(points)


def confirmed(boxes, points, confirmedAt, unconfirmedAt):
    inside = containment(points, boxes).any(axis=0)
    return [b for b, hasTop in zip(boxes, inside)
            if b["score"] >= (confirmedAt if hasTop else unconfirmedAt)]


def agreement(boxes, points, threshold):
    return confirmed(boxes, points, threshold, np.inf)


def boxMergedPoints(boxes, points, threshold):
    """
    Tops sharing a kept box collapse to the highest; others are untouched.

    A top inside several overlapping boxes is grouped with the highest-scoring
    one, so each top belongs to at most one group.
    """
    kept = boxesAlone(boxes, points, threshold)
    if not kept or not points:
        return list(points)
    inside = containment(points, kept)
    scores = np.array([b["score"] for b in kept])
    owner = np.where(inside.any(axis=1),
                     np.argmax(np.where(inside, scores[None, :], -np.inf),
                               axis=1), -1)
    best = {}
    for index, box in enumerate(owner):
        if box < 0:
            continue
        if box not in best or points[index]["score"] > \
                points[best[box]]["score"]:
            best[box] = index
    grouped = set(best.values())
    return [p for index, p in enumerate(points)
            if owner[index] < 0 or index in grouped]


def union(boxes, points, threshold):
    kept = boxesAlone(boxes, points, threshold)
    covered = containment(points, kept).any(axis=1) if kept else \
        np.zeros(len(points), bool)
    return kept + [dict(p, score=POINT_SCORE)
                   for p, isCovered in zip(points, covered) if not isCovered]


STRATEGIES = {
    "boxes": (boxesAlone, {"threshold": THRESHOLDS}),
    "points": (pointsAlone, {}),
    "confirmed": (confirmed, {"confirmedAt": THRESHOLDS,
                              "unconfirmedAt": THRESHOLDS + [np.inf]}),
    "agreement": (agreement, {"threshold": THRESHOLDS}),
    "boxMergedPoints": (boxMergedPoints, {"threshold": THRESHOLDS}),
    "union": (union, {"threshold": THRESHOLDS}),
}


def _grid(parameters):
    keys = list(parameters)
    return [dict(zip(keys, values))
            for values in itertools.product(*[parameters[k] for k in keys])]


# ---------------------------------------------------------------------- #
# cross-validation
# ---------------------------------------------------------------------- #

class FusionCrossValidation(object):
    """
    Every strategy under leave-one-block-out, tuned on validation blocks.

    `boxRun` supplies boxes with scores and validation predictions. `pointRun`
    supplies the tops; a connected-component run saves its detections over the
    whole area per fold, so its validation-block tops are those of the same
    fold. If `boxRun` is itself a point detector, `inflateM` gives its points a
    box so the geometry still applies, and its thresholds are ignored.
    """

    def __init__(self, crowns, blocks, boxRun, pointRun, inflateM=None,
                 strategies=None, verbose=True):
        # a list of sizes is tuned per fold like any other parameter
        self.inflateSizes = (list(inflateM) if isinstance(inflateM,
                                                          (list, tuple))
                             else ([inflateM] if inflateM else [None]))
        self.crowns = crowns
        self.blocks = dict(blocks)
        self.boxRun = boxRun
        self.pointRun = pointRun
        self.inflateM = inflateM
        self.strategies = strategies or list(STRATEGIES)
        self.verbose = verbose

    def _foldInputs(self, block):
        """Boxes and points on this fold's validation and test blocks."""
        validationBlock, boxValidation = self.boxRun.validation(block)
        _, pointArea = self.pointRun.validation(block)
        boxTest = self.boxRun.record(block)["predictions"]
        region = self.blocks[validationBlock]
        return {
            "validationBlock": validationBlock,
            "validation": (dc.predictionsInRegion(boxValidation, region),
                           dc.predictionsInRegion(pointArea, region)),
            "test": (dc.predictionsInRegion(boxTest, self.blocks[block]),
                     dc.predictionsInRegion(pointArea, self.blocks[block])),
        }

    def _score(self, predictions, block):
        return dc.evaluateDetections(predictions, self.crowns,
                                     self.blocks[block])

    def tune(self, name, inputs):
        """The strategy's setting, box size included, best on validation."""
        function, parameters = STRATEGIES[name]
        if self.inflateM:
            # every parameter is a score threshold, and a point detector's
            # score is a height, not a confidence: keep everything
            parameters = {key: [-np.inf] for key in parameters}
        boxes, points = inputs["validation"]
        best, bestF1 = {}, -1.0
        for size in self.inflateSizes:
            sized = inflate(boxes, size) if size else boxes
            for setting in _grid(parameters):
                f1 = self._score(function(sized, points, **setting),
                                 inputs["validationBlock"])["f1"]
                if f1 > bestF1:
                    best, bestF1 = dict(setting, inflateM=size), f1
        return best, bestF1

    def apply(self, name, boxes, points, setting):
        setting = dict(setting)
        size = setting.pop("inflateM", None)
        return STRATEGIES[name][0](inflate(boxes, size) if size else boxes,
                                   points, **setting)

    def runFold(self, block):
        inputs = self._foldInputs(block)
        boxes, points = inputs["test"]
        results = {}
        for name in self.strategies:
            setting, validationF1 = self.tune(name, inputs)
            result = self._score(self.apply(name, boxes, points, setting),
                                 block)
            result.update(block=block, setting=setting,
                          validationF1=validationF1)
            results[name] = result
        return results

    def run(self):
        perStrategy = {name: [] for name in self.strategies}
        for block in sorted(self.blocks):
            for name, result in self.runFold(block).items():
                perStrategy[name].append(result)
        summary = {name: {"folds": folds, "pooled": dc.averageFolds(folds)}
                   for name, folds in perStrategy.items()}
        if self.verbose:
            printSummary(summary)
        return summary


def printSummary(summary, baseline="boxes"):
    reference = [f["f1"] for f in summary[baseline]["folds"]] \
        if baseline in summary else None
    print("%-16s %7s %7s %7s %8s %7s" % ("strategy", "recall", "prec", "F1",
                                         "vs boxes", "wins"))
    print("-" * 58)
    for name, entry in sorted(summary.items(),
                              key=lambda item: -item[1]["pooled"]["f1"]):
        pooled = entry["pooled"]
        line = "%-16s %7.3f %7.3f %7.3f" % (name, pooled["recall"],
                                            pooled["precision"], pooled["f1"])
        if reference is not None and name != baseline:
            difference = np.array([f["f1"] for f in entry["folds"]]) - \
                np.array(reference)
            line += " %+8.3f %4d/%d" % (difference.mean(),
                                        int((difference > 0).sum()),
                                        len(difference))
        print(line)
