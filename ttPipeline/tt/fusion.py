"""
Combining a box detector (Mask R-CNN, YOLO) with the point detector
(connected components), evaluated under the same cross-validation as each part.

The two fail differently. The box detector sees crown shape and texture but
its score is uncalibrated and its repeats are split crowns; the point detector
knows exactly where the apexes are but not where one crown ends. Each strategy
below lets one correct a specific failure of the other:

    confirmed        a box containing a height-model detection is more likely
                     a tree. Two thresholds, one for boxes a top confirms and
                     one for boxes none does. This is "extra score for boxes
                     with a point inside", in its general form: a flat bonus is
                     the special case of the two thresholds differing by it.
    boxMergedPoints  height-model detections that fall inside the same kept
                     box are fused into the highest of them. The box acts as
                     the merge prior the saddle rule approximates, which
                     targets exactly the repeats the height model makes.
    union            kept boxes plus every detection no kept box covers. The
                     recall extreme: a tree either method found is kept.
    unionSaddleCross union, with the saddle rule on the height model deciding
                     which height detections duplicate a kept RGB detection,
                     in place of "inside its box". Exploratory.
    unionSaddlePool  union with all detections pooled and merged by the saddle
                     rule, which also merges the RGB model's split crowns.
                     Exploratory.
    weakBoxMergedPoints
                     boxMergedPoints with weak boxes also used as merge
                     evidence when they overlap every stronger accepted box by
                     at most a quarter of the smaller box. Exploratory: it was
                     designed after the others' results were seen.

Every parameter, the thresholds included, is chosen per fold on that fold's
validation block and applied unchanged to its test block. A combination tuned
and scored on the same block would look better than it is by exactly the
margin under test.

Inputs are MethodRun objects from comparison.py. The box run must have saved
validation predictions (runs made by this version of the code do).
"""

import itertools

import numpy as np
from rasterio.features import geometry_mask
from rasterio.windows import Window, transform as windowTransform
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from .dl import dlCommon as dc
from .merging import saddleDrop

THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
POINT_SCORE = 0.0  # a top brought in by union never outranks a box in a crown
WEAK_OVERLAP = 0.25  # fixed in advance, not tuned
SADDLE_RADIUS_M = 8.0  # the detector's own merge radius
POINT_WIDTH_M = 1.01  # a "box" this narrow is a point detection, not a box


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


class SaddleSurface(object):
    """
    A height model and the saddle rule on it, for the saddle strategies.

    A detection is placed at its apex: the highest CHM pixel inside its mask
    outline, or inside its box when it has none. A box centre can fall in a
    gap between branches, and a saddle test started there would see a dip
    that is not a valley between trees. Point detections are already at an
    apex and are used as they are.
    """

    def __init__(self, chm, transform, pixelSize, dropM,
                 radiusM=SADDLE_RADIUS_M):
        self.chm = chm
        self.transform = transform
        self.inverse = ~transform
        self.pixelSize = float(pixelSize)
        self.dropM = float(dropM)
        self.radiusPixels = radiusM / self.pixelSize

    @classmethod
    def fromScene(cls, scene, dropM, radiusM=SADDLE_RADIUS_M):
        return cls(scene.chm, scene.transform, scene.pixelSize, dropM,
                   radiusM)

    def pixel(self, x, y):
        column, row = self.inverse @ (x, y)
        rows, columns = self.chm.shape
        return (int(np.clip(row, 0, rows - 1)),
                int(np.clip(column, 0, columns - 1)))

    def apex(self, prediction):
        """(row, column) of the prediction's highest point, cached on it."""
        if "apex" not in prediction:
            prediction["apex"] = self._findApex(prediction)
        return prediction["apex"]

    def _findApex(self, prediction):
        box = prediction["box"]
        if box[2] - box[0] <= POINT_WIDTH_M:
            return self.pixel(prediction["centreX"], prediction["centreY"])
        r0, c0 = self.pixel(box[0], box[3])
        r1, c1 = self.pixel(box[2], box[1])
        r0, r1 = sorted((r0, r1))
        c0, c1 = sorted((c0, c1))
        heights = self.chm[r0:r1 + 1, c0:c1 + 1]
        if heights.size == 0:
            return self.pixel(prediction["centreX"], prediction["centreY"])
        ring = prediction.get("polygon")
        if ring and len(ring) >= 3:
            inside = geometry_mask(
                [Polygon(ring)], out_shape=heights.shape, invert=True,
                transform=self._windowTransform(r0, c0, heights.shape))
            if inside.any():
                heights = np.where(inside, heights, -np.inf)
        row, column = np.unravel_index(int(np.argmax(heights)),
                                       heights.shape)
        return r0 + int(row), c0 + int(column)

    def _windowTransform(self, r0, c0, shape):
        return windowTransform(Window(c0, r0, shape[1], shape[0]),
                               self.transform)

    def sameTree(self, a, b):
        """Within the radius, and no dip deeper than dropM between them."""
        if np.hypot(a[0] - b[0], a[1] - b[1]) > self.radiusPixels:
            return False
        return saddleDrop(self.chm, a[0], a[1], b[0], b[1]) <= self.dropM


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


def boxMergedPoints(boxes, points, threshold):
    """
    Height detections sharing a kept box collapse to the best of them; the
    others are untouched.
    """
    return mergeWithin(boxesAlone(boxes, points, threshold), points)


def weakBoxMergedPoints(boxes, points, threshold):
    """
    boxMergedPoints with weak boxes as extra merge evidence.

    Every box scoring >= threshold is accepted. The weaker boxes, down to the
    saved score floor, are then taken strongest first, and each is accepted
    only if its overlap with every box accepted so far, strong or weak, is at
    most WEAK_OVERLAP of the smaller box's area. A weak box standing mostly
    clear of the confident ones is taken to mark a tree of its own.

    Designed after the other strategies' results were seen; WEAK_OVERLAP was
    fixed in advance rather than tuned, and threshold is the only parameter.
    """
    accepted = boxesAlone(boxes, points, threshold)
    weak = sorted((b for b in boxes if b["score"] < threshold),
                  key=lambda b: -b["score"])
    corners = [b["box"] for b in accepted]
    for box in weak:
        if not corners or overlapOverSmaller(box["box"],
                                             np.array(corners)).max() \
                <= WEAK_OVERLAP:
            accepted.append(box)
            corners.append(box["box"])
    return mergeWithin(accepted, points)


def overlapOverSmaller(box, others):
    """
    Shared area of `box` with each row of `others`, as a share of the smaller
    of the two boxes. A small box mostly inside a large one scores near 1,
    where its IoU would be low.
    """
    width = np.clip(np.minimum(box[2], others[:, 2])
                    - np.maximum(box[0], others[:, 0]), 0, None)
    height = np.clip(np.minimum(box[3], others[:, 3])
                     - np.maximum(box[1], others[:, 1]), 0, None)
    shared = width * height
    own = (box[2] - box[0]) * (box[3] - box[1])
    theirs = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    smaller = np.minimum(own, theirs)
    return np.where(smaller > 0, shared / np.where(smaller > 0, smaller, 1),
                    (shared > 0).astype(float))


def mergeWithin(boxes, points):
    """
    Keep one height detection per box: the best-scoring one. A detection
    inside several boxes belongs to the highest-scoring of them, so each
    belongs to at most one group; detections in no box are kept.
    """
    if not boxes or not points:
        return list(points)
    inside = containment(points, boxes)
    scores = np.array([b["score"] for b in boxes])
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


def unionSaddleCross(boxes, points, threshold, surface):
    """
    Union with the saddle rule deciding which height detections duplicate a
    kept RGB detection, in place of "inside its box". RGB detections are not
    merged with one another. Exploratory.
    """
    kept = boxesAlone(boxes, points, threshold)
    if not kept or not points:
        return kept + [dict(p, score=POINT_SCORE) for p in points]
    boxApex = [surface.apex(b) for b in kept]
    tree = cKDTree(np.array(boxApex, float))
    added = []
    for point in points:
        apex = surface.apex(point)
        near = tree.query_ball_point(apex, surface.radiusPixels)
        if not any(surface.sameTree(apex, boxApex[j]) for j in near):
            added.append(dict(point, score=POINT_SCORE))
    return kept + added


def unionSaddlePool(boxes, points, threshold, surface):
    """
    Union with every detection pooled and merged by the saddle rule: kept RGB
    detections first, by score, then height detections, each absorbing the
    later ones it is the same tree as. Also merges the RGB model's split crowns
    and the height model's repeats. Exploratory.
    """
    kept = sorted(boxesAlone(boxes, points, threshold),
                  key=lambda b: -b["score"])
    pool = kept + [dict(p, score=POINT_SCORE)
                   for p in sorted(points, key=lambda p: -p["score"])]
    if not pool:
        return []
    apexes = [surface.apex(p) for p in pool]
    tree = cKDTree(np.array(apexes, float))
    removed = np.zeros(len(pool), bool)
    for index in range(len(pool)):
        if removed[index]:
            continue
        for other in tree.query_ball_point(apexes[index],
                                           surface.radiusPixels):
            if other > index and not removed[other] and \
                    surface.sameTree(apexes[index], apexes[other]):
                removed[other] = True
    return [p for p, gone in zip(pool, removed) if not gone]


SURFACE_STRATEGIES = {"unionSaddleCross", "unionSaddlePool"}

STRATEGIES = {
    "boxes": (boxesAlone, {"threshold": THRESHOLDS}),
    "points": (pointsAlone, {}),
    "confirmed": (confirmed, {"confirmedAt": THRESHOLDS,
                              "unconfirmedAt": THRESHOLDS + [np.inf]}),
    "boxMergedPoints": (boxMergedPoints, {"threshold": THRESHOLDS}),
    "weakBoxMergedPoints": (weakBoxMergedPoints, {"threshold": THRESHOLDS}),
    "unionSaddleCross": (unionSaddleCross, {"threshold": THRESHOLDS}),
    "unionSaddlePool": (unionSaddlePool, {"threshold": THRESHOLDS}),
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

    `boxRun` is the RGB Mask R-CNN: boxes with scores. `pointRun` is the
    detector working on the height model, either connected components (tops,
    no confidence) or a Mask R-CNN on the CHM, whose boxes enter through their
    centres. Each enters at the operating point its own cross-validation chose
    for the fold, recorded beside its predictions, so the "alone" rows repeat
    each model's own fold results and every other threshold is tuned on the
    validation block.
    """

    def __init__(self, crowns, blocks, boxRun, pointRun, surface=None,
                 strategies=None, verbose=True):
        self.crowns = crowns
        self.blocks = dict(blocks)
        self.boxRun = boxRun
        self.pointRun = pointRun
        self.surface = surface
        available = [n for n in STRATEGIES
                     if surface is not None or n not in SURFACE_STRATEGIES]
        self.strategies = strategies or available
        self.verbose = verbose

    def apply(self, name, boxes, points, setting):
        extra = {"surface": self.surface} if name in SURFACE_STRATEGIES \
            else {}
        return STRATEGIES[name][0](boxes, points, **dict(setting, **extra))

    def _foldInputs(self, block):
        """Boxes and points on this fold's validation and test blocks."""
        validationBlock, boxValidation = self.boxRun.validation(block)
        _, pointValidation = self.pointRun.validation(block)
        boxTest = self.boxRun.record(block)["predictions"]
        pointTest = self.pointRun.record(block)["predictions"]
        floor = self.pointRun.threshold(block)
        pointValidation = [p for p in pointValidation if p["score"] >= floor]
        pointTest = [p for p in pointTest if p["score"] >= floor]

        region, test = self.blocks[validationBlock], self.blocks[block]
        inputs = {
            "validationBlock": validationBlock,
            "boxThreshold": self.boxRun.threshold(block),
            "validation": (dc.predictionsInRegion(boxValidation, region),
                           dc.predictionsInRegion(pointValidation, region)),
            "test": (dc.predictionsInRegion(boxTest, test),
                     dc.predictionsInRegion(pointTest, test)),
        }
        if self.surface is not None:
            # once per fold, before any strategy copies a prediction
            for split in ("validation", "test"):
                for predictions in inputs[split]:
                    for prediction in predictions:
                        self.surface.apex(prediction)
        return inputs

    def _score(self, predictions, block):
        return dc.evaluateDetections(predictions, self.crowns,
                                     self.blocks[block])

    def tune(self, name, inputs):
        """The strategy's setting that scores best on the validation block."""
        parameters = STRATEGIES[name][1]
        if name == "boxes":
            # the box model alone, at the threshold it chose for itself
            return {"threshold": inputs["boxThreshold"]}, None
        boxes, points = inputs["validation"]
        best, bestF1 = {}, -1.0
        for setting in _grid(parameters):
            f1 = self._score(self.apply(name, boxes, points, setting),
                             inputs["validationBlock"])["f1"]
            if f1 > bestF1:
                best, bestF1 = setting, f1
        return best, bestF1

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
