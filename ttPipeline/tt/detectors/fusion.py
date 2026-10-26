"""
Combining a box detector (Mask R-CNN, YOLO, DeepForest) with a point detector
(connected components, or a network on the CHM entering through its centres).

The two fail differently. The box detector sees crown shape and texture but
its score is uncalibrated and its repeats are split crowns; the point detector
knows where the apexes are but not where one crown ends. Each strategy lets
one correct a specific failure of the other:

    BoxesAlone, PointsAlone   each part on its own, for reference
    Confirmed           two thresholds: one for boxes a point confirms (a
                        point inside), one for boxes none does
    BoxMergedPoints     points inside the same kept box fuse into the best of
                        them: the box is the merge prior the saddle rule
                        approximates
    WeakBoxMergedPoints BoxMergedPoints with weak boxes as extra merge
                        evidence when they overlap every stronger accepted box
                        by at most WEAK_OVERLAP of the smaller (exploratory,
                        designed after the others' results were seen)
    Union               kept boxes plus every point no kept box covers
    UnionSaddleCross    Union, with the saddle rule on the height model
                        deciding which points duplicate a kept box
    UnionSaddlePool     everything pooled and merged by the saddle rule
    BoxesHigh, UnionHigh
                        boxes over low vegetation removed first (the apex
                        under the box must reach boxMinHeightM)

A strategy is a combine function plus the grid its parameters are tuned over;
the tuning, per fold on the validation block, is the experiment's job
(tt.experiments.fusion). CombinedDetector runs member detectors and combines
their output with one strategy at fixed settings.
"""

from affine import Affine
import numpy as np
from rasterio.features import geometry_mask
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from ..data.detections import Detections
from .base import Detector
from .merging import saddleDrop

THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BOX_MIN_HEIGHTS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
POINT_SCORE = 0.0      # a point brought in by a union never outranks a box
WEAK_OVERLAP = 0.25    # fixed in advance, not tuned
SADDLE_RADIUS_M = 8.0  # the detector's own merge radius
POINT_WIDTH_M = 1.01   # a "box" this narrow is a point detection, not a box


# ---------------------------------------------------------------------- #
# the height model the saddle strategies and the height filter read
# ---------------------------------------------------------------------- #

class SaddleSurface(object):
    """
    A height model and the saddle rule on it. A detection is placed at its
    apex: the highest pixel inside its mask outline, or inside its box when it
    has none (a box centre can fall in a gap between branches). Points are
    already at an apex. Apexes are cached on the Detections, so a strategy
    tried at many settings finds each once.
    """

    def __init__(self, chm, transform, pixelSize, dropM=0.5,
                 radiusM=SADDLE_RADIUS_M):
        self.chm = chm
        self.transform = transform
        self.inverse = ~transform
        self.pixelSize = float(pixelSize)
        self.dropM = float(dropM)
        self.radiusPixels = radiusM / self.pixelSize

    @classmethod
    def fromSurface(cls, surface, dropM=0.5, radiusM=SADDLE_RADIUS_M):
        """From a HeightSurface (or anything with chm, transform, pixelSize)."""
        return cls(surface.chm, surface.transform, surface.pixelSize, dropM,
                   radiusM)

    fromScene = fromSurface

    def pixel(self, x, y):
        column, row = self.inverse @ (x, y)
        rows, columns = self.chm.shape
        return (int(np.clip(row, 0, rows - 1)),
                int(np.clip(column, 0, columns - 1)))

    def apexes(self, detections):
        """(row, column) of each detection's highest point, cached on it."""
        key = "apex%d" % id(self)
        if key not in detections.cache:
            boxes = detections.boxesOrPoints()
            detections.cache[key] = np.array(
                [self._findApex(detections, i, boxes[i])
                 for i in range(len(detections))], int).reshape(-1, 2)
        return detections.cache[key]

    def heights(self, detections):
        """The height at each detection's apex."""
        apexes = self.apexes(detections)
        return self.chm[apexes[:, 0], apexes[:, 1]].astype(float) \
            if len(apexes) else np.zeros(0)

    def _findApex(self, detections, index, box):
        x, y = detections.x[index], detections.y[index]
        if box[2] - box[0] <= POINT_WIDTH_M:
            return self.pixel(x, y)
        r0, c0 = self.pixel(box[0], box[3])
        r1, c1 = self.pixel(box[2], box[1])
        r0, r1 = sorted((r0, r1))
        c0, c1 = sorted((c0, c1))
        heights = self.chm[r0:r1 + 1, c0:c1 + 1]
        if heights.size == 0:
            return self.pixel(x, y)
        ring = (detections.polygons[index]
                if detections.polygons is not None else None)
        if ring and len(ring) >= 3:
            inside = geometry_mask(
                [Polygon(ring)], out_shape=heights.shape, invert=True,
                transform=self.transform @ Affine.translation(c0, r0))
            if inside.any():
                heights = np.where(inside, heights, -np.inf)
        row, column = np.unravel_index(int(np.argmax(heights)), heights.shape)
        return r0 + int(row), c0 + int(column)

    def sameTree(self, a, b):
        """Within the radius, and no dip deeper than dropM between them."""
        if np.hypot(a[0] - b[0], a[1] - b[1]) > self.radiusPixels:
            return False
        return saddleDrop(self.chm, a[0], a[1], b[0], b[1]) <= self.dropM


# ---------------------------------------------------------------------- #
# geometry
# ---------------------------------------------------------------------- #

def containment(points, boxes):
    """Boolean (points x boxes): is each point's centre inside each box."""
    if not len(points) or not len(boxes):
        return np.zeros((len(points), len(boxes)), bool)
    b = boxes.boxesOrPoints()
    px, py = points.x[:, None], points.y[:, None]
    return ((px >= b[None, :, 0]) & (px <= b[None, :, 2])
            & (py >= b[None, :, 1]) & (py <= b[None, :, 3]))


def overlapOverSmaller(box, others):
    """
    Shared area of `box` with each row of `others`, as a share of the smaller
    of the two. A small box mostly inside a large one scores near 1, where its
    IoU would be low.
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
    One point per box, the best-scoring; a point inside several boxes belongs
    to the highest-scoring of them. Points in no box are kept.
    """
    if not len(boxes) or not len(points):
        return points
    inside = containment(points, boxes)
    owner = np.where(inside.any(axis=1),
                     np.argmax(np.where(inside, boxes.scores[None, :],
                                        -np.inf), axis=1), -1)
    best = {}
    for index, box in enumerate(owner):
        if box < 0:
            continue
        if box not in best or points.scores[index] > points.scores[best[box]]:
            best[box] = index
    grouped = set(best.values())
    return points.subset([i for i in range(len(points))
                          if owner[i] < 0 or i in grouped])


# ---------------------------------------------------------------------- #
# strategies
# ---------------------------------------------------------------------- #

class FusionStrategy(object):
    """combine(boxes, points, **setting) over the grid in `parameters`."""

    name = "strategy"
    parameters = {}
    needsSurface = False        # the saddle rule's height model
    needsHeights = False        # a height model with no minimum height

    def __init__(self, surface=None, heightSurface=None):
        self.surface = surface
        self.heightSurface = heightSurface

    def __repr__(self):
        return type(self).__name__ + "()"

    def combine(self, boxes, points, **setting):
        raise NotImplementedError

    def tallEnough(self, boxes, minHeightM):
        """The boxes whose apex stands at least minHeightM above ground."""
        if minHeightM <= 0 or self.heightSurface is None:
            return boxes
        return boxes.subset(self.heightSurface.heights(boxes) >= minHeightM)


class BoxesAlone(FusionStrategy):
    name = "boxes"
    parameters = {"threshold": THRESHOLDS}

    def combine(self, boxes, points, threshold):
        return boxes.aboveScore(threshold)


class PointsAlone(FusionStrategy):
    name = "points"

    def combine(self, boxes, points):
        return points


class Confirmed(FusionStrategy):
    name = "confirmed"
    parameters = {"confirmedAt": THRESHOLDS,
                  "unconfirmedAt": THRESHOLDS + [np.inf]}

    def combine(self, boxes, points, confirmedAt, unconfirmedAt):
        hasTop = containment(points, boxes).any(axis=0)
        return boxes.subset(boxes.scores >= np.where(hasTop, confirmedAt,
                                                     unconfirmedAt))


class BoxMergedPoints(FusionStrategy):
    name = "boxMergedPoints"
    parameters = {"threshold": THRESHOLDS}

    def combine(self, boxes, points, threshold):
        return mergeWithin(boxes.aboveScore(threshold), points)


class WeakBoxMergedPoints(FusionStrategy):
    name = "weakBoxMergedPoints"
    parameters = {"threshold": THRESHOLDS}

    def combine(self, boxes, points, threshold):
        accepted = list(np.flatnonzero(boxes.scores >= threshold))
        weak = np.flatnonzero(boxes.scores < threshold)
        weak = weak[np.argsort(-boxes.scores[weak], kind="stable")]
        corners = boxes.boxesOrPoints()
        kept = [corners[i] for i in accepted]
        for index in weak:
            if not kept or overlapOverSmaller(corners[index], np.array(
                    kept)).max() <= WEAK_OVERLAP:
                accepted.append(index)
                kept.append(corners[index])
        return mergeWithin(boxes.subset(accepted), points)


class Union(FusionStrategy):
    name = "union"
    parameters = {"threshold": THRESHOLDS}

    def combine(self, boxes, points, threshold):
        kept = boxes.aboveScore(threshold)
        covered = containment(points, kept).any(axis=1)
        return Detections.concat(
            [kept, points.subset(~covered).withScores(POINT_SCORE)])


class UnionSaddleCross(FusionStrategy):
    """Union, with the saddle rule deciding which points duplicate a box."""

    name = "unionSaddleCross"
    parameters = {"threshold": THRESHOLDS}
    needsSurface = True

    def combine(self, boxes, points, threshold):
        kept = boxes.aboveScore(threshold)
        if not len(kept) or not len(points):
            return Detections.concat([kept, points.withScores(POINT_SCORE)])
        surface = self.surface
        boxApex = surface.apexes(kept)
        tree = cKDTree(boxApex.astype(float))
        added = []
        for index, apex in enumerate(surface.apexes(points)):
            near = tree.query_ball_point(apex, surface.radiusPixels)
            if not any(surface.sameTree(apex, boxApex[j]) for j in near):
                added.append(index)
        return Detections.concat(
            [kept, points.subset(added).withScores(POINT_SCORE)])


class UnionSaddlePool(FusionStrategy):
    """
    Everything pooled, kept boxes first by score then points by score, each
    absorbing the later ones it is the same tree as. Also merges the box
    model's split crowns and the point model's repeats.
    """

    name = "unionSaddlePool"
    parameters = {"threshold": THRESHOLDS}
    needsSurface = True

    def combine(self, boxes, points, threshold):
        kept = boxes.aboveScore(threshold)
        kept = kept.subset(np.argsort(-kept.scores, kind="stable"))
        ordered = points.subset(np.argsort(-points.scores, kind="stable"))
        pool = Detections.concat([kept, ordered.withScores(POINT_SCORE)])
        if not len(pool):
            return pool
        apexes = self.surface.apexes(pool)
        tree = cKDTree(apexes.astype(float))
        removed = np.zeros(len(pool), bool)
        for index in range(len(pool)):
            if removed[index]:
                continue
            for other in tree.query_ball_point(apexes[index],
                                               self.surface.radiusPixels):
                if other > index and not removed[other] and \
                        self.surface.sameTree(apexes[index], apexes[other]):
                    removed[other] = True
        return pool.subset(~removed)


class BoxesHigh(FusionStrategy):
    """The box model alone, its boxes over low vegetation removed."""

    name = "boxesHigh"
    parameters = {"threshold": THRESHOLDS, "boxMinHeightM": BOX_MIN_HEIGHTS}
    needsHeights = True

    def combine(self, boxes, points, threshold, boxMinHeightM):
        return self.tallEnough(boxes, boxMinHeightM).aboveScore(threshold)


class UnionHigh(Union):
    """The union, its boxes over low vegetation removed first."""

    name = "unionHigh"
    parameters = {"threshold": THRESHOLDS, "boxMinHeightM": BOX_MIN_HEIGHTS}
    needsHeights = True

    def combine(self, boxes, points, threshold, boxMinHeightM):
        return Union.combine(self, self.tallEnough(boxes, boxMinHeightM),
                             points, threshold)


STRATEGIES = [BoxesAlone, PointsAlone, Confirmed, BoxMergedPoints,
              WeakBoxMergedPoints, UnionSaddleCross, UnionSaddlePool, Union]
HEIGHT_STRATEGIES = [BoxesHigh, UnionHigh]
BY_NAME = {s.name: s for s in STRATEGIES + HEIGHT_STRATEGIES}


class CombinedDetector(Detector):
    """
    Member detectors run on the site and their output combined by one
    strategy at fixed settings: members[0] gives the boxes, members[1] the
    points.
    """

    method = "combined"
    geometry = "boxes"

    def __init__(self, members, strategy, setting=None, postprocessors=None):
        self.defaults = dict(setting or {})
        super().__init__(postprocessors=postprocessors, **dict(setting or {}))
        self.members = list(members)
        self.strategy = strategy

    def __repr__(self):
        return "CombinedDetector(%r, %s)" % (self.strategy, self.members)

    def withParams(self, **params):
        return CombinedDetector(self.members, self.strategy,
                                dict(self.params, **params),
                                self.postprocessors)

    def _detect(self, data):
        boxes, points = (member.detect(data) for member in self.members[:2])
        return self.strategy.combine(boxes, points, **self.params)
