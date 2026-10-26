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

import itertools
import weakref

from affine import Affine
import numpy as np
from rasterio.features import geometry_mask
from scipy.ndimage import label
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
PROMINENCE_RADIUS_M = 10.0
MATCH_IOU = 0.3        # two networks' boxes overlapping this much are one
HALF = 0.5             # agreeClean: both at half their thresholds; not tuned
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

    def prominences(self, detections, radiusM=PROMINENCE_RADIUS_M):
        """Each detection's prominence (prominenceAt), cached on it."""
        key = "prominence%d" % id(self)
        if key not in detections.cache:
            detections.cache[key] = np.array(
                [prominenceAt(self, x, y, radiusM)
                 for x, y in zip(detections.x, detections.y)], float)
        return detections.cache[key]

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
    """
    combine(*members, **setting) over the grid in `parameters`. Two-member
    strategies take (boxes, points); three-member ones (rgb boxes, chm boxes,
    points). Before each fold the experiment sets memberThresholds, the
    operating point each member's own run chose for it.
    """

    name = "strategy"
    members = 2
    parameters = {}
    needsSurface = False        # the saddle rule's height model
    needsHeights = False        # a height model with no minimum height
    infiniteAsNone = False      # write "off" (infinite) settings as null

    def __init__(self, surface=None, heightSurface=None):
        self.surface = surface
        self.heightSurface = heightSurface
        self.memberThresholds = None

    def settings(self):
        """The settings tuned over on the validation block."""
        keys = list(self.parameters)
        return [dict(zip(keys, values)) for values in
                itertools.product(*[self.parameters[k] for k in keys])]

    @property
    def tuned(self):
        return True

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
    """The box model alone, at the threshold its own run chose."""

    name = "boxes"
    parameters = {"threshold": THRESHOLDS}

    @property
    def tuned(self):
        return False

    def settings(self):
        return [{"threshold": self.memberThresholds[0]}]

    def combine(self, boxes, *others, threshold):
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

    def settings(self):
        # the box model's own threshold: only the height filter is tuned
        return [{"threshold": self.memberThresholds[0], "boxMinHeightM": h}
                for h in BOX_MIN_HEIGHTS]

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


# ---------------------------------------------------------------------- #
# three members: the RGB network, the CHM network and CC
# ---------------------------------------------------------------------- #

def iouMatrix(a, b):
    """IoU between every box of two detection sets."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    A = a.boxesOrPoints()[:, None, :]
    B = b.boxesOrPoints()[None, :, :]
    w = np.clip(np.minimum(A[..., 2], B[..., 2])
                - np.maximum(A[..., 0], B[..., 0]), 0, None)
    h = np.clip(np.minimum(A[..., 3], B[..., 3])
                - np.maximum(A[..., 1], B[..., 1]), 0, None)
    inter = w * h
    area = lambda X: (X[..., 2] - X[..., 0]) * (X[..., 3] - X[..., 1])
    return inter / np.maximum(area(A) + area(B) - inter, 1e-12)


def pairBoxes(rgb, chm, minIou=MATCH_IOU):
    """Greedy one-to-one matching by IoU, highest first: [(i, j)] plus the
    unmatched indices of each set."""
    iou = iouMatrix(rgb, chm)
    pairs, usedA, usedB = [], set(), set()
    if iou.size:
        for flat in np.argsort(-iou, axis=None):
            i, j = np.unravel_index(flat, iou.shape)
            if iou[i, j] < minIou:
                break
            if i in usedA or j in usedB:
                continue
            pairs.append((int(i), int(j)))
            usedA.add(i)
            usedB.add(j)
    return (pairs, [i for i in range(len(rgb)) if i not in usedA],
            [j for j in range(len(chm)) if j not in usedB])


def pick(rgb, chm, chosen):
    """Detections taken in order from either set: chosen = [(0|1, index)]."""
    fromRgb = [i for source, i in chosen if source == 0]
    fromChm = [i for source, i in chosen if source == 1]
    both = Detections.concat([rgb.subset(fromRgb), chm.subset(fromChm)])
    order, r, c = [], 0, len(fromRgb)
    for source, _ in chosen:
        if source == 0:
            order.append(r)
            r += 1
        else:
            order.append(c)
            c += 1
    return both.subset(order)


def withinBoxes(tops, boxes, rank):
    """
    Tops sharing a box collapse to the one ranking highest (strictly; the
    first on a tie); a top inside several boxes belongs to the best-scoring.
    Tops in no box are kept.
    """
    inside = containment(tops, boxes)
    owner = np.where(inside.any(axis=1),
                     np.argmax(np.where(inside, boxes.scores[None, :],
                                        -np.inf), axis=1), -1)
    best = {}
    for index, box in enumerate(owner):
        if box >= 0 and (box not in best or rank[index] > rank[best[box]]):
            best[box] = index
    grouped = set(best.values())
    return tops.subset([i for i in range(len(tops))
                        if owner[i] < 0 or i in grouped])


class ThreeWay(FusionStrategy):
    """Three members: (rgb boxes, chm boxes, points), points' prominence
    read from the surface."""

    members = 3
    needsSurface = True
    infiniteAsNone = True

    def prominence(self, points):
        return self.surface.prominences(points)

    def addedTops(self, kept, points):
        """The points the union adds: those inside no kept box, at score 0."""
        covered = containment(points, kept).any(axis=1)
        return points.subset(~covered).withScores(POINT_SCORE)

    def cleanedTops(self, tops, chm, cleanAt):
        """
        Tops sharing a CHM network box (score >= cleanAt) collapse to the most
        prominent; cleanAt infinite leaves them as they are.
        """
        if not np.isfinite(cleanAt) or not len(tops):
            return tops
        boxes = chm.aboveScore(cleanAt)
        if not len(boxes):
            return tops
        return withinBoxes(tops, boxes, self.prominence(tops))


class Triple(ThreeWay):
    """
    1. agreement: an RGB box and a CHM box overlapping by IoU >= MATCH_IOU are
       one detection, kept when either score >= high or both >= low; a box
       with no partner is kept when its score >= high.
    2. CC's vote: a box not kept by 1 is kept when it holds a CC top and its
       score >= vote.
    3. CC's trees: a top inside no kept box is added when its prominence >=
       prominence; among such tops inside one CHM box, only the most
       prominent.
    """

    name = "triple"
    parameters = {"high": [0.5, 0.7, 0.9],
                  "low": [0.2, 0.3, 0.4, 0.5, np.inf],
                  "vote": [0.1, 0.2, 0.3, np.inf],
                  "prominence": [0.5, 1.0, 2.0, 4.0, np.inf]}

    def __init__(self, surface=None, heightSurface=None):
        super().__init__(surface, heightSurface)
        self._proposals = weakref.WeakKeyDictionary()

    def proposals(self, rgb, chm):
        """
        One detection per matched pair (the more confident box) and one per
        unmatched box, with the best and worst score behind each (worst NaN
        for an unmatched one). Built once per pair of inputs.
        """
        cached = self._proposals.get(rgb, {}).get(id(chm))
        if cached is not None:
            return cached
        pairs, loneA, loneB = pairBoxes(rgb, chm)
        chosen, best, worst = [], [], []
        for i, j in pairs:
            a, b = rgb.scores[i], chm.scores[j]
            chosen.append((0, i) if a >= b else (1, j))
            best.append(max(a, b))
            worst.append(min(a, b))
        chosen += [(0, i) for i in loneA] + [(1, j) for j in loneB]
        best += [rgb.scores[i] for i in loneA] + [chm.scores[j] for j in loneB]
        worst += [np.nan] * (len(loneA) + len(loneB))
        result = (pick(rgb, chm, chosen), np.array(best, float),
                  np.array(worst, float))
        self._proposals.setdefault(rgb, {})[id(chm)] = result
        return result

    def combine(self, rgb, chm, cc, high, low, vote, prominence):
        proposed, best, worst = self.proposals(rgb, chm)
        paired = ~np.isnan(worst)
        keep = (best >= high) | (paired & (np.nan_to_num(worst, nan=-np.inf)
                                           >= low))
        if np.isfinite(vote) and len(cc):
            hasTop = containment(cc, proposed).any(axis=0)
            keep |= hasTop & (best >= vote)
        kept = proposed.subset(keep)
        if not (np.isfinite(prominence) and len(cc)):
            return kept
        free = ~containment(cc, kept).any(axis=1)
        loose = cc.subset(free & (self.prominence(cc) >= prominence))
        if len(loose) and len(chm):
            within = containment(loose, chm)
            rank = self.prominence(loose)
            winner = {}
            for i in range(len(loose)):
                hits = np.nonzero(within[i])[0]
                key = ("box", int(hits[0])) if len(hits) else ("own", i)
                if key not in winner or rank[i] > rank[winner[key]]:
                    winner[key] = i
            loose = loose.subset(sorted(winner.values()))
        return Detections.concat([kept, loose])


CLEAN_AT = [np.inf, 0.7, 0.5, 0.3, 0.2, 0.1]   # no cleaning first: ties keep it


class UnionOfThree(ThreeWay):
    """The two-way union of the RGB boxes and the points (the CHM ignored)."""

    name = "union"
    parameters = {"threshold": THRESHOLDS}

    def combine(self, rgb, chm, cc, threshold):
        return Union.combine(self, rgb, cc, threshold)


class UnionClean(ThreeWay):
    """The union, its added tops cleaned by the CHM network's boxes."""

    name = "unionClean"
    parameters = {"threshold": THRESHOLDS, "cleanAt": CLEAN_AT}

    def combine(self, rgb, chm, cc, threshold, cleanAt):
        kept = rgb.aboveScore(threshold)
        return Detections.concat(
            [kept, self.cleanedTops(self.addedTops(kept, cc), chm, cleanAt)])


class AgreeClean(ThreeWay):
    """
    Both networks' boxes, one per overlapping pair, kept when either score
    reaches its network's own operating point or both reach HALF of theirs;
    plus the tops they leave, cleaned by the CHM boxes.
    """

    name = "agreeClean"
    parameters = {"cleanAt": CLEAN_AT}

    def agreed(self, rgb, chm):
        rgbAt, chmAt = self.memberThresholds[0], self.memberThresholds[1]
        pairs, loneRgb, loneChm = pairBoxes(rgb, chm)
        chosen = []
        for i, j in pairs:
            a, b = rgb.scores[i], chm.scores[j]
            if a >= rgbAt or b >= chmAt or (a >= HALF * rgbAt
                                            and b >= HALF * chmAt):
                chosen.append((0, i) if a >= b else (1, j))
        chosen += [(0, i) for i in loneRgb if rgb.scores[i] >= rgbAt]
        chosen += [(1, j) for j in loneChm if chm.scores[j] >= chmAt]
        return pick(rgb, chm, chosen)

    def combine(self, rgb, chm, cc, cleanAt):
        kept = self.agreed(rgb, chm)
        return Detections.concat(
            [kept, self.cleanedTops(self.addedTops(kept, cc), chm, cleanAt)])


def prominenceAt(surface, x, y, radiusM=PROMINENCE_RADIUS_M):
    """
    Height of the surface at (x, y)'s highest nearby pixel above the level at
    which its region joins higher ground within radiusM (binary search); the
    full height if no higher ground is that close.
    """
    chm = surface.chm
    row, col = surface.pixel(x, y)
    r = int(round(radiusM / surface.pixelSize))
    r0, c0 = max(0, row - 2), max(0, col - 2)
    patch = chm[r0:row + 3, c0:col + 3]
    if patch.size == 0:
        return 0.0
    dr, dc_ = np.unravel_index(np.argmax(patch), patch.shape)
    row, col = r0 + dr, c0 + dc_
    w0, v0 = max(0, row - r), max(0, col - r)
    win = chm[w0:row + r + 1, v0:col + r + 1]
    pr, pc, h = row - w0, col - v0, float(chm[row, col])
    if h <= 0:
        return 0.0
    if not (win > h).any():
        return h
    structure = np.ones((3, 3), bool)

    def joined(level):
        lab, _ = label(win >= level, structure=structure)
        return bool((win[lab == lab[pr, pc]] > h).any())

    if joined(h):
        return 0.0
    lowLevel, highLevel = 0.0, h
    for _ in range(12):
        mid = (lowLevel + highLevel) / 2
        lowLevel, highLevel = (mid, highLevel) if joined(mid) \
            else (lowLevel, mid)
    return h - lowLevel


STRATEGIES = [BoxesAlone, PointsAlone, Confirmed, BoxMergedPoints,
              WeakBoxMergedPoints, UnionSaddleCross, UnionSaddlePool, Union]
HEIGHT_STRATEGIES = [BoxesHigh, UnionHigh]
BY_NAME = {s.name: s for s in STRATEGIES + HEIGHT_STRATEGIES}
TRIPLE = [Triple, BoxesAlone]
REDUCED = [BoxesAlone, UnionOfThree, UnionClean, AgreeClean]


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
