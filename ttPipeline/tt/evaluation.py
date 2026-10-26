"""
CrownEvaluator — scoring detections against hand-drawn crowns.

This lived inside the detector, which meant every analysis script had to build
a detector to reach it, and the detector was being asked to judge itself. It is
a separate concern and now a separate class.

The rule: a detection is reduced to its point, a crown counts as found if any
point falls inside it, the highest-scoring point in a crown is the hit and any
others are repeats, and a point in no crown is a false positive. Detections
inside several overlapping crowns are assigned to the smallest, the most
specific claim.
"""

import numpy as np
from shapely.geometry import Point

from .analysis.statistics import splitBackground
from .scoring.evaluator import assignToCrowns  # noqa: F401
from .tops import Tops


class Classification(object):
    """Per-detection verdicts, and which crowns were reached."""

    def __init__(self, kinds, hitCrowns, assigned):
        self.kinds = kinds
        self.hitCrowns = hitCrowns
        self.assigned = assigned

    def indicesOf(self, kind):
        return [i for i, k in enumerate(self.kinds) if k == kind]

    def count(self, kind):
        return sum(1 for k in self.kinds if k == kind)

    def __repr__(self):
        return ("Classification(hit=%d, repeat=%d, background=%d)"
                % (self.count("hit"), self.count("repeat"),
                   self.count("background")))


class CrownEvaluator(object):

    def __init__(self, scene):
        self.scene = scene
        if scene.crowns is None:
            raise ValueError("this scene has no crowns to evaluate against")
        self._areas = scene.crowns.geometry.area.to_numpy()

    # ------------------------------------------------------------------ #

    def classify(self, tops, crowns=None, scores=None):
        """
        Label every detection hit / repeat / background.

        `scores` decides which detection is primary within a crown; height is
        the natural stand-in when a detector has no confidence of its own.
        """
        crowns = self.scene.crowns if crowns is None else crowns
        if len(tops) == 0:
            return Classification([], set(), {})
        world = tops.world(self.scene)
        scores = tops.heights if scores is None else np.asarray(scores)
        kinds, hitCrowns, assigned = assignToCrowns(world[:, 0], world[:, 1],
                                                    scores, crowns)
        return Classification(kinds, hitCrowns, assigned)

    def score(self, tops, classification=None, crowns=None):
        """Recall, precision and F1, plus the counts behind them."""
        crowns = self.scene.crowns if crowns is None else crowns
        classification = classification or self.classify(tops, crowns=crowns)

        total = max(len(tops), 1)
        nCrowns = max(len(crowns), 1)
        hits = classification.count("hit")
        repeats = classification.count("repeat")
        background = classification.count("background")

        recall = len(classification.hitCrowns) / float(nCrowns)
        precision = hits / float(total)
        f1 = (2 * recall * precision / (recall + precision)
              if (recall + precision) else 0.0)

        return {"crowns": len(crowns), "detections": len(tops),
                "hits": hits, "repeats": repeats, "background": background,
                "recall": recall, "precision": precision, "f1": f1,
                "repeatShare": repeats / float(total),
                "backgroundShare": background / float(total)}

    def scoreWithin(self, tops, region, scores=None):
        """
        Score only what falls inside a region — one block of a cross-validation
        fold. Crowns are selected by centroid and detections by position, so a
        tree on a boundary is counted by exactly one fold.
        """

        crowns = self.scene.crowns
        inside = crowns[crowns.geometry.centroid.within(region)]
        inside = inside.reset_index(drop=True)

        keep, keepScores = [], []
        allScores = tops.heights if scores is None else scores
        for index, point in enumerate(tops.points):
            east, north = self.scene.toWorld(point[0], point[1])
            if region.contains(Point(east, north)):
                keep.append(point)
                keepScores.append(allScores[index])

        subset = Tops(keep)
        if not len(subset) or not len(inside):
            return {"crowns": len(inside), "detections": len(subset),
                    "hits": 0, "repeats": 0, "background": len(subset),
                    "recall": 0.0, "precision": 0.0, "f1": 0.0,
                    "repeatShare": 0.0, "backgroundShare": 1.0}

        classification = self.classify(subset, crowns=inside,
                                       scores=np.array(keepScores))
        return self.score(subset, classification, crowns=inside)

    # ------------------------------------------------------------------ #

    def missedCrowns(self, classification):
        """Row indices of crowns nothing landed in."""
        return [i for i in range(len(self.scene.crowns))
                if i not in classification.hitCrowns]

    def splitBackground(self, tops, classification, neighbourhood=12.0,
                        canopyFraction=0.7):
        """
        Separate the false positives that are probably real trees from the ones
        that are not; see splitBackground at module level.
        """
        world = tops.world(self.scene)
        return splitBackground(world[:, 0], world[:, 1], tops.heights,
                               classification.kinds, neighbourhood,
                               canopyFraction)
