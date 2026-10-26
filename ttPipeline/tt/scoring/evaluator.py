"""
Evaluator — scoring detections against a reference.

The rule: a detection is reduced to its centre, a crown counts as found if any
centre falls inside it, the highest-scoring centre in a crown is the hit and
the rest are repeats, and a centre in no crown is a false positive. A centre
inside several overlapping crowns goes to the smallest, the most specific
claim. Every score in the package goes through assignToCrowns.

Within a region only crowns whose centroid is in it, and only detections whose
centre is in it, count, so a tree on a block boundary is scored by exactly one
fold.
"""

import geopandas as gpd
import numpy as np

from .objective import Objective, f1Of, weightedOf


def assignToCrowns(xs, ys, scores, crowns):
    """
    Returns (kinds, hitCrowns, assigned): a verdict per point ("hit",
    "repeat", "background"), the set of crown rows reached, and point index ->
    crown row.
    """
    count = len(xs)
    if count == 0 or len(crowns) == 0:
        return ["background"] * count, set(), {}

    points = gpd.GeoDataFrame({"point": np.arange(count)},
                              geometry=gpd.points_from_xy(xs, ys),
                              crs=crowns.crs)
    joined = gpd.sjoin(points, crowns[["geometry"]], how="inner",
                       predicate="within")

    areas = crowns.geometry.area.to_numpy()
    assigned = {}
    for point, crown in zip(joined["point"].to_numpy(),
                            joined["index_right"].to_numpy()):
        point, crown = int(point), int(crown)
        current = assigned.get(point)
        if current is None or areas[crown] < areas[current]:
            assigned[point] = crown

    best = {}
    for point, crown in assigned.items():
        if crown not in best or scores[point] > best[crown][0]:
            best[crown] = (scores[point], point)
    primary = {point for _, point in best.values()}

    kinds = ["background"] * count
    for point in assigned:
        kinds[point] = "hit" if point in primary else "repeat"
    return kinds, set(assigned.values()), assigned


class Matching(object):
    """
    The verdicts of one scoring: a kind per detection, which crowns were
    reached, and detection index -> crown row. Indices refer to the
    `detections` and `crowns` it holds (those in the region scored).
    """

    def __init__(self, detections, crowns, kinds, hitCrowns, assigned):
        self.detections = detections
        self.crowns = crowns
        self.kinds = kinds
        self.hitCrowns = hitCrowns
        self.assigned = assigned

    def __repr__(self):
        return "Matching(hit=%d, repeat=%d, background=%d, missed=%d)" % (
            self.count("hit"), self.count("repeat"), self.count("background"),
            len(self.missed))

    def count(self, kind):
        return sum(1 for k in self.kinds if k == kind)

    def indicesOf(self, kind):
        return [i for i, k in enumerate(self.kinds) if k == kind]

    @property
    def missed(self):
        """Crown rows nothing landed in."""
        return [i for i in range(len(self.crowns)) if i not in self.hitCrowns]

    def scores(self):
        """Counts, recall, precision, F1 and the weighted mean."""
        crowns, found = len(self.crowns), len(self.detections)
        result = {"crowns": crowns, "predictions": found, "hits": 0,
                  "repeats": 0, "falsePositives": 0,
                  "recall": 0.0, "precision": 0.0, "f1": 0.0, "weighted": 0.0}
        if not crowns or not found:
            return result       # counts as runs have always recorded them
        hits = self.count("hit")
        result.update(hits=hits, repeats=self.count("repeat"),
                      falsePositives=self.count("background"))
        recall, precision = hits / float(crowns), hits / float(found)
        result.update(recall=recall, precision=precision,
                      f1=f1Of(recall, precision),
                      weighted=weightedOf(recall, precision))
        return result


class Evaluator(object):

    def __init__(self, reference, objective=None):
        self.reference = reference
        self.objective = objective or Objective.fromEnvironment()

    def __repr__(self):
        return "Evaluator(%s, %r)" % (type(self.reference).__name__,
                                      self.objective)

    def match(self, detections, region=None):
        crowns = self.reference.crownsIn(region)
        kept = detections if region is None else detections.inRegion(region)
        if not len(kept) or not len(crowns):
            return Matching(kept, crowns, ["background"] * len(kept), set(), {})
        kinds, hitCrowns, assigned = assignToCrowns(kept.x, kept.y,
                                                    kept.scores, crowns)
        return Matching(kept, crowns, kinds, hitCrowns, assigned)

    def score(self, detections, region=None):
        return self.match(detections, region).scores()

    def objectiveOf(self, scores):
        return self.objective.value(scores)
