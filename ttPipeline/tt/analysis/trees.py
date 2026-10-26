"""
TreeTable — detectors compared crown by crown from their held-out detections.

Pooled F1 says how well each method does; it does not say whether two methods
miss the same trees. That matters for which method to use and for whether
combining them can help, since a combination can only recover trees at least
one of its parts found. Every method is represented by its held-out
detections (RunSet.heldOut): in each block, what the model made without
seeing that block's crowns.
"""

import numpy as np
from shapely.geometry import Polygon, box as shapelyBox

from ..scoring.evaluator import assignToCrowns
from .statistics import nearestDistances, profile, splitBackground

# radius over which a box detector's centre is given a height: the centre of a
# crown box can fall in a gap between branches, and the top it stands for is
# the highest point near it
HEIGHT_RADIUS_M = 0.75


class TreeTable(object):

    def __init__(self, crowns, crownPeak, surface, detections):
        """
        crowns: the scored crowns; crownPeak: each one's highest height;
        surface: the HeightSurface heights are read from; detections:
        {method: held-out Detections}.
        """
        self.crowns = crowns
        self.crownPeak = crownPeak
        self.surface = surface
        self.detections = detections
        self.crownArea = crowns.geometry.area.to_numpy()
        self.nearest = nearestDistances(crowns.geometry.centroid.x.to_numpy(),
                                        crowns.geometry.centroid.y.to_numpy())
        self.results = {name: self._classify(d)
                        for name, d in detections.items()}

    def _classify(self, detections):
        xs, ys, scores = detections.x, detections.y, detections.scores
        kinds, hitCrowns, assigned = assignToCrowns(xs, ys, scores,
                                                    self.crowns)
        primary = {crown: point for point, crown in assigned.items()
                   if kinds[point] == "hit"}
        heights = self.heightsAt(xs, ys)
        return {"xs": xs, "ys": ys, "scores": scores, "kinds": kinds,
                "heights": heights, "found": set(primary), "primary": primary,
                "split": splitBackground(xs, ys, heights, kinds)}

    def heightsAt(self, xs, ys):
        """The highest height within HEIGHT_RADIUS_M of each point."""
        radius = self.surface.metresToPixels(HEIGHT_RADIUS_M)
        heights = np.zeros(len(xs))
        for index, (x, y) in enumerate(zip(xs, ys)):
            column, row = self.surface.toPixel(x, y)
            column, row = int(column), int(row)
            window = self.surface.chm[max(0, row - radius):row + radius + 1,
                                      max(0, column - radius):
                                      column + radius + 1]
            heights[index] = float(window.max()) if window.size else 0.0
        return heights

    # ------------------------------------------------------------------ #

    def scores(self, name):
        result = self.results[name]
        total = max(len(result["kinds"]), 1)
        hits = len(result["found"])
        recall = hits / float(len(self.crowns))
        precision = hits / float(total)
        return {"method": name, "detections": len(result["kinds"]),
                "hits": hits, "recall": recall, "precision": precision,
                "f1": (2 * recall * precision / (recall + precision)
                       if recall + precision else 0.0)}

    def errorBreakdown(self, name):
        """Where the detections that are not hits went, as counts."""
        result = self.results[name]
        split = result["split"]
        return {"detections": len(result["kinds"]),
                "hits": result["kinds"].count("hit"),
                "repeats": result["kinds"].count("repeat"),
                "canopyLevel": len(split["canopy"]),
                "belowCanopy": len(split["low"]),
                "noNeighbour": len(split["unknown"])}

    def foundMask(self, name):
        found = np.zeros(len(self.crowns), bool)
        found[list(self.results[name]["found"])] = True
        return found

    def missedProfile(self, name):
        """How the crowns a method missed differ from the ones it found."""
        found = self.foundMask(name)
        return {"missed": int((~found).sum()),
                "found": profile(found, self.crownArea, self.crownPeak,
                                 self.nearest),
                "missedCrowns": profile(~found, self.crownArea,
                                        self.crownPeak, self.nearest)}

    def agreement(self, first, second):
        """Crowns found by both, by one only, and by neither."""
        a, b = self.results[first]["found"], self.results[second]["found"]
        everything = set(range(len(self.crowns)))
        groups = {"both": a & b, "only " + first: a - b,
                  "only " + second: b - a, "neither": everything - a - b}
        out = {}
        for label, members in groups.items():
            mask = np.zeros(len(self.crowns), bool)
            mask[list(members)] = True
            out[label] = profile(mask, self.crownArea, self.crownPeak,
                                 self.nearest)
        return out

    def crownTable(self):
        """Per crown: size, height, crowding, and which methods found it."""
        table = {"area": self.crownArea, "peak": self.crownPeak,
                 "nearest": self.nearest}
        for name in self.results:
            table["found_" + name] = self.foundMask(name)
        return table

    def boxQuality(self, name, polygons=None):
        """
        How well each hit's shape matches its crown. Box IoU compares the
        detection's box with the crown's bounding box; shape IoU its outline,
        when it has one, with the crown. `polygons` supplies outlines for
        methods without them, such as pseudo-crowns for the point detector.
        """
        result = self.results[name]
        detections = self.detections[name]
        boxes = detections.boxesOrPoints()
        boxIou, shapeIou = [], []
        for crown, point in result["primary"].items():
            geometry = self.crowns.geometry.iloc[crown]
            if polygons is None:
                outline = _outline(detections.polygons[point]
                                   if detections.polygons is not None
                                   else None)
                predictedBox = shapelyBox(*boxes[point])
            else:
                outline = polygons[point]
                predictedBox = (shapelyBox(*outline.bounds)
                                if outline is not None else None)
            if predictedBox is not None:
                boxIou.append(_iou(predictedBox, shapelyBox(*geometry.bounds)))
            if outline is not None:
                shapeIou.append(_iou(outline, geometry))
        return {"boxIou": np.array(boxIou), "shapeIou": np.array(shapeIou)}


def _outline(ring):
    if not ring or len(ring) < 3:
        return None
    polygon = Polygon(ring)
    return polygon if polygon.is_valid else polygon.buffer(0)


def _iou(first, second):
    union = first.union(second).area
    return first.intersection(second).area / union if union > 0 else 0.0
