"""
Detections — what any detector returns, in world coordinates.

Points (CC, local maxima, watershed) and boxes or masks (the networks) share
one type: every detection has a centre, a score and optionally a height, a box
and a mask outline. Scoring, fusion and rendering read centres and boxes and
never need to know which detector made them.

Files keep the format runs already use (predictions_<block>.json holding a
list of prediction dicts), so existing results load unchanged.
"""

import json
import os

import numpy as np
from shapely import contains_xy

GEOMETRIES = ("points", "boxes", "masks")
POINT_BOX_M = 0.5          # half-width of the box written round a point
_KNOWN = ("box", "centreX", "centreY", "score", "polygon", "height")


class Detections(object):

    def __init__(self, x, y, scores=None, heights=None, boxes=None,
                 polygons=None, geometry="points", crs=None, detector=None,
                 params=None, block=None, extra=None):
        if geometry not in GEOMETRIES:
            raise ValueError("geometry must be one of %s" % (GEOMETRIES,))
        self.x = np.asarray(x, np.float64).reshape(-1)
        self.y = np.asarray(y, np.float64).reshape(-1)
        count = len(self.x)
        self.scores = (np.ones(count) if scores is None
                       else np.asarray(scores, np.float64).reshape(-1))
        self.heights = (None if heights is None
                        else np.asarray(heights, np.float64).reshape(-1))
        self.boxes = (None if boxes is None
                      else np.asarray(boxes, np.float64).reshape(-1, 4))
        self.polygons = list(polygons) if polygons is not None else None
        self.extra = list(extra) if extra is not None else None
        self.geometry = geometry
        self.crs = crs
        self.detector = detector
        self.params = dict(params or {})
        self.block = block

    def __len__(self):
        return len(self.x)

    def __repr__(self):
        return "Detections(%d %s, %s)" % (len(self), self.geometry,
                                          self.detector or "?")

    # ------------------------------------------------------------------ #

    @classmethod
    def fromPixels(cls, points, surface, **keywords):
        """(column, row, height) points on a HeightSurface; height is the score."""
        points = np.asarray(points, np.float64).reshape(-1, 3)
        world = np.array([surface.toWorld(c, r) for c, r, _ in points]
                         ).reshape(-1, 2)
        return cls(world[:, 0], world[:, 1], scores=points[:, 2],
                   heights=points[:, 2], geometry="points", crs=surface.crs,
                   **keywords)

    def subset(self, keep):
        """The detections selected by a boolean mask or index array."""
        keep = np.asarray(keep)
        if keep.dtype == bool:
            keep = np.flatnonzero(keep)
        pick = lambda items: None if items is None else [items[i] for i in keep]
        return Detections(
            self.x[keep], self.y[keep], self.scores[keep],
            None if self.heights is None else self.heights[keep],
            None if self.boxes is None else self.boxes[keep],
            pick(self.polygons), self.geometry, self.crs, self.detector,
            self.params, self.block, pick(self.extra))

    def inRegion(self, region):
        """Those whose centre lies in `region`."""
        if not len(self):
            return self.subset(np.zeros(0, int))
        return self.subset(contains_xy(region, self.x, self.y))

    def aboveScore(self, threshold):
        return self.subset(self.scores >= threshold)

    def boxesOrPoints(self, halfWidthM=POINT_BOX_M):
        """Boxes, with a small square round each point when there are none."""
        if self.boxes is not None:
            return self.boxes
        return np.column_stack([self.x - halfWidthM, self.y - halfWidthM,
                                self.x + halfWidthM, self.y + halfWidthM])

    # ------------------------------------------------------------------ #
    # the prediction-dict form every existing run file uses
    # ------------------------------------------------------------------ #

    def toPredictions(self):
        boxes = self.boxesOrPoints()
        out = []
        for i in range(len(self)):
            prediction = dict(self.extra[i]) if self.extra else {}
            prediction.update({"box": [float(v) for v in boxes[i]],
                               "centreX": float(self.x[i]),
                               "centreY": float(self.y[i]),
                               "score": float(self.scores[i])})
            if self.polygons is not None and self.polygons[i] is not None:
                prediction["polygon"] = self.polygons[i]
            if self.heights is not None and self.geometry != "points":
                prediction["height"] = float(self.heights[i])
            out.append(prediction)
        return out

    @classmethod
    def fromPredictions(cls, predictions, geometry=None, **keywords):
        """
        Prediction dicts back to Detections. Without `geometry`, a run whose
        boxes are all the small point squares reads as points, else as boxes
        (masks when any prediction carries an outline).
        """
        x = [p["centreX"] for p in predictions]
        y = [p["centreY"] for p in predictions]
        scores = [p.get("score", 1.0) for p in predictions]
        boxes = (np.array([p["box"] for p in predictions],
                          np.float64).reshape(-1, 4)
                 if all("box" in p for p in predictions) else None)
        hasPolygons = any(p.get("polygon") for p in predictions)
        if geometry is None:
            geometry = ("masks" if hasPolygons else
                        "points" if boxes is None or _arePointBoxes(boxes, x, y)
                        else "boxes")
        heights = None
        if geometry == "points":
            heights = scores
        elif all("height" in p for p in predictions) and predictions:
            heights = [p["height"] for p in predictions]
        extra = [{k: v for k, v in p.items() if k not in _KNOWN}
                 for p in predictions]
        return cls(x, y, scores, heights,
                   None if geometry == "points" else boxes,
                   [p.get("polygon") for p in predictions]
                   if hasPolygons else None,
                   geometry, extra=extra if any(extra) else None, **keywords)

    def save(self, path, settings=None, region="area"):
        """predictions_<block>.json, as runs have always written it."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        content = {"block": self.block,
                   "settings": self.params if settings is None else settings,
                   "region": region, "predictions": self.toPredictions()}
        with open(path, "w") as handle:
            json.dump(content, handle, indent=2, default=float)
        return path

    @classmethod
    def load(cls, path, geometry=None):
        with open(path) as handle:
            content = json.load(handle)
        return cls.fromPredictions(content["predictions"], geometry,
                                   params=content.get("settings"),
                                   block=content.get("block"))


def _arePointBoxes(boxes, x, y):
    if not len(boxes):
        return True
    half = np.column_stack([np.asarray(x) - boxes[:, 0],
                            boxes[:, 2] - np.asarray(x),
                            np.asarray(y) - boxes[:, 1],
                            boxes[:, 3] - np.asarray(y)])
    return bool(np.allclose(half, POINT_BOX_M, atol=1e-6))
