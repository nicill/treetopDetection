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
        # what a run file holds besides the block and the predictions
        # (settings, threshold, validation predictions, ...); None writes
        # the settings and region as the tuned detectors' files always had
        self.fileFields = None
        # per-detection arrays computed by others and carried through subset
        # and concat (e.g. the apex a height surface found for each)
        self.cache = {}

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
        out = Detections(
            self.x[keep], self.y[keep], self.scores[keep],
            None if self.heights is None else self.heights[keep],
            None if self.boxes is None else self.boxes[keep],
            pick(self.polygons), self.geometry, self.crs, self.detector,
            self.params, self.block, pick(self.extra))
        out.cache = {k: v[keep] for k, v in self.cache.items()}
        return out

    def withScores(self, score):
        """A copy with every score set to `score`."""
        out = self.subset(np.arange(len(self)))
        out.scores = np.full(len(self), float(score))
        return out

    @staticmethod
    def concat(parts):
        """
        Several detection sets as one, in order. Points among boxes keep their
        small square as a box; a part without outlines contributes none.
        """
        parts = [p for p in parts if p is not None]
        geometries = {p.geometry for p in parts if len(p)}
        geometry = ("points" if geometries <= {"points"} else
                    "masks" if "masks" in geometries else "boxes")
        join = lambda name: np.concatenate(
            [getattr(p, name) for p in parts]) if parts else np.zeros(0)
        heights = (join("heights") if parts and all(p.heights is not None
                                                     for p in parts) else None)
        boxes = None if geometry == "points" else np.concatenate(
            [p.boxesOrPoints() for p in parts]).reshape(-1, 4)
        polygons = None
        if any(p.polygons is not None for p in parts):
            polygons = [poly for p in parts for poly in
                        (p.polygons if p.polygons is not None
                         else [None] * len(p))]
        extra = None
        if any(p.extra is not None for p in parts):
            extra = [e for p in parts for e in
                     (p.extra if p.extra is not None else [{}] * len(p))]
        out = Detections(join("x"), join("y"), join("scores"), heights, boxes,
                         polygons, geometry,
                         parts[0].crs if parts else None, extra=extra)
        shared = set.intersection(*[set(p.cache) for p in parts]) \
            if parts else set()
        out.cache = {k: np.concatenate([p.cache[k] for p in parts])
                     for k in shared}
        return out

    def slim(self):
        """Prediction dicts with only where, how sure, and the box."""
        boxes = self.boxesOrPoints()
        return [{"centreX": float(self.x[i]), "centreY": float(self.y[i]),
                 "score": float(self.scores[i]),
                 "box": [float(v) for v in boxes[i]]}
                for i in range(len(self))]

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
            if self.polygons is not None:
                # a mask model writes the key even when no outline was traced
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
        hasPolygons = any("polygon" in p for p in predictions)
        if geometry is None:
            geometry = ("masks" if any(p.get("polygon") for p in predictions)
                        else
                        "points" if boxes is None or _arePointBoxes(boxes, x, y)
                        else "boxes")
        heights = None
        if geometry == "points":
            heights = scores
        elif all("height" in p for p in predictions) and predictions:
            heights = [p["height"] for p in predictions]
        extra = [{k: v for k, v in p.items() if k not in _KNOWN}
                 for p in predictions]
        if hasPolygons and geometry == "boxes":
            geometry = "masks"
        return cls(x, y, scores, heights,
                   None if geometry == "points" else boxes,
                   [p.get("polygon") for p in predictions]
                   if hasPolygons else None,
                   geometry, extra=extra if any(extra) else None, **keywords)

    def save(self, path):
        """
        predictions_<block>.json: the block, the file fields (by default the
        settings and region), and the predictions. Detections among the file
        fields are written as prediction lists.
        """
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fields = (self.fileFields if self.fileFields is not None
                  else {"settings": self.params, "region": "area"})
        content = {"block": self.block}
        for key, value in fields.items():
            content[key] = (value.toPredictions()
                            if isinstance(value, Detections) else value)
        content["predictions"] = self.toPredictions()
        with open(path, "w") as handle:
            json.dump(content, handle, indent=2, default=float)
        return path

    @classmethod
    def load(cls, path, geometry=None):
        with open(path) as handle:
            content = json.load(handle)
        detections = cls.fromPredictions(content["predictions"], geometry,
                                         params=content.get("settings"),
                                         block=content.get("block"))
        detections.fileFields = {k: v for k, v in content.items()
                                 if k not in ("block", "predictions")}
        return detections


def _arePointBoxes(boxes, x, y):
    if not len(boxes):
        return True
    half = np.column_stack([np.asarray(x) - boxes[:, 0],
                            boxes[:, 2] - np.asarray(x),
                            np.asarray(y) - boxes[:, 1],
                            boxes[:, 3] - np.asarray(y)])
    return bool(np.allclose(half, POINT_BOX_M, atol=1e-6))
