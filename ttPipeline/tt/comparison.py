"""
Tree-by-tree comparison of detectors from their cross-validated predictions.

Pooled F1 says how well each method does; it does not say whether two methods
miss the same trees. That matters for two decisions: which method to use, and
whether combining them can help, since a combination can only recover trees at
least one of its parts found.

Every method is represented by its held-out predictions: for each fold, what
the model trained without that block predicted inside it, at the operating
point chosen on its validation block. Stitched across the eight folds that is
one prediction set covering the whole area in which no prediction was made by
a model that had seen the crowns it lands on.

Run directories are those written by `python -m tt.dl concomp|maskrcnn|yolo`,
each holding one predictions_<block>.json per fold.
"""

import glob
import os

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, box as shapelyBox

from .evaluation import assignToCrowns, splitBackground
from .dl import dlCommon as dc

# radius over which a box detector's centre is given a height: the centre of a
# crown box can fall in a gap between branches, and the top it stands for is
# the highest point near it
HEIGHT_RADIUS_M = 0.75


class MethodRun(object):
    """One method's held-out predictions, stitched across folds."""

    def __init__(self, name, runDir, blocks):
        self.name = name
        self.runDir = runDir
        self.blocks = dict(blocks)
        self.predictions = self._load()

    def __repr__(self):
        return "MethodRun(%s, %d held-out predictions)" % (self.name,
                                                          len(self.predictions))

    def _load(self):
        paths = sorted(glob.glob(os.path.join(self.runDir,
                                              "predictions_*.json")))
        if not paths:
            raise FileNotFoundError("no predictions_<block>.json in %s; runs "
                                    "from before predictions were saved need "
                                    "repeating" % self.runDir)
        held = []
        for path in paths:
            record = dc.loadJson(path)
            region = self.blocks[matchBlock(record["block"], self.blocks)]
            threshold = record.get("threshold", -np.inf)
            inBlock = dc.predictionsInRegion(record["predictions"], region)
            for prediction in inBlock:
                if prediction["score"] >= threshold:
                    held.append(dict(prediction, block=matchBlock(
                        record["block"], self.blocks)))
        return held

    def validation(self, block):
        """
        (validation block, predictions there) for one fold.

        A learned model records both. A connected-component run saves the
        whole area per fold and records no validation block, since it tunes on
        all seven training blocks; for it the block is chosen by the rule the
        learned models use, so the two can be paired fold for fold.
        """
        record = self.record(block)
        if "validationPredictions" in record:
            # the run names blocks after its own dataset (rgb_b01); translate
            # to the names the caller's blocks use (lidar_b01)
            return (matchBlock(record["validationBlock"], self.blocks),
                    record["validationPredictions"])
        return nextBlock(block, self.blocks), record["predictions"]

    def threshold(self, block):
        """The operating point this run chose for the fold, if it chose one."""
        return self.record(block).get("threshold", -np.inf)

    def has(self, block):
        """Whether this run has a fold for the block (exact or by _bNN)."""
        if os.path.exists(os.path.join(self.runDir,
                                       "predictions_%s.json" % block)):
            return True
        suffix = block[block.rindex("_b"):] if "_b" in block else block
        return len(glob.glob(os.path.join(self.runDir, "predictions_*%s.json"
                                          % suffix))) == 1

    def record(self, block):
        """
        One fold's saved predictions. Datasets prefix their blocks with their
        own name (lidar_b00, p1_b00) over identical geometry, so a block is
        matched by its _bNN suffix when the exact name is not found.
        """
        path = os.path.join(self.runDir, "predictions_%s.json" % block)
        if not os.path.exists(path):
            suffix = block[block.rindex("_b"):]
            matches = glob.glob(os.path.join(self.runDir,
                                             "predictions_*%s.json" % suffix))
            if len(matches) != 1:
                raise FileNotFoundError("no fold %s in %s" % (block,
                                                              self.runDir))
            path = matches[0]
        return dc.loadJson(path)

    @property
    def isBoxDetector(self):
        return any(p["box"][2] - p["box"][0] > 1.01 for p in self.predictions)


def sharedBlocks(blocks, runs, verbose=True):
    """
    The blocks every run has a fold for. A learned model skips a block it
    cannot train or test on while connected components scores every block,
    so a comparison or combination is made on the blocks they share, and the
    ones left out are named.
    """
    blocks = dict(blocks)
    kept = {b: g for b, g in blocks.items() if all(r.has(b) for r in runs)}
    if verbose and len(kept) < len(blocks):
        print("[blocks] left out, not in every run: %s"
              % ", ".join(sorted(set(blocks) - set(kept))))
    return kept


class TreeComparison(object):

    def __init__(self, scene, runs):
        self.scene = scene
        self.runs = runs
        self.crowns = scene.crowns
        _, self.crownPeak = scene.crownPeaks()
        self.crownArea = self.crowns.geometry.area.to_numpy()
        centroids = np.column_stack([self.crowns.geometry.centroid.x,
                                     self.crowns.geometry.centroid.y])
        self.nearest = cKDTree(centroids).query(centroids, k=2)[0][:, 1]
        self.results = {name: self._classify(run)
                        for name, run in runs.items()}

    # ------------------------------------------------------------------ #

    def _classify(self, run):
        predictions = run.predictions
        xs = np.array([p["centreX"] for p in predictions])
        ys = np.array([p["centreY"] for p in predictions])
        scores = np.array([p["score"] for p in predictions])
        kinds, hitCrowns, assigned = assignToCrowns(xs, ys, scores,
                                                    self.crowns)
        primary = {crown: point for point, crown in assigned.items()
                   if kinds[point] == "hit"}
        heights = self.heightsAt(xs, ys)
        return {"xs": xs, "ys": ys, "scores": scores, "kinds": kinds,
                "heights": heights, "found": set(primary), "primary": primary,
                "split": splitBackground(xs, ys, heights, kinds)}

    def heightsAt(self, xs, ys):
        """The highest CHM value within HEIGHT_RADIUS_M of each point."""
        radius = self.scene.metresToPixels(HEIGHT_RADIUS_M)
        rows, columns = self.scene.chm.shape
        heights = np.zeros(len(xs))
        for index, (x, y) in enumerate(zip(xs, ys)):
            column, row = self.scene.toPixel(x, y)
            column, row = int(column), int(row)
            window = self.scene.chm[max(0, row - radius):row + radius + 1,
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

    def missedProfile(self, name):
        """How the crowns a method missed differ from the ones it found."""
        found = np.zeros(len(self.crowns), bool)
        found[list(self.results[name]["found"])] = True
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
        for name, result in self.results.items():
            found = np.zeros(len(self.crowns), bool)
            found[list(result["found"])] = True
            table["found_" + name] = found
        return table

    # ------------------------------------------------------------------ #

    def boxQuality(self, name, polygons=None):
        """
        How well each hit's shape matches its crown.

        Box IoU compares the detection's box with the crown's bounding box;
        shape IoU compares its outline, when it has one, with the crown itself.
        `polygons` supplies outlines for methods that do not produce them, such
        as pseudo-crowns for the point detector.
        """
        result = self.results[name]
        predictions = self.runs[name].predictions
        boxIou, shapeIou = [], []
        for crown, point in result["primary"].items():
            geometry = self.crowns.geometry.iloc[crown]
            outline, predictedBox = _shapes(predictions[point], polygons,
                                            point)
            if predictedBox is not None:
                boxIou.append(_iou(predictedBox,
                                   shapelyBox(*geometry.bounds)))
            if outline is not None:
                shapeIou.append(_iou(outline, geometry))
        return {"boxIou": np.array(boxIou), "shapeIou": np.array(shapeIou)}


# ---------------------------------------------------------------------- #

def matchBlock(name, blocks):
    """The block called `name`, or the one sharing its _bNN suffix."""
    if name in blocks:
        return name
    suffix = name[name.rindex("_b"):]
    matches = [b for b in blocks if b.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError("no block matching %s" % name)
    return matches[0]


def nextBlock(block, blocks):
    """The validation-block rule of dlCommon.chooseValidationBlock."""
    others = sorted(name for name in blocks if name != block)
    later = [name for name in others if name > block]
    return later[0] if later else others[0]


def profile(mask, area, peak, nearest):
    """Count and medians of area, height and nearest-neighbour distance."""
    if not mask.any():
        return {"count": 0, "area": None, "peak": None, "nearest": None}
    return {"count": int(mask.sum()),
            "area": float(np.median(area[mask])),
            "peak": float(np.median(peak[mask])),
            "nearest": float(np.median(nearest[mask])),
            "areaQuartiles": np.percentile(area[mask], [25, 75]).tolist(),
            "peakQuartiles": np.percentile(peak[mask], [25, 75]).tolist()}


def _shapes(prediction, polygons, index):
    """
    A hit's outline and box. A box detector supplies both itself; a point
    detector's come from the supplied pseudo-crowns, box being its bounds.
    """
    if polygons is None:
        return _outline(prediction), shapelyBox(*prediction["box"])
    outline = polygons[index]
    return outline, (shapelyBox(*outline.bounds) if outline is not None
                     else None)


def _outline(prediction):
    ring = prediction.get("polygon")
    if not ring or len(ring) < 3:
        return None
    polygon = Polygon(ring)
    return polygon if polygon.is_valid else polygon.buffer(0)


def _iou(first, second):
    union = first.union(second).area
    return first.intersection(second).area / union if union > 0 else 0.0


def loadRuns(specs, blocks):
    """{name: MethodRun} from "name=runDir" strings, skipping missing ones."""
    runs = {}
    for spec in specs:
        name, runDir = spec.split("=", 1)
        try:
            runs[name] = MethodRun(name, runDir, blocks)
        except FileNotFoundError as error:
            print("[compare] %s skipped: %s" % (name, error))
    return runs
