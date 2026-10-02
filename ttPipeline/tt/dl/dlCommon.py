"""
dlCommon.py

Shared machinery for benchmarking treetop / crown detectors against each other:
tiling, spatial blocking for cross-validation, prediction stitching, and one
evaluation function that every method is scored through.

Why the pieces are shaped this way
----------------------------------

*Spatial blocks, not random splits.* Neighbouring tiles of a forest share
individual trees, illumination and stand structure. A random tile split leaks
the test set into training and flatters every learned model. The area is cut
into a grid of blocks; a fold holds out one whole block, and training tiles must
lie a buffer away from it so no crown appears on both sides.

*Stitched evaluation.* Tiles overlap so that trees near an edge are seen whole
somewhere. Scoring tile by tile then counts those trees more than once and
inflates the result — Burmaa's own README measured that at 0.17 to 0.25 of
detection F1. Predictions are merged back into world coordinates, deduplicated
with NMS, and each tree is scored once.

*One metric for everything.* The connected-component detector produces points;
Mask R-CNN produces masks; YOLO produces boxes or polygons. They are compared on
the crown-level rule already used in conCompTreetopDetection: a prediction is
reduced to its centre, a crown counts as found if any centre falls inside it,
the highest-scoring centre in a crown is the hit and any others are repeats, and
a centre in no crown is a false positive. That is the number reported as recall,
precision and F1, so the learned models and the hand-tuned one are directly
comparable. Box AP50 is reported alongside for readers who expect it, but it is
not the headline.

Normalisation is fixed once per dataset and stored, so a model never sees test
tiles scaled differently from the tiles it trained on.
"""

import datetime
import json
import os
import sys

from affine import Affine
import numpy as np
from shapely import contains_xy
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.geometry import Point, box

from ..evaluation import assignToCrowns



# ---------------------------------------------------------------------- #
# source definitions
# ---------------------------------------------------------------------- #

class ChannelSpec(object):
    """One contribution to a tile's channels: a height model or a photo."""

    def __init__(self, kind, path, bands=None):
        if kind not in ("chm", "rgb"):
            raise ValueError("kind must be 'chm' or 'rgb', got %r" % kind)
        self.kind = kind
        self.path = path
        self.bands = list(bands) if bands else ([1, 2, 3] if kind == "rgb"
                                                else [1])

    @property
    def channelCount(self):
        return 3 if self.kind == "rgb" else 1

    def describe(self):
        return {"kind": self.kind, "path": self.path, "bands": self.bands}


def parseSource(text):
    """
    'rgb:ortho.tif+chm:lidar.tif' -> [ChannelSpec('rgb', ...),
                                      ChannelSpec('chm', ...)]
    Channels are stacked in the order given, so RGB first then CHM produces the
    4-channel input with the photo in the first three bands.
    """
    specs = []
    for part in text.split("+"):
        part = part.strip()
        if ":" not in part:
            raise ValueError("expected kind:path, got %r" % part)
        kind, path = part.split(":", 1)
        specs.append(ChannelSpec(kind.strip(), path.strip()))
    return specs


# ---------------------------------------------------------------------- #
# the shared grid
# ---------------------------------------------------------------------- #

def buildGrid(referencePath, boundary, resolution):
    """
    A grid covering `boundary` at `resolution`, aligned to the reference raster
    so no resampling shifts anything. Returns (transform, width, height, crs).
    """
    with rasterio.open(referencePath) as src:
        crs = src.crs
        native = src.transform

    minX, minY, maxX, maxY = boundary.bounds
    # snap the origin to a multiple of the resolution from the raster origin
    offsetX = np.floor((minX - native.c) / resolution) * resolution
    offsetY = np.ceil((native.f - maxY) / resolution) * resolution
    originX = native.c + offsetX
    originY = native.f - offsetY

    width = int(np.ceil((maxX - originX) / resolution))
    height = int(np.ceil((originY - minY) / resolution))
    # Built directly rather than through rasterio.transform.from_origin, which
    # still multiplies Affines with * and so warns under affine 3.0.
    transform = Affine(resolution, 0.0, originX, 0.0, -resolution, originY)
    return transform, width, height, crs


def readWindow(spec, transform, c0, r0, c1, r1, heightRange=None,
               source=None):
    """
    Read one channel spec over a grid window, returned as uint8 (C, H, W).

    A photo is percentile-stretched per band over the whole dataset, not per
    tile, so brightness is comparable between tiles. A height model is mapped
    linearly from `heightRange` to 1..255 with 0 reserved for no canopy, which
    keeps "no canopy" distinguishable from "canopy at the lowest height".

    Pass an already open `source` when reading many windows: opening the file
    per tile re-parses its header every time.
    """
    if source is None:
        with rasterio.open(spec.path) as opened:
            return readWindow(spec, transform, c0, r0, c1, r1, heightRange,
                              source=opened)

    height, width = r1 - r0, c1 - c0
    west, north = transform @ (c0, r0)
    east, south = transform @ (c1, r1)
    window = from_bounds(min(west, east), min(north, south),
                         max(west, east), max(north, south),
                         transform=source.transform)
    data = source.read(indexes=spec.bands, window=window,
                       out_shape=(len(spec.bands), height, width),
                       resampling=Resampling.average,
                       boundless=True, fill_value=0).astype(np.float32)

    if spec.kind == "rgb":
        lo, hi = heightRange  # reused as the per-source stretch range
        scaled = np.clip((data - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return (255 * scaled).astype(np.uint8)

    band = data[0]
    band[~np.isfinite(band)] = 0.0
    lo, hi = heightRange
    band[band < lo] = 0.0
    canopy = band > 0
    out = np.zeros(band.shape, np.uint8)
    out[canopy] = (1 + 254 * np.clip((band[canopy] - lo) / max(hi - lo, 1e-6),
                                     0.0, 1.0)).astype(np.uint8)
    return out[None, :, :]


def measureRanges(specs, transform, width, height, minHeight=2.0,
                  sampleShape=1024, verbose=True):
    """
    Work out, once for the whole dataset, the value range each source is scaled
    with. Doing this per tile would make identical trees look different from one
    tile to the next.
    """
    ranges = []
    for spec in specs:
        step = max(1, int(max(width, height) / sampleShape))
        west, north = transform @ (0, 0)
        east, south = transform @ (width, height)
        with rasterio.open(spec.path) as src:
            window = from_bounds(min(west, east), min(north, south),
                                 max(west, east), max(north, south),
                                 transform=src.transform)
            data = src.read(indexes=spec.bands, window=window,
                            out_shape=(len(spec.bands),
                                       max(1, height // step),
                                       max(1, width // step)),
                            resampling=Resampling.average,
                            boundless=True, fill_value=0).astype(np.float32)
        if spec.kind == "rgb":
            values = data[np.isfinite(data) & (data != 0)]
            lo, hi = (np.percentile(values, (1, 99)) if values.size
                      else (0.0, 255.0))
        else:
            band = data[0]
            values = band[np.isfinite(band) & (band >= minHeight)]
            lo = minHeight
            hi = float(np.percentile(values, 99.5)) if values.size else 30.0
        ranges.append((float(lo), float(hi)))
        if verbose:
            print("[data] %s %s -> scaled from %.2f to %.2f"
                  % (spec.kind, os.path.basename(spec.path), lo, hi))
    return ranges


# ---------------------------------------------------------------------- #
# blocks and tiles
# ---------------------------------------------------------------------- #

def makeBlocks(boundary, blockCols, blockRows, areaName="area"):
    """Cut the boundary into a grid of blocks. Returns [(name, geometry), ...]."""

    minX, minY, maxX, maxY = boundary.bounds
    stepX = (maxX - minX) / blockCols
    stepY = (maxY - minY) / blockRows

    blocks = []
    for row in range(blockRows):
        for col in range(blockCols):
            cell = box(minX + col * stepX, maxY - (row + 1) * stepY,
                       minX + (col + 1) * stepX, maxY - row * stepY)
            piece = cell.intersection(boundary)
            if piece.is_empty or piece.area <= 0:
                continue
            blocks.append(("%s_b%d%d" % (areaName, row, col), piece))
    return blocks


def makeTiles(transform, width, height, tileSize, stride, boundary,
              minCoverage=0.5):
    """
    Tile windows over the grid, keeping those sufficiently inside the boundary.
    Returns [{"c0","r0","c1","r1","centreX","centreY"}, ...] in world units.
    """

    tiles = []
    for r0 in range(0, max(1, height - tileSize + 1), stride):
        for c0 in range(0, max(1, width - tileSize + 1), stride):
            c1, r1 = c0 + tileSize, r0 + tileSize
            if c1 > width or r1 > height:
                continue
            west, north = transform @ (c0, r0)
            east, south = transform @ (c1, r1)
            extent = box(min(west, east), min(north, south),
                         max(west, east), max(north, south))
            if extent.intersection(boundary).area < minCoverage * extent.area:
                continue
            centre = extent.centroid
            tiles.append({"c0": c0, "r0": r0, "c1": c1, "r1": r1,
                          "centreX": float(centre.x),
                          "centreY": float(centre.y)})
    return tiles


def assignTilesToBlocks(tiles, blocks):
    """Label each tile with the block its centre falls in."""

    for tile in tiles:
        tile["block"] = None
        point = Point(tile["centreX"], tile["centreY"])
        for name, geometry in blocks:
            if geometry.contains(point):
                tile["block"] = name
                break
    return tiles


def foldSplit(tiles, blocks, heldOut, bufferM):
    """
    Training tiles for a fold: those whose tile extent stays at least `bufferM`
    from the held-out block, so no tree is in both. Test tiles: those centred in
    the held-out block.
    """

    geometry = dict(blocks)[heldOut]
    guard = geometry.buffer(bufferM)

    train, test = [], []
    for tile in tiles:
        if tile["block"] == heldOut:
            test.append(tile)
            continue
        extent = box(min(tile["west"], tile["east"]),
                     min(tile["north"], tile["south"]),
                     max(tile["west"], tile["east"]),
                     max(tile["north"], tile["south"]))
        if not extent.intersects(guard):
            train.append(tile)
    return train, test


# ---------------------------------------------------------------------- #
# crown labels per tile
# ---------------------------------------------------------------------- #

def cropCrowns(crowns, crownBounds, transform, c0, r0, c1, r1,
               minInsideFraction=0.5, minAreaPx=16):
    """
    Crowns for one tile, in tile-pixel coordinates.

    A crown clipped to less than `minInsideFraction` of its area is dropped:
    training a detector on a sliver teaches it that a third of a tree is a tree.
    Crowns near the edge are still seen whole in an overlapping neighbour tile.

    Returns [{"box": [x0,y0,x1,y1], "polygon": [[x,y], ...]}, ...]
    """
    west, north = transform @ (c0, r0)
    east, south = transform @ (c1, r1)
    minX, maxX = min(west, east), max(west, east)
    minY, maxY = min(north, south), max(north, south)
    extent = box(minX, minY, maxX, maxY)
    candidates = np.nonzero((crownBounds[:, 0] <= maxX)
                            & (crownBounds[:, 2] >= minX)
                            & (crownBounds[:, 1] <= maxY)
                            & (crownBounds[:, 3] >= minY))[0]

    inverse = ~transform
    out = []
    for index in candidates:
        piece = _clipToTile(crowns.geometry.iloc[index], extent,
                            minInsideFraction)
        if piece is None:
            continue
        ring = _tilePixels(np.asarray(piece.exterior.coords), inverse,
                           c0, r0, c1 - c0, r1 - r0)
        x0, y0 = ring[:, 0].min(), ring[:, 1].min()
        x1, y1 = ring[:, 0].max(), ring[:, 1].max()
        if (x1 - x0) * (y1 - y0) < minAreaPx:
            continue
        out.append({"box": [float(x0), float(y0), float(x1), float(y1)],
                    "polygon": ring.tolist(), "crownIndex": int(index)})
    return out


def _clipToTile(geometry, extent, minInsideFraction):
    """The largest piece of a crown inside the tile, or None if too little."""
    if geometry is None or geometry.is_empty:
        return None
    clipped = geometry.intersection(extent)
    if clipped.is_empty or clipped.area < minInsideFraction * geometry.area:
        return None
    pieces = [p for p in getattr(clipped, "geoms", [clipped])
              if getattr(p, "exterior", None) is not None]
    return max(pieces, key=lambda p: p.area) if pieces else None


def _tilePixels(coordinates, inverse, c0, r0, width, height):
    """
    World coordinates to tile pixels, clipped to the tile.

    Vectorised, but in affine's own arithmetic order — x*a + y*b + c — so the
    result is bit-identical to applying the transform one vertex at a time.
    """
    a, b, c, d, e, f = inverse[:6]
    x, y = coordinates[:, 0], coordinates[:, 1]
    ring = np.column_stack([x * a + y * b + c - c0, x * d + y * e + f - r0])
    ring[:, 0] = np.clip(ring[:, 0], 0, width)
    ring[:, 1] = np.clip(ring[:, 1], 0, height)
    return ring


# ---------------------------------------------------------------------- #
# stitching
# ---------------------------------------------------------------------- #

def tileToWorld(predictions, transform, c0, r0):
    """
    Tile-pixel predictions to world coordinates, adding a centre.

    Fields other than the box are carried through, and a tile-pixel "polygon"
    is converted with the box, so a mask outline survives stitching and NMS
    and can be saved beside the score.
    """
    out = []
    for prediction in predictions:
        x0, y0, x1, y1 = prediction["box"]
        west, north = transform @ (c0 + x0, r0 + y0)
        east, south = transform @ (c0 + x1, r0 + y1)
        centreX, centreY = transform @ (c0 + 0.5 * (x0 + x1),
                                        r0 + 0.5 * (y0 + y1))
        world = dict(prediction)
        world.update({"box": [min(west, east), min(north, south),
                              max(west, east), max(north, south)],
                      "centreX": float(centreX), "centreY": float(centreY),
                      "score": float(prediction.get("score", 1.0))})
        if prediction.get("polygon"):
            ring = np.asarray(prediction["polygon"], np.float64)
            eastings = transform.c + transform.a * (c0 + ring[:, 0])
            northings = transform.f + transform.e * (r0 + ring[:, 1])
            world["polygon"] = np.column_stack([eastings,
                                                northings]).tolist()
        out.append(world)
    return out


def nonMaximumSuppression(predictions, iouThreshold=0.4):
    """Plain NMS in world coordinates, to merge the same tree seen in two tiles."""
    if not predictions:
        return []
    boxes = np.array([p["box"] for p in predictions], np.float64)
    scores = np.array([p["score"] for p in predictions], np.float64)
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * \
        np.maximum(0, boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-scores)

    keep = []
    while order.size:
        current = order[0]
        keep.append(int(current))
        if order.size == 1:
            break
        rest = order[1:]
        x0 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y0 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x1 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y1 = np.minimum(boxes[current, 3], boxes[rest, 3])
        overlap = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
        union = areas[current] + areas[rest] - overlap
        iou = overlap / np.maximum(union, 1e-9)
        order = rest[iou <= iouThreshold]

    return [predictions[i] for i in keep]


# ---------------------------------------------------------------------- #
# the tuning objective
# ---------------------------------------------------------------------- #

# Every choice made on held-out-free data (a connected-component setting per
# fold, a network's score threshold, a combination's settings, a pseudo-tuned
# setting) maximises one objective, chosen by the environment so a whole run
# uses one: TT_OBJECTIVE=f1 (the default) or TT_OBJECTIVE=weighted, the
# weighted mean RECALL_WEIGHT * recall + (1 - RECALL_WEIGHT) * precision.
# Every result carries both "f1" and "weighted", whichever was tuned on.
RECALL_WEIGHT = 0.6
OBJECTIVES = ("f1", "weighted")


def tuningObjective():
    objective = os.environ.get("TT_OBJECTIVE", "f1")
    if objective not in OBJECTIVES:
        raise ValueError("TT_OBJECTIVE must be one of %s, not %r"
                         % (OBJECTIVES, objective))
    return objective


def weightedScore(recall, precision):
    return RECALL_WEIGHT * recall + (1.0 - RECALL_WEIGHT) * precision


def objectiveOf(result):
    """The value the tuning maximises for one scored result."""
    if tuningObjective() == "weighted":
        return result.get("weighted",
                          weightedScore(result["recall"], result["precision"]))
    return result["f1"]


def _scores(recall, precision):
    f1 = (2 * recall * precision / (recall + precision)
          if recall + precision else 0.0)
    return f1, weightedScore(recall, precision)


# ---------------------------------------------------------------------- #
# the one evaluation
# ---------------------------------------------------------------------- #

def evaluateDetections(predictions, crowns, blockGeometry, verbose=False):
    """
    Score predictions against crowns inside one block.

    Only crowns whose centroid is in the block, and only predictions whose
    centre is in the block, so a tree on a block boundary is scored by exactly
    one fold. The scoring rule itself is tt.evaluation.assignToCrowns, the same
    one every other score in the package uses.
    """
    inBlock = crownsInRegion(crowns, blockGeometry)
    kept = predictionsInRegion(predictions, blockGeometry)

    result = {"crowns": len(inBlock), "predictions": len(kept),
              "hits": 0, "repeats": 0, "falsePositives": 0,
              "recall": 0.0, "precision": 0.0, "f1": 0.0, "weighted": 0.0}
    if not len(inBlock) or not kept:
        return result

    kinds, _, _ = assignToCrowns(
        np.array([p["centreX"] for p in kept]),
        np.array([p["centreY"] for p in kept]),
        np.array([p["score"] for p in kept]), inBlock)

    hits = kinds.count("hit")
    recall = hits / float(len(inBlock))
    precision = hits / float(len(kept))
    result.update({"hits": hits, "repeats": kinds.count("repeat"),
                   "falsePositives": kinds.count("background"),
                   "recall": recall, "precision": precision})
    result["f1"], result["weighted"] = _scores(recall, precision)
    if verbose:
        print("    crowns %d, predictions %d -> hits %d, repeats %d, fp %d  "
              "| R %.3f P %.3f F1 %.3f"
              % (result["crowns"], result["predictions"], hits,
                 result["repeats"], result["falsePositives"],
                 recall, precision, result["f1"]))
    return result


def crownsInRegion(crowns, region):
    """Crowns whose centroid lies in `region`, re-indexed from zero."""
    centroids = crowns.geometry.centroid
    inside = contains_xy(region, centroids.x.to_numpy(), centroids.y.to_numpy())
    return crowns[inside].reset_index(drop=True)


def predictionsInRegion(predictions, region):
    """Predictions whose centre lies in `region`, tested in one vectorised call."""
    if not predictions:
        return []
    inside = contains_xy(region,
                         np.array([p["centreX"] for p in predictions]),
                         np.array([p["centreY"] for p in predictions]))
    return [p for p, keep in zip(predictions, inside) if keep]


def chooseValidationBlock(trainTiles, heldOut):
    """
    One training block set aside for choosing the operating point.

    Deterministic: the training block following the held-out one in sorted
    order, so a fold always makes the same choice and reruns are comparable.
    """
    blocks = sorted({tile["block"] for tile in trainTiles})
    if not blocks:
        return None
    later = [b for b in blocks if b > heldOut]
    return later[0] if later else blocks[0]


THRESHOLD_CANDIDATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                        0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def selectThreshold(predictions, crowns, blockGeometry,
                    candidates=THRESHOLD_CANDIDATES, verbose=True):
    """
    Pick the score threshold that maximises the tuning objective (F1, or the
    recall-weighted mean: tuningObjective) on a validation block.

    Without this the comparison is between thresholds, not methods: Mask R-CNN
    and YOLO default to different confidence scales, and comparing each at its
    own default measured the defaults. A detector is judged at its own best
    operating point, chosen somewhere other than the block it is scored on.

    Which crown a point belongs to does not depend on the threshold, so points
    are assigned once. A crown is then hit at threshold t exactly when the best
    score among its points clears t, which answers every candidate from one
    spatial join instead of one join per candidate.

    Returns (threshold, objectiveAtThatThreshold). When no candidate keeps
    anything the lowest is returned with 0.0, and said so.
    """
    inBlock = crownsInRegion(crowns, blockGeometry)
    kept = predictionsInRegion(predictions, blockGeometry)
    curve = thresholdCurve(kept, inBlock, candidates)
    if not curve:
        if verbose:
            print("    no predictions on the validation block at any "
                  "threshold; falling back to %.2f" % candidates[0])
        return candidates[0], 0.0

    best, bestScore = max(curve, key=lambda item: item[1])
    if verbose:
        print("    operating point chosen on the validation block: "
              "score >= %.2f (%s %.3f there)"
              % (best, tuningObjective(), bestScore))
    return best, bestScore


def thresholdCurve(predictions, crowns, candidates):
    """
    [(threshold, objective)] for every candidate that keeps at least one
    point; the objective is the tuning one (F1 unless TT_OBJECTIVE says).
    """
    if not predictions or not len(crowns):
        return []
    scores = np.array([p["score"] for p in predictions])
    _, _, assigned = assignToCrowns(
        np.array([p["centreX"] for p in predictions]),
        np.array([p["centreY"] for p in predictions]), scores, crowns)

    bestPerCrown = {}
    for point, crown in assigned.items():
        bestPerCrown[crown] = max(bestPerCrown.get(crown, -np.inf),
                                  scores[point])
    crownBest = np.array(list(bestPerCrown.values()))

    curve = []
    for threshold in candidates:
        kept = int((scores >= threshold).sum())
        if not kept:
            continue
        hits = int((crownBest >= threshold).sum())
        recall, precision = hits / float(len(crowns)), hits / float(kept)
        f1, weighted = _scores(recall, precision)
        curve.append((threshold, weighted if tuningObjective() == "weighted"
                      else f1))
    return curve


def averageFolds(foldResults):
    """
    Pool the folds. Counts are summed before the ratios are formed, so a block
    with few trees does not weigh the same as a block with many — a mean of
    per-fold F1 would do that and is the more common mistake.
    """
    total = {"crowns": 0, "predictions": 0, "hits": 0, "repeats": 0,
             "falsePositives": 0}
    for fold in foldResults:
        for key in total:
            total[key] += fold[key]

    recall = total["hits"] / float(total["crowns"]) if total["crowns"] else 0.0
    precision = total["hits"] / float(total["predictions"]) \
        if total["predictions"] else 0.0
    f1, weighted = _scores(recall, precision)

    perFoldF1 = [f["f1"] for f in foldResults if f["crowns"] > 0]
    total.update({"recall": recall, "precision": precision, "f1": f1,
                  "weighted": weighted, "objective": tuningObjective(),
                  "folds": len(foldResults),
                  "perFoldF1Mean": float(np.mean(perFoldF1))
                  if perFoldF1 else 0.0,
                  "perFoldF1Std": float(np.std(perFoldF1))
                  if perFoldF1 else 0.0})
    return total


# ---------------------------------------------------------------------- #

class Tee(object):
    """Duplicate a stream to a file, so a run is both watchable and recorded."""

    def __init__(self, stream, handle):
        self.stream = stream
        self.handle = handle

    def write(self, text):
        self.stream.write(text)
        self.handle.write(text)
        self.handle.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.handle.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


def startLogging(path):
    """
    Send everything printed from here on to `path` as well as the terminal.

    Progress bars write carriage returns, so a log file ends up holding every
    intermediate state of the bar. Worth knowing when reading one back; `col -b`
    or `sed 's/.*\r//'` cleans it up.
    """

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    handle = open(path, "a", buffering=1)
    handle.write("\n%s\n=== %s ===\n" % ("=" * 70, _timestamp()))
    sys.stdout = Tee(sys.stdout, handle)
    sys.stderr = Tee(sys.stderr, handle)
    return handle


def _timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def saveJson(obj, path):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    return path


def loadJson(path):
    with open(path) as f:
        return json.load(f)
