"""
DeepForest's prebuilt tree-crown model, used as downloaded, written as a run
like any other so connected components can be tuned on its boxes with no
annotation of the site at all.

    python -m tt.deepForestDetect --dataset <site>/ds/rgb \\
        --output <out>/<site>/runs/deepforestRgb

Needs deepforest >= 2.0 (its own environment; see deepForestStudy.sh). The
model is a RetinaNet trained on NEON's 10 cm RGB, so the mosaic is resampled
to 0.1 m (pixel averages) over the blocks' extent and predicted with
DeepForest's own defaults (400 px windows, 5% overlap). Nothing is tuned.

The run holds, per block, every box above the score floor over the whole
area, with the operating point THRESHOLD, fixed before any result: there is
no annotation here to choose one, so tt.pseudoTuning and the comparison keep
the boxes at or above it, as they keep a network's above its own.
"""

import argparse
import math
import os
import sys

import numpy as np
import rasterio
from affine import Affine
from deepforest import main as dfMain
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely.ops import unary_union

from .dl import dlCommon as dc
from .dl import dlPrepare as dp
from .scene import readCrowns

THRESHOLD = 0.3
FLOOR = 0.05            # boxes kept below the threshold, as the networks keep


def mosaicPath(meta, override=None):
    if override:
        return override
    return next(c["path"] for c in meta["channels"] if c["kind"] == "rgb")


def toByte(bands):
    """Three bands as 0..255: unchanged if 8-bit, else a 2-98% stretch."""
    if bands.dtype == np.uint8:
        return bands.astype(np.float32)
    out = np.zeros(bands.shape, np.float32)
    for i, band in enumerate(bands.astype(np.float64)):
        valid = band[np.isfinite(band) & (band > 0)]
        if not valid.size:
            continue
        low, high = np.percentile(valid, [2, 98])
        out[i] = np.clip((band - low) / max(high - low, 1e-9), 0, 1) * 255
    return out


def readMosaic(path, bounds, resolution):
    """(H x W x 3 float image 0..255, its transform) over bounds."""
    west, south, east, north = bounds
    width = int(math.ceil((east - west) / resolution))
    height = int(math.ceil((north - south) / resolution))
    with rasterio.open(path) as source:
        window = from_bounds(west, south, east, north, source.transform)
        bands = source.read([1, 2, 3], window=window, out_shape=(3, height, width),
                            resampling=Resampling.average, boundless=True,
                            fill_value=0)
    transform = Affine(resolution, 0.0, west, 0.0, -resolution, north)
    return np.moveaxis(toByte(bands), 0, -1), transform


def loadModel(floor):
    """The prebuilt tree model, keeping boxes down to the floor."""
    return dfMain.deepforest(config_args={"score_thresh": floor})


def toPredictions(frame, transform):
    """DeepForest's pixel boxes as world-space predictions."""
    if frame is None or not len(frame):
        return []
    predictions = []
    for row in frame.itertuples():
        x0, y0 = transform * (row.xmin, row.ymax)
        x1, y1 = transform * (row.xmax, row.ymin)
        predictions.append({"box": [float(x0), float(y0), float(x1), float(y1)],
                            "centreX": float((x0 + x1) / 2),
                            "centreY": float((y0 + y1) / 2),
                            "score": float(row.score)})
    return predictions


def writeRun(output, blocks, predictions, crowns, threshold):
    """One predictions_<block>.json per block, and the pooled results."""
    os.makedirs(output, exist_ok=True)
    folds = []
    for name, geometry in blocks:
        dc.saveJson({"block": name, "threshold": threshold,
                     "scoreFloor": FLOOR, "predictions": predictions},
                    os.path.join(output, "predictions_%s.json" % name))
        result = dc.evaluateDetections(
            [p for p in predictions if p["score"] >= threshold], crowns,
            geometry)
        folds.append(dict(result, block=name, threshold=threshold))
    pooled = dc.averageFolds(folds)
    dc.saveJson({"method": "DeepForest prebuilt (as downloaded)",
                 "threshold": threshold, "folds": folds, "pooled": pooled},
                os.path.join(output, "results.json"))
    return pooled


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mosaic", default=None,
                        help="the RGB mosaic, if the dataset's path moved")
    parser.add_argument("--resolution", type=float, default=0.1)
    args = parser.parse_args(argv)

    meta = dp.loadDataset(args.dataset)
    blocks = meta["blockGeometries"]
    bounds = unary_union([g for _, g in blocks]).bounds
    image, transform = readMosaic(mosaicPath(meta, args.mosaic), bounds,
                                  args.resolution)
    print("[deepforest] %s: %d x %d px at %.2f m"
          % (meta["name"], image.shape[1], image.shape[0], args.resolution))
    frame = loadModel(FLOOR).predict_tile(image=image)
    predictions = toPredictions(frame, transform)
    crowns, _ = readCrowns(meta["crownsPath"], meta["crs"])
    pooled = writeRun(args.output, blocks, predictions, crowns, THRESHOLD)
    print("[deepforest] %d boxes; at %.2f: R %.3f P %.3f F1 %.3f W %.3f"
          % (len(predictions), THRESHOLD, pooled["recall"],
             pooled["precision"], pooled["f1"], pooled["weighted"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
