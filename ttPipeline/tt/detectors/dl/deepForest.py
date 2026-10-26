"""
PretrainedDetector — DeepForest's prebuilt tree-crown model, used as
downloaded, with no training and no tuning on the site.

The model is a RetinaNet trained on NEON's 10 cm RGB, so the mosaic is
resampled to 0.1 m (pixel averages) over the area of the site's blocks and
predicted with DeepForest's own defaults (400 px windows, 5% overlap). The
operating point is fixed before any result: there is no annotation to choose
one with. Needs deepforest >= 2.0, in its own environment.
"""

import math

import numpy as np
import rasterio
from affine import Affine
from deepforest import main as dfMain
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from ...data.detections import Detections
from ..base import Detector

THRESHOLD = 0.3
FLOOR = 0.05            # boxes kept below the threshold, as the networks keep


class PretrainedDetector(Detector):

    method = "DeepForest prebuilt (as downloaded)"
    geometry = "boxes"
    defaults = {"resolution": 0.1, "threshold": THRESHOLD,
                "scoreFloor": FLOOR, "mosaic": None}

    def _detect(self, data):
        """Every box above the score floor over the blocks' extent."""
        path = self.params["mosaic"] or data.layer("RGB").path
        image, transform = readMosaic(path, data.blockArea.bounds,
                                      self.params["resolution"])
        print("[deepforest] %s: %d x %d px at %.2f m"
              % (data.name, image.shape[1], image.shape[0],
                 self.params["resolution"]))
        model = dfMain.deepforest(
            config_args={"score_thresh": self.params["scoreFloor"]})
        return Detections.fromPredictions(
            toPredictions(model.predict_tile(image=image), transform),
            "boxes")


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
        bands = source.read([1, 2, 3], window=window,
                            out_shape=(3, height, width),
                            resampling=Resampling.average, boundless=True,
                            fill_value=0)
    transform = Affine(resolution, 0.0, west, 0.0, -resolution, north)
    return np.moveaxis(toByte(bands), 0, -1), transform


def toPredictions(frame, transform):
    """DeepForest's pixel boxes as world-space predictions."""
    if frame is None or not len(frame):
        return []
    predictions = []
    for row in frame.itertuples():
        x0, y0 = transform @ (row.xmin, row.ymax)
        x1, y1 = transform @ (row.xmax, row.ymin)
        predictions.append({"box": [float(x0), float(y0), float(x1), float(y1)],
                            "centreX": float((x0 + x1) / 2),
                            "centreY": float((y0 + y1) / 2),
                            "score": float(row.score)})
    return predictions
