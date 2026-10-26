"""
Detectors that work on a height model: connected components (CC), local
maxima in a variable window, and marker-controlled watershed.

Each reads its height surface from the site through the shared parameters
(which layer, resolution, minimum height, smoothing), so those are tuned the
same way for all three, and returns points scored by their height.
"""

import numpy as np
from scipy.ndimage import label, maximum_filter, watershed_ift

from ..data.detections import Detections
from ..data.raster import LOCAL_HEIGHT_M
from .base import Detector
from .concomp import ConnectedComponents
from .merging import TopMerger

MAX_WINDOW_M = 12.0     # no local-maximum window larger than this


class HeightDetector(Detector):
    """A detector on one height model of the site."""

    surfaceDefaults = {
        "heightKind": "CHM", "heightSource": None, "resolution": 0.25,
        "minHeight": 2.0, "smoothM": 0.0, "smoothSlope": 0.0,
        "smoothWindowM": LOCAL_HEIGHT_M,
    }

    def surface(self, data):
        p = self.params
        return data.heightSurface(
            p["heightKind"], p["heightSource"], p["resolution"],
            p["minHeight"], p["smoothM"], p["smoothSlope"], p["smoothWindowM"])

    def prepare(self, data):
        self.surface(data)

    def _detect(self, data):
        surface = self.surface(data)
        return Detections.fromPixels(self.tops(surface), surface)

    def tops(self, surface):
        """(column, row, height) tops on the surface."""
        raise NotImplementedError


class CCDetector(HeightDetector):
    """
    Connected components with band descent. The merge rule is part of the
    algorithm (the elder rule is applied during the descent), so its
    parameters are CC parameters and tuned with the rest.
    """

    method = "concomp"
    defaults = dict(HeightDetector.surfaceDefaults, **{
        "windowSizeM": 40.0, "windowOverlap": 0.2, "lowerPercentile": 20,
        "upperPercentile": 99, "erosionKernelSize": 3, "erosionIterations": 1,
        "minTreeAreaM2": 0.5, "minTopAreaM2": 0.25, "minTopAreaSlope": 0.0,
        "topStepM": 0.25, "refine": True, "mergeMetric": "saddle",
        "saddleEpsM": 8.0, "heightWeight": 0.7, "saddleDropM": 0.5,
        "saddleDropSlope": 0.0,
    })
    # the merge threshold is tuned with the detector: a base that
    # over-detects less needs a gentler merge
    defaultGrid = {
        "lowerPercentile": [20, 30], "minTopAreaM2": [0.12, 0.06],
        "topStepM": [0.25], "erosionIterations": [1, 2],
        "saddleDropM": [0.2, 0.3, 0.5],
    }

    def algorithm(self):
        p = self.params
        merger = TopMerger(p["mergeMetric"], epsM=p["saddleEpsM"],
                           heightWeight=p["heightWeight"],
                           saddleDropM=p["saddleDropM"],
                           saddleDropSlope=p["saddleDropSlope"])
        return ConnectedComponents(
            windowSizeM=p["windowSizeM"], windowOverlap=p["windowOverlap"],
            lowerPercentile=p["lowerPercentile"],
            upperPercentile=p["upperPercentile"],
            erosionKernelSize=p["erosionKernelSize"],
            erosionIterations=p["erosionIterations"],
            minTreeAreaM2=p["minTreeAreaM2"], minTopAreaM2=p["minTopAreaM2"],
            topStepM=p["topStepM"], minTopAreaSlope=p["minTopAreaSlope"],
            merger=merger, refine=p["refine"], verbose=False)

    def tops(self, surface):
        return self.algorithm().detect(surface)


class LocalMaximaDetector(HeightDetector):
    """
    Local maxima in a window whose side grows with height (Popescu and Wynne
    2004): a pixel is a top when it is the highest in the square window of
    side windowBaseM + windowSlope x its own height. Slope 0 is the plain
    fixed-window local maximum.
    """

    method = "baseline-lmvw"
    defaults = dict(HeightDetector.surfaceDefaults,
                    windowBaseM=1.0, windowSlope=0.0)
    defaultGrid = {"windowBaseM": [0.5, 1.0, 1.5, 2.0],
                   "windowSlope": [0.0, 0.05, 0.1, 0.15, 0.2]}

    def tops(self, surface):
        return localMaxima(surface.chm, surface.pixelSize,
                           self.params["windowBaseM"],
                           self.params["windowSlope"])


class WatershedDetector(HeightDetector):
    """
    Marker-controlled watershed: markers are the local maxima in a fixed
    window, the surface is flooded from them, and a marker is kept as a tree
    when its region covers at least the minimum crown area.
    """

    method = "baseline-watershed"
    defaults = dict(HeightDetector.surfaceDefaults,
                    markerWindowM=1.0, minCrownAreaM2=0.0)
    defaultGrid = {"markerWindowM": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
                   "minCrownAreaM2": [0.0, 0.5, 1.0, 2.0]}

    def tops(self, surface):
        return markerWatershed(surface.chm, surface.pixelSize,
                               self.params["markerWindowM"],
                               self.params["minCrownAreaM2"])


# ---------------------------------------------------------------------- #
# the baseline algorithms, on a bare array
# ---------------------------------------------------------------------- #

def oddPixels(metres, pixel):
    """A window side in pixels, odd, at least 3."""
    side = int(round(metres / pixel))
    return max(3, side + (side + 1) % 2)


def plateauTops(chm, candidates):
    """One top per 8-connected group of candidate pixels: its highest pixel."""
    labels, count = label(candidates, structure=np.ones((3, 3), bool))
    if count == 0:
        return []
    rows, cols = np.nonzero(labels)
    order = np.lexsort((-chm[rows, cols], labels[rows, cols]))
    rows, cols = rows[order], cols[order]
    first = np.r_[True, np.diff(labels[rows, cols]) != 0]
    return [(int(c), int(r), float(chm[r, c]))
            for r, c in zip(rows[first], cols[first])]


def localMaxima(chm, pixel, windowBaseM, windowSlope):
    valid = chm > 0
    window = np.minimum(windowBaseM + windowSlope * chm, MAX_WINDOW_M)
    sides = np.zeros(chm.shape, int)
    sides[valid] = np.maximum(3, np.rint(window[valid] / pixel).astype(int))
    sides[valid] += (sides[valid] + 1) % 2
    candidates = np.zeros(chm.shape, bool)
    for side in np.unique(sides[valid]):
        here = valid & (sides == side)
        candidates |= here & (chm >= maximum_filter(chm, size=int(side)))
    return plateauTops(chm, candidates)


def markerWatershed(chm, pixel, markerWindowM, minCrownAreaM2):
    valid = chm > 0
    side = oddPixels(markerWindowM, pixel)
    peaks = valid & (chm >= maximum_filter(chm, size=side))
    tops = plateauTops(chm, peaks)
    if not tops:
        return []
    markers = np.zeros(chm.shape, np.int32)
    for index, (x, y, _) in enumerate(tops, 1):
        markers[y, x] = index
    markers[~valid] = -1                     # ground floods as background
    top = float(chm.max())
    depth = np.round((top - chm) / max(top, 1e-9) * 65000).astype(np.uint16)
    regions = watershed_ift(depth, markers)
    areas = np.bincount(regions[regions > 0].ravel(),
                        minlength=len(tops) + 1) * pixel * pixel
    return [t for i, t in enumerate(tops, 1) if areas[i] >= minCrownAreaM2]
