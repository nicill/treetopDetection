"""
Alignment — how far apart two layers are, in metres.

Replaces measureShift.py, nccAlign.py and registrationDiagnostic.py, which were
three tools for one question that had drifted into disagreeing with each other.

Normalised cross-correlation, not phase correlation. Phase correlation whitens
the spectrum, and a densely annotated stand at ~50% crown cover is close to
periodic, so the peak goes ambiguous and every tile falls below the confidence
floor. That reads as "no match" when the layers are aligned to 0.1 m — it
happened, and only looking at the images caught it. The trade is a bounded
search: maxShiftM is the largest offset findable.

Validated by injecting known shifts into a real annotation: 0.5, 1.0, 1.5 and
2.0 m all recovered within 0.2 m.
"""

import itertools
import json

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.windows import from_bounds

from .scene import Scene, readCrowns


def nccShift(a, b, pixelSize, maxShiftM=4.0):
    """
    Where b's features sit relative to a's, by normalised cross-correlation.

    Returns (dxEastM, dySouthM, peak, margin) or None. `margin` is how many
    standard deviations the peak stands above the rest of the search window;
    it is informative but broad peaks on smooth canopy keep it modest, so it is
    reported rather than used as a hard filter.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a - a.mean()
    b = b - b.mean()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return None
    a /= a.std()
    b /= b.std()

    spectrumA = np.fft.rfft2(a)
    spectrumB = np.fft.rfft2(b)
    correlation = np.fft.irfft2(spectrumA * np.conj(spectrumB), s=a.shape)
    correlation = np.fft.fftshift(correlation / a.size)

    centreRow, centreCol = np.array(correlation.shape) // 2
    radius = int(round(maxShiftM / pixelSize))
    if radius < 1 or 2 * radius + 1 > min(correlation.shape):
        return None
    window = correlation[centreRow - radius:centreRow + radius + 1,
                         centreCol - radius:centreCol + radius + 1]

    peakIndex = np.unravel_index(int(np.argmax(window)), window.shape)
    peak = float(window[peakIndex])

    surroundings = np.ones_like(window, bool)
    surroundings[max(0, peakIndex[0] - 3):peakIndex[0] + 4,
                 max(0, peakIndex[1] - 3):peakIndex[1] + 4] = False
    margin = float((peak - window[surroundings].mean())
                   / (window[surroundings].std() + 1e-12))

    dy = peakIndex[0] - radius
    dx = peakIndex[1] - radius
    return -dx * pixelSize, -dy * pixelSize, peak, margin


class LayerReader(object):
    """Reads any layer as a surface on a shared working grid."""

    def __init__(self, name, path, crs, minHeight=2.0, verbose=True):
        self.name = name
        self.path = path
        self.crs = crs
        self.minHeight = float(minHeight)
        self.kind = None
        self.crowns = None

        if path.lower().endswith(".shp"):
            self.kind = "crowns"
            self.crowns, _ = readCrowns(path, crs)
            if verbose:
                print("[align] %s: %d crown polygons" % (name, len(self.crowns)))
        else:
            with rasterio.open(path) as src:
                self.kind = "photo" if src.count >= 3 else "height"
                if verbose:
                    print("[align] %s: %s, %d x %d px, %.5f m, %d bands, %s"
                          % (name, self.kind, src.width, src.height,
                             abs(src.transform.a), src.count, src.crs))
                if src.crs != crs:
                    print("[align] WARNING: %s is in %s but the reference is "
                          "in %s" % (name, src.crs, crs))

    def surface(self, transform, c0, r0, c1, r1):
        height, width = r1 - r0, c1 - c0
        west, north = transform @ (c0, r0)
        east, south = transform @ (c1, r1)

        if self.kind == "crowns":
            windowTransform = transform @ transform.translation(c0, r0)
            subset = self.crowns.cx[min(west, east):max(west, east),
                                    min(north, south):max(north, south)]
            if subset.empty:
                return None
            surface = np.zeros((height, width), np.float32)
            for geom in subset.geometry:
                if geom is None or geom.is_empty:
                    continue
                mask = rasterize([(geom, 1)], out_shape=surface.shape,
                                 transform=windowTransform, fill=0,
                                 dtype=np.uint8, all_touched=True)
                surface = np.maximum(
                    surface, cv2.distanceTransform(mask, cv2.DIST_L2, 3))
            return surface if np.count_nonzero(surface) > 0.02 * surface.size \
                else None

        bands = [1, 2, 3] if self.kind == "photo" else [1]
        with rasterio.open(self.path) as src:
            window = from_bounds(min(west, east), min(north, south),
                                 max(west, east), max(north, south),
                                 transform=src.transform)
            data = src.read(indexes=bands, window=window,
                            out_shape=(len(bands), height, width),
                            resampling=Resampling.average,
                            boundless=True, fill_value=0).astype(np.float32)

        if self.kind == "photo":
            return (2.0 * data[1] - data[0] - data[2]) if np.any(data) else None

        heights = data[0].copy()
        heights[~np.isfinite(heights)] = 0.0
        heights[heights < self.minHeight] = 0.0
        heights[heights > 100.0] = 0.0
        return heights if np.count_nonzero(heights) > 0.05 * heights.size \
            else None


class Alignment(object):
    """Measures every pair among a reference CHM and the layers added to it."""

    def __init__(self, referencePath, resolution=0.10, minHeight=2.0,
                 boundaryPath=None, verbose=True):
        self.scene = Scene(referencePath, boundaryPath=boundaryPath,
                           resolution=resolution, minHeight=minHeight,
                           restrict=False, verbose=verbose)
        self.readers = {"reference": LayerReader("reference", referencePath,
                                                 self.scene.crs, minHeight,
                                                 verbose)}
        self.minHeight = minHeight
        self.verbose = verbose
        self.results = {}

    def add(self, name, path):
        self.readers[name] = LayerReader(name, path, self.scene.crs,
                                         self.minHeight, self.verbose)
        return self

    def measure(self, tileM=40.0, maxShiftM=4.0):
        scene = self.scene
        tile = scene.metresToPixels(tileM, 32)
        rows, columns = scene.chm.shape
        pairs = list(itertools.combinations(self.readers, 2))
        collected = {pair: [] for pair in pairs}

        for r0 in range(0, rows - tile + 1, tile):
            for c0 in range(0, columns - tile + 1, tile):
                if scene.regionMask is not None and \
                        np.count_nonzero(scene.regionMask[r0:r0 + tile,
                                                          c0:c0 + tile]) \
                        < 0.5 * tile * tile:
                    continue
                surfaces = {name: reader.surface(scene.transform, c0, r0,
                                                 c0 + tile, r0 + tile)
                            for name, reader in self.readers.items()}
                for first, second in pairs:
                    if surfaces[first] is None or surfaces[second] is None:
                        continue
                    measured = nccShift(surfaces[first], surfaces[second],
                                        scene.pixelSize, maxShiftM=maxShiftM)
                    if measured:
                        collected[(first, second)].append(measured)

        self.results = {"%s -> %s" % pair: _summarise(rows)
                        for pair, rows in collected.items()}
        return self.results

    def report(self):
        print()
        print("%-24s %5s %9s %9s %9s %8s"
              % ("pair", "n", "dx(m)", "dy(m)", "dist(m)", "MAD"))
        print("-" * 68)
        for label, summary in self.results.items():
            if summary is None:
                print("%-24s  too few tiles" % label)
                continue
            print("%-24s %5d %+9.3f %+9.3f %9.3f %8.3f"
                  % (label, summary["tiles"], summary["eastM"],
                     summary["southM"], summary["magnitudeM"],
                     summary["madM"]))
        worst = max((s for s in self.results.values() if s),
                    key=lambda s: s["magnitudeM"], default=None)
        print()
        if worst and worst["magnitudeM"] < 0.3:
            print("Every pair agrees to better than 0.3 m.")
        elif worst:
            print("Largest offset %.2f m — check which pair before correcting."
                  % worst["magnitudeM"])
        print("A null result is worth a control: shift one layer by a known "
              "amount and confirm it comes back.")
        return self.results

    def save(self, path):
        with open(path, "w") as handle:
            json.dump(self.results, handle, indent=2, default=float)
        return path


def _summarise(rows):
    if len(rows) < 3:
        return None
    east = np.array([r[0] for r in rows])
    south = np.array([r[1] for r in rows])
    medianEast, medianSouth = float(np.median(east)), float(np.median(south))
    return {"tiles": len(rows), "eastM": medianEast, "southM": medianSouth,
            "magnitudeM": float(np.hypot(medianEast, medianSouth)),
            "madM": max(float(np.median(np.abs(east - medianEast))),
                        float(np.median(np.abs(south - medianSouth)))),
            "withinHalfMetre": float(np.mean(np.hypot(east, south) < 0.5))}
