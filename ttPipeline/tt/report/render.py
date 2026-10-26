"""
TileRenderer — the diagnostic images: a site's height surface in grey, crowns
outlined, detections as dots coloured by what they are. It takes a height
surface and crowns and whatever is to be drawn on it.

TDData.show draws a site with any runs on it through the same renderer.
"""

import os

import cv2
import numpy as np


COLOURS = {"hit": (0, 200, 0),        # green
           "repeat": (0, 140, 255),   # orange
           "canopy": (255, 80, 0),    # blue: outside a crown, canopy height
           "low": (0, 0, 255),        # red: outside a crown, below canopy
           "background": (0, 0, 255)}
MISSED = (0, 255, 255)                # yellow outline
FOUND = (255, 255, 0)                 # cyan outline


class TileRenderer(object):

    def __init__(self, surface, crowns=None, tileM=40.0, minSidePx=900,
                 dotRadiusM=0.6, lineWidthM=0.15):
        """surface: a HeightSurface (drawn in grey); crowns: a GeoDataFrame."""
        self.surface = surface
        self.crowns = crowns
        self.tileM = float(tileM)
        self.minSidePx = int(minSidePx)
        self.dotRadiusM = float(dotRadiusM)
        self.lineWidthM = float(lineWidthM)

    # ------------------------------------------------------------------ #

    def tiles(self):
        """Window bounds covering the scene, skipping empty ground."""
        size = self.surface.metresToPixels(self.tileM, 64)
        rows, columns = self.surface.chm.shape
        for r0 in range(0, rows, size):
            for c0 in range(0, columns, size):
                r1, c1 = min(r0 + size, rows), min(c0 + size, columns)
                crop = self.surface.chm[r0:r1, c0:c1]
                if np.count_nonzero(crop) >= 0.05 * crop.size:
                    yield c0, r0, c1, r1

    def background(self, c0, r0, c1, r1):
        """The CHM crop as a BGR image, plus the upscaling factor used."""

        crop = self.surface.chm[r0:r1, c0:c1]
        valid = self.surface.chm[self.surface.chm > 0]
        low = self.surface.minHeight
        high = float(valid.max()) if valid.size else low + 1.0

        gray = np.zeros(crop.shape, np.uint8)
        canopy = crop > 0
        scaled = np.clip((crop[canopy] - low) / max(high - low, 1e-6), 0, 1)
        gray[canopy] = (1 + 254 * scaled).astype(np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        scale = max(1, int(np.ceil(self.minSidePx / max(r1 - r0, c1 - c0))))
        if scale > 1:
            image = cv2.resize(image, (image.shape[1] * scale,
                                       image.shape[0] * scale),
                               interpolation=cv2.INTER_NEAREST)
        return image, scale

    # ------------------------------------------------------------------ #

    def drawCrowns(self, image, indices, c0, r0, scale, colour):
        width = max(1, int(round(self.lineWidthM / self.surface.pixelSize
                                 * scale)))
        inverse = ~self.surface.transform
        for index in indices:
            geometry = self.crowns.geometry.iloc[index]
            for ring in pixelRings(geometry, inverse, c0, r0, scale):
                cv2.polylines(image, [ring], True, colour, width)

    def drawTops(self, image, detections, kinds, c0, r0, c1, r1, scale,
                 colours=COLOURS):
        """Detections as dots, coloured by kind (None: not drawn)."""
        radius = max(2, int(round(self.dotRadiusM / self.surface.pixelSize
                                  * scale)))
        counts = {}
        for index in range(len(detections)):
            x, y = (int(np.floor(v)) for v in self.surface.toPixel(
                detections.x[index], detections.y[index]))
            if not (c0 <= x < c1 and r0 <= y < r1):
                continue
            kind = kinds[index]
            if kind is None:
                continue
            cv2.circle(image, (int((x - c0) * scale + scale // 2),
                               int((y - r0) * scale + scale // 2)),
                       radius, colours.get(kind, (200, 200, 200)), -1)
            counts[kind] = counts.get(kind, 0) + 1
        return counts

    @staticmethod
    def caption(image, text):
        cv2.rectangle(image, (0, 0), (image.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(image, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return image

    # ------------------------------------------------------------------ #

    def writeTiles(self, outputDir, tops, kinds, missedCrowns=(),
                   foundCrowns=None, label=""):
        """One PNG per tile. Returns the paths written."""

        os.makedirs(outputDir, exist_ok=True)
        bounds = self.crowns.geometry.bounds.to_numpy()
        missed = set(missedCrowns)
        written = []

        for c0, r0, c1, r1 in self.tiles():
            image, scale = self.background(c0, r0, c1, r1)
            nearby = self._crownsNear(bounds, c0, r0, c1, r1)

            self.drawCrowns(image, [i for i in nearby if i in missed],
                            c0, r0, scale, MISSED)
            if foundCrowns is not None:
                self.drawCrowns(image,
                                [i for i in nearby if i in set(foundCrowns)],
                                c0, r0, scale, FOUND)

            counts = self.drawTops(image, tops, kinds, c0, r0, c1, r1, scale)
            summary = "  ".join("%s %d" % (k, v) for k, v in
                                sorted(counts.items()))
            self.caption(image, "%s%s" % (label + "  " if label else "",
                                          summary))

            path = os.path.join(outputDir, "tile_r%04d_c%04d.png" % (r0, c0))
            cv2.imwrite(path, image)
            written.append(path)
        return written

    def _crownsNear(self, bounds, c0, r0, c1, r1):
        minX, maxY = self.surface.transform @ (c0, r0)
        maxX, minY = self.surface.transform @ (c1, r1)
        return np.nonzero((bounds[:, 0] <= maxX) & (bounds[:, 2] >= minX)
                          & (bounds[:, 1] <= maxY)
                          & (bounds[:, 3] >= minY))[0].tolist()


def pixelRings(geometry, inverse, c0, r0, scale):
    """Polygon rings as integer pixel arrays, holes included."""
    parts = getattr(geometry, "geoms", None)
    polygons = list(parts) if parts is not None else [geometry]
    rings = []
    for polygon in polygons:
        exterior = getattr(polygon, "exterior", None)
        if exterior is None:
            continue
        for coordinates in [exterior.coords] + [h.coords
                                                for h in polygon.interiors]:
            points = [[(inverse @ (x, y))[0] - c0, (inverse @ (x, y))[1] - r0]
                      for x, y in coordinates]
            array = np.round(np.asarray(points) * scale).astype(np.int32)
            rings.append(array)
    return rings


RUN_COLOURS = [(0, 200, 0), (255, 80, 0), (0, 140, 255), (255, 0, 255),
               (0, 220, 220), (128, 0, 255)]


def drawSite(surface, crowns, runs, path, window=None, minSidePx=1200):
    """
    One image of a site: the surface in grey, crowns outlined in cyan, and
    each run's detections as dots in its own colour. window: (c0, r0, c1, r1)
    pixels of the surface (all of it by default).
    """
    renderer = TileRenderer(surface, crowns, minSidePx=minSidePx)
    rows, columns = surface.chm.shape
    c0, r0, c1, r1 = window or (0, 0, columns, rows)
    image, scale = renderer.background(c0, r0, c1, r1)
    if crowns is not None:
        renderer.drawCrowns(image, range(len(crowns)), c0, r0, scale,
                            (255, 255, 0))
    legend = []
    for index, (name, detections) in enumerate(runs.items()):
        colour = RUN_COLOURS[index % len(RUN_COLOURS)]
        renderer.drawTops(image, detections, ["run"] * len(detections),
                          c0, r0, c1, r1, scale, {"run": colour})
        legend.append("%s %d" % (name, len(detections)))
    renderer.caption(image, "  ".join(legend) or "no detections")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    cv2.imwrite(path, image)
    return path
