"""
TileRenderer — the diagnostic images.

Rendering lived inside the detector, which is why looking at results required
constructing one. It takes a Scene and whatever you want drawn on it.
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

    def __init__(self, scene, tileM=40.0, minSidePx=900,
                 dotRadiusM=0.6, lineWidthM=0.15):
        self.scene = scene
        self.tileM = float(tileM)
        self.minSidePx = int(minSidePx)
        self.dotRadiusM = float(dotRadiusM)
        self.lineWidthM = float(lineWidthM)

    # ------------------------------------------------------------------ #

    def tiles(self):
        """Window bounds covering the scene, skipping empty ground."""
        size = self.scene.metresToPixels(self.tileM, 64)
        rows, columns = self.scene.chm.shape
        for r0 in range(0, rows, size):
            for c0 in range(0, columns, size):
                r1, c1 = min(r0 + size, rows), min(c0 + size, columns)
                crop = self.scene.chm[r0:r1, c0:c1]
                if np.count_nonzero(crop) >= 0.05 * crop.size:
                    yield c0, r0, c1, r1

    def background(self, c0, r0, c1, r1):
        """The CHM crop as a BGR image, plus the upscaling factor used."""

        crop = self.scene.chm[r0:r1, c0:c1]
        valid = self.scene.chm[self.scene.chm > 0]
        low = self.scene.minHeight
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
        width = max(1, int(round(self.lineWidthM / self.scene.pixelSize
                                 * scale)))
        inverse = ~self.scene.transform
        for index in indices:
            geometry = self.scene.crowns.geometry.iloc[index]
            for ring in pixelRings(geometry, inverse, c0, r0, scale):
                cv2.polylines(image, [ring], True, colour, width)

    def drawTops(self, image, tops, kinds, c0, r0, c1, r1, scale):
        radius = max(2, int(round(self.dotRadiusM / self.scene.pixelSize
                                  * scale)))
        counts = {}
        for index, point in enumerate(tops.points):
            x, y = point[0], point[1]
            if not (c0 <= x < c1 and r0 <= y < r1):
                continue
            kind = kinds[index]
            if kind is None:
                continue
            cv2.circle(image, (int((x - c0) * scale + scale // 2),
                               int((y - r0) * scale + scale // 2)),
                       radius, COLOURS.get(kind, (200, 200, 200)), -1)
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
        bounds = self.scene.crownBounds()
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
        minX, maxY = self.scene.transform @ (c0, r0)
        maxX, minY = self.scene.transform @ (c1, r1)
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
