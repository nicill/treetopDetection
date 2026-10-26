"""
Annotations of a site: the hand-drawn crowns and the scored area.

readCrowns is the one crown loader in the package: every reader goes through
it, so the reprojection, the repair of invalid geometry, the alignment shift
and the clip to the area are decided once.
"""

import geopandas as gpd
import numpy as np
from shapely import affinity, contains_xy, make_valid


def readCrowns(path, crs, boundary=None, shiftEastM=0.0, shiftSouthM=0.0):
    """
    Crown polygons, reprojected to `crs`, repaired, optionally shifted and
    clipped to `boundary` (by centroid). Returns (crowns, number repaired).

    The shift is where the crowns SIT relative to the height model, as the
    alignment tools report it, so the polygons are moved by its negative.
    East is +x, south is -y, hence the sign flip on only one axis.
    """
    crowns = gpd.read_file(path)
    if crowns.crs is None:
        raise ValueError("%s has no CRS" % path)
    if crs is not None and crowns.crs != crs:
        crowns = crowns.to_crs(crs)

    invalid = ~crowns.geometry.is_valid
    if invalid.any():
        crowns.loc[invalid, "geometry"] = \
            crowns.loc[invalid, "geometry"].apply(make_valid)

    if shiftEastM or shiftSouthM:
        crowns["geometry"] = crowns.geometry.apply(
            lambda g: affinity.translate(g, xoff=-shiftEastM, yoff=shiftSouthM)
            if g is not None and not g.is_empty else g)

    if boundary is not None:
        crowns = crowns[crowns.geometry.centroid.within(boundary)]

    return crowns.reset_index(drop=True), int(invalid.sum())


def readArea(path, crs):
    """
    The scored area: the largest polygon of the file. These files often carry
    a nested clip outline beside the real boundary, and treating the two as
    separate plots double-counts the same ground.
    """
    frame = gpd.read_file(path)
    if frame.crs is not None and crs is not None and frame.crs != crs:
        frame = frame.to_crs(crs)
    return max(frame.geometry, key=lambda g: g.area)


def crownsInRegion(crowns, region):
    """Crowns whose centroid lies in `region`, re-indexed from zero."""
    centroids = crowns.geometry.centroid
    inside = contains_xy(region, centroids.x.to_numpy(), centroids.y.to_numpy())
    return crowns[inside].reset_index(drop=True)


class Annotations(object):
    """The crowns drawn at a site and the area they were drawn over."""

    def __init__(self, crownsPath=None, areaPath=None, crs=None,
                 shiftEastM=0.0, shiftSouthM=0.0):
        self.crownsPath = crownsPath
        self.areaPath = areaPath
        self.crs = crs
        self.shiftEastM = float(shiftEastM)
        self.shiftSouthM = float(shiftSouthM)
        self._area = None
        self._crowns = None
        self.repaired = 0

    def __repr__(self):
        count = "?" if self._crowns is None else len(self._crowns)
        return "Annotations(%s crowns)" % count

    @property
    def area(self):
        if self._area is None and self.areaPath:
            self._area = readArea(self.areaPath, self.crs)
        return self._area

    @property
    def crowns(self):
        """Every crown whose centroid is in the area (all, if no area)."""
        if self._crowns is None:
            if not self.crownsPath:
                raise ValueError("these annotations have no crowns")
            self._crowns, self.repaired = readCrowns(
                self.crownsPath, self.crs, boundary=self.area,
                shiftEastM=self.shiftEastM, shiftSouthM=self.shiftSouthM)
        return self._crowns

    def crownsIn(self, region):
        """The crowns whose centroid lies in `region`."""
        return crownsInRegion(self.crowns, region)

    def tops(self, surface):
        """
        The highest surface pixel of every crown, as world (x, y, height);
        NaN for a crown with no canopy under it. Each polygon is rasterised
        alone: these crowns overlap and a shared label raster loses overlaps.
        """
        from rasterio.features import geometry_mask
        out = np.full((len(self.crowns), 3), np.nan)
        for index, geometry in enumerate(self.crowns.geometry):
            if geometry is None or geometry.is_empty:
                continue
            minX, minY, maxX, maxY = geometry.bounds
            c0, r1 = surface.toPixel(minX, minY)
            c1, r0 = surface.toPixel(maxX, maxY)
            c0, r0 = max(0, int(np.floor(c0))), max(0, int(np.floor(r0)))
            c1 = min(surface.shape[1], int(np.ceil(c1)) + 1)
            r1 = min(surface.shape[0], int(np.ceil(r1)) + 1)
            if c1 <= c0 or r1 <= r0:
                continue
            window = surface.chm[r0:r1, c0:c1]
            transform = surface.transform @ surface.transform.translation(c0, r0)
            inside = ~geometry_mask([geometry], out_shape=window.shape,
                                    transform=transform, all_touched=True)
            masked = np.where(inside, window, 0.0)
            if masked.max() <= 0:
                continue
            row, column = np.unravel_index(int(masked.argmax()), masked.shape)
            east, north = surface.toWorld(c0 + column, r0 + row)
            out[index] = (east, north, masked[row, column])
        return out
