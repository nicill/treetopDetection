"""
Scene — a canopy height model, the crowns drawn on it, and the area of interest,
all on one grid.

This is the object that was missing. Ten separate scripts each loaded a CHM,
loaded crowns, reprojected them, repaired invalid geometry, burned a plot
raster and worked out a pixel size. They drifted: some applied the crown shift,
some did not; some restricted to the boundary, some did not. Every stage now
takes a Scene and those decisions are made once, here.
"""

import os

import geopandas as gpd
import numpy as np
from scipy.ndimage import gaussian_filter
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.features import rasterize
from shapely import affinity
from shapely import make_valid


RESAMPLING = {
    "average": Resampling.average,
    "max": Resampling.max,
    "bilinear": Resampling.bilinear,
    "nearest": Resampling.nearest,
}


def readCrowns(path, crs, boundary=None, shiftEastM=0.0, shiftSouthM=0.0):
    """
    Crown polygons, reprojected to `crs`, repaired, optionally shifted and
    clipped to `boundary`. Returns (crowns, number repaired).

    The one crown loader. Scene, Alignment and the benchmark all go through it,
    because three copies had been written and had begun to disagree about
    whether the shift was applied and in which direction.

    The shift is where the crowns SIT relative to the CHM, as the alignment
    tools report it, so the polygons are moved by its negative. East is +x,
    south is -y, hence the sign flip on only one axis.
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
            lambda g: affinity.translate(g, xoff=-shiftEastM,
                                         yoff=shiftSouthM)
            if g is not None and not g.is_empty else g)

    if boundary is not None:
        crowns = crowns[crowns.geometry.centroid.within(boundary)]

    return crowns.reset_index(drop=True), int(invalid.sum())


class Scene(object):

    def __init__(self, chmPath, crownsPath=None, boundaryPath=None,
                 resolution=0.25, minHeight=2.0, resampling="average",
                 crownShiftEast=0.0, crownShiftSouth=0.0,
                 classField="tree_class", restrict=True, verbose=True,
                 smoothM=0.0):
        # smoothM: a Gaussian of this standard deviation (m) over the CHM
        # before the minimum height is applied; 0 leaves it as read
        self.smoothM = float(smoothM)
        self.chmPath = chmPath
        self.crownsPath = crownsPath
        self.boundaryPath = boundaryPath
        self.resolution = float(resolution)
        self.minHeight = float(minHeight)
        self.resampling = resampling
        self.crownShiftEast = float(crownShiftEast)
        self.crownShiftSouth = float(crownShiftSouth)
        self.classField = classField
        self.verbose = verbose

        self.chm = None
        self.transform = None
        self.crs = None
        self.pixelSize = None
        self.crowns = None
        self.boundary = None
        self.regionMask = None

        self._loadChm()
        if boundaryPath:
            self._loadBoundary(restrict=restrict)
        if crownsPath:
            self._loadCrowns()

    # ------------------------------------------------------------------ #

    @property
    def shape(self):
        return self.chm.shape

    def __len__(self):
        return 0 if self.crowns is None else len(self.crowns)

    def __repr__(self):
        return ("Scene(%s, %d x %d px at %.3f m, %d crowns)"
                % (os.path.basename(self.chmPath), self.chm.shape[1],
                   self.chm.shape[0], self.pixelSize, len(self)))

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #

    def _loadChm(self):
        """
        Read the CHM decimated to the working resolution.

        NoData, negatives and implausible values all become 0, so "0" means
        "no canopy here" everywhere downstream. The decimation happens inside
        GDAL, so a 1.5e9 pixel raster is never held whole.
        """
        with rasterio.open(self.chmPath) as source:
            native = abs(source.transform.a)
            if self.resolution <= native:
                shape = (source.height, source.width)
            else:
                scale = native / self.resolution
                shape = (max(1, int(round(source.height * scale))),
                         max(1, int(round(source.width * scale))))
            data = source.read(1, out_shape=shape, masked=True,
                               resampling=RESAMPLING[self.resampling])
            self.transform = source.transform @ source.transform.scale(
                source.width / float(shape[1]),
                source.height / float(shape[0]))
            self.crs = source.crs
            self.pixelSize = native * (source.height / float(shape[0]))

        chm = np.asarray(data.filled(0.0), dtype=np.float32)
        chm[~np.isfinite(chm)] = 0.0
        chm[chm < 0.0] = 0.0
        chm[chm > 100.0] = 0.0
        if self.smoothM > 0:
            chm = gaussian_filter(chm, sigma=self.smoothM / self.pixelSize)
        chm[chm < self.minHeight] = 0.0
        self.chm = chm

        if self.verbose:
            valid = chm[chm > 0]
            print("[scene] %s: %d x %d px at %.4f m, canopy %.1f%%, %.2f-%.2f m"
                  % (os.path.basename(self.chmPath), chm.shape[1],
                     chm.shape[0], self.pixelSize,
                     100.0 * valid.size / chm.size,
                     valid.min() if valid.size else 0,
                     valid.max() if valid.size else 0))

    def _loadBoundary(self, restrict):
        """
        The area of interest, and optionally the CHM zeroed outside it.

        The largest polygon wins: these files often carry a nested clip
        outline alongside the real boundary, and treating the two as separate
        plots double-counts the same ground.
        """

        frame = gpd.read_file(self.boundaryPath)
        if frame.crs is not None and frame.crs != self.crs:
            frame = frame.to_crs(self.crs)
        self.boundary = max(frame.geometry, key=lambda g: g.area)

        self.regionMask = rasterize(
            [(self.boundary, 1)], out_shape=self.chm.shape,
            transform=self.transform, fill=0, dtype=np.uint8) > 0

        if restrict:
            before = int(np.count_nonzero(self.chm))
            self.chm[~self.regionMask] = 0.0
            if self.verbose:
                print("[scene] restricted to the boundary: canopy %d -> %d px"
                      % (before, int(np.count_nonzero(self.chm))))

    def _loadCrowns(self):
        self.crowns, repaired = readCrowns(
            self.crownsPath, self.crs, boundary=self.boundary,
            shiftEastM=self.crownShiftEast, shiftSouthM=self.crownShiftSouth)
        if self.verbose:
            print("[scene] %d crowns, %d repaired, median area %.1f m2"
                  % (len(self.crowns), repaired,
                     self.crowns.geometry.area.median()))

    # ------------------------------------------------------------------ #
    # coordinates
    # ------------------------------------------------------------------ #

    def toWorld(self, column, row):
        return self.transform @ (column + 0.5, row + 0.5)

    def toPixel(self, east, north):
        column, row = ~self.transform @ (east, north)
        return column, row

    def metresToPixels(self, metres, minimum=1):
        return max(minimum, int(round(metres / self.pixelSize)))

    def crownBounds(self):
        return self.crowns.geometry.bounds.to_numpy()

    def heightsIn(self, geometry):
        """The CHM values inside one polygon, as a flat array."""

        minX, minY, maxX, maxY = geometry.bounds
        corners = [self.toPixel(x, y) for x, y in
                   ((minX, minY), (minX, maxY), (maxX, minY), (maxX, maxY))]
        c0 = max(0, int(np.floor(min(c for c, _ in corners))))
        c1 = min(self.chm.shape[1], int(np.ceil(max(c for c, _ in corners))) + 1)
        r0 = max(0, int(np.floor(min(r for _, r in corners))))
        r1 = min(self.chm.shape[0], int(np.ceil(max(r for _, r in corners))) + 1)
        if c1 <= c0 or r1 <= r0:
            return np.empty(0, np.float32), (0, 0)

        window = self.chm[r0:r1, c0:c1]
        windowTransform = self.transform @ self.transform.translation(c0, r0)
        inside = ~geometry_mask([geometry], out_shape=window.shape,
                                transform=windowTransform,
                                all_touched=True, invert=False)
        return np.where(inside, window, 0.0), (c0, r0)

    def crownPeaks(self):
        """
        The highest CHM pixel of every crown, as (x, y, height) plus the peak
        heights. Each polygon is rasterised alone: a shared label raster loses
        the overlaps, and these crowns overlap heavily.
        """
        points, heights = [], []
        for geometry in self.crowns.geometry:
            if geometry is None or geometry.is_empty:
                points.append(None)
                heights.append(0.0)
                continue
            masked, (c0, r0) = self.heightsIn(geometry)
            if masked.size == 0 or masked.max() <= 0:
                points.append(None)
                heights.append(0.0)
                continue
            row, column = np.unravel_index(int(masked.argmax()), masked.shape)
            points.append((int(c0 + column), int(r0 + row)))
            heights.append(float(masked[row, column]))
        return points, np.array(heights)
