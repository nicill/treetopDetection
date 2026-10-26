"""
Raster layers of a site: the RGB mosaic and the height models.

A height model knows what it is (a DSM or a CHM) and where it came from (P1
photogrammetry or LiDAR), so a site can hold several and a detector asks for
the one it needs by kind and source instead of by path.

HeightSurface is a height model read for detection: decimated to a working
resolution, cleaned, optionally smoothed, thresholded at a minimum height and
restricted to the scored area. Everything that detects on heights works on one.
"""

import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from scipy.ndimage import gaussian_filter, maximum_filter


HEIGHT_KINDS = ("CHM", "DSM")
LOCAL_HEIGHT_M = 2.0       # tree height: the tallest canopy within this window
SMOOTH_STEP_PX = 0.5       # height-dependent smoothing, in steps of this
MAX_PLAUSIBLE_M = 100.0    # heights above this are read as errors

RESAMPLING = {
    "average": Resampling.average,
    "max": Resampling.max,
    "bilinear": Resampling.bilinear,
    "nearest": Resampling.nearest,
}


def heightSmoothed(chm, smoothM, slope, pixel, windowM=LOCAL_HEIGHT_M):
    """
    The CHM smoothed by a Gaussian whose standard deviation grows with tree
    height: smoothM + slope x h metres, h the tallest canopy within windowM.
    Large crowns, whose branches give several peaks, are smoothed more; small
    trees beside them less. Deviations are rounded to SMOOTH_STEP_PX pixels and
    each pixel takes the CHM smoothed at its own step.
    """
    size = max(1, int(round(windowM / pixel)))
    height = maximum_filter(chm, size=size)
    sigma = (smoothM + slope * height) / pixel
    steps = np.round(sigma / SMOOTH_STEP_PX) * SMOOTH_STEP_PX
    out = chm.copy()
    for step in np.unique(steps):
        if step <= 0:
            continue
        where = steps == step
        out[where] = gaussian_filter(chm, sigma=float(step))[where]
    return out


class Grid(object):
    """A raster grid: affine transform, shape, CRS, and pixel/world maps."""

    def __init__(self, transform, shape, crs):
        self.transform = transform
        self.shape = tuple(shape)
        self.crs = crs
        self.pixelSize = abs(transform.a)

    def toWorld(self, column, row):
        return self.transform @ (column + 0.5, row + 0.5)

    def toPixel(self, east, north):
        column, row = ~self.transform @ (east, north)
        return column, row

    def metresToPixels(self, metres, minimum=1):
        return max(minimum, int(round(metres / self.pixelSize)))

    def mask(self, geometry):
        """True inside `geometry`, on this grid."""
        return rasterize([(geometry, 1)], out_shape=self.shape,
                         transform=self.transform, fill=0,
                         dtype=np.uint8) > 0


class Raster(object):
    """One georeferenced raster file."""

    def __init__(self, path):
        self.path = path
        with rasterio.open(path) as source:
            self.crs = source.crs
            self.nativeGrid = Grid(source.transform,
                                   (source.height, source.width), source.crs)
            self.bandCount = source.count

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, os.path.basename(self.path))

    def read(self, resolution=None, resampling="average", bands=None):
        """
        The raster decimated to `resolution` (no upsampling), as (array, Grid).
        The decimation happens inside GDAL, so a huge raster is never held
        whole. Single-band reads return 2D masked arrays, multi-band 3D.
        """
        with rasterio.open(self.path) as source:
            native = abs(source.transform.a)
            if resolution is None or resolution <= native:
                shape = (source.height, source.width)
            else:
                scale = native / resolution
                shape = (max(1, int(round(source.height * scale))),
                         max(1, int(round(source.width * scale))))
            indexes = bands if bands is not None else 1
            outShape = shape if np.isscalar(indexes) else (len(indexes),) + shape
            data = source.read(indexes, out_shape=outShape, masked=True,
                               resampling=RESAMPLING[resampling])
            transform = source.transform @ source.transform.scale(
                source.width / float(shape[1]), source.height / float(shape[0]))
        grid = Grid(transform, shape, self.crs)
        # keep the exact pixel size the decimation produced (rows decide it)
        grid.pixelSize = native * (self.nativeGrid.shape[0] / float(shape[0]))
        return data, grid


class RGBMosaic(Raster):
    """The orthomosaic."""

    kind = "RGB"
    source = "RGB"


class HeightModel(Raster):
    """A DSM or a CHM, from P1 photogrammetry or LiDAR."""

    def __init__(self, path, kind="CHM", source="unknown"):
        if kind not in HEIGHT_KINDS:
            raise ValueError("kind must be one of %s, not %r"
                             % (HEIGHT_KINDS, kind))
        super().__init__(path)
        self.kind = kind
        self.source = source

    def __repr__(self):
        return "HeightModel(%s %s, %s)" % (self.kind, self.source,
                                          os.path.basename(self.path))

    def surface(self, resolution=0.25, minHeight=2.0, smoothM=0.0,
                smoothSlope=0.0, smoothWindowM=LOCAL_HEIGHT_M,
                resampling="average", region=None):
        """
        The heights as a HeightSurface. NoData, negatives and implausible
        values become 0, so 0 means "no canopy" everywhere downstream. The
        smoothing is applied before the minimum height, and the surface is
        zeroed outside `region` last.
        """
        data, grid = self.read(resolution, resampling)
        heights = np.asarray(data.filled(0.0), dtype=np.float32)
        heights[~np.isfinite(heights)] = 0.0
        heights[heights < 0.0] = 0.0
        heights[heights > MAX_PLAUSIBLE_M] = 0.0
        if smoothSlope > 0:
            heights = heightSmoothed(heights, smoothM, smoothSlope,
                                     grid.pixelSize, smoothWindowM)
        elif smoothM > 0:
            heights = gaussian_filter(heights, sigma=smoothM / grid.pixelSize)
        heights[heights < minHeight] = 0.0
        if region is not None:
            heights[~grid.mask(region)] = 0.0
        return HeightSurface(heights, grid, self)


class HeightSurface(object):
    """
    Heights on a working grid, ready for detection. Exposes the grid's
    coordinate maps directly (chm, transform, pixelSize, toWorld, ...), which is
    all the height detectors use.
    """

    def __init__(self, heights, grid, model=None):
        self.chm = heights
        self.grid = grid
        self.model = model

    def __repr__(self):
        return "HeightSurface(%d x %d px at %.3f m)" % (
            self.chm.shape[1], self.chm.shape[0], self.pixelSize)

    @property
    def shape(self):
        return self.chm.shape

    @property
    def transform(self):
        return self.grid.transform

    @property
    def crs(self):
        return self.grid.crs

    @property
    def pixelSize(self):
        return self.grid.pixelSize

    def toWorld(self, column, row):
        return self.grid.toWorld(column, row)

    def toPixel(self, east, north):
        return self.grid.toPixel(east, north)

    def metresToPixels(self, metres, minimum=1):
        return self.grid.metresToPixels(metres, minimum)

    def heightAt(self, east, north):
        """The surface height under world points (0 outside the grid)."""
        east, north = np.atleast_1d(east), np.atleast_1d(north)
        inverse = ~self.transform
        columns = np.floor(inverse.a * east + inverse.b * north
                           + inverse.c).astype(int)
        rows = np.floor(inverse.d * east + inverse.e * north
                        + inverse.f).astype(int)
        inside = ((rows >= 0) & (rows < self.chm.shape[0])
                  & (columns >= 0) & (columns < self.chm.shape[1]))
        out = np.zeros(east.shape, np.float64)
        out[inside] = self.chm[rows[inside], columns[inside]]
        return out
