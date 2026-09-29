"""
Height above ground from a surface model alone, for sites with no ground model.

    python -m tt.groundModel --dsm ROI_DEM2.tif --output chm.tif

Ground is taken to be the low points the surface model reaches through gaps in
the canopy: the 2nd percentile of the surface in 10 m windows, on a grid of
window centres 5 m apart, filled where a window had too little valid surface,
smoothed over one grid cell, and interpolated back to the full resolution. The
height model is the surface minus that ground, with negative values set to 0.

This measures height above the lowest surface visible nearby, which is ground
only where the photogrammetry saw ground. Two consequences:

  * narrow gaps are smoothed over by photogrammetry, so ground comes out a
    little high and heights a little low;
  * where the canopy is closed across a whole window, the "ground" is lower
    canopy, and heights there are relative to it.

The detector works on relief within its own windows, so both matter less for
finding tops than they would for measuring trees. On the Sergi site a 10 m
window typically spans 7 m of relief with the terrain tilting about 0.6 m
across it, so the low point is usually well below the crowns.

The constants were fixed before the height model was first computed.
"""

import argparse
import sys

import cv2
import numpy as np
import rasterio
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import gaussian_filter

WINDOW_M = 10.0
GROUND_PERCENTILE = 2
STRIDE_M = 5.0            # window centres, half a window apart
MIN_VALID_SHARE = 0.2     # windows with less valid surface give no ground
SMOOTH_CELLS = 1.0        # Gaussian sigma on the grid of window centres
OUTPUT_NODATA = -9999.0


class GroundModel(object):

    def __init__(self, surface, pixelSize):
        self.surface = surface          # float array, NaN where no data
        self.pixelSize = float(pixelSize)
        self.window = max(3, int(round(WINDOW_M / self.pixelSize)))
        self.stride = max(1, int(round(STRIDE_M / self.pixelSize)))

    def lowPoints(self):
        """The ground percentile in each window, on the grid of centres."""
        rows, columns = self.surface.shape
        centreRows = np.arange(0, rows, self.stride)
        centreColumns = np.arange(0, columns, self.stride)
        half = self.window // 2
        grid = np.full((len(centreRows), len(centreColumns)), np.nan)
        for i, r in enumerate(centreRows):
            for j, c in enumerate(centreColumns):
                patch = self.surface[max(0, r - half):r + half,
                                     max(0, c - half):c + half]
                values = patch[np.isfinite(patch)]
                if values.size >= MIN_VALID_SHARE * patch.size:
                    grid[i, j] = np.percentile(values, GROUND_PERCENTILE)
        return grid

    @staticmethod
    def fill(grid):
        """Windows without enough surface take their nearest neighbour's."""
        known = np.isfinite(grid)
        if known.all() or not known.any():
            return grid
        points = np.argwhere(known)
        fill = NearestNDInterpolator(points, grid[known])
        filled = grid.copy()
        filled[~known] = fill(np.argwhere(~known))
        return filled

    def ground(self):
        grid = gaussian_filter(self.fill(self.lowPoints()), SMOOTH_CELLS,
                               mode="nearest")
        rows, columns = self.surface.shape
        # grid cell (i, j) sits at pixel (i*stride, j*stride); stretch it so
        # its first and last centres land on the first and last pixels
        return cv2.resize(grid.astype(np.float32),
                          (grid.shape[1] * self.stride,
                           grid.shape[0] * self.stride),
                          interpolation=cv2.INTER_LINEAR)[:rows, :columns]

    def heightAboveGround(self):
        height = self.surface - self.ground()
        height[height < 0] = 0.0
        height[~np.isfinite(self.surface)] = np.nan
        return height


def readSurface(path):
    with rasterio.open(path) as source:
        data = source.read(1, masked=True).astype(np.float32)
        profile = source.profile
        pixelSize = abs(source.transform.a)
    return data.filled(np.nan), profile, pixelSize


def writeHeight(height, profile, path):
    profile = dict(profile, dtype="float32", nodata=OUTPUT_NODATA, count=1,
                   compress="deflate")
    out = np.where(np.isfinite(height), height, OUTPUT_NODATA)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(out.astype(np.float32), 1)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Height above ground from a surface model alone.")
    parser.add_argument("--dsm", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    surface, profile, pixelSize = readSurface(args.dsm)
    height = GroundModel(surface, pixelSize).heightAboveGround()
    writeHeight(height, profile, args.output)
    valid = height[np.isfinite(height)]
    print("[ground] %s: height above ground %.1f-%.1f m, median %.1f m, "
          "%.1f%% of the surface under 2 m"
          % (args.output, valid.min(), valid.max(), np.median(valid),
             100.0 * np.mean(valid < 2.0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
