"""
CHMs from a finer surface model (Sergi's ROI_DEM.tif at Koiwai), and whether
they show more of the annotated trees than the CHM in use. A check made before
any detector is run: if the new CHM shows no more crowns with a peak of their
own, rerunning the detectors is not worth it.

    python -m tt.prepare.dsmChm --site Data/sergiOut/koiwainoujo220616 \\
        --dsm ROI_DEM.tif --output Data/sergiDsm

Per resolution (--resolutions), a site folder <output>/<site>_dsm<res>/ in
the layout the study scripts read: chm.tif and the site's layers (crowns,
areas; rgb.tif linked). The CHM is the surface model's highest value per cell
minus the site's own ground estimate (ground.tif, the one the current CHM was
made with), so the two CHMs differ only by their surface: the point cloud's
highest point per cell against the surface model. Cells with no surface take
their nearest neighbour's; 0 outside the area.

Then, for the current CHM and every new one, per annotated crown (crowns.shp):

  own peak       the crown holds a peak rising at least H m above where it
                 joins higher ground (its dynamic, or prominence), for each H
                 of --heights: a crown without one has no top a height
                 method can find
  several peaks  it holds two or more peaks of at least the largest H: the
                 crown may be split
  top on outline its highest cell lies on its outline: the outline is shifted
                 from the CHM, or the tree leans on a taller one

printed as one table and written to <output>/visibility.json. The measures are
the same for every CHM, so the comparison is fair; they are close to the
paper's site check but not computed by the same code, so compare the rows
with each other, not with the paper's table.
"""

import argparse
import json
import os
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy import ndimage

from . import quebec as qp
from .quebecPhotogrammetry import copySite


def onGrid(path, transform, shape, crs, resampling):
    """path's band 1 on the grid, NaN where it has no data."""
    out = np.full(shape, np.nan, np.float32)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), out, src_transform=src.transform,
                  src_crs=src.crs or crs, src_nodata=src.nodata,
                  dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                  resampling=resampling)
    return out


def buildChm(siteDir, dsmPath, resolution, outputPath):
    """The surface model's CHM at resolution over the site's area; share filled."""
    area = gpd.read_file(os.path.join(siteDir, "area.shp"))
    west, south, east, north = qp.gridBounds(area, resolution)
    shape = (int(round((north - south) / resolution)),
             int(round((east - west) / resolution)))
    transform = from_origin(west, north, resolution, resolution)
    surface = onGrid(dsmPath, transform, shape, area.crs, Resampling.max)
    ground = onGrid(os.path.join(siteDir, "ground.tif"), transform, shape,
                    area.crs, Resampling.bilinear)
    chm = surface - ground
    outside = geometry_mask(area.geometry, shape, transform)
    chm[~np.isfinite(chm)] = -1.0
    chm[np.isfinite(surface) & np.isfinite(ground) & (chm < 0)] = 0.0
    filled = qp.fillEmpty(chm, outside)
    chm[outside | (chm < 0)] = 0.0
    qp.writeRaster(outputPath, chm[None], transform, area.crs, "float32")
    return filled


def peaksAbove(chm, height):
    """Labelled regional maxima whose dynamic (prominence) is >= height."""
    from skimage.morphology import h_maxima
    maxima = h_maxima(chm.astype(np.float64), height).astype(bool)
    labels, count = ndimage.label(maxima, structure=np.ones((3, 3)))
    return labels, count


def crownVisibility(chmPath, crowns, heights):
    """Shares of crowns with an own peak, several peaks, top on outline."""
    with rasterio.open(chmPath) as src:
        chm, transform = src.read(1).astype(np.float32), src.transform
    ids = rasterize(((g, i + 1) for i, g in enumerate(crowns.geometry)),
                    out_shape=chm.shape, transform=transform, fill=0,
                    dtype="int32")
    index = np.arange(1, len(crowns) + 1)
    present = ndimage.sum(np.ones_like(ids), ids, index) > 0
    result = {"resolution": abs(transform.a), "crowns": int(present.sum())}
    for h in heights:
        labels, _ = peaksAbove(chm, h)
        count = peakCounts(ids, labels, len(crowns))
        result["ownPeak%gm" % h] = float(np.mean(count[present] >= 1))
    result["severalPeaks%gm" % max(heights)] = float(
        np.mean(count[present] >= 2))
    result["topOnOutline"] = topOnOutline(chm, ids, index, present)
    return result


def peakCounts(ids, labels, crownCount):
    """How many distinct peaks reach into each crown."""
    both = (ids > 0) & (labels > 0)
    pairs = np.unique(np.stack([ids[both], labels[both]]), axis=1)
    return np.bincount(pairs[0], minlength=crownCount + 1)[1:]


def topOnOutline(chm, ids, index, present):
    """Share of crowns whose highest cell touches another label (the rim)."""
    padded = np.pad(ids, 1, mode="constant")
    interior = np.ones(ids.shape, bool)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        interior &= padded[1 + dr:1 + dr + ids.shape[0],
                           1 + dc:1 + dc + ids.shape[1]] == ids
    tops = ndimage.maximum_position(chm, ids, index)
    rim = [not interior[r, c] for (r, c), p in zip(tops, present) if p]
    return float(np.mean(rim)) if rim else float("nan")


def printTable(rows, heights):
    keys = ["ownPeak%gm" % h for h in heights] + \
        ["severalPeaks%gm" % max(heights), "topOnOutline"]
    print("%-24s %6s %7s %s" % ("CHM", "res m", "crowns",
                                " ".join("%14s" % k for k in keys)))
    for name, r in rows.items():
        print("%-24s %6.2f %7d %s" % (name, r["resolution"], r["crowns"],
                                      " ".join("%13.1f%%" % (100 * r[k])
                                               for k in keys)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site", required=True,
                        help="the prepared site (chm.tif, ground.tif, area.shp, "
                             "crowns.shp)")
    parser.add_argument("--dsm", required=True, help="the finer surface model")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resolutions", default="0.25,0.1,0.05",
                        help="CHM cell sizes to build, m")
    parser.add_argument("--heights", default="0.1,0.5",
                        help="peak prominences checked, m")
    args = parser.parse_args(argv)
    heights = sorted(float(v) for v in args.heights.split(","))
    site = os.path.basename(os.path.normpath(args.site))
    crowns = gpd.read_file(os.path.join(args.site, "crowns.shp"))
    rows = {"current": crownVisibility(os.path.join(args.site, "chm.tif"),
                                       crowns, heights)}
    for res in (float(v) for v in args.resolutions.split(",")):
        outDir = os.path.join(args.output, "%s_dsm%g" % (site, res))
        copySite(args.site, outDir)
        chmPath = os.path.join(outDir, "chm.tif")
        filled = buildChm(args.site, args.dsm, res, chmPath)
        rows["dsm %g m" % res] = dict(crownVisibility(chmPath, crowns, heights),
                                      cellsFilled=filled, site=outDir)
        print("[dsmChm] %s: %.1f%% of cells filled" % (outDir, 100 * filled),
              flush=True)
    printTable(rows, heights)
    with open(os.path.join(args.output, "visibility.json"), "w") as handle:
        json.dump(rows, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
