"""
Quebec Plantations: a photogrammetric CHM for sites already prepared by
tt.qpPrepare, so every method can run on photogrammetry instead of LiDAR.

    python -m tt.qpPhotogrammetry --prepared "$QP" --output "$QP_P1"

Per site, the dataset's photogrammetric surface model
(Photogrammetry_Products/<site>_p1/<site>_p1_dsm_highdis.cog.tif) is read from
the FRDR server for the site's grid only, at the LiDAR CHM's resolution, from
the overview closest to it. Ground comes from the surface model alone
(tt.groundModel: low points through gaps in the canopy), so the CHM needs no
LiDAR. The surface is read --marginM beyond the site, so the ground at the
site's edge is estimated with a full window around it.

The new CHM is on exactly the LiDAR CHM's grid, 0 outside area.shp, cells
without surface filled from their nearest neighbour (chmCellsFilled). Only the
height model changes: crowns, the scored area and the don't-care areas are
copied from the LiDAR site unchanged, so both CHMs are scored on the same
trees. rgb.tif is linked (relative link, so it works in a synced folder on any
computer).

Per site, in <output>/<site>/: chm.tif, the copied layers, rgb.tif (link),
quicklook.png (LiDAR and photogrammetric CHM with the crowns) and check.json:
the CHM against the field heights (as qpPrepare reports for LiDAR, both given),
crown tops against the LiDAR CHM's, and the scored crowns whose photogrammetric
top is below the site's visibleM. check.json is written last: a site with one
is prepared and is skipped.
"""

import argparse
import json
import os
import shutil
import sys

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

from tt import qpPrepare as qp
from tt.groundModel import GroundModel

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)

SELECTED = ("20230712_afcahoule_itrf20", "20230712_afcamoisan_itrf20",
            "20230606_cbblackburn5", "20230606_cbblackburn6",
            "20230608_cbbernard4")
COPIED = ("area", "scoredArea", "crowns", "ignored", "uncovered", "trees")
SHAPE_PARTS = (".shp", ".shx", ".dbf", ".prj", ".cpg")
BAD_SURFACE_M = -1000.0      # below this a value is an undeclared nodata


def dsmUrl(site):
    return "%sPhotogrammetry_Products/%s_p1/%s_p1_dsm_highdis.cog.tif" % (
        qp.BASE, site, site)


def chmGrid(path, marginM):
    """The LiDAR CHM's grid, and the same grid grown by marginM per side."""
    with rasterio.open(path) as src:
        t, width, height, crs = src.transform, src.width, src.height, src.crs
    resolution = t.a
    cells = int(np.ceil(marginM / resolution))
    grid = {"transform": t, "width": width, "height": height, "crs": crs,
            "resolution": resolution, "margin": cells}
    west, north = t.c - cells * resolution, t.f + cells * resolution
    grid["bounds"] = (west, north - (height + 2 * cells) * resolution,
                      west + (width + 2 * cells) * resolution, north)
    return grid


def readInside(src, bounds, resolution, out):
    """Fill out (on the bounds' grid) where the source has surface."""
    west, south, east, north = bounds
    x0, y0 = max(west, src.bounds.left), max(south, src.bounds.bottom)
    x1, y1 = min(east, src.bounds.right), min(north, src.bounds.top)
    c0 = int(round((x0 - west) / resolution))
    c1 = int(round((x1 - west) / resolution))
    r0 = int(round((north - y1) / resolution))
    r1 = int(round((north - y0) / resolution))
    if c1 <= c0 or r1 <= r0:
        return
    window = from_bounds(west + c0 * resolution, north - r1 * resolution,
                         west + c1 * resolution, north - r0 * resolution,
                         src.transform)
    data = src.read(1, window=window, out_shape=(r1 - r0, c1 - c0),
                    resampling=Resampling.average, masked=True)
    data = data.astype(np.float32).filled(np.nan)
    data[data < BAD_SURFACE_M] = np.nan
    out[r0:r1, c0:c1] = data


def readSurface(source, grid):
    """
    The surface over the grown grid, NaN where there is none. Read from the
    overview closest to the resolution, only where the image is (a boundless
    read would fetch the full-resolution model over HTTPS).
    """
    west, south, east, north = grid["bounds"]
    resolution = grid["resolution"]
    shape = (int(round((north - south) / resolution)),
             int(round((east - west) / resolution)))
    surface = np.full(shape, np.nan, np.float32)
    path = source if not source.startswith("http") else "/vsicurl/" + source
    with rasterio.Env(**qp.GDAL_HTTP):
        with rasterio.open(path) as src:
            level, crs = qp.overviewLevel(src, resolution), src.crs
        if crs != grid["crs"]:
            raise ValueError("surface model in %s, CHM in %s: not the same "
                             "coordinate system" % (crs, grid["crs"]))
        options = {} if level is None else {"overview_level": level}
        with rasterio.open(path, **options) as src:
            readInside(src, grid["bounds"], resolution, surface)
    return surface


def photogrammetricChm(surface, grid, area):
    """Height above the surface's own ground, on the CHM grid; share filled."""
    height = GroundModel(surface, grid["resolution"]).heightAboveGround()
    m = grid["margin"]
    chm = height[m:m + grid["height"], m:m + grid["width"]].copy()
    outside = geometry_mask(area.geometry, chm.shape, grid["transform"])
    chm[~np.isfinite(chm)] = -1.0          # qp.fillEmpty's empty cells
    filled = qp.fillEmpty(chm, outside)
    chm[outside | (chm < 0)] = 0.0
    return chm, filled


def copySite(site, output):
    """The LiDAR site's layers copied, rgb.tif linked."""
    os.makedirs(output, exist_ok=True)
    for stem in COPIED:
        for part in SHAPE_PARTS:
            source = os.path.join(site, stem + part)
            if os.path.exists(source):
                shutil.copyfile(source, os.path.join(output, stem + part))
    source, target = os.path.join(site, "rgb.tif"), os.path.join(output,
                                                                 "rgb.tif")
    if os.path.lexists(target):
        return
    try:
        os.symlink(os.path.relpath(source, output), target)
    except OSError:                       # no links here: a copy instead
        shutil.copyfile(source, target)


def crownComparison(siteDir, chmPath, visibleM):
    """Crown tops, photogrammetric against LiDAR, on the scored crowns."""
    crowns = gpd.read_file(os.path.join(siteDir, "crowns.shp"))
    photo = qp.crownTops(chmPath, crowns)
    lidar = qp.crownTops(os.path.join(siteDir, "chm.tif"), crowns)
    difference = photo - lidar
    return {"crowns": int(len(crowns)),
            "topMinusLidarMedianM": float(np.median(difference)),
            "topMinusLidarMeanAbsM": float(np.mean(np.abs(difference))),
            "topCorrelation": float(np.corrcoef(photo, lidar)[0, 1])
            if len(crowns) > 1 else float("nan"),
            "belowVisibleM": int((photo < visibleM).sum()),
            "visibleM": visibleM}


def fieldComparison(siteDir, chmPath):
    """The CHM against the field heights, as qpPrepare checks the LiDAR."""
    path = os.path.join(siteDir, "trees.shp")
    if not os.path.exists(path):
        return {}, {}
    trees = gpd.read_file(path)
    trees["chmHm"] = qp.chmAtTrees(chmPath, trees)
    return qp.heightCheck(trees, "fieldHm"), qp.heightCheck(trees, "noShootHm")


def readScaled(path, maxPixels):
    with rasterio.open(path) as src:
        scale = max(1.0, max(src.width, src.height) / float(maxPixels))
        data = src.read(1, out_shape=(int(src.height / scale),
                                      int(src.width / scale)),
                        resampling=Resampling.average)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom,
                  src.bounds.top)
    return data, extent


def quicklook(siteDir, chmPath, outputPath, maxPixels=1500):
    crowns = gpd.read_file(os.path.join(siteDir, "crowns.shp"))
    pairs = (("LiDAR", os.path.join(siteDir, "chm.tif")),
             ("photogrammetry", chmPath))
    images = [readScaled(p, maxPixels) for _, p in pairs]
    top = max(float(np.percentile(d[d > 0], 99)) if (d > 0).any() else 1.0
              for d, _ in images)
    figure, axes = plt.subplots(1, 2, figsize=(16, 8))
    for axis, (name, _), (data, extent) in zip(axes, pairs, images):
        shown = axis.imshow(data, extent=extent, vmin=0, vmax=top)
        crowns.boundary.plot(ax=axis, color="white", linewidth=0.3)
        axis.set_title(name)
        axis.set_axis_off()
    figure.colorbar(shown, ax=axes, shrink=0.7, label="height (m)")
    figure.savefig(outputPath, dpi=150, bbox_inches="tight")
    plt.close(figure)


def lidarCheck(siteDir):
    path = os.path.join(siteDir, "check.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def prepareSite(site, args):
    siteDir, outDir = (os.path.join(d, site) for d in (args.prepared,
                                                        args.output))
    copySite(siteDir, outDir)
    area = gpd.read_file(os.path.join(siteDir, "area.shp"))
    grid = chmGrid(os.path.join(siteDir, "chm.tif"), args.marginM)
    source = args.dsmSource(site)
    print("[qpP1] %s: reading the surface model, %.0f x %.0f m at %.2f m"
          % (site, grid["bounds"][2] - grid["bounds"][0],
             grid["bounds"][3] - grid["bounds"][1], grid["resolution"]),
          flush=True)
    chm, filled = photogrammetricChm(readSurface(source, grid), grid, area)
    chmPath = os.path.join(outDir, "chm.tif")
    qp.writeRaster(chmPath, chm[None], grid["transform"], grid["crs"],
                   "float32")
    return finishSite(site, siteDir, outDir, chmPath, source, filled, args)


def finishSite(site, siteDir, outDir, chmPath, source, filled, args):
    lidar = lidarCheck(siteDir)
    heights, noShoot = fieldComparison(siteDir, chmPath)
    quicklook(siteDir, chmPath, os.path.join(outDir, "quicklook.png"))
    check = {"site": site, "heightSource": "photogrammetry",
             "ground": "from the surface model (tt.groundModel)",
             "surfaceModel": source, "marginM": args.marginM,
             "chmCellsFilled": filled,
             "crownTops": crownComparison(siteDir, chmPath,
                                          lidar.get("visibleM", 1.0)),
             "heights": heights, "heightsNoShoot": noShoot,
             "lidarHeights": lidar.get("heights", {}),
             "lidarHeightsNoShoot": lidar.get("heightsNoShoot", {})}
    with open(os.path.join(outDir, "check.json"), "w") as handle:
        json.dump(check, handle, indent=2)
    reportSite(check)
    return check


def reportSite(check):
    site, tops = check["site"], check["crownTops"]
    print("[qpP1] %s: cells filled %.1f%%; crown tops minus LiDAR's median "
          "%+.2f m, mean |diff| %.2f m, r %.2f; %d of %d scored crowns below "
          "%.1f m" % (site, 100 * check["chmCellsFilled"],
                      tops["topMinusLidarMedianM"],
                      tops["topMinusLidarMeanAbsM"], tops["topCorrelation"],
                      tops["belowVisibleM"], tops["crowns"],
                      tops["visibleM"]), flush=True)
    for label, mine, theirs in (("total", "heights", "lidarHeights"),
                                ("no shoot", "heightsNoShoot",
                                 "lidarHeightsNoShoot")):
        a, b = check[mine], check[theirs]
        print("[qpP1] %s: CHM - field %-8s photogrammetry %+.2f m (|diff| "
              "%.2f), LiDAR %+.2f m (|diff| %.2f), %d trees"
              % (site, label, a.get("medianChmMinusFieldM", np.nan),
                 a.get("meanAbsDiffM", np.nan),
                 b.get("medianChmMinusFieldM", np.nan),
                 b.get("meanAbsDiffM", np.nan), a.get("trees", 0)),
              flush=True)


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Quebec Plantations: photogrammetric CHM (ground from the "
                    "surface model) for sites prepared by tt.qpPrepare.")
    parser.add_argument("--prepared", required=True,
                        help="The folder of LiDAR sites written by qpPrepare")
    parser.add_argument("--output", required=True,
                        help="Where the photogrammetric sites go (not the "
                             "--prepared folder)")
    parser.add_argument("--sites", nargs="*", default=list(SELECTED),
                        help="Default: the five selected sites")
    parser.add_argument("--marginM", type=float, default=15.0,
                        help="Surface read this far beyond the site, so the "
                             "ground at its edge has a full window")
    args = parser.parse_args(argv)
    if os.path.abspath(args.output) == os.path.abspath(args.prepared):
        parser.error("--output must differ from --prepared")
    args.dsmSource = dsmUrl
    return args


def main(argv=None):
    args = parseArguments(argv)
    failed = []
    for site in args.sites:
        if os.path.exists(os.path.join(args.output, site, "check.json")):
            print("[qpP1] %s: already prepared, skipped" % site)
            continue
        if not os.path.exists(os.path.join(args.prepared, site, "chm.tif")):
            print("[qpP1] %s: no LiDAR site in %s, skipped" % (site,
                                                               args.prepared))
            failed.append(site)
            continue
        try:
            prepareSite(site, args)
        except (RuntimeError, OSError, ValueError) as error:
            print("[qpP1] %s: FAILED, skipped: %s" % (site, error), flush=True)
            failed.append(site)
    if failed:
        print("[qpP1] %d site(s) not done; run the same command again to "
              "retry them: %s" % (len(failed), " ".join(failed)), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
