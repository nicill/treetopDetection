"""
Koiwainoujo (Sergi) as the same per-site products qpPrepare writes, so the
rest of the pipeline can treat it as one more site.

    python -m tt.sergiPrepare --data Data/sergi --output Data/sergiOut \\
        --pdal ~/anaconda3/envs/pdal/bin/pdal

What differs from the Quebec data, and what this does about it:

  crowns       Shapefiles/CanopyMerge.shp, the hand-made crown polygons (in
               lat/lon, reprojected to the LiDAR's UTM). One class, so no
               'other' crowns.
  area         Shapefiles/LarchROI.shp. Every crown lies inside it. The
               Quebec code builds the area by buffering the crowns instead.
  ground       there is none: no ground model exists, and the floor is barely
               visible in the photogrammetric cloud. The CHM is the highest
               point per cell minus a ground estimate: the lowest point per
               cell, its --groundPercentile over --groundWindowM, smoothed
               (--groundSmoothM). Heights are therefore relative, and too low
               wherever no ground point was seen. ground.tif is written so
               the estimate can be inspected.
  heights      no field heights: no trees.shp, no height checks.
  LiDAR, RGB   local files: the COPC made from the .las (a plain .las has no
               spatial index, so every tile would read all of it) and the
               orthomosaic.

Per site, in <output>/<site>/: area.shp, chm.tif, ground.tif, rgb.tif,
crowns.shp, ignored.shp, scoredArea.shp, uncovered.shp, check.json and
quicklook.png, as described in qpPrepare. A site counts as prepared once
check.json exists; --redo prepares it again.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.merge import merge
from scipy import ndimage

from tt import qpPrepare as qp

NODATA = -9999.0
TOP_PERCENTILES = (5, 25, 50, 75, 95)


def readCrowns(path, crs):
    """The crown polygons in the working CRS, all one class."""
    crowns = gpd.read_file(path).to_crs(crs)
    crowns = crowns[crowns.geometry.notna() & ~crowns.geometry.is_empty]
    return crowns.assign(class_code="tree")[["class_code", "geometry"]] \
        .reset_index(drop=True)


def readArea(path, crs):
    """The annotated area: the ROI polygons merged into one geometry."""
    roi = gpd.read_file(path).to_crs(crs)
    return gpd.GeoDataFrame({"name": ["area"]}, crs=roi.crs,
                            geometry=[roi.geometry.union_all()])


def pipeline(source, tile, marginM, resolution, path, kind):
    """
    PDAL stages: read with margin, drop outliers, write the highest ('max')
    or the lowest ('min') point per cell, as height above sea level, not
    above ground. One writer per pipeline: with two, PDAL accepted the
    pipeline but wrote only the first raster.
    """
    x0, y0, x1, y1 = tile
    read = "([%f,%f],[%f,%f])" % (x0 - marginM, x1 + marginM,
                                  y0 - marginM, y1 + marginM)
    # one cell beyond the tile on every side, as in qpPrepare
    write = "([%f,%f],[%f,%f])" % (x0 - resolution, x1 + resolution,
                                   y0 - resolution, y1 + resolution)
    return [
        {"type": "readers.copc", "filename": source, "bounds": read},
        {"type": "filters.outlier", "method": "statistical",
         "mean_k": 8, "multiplier": 3.0},
        {"type": "filters.range", "limits": "Classification![7:7]"},
        {"type": "writers.gdal", "filename": path, "dimension": "Z",
         "output_type": kind, "resolution": resolution, "bounds": write,
         "nodata": NODATA, "gdalopts": "COMPRESS=DEFLATE"}]


def buildTile(source, tile, args, index, tileDir):
    """The highest and the lowest surface of one tile: two PDAL runs."""
    paths = []
    for kind in ("max", "min"):
        path = os.path.join(tileDir, "%s%04d.tif" % (kind, index))
        qp.runPdal(args.pdal, pipeline(source, tile, args.marginM,
                                       args.chmResolution, path, kind))
        if not os.path.exists(path):
            raise RuntimeError("pdal finished but wrote no %s raster for "
                               "tile %d" % (kind, index))
        paths.append(path)
    return tuple(paths)


def mosaic(paths, bounds, resolution):
    """Tiles merged on the CHM grid: (values, transform, crs)."""
    sources = [rasterio.open(p) for p in paths]
    try:
        data, transform = merge(sources, bounds=bounds, res=resolution,
                                nodata=NODATA)
        crs = sources[0].crs
    finally:
        for s in sources:
            s.close()
    return data[0].astype(np.float32), transform, crs


def buildSurfaces(source, area, args, tileDir):
    """Highest and lowest point per cell over the area, tiles run --jobs at a
    time, each its own PDAL process."""
    bounds = qp.gridBounds(area, args.chmResolution)
    work = list(qp.tiles(bounds, area, args.tileM))
    results = [None] * len(work)
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(buildTile, source, tile, args, i, tileDir): i
                   for i, tile in enumerate(work)}
        for done, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            print("[sergi]   tile %d of %d done, %s, %.1f min per tile"
                  % (done, len(work), time.strftime("%H:%M"),
                     (time.time() - start) / 60 / done), flush=True)
    qp.checkTileGrid(results[0][0], work[0], args.chmResolution)
    high = mosaic([r[0] for r in results], bounds, args.chmResolution)
    low = mosaic([r[1] for r in results], bounds, args.chmResolution)
    return high, low


def nearestFill(values, valid):
    """Cells that are not valid take the value of their nearest valid cell."""
    if not valid.any():
        raise ValueError("no LiDAR points in the area")
    _, (rows, columns) = ndimage.distance_transform_edt(~valid,
                                                        return_indices=True)
    return values[rows, columns]


def groundSurface(low, valid, resolution, windowM, percentile, smoothM):
    """
    Ground estimate from the lowest point per cell: a low percentile over a
    window wider than a crown (so it follows the ground where some is seen
    and ignores the canopy floor where none is), then smoothed.
    """
    filled = nearestFill(low, valid)
    size = max(int(round(windowM / resolution)) // 2 * 2 + 1, 3)
    ground = ndimage.percentile_filter(filled, percentile, size=size,
                                       mode="nearest")
    return ndimage.gaussian_filter(ground, smoothM / resolution,
                                   mode="nearest")


def buildChm(source, area, args, tileDir, chmPath, groundPath):
    """CHM = highest point - ground estimate; 0 outside the area. Returns the
    share of the area's cells that had no point and were filled."""
    (high, transform, crs), (low, _, _) = buildSurfaces(source, area, args,
                                                         tileDir)
    outside = geometry_mask(area.geometry, high.shape, transform)
    validHigh, validLow = high > NODATA / 2, low > NODATA / 2
    empty = ~validHigh & ~outside
    filled = float(empty.sum()) / max(int((~outside).sum()), 1)
    surface = nearestFill(high, validHigh)
    ground = groundSurface(low, validLow, args.chmResolution,
                           args.groundWindowM, args.groundPercentile,
                           args.groundSmoothM)
    chm = np.clip(surface - ground, 0, None).astype(np.float32)
    chm[outside] = 0.0
    qp.writeRaster(chmPath, chm[None], transform, crs, "float32")
    qp.writeRaster(groundPath, ground.astype(np.float32)[None], transform,
                   crs, "float32")
    return filled


def finishSite(site, outDir, crowns, area, chmPath, rgbPath, filled, inside,
               args):
    coverage, uncovered = qp.uncoveredCanopy(chmPath, crowns, area, args)
    tops = qp.crownTops(chmPath, crowns)
    scored, ignored, region, counts = qp.scoredRegion(crowns, area, uncovered,
                                                      tops, args.visibleM)
    qp.writeLayers(outDir, {"uncovered": uncovered,
                            "crowns": scored[["class_code", "geometry"]],
                            "ignored": ignored, "scoredArea": region})
    qp.quicklook(rgbPath, scored, area, ignored,
                 os.path.join(outDir, "quicklook.png"))
    check = {"site": site, "crowns": int(len(scored)),
             "annotatedCrowns": int(len(crowns)),
             "dontCareCrowns": counts["other"],
             "invisibleCrowns": counts["invisible"],
             "visibleM": args.visibleM,
             "crownsInsideArea": inside,
             "areaM2": float(area.area.iloc[0]),
             "scoredAreaM2": float(region.area.iloc[0]),
             "medianCrownM2": float(scored.area.median()),
             "chmCellsFilled": filled, "coverage": coverage,
             "crownTopM": {str(p): float(v) for p, v in
                           zip(TOP_PERCENTILES,
                               np.percentile(tops, TOP_PERCENTILES))},
             "ground": {"windowM": args.groundWindowM,
                        "percentile": args.groundPercentile,
                        "smoothM": args.groundSmoothM},
             "sizesMB": {n: os.path.getsize(os.path.join(outDir, n)) / 1e6
                         for n in ("chm.tif", "ground.tif", "rgb.tif")}}
    with open(os.path.join(outDir, "check.json"), "w") as handle:
        json.dump(check, handle, indent=2)
    reportSite(check)
    return check


def reportSite(check):
    site, c = check["site"], check["coverage"]
    print("[sergi] %s: %d of %d crowns scored, %.0f%% inside the area; "
          "don't-care: %d 'other', %d invisible (CHM top < %.1f m); scored "
          "area %.0f of %.0f m2; CHM cells filled %.1f%%"
          % (site, check["crowns"], check["annotatedCrowns"],
             100 * check["crownsInsideArea"], check["dontCareCrowns"],
             check["invisibleCrowns"], check["visibleM"],
             check["scoredAreaM2"], check["areaM2"],
             100 * check["chmCellsFilled"]), flush=True)
    print("[sergi] %s: uncovered canopy (>= %.1f m) %.1f%% at the edge, "
          "%.1f%% inside (%d interior patches)"
          % (site, c["canopyM"], 100 * c["edgeShare"],
             100 * c["interiorShare"], c["interiorPatches"]), flush=True)
    print("[sergi] %s: crown top above the ground estimate, percentiles %s: "
          "%s m" % (site, "/".join(str(p) for p in TOP_PERCENTILES),
                    "/".join("%.1f" % v for v in check["crownTopM"].values())),
          flush=True)


def prepare(args):
    outDir = os.path.join(args.output, args.site)
    os.makedirs(outDir, exist_ok=True)
    path = lambda name: os.path.join(args.data, name)
    crowns = readCrowns(path(args.crowns), args.crs)
    area = readArea(path(args.roi), args.crs)
    inside = float(crowns.within(area.geometry.iloc[0]).mean())
    area.to_file(os.path.join(outDir, "area.shp"))
    print("[sergi] %s: %d crowns, %.0f%% inside the area, area %.0f m2"
          % (args.site, len(crowns), 100 * inside, area.area.iloc[0]),
          flush=True)
    chmPath, groundPath, rgbPath = (os.path.join(outDir, n) for n in
                                    ("chm.tif", "ground.tif", "rgb.tif"))
    with tempfile.TemporaryDirectory(dir=args.workDir) as tileDir:
        filled = buildChm(path(args.lidar), area, args, tileDir, chmPath,
                          groundPath)
    qp.readRgb(path(args.rgb), area, args.rgbResolution, rgbPath)
    return finishSite(args.site, outDir, crowns, area, chmPath, rgbPath,
                      filled, inside, args)


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Koiwainoujo (Sergi): compact CHM, RGB and crowns in the "
                    "layout qpPrepare writes.")
    parser.add_argument("--pdal", required=True, help="Path to the pdal program")
    parser.add_argument("--data", default="Data/sergi",
                        help="Folder with Shapefiles/, DPC/, Orthomosaic/")
    parser.add_argument("--output", default="Data/sergiOut")
    parser.add_argument("--site", default="koiwainoujo220616")
    parser.add_argument("--crowns", default="Shapefiles/CanopyMerge.shp")
    parser.add_argument("--roi", default="Shapefiles/LarchROI.shp")
    parser.add_argument("--lidar", default="DPC/koiwainoujo.copc.laz")
    parser.add_argument("--rgb", default="Orthomosaic/Koiwainoujo220616NEW.tif")
    parser.add_argument("--crs", default="EPSG:32654",
                        help="The LiDAR's CRS, UTM 54N")
    parser.add_argument("--chmResolution", type=float, default=0.25,
                        help="About 140 points/m2: 5 cm cells would be mostly "
                             "empty")
    parser.add_argument("--rgbResolution", type=float, default=0.02)
    parser.add_argument("--groundWindowM", type=float, default=10.0)
    parser.add_argument("--groundPercentile", type=float, default=5.0)
    parser.add_argument("--groundSmoothM", type=float, default=5.0)
    parser.add_argument("--tileM", type=float, default=50.0)
    parser.add_argument("--marginM", type=float, default=5.0)
    parser.add_argument("--canopyM", type=float, default=1.5,
                        help="Canopy height for the coverage check")
    parser.add_argument("--crownMarginM", type=float, default=0.2,
                        help="Crowns grown by this before the check")
    parser.add_argument("--minPatchM2", type=float, default=0.5,
                        help="Smallest uncovered patch counted")
    parser.add_argument("--visibleM", type=float, default=0.0,
                        help="Crowns whose CHM top is below this are "
                             "invisible: reported, not scored. 0 scores every "
                             "crown; without a ground model there is no "
                             "herb-layer height to set it by")
    parser.add_argument("--jobs", type=int, default=4,
                        help="Tiles processed at once (each a PDAL process)")
    parser.add_argument("--workDir", default=None,
                        help="Where the temporary PDAL tiles go")
    parser.add_argument("--redo", action="store_true",
                        help="Prepare the site again if it already exists")
    return parser.parse_args(argv)


def main(argv=None):
    args = parseArguments(argv)
    done = os.path.exists(os.path.join(args.output, args.site, "check.json"))
    if done and not args.redo:
        print("[sergi] %s: already prepared, skipped (--redo to redo)"
              % args.site)
        return 0
    try:
        prepare(args)
    except (RuntimeError, OSError, ValueError) as error:
        print("[sergi] %s: FAILED: %s" % (args.site, error), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
