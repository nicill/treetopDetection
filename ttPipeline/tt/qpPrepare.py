"""
Quebec Plantations (Lefebvre & Laliberte 2024) as compact per-site products.

    python -m tt.qpPrepare --vectors Data/QuebecPlantations/Dataset/Vector_Data \\
        --pdal ~/anaconda3/envs/pdal/bin/pdal --output Data/qp \\
        --sites 20230609_cbserpentin1

The raw LiDAR (up to 400 million points per site) and the 0.5 cm orthomosaics
are never downloaded: both are cloud-optimised (COPC, COG) and are read from
the FRDR server for the annotated area only. Per site, in <output>/<site>/:

  area.shp     the annotated area: all crowns, buffered by --areaBufferM,
               holes filled. Nothing says where annotation stopped, so ground
               far from every crown is left out rather than scored.
  chm.tif      height above ground from the UAV LiDAR, highest point per cell
               at --chmResolution. Built in tiles by PDAL: statistical outlier
               removal, SMRF ground, height above the Delaunay ground. The
               points carry no classification, so ground is classified here.
  rgb.tif      the orthomosaic over the area at --rgbResolution
  crowns.shp   the hand-made crown polygons
  trees.shp    the field-measured trees; fieldHm is the highest of the three
               total heights, chmHm the CHM's highest cell within 0.3 m
  check.json   CHM against field heights, and the product sizes
  quicklook.png  crowns over the RGB, to judge annotation coverage by eye

PDAL runs as a program (--pdal) so it can live in its own environment.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.windows import from_bounds
from shapely.geometry import Polygon, box

try:                                    # geopandas' newer backend
    from pyogrio import list_layers as _listLayers
except ImportError:                     # older installations: fiona
    from fiona import listlayers as _listLayers

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)

BASE = ("https://g-154be2.cd4fe.0ec8.data.globus.org/13/published/"
        "publication_974/submitted_data/Dataset/")
FIELD_HEIGHTS = ("total_height1_cm", "total_height2_cm", "total_height3_cm")
TREE_RADIUS_M = 0.3


def lidarUrl(site):
    return "%sLiDAR/%s_l1/%s_l1_lidar.copc.laz" % (BASE, site, site)


def rgbUrl(site):
    return "%sPhotogrammetry_Products/%s_p1/%s_p1_rgb.cog.tif" % (BASE, site,
                                                                 site)


def sitesIn(vectorDir):
    return sorted(f[:-len("_p1.gpkg")] for f in os.listdir(vectorDir)
                  if f.endswith("_p1.gpkg"))


def layerNames(gpkgPath):
    names = _listLayers(gpkgPath)
    return [n if isinstance(n, str) else n[0] for n in names]


def readLayers(gpkgPath):
    """(crown polygons, measured tree points) from a site's geopackage."""
    frames = [gpd.read_file(gpkgPath, layer=layer)
              for layer in layerNames(gpkgPath)]
    polygons = [f for f in frames if f.geom_type.str.contains("Polygon").all()]
    points = [f for f in frames if (f.geom_type == "Point").all()]
    if len(polygons) != 1 or len(points) != 1:
        raise ValueError("%s: expected one polygon and one point layer"
                         % gpkgPath)
    return polygons[0], points[0]


def annotatedArea(crowns, bufferM):
    """All crowns buffered and merged, holes filled."""
    merged = crowns.geometry.buffer(bufferM).unary_union
    parts = getattr(merged, "geoms", [merged])
    return gpd.GeoDataFrame({"name": ["area"]}, crs=crowns.crs, geometry=[
        gpd.GeoSeries([Polygon(p.exterior) for p in parts]).unary_union])


def fieldHeight(trees):
    """Highest of the three total heights, in metres; NaN when none."""
    values = trees.reindex(columns=list(FIELD_HEIGHTS))
    values = values.apply(lambda c: pd.to_numeric(c, errors="coerce"))
    return values.max(axis=1, skipna=True) / 100.0


def gridBounds(area, resolution):
    """Area bounds snapped outwards to the CHM grid."""
    west, south, east, north = area.total_bounds
    snap = lambda v, f: f(v / resolution) * resolution
    return (snap(west, np.floor), snap(south, np.floor),
            snap(east, np.ceil), snap(north, np.ceil))


def tiles(bounds, area, sizeM):
    """Tile boxes over the bounds that touch the area."""
    west, south, east, north = bounds
    shape = area.geometry.iloc[0]
    for x in np.arange(west, east, sizeM):
        for y in np.arange(south, north, sizeM):
            tile = (x, y, min(x + sizeM, east), min(y + sizeM, north))
            if box(*tile).intersects(shape):
                yield tile


def pipeline(source, tile, marginM, resolution, output):
    """PDAL stages: read with margin, clean, ground, height, rasterise."""
    x0, y0, x1, y1 = tile
    read = "([%f,%f],[%f,%f])" % (x0 - marginM, x1 + marginM,
                                  y0 - marginM, y1 + marginM)
    write = "([%f,%f],[%f,%f])" % (x0, x1 - resolution, y0, y1 - resolution)
    return [
        {"type": "readers.copc", "filename": source, "bounds": read},
        {"type": "filters.outlier", "method": "statistical",
         "mean_k": 8, "multiplier": 3.0},
        {"type": "filters.range", "limits": "Classification![7:7]"},
        {"type": "filters.smrf"},
        {"type": "filters.hag_delaunay"},
        {"type": "writers.gdal", "filename": output,
         "dimension": "HeightAboveGround", "output_type": "max",
         "resolution": resolution, "bounds": write, "nodata": -9999,
         "gdalopts": "COMPRESS=DEFLATE"}]


def runPdal(pdalPath, stages):
    with tempfile.NamedTemporaryFile("w", suffix=".json",
                                     delete=False) as handle:
        json.dump(stages, handle)
    try:
        subprocess.run([pdalPath, "pipeline", handle.name], check=True,
                       capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("pdal failed: %s" % error.stderr.strip()[-2000:])
    finally:
        os.remove(handle.name)


def buildChm(source, area, args, tileDir, outputPath):
    bounds = gridBounds(area, args.chmResolution)
    parts = []
    for index, tile in enumerate(tiles(bounds, area, args.tileM)):
        part = os.path.join(tileDir, "tile%04d.tif" % index)
        runPdal(args.pdal, pipeline(source, tile, args.marginM,
                                    args.chmResolution, part))
        if not parts:
            checkTileGrid(part, tile, args.chmResolution)
        parts.append(part)
        print("[qp]   tile %d done" % (index + 1), flush=True)
    writeMosaic(parts, bounds, args.chmResolution, area, outputPath)


def checkTileGrid(path, tile, resolution):
    """Warn when PDAL placed a tile's grid off the expected cell edges."""
    with rasterio.open(path) as src:
        offset = max(abs(src.transform.c - tile[0]),
                     abs(src.transform.f - tile[3]))
        size = src.res[0]
    if offset > resolution / 10 or abs(size - resolution) > 1e-9:
        print("[qp] WARNING first tile origin off by %.3f m, cell %.3f m: "
              "the CHM grid is shifted" % (offset, size), flush=True)


def writeMosaic(parts, bounds, resolution, area, outputPath):
    """Tiles merged; no-data and ground outside the area set to 0."""
    sources = [rasterio.open(p) for p in parts]
    try:
        mosaic, transform = merge(sources, bounds=bounds, res=resolution,
                                  nodata=-9999)
        crs = sources[0].crs or area.crs
    finally:
        for s in sources:
            s.close()
    chm = np.where(mosaic[0] < 0, 0.0, mosaic[0]).astype(np.float32)
    outside = geometry_mask(area.geometry, chm.shape, transform)
    chm[outside] = 0.0
    writeRaster(outputPath, chm[None], transform, crs, "float32")


def writeRaster(path, data, transform, crs, dtype):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[1],
                       width=data.shape[2], count=data.shape[0], dtype=dtype,
                       crs=crs, transform=transform,
                       compress="DEFLATE", tiled=True) as destination:
        destination.write(data.astype(dtype))


def readRgb(source, area, resolution, outputPath):
    """The RGB window over the area, resampled; overviews do the reduction."""
    west, south, east, north = gridBounds(area, resolution)
    width = int(round((east - west) / resolution))
    height = int(round((north - south) / resolution))
    path = source if not source.startswith("http") else "/vsicurl/" + source
    with rasterio.open(path) as src:
        window = from_bounds(west, south, east, north, src.transform)
        data = src.read([1, 2, 3], window=window, out_shape=(3, height, width),
                        resampling=Resampling.average, boundless=True,
                        fill_value=0)
        crs = src.crs
    transform = rasterio.transform.from_origin(west, north, resolution,
                                               resolution)
    writeRaster(outputPath, data, transform, crs, "uint8")


def chmAtTrees(chmPath, trees):
    """The CHM's highest cell within TREE_RADIUS_M of each tree."""
    with rasterio.open(chmPath) as src:
        chm = src.read(1)
        reach = int(np.ceil(TREE_RADIUS_M / src.res[0]))
        values = []
        for point in trees.geometry:
            row, column = src.index(point.x, point.y)
            inside = 0 <= row < src.height and 0 <= column < src.width
            patch = chm[max(row - reach, 0):row + reach + 1,
                        max(column - reach, 0):column + reach + 1]
            values.append(float(patch.max()) if inside else np.nan)
    return np.array(values)


def heightCheck(trees):
    """Field against CHM heights, for trees measured and inside the CHM."""
    both = trees[np.isfinite(trees["fieldHm"]) & np.isfinite(trees["chmHm"])]
    if both.empty:
        return {"trees": 0}
    difference = both["chmHm"] - both["fieldHm"]
    return {"trees": int(len(both)),
            "medianFieldHm": float(both["fieldHm"].median()),
            "medianChmMinusFieldM": float(difference.median()),
            "meanAbsDiffM": float(difference.abs().mean()),
            "correlation": float(np.corrcoef(both["fieldHm"],
                                             both["chmHm"])[0, 1])}


def quicklook(rgbPath, crowns, area, outputPath, maxPixels=2000):
    with rasterio.open(rgbPath) as src:
        scale = max(src.width, src.height) / float(maxPixels)
        shape = (3, int(src.height / max(scale, 1)),
                 int(src.width / max(scale, 1)))
        image = src.read(out_shape=shape, resampling=Resampling.average)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom,
                  src.bounds.top)
    figure, axis = plt.subplots(figsize=(10, 10 * shape[1] / shape[2]))
    axis.imshow(np.moveaxis(image, 0, -1), extent=extent)
    crowns.boundary.plot(ax=axis, color="yellow", linewidth=0.4)
    area.boundary.plot(ax=axis, color="red", linewidth=1.0)
    axis.set_axis_off()
    figure.savefig(outputPath, dpi=150, bbox_inches="tight")
    plt.close(figure)


def prepareSite(site, args):
    outDir = os.path.join(args.output, site)
    os.makedirs(outDir, exist_ok=True)
    crowns, trees = readLayers(os.path.join(args.vectors, site + "_p1.gpkg"))
    area = annotatedArea(crowns, args.areaBufferM)
    area.to_file(os.path.join(outDir, "area.shp"))
    crowns[["class_code", "geometry"]].to_file(
        os.path.join(outDir, "crowns.shp"))
    print("[qp] %s: %d crowns, area %.0f m2" % (site, len(crowns),
                                               area.area.iloc[0]), flush=True)
    chmPath, rgbPath = (os.path.join(outDir, n) for n in ("chm.tif", "rgb.tif"))
    with tempfile.TemporaryDirectory(dir=outDir) as tileDir:
        buildChm(args.lidarSource(site), area, args, tileDir, chmPath)
    readRgb(args.rgbSource(site), area, args.rgbResolution, rgbPath)
    return finishSite(site, outDir, crowns, trees, area, chmPath, rgbPath)


def finishSite(site, outDir, crowns, trees, area, chmPath, rgbPath):
    trees = trees[["class_code", "geometry"]].assign(
        fieldHm=fieldHeight(trees).values)
    trees["chmHm"] = chmAtTrees(chmPath, trees)
    trees.to_file(os.path.join(outDir, "trees.shp"))
    quicklook(rgbPath, crowns, area, os.path.join(outDir, "quicklook.png"))
    check = {"site": site, "crowns": int(len(crowns)),
             "areaM2": float(area.area.iloc[0]),
             "medianCrownM2": float(crowns.area.median()),
             "heights": heightCheck(trees),
             "sizesMB": {n: os.path.getsize(os.path.join(outDir, n)) / 1e6
                         for n in ("chm.tif", "rgb.tif")}}
    with open(os.path.join(outDir, "check.json"), "w") as handle:
        json.dump(check, handle, indent=2)
    h = check["heights"]
    print("[qp] %s: CHM-field median %+.2f m, |diff| %.2f m, r %.2f over %d "
          "trees; chm %.0f MB, rgb %.0f MB"
          % (site, h.get("medianChmMinusFieldM", np.nan),
             h.get("meanAbsDiffM", np.nan), h.get("correlation", np.nan),
             h["trees"], check["sizesMB"]["chm.tif"],
             check["sizesMB"]["rgb.tif"]), flush=True)
    return check


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Quebec Plantations: compact CHM, RGB and crowns per "
                    "site, read from the server for the annotated area only.")
    parser.add_argument("--vectors", required=True,
                        help="Folder of the <site>_p1.gpkg files")
    parser.add_argument("--pdal", required=True, help="Path to the pdal program")
    parser.add_argument("--output", default="Data/qp")
    parser.add_argument("--sites", nargs="*", help="Default: every site")
    parser.add_argument("--chmResolution", type=float, default=0.05)
    parser.add_argument("--rgbResolution", type=float, default=0.02)
    parser.add_argument("--areaBufferM", type=float, default=2.0)
    parser.add_argument("--tileM", type=float, default=50.0)
    parser.add_argument("--marginM", type=float, default=5.0)
    args = parser.parse_args(argv)
    args.lidarSource, args.rgbSource = lidarUrl, rgbUrl
    return args


def main(argv=None):
    args = parseArguments(argv)
    sites = args.sites or sitesIn(args.vectors)
    for site in sites:
        if os.path.exists(os.path.join(args.output, site, "check.json")):
            print("[qp] %s: already prepared, skipped" % site)
            continue
        prepareSite(site, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
