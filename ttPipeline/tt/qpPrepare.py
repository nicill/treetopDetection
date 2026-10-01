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
               Cells without a point take their nearest filled cell; the
               share is chmCellsFilled in check.json.
  rgb.tif      the orthomosaic over the area at --rgbResolution
  crowns.shp   the hand-made crown polygons to score: all but 'other' and
               the invisible ones
  ignored.shp  don't-care areas: 'other' crowns (thickets drawn as one
               polygon), invisible crowns (CHM top below --visibleM: trees
               no taller than the herb layer, counted in check.json and
               reported separately) and uncovered canopy (below); reason
               says which
  scoredArea.shp  area.shp minus ignored.shp: the boundary to score in
  trees.shp    the field-measured trees; fieldHm is the highest of the three
               total heights, noShootHm the same without this year's shoot
               (the fair one when the flight was earlier in the season),
               chmHm the CHM's highest cell within 0.3 m
  uncovered.shp  canopy (CHM >= --canopyM) in the area that no crown covers,
               as patches of at least --minPatchM2: unannotated trees, or
               tall shrubs and herbs. edge marks patches on the area's
               boundary, where the buffer reaches into vegetation nobody
               meant to annotate; interior ones mean missing annotations.
  check.json   CHM against field heights, the uncovered share of the area's
               canopy, and the product sizes
  quicklook.png  scored crowns (yellow), area (red), uncovered canopy (cyan)
               'other' crowns (magenta) and invisible crowns (orange)

--recheck redoes the checks for sites already prepared, from their saved
chm.tif and rgb.tif, without reading anything from the server.

PDAL runs as a program (--pdal) so it can live in its own environment.
A site counts as prepared once its check.json exists, which is written last,
so --output can be a synced folder (Dropbox) and a run interrupted on one
computer can be continued on another: finished sites are skipped.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, shapes
from rasterio.merge import merge
from rasterio.windows import Window, from_bounds
from scipy.ndimage import distance_transform_edt, label
from shapely.geometry import Polygon, box, shape

try:                                    # geopandas' newer backend
    from pyogrio import list_layers as _listLayers
except ImportError:                     # older installations: fiona
    from fiona import listlayers as _listLayers

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)

BASE = ("https://g-154be2.cd4fe.0ec8.data.globus.org/13/published/"
        "publication_974/submitted_data/Dataset/")
FIELD_HEIGHTS = ("total_height1_cm", "total_height2_cm", "total_height3_cm")
NO_SHOOT_HEIGHTS = ("height1_no_shoot_cm", "height2_no_shoot_cm",
                    "height3_no_shoot_cm")
DONT_CARE_CLASS = "other"
EDGE_M = 0.5
TREE_RADIUS_M = 0.3
# GDAL over HTTPS: merge neighbouring range requests, cache, and let a
# stalled request time out and be retried rather than hang
GDAL_HTTP = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
             "GDAL_HTTP_MULTIRANGE": "YES",
             "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
             "VSI_CACHE": "TRUE", "VSI_CACHE_SIZE": str(512 * 1024 * 1024),
             "GDAL_HTTP_TIMEOUT": "120", "GDAL_HTTP_MAX_RETRY": "5",
             "GDAL_HTTP_RETRY_DELAY": "10"}


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
    merged = crowns.geometry.buffer(bufferM).union_all()
    parts = getattr(merged, "geoms", [merged])
    return gpd.GeoDataFrame({"name": ["area"]}, crs=crowns.crs, geometry=[
        gpd.GeoSeries([Polygon(p.exterior) for p in parts]).union_all()])


def fieldHeight(trees, columns=FIELD_HEIGHTS):
    """
    Highest of the three measured heights, in metres; NaN when none. Zero
    or negative values are placeholders (trees measured once carry 0 in the
    no-shoot fields), so they count as missing.
    """
    values = trees.reindex(columns=list(columns))
    values = values.apply(lambda c: pd.to_numeric(c, errors="coerce"))
    return values.where(values > 0).max(axis=1, skipna=True) / 100.0


def measurementDays(trees):
    """Each tree's measurement day; the column is spelt differently."""
    columns = [c for c in trees.columns if "date" in c.lower()]
    if not columns:
        return pd.Series(["unknown"] * len(trees), index=trees.index)
    days = pd.to_datetime(trees[columns[0]], errors="coerce", utc=True)
    return days.dt.strftime("%Y-%m-%d").fillna("unknown")


def measurementDates(trees):
    return {str(k): int(v) for k, v in
            measurementDays(trees).value_counts().items()}


def isDontCare(crowns):
    return crowns["class_code"].astype(str).str.lower().eq(DONT_CARE_CLASS)


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
    # one cell beyond the tile on every side: PDAL may start its grid a whole
    # cell off, and the overlap guarantees no gap where tiles meet (the
    # overlapping cells come from the same points, so they agree)
    write = "([%f,%f],[%f,%f])" % (x0 - resolution, x1 + resolution,
                                   y0 - resolution, y1 + resolution)
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


def runPdal(pdalPath, stages, timeoutS=900):
    with tempfile.NamedTemporaryFile("w", suffix=".json",
                                     delete=False) as handle:
        json.dump(stages, handle)
    try:
        subprocess.run([pdalPath, "pipeline", handle.name], check=True,
                       capture_output=True, text=True, timeout=timeoutS)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("pdal failed: %s" % error.stderr.strip()[-2000:])
    except subprocess.TimeoutExpired:
        raise RuntimeError("pdal timed out after %d s" % timeoutS)
    finally:
        os.remove(handle.name)



def buildTile(source, tile, args, part):
    """One tile, retried: reading over HTTPS fails now and then."""
    stages = pipeline(source, tile, args.marginM, args.chmResolution, part)
    for attempt in range(1, args.retries + 2):
        try:
            runPdal(args.pdal, stages, args.tileTimeoutMin * 60)
            return part
        except RuntimeError as error:
            if attempt > args.retries:
                raise
            lines = [l for l in str(error).splitlines() if l.strip()]
            print("[qp]   tile failed (attempt %d), retrying in %d s: %s"
                  % (attempt, args.retryWaitS * attempt,
                     (lines[-1] if lines else "no message")[:200]),
                  flush=True)
            time.sleep(args.retryWaitS * attempt)


def buildChm(source, area, args, tileDir, outputPath):
    """
    The tiles run --jobs at a time, each its own PDAL process; the CHM is
    the same whatever the number, only the order they finish in changes.
    """
    bounds = gridBounds(area, args.chmResolution)
    work = list(tiles(bounds, area, args.tileM))
    parts = [os.path.join(tileDir, "tile%04d.tif" % i) for i in range(len(work))]
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(buildTile, source, tile, args, part): tile
                   for tile, part in zip(work, parts)}
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            print("[qp]   tile %d of %d done, %s, %.1f min per tile"
                  % (done, len(work), time.strftime("%H:%M"),
                     (time.time() - start) / 60 / done), flush=True)
    checkTileGrid(parts[0], work[0], args.chmResolution)
    return writeMosaic(parts, bounds, args.chmResolution, area, outputPath)


def latticeOffset(value, reference, resolution):
    """Distance of value from the reference lattice, within half a cell."""
    remainder = (value - reference) % resolution
    return min(remainder, resolution - remainder)


def checkTileGrid(path, tile, resolution):
    """
    Warn when PDAL's grid is off the CHM lattice by a fraction of a cell,
    which would shift the data. Whole-cell offsets only change the extent.
    """
    with rasterio.open(path) as src:
        offset = max(latticeOffset(src.transform.c, tile[0], resolution),
                     latticeOffset(src.transform.f, tile[3], resolution))
        size = src.res[0]
    if offset > resolution / 10 or abs(size - resolution) > 1e-9:
        print("[qp] WARNING tile grid off the CHM lattice by %.3f m, cell "
              "%.3f m: the CHM would be shifted" % (offset, size), flush=True)


def fillEmpty(chm, outside):
    """
    Cells without a point take their nearest cell with one. At 5 cm and
    about 1,700 points per m2 some 1-2% of cells get no point; left at 0 each
    would be a pit inside a crown. Returns the share of the area filled.
    """
    empty = (chm < 0) & ~outside
    if (chm >= 0).any():
        _, (rows, columns) = distance_transform_edt(chm < 0,
                                                    return_indices=True)
        chm[empty] = chm[rows, columns][empty]
    return float(empty.sum()) / max(int((~outside).sum()), 1)


def writeMosaic(parts, bounds, resolution, area, outputPath):
    """Tiles merged, empty cells filled, everything outside the area 0."""
    sources = [rasterio.open(p) for p in parts]
    try:
        mosaic, transform = merge(sources, bounds=bounds, res=resolution,
                                  nodata=-9999)
        crs = sources[0].crs or area.crs
    finally:
        for s in sources:
            s.close()
    chm = mosaic[0].astype(np.float32)
    outside = geometry_mask(area.geometry, chm.shape, transform)
    filled = fillEmpty(chm, outside)
    chm[outside | (chm < 0)] = 0.0
    writeRaster(outputPath, chm[None], transform, crs, "float32")
    return filled


def writeRaster(path, data, transform, crs, dtype):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[1],
                       width=data.shape[2], count=data.shape[0], dtype=dtype,
                       crs=crs, transform=transform,
                       compress="DEFLATE", tiled=True) as destination:
        destination.write(data.astype(dtype))


def overviewLevel(src, resolution):
    """Index of the coarsest overview still at least as fine as resolution."""
    factors = src.overviews(1)
    fine = [i for i, f in enumerate(factors)
            if src.res[0] * f <= resolution * 1.0001]
    return fine[-1] if fine else None


def readRgb(source, area, resolution, outputPath):
    """
    The RGB over the area at resolution, read from the overview closest to
    it and only where the image is; the rest of the grid stays 0. A
    boundless read would bypass the overviews and fetch the full 0.5 cm
    image block by block, which over HTTPS takes hours.
    """
    west, south, east, north = gridBounds(area, resolution)
    width = int(round((east - west) / resolution))
    height = int(round((north - south) / resolution))
    data = np.zeros((3, height, width), dtype=np.uint8)
    path = source if not source.startswith("http") else "/vsicurl/" + source
    with rasterio.Env(**GDAL_HTTP):
        with rasterio.open(path) as src:
            level, crs = overviewLevel(src, resolution), src.crs
        options = {} if level is None else {"overview_level": level}
        with rasterio.open(path, **options) as src:
            readInside(src, (west, south, east, north), resolution, data)
    transform = rasterio.transform.from_origin(west, north, resolution,
                                               resolution)
    writeRaster(outputPath, data, transform, crs, "uint8")


def readInside(src, bounds, resolution, data):
    """Fill data (on the bounds' grid) where the source image has pixels."""
    west, south, east, north = bounds
    x0, y0 = max(west, src.bounds.left), max(south, src.bounds.bottom)
    x1, y1 = min(east, src.bounds.right), min(north, src.bounds.top)
    c0, c1 = int(round((x0 - west) / resolution)), int(round((x1 - west) /
                                                             resolution))
    r0, r1 = int(round((north - y1) / resolution)), int(round((north - y0) /
                                                              resolution))
    if c1 <= c0 or r1 <= r0:
        return
    window = from_bounds(west + c0 * resolution, north - r1 * resolution,
                         west + c1 * resolution, north - r0 * resolution,
                         src.transform)
    data[:, r0:r1, c0:c1] = src.read([1, 2, 3], window=window,
                                     out_shape=(3, r1 - r0, c1 - c0),
                                     resampling=Resampling.average)


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


def heightCheck(trees, column="fieldHm"):
    """Field against CHM heights, for trees measured and inside the CHM."""
    both = trees[np.isfinite(trees[column]) & np.isfinite(trees["chmHm"])]
    if both.empty:
        return {"trees": 0}
    difference = both["chmHm"] - both[column]
    spread = len(both) > 1 and both[column].std() > 0 \
        and both["chmHm"].std() > 0
    return {"trees": int(len(both)),
            "medianFieldHm": float(both[column].median()),
            "medianChmMinusFieldM": float(difference.median()),
            "meanAbsDiffM": float(difference.abs().mean()),
            "correlation": float(np.corrcoef(both[column], both["chmHm"])[0, 1])
            if spread else float("nan")}


def heightsByDay(table):
    """The total-height check per measurement day: growth after the flight
    shows as a bias that grows with the day."""
    return {day: heightCheck(group, "fieldHm")
            for day, group in table.groupby("day")}


def uncoveredCanopy(chmPath, crowns, area, args):
    """
    Canopy in the area outside every crown (grown by --crownMarginM), as
    patches of at least --minPatchM2. Returns (summary, patch polygons).
    """
    with rasterio.open(chmPath) as src:
        chm, transform, cell = src.read(1), src.transform, src.res[0] ** 2
    inArea = ~geometry_mask(area.geometry, chm.shape, transform)
    grown = crowns.geometry.buffer(args.crownMarginM)
    inCrowns = ~geometry_mask(grown, chm.shape, transform)
    canopy = (chm >= args.canopyM) & inArea
    labels, count = label(canopy & ~inCrowns, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel(), minlength=count + 1) * cell
    keep = np.flatnonzero(sizes >= args.minPatchM2)
    keep = keep[keep > 0]
    patches = np.isin(labels, keep)
    polygons = [shape(g) for g, v in shapes(labels.astype(np.int32),
                                            mask=patches, transform=transform)]
    edge = area.geometry.iloc[0].boundary
    frame = gpd.GeoDataFrame(
        {"areaM2": [p.area for p in polygons],
         "edge": [p.distance(edge) < EDGE_M for p in polygons]},
        geometry=polygons, crs=crowns.crs)
    canopyM2 = max(float(canopy.sum() * cell), 1e-9)
    interiorM2 = float(frame.loc[~frame["edge"], "areaM2"].sum())
    edgeM2 = float(frame.loc[frame["edge"], "areaM2"].sum())
    summary = {"canopyM": args.canopyM, "canopyM2": canopyM2,
               "uncoveredShare": (interiorM2 + edgeM2) / canopyM2,
               "edgeShare": edgeM2 / canopyM2,
               "interiorShare": interiorM2 / canopyM2,
               "patches": int(len(frame)),
               "interiorPatches": int((~frame["edge"]).sum())}
    return summary, frame


def crownTops(chmPath, crowns):
    """Highest CHM cell inside each crown (0 when it holds no cell)."""
    with rasterio.open(chmPath) as src:
        chm, transform = src.read(1), src.transform
    tops = []
    for polygon in crowns.geometry:
        window = from_bounds(*polygon.bounds, transform).round_offsets() \
            .round_lengths()
        row0, col0 = max(int(window.row_off), 0), max(int(window.col_off), 0)
        rows = slice(row0, max(int(window.row_off + window.height) + 1, row0))
        cols = slice(col0, max(int(window.col_off + window.width) + 1, col0))
        patch = chm[rows, cols]
        if patch.size == 0:
            tops.append(0.0)
            continue
        inside = ~geometry_mask([polygon], patch.shape, rasterio.windows
                                .transform(Window(col0, row0, patch.shape[1],
                                                  patch.shape[0]), transform))
        tops.append(float(patch[inside].max()) if inside.any() else 0.0)
    return np.array(tops)


def scoredRegion(crowns, area, uncovered, tops, visibleM):
    """
    Crowns to score and the area to score them in. Don't-care, cut out of
    the area so a detection there counts neither as a hit nor as a false
    positive: 'other' thickets, uncovered canopy, and invisible crowns,
    whose highest CHM cell is below visibleM (trees no taller than the herb
    layer the LiDAR takes for ground). visibleM is fixed for every setting
    tried, so the set of scored trees never moves with the tuning.
    """
    other = isDontCare(crowns).values
    invisible = ~other & (tops < visibleM)
    parts = (("other", crowns.geometry[other]),
             ("invisible", crowns.geometry[invisible]),
             ("uncovered", uncovered.geometry))
    ignored = gpd.GeoDataFrame(
        {"reason": [r for r, g in parts for _ in range(len(g))]},
        geometry=[p for _, g in parts for p in g], crs=crowns.crs)
    shape = area.geometry.iloc[0]
    if len(ignored):
        shape = shape.difference(ignored.geometry.union_all())
    region = gpd.GeoDataFrame({"name": ["scored"]}, geometry=[shape],
                              crs=crowns.crs)
    counts = {"other": int(other.sum()), "invisible": int(invisible.sum())}
    return crowns[~other & ~invisible], ignored, region, counts


def quicklook(rgbPath, crowns, area, ignored, outputPath, maxPixels=2000):
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
    for reason, colour in (("uncovered", "cyan"), ("other", "magenta"),
                           ("invisible", "orange")):
        part = ignored[ignored["reason"] == reason]
        if len(part):
            part.boundary.plot(ax=axis, color=colour, linewidth=0.6)
    axis.set_axis_off()
    figure.savefig(outputPath, dpi=150, bbox_inches="tight")
    plt.close(figure)


def prepareSite(site, args):
    outDir = os.path.join(args.output, site)
    os.makedirs(outDir, exist_ok=True)
    crowns, trees = readLayers(os.path.join(args.vectors, site + "_p1.gpkg"))
    area = annotatedArea(crowns, args.areaBufferM)
    area.to_file(os.path.join(outDir, "area.shp"))
    print("[qp] %s: %d crowns, area %.0f m2" % (site, len(crowns),
                                               area.area.iloc[0]), flush=True)
    chmPath, rgbPath = (os.path.join(outDir, n) for n in ("chm.tif", "rgb.tif"))
    with tempfile.TemporaryDirectory(dir=args.workDir) as tileDir:
        filled = buildChm(args.lidarSource(site), area, args, tileDir, chmPath)
    readRgb(args.rgbSource(site), area, args.rgbResolution, rgbPath)
    return finishSite(site, outDir, crowns, trees, area, chmPath, rgbPath,
                      filled, args)


def removeShapefile(path):
    """A shapefile and its side files, if present (a recheck may find none)."""
    stem = os.path.splitext(path)[0]
    for extension in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        if os.path.exists(stem + extension):
            os.remove(stem + extension)


def recheckSite(site, args):
    """The checks again, from a prepared site's saved products."""
    outDir = os.path.join(args.output, site)
    crowns, trees = readLayers(os.path.join(args.vectors, site + "_p1.gpkg"))
    area = gpd.read_file(os.path.join(outDir, "area.shp"))
    with open(os.path.join(outDir, "check.json")) as handle:
        filled = json.load(handle).get("chmCellsFilled", float("nan"))
    return finishSite(site, outDir, crowns, trees, area,
                      os.path.join(outDir, "chm.tif"),
                      os.path.join(outDir, "rgb.tif"), filled, args)


def treeTable(trees, chmPath):
    table = trees[["class_code", "geometry"]].assign(
        fieldHm=fieldHeight(trees).values,
        noShootHm=fieldHeight(trees, NO_SHOOT_HEIGHTS).values,
        day=measurementDays(trees).values)
    table["chmHm"] = chmAtTrees(chmPath, table)
    return table


def writeLayers(outDir, layers):
    """Shapefiles written afresh; an empty layer leaves no stale file."""
    for name, frame in layers.items():
        path = os.path.join(outDir, name + ".shp")
        removeShapefile(path)
        if len(frame):
            frame.to_file(path)


def finishSite(site, outDir, crowns, trees, area, chmPath, rgbPath, filled,
               args):
    table = treeTable(trees, chmPath)
    coverage, uncovered = uncoveredCanopy(chmPath, crowns, area, args)
    tops = crownTops(chmPath, crowns)
    scored, ignored, region, counts = scoredRegion(crowns, area, uncovered,
                                                   tops, args.visibleM)
    writeLayers(outDir, {"trees": table, "uncovered": uncovered,
                         "crowns": scored[["class_code", "geometry"]],
                         "ignored": ignored, "scoredArea": region})
    quicklook(rgbPath, scored, area, ignored,
              os.path.join(outDir, "quicklook.png"))
    check = {"site": site, "crowns": int(len(scored)),
             "annotatedCrowns": int(len(crowns)),
             "dontCareCrowns": counts["other"],
             "invisibleCrowns": counts["invisible"],
             "visibleM": args.visibleM,
             "areaM2": float(area.area.iloc[0]),
             "scoredAreaM2": float(region.area.iloc[0]),
             "medianCrownM2": float(scored.area.median()),
             "chmCellsFilled": filled, "coverage": coverage,
             "measured": measurementDates(trees),
             "heights": heightCheck(table, "fieldHm"),
             "heightsNoShoot": heightCheck(table, "noShootHm"),
             "heightsByDay": heightsByDay(table),
             "sizesMB": {n: os.path.getsize(os.path.join(outDir, n)) / 1e6
                         for n in ("chm.tif", "rgb.tif")}}
    with open(os.path.join(outDir, "check.json"), "w") as handle:
        json.dump(check, handle, indent=2)
    reportSite(check)
    return check


def reportSite(check):
    site, c = check["site"], check["coverage"]
    print("[qp] %s: %d of %d crowns scored; don't-care: %d 'other', %d "
          "invisible (CHM top < %.1f m, %.1f%%); scored area %.0f of %.0f m2; "
          "CHM cells filled %.1f%%"
          % (site, check["crowns"], check["annotatedCrowns"],
             check["dontCareCrowns"], check["invisibleCrowns"],
             check["visibleM"], 100.0 * check["invisibleCrowns"] /
             max(check["annotatedCrowns"], 1), check["scoredAreaM2"],
             check["areaM2"], 100 * check["chmCellsFilled"]), flush=True)
    print("[qp] %s: uncovered canopy (>= %.1f m) %.1f%% at the edge, %.1f%% "
          "inside (%d interior patches)"
          % (site, c["canopyM"], 100 * c["edgeShare"],
             100 * c["interiorShare"], c["interiorPatches"]), flush=True)
    for label, key in (("total", "heights"), ("no shoot", "heightsNoShoot")):
        h = check[key]
        print("[qp] %s: CHM - field %-8s median %+.2f m, |diff| %.2f m, "
              "r %.2f over %d trees"
              % (site, label, h.get("medianChmMinusFieldM", np.nan),
                 h.get("meanAbsDiffM", np.nan), h.get("correlation", np.nan),
                 h["trees"]), flush=True)
    for day, h in check["heightsByDay"].items():
        print("[qp] %s:   measured %s: %3d trees, CHM - field median %+.2f m"
              % (site, day, h["trees"], h.get("medianChmMinusFieldM", np.nan)),
              flush=True)


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
    parser.add_argument("--canopyM", type=float, default=1.5,
                        help="Canopy height for the coverage check")
    parser.add_argument("--crownMarginM", type=float, default=0.2,
                        help="Crowns grown by this before the check")
    parser.add_argument("--minPatchM2", type=float, default=0.5,
                        help="Smallest uncovered patch counted")
    parser.add_argument("--jobs", type=int, default=4,
                        help="Tiles processed at once (each a PDAL process "
                             "of 1-2 GB)")
    parser.add_argument("--retries", type=int, default=5,
                        help="Retries of a failed tile before its site fails")
    parser.add_argument("--retryWaitS", type=float, default=60.0,
                        help="Wait before retry n is n times this")
    parser.add_argument("--tileTimeoutMin", type=float, default=15.0,
                        help="A PDAL process running longer is killed and "
                             "retried (a hung HTTPS read never ends)")
    parser.add_argument("--workDir", default=None,
                        help="Where the temporary PDAL tiles go (default: the "
                             "system's temporary folder, so an --output in "
                             "Dropbox syncs only finished products)")
    parser.add_argument("--visibleM", type=float, default=1.0,
                        help="Crowns whose CHM top is below this are "
                             "invisible: reported, not scored. Keep it fixed "
                             "across every setting tried")
    parser.add_argument("--recheck", action="store_true",
                        help="Redo the checks of prepared sites only")
    args = parser.parse_args(argv)
    args.lidarSource, args.rgbSource = lidarUrl, rgbUrl
    return args


def main(argv=None):
    args = parseArguments(argv)
    sites = args.sites or sitesIn(args.vectors)
    failed = []
    for site in sites:
        prepared = os.path.exists(os.path.join(args.output, site,
                                               "check.json"))
        if args.recheck:
            if prepared:
                recheckSite(site, args)
            continue
        if prepared:
            print("[qp] %s: already prepared, skipped" % site)
            continue
        try:
            prepareSite(site, args)
        except (RuntimeError, OSError, ValueError) as error:
            print("[qp] %s: FAILED, skipped: %s" % (site, error), flush=True)
            failed.append(site)
    if failed:
        print("[qp] %d site(s) failed; run the same command again to retry "
              "them: %s" % (len(failed), " ".join(failed)), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
