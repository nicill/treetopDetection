"""Synthetic sites with known answers, shared by the tests."""

import json
import os
import subprocess
import sys

import numpy as np


def buildFixture(directory, trees, resolution=0.25, size=240):
    """
    Write a CHM and crown shapefile for `trees`, each (columnPx, rowPx,
    radiusM, heightM). Cones on a flat floor: crisp apexes at known pixels.
    """
    import geopandas as gpd
    import rasterio
    from affine import Affine
    from shapely.geometry import Point, box

    originX, originY = 500000.0, 5000000.0
    rows, columns = np.mgrid[0:size, 0:size]
    chm = np.zeros((size, size), np.float32)
    records = []

    for columnPx, rowPx, radiusM, heightM in trees:
        radiusPx = radiusM / resolution
        distance = np.hypot(rows - rowPx, columns - columnPx)
        chm = np.maximum(chm, heightM * np.clip(1 - distance / radiusPx, 0, 1))
        east = originX + (columnPx + 0.5) * resolution
        north = originY - (rowPx + 0.5) * resolution
        records.append({"geometry": Point(east, north).buffer(radiusM * 0.95),
                        "tree_class": 1})

    # -3.4028235e+38 is a rounded-up float64 literal that sits just below
    # float32's minimum, and newer rasterio rejects it. A plain sentinel avoids
    # the whole question.
    nodata = -9999.0
    stored = chm.copy()
    stored[stored < 2.0] = nodata
    transform = Affine(resolution, 0.0, originX, 0.0, -resolution, originY)

    chmPath = os.path.join(directory, "chm.tif")
    with rasterio.open(chmPath, "w", driver="GTiff", height=size, width=size,
                       count=1, dtype="float32", crs="EPSG:32648",
                       transform=transform, nodata=nodata) as destination:
        destination.write(stored, 1)

    crownsPath = os.path.join(directory, "crowns.shp")
    gpd.GeoDataFrame(records, crs="EPSG:32648").to_file(crownsPath)

    boundaryPath = os.path.join(directory, "area.shp")
    gpd.GeoDataFrame(
        [{"geometry": box(originX, originY - size * resolution,
                          originX + size * resolution, originY)}],
        crs="EPSG:32648").to_file(boundaryPath)

    return chmPath, crownsPath, boundaryPath


def buildDataset(directory, chmPath, crownsPath, boundaryPath, blockCols=2,
                 blockRows=2, name="t"):
    """A prepared dataset folder (dataset.json) over a fixture."""
    out = os.path.join(directory, "ds")
    if not os.path.exists(os.path.join(out, "dataset.json")):
        subprocess.run([sys.executable, "-m", "tt.dl", "prepare",
                        "--source", "chm:" + chmPath, "--crowns", crownsPath,
                        "--boundary", boundaryPath, "--output", out,
                        "--name", name, "--resolution", "0.25",
                        "--minHeight", "1.0", "--blockCols", str(blockCols),
                        "--blockRows", str(blockRows)], check=True,
                       capture_output=True,
                       cwd=os.path.dirname(os.path.dirname(
                           os.path.abspath(__file__))))
    return out


def crowdedTrees(count=60, seed=3, size=240):
    """Random overlapping cones: a stand where settings make a difference."""
    rng = np.random.default_rng(seed)
    return [(int(rng.integers(15, size - 15)), int(rng.integers(15, size - 15)),
             float(rng.uniform(1.5, 4.0)), float(rng.uniform(3, 20)))
            for _ in range(count)]


def plain(value):
    """JSON round trip, so dicts compare as files would."""
    return json.loads(json.dumps(value, default=float))
