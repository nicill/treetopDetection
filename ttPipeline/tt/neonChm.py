"""
A finer canopy height model for each NEON tile, built from its point cloud.

    python -m tt.neonChm --manifest Data/neon/eval/manifest.csv --resolution 0.5

NEON's own CHM is 1 m, too coarse to separate crowns of a few square metres.
The point clouds are dense enough for finer grids (about 20 points per m2 at
NIWO) and carry elevations with the ground classified, so for each tile:

  ground   the class-2 points, interpolated linearly (nearest beyond their
           hull) to every point's position
  height   each point's elevation minus the ground under it; noise classes
           (7, 18) are dropped, negatives clipped to 0
  CHM      per cell of the RGB tile's own grid at --resolution, the highest
           point; empty cells take their nearest filled neighbour

Point cloud: <root>/LiDAR/<tile>.laz or .las, beside <root>/RGB/<tile>.tif.
Output: <root>/<folder>/<tile>_CHM.tif, folder "CHM050" for 0.5 m by default,
so tt.neonRun --chmFolder CHM050 scores it.

Check: each built CHM is warped onto NEON's 1 m grid of the same tile, taking
the maximum per cell as NEON's CHM does, and the mean absolute and signed
differences are printed per tile and written to <root>/<folder>/check.csv.
Tiles where over 5% of the cells had no point are marked sparse: the cloud has
gaps there and the filled cells come out too low.
"""

import argparse
import csv
import os
import sys

import laspy
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import distance_transform_edt

from .neonRun import readManifest

GROUND = 2
NOISE = (7, 18)
MAX_HEIGHT_M = 100.0
SPARSE_EMPTY_SHARE = 0.05   # more cells without a point: the cloud has gaps


def cloudFor(rgbPath, tile):
    root = os.path.dirname(os.path.dirname(rgbPath))
    for extension in (".laz", ".las"):
        path = os.path.join(root, "LiDAR", tile + extension)
        if os.path.exists(path):
            return path
    return None


def readCloud(path):
    """x, y, z and classification, noise removed."""
    cloud = laspy.read(path)
    kind = np.asarray(cloud.classification)
    keep = ~np.isin(kind, NOISE)
    return (np.asarray(cloud.x)[keep], np.asarray(cloud.y)[keep],
            np.asarray(cloud.z)[keep], kind[keep])


def groundUnder(x, y, z, kind):
    """Ground elevation under every point, from the class-2 points."""
    ground = kind == GROUND
    if ground.sum() < 3:
        raise ValueError("fewer than 3 ground points")
    points = np.column_stack([x[ground], y[ground]])
    linear = LinearNDInterpolator(points, z[ground])(x, y)
    nearest = NearestNDInterpolator(points, z[ground])(x, y)
    return np.where(np.isfinite(linear), linear, nearest)


def heights(x, y, z, kind):
    h = z - groundUnder(x, y, z, kind)
    return np.clip(h, 0.0, MAX_HEIGHT_M)


def rasterize(x, y, h, bounds, resolution):
    """Highest point per cell; empty cells from their nearest filled cell."""
    west, south, east, north = bounds
    width = int(round((east - west) / resolution))
    height = int(round((north - south) / resolution))
    column = np.floor((x - west) / resolution).astype(int)
    row = np.floor((north - y) / resolution).astype(int)
    inside = (column >= 0) & (column < width) & (row >= 0) & (row < height)
    chm = np.full((height, width), -1.0, dtype=np.float32)
    np.maximum.at(chm, (row[inside], column[inside]), h[inside])
    empty = chm < 0
    if empty.all():
        raise ValueError("no point falls on the tile")
    _, (rows, columns) = distance_transform_edt(empty, return_indices=True)
    return chm[rows, columns], float(empty.mean())


def writeChm(path, chm, bounds, resolution, crs):
    west, _, _, north = bounds
    transform = rasterio.transform.from_origin(west, north, resolution,
                                               resolution)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=chm.shape[0],
                       width=chm.shape[1], count=1, dtype="float32", crs=crs,
                       transform=transform, nodata=None) as destination:
        destination.write(chm, 1)


def differenceFromNeon(builtPath, neonPath):
    """
    Mean |built - NEON| and mean built - NEON on NEON's own 1 m grid, where
    NEON has canopy. The built CHM is warped onto that grid taking the maximum
    per cell, as NEON's CHM keeps the highest return; an average would lower
    every peak and bias the comparison.
    """
    with rasterio.open(neonPath) as neon, rasterio.open(builtPath) as built:
        reference = neon.read(1, masked=True).filled(0.0)
        onNeon = np.zeros(reference.shape, dtype=np.float32)
        reproject(built.read(1), onNeon, src_transform=built.transform,
                  src_crs=built.crs, dst_transform=neon.transform,
                  dst_crs=neon.crs, resampling=Resampling.max)
    canopy = reference > 0
    if not canopy.any():
        return float("nan"), float("nan")
    difference = onNeon[canopy] - reference[canopy]
    return float(np.abs(difference).mean()), float(difference.mean())


def buildTile(row, resolution, folder):
    cloudPath = cloudFor(row["rgb"], row["tile"])
    if cloudPath is None:
        return None
    with rasterio.open(row["rgb"]) as rgb:
        bounds, crs = tuple(rgb.bounds), rgb.crs
    x, y, z, kind = readCloud(cloudPath)
    chm, emptyShare = rasterize(x, y, heights(x, y, z, kind), bounds,
                                resolution)
    root = os.path.dirname(os.path.dirname(row["rgb"]))
    path = os.path.join(root, folder, row["tile"] + "_CHM.tif")
    writeChm(path, chm, bounds, resolution, crs)
    neonPath = os.path.join(root, "CHM", row["tile"] + "_CHM.tif")
    absolute, signed = differenceFromNeon(path, neonPath) \
        if os.path.exists(neonPath) else (float("nan"), float("nan"))
    return {"tile": row["tile"], "chm": path, "emptyCells": emptyShare,
            "sparse": emptyShare > SPARSE_EMPTY_SHARE,
            "meanAbsDiffM": absolute, "meanDiffM": signed}


def build(manifestPath, resolution, folder):
    rows, missing = [], []
    for entry in readManifest(manifestPath):
        result = buildTile(entry, resolution, folder)
        if result is None:
            missing.append(entry["tile"])
            continue
        rows.append(result)
        print("[chm] %-45s empty %4.1f%%  |built-NEON| %.2f m  "
              "built-NEON %+.2f m%s"
              % (result["tile"], 100 * result["emptyCells"],
                 result["meanAbsDiffM"], result["meanDiffM"],
                 "  SPARSE" if result["sparse"] else ""))
    if missing:
        print("[chm] %d tile(s) without a point cloud: %s"
              % (len(missing), ", ".join(missing)))
    if not rows:
        raise SystemExit("no tile had a point cloud")
    writeCheck(rows, os.path.dirname(rows[0]["chm"]))
    return rows


def writeCheck(rows, directory):
    with open(os.path.join(directory, "check.csv"), "w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for label, subset in (("all", rows),
                          ("not sparse", [r for r in rows if not r["sparse"]])):
        if not subset:
            continue
        absolute = np.array([r["meanAbsDiffM"] for r in subset])
        signed = np.array([r["meanDiffM"] for r in subset])
        print("[chm] %-10s %3d CHMs; |built-NEON| median %.2f m, worst %.2f m;"
              " built-NEON median %+.2f m"
              % (label, len(subset), np.nanmedian(absolute),
                 np.nanmax(absolute), np.nanmedian(signed)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Finer CHMs for the NEON tiles from their point clouds.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--resolution", type=float, default=0.5)
    parser.add_argument("--folder", default=None,
                        help="Output folder beside RGB/ (default CHM050 "
                             "for 0.5 m)")
    args = parser.parse_args(argv)
    folder = args.folder or "CHM%03d" % round(args.resolution * 100)
    build(args.manifest, args.resolution, folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
