"""
Crown annotation delivered as a label raster, converted to what the pipeline
reads: a crown shapefile and a boundary shapefile.

    python -m tt.labelImport --labels Label_image.tif --reference ROI_DEM2.tif \\
        --output Data/sergi

Each non-zero value of the label raster is one crown. The raster can carry its
own georeferencing or none that matches: when it has the reference raster's
exact pixel dimensions and covers the same ground, it is taken to be on the
reference grid and given the reference's transform and CRS. That is the case
the Sergi data arrived in — the label image tagged in geographic coordinates,
the DEM in UTM, the same 1152 x 4384 pixels over the same footprint.

The boundary is the reference raster's valid area: every connected piece of
real size, as one (multi)polygon. Not only the largest: the Sergi DEM covers two
separate pieces, and the smaller one holds 169 of the 1955 crowns.
"""

import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_bounds
from shapely.geometry import shape
from shapely.ops import unary_union

BOUNDS_TOLERANCE_DEG = 1e-5   # about a metre: "covers the same ground"
BOUNDARY_SIMPLIFY_M = 0.5
MIN_PIECE_M2 = 10.0           # smaller valid specks are edge noise


def sameGrid(labels, reference):
    """True when the label raster is the reference grid, whatever its tags."""
    if (labels.width, labels.height) != (reference.width, reference.height):
        return False
    if labels.crs == reference.crs and labels.transform == reference.transform:
        return True
    if labels.crs is None:
        return True
    try:
        mine = transform_bounds(labels.crs, "EPSG:4326", *labels.bounds)
        theirs = transform_bounds(reference.crs, "EPSG:4326",
                                  *reference.bounds)
    except Exception:
        return False
    return max(abs(a - b) for a, b in zip(mine, theirs)) \
        < BOUNDS_TOLERANCE_DEG


def crownPolygons(labelArray, transform, crs):
    """One polygon (or multipolygon) per crown id."""
    pieces = {}
    for geometry, value in shapes(labelArray, mask=labelArray > 0,
                                  transform=transform):
        pieces.setdefault(int(value), []).append(shape(geometry))
    ids = sorted(pieces)
    geometries = [unary_union(pieces[i]) for i in ids]
    frame = gpd.GeoDataFrame({"crownId": ids}, geometry=geometries, crs=crs)
    frame["area"] = frame.geometry.area
    return frame


def validBoundary(reference):
    """Every piece of the valid area above MIN_PIECE_M2, as one feature."""
    mask = reference.read_masks(1) > 0
    polygons = [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask,
                                            transform=reference.transform)
                if v == 1]
    kept = [p for p in polygons if p.area >= MIN_PIECE_M2]
    area = unary_union(kept).simplify(BOUNDARY_SIMPLIFY_M)
    print("[labels] valid area: %d piece(s) kept of %d"
          % (len(kept), len(polygons)))
    return gpd.GeoDataFrame({"name": ["area"]}, geometry=[area],
                            crs=reference.crs)


def convert(labelPath, referencePath, outputDir):
    os.makedirs(outputDir, exist_ok=True)
    with rasterio.open(labelPath) as labels, \
            rasterio.open(referencePath) as reference:
        if labels.crs == reference.crs and \
                labels.transform == reference.transform:
            transform, crs = labels.transform, labels.crs
        elif sameGrid(labels, reference):
            print("[labels] %s is on the reference grid; using the "
                  "reference's georeferencing (%s)"
                  % (os.path.basename(labelPath), reference.crs))
            transform, crs = reference.transform, reference.crs
        else:
            raise SystemExit("%s is not on the grid of %s; reproject it first"
                             % (labelPath, referencePath))
        crowns = crownPolygons(labels.read(1).astype(np.int32), transform,
                               crs)
        boundary = validBoundary(reference)

    crownPath = os.path.join(outputDir, "crowns.shp")
    boundaryPath = os.path.join(outputDir, "area.shp")
    crowns.to_file(crownPath)
    boundary.to_file(boundaryPath)
    print("[labels] %d crowns, median area %.1f m2 -> %s"
          % (len(crowns), crowns["area"].median(), crownPath))
    print("[labels] boundary %.0f m2 -> %s"
          % (boundary.geometry.iloc[0].area, boundaryPath))
    return crownPath, boundaryPath


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Label raster to crown and boundary shapefiles.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--reference", required=True,
                        help="The raster whose grid the labels are on")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    convert(args.labels, args.reference, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
