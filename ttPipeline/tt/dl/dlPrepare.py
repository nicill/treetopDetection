#!/usr/bin/env python
"""
dlPrepare.py

Turn one annotated area into a tiled dataset that Mask R-CNN, YOLO and the
connected-component detector can all be run on, with spatial blocks for
cross-validation.

    python dlPrepare.py \\
        --source "chm:chm_lidar.tif" \\
        --crowns shp/annotation_new_0925.shp \\
        --boundary shp/area_p1.shp \\
        --output datasets/lidarChm --name lidar

    python dlPrepare.py \\
        --source "rgb:ortho.tif+chm:chm_lidar.tif" \\
        ... --output datasets/rgbLidar

The `--source` string decides the channels: `chm:` contributes one band,
`rgb:` contributes three, and several can be combined with `+` in the order
they should be stacked. Putting rgb first gives the 4-channel input with the
photo in bands 1 to 3 and the height model in band 4.

What it writes
--------------
    tiles/tile_0000.npy      uint8, (C, H, W)
    tiles/tile_0000.json     crowns in that tile, as boxes and polygons
    dataset.json             grid, blocks, tile index, scaling, provenance

Tiles are stored once. The per-model exporters build whatever directory layout
their framework wants from this, per fold, so the pixels are never duplicated
per experiment.

Scaling ranges are measured once over the whole area and recorded, so a tile
in the test fold is never scaled differently from the tiles a model trained on.
"""

import argparse
import os
import sys

import rasterio
from shapely import wkt

from ..data.patches import PatchSetBuilder
from . import dlCommon as dc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a tiled, spatially blocked dataset from one area.")
    parser.add_argument("--source", required=True,
                        help="Channels, e.g. 'chm:lidar.tif' or "
                             "'rgb:ortho.tif+chm:lidar.tif'")
    parser.add_argument("--crowns", required=True)
    parser.add_argument("--boundary", required=True,
                        help="Area polygon shapefile. If it holds several "
                             "polygons the largest is used, since nested clip "
                             "boundaries are common")
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", default="area",
                        help="Short name, used to prefix block names so "
                             "several areas can be pooled later")

    parser.add_argument("--resolution", type=float, default=0.05,
                        help="Metres per pixel for the tiles (default: 0.05). "
                             "Crowns here are about 4 m across, so this makes "
                             "them roughly 80 px — a normal object scale for "
                             "a detector")
    parser.add_argument("--tileSize", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Tile overlap (default: 0.5). Overlap is what "
                             "lets every tree be seen whole in some tile; the "
                             "double counting it causes is removed by the "
                             "stitched evaluation")
    parser.add_argument("--blockCols", type=int, default=4)
    parser.add_argument("--blockRows", type=int, default=2)
    parser.add_argument("--minHeight", type=float, default=2.0)
    parser.add_argument("--minInsideFraction", type=float, default=0.5)
    return build(parser.parse_args(argv))


def build(args):
    """Build the dataset through tt.data.PatchSet."""
    PatchSetBuilder(args.output, dc.parseSource(args.source), args.crowns,
                    args.boundary, args.name, args.resolution, args.tileSize,
                    args.overlap, args.blockCols, args.blockRows,
                    args.minHeight, args.minInsideFraction,
                    source=args.source).build()
    return 0


def loadDataset(path):
    """Read a prepared dataset back, with shapely geometries restored."""

    meta = dc.loadJson(os.path.join(path, "dataset.json"))
    meta["root"] = path
    meta["blockGeometries"] = [(b["name"], wkt.loads(b["wkt"]))
                               for b in meta["blocks"]]
    meta["transformObject"] = rasterio.Affine(*meta["transform"])
    return meta


if __name__ == "__main__":
    sys.exit(main())
