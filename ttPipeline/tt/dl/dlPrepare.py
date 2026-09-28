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

import geopandas as gpd
import numpy as np
import rasterio
from shapely import wkt

from ..scene import readCrowns
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


class DatasetBuilder(object):
    """
    One area rasterised into overlapping tiles with crown labels, and the
    spatial blocks cross-validation splits on.

    Every raster is opened once for the whole build rather than once per tile:
    GDAL re-reads the header on every open, and that was paid 162 times per
    source.
    """

    def __init__(self, args):
        self.args = args
        self.specs = dc.parseSource(args.source)
        self.boundary = self._boundary()
        self.transform, self.width, self.height, self.crs = dc.buildGrid(
            self.specs[0].path, self.boundary, args.resolution)
        print("[data] grid %d x %d px at %.3f m, CRS %s"
              % (self.width, self.height, args.resolution, self.crs))
        self.crowns = self._crowns()
        self.crownBounds = self.crowns.geometry.bounds.to_numpy()

    def _boundary(self):
        frame = gpd.read_file(self.args.boundary)
        boundary = max(frame.geometry, key=lambda g: g.area)
        if len(frame) > 1:
            print("[data] boundary has %d polygons; using the largest (%.0f "
                  "m2). Nested clip outlines are common and would otherwise "
                  "be treated as separate plots." % (len(frame), boundary.area))
        return boundary

    def _crowns(self):
        crowns, repaired = readCrowns(self.args.crowns, self.crs,
                                      boundary=self.boundary)
        if repaired:
            print("[data] repaired %d invalid crown polygons" % repaired)
        print("[data] %d crowns inside the boundary, median area %.1f m2"
              % (len(crowns), crowns.geometry.area.median()))
        return crowns

    def blocks(self):
        args = self.args
        blocks = dc.makeBlocks(self.boundary, args.blockCols, args.blockRows,
                               areaName=args.name)
        minX, minY, maxX, maxY = self.boundary.bounds
        print("[data] %d blocks of about %.0f x %.0f m"
              % (len(blocks), (maxX - minX) / args.blockCols,
                 (maxY - minY) / args.blockRows))
        return blocks

    def tiles(self, blocks):
        args = self.args
        stride = max(1, int(round(args.tileSize * (1.0 - args.overlap))))
        tiles = dc.assignTilesToBlocks(
            dc.makeTiles(self.transform, self.width, self.height,
                         args.tileSize, stride, self.boundary), blocks)
        tiles = [t for t in tiles if t["block"] is not None]
        print("[data] %d tiles of %d px (%.1f m), stride %d px (%.1f m)"
              % (len(tiles), args.tileSize, args.tileSize * args.resolution,
                 stride, stride * args.resolution))
        return tiles, stride

    def writeTiles(self, tiles, ranges):
        tileDir = os.path.join(self.args.output, "tiles")
        os.makedirs(tileDir, exist_ok=True)
        sources = [rasterio.open(spec.path) for spec in self.specs]
        try:
            index = [self._writeTile(position, tile, ranges, sources, tileDir)
                     for position, tile in enumerate(tiles)]
        finally:
            for source in sources:
                source.close()
        return index

    def _writeTile(self, position, tile, ranges, sources, tileDir):
        window = (tile["c0"], tile["r0"], tile["c1"], tile["r1"])
        stack = np.concatenate(
            [dc.readWindow(spec, self.transform, *window, heightRange=value,
                           source=source)
             for spec, value, source in zip(self.specs, ranges, sources)],
            axis=0)
        labels = dc.cropCrowns(self.crowns, self.crownBounds, self.transform,
                               *window,
                               minInsideFraction=self.args.minInsideFraction)
        stem = "tile_%04d" % position
        np.save(os.path.join(tileDir, stem + ".npy"), stack)
        dc.saveJson({"crowns": labels}, os.path.join(tileDir, stem + ".json"))

        west, north = self.transform @ (tile["c0"], tile["r0"])
        east, south = self.transform @ (tile["c1"], tile["r1"])
        entry = dict(tile)
        entry.update({"stem": stem, "crownCount": len(labels),
                      "west": float(west), "north": float(north),
                      "east": float(east), "south": float(south)})
        return entry

    def metadata(self, ranges, stride, blocks, index):
        args = self.args
        return {
            "name": args.name, "source": args.source,
            "channels": [spec.describe() for spec in self.specs],
            "channelCount": sum(spec.channelCount for spec in self.specs),
            "scaleRanges": ranges, "resolution": args.resolution,
            "tileSize": args.tileSize, "overlap": args.overlap,
            "stride": stride, "minHeight": args.minHeight,
            "crs": str(self.crs), "transform": list(self.transform)[:6],
            "width": self.width, "height": self.height,
            "crownsPath": os.path.abspath(args.crowns),
            "boundaryPath": os.path.abspath(args.boundary),
            "crownCount": int(len(self.crowns)),
            "blocks": [{"name": name, "wkt": geometry.wkt}
                       for name, geometry in blocks],
            "tiles": index,
        }

    def build(self):
        ranges = dc.measureRanges(self.specs, self.transform, self.width,
                                  self.height, minHeight=self.args.minHeight)
        blocks = self.blocks()
        tiles, stride = self.tiles(blocks)
        index = self.writeTiles(tiles, ranges)

        perBlock = {}
        for entry in index:
            perBlock[entry["block"]] = perBlock.get(entry["block"], 0) + 1
        print("[data] tiles per block: %s" % perBlock)
        print("[data] %d tiles hold no crown (kept: background matters for "
              "precision)" % sum(1 for e in index if not e["crownCount"]))

        path = os.path.join(self.args.output, "dataset.json")
        dc.saveJson(self.metadata(ranges, stride, blocks, index), path)
        print("[data] wrote %s" % path)
        return 0


def build(args):
    return DatasetBuilder(args).build()


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
