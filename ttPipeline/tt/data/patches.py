"""
Patches — a site cut into overlapping tiles for the networks, and the spatial
blocks cross-validation splits on.

A PatchSet is built once per site and set of channels and stored as a folder:

    tiles/tile_0000.npy      uint8, (C, H, W)
    tiles/tile_0000.json     crowns in that tile, as boxes and polygons
    dataset.json             grid, blocks, tile index, scaling, provenance

Scaling ranges are measured once over the whole area and recorded, so a tile
in a test fold is never scaled differently from the tiles a model trained on.

Spatial blocks, not random splits: neighbouring tiles share trees, so a fold
holds out one whole block and its training tiles must lie a buffer away from it.
Tiles overlap so every tree is seen whole somewhere; predictions are stitched
back into world coordinates and deduplicated before scoring, so each tree is
scored once.

The tile records stored in dataset.json are dicts; Patch is the view the rest
of the code uses. The record-level functions below are kept for the modules
that still read dataset.json directly.
"""

import json
import math
import os
import zlib

from affine import Affine
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from shapely import wkt
from shapely.geometry import Point, box
from shapely.ops import unary_union

from .annotations import readCrowns


class ChannelSpec(object):
    """One contribution to a tile's channels: a height model or a photo."""

    def __init__(self, kind, path, bands=None):
        if kind not in ("chm", "rgb"):
            raise ValueError("kind must be 'chm' or 'rgb', got %r" % kind)
        self.kind = kind
        self.path = path
        self.bands = list(bands) if bands else ([1, 2, 3] if kind == "rgb"
                                                else [1])

    @property
    def channelCount(self):
        return 3 if self.kind == "rgb" else 1

    def describe(self):
        return {"kind": self.kind, "path": self.path, "bands": self.bands}


def parseSource(text):
    """
    'rgb:ortho.tif+chm:lidar.tif' -> [ChannelSpec('rgb', ...),
                                      ChannelSpec('chm', ...)]
    Channels are stacked in the order given, so RGB first then CHM produces the
    4-channel input with the photo in the first three bands.
    """
    specs = []
    for part in text.split("+"):
        part = part.strip()
        if ":" not in part:
            raise ValueError("expected kind:path, got %r" % part)
        kind, path = part.split(":", 1)
        specs.append(ChannelSpec(kind.strip(), path.strip()))
    return specs


def buildGrid(referencePath, boundary, resolution):
    """
    A grid covering `boundary` at `resolution`, aligned to the reference raster
    so no resampling shifts anything. Returns (transform, width, height, crs).
    """
    with rasterio.open(referencePath) as src:
        crs = src.crs
        native = src.transform

    minX, minY, maxX, maxY = boundary.bounds
    # snap the origin to a multiple of the resolution from the raster origin
    offsetX = np.floor((minX - native.c) / resolution) * resolution
    offsetY = np.ceil((native.f - maxY) / resolution) * resolution
    originX = native.c + offsetX
    originY = native.f - offsetY

    width = int(np.ceil((maxX - originX) / resolution))
    height = int(np.ceil((originY - minY) / resolution))
    # Built directly rather than through rasterio.transform.from_origin, which
    # still multiplies Affines with * and so warns under affine 3.0.
    transform = Affine(resolution, 0.0, originX, 0.0, -resolution, originY)
    return transform, width, height, crs


def readWindow(spec, transform, c0, r0, c1, r1, heightRange=None,
               source=None):
    """
    Read one channel spec over a grid window, returned as uint8 (C, H, W).

    A photo is percentile-stretched per band over the whole dataset, not per
    tile, so brightness is comparable between tiles. A height model is mapped
    linearly from `heightRange` to 1..255 with 0 reserved for no canopy, which
    keeps "no canopy" distinguishable from "canopy at the lowest height".

    Pass an already open `source` when reading many windows: opening the file
    per tile re-parses its header every time.
    """
    if source is None:
        with rasterio.open(spec.path) as opened:
            return readWindow(spec, transform, c0, r0, c1, r1, heightRange,
                              source=opened)

    height, width = r1 - r0, c1 - c0
    west, north = transform @ (c0, r0)
    east, south = transform @ (c1, r1)
    window = from_bounds(min(west, east), min(north, south),
                         max(west, east), max(north, south),
                         transform=source.transform)
    data = source.read(indexes=spec.bands, window=window,
                       out_shape=(len(spec.bands), height, width),
                       resampling=Resampling.average,
                       boundless=True, fill_value=0).astype(np.float32)

    if spec.kind == "rgb":
        lo, hi = heightRange  # reused as the per-source stretch range
        scaled = np.clip((data - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return (255 * scaled).astype(np.uint8)

    band = data[0]
    band[~np.isfinite(band)] = 0.0
    lo, hi = heightRange
    band[band < lo] = 0.0
    canopy = band > 0
    out = np.zeros(band.shape, np.uint8)
    out[canopy] = (1 + 254 * np.clip((band[canopy] - lo) / max(hi - lo, 1e-6),
                                     0.0, 1.0)).astype(np.uint8)
    return out[None, :, :]


def measureRanges(specs, transform, width, height, minHeight=2.0,
                  sampleShape=1024, verbose=True):
    """
    Work out, once for the whole dataset, the value range each source is scaled
    with. Doing this per tile would make identical trees look different from one
    tile to the next.
    """
    ranges = []
    for spec in specs:
        step = max(1, int(max(width, height) / sampleShape))
        west, north = transform @ (0, 0)
        east, south = transform @ (width, height)
        with rasterio.open(spec.path) as src:
            window = from_bounds(min(west, east), min(north, south),
                                 max(west, east), max(north, south),
                                 transform=src.transform)
            data = src.read(indexes=spec.bands, window=window,
                            out_shape=(len(spec.bands),
                                       max(1, height // step),
                                       max(1, width // step)),
                            resampling=Resampling.average,
                            boundless=True, fill_value=0).astype(np.float32)
        if spec.kind == "rgb":
            values = data[np.isfinite(data) & (data != 0)]
            lo, hi = (np.percentile(values, (1, 99)) if values.size
                      else (0.0, 255.0))
        else:
            band = data[0]
            values = band[np.isfinite(band) & (band >= minHeight)]
            lo = minHeight
            hi = float(np.percentile(values, 99.5)) if values.size else 30.0
        ranges.append((float(lo), float(hi)))
        if verbose:
            print("[data] %s %s -> scaled from %.2f to %.2f"
                  % (spec.kind, os.path.basename(spec.path), lo, hi))
    return ranges


def makeBlocks(boundary, blockCols, blockRows, areaName="area"):
    """Cut the boundary into a grid of blocks. Returns [(name, geometry), ...]."""

    minX, minY, maxX, maxY = boundary.bounds
    stepX = (maxX - minX) / blockCols
    stepY = (maxY - minY) / blockRows

    blocks = []
    for row in range(blockRows):
        for col in range(blockCols):
            cell = box(minX + col * stepX, maxY - (row + 1) * stepY,
                       minX + (col + 1) * stepX, maxY - row * stepY)
            piece = cell.intersection(boundary)
            if piece.is_empty or piece.area <= 0:
                continue
            blocks.append(("%s_b%d%d" % (areaName, row, col), piece))
    return blocks


def makeTiles(transform, width, height, tileSize, stride, boundary,
              minCoverage=0.5):
    """
    Tile windows over the grid, keeping those sufficiently inside the boundary.
    Returns [{"c0","r0","c1","r1","centreX","centreY"}, ...] in world units.
    """

    tiles = []
    for r0 in range(0, max(1, height - tileSize + 1), stride):
        for c0 in range(0, max(1, width - tileSize + 1), stride):
            c1, r1 = c0 + tileSize, r0 + tileSize
            if c1 > width or r1 > height:
                continue
            west, north = transform @ (c0, r0)
            east, south = transform @ (c1, r1)
            extent = box(min(west, east), min(north, south),
                         max(west, east), max(north, south))
            if extent.intersection(boundary).area < minCoverage * extent.area:
                continue
            centre = extent.centroid
            tiles.append({"c0": c0, "r0": r0, "c1": c1, "r1": r1,
                          "centreX": float(centre.x),
                          "centreY": float(centre.y)})
    return tiles


def assignTilesToBlocks(tiles, blocks):
    """Label each tile with the block its centre falls in."""

    for tile in tiles:
        tile["block"] = None
        point = Point(tile["centreX"], tile["centreY"])
        for name, geometry in blocks:
            if geometry.contains(point):
                tile["block"] = name
                break
    return tiles


def blockSuffix(name):
    """rgb_b03 -> _b03: the part of a block name every dataset shares."""
    return name[name.rindex("_b"):] if "_b" in name else name


def tileExtent(tile):
    return box(min(tile["west"], tile["east"]), min(tile["north"], tile["south"]),
               max(tile["west"], tile["east"]), max(tile["north"], tile["south"]))


def drawTiles(tiles, fraction, seed, heldOut):
    """
    A fold's annotated subset for the learning curve: from each training block
    the same fraction of its tiles (at least one), drawn at random. Whole
    tiles, so every tree in a drawn tile is annotated: a learned model treats
    an unannotated tree in a training tile as background.

    Deterministic in (seed, held-out block suffix, tile positions), so the RGB
    and CHM datasets of a site, whose tiles coincide, draw the same tiles, and
    connected components and Mask R-CNN are given the same annotation.
    """
    if fraction >= 1:
        return list(tiles)
    rng = np.random.default_rng([int(seed),
                                 zlib.crc32(blockSuffix(heldOut).encode())])
    chosen = []
    for block in sorted({t["block"] for t in tiles}, key=blockSuffix):
        members = sorted((t for t in tiles if t["block"] == block),
                         key=lambda t: (t["r0"], t["c0"]))
        count = max(1, int(math.ceil(fraction * len(members))))
        picks = rng.choice(len(members), size=count, replace=False)
        chosen.extend(members[i] for i in sorted(picks))
    return chosen


def tileRegion(tiles):
    """The ground the tiles cover: where their annotation is known."""
    return unary_union([tileExtent(t) for t in tiles])


def foldSplit(tiles, blocks, heldOut, bufferM):
    """
    Training tiles for a fold: those whose tile extent stays at least `bufferM`
    from the held-out block, so no tree is in both. Test tiles: those centred in
    the held-out block.
    """

    geometry = dict(blocks)[heldOut]
    guard = geometry.buffer(bufferM)

    train, test = [], []
    for tile in tiles:
        if tile["block"] == heldOut:
            test.append(tile)
            continue
        if not tileExtent(tile).intersects(guard):
            train.append(tile)
    return train, test


def cropCrowns(crowns, crownBounds, transform, c0, r0, c1, r1,
               minInsideFraction=0.5, minAreaPx=16):
    """
    Crowns for one tile, in tile-pixel coordinates.

    A crown clipped to less than `minInsideFraction` of its area is dropped:
    training a detector on a sliver teaches it that a third of a tree is a tree.
    Crowns near the edge are still seen whole in an overlapping neighbour tile.

    Returns [{"box": [x0,y0,x1,y1], "polygon": [[x,y], ...]}, ...]
    """
    west, north = transform @ (c0, r0)
    east, south = transform @ (c1, r1)
    minX, maxX = min(west, east), max(west, east)
    minY, maxY = min(north, south), max(north, south)
    extent = box(minX, minY, maxX, maxY)
    candidates = np.nonzero((crownBounds[:, 0] <= maxX)
                            & (crownBounds[:, 2] >= minX)
                            & (crownBounds[:, 1] <= maxY)
                            & (crownBounds[:, 3] >= minY))[0]

    inverse = ~transform
    out = []
    for index in candidates:
        piece = _clipToTile(crowns.geometry.iloc[index], extent,
                            minInsideFraction)
        if piece is None:
            continue
        ring = _tilePixels(np.asarray(piece.exterior.coords), inverse,
                           c0, r0, c1 - c0, r1 - r0)
        x0, y0 = ring[:, 0].min(), ring[:, 1].min()
        x1, y1 = ring[:, 0].max(), ring[:, 1].max()
        if (x1 - x0) * (y1 - y0) < minAreaPx:
            continue
        out.append({"box": [float(x0), float(y0), float(x1), float(y1)],
                    "polygon": ring.tolist(), "crownIndex": int(index)})
    return out


def _clipToTile(geometry, extent, minInsideFraction):
    """The largest piece of a crown inside the tile, or None if too little."""
    if geometry is None or geometry.is_empty:
        return None
    clipped = geometry.intersection(extent)
    if clipped.is_empty or clipped.area < minInsideFraction * geometry.area:
        return None
    pieces = [p for p in getattr(clipped, "geoms", [clipped])
              if getattr(p, "exterior", None) is not None]
    return max(pieces, key=lambda p: p.area) if pieces else None


def _tilePixels(coordinates, inverse, c0, r0, width, height):
    """
    World coordinates to tile pixels, clipped to the tile.

    Vectorised, but in affine's own arithmetic order — x*a + y*b + c — so the
    result is bit-identical to applying the transform one vertex at a time.
    """
    a, b, c, d, e, f = inverse[:6]
    x, y = coordinates[:, 0], coordinates[:, 1]
    ring = np.column_stack([x * a + y * b + c - c0, x * d + y * e + f - r0])
    ring[:, 0] = np.clip(ring[:, 0], 0, width)
    ring[:, 1] = np.clip(ring[:, 1], 0, height)
    return ring


def tileToWorld(predictions, transform, c0, r0):
    """
    Tile-pixel predictions to world coordinates, adding a centre.

    Fields other than the box are carried through, and a tile-pixel "polygon"
    is converted with the box, so a mask outline survives stitching and NMS
    and can be saved beside the score.
    """
    out = []
    for prediction in predictions:
        x0, y0, x1, y1 = prediction["box"]
        west, north = transform @ (c0 + x0, r0 + y0)
        east, south = transform @ (c0 + x1, r0 + y1)
        centreX, centreY = transform @ (c0 + 0.5 * (x0 + x1),
                                        r0 + 0.5 * (y0 + y1))
        world = dict(prediction)
        world.update({"box": [min(west, east), min(north, south),
                              max(west, east), max(north, south)],
                      "centreX": float(centreX), "centreY": float(centreY),
                      "score": float(prediction.get("score", 1.0))})
        if prediction.get("polygon"):
            ring = np.asarray(prediction["polygon"], np.float64)
            eastings = transform.c + transform.a * (c0 + ring[:, 0])
            northings = transform.f + transform.e * (r0 + ring[:, 1])
            world["polygon"] = np.column_stack([eastings,
                                                northings]).tolist()
        out.append(world)
    return out


def chooseValidationBlock(trainTiles, heldOut):
    """
    One training block set aside for choosing the operating point.

    Deterministic: the training block following the held-out one in sorted
    order, so a fold always makes the same choice and reruns are comparable.
    """
    blocks = sorted({tile["block"] for tile in trainTiles})
    if not blocks:
        return None
    later = [b for b in blocks if b > heldOut]
    return later[0] if later else blocks[0]



# ---------------------------------------------------------------------- #
# the classes
# ---------------------------------------------------------------------- #

class Patch(object):
    """One tile: a window of the patch grid, the block it belongs to."""

    def __init__(self, record, patchSet=None):
        self.record = record            # the stored dict, shared, not copied
        self.patchSet = patchSet

    def __repr__(self):
        return "Patch(%s, %s)" % (self.stem, self.block)

    @property
    def stem(self):
        return self.record.get("stem")

    @property
    def block(self):
        return self.record.get("block")

    @property
    def window(self):
        r = self.record
        return r["c0"], r["r0"], r["c1"], r["r1"]

    @property
    def extent(self):
        return tileExtent(self.record)

    @property
    def crownCount(self):
        return self.record.get("crownCount", 0)

    def image(self):
        """uint8 (C, H, W)."""
        return self.patchSet.image(self)

    def crowns(self):
        """[{"box", "polygon", "crownIndex"}] in tile pixels."""
        return self.patchSet.labels(self)


class PatchSet(object):

    def __init__(self, root, meta):
        self.root = root
        self.meta = meta
        self.patches = [Patch(record, self) for record in meta["tiles"]]

    def __repr__(self):
        return "PatchSet(%s: %d tiles of %d px, %d channels)" % (
            self.name, len(self.patches), self.tileSize, self.channelCount)

    def __len__(self):
        return len(self.patches)

    def __iter__(self):
        return iter(self.patches)

    @property
    def name(self):
        return self.meta["name"]

    @property
    def channelCount(self):
        return self.meta["channelCount"]

    @property
    def tileSize(self):
        return self.meta["tileSize"]

    @property
    def transform(self):
        return Affine(*self.meta["transform"])

    @property
    def crs(self):
        return self.meta["crs"]

    @property
    def blocks(self):
        """[(name, geometry)]"""
        return [(b["name"], wkt.loads(b["wkt"])) for b in self.meta["blocks"]]

    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, root):
        with open(os.path.join(root, "dataset.json")) as handle:
            return cls(root, json.load(handle))

    @classmethod
    def build(cls, output, specs, crownsPath, areaPath, name="area",
              resolution=0.05, tileSize=512, overlap=0.5, blockCols=4,
              blockRows=2, minHeight=2.0, minInsideFraction=0.5,
              source=None):
        """Cut the area into tiles and blocks, write them, and load them."""
        PatchSetBuilder(output, specs, crownsPath, areaPath, name, resolution,
                        tileSize, overlap, blockCols, blockRows, minHeight,
                        minInsideFraction, source).build()
        return cls.load(output)

    # ------------------------------------------------------------------ #

    def image(self, patch):
        return np.load(os.path.join(self.root, "tiles", patch.stem + ".npy"))

    def labels(self, patch):
        with open(os.path.join(self.root, "tiles", patch.stem + ".json")) as h:
            return json.load(h)["crowns"]

    def inBlocks(self, names):
        names = set(names)
        return [p for p in self.patches if p.block in names]

    def foldSplit(self, heldOut, bufferM):
        """(training patches a buffer away from the block, its test patches)"""
        train, test = foldSplit(self.meta["tiles"], self.blocks, heldOut,
                                bufferM)
        return self._wrap(train), self._wrap(test)

    def validationBlock(self, patches, heldOut):
        """The training block set aside to choose the operating point."""
        return chooseValidationBlock([p.record for p in patches], heldOut)

    def draw(self, patches, fraction, seed, heldOut):
        """The learning curve's annotated subset: whole tiles per block."""
        return self._wrap(drawTiles([p.record for p in patches], fraction,
                                    seed, heldOut))

    @staticmethod
    def region(patches):
        """The ground the patches cover."""
        return tileRegion([p.record for p in patches])

    def toWorld(self, predictions, patch):
        """Tile-pixel predictions of one patch to world coordinates."""
        c0, r0, _, _ = patch.window
        return tileToWorld(predictions, self.transform, c0, r0)

    def _wrap(self, records):
        return [Patch(r, self) for r in records]


class PatchSetBuilder(object):
    """
    One area rasterised into overlapping tiles with crown labels. Every raster
    is opened once for the whole build: GDAL re-reads the header per open.
    """

    def __init__(self, output, specs, crownsPath, areaPath, name, resolution,
                 tileSize, overlap, blockCols, blockRows, minHeight,
                 minInsideFraction, source=None):
        self.output, self.specs, self.name = output, specs, name
        self.crownsPath, self.areaPath = crownsPath, areaPath
        self.resolution, self.tileSize, self.overlap = (resolution, tileSize,
                                                        overlap)
        self.blockCols, self.blockRows = blockCols, blockRows
        self.minHeight, self.minInsideFraction = minHeight, minInsideFraction
        self.source = source or "+".join("%s:%s" % (s.kind, s.path)
                                         for s in specs)
        self.boundary = self._boundary()
        self.transform, self.width, self.height, self.crs = buildGrid(
            specs[0].path, self.boundary, resolution)
        print("[data] grid %d x %d px at %.3f m, CRS %s"
              % (self.width, self.height, resolution, self.crs))
        self.crowns = self._crowns()
        self.crownBounds = self.crowns.geometry.bounds.to_numpy()

    def _boundary(self):
        frame = gpd.read_file(self.areaPath)
        boundary = max(frame.geometry, key=lambda g: g.area)
        if len(frame) > 1:
            print("[data] boundary has %d polygons; using the largest (%.0f "
                  "m2)" % (len(frame), boundary.area))
        return boundary

    def _crowns(self):
        crowns, repaired = readCrowns(self.crownsPath, self.crs,
                                      boundary=self.boundary)
        if repaired:
            print("[data] repaired %d invalid crown polygons" % repaired)
        print("[data] %d crowns inside the boundary, median area %.1f m2"
              % (len(crowns), crowns.geometry.area.median()))
        return crowns

    def blocks(self):
        blocks = makeBlocks(self.boundary, self.blockCols, self.blockRows,
                            areaName=self.name)
        minX, minY, maxX, maxY = self.boundary.bounds
        print("[data] %d blocks of about %.0f x %.0f m"
              % (len(blocks), (maxX - minX) / self.blockCols,
                 (maxY - minY) / self.blockRows))
        return blocks

    def tiles(self, blocks):
        stride = max(1, int(round(self.tileSize * (1.0 - self.overlap))))
        tiles = assignTilesToBlocks(
            makeTiles(self.transform, self.width, self.height, self.tileSize,
                      stride, self.boundary), blocks)
        tiles = [t for t in tiles if t["block"] is not None]
        print("[data] %d tiles of %d px (%.1f m), stride %d px (%.1f m)"
              % (len(tiles), self.tileSize, self.tileSize * self.resolution,
                 stride, stride * self.resolution))
        return tiles, stride

    def writeTiles(self, tiles, ranges):
        tileDir = os.path.join(self.output, "tiles")
        os.makedirs(tileDir, exist_ok=True)
        sources = [rasterio.open(spec.path) for spec in self.specs]
        try:
            return [self._writeTile(position, tile, ranges, sources, tileDir)
                    for position, tile in enumerate(tiles)]
        finally:
            for source in sources:
                source.close()

    def _writeTile(self, position, tile, ranges, sources, tileDir):
        window = (tile["c0"], tile["r0"], tile["c1"], tile["r1"])
        stack = np.concatenate(
            [readWindow(spec, self.transform, *window, heightRange=value,
                        source=source)
             for spec, value, source in zip(self.specs, ranges, sources)],
            axis=0)
        labels = cropCrowns(self.crowns, self.crownBounds, self.transform,
                            *window, minInsideFraction=self.minInsideFraction)
        stem = "tile_%04d" % position
        np.save(os.path.join(tileDir, stem + ".npy"), stack)
        _saveJson({"crowns": labels}, os.path.join(tileDir, stem + ".json"))
        west, north = self.transform @ (tile["c0"], tile["r0"])
        east, south = self.transform @ (tile["c1"], tile["r1"])
        entry = dict(tile)
        entry.update({"stem": stem, "crownCount": len(labels),
                      "west": float(west), "north": float(north),
                      "east": float(east), "south": float(south)})
        return entry

    def metadata(self, ranges, stride, blocks, index):
        return {
            "name": self.name, "source": self.source,
            "channels": [spec.describe() for spec in self.specs],
            "channelCount": sum(spec.channelCount for spec in self.specs),
            "scaleRanges": ranges, "resolution": self.resolution,
            "tileSize": self.tileSize, "overlap": self.overlap,
            "stride": stride, "minHeight": self.minHeight,
            "crs": str(self.crs), "transform": list(self.transform)[:6],
            "width": self.width, "height": self.height,
            "crownsPath": os.path.abspath(self.crownsPath),
            "boundaryPath": os.path.abspath(self.areaPath),
            "crownCount": int(len(self.crowns)),
            "blocks": [{"name": name, "wkt": geometry.wkt}
                       for name, geometry in blocks],
            "tiles": index,
        }

    def build(self):
        ranges = measureRanges(self.specs, self.transform, self.width,
                               self.height, minHeight=self.minHeight)
        blocks = self.blocks()
        tiles, stride = self.tiles(blocks)
        index = self.writeTiles(tiles, ranges)
        perBlock = {}
        for entry in index:
            perBlock[entry["block"]] = perBlock.get(entry["block"], 0) + 1
        print("[data] tiles per block: %s" % perBlock)
        print("[data] %d tiles hold no crown (kept: background matters for "
              "precision)" % sum(1 for e in index if not e["crownCount"]))
        path = os.path.join(self.output, "dataset.json")
        _saveJson(self.metadata(ranges, stride, blocks, index), path)
        print("[data] wrote %s" % path)
        return path


def _saveJson(content, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(content, handle, indent=2, default=float)
    return path
