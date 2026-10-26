"""
TDData — one site: its rasters, its annotations and its cross-validation
blocks, plus the runs made on it.

A site holds any number of layers keyed by (kind, source): ("RGB", "RGB"),
("CHM", "LiDAR"), ("CHM", "P1"), ("DSM", "P1"), ... A detector asks for the
layer it needs; nothing outside this class handles paths.

Sites load from the dataset folders the pipeline already writes
(dataset.json), or are built from files directly.
"""

import json
import os

from shapely import wkt
from shapely.ops import unary_union

import numpy as np

from ..report.render import drawSite
from .annotations import Annotations
from .patches import ChannelSpec, PatchSet, makeBlocks
from .raster import HeightModel, LOCAL_HEIGHT_M, RGBMosaic


def guessHeightSource(path):
    """'LiDAR' or 'P1' from a file name, else 'unknown'."""
    name = os.path.basename(path).lower()
    if "lidar" in name:
        return "LiDAR"
    if "p1" in name or "photo" in name:
        return "P1"
    return "unknown"


class Block(object):
    """One cross-validation block: a named piece of the scored area."""

    def __init__(self, name, geometry):
        self.name = name
        self.geometry = geometry

    def __repr__(self):
        return "Block(%s)" % self.name


class TDData(object):

    def __init__(self, name, layers=None, annotations=None, blocks=None,
                 source=None):
        self.name = name
        self.layers = dict(layers or {})
        self.annotations = annotations
        self.blocks = list(blocks or [])
        self.source = source           # provenance text, kept for run files
        self.runs = []
        self.patchSet = None           # the site cut into tiles, if it has been
        self._surfaces = {}

    def __repr__(self):
        return "TDData(%s, layers %s, %d blocks)" % (
            self.name, sorted("%s/%s" % k for k in self.layers),
            len(self.blocks))

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    @classmethod
    def fromFiles(cls, name, chm=None, dsm=None, rgb=None, crowns=None,
                  area=None, heightSource=None, blocks=None,
                  crownShiftEastM=0.0, crownShiftSouthM=0.0):
        """
        A site from file paths. chm and dsm may be a path or a dict
        {source: path} for a site with several (e.g. {"LiDAR": a, "P1": b}).
        """
        layers = {}
        for kind, given in (("CHM", chm), ("DSM", dsm)):
            if not given:
                continue
            if not isinstance(given, dict):
                given = {heightSource or guessHeightSource(given): given}
            for source, path in given.items():
                layers[(kind, source)] = HeightModel(path, kind, source)
        if rgb:
            layers[("RGB", "RGB")] = RGBMosaic(rgb)
        crs = next(iter(layers.values())).crs if layers else None
        annotations = Annotations(crowns, area, crs, crownShiftEastM,
                                  crownShiftSouthM) if (crowns or area) \
            else None
        return cls(name, layers, annotations, blocks)

    @classmethod
    def fromDataset(cls, path, heightSource=None):
        """A site from a prepared dataset folder (its dataset.json)."""
        with open(os.path.join(path, "dataset.json")) as handle:
            meta = json.load(handle)
        layers = {}
        for channel in meta["channels"]:
            if channel["kind"] == "chm":
                source = heightSource or guessHeightSource(channel["path"])
                layers[("CHM", source)] = HeightModel(channel["path"], "CHM",
                                                      source)
            elif channel["kind"] == "rgb":
                layers[("RGB", "RGB")] = RGBMosaic(channel["path"])
        crs = next(iter(layers.values())).crs if layers else None
        annotations = Annotations(meta.get("crownsPath"),
                                  meta.get("boundaryPath"), crs)
        blocks = [Block(b["name"], wkt.loads(b["wkt"]))
                  for b in meta.get("blocks", [])]
        data = cls(meta["name"], layers, annotations, blocks,
                   source=meta.get("source"))
        data.datasetPath = path
        data.patchSet = PatchSet(path, meta)
        return data

    # ------------------------------------------------------------------ #
    # layers
    # ------------------------------------------------------------------ #

    def layer(self, kind, source=None):
        """
        The layer of this kind (and source). Without a source, the only layer
        of that kind; asking for one of several is an error, not a guess.
        """
        if source is not None:
            if (kind, source) not in self.layers:
                raise KeyError("%s has no %s from %s; it has %s"
                               % (self.name, kind, source, sorted(self.layers)))
            return self.layers[(kind, source)]
        matches = [v for (k, _), v in self.layers.items() if k == kind]
        if len(matches) != 1:
            raise KeyError("%s has %d %s layers (%s): name the source"
                           % (self.name, len(matches), kind,
                              sorted(self.layers)))
        return matches[0]

    def heightSurface(self, kind="CHM", source=None, resolution=0.25,
                      minHeight=2.0, smoothM=0.0, smoothSlope=0.0,
                      smoothWindowM=LOCAL_HEIGHT_M, resampling="average",
                      restrict=True):
        """
        A height model read for detection, restricted to the scored area
        unless restrict is False. Built once per setting and kept, because a
        parameter grid asks for the same few surfaces many times.
        """
        model = self.layer(kind, source)
        key = (kind, model.source, resolution, minHeight, smoothM,
               smoothSlope, smoothWindowM, resampling, restrict)
        if key not in self._surfaces:
            region = self.area if restrict else None
            self._surfaces[key] = model.surface(
                resolution, minHeight, smoothM, smoothSlope, smoothWindowM,
                resampling, region)
        return self._surfaces[key]

    # ------------------------------------------------------------------ #
    # annotations and blocks
    # ------------------------------------------------------------------ #

    @property
    def area(self):
        return None if self.annotations is None else self.annotations.area

    @property
    def crowns(self):
        if self.annotations is None:
            raise ValueError("%s has no annotations" % self.name)
        return self.annotations.crowns

    def block(self, name):
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError("%s has no block %s" % (self.name, name))

    @property
    def blockArea(self):
        """The union of every block: where cross-validation scores."""
        return unary_union([b.geometry for b in self.blocks])

    def trainingRegion(self, heldOut):
        """Every block but the held-out one."""
        return unary_union([b.geometry for b in self.blocks
                            if b.name != heldOut])

    def makeBlocks(self, columns, rows):
        """Cut the area into a grid of cross-validation blocks."""
        self.blocks = [Block(name, geometry) for name, geometry in
                       makeBlocks(self.area, columns, rows, areaName=self.name)]
        return self.blocks

    def patches(self, output, layers, resolution=0.05, tileSize=512,
                overlap=0.5, blockCols=4, blockRows=2, minHeight=2.0,
                minInsideFraction=0.5):
        """
        Cut the site into tiles of these layers, stacked in the order given
        (e.g. [("RGB", None), ("CHM", "LiDAR")] for the four-channel input),
        write them to `output` and keep them as this site's PatchSet. The
        blocks are cut with them.
        """
        specs = [ChannelSpec("rgb" if kind == "RGB" else "chm",
                             self.layer(kind, source).path)
                 for kind, source in layers]
        self.patchSet = PatchSet.build(
            output, specs, self.annotations.crownsPath,
            self.annotations.areaPath, self.name, resolution, tileSize,
            overlap, blockCols, blockRows, minHeight, minInsideFraction)
        self.blocks = [Block(name, geometry)
                       for name, geometry in self.patchSet.blocks]
        return self.patchSet

    def show(self, path, runs=None, region=None, kind="CHM", source=None,
             minHeight=2.0, minSidePx=1200):
        """
        Draw the site: a height layer in grey, the crowns outlined, and each
        run's detections as dots in its own colour. runs: {name: Detections
        or RunSet} (a RunSet shows its held-out detections); region: a
        polygon to draw (the whole layer by default). Returns the path.
        """
        surface = self.heightSurface(kind, source, minHeight=minHeight,
                                     restrict=False)
        crowns = None
        if self.annotations is not None and self.annotations.crownsPath:
            crowns = self.crowns if region is None \
                else self.annotations.crownsIn(region)
        blocks = {b.name: b.geometry for b in self.blocks}
        shown = {}
        for name, run in (runs or {}).items():
            detections = run.heldOut(blocks) if hasattr(run, "heldOut") \
                else run
            shown[name] = (detections if region is None
                           else detections.inRegion(region))
        window = None
        if region is not None:
            minX, minY, maxX, maxY = region.bounds
            c0, r0 = surface.toPixel(minX, maxY)
            c1, r1 = surface.toPixel(maxX, minY)
            rows, columns = surface.shape
            window = (max(0, int(c0)), max(0, int(r0)),
                      min(columns, int(np.ceil(c1))),
                      min(rows, int(np.ceil(r1))))
        return drawSite(surface, crowns, shown, path, window, minSidePx)

    def addRun(self, detections):
        self.runs.append(detections)
        return detections
