"""
One image per held-out block for every result: each run, and each
combination's output.

    python -m tt.report.blockImages --chm lidar.tif --crowns crowns.shp \\
        --boundary area.shp --dataset ds/lidar \\
        --run ccLidar=runs/ccLidar --run mrcnnRgb=runs/mrcnnRgb \\
        --fused report/fused --output images

Colour code, on the CHM:

    green    a hit: the detection a crown is credited with
    orange   a repeat: a further detection in a crown already hit
    blue     outside every crown, at canopy height: likely an unannotated tree
    red      outside every crown and below the local canopy
    yellow   the outline of a crown nothing hit

Each block shows only what was predicted there by a model that never saw it:
a run's held-out predictions at its own operating point, or a combination's
held-out output. Images already on disk are skipped, so rerunning after each
stage draws only what is new.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
from shapely import contains_xy

from ..analysis import Analysis
from ..analysis.trees import heightsNear
from ..analysis.statistics import splitBackground
from ..data.site import Block, TDData
from ..dl import dlPrepare as dp
from ..experiments import RunSet
from ..scoring.evaluator import assignToCrowns
from .render import TileRenderer

FATE_COLOURS = {"hit": (0, 190, 0), "repeat": (0, 150, 255),     # BGR
                "canopy": (255, 90, 0), "low": (0, 0, 230)}
MISSED = (0, 255, 255)
JPEG_QUALITY = 88


class BlockRenderer(object):
    """Every block of a run drawn: fates coloured, missed crowns outlined."""

    def __init__(self, analysis, blocks, minSidePx=900):
        self.analysis = analysis
        self.surface = analysis.surface
        self.blocks = dict(blocks)
        self.renderer = TileRenderer(analysis.surface, analysis.crowns,
                                     minSidePx=minSidePx)
        centroids = analysis.crowns.geometry.centroid
        self.centroidX = centroids.x.to_numpy()
        self.centroidY = centroids.y.to_numpy()

    def pixelBounds(self, block):
        minX, minY, maxX, maxY = self.blocks[block].bounds
        c0, r0 = self.surface.toPixel(minX, maxY)
        c1, r1 = self.surface.toPixel(maxX, minY)
        rows, columns = self.surface.chm.shape
        return (int(max(0, c0)), int(max(0, r0)),
                int(min(columns, c1)), int(min(rows, r1)))

    def classify(self, detections, block):
        """Fates of the block's detections, and the block's missed crowns."""
        inBlock = np.nonzero(contains_xy(self.blocks[block], self.centroidX,
                                         self.centroidY))[0]
        crowns = self.analysis.crowns.iloc[inBlock].reset_index(drop=True)
        xs, ys = detections.x, detections.y
        kinds, hit, _ = assignToCrowns(xs, ys, detections.scores, crowns)
        heights = heightsNear(self.surface, xs, ys)
        split = splitBackground(xs, ys, heights, kinds)
        fates = list(kinds)
        for index in split["canopy"]:
            fates[index] = "canopy"
        for index in split["low"]:
            fates[index] = "low"
        missed = [int(inBlock[i]) for i in range(len(inBlock)) if i not in hit]
        return fates, missed, len(inBlock)

    def render(self, detections, block, title, path):
        fates, missed, crownCount = self.classify(detections, block)
        c0, r0, c1, r1 = self.pixelBounds(block)
        image, scale = self.renderer.background(c0, r0, c1, r1)
        self.renderer.drawCrowns(image, missed, c0, r0, scale, MISSED)
        radius = max(4, int(0.45 / self.surface.pixelSize * scale))
        for x, y, fate in zip(detections.x, detections.y, fates):
            if fate not in FATE_COLOURS:
                continue
            column, row = self.surface.toPixel(x, y)
            cv2.circle(image, (int((column - c0) * scale),
                               int((row - r0) * scale)),
                       radius, FATE_COLOURS[fate], -1)
        hits = fates.count("hit")
        caption = ("%s | %s | R %.2f P %.2f | hit %d, repeat %d, canopy %d, "
                   "low %d, missed %d"
                   % (title, block, hits / float(max(crownCount, 1)),
                      hits / float(max(len(fates), 1)), hits,
                      fates.count("repeat"), fates.count("canopy"),
                      fates.count("low"), len(missed)))
        self.renderer.caption(image, caption)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return path

    def renderRun(self, name, runDir, outputDir):
        """Every block of one run; returns how many images were drawn."""
        held = RunSet.load(runDir).heldOut(self.blocks)
        drawn = 0
        for block in sorted(self.blocks):
            path = os.path.join(outputDir, "%s.jpg" % block)
            if os.path.exists(path):
                continue
            self.render(held.subset(held.cache["block"] == block), block,
                        name, path)
            drawn += 1
        return drawn


def fusedRuns(fusedDir):
    """(name, directory) for every combination output worth drawing."""
    found = []
    for folder in sorted(glob.glob(os.path.join(fusedDir, "*", "*"))):
        strategy = os.path.basename(folder)
        if strategy in ("boxes", "points"):
            continue        # identical to the runs themselves
        pair = os.path.basename(os.path.dirname(folder))
        found.append(("%s %s" % (pair, strategy), folder,
                      os.path.join(pair, strategy)))
    return found


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Draw every held-out block of every result.")
    parser.add_argument("--chm", required=True,
                        help="The CHM drawn underneath (the LiDAR one reads "
                             "best)")
    parser.add_argument("--crowns", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--dataset", required=True,
                        help="Any prepared dataset; only its blocks are used")
    parser.add_argument("--run", action="append", default=[],
                        metavar="NAME=RUNDIR")
    parser.add_argument("--fused", default=None,
                        help="A report's fused/ folder of combination outputs")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parseArguments(argv)
    blocks = dp.loadDataset(args.dataset)["blockGeometries"]
    site = TDData.fromFiles("site", chm=args.chm, crowns=args.crowns,
                            area=args.boundary,
                            blocks=[Block(n, g) for n, g in blocks])
    renderer = BlockRenderer(Analysis(site), blocks)
    jobs = [(spec.split("=", 1)[0], spec.split("=", 1)[1],
             spec.split("=", 1)[0]) for spec in args.run]
    if args.fused and os.path.isdir(args.fused):
        jobs += [(name, folder, os.path.join("fused", sub))
                 for name, folder, sub in fusedRuns(args.fused)]
    total = 0
    for name, folder, sub in jobs:
        try:
            drawn = renderer.renderRun(name, folder,
                                       os.path.join(args.output, sub))
        except FileNotFoundError as error:
            print("[images] %s skipped: %s" % (name, error))
            continue
        total += drawn
        if drawn:
            print("[images] %s: %d block images" % (name, drawn))
    print("[images] %d new images under %s" % (total, args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
