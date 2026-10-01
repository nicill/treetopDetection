"""
Calibrating the connected-component detector from the RGB Mask R-CNN.

The RGB model's high-confidence boxes are used as automatic testing zones.
Inside a zone the detector must behave correctly: find the tree, and find it
once. Outside every zone it is free — that is where it is expected to find what
the RGB model did not — and nothing there is judged.

Per block and per confidence level:

  1. zones   the RGB predictions at or above the confidence level, each with
             its apex: the highest CHM point inside the box
  2. propose each parameter from a rule on the zones (ParameterProposal)
  3. detect  connected components as usual, saddle merge included
  4. check   coverage (zones holding a top) and multiplicity (zones holding
             two or more), inside the zones only; if a check fails, adjust the
             parameter it points to, once, and detect again
  5. merge   tops inside the same zone collapse to the highest

The rules' constants were fixed before any calibration result existed and are
collected below. Changing them after seeing results would turn the proposal
into a fit to this site.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon

from .comparison import MethodRun
from .detector import ConCompDetector
from .dl import dlCommon as dc
from .dl import dlPrepare as dp
from .fusion import mergeWithin, overlapOverSmaller
from .merging import TopMerger, saddleDrop
from .scene import Scene

# --- the rules' constants, fixed in advance -------------------------------- #
APEX_COVER = 0.95          # zone apexes the percentile cut must keep
APEX_COVER_ADJUSTED = 0.99  # what it must keep after a coverage adjustment
SMALL_ZONE_QUANTILE = 10   # "the smallest zones": their 10th percentile
EROSION_WIDTH_SHARE = 0.25  # total width erosion may strip from a small zone
EROSION_MAX = 2
MIN_TOP_FRACTION = 0.02    # min top area as a share of a small zone's area
NEIGHBOUR_DIP_QUANTILE = 10  # saddle dip below 90% of between-zone dips
NEIGHBOUR_DIP_CAP_QUANTILE = 25  # an adjusted dip stays below 75% of them
WITHIN_DIP_QUANTILE = 90   # adjusted dip covers 90% of within-zone dips
SAME_TREE_OVERLAP = 0.25   # zones overlapping more than this are one tree
DIP_FLOOR_M = 0.1          # CHM noise: no dip below this counts as a valley
COVERAGE_REQUIRED = 0.95
MULTIPLICITY_ALLOWED = 0.10

# --- fixed detector settings, not proposed --------------------------------- #
# (defaults; the resolution, minimum height and minimum tree area can be set on
# the command line, to match the connected-component run on another site)
WINDOW_M = 40.0
WINDOW_OVERLAP = 0.2
MIN_TREE_AREA_M2 = 0.5
MERGE_RADIUS_M = 8.0
POINT_BOX_M = 0.5
CONFIDENCE_LEVELS = (0.5, 0.7, 0.9)


class ConfidentZones(object):
    """The high-confidence RGB boxes of one block, on the CHM."""

    def __init__(self, predictions, scene, threshold, region):
        self.scene = scene
        kept = [p for p in dc.predictionsInRegion(predictions, region)
                if p["score"] >= threshold]
        self.boxes = np.array([p["box"] for p in kept], float).reshape(-1, 4)
        self.scores = np.array([p["score"] for p in kept], float)
        self.predictions = kept
        self.apexes = [self._apex(box) for box in self.boxes]

    def __len__(self):
        return len(self.boxes)

    def _pixelBox(self, box):
        c0, r0 = self.scene.toPixel(box[0], box[3])
        c1, r1 = self.scene.toPixel(box[2], box[1])
        rows, columns = self.scene.chm.shape
        r0, r1 = sorted((int(np.clip(r0, 0, rows - 1)),
                         int(np.clip(r1, 0, rows - 1))))
        c0, c1 = sorted((int(np.clip(c0, 0, columns - 1)),
                         int(np.clip(c1, 0, columns - 1))))
        return r0, c0, r1, c1

    def _apex(self, box):
        r0, c0, r1, c1 = self._pixelBox(box)
        heights = self.scene.chm[r0:r1 + 1, c0:c1 + 1]
        row, column = np.unravel_index(int(np.argmax(heights)), heights.shape)
        return r0 + int(row), c0 + int(column), float(heights[row, column])

    @property
    def widthsM(self):
        return np.minimum(self.boxes[:, 2] - self.boxes[:, 0],
                          self.boxes[:, 3] - self.boxes[:, 1])

    @property
    def areasM2(self):
        return (self.boxes[:, 2] - self.boxes[:, 0]) * \
            (self.boxes[:, 3] - self.boxes[:, 1])

    def neighbourDips(self):
        """
        Dips between the apexes of neighbouring zones, taken to be two trees.

        Zones overlapping by more than SAME_TREE_OVERLAP of the smaller are
        skipped: two confident boxes that much on top of each other are more
        likely one crown split than two neighbours.
        """
        if len(self) < 2:
            return np.array([])
        points = np.array([(r, c) for r, c, _ in self.apexes], float)
        radius = MERGE_RADIUS_M / self.scene.pixelSize
        dips = []
        for i, j in cKDTree(points).query_pairs(radius):
            if overlapOverSmaller(self.boxes[i], self.boxes[j:j + 1])[0] \
                    > SAME_TREE_OVERLAP:
                continue
            a, b = self.apexes[i], self.apexes[j]
            dips.append(saddleDrop(self.scene.chm, a[0], a[1], b[0], b[1]))
        return np.array(dips)

    def apexRanks(self):
        """Each apex's percentile rank among its detector window's heights."""
        size = self.scene.metresToPixels(WINDOW_M, 8)
        step = max(1, int(size * (1.0 - WINDOW_OVERLAP)))
        rows, columns = self.scene.chm.shape
        ranks = []
        for row, column, height in self.apexes:
            r0 = int(np.clip((row // step) * step, 0, max(0, rows - size)))
            c0 = int(np.clip((column // step) * step, 0,
                             max(0, columns - size)))
            window = self.scene.chm[r0:r0 + size, c0:c0 + size]
            values = window[window > 0]
            ranks.append(100.0 * np.mean(values <= height)
                         if values.size else 100.0)
        return np.array(ranks)


class ParameterProposal(object):
    """Connected-component parameters proposed from the zones."""

    def __init__(self, zones):
        self.zones = zones
        self.ranks = zones.apexRanks()
        self.betweenDips = zones.neighbourDips()

    def lowerPercentile(self, cover=APEX_COVER):
        if not self.ranks.size:
            return 20
        cut = np.percentile(self.ranks, 100.0 * (1.0 - cover))
        return int(np.clip(np.floor(cut), 1, 50))

    def erosionIterations(self):
        if not len(self.zones):
            return 1
        smallWidth = np.percentile(self.zones.widthsM, SMALL_ZONE_QUANTILE)
        stripped = EROSION_WIDTH_SHARE * smallWidth / self.zones.scene.pixelSize
        return int(np.clip(np.floor(stripped / 2.0), 0, EROSION_MAX))

    def minTopAreaM2(self):
        if not len(self.zones):
            return 0.12
        smallArea = np.percentile(self.zones.areasM2, SMALL_ZONE_QUANTILE)
        floor = self.zones.scene.pixelSize ** 2
        return float(max(floor, MIN_TOP_FRACTION * smallArea))

    def saddleDropM(self):
        if not self.betweenDips.size:
            return 0.3
        return float(max(DIP_FLOOR_M, np.percentile(
            self.betweenDips, NEIGHBOUR_DIP_QUANTILE)))

    def dropCapM(self):
        if not self.betweenDips.size:
            return 0.5
        return float(max(DIP_FLOOR_M, np.percentile(
            self.betweenDips, NEIGHBOUR_DIP_CAP_QUANTILE)))

    def settings(self):
        drop = self.saddleDropM()
        return {"lowerPercentile": self.lowerPercentile(),
                "erosionIterations": self.erosionIterations(),
                "minTopAreaM2": self.minTopAreaM2(),
                "saddleDropM": drop, "topStepM": topStep(drop)}


def topStep(drop):
    """Half the saddle dip, so the descent resolves the gap between trees."""
    return float(np.clip(drop / 2.0, 0.05, 0.5))


def buildDetector(settings, minTreeAreaM2=MIN_TREE_AREA_M2):
    return ConCompDetector(
        windowSizeM=WINDOW_M, windowOverlap=WINDOW_OVERLAP,
        minTreeAreaM2=minTreeAreaM2,
        lowerPercentile=settings["lowerPercentile"],
        minTopAreaM2=settings["minTopAreaM2"], topStepM=settings["topStepM"],
        erosionIterations=settings["erosionIterations"],
        merger=TopMerger("saddle", epsM=MERGE_RADIUS_M,
                         saddleDropM=settings["saddleDropM"]),
        verbose=False)


# ---------------------------------------------------------------------- #

def asPredictions(tops, scene):
    """Tops as the prediction dicts the scorer and the merge expect."""
    out = []
    for x, y, height in tops.points:
        east, north = scene.toWorld(x, y)
        out.append({"centreX": float(east), "centreY": float(north),
                    "score": float(height),
                    "box": [east - POINT_BOX_M, north - POINT_BOX_M,
                            east + POINT_BOX_M, north + POINT_BOX_M]})
    return out


class ZoneCheck(object):
    """Coverage and multiplicity of detections inside the zones."""

    def __init__(self, predictions, zones):
        self.zones = zones
        self.members = self._members(predictions)
        self.predictions = predictions

    def _members(self, predictions):
        """Each zone's detections; a detection in several goes to the best."""
        members = [[] for _ in range(len(self.zones))]
        if not predictions or not len(self.zones):
            return members
        xs = np.array([p["centreX"] for p in predictions])[:, None]
        ys = np.array([p["centreY"] for p in predictions])[:, None]
        b = self.zones.boxes
        inside = ((xs >= b[None, :, 0]) & (xs <= b[None, :, 2])
                  & (ys >= b[None, :, 1]) & (ys <= b[None, :, 3]))
        owners = np.argmax(np.where(inside, self.zones.scores[None, :],
                                    -np.inf), axis=1)
        for index, owner in enumerate(owners):
            if inside[index].any():
                members[owner].append(index)
        return members

    @property
    def coverage(self):
        counts = [len(m) for m in self.members]
        return float(np.mean([c >= 1 for c in counts])) if counts else 1.0

    @property
    def multiplicity(self):
        counts = [len(m) for m in self.members]
        return float(np.mean([c >= 2 for c in counts])) if counts else 0.0

    def uncovered(self):
        return [z for z, m in enumerate(self.members) if not m]

    def withinDips(self, scene):
        """Dips between each zone's best detection and its other ones."""
        dips = []
        for members in self.members:
            if len(members) < 2:
                continue
            ordered = sorted(members,
                             key=lambda i: -self.predictions[i]["score"])
            first = self._pixel(self.predictions[ordered[0]], scene)
            for index in ordered[1:]:
                other = self._pixel(self.predictions[index], scene)
                dips.append(saddleDrop(scene.chm, first[0], first[1],
                                       other[0], other[1]))
        return np.array(dips)

    @staticmethod
    def _pixel(prediction, scene):
        column, row = scene.toPixel(prediction["centreX"],
                                    prediction["centreY"])
        return int(row), int(column)

    def summary(self):
        return {"zones": len(self.zones), "coverage": self.coverage,
                "multiplicity": self.multiplicity}


# ---------------------------------------------------------------------- #

class CalibratedDetector(object):
    """Steps 2 to 5 for one set of zones."""

    def __init__(self, scene, zones, minTreeAreaM2=MIN_TREE_AREA_M2):
        self.scene = scene
        self.zones = zones
        self.minTreeAreaM2 = minTreeAreaM2
        self.proposal = ParameterProposal(zones)
        self.log = {}

    def detect(self, settings):
        tops = buildDetector(settings, self.minTreeAreaM2).detect(self.scene)
        return asPredictions(tops, self.scene)

    def adjust(self, settings, check):
        """At most one change per failed check; returns the new settings."""
        changed, notes = dict(settings), []
        if check.coverage < COVERAGE_REQUIRED:
            notes.append(self._adjustCoverage(changed, check))
        if check.multiplicity > MULTIPLICITY_ALLOWED:
            notes.append(self._adjustMultiplicity(changed, check))
        return changed, [n for n in notes if n]

    def _adjustCoverage(self, settings, check):
        uncovered = check.uncovered()
        ranks = self.proposal.ranks[uncovered]
        belowCut = np.mean(ranks < settings["lowerPercentile"]) \
            if ranks.size else 0.0
        if belowCut > 0.5:
            settings["lowerPercentile"] = self.proposal.lowerPercentile(
                APEX_COVER_ADJUSTED)
            return "coverage: lowered the cut to %d" % \
                settings["lowerPercentile"]
        if settings["erosionIterations"] > 0:
            settings["erosionIterations"] -= 1
            return "coverage: erosion to %d" % settings["erosionIterations"]
        settings["minTopAreaM2"] /= 2.0
        return "coverage: min top area to %.3f" % settings["minTopAreaM2"]

    def _adjustMultiplicity(self, settings, check):
        dips = check.withinDips(self.scene)
        if not dips.size:
            return None
        target = min(np.percentile(dips, WITHIN_DIP_QUANTILE),
                     self.proposal.dropCapM())
        if target <= settings["saddleDropM"]:
            return "multiplicity: dip already at its cap, not changed"
        settings["saddleDropM"] = float(target)
        settings["topStepM"] = topStep(target)
        return "multiplicity: saddle dip to %.2f" % target

    def run(self):
        settings = self.proposal.settings()
        predictions = self.detect(settings)
        before = ZoneCheck(predictions, self.zones)
        adjusted, notes = self.adjust(settings, before)
        if notes:
            predictions = self.detect(adjusted)
        after = ZoneCheck(predictions, self.zones)
        self.log = {"proposed": settings, "used": adjusted,
                    "adjustments": notes, "checkBefore": before.summary(),
                    "checkAfter": after.summary()}
        return finalMerge(predictions, self.zones)


def finalMerge(predictions, zones):
    """Tops inside the same zone collapse to the highest (step 5)."""
    boxes = [{"box": list(b), "score": float(s)}
             for b, s in zip(zones.boxes, zones.scores)]
    return mergeWithin(boxes, predictions)


# ---------------------------------------------------------------------- #

class CalibrationExperiment(object):
    """
    The per-block test. Each block's held-out RGB predictions, made by a model
    that never saw the block, drive the zones; the result is scored against
    that block's real crowns, which the calibration never sees.
    """

    def __init__(self, scene, blocks, rgbRun, ccRun,
                 levels=CONFIDENCE_LEVELS, minTreeAreaM2=MIN_TREE_AREA_M2):
        self.scene = scene
        self.blocks = dict(blocks)
        self.rgbRun = rgbRun
        self.ccRun = ccRun
        self.levels = levels
        self.minTreeAreaM2 = minTreeAreaM2

    def _score(self, predictions, block):
        return dc.evaluateDetections(predictions, self.scene.crowns,
                                     self.blocks[block])

    def baselines(self, block, zones):
        """CC at its own cross-validated setting, alone and merged."""
        cc = self.ccRun.record(block)["predictions"]
        return {"cc": self._score(cc, block),
                "ccMerged": self._score(finalMerge(cc, zones), block)}

    def runBlock(self, block):
        region = self.blocks[block]
        rgb = self.rgbRun.record(block)["predictions"]
        rows = {}
        for level in self.levels:
            zones = ConfidentZones(rgb, self.scene, level, region)
            calibrated = CalibratedDetector(self.scene, zones,
                                            self.minTreeAreaM2)
            result = self._score(calibrated.run(), block)
            result.update(calibrated.log)
            rows[level] = dict(self.baselines(block, zones),
                               calibrated=result, zones=len(zones))
        return rows

    def run(self, verbose=True):
        results = {}
        for block in sorted(self.blocks):
            results[block] = self.runBlock(block)
            if verbose:
                printBlock(block, results[block])
        return results


def printBlock(block, rows):
    for level, row in rows.items():
        c = row["calibrated"]
        print("  %s  conf %.1f  %3d zones | CC %.3f  CC+merge %.3f  "
              "calibrated %.3f (R %.3f P %.3f)  %s"
              % (block, level, row["zones"], row["cc"]["f1"],
                 row["ccMerged"]["f1"], c["f1"], c["recall"], c["precision"],
                 "; ".join(c["adjustments"]) or "no adjustment"))


def pooled(results, level, key):
    folds = [results[b][level][key] for b in sorted(results)]
    return dc.averageFolds(folds), np.array([f["f1"] for f in folds])


def summarise(results, levels):
    table = {}
    for level in levels:
        cc, ccF1 = pooled(results, level, "cc")
        merged, mergedF1 = pooled(results, level, "ccMerged")
        calibrated, calF1 = pooled(results, level, "calibrated")
        table[level] = {"cc": cc, "ccMerged": merged, "calibrated": calibrated,
                        "calibratedVsCc": pairedStats(calF1 - ccF1),
                        "mergedVsCc": pairedStats(mergedF1 - ccF1),
                        "adjustedBlocks": sum(
                            1 for b in results
                            if results[b][level]["calibrated"]["adjustments"])}
    return table


def pairedStats(difference):
    p = wilcoxon(difference).pvalue if np.any(difference) else 1.0
    return {"meanDifference": float(difference.mean()),
            "wins": int((difference > 0).sum()), "blocks": len(difference),
            "p": float(p)}


def printSummary(table):
    print("\n%-6s %-12s %7s %7s %7s   %s" % ("conf", "", "recall", "prec",
                                            "F1", "vs CC (mean, wins, p)"))
    for level, row in table.items():
        for key, name in (("cc", "CC"), ("ccMerged", "CC + merge"),
                          ("calibrated", "calibrated")):
            p = row[key]
            versus = {"ccMerged": row["mergedVsCc"],
                      "calibrated": row["calibratedVsCc"]}.get(key)
            tail = ("%+.3f, %d/%d, p=%.2f" % (versus["meanDifference"],
                                             versus["wins"], versus["blocks"],
                                             versus["p"]) if versus else "")
            print("%-6.1f %-12s %7.3f %7.3f %7.3f   %s"
                  % (level, name, p["recall"], p["precision"], p["f1"], tail))
        print("       adjusted on %d of %d blocks" % (row["adjustedBlocks"],
                                                     row["calibratedVsCc"]
                                                     ["blocks"]))


# ---------------------------------------------------------------------- #

def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Calibrate connected components from RGB Mask R-CNN "
                    "high-confidence zones, block by block.")
    parser.add_argument("--chm", required=True,
                        help="The CHM connected components runs on")
    parser.add_argument("--crowns", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--dataset", required=True,
                        help="Any prepared dataset; only its blocks are used")
    parser.add_argument("--rgbRun", required=True,
                        help="Run directory of the RGB Mask R-CNN")
    parser.add_argument("--ccRun", required=True,
                        help="Connected-component run on the same CHM, for "
                             "the baselines")
    parser.add_argument("--output", default="calibration")
    # fixed detector settings, not proposed: set them as in the connected-
    # component run the calibration is compared with (tt.dl concomp)
    parser.add_argument("--resolution", type=float, default=0.25,
                        help="CHM resolution the detector works at")
    parser.add_argument("--minHeight", type=float, default=2.0)
    parser.add_argument("--minTreeArea", type=float, default=MIN_TREE_AREA_M2)
    return parser.parse_args(argv)


def main(argv=None):
    args = parseArguments(argv)
    blocks = dp.loadDataset(args.dataset)["blockGeometries"]
    scene = Scene(args.chm, crownsPath=args.crowns,
                  boundaryPath=args.boundary, resolution=args.resolution,
                  minHeight=args.minHeight, verbose=False)
    experiment = CalibrationExperiment(
        scene, blocks, MethodRun("rgb", args.rgbRun, blocks),
        MethodRun("cc", args.ccRun, blocks),
        minTreeAreaM2=args.minTreeArea)
    results = experiment.run()
    table = summarise(results, experiment.levels)
    printSummary(table)
    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, "calibration.json")
    with open(path, "w") as handle:
        json.dump({"blocks": {b: {str(k): v for k, v in rows.items()}
                              for b, rows in results.items()},
                   "summary": {str(k): v for k, v in table.items()}},
                  handle, indent=1, default=float)
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
