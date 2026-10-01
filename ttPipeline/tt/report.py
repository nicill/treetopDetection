"""
The results report, as an OpenDocument text file.

    python -m tt.report --lidarChm lidar.tif --p1Chm p1.tif \\
        --crowns crowns.shp --boundary area.shp --dataset ds/lidar \\
        --cv ccLidar=runs/ccLidar --cv mrcnnLidar=runs/mrcnnLidar ... \\
        --sweep ccLidar=sweepLidar.json --output report

Built from whatever evidence exists. Every run contributes its cross-validated
scores from results.json; runs that also saved predictions_<block>.json
contribute to the tree-by-tree sections and to the combinations. So the same
command produces a partial report before the learned models have been re-run
with prediction saving, and the complete one after.
"""

import argparse
import itertools
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

from . import reportFigures as figures
from .comparison import MethodRun, TreeComparison, matchBlock
from .dl import dlCommon as dc
from .dl import dlPrepare as dp
from .fusion import FusionCrossValidation, SaddleSurface
from .odt import OdtDocument
from .pseudoCrowns import PseudoCrowns
from .reportText import (HEIGHT_METHODS, HEIGHT_SOURCE, RGB_BOX_METHODS,
                         SADDLE_DROP, SECTIONS, label)
from .scene import Scene
from .tops import Tops

# ---------------------------------------------------------------------- #
# evidence
# ---------------------------------------------------------------------- #

class Evidence(object):
    """Everything the report states, computed once."""

    def __init__(self, args):
        self.args = args
        self.meta = dp.loadDataset(args.dataset)
        self.blocks = self.meta["blockGeometries"]
        self.scene = Scene(args.lidarChm, crownsPath=args.crowns,
                           boundaryPath=args.boundary, verbose=False)
        self.p1Scene = (Scene(args.p1Chm, crownsPath=args.crowns,
                              boundaryPath=args.boundary, verbose=False)
                        if args.p1Chm else None)
        self.cv = self._loadCv(args.cv)
        self.sweeps = {spec.split("=", 1)[0]: dc.loadJson(spec.split("=", 1)[1])
                       for spec in args.sweep}
        self.runs = self._loadPredictions(args.cv)
        self.comparison = (TreeComparison(self.scene, self.runs)
                           if self.runs else None)
        self.fusion = self._fusion()

    def _loadCv(self, specs):
        cv = {}
        for spec in specs:
            name, runDir = spec.split("=", 1)
            path = os.path.join(runDir, "results.json")
            if os.path.exists(path):
                cv[name] = dc.loadJson(path)
        return cv

    def _loadPredictions(self, specs):
        runs = {}
        for spec in specs:
            name, runDir = spec.split("=", 1)
            if name.endswith("Old"):
                continue
            try:
                runs[name] = MethodRun(name, runDir, self.blocks)
            except FileNotFoundError:
                pass
        return runs

    def _fusion(self):
        """
        Each mosaic detector with each height detector that has predictions,
        on the LiDAR or on the P1 CHM. Each combination's held-out output is
        saved beside the report, in the layout the runs use, so it can be
        rendered and compared like any run.
        """
        results = {}
        for box in RGB_BOX_METHODS:
            for height in HEIGHT_METHODS:
                if box in self.runs and height in self.runs:
                    results[(box, height)] = self._fusePair(box, height)
        return results

    def _fusePair(self, box, height):
        validation = FusionCrossValidation(
            self.scene.crowns, self.blocks, self.runs[box], self.runs[height],
            surface=self.surfaceFor(height), verbose=False)
        summary = validation.run()
        folder = os.path.join(self.args.output, "fused", "%s_%s" % (box, height))
        saveFusedOutputs(folder, validation.outputs)
        # the scores too, so results can be gathered across sites
        dc.saveJson(summary, os.path.join(folder, "summary.json"))
        return summary

    def surfaceFor(self, height):
        source = HEIGHT_SOURCE[height]
        scene = self.p1Scene if source == "p1" else self.scene
        if scene is None:
            return None
        return SaddleSurface.fromScene(scene, SADDLE_DROP[source])

    # ------------------------------------------------------------------ #

    def cvRows(self, include=None):
        rows = []
        for name, data in self.cv.items():
            if include and name not in include:
                continue
            pooled = data["pooled"]
            rows.append({"name": name, "label": label(name),
                         "short": label(name, True),
                         "recall": pooled["recall"],
                         "precision": pooled["precision"],
                         "f1": pooled["f1"], "std": pooled["perFoldF1Std"]})
        return sorted(rows, key=lambda r: -r["f1"])

    def foldF1(self, name):
        """{block suffix: F1} for one run."""
        return {f["block"][f["block"].rindex("_b"):]: f["f1"]
                for f in self.cv[name]["folds"]}

    def paired(self, names):
        rows = []
        for first, second in itertools.combinations(names, 2):
            a, b = self.foldF1(first), self.foldF1(second)
            shared = sorted(set(a) & set(b))
            difference = np.array([a[k] - b[k] for k in shared])
            p = wilcoxon(difference).pvalue if np.any(difference) else 1.0
            rows.append((label(first, True), label(second, True),
                         difference.mean(), int((difference > 0).sum()),
                         len(shared), p))
        return rows

    def sweepPoints(self, name, minimumRecall=0.90):
        rows = self.sweeps[name]
        best = max(rows, key=lambda r: r["f1"])
        recallRows = [r for r in rows if r["recall"] >= minimumRecall]
        highRecall = max(recallRows, key=lambda r: r["precision"]) \
            if recallRows else None
        ceiling = max(rows, key=lambda r: r["recall"])
        return best, highRecall, ceiling

    def pseudoCrowns(self, name):
        """Pseudo-crown outlines for a point method's held-out tops."""
        scene = self.p1Scene if (name.endswith("P1") and self.p1Scene) \
            else self.scene
        points = []
        for p in self.runs[name].predictions:
            column, row = scene.toPixel(p["centreX"], p["centreY"])
            column, row = int(column), int(row)
            points.append((column, row, float(scene.chm[row, column])))
        crowns = PseudoCrowns(scene, Tops(points), verbose=False)
        crowns.build()
        return crowns.polygons()

    def blockName(self, name):
        return matchBlock(name, dict(self.blocks))


# ---------------------------------------------------------------------- #
# the document
# ---------------------------------------------------------------------- #

class ResultsReport(object):

    def __init__(self, evidence, outputDir):
        self.e = evidence
        self.outputDir = outputDir
        self.figureDir = figures.ensureDirectory(os.path.join(outputDir,
                                                              "figures"))
        self.doc = OdtDocument("Treetop detection on the Ulaanbaatar site: "
                               "connected components and Mask R-CNN")

    def figurePath(self, name):
        return os.path.join(self.figureDir, name + ".png")

    def write(self):
        """
        Every section; one that fails is reported in its place and the rest
        still written, so an unattended run on a new site always ends with a
        document (and the combination scores, saved before any text).
        """
        for section in SECTIONS:
            try:
                section(self)
            except Exception as error:   # noqa: BLE001 — reported, not hidden
                message = "section %s failed: %s: %s" % (
                    section.__name__, type(error).__name__, error)
                print("[report] " + message, file=sys.stderr)
                self.doc.paragraph("[" + message + "]")
        return self.doc.save(os.path.join(self.outputDir, "report.odt"))


def saveFusedOutputs(directory, outputs):
    """One folder per strategy, one predictions_<block>.json per fold."""
    for strategy, perBlock in outputs.items():
        folder = os.path.join(directory, strategy)
        os.makedirs(folder, exist_ok=True)
        for block, predictions in perBlock.items():
            dc.saveJson({"block": block, "predictions": predictions},
                        os.path.join(folder, "predictions_%s.json" % block))


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(description="Write the results report.")
    parser.add_argument("--lidarChm", required=True)
    parser.add_argument("--p1Chm", default=None)
    parser.add_argument("--crowns", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--dataset", required=True,
                        help="Any prepared dataset of the area; only its "
                             "block geometry is used")
    parser.add_argument("--cv", action="append", default=[],
                        metavar="NAME=RUNDIR")
    parser.add_argument("--sweep", action="append", default=[],
                        metavar="NAME=SWEEP.json")
    parser.add_argument("--output", default="report")
    return parser.parse_args(argv)


def main(argv=None):
    args = parseArguments(argv)
    evidence = Evidence(args)
    path = ResultsReport(evidence, args.output).write()
    print("wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
