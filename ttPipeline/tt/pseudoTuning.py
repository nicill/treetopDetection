"""
Can the RGB Mask R-CNN stand in for the crowns when tuning connected
components? Measured before anything is built on it.

    python -m tt.pseudoTuning --ccRun <site>/runs/ccP1 --rgbRun <site>/runs/mrcnnRgb \\
        --dataset <site>/ds/chm --jobs 16 --output <site>/pseudoTuning

For each held-out block, every setting of the connected-component run's grid
is scored against:

  real       the block's annotated crowns (what tuning on the block itself
             would pick: an optimistic bound, never a method)
  pseudoF1   the RGB Mask R-CNN's held-out boxes for the block, above the
             operating point its own cross-validation chose, as crowns. A
             tree the network missed counts against a setting that finds it.
  zones@c    the network's boxes at confidence c or more, as in the
             calibration: the share holding exactly one top. Nothing outside
             a box counts, so finding trees the network missed is free.

The network made those predictions without seeing the block, and none of the
pseudo scorers looks at the block's crowns. For each scorer the setting it
picks is then scored against the real crowns, pooled over blocks like any
fold, and its ranking of all settings is compared with the real ranking
(Spearman). The connected-component run's own cross-validated score, tuned on
the other blocks' real crowns, is the target an automatic tuning has to
reach.

The detections of every setting are taken from the run's detections.npz
(tt.dl concomp --saveDetections) or, when it has none, recomputed from the
grid the run recorded and cached in --output.
"""

import argparse
import multiprocessing
import os
import sys

import geopandas as gpd
import numpy as np
from scipy.stats import spearmanr
from shapely.geometry import box

from .comparison import MethodRun, sharedBlocks
from .dl import dlCommon as dc
from .dl import dlPrepare as dp
from .dl.dlConComp import (ConCompCrossValidation, gridFromArguments,
                           loadDetections, saveDetections)

ZONE_LEVELS = (0.5, 0.7, 0.9)
# the scorer whose choices become a run of their own (runs/ccAuto<height>),
# fixed before any result: plain F1 (or the tuning objective) against the boxes
AUTO_SCORER = "pseudoF1"
GRID_OPTIONS = ("percentiles", "minTopAreas", "topSteps", "erosions",
                "saddleDrops", "minHeights", "minTreeAreas", "mergeMetrics",
                "smooths")


def runGrid(ccRun):
    """The grid and fixed settings the connected-component run used."""
    recorded = dc.loadJson(os.path.join(ccRun, "results.json"))["settings"]
    options = argparse.Namespace(**{k: recorded.get(k) for k in GRID_OPTIONS})
    return gridFromArguments(options), recorded


def detections(args, validation):
    """Every setting's detections: the run's, the cache, or computed now."""
    for path in (os.path.join(args.ccRun, "detections.npz"),
                 os.path.join(args.output, "detections.npz")):
        if os.path.exists(path):
            print("[pseudo] detections from %s" % path)
            return loadDetections(path)
    print("[pseudo] computing %d settings, %d jobs"
          % (len(validation.settings), validation.jobs))
    found = validation.detectAll()
    saveDetections(os.path.join(args.output, "detections.npz"), found)
    return found


def pseudoCrowns(predictions, crs):
    return gpd.GeoDataFrame(geometry=[box(*p["box"]) for p in predictions],
                            crs=crs)


def exactlyOneShare(points, boxes):
    """Share of boxes holding exactly one point."""
    if not len(boxes):
        return 0.0
    if not len(points):
        return 0.0
    x, y = points[:, 0][None, :], points[:, 1][None, :]
    inside = ((x >= boxes[:, 0:1]) & (x <= boxes[:, 2:3]) &
              (y >= boxes[:, 1:2]) & (y <= boxes[:, 3:4]))
    return float((inside.sum(axis=1) == 1).mean())


_BLOCK = None      # the block being scored, shared with forked workers


def _scoreReal(index):
    return dc.evaluateDetections(_BLOCK.inBlock[index], _BLOCK.crowns,
                                 _BLOCK.region)


def _scorePseudo(index):
    return dc.objectiveOf(dc.evaluateDetections(_BLOCK.inBlock[index],
                                                _BLOCK.pseudo, _BLOCK.region))


class BlockScores(object):
    """One held-out block: every setting under every scorer."""

    def __init__(self, block, region, crowns, rgbRecord, allDetections,
                 jobs=1):
        self.block = block
        self.region = region
        self.crowns = crowns
        self.jobs = max(1, int(jobs))
        self.inBlock = [dc.predictionsInRegion(d, region) for d in allDetections]
        held = dc.predictionsInRegion(rgbRecord["predictions"], region)
        threshold = rgbRecord.get("threshold", -np.inf)
        self.pseudo = pseudoCrowns([p for p in held if p["score"] >= threshold],
                                   crowns.crs)
        self.zones = {c: np.array([p["box"] for p in held if p["score"] >= c],
                                  float).reshape(-1, 4) for c in ZONE_LEVELS}

    def _map(self, function):
        """function(i) for every setting, --jobs at a time; same results."""
        global _BLOCK
        _BLOCK = self
        try:
            indices = range(len(self.inBlock))
            if self.jobs == 1:
                return [function(i) for i in indices]
            with multiprocessing.get_context("fork").Pool(self.jobs) as pool:
                return pool.map(function, indices, chunksize=max(
                    1, len(self.inBlock) // (self.jobs * 8)))
        finally:
            _BLOCK = None

    def realScores(self):
        if not hasattr(self, "_real"):
            self._real = self._map(_scoreReal)
        return self._real

    def scorers(self):
        """{scorer: score per setting}, none of them using the crowns."""
        points = [np.array([[p["centreX"], p["centreY"]] for p in d],
                           float).reshape(-1, 2) for d in self.inBlock]
        out = {"pseudoF1": self._map(_scorePseudo)}
        for c, boxes in self.zones.items():
            out["zones@%.1f" % c] = [exactlyOneShare(p, boxes) for p in points]
        return out


def tuneBlock(scores):
    """
    Per scorer: the chosen setting's real result and the rank agreement;
    None for a block without crowns (it has nothing to score).
    """
    real = scores.realScores()
    if not real or not real[0]["crowns"]:
        return None
    realF1 = np.array([dc.objectiveOf(r) for r in real])
    rows = {"real": dict(real[int(np.argmax(realF1))], spearman=1.0,
                         chosen=int(np.argmax(realF1)))}
    for name, values in scores.scorers().items():
        values = np.array(values)
        chosen = int(np.argmax(values))
        rho = spearmanr(values, realF1).correlation \
            if np.ptp(values) > 0 and np.ptp(realF1) > 0 else float("nan")
        rows[name] = dict(real[chosen], spearman=float(rho), chosen=chosen)
    return rows


def pool(perBlock, name):
    folds = [rows[name] for rows in perBlock.values()]
    pooled = dc.averageFolds(folds)
    rhos = [f["spearman"] for f in folds if np.isfinite(f["spearman"])]
    pooled["meanSpearman"] = float(np.mean(rhos)) if rhos else float("nan")
    return pooled


def printTable(table, target):
    print("\n%-26s %7s %7s %7s %10s" % ("chosen by", "recall", "prec", "F1",
                                         "Spearman"))
    print("%-26s %7.3f %7.3f %7.3f %10s" % (
        "CV on real crowns (target)", target["recall"], target["precision"],
        target["f1"], "-"))
    for name, p in table.items():
        label = "own block's crowns (bound)" if name == "real" else name
        print("%-26s %7.3f %7.3f %7.3f %10.2f" % (
            label, p["recall"], p["precision"], p["f1"], p["meanSpearman"]))


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Tune connected components on the RGB Mask R-CNN's "
                    "boxes instead of the crowns, and measure how well that "
                    "works.")
    parser.add_argument("--ccRun", required=True)
    parser.add_argument("--rgbRun", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--autoRun", default=None,
                        help="where the automatic run goes (default: beside "
                             "--ccRun, as runs/ccAuto<height>)")
    return parser.parse_args(argv)


def autoFromSaved(args):
    """
    The automatic run from a finished pseudo-tuning: the choices recorded in
    pseudoTuning.json and the detections saved with the run (or cached),
    without scoring anything again. False when there is nothing to use.
    """
    saved = os.path.join(args.output, "pseudoTuning.json")
    if not os.path.exists(saved):
        return False
    paths = [p for p in (os.path.join(args.ccRun, "detections.npz"),
                         os.path.join(args.output, "detections.npz"))
             if os.path.exists(p)]
    if not paths:
        return False
    record = dc.loadJson(saved)
    found = loadDetections(paths[0])
    writeAutoRun(args.ccRun, record["blocks"], found,
                 record["pooled"][AUTO_SCORER], args.autoRun)
    return True


def main(argv=None):
    args = parseArguments(argv)
    os.makedirs(args.output, exist_ok=True)
    autoRun = args.autoRun or autoRunDir(args.ccRun)
    if not os.path.exists(os.path.join(autoRun, "results.json")) \
            and autoFromSaved(args):
        print("[pseudo] automatic run written from the saved pseudo-tuning")
        return 0
    grid, fixed = runGrid(args.ccRun)
    validation = ConCompCrossValidation(
        args.dataset, grid=grid, resolution=fixed["resolution"],
        minHeight=fixed["minHeight"], minTreeAreaM2=fixed["minTreeArea"],
        verbose=False, jobs=args.jobs)
    found = detections(args, validation)
    blocks = dict(validation.blocks)
    rgb = MethodRun("rgb", args.rgbRun, blocks)
    perBlock = {}
    for name, region in sorted(sharedBlocks(blocks, (rgb,)).items()):
        rows = tuneBlock(BlockScores(name, region, validation.scene.crowns,
                                     rgb.record(name), found, args.jobs))
        if rows is not None:
            perBlock[name] = rows
    names = list(next(iter(perBlock.values())))
    table = {n: pool(perBlock, n) for n in names}
    target = dc.loadJson(os.path.join(args.ccRun, "results.json"))["pooled"]
    for rows in perBlock.values():
        for name, row in rows.items():
            row["settings"] = validation.settings[row["chosen"]]
    dc.saveJson({"blocks": perBlock, "pooled": table, "target": target},
                os.path.join(args.output, "pseudoTuning.json"))
    writeAutoRun(args.ccRun, perBlock, found, table[AUTO_SCORER], args.autoRun)
    printTable(table, target)
    return 0


def autoRunDir(ccRun):
    """runs/ccLidar -> runs/ccAutoLidar (beside the run it was tuned from)."""
    name = os.path.basename(os.path.normpath(ccRun))
    auto = "ccAuto" + name[2:] if name.startswith("cc") else name + "Auto"
    return os.path.join(os.path.dirname(os.path.normpath(ccRun)), auto)


def writeAutoRun(ccRun, perBlock, found, pooled, directory=None):
    """
    The AUTO_SCORER's chosen setting per block, as a run like any other: per
    block the whole-area detections of that setting (as connected-component
    runs store them), and the pooled results. The report then combines it
    with the RGB networks like any height method.
    """
    directory = directory or autoRunDir(ccRun)
    os.makedirs(directory, exist_ok=True)
    folds = []
    for block, rows in sorted(perBlock.items()):
        row = rows[AUTO_SCORER]
        dc.saveJson({"block": block, "predictions": found[row["chosen"]]},
                    os.path.join(directory, "predictions_%s.json" % block))
        folds.append(dict(row, block=block))
    dc.saveJson({"method": "connected components tuned on the RGB Mask "
                           "R-CNN's held-out boxes (%s)" % AUTO_SCORER,
                 "folds": folds, "pooled": pooled},
                os.path.join(directory, "results.json"))
    print("[pseudo] wrote the automatic run %s" % directory)
    return directory


if __name__ == "__main__":
    sys.exit(main())
