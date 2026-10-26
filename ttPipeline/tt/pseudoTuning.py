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

The scorers and the tuning are tt.experiments.pseudo.PseudoTuning.

The detections of every setting are taken from the run's detections.npz
(tt.dl concomp --saveDetections) or, when it has none, recomputed from the
grid the run recorded and cached in --output.
"""

import argparse
import os
import sys

from .data.site import TDData
from .detectors.height import CCDetector
from .dl import dlCommon as dc
from .dl.dlConComp import gridFromArguments
from .experiments import CrossValidation, ParameterGrid, RunSet
from .experiments.crossValidation import loadAllDetections
from .experiments.pseudo import PseudoTuning

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


def savedDetections(ccRun, output):
    """The detections.npz saved with the run, or cached in output."""
    for path in (os.path.join(ccRun, "detections.npz"),
                 os.path.join(output, "detections.npz")):
        if os.path.exists(path):
            return path
    return None


def experiment(args):
    """The cross-validation the CC run made, and every setting's detections."""
    grid, fixed = runGrid(args.ccRun)
    site = TDData.fromDataset(args.dataset)
    validation = CrossValidation(
        site, CCDetector(resolution=fixed["resolution"],
                         minHeight=fixed["minHeight"],
                         minTreeAreaM2=fixed["minTreeArea"]),
        ParameterGrid(grid), verbose=False, jobs=args.jobs)
    path = savedDetections(args.ccRun, args.output)
    if path:
        print("[pseudo] detections from %s" % path)
        found = loadAllDetections(path)
    else:
        print("[pseudo] computing %d settings, %d jobs"
              % (len(validation.settings), validation.jobs))
        found = validation.detectAll()
        validation.saveAllDetections(os.path.join(args.output,
                                                  "detections.npz"))
    return site, validation, found


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
    parser.add_argument("--autoScorer", default=AUTO_SCORER,
                        help="the scorer whose choices make the automatic run "
                             "(default %s; e.g. zones@0.7 scores the boxes "
                             "zone by zone)" % AUTO_SCORER)
    return parser.parse_args(argv)


def autoRunDir(ccRun):
    """runs/ccLidar -> runs/ccAutoLidar (beside the run it was tuned from)."""
    name = os.path.basename(os.path.normpath(ccRun))
    auto = "ccAuto" + name[2:] if name.startswith("cc") else name + "Auto"
    return os.path.join(os.path.dirname(os.path.normpath(ccRun)), auto)


def writeAutoRun(tuning, perBlock, pooled, directory, scorer):
    tuning.asRun(perBlock, pooled, scorer).save(directory)
    print("[pseudo] wrote the automatic run %s" % directory)
    return directory


def main(argv=None):
    args = parseArguments(argv)
    os.makedirs(args.output, exist_ok=True)
    autoRun = args.autoRun or autoRunDir(args.ccRun)
    saved = os.path.join(args.output, "pseudoTuning.json")
    rgb = RunSet.load(args.rgbRun)
    if not os.path.exists(os.path.join(autoRun, "results.json")) \
            and os.path.exists(saved) \
            and savedDetections(args.ccRun, args.output):
        # the automatic run from a finished pseudo-tuning, scoring nothing
        record = dc.loadJson(saved)
        site = TDData.fromDataset(args.dataset)
        found = loadAllDetections(savedDetections(args.ccRun, args.output))
        tuning = PseudoTuning(site, None, found, rgb)
        writeAutoRun(tuning, record["blocks"],
                     record["pooled"][args.autoScorer], autoRun,
                     args.autoScorer)
        print("[pseudo] automatic run written from the saved pseudo-tuning")
        return 0
    site, validation, found = experiment(args)
    tuning = PseudoTuning(site, validation.settings, found, rgb,
                          jobs=args.jobs)
    perBlock, table = tuning.run()
    target = dc.loadJson(os.path.join(args.ccRun, "results.json"))["pooled"]
    dc.saveJson({"blocks": perBlock, "pooled": table, "target": target},
                saved)
    writeAutoRun(tuning, perBlock, table[args.autoScorer], autoRun,
                 args.autoScorer)
    printTable(table, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
