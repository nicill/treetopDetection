#!/usr/bin/env python
"""
dlConComp.py

The connected-component detector run through the same leave-one-block-out
scheme as the learned models, so all three can share one table.

    python -m tt.dl concomp --dataset datasets/lidarChm --output runs/cc

Each fold tunes on the training blocks and is scored on the held-out block it
never saw — exactly what the learned models get. Without that the comparison
would be rigged in favour of the hand-tuned method.

How the work is organised
-------------------------
Every setting in the tuning grid is run once, over the whole area, and the
folds are then scored from those detections by selecting what falls in each
block. An earlier version re-ran every setting inside every fold on that fold's
training blocks; since each block is a training block in seven of eight folds,
every detection was computed seven times over — 200 detector runs where 24
suffice.

Running over the whole area is not leakage. The CHM is the input, not the
label: tuning still scores against training-block crowns only, and the held-out
block's crowns are never consulted until its own fold. What changes is that the
detector no longer sees an artificial cliff where the held-out block was zeroed
out, so trees near block borders are detected as they would be in practice.

The detector has no notion of confidence, so a detection's height stands in as
its score. That only decides which detection is primary inside a crown when
several land in the same one, the rule the detector already uses internally.
"""

import argparse
import os
import sys


from ..data.site import TDData
from ..detectors.height import CCDetector
from ..experiments import CrossValidation, ParameterGrid
from . import dlCommon as dc

# The merge threshold is tuned alongside the detector, because the two interact:
# a base that over-detects less needs a gentler merge, so tuning them separately
# finds the best of neither.
DEFAULT_GRID = {
    "lowerPercentile": [20, 30],
    "minTopAreaM2": [0.12, 0.06],
    "topStepM": [0.25],
    "erosionIterations": [1, 2],
    "saddleDropM": [0.2, 0.3, 0.5],
}
GRID_CASTS = {"lowerPercentile": int, "erosionIterations": int}
# the minimum height and minimum tree area are fixed unless given as lists
# (--minHeights, --minTreeAreas); then they are tuned like the rest. The crowns
# scored never depend on them: only the detector's input does.
SCENE_KEY = "minHeight"
SMOOTH_SLOPE_KEY = "smoothSlope"   # smoothing grows with tree height (m/m)
SMOOTH_WINDOW_KEY = "smoothWindowM"   # tree height: tallest canopy within this
SMOOTH_KEY = "smoothM"      # Gaussian smoothing of the CHM (m); tuned when
                            # --smooths lists values, else none
TREE_AREA_KEY = "minTreeAreaM2"
# the merge rule: "saddle" (straight-line drop) or "prominence" (elder rule in
# the descent); a grid dimension when --mergeMetrics lists both
MERGE_KEY = "mergeMetric"
DROP_SLOPE_KEY = "saddleDropSlope"     # drop(h) = saddleDropM + slope * h
TOP_SLOPE_KEY = "minTopAreaSlope"      # minTop(h) = minTopAreaM2 + slope * h

def gridFromArguments(args):
    grid = dict(DEFAULT_GRID)
    for key, option in (("lowerPercentile", args.percentiles),
                        ("minTopAreaM2", args.minTopAreas),
                        ("topStepM", args.topSteps),
                        ("erosionIterations", args.erosions),
                        ("saddleDropM", args.saddleDrops)):
        if option:
            cast = GRID_CASTS.get(key, float)
            grid[key] = [cast(v) for v in option.split(",") if v.strip()]
    for key, option in ((SCENE_KEY, getattr(args, "minHeights", None)),
                        (TREE_AREA_KEY, getattr(args, "minTreeAreas", None))):
        if option:
            grid[key] = [float(v) for v in option.split(",") if v.strip()]
    smooths = getattr(args, "smooths", None)
    if smooths:
        grid[SMOOTH_KEY] = [float(v) for v in smooths.split(",") if v.strip()]
    metrics = getattr(args, "mergeMetrics", None)
    if metrics:
        grid[MERGE_KEY] = [v.strip() for v in metrics.split(",") if v.strip()]
    # thresholds that grow with tree height; dimensions only when given, so
    # runs without them keep their settings and labels unchanged
    for key, option in ((DROP_SLOPE_KEY, getattr(args, "saddleDropSlopes", None)),
                        (TOP_SLOPE_KEY, getattr(args, "minTopAreaSlopes", None)),
                        (SMOOTH_SLOPE_KEY, getattr(args, "smoothSlopes", None)),
                        (SMOOTH_WINDOW_KEY, getattr(args, "smoothWindows", None))):
        if option:
            grid[key] = [float(v) for v in option.split(",") if v.strip()]
    return grid


def crossValidate(args):
    """The `concomp` command, run through the experiment classes."""
    os.makedirs(args.output, exist_ok=True)
    dc.startLogging(args.log or os.path.join(args.output, "run.log"))
    data = TDData.fromDataset(args.dataset)
    detector = CCDetector(resolution=args.resolution, minHeight=args.minHeight,
                          minTreeAreaM2=args.minTreeArea)
    validation = CrossValidation(data, detector,
                                 ParameterGrid(gridFromArguments(args)),
                                 jobs=args.jobs, output=args.output)
    validation.progress = bool(getattr(args, "progress", False))
    runSet = validation.run()
    runSet.settings = vars(args)
    runSet.save(args.output)
    validation.writeSettingsTable(os.path.join(args.output, "settings.tsv"))
    if args.saveDetections:
        validation.saveAllDetections(os.path.join(args.output,
                                                  "detections.npz"))
    pooled = runSet.pooled
    print("\n[concomp] pooled over %d folds: R %.3f  P %.3f  F1 %.3f "
          "(per-fold F1 %.3f +- %.3f)"
          % (pooled["folds"], pooled["recall"], pooled["precision"],
             pooled["f1"], pooled["perFoldF1Mean"], pooled["perFoldF1Std"]))
    print("[concomp] objective %s: R %.3f P %.3f F1 %.3f weighted %.3f"
          % (pooled["objective"], pooled["recall"], pooled["precision"],
             pooled["f1"], pooled["weighted"]))
    print("[concomp] tuning optimism: %+.3f" % pooled["tuningOptimism"])
    for metric, p in pooled["byMergeMetric"].items():
        print("[concomp] %-10s only: R %.3f  P %.3f  F1 %.3f"
              % (metric, p["recall"], p["precision"], p["f1"]))
    for smooth, p in pooled["bySmoothing"].items():
        print("[concomp] smoothing %s m only: R %.3f  P %.3f  F1 %.3f"
              % (smooth, p["recall"], p["precision"], p["f1"]))
    ceiling = pooled["wholeAreaBest"]
    print("[concomp] whole-area best (optimistic ceiling): R %.3f  P %.3f  "
          "F1 %.3f at %s" % (ceiling["recall"], ceiling["precision"],
                             ceiling["f1"], ceiling["settings"]))
    print("[concomp] wrote %s" % os.path.join(args.output, "results.json"))
    return pooled


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Connected-component detection under leave-one-block-out.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", default=None,
                        help="Also write everything printed to this file. "
                             "Defaults to <output>/run.log")
    parser.add_argument("--resolution", type=float, default=0.25,
                        help="Working resolution for the detector (default: "
                             "0.25). Independent of the tile resolution the "
                             "learned models use; each method gets its own")
    parser.add_argument("--minHeight", type=float, default=2.0)
    parser.add_argument("--minTreeArea", type=float, default=0.5,
                        help="Smallest blob searched for tops, m2 (default: "
                             "0.5, matching the sweeps and the CLI)")
    parser.add_argument("--percentiles", default=None,
                        help="Override the tuning grid, comma separated")
    parser.add_argument("--minTopAreas", default=None)
    parser.add_argument("--topSteps", default=None)
    parser.add_argument("--erosions", default=None)
    parser.add_argument("--saddleDrops", default=None)
    parser.add_argument("--minHeights", default=None,
                        help="Minimum heights to tune over (else --minHeight "
                             "is fixed)")
    parser.add_argument("--minTreeAreas", default=None,
                        help="Minimum tree areas to tune over (else "
                             "--minTreeArea is fixed)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Settings detected at once, in parallel")
    parser.add_argument("--smooths", default=None,
                        help="Gaussian smoothing of the CHM (sigma, m) to "
                             "tune over, e.g. 0,0.1,0.2 (default: none)")
    parser.add_argument("--mergeMetrics", default=None,
                        help="Merge rules to tune over: saddle,prominence "
                             "(default: saddle only)")
    parser.add_argument("--saddleDropSlopes", default=None,
                        help="dip threshold growth per metre of tree height "
                             "(grid dimension only when given)")
    parser.add_argument("--smoothWindows", default=None,
                        help="window (m) in which a pixel's tree height is the "
                             "tallest canopy, for --smoothSlopes (default 2)")
    parser.add_argument("--smoothSlopes", default=None,
                        help="smoothing growth per metre of tree height, m/m "
                             "(grid dimension only when given)")
    parser.add_argument("--minTopAreaSlopes", default=None,
                        help="minimum top area growth, m2 per metre of height")
    parser.add_argument("--progress", action="store_true",
                        help="print each setting's whole-area recall and "
                             "precision as it is detected (progress.tsv)")
    parser.add_argument("--saveDetections", action="store_true",
                        help="Keep every setting's detections in "
                             "detections.npz (for tt.pseudoTuning)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(0 if crossValidate(parseArguments()) else 1)
