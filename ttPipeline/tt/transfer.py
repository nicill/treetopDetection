"""
Does a site need its own tuning? Connected components at settings fixed
elsewhere, scored block by block on a new site, against the site's own
cross-validated tuning.

    python -m tt.transfer --chm sergi/chm.tif --crowns sergi/crowns.shp \\
        --boundary sergi/area.shp --dataset ds/sergi \\
        --tuned runs/ccSergi --modalFrom ccLidar=runs/ccLidar \\
        --modalFrom ccP1=runs/ccP1 --output transfer

The fixed settings:

  defaults         the detector's own defaults, those `tt detect` uses
  modal <run>      for each --modalFrom run, the setting its cross-validation
                   chose in the most folds

--compare adds other cross-validated runs on the same site, such as a learned
model, to the same block-by-block comparison with the tuned run.

The modal settings are read from the runs' results.json, never typed in, so
they are exactly what the other site's tuning chose. Nothing about the new
site's crowns enters them, so each is scored once over the whole area, as it
would be used, and then block by block.

Each fixed setting is written as a run — one predictions_<block>.json per
block, holding the whole area as a connected-component run does — so it can
be drawn by tt.blockImages and compared with any other run. transfer.json
holds every block's F1 and the paired comparison with the tuned run.
"""

import argparse
import collections
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

from .detector import ConCompDetector
from .dl import dlCommon as dc
from .dl import dlPrepare as dp
from .merging import TopMerger
from .scene import Scene

# the fixed parts, as in the connected-component cross-validation
WINDOW_M = 40.0
MIN_TREE_AREA_M2 = 0.5
MERGE_RADIUS_M = 8.0
POINT_BOX_M = 0.5
SETTING_KEYS = ("lowerPercentile", "minTopAreaM2", "topStepM",
                "erosionIterations", "saddleDropM")
DEFAULTS = {"lowerPercentile": 10, "minTopAreaM2": 0.12, "topStepM": 0.12,
            "erosionIterations": 1, "saddleDropM": 0.5}


def modalSetting(resultsPath):
    """The setting a cross-validation chose in the most folds."""
    folds = dc.loadJson(resultsPath)["folds"]
    chosen = collections.Counter(
        tuple(fold["settings"][key] for key in SETTING_KEYS)
        for fold in folds if "settings" in fold)
    if not chosen:
        raise ValueError("%s records no per-fold settings" % resultsPath)
    values, count = chosen.most_common(1)[0]
    return dict(zip(SETTING_KEYS, values)), count, len(folds)


def detect(scene, setting):
    detector = ConCompDetector(
        windowSizeM=WINDOW_M, minTreeAreaM2=MIN_TREE_AREA_M2,
        lowerPercentile=setting["lowerPercentile"],
        minTopAreaM2=setting["minTopAreaM2"], topStepM=setting["topStepM"],
        erosionIterations=setting["erosionIterations"],
        merger=TopMerger("saddle", epsM=MERGE_RADIUS_M,
                         saddleDropM=setting["saddleDropM"]),
        verbose=False)
    tops = detector.detect(scene)
    return [{"centreX": float(x), "centreY": float(y), "score": float(h),
             "box": [x - POINT_BOX_M, y - POINT_BOX_M,
                     x + POINT_BOX_M, y + POINT_BOX_M]}
            for (x, y), h in zip(tops.world(scene), tops.heights)]


def writeRun(directory, blocks, predictions, setting, folds, whole):
    """The fixed setting as a run: the layout every other tool reads."""
    os.makedirs(directory, exist_ok=True)
    for name, _ in blocks:
        dc.saveJson({"block": name, "settings": setting, "region": "area",
                     "predictions": predictions},
                    os.path.join(directory, "predictions_%s.json" % name))
    dc.saveJson({"method": "concomp", "settings": setting, "folds": folds,
                 "pooled": dc.averageFolds(folds), "wholeArea": whole},
                os.path.join(directory, "results.json"))


def paired(tunedFolds, fixedFolds):
    tuned = {f["block"]: f["f1"] for f in tunedFolds}
    fixed = {f["block"]: f["f1"] for f in fixedFolds}
    shared = sorted(set(tuned) & set(fixed))
    difference = np.array([tuned[b] - fixed[b] for b in shared])
    p = wilcoxon(difference).pvalue if np.any(difference) else 1.0
    return {"tunedMinusFixed": float(difference.mean()),
            "tunedBetter": int((difference > 0).sum()),
            "blocks": len(shared), "p": float(p)}


class Transfer(object):

    def __init__(self, scene, blocks, tunedDir):
        self.scene = scene
        self.blocks = blocks
        self.tuned = dc.loadJson(os.path.join(tunedDir, "results.json"))

    def score(self, name, setting, outputDir):
        predictions = detect(self.scene, setting)
        folds = []
        for block, geometry in self.blocks:
            result = dc.evaluateDetections(predictions, self.scene.crowns,
                                           geometry)
            result["block"] = block
            folds.append(result)
        whole = dc.evaluateDetections(predictions, self.scene.crowns,
                                      self.scene.boundary)
        writeRun(os.path.join(outputDir, "runs", name), self.blocks,
                 predictions, setting, folds, whole)
        return {"setting": setting, "wholeArea": whole,
                "pooled": dc.averageFolds(folds), "folds": folds,
                "vsTuned": paired(self.tuned["folds"], folds)}


def printTable(tuned, rows):
    print("\n%-22s %7s %7s %7s   %s" % ("setting", "recall", "prec", "F1",
                                        "site-tuned minus it"))
    pooled = tuned["pooled"]
    print("%-22s %7.3f %7.3f %7.3f   (cross-validated)"
          % ("site-tuned", pooled["recall"], pooled["precision"],
             pooled["f1"]))
    for name, row in rows.items():
        p, v = row["pooled"], row["vsTuned"]
        print("%-22s %7.3f %7.3f %7.3f   %+.3f, tuned better on %d/%d, "
              "p=%.2f" % (name, p["recall"], p["precision"], p["f1"],
                          v["tunedMinusFixed"], v["tunedBetter"],
                          v["blocks"], v["p"]))


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Fixed settings from elsewhere against site tuning.")
    parser.add_argument("--chm", required=True)
    parser.add_argument("--crowns", required=True)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--dataset", required=True,
                        help="The site's prepared dataset; its blocks")
    parser.add_argument("--tuned", required=True,
                        help="The site's own connected-component run")
    parser.add_argument("--modalFrom", action="append", default=[],
                        metavar="NAME=RUNDIR",
                        help="A run elsewhere whose modal setting to test")
    parser.add_argument("--compare", action="append", default=[],
                        metavar="NAME=RUNDIR",
                        help="Another cross-validated run on this site, such "
                             "as a learned model, compared the same way")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parseArguments(argv)
    scene = Scene(args.chm, crownsPath=args.crowns,
                  boundaryPath=args.boundary, verbose=False)
    blocks = dp.loadDataset(args.dataset)["blockGeometries"]
    transfer = Transfer(scene, blocks, args.tuned)
    settings = {"defaults": DEFAULTS}
    for spec in args.modalFrom:
        name, runDir = spec.split("=", 1)
        setting, count, total = modalSetting(os.path.join(runDir,
                                                          "results.json"))
        print("[transfer] modal %s: %s (chosen in %d of %d folds)"
              % (name, setting, count, total))
        settings["modal_" + name] = setting
    rows = {name: transfer.score(name, setting, args.output)
            for name, setting in settings.items()}
    for spec in args.compare:
        name, runDir = spec.split("=", 1)
        path = os.path.join(runDir, "results.json")
        if not os.path.exists(path):
            print("[transfer] %s skipped: no results.json" % name)
            continue
        other = dc.loadJson(path)
        rows[name] = {"pooled": other["pooled"], "folds": other["folds"],
                      "vsTuned": paired(transfer.tuned["folds"],
                                        other["folds"])}
    printTable(transfer.tuned, rows)
    dc.saveJson({"tuned": transfer.tuned["pooled"], "fixed": rows},
                os.path.join(args.output, "transfer.json"))
    print("\nwrote %s" % os.path.join(args.output, "transfer.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
