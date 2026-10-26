"""
RGB detections over low vegetation removed with the height model, on a site's
existing runs: no network is retrained and the report is not rerun.

    python -m tt.heightFilter --site study_procedure/enshurin
    python -m tt.heightFilter --site study_procedure/terelj_lidar --site ...

For every combination the site's report computed (report/fused/<box>_<height>),
two strategies are added under the same leave-one-block-out tuning:
  boxesHigh   the RGB model alone, at the score threshold it chose for
              itself, minus its boxes whose apex stands below a minimum
              height (tuned per fold on the validation block, 0 = keep all)
  unionHigh   the union, after the same filter on its boxes (threshold and
              minimum height tuned together)
Heights come from the site's CHM with no minimum height applied, so a 1.5 m
tree is measured as 1.5 m. The crowns and blocks are the report's own, and
"boxes" and "union" are scored again and checked against the report's saved
results, so the new rows compare like with like. The new strategies' scores
are added to the combination's summary.json (nothing else in it changes) and
their held-out outputs saved beside the others.
"""

import argparse
import os
import sys

from tt.dl import dlCommon as dc
from tt.dl import dlPrepare as dp
from tt.detectors.fusion import BOX_MIN_HEIGHTS, SaddleSurface
from tt.experiments import RunSet
from tt.experiments.fusion import FusionCrossValidation
from tt.scene import Scene
from tt.scoring import CrownFrame

STRATEGIES = ["boxes", "union", "boxesHigh", "unionHigh"]
CHECKED = ("boxes", "union")


def siteInputs(site):
    meta = dp.loadDataset(os.path.join(site, "ds", "chm"))
    chm = next(c["path"] for c in meta["channels"] if c["kind"] == "chm")
    # the report's scene (its crowns), and the same CHM unclipped for heights
    scene = Scene(chm, crownsPath=meta["crownsPath"],
                  boundaryPath=meta["boundaryPath"], verbose=False)
    heights = Scene(chm, minHeight=0.0, verbose=False)
    return meta, scene, SaddleSurface.fromScene(heights, dropM=0.0)


def pairs(site):
    fused = os.path.join(site, "report", "fused")
    if not os.path.isdir(fused):
        return []
    found = []
    for name in sorted(os.listdir(fused)):
        if "_" in name and os.path.exists(os.path.join(fused, name,
                                                       "summary.json")):
            box, height = name.split("_", 1)
            found.append((box, height, os.path.join(fused, name)))
    return found


def chosenHeights(summary, name):
    return [f["setting"]["boxMinHeightM"] for f in summary[name]["folds"]]


def filterSite(site, verbose=True):
    meta, scene, surface = siteInputs(site)
    blocks = meta["blockGeometries"]
    report = []
    for box, height, folder in pairs(site):
        try:
            boxRun = RunSet.load(os.path.join(site, "runs", box))
            pointRun = RunSet.load(os.path.join(site, "runs", height))
        except FileNotFoundError as error:
            report.append("%s_%s: skipped (%s)" % (box, height, error))
            continue
        validation = FusionCrossValidation.pair(
            CrownFrame(scene.crowns), dict(blocks), boxRun, pointRun,
            strategies=STRATEGIES, verbose=False,
            heightSurface=surface)
        summary = validation.run()
        saved = dc.loadJson(os.path.join(folder, "summary.json"))
        for name in CHECKED:
            if name in saved:
                a, b = saved[name]["pooled"], summary[name]["pooled"]
                if (a["hits"], a["predictions"]) != (b["hits"], b["predictions"]):
                    raise RuntimeError(
                        "%s_%s: '%s' scores %d hits / %d detections here but "
                        "%d / %d in the report: not the same inputs"
                        % (box, height, name, b["hits"], b["predictions"],
                           a["hits"], a["predictions"]))
        for name in ("boxesHigh", "unionHigh"):
            saved[name] = summary[name]
        dc.saveJson(saved, os.path.join(folder, "summary.json"))
        validation.outputs = {n: validation.outputs[n]
                              for n in ("boxesHigh", "unionHigh")}
        validation.saveOutputs(folder)
        line = "%s_%s:" % (box, height)
        for plain, high in (("boxes", "boxesHigh"), ("union", "unionHigh")):
            p, q = summary[plain]["pooled"], summary[high]["pooled"]
            line += ("  %s F1 %.3f W %.3f -> %s F1 %.3f W %.3f (heights %s)"
                     % (plain, p["f1"], p["weighted"], high, q["f1"],
                        q["weighted"], chosenHeights(summary, high)))
        report.append(line)
    if verbose:
        print("[heightFilter] %s" % os.path.basename(os.path.normpath(site)))
        for line in report:
            print("  " + line)
    return report


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site", action="append", required=True,
                        help="a site folder of the procedure (repeatable)")
    return parser.parse_args(argv)


def main(argv=None):
    for site in parseArguments(argv).site:
        filterSite(site)
    print("[heightFilter] minimum heights tried: %s m" % BOX_MIN_HEIGHTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
