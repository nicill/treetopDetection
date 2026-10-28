"""
Experiment 3 — the best combination, with enough good annotations: early
fusion (one Mask R-CNN on four channels, RGB and the P1 CHM: runs/mrcnnRgbP1,
made by exp3Run.sh) against late fusion of RGB Mask R-CNN with CC (ccP1), and
of RGB Mask R-CNN with the CHM network (mrcnnP1), against RGB alone.

The late fusions are made here from the saved runs, with every strategy of
tt.detectors.fusion, each tuned per fold on the validation block, each part
entering at its own operating point:
  RGB+CC:<s> and RGB+CHMnet:<s> for the two-member strategies (confirmed,
    boxMergedPoints, weakBoxMergedPoints, union, unionSaddleCross,
    unionSaddlePool, boxesHigh, unionHigh), plus CC alone and CHMnet alone;
  3way:<s> for the three-member ones (triple, unionClean, agreeClean).
The saddle strategies read the P1 CHM as the reports do (0.25 m, minimum
height 2 m, 0.5 m drop); the height filter and the three-member strategies
read it unclipped, with no minimum height, as heightFilter and tripleFusion do.
Values only, no tests: per site, every method's recall, precision and F1,
pooled over the blocks every method has; then one table of F1, methods by
sites, with the mean over sites. No LiDAR.

Run from ttPipeline (it needs tt):

    python experiment3.py                   # every site; results also in experiment3.json
    python experiment3.py --site name=FOLDER
"""

import argparse
import json
import os

import numpy as np

from tt.analysis.statistics import blockKey
from tt.data import TDData
from tt.experiments import RunSet
from tt.detectors.fusion import HEIGHT_STRATEGIES, STRATEGIES, SaddleSurface
from tt.experiments.fusion import FusionCrossValidation
from tt.runCompare import loadRun
from tt.scoring import AnnotatedCrowns

DB = os.environ.get("DROP") or os.path.expanduser("~/Yago Lab Dropbox/Meetings/2026TreeDet")
QUEBEC = ["20230712_afcamoisan_itrf20", "20230712_afcahoule_itrf20",
          "20230606_cbblackburn5", "20230608_cbbernard4", "20230606_cbblackburn6"]
SITES = ([(s.split("_")[1], os.path.join(DB, "study_edge", s)) for s in QUEBEC]
         + [("koiwai", os.path.join(DB, "study_edge", "koiwainoujo220616"))]
         + [("terelj" + a, os.path.join(DB, "study_terelj" + a, "terelj" + a))
            for a in ("First", "Second", "Third")]
         + [("enshurinV2", os.path.join(DB, "study_enshurinVlad2"))])
DATASETS = ("P1", "chm")


TWO = [c.name for c in STRATEGIES + HEIGHT_STRATEGIES if c.name not in ("boxes", "points")]
THREE = ("triple", "unionClean", "agreeClean")
SADDLE_DROP_P1 = 0.5


def has(runs, name):
    return os.path.exists(os.path.join(runs, name, "results.json"))


def byBlock(folds):
    return {blockKey(f["block"]): f for f in folds}


def pooled(folds):
    """Recall, precision and F1 over the blocks, from the folds' counts."""
    hits = sum(f["hits"] for f in folds.values())
    crowns = sum(f["crowns"] for f in folds.values())
    found = sum(f["predictions"] for f in folds.values())
    recall, precision = hits / max(crowns, 1), hits / max(found, 1)
    return recall, precision, 2 * recall * precision / max(recall + precision, 1e-9)


def lateFusions(reference, blocks, runs, saddle, heights):
    """{method: {blockKey: fold}} for every late fusion of the site."""
    rgb = RunSet.load(os.path.join(runs, "mrcnnRgb"))
    cc = RunSet.load(os.path.join(runs, "ccP1"))
    chm = RunSet.load(os.path.join(runs, "mrcnnP1"))
    out = {}
    for label, points in (("RGB+CC", cc), ("RGB+CHMnet", chm)):
        summary = FusionCrossValidation.pair(reference, blocks, rgb, points, surface=saddle,
                                             heightSurface=heights, verbose=False).run()
        out[label.split("+")[1] + " alone"] = byBlock(summary["points"]["folds"])
        for strategy in TWO:
            out["%s:%s" % (label, strategy)] = byBlock(summary[strategy]["folds"])
    for reduced in (False, True):
        summary = FusionCrossValidation.threeWay(reference, blocks, rgb, chm, cc, heights,
                                                 reduced=reduced, verbose=False).run()
        for strategy in THREE:
            if strategy in summary:
                out["3way:" + strategy] = byBlock(summary[strategy]["folds"])
    return out


def site(name, folder):
    runs = os.path.join(folder, "runs")
    missing = [r for r in ("mrcnnRgb", "ccP1", "mrcnnP1") if not has(runs, r)]
    if missing:
        print("== %s: missing %s in %s; skipped" % (name, ", ".join(missing), runs))
        return None
    dataset = next((os.path.join(folder, "ds", d) for d in DATASETS
                    if os.path.exists(os.path.join(folder, "ds", d, "dataset.json"))), None)
    if dataset is None:
        print("== %s: no ds/P1 or ds/chm dataset; skipped" % name)
        return None
    print("== %s" % name, flush=True)
    data = TDData.fromDataset(dataset)
    reference = AnnotatedCrowns(data.annotations)
    blocks = {b.name: b.geometry for b in data.blocks}
    saddle = SaddleSurface.fromSurface(data.heightSurface(), SADDLE_DROP_P1)
    heights = SaddleSurface.fromSurface(data.heightSurface(minHeight=0.0, restrict=False), 0.0)
    methods = {"RGB": loadRun(runs, "mrcnnRgb")[1]}
    if has(runs, "mrcnnRgbP1"):
        methods["4-channel"] = loadRun(runs, "mrcnnRgbP1")[1]
    methods.update(lateFusions(reference, blocks, runs, saddle, heights))
    shared = set.intersection(*[set(f) for f in methods.values()])
    methods = {m: {b: f for b, f in folds.items() if b in shared} for m, folds in methods.items()}
    record = {"site": name, "blocks": len(shared), "methods": {}}
    print("    %-31s %6s %6s %6s" % ("method (%d blocks)" % len(shared), "R", "P", "F1"))
    for method, folds in methods.items():
        r, p, f1 = pooled(folds)
        print("    %-31s %6.3f %6.3f %6.3f" % (method, r, p, f1))
        record["methods"][method] = {"recall": r, "precision": p, "f1": f1}
    if "4-channel" not in methods:
        print("    (4-channel not run yet: runs/mrcnnRgbP1, exp3Run.sh)")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site", action="append", default=None,
                        help="name=FOLDER, instead of the default sites")
    parser.add_argument("--output", default="experiment3.json")
    args = parser.parse_args(argv)
    sites = [tuple(s.split("=", 1)) for s in args.site] if args.site else SITES
    records = [r for r in (site(n, f) for n, f in sites) if r]
    names = []
    for r in records:
        names += [m for m in r["methods"] if m not in names]
    short = [r["site"][:9] for r in records]
    print("\nF1, methods by sites (blank: not run there)")
    print("%-31s %s  %6s" % ("method", " ".join("%9s" % c for c in short), "mean"))
    for method in names:
        values = [r["methods"].get(method, {}).get("f1") for r in records]
        known = [v for v in values if v is not None]
        print("%-31s %s  %6s" % (method, " ".join("%9s" % ("%.3f" % v if v is not None else "")
                                                for v in values),
                                 "%.3f" % float(np.mean(known)) if known else ""))
    with open(args.output, "w") as handle:
        json.dump(records, handle, indent=1, default=float)
    print("\nwritten to %s" % args.output)


if __name__ == "__main__":
    main()
