"""
Experiment 4 — what is left with little or no annotation. Values only, per
site and across sites, from the runs exp4.sh makes:

  automatic hybrid (the RGB network's annotation only)
    RGB             Mask R-CNN on RGB (runs/mrcnnRgb)
    CC              CC tuned on the crowns (runs/ccP1)
    CC auto         CC tuned on the RGB network's held-out boxes, pseudoF1
                    (exp4/runs/ccAutoRgbP1)
    RGB+CC          their union, as in experiment 3
    RGB+CC auto     the union with the automatic CC
  no annotation of the site at all
    DeepForest      as downloaded, at its fixed threshold (exp4/runs/deepforestRgb)
    CC on DF        CC tuned on DeepForest's boxes (exp4/runs/ccAutoDfP1)
    DF+CC on DF     their union at DeepForest's fixed threshold
  learning curves (DROP/exp4/curves, tt.learningCurve)
    F1 of CC and of the RGB network at each annotated fraction, mean over draws

The unions with the RGB network take their threshold per fold on the
validation block, as in experiment 3 (union only); the DeepForest union takes
nothing from the crowns. Scores are pooled over the blocks every method of a
part shares.

    python experiment4.py [--output experiment4.json] [--site name=FOLDER ...]
"""

import argparse
import json
import os

import numpy as np

from tt.analysis.statistics import blockKey
from tt.data import TDData
from tt.deepForestCompare import blockScores, unionScores
from tt.experiments import RunSet
from tt.experiments.fusion import FusionCrossValidation
from tt.learningCurve import siteRows
from tt.runCompare import loadRun
from tt.scoring import AnnotatedCrowns, Evaluator

DB = os.environ.get("DROP") or os.path.expanduser("~/Yago Lab Dropbox/Meetings/2026TreeDet")
QUEBEC = ["20230712_afcamoisan_itrf20", "20230712_afcahoule_itrf20",
          "20230606_cbblackburn5", "20230608_cbbernard4", "20230606_cbblackburn6"]
SITES = ([(s.split("_")[1], os.path.join(DB, "study_edge", s)) for s in QUEBEC]
         + [("koiwai", os.path.join(DB, "study_edge", "koiwainoujo220616"))]
         + [("terelj" + a, os.path.join(DB, "study_terelj" + a, "terelj" + a))
            for a in ("First", "Second", "Third")]
         + [("enshurinV2", os.path.join(DB, "study_enshurinVlad2"))])
DATASETS = ("P1", "chm")
HYBRID = ("RGB", "CC", "CC auto", "RGB+CC", "RGB+CC auto")
NONE = ("DeepForest", "CC on DF", "DF+CC on DF")


def has(path):
    return os.path.exists(os.path.join(path, "results.json"))


def pooled(folds):
    """Recall, precision and F1 over the blocks, from the folds' counts."""
    hits = sum(f["hits"] for f in folds.values())
    crowns = sum(f["crowns"] for f in folds.values())
    found = sum(f["predictions"] for f in folds.values())
    recall, precision = hits / max(crowns, 1), hits / max(found, 1)
    return {"recall": recall, "precision": precision,
            "f1": 2 * recall * precision / max(recall + precision, 1e-9)}


def shared(methods):
    """The methods' folds on the blocks all of them have."""
    common = set.intersection(*[set(f) for f in methods.values()])
    return {m: {b: f for b, f in folds.items() if b in common} for m, folds in methods.items()}, len(common)


def union(reference, blocks, boxes, points):
    folds = FusionCrossValidation.pair(reference, blocks, RunSet.load(boxes), RunSet.load(points),
                                       strategies=["union"], verbose=False).run()["union"]["folds"]
    return {blockKey(f["block"]): f for f in folds}


def site(name, folder):
    runs, exp4 = os.path.join(folder, "runs"), os.path.join(folder, "exp4", "runs")
    dataset = next((os.path.join(folder, "ds", d) for d in DATASETS
                    if os.path.exists(os.path.join(folder, "ds", d, "dataset.json"))), None)
    if dataset is None or not (has(os.path.join(runs, "mrcnnRgb")) and has(os.path.join(runs, "ccP1"))):
        print("== %s: no CHM dataset or no mrcnnRgb and ccP1 runs; skipped" % name)
        return None
    print("== %s" % name, flush=True)
    data = TDData.fromDataset(dataset)
    reference = AnnotatedCrowns(data.annotations)
    blocks = {b.name: b.geometry for b in data.blocks}
    record = {"site": name}

    hybrid = {"RGB": loadRun(runs, "mrcnnRgb")[1], "CC": loadRun(runs, "ccP1")[1],
              "RGB+CC": union(reference, blocks, os.path.join(runs, "mrcnnRgb"), os.path.join(runs, "ccP1"))}
    auto = os.path.join(exp4, "ccAutoRgbP1")
    if has(auto):
        hybrid["CC auto"] = loadRun(exp4, "ccAutoRgbP1")[1]
        hybrid["RGB+CC auto"] = union(reference, blocks, os.path.join(runs, "mrcnnRgb"), auto)
    hybrid, n = shared(hybrid)
    record["hybrid"] = {m: pooled(f) for m, f in hybrid.items()}
    print("    automatic hybrid (%d blocks)          R      P      F1" % n)
    for method in HYBRID:
        if method in record["hybrid"]:
            p = record["hybrid"][method]
            print("      %-30s %.3f  %.3f  %.3f" % (method, p["recall"], p["precision"], p["f1"]))
        else:
            print("      %-30s not run yet" % method)

    df, ccDf = os.path.join(exp4, "deepforestRgb"), os.path.join(exp4, "ccAutoDfP1")
    if has(df):
        evaluator = Evaluator(reference)
        none = {"DeepForest": blockScores(RunSet.load(df), blocks, evaluator, blocks)}
        if has(ccDf):
            none["CC on DF"] = blockScores(RunSet.load(ccDf), blocks, evaluator, blocks)
            none["DF+CC on DF"] = unionScores(RunSet.load(df), RunSet.load(ccDf), blocks, evaluator)
        none = {m: {blockKey(b): s for b, s in scores.items()} for m, scores in none.items()}
        none, n = shared(none)
        record["noAnnotation"] = {m: pooled(f) for m, f in none.items()}
        print("    no annotation (%d blocks)" % n)
        for method in NONE:
            if method in record["noAnnotation"]:
                p = record["noAnnotation"][method]
                print("      %-30s %.3f  %.3f  %.3f" % (method, p["recall"], p["precision"], p["f1"]))
            else:
                print("      %-30s not run yet" % method)
    else:
        print("    no annotation: no DeepForest run yet")

    curve = os.path.join(DB, "exp4", "curves", name)
    rows = siteRows(curve) if os.path.isdir(curve) else []
    if rows:
        record["curve"] = rows
        print("    learning curve (F1, mean over draws +- sd)")
        for r in rows:
            print("      %-10s %-10s draws %d  F1 %.3f +- %.3f  (crowns per fold %.0f)"
                  % (r["method"], r["point"], r["draws"], r["f1"], r["f1Sd"], r["crownsUsedPerFold"]))
    return record


def means(records, part, methods):
    print("\n%s, mean over sites" % part)
    for method in methods:
        values = [r[part][method] for r in records if part in r and method in r[part]]
        if values:
            print("  %-30s sites %2d  R %.3f  P %.3f  F1 %.3f" % (
                method, len(values), *[float(np.mean([v[k] for v in values]))
                                       for k in ("recall", "precision", "f1")]))


def curveMeans(records):
    points = {}
    for r in records:
        for row in r.get("curve", []):
            points.setdefault((row["method"], row["point"]), []).append(row["f1"])
    if not points:
        return
    print("\nlearning curve, F1 mean over sites")
    order = lambda p: float(p[1].rstrip("%")) if p[1].endswith("%") else -1.0
    for key in sorted(points, key=lambda k: (k[0], order(k))):
        print("  %-10s %-10s sites %2d  F1 %.3f" % (key[0], key[1], len(points[key]), np.mean(points[key])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site", action="append", default=None,
                        help="name=FOLDER, instead of the default sites")
    parser.add_argument("--output", default="experiment4.json")
    args = parser.parse_args(argv)
    sites = [tuple(s.split("=", 1)) for s in args.site] if args.site else SITES
    records = [r for r in (site(n, f) for n, f in sites) if r]
    means(records, "hybrid", HYBRID)
    means(records, "noAnnotation", NONE)
    curveMeans(records)
    with open(args.output, "w") as handle:
        json.dump(records, handle, indent=1, default=float)
    print("\nwritten to %s" % args.output)


if __name__ == "__main__":
    main()
