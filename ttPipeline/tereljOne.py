"""
Terelj as one site: the three areas' held-out blocks pooled (each area keeps its own cross-validation;
nothing is retrained), and every number the paper reports for a site recomputed on the pooled blocks.

  exp1   CC, local maxima and watershed on the P1 and the LiDAR CHM: recall, precision, F1; P1 minus
         LiDAR per method; CC minus each baseline per CHM (Holm over the two)
  exp2   Mask R-CNN on RGB against the P1 CHM
  exp3   every method of experiment 3 (experiment3.py's own), and the three tests fixed in advance
         against the RGB network (Holm over the three)
  exp4   the automatic hybrid, the no-annotation methods, and the learning curves

A comparison is the pooled F1 difference, its 95% bootstrap interval over the pooled blocks, the blocks
won and the paired Wilcoxon p over them.

    cd ttPipeline && python tereljOne.py            # writes DROP/tereljOne.json
"""

import argparse
import json
import os

import numpy as np

import experiment3 as e3
import experiment4 as e4
from tt.analysis.statistics import blockKey, compareFolds, holm
from tt.data import TDData
from tt.deepForestCompare import blockScores, unionScores
from tt.experiments import RunSet
from tt.learningCurve import dc, networkPoints, suffixes
from tt.runCompare import loadRun
from tt.scoring import AnnotatedCrowns, Evaluator

DB = os.environ.get("DROP") or os.path.expanduser("~/Yago Lab Dropbox/Meetings/2026TreeDet")
AREAS = [("terelj" + a, os.path.join(DB, "study_terelj" + a, "terelj" + a)) for a in ("First", "Second", "Third")]
BOOTSTRAP = 10000


def exists(folder, *parts):
    return os.path.exists(os.path.join(folder, *parts, "results.json"))


def merge(perArea):
    """{area: {block: fold}} -> {"area/block": fold}: the areas' blocks as one site's."""
    return {"%s/%s" % (area, block): fold for area, folds in perArea.items() for block, fold in folds.items()}


def pooled(folds):
    values = folds.values() if isinstance(folds, dict) else folds
    hits = sum(f["hits"] for f in values)
    crowns = sum(f["crowns"] for f in values)
    found = sum(f["predictions"] for f in values)
    recall, precision = hits / max(crowns, 1), hits / max(found, 1)
    return {"blocks": len(values), "recall": recall, "precision": precision,
            "f1": 2 * recall * precision / max(recall + precision, 1e-9)}


def comparison(first, second, rng):
    """first minus second over the blocks both have."""
    r = compareFolds(first, second, BOOTSTRAP, rng)
    return {k: r[k] for k in ("pooledReference", "pooledOther", "pooledDifference", "interval95",
                              "referenceWins", "blocks", "p")}


def runsOf(areas, name, sub=("runs",)):
    """The run's folds merged over the areas, or None if an area lacks it."""
    if not all(exists(folder, *sub, name) for _, folder in areas):
        return None
    return merge({area: loadRun(os.path.join(folder, *sub), name)[1] for area, folder in areas})


def chmDataset(folder):
    return next(os.path.join(folder, "ds", d) for d in ("P1", "chm")
                if os.path.exists(os.path.join(folder, "ds", d, "dataset.json")))


# ------------------------------------------------------------------ experiments

def experiment1(areas, rng):
    out = {"f1": {}, "p1MinusLidar": {}, "ccMinusBaseline": {}}
    folds = {(m, s): runsOf(areas, m + s) for m in ("cc", "lmvw", "watershed") for s in ("P1", "Lidar")}
    for (m, s), f in folds.items():
        if f is not None:
            out["f1"]["%s%s" % (m, s)] = pooled(f)
    for m in ("cc", "lmvw", "watershed"):
        if folds[(m, "P1")] and folds[(m, "Lidar")]:
            out["p1MinusLidar"][m] = comparison(folds[(m, "P1")], folds[(m, "Lidar")], rng)
    for s in ("P1", "Lidar"):
        rows = {b: comparison(folds[("cc", s)], folds[(b, s)], rng) for b in ("lmvw", "watershed")
                if folds[("cc", s)] and folds[(b, s)]}
        for row, adjusted in zip(rows.values(), holm([r["p"] for r in rows.values()]) if rows else []):
            row["pHolm"] = float(adjusted)
        out["ccMinusBaseline"][s] = rows
    return out


def experiment2(areas, rng):
    rgb, chm = runsOf(areas, "mrcnnRgb"), runsOf(areas, "mrcnnP1")
    return {"rgb": pooled(rgb), "chm": pooled(chm), "rgbVsChm": comparison(rgb, chm, rng)}


def methods3(folder):
    """One area's experiment 3 methods, exactly as experiment3.py builds them."""
    runs = os.path.join(folder, "runs")
    data = TDData.fromDataset(chmDataset(folder))
    reference = AnnotatedCrowns(data.annotations)
    blocks = {b.name: b.geometry for b in data.blocks}
    saddle = e3.SaddleSurface.fromSurface(data.heightSurface(), e3.SADDLE_DROP_P1)
    heights = e3.SaddleSurface.fromSurface(data.heightSurface(minHeight=0.0, restrict=False), 0.0)
    methods = {"RGB": loadRun(runs, "mrcnnRgb")[1]}
    if e3.has(runs, "mrcnnRgbP1"):
        methods["4-channel"] = loadRun(runs, "mrcnnRgbP1")[1]
    methods.update(e3.lateFusions(reference, blocks, runs, saddle, heights))
    common = set.intersection(*[set(f) for f in methods.values()])
    return {m: {b: f for b, f in folds.items() if b in common} for m, folds in methods.items()}


def mergeMethods(perArea):
    names = [m for m in next(iter(perArea.values())) if all(m in methods for methods in perArea.values())]
    return {m: merge({area: methods[m] for area, methods in perArea.items()}) for m in names}


def experiment3(areas, rng):
    methods = mergeMethods({area: methods3(folder) for area, folder in areas})
    out = {"methods": {m: pooled(f) for m, f in methods.items()}, "tests": {}}
    rows = {m: comparison(methods[m], methods["RGB"], rng)
            for m in ("4-channel", "RGB+CC:union", "RGB+CHMnet:union") if m in methods}
    for row, adjusted in zip(rows.values(), holm([r["p"] for r in rows.values()]) if rows else []):
        row["pHolm"] = float(adjusted)
    out["tests"] = rows
    return out


def hybrid(folder):
    runs, exp4 = os.path.join(folder, "runs"), os.path.join(folder, "exp4", "runs")
    data = TDData.fromDataset(chmDataset(folder))
    reference = AnnotatedCrowns(data.annotations)
    blocks = {b.name: b.geometry for b in data.blocks}
    methods = {"RGB": loadRun(runs, "mrcnnRgb")[1], "CC": loadRun(runs, "ccP1")[1],
               "RGB+CC": e4.union(reference, blocks, os.path.join(runs, "mrcnnRgb"), os.path.join(runs, "ccP1"))}
    if exists(exp4, "ccAutoRgbP1"):
        methods["CC auto"] = loadRun(exp4, "ccAutoRgbP1")[1]
        methods["RGB+CC auto"] = e4.union(reference, blocks, os.path.join(runs, "mrcnnRgb"),
                                           os.path.join(exp4, "ccAutoRgbP1"))
    none = {}
    if exists(exp4, "deepforestRgb"):
        evaluator = Evaluator(reference)
        df = RunSet.load(os.path.join(exp4, "deepforestRgb"))
        none["DeepForest"] = blockScores(df, blocks, evaluator, blocks)
        if exists(exp4, "ccAutoDfP1"):
            cc = RunSet.load(os.path.join(exp4, "ccAutoDfP1"))
            none["CC on DF"] = blockScores(cc, blocks, evaluator, blocks)
            none["DF+CC on DF"] = unionScores(df, cc, blocks, evaluator)
        none = {m: {blockKey(b): s for b, s in scores.items()} for m, scores in none.items()}
    return e4.shared(methods)[0], (e4.shared(none)[0] if none else {})


def curvePoints(areas, curves):
    """{method: {(fraction, seed): folds of all areas}}, each area restricted to its full run's blocks."""
    out = {"network": {}, "cc": {}}
    perArea = {}
    for area, _ in areas:
        folder = os.path.join(curves, area)
        network = networkPoints(folder)
        keep = suffixes(network[(1.0, 1)]) if (1.0, 1) in network else None
        path = os.path.join(folder, "ccCurve_ccP1.json")
        cc = {}
        if os.path.exists(path):
            with open(path) as handle:
                cc = {(p["fraction"], p["seed"]): p["folds"] for p in json.load(handle)["points"]}
        restrict = lambda folds: [f for f in folds if keep is None or dc.blockSuffix(f["block"]) in keep]
        perArea[area] = {"network": {k: restrict(v) for k, v in network.items()},
                         "cc": {k: restrict(v) for k, v in cc.items()}}
    for method in out:
        keys = set.intersection(*[set(perArea[a][method]) for a, _ in areas])
        for key in keys:
            out[method][key] = [f for a, _ in areas for f in perArea[a][method][key]]
    return out


def curve(areas, curves):
    points = curvePoints(areas, curves)
    # the crowns used are recorded by the CC curve; the network shares its draws
    crowns = {key: float(np.mean([f.get("crownsUsed", 0) for f in folds])) for key, folds in points["cc"].items()}
    rows = []
    for method, byDraw in points.items():
        for fraction in sorted({f for f, _ in byDraw}):
            keys = [k for k in byDraw if k[0] == fraction]
            draws = [pooled(byDraw[k]) for k in keys]
            used = [crowns[k] for k in keys if k in crowns]
            rows.append({"method": method, "fraction": fraction, "draws": len(draws),
                         "f1": float(np.mean([d["f1"] for d in draws])),
                         "f1Sd": float(np.std([d["f1"] for d in draws])),
                         "recall": float(np.mean([d["recall"] for d in draws])),
                         "precision": float(np.mean([d["precision"] for d in draws])),
                         "crownsUsedPerFold": 0.0 if fraction == 0 else (float(np.mean(used)) if used else None)})
    return rows


def experiment4(areas, curves):
    parts = [hybrid(folder) for _, folder in areas]
    names = [a for a, _ in areas]
    out = {"hybrid": {m: pooled(f) for m, f in mergeMethods({a: p[0] for a, p in zip(names, parts)}).items()}}
    if all(p[1] for p in parts):
        out["noAnnotation"] = {m: pooled(f) for m, f in mergeMethods({a: p[1] for a, p in zip(names, parts)}).items()}
    out["curve"] = curve(areas, curves)
    return out


def report(out):
    e = out["exp1"]
    print("exp1  F1: " + "  ".join("%s %.3f" % (k, v["f1"]) for k, v in sorted(e["f1"].items())))
    for m, r in e["p1MinusLidar"].items():
        print("      P1 - LiDAR, %-9s %+.3f [%+.3f, %+.3f]  p %.3f" % (m, r["pooledDifference"], *r["interval95"], r["p"]))
    for s, rows in e["ccMinusBaseline"].items():
        for b, r in rows.items():
            print("      CC - %-9s on %-5s %+.3f  Holm p %.3f" % (b, s, r["pooledDifference"], r["pHolm"]))
    r = out["exp2"]["rgbVsChm"]
    print("exp2  RGB %.3f  CHM %.3f  RGB - CHM %+.3f [%+.3f, %+.3f]  wins %d/%d  p %.3f" % (
        r["pooledReference"], r["pooledOther"], r["pooledDifference"], *r["interval95"], r["referenceWins"], r["blocks"], r["p"]))
    for m in ("RGB", "4-channel", "CC alone", "CHMnet alone", "RGB+CC:union", "RGB+CHMnet:union"):
        if m in out["exp3"]["methods"]:
            v = out["exp3"]["methods"][m]
            print("exp3  %-18s R %.3f  P %.3f  F1 %.3f" % (m, v["recall"], v["precision"], v["f1"]))
    for m, r in out["exp3"]["tests"].items():
        print("      %-18s - RGB %+.3f  Holm p %.3f" % (m, r["pooledDifference"], r["pHolm"]))
    for part in ("hybrid", "noAnnotation"):
        for m, v in out["exp4"].get(part, {}).items():
            print("exp4  %-13s %-12s R %.3f  P %.3f  F1 %.3f" % (part, m, v["recall"], v["precision"], v["f1"]))
    for r in out["exp4"]["curve"]:
        print("exp4  curve %-8s %5.1f%%  draws %d  F1 %.3f  (crowns per fold %s)" % (
            r["method"], 100 * r["fraction"], r["draws"], r["f1"],
            "-" if r["crownsUsedPerFold"] is None else "%.0f" % r["crownsUsedPerFold"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--area", action="append", default=None, help="name=FOLDER, instead of the three areas")
    parser.add_argument("--curves", default=os.path.join(DB, "exp4", "curves"))
    parser.add_argument("--output", default=os.path.join(DB, "tereljOne.json"))
    args = parser.parse_args(argv)
    areas = [tuple(a.split("=", 1)) for a in args.area] if args.area else AREAS
    rng = np.random.default_rng(0)
    out = {"areas": [a for a, _ in areas],
           "exp1": experiment1(areas, rng), "exp2": experiment2(areas, rng),
           "exp3": experiment3(areas, rng), "exp4": experiment4(areas, args.curves)}
    report(out)
    with open(args.output, "w") as handle:
        json.dump(out, handle, indent=1, default=float)
    print("written to %s" % args.output)


if __name__ == "__main__":
    main()
