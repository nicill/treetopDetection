"""
Experiment 2 — Mask R-CNN on RGB against Mask R-CNN on the CHM (P1), at every
site. No LiDAR.

Per site:
  - each network's pooled recall, precision and F1 (its own cross-validated run);
  - RGB against the CHM network (P1): the pooled F1 difference with a 95%
    bootstrap interval over blocks, the blocks RGB wins, the Wilcoxon over blocks;
  - tree by tree, RGB against the CHM network: the crowns found by both,
    only by RGB, only by the CHM network, by neither, with their median top
    height and crown area; and the union's recall (the most a combination of
    the two could find, experiment 3's ceiling).
Then across sites: RGB minus CHM, sites in RGB's favour, the Wilcoxon over sites.

Run from ttPipeline (it needs tt):

    python experiment2.py                       # every site, results also in experiment2.json
    python experiment2.py --site name=FOLDER    # other sites (FOLDER holds ds/ and runs/)
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import wilcoxon

from tt.analysis import Analysis
from tt.data import TDData
from tt.experiments import RunSet
from tt.runCompare import compare, loadRun

DB = os.environ.get("DROP") or os.path.expanduser("~/Yago Lab Dropbox/Meetings/2026TreeDet")
QUEBEC = ["20230712_afcamoisan_itrf20", "20230712_afcahoule_itrf20",
          "20230606_cbblackburn5", "20230608_cbbernard4", "20230606_cbblackburn6"]
SITES = ([(s.split("_")[1], os.path.join(DB, "study_edge", s)) for s in QUEBEC]
         + [("koiwai", os.path.join(DB, "study_edge", "koiwainoujo220616"))]
         + [("terelj" + a, os.path.join(DB, "study_terelj" + a, "terelj" + a))
            for a in ("First", "Second", "Third")]
         + [("enshurinV2", os.path.join(DB, "study_enshurinVlad2"))])
DATASETS = ("P1", "chm")            # the dataset that gives crowns, area, blocks
BOOTSTRAP = 10000


def has(runs, name):
    return os.path.exists(os.path.join(runs, name, "results.json"))


def pooledLine(runs, name):
    p = json.load(open(os.path.join(runs, name, "results.json")))["pooled"]
    return "    %-11s R %.3f  P %.3f  F1 %.3f" % (name, p["recall"], p["precision"], p["f1"])


def site(name, folder, rng):
    runs = os.path.join(folder, "runs")
    if not (has(runs, "mrcnnRgb") and has(runs, "mrcnnP1")):
        print("== %s: no mrcnnRgb and mrcnnP1 in %s; skipped" % (name, runs))
        return None
    print("== %s" % name)
    record = {"site": name}
    for run in ("mrcnnRgb", "mrcnnP1"):
        print(pooledLine(runs, run))

    row = compare(loadRun(runs, "mrcnnRgb")[1], loadRun(runs, "mrcnnP1")[1], BOOTSTRAP, rng)
    print("    RGB - CHM  %+.3f  [%+.3f, %+.3f]  RGB wins %d of %d blocks  p %.3f"
          % (row["pooledDifference"], row["interval95"][0], row["interval95"][1],
             row["referenceWins"], row["blocks"], row["p"]))
    record["rgbVsChm"] = row

    dataset = next((os.path.join(folder, "ds", d) for d in DATASETS
                    if os.path.exists(os.path.join(folder, "ds", d, "dataset.json"))), None)
    if dataset is None:
        print("    tree by tree: no ds/P1 or ds/chm dataset; skipped")
        return record
    table = Analysis(TDData.fromDataset(dataset)).treeByTree(
        {"RGB": RunSet.load(os.path.join(runs, "mrcnnRgb")),
         "CHM": RunSet.load(os.path.join(runs, "mrcnnP1"))})
    groups = table.agreement("RGB", "CHM")
    total = sum(g["count"] for g in groups.values())
    print("    tree by tree (%d crowns)       crowns  share  median top m  median area m2" % total)
    for label in ("both", "only RGB", "only CHM", "neither"):
        g = groups[label]
        print("      %-24s %6d  %4.0f%%  %12s  %14s" % (
            label, g["count"], 100.0 * g["count"] / max(total, 1),
            "%.1f" % g["peak"] if g["peak"] is not None else "-",
            "%.1f" % g["area"] if g["area"] is not None else "-"))
    union = 1.0 - groups["neither"]["count"] / float(max(total, 1))
    print("      union recall (either network finds it)  %.3f" % union)
    record["agreement"] = {k: {"count": v["count"], "peak": v["peak"], "area": v["area"]}
                           for k, v in groups.items()}
    record["unionRecall"] = union
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site", action="append", default=None,
                        help="name=FOLDER, instead of the default sites")
    parser.add_argument("--output", default="experiment2.json")
    args = parser.parse_args(argv)
    sites = ([tuple(s.split("=", 1)) for s in args.site] if args.site else SITES)
    rng = np.random.default_rng(0)
    records = [r for r in (site(n, f, rng) for n, f in sites) if r]

    diffs = [r["rgbVsChm"]["pooledDifference"] for r in records]
    if len(diffs) >= 2:
        print("\nacross %d sites: RGB minus the CHM network (P1), pooled F1: RGB ahead at %d, "
              "mean %+.3f, Wilcoxon over sites p %.4f"
              % (len(diffs), sum(d > 0 for d in diffs), float(np.mean(diffs)),
                 wilcoxon(diffs).pvalue if any(diffs) else 1.0))
        only = [(r["agreement"]["only RGB"]["count"], r["agreement"]["only CHM"]["count"])
                for r in records if "agreement" in r]
        if only:
            print("crowns found by one network only, summed over sites: only RGB %d, only CHM %d"
                  % (sum(a for a, _ in only), sum(b for _, b in only)))
    with open(args.output, "w") as handle:
        json.dump(records, handle, indent=1, default=float)
    print("\nwritten to %s" % args.output)


if __name__ == "__main__":
    main()
