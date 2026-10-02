"""
Across sites: does a hybrid of connected components and deep learning beat
the best of its parts?

    python -m tt.studySummary --root "<study folder>" --output summary

Reads, for every site folder under --root (as tt's siteStudy.sh writes them):

  runs/<name>/results.json            each single method's cross-validation
  report/fused/<box>_<height>/summary.json   each combination, every strategy
  calibration/calibration.json        CC calibrated from RGB Mask R-CNN zones

and writes <output>/sites.csv (one row per site and method) and
<output>/summary.json, and prints two tables:

  per site     F1 of every single method and of the calibrated CC
  per hybrid   each combination strategy (and each calibration confidence)
               against the best single method at the same site: mean
               difference, sites won, Wilcoxon over sites

The comparison is deliberately hard on the hybrids. A hybrid is one fixed
strategy across all sites, never the best strategy picked per site, while
the single method it is compared with is whichever did best at that site.
So a hybrid that wins here beats each of its parts, not just the weaker one.
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

ALONE = {"boxes", "points"}      # a combination's parts, repeated in its table
RECALL_WEIGHT = 0.6              # as tt.dl.dlCommon.RECALL_WEIGHT
METRICS = ("f1", "weighted")


def score(pooled, metric):
    """F1, or the recall-weighted mean (computed when an old run lacks it)."""
    if metric == "weighted" and "weighted" not in pooled:
        return (RECALL_WEIGHT * pooled["recall"]
                + (1 - RECALL_WEIGHT) * pooled["precision"])
    return pooled[metric]


def loadJson(path):
    with open(path) as handle:
        return json.load(handle)


def singles(siteDir):
    """
    ({method: pooled scores}, {method ceiling: whole-area best}) from the
    site's cross-validated runs. The ceilings (connected components tuned
    and scored on the whole area) are optimistic, so they are shown but
    never count as a method.
    """
    out, ceilings = {}, {}
    for path in sorted(glob.glob(os.path.join(siteDir, "runs", "*",
                                              "results.json"))):
        data = loadJson(path)
        if "pooled" in data:
            name = os.path.basename(os.path.dirname(path))
            out[name] = data["pooled"]
            if "wholeAreaBest" in data["pooled"]:
                ceilings[name + " ceiling"] = data["pooled"]["wholeAreaBest"]
    return out, ceilings


def hybrids(siteDir):
    """{'box+height:strategy': pooled} from the report's combinations."""
    out = {}
    pattern = os.path.join(siteDir, "report", "fused", "*", "summary.json")
    for path in sorted(glob.glob(pattern)):
        pair = os.path.basename(os.path.dirname(path)).replace("_", "+", 1)
        for strategy, entry in loadJson(path).items():
            if strategy not in ALONE:
                out["%s:%s" % (pair, strategy)] = entry["pooled"]
    return out


def calibrated(siteDir):
    """{'calibrated@<confidence>': pooled} from the calibration."""
    path = os.path.join(siteDir, "calibration", "calibration.json")
    if not os.path.exists(path):
        return {}
    return {"calibrated@%s" % level: row["calibrated"]
            for level, row in loadJson(path)["summary"].items()}


def pseudoTuned(siteDir):
    """
    ({'pseudo:<scorer>': pooled}, {bound}) from tt.pseudoTuning: CC tuned on
    the RGB Mask R-CNN's boxes is a hybrid; tuning on each block's own crowns
    is an optimistic bound, shown with the ceilings.
    """
    path = os.path.join(siteDir, "pseudoTuning", "pseudoTuning.json")
    if not os.path.exists(path):
        return {}, {}
    pooled = loadJson(path)["pooled"]
    tuned = {"pseudo:%s" % k: v for k, v in pooled.items() if k != "real"}
    bound = {"cc own-block bound": pooled["real"]} if "real" in pooled else {}
    return tuned, bound


def readSite(siteDir):
    methods, ceilings = singles(siteDir)
    tuned, bound = pseudoTuned(siteDir)
    ceilings.update(bound)
    site = {"singles": methods, "ceilings": ceilings,
            "hybrids": dict(hybrids(siteDir))}
    site["hybrids"].update(calibrated(siteDir))
    site["hybrids"].update(tuned)
    if site["singles"]:
        site["bestSingle"] = {m: max(site["singles"], key=lambda n: score(
            site["singles"][n], m)) for m in METRICS}
    return site


def readStudy(root):
    sites = {}
    for siteDir in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(siteDir):
            site = readSite(siteDir)
            if site["singles"]:
                sites[os.path.basename(siteDir)] = site
    return sites


def compareHybrids(sites, metric="f1"):
    """Each hybrid against the best single method (by metric) at each site."""
    names = sorted({h for s in sites.values() for h in s["hybrids"]})
    rows = []
    for name in names:
        differences = [score(s["hybrids"][name], metric) -
                       score(s["singles"][s["bestSingle"][metric]], metric)
                       for s in sites.values() if name in s["hybrids"]]
        d = np.array(differences)
        p = wilcoxon(d).pvalue if len(d) > 1 and np.any(d) else float("nan")
        rows.append({"hybrid": name, "sites": len(d),
                     "meanDifference": float(d.mean()),
                     "wins": int((d > 0).sum()), "p": float(p)})
    return sorted(rows, key=lambda r: -r["meanDifference"])


def writeCsv(sites, path):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["site", "kind", "method", "recall", "precision",
                         "f1", "weighted", "crowns"])
        for name, site in sites.items():
            for kind in ("singles", "ceilings", "hybrids"):
                for method, p in sorted(site[kind].items()):
                    writer.writerow([name, kind[:-1], method,
                                     "%.4f" % p["recall"],
                                     "%.4f" % p["precision"],
                                     "%.4f" % p["f1"],
                                     "%.4f" % score(p, "weighted"),
                                     p.get("crowns", "")])


def shortLabel(name):
    """Column headings short enough to stay distinct."""
    for old, new in (("pseudo:pseudoF1", "pseudoF1"), ("pseudo:zones", "pzones"),
                     ("calibrated", "calib"), ("cc own-block bound", "ownBlock*"),
                     (" ceiling", " ceil*")):
        name = name.replace(old, new)
    return name


def printSites(sites, metric="f1"):
    methods = sorted({m for s in sites.values() for m in s["singles"]})
    cal = sorted({h for s in sites.values() for h in s["hybrids"]
                  if h.startswith(("calibrated", "pseudo"))})
    ceilings = sorted({c for s in sites.values() for c in s["ceilings"]})
    columns = methods + ceilings + cal
    print("\n%s per site. Best single method marked *. Columns ending in * are "
          "optimistic bounds, never methods: 'ceil' tuned and scored on the "
          "whole site, 'ownBlock' tuned on each block's own crowns."
          % {"f1": "F1", "weighted": "0.6 R + 0.4 P"}[metric])
    print("%-34s " % "site" + " ".join("%14s" % shortLabel(c)[:14]
                                       for c in columns))
    for name, site in sites.items():
        cells = []
        for c in columns:
            p = (site["singles"].get(c) or site["ceilings"].get(c)
                 or site["hybrids"].get(c))
            mark = "*" if c == site.get("bestSingle", {}).get(metric) else " "
            cells.append("%13s%s" % ("%.3f" % score(p, metric) if p else "-",
                                     mark))
        print("%-34s " % name[:34] + " ".join(cells))


def printHybrids(rows, metric="f1"):
    print("\nEach hybrid against the best single method at the same site, by %s"
          % {"f1": "F1", "weighted": "0.6 R + 0.4 P"}[metric])
    print("%-52s %5s %8s %6s %7s" % ("hybrid", "sites", "mean dF1", "wins",
                                      "p"))
    for r in rows:
        print("%-52s %5d %+8.3f %3d/%-2d %7.3f"
              % (r["hybrid"][:52], r["sites"], r["meanDifference"], r["wins"],
                 r["sites"], r["p"]))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Single methods, combinations and calibration, across "
                    "sites.")
    parser.add_argument("--root", required=True,
                        help="Folder holding one study folder per site")
    parser.add_argument("--output", default="studySummary")
    args = parser.parse_args(argv)
    sites = readStudy(args.root)
    if not sites:
        print("no site with results under %s" % args.root)
        return 1
    os.makedirs(args.output, exist_ok=True)
    rows = {m: compareHybrids(sites, m) for m in METRICS}
    writeCsv(sites, os.path.join(args.output, "sites.csv"))
    with open(os.path.join(args.output, "summary.json"), "w") as handle:
        json.dump({"sites": sites, "hybrids": rows}, handle, indent=1)
    for metric in METRICS:
        printSites(sites, metric)
        printHybrids(rows[metric], metric)
    return 0


if __name__ == "__main__":
    sys.exit(main())
