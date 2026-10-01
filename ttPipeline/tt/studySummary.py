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


def loadJson(path):
    with open(path) as handle:
        return json.load(handle)


def singles(siteDir):
    """{method: pooled scores} from the site's cross-validated runs."""
    out = {}
    for path in sorted(glob.glob(os.path.join(siteDir, "runs", "*",
                                              "results.json"))):
        data = loadJson(path)
        if "pooled" in data:
            out[os.path.basename(os.path.dirname(path))] = data["pooled"]
    return out


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


def readSite(siteDir):
    site = {"singles": singles(siteDir), "hybrids": dict(hybrids(siteDir))}
    site["hybrids"].update(calibrated(siteDir))
    if site["singles"]:
        best = max(site["singles"], key=lambda n: site["singles"][n]["f1"])
        site["bestSingle"] = best
    return site


def readStudy(root):
    sites = {}
    for siteDir in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(siteDir):
            site = readSite(siteDir)
            if site["singles"]:
                sites[os.path.basename(siteDir)] = site
    return sites


def compareHybrids(sites):
    """Each hybrid against the best single method at each site it ran on."""
    names = sorted({h for s in sites.values() for h in s["hybrids"]})
    rows = []
    for name in names:
        differences = [s["hybrids"][name]["f1"] -
                       s["singles"][s["bestSingle"]]["f1"]
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
                         "f1", "crowns"])
        for name, site in sites.items():
            for kind in ("singles", "hybrids"):
                for method, p in sorted(site[kind].items()):
                    writer.writerow([name, kind[:-1], method,
                                     "%.4f" % p["recall"],
                                     "%.4f" % p["precision"],
                                     "%.4f" % p["f1"], p.get("crowns", "")])


def printSites(sites):
    methods = sorted({m for s in sites.values() for m in s["singles"]})
    cal = sorted({h for s in sites.values() for h in s["hybrids"]
                  if h.startswith("calibrated")})
    columns = methods + cal
    print("F1 per site (best single marked *)")
    print("%-34s " % "site" + " ".join("%14s" % c[:14] for c in columns))
    for name, site in sites.items():
        cells = []
        for c in columns:
            p = site["singles"].get(c) or site["hybrids"].get(c)
            mark = "*" if c == site.get("bestSingle") else " "
            cells.append("%13s%s" % ("%.3f" % p["f1"] if p else "-", mark))
        print("%-34s " % name[:34] + " ".join(cells))


def printHybrids(rows):
    print("\nEach hybrid against the best single method at the same site")
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
    rows = compareHybrids(sites)
    writeCsv(sites, os.path.join(args.output, "sites.csv"))
    with open(os.path.join(args.output, "summary.json"), "w") as handle:
        json.dump({"sites": sites, "hybrids": rows}, handle, indent=1)
    printSites(sites)
    printHybrids(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
