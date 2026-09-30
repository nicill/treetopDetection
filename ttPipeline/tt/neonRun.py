"""
Connected components over every NEON tile, at fixed settings, pooled.

    python -m tt.neonRun --manifest Data/neon/manifest.csv \\
        --modalFrom ccLidar=runs/ccLidar --modalFrom ccP1=runs/ccP1 \\
        --output neonCC

Reads the manifest tt.neonImport wrote. Each tile's CHM is looked up beside
its RGB: <root>/RGB/<tile>.tif -> <root>/CHM/<tile>_CHM.tif, or another
folder with --chmFolder (CHM050 for the 0.5 m CHM tt.neonChm builds). Tiles
without one are listed and skipped.

The settings are fixed, never tuned on NEON: the detector's defaults and, for
each --modalFrom run, the setting its cross-validation chose most often
(tt.transfer.modalSetting). So nothing about NEON's crowns enters them and
every tile is a held-out test.

Scores are pooled by summing counts over tiles before forming the ratios,
overall and per site, as tt.dl.dlCommon.averageFolds does for folds.

--minHeight defaults to 3 m: the NEON annotators removed vegetation below 3 m,
so a top below it could only ever count as a false positive.

Output: <output>/results.json with every tile, site and setting, and a table.
"""

import argparse
import csv
import os
import re
import sys

from .dl import dlCommon as dc
from .scene import Scene
from .transfer import DEFAULTS, detect, modalSetting

SITE = re.compile(r"[A-Z]{4}")


def chmFor(rgbPath, tile, folder="CHM"):
    root = os.path.dirname(os.path.dirname(rgbPath))
    path = os.path.join(root, folder, tile + "_CHM.tif")
    return path if os.path.exists(path) else None


def siteOf(tile):
    match = SITE.search(tile)
    return match.group(0) if match else "unknown"


def readManifest(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def settingsFrom(modalFrom):
    """Name -> setting: the defaults and each run's modal choice."""
    settings = {"defaults": dict(DEFAULTS)}
    for item in modalFrom:
        name, run = item.split("=", 1)
        setting, count, total = modalSetting(os.path.join(run, "results.json"))
        print("[neon] %s: modal setting chosen in %d of %d folds: %s"
              % (name, count, total, setting))
        settings[name] = setting
    return settings


def scoreTile(row, chmPath, settings, minHeight):
    scene = Scene(chmPath, crownsPath=row["crownPath"],
                  boundaryPath=row["boundaryPath"], resolution=0.0,
                  minHeight=minHeight, verbose=False)
    results = {}
    for name, setting in settings.items():
        result = dc.evaluateDetections(detect(scene, setting), scene.crowns,
                                       scene.boundary)
        result.update({"tile": row["tile"], "site": siteOf(row["tile"]),
                       "pixelM": scene.pixelSize})
        results[name] = result
    return results


def pool(rows):
    """Overall and per-site pooled scores for one setting."""
    sites = sorted({r["site"] for r in rows})
    return {"overall": dc.averageFolds(rows),
            "sites": {s: dc.averageFolds([r for r in rows if r["site"] == s])
                      for s in sites}}


def printTables(pooled):
    print("\n%-12s %6s %7s %7s %7s %7s" % ("setting", "tiles", "crowns",
                                           "recall", "prec", "F1"))
    for name, p in pooled.items():
        o = p["overall"]
        print("%-12s %6d %7d %7.3f %7.3f %7.3f" % (
            name, o["folds"], o["crowns"], o["recall"], o["precision"],
            o["f1"]))
    names = list(pooled)
    print("\nF1 per site\n%-6s %6s %7s " % ("site", "tiles", "crowns")
          + " ".join("%10s" % n[:10] for n in names))
    for site, first in pooled[names[0]]["sites"].items():
        print("%-6s %6d %7d " % (site, first["folds"], first["crowns"])
              + " ".join("%10.3f" % pooled[n]["sites"][site]["f1"]
                         for n in names))


def run(manifestPath, settings, minHeight, outputDir, chmFolder="CHM"):
    perSetting = {name: [] for name in settings}
    missing = []
    for row in readManifest(manifestPath):
        chmPath = chmFor(row["rgb"], row["tile"], chmFolder)
        if chmPath is None:
            missing.append(row["tile"])
            continue
        for name, result in scoreTile(row, chmPath, settings,
                                      minHeight).items():
            perSetting[name].append(result)
    if missing:
        print("[neon] %d tile(s) without a CHM skipped: %s"
              % (len(missing), ", ".join(missing)))
    pooled = {name: pool(rows) for name, rows in perSetting.items() if rows}
    if not pooled:
        raise SystemExit("no tile had a CHM; nothing scored")
    dc.saveJson({"settings": settings, "minHeight": minHeight,
                 "chmFolder": chmFolder,
                 "skipped": missing, "pooled": pooled, "tiles": perSetting},
                os.path.join(outputDir, "results.json"))
    printTables(pooled)
    return pooled


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Connected components over the NEON tiles at fixed "
                    "settings, pooled overall and per site.")
    parser.add_argument("--manifest", required=True,
                        help="manifest.csv written by tt.neonImport")
    parser.add_argument("--modalFrom", action="append", default=[],
                        metavar="NAME=RUN",
                        help="Also score this run's modal setting")
    parser.add_argument("--minHeight", type=float, default=3.0)
    parser.add_argument("--chmFolder", default="CHM",
                        help="Folder beside RGB/ holding <tile>_CHM.tif; "
                             "CHM050 for the one tt.neonChm builds at 0.5 m")
    parser.add_argument("--output", default="neonCC")
    args = parser.parse_args(argv)
    run(args.manifest, settingsFrom(args.modalFrom), args.minHeight,
        args.output, args.chmFolder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
