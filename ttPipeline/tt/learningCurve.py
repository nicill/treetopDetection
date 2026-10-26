"""
How much annotation does each method need? The learning curve.

    python -m tt.learningCurve cc --ccRun <site>/runs/ccP1 \\
        --dataset <site>/ds/chm --tiles <site>/ds/rgb \\
        --transferFrom <other site>/runs/ccP1 --jobs 20 \\
        --output <curve>/<site>/ccCurve_ccP1.json
    python -m tt.learningCurve summary --root <curve>

For every held-out block, the annotation is cut down to a fraction of each
training block's tiles (dlCommon.drawTiles: whole tiles, at random, the same
fraction from every block), drawn several times with fixed seeds. Connected
components chooses its setting on the crowns of the drawn tiles only; Mask
R-CNN (tt.dl maskrcnn --trainFraction --subsetSeed) trains and chooses its
operating point on the same tiles, since the draw depends only on the seed,
the block and the tile positions, which the RGB and CHM datasets share. Both
are scored on the whole held-out block, as in every fold.

Connected components needs no detection here: every setting's tops are in
the run's detections.npz (tt.dl concomp --saveDetections), and only the
choice among them changes with the annotation.

The point with no annotation of the site:
  connected components  the other site's best setting over its whole area
                        (results.json pooled.wholeAreaBest), from the same
                        grid, so it is a lookup
  Mask R-CNN            a network trained on every block of the other site
                        (tt.dl maskrcnn --transferFrom)
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

from .data.annotations import readCrowns
from .data.patches import PatchSet
from .data.site import TDData
from .dl import dlCommon as dc
from .experiments import ParameterGrid
from .experiments.crossValidation import loadAllDetections
from .experiments.fixed import LearningCurve, poolPoint
from .pseudoTuning import runGrid
from .scoring import CrownFrame, Evaluator

def ccCurve(args):
    """The curve of one CC run, from the detections it saved."""
    grid, _ = runGrid(args.ccRun)
    site = TDData.fromDataset(args.dataset)
    # every crown in the file: drawn tiles may reach past the scored area
    crowns, _ = readCrowns(site.annotations.crownsPath, site.annotations.crs)
    return LearningCurve(
        site, ParameterGrid(grid).expand(),
        loadAllDetections(os.path.join(args.ccRun, "detections.npz")),
        PatchSet.load(args.tiles or args.dataset), Evaluator(CrownFrame(crowns)),
        [float(v) for v in args.fractions.split(",")],
        [int(v) for v in args.seeds.split(",")], args.buffer, args.jobs)


def ccCommand(args):
    curve = ccCurve(args)
    record = {"ccRun": args.ccRun, "objective": dc.tuningObjective(),
              "points": curve.run()}
    if args.transferFrom:
        other = dc.loadJson(os.path.join(args.transferFrom, "results.json"))
        record["transfer"] = dict(
            {"from": args.transferFrom},
            **curve.transfer(other["pooled"]["wholeAreaBest"]["settings"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    dc.saveJson(record, args.output)
    for point in record["points"]:
        p = point["pooled"]
        print("  %4.0f%% seed %d: %5.0f crowns/fold  R %.3f P %.3f F1 %.3f "
              "W %.3f" % (100 * point["fraction"], point["seed"],
                          p["crownsUsedPerFold"], p["recall"], p["precision"],
                          p["f1"], p["weighted"]))
    if args.transferFrom:
        p = record["transfer"]["pooled"]
        print("  other site's setting: R %.3f P %.3f F1 %.3f W %.3f"
              % (p["recall"], p["precision"], p["f1"], p["weighted"]))
    print("[curve] wrote %s" % args.output)
    return 0


# ---------------------------------------------------------------------- #
# summary
# ---------------------------------------------------------------------- #



def suffixes(folds):
    return {dc.blockSuffix(f["block"]) for f in folds}


def repool(folds, keep):
    """The folds whose block is in keep, pooled; all of them if keep is None."""
    chosen = [f for f in folds if keep is None or dc.blockSuffix(f["block"])
              in keep]
    return poolPoint([dict(f, crownsUsed=f.get("crownsUsed", 0))
                      for f in chosen])


def networkPoints(siteDir, prefix="mrcnn"):
    """{(fraction, seed): folds} of the site's Mask R-CNN runs (folders
    <prefix>_f<fraction>_s<seed>; "mrcnnChm" for the CHM network's)."""
    points = {}
    run = re.compile(r"^%s_f([0-9.]+)_s(\d+)$" % re.escape(prefix))
    for path in glob.glob(os.path.join(siteDir, prefix + "_f*_s*",
                                       "results.json")):
        match = run.search(os.path.basename(os.path.dirname(path)))
        if match:
            points[(float(match.group(1)), int(match.group(2)))] = \
                dc.loadJson(path)["folds"]
    return points


def crownsByDraw(curves):
    """{(fraction, seed): {block suffix: crowns used}} from any CC curve."""
    found = {}
    for record in curves:
        for point in record["points"]:
            found.setdefault((point["fraction"], point["seed"]), {
                dc.blockSuffix(f["block"]): f["crownsUsed"]
                for f in point["folds"]})
    return found


def rowsFor(name, points, keep, crowns):
    """One row per fraction: mean over draws, and the F1 spread."""
    rows = []
    for fraction in sorted({f for f, _ in points}):
        pooled = []
        for (f, seed), folds in sorted(points.items()):
            if f != fraction:
                continue
            p = repool(folds, keep)
            used = crowns.get((f, seed), {})
            p["crownsUsedPerFold"] = float(np.mean(
                [used[dc.blockSuffix(x["block"])] for x in folds
                 if dc.blockSuffix(x["block"]) in used])) if used else np.nan
            pooled.append(p)
        rows.append(summaryRow(name, "%g%%" % (100 * fraction), pooled))
    return rows


def summaryRow(method, point, pooled):
    mean = {k: float(np.mean([p[k] for p in pooled]))
            for k in ("recall", "precision", "f1", "weighted",
                      "crownsUsedPerFold")}
    mean.update(method=method, point=point, draws=len(pooled),
                f1Sd=float(np.std([p["f1"] for p in pooled])))
    return mean


def siteRows(siteDir, prefix="mrcnn", name="mrcnnRgb"):
    """The site's curve rows; prefix and name select another network's runs
    (the CHM network: "mrcnnChm", "mrcnnChm"), pooled over its own blocks."""
    curves = [dc.loadJson(p) for p in sorted(glob.glob(
        os.path.join(siteDir, "ccCurve_*.json")))]
    networks = networkPoints(siteDir, prefix)
    full = networks.get((1.0, 1))
    keep = suffixes(full) if full else None
    crowns = crownsByDraw(curves)
    rows = []
    transfer = os.path.join(siteDir, prefix + "_transfer", "results.json")
    if os.path.exists(transfer):
        rows.append(summaryRow(name, "other site", [dict(
            repool(dc.loadJson(transfer)["folds"], keep),
            crownsUsedPerFold=0.0)]))
    rows += rowsFor(name, networks, keep, crowns)
    for record in curves:
        name = os.path.basename(os.path.normpath(record["ccRun"]))
        if "transfer" in record:
            rows.append(summaryRow(name, "other site", [dict(
                repool(record["transfer"]["folds"], keep),
                crownsUsedPerFold=0.0)]))
        points = {(p["fraction"], p["seed"]): p["folds"]
                  for p in record["points"]}
        rows += rowsFor(name, points, keep, crowns)
    return rows


def summaryCommand(args):
    lines = []
    for siteDir in sorted(glob.glob(os.path.join(args.root, "*", ""))):
        rows = siteRows(siteDir)
        if not rows:
            continue
        lines.append("\n%s (objective %s; mean over draws)"
                     % (os.path.basename(os.path.normpath(siteDir)),
                        dc.tuningObjective()))
        lines.append("  %-10s %-10s %6s %5s  %5s  %5s  %5s  %5s  %5s"
                     % ("method", "annotated", "crowns", "draws", "R", "P",
                        "F1", "+-", "W"))
        for r in rows:
            lines.append("  %-10s %-10s %6.0f %5d  %.3f  %.3f  %.3f  %.3f  "
                         "%.3f" % (r["method"], r["point"],
                                   r["crownsUsedPerFold"], r["draws"],
                                   r["recall"], r["precision"], r["f1"],
                                   r["f1Sd"], r["weighted"]))
    text = "\n".join(lines) + "\n"
    print(text)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(text)
    return 0


# ---------------------------------------------------------------------- #

def parseArguments(argv=None):
    parser = argparse.ArgumentParser(description="Learning curves.")
    commands = parser.add_subparsers(dest="command", required=True)
    cc = commands.add_parser("cc", help="connected components' curve")
    cc.add_argument("--ccRun", required=True)
    cc.add_argument("--dataset", required=True,
                    help="the CHM dataset the run was made on")
    cc.add_argument("--tiles", default=None,
                    help="dataset whose tiles are drawn (the RGB one, so the "
                         "draw is Mask R-CNN's); default --dataset")
    cc.add_argument("--fractions", default="0.05,0.1,0.25,0.5,1")
    cc.add_argument("--seeds", default="1,2,3")
    cc.add_argument("--buffer", type=float, default=10.0,
                    help="m kept clear of the held-out block, as Mask R-CNN")
    cc.add_argument("--transferFrom", default=None,
                    help="the other site's CC run: its whole-area best "
                         "setting, untuned here")
    cc.add_argument("--jobs", type=int, default=8)
    cc.add_argument("--output", required=True)
    summary = commands.add_parser("summary", help="tables of every site")
    summary.add_argument("--root", required=True)
    summary.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parseArguments(argv)
    return ccCommand(args) if args.command == "cc" else summaryCommand(args)


if __name__ == "__main__":
    sys.exit(main())
