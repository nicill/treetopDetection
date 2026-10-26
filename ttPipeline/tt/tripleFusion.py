"""
RGB Mask R-CNN, CHM Mask R-CNN and CC combined: the three find partly
different trees. On a site's saved predictions, no retraining:

    python -m tt.tripleFusion --site study_procedure/koiwainoujo220616

The strategies are tt.detectors.fusion.Triple (and, with --reduced, the
union, UnionClean and AgreeClean), run by tt.experiments.fusion.

The rule, its thresholds tuned per fold on the validation block like every
other combination (the "off" values are in the grid):

  1. agreement   an RGB box and a CHM box overlapping by IoU >= MATCH_IOU are
                 one detection; it is kept when either score >= high, or both
                 >= low. A box with no partner is kept when its score >= high.
  2. CC's vote   a box not kept by 1 is kept when it holds a CC top and its
                 score >= vote.
  3. CC's trees  a CC top inside no kept box is added when its prominence (how
                 far the CHM falls from it before reaching higher ground) >=
                 prominence; among such tops inside one CHM box, only the most
                 prominent (CC's over-splitting cleaned by the CHM network).

Writes report/fused/mrcnnRgb_triple/summary.json beside the report's
combinations (the summary reads it as "mrcnnRgb+triple:triple"), with "boxes"
(the RGB network alone) checked against the report's.
"""

import argparse
import os
import sys

from tt.dl import dlCommon as dc
from tt.experiments import RunSet
from tt.experiments.fusion import FusionCrossValidation
from tt.heightFilter import siteInputs
from tt.scoring import CrownFrame


def height(site):
    return "Lidar" if os.path.isdir(os.path.join(site, "runs", "ccLidar")) else "P1"


def fuseSite(site, ccName=None, verbose=True):
    h = height(site)
    meta, crowns, surface = siteInputs(site)
    blocks = meta["blockGeometries"]
    ccName = ccName or "cc" + h
    runs = [RunSet.load(os.path.join(site, "runs", n))
            for n in ("mrcnnRgb", "mrcnn" + h, ccName)]
    validation = FusionCrossValidation.threeWay(
        CrownFrame(crowns), dict(blocks), *runs, surface=surface,
        verbose=verbose)
    summary = validation.run()
    folder = os.path.join(site, "report", "fused", "mrcnnRgb_triple")
    saved = os.path.join(site, "report", "fused", "mrcnnRgb_cc" + h, "summary.json")
    if os.path.exists(saved):
        a = dc.loadJson(saved)["boxes"]["pooled"]
        b = summary["boxes"]["pooled"]
        if (a["hits"], a["predictions"]) != (b["hits"], b["predictions"]):
            raise RuntimeError("%s: the RGB network alone scores %d / %d here but "
                               "%d / %d in the report: not the same inputs"
                               % (site, b["hits"], b["predictions"], a["hits"],
                                  a["predictions"]))
    os.makedirs(folder, exist_ok=True)
    dc.saveJson(summary, os.path.join(folder, "summary.json"))
    validation.saveOutputs(folder)
    return summary


def fuseSiteReduced(site, ccName=None, verbose=True):
    """The reduced versions at one site, into report/fused/mrcnnRgb_lite; the
    union scored here must match the report's, so the rows compare like with like."""
    h = height(site)
    meta, crowns, surface = siteInputs(site)
    blocks = meta["blockGeometries"]
    ccName = ccName or "cc" + h
    runs = [RunSet.load(os.path.join(site, "runs", n))
            for n in ("mrcnnRgb", "mrcnn" + h, ccName)]
    validation = FusionCrossValidation.threeWay(
        CrownFrame(crowns), dict(blocks), *runs, surface=surface,
        reduced=True, verbose=verbose)
    summary = validation.run()
    saved = os.path.join(site, "report", "fused", "mrcnnRgb_" + ccName, "summary.json")
    if os.path.exists(saved):
        reference = dc.loadJson(saved)
        for name in ("boxes", "union"):
            a, b = reference[name]["pooled"], summary[name]["pooled"]
            if (a["hits"], a["predictions"]) != (b["hits"], b["predictions"]):
                raise RuntimeError("%s: '%s' scores %d / %d here but %d / %d in the "
                                   "report: not the same inputs"
                                   % (site, name, b["hits"], b["predictions"],
                                      a["hits"], a["predictions"]))
    folder = os.path.join(site, "report", "fused", "mrcnnRgb_lite")
    os.makedirs(folder, exist_ok=True)
    dc.saveJson(summary, os.path.join(folder, "summary.json"))
    validation.saveOutputs(folder)
    return summary


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--reduced", action="store_true",
                        help="the reduced versions (union, unionClean, agreeClean) "
                             "instead of the three-way rule")
    parser.add_argument("--site", action="append", required=True,
                        help="a site folder of the procedure (repeatable)")
    parser.add_argument("--cc", default=None,
                        help="the CC run in the site's runs/ (default cc<height>, "
                             "tuned on crowns; e.g. ccAutoLidar)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parseArguments(argv)
    for site in args.site:
        print("[triple] %s" % os.path.basename(os.path.normpath(site)))
        (fuseSiteReduced if args.reduced else fuseSite)(site, args.cc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
