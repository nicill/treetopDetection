"""
A box run and any point run of a site combined, as the report combines its
runs, without rerunning the report: for CC runs made after it (a zone-tuned
automatic CC, a run with height-dependent smoothing, ...).

    python -m tt.fuseRuns --site study_procedure/enshurin \\
        --points study_procedure/enshurin/slopes2/smooth/runs/slope --name ccSmooth

Writes report/fused/<box>_<name>/summary.json (and the held-out outputs) with
the box model alone and the union, under the report's own crowns, blocks and
leave-one-block-out tuning, so they sit beside the report's combinations and
the summary reads them as hybrids "<box>+<name>:union".
"""

import argparse
import os
import sys

from tt.dl import dlCommon as dc
from tt.experiments import RunSet
from tt.experiments.fusion import FusionCrossValidation, printSummary
from tt.heightFilter import siteInputs
from tt.scoring import CrownFrame

STRATEGIES = ["boxes", "union"]


def fuse(site, box, points, name, verbose=True):
    meta, crowns, _ = siteInputs(site)
    blocks = meta["blockGeometries"]
    boxRun = RunSet.load(os.path.join(site, "runs", box))
    pointRun = RunSet.load(points)
    validation = FusionCrossValidation.pair(CrownFrame(crowns), dict(blocks),
                                       boxRun, pointRun,
                                       strategies=STRATEGIES, verbose=False)
    summary = validation.run()
    folder = os.path.join(site, "report", "fused", "%s_%s" % (box, name))
    os.makedirs(folder, exist_ok=True)
    dc.saveJson(summary, os.path.join(folder, "summary.json"))
    validation.saveOutputs(folder)
    if verbose:
        print("[fuse] %s: %s + %s" % (os.path.basename(os.path.normpath(site)),
                                      box, name))
        printSummary(summary)
    return summary


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site", required=True, help="a site of the procedure")
    parser.add_argument("--box", default="mrcnnRgb",
                        help="the box run, in the site's runs/ (mrcnnRgb)")
    parser.add_argument("--points", required=True, help="a point run's folder")
    parser.add_argument("--name", required=True,
                        help="its name in the combination (e.g. ccSmooth)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parseArguments(argv)
    fuse(args.site, args.box, args.points, args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
