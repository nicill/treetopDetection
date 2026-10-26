"""
One run against others on the same blocks: the pooled difference (every tree
counts once, as in the pooled F1) with a bootstrap interval over blocks, and
the per-block test (every block counts once).

    python -m tt.runCompare --root "$DB/study_enshurinNew/runs" \\
        --reference ccP1 --others lmvwP1 watershedP1 mrcnnRgb mrcnnP1

A run is a folder with results.json (a name under --root, or a path). Per
block, the hits, misses and false detections are rebuilt from the block's
crowns, recall and precision, so the pooled F1 of any set of blocks can be
recomputed: the pooled difference is reference minus other, and its 95%
interval comes from resampling the blocks with replacement (--bootstrap
times): it says how much the difference depends on which blocks there are.
The per-block test is the paired Wilcoxon on the blocks' F1, with the number
of blocks the reference wins, and a Holm adjustment over the comparisons made.

Why both: a test over blocks weighs a block of 30 trees like one of 400; the
pooled difference weighs every tree the same. When they disagree, small
blocks are deciding the test.
"""

import argparse
import json
import os
import sys

import numpy as np

from .analysis.statistics import blockKey, compareFolds as compare, holm


def loadRun(root, name):
    path = name if os.path.isdir(name) else os.path.join(root or "", name)
    with open(os.path.join(path, "results.json")) as handle:
        result = json.load(handle)
    return result["pooled"], {blockKey(f["block"]): f for f in result["folds"]}


def label(name):
    """A run's name in the table: its folder (the path is in --output)."""
    return os.path.basename(os.path.normpath(name))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=None)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--others", nargs="+", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    pooled, reference = loadRun(args.root, args.reference)
    rows = []
    for name in args.others:
        _, other = loadRun(args.root, name)
        rows.append(dict(compare(reference, other, args.bootstrap, rng),
                         other=label(name), otherPath=name))
    for row, adjusted in zip(rows, holm(np.array([r["p"] for r in rows]))):
        row["pHolm"] = float(adjusted)
    print("%s against each other run; pooled F1 difference with a 95%% "
          "interval over blocks, and the per-block test" % label(args.reference))
    print("%-14s %7s %7s %8s %18s %6s %6s %7s %7s" % (
        "other", "ref", "other", "diff", "95% interval", "wins", "blocks",
        "p", "p Holm"))
    for r in rows:
        print("%-14s %7.3f %7.3f %+8.3f   [%+.3f, %+.3f] %6d %6d %7.3f %7.3f"
              % (r["other"], r["pooledReference"], r["pooledOther"],
                 r["pooledDifference"], r["interval95"][0], r["interval95"][1],
                 r["referenceWins"], r["blocks"], r["p"], r["pHolm"]))
    if args.output:
        with open(args.output, "w") as handle:
            json.dump({"reference": args.reference, "rows": rows}, handle,
                      indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
