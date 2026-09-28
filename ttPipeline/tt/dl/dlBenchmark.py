#!/usr/bin/env python
"""
dlBenchmark.py

Run the whole comparison and print one table.

    python dlBenchmark.py --config benchmark.json

    # or, for the standard four experiments on one area:
    python dlBenchmark.py --auto \\
        --lidarChm chm_lidar_l2_2025_clipped_modified_last.tif \\
        --p1Chm    CHM_P1_2026_clipped_last.tif \\
        --ortho    ortho_p1_2026_clipped_last.tif \\
        --crowns   shp/annotation_new_0925.shp \\
        --boundary shp/area_p1.shp \\
        --output   benchmark

The matrix:

    lidarChm    1 channel   concomp, maskrcnn, yolo
    p1Chm       1 channel   concomp, maskrcnn, yolo
    rgb         3 channels  maskrcnn, yolo
    rgbLidar    4 channels  maskrcnn, yolo

Every cell is leave-one-block-out over the same eight spatial blocks, scored by
the same stitched crown-level rule, so the numbers belong in one table. The
connected-component rows are tuned per fold on the training blocks only, which
is what makes them comparable rather than flattering.

Read `perFoldF1Std` before believing any difference between two rows. On this
area a fold holds roughly 115 crowns and the per-fold spread is around 0.03 F1,
so two methods within about 0.03 of each other are indistinguishable here. That
is a statement about how much data there is, not about the methods.

--stages lets the slow parts be run separately, since a full matrix is several
hours on one GPU:

    --stages prepare              build the datasets only
    --stages concomp              the cheap baseline
    --stages maskrcnn,yolo        the learned models
    --stages report               re-print the table from existing results
"""

import argparse
import os
import subprocess
import sys

import numpy as np

from . import dlCommon as dc


TORCHVISION_MODELS = ["maskrcnn", "fasterrcnn", "fasterrcnnmb", "retinanet",
                      "fcos", "ssd"]

EXPERIMENTS = [
    # name,        source template,                          methods
    ("lidarChm", "chm:{lidarChm}", ["concomp", "maskrcnn", "yolo"]),
    ("p1Chm", "chm:{p1Chm}", ["concomp", "maskrcnn", "yolo"]),
    ("rgb", "rgb:{ortho}", ["maskrcnn", "yolo"]),
    ("rgbLidar", "rgb:{ortho}+chm:{lidarChm}", ["maskrcnn", "yolo"]),
]


def expandExperiments(extraModels):
    """Add the extra torchvision architectures to every raster source."""
    if not extraModels:
        return EXPERIMENTS
    expanded = []
    for name, source, methods in EXPERIMENTS:
        methods = list(methods)
        for model in extraModels:
            if model not in methods:
                methods.append(model)
        expanded.append((name, source, methods))
    return expanded


def run(command, description, logPath=None):
    """
    Run a stage, streaming its output to the terminal and to a log.

    A full matrix is hours of output and the interesting lines — the per-fold
    scores — are buried in thousands of progress lines. Everything is captured
    so a finished run can be read back, while still being watchable live.
    """
    print("\n>>> %s" % description)
    print("    %s" % " ".join(command))

    if logPath is None:
        return subprocess.run(command).returncode == 0

    directory = os.path.dirname(os.path.abspath(logPath))
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(logPath, "a", buffering=1) as handle:
        handle.write("\n%s\n=== %s : %s ===\n"
                     % ("=" * 70, dc._timestamp(), description))
        handle.write("%s\n" % " ".join(command))
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
        process.wait()

    if process.returncode != 0:
        print("    FAILED (exit %d) — see %s" % (process.returncode, logPath))
        return False
    return True


def stageCommand(stage, *arguments):
    """A stage of the benchmark, run through the package so paths never matter."""
    return [sys.executable, "-m", "tt.dl", stage] + [str(a) for a in arguments]


def prepare(args, name, sourceTemplate):
    source = sourceTemplate.format(lidarChm=args.lidarChm, p1Chm=args.p1Chm,
                                   ortho=args.ortho)
    datasetDir = os.path.join(args.output, "datasets", name)
    if os.path.exists(os.path.join(datasetDir, "dataset.json")) \
            and not args.rebuild:
        print("    dataset %s already built" % name)
        return datasetDir
    ok = run(stageCommand("prepare", "--source", source, "--crowns", args.crowns,
                          "--boundary", args.boundary, "--output", datasetDir,
                          "--name", name, "--resolution", args.resolution,
                          "--tileSize", args.tileSize,
                          "--blockCols", args.blockCols,
                          "--blockRows", args.blockRows),
             "prepare %s" % name,
             logPath=os.path.join(args.output, "logs", "prepare.log"))
    return datasetDir if ok else None


def methodCommand(args, method, dataset, outputDir):
    """The command line for one method on one dataset."""
    common = ["--dataset", dataset, "--output", outputDir]
    device = ["--device", args.device] if args.device else []
    if method == "concomp":
        return stageCommand("concomp", *common)
    if method in TORCHVISION_MODELS:
        return stageCommand("maskrcnn", *(common + [
            "--modelType", method, "--optimiser", args.optimiser,
            "--epochs", args.epochs, "--batchSize", args.batchSize] + device))
    return stageCommand("yolo", *(common + [
        "--epochs", args.yoloEpochs, "--batchSize", args.yoloBatch] + device))


def runStages(args, stages, experiments):
    datasets = {}
    for name, sourceTemplate, _ in experiments:
        datasets[name] = (prepare(args, name, sourceTemplate)
                          if "prepare" in stages else
                          os.path.join(args.output, "datasets", name))

    for name, _, methods in experiments:
        if not datasets.get(name):
            continue
        for method in methods:
            if method not in stages:
                continue
            outputDir = os.path.join(args.output, "runs",
                                     "%s_%s" % (method, name))
            run(methodCommand(args, method, datasets[name], outputDir),
                "%s on %s" % (method, name),
                logPath=os.path.join(args.output, "logs",
                                     "%s_%s.log" % (method, name)))


# ---------------------------------------------------------------------- #
# report
# ---------------------------------------------------------------------- #

def collectRows(args):
    rows = []
    for name, _, methods in expandExperiments(args.extraModels):
        for method in methods:
            path = os.path.join(args.output, "runs", "%s_%s" % (method, name),
                                "results.json")
            if os.path.exists(path):
                rows.append(summariseRun(name, method, dc.loadJson(path)))
    return rows


def summariseRun(source, method, data):
    pooled = data["pooled"]
    return {"source": source, "method": method,
            "channels": data.get("channelCount", 1),
            "recall": pooled["recall"], "precision": pooled["precision"],
            "f1": pooled["f1"], "std": pooled["perFoldF1Std"],
            "folds": pooled["folds"], "crowns": pooled["crowns"],
            "predictions": pooled["predictions"],
            "optimism": pooled.get("tuningOptimism"),
            "thresholds": sorted({f.get("threshold") for f in data["folds"]
                                  if f.get("threshold") is not None})}


def printTable(rows):
    print()
    print("=" * 84)
    print("Leave-one-block-out, stitched crown-level scoring")
    print("=" * 84)
    print("%-10s %-9s %3s %8s %8s %8s %8s %7s"
          % ("source", "method", "ch", "recall", "prec", "F1", "F1 sd",
             "preds"))
    print("-" * 84)
    for row in sorted(rows, key=lambda r: -r["f1"]):
        print("%-10s %-9s %3d %7.1f%% %7.1f%% %7.1f%% %7.3f %7d"
              % (row["source"], row["method"], row["channels"],
                 100 * row["recall"], 100 * row["precision"],
                 100 * row["f1"], row["std"], row["predictions"]))
    print("-" * 84)


def printNotes(rows):
    best = max(rows, key=lambda r: r["f1"])
    close = [r for r in rows
             if r is not best and abs(r["f1"] - best["f1"]) <= best["std"]]
    print("Best: %s / %s at F1 %.1f%%." % (best["source"], best["method"],
                                           100 * best["f1"]))
    if close:
        print("Within one per-fold standard deviation of it, so not separable "
              "on this much data: %s"
              % ", ".join("%s/%s" % (r["source"], r["method"]) for r in close))
    for row in rows:
        if len(row.get("thresholds") or []) > 1:
            print("%s/%s chose thresholds %s across folds, on its validation "
                  "block each time."
                  % (row["source"], row["method"],
                     ", ".join("%.2f" % t for t in row["thresholds"])))
    for row in rows:
        if row["optimism"] is not None:
            print("Connected components on %s would have read %+.3f F1 higher "
                  "if scored on its own tuning blocks."
                  % (row["source"], row["optimism"]))


def report(args):
    rows = collectRows(args)
    if not rows:
        print("\nNo results found under %s/runs." % args.output)
        return
    printTable(rows)
    printNotes(rows)
    pairedComparison(args, rows)
    path = os.path.join(args.output, "summary.json")
    dc.saveJson(rows, path)
    print("\nwrote %s" % path)


def pairedComparison(args, rows, baseline=("lidarChm", "concomp")):
    """
    Compare every method against a baseline block by block.

    Pooled F1 against a per-fold spread is a weak test, because blocks differ
    in difficulty far more than methods differ from each other — the
    connected-component detector scored 0.787 on one block here and 0.860 on
    another. Pairing by block cancels that: what matters is whether a method
    beats the baseline on the *same* trees, fold after fold. Eight blocks is
    few, so the count of wins is reported alongside the mean difference; a
    method that wins on 7 or 8 of 8 is saying something a mean difference of
    the same size might not.
    """

    def foldMap(source, method):
        path = os.path.join(args.output, "runs", "%s_%s" % (method, source),
                            "results.json")
        if not os.path.exists(path):
            return None
        data = dc.loadJson(path)
        return {f["block"].split("_b")[-1]: f["f1"] for f in data["folds"]}

    reference = foldMap(*baseline)
    if reference is None:
        return

    print()
    print("Paired against %s/%s, block by block" % baseline)
    print("%-10s %-9s %10s %8s %9s" % ("source", "method", "mean diff",
                                       "wins", "worst block"))
    print("-" * 52)
    for row in sorted(rows, key=lambda r: -r["f1"]):
        if (row["source"], row["method"]) == baseline:
            continue
        other = foldMap(row["source"], row["method"])
        if not other:
            continue
        shared = sorted(set(reference) & set(other))
        if len(shared) < 3:
            continue
        differences = np.array([other[b] - reference[b] for b in shared])
        wins = int((differences > 0).sum())
        worst = shared[int(np.argmin(differences))]
        print("%-10s %-9s %+10.3f %5d/%-2d %9s"
              % (row["source"], row["method"], differences.mean(), wins,
                 len(shared), "b" + worst))


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Run and report the full detector comparison.")
    parser.add_argument("--auto", action="store_true",
                        help="Use the standard four experiments")
    for name in ("lidarChm", "p1Chm", "ortho", "crowns", "boundary"):
        parser.add_argument("--" + name)
    parser.add_argument("--output", default="benchmark")
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--tileSize", type=int, default=512)
    parser.add_argument("--blockCols", type=int, default=4)
    parser.add_argument("--blockRows", type=int, default=2)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--yoloEpochs", type=int, default=150)
    parser.add_argument("--batchSize", type=int, default=2)
    parser.add_argument("--yoloBatch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--models", dest="extraModels", default="",
                        type=lambda t: [m for m in t.split(",") if m],
                        help="Extra torchvision architectures to add to every "
                             "source, comma separated, from: %s. maskrcnn is "
                             "always included" % ", ".join(TORCHVISION_MODELS))
    parser.add_argument("--optimiser", default="adamw",
                        choices=["adamw", "sgd"])
    parser.add_argument("--stages",
                        default="prepare,concomp,maskrcnn,yolo,report")
    return parser.parse_args(argv)


def main(argv=None):
    args = parseArguments(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if stages == ["report"]:
        report(args)
        return 0

    missing = [name for name in ("lidarChm", "p1Chm", "ortho", "crowns",
                                 "boundary") if not getattr(args, name)]
    if missing:
        print("--%s is required unless only reporting" % missing[0])
        return 1

    os.makedirs(args.output, exist_ok=True)
    runStages(args, stages, expandExperiments(args.extraModels))
    if "report" in stages:
        report(args)
    print("\nLogs under %s/logs/, one per stage, plus a run.log inside each "
          "run directory." % args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
