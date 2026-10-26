#!/usr/bin/env python
"""
The `yolo` command: ultralytics YOLO (tt.detectors.dl.yolo) under
leave-one-block-out cross-validation (tt.experiments.trained).

    python -m tt.dl yolo --dataset datasets/lidarChm \\
                         --output runs/yolo_lidar --epochs 100
"""

import argparse
import os
import sys

from ..data.site import TDData
from ..detectors.base import NMS
from ..detectors.dl.yolo import YOLODetector
from ..experiments.trained import TrainedCrossValidation
from . import dlCommon as dc


class FoldDirectories(TrainedCrossValidation):
    """Each fold trains in its own folder under <output>/folds."""

    def runFold(self, block):
        self.detector = self.detector.withParams(
            workDir=os.path.join(self.output, "folds", block.name))
        return super().runFold(block)


def crossValidate(args):
    os.makedirs(args.output, exist_ok=True)
    dc.startLogging(args.log or os.path.join(args.output, "run.log"))
    detector = YOLODetector(
        weights=args.weights, task=args.task, epochs=args.epochs,
        patience=args.patience, batchSize=args.batchSize,
        workers=args.workers, device=args.device, mosaic=args.mosaic,
        scale=args.scale, pretrained=args.pretrained,
        verboseTraining=args.verboseTraining, keepImages=args.keepFolds,
        threshold=args.scoreThreshold, postprocessors=[NMS(args.nmsIou)])
    experiment = FoldDirectories(
        TDData.fromDataset(args.dataset), detector, bufferM=args.buffer,
        fixedThreshold=args.fixedThreshold, output=args.output,
        folds=args.folds.split(",") if args.folds else None)
    runSet = experiment.run()
    runSet.settings = vars(args)
    path = runSet.save(args.output)
    pooled = runSet.pooled
    print("\n[yolo] pooled over %d folds: R %.3f  P %.3f  F1 %.3f "
          "(per-fold F1 %.3f +- %.3f)"
          % (pooled["folds"], pooled["recall"], pooled["precision"],
             pooled["f1"], pooled["perFoldF1Mean"], pooled["perFoldF1Std"]))
    print("[yolo] wrote %s and one predictions_<block>.json per fold" % path)
    return pooled


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="YOLO with leave-one-block-out cross-validation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", default=None,
                        help="Also write everything printed to this file. "
                             "Defaults to <output>/run.log")
    parser.add_argument("--weights", default="yolo11s-seg.pt")
    parser.add_argument("--task", default="segment",
                        choices=["segment", "detect"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30,
                        help="Early-stopping patience (default: 30). The "
                             "ultralytics default of 100 means a run that "
                             "peaks at epoch 19 still grinds through 119")
    parser.add_argument("--batchSize", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--folds", default=None,
                        help="Comma-separated block names to run; default is "
                             "all of them. Useful for resuming a long run or "
                             "debugging a single fold")
    parser.add_argument("--buffer", type=float, default=10.0)
    parser.add_argument("--scoreThreshold", type=float, default=0.25,
                        help="Fallback operating point. By default it is "
                             "replaced per fold by the threshold that "
                             "maximises F1 on the validation block")
    parser.add_argument("--fixedThreshold", action="store_true")
    parser.add_argument("--nmsIou", type=float, default=0.4)
    parser.add_argument("--mosaic", type=float, default=0.0,
                        help="Mosaic augmentation (default: 0). It stitches "
                             "four tiles together, which invents crown "
                             "adjacencies that cannot occur and breaks the "
                             "constant-scale assumption this task relies on")
    parser.add_argument("--scale", type=float, default=0.1,
                        help="Scale jitter (default: 0.1, nearly off). Crown "
                             "size in pixels is fixed by the ground "
                             "resolution and is information, not nuisance")
    parser.add_argument("--noPretrained", dest="pretrained",
                        action="store_false")
    parser.add_argument("--keepFolds", action="store_true")
    parser.add_argument("--verboseTraining", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(0 if crossValidate(parseArguments()) else 1)
