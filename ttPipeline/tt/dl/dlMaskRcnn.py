#!/usr/bin/env python
"""
The `maskrcnn` command: a torchvision detector (tt.detectors.dl.
torchvisionDetector) under leave-one-block-out cross-validation
(tt.experiments.trained), or trained on another site with --transferFrom.

    python -m tt.dl maskrcnn --dataset datasets/lidarChm \\
                             --output runs/maskrcnn_lidar --epochs 30

`--modelType` selects from the zoo: maskrcnn, fasterrcnn, fasterrcnnmb,
retinanet, retinanetv1, fcos, ssd.
"""

import argparse
import os
import sys

from ..data.patches import PatchSet
from ..data.site import TDData
from ..detectors.base import NMS
from ..detectors.dl import models
from ..detectors.dl.torchvisionDetector import TorchvisionDetector
from ..experiments.trained import TrainedCrossValidation, TrainedTransfer
from . import dlCommon as dc


def detectorFrom(args):
    return TorchvisionDetector(
        modelType=args.modelType, epochs=args.epochs,
        batchSize=args.batchSize, learningRate=args.learningRate,
        optimiser=args.optimiser, workers=args.workers, device=args.device,
        pretrained=args.pretrained, imageSize=args.imageSize,
        initWeights=args.initWeights, inferenceBatch=args.inferenceBatch,
        threshold=args.scoreThreshold, postprocessors=[NMS(args.nmsIou)])


def experimentFrom(args):
    site = TDData.fromDataset(args.dataset)
    detector = detectorFrom(args)
    common = dict(bufferM=args.buffer, fixedThreshold=args.fixedThreshold,
                  saveModels=args.saveModels, output=args.output,
                  folds=args.folds.split(",") if args.folds else None)
    if args.transferFrom:
        return TrainedTransfer(site, detector,
                               TDData.fromDataset(args.transferFrom), **common)
    return TrainedCrossValidation(
        site, detector, trainFraction=args.trainFraction,
        subsetSeed=args.subsetSeed,
        drawFrom=PatchSet.load(args.drawFrom) if args.drawFrom else None,
        **common)


def crossValidate(args):
    os.makedirs(args.output, exist_ok=True)
    dc.startLogging(args.log or os.path.join(args.output, "run.log"))
    runSet = experimentFrom(args).run()
    runSet.settings = vars(args)
    path = runSet.save(args.output)
    pooled = runSet.pooled
    print("\n[torchvision] pooled over %d folds: R %.3f  P %.3f  F1 %.3f "
          "(per-fold F1 %.3f +- %.3f)"
          % (pooled["folds"], pooled["recall"], pooled["precision"],
             pooled["f1"], pooled["perFoldF1Mean"], pooled["perFoldF1Std"]))
    print("[torchvision] wrote %s and one predictions_<block>.json per fold"
          % path)
    return pooled


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Mask R-CNN with leave-one-block-out cross-validation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log", default=None,
                        help="Also write everything printed to this file. "
                             "Defaults to <output>/run.log")
    parser.add_argument("--modelType", default="maskrcnn",
                        choices=models.MODEL_TYPES,
                        help="Which torchvision detector (default: maskrcnn). "
                             "Only maskrcnn produces masks; the rest are "
                             "box-only, which this metric does not penalise")
    parser.add_argument("--optimiser", default="adamw",
                        choices=["adamw", "sgd"],
                        help="adamw with OneCycle (default), or sgd with the "
                             "StepLR(3, 0.1) schedule from your train.py, for "
                             "comparability with earlier runs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batchSize", type=int, default=2)
    parser.add_argument("--learningRate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default=None,
                        help="0, cuda:0, cpu, mps, or blank for automatic")
    parser.add_argument("--folds", default=None,
                        help="Comma-separated block names to run; default is "
                             "all of them. Useful for resuming a long run or "
                             "debugging a single fold")
    parser.add_argument("--buffer", type=float, default=10.0,
                        help="Metres of separation between training tiles and "
                             "the held-out block (default: 10). Must exceed a "
                             "crown diameter or the same tree appears on both "
                             "sides of the split")
    parser.add_argument("--scoreThreshold", type=float, default=0.5,
                        help="Fallback operating point. By default it is "
                             "replaced per fold by the threshold that "
                             "maximises F1 on the validation block")
    parser.add_argument("--fixedThreshold", action="store_true",
                        help="Use --scoreThreshold as given instead of "
                             "choosing one per fold")
    parser.add_argument("--nmsIou", type=float, default=0.4)
    parser.add_argument("--noPretrained", dest="pretrained",
                        action="store_false")
    parser.add_argument("--saveModels", action="store_true")
    parser.add_argument("--trainFraction", type=float, default=1.0,
                        help="learning curve: train (and choose the "
                             "operating point) on this fraction of each "
                             "training block's tiles, whole tiles drawn at "
                             "random (dlCommon.drawTiles); 1 = all")
    parser.add_argument("--drawFrom", default=None,
                        help="draw the --trainFraction subset on this dataset's "
                             "tiles (e.g. the site's RGB dataset when training on "
                             "its CHM) and keep those this dataset has")
    parser.add_argument("--subsetSeed", type=int, default=1,
                        help="which random draw of --trainFraction")
    parser.add_argument("--transferFrom", default=None,
                        help="learning curve, no annotation here: train one "
                             "network on every block of this other dataset "
                             "and apply it to every block of --dataset")
    parser.add_argument("--initWeights", default=None,
                        help="start from these weights (a one-class RetinaNet "
                             "state dict, e.g. DeepForest's tree model from "
                             "exportDeepForest.py) with --modelType retinanetv1")
    parser.add_argument("--imageSize", type=int, default=0,
                        help="Pixels the model runs at. Default 0 means the "
                             "tile size, i.e. native resolution. Pass 800 to "
                             "reproduce torchvision's default upscale, which "
                             "earlier runs used without saying so")
    parser.add_argument("--inferenceBatch", type=int, default=4,
                        help="Tiles per forward pass at prediction time "
                             "(default: 4)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(0 if crossValidate(parseArguments()) else 1)
