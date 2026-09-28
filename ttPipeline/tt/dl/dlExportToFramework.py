#!/usr/bin/env python
"""
dlExportToFramework.py

Write the folds of a dlPrepare dataset in the folder format your existing
experiment framework (datasets.TDDataset / train.py / predict.py) already
reads, so its whole model zoo — maskrcnn, fasterrcnn, convnextmaskrcnn,
retinanet, fcos, ssd, DETR, Deformable DETR, YOLO — can be run on spatially
blocked splits instead of random ones.

    python dlExportToFramework.py --dataset datasets/lidarChm \\
                                  --output frameworkFolds

    frameworkFolds/lidar_b00/train/Tile0000.png
                                  /Tile0000Labels.tif
                                  /Tile0000Boxes.txt
                             /test/...
    frameworkFolds/lidar_b01/...

One folder pair per fold, matching TDDataset's convention: a PNG, a Labels.tif,
and a Boxes.txt holding `class px py w h` per line — the order
imageUtils.boxCoordsToFile writes and datasets.TDDataset.__getitem__ reads.

Why this exists
---------------
dataHandling.prepareDataKoi assigns each tile to train or test with

    outFolder = ".../train" if randint(1,100) < trainPerc else ".../test"

while sliding_window steps by `int(slice*0.8)`, a 20% overlap on every side.
So 64% of each tile's area is also inside a neighbouring tile, and with an
80-90% training draw every test tile has essentially all eight of its
neighbours in the training set. Most trees scored as held-out were seen during
training, in a neighbouring tile, at a slightly different offset. That inflates
every model equally and cannot be corrected after the fact.

This exporter splits by spatial block with a buffer instead, so no tree appears
on both sides. Nothing else about your pipeline needs to change.

Two things to know before running Mask R-CNN on the output
----------------------------------------------------------
1. `datasets.extractMask` builds an instance mask as
   `labelIm[box] == cat`. With every tree in one class that returns *every*
   tree pixel inside the box, not the one instance — and these crowns overlap
   heavily, so the masks would be badly contaminated. Boxes are unaffected, so
   fasterrcnn, retinanet, fcos, ssd, DETR and YOLO are all fine as they stand.
   For correct masks, `--instanceLabels` writes Labels.tif with a unique id per
   crown and Boxes.txt carrying that id in the class column, and you change one
   line so the label tensor stays at 1:

       labels.append(1)              # instead of self.classDict[cat] / cat

2. `datasets.TDDataset.findMaxClass` unpacks `px,py,w,h,cat` while the file is
   written and read everywhere else as `cat,px,py,w,h`. It therefore returns
   the largest box *height* as the class count, and line 82 uses it
   unconditionally, overriding the classDict branch above it. It does not bite
   here because PyTorchModelExperiment takes num_classes from the config, but
   `getNumClasses()` is wrong for any caller that trusts it.
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np

from . import dlCommon as dc
from . import dlPrepare as dp


def writeTile(array, labels, outDir, stem, instanceLabels):
    """One tile in TDDataset's three-file convention."""
    channels = array.shape[0]
    if channels == 1:
        image = cv2.cvtColor(array[0], cv2.COLOR_GRAY2BGR)
    elif channels == 3:
        image = cv2.merge([array[2], array[1], array[0]])
    elif channels == 4:
        # the framework reads PNGs with PIL, which would drop a fourth band;
        # the height model is written beside the photo instead
        image = cv2.merge([array[2], array[1], array[0]])
        cv2.imwrite(os.path.join(outDir, stem + "Height.png"), array[3])
    else:
        raise ValueError("unsupported channel count %d" % channels)

    cv2.imwrite(os.path.join(outDir, stem + ".png"), image)

    height, width = array.shape[1], array.shape[2]
    labelImage = np.zeros((height, width), np.uint16)
    lines = []
    for position, crown in enumerate(labels):
        polygon = np.round(np.array(crown["polygon"])).astype(np.int32)
        value = (position + 1) if instanceLabels else 1
        cv2.fillPoly(labelImage, [polygon], int(value))
        x0, y0, x1, y1 = crown["box"]
        px, py = int(round(x0)), int(round(y0))
        w, h = int(round(x1 - x0)), int(round(y1 - y0))
        if w <= 0 or h <= 0:
            continue
        lines.append("%d %d %d %d %d" % (value, px, py, w, h))

    cv2.imwrite(os.path.join(outDir, stem + "Labels.tif"), labelImage)
    with open(os.path.join(outDir, stem + "Boxes.txt"), "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def writeSplit(root, tiles, splitDir, instanceLabels):
    """Write one split's tiles; returns how many crowns they carried."""
    os.makedirs(splitDir, exist_ok=True)
    crowns = 0
    for position, entry in enumerate(tiles):
        array = np.load(os.path.join(root, "tiles", entry["stem"] + ".npy"))
        labels = dc.loadJson(os.path.join(root, "tiles",
                                          entry["stem"] + ".json"))["crowns"]
        writeTile(array, labels, splitDir, "Tile%04d" % position,
                  instanceLabels)
        crowns += len(labels)
    return crowns


def exportFold(meta, name, output, bufferM, instanceLabels):
    """One fold in the framework's train/test layout, or None if empty."""
    trainTiles, testTiles = dc.foldSplit(meta["tiles"],
                                         meta["blockGeometries"], name,
                                         bufferM)
    if not trainTiles or not testTiles:
        print("  %s: skipped (train %d, test %d)"
              % (name, len(trainTiles), len(testTiles)))
        return None

    foldDir = os.path.join(output, name)
    if os.path.exists(foldDir):
        shutil.rmtree(foldDir)
    trainCrowns = writeSplit(meta["root"], trainTiles,
                             os.path.join(foldDir, "train"), instanceLabels)
    testCrowns = writeSplit(meta["root"], testTiles,
                            os.path.join(foldDir, "test"), instanceLabels)
    buffered = len(meta["tiles"]) - len(trainTiles) - len(testTiles)
    print("  %s: train %d tiles / %d crowns, test %d tiles / %d crowns, "
          "%d tiles held in the buffer"
          % (name, len(trainTiles), trainCrowns, len(testTiles), testCrowns,
             buffered))
    return {"fold": name, "path": os.path.abspath(foldDir),
            "trainTiles": len(trainTiles), "trainCrowns": trainCrowns,
            "testTiles": len(testTiles), "testCrowns": testCrowns,
            "bufferedTiles": buffered}


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Export dlPrepare folds into the TDDataset folder format.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--buffer", type=float, default=10.0,
                        help="Metres between training tiles and the held-out "
                             "block (default: 10)")
    parser.add_argument("--instanceLabels", action="store_true",
                        help="Write a unique id per crown into Labels.tif and "
                             "into the class column of Boxes.txt. Needed for "
                             "correct Mask R-CNN masks; see the note at the "
                             "top of this file about the one-line change that "
                             "goes with it")
    parser.add_argument("--folds", default=None,
                        help="Comma-separated block names; default is all")
    return parser.parse_args(argv)


def main(argv=None):
    args = parseArguments(argv)
    meta = dp.loadDataset(args.dataset)
    wanted = set(args.folds.split(",")) if args.folds else None
    os.makedirs(args.output, exist_ok=True)

    summary = [fold for fold in
               (exportFold(meta, name, args.output, args.buffer,
                           args.instanceLabels)
                for name, _ in meta["blockGeometries"]
                if not wanted or name in wanted)
               if fold is not None]

    dc.saveJson({"dataset": meta["name"], "source": meta["source"],
                 "channelCount": meta["channelCount"],
                 "tileSize": meta["tileSize"],
                 "resolution": meta["resolution"],
                 "instanceLabels": args.instanceLabels, "folds": summary},
                os.path.join(args.output, "folds.json"))
    print("\nwrote %d folds to %s" % (len(summary), args.output))
    print("Point the framework's config at one fold at a time: "
          "TV_dir = %s/<fold>, Test_dir = test, slice = %d"
          % (os.path.abspath(args.output), meta["tileSize"]))
    print("Its own scores are per tile and count overlap trees more than "
          "once; dlCommon.evaluateDetections scores each tree once.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
