#!/usr/bin/env python
"""
dlYolo.py

YOLO (ultralytics) on a dataset built by dlPrepare.py, with leave-one-block-out
cross-validation and the same stitched, crown-level scoring as everything else.

    python dlYolo.py --dataset datasets/lidarChm \\
                     --output runs/yolo_lidar --epochs 100

Segmentation weights (`yolo11s-seg.pt`) are the default so the model is doing
the same job as Mask R-CNN. `--task detect` switches to boxes only, which
trains faster and, on this metric, loses nothing: crowns are scored by whether a
prediction's centre lands inside them, and a box centre is as good as a mask
centroid for that.

Channels
--------
    1   the height model is written as a grey PNG with the band repeated three
        times. That keeps the pretrained stem intact, which matters far more
        on a few hundred tiles than the wasted duplication does
    3   written straight out as a colour PNG
    4   written as a four-channel PNG, with the first convolution rebuilt for
        four inputs and ultralytics' image reader patched to stop discarding
        the fourth band

The four-channel path reaches into ultralytics internals and so is the part
most likely to break on a version bump. It checks itself at import and reports
clearly rather than silently training on three channels, which is the failure
that would otherwise go unnoticed and quietly invalidate the comparison.
"""

import argparse
import os
import shutil
import sys
import sys as _sys

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset
import ultralytics.utils.patches as patches

from ..scene import readCrowns
from . import dlCommon as dc
from . import dlPrepare as dp


# ---------------------------------------------------------------------- #
# four-channel support
# ---------------------------------------------------------------------- #

def enableFourChannelReading():
    """
    Make ultralytics read the fourth band instead of dropping it.

    Its loader calls cv2.imdecode with IMREAD_COLOR, which silently returns
    three channels for an RGBA file. Patched in every module that imported the
    function, since a plain attribute swap on the source module does not reach
    names already bound elsewhere.
    """

    original = patches.imread

    def imreadUnchanged(filename, flags=cv2.IMREAD_UNCHANGED):
        image = original(filename, flags=cv2.IMREAD_UNCHANGED)
        if image is not None and image.ndim == 2:
            image = image[:, :, None]
        return image

    patched = ["ultralytics.utils.patches"]
    patches.imread = imreadUnchanged
    for moduleName, module in list(_sys.modules.items()):
        if not moduleName.startswith("ultralytics"):
            continue
        if getattr(module, "imread", None) is original:
            module.imread = imreadUnchanged
            patched.append(moduleName)
    return patched


def patchFirstConvolution(model, channelCount):
    """Rebuild YOLO's stem for `channelCount` inputs, carrying weights over."""

    core = model.model
    first = core.model[0]
    old = first.conv
    if old.in_channels == channelCount:
        return model

    new = torch.nn.Conv2d(channelCount, old.out_channels,
                          kernel_size=old.kernel_size, stride=old.stride,
                          padding=old.padding, bias=old.bias is not None)
    with torch.no_grad():
        weight = old.weight.detach()
        if channelCount == 1:
            new.weight.copy_(weight.sum(dim=1, keepdim=True))
        elif channelCount > 3:
            new.weight[:, :3].copy_(weight)
            mean = weight.mean(dim=1, keepdim=True)
            for extra in range(3, channelCount):
                new.weight[:, extra:extra + 1].copy_(mean)
        else:
            new.weight.copy_(weight[:, :channelCount])
    first.conv = new
    core.yaml["ch"] = channelCount
    if hasattr(core, "ch"):
        core.ch = channelCount
    return model


# ---------------------------------------------------------------------- #
# export
# ---------------------------------------------------------------------- #

def writeTileImage(array, path):
    """Write a tile as a PNG ultralytics can read, preserving every channel."""
    channels = array.shape[0]
    if channels == 1:
        image = cv2.cvtColor(array[0], cv2.COLOR_GRAY2BGR)
    elif channels == 3:
        image = cv2.merge([array[2], array[1], array[0]])
    elif channels == 4:
        image = cv2.merge([array[2], array[1], array[0], array[3]])
    else:
        raise ValueError("unsupported channel count %d" % channels)
    cv2.imwrite(path, image)


def writeLabels(labels, tileSize, path, task):
    """YOLO labels: one line per crown, normalised, class 0 (tree)."""
    lines = []
    for crown in labels:
        if task == "segment":
            polygon = np.asarray(crown["polygon"], np.float64) / tileSize
            polygon = np.clip(polygon, 0.0, 1.0)
            if len(polygon) < 3:
                continue
            flat = " ".join("%.6f %.6f" % (x, y) for x, y in polygon)
            lines.append("0 " + flat)
        else:
            x0, y0, x1, y1 = crown["box"]
            cx = 0.5 * (x0 + x1) / tileSize
            cy = 0.5 * (y0 + y1) / tileSize
            w = (x1 - x0) / tileSize
            h = (y1 - y0) / tileSize
            if w <= 0 or h <= 0:
                continue
            lines.append("0 %.6f %.6f %.6f %.6f"
                         % (np.clip(cx, 0, 1), np.clip(cy, 0, 1),
                            np.clip(w, 0, 1), np.clip(h, 0, 1)))
    with open(path, "w") as f:
        f.write("\n".join(lines))


def exportFold(meta, trainTiles, testTiles, foldDir, task, heldOut=None):
    """
    Build the directory layout ultralytics expects, plus a separate test split.

    train/  the training blocks, minus one kept for validation
    val/    that one block — what ultralytics early-stops and selects on
    test/   the held-out block, never seen by training or selection, and not
            referenced from data.yaml
    """
    root = meta["root"]
    tileSize = meta["tileSize"]

    # Ultralytics validates every epoch, early-stops and saves best.pt by the
    # val split. Pointing that at the held-out block hands YOLO checkpoint
    # selection on the very trees it is then scored on, so the split has to
    # come out of the training blocks. The same block then chooses the score
    # threshold, exactly as it does for the torchvision models.
    validationBlock = dc.chooseValidationBlock(trainTiles, heldOut or "")
    valTiles = [t for t in trainTiles if t["block"] == validationBlock]
    fitTiles = [t for t in trainTiles if t["block"] != validationBlock]
    if not valTiles or not fitTiles:
        fitTiles, valTiles = trainTiles, trainTiles[:max(1, len(trainTiles) // 8)]

    for split, tiles in (("train", fitTiles), ("val", valTiles),
                         ("test", testTiles)):
        imageDir = os.path.join(foldDir, "images", split)
        labelDir = os.path.join(foldDir, "labels", split)
        os.makedirs(imageDir, exist_ok=True)
        os.makedirs(labelDir, exist_ok=True)
        for entry in tiles:
            array = np.load(os.path.join(root, "tiles",
                                         entry["stem"] + ".npy"))
            labels = dc.loadJson(os.path.join(root, "tiles",
                                              entry["stem"] + ".json"))["crowns"]
            writeTileImage(array, os.path.join(imageDir,
                                               entry["stem"] + ".png"))
            writeLabels(labels, tileSize,
                        os.path.join(labelDir, entry["stem"] + ".txt"), task)

    yamlPath = os.path.join(foldDir, "data.yaml")
    with open(yamlPath, "w") as f:
        f.write("path: %s\n" % os.path.abspath(foldDir))
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n  0: tree\n")
    return yamlPath, validationBlock, len(fitTiles), len(valTiles)


# ---------------------------------------------------------------------- #

# predictions are kept down to this score and the operating point chosen
# afterwards, so the saved files support any threshold
SCORE_FLOOR = 0.05

FALLBACK_WEIGHTS = ["yolo11s-seg.pt", "yolo11n-seg.pt",
                    "yolov8s-seg.pt", "yolov8n-seg.pt"]
FALLBACK_DETECT = ["yolo11s.pt", "yolo11n.pt", "yolov8s.pt", "yolov8n.pt"]


def resolveWeights(requested, task):
    """
    Get a usable checkpoint, or explain precisely why not.

    Older ultralytics releases do not list the yolo11 assets, so asking for
    `yolo11s-seg.pt` there does not trigger a download and fails with a bare
    FileNotFoundError from torch.load. This tries the request, then falls back
    through earlier generations, and if all of them fail says what to do rather
    than leaving the traceback to be decoded.
    """

    candidates = [requested]
    candidates += [w for w in (FALLBACK_WEIGHTS if task == "segment"
                               else FALLBACK_DETECT) if w != requested]

    tried = []
    for candidate in candidates:
        if os.path.exists(candidate):
            if candidate != requested:
                print("[yolo] '%s' unavailable; using local '%s'"
                      % (requested, candidate))
            return candidate
        try:
            path = attempt_download_asset(candidate)
            if path and os.path.exists(path):
                if candidate != requested:
                    print("[yolo] '%s' is not an asset in ultralytics %s; "
                          "downloaded '%s' instead"
                          % (requested, ultralytics.__version__, candidate))
                return path
        except Exception as error:
            tried.append("%s (%s)" % (candidate, type(error).__name__))
            continue
        tried.append("%s (not found)" % candidate)

    raise RuntimeError(
        "No usable YOLO weights. ultralytics %s tried: %s.\n"
        "Either upgrade ultralytics (pip install -U ultralytics), or download "
        "a checkpoint by hand into this directory and pass it with "
        "--weights, for example:\n"
        "  wget https://github.com/ultralytics/assets/releases/download/"
        "v8.3.0/yolo11s-seg.pt"
        % (ultralytics.__version__, ", ".join(tried)))


class YoloCrossValidation(object):

    def __init__(self, args):
        self.args = args
        self.meta = dp.loadDataset(args.dataset)
        self.blocks = self.meta["blockGeometries"]
        self.crowns, _ = readCrowns(self.meta["crownsPath"], self.meta["crs"])
        self.channelCount = self.meta["channelCount"]
        self.tileSize = self.meta["tileSize"]
        self.device = args.device if args.device else \
            (0 if torch.cuda.is_available() else "cpu")
        self.weights = resolveWeights(args.weights, args.task)
        if self.channelCount == 4:
            patched = enableFourChannelReading()
            print("[yolo] four-channel reading patched into %d module(s)"
                  % len(patched))

    def buildModel(self):
        model = YOLO(self.weights)
        if self.channelCount == 4:
            model = patchFirstConvolution(model, 4)
            actual = model.model.model[0].conv.in_channels
            if actual != 4:
                raise RuntimeError("the first convolution is still %d-channel; "
                                   "the four-channel patch did not take"
                                   % actual)
        return model

    def train(self, model, dataYaml, foldDir):
        args = self.args
        model.train(data=dataYaml, epochs=args.epochs, imgsz=self.tileSize,
                    batch=args.batchSize, device=self.device,
                    workers=args.workers, project=foldDir, name="train",
                    exist_ok=True, pretrained=args.pretrained,
                    verbose=args.verboseTraining, seed=0, deterministic=False,
                    patience=args.patience, degrees=180.0, fliplr=0.5,
                    flipud=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                    mosaic=args.mosaic, scale=args.scale)

    def predict(self, model, foldDir, tiles, split):
        """
        Stitched, NMS-merged world-space predictions for one split.

        The whole split goes to ultralytics as one list, which it batches
        itself; predicting file by file left the GPU idle between calls.
        Segmentation results carry their outlines, in tile pixels, through.
        """
        entries = [e for e in tiles if os.path.exists(
            os.path.join(foldDir, "images", split, e["stem"] + ".png"))]
        if not entries:
            return []
        paths = [os.path.join(foldDir, "images", split, e["stem"] + ".png")
                 for e in entries]
        results = model.predict(paths, conf=SCORE_FLOOR, imgsz=self.tileSize,
                                device=self.device, batch=self.args.batchSize,
                                verbose=False)
        gathered = []
        for entry, output in zip(entries, results):
            gathered.extend(dc.tileToWorld(yoloPredictions(output),
                                           self.meta["transformObject"],
                                           entry["c0"], entry["r0"]))
        return dc.nonMaximumSuppression(gathered, self.args.nmsIou)

    def chooseThreshold(self, model, foldDir, trainTiles, validationBlock):
        """The operating point, and the validation predictions it came from."""
        if not validationBlock:
            return self.args.scoreThreshold, []
        valTiles = [t for t in trainTiles if t["block"] == validationBlock]
        merged = self.predict(model, foldDir, valTiles, "val")
        if self.args.fixedThreshold:
            return self.args.scoreThreshold, merged
        threshold, _ = dc.selectThreshold(merged, self.crowns,
                                          dict(self.blocks)[validationBlock])
        return threshold, merged

    def runFold(self, name, geometry):
        trainTiles, testTiles = dc.foldSplit(self.meta["tiles"], self.blocks,
                                             name, self.args.buffer)
        if not trainTiles or not testTiles:
            print("  fold %s: skipped, empty split" % name)
            return None

        foldDir = os.path.join(self.args.output, "folds", name)
        if os.path.exists(foldDir):
            shutil.rmtree(foldDir)
        dataYaml, validationBlock, fitCount, valCount = exportFold(
            self.meta, trainTiles, testTiles, foldDir, self.args.task,
            heldOut=name)
        print("  fold %s: fit %d tiles, validate on %s (%d), test %d"
              % (name, fitCount, validationBlock, valCount, len(testTiles)))

        model = self.buildModel()
        self.train(model, dataYaml, foldDir)
        threshold, validation = self.chooseThreshold(model, foldDir,
                                                     trainTiles,
                                                     validationBlock)
        merged = self.predict(model, foldDir, testTiles, "test")
        result = dc.evaluateDetections(
            [p for p in merged if p["score"] >= threshold], self.crowns,
            geometry, verbose=True)
        result.update(block=name, validationBlock=validationBlock,
                      threshold=threshold, mergedPredictions=len(merged))

        dc.saveJson({"block": name, "threshold": threshold,
                     "scoreFloor": SCORE_FLOOR, "predictions": merged,
                     "validationBlock": validationBlock,
                     "validationPredictions": validation},
                    os.path.join(self.args.output,
                                 "predictions_%s.json" % name))
        if not self.args.keepFolds:
            for split in ("train", "val", "test"):
                shutil.rmtree(os.path.join(foldDir, "images", split),
                              ignore_errors=True)
        return result

    def run(self):
        print("[yolo] ultralytics %s, weights %s"
              % (ultralytics.__version__, self.weights))
        print("[yolo] dataset %s, %d channels, %d tiles, %d blocks, device %s"
              % (self.meta["name"], self.channelCount,
                 len(self.meta["tiles"]), len(self.blocks), self.device))
        wanted = set(self.args.folds.split(",")) if self.args.folds else None
        folds = []
        for name, geometry in self.blocks:
            if wanted and name not in wanted:
                continue
            result = self.runFold(name, geometry)
            if result is not None:
                folds.append(result)
        return folds, dc.averageFolds(folds)


def yoloPredictions(output):
    """One ultralytics result as tile-pixel predictions, outlines included."""
    if output.boxes is None or len(output.boxes) == 0:
        return []
    boxes = output.boxes.xyxy.cpu().numpy()
    scores = output.boxes.conf.cpu().numpy()
    outlines = output.masks.xy if output.masks is not None else None
    predictions = []
    for index, (box, score) in enumerate(zip(boxes, scores)):
        prediction = {"box": box.tolist(), "score": float(score)}
        if outlines is not None and index < len(outlines) and \
                len(outlines[index]) >= 3:
            prediction["polygon"] = np.asarray(outlines[index],
                                               float).tolist()
        predictions.append(prediction)
    return predictions


def crossValidate(args):
    os.makedirs(args.output, exist_ok=True)
    dc.startLogging(args.log or os.path.join(args.output, "run.log"))

    validation = YoloCrossValidation(args)
    folds, pooled = validation.run()

    print("\n[yolo] pooled over %d folds: R %.3f  P %.3f  F1 %.3f "
          "(per-fold F1 %.3f +- %.3f)"
          % (pooled["folds"], pooled["recall"], pooled["precision"],
             pooled["f1"], pooled["perFoldF1Mean"], pooled["perFoldF1Std"]))
    path = os.path.join(args.output, "results.json")
    dc.saveJson({"method": "yolo", "dataset": validation.meta["name"],
                 "source": validation.meta["source"],
                 "channelCount": validation.channelCount,
                 "settings": vars(args), "folds": folds, "pooled": pooled},
                path)
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
