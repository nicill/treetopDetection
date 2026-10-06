#!/usr/bin/env python
"""
dlMaskRcnn.py

A torchvision detector on a dataset built by dlPrepare.py, with
leave-one-block-out cross-validation and stitched, crown-level scoring.

    python dlMaskRcnn.py --dataset datasets/lidarChm \\
                         --output runs/maskrcnn_lidar --epochs 30

    python dlMaskRcnn.py --dataset datasets/lidarChm --modelType retinanet \\
                         --output runs/retinanet_lidar

`--modelType` selects from dlModels.MODEL_TYPES: maskrcnn, fasterrcnn,
fasterrcnnmb, retinanet, fcos, ssd. The file name is now a slight misnomer,
kept so existing commands still work.

One class: tree. The health classes are ignored deliberately — with 828 healthy,
90 mild and 2 severe in this area there is no prospect of learning three tiers,
and mixing them in would only depress the detection numbers being compared.

Channel handling
----------------
torchvision ships a three-channel model. For any other channel count the first
convolution is rebuilt and the pretrained weights are carried over rather than
discarded:

    1 channel  the three RGB filters are summed, the standard way to move a
               pretrained stem to grayscale; it preserves the filter's response
               to edges instead of starting from noise
    4 channels the first three keep their weights and the fourth starts as the
               mean of them, so the height band begins as a neutral copy of a
               colour band and is free to specialise

`transform.image_mean` and `image_std` are resized to match, otherwise
torchvision normalises with the wrong number of statistics and fails silently
on the extra band.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..scene import readCrowns
from . import dlCommon as dc
from . import dlModels as dm
from . import dlPrepare as dp

# predictions are kept down to this score and the operating point is chosen
# afterwards, so the saved files support any threshold, not just the chosen one
SCORE_FLOOR = 0.05


# ---------------------------------------------------------------------- #
# dataset
# ---------------------------------------------------------------------- #

class TileDataset(Dataset):
    """
    Tiles as (C, H, W) float tensors in 0..1, with per-instance masks.

    Augmentation is flips and 90-degree rotations only. Those are the
    transformations a nadir canopy raster is genuinely invariant to: a forest
    seen from above has no up. Colour jitter would break the CHM channel, whose
    values mean metres, and scaling would break the one strong prior the model
    has, that crowns are a known size in pixels.
    """

    def __init__(self, root, tiles, augment=False, withMasks=True):
        self.root = root
        self.tiles = tiles
        self.augment = augment
        self.withMasks = withMasks

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index):
        image, masks, boxes = self._load(self.tiles[index]["stem"])
        if self.augment:
            image, masks, boxes = self._augment(image, masks, boxes)
        keep = ((boxes[:, 2] - boxes[:, 0]) > 1) & \
            ((boxes[:, 3] - boxes[:, 1]) > 1)
        return torch.as_tensor(image), self._target(boxes[keep], masks[keep],
                                                    index)

    def _load(self, stem):
        array = np.load(os.path.join(self.root, "tiles", stem + ".npy"))
        crowns = dc.loadJson(os.path.join(self.root, "tiles",
                                          stem + ".json"))["crowns"]
        _, height, width = array.shape
        masks = np.zeros((len(crowns), height, width), np.uint8)
        boxes = np.zeros((len(crowns), 4), np.float32)
        for position, crown in enumerate(crowns):
            polygon = np.round(np.array(crown["polygon"])).astype(np.int32)
            cv2.fillPoly(masks[position], [polygon], 1)
            boxes[position] = crown["box"]
        return array.astype(np.float32) / 255.0, masks, boxes

    @staticmethod
    def _augment(image, masks, boxes):
        _, height, width = image.shape
        if np.random.rand() < 0.5:
            image = image[:, :, ::-1].copy()
            masks = masks[:, :, ::-1].copy()
            boxes = boxes[:, [2, 1, 0, 3]].copy()
            boxes[:, 0] = width - boxes[:, 0]
            boxes[:, 2] = width - boxes[:, 2]
        if np.random.rand() < 0.5:
            image = image[:, ::-1, :].copy()
            masks = masks[:, ::-1, :].copy()
            boxes = boxes[:, [0, 3, 2, 1]].copy()
            boxes[:, 1] = height - boxes[:, 1]
            boxes[:, 3] = height - boxes[:, 3]
        turns = np.random.randint(0, 4)
        if turns:
            image = np.rot90(image, turns, axes=(1, 2)).copy()
            masks = np.rot90(masks, turns, axes=(1, 2)).copy()
            boxes = boxesFromMasks(masks, boxes.shape[0])
        return image, masks, boxes

    def _target(self, boxes, masks, index):
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32),
                  "labels": torch.ones((len(boxes),), dtype=torch.int64),
                  "image_id": torch.tensor([index])}
        if self.withMasks:
            # box-only architectures reject an unexpected masks key, and
            # carrying full-resolution masks that are never used is the largest
            # avoidable cost in the data loader
            target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
        return target


def boxesFromMasks(masks, count):
    """Recompute boxes after a rotation, from the masks themselves."""
    boxes = np.zeros((count, 4), np.float32)
    for position in range(min(count, masks.shape[0])):
        rows, cols = np.nonzero(masks[position])
        if rows.size:
            boxes[position] = [cols.min(), rows.min(), cols.max() + 1,
                               rows.max() + 1]
    return boxes


def collate(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------- #
# model
# ---------------------------------------------------------------------- #

def buildModel(channelCount, modelType="maskrcnn", pretrained=True,
               scoreThreshold=None, nmsThreshold=None, imageSize=None):
    """One detector from the zoo in dlModels, ready for this many channels."""
    return dm.buildDetector(modelType, channelCount, numClasses=2,
                            pretrained=pretrained,
                            scoreThreshold=scoreThreshold,
                            nmsThreshold=nmsThreshold, imageSize=imageSize)


def trainFold(model, tiles, root, device, epochs, batchSize, learningRate,
              workers=2, verbose=True, withMasks=True, optimiser="adamw"):
    loader = DataLoader(TileDataset(root, tiles, augment=True,
                                    withMasks=withMasks),
                        batch_size=batchSize, shuffle=True,
                        num_workers=workers, collate_fn=collate)
    model.to(device).train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optim, scheduler, stepWhen = dm.buildOptimiser(
        model, optimiser, learningRate, epochs, len(loader))

    for epoch in range(epochs):
        started = time.time()
        loss = _trainEpoch(model, loader, device, parameters, optim,
                           scheduler if stepWhen == "step" else None)
        if stepWhen == "epoch":
            scheduler.step()
        if verbose:
            print("      epoch %2d/%d  loss %.4f  (%.1fs)"
                  % (epoch + 1, epochs, loss, time.time() - started))
    return model


def _trainEpoch(model, loader, device, parameters, optim, stepScheduler):
    running = 0.0
    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = sum(model(images, targets).values())
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optim.step()
        if stepScheduler is not None:
            stepScheduler.step()
        running += float(loss)
    return running / max(1, len(loader))


# ---------------------------------------------------------------------- #
# predict
# ---------------------------------------------------------------------- #

def predictTiles(model, tiles, root, device, transform, floor=SCORE_FLOOR,
                 batchSize=4):
    """
    Run the model over `tiles` and return world-space predictions.

    Tiles go through the model `batchSize` at a time rather than one by one;
    torchvision's detectors take a list of images and the GPU is otherwise
    mostly idle between calls. Mask models also return each detection's
    outline, converted to world coordinates with its box.
    """
    model.to(device).eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(tiles), batchSize):
            batch = tiles[start:start + batchSize]
            images = [torch.as_tensor(
                np.load(os.path.join(root, "tiles", entry["stem"] + ".npy"))
                .astype(np.float32) / 255.0).to(device) for entry in batch]
            for entry, output in zip(batch, model(images)):
                predictions.extend(dc.tileToWorld(
                    _tilePredictions(output, floor), transform,
                    entry["c0"], entry["r0"]))
    return predictions


def _tilePredictions(output, floor):
    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    masks = output.get("masks")
    keep = np.nonzero(scores >= floor)[0]
    result = []
    for index in keep:
        prediction = {"box": boxes[index].tolist(),
                      "score": float(scores[index])}
        if masks is not None:
            prediction["polygon"] = maskOutline(
                masks[index, 0].cpu().numpy(), boxes[index])
        result.append(prediction)
    return result


def maskOutline(probability, box, level=0.5):
    """
    The outline of one predicted mask, in tile pixels, or None.

    Traced inside the detection's own box: the mask is full-tile but empty
    outside it, and contouring 512 x 512 per detection is most of the cost.
    """
    height, width = probability.shape
    x0 = max(0, int(np.floor(box[0])))
    y0 = max(0, int(np.floor(box[1])))
    x1 = min(width, int(np.ceil(box[2])) + 1)
    y1 = min(height, int(np.ceil(box[3])) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    binary = (probability[y0:y1, x0:x1] >= level).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    outline = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(outline) < 3:
        return None
    return (outline + [x0, y0]).astype(float).tolist()


# ---------------------------------------------------------------------- #
# cross-validation
# ---------------------------------------------------------------------- #

class MaskRcnnCrossValidation(object):

    def __init__(self, args):
        self.args = args
        self.meta = dp.loadDataset(args.dataset)
        self.blocks = self.meta["blockGeometries"]
        self.crowns, _ = readCrowns(self.meta["crownsPath"], self.meta["crs"])
        self.device = dm.torchDevice(args.device)

    def splitFold(self, name):
        """Fit tiles, validation tiles, the validation block, test tiles."""
        trainTiles, testTiles = dc.foldSplit(self.meta["tiles"], self.blocks,
                                             name, self.args.buffer)
        if not trainTiles or not testTiles:
            return None
        # the learning curve's annotated subset; all of them by default
        trainTiles = dc.drawTiles(trainTiles, self.args.trainFraction,
                                  self.args.subsetSeed, name)
        validationBlock = dc.chooseValidationBlock(trainTiles, name)
        valTiles = [t for t in trainTiles if t["block"] == validationBlock]
        fitTiles = [t for t in trainTiles if t["block"] != validationBlock]
        if not valTiles or not fitTiles:
            fitTiles, valTiles = trainTiles, trainTiles[:1]
        return fitTiles, valTiles, validationBlock, testTiles

    def predict(self, model, tiles, meta=None):
        meta = meta or self.meta
        raw = predictTiles(model, tiles, meta["root"], self.device,
                           meta["transformObject"],
                           batchSize=self.args.inferenceBatch)
        return raw, dc.nonMaximumSuppression(raw, self.args.nmsIou)

    def validationRegion(self, validationBlock, valTiles):
        """
        Where the operating point is chosen: the validation block, or with an
        annotated subset only the part of it its drawn tiles cover.
        """
        geometry = dict(self.blocks)[validationBlock]
        if self.args.trainFraction >= 1:
            return geometry
        return geometry.intersection(dc.tileRegion(valTiles))

    def chooseThreshold(self, model, valTiles, region, crowns=None,
                        meta=None):
        """The operating point, and the validation predictions it came from."""
        _, merged = self.predict(model, valTiles, meta)
        if self.args.fixedThreshold:
            return self.args.scoreThreshold, merged
        threshold, _ = dc.selectThreshold(
            merged, self.crowns if crowns is None else crowns, region)
        return threshold, merged

    def runFold(self, name, geometry):
        split = self.splitFold(name)
        if split is None:
            print("  fold %s: skipped, empty split" % name)
            return None
        fitTiles, valTiles, validationBlock, testTiles = split
        print("  fold %s: fit %d tiles, validate on %s (%d), test %d"
              % (name, len(fitTiles), validationBlock, len(valTiles),
                 len(testTiles)))

        args = self.args
        model = buildModel(self.meta["channelCount"],
                           modelType=args.modelType,
                           pretrained=args.pretrained,
                           imageSize=args.imageSize or self.meta["tileSize"])
        trainFold(model, fitTiles, self.meta["root"], self.device,
                  args.epochs, args.batchSize, args.learningRate,
                  workers=args.workers,
                  withMasks=dm.needsMasks(args.modelType),
                  optimiser=args.optimiser)

        threshold, validation = self.chooseThreshold(
            model, valTiles, self.validationRegion(validationBlock, valTiles))
        raw, merged = self.predict(model, testTiles)
        result = dc.evaluateDetections(
            [p for p in merged if p["score"] >= threshold], self.crowns,
            geometry, verbose=True)
        result.update(block=name, validationBlock=validationBlock,
                      threshold=threshold, rawPredictions=len(raw),
                      fitTiles=len(fitTiles), validationTiles=len(valTiles),
                      trainFraction=self.args.trainFraction,
                      subsetSeed=self.args.subsetSeed)

        self.savePredictions(name, merged, threshold, validationBlock,
                             validation)
        if args.saveModels:
            torch.save(model.state_dict(), os.path.join(
                args.output, "%s_%s.pt" % (args.modelType, name)))
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    def savePredictions(self, name, merged, threshold, validationBlock,
                        validation):
        """
        Every prediction down to the score floor, on the test block and on the
        validation block, with the threshold the fold chose. The test set lets
        methods be compared tree by tree at any operating point; the
        validation set lets a combination be tuned without touching the test
        block.
        """
        dc.saveJson({"block": name, "threshold": threshold,
                     "scoreFloor": SCORE_FLOOR, "predictions": merged,
                     "validationBlock": validationBlock,
                     "validationPredictions": validation},
                    os.path.join(self.args.output,
                                 "predictions_%s.json" % name))

    def run(self):
        print("[torchvision] model %s, dataset %s, %d channels, %d tiles, "
              "%d blocks, optimiser %s, device %s"
              % (self.args.modelType, self.meta["name"],
                 self.meta["channelCount"], len(self.meta["tiles"]),
                 len(self.blocks), self.args.optimiser, self.device))
        wanted = set(self.args.folds.split(",")) if self.args.folds else None
        folds = []
        for name, geometry in self.blocks:
            if wanted and name not in wanted:
                continue
            result = self.runFold(name, geometry)
            if result is not None:
                folds.append(result)
        return folds, dc.averageFolds(folds)


class TransferRun(MaskRcnnCrossValidation):
    """
    The learning curve's point with no annotation of this site: one network
    trained on every block of another site (one of its blocks set aside to
    choose the operating point there), applied unchanged to every block here.
    """

    def trainOnSource(self):
        """The source site's network and the operating point chosen there."""
        source = dp.loadDataset(self.args.transferFrom)
        if source["channelCount"] != self.meta["channelCount"]:
            raise ValueError("the source dataset has %d channels, this one %d"
                             % (source["channelCount"],
                                self.meta["channelCount"]))
        crowns, _ = readCrowns(source["crownsPath"], source["crs"])
        tiles = [t for t in source["tiles"] if t["block"]]
        validationBlock = dc.chooseValidationBlock(tiles, "")
        fitTiles = [t for t in tiles if t["block"] != validationBlock]
        valTiles = [t for t in tiles if t["block"] == validationBlock]
        print("  source %s: fit %d tiles, validate on %s (%d)"
              % (source["name"], len(fitTiles), validationBlock, len(valTiles)))
        args = self.args
        model = buildModel(source["channelCount"], modelType=args.modelType,
                           pretrained=args.pretrained,
                           imageSize=args.imageSize or source["tileSize"])
        trainFold(model, fitTiles, source["root"], self.device, args.epochs,
                  args.batchSize, args.learningRate, workers=args.workers,
                  withMasks=dm.needsMasks(args.modelType),
                  optimiser=args.optimiser)
        region = dict(source["blockGeometries"])[validationBlock]
        threshold, _ = self.chooseThreshold(model, valTiles, region, crowns,
                                            source)
        return model, threshold, source["name"]

    def testBlock(self, model, threshold, sourceName, name, geometry):
        tiles = [t for t in self.meta["tiles"] if t["block"] == name]
        if not tiles:
            return None
        raw, merged = self.predict(model, tiles)
        result = dc.evaluateDetections(
            [p for p in merged if p["score"] >= threshold], self.crowns,
            geometry, verbose=True)
        result.update(block=name, threshold=threshold, rawPredictions=len(raw),
                      transferredFrom=sourceName)
        dc.saveJson({"block": name, "threshold": threshold,
                     "scoreFloor": SCORE_FLOOR, "predictions": merged,
                     "transferredFrom": sourceName},
                    os.path.join(self.args.output, "predictions_%s.json" % name))
        return result

    def run(self):
        model, threshold, sourceName = self.trainOnSource()
        print("[torchvision] trained on %s, threshold %.2f, applied to %s"
              % (sourceName, threshold, self.meta["name"]))
        folds = [r for r in (self.testBlock(model, threshold, sourceName,
                                            name, geometry)
                             for name, geometry in self.blocks)
                 if r is not None]
        return folds, dc.averageFolds(folds)


def crossValidate(args):
    os.makedirs(args.output, exist_ok=True)
    dc.startLogging(args.log or os.path.join(args.output, "run.log"))

    validation = (TransferRun if args.transferFrom
                  else MaskRcnnCrossValidation)(args)
    folds, pooled = validation.run()

    print("\n[torchvision] pooled over %d folds: R %.3f  P %.3f  F1 %.3f "
          "(per-fold F1 %.3f +- %.3f)"
          % (pooled["folds"], pooled["recall"], pooled["precision"],
             pooled["f1"], pooled["perFoldF1Mean"], pooled["perFoldF1Std"]))
    path = os.path.join(args.output, "results.json")
    dc.saveJson({"method": args.modelType, "dataset": validation.meta["name"],
                 "source": validation.meta["source"],
                 "channelCount": validation.meta["channelCount"],
                 "settings": vars(args), "folds": folds, "pooled": pooled},
                path)
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
                        choices=dm.MODEL_TYPES,
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
    parser.add_argument("--subsetSeed", type=int, default=1,
                        help="which random draw of --trainFraction")
    parser.add_argument("--transferFrom", default=None,
                        help="learning curve, no annotation here: train one "
                             "network on every block of this other dataset "
                             "and apply it to every block of --dataset")
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
