"""
TorchvisionDetector — the torchvision zoo (Mask R-CNN, Faster R-CNN, RetinaNet,
FCOS, SSD; models.py) as a DLDetector.

One class: tree. Tiles go in as (C, H, W) floats in 0..1 with any number of
channels (the first convolution is rebuilt and its pretrained weights carried
over, models.retargetChannels). Mask models also return each detection's
outline, traced inside its own box and stitched with it.
"""

import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import models
from .base import DLDetector

TORCHVISION_DEFAULTS = {
    "modelType": "maskrcnn", "epochs": 30, "batchSize": 2,
    "learningRate": 1e-4, "optimiser": "adamw", "workers": 2,
    "device": None, "pretrained": True, "imageSize": 0, "initWeights": None,
    "inferenceBatch": 4,
}


class TorchvisionDetector(DLDetector):

    defaults = dict(DLDetector.defaults, **TORCHVISION_DEFAULTS)

    def __init__(self, postprocessors=None, **params):
        super().__init__(postprocessors, **params)
        if self.params["modelType"] not in models.MODEL_TYPES:
            raise ValueError("modelType must be one of %s"
                             % (models.MODEL_TYPES,))
        self.model = None
        self.device = models.torchDevice(self.params["device"])

    @property
    def method(self):
        return self.params["modelType"]

    @property
    def geometry(self):
        return "masks" if models.needsMasks(self.params["modelType"]) \
            else "boxes"

    def reset(self, patchSet):
        p = self.params
        self.model = models.buildDetector(
            p["modelType"], patchSet.channelCount, numClasses=2,
            pretrained=p["pretrained"],
            imageSize=p["imageSize"] or patchSet.tileSize,
            initWeights=p["initWeights"])
        return self.model

    def fit(self, patchSet, patches, validation=None):
        """Train on these patches (validation is not used by torchvision)."""
        p = self.params
        loader = DataLoader(TileDataset(patches, augment=True,
                                        withMasks=models.needsMasks(
                                            p["modelType"])),
                            batch_size=p["batchSize"], shuffle=True,
                            num_workers=p["workers"], collate_fn=collate)
        self.model.to(self.device).train()
        if p["epochs"] <= 0:                 # the starting weights as they are
            return self.model
        parameters = [q for q in self.model.parameters() if q.requires_grad]
        optim, scheduler, stepWhen = models.buildOptimiser(
            self.model, p["optimiser"], p["learningRate"], p["epochs"],
            len(loader))
        for epoch in range(p["epochs"]):
            started = time.time()
            loss = _trainEpoch(self.model, loader, self.device, parameters,
                               optim, scheduler if stepWhen == "step"
                               else None)
            if stepWhen == "epoch":
                scheduler.step()
            print("      epoch %2d/%d  loss %.4f  (%.1fs)"
                  % (epoch + 1, p["epochs"], loss, time.time() - started))
        return self.model

    def predictPatches(self, patchSet, patches):
        """Tiles go through the network inferenceBatch at a time."""
        self.model.to(self.device).eval()
        floor, batchSize = self.params["scoreFloor"], \
            self.params["inferenceBatch"]
        predictions = []
        with torch.no_grad():
            for start in range(0, len(patches), batchSize):
                batch = patches[start:start + batchSize]
                images = [torch.as_tensor(patch.image().astype(np.float32)
                                          / 255.0).to(self.device)
                          for patch in batch]
                for patch, output in zip(batch, self.model(images)):
                    predictions.extend(patchSet.toWorld(
                        _tilePredictions(output, floor), patch))
        return predictions

    def saveWeights(self, path):
        torch.save(self.model.state_dict(), path)

    def release(self):
        """Drop the network and free the GPU memory it held."""
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class TileDataset(Dataset):
    """
    Tiles as (C, H, W) float tensors in 0..1, with per-instance masks.

    Augmentation is flips and 90-degree rotations only. Those are the
    transformations a nadir canopy raster is genuinely invariant to: a forest
    seen from above has no up. Colour jitter would break the CHM channel, whose
    values mean metres, and scaling would break the one strong prior the model
    has, that crowns are a known size in pixels.
    """

    def __init__(self, patches, augment=False, withMasks=True):
        self.patches = patches
        self.augment = augment
        self.withMasks = withMasks

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        image, masks, boxes = self._load(self.patches[index])
        if self.augment:
            image, masks, boxes = self._augment(image, masks, boxes)
        keep = ((boxes[:, 2] - boxes[:, 0]) > 1) & \
            ((boxes[:, 3] - boxes[:, 1]) > 1)
        return torch.as_tensor(image), self._target(boxes[keep], masks[keep],
                                                    index)

    @staticmethod
    def _load(patch):
        array = patch.image()
        crowns = patch.crowns()
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
