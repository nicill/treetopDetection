"""
YOLODetector — ultralytics YOLO as a DLDetector.

Segmentation weights are the default so the model does the same job as Mask
R-CNN; task "detect" trains boxes only, which on this metric loses nothing
(crowns are scored by where a detection's centre lands).

Channels: one band is written as a grey PNG repeated three times, keeping the
pretrained stem intact; three as a colour PNG; four as a four-channel PNG,
with the first convolution rebuilt and ultralytics' image reader patched to
stop discarding the fourth band. That path reaches into ultralytics internals,
so it checks itself and refuses to train on three channels by mistake.

Ultralytics trains from a folder layout (images/ and labels/ per split, and a
data.yaml); fit writes it under workDir. Its own validation split, which it
early-stops and selects the checkpoint on, must come out of the training
blocks, never the block then scored.
"""

import os
import shutil
import sys as _sys

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset
import ultralytics.utils.patches as patches

from .base import DLDetector


class YOLODetector(DLDetector):

    method = "yolo"
    defaults = dict(DLDetector.defaults, **{
        "threshold": 0.25, "weights": "yolo11s-seg.pt", "task": "segment",
        "epochs": 100, "patience": 30, "batchSize": 8, "workers": 4,
        "device": None, "mosaic": 0.0, "scale": 0.1, "pretrained": True,
        "verboseTraining": False, "workDir": "yoloWork", "keepImages": False,
    })

    def __init__(self, postprocessors=None, **params):
        super().__init__(postprocessors, **params)
        self.model = None
        self.weightsPath = None
        self.device = self.params["device"] if self.params["device"] \
            else (0 if torch.cuda.is_available() else "cpu")

    @property
    def geometry(self):
        return "masks" if self.params["task"] == "segment" else "boxes"

    def reset(self, patchSet):
        if self.weightsPath is None:
            self.weightsPath = resolveWeights(self.params["weights"],
                                              self.params["task"])
        self.channelCount, self.tileSize = (patchSet.channelCount,
                                            patchSet.tileSize)
        if self.channelCount == 4:
            enableFourChannelReading()
        model = YOLO(self.weightsPath)
        if self.channelCount == 4:
            model = patchFirstConvolution(model, 4)
            actual = model.model.model[0].conv.in_channels
            if actual != 4:
                raise RuntimeError("the first convolution is still %d-channel; "
                                   "the four-channel patch did not take"
                                   % actual)
        self.model = model
        return model

    def fit(self, patchSet, patches, validation=None):
        """Write the train and val splits under workDir and train there."""
        p = self.params
        work = p["workDir"]
        if os.path.exists(work):
            shutil.rmtree(work)
        validation = validation or patches[:max(1, len(patches) // 8)]
        for split, chosen in (("train", patches), ("val", validation)):
            self._export(patchSet, chosen, split, withLabels=True)
        dataYaml = os.path.join(work, "data.yaml")
        with open(dataYaml, "w") as handle:
            handle.write("path: %s\n" % os.path.abspath(work))
            handle.write("train: images/train\nval: images/val\n")
            handle.write("names:\n  0: tree\n")
        self.model.train(data=dataYaml, epochs=p["epochs"],
                         imgsz=self.tileSize, batch=p["batchSize"],
                         device=self.device, workers=p["workers"],
                         project=work, name="train", exist_ok=True,
                         pretrained=p["pretrained"],
                         verbose=p["verboseTraining"], seed=0,
                         deterministic=False, patience=p["patience"],
                         degrees=180.0, fliplr=0.5, flipud=0.5, hsv_h=0.0,
                         hsv_s=0.0, hsv_v=0.0, mosaic=p["mosaic"],
                         scale=p["scale"])
        if not p["keepImages"]:
            for split in ("train", "val"):
                shutil.rmtree(os.path.join(work, "images", split),
                              ignore_errors=True)
        return self.model

    def predictPatches(self, patchSet, patches):
        """The whole list goes to ultralytics at once; it batches itself."""
        if not patches:
            return []
        paths = self._export(patchSet, patches, "predict", withLabels=False)
        results = self.model.predict(paths, conf=self.params["scoreFloor"],
                                     imgsz=self.tileSize, device=self.device,
                                     batch=self.params["batchSize"],
                                     verbose=False)
        gathered = []
        for patch, output in zip(patches, results):
            gathered.extend(patchSet.toWorld(yoloPredictions(output), patch))
        if not self.params["keepImages"]:
            shutil.rmtree(os.path.join(self.params["workDir"], "images",
                                       "predict"), ignore_errors=True)
        return gathered

    def saveWeights(self, path):
        self.model.save(path)

    def release(self):
        self.model = None

    def _export(self, patchSet, patches, split, withLabels):
        work = self.params["workDir"]
        imageDir = os.path.join(work, "images", split)
        labelDir = os.path.join(work, "labels", split)
        os.makedirs(imageDir, exist_ok=True)
        if withLabels:
            os.makedirs(labelDir, exist_ok=True)
        paths = []
        for patch in patches:
            path = os.path.join(imageDir, patch.stem + ".png")
            writeTileImage(patch.image(), path)
            if withLabels:
                writeLabels(patch.crowns(), patchSet.tileSize,
                            os.path.join(labelDir, patch.stem + ".txt"),
                            self.params["task"])
            paths.append(path)
        return paths


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
