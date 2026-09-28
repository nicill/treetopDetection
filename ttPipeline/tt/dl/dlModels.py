"""
dlModels.py

The torchvision detector zoo, built so every architecture can be run through
the same cross-validation and the same evaluation. Follows the model choices
already in your train.py (`get_model_instance_segmentation`), with the
per-architecture head replacement and the arbitrary-channel stem handled in one
place.

    maskrcnn      maskrcnn_resnet50_fpn_v2        masks + boxes
    fasterrcnn    fasterrcnn_resnet50_fpn_v2      boxes
    fasterrcnnmb  fasterrcnn_mobilenet_v3_large_fpn   boxes, the one your
                                                  train.py uses; much lighter
                                                  and a fair bit weaker
    retinanet     retinanet_resnet50_fpn_v2       boxes, one stage
    fcos          fcos_resnet50_fpn               boxes, anchor free
    ssd           ssd300_vgg16                    boxes, fixed 300 px input

`convnextmaskrcnn` is deliberately absent: it is your own
`get_maskrcnn_convnext`, and half-copying it here would leave two versions to
keep in step. Import it and pass it in if you want it in the table.

Channel handling
----------------
The stem is found by walking the model and taking the first Conv2d, rather than
hard-coding a path per architecture — resnet backbones keep it at
`backbone.body.conv1`, mobilenet at `backbone.body[0][0]`, vgg at
`backbone.features[0]`, and those paths drift between torchvision versions.
Weights are carried over rather than reinitialised: summed for one channel,
kept plus a neutral mean-initialised copy for four.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import _utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.fcos import FCOSClassificationHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection.ssd import SSDClassificationHead


MODEL_TYPES = ["maskrcnn", "fasterrcnn", "fasterrcnnmb", "retinanet", "fcos",
               "ssd"]

MASK_MODELS = {"maskrcnn"}


def needsMasks(modelType):
    return modelType in MASK_MODELS


# ---------------------------------------------------------------------- #

def findFirstConvolution(model):
    """Return (parentModule, attributeName, module) for the first Conv2d."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                # getattr first: nn.Module resolves submodules registered by
                # name, including the string keys of a ModuleDict. Indexing
                # with int() breaks there, because IntermediateLayerGetter —
                # what the mobilenet backbone uses — keys its children as the
                # strings "0", "1", ... and raises KeyError on the integer.
                if hasattr(parent, part):
                    parent = getattr(parent, part)
                else:
                    try:
                        parent = parent[int(part)]
                    except (KeyError, TypeError, ValueError):
                        parent = parent[part]
            return parent, parts[-1], module
    raise RuntimeError("no Conv2d found in this model")


def retargetChannels(model, channelCount):
    """Rebuild the stem for `channelCount` inputs, carrying the weights over."""
    if channelCount == 3:
        return model

    parent, attribute, old = findFirstConvolution(model)
    new = nn.Conv2d(channelCount, old.out_channels,
                    kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, dilation=old.dilation,
                    groups=1, bias=old.bias is not None)
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
        if old.bias is not None:
            new.bias.copy_(old.bias.detach())
    setattr(parent, attribute, new)

    transform = getattr(model, "transform", None)
    if transform is not None and hasattr(transform, "image_mean"):
        baseMean = [0.485, 0.456, 0.406]
        baseStd = [0.229, 0.224, 0.225]
        if channelCount == 1:
            transform.image_mean = [float(sum(baseMean) / 3)]
            transform.image_std = [float(sum(baseStd) / 3)]
        else:
            mean = list(baseMean) + [float(sum(baseMean) / 3)] * \
                max(0, channelCount - 3)
            std = list(baseStd) + [float(sum(baseStd) / 3)] * \
                max(0, channelCount - 3)
            transform.image_mean = mean[:channelCount]
            transform.image_std = std[:channelCount]
    return model


# ---------------------------------------------------------------------- #

def _replaceRoiPredictor(model, numClasses):
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, numClasses)


def _maskrcnn(weights, numClasses):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=weights, weights_backbone=weights)
    _replaceRoiPredictor(model, numClasses)
    features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(features, 256,
                                                       numClasses)
    return model


def _fasterrcnn(weights, numClasses):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=weights, weights_backbone=weights)
    _replaceRoiPredictor(model, numClasses)
    return model


def _fasterrcnnMobile(weights, numClasses):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, weights_backbone=weights)
    _replaceRoiPredictor(model, numClasses)
    return model


def _retinanet(weights, numClasses):
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights=weights, weights_backbone=weights)
    head = model.head.classification_head
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=head.conv[0][0].in_channels,
        num_anchors=model.anchor_generator.num_anchors_per_location()[0],
        num_classes=numClasses,
        norm_layer=lambda channels: nn.GroupNorm(32, channels))
    return model


def _fcos(weights, numClasses):
    model = torchvision.models.detection.fcos_resnet50_fpn(
        weights=weights, weights_backbone=weights)
    head = model.head.classification_head
    # FCOS's head.conv is a flat Sequential of Conv2d/GroupNorm/ReLU, unlike
    # RetinaNet's Sequential of Conv2dNormActivation blocks
    model.head.classification_head = FCOSClassificationHead(
        in_channels=head.conv[0].in_channels,
        num_anchors=model.anchor_generator.num_anchors_per_location()[0],
        num_classes=numClasses)
    return model


def _ssd(weights, numClasses):
    model = torchvision.models.detection.ssd300_vgg16(
        weights=weights, weights_backbone=weights)
    model.head.classification_head = SSDClassificationHead(
        in_channels=_utils.retrieve_out_channels(model.backbone, (300, 300)),
        num_anchors=model.anchor_generator.num_anchors_per_location(),
        num_classes=numClasses)
    return model


BUILDERS = {"maskrcnn": _maskrcnn, "fasterrcnn": _fasterrcnn,
            "fasterrcnnmb": _fasterrcnnMobile, "retinanet": _retinanet,
            "fcos": _fcos, "ssd": _ssd}


def buildDetector(modelType, channelCount, numClasses=2, pretrained=True,
                  scoreThreshold=None, nmsThreshold=None, imageSize=None):
    """
    One detector, head replaced for `numClasses` (background included), stem
    rebuilt for `channelCount`, and optionally pinned to `imageSize` pixels.
    """
    if modelType not in BUILDERS:
        raise ValueError("unknown modelType %r; expected one of %s"
                         % (modelType, MODEL_TYPES))
    model = BUILDERS[modelType]("DEFAULT" if pretrained else None, numClasses)
    model = retargetChannels(model, channelCount)
    if imageSize:
        setInputSize(model, imageSize)
    setThresholds(model, scoreThreshold, nmsThreshold)
    return model


def setThresholds(model, scoreThreshold=None, nmsThreshold=None):
    """Two-stage models keep these on roi_heads, one-stage ones on the model."""
    owner = model.roi_heads if hasattr(model, "roi_heads") else model
    if scoreThreshold is not None:
        owner.score_thresh = scoreThreshold
    if nmsThreshold is not None:
        owner.nms_thresh = nmsThreshold
    return model


# ---------------------------------------------------------------------- #

def buildOptimiser(model, kind, learningRate, epochs, stepsPerEpoch,
                   weightDecay=1e-4):
    """
    Two schedules.

    `adamw` is the default: AdamW with OneCycle. On a few hundred tiles and a
    few dozen epochs it reaches a usable model faster and is far less sensitive
    to the learning rate being slightly wrong.

    `sgd` reproduces the schedule in your train.py — SGD lr 0.005, momentum
    0.9, StepLR(step_size=3, gamma=0.1) — so numbers can be compared with runs
    you have already done. Worth knowing what that schedule does over a long
    run: gamma 0.1 every 3 epochs drops the rate by 10x eight times in 24
    epochs, so from about epoch 12 onwards it is training at 5e-9 and nothing
    further is learned. It is fine for the ~10 epoch runs it was written for
    and wasteful past that.
    """
    parameters = [p for p in model.parameters() if p.requires_grad]

    if kind == "sgd":
        optimiser = torch.optim.SGD(parameters, lr=learningRate,
                                    momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=3,
                                                    gamma=0.1)
        return optimiser, scheduler, "epoch"

    optimiser = torch.optim.AdamW(parameters, lr=learningRate,
                                  weight_decay=weightDecay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=learningRate,
        total_steps=max(1, epochs * stepsPerEpoch))
    return optimiser, scheduler, "step"


def torchDevice(spec):
    """
    Accept what people actually type: 0, "0", "cuda:1", "cpu", "mps", or
    nothing. torch.device("0") raises, and "0" is what nvidia-smi and
    ultralytics both use, so it is the obvious thing to pass.
    """

    if spec is None or spec == "":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = str(spec).strip()
    if text.isdigit():
        return torch.device("cuda:%s" % text)
    return torch.device(text)


def setInputSize(model, size):
    """
    Run the model at `size` pixels instead of torchvision's default.

    torchvision's detectors resize every input to a shortest side of 800 px
    unless told otherwise. On 512 px tiles that is a silent 1.56x upscale: the
    model sees crowns half again as wide as they are, and pays about 1.4x the
    compute for it. It also made the comparison uneven, since YOLO trains at
    the tile size. Pinning both bounds to the tile size runs it at native
    resolution.
    """
    transform = getattr(model, "transform", None)
    if transform is None or not hasattr(transform, "min_size"):
        return model
    if getattr(transform, "fixed_size", None):
        # SSD300 is defined at a fixed 300 x 300; min and max are ignored there
        return model
    transform.min_size = (int(size),)
    transform.max_size = int(size)
    return model
