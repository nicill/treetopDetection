"""Detectors: settings in, Detections out."""

from .base import NMS, Detector, PostProcessor, nmsKeep
from .concomp import ConnectedComponents
from .height import (CCDetector, HeightDetector, LocalMaximaDetector,
                     WatershedDetector)
from .merging import TopMerger, saddleDrop

__all__ = ["CCDetector", "ConnectedComponents", "Detector", "HeightDetector",
           "LocalMaximaDetector", "NMS", "PostProcessor", "TopMerger",
           "WatershedDetector", "nmsKeep", "saddleDrop"]
