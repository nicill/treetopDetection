"""Treetop detection and crown-level evaluation on canopy height models."""

import warnings

from .data import Annotations, Detections, HeightModel, RGBMosaic, TDData
from .detector import ConCompDetector
from .detectors import (CCDetector, Detector, LocalMaximaDetector, NMS,
                        PostProcessor, WatershedDetector)
from .experiments import CrossValidation, ParameterGrid, RunSet
from .scoring import AnnotatedCrowns, Evaluator, F1, Weighted
from .evaluation import CrownEvaluator, Classification
from .merging import TopMerger
from .pseudoCrowns import PseudoCrowns
from .scene import Scene, readCrowns
from .tops import Tops

# affine 3.0 deprecates * on Affine in favour of @. This package uses @
# throughout; rasterio still uses * inside its own windowed-read code, which
# fires on every tile read. The filter is scoped to warnings raised from
# rasterio's modules, so a regression back to * in this package still shows.
warnings.filterwarnings("ignore", category=PendingDeprecationWarning,
                        module=r"rasterio(\.|$)")


__all__ = [
    # the redesigned classes
    "TDData", "HeightModel", "RGBMosaic", "Annotations", "Detections",
    "Detector", "CCDetector", "LocalMaximaDetector", "WatershedDetector",
    "PostProcessor", "NMS", "Evaluator", "AnnotatedCrowns", "F1", "Weighted",
    "ParameterGrid", "CrossValidation", "RunSet",
    # still used by modules not yet moved to them
    "Scene", "Tops", "ConCompDetector", "CrownEvaluator", "Classification",
    "TopMerger", "PseudoCrowns", "readCrowns"]
