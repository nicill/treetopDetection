"""Treetop detection and crown-level evaluation on canopy height models."""

import warnings

from .analysis import Analysis
from .data import (Annotations, Detections, HeightModel, PatchSet, RGBMosaic,
                   TDData)
from .detectors import (CCDetector, Detector, LocalMaximaDetector, NMS,
                        PostProcessor, TopMerger, WatershedDetector)
from .experiments import CrossValidation, ParameterGrid, RunSet
from .scoring import AnnotatedCrowns, Evaluator, F1, Weighted
from .scoring.pseudoCrowns import PseudoCrowns

# affine 3.0 deprecates * on Affine in favour of @. This package uses @
# throughout; rasterio still uses * inside its own windowed-read code, which
# fires on every tile read. The filter is scoped to warnings raised from
# rasterio's modules, so a regression back to * in this package still shows.
warnings.filterwarnings("ignore", category=PendingDeprecationWarning,
                        module=r"rasterio(\.|$)")


__all__ = [
    "TDData", "HeightModel", "RGBMosaic", "Annotations", "Detections",
    "PatchSet", "Detector", "CCDetector", "LocalMaximaDetector",
    "WatershedDetector", "TopMerger", "PostProcessor", "NMS", "Evaluator",
    "AnnotatedCrowns", "F1", "Weighted", "ParameterGrid", "CrossValidation",
    "RunSet", "Analysis", "PseudoCrowns"]
