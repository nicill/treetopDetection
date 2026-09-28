"""Treetop detection and crown-level evaluation on canopy height models."""

import warnings

from .detector import ConCompDetector
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


__all__ = ["Scene", "Tops", "ConCompDetector", "CrownEvaluator",
           "Classification", "TopMerger", "PseudoCrowns", "readCrowns"]
