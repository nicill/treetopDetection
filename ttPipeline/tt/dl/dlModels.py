"""Compatibility: the model zoo now lives in tt.detectors.dl.models."""

from ..detectors.dl.models import *  # noqa: F401,F403
from ..detectors.dl.models import (MASK_MODELS, MODEL_TYPES,  # noqa: F401
                                   loadTreeWeights)
