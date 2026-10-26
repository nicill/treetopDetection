"""Experiments: detectors tried over parameter grids and scored."""

from .grid import ParameterGrid
from .runset import RunSet, poolFolds
from .crossValidation import CrossValidation, Experiment

__all__ = ["CrossValidation", "Experiment", "ParameterGrid", "RunSet",
           "poolFolds"]
