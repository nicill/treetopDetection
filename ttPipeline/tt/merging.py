"""Compatibility: TopMerger now lives in tt.detectors.merging."""

from .detectors.merging import METRICS, TopMerger, saddleDrop

__all__ = ["METRICS", "TopMerger", "saddleDrop"]
