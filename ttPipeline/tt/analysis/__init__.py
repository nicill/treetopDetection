"""Analysis of sites and of results."""

from .analysis import Analysis
from .statistics import (compareFolds, holm, pairedTest, profile, quantiles,
                         splitBackground)
from .trees import TreeTable

__all__ = ["Analysis", "TreeTable", "compareFolds", "holm", "pairedTest",
           "profile", "quantiles", "splitBackground"]
