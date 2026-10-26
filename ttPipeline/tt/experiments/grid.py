"""ParameterGrid — the settings a detector is tried at."""

import itertools


class ParameterGrid(object):
    """
    Named axes, each a list of values; every combination is one setting.
    Settings are expanded in axis order (the last axis varies fastest), the
    order run files index them by, so keep axes in the order given.
    """

    def __init__(self, axes=None):
        self.axes = {k: list(v) for k, v in (axes or {}).items()}

    def __repr__(self):
        return "ParameterGrid(%d settings: %s)" % (
            len(self), ", ".join("%s x%d" % (k, len(v))
                                 for k, v in self.axes.items()))

    def __len__(self):
        count = 1
        for values in self.axes.values():
            count *= len(values)
        return count

    def expand(self):
        keys = list(self.axes)
        return [dict(zip(keys, values))
                for values in itertools.product(*[self.axes[k] for k in keys])]

    def withAxes(self, **axes):
        """A copy with these axes replaced or added (added at the end)."""
        merged = dict(self.axes)
        merged.update({k: list(v) for k, v in axes.items()})
        return ParameterGrid(merged)

    def varying(self):
        """The axes with more than one value."""
        return [k for k, v in self.axes.items() if len(v) > 1]
