"""
Detector — settings in, Detections out — and the PostProcessor steps a
detector applies to its own output.
"""

import numpy as np

from ..experiments.grid import ParameterGrid


class Detector(object):
    """
    Subclasses declare `defaults` (every parameter they accept, with its
    default), `defaultGrid` (the axes tuned when nothing else is said) and
    `method` (the name written into run files), and implement _detect.
    """

    method = "detector"
    defaults = {}
    defaultGrid = {}
    geometry = "points"

    def __init__(self, postprocessors=None, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError("%s does not take %s; it takes %s"
                             % (type(self).__name__, sorted(unknown),
                                sorted(self.defaults)))
        self.params = dict(self.defaults, **params)
        self.postprocessors = list(postprocessors or [])

    def __repr__(self):
        changed = {k: v for k, v in self.params.items()
                   if self.defaults.get(k) != v}
        return "%s(%s)" % (type(self).__name__, ", ".join(
            "%s=%s" % item for item in sorted(changed.items())))

    def withParams(self, **params):
        """A detector like this one with some parameters changed."""
        return type(self)(postprocessors=self.postprocessors,
                          **dict(self.params, **params))

    @classmethod
    def parameterGrid(cls, **axes):
        """The default grid, with any axes given here replaced or added."""
        return ParameterGrid(cls.defaultGrid).withAxes(**axes)

    def prepare(self, data):
        """Build anything detect() reads and can share (before forking)."""

    def detect(self, data, region=None):
        """Detections on a TDData, postprocessed, optionally cut to a region."""
        detections = self._detect(data)
        for step in self.postprocessors:
            detections = step.apply(detections, data)
        if region is not None:
            detections = detections.inRegion(region)
        detections.detector = self.method
        detections.params = dict(self.params)
        return detections

    def _detect(self, data):
        raise NotImplementedError


class PostProcessor(object):
    """One step applied to a detector's output."""

    def apply(self, detections, data):
        raise NotImplementedError

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, ", ".join(
            "%s=%s" % item for item in sorted(vars(self).items())))


class NMS(PostProcessor):
    """
    Non-maximum suppression on boxes (a point counts as its small square), to
    merge the same tree seen in two overlapping tiles.
    """

    def __init__(self, iouThreshold=0.4):
        self.iouThreshold = float(iouThreshold)

    def apply(self, detections, data=None):
        return detections.subset(nmsKeep(detections.boxesOrPoints(),
                                         detections.scores, self.iouThreshold))


def nmsKeep(boxes, scores, iouThreshold):
    """Indices kept by plain NMS, highest score first."""
    if not len(boxes):
        return []
    boxes = np.asarray(boxes, np.float64)
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * \
        np.maximum(0, boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-np.asarray(scores, np.float64))
    keep = []
    while order.size:
        current = order[0]
        keep.append(int(current))
        if order.size == 1:
            break
        rest = order[1:]
        x0 = np.maximum(boxes[current, 0], boxes[rest, 0])
        y0 = np.maximum(boxes[current, 1], boxes[rest, 1])
        x1 = np.minimum(boxes[current, 2], boxes[rest, 2])
        y1 = np.minimum(boxes[current, 3], boxes[rest, 3])
        overlap = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
        union = areas[current] + areas[rest] - overlap
        iou = overlap / np.maximum(union, 1e-9)
        order = rest[iou <= iouThreshold]
    return keep
