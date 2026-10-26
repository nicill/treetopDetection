"""
DLDetector — a network that is trained on patches and predicts on patches.

A network predicts tile by tile; the predictions are stitched into world
coordinates and passed through the detector's postprocessors (by default NMS,
which merges the same tree seen in two overlapping tiles). Everything above
the score floor is kept, so the operating point can be chosen afterwards on
data the network was not scored on; `threshold` is the operating point used
when the detector is run on its own.
"""

from ...data.detections import Detections
from ..base import NMS, Detector

SCORE_FLOOR = 0.05


class DLDetector(Detector):

    defaults = {"scoreFloor": SCORE_FLOOR, "threshold": 0.5}
    geometry = "boxes"

    def __init__(self, postprocessors=None, **params):
        if postprocessors is None:
            postprocessors = [NMS(0.4)]
        super().__init__(postprocessors=postprocessors, **params)

    # ------------------------------------------------------------------ #
    # what a network implements
    # ------------------------------------------------------------------ #

    def reset(self, patchSet):
        """A fresh, untrained network for these patches' channels."""
        raise NotImplementedError

    def fit(self, patchSet, patches, validation=None):
        """
        Train on these patches (their crowns are the labels). `validation`
        patches are for networks that select a checkpoint while training;
        they come from the training blocks, never the held-out one.
        """
        raise NotImplementedError

    def predictPatches(self, patchSet, patches):
        """World-space prediction dicts above the score floor, unmerged."""
        raise NotImplementedError

    def saveWeights(self, path):
        raise NotImplementedError

    def release(self):
        """Drop the trained network (between folds)."""

    # ------------------------------------------------------------------ #

    def predict(self, patchSet, patches, data=None):
        """(number of raw predictions, the postprocessed Detections)."""
        raw = self.predictPatches(patchSet, patches)
        detections = Detections.fromPredictions(raw, self.geometry,
                                                crs=patchSet.crs)
        for step in self.postprocessors:
            detections = step.apply(detections, data)
        detections.detector = self.method
        return len(raw), detections

    def detect(self, data, region=None):
        """Every patch of the site, at the operating point."""
        patches = data.patchSet.patches
        _, detections = self.predict(data.patchSet, patches, data)
        detections = detections.aboveScore(self.params["threshold"])
        if region is not None:
            detections = detections.inRegion(region)
        detections.params = dict(self.params)
        return detections
