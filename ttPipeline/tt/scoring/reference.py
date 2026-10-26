"""
What detections are scored against: the annotated crowns, or crowns that
stand in for them (pseudo-crowns built from another detector's output).
"""

from ..data.annotations import crownsInRegion


class Reference(object):
    """Crown polygons to score against, selectable by region."""

    def crowns(self):
        raise NotImplementedError

    def crownsIn(self, region):
        """Crowns whose centroid lies in `region` (all, if region is None)."""
        crowns = self.crowns()
        if region is None:
            return crowns.reset_index(drop=True)
        return crownsInRegion(crowns, region)


class AnnotatedCrowns(Reference):
    """The hand-drawn crowns of a site."""

    def __init__(self, annotations):
        self.annotations = annotations

    def crowns(self):
        return self.annotations.crowns


class CrownFrame(Reference):
    """Any GeoDataFrame of crown polygons (e.g. pseudo-crowns)."""

    def __init__(self, frame):
        self.frame = frame

    def crowns(self):
        return self.frame
