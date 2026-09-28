"""
TopMerger — deciding when two tops are the same tree.

Previously this was spread over three places in the detector, in two coordinate
systems, and a metric added to one of them silently did nothing in the other.
That is exactly how the saddle metric came to look useless: the test was
implemented in the global pass but not in the per-blob pass, where most merging
happens. Every proximity decision now goes through this class.

Four metrics:

    "2d"         horizontal distance. The original.
    "3d"         sqrt(dxy^2 + dz^2); a metre of height counts as a metre of
                 ground.
    "composite"  heightWeight*|dz| + (1-heightWeight)*dxy. Note the horizontal
                 reach is eps/(1-heightWeight), so at weight 0.7 an eps of 1 m
                 still merges tops 3.3 m apart at equal height.
    "saddle"     merge only if the canopy between the two tops never dips more
                 than saddleDropM below the lower of them.

"saddle" is the one that works, and it needs a large eps to do anything.
Measured on real data: tops in one crown sit a median 0.25 m apart with a
median dip of 0.00 m between them; tops in different crowns sit 5.84 m apart
with a median dip of 6.63 m. At eps 2 m, distance has already excluded almost
every different-crown pair, so the dip test has nothing left to reject. Give it
eps 5 to 12 m and let saddleDropM discriminate. At drop > 2 m the dip keeps
apart 82.9% of different-crown pairs while wrongly splitting 3.0% of same-crown
ones.
"""

import numpy as np
from scipy.spatial import cKDTree


METRICS = ("2d", "3d", "composite", "saddle")


class TopMerger(object):

    def __init__(self, metric="2d", epsM=0.70, heightWeight=0.7,
                 saddleDropM=0.5):
        if metric not in METRICS:
            raise ValueError("metric must be one of %s" % (METRICS,))
        self.metric = metric
        self.epsM = float(epsM)
        self.heightWeight = float(heightWeight)
        self.saddleDropM = float(saddleDropM)

    def __repr__(self):
        detail = ""
        if self.metric == "composite":
            detail = ", w=%.2f" % self.heightWeight
        elif self.metric == "saddle":
            detail = ", drop=%.2f" % self.saddleDropM
        return "TopMerger(%s, eps=%.2f m%s)" % (self.metric, self.epsM, detail)

    @property
    def horizontalReachM(self):
        """The furthest apart two tops can be and still merge."""
        if self.metric == "composite":
            return self.epsM / max(1e-6, 1.0 - self.heightWeight)
        return self.epsM

    # ------------------------------------------------------------------ #

    def distance(self, horizontalM, heightDifferenceM):
        """Separation in metres under the configured metric."""
        horizontal = float(horizontalM)
        difference = abs(float(heightDifferenceM))
        if self.metric == "3d":
            return float(np.hypot(horizontal, difference))
        if self.metric == "composite":
            return (self.heightWeight * difference
                    + (1.0 - self.heightWeight) * horizontal)
        return horizontal  # "2d" and "saddle" both start here

    def areSame(self, horizontalM, heightDifferenceM,
                surface=None, a=None, b=None):
        """Should these two tops be merged?"""
        if horizontalM >= self.horizontalReachM:
            return False
        if self.metric == "saddle":
            if surface is None:
                return horizontalM < self.epsM
            return (horizontalM < self.epsM
                    and saddleDrop(surface, a[0], a[1], b[0], b[1])
                    <= self.saddleDropM)
        return self.distance(horizontalM, heightDifferenceM) < self.epsM

    # ------------------------------------------------------------------ #

    def refineWithinBlob(self, grouped, scene, metresPerLevel, window):
        """
        Collapse tops inside one sub-blob, keeping the highest of any group.

        Positions are window pixels and heights are levels of the 0..255
        stretch, so both are converted before the metric sees them. The dip is
        measured on the window's real heights rather than the stretched
        single-label blob, because a straight line between two tops of one blob
        can leave the label and the zeros outside would read as a chasm.
        """
        result = {}
        for label, entries in grouped.items():
            if not entries:
                continue
            ordered = sorted(entries, key=lambda item: item[0], reverse=True)
            kept = [ordered[0]]
            for candidate in ordered[1:]:
                if not self._clashes(candidate, kept, scene, metresPerLevel,
                                     window):
                    kept.append(candidate)
            result[label] = kept
        return result

    def _clashes(self, candidate, kept, scene, metresPerLevel, window):
        for height, position in kept:
            horizontal = float(np.hypot(candidate[1][0] - position[0],
                                        candidate[1][1] - position[1])) \
                * scene.pixelSize
            difference = abs(candidate[0] - height) * (metresPerLevel or 0.0)
            if self.areSame(horizontal, difference, surface=window,
                            a=candidate[1], b=position):
                return True
        return False

    def merge(self, seeds, scene):
        """
        Merge the global seed set: take the highest top, discard everything it
        absorbs, repeat.

        Horizontal distance bounds every metric from below, so a KD-tree radius
        query prunes the candidates and only survivors need the full test.
        Without it this is quadratic in a seed count that runs to thousands.
        """

        if not seeds:
            return []

        positions = np.array(seeds, dtype=np.float64)
        heights = np.array([scene.chm[int(y), int(x)] for x, y in seeds])
        radiusPixels = self.horizontalReachM / scene.pixelSize

        tree = cKDTree(positions)
        removed = np.zeros(len(seeds), bool)
        kept = []

        for index in np.argsort(-heights):
            if removed[index]:
                continue
            kept.append(index)
            for other in tree.query_ball_point(positions[index], radiusPixels):
                if other == index or removed[other]:
                    continue
                horizontal = float(np.hypot(*(positions[other]
                                              - positions[index]))) \
                    * scene.pixelSize
                if self.areSame(horizontal,
                                heights[other] - heights[index],
                                surface=scene.chm,
                                a=(int(positions[index][1]),
                                   int(positions[index][0])),
                                b=(int(positions[other][1]),
                                   int(positions[other][0]))):
                    removed[other] = True

        return [seeds[i] for i in sorted(kept)]


def saddleDrop(surface, rowA, columnA, rowB, columnB):
    """
    How far the surface dips between two points, below the lower of them.

    This is what a geodesic over the canopy is really responding to: two apexes
    on one crown are joined by a path that stays near the top, while two on
    neighbouring crowns are joined by one that descends into the gap and climbs
    back. Measuring the depth of that gap is the same discrimination at a
    fraction of the cost of tracing a path.
    """
    steps = max(2, int(np.hypot(rowB - rowA, columnB - columnA)) + 1)
    # The displacement from A is rounded, and A added back as an integer.
    # Rounding absolute coordinates instead is not translation-invariant:
    # np.rint rounds half to even, so a sample 2.5 px along comes out as 3 from
    # an odd origin and 2 from an even one. Windows overlap with different
    # origins, so the old form could give the same two treetops a different
    # saddle verdict depending on which window evaluated them.
    #
    # Built directly rather than through np.linspace, whose Python overhead
    # dominates a call this small and this runs ~10^5 times per scene. No clip:
    # a straight line between two in-bounds pixels cannot leave the grid.
    fraction = np.arange(steps) / (steps - 1)
    rows = rowA + np.rint(fraction * (rowB - rowA)).astype(np.intp)
    columns = columnA + np.rint(fraction * (columnB - columnA)).astype(np.intp)
    profile = surface[rows, columns]
    lower = min(float(surface[rowA, columnA]), float(surface[rowB, columnB]))
    return lower - float(profile.min())
