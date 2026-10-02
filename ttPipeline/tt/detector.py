"""
ConCompDetector — treetops from a CHM by connected components and band descent.

A window slides over the CHM. In each window the heights are stretched between
two percentiles and eroded, which drops the low connecting canopy so crowns
begin to separate, and the result is labelled. Each blob big enough to hold a
tree is then walked down in height steps: at every step the blob is thresholded
and relabelled, and a sub-blob appearing for the first time contributes its
highest pixel as a new top. That descent does the real work, because crowns are
separate at the top and fused at the bottom. The seeds from every window are
merged at the end.

Performance notes, because they shaped the structure:

* Every blob is cropped to its bounding box before the descent. The descent
  thresholds and relabels the blob dozens of times; doing that on the whole
  160 x 160 window for a blob that occupies 15 x 15 of it was most of the cost.
* Within the descent, sub-blob maxima are found on each sub-blob's own bounding
  box from connectedComponentsWithStats, not by masking the full blob with
  np.where once per label, which was quadratic in the label count.
* The threshold buffer is allocated once per blob and reused across levels.
"""

import cv2
import numpy as np

from .merging import TopMerger
from .tops import Tops


AREA = cv2.CC_STAT_AREA
LEFT, TOP = cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP
WIDTH, HEIGHT = cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT


class ConCompDetector(object):

    def __init__(self,
                 windowSizeM=40.0, windowOverlap=0.2,
                 lowerPercentile=20, upperPercentile=99,
                 erosionKernelSize=3, erosionIterations=1,
                 minTreeAreaM2=0.5, minTopAreaM2=0.25, topStepM=0.25,
                 merger=None, refine=True, verbose=True):
        self.windowSizeM = float(windowSizeM)
        self.windowOverlap = float(windowOverlap)
        self.lowerPercentile = int(lowerPercentile)
        self.upperPercentile = int(upperPercentile)
        self.erosionKernelSize = int(erosionKernelSize)
        self.erosionIterations = int(erosionIterations)
        self.minTreeAreaM2 = float(minTreeAreaM2)
        self.minTopAreaM2 = float(minTopAreaM2)
        self.topStepM = float(topStepM)
        self.merger = merger or TopMerger()
        self.refine = bool(refine)
        self.verbose = verbose
        self._kernel = np.ones((self.erosionKernelSize,
                                self.erosionKernelSize), np.uint8)

    def __repr__(self):
        return ("ConCompDetector(pct=%d, minTop=%.2f, step=%.2f, erode=%d, %r)"
                % (self.lowerPercentile, self.minTopAreaM2, self.topStepM,
                   self.erosionIterations, self.merger))

    # ------------------------------------------------------------------ #

    def detect(self, scene):
        """Run over a Scene and return Tops."""
        geometry = self._pixelParameters(scene)
        seeds = []
        windows = 0
        for column, row, window in self._windows(scene, geometry):
            windows += 1
            if np.count_nonzero(window) <= geometry["minPixTree"]:
                continue
            for localRow, localColumn in self._topsInWindow(scene, window,
                                                            geometry):
                seeds.append((column + localColumn, row + localRow))

        if self.verbose:
            print("[detect] %d windows -> %d raw seeds" % (windows, len(seeds)))

        if self.refine and len(seeds) >= 2:
            seeds = self.merger.merge(seeds, scene)
            if self.verbose:
                print("[detect] after merging: %d seeds" % len(seeds))

        return Tops([(int(x), int(y), float(scene.chm[int(y), int(x)]))
                     for x, y in seeds])

    # ------------------------------------------------------------------ #

    def _pixelParameters(self, scene):
        area = scene.pixelSize ** 2
        return {
            "window": scene.metresToPixels(self.windowSizeM, 8),
            "minPixTree": max(1, int(round(self.minTreeAreaM2 / area))),
            "minPixTop": max(1, int(round(self.minTopAreaM2 / area))),
        }

    def _windows(self, scene, geometry):
        size = geometry["window"]
        step = max(1, int(size * (1.0 - self.windowOverlap)))
        for row in range(0, scene.chm.shape[0], step):
            for column in range(0, scene.chm.shape[1], step):
                yield (column, row,
                       scene.chm[row:row + size, column:column + size])

    def _stretch(self, window):
        """
        Percentile-stretch a window to 0..255 and erode it.

        The grayscale values are kept rather than thresholded away: the descent
        needs the height ordering inside each blob. Returns the stretched image
        and how many metres one level of it is worth.
        """
        values = window[window > 0]
        if values.size == 0:
            return None
        low = np.percentile(values, self.lowerPercentile)
        high = np.percentile(values, self.upperPercentile)
        if high <= low:
            return None

        stretched = np.clip(window, None, high).astype(np.float32) - low
        stretched[stretched < 0] = 0
        stretched *= 255.0 / (high - low)
        stretched[window <= 0] = 0
        eroded = cv2.erode(stretched.astype(np.uint8), self._kernel,
                           iterations=self.erosionIterations)
        return eroded, (high - low) / 255.0

    # ------------------------------------------------------------------ #

    def _topsInWindow(self, scene, window, geometry):
        """Tops found in one window, as (row, column) local to it."""
        result = self._stretch(window)
        if result is None:
            return []
        stretched, metresPerLevel = result

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            stretched, connectivity=8)

        tops = []
        for label in range(1, count):
            if stats[label, AREA] <= geometry["minPixTree"]:
                continue
            x0, y0 = stats[label, LEFT], stats[label, TOP]
            x1 = x0 + stats[label, WIDTH]
            y1 = y0 + stats[label, HEIGHT]

            blob = stretched[y0:y1, x0:x1].copy()
            blob[labels[y0:y1, x0:x1] != label] = 0
            for row, column in self._descend(
                    scene, blob, window[y0:y1, x0:x1], metresPerLevel,
                    geometry):
                tops.append((row + y0, column + x0))
        return tops

    def _descend(self, scene, blob, heights, metresPerLevel, geometry):
        if self.merger.metric == "prominence":
            return self._descendProminence(blob, metresPerLevel, geometry)
        return self._descendSaddle(scene, blob, heights, metresPerLevel,
                                   geometry)

    def _descendProminence(self, blob, metresPerLevel, geometry):
        """
        The descent with the elder rule. Every surviving top belongs to a
        cluster whose representative is its highest top. When a level joins
        components holding different clusters, the cluster with the highest
        representative survives; every other cluster's representative dies
        there, with prominence = its height - this level, and is removed if
        that is below the merger's drop; its cluster joins the elder's either
        way, so its other tops stay as they were decided. A chain of branch
        tops cannot survive through a removed one: the components join where
        the canopy joins them, whatever was removed before.

        The join happened between the previous level and this one, so the
        prominence is measured at most one step high, never low.
        """
        values = blob[blob > 0]
        if values.size == 0:
            return []
        highest, lowest = float(values.max()), float(values.min())
        if highest <= lowest:
            return []
        step = max(1.0, self.topStepM / max(metresPerLevel, 1e-6))
        threshold = self.merger.saddleDropM / max(metresPerLevel, 1e-6)
        above = np.empty(blob.shape, bool)
        tops = []          # [height, (row, column), cluster]
        clusters = 0
        level = highest - step
        while level > lowest:
            np.greater_equal(blob, level, out=above)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                above.view(np.uint8), connectivity=8)
            byLabel = {}
            for top in tops:
                byLabel.setdefault(int(labels[top[1]]), []).append(top)
            removed = set()
            for label, members in byLabel.items():
                elder = max(members, key=lambda t: t[0])[2]
                for cluster in {t[2] for t in members} - {elder}:
                    inCluster = [t for t in members if t[2] == cluster]
                    representative = max(inCluster, key=lambda t: t[0])
                    if representative[0] - level < threshold:
                        removed.add(id(representative))
                    for t in inCluster:
                        t[2] = elder
                peak, position = self._peakIn(blob, labels, stats, label)
                if peak > max(t[0] for t in members):
                    tops.append([peak, position, elder])
            tops = [t for t in tops if id(t) not in removed]
            for label in range(1, count):
                if label in byLabel or stats[label, AREA] <= geometry["minPixTop"]:
                    continue
                peak, position = self._peakIn(blob, labels, stats, label)
                tops.append([peak, position, clusters])
                clusters += 1
            level -= step
        return [t[1] for t in tops]

    def _descendSaddle(self, scene, blob, heights, metresPerLevel, geometry):
        """
        Walk one blob down in height steps, registering each new sub-blob.

        `blob` and `heights` are already cropped to the blob's bounding box,
        and positions returned are relative to that crop.
        """
        values = blob[blob > 0]
        if values.size == 0:
            return []
        highest, lowest = float(values.max()), float(values.min())
        if highest <= lowest:
            return []

        step = max(1.0, self.topStepM / max(metresPerLevel, 1e-6))
        above = np.empty(blob.shape, bool)
        tops = []
        level = highest - step

        while level > lowest:
            np.greater_equal(blob, level, out=above)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                above.view(np.uint8), connectivity=8)
            if count > 1:
                grouped = self._groupByLabel(tops, blob, labels)
                self._addTrueMaxima(grouped, blob, labels, stats)
                grouped = self.merger.refineWithinBlob(
                    grouped, scene, metresPerLevel, heights)
                tops = [position for entries in grouped.values()
                        for _, position in entries]
                tops.extend(self._newBlobs(blob, labels, stats, count,
                                           grouped, geometry["minPixTop"]))
            level -= step
        return tops

    # ------------------------------------------------------------------ #

    @staticmethod
    def _groupByLabel(tops, blob, labels):
        grouped = {}
        for row, column in tops:
            grouped.setdefault(int(labels[row, column]), []).append(
                (float(blob[row, column]), (row, column)))
        return grouped

    @staticmethod
    def _peakIn(blob, labels, stats, label):
        """
        The highest pixel of one sub-blob, searched only inside its own box.

        argmax on the box slice, not argwhere: argwhere materialises every
        coordinate, and all that is wanted here is one.
        """
        x0, y0 = stats[label, LEFT], stats[label, TOP]
        x1, y1 = x0 + stats[label, WIDTH], y0 + stats[label, HEIGHT]
        region = np.where(labels[y0:y1, x0:x1] == label,
                          blob[y0:y1, x0:x1], 0)
        row, column = np.unravel_index(int(region.argmax()), region.shape)
        return float(region[row, column]), (int(row + y0), int(column + x0))

    def _addTrueMaxima(self, grouped, blob, labels, stats):
        """
        If a sub-blob's highest pixel is above every top it already holds, that
        pixel has to be registered too, or the descent can keep a shoulder and
        lose the actual apex.
        """
        for label in list(grouped):
            if label == 0:
                continue
            peak, position = self._peakIn(blob, labels, stats, label)
            if peak > max(height for height, _ in grouped[label]):
                grouped[label].append((peak, position))

    def _newBlobs(self, blob, labels, stats, count, grouped, minPixTop):
        fresh = []
        for label in range(1, count):
            if label in grouped or stats[label, AREA] <= minPixTop:
                continue
            fresh.append(self._peakIn(blob, labels, stats, label)[1])
        return fresh
