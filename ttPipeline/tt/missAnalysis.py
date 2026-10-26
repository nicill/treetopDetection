"""
MissAnalysis — what the detector gets wrong, and why.

Answers two questions at one operating point. Which crowns are missed, compared
against the found ones on height, area, crowding and whether their apex even
survives the percentile cut. And what the false positives are: repeats inside a
crown already found, detections on unannotated trees, and detections on nothing.

The three kinds of error want different fixes, which is the reason for
separating them rather than reporting one precision figure.
"""

import json
import os

import numpy as np
from scipy.spatial import cKDTree

from .render import TileRenderer


class MissAnalysis(object):

    def __init__(self, scene, tops, evaluator, classification=None,
                 neighbourhood=12.0, canopyFraction=0.7):
        self.scene = scene
        self.tops = tops
        self.evaluator = evaluator
        self.classification = classification or evaluator.classify(tops)
        self.split = evaluator.splitBackground(tops, self.classification,
                                               neighbourhood, canopyFraction)
        self.canopyFraction = canopyFraction
        self._peaks = None
        self._nn = None

    # ------------------------------------------------------------------ #

    @property
    def crownPeaks(self):
        if self._peaks is None:
            _, self._peaks = self.scene.crownPeaks()
        return self._peaks

    @property
    def nearestNeighbour(self):
        if self._nn is None:
            centroids = np.array([[g.centroid.x, g.centroid.y]
                                  for g in self.scene.crowns.geometry])
            self._nn = cKDTree(centroids).query(centroids, k=2)[0][:, 1]
        return self._nn

    def foundMask(self):
        return np.array([i in self.classification.hitCrowns
                         for i in range(len(self.scene.crowns))])

    def belowWindowCut(self, detector):
        """
        Which crowns have their apex under their own window's percentile cut.

        A crown below the cut cannot be found at any setting of the other
        parameters, so it is a thresholding failure rather than a detection
        one, and the two want opposite changes.
        """
        window = self.scene.metresToPixels(detector.windowSizeM, 8)
        step = max(1, int(window * (1.0 - detector.windowOverlap)))
        rows, columns = self.scene.chm.shape

        cut = np.full(len(self.scene.crowns), np.nan)
        for index, geometry in enumerate(self.scene.crowns.geometry):
            column, row = self.scene.toPixel(geometry.centroid.x,
                                             geometry.centroid.y)
            c0 = int(np.clip((column // step) * step, 0, max(0, columns - window)))
            r0 = int(np.clip((row // step) * step, 0, max(0, rows - window)))
            values = self.scene.chm[r0:r0 + window, c0:c0 + window]
            values = values[values > 0]
            if values.size:
                cut[index] = np.percentile(values, detector.lowerPercentile)
        return self.crownPeaks < cut

    # ------------------------------------------------------------------ #

    def summary(self, detector=None):
        found = self.foundMask()
        missed = ~found
        total = max(len(self.tops), 1)

        result = {
            "crowns": len(self.scene.crowns),
            "detections": len(self.tops),
            "recall": len(self.classification.hitCrowns)
            / float(max(len(self.scene.crowns), 1)),
            "hits": self.classification.count("hit"),
            "repeats": self.classification.count("repeat"),
            "backgroundCanopy": len(self.split["canopy"]),
            "backgroundLow": len(self.split["low"]),
            "backgroundUnknown": len(self.split["unknown"]),
            "missed": int(missed.sum()),
        }
        result["precision"] = result["hits"] / float(total)
        result["precisionIfCanopyAreTrees"] = \
            (result["hits"] + result["backgroundCanopy"]) / float(total)

        for name, values in (("Height", self.crownPeaks),
                             ("Area", self.scene.crowns.geometry.area.to_numpy()),
                             ("NearestNeighbour", self.nearestNeighbour)):
            result["found%sMedian" % name] = float(np.median(values[found])) \
                if found.any() else None
            result["missed%sMedian" % name] = float(np.median(values[missed])) \
                if missed.any() else None

        if detector is not None and missed.any():
            below = self.belowWindowCut(detector)
            result["missedBelowWindowCut"] = int((below & missed).sum())
        return result

    def report(self, detector=None):
        """Print the summary in the form that is actually readable."""
        summary = self.summary(detector)
        found, missed = self.foundMask(), ~self.foundMask()
        total = max(len(self.tops), 1)

        print("=" * 70)
        print("%d crowns, %d detections | recall %.1f%% | precision %.1f%%"
              % (summary["crowns"], summary["detections"],
                 100 * summary["recall"], 100 * summary["precision"]))
        print("=" * 70)
        print("WHERE THE DETECTIONS GO")
        for label, key, note in (
                ("in a crown, primary", "hits", ""),
                ("in a crown, repeat", "repeats", "over-segmented trees"),
                ("outside, canopy-level", "backgroundCanopy",
                 "probably unannotated trees"),
                ("outside, below canopy", "backgroundLow",
                 "the real false positives"),
                ("outside, no neighbours", "backgroundUnknown", "")):
            count = summary[key]
            if count or key in ("hits", "repeats"):
                print("  %-24s %5d  %5.1f%%   %s"
                      % (label, count, 100.0 * count / total, note))

        print()
        print("MISSED CROWNS: %d of %d (%.1f%%)"
              % (summary["missed"], summary["crowns"],
                 100.0 * summary["missed"] / max(summary["crowns"], 1)))
        for name, values, unit in (
                ("height", self.crownPeaks, "m"),
                ("area", self.scene.crowns.geometry.area.to_numpy(), "m2"),
                ("nn distance", self.nearestNeighbour, "m")):
            print("  %-14s found %7.2f %-3s   missed %7.2f %s"
                  % (name, np.median(values[found]), unit,
                     np.median(values[missed]) if missed.any() else float("nan"),
                     unit))
        if "missedBelowWindowCut" in summary:
            print("  %d of the missed have their apex below their window's "
                  "percentile cut" % summary["missedBelowWindowCut"])

        print()
        print("  precision as scored                          %.1f%%"
              % (100 * summary["precision"]))
        print("  precision if canopy-level ones are real      %.1f%%"
              % (100 * summary["precisionIfCanopyAreTrees"]))
        return summary

    # ------------------------------------------------------------------ #

    def kindsForRendering(self):
        """Per-detection labels with the background split into two colours."""
        canopy, low = set(self.split["canopy"]), set(self.split["low"])
        kinds = []
        for index, kind in enumerate(self.classification.kinds):
            if kind == "background":
                kind = "canopy" if index in canopy else (
                    "low" if index in low else None)
            kinds.append(kind)
        return kinds

    def writeOutputs(self, outputDir, detector=None, images=True):

        os.makedirs(outputDir, exist_ok=True)
        with open(os.path.join(outputDir, "analysis.json"), "w") as handle:
            json.dump(self.summary(detector), handle, indent=2, default=float)

        missed = self.evaluator.missedCrowns(self.classification)
        if missed:
            frame = self.scene.crowns.iloc[missed].copy()
            frame["peak"] = self.crownPeaks[missed]
            frame["nnDist"] = self.nearestNeighbour[missed]
            frame.to_file(os.path.join(outputDir, "missedCrowns.shp"))

        if images:
            renderer = TileRenderer(self.scene)
            paths = renderer.writeTiles(
                os.path.join(outputDir, "images"), self.tops,
                self.kindsForRendering(), missedCrowns=missed,
                label="green hit | orange repeat | blue unannotated? | "
                      "red low | yellow missed")
            print("  wrote %d tile images" % len(paths))
        return outputDir
