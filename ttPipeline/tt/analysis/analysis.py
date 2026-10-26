"""
Analysis — what a site is like, and what a detector made of it.

The same measures serve both: topStats describes any set of tops (heights,
prominence, spacing, density, the share above given heights) whether they are
the annotated crowns' apexes (exploration: what kind of forest is this) or a
detector's output (what kind of tops does it find). errors explains a
detector's result against the crowns: which crowns it missed and how they
differ from the found ones, and what its false positives are (repeats inside
a found crown, detections at canopy level that are probably undrawn trees,
and detections below the canopy). treeByTree compares methods crown by crown.
"""

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Point
from shapely.ops import unary_union

from ..data.annotations import crownTops
from ..data.detections import Detections
from ..detectors.fusion import SaddleSurface
from ..scoring.evaluator import Evaluator
from ..scoring.reference import AnnotatedCrowns, CrownFrame
from .statistics import (nearestDistances, profile, quantiles,
                         splitBackground)
from .trees import TreeTable

HEIGHT_CLASSES_M = (5.0, 10.0, 20.0)
NEAR_M = 0.5               # an outside top this close to a crown is "near"
SHIFT_M = 1.0              # the offset scan's reach and step
SHIFT_STEP_M = 0.1
TOLERANCES_M = (0.25, 0.5)


class Analysis(object):

    def __init__(self, data, surface=None, reference=None, objective=None):
        """
        data: a TDData; surface: the HeightSurface heights and prominence are
        read from (the site's CHM at the default settings when not given);
        reference: what results are scored against (its annotated crowns).
        """
        self.data = data
        # False: no surface (only the scoring measures are wanted)
        self.surface = data.heightSurface() if surface is None \
            else (None if surface is False else surface)
        self.reference = reference or AnnotatedCrowns(data.annotations)
        self.evaluator = Evaluator(self.reference, objective)
        self._saddle = None
        self._crowns = None

    @property
    def saddle(self):
        if self._saddle is None:
            self._saddle = SaddleSurface.fromSurface(self.surface)
        return self._saddle

    # ------------------------------------------------------------------ #
    # the crowns
    # ------------------------------------------------------------------ #

    def crownMeasures(self):
        """Per crown: its apex (x, y), peak height, area and spacing."""
        if self._crowns is None:
            crowns = self.reference.crowns()
            tops = crownTops(crowns, self.surface)
            centroids = crowns.geometry.centroid
            self._crowns = {
                "x": tops[:, 0], "y": tops[:, 1],
                "peak": np.nan_to_num(tops[:, 2], nan=0.0),
                "area": crowns.geometry.area.to_numpy(),
                "nearest": nearestDistances(centroids.x.to_numpy(),
                                            centroids.y.to_numpy())}
        return self._crowns

    def crownTops(self):
        """The crowns' apexes as Detections (crowns with no canopy left out)."""
        m = self.crownMeasures()
        keep = np.isfinite(m["x"])
        return Detections(m["x"][keep], m["y"][keep], m["peak"][keep],
                          heights=m["peak"][keep], crs=self.surface.crs)

    # ------------------------------------------------------------------ #
    # exploration
    # ------------------------------------------------------------------ #

    def topStats(self, tops, region=None, heightClasses=HEIGHT_CLASSES_M):
        """
        Any set of tops described: count, density per hectare (over region,
        or the site's area), heights, prominence and spacing as 10/50/90th
        percentiles, and the share of tops above each height class.
        """
        region = region if region is not None else self.data.area
        inside = tops.inRegion(region) if region is not None else tops
        heights = (inside.heights if inside.heights is not None
                   else self.surface.heightAt(inside.x, inside.y))
        areaHa = region.area / 1e4 if region is not None else np.nan
        return {"count": len(inside),
                "perHectare": len(inside) / areaHa if areaHa else np.nan,
                "heights": quantiles(heights),
                "prominence": quantiles(self.saddle.prominences(inside)),
                "nearest": quantiles(nearestDistances(inside.x, inside.y)),
                "shareAbove": {"%g" % h: float(np.mean(heights >= h))
                               if len(heights) else 0.0
                               for h in heightClasses}}

    def siteStats(self, region=None):
        """The site's crowns: how many, how big, and their tops."""
        region = region if region is not None else self.data.area
        crowns = self.reference.crownsIn(region)
        return {"crowns": len(crowns),
                "areaHa": region.area / 1e4,
                "crownArea": quantiles(crowns.geometry.area.to_numpy()),
                "tops": self.topStats(self.crownTops(), region)}

    # ------------------------------------------------------------------ #
    # a result
    # ------------------------------------------------------------------ #

    def errors(self, detections, region=None, neighbourhood=12.0,
               canopyFraction=0.7):
        """
        A result explained: scores; where the detections went (hits, repeats,
        and false positives split by height into canopy-level, below canopy
        and with no found tree near); and how the missed crowns differ from
        the found ones in height, area and crowding.
        """
        matching = self.evaluator.match(detections, region)
        kept = matching.detections
        heights = (kept.heights if kept.heights is not None
                   else self.surface.heightAt(kept.x, kept.y))
        split = splitBackground(kept.x, kept.y, heights, matching.kinds,
                                neighbourhood, canopyFraction)
        crowns = matching.crowns
        found = np.zeros(len(crowns), bool)
        found[list(matching.hitCrowns)] = True
        peaks = np.nan_to_num(crownTops(crowns, self.surface)[:, 2], nan=0.0)
        nearest = nearestDistances(crowns.geometry.centroid.x.to_numpy(),
                                   crowns.geometry.centroid.y.to_numpy())
        area = crowns.geometry.area.to_numpy()
        total = max(len(kept), 1)
        return dict(matching.scores(),
                    backgroundCanopy=len(split["canopy"]),
                    backgroundLow=len(split["low"]),
                    backgroundUnknown=len(split["unknown"]),
                    precisionIfCanopyAreTrees=(matching.count("hit")
                                               + len(split["canopy"]))
                    / float(total),
                    missed=int((~found).sum()),
                    foundCrowns=profile(found, area, peaks, nearest),
                    missedCrowns=profile(~found, area, peaks, nearest))

    def belowWindowCut(self, detector):
        """
        Which crowns have their apex under their own window's percentile cut:
        a crown below the cut cannot be found at any setting of the other CC
        parameters, so it is a thresholding failure, not a detection one.
        """
        p = detector.params
        surface = self.surface
        window = surface.metresToPixels(p["windowSizeM"], 8)
        step = max(1, int(window * (1.0 - p["windowOverlap"])))
        rows, columns = surface.chm.shape
        crowns = self.reference.crowns()
        cut = np.full(len(crowns), np.nan)
        for index, geometry in enumerate(crowns.geometry):
            column, row = surface.toPixel(geometry.centroid.x,
                                          geometry.centroid.y)
            c0 = int(np.clip((column // step) * step, 0,
                             max(0, columns - window)))
            r0 = int(np.clip((row // step) * step, 0, max(0, rows - window)))
            values = surface.chm[r0:r0 + window, c0:c0 + window]
            values = values[values > 0]
            if values.size:
                cut[index] = np.percentile(values, p["lowerPercentile"])
        return self.crownMeasures()["peak"] < cut

    def outsideDetail(self, detections, region):
        """Heights of hits and outside detections, and how many outside ones
        lie within NEAR_M of a drawn crown."""
        matching = self.evaluator.match(detections, region)
        kept, kinds = matching.detections, np.array(matching.kinds)
        h = kept.heights if kept.heights is not None else kept.scores
        out = kinds == "background"
        drawn = unary_union(list(matching.crowns.geometry))
        near = [drawn.distance(Point(x, y)) <= NEAR_M
                for x, y in zip(kept.x[out], kept.y[out])]
        return {"hitHeights": quantiles(h[kinds == "hit"]),
                "outsideHeights": quantiles(h[out]),
                "outsideNear": float(np.mean(near)) if near else 0.0}

    def offsetScan(self, detections, region):
        """
        F1 for every shift of the crowns up to SHIFT_M: a misalignment between
        the height model and the imagery the crowns were drawn on shows as a
        clear gain at one shift. The centre of the shifts reaching the best is
        the estimate.
        """
        crowns = self.reference.crowns()
        steps = np.round(np.arange(-SHIFT_M, SHIFT_M + 1e-9, SHIFT_STEP_M), 3)
        scores = {}
        for dx in steps:
            for dy in steps:
                moved = crowns.copy()
                moved["geometry"] = [translate(g, dx, dy)
                                     for g in crowns.geometry]
                scores[(float(dx), float(dy))] = Evaluator(
                    CrownFrame(moved), self.evaluator.objective).score(
                        detections, region)["f1"]
        top = max(scores.values())
        reaching = np.array([k for k, v in scores.items() if v >= top - 1e-9])
        centre = [round(float(v), 2) + 0.0 for v in reaching.mean(axis=0)]
        return {"bestShiftM": centre, "bestF1": top,
                "unshiftedF1": scores[(0.0, 0.0)],
                "gain": top - scores[(0.0, 0.0)],
                "shiftsAtBest": int(len(reaching))}

    def tolerance(self, detections, region):
        """
        Scores with the crowns grown by each of TOLERANCES_M: a method whose
        detections sit at crown edges gains far more than one at centres.
        """
        crowns = self.reference.crowns()
        out = {}
        for grow in TOLERANCES_M:
            grown = crowns.copy()
            grown["geometry"] = crowns.geometry.buffer(grow)
            r = Evaluator(CrownFrame(grown), self.evaluator.objective).score(
                detections, region)
            out["%.2f" % grow] = {"recall": r["recall"],
                                  "precision": r["precision"], "f1": r["f1"]}
        return out

    # ------------------------------------------------------------------ #

    def treeByTree(self, runs, blocks=None):
        """
        A TreeTable of {name: RunSet}: each run's held-out detections,
        stitched over the blocks ({name: geometry}, the site's by default).
        """
        blocks = blocks or {b.name: b.geometry for b in self.data.blocks}
        return TreeTable(self.reference.crowns(),
                         self.crownMeasures()["peak"], self.surface,
                         {name: run.heldOut(blocks)
                          for name, run in runs.items()})

