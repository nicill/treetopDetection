"""
The core on a clean synthetic site: the height surface, tops output, the
merge rule, the CC detector and the scorer, and properties found broken
during the refactors, pinned so they stay fixed.
"""

import os
import unittest

import numpy as np

from .base import SiteCase


class TestHeightSurface(SiteCase):

    def testLoadsOnOneGrid(self):
        surface = self.site().heightSurface()
        self.assertEqual(surface.shape, (240, 240))
        self.assertAlmostEqual(surface.pixelSize, 0.25, places=6)
        self.assertEqual(str(surface.crs), "EPSG:32648")
        self.assertEqual(len(self.site().crowns), 9)

    def testBelowMinHeightIsZeroed(self):
        surface = self.site().heightSurface(minHeight=8.0)
        canopy = surface.chm[surface.chm > 0]
        self.assertTrue((canopy >= 8.0).all())

    def testRoundTripThroughWorldCoordinates(self):
        surface = self.site().heightSurface()
        column, row = surface.toPixel(*surface.toWorld(100, 50))
        self.assertAlmostEqual(column, 100.5, places=3)
        self.assertAlmostEqual(row, 50.5, places=3)

    def testCrownTopsSitOnTheApexes(self):
        from tt.data.annotations import crownTops
        site = self.site()
        tops = crownTops(site.crowns, site.heightSurface())
        built = sorted(h for *_, h in self.trees)
        for expected, found in zip(built, sorted(tops[:, 2])):
            self.assertAlmostEqual(expected, found, delta=0.15)

    def testCrownShiftMovesThePolygons(self):
        plain = self.site().crowns.geometry.iloc[0].centroid.x
        shifted = self.site(crownShiftEastM=2.0).crowns.geometry.iloc[0] \
            .centroid.x
        self.assertAlmostEqual(shifted - plain, -2.0, places=3,
                               msg="a crown reported 2 m east of the CHM "
                                   "must move 2 m west to correct it")


class TestTopsOutput(SiteCase):

    def testListAndMask(self):
        import cv2
        from tt.data import Detections
        from tt.data.detections import writeList, writeMask
        surface = self.site().heightSurface()
        world = [surface.toWorld(20, 20), surface.toWorld(60, 60)]
        detections = Detections([w[0] for w in world], [w[1] for w in world],
                                [10.0, 12.0], heights=[10.0, 12.0])
        lines = open(writeList(detections, os.path.join(
            self.directory, "tops.txt"), surface)).read().splitlines()
        self.assertEqual(lines[1].split()[:3], ["20", "20", "10.000"])
        path = writeMask(detections, os.path.join(self.directory, "m.png"),
                         surface, radius=3)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        count, _ = cv2.connectedComponents((mask == 0).astype(np.uint8))
        self.assertEqual(count - 1, 2)


class TestMerger(unittest.TestCase):

    def testMetricsOrderAsExpected(self):
        from tt.detectors import TopMerger
        self.assertAlmostEqual(TopMerger("2d", epsM=1.0).distance(0.7, 3.0),
                               0.7)
        self.assertAlmostEqual(TopMerger("3d", epsM=1.0).distance(0.7, 3.0),
                               np.hypot(0.7, 3.0))
        mixed = TopMerger("composite", epsM=1.0, heightWeight=0.7)
        self.assertAlmostEqual(mixed.distance(0.7, 3.0), 0.7 * 3.0 + 0.3 * 0.7)

    def testCompositeReachIsWiderThanEpsilon(self):
        """The trap: a weighted sum lets tops merge far beyond eps."""
        from tt.detectors import TopMerger
        merger = TopMerger("composite", epsM=1.0, heightWeight=0.7)
        self.assertAlmostEqual(merger.horizontalReachM, 1.0 / 0.3, places=6)
        self.assertTrue(merger.areSame(3.0, 0.0))

    def testHeightSeparatesWhatDistanceWouldMerge(self):
        from tt.detectors import TopMerger
        merger = TopMerger("composite", epsM=1.0, heightWeight=0.7)
        self.assertTrue(merger.areSame(0.7, 0.0))
        self.assertFalse(merger.areSame(0.7, 3.0))

    def testSaddleDropFindsTheValley(self):
        from tt.detectors import saddleDrop
        surface = np.zeros((10, 40), np.float32)
        surface[5, :] = 20.0
        surface[5, 18:22] = 14.0          # a 6 m notch between two peaks
        self.assertAlmostEqual(saddleDrop(surface, 5, 5, 5, 35), 6.0, places=3)
        self.assertAlmostEqual(saddleDrop(surface, 5, 5, 5, 10), 0.0, places=3)

    def testSaddleRefusesAcrossAValley(self):
        from tt.detectors import TopMerger
        surface = np.zeros((10, 40), np.float32)
        surface[5, :] = 20.0
        surface[5, 18:22] = 14.0
        merger = TopMerger("saddle", epsM=100.0, saddleDropM=1.0)
        self.assertFalse(merger.areSame(7.5, 0.0, surface=surface,
                                        a=(5, 5), b=(5, 35)))
        self.assertTrue(merger.areSame(1.25, 0.0, surface=surface,
                                       a=(5, 5), b=(5, 10)))


SETTINGS = dict(lowerPercentile=10, minTopAreaM2=0.12, topStepM=0.12,
                erosionIterations=1, minTreeAreaM2=0.5, windowSizeM=30.0,
                saddleDropM=0.5)


class TestCCDetector(SiteCase):

    def testFindsEveryTreeOnACleanSite(self):
        from tt.detectors import CCDetector
        from tt.scoring import AnnotatedCrowns, Evaluator
        site = self.site()
        result = Evaluator(AnnotatedCrowns(site.annotations)).score(
            CCDetector(**SETTINGS).detect(site))
        self.assertEqual(result["recall"], 1.0)
        self.assertGreaterEqual(result["precision"], 0.5)

    def testMergingNeverAddsDetections(self):
        from tt.detectors import CCDetector
        site = self.site()
        raw = CCDetector(refine=False, **SETTINGS).detect(site)
        merged = CCDetector(**SETTINGS).detect(site)
        self.assertLessEqual(len(merged), len(raw))

    def testDetectionsCarryTheirHeight(self):
        from tt.detectors import CCDetector
        site = self.site()
        detections = CCDetector(windowSizeM=30.0).detect(site)
        np.testing.assert_allclose(
            detections.heights,
            site.heightSurface().heightAt(detections.x, detections.y))


class TestScorer(SiteCase):

    def evaluator(self):
        from tt.scoring import AnnotatedCrowns, Evaluator
        return Evaluator(AnnotatedCrowns(self.site().annotations))

    def apexes(self):
        from tt.data.annotations import crownTops
        return crownTops(self.site().crowns, self.site().heightSurface())

    def testHitRepeatAndBackgroundAreDistinguished(self):
        from tt.data import Detections
        x, y, _ = self.apexes()[0]
        # on a crown apex, beside it in the same crown, and far away
        detections = Detections([x, x + 0.5, x + 25], [y, y - 0.5, y],
                                [20.0, 10.0, 3.0])
        self.assertEqual(self.evaluator().match(detections).kinds,
                         ["hit", "repeat", "background"])

    def testPerfectDetectionScoresOne(self):
        from tt.data import Detections
        tops = self.apexes()
        result = self.evaluator().score(Detections(tops[:, 0], tops[:, 1],
                                                   tops[:, 2]))
        self.assertEqual((result["recall"], result["precision"]), (1.0, 1.0))

    def testBackgroundSplitsByLocalCanopyHeight(self):
        from tt.analysis import splitBackground
        from tt.data import Detections
        tops = self.apexes()
        xs = np.r_[tops[:, 0], tops[0, 0] + 6.0, tops[0, 0] + 7.0]
        ys = np.r_[tops[:, 1], tops[0, 1], tops[0, 1]]
        heights = np.r_[tops[:, 2], 14.0, 2.5]
        kinds = self.evaluator().match(Detections(xs, ys, heights)).kinds
        split = splitBackground(xs, ys, heights, kinds)
        self.assertIn(len(xs) - 2, split["canopy"])
        self.assertIn(len(xs) - 1, split["low"])


class TestInvariants(SiteCase):

    def testSaddleDropIsTranslationInvariant(self):
        """np.rint rounds half to even, so rounding absolute coordinates made
        the verdict depend on where a window started."""
        from tt.detectors import saddleDrop
        rng = np.random.default_rng(3)
        surface = rng.random((200, 200)).astype(np.float32)
        for _ in range(2000):
            a, b = rng.integers(40, 160, 2), rng.integers(40, 160, 2)
            dy, dx = rng.integers(0, 40, 2)
            self.assertEqual(
                saddleDrop(surface, a[0], a[1], b[0], b[1]),
                saddleDrop(surface[dy:, dx:], a[0] - dy, a[1] - dx,
                           b[0] - dy, b[1] - dx))

    def testCroppedDescentMatchesFullWindow(self):
        """CC crops each blob to its box before descending; that must change
        nothing."""
        import cv2
        from tt.detectors import CCDetector
        surface = self.site().heightSurface()
        algorithm = CCDetector(windowSizeM=30.0).algorithm()
        geometry = algorithm._pixelParameters(surface)
        for column, row, window in algorithm._windows(surface, geometry):
            result = algorithm._stretch(window)
            if result is None:
                continue
            stretched, perLevel = result
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                stretched, connectivity=8)
            for label in range(1, count):
                if stats[label, 4] <= geometry["minPixTree"]:
                    continue
                x0, y0, w, h = stats[label, :4]
                full = stretched.copy()
                full[labels != label] = 0
                whole = sorted(algorithm._descend(surface, full, window,
                                                  perLevel, geometry))
                cropped = sorted(
                    (r + y0, c + x0) for r, c in algorithm._descend(
                        surface, full[y0:y0 + h, x0:x0 + w],
                        window[y0:y0 + h, x0:x0 + w], perLevel, geometry))
                self.assertEqual(whole, cropped)

    def testThresholdCurveMatchesRescoring(self):
        """One spatial join answers every threshold; it must agree with
        re-scoring at each one."""
        from tt.data import Detections
        from tt.data.annotations import crownTops
        from tt.scoring import AnnotatedCrowns, Evaluator
        from tt.scoring.evaluator import THRESHOLD_CANDIDATES
        site = self.site()
        rng = np.random.default_rng(5)
        tops = crownTops(site.crowns, site.heightSurface())
        xs = np.repeat(tops[:, 0], 3) + rng.uniform(-1.5, 1.5, 3 * len(tops))
        ys = np.repeat(tops[:, 1], 3) + rng.uniform(-1.5, 1.5, 3 * len(tops))
        detections = Detections(xs, ys, rng.random(len(xs)))
        evaluator = Evaluator(AnnotatedCrowns(site.annotations))
        for threshold, value in evaluator.thresholdCurve(
                detections, site.area, THRESHOLD_CANDIDATES):
            self.assertAlmostEqual(value, evaluator.objectiveOf(
                evaluator.score(detections.aboveScore(threshold),
                                site.area)), places=12)

    def testStitchingKeepsOutlines(self):
        from affine import Affine
        from tt.data.patches import tileToWorld
        transform = Affine(0.1, 0, 500000, 0, -0.1, 5000000)
        world = tileToWorld([{"box": [10, 20, 30, 40], "score": 0.9,
                              "polygon": [[10, 20], [30, 20], [30, 40]]}],
                            transform, 100, 200)[0]
        self.assertAlmostEqual(world["polygon"][0][0], world["box"][0])
        self.assertAlmostEqual(world["polygon"][0][1], world["box"][3])


if __name__ == "__main__":
    unittest.main()
