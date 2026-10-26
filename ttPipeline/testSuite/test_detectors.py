"""The detector hierarchy and its postprocessors."""

import os
import unittest

import numpy as np

from .base import SiteCase


class TestHeightDetectors(SiteCase):

    def testCCFindsEveryTree(self):
        from tt.detectors import CCDetector
        detections = CCDetector(minHeight=1.0).detect(self.site())
        self.assertEqual(len(detections), len(self.trees))
        self.assertEqual(detections.geometry, "points")
        self.assertEqual(detections.detector, "concomp")
        np.testing.assert_array_equal(detections.scores, detections.heights)

    def testBaselines(self):
        from tt.detectors import LocalMaximaDetector, WatershedDetector
        for detector in (LocalMaximaDetector(minHeight=1.0, windowBaseM=2.0),
                         WatershedDetector(minHeight=1.0, markerWindowM=2.0,
                                           minCrownAreaM2=1.0)):
            self.assertEqual(len(detector.detect(self.site())),
                             len(self.trees), detector)

    def testUnknownParameterIsRefused(self):
        from tt.detectors import CCDetector
        with self.assertRaises(ValueError):
            CCDetector(windowSize=40)

    def testWithParamsKeepsTheRest(self):
        from tt.detectors import CCDetector
        detector = CCDetector(minHeight=1.0).withParams(saddleDropM=0.2)
        self.assertEqual((detector.params["minHeight"],
                          detector.params["saddleDropM"]), (1.0, 0.2))

    def testDefaultGrid(self):
        from tt.detectors import CCDetector
        grid = CCDetector.parameterGrid(saddleDropM=[0.5])
        self.assertEqual(len(grid), 8)
        self.assertEqual(grid.expand()[0]["lowerPercentile"], 20)

    def testRegionCutsTheOutput(self):
        from shapely.geometry import box
        from tt.detectors import CCDetector
        site = self.site()
        minX, minY, maxX, maxY = site.area.bounds
        half = box(minX, minY, (minX + maxX) / 2, maxY)
        everything = CCDetector(minHeight=1.0).detect(site)
        some = CCDetector(minHeight=1.0).detect(site, region=half)
        self.assertEqual(len(some), len(everything.inRegion(half)))
        self.assertLess(len(some), len(everything))


class TestPostProcessors(SiteCase):

    def testNmsMergesOverlappingBoxes(self):
        from tt.data import Detections
        from tt.detectors import NMS
        detections = Detections([1, 1.2, 9], [1, 1.1, 9], [0.9, 0.5, 0.8],
                                boxes=[[0, 0, 2, 2], [0.2, 0.1, 2.2, 2.1],
                                       [8, 8, 10, 10]], geometry="boxes")
        kept = NMS(0.4).apply(detections)
        self.assertEqual(list(kept.scores), [0.9, 0.8])

    def testDetectorAppliesItsPostprocessors(self):
        from tt.detectors import CCDetector, PostProcessor

        class TallOnly(PostProcessor):
            def apply(self, detections, data):
                return detections.subset(detections.heights > 15)

        detector = CCDetector(minHeight=1.0, postprocessors=[TallOnly()])
        tall = sum(1 for *_, h in self.trees if h > 15)
        self.assertEqual(len(detector.detect(self.site())), tall)
        self.assertEqual(len(detector.withParams(minHeight=2.0)
                             .postprocessors), 1)


class TestBaselineAlgorithms(unittest.TestCase):
    """Local maxima and marker watershed on hand-made height models."""

    @staticmethod
    def bumps(centres, heights, size=60, sigma=4.0):
        rows, cols = np.indices((size, size))
        chm = np.zeros((size, size))
        for (r, c), h in zip(centres, heights):
            chm = np.maximum(chm, h * np.exp(-((rows - r) ** 2 + (cols - c) ** 2)
                                             / (2 * sigma ** 2)))
        chm[chm < 0.5] = 0
        return chm

    def testVariableWindowGrowsWithHeight(self):
        from tt.detectors.height import localMaxima
        # two tops 3 m apart: separate in a 1 m window; one when the window
        # grows with height to 1 + 0.6 x 9 = 6.4 m (3.2 m either side)
        chm = self.bumps([(30, 24), (30, 36)], [10.0, 9.0])
        self.assertEqual(len(localMaxima(chm, 0.25, 1.0, 0.0)), 2)
        large = localMaxima(chm, 0.25, 1.0, 0.6)
        self.assertEqual([(x, y) for x, y, _ in large], [(24, 30)])

    def testPlateauGivesOneTop(self):
        from tt.detectors.height import localMaxima
        chm = np.zeros((20, 20))
        chm[8:12, 8:12] = 5.0
        self.assertEqual(len(localMaxima(chm, 0.25, 0.5, 0.0)), 1)

    def testWatershedDropsSmallCrowns(self):
        from tt.detectors.height import markerWatershed
        chm = self.bumps([(20, 20), (45, 45)], [8.0, 6.0])
        chm += self.bumps([(20, 45)], [3.0], sigma=1.0)       # a tiny one
        self.assertEqual(len(markerWatershed(chm, 0.25, 1.0, 0.0)), 3)
        big = markerWatershed(chm, 0.25, 1.0, 1.0)
        self.assertEqual(sorted((x, y) for x, y, _ in big),
                         [(20, 20), (45, 45)])


class TestProminence(unittest.TestCase):
    """A chain of branch pseudotops: the saddle test keeps the far one, the
    elder rule merges it through the branch."""

    def setUp(self):
        import rasterio
        import tempfile
        from affine import Affine
        self.directory = tempfile.mkdtemp(prefix="ttProm")
        chm = np.full((100, 100), 2.0, np.float32)            # low canopy
        rows, columns = np.mgrid[0:100, 0:100]
        chm[47:54, 30:63] = 9.0                               # east arm
        chm[18:54, 57:64] = 9.0                               # north arm
        distance = np.hypot(rows - 50, columns - 30)
        chm[distance < 8] = (9.6 + 0.4 * (1 - distance / 8))[distance < 8]
        chm[np.hypot(rows - 50, columns - 60) < 2.5] = 9.5    # P1, at the bend
        chm[np.hypot(rows - 20, columns - 60) < 2.5] = 9.4    # P2, arm's end
        chm[58:72, 24:37] = 5.0      # lower canopy, so the descent goes
                                     # below the branches, as in a real CHM
        self.path = os.path.join(self.directory, "branch.tif")
        with rasterio.open(self.path, "w", driver="GTiff", height=100,
                           width=100, count=1, dtype="float32",
                           crs="EPSG:32619",
                           transform=Affine(0.1, 0, 0, 0, -0.1, 10)) as d:
            d.write(chm, 1)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.directory, ignore_errors=True)

    def raiseP2(self):
        import rasterio
        with rasterio.open(self.path, "r+") as d:
            chm = d.read(1)
            chm[np.hypot(*np.mgrid[0:100, 0:100] - np.array([[[20]], [[60]]]))
                < 2.5] = 11.0                                 # P2 now a tree
            d.write(chm, 1)

    def tops(self, metric, **params):
        from tt.data import TDData
        from tt.detectors import CCDetector
        site = TDData.fromFiles("branch", chm=self.path)
        detector = CCDetector(
            resolution=0.1, minHeight=1.0, windowSizeM=40.0,
            lowerPercentile=0, erosionIterations=0, minTreeAreaM2=0.05,
            minTopAreaM2=0.01, topStepM=0.1, mergeMetric=metric,
            saddleEpsM=8.0, saddleDropM=0.6, **params)
        surface = detector.surface(site)
        detections = detector.detect(site)
        pixels = [surface.toPixel(x, y) for x, y in zip(detections.x,
                                                        detections.y)]
        return sorted((int(row), int(column)) for column, row in pixels)

    def testSaddleKeepsASpuriousTopOnTheBranch(self):
        # a straight line from the apex to a pseudotop leaves the narrow
        # branch and dips to the ground, so the saddle test keeps a top there
        tops = self.tops("saddle")
        self.assertEqual(len(tops), 2)
        self.assertTrue(any(c > 50 for r, c in tops))       # on the branch

    def testProminenceMergesThePseudotopsIntoTheTree(self):
        tops = self.tops("prominence")
        self.assertEqual(len(tops), 1)
        row, column = tops[0]
        self.assertLess(np.hypot(row - 50, column - 30), 8)   # on T's crown

    def testSmoothingFlattensASpikeButKeepsTheCrown(self):
        import rasterio
        from tt.data import TDData
        with rasterio.open(self.path, "r+") as d:
            chm = d.read(1)
            chm[80, 80] = 12.0                    # one noise pixel on low canopy
            d.write(chm, 1)
        site = TDData.fromFiles("branch", chm=self.path)
        plain = site.heightSurface(resolution=0.1, minHeight=1.0)
        smooth = site.heightSurface(resolution=0.1, minHeight=1.0, smoothM=0.2)
        self.assertAlmostEqual(float(plain.chm[80, 80]), 12.0)
        self.assertLess(float(smooth.chm[80, 80]), 3.0)   # the spike is gone
        self.assertGreater(float(smooth.chm[50, 30]), 9.5)  # T still stands

    def testAProminentNeighbourStaysApart(self):
        self.raiseP2()
        self.assertEqual(len(self.tops("prominence")), 2)

    def testDropSlopeMergesAtTheTopsHeight(self):
        """P2 raised to 11 m becomes the elder; T (10 m) dies where the
        branches join at 9 m, prominence 1 m. A fixed 0.6 m drop keeps T;
        0.6 + 0.05 * 10 = 1.1 m at T's height removes it."""
        self.raiseP2()
        self.assertEqual(len(self.tops("prominence")), 2)
        self.assertEqual(len(self.tops("prominence", saddleDropSlope=0.05)), 1)


class TestHeightSlopes(unittest.TestCase):
    """Parameters that grow with tree height; a slope of 0 is the old
    fixed parameter."""

    def testDropGrowsWithHeight(self):
        from tt.detectors import TopMerger
        merger = TopMerger("prominence", saddleDropM=0.5, saddleDropSlope=0.05)
        self.assertAlmostEqual(merger.dropAt(0.0), 0.5)
        self.assertAlmostEqual(merger.dropAt(20.0), 1.5)
        self.assertAlmostEqual(TopMerger(saddleDropM=0.5).dropAt(20.0), 0.5)

    def testSaddleDropIsTakenAtTheLowerTop(self):
        """The same 1 m dip separates two 5 m trees but not two branch tops
        of a 20 m tree."""
        from tt.detectors import TopMerger
        merger = TopMerger("saddle", epsM=100.0, saddleDropM=0.5,
                           saddleDropSlope=0.05)
        for height, same in ((20.0, True), (5.0, False)):
            surface = np.full((10, 40), height, np.float32)
            surface[5, 18:22] = height - 1.0
            self.assertEqual(merger.areSame(7.5, 0.0, surface=surface,
                                            a=(5, 5), b=(5, 35)), same)

    def testMinimumTopAreaGrowsWithHeight(self):
        from tt.detectors import CCDetector
        geometry = {"minPixTop": 2, "pixelArea": 0.0625}
        fixed = CCDetector(minTopAreaM2=0.12).algorithm()
        self.assertEqual(fixed._minPixTop(geometry, 20.0), 2)
        sloped = CCDetector(minTopAreaM2=0.12,
                            minTopAreaSlope=0.02).algorithm()
        self.assertEqual(sloped._minPixTop(geometry, 20.0), 8)  # 0.52 m2
        with self.assertRaises(ValueError):
            CCDetector(minTopAreaSlope=-0.01).algorithm()


if __name__ == "__main__":
    unittest.main()
