"""The detector hierarchy and its postprocessors."""

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

    def testCCMatchesTheOldDetector(self):
        from tt import ConCompDetector, Scene, TopMerger
        from tt.detectors import CCDetector
        scene = Scene(self.chmPath, crownsPath=self.crownsPath,
                      boundaryPath=self.areaPath, verbose=False)
        old = ConCompDetector(merger=TopMerger("saddle", epsM=8.0),
                              verbose=False).detect(scene)
        new = CCDetector().detect(self.site())
        world = old.world(scene)
        np.testing.assert_array_equal(world[:, 0], new.x)
        np.testing.assert_array_equal(old.heights, new.scores)

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


if __name__ == "__main__":
    unittest.main()
