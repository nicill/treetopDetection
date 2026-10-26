"""TDData, its layers, annotations and Detections."""

import os
import unittest

import numpy as np

from .base import SiteCase


class TestTDData(SiteCase):

    def testLayersByKindAndSource(self):
        site = self.site()
        self.assertEqual(site.layer("CHM").source, "LiDAR")
        self.assertIs(site.layer("CHM", "LiDAR"), site.layer("CHM"))
        with self.assertRaises(KeyError):
            site.layer("CHM", "P1")
        with self.assertRaises(KeyError):
            site.layer("RGB")

    def testAskingForOneOfSeveralIsAnError(self):
        from tt.data import TDData
        site = TDData.fromFiles("two", chm={"LiDAR": self.chmPath,
                                            "P1": self.chmPath})
        with self.assertRaises(KeyError):
            site.layer("CHM")
        self.assertEqual(site.layer("CHM", "P1").source, "P1")

    def testSurfacesAreBuiltOnce(self):
        site = self.site()
        self.assertIs(site.heightSurface(minHeight=1.0),
                      site.heightSurface(minHeight=1.0))

    def testCrownsAndTheirTops(self):
        site = self.site()
        self.assertEqual(len(site.crowns), len(self.trees))
        tops = site.annotations.tops(site.heightSurface(minHeight=1.0))
        expected = sorted(h for *_, h in self.trees)
        np.testing.assert_allclose(sorted(tops[:, 2]), expected, atol=0.5)

    def testFromDatasetReadsBlocks(self):
        from tt.data import TDData
        site = TDData.fromDataset(self.dataset(), heightSource="P1")
        self.assertEqual(len(site.blocks), 4)
        self.assertEqual(site.layer("CHM").source, "P1")
        self.assertAlmostEqual(site.blockArea.area, site.area.area, places=3)


class TestShow(SiteCase):

    def testTheSiteWithARunOnIt(self):
        import tempfile
        from tt.detectors import CCDetector
        site = self.site()
        detections = CCDetector(minHeight=1.0).detect(site)
        path = site.show(os.path.join(tempfile.mkdtemp(), "site.png"),
                         {"cc": detections}, minSidePx=200)
        self.assertGreater(os.path.getsize(path), 1000)
        minX, minY, maxX, maxY = site.area.bounds
        from shapely.geometry import box
        half = box(minX, minY, (minX + maxX) / 2, maxY)
        path = site.show(os.path.join(tempfile.mkdtemp(), "half.png"),
                         {"cc": detections}, region=half, minSidePx=200)
        self.assertGreater(os.path.getsize(path), 1000)


class TestDetections(unittest.TestCase):

    def detections(self):
        from tt.data import Detections
        return Detections([0.0, 10.0, 20.0], [0.0, 0.0, 5.0],
                          scores=[3.0, 2.0, 1.0], heights=[3.0, 2.0, 1.0])

    def testPointsRoundTripThroughRunFiles(self):
        import tempfile
        from tt.data import Detections
        detections = self.detections()
        detections.block, detections.params = "a_b00", {"k": 1}
        path = os.path.join(tempfile.mkdtemp(), "predictions_a_b00.json")
        detections.save(path)
        back = Detections.load(path)
        self.assertEqual(back.geometry, "points")
        self.assertEqual((back.block, back.params), ("a_b00", {"k": 1}))
        np.testing.assert_array_equal(back.x, detections.x)
        np.testing.assert_array_equal(back.heights, detections.heights)

    def testBoxesAndExtraFieldsSurvive(self):
        from tt.data import Detections
        predictions = [{"box": [0, 0, 4, 4], "centreX": 2.0, "centreY": 2.0,
                        "score": 0.9, "label": 1,
                        "polygon": [[0, 0], [4, 0], [4, 4]]}]
        back = Detections.fromPredictions(predictions)
        self.assertEqual(back.geometry, "masks")
        self.assertEqual(back.toPredictions(), predictions)

    def testRegionAndThreshold(self):
        from shapely.geometry import box
        detections = self.detections()
        self.assertEqual(len(detections.inRegion(box(-1, -1, 15, 1))), 2)
        self.assertEqual(len(detections.aboveScore(2.0)), 2)


if __name__ == "__main__":
    unittest.main()
