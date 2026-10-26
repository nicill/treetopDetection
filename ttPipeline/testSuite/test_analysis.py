"""Analysis: the site, a result, and methods tree by tree."""

import os
import unittest

import numpy as np

from .base import SiteCase
from .test_trained import crownBoxes


class TestAnalysis(SiteCase):

    def testSiteStatsFromTheCrowns(self):
        from tt.analysis import Analysis
        site = self.site()
        stats = Analysis(site, site.heightSurface(minHeight=1.0)).siteStats()
        self.assertEqual(stats["crowns"], len(self.trees))
        self.assertEqual(stats["tops"]["count"], len(self.trees))
        self.assertAlmostEqual(stats["tops"]["perHectare"],
                               len(self.trees) / stats["areaHa"])
        heights = sorted(h for *_, h in self.trees)
        self.assertAlmostEqual(stats["tops"]["heights"][1],
                               float(np.median(heights)), delta=0.5)

    def testTopStatsIsTheSameMeasureForDetections(self):
        from tt.analysis import Analysis
        from tt.detectors import CCDetector
        site = self.site()
        analysis = Analysis(site, site.heightSurface(minHeight=1.0))
        crowns = analysis.topStats(analysis.crownTops())
        found = analysis.topStats(CCDetector(minHeight=1.0).detect(site))
        self.assertEqual(crowns["count"], found["count"])
        np.testing.assert_allclose(crowns["heights"], found["heights"],
                                   atol=0.5)

    def testErrorsSplitTheFalsePositives(self):
        from tt.analysis import Analysis
        from tt.data import Detections
        site = self.site()
        analysis = Analysis(site, site.heightSurface(minHeight=1.0))
        tops = analysis.crownTops()
        # every tree once, one repeat, and one detection on bare ground
        x = np.r_[tops.x, tops.x[0] + 0.3, site.area.bounds[0] + 1.0]
        y = np.r_[tops.y, tops.y[0], site.area.bounds[1] + 1.0]
        h = np.r_[tops.heights, tops.heights[0] - 1, 0.1]
        errors = analysis.errors(Detections(x, y, h, heights=h))
        self.assertEqual((errors["hits"], errors["repeats"]),
                         (len(self.trees), 1))
        self.assertEqual(errors["backgroundLow"] + errors["backgroundUnknown"],
                         1)
        self.assertEqual(errors["missed"], 0)

    def testTreeByTree(self):
        from tt.analysis import Analysis
        from tt.data import TDData
        from tt.experiments.trained import TrainedCrossValidation
        directory = os.path.join(self.directory, "treeTiles")
        self.site().patches(directory, [("CHM", "LiDAR")], resolution=0.25,
                            tileSize=64, blockCols=2, blockRows=2,
                            minHeight=1.0)
        site = TDData.fromDataset(directory)
        run = TrainedCrossValidation(site, crownBoxes(), bufferM=2.0).run()
        table = Analysis(site).treeByTree({"boxes": run, "again": run})
        agreement = table.agreement("boxes", "again")
        self.assertEqual(agreement["only boxes"]["count"], 0)
        self.assertEqual(table.scores("boxes")["hits"],
                         len(table.results["boxes"]["found"]))


class TestStatistics(unittest.TestCase):

    def testHolmAndThePooledDifference(self):
        from tt.analysis import compareFolds, holm
        np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        a = {b: {"crowns": 10, "recall": 0.9, "precision": 0.9, "f1": 0.9}
             for b in ("b00", "b01", "b02")}
        b = {k: dict(v, recall=0.8, f1=0.85) for k, v in a.items()}
        result = compareFolds(a, b, 200, np.random.default_rng(0))
        self.assertGreater(result["pooledDifference"], 0)
        self.assertEqual(result["referenceWins"], 3)


if __name__ == "__main__":
    unittest.main()
