"""Rendering: block images and pseudo-crowns' canopy."""

import os
import tempfile
import unittest

import numpy as np

from .base import SiteCase


class TestBlockImages(SiteCase):
    """Fates in the images must be the scorer's own classification."""

    def renderer(self):
        from tt.analysis import Analysis
        from tt.report.blockImages import BlockRenderer
        site = self.site()
        analysis = Analysis(site)
        return analysis, BlockRenderer(analysis, [("all", site.area)],
                                       minSidePx=200)

    def tops(self, analysis):
        from tt.data import Detections
        m = analysis.crownMeasures()
        return Detections(m["x"], m["y"], np.full(len(m["x"]), 10.0))

    def testFatesMatchTheScorer(self):
        from tt.data import Detections
        analysis, renderer = self.renderer()
        tops = self.tops(analysis)
        extra = Detections([tops.x[0] + 0.3], [tops.y[0]], [9.0])
        rest = tops.subset(np.arange(1, len(tops)))
        fates, missed, crowns = renderer.classify(
            Detections.concat([rest, extra]), "all")
        self.assertEqual((crowns, fates.count("hit"), missed), (9, 9, []))
        fates, missed, _ = renderer.classify(rest, "all")
        self.assertEqual(len(missed), 1)

    def testImageIsWritten(self):
        analysis, renderer = self.renderer()
        path = os.path.join(tempfile.mkdtemp(), "sub", "all.jpg")
        renderer.render(self.tops(analysis), "all", "test", path)
        self.assertGreater(os.path.getsize(path), 1000)


class TestPseudoCrownCanopy(SiteCase):
    """The canopy mask is the whole kept CHM unless a percentile is asked for."""

    def testWholeChmByDefaultSmallerWithAPercentile(self):
        from tt.scoring.pseudoCrowns import PseudoCrowns
        surface = self.site().heightSurface()
        whole = PseudoCrowns(surface, None, verbose=False).canopyMask() > 0
        np.testing.assert_array_equal(whole, surface.chm > 0)
        cut = PseudoCrowns(surface, None, percentile=20,
                           verbose=False).canopyMask() > 0
        self.assertTrue((cut <= whole).all())
        self.assertLess(cut.sum(), whole.sum())


class TestRealSite(unittest.TestCase):
    """
    The regression test on real data: skipped unless TT_CHM, TT_CROWNS and
    TT_BOUNDARY point at it. The published operating point must still give
    the published numbers.
    """

    @classmethod
    def setUpClass(cls):
        cls.chm = os.environ.get("TT_CHM")
        cls.crowns = os.environ.get("TT_CROWNS")
        cls.boundary = os.environ.get("TT_BOUNDARY")
        if not (cls.chm and cls.crowns and cls.boundary):
            raise unittest.SkipTest("set TT_CHM, TT_CROWNS and TT_BOUNDARY "
                                    "to run the regression test")

    def testPublishedOperatingPoint(self):
        from tt.data import TDData
        from tt.detectors import CCDetector
        from tt.scoring import AnnotatedCrowns, Evaluator
        site = TDData.fromFiles("real", chm=self.chm, crowns=self.crowns,
                                area=self.boundary)
        detections = CCDetector(
            lowerPercentile=10, minTopAreaM2=0.12, topStepM=0.12,
            erosionIterations=1, minTreeAreaM2=0.5, windowSizeM=40.0,
            saddleDropM=0.5).detect(site)
        result = Evaluator(AnnotatedCrowns(site.annotations)).score(detections)
        print("\n    real site: R %.3f  P %.3f  F1 %.3f  (%d detections)"
              % (result["recall"], result["precision"], result["f1"],
                 result["predictions"]))
        self.assertAlmostEqual(result["recall"], 0.933, delta=0.01)
        self.assertAlmostEqual(result["precision"], 0.810, delta=0.01)
        self.assertAlmostEqual(result["f1"], 0.867, delta=0.01)


if __name__ == "__main__":
    unittest.main()
