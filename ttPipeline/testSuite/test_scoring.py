"""The evaluator, its references and objectives."""

import os
import unittest

from .base import SiteCase


class TestEvaluator(SiteCase):

    def evaluator(self, site):
        from tt.scoring import AnnotatedCrowns, Evaluator, F1
        return Evaluator(AnnotatedCrowns(site.annotations), F1())

    def testPerfectDetection(self):
        from tt.detectors import CCDetector
        site = self.site()
        scores = self.evaluator(site).score(CCDetector(minHeight=1.0)
                                            .detect(site))
        self.assertEqual((scores["recall"], scores["precision"]), (1.0, 1.0))

    def testMissedAndRepeated(self):
        from tt.data import Detections
        site = self.site()
        tops = site.annotations.tops(site.heightSurface(minHeight=1.0))
        x = list(tops[1:, 0]) + [tops[1, 0] + 0.2]
        y = list(tops[1:, 1]) + [tops[1, 1]]
        matching = self.evaluator(site).match(Detections(x, y))
        self.assertEqual(matching.missed, [0])
        self.assertEqual(matching.count("repeat"), 1)
        scores = matching.scores()
        self.assertAlmostEqual(scores["recall"], 8 / 9.0)
        self.assertAlmostEqual(scores["precision"], 8 / 9.0)

    def testObjectives(self):
        from tt.scoring import F1, Objective, Weighted
        scores = {"recall": 1.0, "precision": 0.5, "f1": 2 / 3.0}
        self.assertAlmostEqual(F1().value(scores), 2 / 3.0)
        self.assertAlmostEqual(Weighted().value(scores), 0.8)
        self.assertAlmostEqual(Weighted(0.5).value(scores), 0.75)
        os.environ["TT_OBJECTIVE"] = "weighted"
        try:
            self.assertEqual(Objective.fromEnvironment().name, "weighted")
        finally:
            del os.environ["TT_OBJECTIVE"]

    def testAnyCrownFrameCanBeTheReference(self):
        from tt.detectors import CCDetector
        from tt.scoring import CrownFrame, Evaluator
        site = self.site()
        half = site.crowns.iloc[:4]
        scores = Evaluator(CrownFrame(half)).score(
            CCDetector(minHeight=1.0).detect(site))
        self.assertEqual(scores["crowns"], 4)
        self.assertEqual(scores["hits"], 4)


if __name__ == "__main__":
    unittest.main()
