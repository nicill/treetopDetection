"""Fixed settings and the learning curve."""

import os
import tempfile
import unittest

from .base import CrowdedSiteCase


class TestFixedAndCurve(CrowdedSiteCase):

    def prepared(self):
        from tt.data import TDData
        directory = os.path.join(self.directory, "curveTiles")
        if not os.path.exists(os.path.join(directory, "dataset.json")):
            self.site().patches(directory, [("CHM", "LiDAR")],
                                resolution=0.25, tileSize=64, blockCols=2,
                                blockRows=2, minHeight=1.0)
        return TDData.fromDataset(directory)

    def testFixedSettingsScoresEveryBlock(self):
        from tt.detectors import CCDetector
        from tt.experiments import RunSet
        from tt.experiments.fixed import FixedSettings
        site = self.prepared()
        runSet = FixedSettings(site, CCDetector(minHeight=1.0),
                               {"saddleDropM": 0.3}).run()
        self.assertEqual(len(runSet.folds), 4)
        self.assertEqual(runSet.settings, {"saddleDropM": 0.3})
        output = tempfile.mkdtemp()
        runSet.save(output)
        back = RunSet.load(output)
        self.assertEqual(back.extra["wholeArea"]["crowns"],
                         len(site.crowns))

    def testMoreAnnotationNeverUsesFewerCrowns(self):
        from tt.detectors import CCDetector
        from tt.experiments import CrossValidation, ParameterGrid
        from tt.experiments.fixed import LearningCurve
        from tt.scoring import AnnotatedCrowns, Evaluator
        site = self.prepared()
        validation = CrossValidation(
            site, CCDetector(minHeight=1.0),
            ParameterGrid({"saddleDropM": [0.3, 1.0]}), verbose=False)
        found = validation.detectAll()
        curve = LearningCurve(site, validation.settings, found,
                              site.patchSet,
                              Evaluator(AnnotatedCrowns(site.annotations)),
                              [0.25, 1.0], [1], bufferM=2.0)
        points = curve.run()
        used = [p["pooled"]["crownsUsedPerFold"] for p in points]
        self.assertLessEqual(used[0], used[1])
        self.assertEqual(curve.transfer(validation.settings[0])["folds"][0]
                         ["chosen"], 0)


class TestGridSearch(CrowdedSiteCase):

    def testEverySettingScoredOnAllCrowns(self):
        from tt.detectors import CCDetector
        from tt.experiments import ParameterGrid
        from tt.experiments.fixed import GridSearch
        search = GridSearch(self.site(), CCDetector(minHeight=1.0),
                            ParameterGrid({"saddleDropM": [0.3, 1.0],
                                           "mergeMetric": ["saddle"]}),
                            verbose=False)
        rows = search.run()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["label"], "metric=saddle drop=0.3")
        self.assertEqual(rows[0]["crowns"], len(self.trees))
        self.assertIs(search.best(), max(rows, key=lambda r: r["f1"]))


if __name__ == "__main__":
    unittest.main()
