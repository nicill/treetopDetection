"""Fusion strategies, CombinedDetector and FusionCrossValidation."""

import os
import tempfile
import unittest

import numpy as np

from .base import CrowdedSiteCase
from .test_trained import crownBoxes


def boxes(xs, ys, scores, half=1.5):
    from tt.data import Detections
    return Detections(xs, ys, scores, geometry="boxes",
                      boxes=[[x - half, y - half, x + half, y + half]
                             for x, y in zip(xs, ys)])


def points(xs, ys, scores):
    from tt.data import Detections
    return Detections(xs, ys, scores, heights=scores)


class TestStrategies(unittest.TestCase):

    def setUp(self):
        self.boxes = boxes([0, 10, 20], [0, 0, 0], [0.9, 0.4, 0.2])
        self.points = points([0.5, -0.5, 10, 30], [0, 0, 0.5, 0],
                             [5.0, 7.0, 6.0, 4.0])

    def testUnionKeepsUncoveredPointsAtZero(self):
        from tt.detectors.fusion import Union
        out = Union().combine(self.boxes, self.points, threshold=0.3)
        self.assertEqual(len(out), 3)                 # two boxes, one point
        self.assertEqual(list(out.scores), [0.9, 0.4, 0.0])
        self.assertEqual(out.x[-1], 30)

    def testBoxMergedPointsKeepsTheBestPerBox(self):
        from tt.detectors.fusion import BoxMergedPoints
        out = BoxMergedPoints().combine(self.boxes, self.points,
                                        threshold=0.3)
        self.assertEqual(sorted(out.scores), [4.0, 6.0, 7.0])

    def testConfirmedTwoThresholds(self):
        from tt.detectors.fusion import Confirmed
        out = Confirmed().combine(self.boxes, self.points, confirmedAt=0.3,
                                  unconfirmedAt=0.1)
        self.assertEqual(list(out.scores), [0.9, 0.4, 0.2])
        out = Confirmed().combine(self.boxes, self.points, confirmedAt=0.3,
                                  unconfirmedAt=np.inf)
        self.assertEqual(list(out.scores), [0.9, 0.4])

    def testWeakBoxesMergeWhenClearOfStrongOnes(self):
        from tt.detectors.fusion import WeakBoxMergedPoints
        out = WeakBoxMergedPoints().combine(self.boxes, self.points,
                                            threshold=0.5)
        self.assertEqual(len(out), 3)                 # merged under box 1


class TestCombinedDetector(CrowdedSiteCase):

    def testUnionOfTwoHeightDetectors(self):
        from tt.detectors import CCDetector, LocalMaximaDetector
        from tt.detectors.fusion import CombinedDetector, Union
        site = self.site()
        lm = LocalMaximaDetector(minHeight=1.0, windowBaseM=2.0)
        cc = CCDetector(minHeight=1.0)
        combined = CombinedDetector([lm, cc], Union(), {"threshold": 5.0})
        out = combined.detect(site)
        alone = lm.detect(site).aboveScore(5.0)
        self.assertGreaterEqual(len(out), len(alone))
        self.assertEqual(combined.withParams(threshold=8.0)
                         .params["threshold"], 8.0)


class TestFusionCrossValidation(CrowdedSiteCase):

    def testEveryStrategyOnSavedRuns(self):
        from tt.data import TDData
        from tt.detectors import CCDetector
        from tt.detectors.fusion import SaddleSurface
        from tt.experiments import CrossValidation, ParameterGrid, RunSet
        from tt.experiments.fusion import FusionCrossValidation
        from tt.experiments.trained import TrainedCrossValidation
        from tt.scoring import AnnotatedCrowns
        directory = os.path.join(self.directory, "fusionTiles")
        if not os.path.exists(os.path.join(directory, "dataset.json")):
            self.site().patches(directory, [("CHM", "LiDAR")],
                                resolution=0.25, tileSize=64, blockCols=2,
                                blockRows=2, minHeight=1.0)
        site = TDData.fromDataset(directory)
        output = tempfile.mkdtemp()
        CrossValidation(site, CCDetector(minHeight=1.0),
                        ParameterGrid({"saddleDropM": [0.3, 1.0]}),
                        verbose=False).run().save(os.path.join(output, "cc"))
        TrainedCrossValidation(site, crownBoxes(), bufferM=2.0).run().save(
            os.path.join(output, "box"))
        blocks = {b.name: b.geometry for b in site.blocks}
        surface = SaddleSurface.fromSurface(site.heightSurface(minHeight=1.0))
        validation = FusionCrossValidation.pair(
            AnnotatedCrowns(site.annotations), blocks,
            RunSet.load(os.path.join(output, "box")),
            RunSet.load(os.path.join(output, "cc")), surface=surface,
            heightSurface=surface, verbose=False)
        summary = validation.run()
        self.assertEqual(len(summary), 10)
        boxRun = RunSet.load(os.path.join(output, "box"))
        for fold in summary["boxes"]["folds"]:
            # the box model alone enters at the threshold it chose itself
            self.assertEqual(fold["setting"]["threshold"],
                             boxRun.threshold(fold["block"]))
        for name, entry in summary.items():
            self.assertEqual(len(entry["folds"]), len(blocks), name)
        validation.saveOutputs(os.path.join(output, "fused"))
        self.assertTrue(os.path.exists(os.path.join(
            output, "fused", "union", "predictions_%s.json"
            % sorted(blocks)[0])))


if __name__ == "__main__":
    unittest.main()
