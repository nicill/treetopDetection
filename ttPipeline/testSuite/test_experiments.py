"""Parameter grids, cross-validation and run sets."""

import os
import tempfile
import unittest

from .base import CrowdedSiteCase
from .fixtures import plain


class TestParameterGrid(unittest.TestCase):

    def testOrderAndOverrides(self):
        from tt.experiments import ParameterGrid
        grid = ParameterGrid({"a": [1, 2], "b": [3]}).withAxes(c=[4, 5])
        self.assertEqual(len(grid), 4)
        self.assertEqual(grid.expand()[1], {"a": 1, "b": 3, "c": 5})
        self.assertEqual(grid.varying(), ["a", "c"])


class TestCrossValidation(CrowdedSiteCase):

    grid = {"lowerPercentile": [10, 20], "minTopAreaM2": [0.12, 0.5],
            "topStepM": [0.25], "erosionIterations": [1],
            "saddleDropM": [0.3, 1.5], "mergeMetric": ["saddle", "prominence"],
            "minHeight": [1.0, 3.0]}

    def crossValidate(self, jobs=1, grid=None):
        from tt.data import TDData
        from tt.detectors import CCDetector
        from tt.experiments import CrossValidation, ParameterGrid
        validation = CrossValidation(TDData.fromDataset(self.dataset()),
                                     CCDetector(),
                                     ParameterGrid(grid or self.grid),
                                     verbose=False, jobs=jobs)
        return validation, validation.run()

    def testSameAsTheOldCrossValidation(self):
        """Folds, pooled scores and saved predictions, number for number."""
        from tt.dl.dlConComp import ConCompCrossValidation
        output = tempfile.mkdtemp()
        old = ConCompCrossValidation(self.dataset(), grid=dict(self.grid),
                                     minHeight=2.0, verbose=False,
                                     output=output)
        folds, pooled = old.run()
        _, runSet = self.crossValidate(jobs=2)
        self.assertEqual(plain(folds), plain(runSet.folds))
        self.assertEqual(plain(pooled), plain(runSet.pooled))
        newOutput = tempfile.mkdtemp()
        runSet.save(newOutput)
        for name, _ in old.blocks:
            fileName = "predictions_%s.json" % name
            with open(os.path.join(output, fileName)) as a, \
                    open(os.path.join(newOutput, fileName)) as b:
                self.assertEqual(plain(__import__("json").load(a)),
                                 plain(__import__("json").load(b)))

    def testComparedAxesAreTunedSeparately(self):
        _, runSet = self.crossValidate()
        self.assertEqual(sorted(runSet.pooled["byMergeMetric"]),
                         ["prominence", "saddle"])
        self.assertEqual(runSet.pooled["bySmoothing"], {})

    def testCeilingIsNoWorseThanCrossValidated(self):
        _, runSet = self.crossValidate()
        self.assertGreaterEqual(runSet.pooled["wholeAreaBest"]["f1"],
                                runSet.pooled["f1"] - 1e-9)

    def testParallelEqualsSerial(self):
        _, serial = self.crossValidate(jobs=1)
        _, parallel = self.crossValidate(jobs=3)
        self.assertEqual(plain(serial.folds), plain(parallel.folds))

    def testRunSetRoundTrip(self):
        from tt.experiments import RunSet
        _, runSet = self.crossValidate()
        output = tempfile.mkdtemp()
        runSet.save(output)
        back = RunSet.load(output)
        self.assertEqual(plain(back.pooled), plain(runSet.pooled))
        self.assertEqual(sorted(back.predictions), sorted(runSet.predictions))
        block = sorted(back.predictions)[0]
        self.assertEqual(len(back.predictions[block]),
                         len(runSet.predictions[block]))

    def testBaselinesSameAsBefore(self):
        from tt.data import TDData
        from tt.detectors import LocalMaximaDetector, WatershedDetector
        from tt.dl.dlBaselines import BaselineCrossValidation
        from tt.experiments import CrossValidation, ParameterGrid
        site = TDData.fromDataset(self.dataset())
        for method, cls in (("lmvw", LocalMaximaDetector),
                            ("watershed", WatershedDetector)):
            grid = dict(cls.defaultGrid, minHeight=[1.0, 2.0])
            folds, pooled = BaselineCrossValidation(
                method, self.dataset(), grid=dict(grid), minHeight=2.0,
                verbose=False).run()
            runSet = CrossValidation(site, cls(), ParameterGrid(grid),
                                     verbose=False).run()
            self.assertEqual(plain(folds), plain(runSet.folds), method)
            self.assertEqual(plain(pooled), plain(runSet.pooled), method)


if __name__ == "__main__":
    unittest.main()
