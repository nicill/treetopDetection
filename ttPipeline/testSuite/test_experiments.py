"""Parameter grids, cross-validation and run sets."""

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

    def testHeightAndTreeAreaAreTunedLikeTheRest(self):
        validation, _ = self.crossValidate(grid={
            "lowerPercentile": [10], "minTopAreaM2": [0.12], "topStepM": [0.25],
            "erosionIterations": [1], "saddleDropM": [0.5],
            "minHeight": [1.0, 3.0], "minTreeAreaM2": [0.5, 40.0]})
        self.assertEqual(len(validation.settings), 4)
        self.assertEqual(sorted({key[3] for key in
                                 validation.data._surfaces}), [1.0, 3.0])
        counts = {(s["minHeight"], s["minTreeAreaM2"]): len(d)
                  for s, d in zip(validation.settings, validation.detections)}
        # a 40 m2 minimum tree area leaves none of the small cones
        self.assertLess(counts[(1.0, 40.0)], counts[(1.0, 0.5)])

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



if __name__ == "__main__":
    unittest.main()
