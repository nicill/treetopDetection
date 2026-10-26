"""The command line, end to end, on the synthetic site."""

import os
import unittest

from .base import SiteCase


class TestCommandLine(SiteCase):
    """
    Every subcommand, end to end, on the synthetic scene.

    These exist because `tt align` shipped with two undefined names in it and
    nothing noticed: no test ran it. A NameError anywhere in a subcommand's
    path now fails here instead of in front of someone's data.
    """

    def invoke(self, *arguments):
        # not "run": that name belongs to unittest.TestCase and shadowing it
        # hands the test runner's result object to argparse
        from tt.cli import main
        scene = ["--chm", self.chmPath, "--crowns", self.crownsPath,
                 "--boundary", self.boundaryPath]
        return main(list(arguments[:1]) + scene + list(arguments[1:]))

    def output(self, name):
        return os.path.join(self.directory, name)

    def testDetect(self):
        self.assertEqual(self.invoke("detect", "--windowSize", "30",
                                  "--output", self.output("detect")), 0)
        self.assertTrue(os.path.exists(self.output("detect/tops.txt")))

    def testAnalyse(self):
        self.assertEqual(self.invoke("analyse", "--windowSize", "30", "--noImages",
                                  "--output", self.output("analyse")), 0)
        self.assertTrue(os.path.exists(self.output("analyse/analysis.json")))

    def testSweep(self):
        self.assertEqual(self.invoke("sweep", "--windowSize", "30",
                                  "--percentiles", "20", "--minTopAreas", "0.12",
                                  "--topSteps", "0.25", "--erosions", "1",
                                  "--saddleDrops", "0.3",
                                  "--output", self.output("sweep.json")), 0)

    def testMerge(self):
        self.assertEqual(self.invoke("merge", "--windowSize", "30",
                                  "--saddleDrops", "0.3", "--epsilons", "8",
                                  "--output", self.output("merge.json")), 0)

    def testCrowns(self):
        self.assertEqual(self.invoke("crowns", "--windowSize", "30",
                                  "--output", self.output("crowns")), 0)
        self.assertTrue(os.path.exists(self.output("crowns/pseudoBoxes.shp")))

    def testAlign(self):
        from tt.cli import main
        self.assertEqual(main(["align", "--chm", self.chmPath,
                               "--layer", "crowns=" + self.crownsPath,
                               "--boundary", self.boundaryPath,
                               "--resolution", "0.25", "--tile", "20"]), 0)

    def testZeroSlopeLeavesTheSweepGridAlone(self):
        from tt.cli import _addSlopes
        grid = {}
        _addSlopes(grid, "saddleDropSlope", "0")
        self.assertEqual(grid, {})
        _addSlopes(grid, "saddleDropSlope", "0,0.05")
        self.assertEqual(grid, {"saddleDropSlope": [0.0, 0.05]})


if __name__ == "__main__":
    unittest.main()
