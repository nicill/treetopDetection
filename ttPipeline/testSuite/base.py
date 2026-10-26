"""A fixture site, built once per test class."""

import shutil
import tempfile
import unittest

from .fixtures import buildDataset, buildFixture, crowdedTrees


class SiteCase(unittest.TestCase):
    """Nine well-separated cones on a 60 x 60 m patch."""

    trees = [(40 + 70 * (i % 3), 40 + 70 * (i // 3), 2.0 + 0.2 * i,
              10.0 + 1.5 * i) for i in range(9)]

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp(prefix="ttSuite")
        cls.chmPath, cls.crownsPath, cls.areaPath = buildFixture(
            cls.directory, cls.trees)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def site(self, **keywords):
        from tt.data import TDData
        return TDData.fromFiles("fixture", chm=self.chmPath,
                                crowns=self.crownsPath, area=self.areaPath,
                                heightSource="LiDAR", **keywords)

    @property
    def boundaryPath(self):
        return self.areaPath

    def dataset(self):
        return buildDataset(self.directory, self.chmPath, self.crownsPath,
                            self.areaPath)


class CrowdedSiteCase(SiteCase):
    """Sixty overlapping cones: settings change the result."""

    trees = crowdedTrees()
