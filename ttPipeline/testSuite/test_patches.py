"""PatchSet: tiles, blocks, splits and stitching."""

import os
import tempfile
import unittest

import numpy as np

from .base import SiteCase


class TestPatchSet(SiteCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from tt.data import TDData
        cls.built = TDData.fromFiles("p", chm=cls.chmPath,
                                     crowns=cls.crownsPath, area=cls.areaPath,
                                     heightSource="LiDAR")
        cls.built.patches(os.path.join(cls.directory, "patches"),
                          [("CHM", "LiDAR")], resolution=0.25, tileSize=64,
                          blockCols=2, blockRows=2, minHeight=1.0)

    def testBuiltFromTheSiteLayers(self):
        patchSet = self.built.patchSet
        self.assertEqual(patchSet.channelCount, 1)
        self.assertEqual(len(self.built.blocks), 4)
        image = patchSet.patches[0].image()
        self.assertEqual(image.shape, (1, 64, 64))
        self.assertEqual(image.dtype, np.uint8)
        total = sum(p.crownCount for p in patchSet)
        self.assertGreaterEqual(total, len(self.trees))

    def testLoadedBackWithTheSite(self):
        from tt.data import TDData
        site = TDData.fromDataset(os.path.join(self.directory, "patches"))
        self.assertEqual(len(site.patchSet), len(self.built.patchSet))
        self.assertEqual([b.name for b in site.blocks],
                         [b.name for b in self.built.blocks])

    def testFoldSplitKeepsTheBuffer(self):
        patchSet = self.built.patchSet
        held = self.built.blocks[0]
        train, test = patchSet.foldSplit(held.name, 3.0)
        self.assertTrue(test and train)
        self.assertTrue(all(p.block == held.name for p in test))
        guard = held.geometry.buffer(3.0)
        self.assertFalse(any(p.extent.intersects(guard) for p in train))

    def testDrawIsDeterministicAndPerBlock(self):
        patchSet = self.built.patchSet
        train, _ = patchSet.foldSplit(self.built.blocks[0].name, 0.0)
        a = patchSet.draw(train, 0.5, 3, self.built.blocks[0].name)
        b = patchSet.draw(train, 0.5, 3, self.built.blocks[0].name)
        self.assertEqual([p.stem for p in a], [p.stem for p in b])
        self.assertEqual({p.block for p in a}, {p.block for p in train})

    def testRecordsAreSharedNotCopied(self):
        patchSet = self.built.patchSet
        train, _ = patchSet.foldSplit(self.built.blocks[0].name, 0.0)
        self.assertTrue(any(p.record is train[0].record for p in patchSet))

    def testStitchingToWorld(self):
        patchSet = self.built.patchSet
        patch = patchSet.patches[0]
        world = patchSet.toWorld([{"box": [0, 0, 64, 64], "score": 0.5}],
                                 patch)[0]
        extent = patch.extent
        self.assertAlmostEqual(world["centreX"], extent.centroid.x)
        self.assertAlmostEqual(world["centreY"], extent.centroid.y)


if __name__ == "__main__":
    unittest.main()
