"""Trained cross-validation, transfer and fixed runs, with a stand-in network."""

import json
import os
import tempfile
import unittest

from .base import CrowdedSiteCase


def crownBoxes(**params):
    """A stand-in network: each tile's crowns as boxes, scores varying."""
    from tt.detectors.dl.base import DLDetector

    class CrownBoxes(DLDetector):
        method = "crownBoxes"

        def reset(self, patchSet):
            self.fitted = None

        def fit(self, patchSet, patches, validation=None):
            self.fitted = (len(patches), len(validation or []))

        def release(self):
            pass

        def predictPatches(self, patchSet, patches):
            out = []
            for patch in patches:
                tile = [{"box": c["box"],
                         "score": 0.3 + 0.06 * (c["crownIndex"] % 10)}
                        for c in patch.crowns()]
                tile.append({"box": [28, 28, 36, 36], "score": 0.2})
                out.extend(patchSet.toWorld(tile, patch))
            return out

    return CrownBoxes(**params)


class TestTrainedCrossValidation(CrowdedSiteCase):

    def site(self):
        from tt.data import TDData
        directory = os.path.join(self.directory, "tiles")
        if not os.path.exists(os.path.join(directory, "dataset.json")):
            site = super().site()
            site.patches(directory, [("CHM", "LiDAR")], resolution=0.25,
                         tileSize=64, blockCols=2, blockRows=2, minHeight=1.0)
        return TDData.fromDataset(directory)

    def testThresholdIsChosenOffTheTestBlock(self):
        from tt.experiments.trained import TrainedCrossValidation
        runSet = TrainedCrossValidation(self.site(), crownBoxes(),
                                        bufferM=2.0).run()
        self.assertEqual(len(runSet.folds), 4)
        for fold in runSet.folds:
            self.assertNotEqual(fold["validationBlock"], fold["block"])
            # the junk boxes score 0.2; the validation block learns to drop
            self.assertGreater(fold["threshold"], 0.2)
        self.assertGreater(runSet.pooled["precision"], 0.5)

    def testPredictionFilesKeepEverythingAboveTheFloor(self):
        from tt.experiments import RunSet
        from tt.experiments.trained import TrainedCrossValidation
        runSet = TrainedCrossValidation(self.site(), crownBoxes(),
                                        bufferM=2.0).run()
        output = tempfile.mkdtemp()
        runSet.save(output)
        block = runSet.folds[0]["block"]
        with open(os.path.join(output, "predictions_%s.json" % block)) as h:
            content = json.load(h)
        self.assertEqual(content["threshold"], runSet.folds[0]["threshold"])
        self.assertTrue(content["validationPredictions"])
        self.assertTrue(any(p["score"] < content["threshold"]
                            for p in content["predictions"]))
        back = RunSet.load(output)
        self.assertEqual(back.extra["channelCount"], 1)

    def testFixedThreshold(self):
        from tt.experiments.trained import TrainedCrossValidation
        runSet = TrainedCrossValidation(self.site(), crownBoxes(threshold=0.1),
                                        bufferM=2.0,
                                        fixedThreshold=True).run()
        self.assertEqual({f["threshold"] for f in runSet.folds}, {0.1})

    def testTransferAndFixedRun(self):
        from tt.experiments.trained import FixedRun, TrainedTransfer
        site = self.site()
        transfer = TrainedTransfer(site, crownBoxes(), self.site()).run()
        self.assertEqual(len(transfer.folds), 4)

        fixed = FixedRun(site, crownBoxes(threshold=0.4)).run()
        self.assertEqual({f["threshold"] for f in fixed.folds}, {0.4})
        self.assertLessEqual(fixed.pooled["predictions"],
                             transfer.pooled["predictions"])


if __name__ == "__main__":
    unittest.main()
