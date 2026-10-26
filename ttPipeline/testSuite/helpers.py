"""Fixtures and helpers shared by several test files."""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

from .fixtures import buildFixture


TESTS_SUPPORT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tests_support")


def ccDataset(case):
    """A one-by-two-block dataset over the fixture (built once per class)."""
    import subprocess
    out = os.path.join(case.directory, "ccds")
    if not os.path.exists(os.path.join(out, "dataset.json")):
        subprocess.run([sys.executable, "-m", "tt.dl", "prepare",
                        "--source", "chm:" + case.chmPath,
                        "--crowns", case.crownsPath,
                        "--boundary", case.boundaryPath, "--output", out,
                        "--name", "t", "--resolution", "0.25",
                        "--minHeight", "1.0", "--blockCols", "2",
                        "--blockRows", "1"], check=True, capture_output=True)
    return out


class Fixture(unittest.TestCase):
    """Nine well-separated trees on a 60 x 60 m patch."""

    trees = [(40 + 70 * (i % 3), 40 + 70 * (i // 3), 2.0 + 0.2 * i,
              10.0 + 1.5 * i) for i in range(9)]

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp(prefix="ttTest")
        cls.chmPath, cls.crownsPath, cls.boundaryPath = buildFixture(
            cls.directory, cls.trees)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)



# ---------------------------------------------------------------------- #


def asBoxes(dicts):
    from tt.data import Detections
    return Detections.fromPredictions(dicts, "boxes")


def asPoints(dicts):
    from tt.data import Detections
    return Detections.fromPredictions(dicts, "points")


def fusionBoxes(*items, half=1.5):
    """Detections from (x, y, score) boxes."""
    from tt.data import Detections
    xs = [x for x, _, _ in items]
    ys = [y for _, y, _ in items]
    return Detections(xs, ys, [sc for _, _, sc in items], geometry="boxes",
                      boxes=[[x - half, y - half, x + half, y + half]
                             for x, y in zip(xs, ys)])


def fusionTops(*items):
    """Point Detections from (x, y, prominence), the prominence given."""
    from tt.data import Detections
    tops = Detections([x for x, _, _ in items], [y for _, y, _ in items],
                      [1.0] * len(items))
    tops.cache["prominence"] = np.array([p for _, _, p in items], float)
    return tops


def givenProminence(cls):
    """A three-way strategy reading the prominence the tops carry."""

    class Given(cls):
        def prominence(self, points):
            return points.cache["prominence"]

    return Given()


class CurveSite(unittest.TestCase):
    """A synthetic site with CHM and RGB datasets and a CC run on it."""

    @classmethod
    def setUpClass(cls):
        import rasterio
        from tt.dl.dlConComp import crossValidate, parseArguments
        from tt.dl.dlPrepare import main as prepare
        cls.directory = tempfile.mkdtemp(prefix="ttCurve")
        d = cls.directory
        trees = [(20 + 25 * (i % 9), 20 + 25 * (i // 9), 1.6 + 0.1 * (i % 3),
                  8.0 + 0.4 * (i % 7)) for i in range(81)]
        chm, crowns, area = buildFixture(d, trees)
        with rasterio.open(chm) as source:
            heights, profile = source.read(1), source.profile
        profile.update(count=3, dtype="uint8", nodata=None)
        with rasterio.open(os.path.join(d, "rgb.tif"), "w", **profile) as out:
            out.write(np.stack([np.clip(np.maximum(heights, 0) * 20, 0, 255)]
                               * 3).astype(np.uint8))
        for kind, path in (("chm", chm), ("rgb", os.path.join(d, "rgb.tif"))):
            prepare(["--source", "%s:%s" % (kind, path), "--crowns", crowns,
                     "--boundary", area, "--output", os.path.join(d, kind),
                     "--name", "synth", "--resolution", "0.25",
                     "--minHeight", "1", "--blockCols", "2", "--blockRows",
                     "2", "--tileSize", "64"])
        cls.ccRun = os.path.join(d, "ccP1")
        crossValidate(parseArguments([
            "--dataset", os.path.join(d, "chm"), "--output", cls.ccRun,
            "--resolution", "0.25", "--minHeight", "1", "--percentiles",
            "0,10", "--saddleDrops", "0.3,0.5", "--erosions", "0",
            "--minTopAreas", "0.06", "--topSteps", "0.25", "--saveDetections"]))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def path(self, *parts):
        return os.path.join(self.directory, *parts)


class TileBoxes(object):
    """A stand-in network: one box per tile, covering it (no torch)."""

    @staticmethod
    def make(**params):
        from tt.detectors.dl.base import DLDetector

        class Boxes(DLDetector):
            method = "boxes"

            def reset(self, patchSet):
                pass

            def fit(self, patchSet, patches, validation=None):
                pass

            def release(self):
                pass

            def predictPatches(self, patchSet, patches):
                return [{"box": [p.record["west"], p.record["south"],
                                 p.record["east"], p.record["north"]],
                         "centreX": p.record["centreX"],
                         "centreY": p.record["centreY"], "score": 0.9}
                        for p in patches]

        return Boxes(**params)
