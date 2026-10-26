"""The commands and the experiments they run, on synthetic sites."""

import argparse
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from .helpers import CurveSite, Fixture, TileBoxes, ccDataset


class TestTransfer(unittest.TestCase):
    """The modal setting and the paired comparison, on hand-built folds."""

    def testModalSettingIsTheMostChosen(self):
        import json
        import tempfile
        from tt.transfer import modalSetting
        a = {"lowerPercentile": 20, "minTopAreaM2": 0.12, "topStepM": 0.12,
             "erosionIterations": 2, "saddleDropM": 0.3}
        b = dict(a, saddleDropM=0.5)
        path = os.path.join(tempfile.mkdtemp(), "results.json")
        with open(path, "w") as handle:
            json.dump({"folds": [{"settings": a}, {"settings": b},
                                 {"settings": a}]}, handle)
        setting, count, total = modalSetting(path)
        self.assertEqual((setting, count, total), (a, 2, 3))

    def testPairedMatchesBlocksByName(self):
        from tt.transfer import paired
        tuned = [{"block": "b0", "f1": 0.8}, {"block": "b1", "f1": 0.7}]
        fixed = [{"block": "b1", "f1": 0.6}, {"block": "b0", "f1": 0.9}]
        result = paired(tuned, fixed)
        self.assertAlmostEqual(result["tunedMinusFixed"], 0.0)
        self.assertEqual(result["tunedBetter"], 1)


class TestStudySummary(unittest.TestCase):
    """Each hybrid is held against the best single method of every site."""

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="ttStudy")
        # site: (cc, mrcnnRgb, union hybrid, calibrated@0.9)
        for site, (cc, rgb, union, cal) in {
                "siteA": (0.80, 0.70, 0.85, 0.82),
                "siteB": (0.60, 0.75, 0.78, 0.55),
                "siteC": (0.70, 0.72, 0.71, 0.74)}.items():
            for name, f1 in (("ccLidar", cc), ("mrcnnRgb", rgb)):
                pooled = self.pooled(f1)
                if name == "ccLidar":
                    pooled["wholeAreaBest"] = self.pooled(min(f1 + 0.1, 1.0))
                self.write(site, "runs/%s/results.json" % name,
                           {"pooled": pooled})
            self.write(site, "report/fused/mrcnnRgb_ccLidar/summary.json",
                       {"boxes": {"pooled": self.pooled(rgb)},
                        "points": {"pooled": self.pooled(cc)},
                        "union": {"pooled": self.pooled(union)}})
            self.write(site, "calibration/calibration.json",
                       {"summary": {"0.9": {"calibrated": self.pooled(cal)}}})

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def pooled(self, f1):
        return {"recall": f1, "precision": f1, "f1": f1, "crowns": 100}

    def write(self, site, relative, data):
        import json
        path = os.path.join(self.directory, site, relative)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as handle:
            json.dump(data, handle)

    def testBestSingleAndDifferences(self):
        from tt.report.studySummary import compareHybrids, readStudy
        sites = readStudy(self.directory)
        self.assertEqual([sites[s]["bestSingle"]["f1"] for s in sorted(sites)],
                         ["ccLidar", "mrcnnRgb", "mrcnnRgb"])
        self.assertNotIn("mrcnnRgb+ccLidar:boxes", sites["siteA"]["hybrids"])
        rows = {r["hybrid"]: r for r in compareHybrids(sites)}
        union = rows["mrcnnRgb+ccLidar:union"]
        self.assertAlmostEqual(union["meanDifference"],
                               ((0.85 - 0.80) + (0.78 - 0.75) + (0.71 - 0.72))
                               / 3)
        self.assertEqual((union["wins"], union["sites"]), (2, 3))
        self.assertEqual(rows["calibrated@0.9"]["wins"], 2)
        self.assertIn("ccLidar ceiling", sites["siteA"]["ceilings"])
        self.assertNotIn("ccLidar ceiling", sites["siteA"]["singles"])

    def testWritesTheTables(self):
        from tt.report.studySummary import main
        out = os.path.join(self.directory, "out")
        self.assertEqual(main(["--root", self.directory, "--output", out]), 0)
        with open(os.path.join(out, "sites.csv")) as handle:
            self.assertEqual(len(handle.readlines()), 1 + 3 * 5)


class TestObjective(unittest.TestCase):
    """F1 picks the strict threshold, 0.6 R + 0.4 P the lenient one."""

    def setUp(self):
        import geopandas as gpd
        from shapely.geometry import box
        self.crowns = gpd.GeoDataFrame(
            geometry=[box(10 * i, 0, 10 * i + 2, 2) for i in range(10)],
            crs="EPSG:32619")
        point = lambda x, y, s: {"centreX": x, "centreY": y, "score": s,
                                 "box": [x - .5, y - .5, x + .5, y + .5]}
        self.predictions = ([point(10 * i + 1, 1, 0.9) for i in range(6)] +
                            [point(10 * i + 1, 1, 0.3) for i in range(6, 9)] +
                            [point(10 * i + 5, 5, 0.3) for i in range(6)])
        self.area = box(-5, -5, 105, 10)
        self.previous = os.environ.get("TT_OBJECTIVE")

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("TT_OBJECTIVE", None)
        else:
            os.environ["TT_OBJECTIVE"] = self.previous

    def threshold(self, objective):
        from tt.dl import dlCommon as dc
        os.environ["TT_OBJECTIVE"] = objective
        return dc.selectThreshold(self.predictions, self.crowns, self.area,
                                  candidates=(0.3, 0.9), verbose=False)[0]

    def testTheObjectiveDecides(self):
        self.assertEqual(self.threshold("f1"), 0.9)
        self.assertEqual(self.threshold("weighted"), 0.3)

    def testEveryResultCarriesBoth(self):
        from tt.dl import dlCommon as dc
        result = dc.evaluateDetections(self.predictions, self.crowns, self.area)
        self.assertAlmostEqual(result["weighted"],
                               0.6 * result["recall"] + 0.4 * result["precision"])
        self.assertIn("f1", result)

    def testAnUnknownObjectiveIsRefused(self):
        from tt.dl import dlCommon as dc
        os.environ["TT_OBJECTIVE"] = "recall"
        with self.assertRaises(ValueError):
            dc.tuningObjective()


class TestSiteAnalysis(unittest.TestCase):
    """Crowns drawn 0.5 m east of the trees: the scan finds the shift back."""

    def setUp(self):
        import geopandas as gpd
        from shapely.geometry import Point, box
        centres = [(2.0 + 3 * i, 2.0) for i in range(8)]
        self.crowns = gpd.GeoDataFrame(
            geometry=[Point(x + 0.5, y).buffer(0.3) for x, y in centres],
            crs="EPSG:32619")
        self.predictions = [{"centreX": x, "centreY": y, "score": 3.0,
                             "box": [x - .5, y - .5, x + .5, y + .5]}
                            for x, y in centres]
        self.region = box(-5, -5, 40, 10)

    def analysis(self):
        import types
        from tt.analysis import Analysis
        from tt.scoring import CrownFrame
        return Analysis(types.SimpleNamespace(area=None, blocks=[]),
                        surface=False, reference=CrownFrame(self.crowns))

    def detections(self):
        from tt.data import Detections
        return Detections.fromPredictions(self.predictions)

    def testTheShiftIsFound(self):
        result = self.analysis().offsetScan(self.detections(), self.region)
        self.assertEqual([round(v, 1) for v in result["bestShiftM"]],
                         [-0.5, 0.0])
        self.assertAlmostEqual(result["unshiftedF1"], 0.0)
        self.assertAlmostEqual(result["bestF1"], 1.0)

    def testGrowingTheCrownsRecoversEdgeTops(self):
        result = self.analysis().tolerance(self.detections(), self.region)
        self.assertAlmostEqual(result["0.25"]["f1"], 1.0)   # 0.3 + 0.25 > 0.5


class TestSharedBlocks(unittest.TestCase):
    """A block a learned model skipped is left out, by name or _bNN suffix."""

    def testOnlyBlocksEveryRunHas(self):
        from tt.data import Detections
        from tt.experiments import RunSet
        from tt.experiments.runset import sharedBlocks
        run = RunSet("m", "site", [], {}, predictions={
            name: Detections([], []) for name in ("site_b00", "site_b01")})
        self.assertTrue(run.has("site_b00"))
        self.assertTrue(run.has("other_b01"))      # matched by suffix
        self.assertFalse(run.has("site_b02"))
        kept = sharedBlocks({"site_b00": 0, "site_b01": 1, "site_b02": 2},
                            (run,), verbose=False)
        self.assertEqual(sorted(kept), ["site_b00", "site_b01"])


class TestPseudoTuning(unittest.TestCase):
    """The pseudo scorers pick settings without the crowns; scored with them."""

    def testExactlyOneShare(self):
        from tt.data import Detections
        from tt.experiments.pseudo import exactlyOneShare
        boxes = np.array([[0, 0, 2, 2], [5, 5, 7, 7], [10, 10, 12, 12]], float)
        points = Detections([1, 6, 6.5], [1, 6, 6.5])
        self.assertAlmostEqual(exactlyOneShare(points, boxes), 1.0 / 3)
        self.assertEqual(exactlyOneShare(Detections([], []), boxes), 0.0)

    def testChosenSettingIsScoredOnTheRealCrowns(self):
        from tt.experiments.pseudo import BlockScores
        from tt.scoring import Evaluator, F1

        class Stub(BlockScores):
            def __init__(self):
                self.real = Evaluator(None, F1())

            def realScores(self):
                return [{"crowns": 10, "f1": f, "hits": 0, "predictions": 0,
                         "repeats": 0, "falsePositives": 0, "recall": f,
                         "precision": f} for f in (0.5, 0.9, 0.7)]

            def scorers(self):
                # pseudoF1 prefers setting 2, the zones setting 1
                return {"pseudoF1": [0.1, 0.2, 0.3], "zones@0.9": [0.2, 0.8, 0.5]}

        rows = Stub().tune()
        self.assertEqual(rows["real"]["chosen"], 1)
        self.assertAlmostEqual(rows["pseudoF1"]["f1"], 0.7)
        self.assertAlmostEqual(rows["zones@0.9"]["f1"], 0.9)
        self.assertAlmostEqual(rows["zones@0.9"]["spearman"], 1.0)
        self.assertAlmostEqual(rows["pseudoF1"]["spearman"], 0.5)

    def testTheAutomaticRunSitsBesideItsSource(self):
        from tt.pseudoTuning import autoRunDir
        self.assertEqual(autoRunDir("/s/runs/ccLidar"), "/s/runs/ccAutoLidar")
        self.assertEqual(autoRunDir("/s/runs/ccP1/"), "/s/runs/ccAutoP1")
        from tt.report.text import HEIGHT_SOURCE
        self.assertEqual(HEIGHT_SOURCE["ccAutoLidar"], "lidar")
        self.assertEqual(HEIGHT_SOURCE["ccAutoP1"], "p1")

    def testDetectionsRoundTrip(self):
        from tt.data import Detections
        from tt.experiments.crossValidation import (CrossValidation,
                                                    loadAllDetections)
        path = os.path.join(tempfile.mkdtemp(), "d.npz")
        holder = object.__new__(CrossValidation)
        holder.detections = [Detections([1.0], [2.0], [3.0]), Detections([], [])]
        holder.saveAllDetections(path)
        back = loadAllDetections(path)
        self.assertEqual(len(back), 2)
        self.assertEqual((back[0].x[0], back[0].scores[0]), (1.0, 3.0))
        self.assertEqual(len(back[1]), 0)


class TestLearningCurve(CurveSite):
    """
    The learning curve and the DeepForest chain; DeepForest itself replaced
    by a stand-in.
    """

    def testDrawIsStratifiedAndShared(self):
        from tt.dl import dlCommon as dc
        from tt.dl import dlPrepare as dp
        draws = []
        for kind in ("rgb", "chm"):
            meta = dp.loadDataset(self.path(kind))
            held = meta["blocks"][0]["name"]
            train, _ = dc.foldSplit(meta["tiles"], meta["blockGeometries"],
                                    held, 2.0)
            drawn = dc.drawTiles(train, 0.25, 3, held)
            self.assertEqual({t["block"] for t in drawn},
                             {t["block"] for t in train})
            self.assertEqual(dc.drawTiles(train, 1.0, 3, held), train)
            draws.append(sorted((t["r0"], t["c0"]) for t in drawn))
        self.assertEqual(draws[0], draws[1])

    def testCcCurveAndTransfer(self):
        from tt.learningCurve import main
        output = self.path("curve", "synth", "ccCurve_ccP1.json")
        self.assertEqual(main(["cc", "--ccRun", self.ccRun, "--dataset",
                               self.path("chm"), "--tiles", self.path("rgb"),
                               "--fractions", "0.25,1", "--seeds", "1,2",
                               "--buffer", "2", "--transferFrom", self.ccRun,
                               "--jobs", "2", "--output", output]), 0)
        from tt.dl import dlCommon as dc
        record = dc.loadJson(output)
        self.assertEqual([(p["fraction"], p["seed"]) for p in
                          record["points"]], [(0.25, 1), (0.25, 2), (1.0, 1)])
        used = [p["pooled"]["crownsUsedPerFold"] for p in record["points"]]
        self.assertLess(used[0], used[2])
        self.assertGreater(record["transfer"]["pooled"]["recall"], 0.9)
        self.assertEqual(main(["summary", "--root", self.path("curve")]), 0)

    def testDeepForestChain(self):
        import types
        import pandas as pd
        from scipy.ndimage import maximum_filter

        class StandIn(object):
            def __init__(self, config_args=None):
                pass

            def predict_tile(self, image=None):
                band = image[..., 0]
                rows, cols = np.nonzero((band == maximum_filter(band, 9)) &
                                        (band > 100))
                return pd.DataFrame({"xmin": cols - 6, "ymin": rows - 6,
                                     "xmax": cols + 6, "ymax": rows + 6,
                                     "score": np.full(len(rows), 0.8)})

        from unittest import mock
        module = types.ModuleType("deepforest")
        module.main = types.SimpleNamespace(deepforest=StandIn)
        with mock.patch.dict(sys.modules, {"deepforest": module}):
            from tt import deepForestCompare, deepForestDetect, pseudoTuning
            from tt.detectors.dl import deepForest
        # the stand-in even where the real DeepForest is installed
        deepForest.dfMain = module.main
        boxes = self.path("df", "deepforestRgb")
        auto = self.path("df", "ccAutoDf")
        deepForestDetect.main(["--dataset", self.path("rgb"),
                               "--output", boxes])
        pseudoTuning.main(["--ccRun", self.ccRun, "--rgbRun", boxes,
                           "--dataset", self.path("chm"), "--jobs", "1",
                           "--output", self.path("df", "pseudo"),
                           "--autoRun", auto])
        self.assertFalse(os.path.exists(self.path("ccAutoP1")))
        rows, tests = deepForestCompare.compare(argparse.Namespace(
            dataset=self.path("chm"), deepforest=boxes, ccAuto=auto,
            reference=["ccP1=" + self.ccRun]))
        byName = {r["name"]: r for r in rows}
        self.assertGreater(byName["deepforest"]["recall"], 0.8)
        self.assertGreaterEqual(byName["union"]["recall"],
                                byName["deepforest"]["recall"])
        self.assertEqual(set(tests), {"deepforest", "ccOnDeepforest", "ccP1"})


class TestNetworkSubsets(CurveSite):
    """The networks' annotated subset and transfer, without training."""

    def crossValidation(self, dataset, drawFrom=None, fraction=0.5,
                        bufferM=10.0):
        from tt.data import PatchSet, TDData
        from tt.experiments.trained import TrainedCrossValidation
        return TrainedCrossValidation(
            TDData.fromDataset(dataset), TileBoxes.make(), bufferM=bufferM,
            trainFraction=fraction, subsetSeed=2,
            drawFrom=PatchSet.load(drawFrom) if drawFrom else None)

    def testSubsetFoldUsesTheDrawnTiles(self):
        from tt.dl import dlCommon as dc
        run = self.crossValidation(self.path("rgb"), fraction=0.25,
                                   bufferM=2.0)
        block = run.data.blocks[0]
        fit, val, validationBlock, _ = run.splitFold(block.name)
        meta = run.patchSet.meta
        train, _ = dc.foldSplit(meta["tiles"], run.patchSet.blocks,
                                block.name, 2.0)
        drawn = dc.drawTiles(train, 0.25, 2, block.name)
        self.assertEqual(sorted(p.stem for p in fit + val),
                         sorted(t["stem"] for t in drawn))
        region = run.validationRegion(validationBlock, val)
        self.assertLess(region.area,
                        run.data.block(validationBlock).geometry.area)
        result, _ = run.runFold(block)
        self.assertEqual((result["fitTiles"], result["trainFraction"]),
                         (len(fit), 0.25))

    def testTransferScoresEveryBlock(self):
        from tt.data import TDData
        from tt.experiments.trained import TrainedTransfer
        runSet = TrainedTransfer(TDData.fromDataset(self.path("rgb")),
                                 TileBoxes.make(),
                                 TDData.fromDataset(self.path("rgb"))).run()
        self.assertEqual(len(runSet.folds), 4)
        output = self.path("mrT")
        runSet.save(output)
        self.assertTrue(os.path.exists(os.path.join(
            output, "predictions_%s.json" % runSet.folds[0]["block"])))

    @staticmethod
    def keys(fold):
        fit, val, _, _ = fold
        return {(p.block, p.record["r0"], p.record["c0"]) for p in fit + val}

    def reducedDataset(self):
        """The CHM dataset's list without the RGB list's busiest tile."""
        import json
        meta = json.load(open(self.path("rgb", "dataset.json")))
        victim = max(meta["tiles"], key=lambda t: t["crownCount"])
        meta["tiles"] = [t for t in meta["tiles"] if t is not victim]
        folder = self.path("reduced")
        os.makedirs(folder, exist_ok=True)
        json.dump(meta, open(os.path.join(folder, "dataset.json"), "w"))
        return folder, (victim["block"], victim["r0"], victim["c0"])

    def testIdenticalDatasetsDrawAsBefore(self):
        plain = self.crossValidation(self.path("chm"))
        drawn = self.crossValidation(self.path("chm"), self.path("rgb"))
        for block in plain.data.blocks:
            a, b = plain.splitFold(block.name), drawn.splitFold(block.name)
            self.assertEqual(self.keys(a) if a else None,
                             self.keys(b) if b else None)

    def testAMissingTileLeavesTheRestOfTheDrawAlone(self):
        folder, victim = self.reducedDataset()
        reference = self.crossValidation(self.path("rgb"))
        reduced = self.crossValidation(folder, self.path("rgb"))
        for block in reference.data.blocks:
            want = reference.splitFold(block.name)
            got = reduced.splitFold(block.name)
            if want is None or got is None:
                continue
            self.assertEqual(self.keys(got), self.keys(want) - {victim})


class TestBaselineCommand(unittest.TestCase):
    """The baselines' command: its grid and the run it writes."""

    def testGridOverride(self):
        from tt.dl.dlBaselines import gridFor, parseArguments
        args = parseArguments(["--method", "watershed", "--dataset", "x",
                               "--output", "y", "--markerWindows", "2,6,8"])
        grid = gridFor(args)
        self.assertEqual(grid["markerWindowM"], [2.0, 6.0, 8.0])
        self.assertEqual(grid["minCrownAreaM2"], [0.0, 0.5, 1.0, 2.0])

    def testCrossValidationWritesARun(self):
        from tt.dl.dlBaselines import crossValidate, parseArguments
        site = CurveSite
        site.setUpClass()
        try:
            output = os.path.join(site.directory, "lmvwP1")
            pooled = crossValidate(parseArguments([
                "--method", "lmvw", "--dataset", os.path.join(site.directory,
                                                              "chm"),
                "--output", output, "--resolution", "0.25", "--minHeight",
                "1", "--minHeights", "0.5,1.0"]))
            self.assertGreater(pooled["f1"], 0.9)
            from tt.report.studySummary import singles
            os.makedirs(os.path.join(site.directory, "runs"), exist_ok=True)
            shutil.move(output, os.path.join(site.directory, "runs", "lmvwP1"))
            methods, ceilings = singles(site.directory)
            self.assertIn("lmvwP1", methods)
            self.assertIn("lmvwP1 ceiling", ceilings)
        finally:
            site.tearDownClass()


class TestConCompSlopes(unittest.TestCase):
    """Height slopes reach the detector as grid dimensions, only when given."""

    def testSlopesAreGridDimensionsOnlyWhenGiven(self):
        from tt.dl import dlConComp as cc
        base = ["--dataset", "x", "--output", "y"]
        plain = cc.gridFromArguments(cc.parseArguments(base))
        self.assertNotIn(cc.DROP_SLOPE_KEY, plain)
        self.assertNotIn(cc.TOP_SLOPE_KEY, plain)
        grid = cc.gridFromArguments(cc.parseArguments(
            base + ["--saddleDropSlopes", "0,0.05", "--minTopAreaSlopes", "0.02"]))
        self.assertEqual(grid[cc.DROP_SLOPE_KEY], [0.0, 0.05])
        self.assertEqual(grid[cc.TOP_SLOPE_KEY], [0.02])

    def testDetectorReceivesTheSlopes(self):
        from tt.detectors import CCDetector
        from tt.dl import dlConComp as cc
        setting = {"lowerPercentile": 0, "minTopAreaM2": 0.1, "topStepM": 0.1,
                   "erosionIterations": 0, "saddleDropM": 0.3,
                   cc.DROP_SLOPE_KEY: 0.05, cc.TOP_SLOPE_KEY: 0.02}
        algorithm = CCDetector().withParams(**setting).algorithm()
        self.assertEqual(algorithm.merger.saddleDropSlope, 0.05)
        self.assertEqual(algorithm.minTopAreaSlope, 0.02)
        del setting[cc.DROP_SLOPE_KEY], setting[cc.TOP_SLOPE_KEY]
        algorithm = CCDetector().withParams(**setting).algorithm()
        self.assertEqual(algorithm.merger.saddleDropSlope, 0.0)
        self.assertEqual(algorithm.minTopAreaSlope, 0.0)


class TestConCompProgress(Fixture):
    """Progress lines and the settings table never change a result."""

    def dataset(self):
        return ccDataset(self)

    def testProgressSameResultsAndTables(self):
        import json
        from tt.dl import dlConComp as cc
        grid = ["--resolution", "0.25", "--minHeight", "1", "--percentiles", "0,10",
                "--saddleDrops", "0.3,0.5", "--erosions", "0", "--minTopAreas",
                "0.06", "--topSteps", "0.25"]
        runs = {}
        for name, extra in (("plain", []), ("progress", ["--progress", "--jobs", "2"])):
            output = os.path.join(self.directory, "cc_" + name)
            cc.crossValidate(cc.parseArguments(
                ["--dataset", self.dataset(), "--output", output] + grid + extra))
            runs[name] = output
        a, b = (json.load(open(os.path.join(runs[n], "results.json")))
                for n in ("plain", "progress"))
        self.assertEqual([(f["hits"], f["predictions"], f["settings"]) for f in a["folds"]],
                         [(f["hits"], f["predictions"], f["settings"]) for f in b["folds"]])
        for name in runs:
            rows = open(os.path.join(runs[name], "settings.tsv")).read().splitlines()
            self.assertEqual(len(rows), 1 + 4)
        progress = open(os.path.join(runs["progress"], "progress.tsv")).read().splitlines()
        self.assertEqual(len(progress), 1 + 4)
        self.assertFalse(os.path.exists(os.path.join(runs["plain"], "progress.tsv")))


class TestHeightSmoothing(unittest.TestCase):
    """Smoothing that grows with tree height: none for slope 0, more on tall trees."""

    def testTallCrownsSmoothedMoreSmallLess(self):
        from scipy.ndimage import gaussian_filter
        from tt.data.raster import heightSmoothed
        rng = np.random.default_rng(0)
        chm = np.zeros((120, 240), np.float32)
        bumps = rng.normal(0, 0.5, chm.shape).astype(np.float32)
        chm[:, :120] = 20 + bumps[:, :120]           # tall canopy, branchy
        chm[:, 120:] = 3 + bumps[:, 120:]            # small trees
        pixel, base = 0.25, 0.25
        constant = gaussian_filter(chm, base / pixel)
        grown = heightSmoothed(chm, base, 0.03, pixel)
        rough = lambda a, cols: float(np.std(a[10:-10, cols]))
        tall, small = slice(10, 110), slice(130, 230)
        # tall canopy: smoother than with the constant sigma
        self.assertLess(rough(grown, tall), 0.7 * rough(constant, tall))
        # small trees: about as before (sigma 0.25 + 0.03 x 3 m, one step)
        self.assertAlmostEqual(rough(grown, small), rough(constant, small),
                               delta=0.35 * rough(constant, small))

    def testNarrowWindowSparesASmallTreeBesideATallOne(self):
        from tt.data.raster import heightSmoothed
        rows, columns = np.mgrid[0:80, 0:80].astype(float)
        chm = np.zeros((80, 80), np.float32)
        # a 20 m crown, and 2 m to its side a 3 m tree with a sharp top
        chm = np.maximum(chm, 20 * np.clip(1 - np.hypot(rows - 40, columns - 30) / 12, 0, 1))
        chm = np.maximum(chm, 3 * np.clip(1 - np.hypot(rows - 40, columns - 50) / 4, 0, 1))
        pixel = 0.25
        top = lambda a: float(a[40, 50])
        wide = heightSmoothed(chm, 0.0, 0.05, pixel, windowM=12.0)
        narrow = heightSmoothed(chm, 0.0, 0.05, pixel, windowM=0.25)
        # the narrow window smooths the small tree by its own height: its top
        # survives better than when it borrows its tall neighbour's height
        self.assertGreater(top(narrow), top(wide))

    def testDefaultWindowUnchanged(self):
        from tt.dl import dlConComp as cc
        from tt.data.raster import LOCAL_HEIGHT_M, heightSmoothed
        rng = np.random.default_rng(1)
        chm = (5 + rng.normal(0, 1, (60, 60))).astype(np.float32)
        self.assertTrue(np.array_equal(
            heightSmoothed(chm, 0.25, 0.02, 0.25),
            heightSmoothed(chm, 0.25, 0.02, 0.25, windowM=LOCAL_HEIGHT_M)))
        base = ["--dataset", "x", "--output", "y"]
        self.assertNotIn(cc.SMOOTH_WINDOW_KEY, cc.gridFromArguments(cc.parseArguments(base)))
        grid = cc.gridFromArguments(cc.parseArguments(base + ["--smoothWindows", "0.25,1,2"]))
        self.assertEqual(grid[cc.SMOOTH_WINDOW_KEY], [0.25, 1.0, 2.0])

    def testSlopeZeroIsTheConstantSmoothing(self):
        from tt.dl import dlConComp as cc
        base = ["--dataset", "x", "--output", "y"]
        self.assertNotIn(cc.SMOOTH_SLOPE_KEY, cc.gridFromArguments(cc.parseArguments(base)))
        grid = cc.gridFromArguments(cc.parseArguments(base + ["--smoothSlopes", "0,0.02"]))
        self.assertEqual(grid[cc.SMOOTH_SLOPE_KEY], [0.0, 0.02])


class TestHeightSmoothingRun(Fixture):
    """In CC's cross-validation, a slope of 0 reproduces the run without it."""

    def dataset(self):
        return ccDataset(self)

    def testZeroSlopeSameFolds(self):
        import json
        from tt.dl import dlConComp as cc
        grid = ["--resolution", "0.25", "--minHeight", "1", "--percentiles", "0",
                "--saddleDrops", "0.3,0.5", "--erosions", "0", "--minTopAreas",
                "0.06", "--topSteps", "0.25", "--smooths", "0,0.25"]
        out = {}
        for name, extra in (("plain", []), ("zero", ["--smoothSlopes", "0"]),
                            ("grown", ["--smoothSlopes", "0,0.03"])):
            output = os.path.join(self.directory, "cs_" + name)
            cc.crossValidate(cc.parseArguments(
                ["--dataset", self.dataset(), "--output", output] + grid + extra))
            out[name] = json.load(open(os.path.join(output, "results.json")))
        key = lambda r: [(f["hits"], f["predictions"]) for f in r["folds"]]
        self.assertEqual(key(out["plain"]), key(out["zero"]))
        self.assertEqual(len(out["grown"]["folds"]), len(out["plain"]["folds"]))


class TestAutoScorer(unittest.TestCase):
    """The automatic run follows the chosen scorer; the default is unchanged."""

    def testEachScorerWritesItsOwnChoices(self):
        import json
        import types
        from tt.data import Detections
        from tt.experiments.pseudo import PseudoTuning
        from tt.pseudoTuning import AUTO_SCORER, parseArguments, writeAutoRun
        d = tempfile.mkdtemp(prefix="ttAuto")
        try:
            found = [Detections([1.0], [1.0], [1.0]), Detections([2.0], [2.0], [1.0])]
            tuning = PseudoTuning.__new__(PseudoTuning)
            tuning.detections, tuning.data = found, types.SimpleNamespace(name="s")
            row = lambda chosen: {"chosen": chosen, "recall": 1.0}
            perBlock = {"s_b00": {AUTO_SCORER: row(0), "zones@0.7": row(1)}}
            for scorer, expected in ((AUTO_SCORER, 1.0), ("zones@0.7", 2.0)):
                out = os.path.join(d, scorer)
                writeAutoRun(tuning, perBlock, {}, out, scorer)
                saved = json.load(open(os.path.join(out, "predictions_s_b00.json")))
                self.assertEqual(saved["predictions"][0]["centreX"], expected)
            args = parseArguments(["--ccRun", "a", "--rgbRun", "b", "--dataset",
                                   "c", "--output", "d"])
            self.assertEqual(args.autoScorer, AUTO_SCORER)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestTreeWeights(unittest.TestCase):
    """DeepForest's one-class RetinaNet loads into ours without changing it."""

    def setUp(self):
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    @staticmethod
    def treeModel():
        """A stand-in for DeepForest's model, built as DeepForest builds it:
        torchvision's COCO-configured RetinaNet (frozen batch normalization,
        eps 0, so no num_batches_tracked buffers), one class."""
        import torch
        from torchvision.models import resnet50
        from torchvision.models.detection._utils import overwrite_eps
        from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
        from torchvision.models.detection.retinanet import RetinaNet
        from torchvision.ops import misc as misc_nn_ops
        from torchvision.ops.feature_pyramid_network import LastLevelP6P7
        torch.manual_seed(0)
        backbone = _resnet_fpn_extractor(
            resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d), 3,
            returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256))
        tree = RetinaNet(backbone, num_classes=1).eval()
        overwrite_eps(tree, 0.0)
        with torch.no_grad():
            for p in tree.head.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        return tree

    def testSameScoresAndBoxesAsTheTreeModel(self):
        import torch
        import torchvision
        from tt.dl import dlModels as dm
        tree = self.treeModel()
        self.assertFalse(any(k.endswith("num_batches_tracked") for k in tree.state_dict()))
        d = tempfile.mkdtemp(prefix="ttTree")
        try:
            path = os.path.join(d, "tree.pt")
            torch.save({"model." + k: v for k, v in tree.state_dict().items()}, path)
            ours = dm.buildDetector("retinanetv1", 3, pretrained=False,
                                    initWeights=path).eval()
            x = torch.rand(1, 3, 256, 256)
            with torch.no_grad():
                a = list(tree.backbone(x).values())
                b = list(ours.backbone(x).values())
                self.assertTrue(all(torch.allclose(u, v, atol=1e-5) for u, v in zip(a, b)))
                ca = tree.head.classification_head(a)
                cb = ours.head.classification_head(b)
                self.assertTrue(torch.allclose(cb[..., 1], ca[..., 0], atol=1e-5))
                self.assertLess(float(torch.sigmoid(cb[..., 0]).max()), 0.011)
                self.assertTrue(torch.allclose(ours.head.regression_head(b),
                                               tree.head.regression_head(a),
                                               atol=1e-5))
            wrong = torchvision.models.detection.retinanet_resnet50_fpn_v2(
                weights=None, weights_backbone=None, num_classes=1)
            bad = os.path.join(d, "v2.pt")
            torch.save(wrong.state_dict(), bad)
            with self.assertRaises(ValueError):
                dm.buildDetector("retinanetv1", 3, pretrained=False, initWeights=bad)
            with self.assertRaises(ValueError):
                dm.buildDetector("maskrcnn", 3, pretrained=False, initWeights=path)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def testBothStartsShareOneConfiguration(self):
        """The COCO start and the tree start: frozen normalization with eps 0,
        and the first two backbone stages frozen, as torchvision builds a
        pretrained RetinaNet."""
        from torchvision.ops import misc as misc_nn_ops
        from tt.dl import dlModels as dm
        model = dm._retinanetV1(None, 2)
        norms = [m for m in model.modules() if isinstance(m, misc_nn_ops.FrozenBatchNorm2d)]
        self.assertTrue(norms)
        self.assertTrue(all(m.eps == 0.0 for m in norms))
        frozen = {n.split(".")[2] for n, p in model.named_parameters()
                  if n.startswith("backbone.body.") and not p.requires_grad}
        self.assertEqual(frozen, {"conv1", "layer1"})


class TestCurveRowsOfAnotherNetwork(unittest.TestCase):
    """siteRows reads the CHM network's runs by prefix, the RGB network's by default."""

    @staticmethod
    def results(f1, blocks=("s_b00", "s_b01")):
        folds = [{"block": b, "crowns": 100, "hits": int(100 * f1),
                  "predictions": 100, "repeats": 0, "falsePositives": 0,
                  "recall": f1, "precision": 1.0, "f1": 2 * f1 / (1 + f1),
                  "weighted": 0.6 * f1 + 0.4} for b in blocks]
        return {"folds": folds}

    def testPrefixSelectsTheNetworkAndDefaultIsUnchanged(self):
        import json
        from tt.learningCurve import siteRows
        d = tempfile.mkdtemp(prefix="ttRows")
        try:
            for name, f1 in (("mrcnn_f1_s1", 0.9), ("mrcnn_f0.5_s1", 0.8),
                             ("mrcnnChm_f1_s1", 0.7), ("mrcnnChm_f0.5_s1", 0.6),
                             ("mrcnnChm_f0.5_s2", 0.5), ("mrcnnChm_transfer", 0.2)):
                os.makedirs(os.path.join(d, name))
                json.dump(self.results(f1), open(os.path.join(d, name, "results.json"), "w"))
            rgb = {r["point"]: r for r in siteRows(d)}
            chm = {r["point"]: r for r in siteRows(d, prefix="mrcnnChm", name="mrcnnChm")}
            self.assertEqual(set(rgb), {"50%", "100%"})              # the CHM runs ignored
            self.assertEqual(set(chm), {"other site", "50%", "100%"})
            self.assertTrue(all(r["method"] == "mrcnnChm" for r in chm.values()))
            self.assertEqual(chm["50%"]["draws"], 2)                  # two draws, one row
            self.assertAlmostEqual(chm["50%"]["recall"], 0.55)
            self.assertAlmostEqual(rgb["50%"]["recall"], 0.8)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestHeightSourceCompare(unittest.TestCase):
    """Per-block paired tests between two studies' height runs."""

    def write(self, root, run, f1s, pooled):
        import json
        path = os.path.join(root, "site", "runs", run)
        os.makedirs(path)
        folds = [{"block": "site_b%02d" % i, "f1": f, "weighted": f}
                 for i, f in enumerate(f1s)]
        with open(os.path.join(path, "results.json"), "w") as handle:
            json.dump({"pooled": {"f1": pooled}, "folds": folds}, handle)

    def testDifferencesAndBetterBaseline(self):
        from tt.heightSourceCompare import main
        import json
        d = tempfile.mkdtemp(prefix="ttHsc")
        try:
            lidar, p1 = os.path.join(d, "l"), os.path.join(d, "p")
            base = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94]
            self.write(lidar, "ccLidar", [b + 0.05 for b in base], 0.92)
            self.write(lidar, "lmvwLidar", base, 0.87)
            self.write(lidar, "watershedLidar", [b + 0.01 for b in base], 0.88)
            self.write(p1, "ccP1", [b + 0.05 for b in base], 0.92)    # same
            self.write(p1, "lmvwP1", [b + 0.03 + 0.001 * i
                                      for i, b in enumerate(base)], 0.90)
            self.write(p1, "watershedP1", base, 0.87)
            out = os.path.join(d, "c.json")
            self.assertEqual(main(["--first", lidar, "--second", p1,
                                   "--output", out]), 0)
            result = json.load(open(out))[0]
            self.assertEqual(result["sources"]["cc"]["f1P"], 1.0)
            self.assertLess(result["sources"]["lmvw"]["f1P"], 0.05)
            self.assertAlmostEqual(result["sources"]["lmvw"]["f1MeanDiff"],
                                   0.0335, places=4)
            self.assertEqual(result["ccVsBaseline"]["Lidar"]["baseline"],
                             "watershed")
            self.assertEqual(result["ccVsBaseline"]["P1"]["baseline"], "lmvw")
            self.assertLess(result["ccVsBaseline"]["Lidar"]["f1P"], 0.05)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestReplayRun(unittest.TestCase):
    """A run's recorded options written back as the options to replay it."""

    helpText = ("usage: x [--dataset D] [--output O] [--epochs E] "
                "[--minHeights M] [--noPretrained] [--fixedThreshold] "
                "[--device D]")

    def testOptionsBackUnderTheirFlags(self):
        from tt.replayRun import words
        settings = {"dataset": "a", "output": "b", "log": None, "epochs": 40,
                    "minHeights": "0.75,1.0", "pretrained": True,
                    "fixedThreshold": False, "device": "0"}
        self.assertEqual(words(settings, self.helpText),
                         ["--epochs", "40", "--minHeights", "0.75,1.0",
                          "--device", "0"])
        self.assertEqual(words({"pretrained": False}, self.helpText),
                         ["--noPretrained"])

    def testUnknownOptionStopsTheReplay(self):
        from tt.replayRun import words
        with self.assertRaises(ValueError):
            words({"minHeight": 2.0}, self.helpText)    # only --minHeights
        with self.assertRaises(ValueError):
            words({"epochs": 40}, "")                   # no --help read

    def testConcompRunReplaysAsRecorded(self):
        from tt.replayRun import main
        import json
        d = tempfile.mkdtemp(prefix="ttReplay")
        try:
            path = os.path.join(d, "results.json")
            json.dump({"settings": {"dataset": "x", "resolution": 0.1,
                                    "minHeights": "0.75,1.0",
                                    "saveDetections": True, "jobs": 16}},
                      open(path, "w"))
            from io import StringIO
            with mock.patch("sys.stdout", new=StringIO()) as out:
                self.assertEqual(main([path, "tt.dl", "concomp"]), 0)
            self.assertEqual(out.getvalue().split(),
                             ["--resolution", "0.1", "--minHeights",
                              "0.75,1.0", "--saveDetections"])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestRunCompare(unittest.TestCase):
    """Pooled difference over blocks with a bootstrap, and the block test."""

    def write(self, root, name, folds):
        import json
        path = os.path.join(root, name)
        os.makedirs(path)
        rows = [{"block": "b%02d" % i, "crowns": c, "recall": r,
                 "precision": p, "f1": 2 * r * p / (r + p)}
                for i, (c, r, p) in enumerate(folds)]
        with open(os.path.join(path, "results.json"), "w") as handle:
            json.dump({"pooled": {"f1": 0}, "folds": rows}, handle)

    def testSmallBlocksDecideTheTestNotThePooledDifference(self):
        import json
        from tt.runCompare import main
        d = tempfile.mkdtemp(prefix="ttRunCompare")
        try:
            big = [(400, 0.80, 0.80)] * 6
            self.write(d, "a", big + [(30, 0.80, 0.70), (40, 0.80, 0.70)])
            # the other: equal on big blocks but a hair worse, collapses on
            # the two small ones
            self.write(d, "b", [(400, 0.79, 0.80)] * 6 +
                       [(30, 0.30, 0.80), (40, 0.30, 0.80)])
            out = os.path.join(d, "c.json")
            self.assertEqual(main(["--root", d, "--reference", "a", "--others",
                                   "b", "--bootstrap", "2000",
                                   "--output", out]), 0)
            row = json.load(open(out))["rows"][0]
            self.assertEqual(row["referenceWins"], 8)
            self.assertLess(row["p"], 0.01)
            # the pooled difference is small: the big blocks dominate
            self.assertLess(row["pooledDifference"], 0.05)
            self.assertGreater(row["meanBlockDifference"],
                               3 * row["pooledDifference"])
            lo, hi = row["interval95"]
            self.assertLessEqual(lo, row["pooledDifference"])
            self.assertGreaterEqual(hi, row["pooledDifference"])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestExperimentReport(unittest.TestCase):
    """Old against new runs, with changed conclusions flagged."""

    def testPlantedChangeIsFlaggedAndNoiseIsNot(self):
        import json
        from tt.report.experiments import report
        d = tempfile.mkdtemp(prefix="ttReport")
        rng = np.random.default_rng(1)
        try:
            def run(path, base, shift=0.0):
                os.makedirs(path)
                folds = []
                for i in range(8):
                    r = base + shift + rng.normal(0, .005)
                    p = base + rng.normal(0, .005)
                    folds.append({"block": "x_b%d%d" % (i // 4, i % 4),
                                  "crowns": 100, "recall": r, "precision": p,
                                  "f1": 2 * r * p / (r + p)})
                json.dump({"pooled": {"f1": base, "recall": base,
                                      "precision": base}, "folds": folds},
                          open(os.path.join(path, "results.json"), "w"))
            s = "20230712_afcamoisan_itrf20"
            for m, b in (("cc", .90), ("lmvw", .85), ("watershed", .86),
                         ("mrcnn", .88)):
                run(os.path.join(d, "study_procedure", s, "runs", m + "Lidar"), b)
                run(os.path.join(d, "study_procedureP1", s, "runs", m + "P1"), b)
                run(os.path.join(d, "study_edge", s, "runs", m + "Lidar"), b)
                run(os.path.join(d, "study_edge", s, "runs", m + "P1"), b,
                    0.06 if m == "lmvw" else 0.0)
            lines = report(d, np.random.default_rng(0), 500)
            changed = lines[1:lines.index("")]
            self.assertTrue(any("lmvw: P1 minus Lidar" in c for c in changed))
            self.assertFalse(any("cc: P1 minus Lidar" in c for c in changed))
            self.assertTrue(any(l.startswith("ccLidar") for l in lines))
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
