#!/usr/bin/env python
"""
Tests for tt.

    python tests.py                 the synthetic tests, no data needed
    TT_CHM=chm.tif TT_CROWNS=crowns.shp TT_BOUNDARY=area.shp python tests.py

Most of this builds its own two-tree world in a temporary directory, so the
behaviour being checked is behaviour with a known right answer rather than
whatever the real data happens to produce. The last test is different: it
re-runs the published operating point on the real scene and checks the numbers
still come out, which is what actually catches a refactor having changed
something subtle.
"""

import argparse
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

# Run from anywhere: the package sits next to this file, and requiring an
# editable install just to run the tests is friction with no upside.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------- #
# a synthetic world with known answers
# ---------------------------------------------------------------------- #

def buildFixture(directory, trees, resolution=0.25, size=240):
    """
    Write a CHM and crown shapefile for `trees`, each (columnPx, rowPx,
    radiusM, heightM). Cones on a flat floor: crisp apexes at known pixels.
    """
    import geopandas as gpd
    import rasterio
    from affine import Affine
    from shapely.geometry import Point, box

    originX, originY = 500000.0, 5000000.0
    rows, columns = np.mgrid[0:size, 0:size]
    chm = np.zeros((size, size), np.float32)
    records = []

    for columnPx, rowPx, radiusM, heightM in trees:
        radiusPx = radiusM / resolution
        distance = np.hypot(rows - rowPx, columns - columnPx)
        chm = np.maximum(chm, heightM * np.clip(1 - distance / radiusPx, 0, 1))
        east = originX + (columnPx + 0.5) * resolution
        north = originY - (rowPx + 0.5) * resolution
        records.append({"geometry": Point(east, north).buffer(radiusM * 0.95),
                        "tree_class": 1})

    # -3.4028235e+38 is a rounded-up float64 literal that sits just below
    # float32's minimum, and newer rasterio rejects it. A plain sentinel avoids
    # the whole question.
    nodata = -9999.0
    stored = chm.copy()
    stored[stored < 2.0] = nodata
    transform = Affine(resolution, 0.0, originX, 0.0, -resolution, originY)

    chmPath = os.path.join(directory, "chm.tif")
    with rasterio.open(chmPath, "w", driver="GTiff", height=size, width=size,
                       count=1, dtype="float32", crs="EPSG:32648",
                       transform=transform, nodata=nodata) as destination:
        destination.write(stored, 1)

    crownsPath = os.path.join(directory, "crowns.shp")
    gpd.GeoDataFrame(records, crs="EPSG:32648").to_file(crownsPath)

    boundaryPath = os.path.join(directory, "area.shp")
    gpd.GeoDataFrame(
        [{"geometry": box(originX, originY - size * resolution,
                          originX + size * resolution, originY)}],
        crs="EPSG:32648").to_file(boundaryPath)

    return chmPath, crownsPath, boundaryPath


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

    def scene(self, **keywords):
        from tt import Scene
        return Scene(self.chmPath, crownsPath=self.crownsPath,
                     boundaryPath=self.boundaryPath, verbose=False, **keywords)


# ---------------------------------------------------------------------- #

class TestScene(Fixture):

    def testLoadsOnOneGrid(self):
        scene = self.scene()
        self.assertEqual(scene.chm.shape, (240, 240))
        self.assertAlmostEqual(scene.pixelSize, 0.25, places=6)
        self.assertEqual(len(scene), 9)
        self.assertEqual(str(scene.crs), "EPSG:32648")

    def testBelowMinHeightIsZeroed(self):
        scene = self.scene(minHeight=8.0)
        canopy = scene.chm[scene.chm > 0]
        self.assertTrue((canopy >= 8.0).all(),
                        "heights below minHeight must not survive loading")

    def testRoundTripThroughWorldCoordinates(self):
        scene = self.scene()
        east, north = scene.toWorld(100, 50)
        column, row = scene.toPixel(east, north)
        self.assertAlmostEqual(column, 100.5, places=3)
        self.assertAlmostEqual(row, 50.5, places=3)

    def testCrownPeaksSitOnTheApexes(self):
        """Each crown's peak must be the height the tree was built with."""
        scene = self.scene()
        points, heights = scene.crownPeaks()
        self.assertEqual(len(points), 9)
        built = sorted(height for _, _, _, height in self.trees)
        for expected, found in zip(built, sorted(heights)):
            self.assertAlmostEqual(expected, found, delta=0.15)

    def testCrownShiftMovesThePolygons(self):
        plain = self.scene()
        shifted = self.scene(crownShiftEast=2.0, crownShiftSouth=0.0)
        before = plain.crowns.geometry.iloc[0].centroid.x
        after = shifted.crowns.geometry.iloc[0].centroid.x
        self.assertAlmostEqual(after - before, -2.0, places=3,
                               msg="a crown reported 2 m east of the CHM must "
                                   "move 2 m west to correct it")


class TestTops(Fixture):

    def testWriteAndReadAreInverse(self):
        from tt import Tops
        original = Tops([(10, 20, 15.5), (30, 40, 12.25)])
        path = os.path.join(self.directory, "tops.txt")
        original.writeList(path)
        recovered = Tops.read(path)
        self.assertEqual(len(recovered), 2)
        for first, second in zip(original.points, recovered.points):
            self.assertEqual(first[0], second[0])
            self.assertEqual(first[1], second[1])
            self.assertAlmostEqual(first[2], second[2], places=3)

    def testMaskHasOneBlackDiscPerTop(self):
        import cv2
        from tt import Tops
        path = os.path.join(self.directory, "mask.png")
        Tops([(20, 20, 10.0), (60, 60, 12.0)]).writeMask(path, (100, 100),
                                                         radius=3)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        count, _ = cv2.connectedComponents((mask == 0).astype(np.uint8))
        self.assertEqual(count - 1, 2)


class TestMerger(unittest.TestCase):

    def testMetricsOrderAsExpected(self):
        from tt import TopMerger
        flat = TopMerger("2d", epsM=1.0)
        self.assertAlmostEqual(flat.distance(0.7, 3.0), 0.7)

        cube = TopMerger("3d", epsM=1.0)
        self.assertAlmostEqual(cube.distance(0.7, 3.0), np.hypot(0.7, 3.0))

        mixed = TopMerger("composite", epsM=1.0, heightWeight=0.7)
        self.assertAlmostEqual(mixed.distance(0.7, 3.0), 0.7 * 3.0 + 0.3 * 0.7)

    def testCompositeReachIsWiderThanEpsilon(self):
        """The trap: a weighted sum lets tops merge far beyond eps."""
        from tt import TopMerger
        merger = TopMerger("composite", epsM=1.0, heightWeight=0.7)
        self.assertAlmostEqual(merger.horizontalReachM, 1.0 / 0.3, places=6)
        self.assertTrue(merger.areSame(3.0, 0.0),
                        "equal-height tops 3 m apart merge at eps 1 m")

    def testHeightSeparatesWhatDistanceWouldMerge(self):
        from tt import TopMerger
        merger = TopMerger("composite", epsM=1.0, heightWeight=0.7)
        self.assertTrue(merger.areSame(0.7, 0.0))
        self.assertFalse(merger.areSame(0.7, 3.0))

    def testSaddleDropFindsTheValley(self):
        from tt.merging import saddleDrop
        surface = np.zeros((10, 40), np.float32)
        surface[5, :] = 20.0
        surface[5, 18:22] = 14.0          # a 6 m notch between two peaks
        self.assertAlmostEqual(saddleDrop(surface, 5, 5, 5, 35), 6.0, places=3)
        self.assertAlmostEqual(saddleDrop(surface, 5, 5, 5, 10), 0.0, places=3)

    def testSaddleRefusesAcrossAValley(self):
        from tt import TopMerger
        surface = np.zeros((10, 40), np.float32)
        surface[5, :] = 20.0
        surface[5, 18:22] = 14.0
        merger = TopMerger("saddle", epsM=100.0, saddleDropM=1.0)
        self.assertFalse(merger.areSame(7.5, 0.0, surface=surface,
                                        a=(5, 5), b=(5, 35)))
        self.assertTrue(merger.areSame(1.25, 0.0, surface=surface,
                                       a=(5, 5), b=(5, 10)))


class TestDetector(Fixture):

    def testFindsEveryTreeOnACleanScene(self):
        from tt import ConCompDetector, CrownEvaluator, TopMerger
        scene = self.scene()
        detector = ConCompDetector(
            lowerPercentile=10, minTopAreaM2=0.12, topStepM=0.12,
            erosionIterations=1, minTreeAreaM2=0.5, windowSizeM=30.0,
            merger=TopMerger("saddle", epsM=8.0, saddleDropM=0.5),
            verbose=False)
        result = CrownEvaluator(scene).score(detector.detect(scene))
        self.assertEqual(result["recall"], 1.0,
                         "nine separated trees should all be found")
        self.assertGreaterEqual(result["precision"], 0.5)

    def testMergingNeverAddsDetections(self):
        from tt import ConCompDetector, TopMerger
        scene = self.scene()
        settings = dict(lowerPercentile=10, minTopAreaM2=0.12, topStepM=0.12,
                        erosionIterations=1, minTreeAreaM2=0.5,
                        windowSizeM=30.0, verbose=False)
        raw = ConCompDetector(refine=False, **settings).detect(scene)
        merged = ConCompDetector(
            merger=TopMerger("saddle", epsM=8.0, saddleDropM=0.5),
            **settings).detect(scene)
        self.assertLessEqual(len(merged), len(raw))

    def testDetectionsCarryTheirHeight(self):
        from tt import ConCompDetector
        scene = self.scene()
        tops = ConCompDetector(windowSizeM=30.0, verbose=False).detect(scene)
        for x, y, height in tops.points:
            self.assertAlmostEqual(height, float(scene.chm[y, x]), places=5)


class TestEvaluator(Fixture):

    def testHitRepeatAndBackgroundAreDistinguished(self):
        from tt import CrownEvaluator, Tops
        scene = self.scene()
        evaluator = CrownEvaluator(scene)

        points, _ = scene.crownPeaks()
        first = points[0]
        # one on a crown apex, one beside it in the same crown, one far away
        beside = (first[0] + 2, first[1] + 2)
        empty = (5, 5)
        tops = Tops([(first[0], first[1], 20.0),
                     (beside[0], beside[1], 10.0),
                     (empty[0], empty[1], 3.0)])

        classification = evaluator.classify(tops)
        self.assertEqual(classification.kinds[0], "hit")
        self.assertEqual(classification.kinds[1], "repeat")
        self.assertEqual(classification.kinds[2], "background")

    def testPerfectDetectionScoresOne(self):
        from tt import CrownEvaluator, Tops
        scene = self.scene()
        points, heights = scene.crownPeaks()
        tops = Tops([(p[0], p[1], h) for p, h in zip(points, heights)])
        result = CrownEvaluator(scene).score(tops)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)

    def testBackgroundSplitsByLocalCanopyHeight(self):
        from tt import CrownEvaluator, Tops
        scene = self.scene()
        evaluator = CrownEvaluator(scene)
        points, heights = scene.crownPeaks()

        entries = [(p[0], p[1], h) for p, h in zip(points, heights)]
        entries.append((points[0][0] + 24, points[0][1], 14.0))  # tall, no crown
        entries.append((points[0][0] + 28, points[0][1], 2.5))   # low, no crown
        tops = Tops(entries)

        classification = evaluator.classify(tops)
        split = evaluator.splitBackground(tops, classification)
        self.assertIn(len(entries) - 2, split["canopy"])
        self.assertIn(len(entries) - 1, split["low"])


class TestInvariants(Fixture):
    """
    Properties found broken during the refactor, pinned so they stay fixed.
    """

    def testSaddleDropIsTranslationInvariant(self):
        """np.rint rounds half to even, so rounding absolute coordinates made
        the verdict depend on where a window started."""
        from tt.merging import saddleDrop
        rng = np.random.default_rng(3)
        surface = rng.random((200, 200)).astype(np.float32)
        for _ in range(2000):
            a, b = rng.integers(40, 160, 2), rng.integers(40, 160, 2)
            dy, dx = rng.integers(0, 40, 2)
            self.assertEqual(
                saddleDrop(surface, a[0], a[1], b[0], b[1]),
                saddleDrop(surface[dy:, dx:], a[0] - dy, a[1] - dx,
                           b[0] - dy, b[1] - dx))

    def testCroppedDescentMatchesFullWindow(self):
        """The detector crops each blob to its box before descending; that
        must change nothing."""
        import cv2
        from tt import ConCompDetector, TopMerger
        scene = self.scene()
        detector = ConCompDetector(windowSizeM=30.0, verbose=False,
                                   merger=TopMerger("saddle", epsM=8.0,
                                                    saddleDropM=0.5))
        geometry = detector._pixelParameters(scene)
        for column, row, window in detector._windows(scene, geometry):
            result = detector._stretch(window)
            if result is None:
                continue
            stretched, perLevel = result
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                stretched, connectivity=8)
            for label in range(1, count):
                if stats[label, 4] <= geometry["minPixTree"]:
                    continue
                x0, y0, w, h = stats[label, :4]
                full = stretched.copy()
                full[labels != label] = 0
                whole = sorted(detector._descend(scene, full, window,
                                                 perLevel, geometry))
                cropped = sorted(
                    (r + y0, c + x0) for r, c in detector._descend(
                        scene, full[y0:y0 + h, x0:x0 + w],
                        window[y0:y0 + h, x0:x0 + w], perLevel, geometry))
                self.assertEqual(whole, cropped)

    def testThresholdCurveMatchesRescoring(self):
        """One spatial join answers every threshold; it must agree with
        re-scoring at each one."""
        from tt.dl import dlCommon as dc
        from shapely.geometry import box as shapelyBox
        scene = self.scene()
        rng = np.random.default_rng(5)
        points, _ = scene.crownPeaks()
        predictions = []
        for column, row in points:
            for _ in range(3):
                east, north = scene.toWorld(column + rng.integers(-6, 7),
                                            row + rng.integers(-6, 7))
                predictions.append({"centreX": east, "centreY": north,
                                    "score": float(rng.random()),
                                    "box": [east, north, east, north]})
        region = shapelyBox(*scene.boundary.bounds)
        curve = dict(dc.thresholdCurve(predictions, scene.crowns,
                                       dc.THRESHOLD_CANDIDATES))
        # the curve holds the tuning objective (TT_OBJECTIVE), F1 by default
        for threshold, value in curve.items():
            kept = [p for p in predictions if p["score"] >= threshold]
            self.assertAlmostEqual(
                value, dc.objectiveOf(dc.evaluateDetections(kept, scene.crowns,
                                                            region)),
                places=12)

    def testOneScorerEverywhere(self):
        """CrownEvaluator and the benchmark scorer must agree exactly."""
        from shapely.geometry import box as shapelyBox
        from tt import ConCompDetector, CrownEvaluator
        from tt.dl import dlCommon as dc
        scene = self.scene()
        tops = ConCompDetector(windowSizeM=30.0, verbose=False).detect(scene)
        local = CrownEvaluator(scene).score(tops)
        world = tops.world(scene)
        benchmark = dc.evaluateDetections(
            [{"centreX": x, "centreY": y, "score": h}
             for (x, y), h in zip(world, tops.heights)],
            scene.crowns, shapelyBox(*scene.boundary.bounds))
        self.assertAlmostEqual(local["recall"], benchmark["recall"])
        self.assertAlmostEqual(local["precision"], benchmark["precision"])

    def testStitchingKeepsOutlines(self):
        from affine import Affine
        from tt.dl import dlCommon as dc
        transform = Affine(0.1, 0, 500000, 0, -0.1, 5000000)
        world = dc.tileToWorld([{"box": [10, 20, 30, 40], "score": 0.9,
                                 "polygon": [[10, 20], [30, 20], [30, 40]]}],
                               transform, 100, 200)[0]
        self.assertAlmostEqual(world["polygon"][0][0], world["box"][0])
        self.assertAlmostEqual(world["polygon"][0][1], world["box"][3])


class TestFusion(unittest.TestCase):
    """The combination strategies, on hand-built boxes and points."""

    @staticmethod
    def box(x0, y0, x1, y1, score):
        return {"box": [x0, y0, x1, y1], "score": score,
                "centreX": (x0 + x1) / 2.0, "centreY": (y0 + y1) / 2.0}

    @staticmethod
    def point(x, y, score=10.0):
        return {"centreX": x, "centreY": y, "score": score,
                "box": [x, y, x, y]}

    def setUp(self):
        self.boxes = [self.box(0, 0, 4, 4, 0.9),     # holds two points
                      self.box(10, 0, 14, 4, 0.6),   # holds none
                      self.box(20, 0, 24, 4, 0.3)]   # holds one, low score
        self.points = [self.point(1, 1, 12.0), self.point(3, 3, 15.0),
                       self.point(21, 1), self.point(40, 40)]

    def testContainment(self):
        from tt.fusion import containment
        inside = containment(self.points, self.boxes)
        self.assertEqual(inside.tolist(),
                         [[True, False, False], [True, False, False],
                          [False, False, True], [False, False, False]])

    def testWeakBoxClearOfStrongOnesMerges(self):
        """A weak box overlapping no stronger box merges the points in it."""
        from tt.fusion import weakBoxMergedPoints
        boxes = [self.box(0, 0, 4, 4, 0.9), self.box(30, 0, 34, 4, 0.1)]
        points = [self.point(1, 1, 12.0), self.point(31, 1, 9.0),
                  self.point(33, 3, 11.0)]
        result = weakBoxMergedPoints(boxes, points, threshold=0.5)
        self.assertEqual(sorted(p["centreX"] for p in result), [1, 33])

    def testWeakBoxOverlappingTooMuchIsIgnored(self):
        """Half of the weak box lies in a strong one: more than 25%."""
        from tt.fusion import weakBoxMergedPoints
        boxes = [self.box(0, 0, 4, 4, 0.9), self.box(2, 0, 6, 4, 0.1)]
        points = [self.point(5, 1, 9.0), self.point(5.5, 3, 11.0)]
        result = weakBoxMergedPoints(boxes, points, threshold=0.5)
        self.assertEqual(len(result), 2)

    def testOverlapIsMeasuredAgainstTheSmallerBox(self):
        """A small box inside a big one overlaps fully, whatever the IoU."""
        from tt.fusion import overlapOverSmaller
        share = overlapOverSmaller([1, 1, 2, 2], np.array([[0, 0, 10, 10]]))
        self.assertAlmostEqual(float(share[0]), 1.0)
        share = overlapOverSmaller([0, 0, 4, 4], np.array([[3, 0, 7, 4]]))
        self.assertAlmostEqual(float(share[0]), 0.25)

    def testWeakBoxesBlockEachOther(self):
        """Taken strongest first, an accepted weak box blocks a weaker one."""
        from tt.fusion import weakBoxMergedPoints
        boxes = [self.box(0, 0, 4, 4, 0.3), self.box(2, 0, 6, 4, 0.2)]
        points = [self.point(1, 1, 9.0), self.point(3, 3, 11.0),
                  self.point(5, 1, 8.0), self.point(5.5, 3, 7.0)]
        result = weakBoxMergedPoints(boxes, points, threshold=0.5)
        # the 0.3 box is accepted and merges its two points to x=3; the 0.2
        # box overlaps it by half and is refused, so its points both survive
        self.assertEqual(sorted(p["centreX"] for p in result), [3, 5, 5.5])

    def testConfirmedUsesTwoThresholds(self):
        from tt.fusion import confirmed
        kept = confirmed(self.boxes, self.points, confirmedAt=0.2,
                         unconfirmedAt=0.5)
        self.assertEqual([b["score"] for b in kept], [0.9, 0.6, 0.3])
        kept = confirmed(self.boxes, self.points, confirmedAt=0.5,
                         unconfirmedAt=0.7)
        self.assertEqual([b["score"] for b in kept], [0.9])

    def testUnionAddsOnlyUncoveredPoints(self):
        from tt.fusion import union
        result = union(self.boxes, self.points, threshold=0.5)
        # boxes 0.9 and 0.6 kept; the points in box 0.9 are covered
        self.assertEqual(len(result), 2 + 2)
        self.assertEqual(sorted(p["centreX"] for p in result[2:]), [21, 40])

    def testBoxMergesPointsItHolds(self):
        from tt.fusion import boxMergedPoints
        result = boxMergedPoints(self.boxes, self.points, threshold=0.2)
        xs = sorted(p["centreX"] for p in result)
        # the two points in the first box collapse to the higher (x=3)
        self.assertEqual(xs, [3, 21, 40])


class TestSaddleUnion(unittest.TestCase):
    """
    Two cones on a 40 m grid, A (16 m) and B (14 m), with bare ground between.
    A's box is drawn wide enough that its corner covers B's top — the case box
    containment gets wrong.
    """

    def setUp(self):
        from affine import Affine
        from tt.fusion import SaddleSurface
        rows, columns = np.mgrid[0:40, 0:40]
        chm = np.zeros((40, 40), np.float32)
        for row, column, height in ((20, 10, 16.0), (20, 25, 14.0)):
            distance = np.hypot(rows - row, columns - column)
            chm = np.maximum(chm, height * np.clip(1 - distance / 7.0, 0, 1))
        self.surface = SaddleSurface(chm, Affine(1, 0, 0, 0, -1, 40), 1.0,
                                     dropM=0.5)

    @staticmethod
    def at(row, column, score=10.0):
        x, y = column + 0.5, 40 - row - 0.5
        return {"centreX": x, "centreY": y, "score": score,
                "box": [x, y, x, y]}

    @staticmethod
    def box(c0, c1, r0, r1, score):
        return {"box": [c0, 40 - r1, c1, 40 - r0], "score": score,
                "centreX": (c0 + c1) / 2.0, "centreY": 40 - (r0 + r1) / 2.0}

    def testBoxContainmentLosesTheNeighbour(self):
        from tt.fusion import union
        wide = self.box(3, 27, 13, 27, 0.9)
        self.assertEqual(len(union([wide], [self.at(20, 25)], 0.5)), 1)

    def testSaddleKeepsTheNeighbour(self):
        from tt.fusion import unionSaddleCross
        wide = self.box(3, 27, 13, 27, 0.9)
        result = unionSaddleCross([wide], [self.at(20, 25)], 0.5,
                                  self.surface)
        self.assertEqual(len(result), 2)

    def testSaddleDropsTheTopOfTheBoxesOwnTree(self):
        from tt.fusion import unionSaddleCross
        wide = self.box(3, 27, 13, 27, 0.9)
        result = unionSaddleCross([wide], [self.at(20, 10)], 0.5,
                                  self.surface)
        self.assertEqual(len(result), 1)

    def testPoolMergesASplitCrownButCrossDoesNot(self):
        from tt.fusion import unionSaddleCross, unionSaddlePool
        halves = [self.box(3, 11, 13, 27, 0.9), self.box(11, 17, 13, 27, 0.6)]
        self.assertEqual(len(unionSaddlePool(halves, [], 0.5,
                                             self.surface)), 1)
        self.assertEqual(len(unionSaddleCross(halves, [], 0.5,
                                              self.surface)), 2)

    def testApexFollowsTheMaskOutline(self):
        """Box over both cones; the outline around B puts the apex on B."""
        wide = self.box(3, 32, 13, 27, 0.9)
        self.assertEqual(self.surface.apex(dict(wide)), (20, 10))
        ring = [[19, 40 - 16], [31, 40 - 16], [31, 40 - 24], [19, 40 - 24]]
        self.assertEqual(self.surface.apex(dict(wide, polygon=ring)),
                         (20, 25))

    def testPoolKeepsTwoTreesApart(self):
        from tt.fusion import unionSaddlePool
        boxes = [self.box(3, 17, 13, 27, 0.9), self.box(18, 32, 13, 27, 0.8)]
        self.assertEqual(len(unionSaddlePool(boxes, [], 0.5,
                                             self.surface)), 2)


class TestCalibration(Fixture):
    """
    The calibration's parts on the nine-tree scene, with zones drawn around
    the known crowns so every expected answer is known.
    """

    def zones(self, extra=()):
        from tt.calibration import ConfidentZones
        scene = self.scene()
        predictions = [{"box": list(g.bounds), "score": 0.95,
                        "centreX": g.centroid.x, "centreY": g.centroid.y}
                       for g in scene.crowns.geometry] + list(extra)
        region = scene.boundary
        return scene, ConfidentZones(predictions, scene, 0.9, region)

    def testZonesSitOnTheApexes(self):
        scene, zones = self.zones()
        _, heights = scene.crownPeaks()
        self.assertEqual(len(zones), 9)
        self.assertEqual(sorted(round(h, 3) for _, _, h in zones.apexes),
                         sorted(round(float(h), 3) for h in heights))

    def testLowConfidenceBoxesAreNotZones(self):
        scene, zones = self.zones(extra=[{"box": [0, 0, 1, 1], "score": 0.5,
                                          "centreX": 0.5, "centreY": 0.5}])
        self.assertEqual(len(zones), 9)

    def testProposalIsWithinItsBounds(self):
        from tt.calibration import (ParameterProposal, EROSION_MAX,
                                    TOP_STEP_CAP_M)
        _, zones = self.zones()
        original = ParameterProposal(zones, "original").settings()
        self.assertTrue(1 <= original["lowerPercentile"] <= 50)
        self.assertTrue(0 <= original["erosionIterations"] <= EROSION_MAX)
        self.assertGreater(original["minTopAreaM2"], 0)
        self.assertAlmostEqual(original["topStepM"],
                               min(0.5, max(0.05,
                                            original["saddleDropM"] / 2)))
        corrected = ParameterProposal(zones, "corrected").settings()
        self.assertTrue(0 <= corrected["lowerPercentile"] <= 50)
        self.assertLessEqual(corrected["topStepM"], TOP_STEP_CAP_M + 1e-9)

    def testCrownBaseCutIsNeverAboveTheApexCut(self):
        from tt.calibration import ParameterProposal
        _, zones = self.zones()
        self.assertLessEqual(
            ParameterProposal(zones, "corrected").lowerPercentile(),
            ParameterProposal(zones, "original").lowerPercentile())

    def testMinTreeAreaReachesTheDetector(self):
        from tt.calibration import CalibratedDetector, buildDetector
        scene, zones = self.zones()
        self.assertAlmostEqual(CalibratedDetector(scene, zones, 0.1)
                               .minTreeAreaM2, 0.1)
        settings = {"lowerPercentile": 10, "erosionIterations": 1,
                    "minTopAreaM2": 0.12, "saddleDropM": 0.5, "topStepM": 0.25}
        detector = buildDetector(settings, 0.1)
        self.assertAlmostEqual(detector.minTreeAreaM2, 0.1)

    def testCheckCountsCoverageAndMultiplicity(self):
        from tt.calibration import ZoneCheck
        scene, zones = self.zones()
        points, _ = scene.crownPeaks()
        tops = []
        for column, row in points:
            east, north = scene.toWorld(column, row)
            tops.append({"centreX": east, "centreY": north, "score": 10.0})
        check = ZoneCheck(tops[1:], zones)     # tree 0 has no top
        self.assertAlmostEqual(check.coverage, 8 / 9.0)
        self.assertAlmostEqual(check.multiplicity, 0.0)
        first = tops[0]
        tops.append(dict(first, centreX=first["centreX"] + 0.3, score=9.0))
        check = ZoneCheck(tops, zones)         # tree 0 has two
        self.assertAlmostEqual(check.coverage, 1.0)
        self.assertAlmostEqual(check.multiplicity, 1 / 9.0)

    def testFinalMergeLeavesOneTopPerZone(self):
        from tt.calibration import ZoneCheck, finalMerge
        scene, zones = self.zones()
        points, _ = scene.crownPeaks()
        tops = []
        for column, row in points:
            east, north = scene.toWorld(column, row)
            tops.append({"centreX": east, "centreY": north, "score": 10.0})
            tops.append({"centreX": east + 0.3, "centreY": north,
                         "score": 9.0})
        tops.append({"centreX": 1.0, "centreY": 1.0, "score": 3.0})
        merged = finalMerge(tops, zones)
        self.assertEqual(len(merged), 9 + 1)
        self.assertAlmostEqual(ZoneCheck(merged, zones).multiplicity, 0.0)

    def testCalibratedRunFindsEveryTree(self):
        from tt import CrownEvaluator
        from tt.calibration import CalibratedDetector
        from shapely.geometry import box as shapelyBox
        from tt.dl import dlCommon as dc
        scene, zones = self.zones()
        calibrated = CalibratedDetector(scene, zones)
        predictions = calibrated.run()
        result = dc.evaluateDetections(predictions, scene.crowns,
                                       shapelyBox(*scene.boundary.bounds))
        self.assertEqual(result["recall"], 1.0)
        self.assertIn("proposed", calibrated.log)


class TestBlockImages(Fixture):
    """Fates in the images must be the scorer's own classification."""

    def renderer(self):
        from shapely.geometry import box as shapelyBox
        from tt.blockImages import BlockRenderer
        scene = self.scene()
        return scene, BlockRenderer(scene, [("all", shapelyBox(
            *scene.boundary.bounds))], minSidePx=200)

    def detections(self, scene):
        points, _ = scene.crownPeaks()
        out = []
        for column, row in points:
            east, north = scene.toWorld(column, row)
            out.append({"centreX": east, "centreY": north, "score": 10.0,
                        "box": [east, north, east, north]})
        return out

    def testFatesMatchTheScene(self):
        scene, renderer = self.renderer()
        predictions = self.detections(scene)
        extra = dict(predictions[0], centreX=predictions[0]["centreX"] + 0.3,
                     score=9.0)
        fates, missed, crowns = renderer.classify(
            predictions[1:] + [extra], "all")
        self.assertEqual(crowns, 9)
        self.assertEqual(fates.count("hit"), 9)
        self.assertEqual(missed, [])
        fates, missed, _ = renderer.classify(predictions[1:], "all")
        self.assertEqual(len(missed), 1)

    def testImageIsWritten(self):
        import tempfile
        scene, renderer = self.renderer()
        path = os.path.join(tempfile.mkdtemp(), "sub", "all.jpg")
        renderer.render(self.detections(scene), "all", "test", path)
        self.assertGreater(os.path.getsize(path), 1000)


class TestFusionOutputs(unittest.TestCase):
    """Each strategy's output is kept apart from its score."""

    def testOutputDoesNotReplaceTheDetectionCount(self):
        from tt.fusion import slim
        full = {"centreX": 1.0, "centreY": 2.0, "score": 0.5,
                "box": [0, 0, 2, 4], "polygon": [[0, 0]], "apex": (1, 1)}
        self.assertEqual(sorted(slim(full)), ["box", "centreX", "centreY",
                                              "score"])


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


class TestNeonImport(unittest.TestCase):
    """NEON pixel boxes land on the right ground, one folder per tile."""

    xml = ("<annotation><filename>PLOT_001_2019.tif</filename>"
           "<object><name>Tree</name><bndbox><xmin>10</xmin><ymin>20</ymin>"
           "<xmax>30</xmax><ymax>60</ymax></bndbox></object>"
           "<object><name>Tree</name><bndbox><xmin>0</xmin><ymin>0</ymin>"
           "<xmax>400</xmax><ymax>400</ymax></bndbox></object>"
           "</annotation>")

    def setUp(self):
        import rasterio
        from affine import Affine
        self.directory = tempfile.mkdtemp(prefix="ttNeon")
        for sub in ("annotations", "RGB"):
            os.makedirs(os.path.join(self.directory, sub))
        with open(os.path.join(self.directory, "annotations",
                               "PLOT_001_2019.xml"), "w") as handle:
            handle.write(self.xml)
        with open(os.path.join(self.directory, "annotations",
                               "ORPHAN.xml"), "w") as handle:
            handle.write(self.xml.replace("PLOT_001_2019", "MISSING"))
        transform = Affine(0.1, 0.0, 500000.0, 0.0, -0.1, 4000040.0)
        with rasterio.open(os.path.join(self.directory, "RGB",
                                        "PLOT_001_2019.tif"), "w",
                           driver="GTiff", height=400, width=400, count=3,
                           dtype="uint8", crs="EPSG:32618",
                           transform=transform) as destination:
            destination.write(np.zeros((3, 400, 400), dtype=np.uint8))

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def convert(self):
        from tt.neonImport import convert
        return convert(os.path.join(self.directory, "annotations"),
                       os.path.join(self.directory, "RGB"),
                       os.path.join(self.directory, "out"))

    def testBoxLandsOnItsGround(self):
        import geopandas as gpd
        rows = self.convert()
        crowns = gpd.read_file(rows[0]["crownPath"])
        np.testing.assert_allclose(crowns.geometry.iloc[0].bounds,
                                   (500001.0, 4000034.0, 500003.0, 4000038.0))
        self.assertAlmostEqual(crowns["area"].iloc[0], 8.0)

    def testBoundaryIsTheTileFootprint(self):
        import geopandas as gpd
        rows = self.convert()
        boundary = gpd.read_file(rows[0]["boundaryPath"])
        np.testing.assert_allclose(boundary.geometry.iloc[0].bounds,
                                   (500000.0, 4000000.0, 500040.0, 4000040.0))
        self.assertEqual(str(boundary.crs), "EPSG:32618")

    def testNestedAnnotationFolderIsSearched(self):
        from tt.neonImport import convert
        outer = os.path.join(self.directory, "outer")
        shutil.copytree(os.path.join(self.directory, "annotations"),
                        os.path.join(outer, "annotations"))
        rows = convert(outer, os.path.join(self.directory, "RGB"),
                       os.path.join(self.directory, "out"))
        self.assertEqual([r["tile"] for r in rows], ["PLOT_001_2019"])

    def testAnnotationWithoutTileIsSkipped(self):
        rows = self.convert()
        self.assertEqual([r["tile"] for r in rows], ["PLOT_001_2019"])
        self.assertEqual(rows[0]["crowns"], 2)


class TestNeonPrune(unittest.TestCase):
    """Only files of annotated tiles survive, and nothing without --delete."""

    files = ["evaluation/RGB/PLOT_001_2019.tif",
             "evaluation/CHM/PLOT_001_2019_CHM.tif",
             "evaluation/LiDAR/PLOT_001_2019.laz",
             "evaluation/Hyperspectral/PLOT_001_2019_hyperspectral.tif",
             "evaluation/RGB/PLOT_099_2019.tif",
             "training/RGB/2018_NIWO_2_450000_4426000_image_crop.tif",
             "training/CHM/2018_NIWO_2_450000_4426000_CHM.tif"]

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="ttPrune")
        annotations = os.path.join(self.directory, "annotations")
        os.makedirs(annotations)
        for tile in ("PLOT_001_2019", "2018_NIWO_2_450000_4426000_image_crop"):
            with open(os.path.join(annotations, tile + ".xml"), "w") as h:
                h.write("<annotation><filename>%s.tif</filename>"
                        "</annotation>" % tile)
        for name in self.files:
            path = os.path.join(self.directory, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").close()

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def prune(self, *extra):
        from tt.neonPrune import main
        return main(["--annotations", os.path.join(self.directory,
                                                   "annotations"),
                     "--roots", os.path.join(self.directory, "evaluation"),
                     os.path.join(self.directory, "training")] + list(extra))

    def remaining(self):
        return sorted(os.path.relpath(os.path.join(f, n), self.directory)
                      for f, _, ns in os.walk(self.directory) for n in ns
                      if not n.endswith(".xml"))

    def testDryRunRemovesNothing(self):
        self.prune()
        self.assertEqual(len(self.remaining()), len(self.files))

    def testDeleteKeepsOnlyAnnotatedTiles(self):
        self.prune("--delete")
        self.assertEqual(self.remaining(), sorted(
            ["evaluation/RGB/PLOT_001_2019.tif",
             "evaluation/CHM/PLOT_001_2019_CHM.tif",
             "evaluation/LiDAR/PLOT_001_2019.laz",
             "training/RGB/2018_NIWO_2_450000_4426000_image_crop.tif",
             "training/CHM/2018_NIWO_2_450000_4426000_CHM.tif"]))

    def testNoAnnotationsRemovesNothing(self):
        from tt.neonPrune import main
        empty = os.path.join(self.directory, "empty")
        os.makedirs(empty)
        code = main(["--annotations", empty, "--roots",
                     os.path.join(self.directory, "evaluation"), "--delete"])
        self.assertEqual(code, 1)
        self.assertEqual(len(self.remaining()), len(self.files))

    def testNestedAnnotationsAreFound(self):
        from tt.neonPrune import main
        nested = os.path.join(self.directory, "outer")
        shutil.copytree(os.path.join(self.directory, "annotations"),
                        os.path.join(nested, "annotations"))
        code = main(["--annotations", nested, "--roots",
                     os.path.join(self.directory, "evaluation"), "--delete"])
        self.assertEqual(code, 0)
        self.assertIn("evaluation/CHM/PLOT_001_2019_CHM.tif",
                      self.remaining())

    def testDropLidar(self):
        self.prune("--delete", "--dropLidar")
        self.assertNotIn("evaluation/LiDAR/PLOT_001_2019.laz",
                         self.remaining())


class TestNeonRun(Fixture):
    """The NEON loop finds each tile's CHM, scores it and pools the counts."""

    def testRunScoresEveryTileWithACHM(self):
        import csv
        from tt.neonRun import main
        root = os.path.join(self.directory, "neon")
        for sub in ("RGB", "CHM"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        shutil.copy(self.chmPath, os.path.join(root, "CHM", "ABCD_001_CHM.tif"))
        manifest = os.path.join(self.directory, "manifest.csv")
        with open(manifest, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "tile", "rgb", "crowns", "crownPath", "boundaryPath"])
            writer.writeheader()
            for tile in ("ABCD_001", "ABCD_002"):
                writer.writerow({"tile": tile,
                                 "rgb": os.path.join(root, "RGB",
                                                     tile + ".tif"),
                                 "crowns": 9, "crownPath": self.crownsPath,
                                 "boundaryPath": self.boundaryPath})
        output = os.path.join(self.directory, "neonOut")
        self.assertEqual(main(["--manifest", manifest, "--minHeight", "2",
                               "--output", output]), 0)
        from tt.dl.dlCommon import loadJson
        result = loadJson(os.path.join(output, "results.json"))
        self.assertEqual(result["skipped"], ["ABCD_002"])
        overall = result["pooled"]["defaults"]["overall"]
        self.assertEqual((overall["folds"], overall["crowns"]), (1, 9))
        self.assertGreater(overall["f1"], 0.8)
        self.assertIn("ABCD", result["pooled"]["defaults"]["sites"])


class TestNeonChm(unittest.TestCase):
    """A cone on sloping ground: the CHM is the cone, the slope is gone."""

    def setUp(self):
        try:
            import laspy
        except ImportError:
            self.skipTest("laspy not installed")
        import csv
        import rasterio
        from affine import Affine
        self.directory = tempfile.mkdtemp(prefix="ttChm")
        root = os.path.join(self.directory, "evaluation")
        for sub in ("RGB", "LiDAR", "CHM"):
            os.makedirs(os.path.join(root, sub))
        west, north = 500000.0, 4000020.0
        with rasterio.open(os.path.join(root, "RGB", "T_001.tif"), "w",
                           driver="GTiff", height=200, width=200, count=1,
                           dtype="uint8", crs="EPSG:32613",
                           transform=Affine(0.1, 0, west, 0, -0.1, north)
                           ) as destination:
            destination.write(np.zeros((1, 200, 200), dtype=np.uint8))
        rng = np.random.default_rng(0)
        x = west + rng.uniform(0, 20, 20000)
        y = north - rng.uniform(0, 20, 20000)
        ground = 3000.0 + 0.2 * (x - west)
        distance = np.hypot(x - west - 10, north - y - 10)
        cone = np.clip(8.0 - 2.0 * distance, 0.0, None)
        kind = np.where(cone > 0, 5, 2).astype(np.uint8)
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = [0.001, 0.001, 0.001]
        header.offsets = [west, north - 20, 3000.0]
        cloud = laspy.LasData(header)
        cloud.x, cloud.y, cloud.z = x, y, ground + cone
        cloud.classification = kind
        cloud.write(os.path.join(root, "LiDAR", "T_001.las"))
        self.manifest = os.path.join(self.directory, "manifest.csv")
        with open(self.manifest, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["tile", "rgb"])
            writer.writeheader()
            writer.writerow({"tile": "T_001", "rgb": os.path.join(
                root, "RGB", "T_001.tif")})

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def testConeStandsOnFlatGround(self):
        import rasterio
        from tt.neonChm import build
        rows = build(self.manifest, 0.5, "CHM050")
        with rasterio.open(rows[0]["chm"]) as source:
            chm = source.read(1)
            self.assertEqual(chm.shape, (40, 40))
            self.assertAlmostEqual(source.transform.a, 0.5)
        self.assertAlmostEqual(float(chm.max()), 8.0, delta=0.6)
        self.assertLess(float(chm[0, 0]), 0.2)
        self.assertLess(float(chm[-1, -1]), 0.2)
        self.assertFalse(rows[0]["sparse"])


class TestNeonCv(Fixture):
    """Leave-one-site-out picks each site's setting from the other sites."""

    def manifest(self):
        import csv
        root = os.path.join(self.directory, "cvNeon")
        for sub in ("RGB", "CHM050"):
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        path = os.path.join(self.directory, "cvManifest.csv")
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "tile", "rgb", "crowns", "crownPath", "boundaryPath"])
            writer.writeheader()
            for tile in ("AAAA_001", "BBBB_001", "BBBB_002"):
                shutil.copy(self.chmPath, os.path.join(
                    root, "CHM050", tile + "_CHM.tif"))
                writer.writerow({"tile": tile, "rgb": os.path.join(
                    root, "RGB", tile + ".tif"), "crowns": 9,
                    "crownPath": self.crownsPath,
                    "boundaryPath": self.boundaryPath})
        return path

    def testFoldsPerSiteAndCachedCounts(self):
        from tt.dl.dlCommon import loadJson
        from tt.neonCv import main
        output = os.path.join(self.directory, "cvOut")
        arguments = ["--manifest", self.manifest(), "--minHeight", "2",
                     "--percentiles", "10", "--minTopAreas", "0.12",
                     "--topSteps", "0.12", "--erosions", "0,1",
                     "--saddleDrops", "0.5", "--jobs", "1",
                     "--output", output]
        self.assertEqual(main(arguments), 0)
        result = loadJson(os.path.join(output, "results.json"))
        self.assertEqual([f["block"] for f in result["folds"]],
                         ["AAAA", "BBBB"])
        self.assertEqual([f["crowns"] for f in result["folds"]], [9, 18])
        self.assertGreater(result["pooled"]["f1"], 0.8)
        self.assertEqual(len(result["grid"]), 2)
        self.assertEqual(main(arguments), 0)   # second run: from the cache

    def testHeldOutSiteNeverChoosesItsOwnSetting(self):
        from tt.neonCv import crossValidate
        from tt.transfer import DEFAULTS
        grid = [dict(a=1), dict(a=2), dict(DEFAULTS)]
        good, bad, half = [10, 10, 10, 0, 0], [10, 10, 0, 0, 10], \
            [10, 10, 5, 0, 5]
        # setting 0 is perfect on A, B and C; setting 1 only on D
        counts = np.array([[good, bad, half], [good, bad, half],
                           [good, bad, half], [bad, good, half]], dtype=float)
        tuned, default = crossValidate(["AAAA_1", "BBBB_1", "CCCC_1",
                                        "DDDD_1"], counts, grid)
        self.assertEqual([f["settings"] for f in tuned], [dict(a=1)] * 4)
        self.assertEqual([f["f1"] for f in tuned], [1.0, 1.0, 1.0, 0.0])
        self.assertEqual([f["f1"] for f in default], [0.5] * 4)


class TestQpPrepare(unittest.TestCase):
    """Quebec Plantations prep on a synthetic site, with a stand-in PDAL."""

    site = "20990101_testsite"
    trees = ((10.0, 10.0, 3.0), (27.0, 12.0, 4.0), (60.0, 30.0, 2.5))
    unannotated = ((12.5, 10.0, 3.0),)   # in the cloud, in no crown
    hidden = ((45.0, 12.0, 0.6),)        # a crown whose top is below 1 m

    def setUp(self):
        try:
            import laspy
        except ImportError:
            self.skipTest("laspy not installed")
        import geopandas as gpd
        import rasterio
        from affine import Affine
        from shapely.geometry import Point
        self.directory = tempfile.mkdtemp(prefix="ttQp")
        self.west, self.south = 300000.0, 5000000.0
        centres = [Point(self.west + x, self.south + y) for x, y, _ in self.trees]
        thicket = Point(self.west + 40, self.south + 30).buffer(1.5)
        low = [Point(self.west + x, self.south + y).buffer(1.0)
               for x, y, _ in self.hidden]
        crowns = gpd.GeoDataFrame(
            {"class_code": ["PIGL"] * 3 + ["other"] + ["PIGL"] * len(low)},
            geometry=[c.buffer(1.0) for c in centres] + [thicket] + low,
            crs="EPSG:32619")
        points = gpd.GeoDataFrame(
            {"class_code": ["PIGL"] * 3,
             "total_height1_cm": [str(int(h * 100)) for *_, h in self.trees],
             "total_height2_cm": ["NA", str(int(self.trees[1][2] * 90)), "NA"],
             "height1_no_shoot_cm": [str(int(h * 80)) for *_, h in self.trees],
             "height2_no_shoot_cm": ["0", "0", "0"],        # placeholders
             "date_mesured": ["2023-07-29 10:00:00"] * 2 + ["2023-10-11 10:00:00"]},
            geometry=centres, crs="EPSG:32619")
        vectors = os.path.join(self.directory, "vectors")
        os.makedirs(vectors)
        gpkg = os.path.join(vectors, self.site + "_p1.gpkg")
        crowns.to_file(gpkg, layer=self.site + "_labels_poly")
        points.to_file(gpkg, layer=self.site + "_labels_pts")
        self.lidar = os.path.join(self.directory, "cloud.las")
        self.writeCloud(laspy)
        self.rgb = os.path.join(self.directory, "rgb.tif")
        with rasterio.open(self.rgb, "w", driver="GTiff", height=800,
                           width=800, count=4, dtype="uint8", crs="EPSG:32619",
                           transform=Affine(0.1, 0, self.west, 0, -0.1,
                                            self.south + 80)) as d:
            d.write(np.full((4, 800, 800), 120, np.uint8))
        self.vectors = vectors

    def writeCloud(self, laspy):
        rng = np.random.default_rng(1)
        x = self.west + rng.uniform(0, 80, 200000)
        y = self.south + rng.uniform(0, 80, 200000)
        z = np.full(x.shape, 100.0)
        for tx, ty, h in self.trees + self.unannotated + self.hidden:
            d = np.hypot(x - self.west - tx, y - self.south - ty)
            z = np.maximum(z, 100.0 + h * np.clip(1 - d / 1.5, 0, None))
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales, header.offsets = [0.001] * 3, [self.west, self.south, 0]
        cloud = laspy.LasData(header)
        cloud.x, cloud.y, cloud.z = x, y, z
        cloud.write(self.lidar)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def arguments(self):
        from tt.qpPrepare import parseArguments
        fake = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "tests_support", "fakePdal.py")
        args = parseArguments(["--vectors", self.vectors, "--pdal", fake,
                               "--output", os.path.join(self.directory, "out"),
                               "--chmResolution", "0.1", "--tileM", "20"])
        args.lidarSource = lambda site: self.lidar
        args.rgbSource = lambda site: self.rgb
        return args

    def testSiteProducts(self):
        import geopandas as gpd
        import rasterio
        from shapely.geometry import Point
        from tt.qpPrepare import prepareSite
        check = prepareSite(self.site, self.arguments())
        out = os.path.join(self.directory, "out", self.site)
        for name in ("area.shp", "chm.tif", "rgb.tif", "crowns.shp",
                     "trees.shp", "check.json", "quicklook.png",
                     "uncovered.shp"):
            self.assertTrue(os.path.exists(os.path.join(out, name)), name)
        self.assertEqual(check["crowns"], 3)
        self.assertEqual(check["dontCareCrowns"], 1)
        self.assertEqual(check["invisibleCrowns"], 1)
        self.assertEqual(check["annotatedCrowns"], 5)
        self.assertEqual(check["measured"], {"2023-07-29": 2, "2023-10-11": 1})
        self.assertEqual(sorted(check["heightsByDay"]), ["2023-07-29",
                                                        "2023-10-11"])
        self.assertEqual(check["heightsByDay"]["2023-10-11"]["trees"], 1)
        self.assertTrue(np.isfinite(
            check["heightsByDay"]["2023-10-11"]["medianChmMinusFieldM"]))
        self.assertAlmostEqual(check["heightsNoShoot"]["medianFieldHm"], 2.4)
        self.assertGreater(check["chmCellsFilled"], 0.3)   # sparse on purpose
        coverage = check["coverage"]
        self.assertEqual(coverage["patches"], 1)          # the unannotated tree
        uncovered = gpd.read_file(os.path.join(out, "uncovered.shp"))
        loose = Point(self.west + 12.5, self.south + 10.0)
        self.assertTrue(uncovered.geometry.iloc[0].intersects(loose.buffer(0.5)))
        self.assertTrue(bool(uncovered["edge"].iloc[0]))   # reaches the edge
        scored = gpd.read_file(os.path.join(out, "scoredArea.shp")).geometry[0]
        whole = gpd.read_file(os.path.join(out, "area.shp")).geometry[0]
        thicket = Point(self.west + 40, self.south + 30)
        hidden = Point(self.west + 45, self.south + 12)
        for place in (loose, thicket, hidden):
            self.assertTrue(whole.contains(place))
            self.assertFalse(scored.contains(place))
        self.assertEqual(len(gpd.read_file(os.path.join(out, "crowns.shp"))), 3)
        ignored = gpd.read_file(os.path.join(out, "ignored.shp"))
        self.assertEqual(sorted(ignored["reason"]),
                         ["invisible", "other", "uncovered"])
        heights = check["heights"]
        self.assertEqual(heights["trees"], 3)
        self.assertLess(abs(heights["medianChmMinusFieldM"]), 0.5)
        trees = gpd.read_file(os.path.join(out, "trees.shp"))
        self.assertAlmostEqual(trees["fieldHm"].iloc[1], 4.0)
        with rasterio.open(os.path.join(out, "chm.tif")) as chm, \
                rasterio.open(os.path.join(out, "rgb.tif")) as rgb:
            self.assertEqual(rgb.count, 3)
            self.assertAlmostEqual(rgb.res[0], 0.02)
            data = chm.read(1)
            self.assertAlmostEqual(chm.res[0], 0.1)
            self.assertGreater(data.max(), 3.4)   # 4 m apex, sampled
            # ground between the trees but outside every buffered crown
            row, column = chm.index(self.west + 40, self.south + 20)
            self.assertEqual(data[row, column], 0.0)
            # tiles start at 7 m and are 20 m wide: tree 2 sits on the seam
            row, column = chm.index(self.west + 27.0, self.south + 12.0)
            self.assertGreater(data[row, column], 3.0)

    def testRecheckRedoesTheChecksOnly(self):
        import json
        from tt.qpPrepare import main, prepareSite
        args = self.arguments()
        prepareSite(self.site, args)
        chm = os.path.join(self.directory, "out", self.site, "chm.tif")
        before = os.path.getmtime(chm)
        argv = ["--vectors", self.vectors, "--pdal", "/nonexistent",
                "--output", os.path.join(self.directory, "out"), "--recheck",
                "--canopyM", "2.0"]
        self.assertEqual(main(argv), 0)
        self.assertEqual(os.path.getmtime(chm), before)
        with open(os.path.join(self.directory, "out", self.site,
                               "check.json")) as handle:
            self.assertEqual(json.load(handle)["coverage"]["canopyM"], 2.0)

    def testZeroHeightsArePlaceholders(self):
        import pandas as pd
        from tt.qpPrepare import NO_SHOOT_HEIGHTS, fieldHeight
        trees = pd.DataFrame({"height1_no_shoot_cm": ["0", "250", "NA"],
                              "height2_no_shoot_cm": [0, 240, 0]})
        heights = fieldHeight(trees, NO_SHOOT_HEIGHTS)
        self.assertTrue(np.isnan(heights.iloc[0]))
        self.assertAlmostEqual(heights.iloc[1], 2.5)
        self.assertTrue(np.isnan(heights.iloc[2]))

    def testInteriorAndEdgePatchesAreTold(self):
        import argparse
        import geopandas as gpd
        import rasterio
        from affine import Affine
        from shapely.geometry import Point, box
        from tt.qpPrepare import uncoveredCanopy
        chm = np.zeros((200, 200), np.float32)          # 20 x 20 m at 0.1 m
        rows, columns = np.mgrid[0:200, 0:200]
        chm[np.hypot(rows - 100, columns - 100) < 10] = 3.0   # middle
        chm[np.hypot(rows - 100, columns - 5) < 10] = 3.0     # on the edge
        path = os.path.join(self.directory, "patches.tif")
        with rasterio.open(path, "w", driver="GTiff", height=200, width=200,
                           count=1, dtype="float32", crs="EPSG:32619",
                           transform=Affine(0.1, 0, 0, 0, -0.1, 20)) as d:
            d.write(chm, 1)
        area = gpd.GeoDataFrame(geometry=[box(0.5, 0.5, 19.5, 19.5)],
                                crs="EPSG:32619")
        crowns = gpd.GeoDataFrame({"class_code": ["PIGL"]},
                                  geometry=[Point(18, 18).buffer(0.5)],
                                  crs="EPSG:32619")
        args = argparse.Namespace(canopyM=1.5, crownMarginM=0.2, minPatchM2=0.5)
        summary, patches = uncoveredCanopy(path, crowns, area, args)
        self.assertEqual(summary["patches"], 2)
        self.assertEqual(summary["interiorPatches"], 1)
        middle = patches[patches.contains(Point(10, 10))]
        self.assertFalse(bool(middle["edge"].iloc[0]))
        self.assertGreater(summary["interiorShare"], 0.4)
        self.assertGreater(summary["edgeShare"], 0.1)

    def testFlakyTileIsRetried(self):
        import tt.qpPrepare as qp
        args = self.arguments()
        args.retries, args.retryWaitS = 2, 0.0
        calls = []
        real = qp.runPdal

        def flaky(pdalPath, stages, timeoutS=None):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("pdal failed: Could not read from 'x'")
            real(pdalPath, stages, timeoutS)
        qp.runPdal = flaky
        try:
            part = os.path.join(self.directory, "tile.tif")
            qp.buildTile(self.lidar, (300007.0, 5000007.0, 300027.0,
                                      5000027.0), args, part)
        finally:
            qp.runPdal = real
        self.assertEqual(len(calls), 3)
        self.assertTrue(os.path.exists(part))

    def testHungPdalIsKilledAndRetried(self):
        import stat
        import time
        import tt.qpPrepare as qp
        slow = os.path.join(self.directory, "slowPdal.sh")
        with open(slow, "w") as handle:
            handle.write("#!/bin/sh\nsleep 30\n")
        os.chmod(slow, os.stat(slow).st_mode | stat.S_IEXEC)
        args = self.arguments()
        args.pdal, args.retries, args.retryWaitS = slow, 1, 0.0
        args.tileTimeoutMin = 1.0 / 60          # one second
        start = time.time()
        with self.assertRaises(RuntimeError) as caught:
            qp.buildTile(self.lidar, (300007.0, 5000007.0, 300027.0,
                                      5000027.0), args,
                         os.path.join(self.directory, "t.tif"))
        self.assertIn("timed out", str(caught.exception))
        self.assertLess(time.time() - start, 10)

    def testFailedSiteDoesNotStopTheRun(self):
        from tt.qpPrepare import main
        argv = ["--vectors", self.vectors, "--pdal", "/nonexistent/pdal",
                "--output", os.path.join(self.directory, "out"),
                "--retries", "0", "--sites", self.site, "20990101_missing"]
        self.assertEqual(main(argv), 1)          # both fail, neither raises

    def testRgbComesFromTheOverviewAndStopsAtTheImage(self):
        import geopandas as gpd
        import rasterio
        from affine import Affine
        from rasterio.enums import Resampling
        from shapely.geometry import box
        from tt.qpPrepare import readRgb
        path = os.path.join(self.directory, "ov.tif")
        image = np.zeros((4, 800, 800), np.uint8)
        image[:, :, 400:] = 200                   # right half bright
        with rasterio.open(path, "w", driver="GTiff", height=800, width=800,
                           count=4, dtype="uint8", crs="EPSG:32619",
                           tiled=True, transform=Affine(0.01, 0, 0, 0, -0.01,
                                                        8)) as d:
            d.write(image)
            d.build_overviews([2, 4], Resampling.average)
        with rasterio.open(path) as src:
            from tt.qpPrepare import overviewLevel
            self.assertEqual(overviewLevel(src, 0.02), 0)
            self.assertEqual(overviewLevel(src, 0.045), 1)
            self.assertIsNone(overviewLevel(src, 0.005))
        # area sticks out 2 m past the image's right edge (x = 8 m)
        area = gpd.GeoDataFrame(geometry=[box(2, 2, 10, 6)], crs="EPSG:32619")
        out = os.path.join(self.directory, "rgbOut.tif")
        readRgb(path, area, 0.04, out)
        with rasterio.open(out) as rgb:
            data = rgb.read()
            self.assertEqual(data.shape, (3, 100, 200))
            self.assertAlmostEqual(rgb.transform.c, 2.0)
            self.assertTrue((data[:, :, :50] == 0).all())     # x 2-4: dark
            self.assertTrue((data[:, :, 60:140] == 200).all())  # x 4.4-7.6
            self.assertTrue((data[:, :, 150:] == 0).all())    # beyond image

    def testWholeCellOffsetIsOnTheLattice(self):
        from tt.qpPrepare import latticeOffset
        self.assertAlmostEqual(latticeOffset(10.05, 10.0, 0.05), 0.0)
        self.assertAlmostEqual(latticeOffset(10.025, 10.0, 0.05), 0.025)

    def testAreaFillsHolesAndMergesNeighbours(self):
        import geopandas as gpd
        from shapely.geometry import Point
        from tt.qpPrepare import annotatedArea
        ring = [Point(np.cos(a) * 5, np.sin(a) * 5).buffer(1.2)
                for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        area = annotatedArea(gpd.GeoDataFrame(geometry=ring,
                                              crs="EPSG:32619"), 1.0)
        shape = area.geometry.iloc[0]
        self.assertEqual(shape.geom_type, "Polygon")
        self.assertTrue(shape.contains(Point(0, 0)))


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
        from tt.studySummary import compareHybrids, readStudy
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
        from tt.studySummary import main
        out = os.path.join(self.directory, "out")
        self.assertEqual(main(["--root", self.directory, "--output", out]), 0)
        with open(os.path.join(out, "sites.csv")) as handle:
            self.assertEqual(len(handle.readlines()), 1 + 3 * 5)


class TestConCompGrid(Fixture):
    """Minimum height and tree area tuned like the rest; jobs change nothing."""

    def dataset(self):
        import subprocess
        out = os.path.join(self.directory, "ccds")
        if not os.path.exists(os.path.join(out, "dataset.json")):
            subprocess.run([sys.executable, "-m", "tt.dl", "prepare",
                            "--source", "chm:" + self.chmPath,
                            "--crowns", self.crownsPath,
                            "--boundary", self.boundaryPath, "--output", out,
                            "--name", "t", "--resolution", "0.25",
                            "--minHeight", "1.0", "--blockCols", "2",
                            "--blockRows", "1"], check=True,
                           capture_output=True)
        return out

    def validation(self, jobs):
        from tt.dl.dlConComp import ConCompCrossValidation
        grid = {"lowerPercentile": [10], "minTopAreaM2": [0.12],
                "topStepM": [0.25], "erosionIterations": [1],
                "saddleDropM": [0.5], "minHeight": [1.0, 3.0],
                "minTreeAreaM2": [0.5, 40.0]}
        return ConCompCrossValidation(self.dataset(), grid=grid,
                                      minHeight=2.0, verbose=False, jobs=jobs)

    def testTheTwoAreTunedDimensions(self):
        validation = self.validation(1)
        self.assertEqual(len(validation.settings), 4)
        validation.detectAll()
        self.assertEqual(sorted(k[0] for k in validation.scenes),
                         [1.0, 2.0, 3.0])
        counts = {(s["minHeight"], s["minTreeAreaM2"]): len(d)
                  for s, d in zip(validation.settings, validation.detections)}
        # a 40 m2 minimum tree area leaves none of the small cones
        self.assertLess(counts[(1.0, 40.0)], counts[(1.0, 0.5)])

    def testEachMergeRuleGetsItsOwnScore(self):
        from tt.dl.dlConComp import ConCompCrossValidation
        grid = {"lowerPercentile": [10], "minTopAreaM2": [0.12],
                "topStepM": [0.25], "erosionIterations": [1],
                "saddleDropM": [0.5], "mergeMetric": ["saddle", "prominence"]}
        validation = ConCompCrossValidation(self.dataset(), grid=grid,
                                            minHeight=2.0, verbose=False)
        folds, pooled = validation.run()
        self.assertEqual(sorted(pooled["byMergeMetric"]),
                         ["prominence", "saddle"])
        self.assertEqual(pooled["bySmoothing"], {})       # not in this grid
        best = max(p["f1"] for p in pooled["byMergeMetric"].values())
        self.assertGreater(best, 0.5)

    def testCeilingIsNoWorseThanAnyFold(self):
        validation = self.validation(1)
        folds, pooled = validation.run()
        self.assertIn(pooled["wholeAreaBest"]["settings"], validation.settings)
        self.assertGreaterEqual(pooled["wholeAreaBest"]["f1"],
                                pooled["f1"] - 1e-9)

    def testParallelEqualsSerial(self):
        serial = self.validation(1)
        parallel = self.validation(3)
        self.assertEqual(serial.detectAll(), parallel.detectAll())

    def testParallelRunEqualsSerialRun(self):
        foldsA, pooledA = self.validation(1).run()
        foldsB, pooledB = self.validation(3).run()
        self.assertEqual([(f["block"], f["f1"], f["settings"]) for f in foldsA],
                         [(f["block"], f["f1"], f["settings"]) for f in foldsB])
        self.assertEqual(pooledA["wholeAreaBest"], pooledB["wholeAreaBest"])


class TestPseudoCrownCanopy(Fixture):
    """The canopy mask is the whole kept CHM unless a percentile is asked for."""

    def testWholeChmByDefaultSmallerWithAPercentile(self):
        from tt.pseudoCrowns import PseudoCrowns
        scene = self.scene()
        whole = PseudoCrowns(scene, None, verbose=False).canopyMask() > 0
        np.testing.assert_array_equal(whole, scene.chm > 0)
        cut = PseudoCrowns(scene, None, percentile=20,
                           verbose=False).canopyMask() > 0
        self.assertTrue((cut <= whole).all())
        self.assertLess(cut.sum(), whole.sum())


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

    def testTheShiftIsFound(self):
        from tt.siteAnalysis import offsetScan
        result = offsetScan(self.predictions, self.crowns, self.region)
        self.assertEqual([round(v, 1) for v in result["bestShiftM"]],
                         [-0.5, 0.0])
        self.assertAlmostEqual(result["unshiftedF1"], 0.0)
        self.assertAlmostEqual(result["bestF1"], 1.0)

    def testGrowingTheCrownsRecoversEdgeTops(self):
        from tt.siteAnalysis import tolerance
        result = tolerance(self.predictions, self.crowns, self.region)
        self.assertAlmostEqual(result["0.25"]["f1"], 1.0)   # 0.3 + 0.25 > 0.5


class TestSharedBlocks(unittest.TestCase):
    """A block a learned model skipped is left out, by name or _bNN suffix."""

    def testOnlyBlocksEveryRunHas(self):
        from tt.comparison import MethodRun, sharedBlocks
        directory = tempfile.mkdtemp()
        for name in ("site_b00", "site_b01"):
            with open(os.path.join(directory, "predictions_%s.json" % name),
                      "w") as handle:
                handle.write("{}")
        run = MethodRun.__new__(MethodRun)
        run.runDir = directory
        self.assertTrue(run.has("site_b00"))
        self.assertTrue(run.has("other_b01"))      # matched by suffix
        self.assertFalse(run.has("site_b02"))
        kept = sharedBlocks({"site_b00": 0, "site_b01": 1, "site_b02": 2},
                            (run,), verbose=False)
        self.assertEqual(sorted(kept), ["site_b00", "site_b01"])


class TestProminence(unittest.TestCase):
    """A chain of branch pseudotops: the saddle test keeps the far one, the
    elder rule merges it through the branch."""

    def setUp(self):
        import rasterio
        from affine import Affine
        self.directory = tempfile.mkdtemp(prefix="ttProm")
        chm = np.full((100, 100), 2.0, np.float32)            # low canopy
        rows, columns = np.mgrid[0:100, 0:100]
        chm[47:54, 30:63] = 9.0                               # east arm
        chm[18:54, 57:64] = 9.0                               # north arm
        distance = np.hypot(rows - 50, columns - 30)
        chm[distance < 8] = (9.6 + 0.4 * (1 - distance / 8))[distance < 8]
        chm[np.hypot(rows - 50, columns - 60) < 2.5] = 9.5    # P1, at the bend
        chm[np.hypot(rows - 20, columns - 60) < 2.5] = 9.4    # P2, arm's end
        chm[58:72, 24:37] = 5.0      # lower canopy, so the descent goes
                                     # below the branches, as in a real CHM
        self.path = os.path.join(self.directory, "branch.tif")
        with rasterio.open(self.path, "w", driver="GTiff", height=100,
                           width=100, count=1, dtype="float32",
                           crs="EPSG:32619",
                           transform=Affine(0.1, 0, 0, 0, -0.1, 10)) as d:
            d.write(chm, 1)

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def tops(self, metric, **mergerKeywords):
        from tt.detector import ConCompDetector
        from tt.merging import TopMerger
        from tt.scene import Scene
        scene = Scene(self.path, resolution=0.1, minHeight=1.0, verbose=False)
        detector = ConCompDetector(
            windowSizeM=40.0, lowerPercentile=0, erosionIterations=0,
            minTreeAreaM2=0.05, minTopAreaM2=0.01, topStepM=0.1,
            merger=TopMerger(metric, epsM=8.0, saddleDropM=0.6,
                             **mergerKeywords),
            verbose=False)
        return sorted((y, x) for x, y, _ in detector.detect(scene))

    def testSaddleKeepsASpuriousTopOnTheBranch(self):
        # a straight line from the apex to a pseudotop leaves the narrow
        # branch and dips to the ground, so the saddle test keeps a top there
        tops = self.tops("saddle")
        self.assertEqual(len(tops), 2)
        self.assertTrue(any(c > 50 for r, c in tops))       # on the branch

    def testProminenceMergesThePseudotopsIntoTheTree(self):
        tops = self.tops("prominence")
        self.assertEqual(len(tops), 1)
        row, column = tops[0]
        self.assertLess(np.hypot(row - 50, column - 30), 8)   # on T's crown

    def testCalibrationDipIsTheMergeLevel(self):
        """Apex T and branch tip P2: the straight line crosses the ground
        (dip 7.4 m); through the branch they join at 9.0 m (dip 0.4 m)."""
        from tt.calibration import ConfidentZones
        from tt.scene import Scene
        scene = Scene(self.path, resolution=0.1, minHeight=1.0, verbose=False)
        zones = ConfidentZones.__new__(ConfidentZones)
        zones.scene = scene
        zones.boxes = np.array([[2.2, 4.2, 3.8, 5.8],    # around T (x, y in m)
                                [5.7, 7.7, 6.3, 8.3]])  # around P2
        zones.apexes = [zones._apex(b) for b in zones.boxes]
        self.assertAlmostEqual(zones._mergeDip(0, 1), 0.4, delta=0.05)

    def testSmoothingFlattensASpikeButKeepsTheCrown(self):
        import rasterio
        from tt.scene import Scene
        with rasterio.open(self.path, "r+") as d:
            chm = d.read(1)
            chm[80, 80] = 12.0                    # one noise pixel on low canopy
            d.write(chm, 1)
        plain = Scene(self.path, resolution=0.1, minHeight=1.0, verbose=False)
        smooth = Scene(self.path, resolution=0.1, minHeight=1.0, verbose=False,
                       smoothM=0.2)
        self.assertAlmostEqual(float(plain.chm[80, 80]), 12.0)
        self.assertLess(float(smooth.chm[80, 80]), 3.0)   # the spike is gone
        self.assertGreater(float(smooth.chm[50, 30]), 9.5)  # T still stands

    def testAProminentNeighbourStaysApart(self):
        from tt.detector import ConCompDetector
        from tt.merging import TopMerger
        from tt.scene import Scene
        import rasterio
        with rasterio.open(self.path, "r+") as d:
            chm = d.read(1)
            chm[np.hypot(*np.mgrid[0:100, 0:100] - np.array([[[20]], [[60]]]))
                < 2.5] = 11.0                                 # P2 now a tree
            d.write(chm, 1)
        self.assertEqual(len(self.tops("prominence")), 2)

    def testDropSlopeMergesAtTheTopsHeight(self):
        """P2 raised to 11 m becomes the elder; T (10 m) dies where the
        branches join at 9 m, prominence 1 m. A fixed 0.6 m drop keeps T;
        0.6 + 0.05 * 10 = 1.1 m at T's height removes it."""
        import rasterio
        with rasterio.open(self.path, "r+") as d:
            chm = d.read(1)
            chm[np.hypot(*np.mgrid[0:100, 0:100] - np.array([[[20]], [[60]]]))
                < 2.5] = 11.0
            d.write(chm, 1)
        self.assertEqual(len(self.tops("prominence")), 2)
        self.assertEqual(len(self.tops("prominence", saddleDropSlope=0.05)), 1)


class TestHeightSlopes(unittest.TestCase):
    """Parameters that grow with tree height; a slope of 0 is the old
    fixed parameter."""

    def testDropGrowsWithHeight(self):
        from tt import TopMerger
        merger = TopMerger("prominence", saddleDropM=0.5, saddleDropSlope=0.05)
        self.assertAlmostEqual(merger.dropAt(0.0), 0.5)
        self.assertAlmostEqual(merger.dropAt(20.0), 1.5)
        self.assertAlmostEqual(TopMerger(saddleDropM=0.5).dropAt(20.0), 0.5)

    def testSaddleDropIsTakenAtTheLowerTop(self):
        """The same 1 m dip separates two 5 m trees but not two branch tops
        of a 20 m tree."""
        from tt import TopMerger
        merger = TopMerger("saddle", epsM=100.0, saddleDropM=0.5,
                           saddleDropSlope=0.05)
        for height, same in ((20.0, True), (5.0, False)):
            surface = np.full((10, 40), height, np.float32)
            surface[5, 18:22] = height - 1.0
            self.assertEqual(merger.areSame(7.5, 0.0, surface=surface,
                                            a=(5, 5), b=(5, 35)), same)

    def testMinimumTopAreaGrowsWithHeight(self):
        from tt.detector import ConCompDetector
        geometry = {"minPixTop": 2, "pixelArea": 0.0625}
        fixed = ConCompDetector(minTopAreaM2=0.12, verbose=False)
        self.assertEqual(fixed._minPixTop(geometry, 20.0), 2)
        sloped = ConCompDetector(minTopAreaM2=0.12, minTopAreaSlope=0.02,
                                 verbose=False)
        self.assertEqual(sloped._minPixTop(geometry, 20.0), 8)  # 0.52 m2
        with self.assertRaises(ValueError):
            ConCompDetector(minTopAreaSlope=-0.01, verbose=False)

    def testZeroSlopeLeavesTheSweepGridAlone(self):
        from tt.cli import _addSlopes
        grid = {}
        _addSlopes(grid, "saddleDropSlope", "0")
        self.assertEqual(grid, {})
        _addSlopes(grid, "saddleDropSlope", "0,0.05")
        self.assertEqual(grid, {"saddleDropSlope": [0.0, 0.05]})


class TestPseudoTuning(unittest.TestCase):
    """The pseudo scorers pick settings without the crowns; scored with them."""

    def testExactlyOneShare(self):
        from tt.pseudoTuning import exactlyOneShare
        boxes = np.array([[0, 0, 2, 2], [5, 5, 7, 7], [10, 10, 12, 12]], float)
        points = np.array([[1, 1], [6, 6], [6.5, 6.5]], float)
        self.assertAlmostEqual(exactlyOneShare(points, boxes), 1.0 / 3)
        self.assertEqual(exactlyOneShare(np.zeros((0, 2)), boxes), 0.0)

    def testChosenSettingIsScoredOnTheRealCrowns(self):
        from tt.pseudoTuning import tuneBlock

        class Stub(object):
            def realScores(self):
                return [{"crowns": 10, "f1": f, "hits": 0, "predictions": 0,
                         "repeats": 0, "falsePositives": 0, "recall": f,
                         "precision": f} for f in (0.5, 0.9, 0.7)]

            def scorers(self):
                # pseudoF1 prefers setting 2, the zones setting 1
                return {"pseudoF1": [0.1, 0.2, 0.3], "zones@0.9": [0.2, 0.8, 0.5]}

        rows = tuneBlock(Stub())
        self.assertEqual(rows["real"]["chosen"], 1)
        self.assertAlmostEqual(rows["pseudoF1"]["f1"], 0.7)
        self.assertAlmostEqual(rows["zones@0.9"]["f1"], 0.9)
        self.assertAlmostEqual(rows["zones@0.9"]["spearman"], 1.0)
        self.assertAlmostEqual(rows["pseudoF1"]["spearman"], 0.5)

    def testTheAutomaticRunSitsBesideItsSource(self):
        from tt.pseudoTuning import autoRunDir
        self.assertEqual(autoRunDir("/s/runs/ccLidar"), "/s/runs/ccAutoLidar")
        self.assertEqual(autoRunDir("/s/runs/ccP1/"), "/s/runs/ccAutoP1")
        from tt.reportText import HEIGHT_SOURCE
        self.assertEqual(HEIGHT_SOURCE["ccAutoLidar"], "lidar")
        self.assertEqual(HEIGHT_SOURCE["ccAutoP1"], "p1")

    def testDetectionsRoundTrip(self):
        from tt.dl.dlConComp import loadDetections, saveDetections
        path = os.path.join(tempfile.mkdtemp(), "d.npz")
        detections = [[{"centreX": 1.0, "centreY": 2.0, "score": 3.0}], []]
        saveDetections(path, detections)
        back = loadDetections(path)
        self.assertEqual(len(back), 2)
        self.assertEqual((back[0][0]["centreX"], back[0][0]["score"]), (1.0, 3.0))
        self.assertEqual(back[1], [])


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
        # the stand-in even where the real DeepForest is installed
        deepForestDetect.dfMain = module.main
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
    """Mask R-CNN's annotated subset and transfer, without training (torch)."""

    def setUp(self):
        try:
            from tt.dl import dlMaskRcnn
        except ImportError:
            self.skipTest("torch not installed")
        self.mr = dlMaskRcnn

        def boxesOnTiles(model, tiles, root, device, transform, **keywords):
            return [{"box": [t["west"], t["south"], t["east"], t["north"]],
                     "centreX": t["centreX"], "centreY": t["centreY"],
                     "score": 0.9} for t in tiles]
        self.patches = [mock.patch.object(dlMaskRcnn, name, value)
                        for name, value in (
                            ("predictTiles", boxesOnTiles),
                            ("buildModel", lambda *a, **k: object()),
                            ("trainFold", lambda *a, **k: None))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in getattr(self, "patches", []):
            p.stop()

    def testSubsetFoldUsesTheDrawnTiles(self):
        from tt.dl import dlCommon as dc
        args = self.mr.parseArguments([
            "--dataset", self.path("rgb"), "--output", self.path("mr"),
            "--trainFraction", "0.25", "--subsetSeed", "2", "--buffer", "2",
            "--device", "cpu"])
        os.makedirs(args.output, exist_ok=True)
        run = self.mr.MaskRcnnCrossValidation(args)
        name, geometry = run.blocks[0]
        fit, val, validationBlock, _ = run.splitFold(name)
        train, _ = dc.foldSplit(run.meta["tiles"], run.blocks, name, 2.0)
        drawn = dc.drawTiles(train, 0.25, 2, name)
        self.assertEqual(sorted(t["stem"] for t in fit + val),
                         sorted(t["stem"] for t in drawn))
        region = run.validationRegion(validationBlock, val)
        self.assertLess(region.area, dict(run.blocks)[validationBlock].area)
        result = run.runFold(name, geometry)
        self.assertEqual((result["fitTiles"], result["trainFraction"]),
                         (len(fit), 0.25))

    def testTransferScoresEveryBlock(self):
        args = self.mr.parseArguments([
            "--dataset", self.path("rgb"), "--output", self.path("mrT"),
            "--transferFrom", self.path("rgb"), "--device", "cpu"])
        os.makedirs(args.output, exist_ok=True)
        folds, _ = self.mr.TransferRun(args).run()
        self.assertEqual(len(folds), 4)
        self.assertTrue(os.path.exists(os.path.join(
            args.output, "predictions_%s.json" % folds[0]["block"])))

    def crossValidation(self, dataset, *extra):
        args = self.mr.parseArguments(["--dataset", dataset, "--output",
                                       self.path("out"), "--device", "cpu",
                                       "--trainFraction", "0.5", "--subsetSeed", "2"]
                                      + list(extra))
        return self.mr.MaskRcnnCrossValidation(args)

    @staticmethod
    def keys(fold):
        fit, val, _, _ = fold
        return {(t["block"], t["r0"], t["c0"]) for t in fit + val}

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
        drawn = self.crossValidation(self.path("chm"), "--drawFrom", self.path("rgb"))
        for block, _ in plain.blocks:
            a, b = plain.splitFold(block), drawn.splitFold(block)
            self.assertEqual(self.keys(a) if a else None, self.keys(b) if b else None)

    def testAMissingTileLeavesTheRestOfTheDrawAlone(self):
        folder, victim = self.reducedDataset()
        reference = self.crossValidation(self.path("rgb"))
        reduced = self.crossValidation(folder, "--drawFrom", self.path("rgb"))
        for block, _ in reference.blocks:
            want, got = reference.splitFold(block), reduced.splitFold(block)
            if want is None or got is None:
                continue
            self.assertEqual(self.keys(got), self.keys(want) - {victim})



class TestBaselines(unittest.TestCase):
    """The classical CHM baselines' detectors, on hand-made height models."""

    class Scene(object):
        def __init__(self, chm, pixel=0.25):
            self.chm, self.pixelSize = chm, pixel

    @staticmethod
    def bumps(centres, heights, size=60, sigma=4.0):
        rows, cols = np.indices((size, size))
        chm = np.zeros((size, size))
        for (r, c), h in zip(centres, heights):
            chm = np.maximum(chm, h * np.exp(-((rows - r) ** 2 + (cols - c) ** 2)
                                             / (2 * sigma ** 2)))
        chm[chm < 0.5] = 0
        return chm

    def testVariableWindowGrowsWithHeight(self):
        from tt.dl.dlBaselines import LocalMaximaVariableWindow
        # two tops 3 m apart: separate in a 1 m window; one when the window
        # grows with height to 1 + 0.6 x 9 = 6.4 m (3.2 m either side)
        chm = self.bumps([(30, 24), (30, 36)], [10.0, 9.0])
        small = LocalMaximaVariableWindow(1.0, 0.0).detect(self.Scene(chm))
        large = LocalMaximaVariableWindow(1.0, 0.6).detect(self.Scene(chm))
        self.assertEqual(len(small.points), 2)
        self.assertEqual(len(large.points), 1)
        self.assertEqual((large.points[0][0], large.points[0][1]), (24, 30))

    def testPlateauGivesOneTop(self):
        from tt.dl.dlBaselines import LocalMaximaVariableWindow
        chm = np.zeros((20, 20))
        chm[8:12, 8:12] = 5.0
        tops = LocalMaximaVariableWindow(0.5, 0.0).detect(self.Scene(chm))
        self.assertEqual(len(tops.points), 1)

    def testWatershedDropsSmallCrowns(self):
        from tt.dl.dlBaselines import MarkerWatershed
        chm = self.bumps([(20, 20), (45, 45)], [8.0, 6.0])
        chm += self.bumps([(20, 45)], [3.0], sigma=1.0)       # a tiny one
        both = MarkerWatershed(1.0, 0.0).detect(self.Scene(chm))
        big = MarkerWatershed(1.0, 1.0).detect(self.Scene(chm))
        self.assertEqual(len(both.points), 3)
        self.assertEqual(sorted((x, y) for x, y, _ in big.points),
                         [(20, 20), (45, 45)])

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
            from tt.studySummary import singles
            os.makedirs(os.path.join(site.directory, "runs"), exist_ok=True)
            shutil.move(output, os.path.join(site.directory, "runs", "lmvwP1"))
            methods, ceilings = singles(site.directory)
            self.assertIn("lmvwP1", methods)
            self.assertIn("lmvwP1 ceiling", ceilings)
        finally:
            site.tearDownClass()


class TestEnshurinPrepare(unittest.TestCase):
    """Crowns from class masks and the area from annotation coverage."""

    def testCoveredCellsAndSpecies(self):
        import geopandas as gpd
        import rasterio
        from rasterio.transform import from_origin
        from tt import enshurinPrepare as ep
        d = tempfile.mkdtemp(prefix="ttEnshurin")
        try:
            crs, x0, y0 = "EPSG:32654", 401800.0, 4268500.0
            rng = np.random.default_rng(0)
            trees = [(x0 + cx, y0 - cy, 2.2, rng.uniform(8, 15),
                      int(rng.integers(1, 5)))
                     for cx in np.arange(4, 58, 6.0)
                     for cy in np.arange(4, 28, 6.0)]
            res = 0.25
            W, H = int(60 / res), int(30 / res)
            cols, rows = np.meshgrid(np.arange(W), np.arange(H))
            X, Y = x0 + (cols + 0.5) * res, y0 - (rows + 0.5) * res
            chm = np.zeros((H, W), np.float32)
            for tx, ty, r, h, _ in trees:
                dist = np.hypot(X - tx, Y - ty)
                chm = np.maximum(chm, np.where(dist < r, h, 0))
            with rasterio.open(os.path.join(d, "chm.tif"), "w", driver="GTiff",
                               height=H, width=W, count=1, dtype="float32",
                               crs=crs, transform=from_origin(x0, y0, res, res)
                               ) as out:
                out.write(chm, 1)
            data = os.path.join(d, "data")
            os.makedirs(os.path.join(data, "raw", "a", "per_class"))
            pr = 0.05
            for i, part in enumerate(("newtrain", "test")):
                px0, w, h = x0 + 30 * i, int(30 / pr), int(30 / pr)
                c, r = np.meshgrid(np.arange(w), np.arange(h))
                PX, PY = px0 + (c + 0.5) * pr, y0 - (r + 0.5) * pr
                mask = np.zeros((h, w), np.uint8)
                rgb = np.full((3, h, w), 60, np.uint8)
                for tx, ty, rad, _, cls in trees:
                    dist = np.hypot(PX - tx, PY - ty)
                    if part == "test" and tx > px0 + 15:
                        continue             # unannotated: other species
                    mask[dist < rad * 0.95] = cls
                t = from_origin(px0, y0, pr, pr)
                for name, array, count in (
                        ("mask_4classes_%s.tif" % part, mask[None], 1),
                        ("08_22_%s.tif" % part, rgb, 3)):
                    with rasterio.open(os.path.join(data, name), "w",
                                       driver="GTiff", height=h, width=w,
                                       count=count, dtype="uint8", crs=crs,
                                       transform=t) as out:
                        out.write(array)
                if part == "newtrain":
                    for value, species in ((1, "beech"), (2, "larch")):
                        with rasterio.open(os.path.join(
                                data, "raw", "a", "per_class",
                                "mask_%s_newtrain.tif" % species), "w",
                                driver="GTiff", height=h, width=w, count=1,
                                dtype="uint8", crs=crs, transform=t) as out:
                            out.write((mask == value).astype(np.uint8)[None])
            check = ep.prepare(ep.parseArguments([
                "--data", data, "--chm", os.path.join(d, "chm.tif"),
                "--output", os.path.join(d, "out"), "--parts", "newtrain",
                "test", "--cellM", "5"]))
            parts = check["parts"]
            self.assertEqual(parts["newtrain"]["species"],
                             {1: "beech", 2: "larch"})
            self.assertEqual(parts["test"]["species"], {1: "beech", 2: "larch"})
            self.assertAlmostEqual(parts["newtrain"]["keptM2"], 900, delta=30)
            # west half annotated (450 m2) plus the last 5 m strip, which has
            # no canopy at all and is kept: nothing there to miss
            self.assertAlmostEqual(parts["test"]["keptM2"], 525, delta=40)
            # 20 trees in newtrain, the 2 annotated columns of test (8)
            self.assertEqual(check["crowns"], 28)
            # the area's edge cuts no annotated crown
            area = gpd.read_file(os.path.join(d, "out", "enshurin",
                                              "area.shp")).geometry.iloc[0]
            species = gpd.read_file(os.path.join(d, "out", "enshurin",
                                                 "crownSpecies.shp"))
            touching = species.geometry[species.intersects(area)]
            self.assertTrue(all(g.buffer(-0.01).within(area)
                                for g in touching))
            for name in ("chm.tif", "rgb.tif", "crowns.shp", "scoredArea.shp"):
                self.assertTrue(os.path.exists(os.path.join(
                    d, "out", "enshurin", name)))
        finally:
            shutil.rmtree(d, ignore_errors=True)


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
        from tt.dl import dlConComp as cc
        validation = object.__new__(cc.ConCompCrossValidation)
        validation.windowSizeM, validation.saddleEpsM = 40.0, 8.0
        validation.minTreeAreaM2 = 0.5
        setting = {"lowerPercentile": 0, "minTopAreaM2": 0.1, "topStepM": 0.1,
                   "erosionIterations": 0, "saddleDropM": 0.3,
                   cc.DROP_SLOPE_KEY: 0.05, cc.TOP_SLOPE_KEY: 0.02}
        detector = validation._detector(setting)
        self.assertEqual(detector.merger.saddleDropSlope, 0.05)
        self.assertEqual(detector.minTopAreaSlope, 0.02)
        del setting[cc.DROP_SLOPE_KEY], setting[cc.TOP_SLOPE_KEY]
        detector = validation._detector(setting)
        self.assertEqual(detector.merger.saddleDropSlope, 0.0)
        self.assertEqual(detector.minTopAreaSlope, 0.0)


class TestCrownEdges(unittest.TestCase):
    """Thin canopy edges outside a crown go to it; anything thicker does not."""

    def testRimsAddedNeighboursSplitUnannotatedTreeLeft(self):
        from tt.crownEdges import findEdges
        pixel, size = 0.05, 400
        rows, cols = np.indices((size, size)) * pixel
        chm = np.zeros((size, size), np.float32)
        ids = np.zeros((size, size), np.int32)
        trees = [(5, 5, 2.0, 1), (5, 8.7, 1.7, 2), (14, 14, 2.0, 0)]
        for cy, cx, r, crown in trees:            # crown 0: not annotated
            d = np.hypot(rows - cy, cols - cx)
            chm = np.maximum(chm, np.where(d < r, 3.0, 0))
            if crown:
                ids[d < 0.85 * r] = crown          # outline 15% short of the tree
        added, rejected = findEdges(chm, ids, pixel, 0.5, 0.5)
        for cy, cx, r, crown in trees[:2]:
            d = np.hypot(rows - cy, cols - cx)
            rim = (d < r) & (ids == 0)
            # every rim pixel went to a crown, and mostly to its own
            self.assertTrue(np.all(added[rim] > 0))
            self.assertGreater(np.mean(added[rim] == crown), 0.9)
        d = np.hypot(rows - 14, cols - 14)
        self.assertTrue(np.all(added[d < 2.0] == 0))
        self.assertTrue(np.all(rejected[d < 1.9]))


class TestEnshurinShift(unittest.TestCase):
    """A shifted source moves only the georeferencing, never the pixels."""

    def testShiftedSourceMovesBoundsKeepsPixels(self):
        import rasterio
        from rasterio.transform import from_origin
        from tt.enshurinPrepare import parseShifts, shiftedSource
        d = tempfile.mkdtemp(prefix="ttShift")
        try:
            path = os.path.join(d, "m.tif")
            data = np.arange(3 * 20 * 30, dtype=np.uint8).reshape(3, 20, 30)
            with rasterio.open(path, "w", driver="GTiff", height=20, width=30,
                               count=3, dtype="uint8", crs="EPSG:32654",
                               transform=from_origin(1000, 2000, 0.5, 0.5)) as out:
                out.write(data)
            vrt = shiftedSource(path, -1.37, -0.43, os.path.join(d, "shifted"))
            with rasterio.open(path) as a, rasterio.open(vrt) as b:
                self.assertTrue(np.array_equal(a.read(), b.read()))
                self.assertAlmostEqual(b.bounds.left, a.bounds.left - 1.37)
                self.assertAlmostEqual(b.bounds.top, a.bounds.top - 0.43)
                self.assertEqual(a.crs, b.crs)
            self.assertEqual(parseShifts(["newtrain=-1.37,-0.43", "val=0,1"]),
                             {"newtrain": (-1.37, -0.43), "val": (0.0, 1.0)})
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestConCompProgress(Fixture):
    """Progress lines and the settings table never change a result."""

    dataset = TestConCompGrid.dataset

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
        from tt.scene import heightSmoothed
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
        from tt.scene import heightSmoothed
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
        from tt.scene import LOCAL_HEIGHT_M, heightSmoothed
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

    dataset = TestConCompGrid.dataset

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


class TestHeightFilter(unittest.TestCase):
    """RGB boxes over low vegetation are removed using the height model."""

    def setUp(self):
        from affine import Affine
        from tt.fusion import SaddleSurface
        rows, columns = np.mgrid[0:60, 0:60]
        chm = np.zeros((60, 60), np.float32)
        # four trees of 8 m, and four bushes of 0.8 m, on a 1 m grid
        self.trees = [(10, 10), (10, 40), (40, 10), (40, 40)]
        self.bushes = [(25, 10), (25, 40), (10, 25), (40, 25)]
        for (r, c), h in [(t, 8.0) for t in self.trees] + [(b, 0.8) for b in self.bushes]:
            chm = np.maximum(chm, h * np.clip(1 - np.hypot(rows - r, columns - c) / 4.0, 0, 1))
        self.surface = SaddleSurface(chm, Affine(1, 0, 0, 0, -1, 60), 1.0, dropM=0.5)

    @staticmethod
    def box(r, c, score=0.9):
        x, y = c + 0.5, 60 - r - 0.5
        return {"box": [x - 3, y - 3, x + 3, y + 3], "score": score,
                "centreX": x, "centreY": y}

    def testLowBoxesRemovedTallKept(self):
        from tt.fusion import tallEnough
        boxes = [self.box(*t) for t in self.trees] + [self.box(*b) for b in self.bushes]
        self.assertEqual(len(tallEnough(boxes, self.surface, 0.0)), 8)
        kept = tallEnough(boxes, self.surface, 1.0)
        self.assertEqual(sorted((round(b["centreX"]), round(b["centreY"])) for b in kept),
                         sorted((c, 60 - r) for r, c in self.trees))

    def testOwnCacheNotAnotherSurfacesApex(self):
        from tt.fusion import tallEnough
        bush = self.box(*self.bushes[0])
        bush["apex"] = (10, 10)          # another surface's apex: a tree top
        self.assertEqual(tallEnough([bush], self.surface, 1.0), [])

    def testUnionHighDropsBushBoxesKeepsPoints(self):
        from tt.fusion import unionHigh
        boxes = [self.box(*t) for t in self.trees] + [self.box(*b) for b in self.bushes]
        point = {"centreX": 55.5, "centreY": 4.5, "score": 10.0, "box": [55.5, 4.5, 55.5, 4.5]}
        out = unionHigh(boxes, [point], threshold=0.5, boxMinHeightM=1.0,
                        surface=self.surface)
        self.assertEqual(len(out), 5)                  # four trees and the point

    def testCrossValidationTunesTheFilterOnBushes(self):
        import geopandas as gpd
        from shapely.geometry import Point, box as shapeBox
        from tt.fusion import FusionCrossValidation

        class Run(object):
            def __init__(self, name, predictions, threshold):
                self.name, self.predictions, self.t = name, predictions, threshold
            def has(self, block):
                return True
            def record(self, block):
                return {"predictions": [dict(p) for p in self.predictions],
                        "threshold": self.t}
            def validation(self, block):
                other = "s_b01" if block == "s_b00" else "s_b00"
                return other, [dict(p) for p in self.predictions]
            def threshold(self, block):
                return self.t

        boxes = [self.box(*t) for t in self.trees] + [self.box(*b) for b in self.bushes]
        crowns = gpd.GeoDataFrame({"class_code": ["tree"] * 4},
                                  geometry=[Point(c + 0.5, 60 - r - 0.5).buffer(3)
                                            for r, c in self.trees])
        blocks = {"s_b00": shapeBox(0, 0, 60, 60), "s_b01": shapeBox(0, 0, 60, 60)}
        validation = FusionCrossValidation(
            crowns, blocks, Run("rgb", boxes, 0.5), Run("cc", [], -np.inf),
            strategies=["boxes", "boxesHigh", "unionHigh"], verbose=False,
            heightSurface=self.surface)
        summary = validation.run()
        self.assertAlmostEqual(summary["boxes"]["pooled"]["precision"], 0.5)
        self.assertAlmostEqual(summary["boxesHigh"]["pooled"]["precision"], 1.0)
        self.assertAlmostEqual(summary["boxesHigh"]["pooled"]["recall"], 1.0)
        self.assertTrue(all(f["setting"]["boxMinHeightM"] >= 1.0
                            for f in summary["boxesHigh"]["folds"]))
        # boxesHigh keeps the model's own score threshold
        self.assertTrue(all(f["setting"]["threshold"] == 0.5
                            for f in summary["boxesHigh"]["folds"]))

    def testReportsUnchangedWithoutTheHeightSurface(self):
        # the reports' strategies never include the height filter
        from tt.fusion import HEIGHT_STRATEGIES, STRATEGIES
        self.assertFalse(set(HEIGHT_STRATEGIES) & set(STRATEGIES))


class TestAutoScorer(unittest.TestCase):
    """The automatic run follows the chosen scorer; the default is unchanged."""

    def testEachScorerWritesItsOwnChoices(self):
        import json
        from tt.pseudoTuning import AUTO_SCORER, parseArguments, writeAutoRun
        d = tempfile.mkdtemp(prefix="ttAuto")
        try:
            found = [[{"centreX": 1.0, "centreY": 1.0, "score": 1.0}],
                     [{"centreX": 2.0, "centreY": 2.0, "score": 1.0}]]
            row = lambda chosen: {"chosen": chosen, "recall": 1.0}
            perBlock = {"s_b00": {AUTO_SCORER: row(0), "zones@0.7": row(1)}}
            for scorer, expected in ((AUTO_SCORER, 1.0), ("zones@0.7", 2.0)):
                out = os.path.join(d, scorer)
                writeAutoRun(os.path.join(d, "ccP1"), perBlock, found, {},
                             out, scorer)
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


class TestTripleFusion(unittest.TestCase):
    """RGB and CHM networks agreeing, CC voting, CC adding prominent tops."""

    @staticmethod
    def box(x, y, score, half=1.5):
        return {"box": [x - half, y - half, x + half, y + half], "score": score,
                "centreX": x, "centreY": y}

    @staticmethod
    def top(x, y, prominence):
        return {"box": [x, y, x, y], "score": 1.0, "centreX": x, "centreY": y,
                "prominence": prominence}

    def testAgreementVoteAndAdditions(self):
        from tt.tripleFusion import triple
        rgb = [self.box(0, 0, 0.40), self.box(10, 0, 0.80), self.box(30, 0, 0.25)]
        chm = [self.box(0.2, 0, 0.40), self.box(20, 0, 0.60), self.box(30.1, 0, 0.25),
               self.box(40, 0, 0.05)]
        cc = [self.top(20, 0.5, 1.0),                    # inside the lone CHM box at 20
              self.top(50, 0, 3.0),                      # free, prominent
              self.top(60, 0, 0.2),                      # free, not prominent
              self.top(39.5, 0, 3.0), self.top(40.5, 0, 2.5)]   # two in one CHM box
        off = float("inf")
        # agreement only: the pair at 0 (both 0.40 >= low) and the confident RGB box
        out = triple(rgb, chm, cc, high=0.7, low=0.3, vote=off, prominence=off)
        self.assertEqual(sorted(round(p["centreX"]) for p in out), [0, 10])
        # the pair at 30 is too weak for low=0.3
        self.assertNotIn(30, [round(p["centreX"]) for p in out])
        # CC's vote rescues the CHM box at 20 (0.60 >= vote) that holds a top
        out = triple(rgb, chm, cc, high=0.7, low=0.3, vote=0.5, prominence=off)
        self.assertEqual(sorted(round(p["centreX"]) for p in out), [0, 10, 20])
        # prominent free tops are added; of the two in one CHM box, one only
        out = triple(rgb, chm, cc, high=0.7, low=0.3, vote=off, prominence=2.0)
        xs = sorted(round(p["centreX"], 1) for p in out)
        self.assertEqual(xs, [0.0, 10.0, 39.5, 50.0])

    def testOnePerPair(self):
        from tt.tripleFusion import triple
        rgb, chm = [self.box(0, 0, 0.9)], [self.box(0.3, 0.2, 0.95)]
        out = triple(rgb, chm, [], high=0.5, low=0.3, vote=float("inf"),
                     prominence=float("inf"))
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["score"], 0.95)

    def testProminence(self):
        from affine import Affine
        from tt.fusion import SaddleSurface
        from tt.tripleFusion import prominenceAt
        rows, columns = np.mgrid[0:80, 0:80].astype(float)
        chm = np.maximum(5 * np.clip(1 - np.hypot(rows - 40, columns - 20) / 8, 0, 1),
                         np.maximum(6 * np.clip(1 - np.hypot(rows - 40, columns - 50) / 8, 0, 1),
                                    0.0)).astype(np.float32)
        surface = SaddleSurface(chm, Affine(1, 0, 0, 0, -1, 80), 1.0, dropM=0.5)
        x, y = 20.5, 80 - 40.5
        self.assertAlmostEqual(prominenceAt(surface, x, y), 5.0, delta=0.15)
        self.assertAlmostEqual(prominenceAt(surface, 50.5, 80 - 40.5), 6.0, delta=0.15)


class TestReducedFusion(unittest.TestCase):
    """The union plus one idea each: off = the union, cleaning keeps one top per CHM box."""

    box = staticmethod(TestTripleFusion.box)
    top = staticmethod(TestTripleFusion.top)

    def testCleaningKeepsTheMostProminentTopPerChmBox(self):
        from tt.tripleFusion import cleanedTops
        chm = [self.box(0, 0, 0.8), self.box(20, 0, 0.15)]
        tops = [self.top(-0.5, 0, 1.0), self.top(0.6, 0, 3.0), self.top(10, 10, 0.5),
                self.top(19.5, 0, 2.0), self.top(20.5, 0, 1.0)]
        xs = lambda out: sorted(round(t["centreX"], 1) for t in out)
        # CHM box at 0 (score 0.8) groups the first two; the one at 20 needs cleanAt <= 0.15
        self.assertEqual(xs(cleanedTops(tops, chm, 0.5)), [0.6, 10, 19.5, 20.5])
        self.assertEqual(xs(cleanedTops(tops, chm, 0.1)), [0.6, 10, 19.5])
        self.assertEqual(xs(cleanedTops(tops, chm, float("inf"))), xs(tops))

    def testOffIsTheUnion(self):
        from tt.fusion import union
        from tt.tripleFusion import unionClean
        rgb = [self.box(0, 0, 0.9), self.box(10, 0, 0.4), self.box(30, 0, 0.7)]
        chm = [self.box(20, 0, 0.9)]
        cc = [self.top(0.2, 0, 1.0), self.top(10, 0, 2.0), self.top(19.5, 0, 1.0),
              self.top(20.5, 0, 3.0), self.top(50, 0, 1.0)]
        for threshold in (0.3, 0.5, 0.8):
            a = union(rgb, cc, threshold)
            b = unionClean(rgb, chm, cc, threshold, float("inf"))
            self.assertEqual([(p["centreX"], p["score"]) for p in a],
                             [(p["centreX"], p["score"]) for p in b])
        cleaned = unionClean(rgb, chm, cc, 0.5, 0.5)
        self.assertEqual(len(cleaned), len(union(rgb, cc, 0.5)) - 1)   # 19.5 and 20.5 -> one

    def testAgreedBoxesFixedRules(self):
        from tt.tripleFusion import agreedBoxes
        rgb = [self.box(0, 0, 0.30), self.box(10, 0, 0.10), self.box(30, 0, 0.60)]
        chm = [self.box(0.1, 0, 0.30), self.box(10.1, 0, 0.40), self.box(50, 0, 0.55),
               self.box(60, 0, 0.10)]
        out = agreedBoxes(rgb, chm, rgbAt=0.5, chmAt=0.5)
        xs = sorted(round(p["centreX"]) for p in out)
        # pair at 0: both at 0.30 >= half of 0.5 -> kept; pair at 10: 0.10 < 0.25 and
        # 0.40 < 0.5 -> dropped; lone RGB 30 (0.60) and lone CHM 50 (0.55) kept; lone 60 dropped
        self.assertEqual(xs, [0, 30, 50])


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


class TestCommandLine(Fixture):
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


class TestRealScene(unittest.TestCase):
    """
    The regression test. Skipped unless the real data is pointed at, because it
    is the only check that the refactor did not quietly change a number.
    """

    @classmethod
    def setUpClass(cls):
        cls.chm = os.environ.get("TT_CHM")
        cls.crowns = os.environ.get("TT_CROWNS")
        cls.boundary = os.environ.get("TT_BOUNDARY")
        if not (cls.chm and cls.crowns and cls.boundary):
            raise unittest.SkipTest("set TT_CHM, TT_CROWNS and TT_BOUNDARY "
                                    "to run the regression test")

    def testPublishedOperatingPoint(self):
        from tt import Scene, ConCompDetector, CrownEvaluator, TopMerger
        scene = Scene(self.chm, crownsPath=self.crowns,
                      boundaryPath=self.boundary, resolution=0.25,
                      verbose=False)
        detector = ConCompDetector(
            lowerPercentile=10, minTopAreaM2=0.12, topStepM=0.12,
            erosionIterations=1, minTreeAreaM2=0.5, windowSizeM=40.0,
            merger=TopMerger("saddle", epsM=8.0, saddleDropM=0.5),
            verbose=False)
        result = CrownEvaluator(scene).score(detector.detect(scene))
        print("\n    real scene: R %.3f  P %.3f  F1 %.3f  (%d detections)"
              % (result["recall"], result["precision"], result["f1"],
                 result["detections"]))
        self.assertAlmostEqual(result["recall"], 0.933, delta=0.01)
        self.assertAlmostEqual(result["precision"], 0.810, delta=0.01)
        self.assertAlmostEqual(result["f1"], 0.867, delta=0.01)


if __name__ == "__main__":
    try:
        import tt  # noqa: F401
    except ImportError:
        print("Cannot import 'tt'. This file expects the package directory "
              "beside it:\n"
              "    %s/\n        tests.py\n        tt/\n            "
              "__init__.py, scene.py, ..."
              % os.path.dirname(os.path.abspath(__file__)))
        sys.exit(1)
    # warnings=False: unittest would otherwise reset the warning filters to
    # "default" for the run, overriding tt's filter for rasterio's own
    # deprecation warnings and filling the output with them
    unittest.main(verbosity=2, warnings=False)
