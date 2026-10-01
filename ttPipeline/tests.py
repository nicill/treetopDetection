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

import os
import shutil
import sys
import tempfile
import unittest

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
        for threshold, f1 in curve.items():
            kept = [p for p in predictions if p["score"] >= threshold]
            self.assertAlmostEqual(
                f1, dc.evaluateDetections(kept, scene.crowns, region)["f1"],
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
        from tt.calibration import ParameterProposal, EROSION_MAX
        _, zones = self.zones()
        settings = ParameterProposal(zones).settings()
        self.assertTrue(1 <= settings["lowerPercentile"] <= 50)
        self.assertTrue(0 <= settings["erosionIterations"] <= EROSION_MAX)
        self.assertGreater(settings["minTopAreaM2"], 0)
        self.assertAlmostEqual(settings["topStepM"],
                               min(0.5, max(0.05,
                                            settings["saddleDropM"] / 2)))

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
