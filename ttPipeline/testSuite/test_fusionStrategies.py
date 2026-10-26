"""The combination strategies and the height filter, on hand-built cases."""

import unittest

import numpy as np

from .helpers import asBoxes, asPoints, fusionBoxes, fusionTops, givenProminence


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
        self.boxes = asBoxes([self.box(0, 0, 4, 4, 0.9),     # holds two points
                              self.box(10, 0, 14, 4, 0.6),   # holds none
                              self.box(20, 0, 24, 4, 0.3)])  # holds one, low score
        self.points = asPoints([self.point(1, 1, 12.0), self.point(3, 3, 15.0),
                                self.point(21, 1), self.point(40, 40)])

    def testContainment(self):
        from tt.detectors.fusion import containment
        inside = containment(self.points, self.boxes)
        self.assertEqual(inside.tolist(),
                         [[True, False, False], [True, False, False],
                          [False, False, True], [False, False, False]])

    def testWeakBoxClearOfStrongOnesMerges(self):
        """A weak box overlapping no stronger box merges the points in it."""
        from tt.detectors.fusion import WeakBoxMergedPoints
        boxes = asBoxes([self.box(0, 0, 4, 4, 0.9), self.box(30, 0, 34, 4, 0.1)])
        points = asPoints([self.point(1, 1, 12.0), self.point(31, 1, 9.0),
                           self.point(33, 3, 11.0)])
        result = WeakBoxMergedPoints().combine(boxes, points, threshold=0.5)
        self.assertEqual(sorted(result.x), [1, 33])

    def testWeakBoxOverlappingTooMuchIsIgnored(self):
        """Half of the weak box lies in a strong one: more than 25%."""
        from tt.detectors.fusion import WeakBoxMergedPoints
        boxes = asBoxes([self.box(0, 0, 4, 4, 0.9), self.box(2, 0, 6, 4, 0.1)])
        points = asPoints([self.point(5, 1, 9.0), self.point(5.5, 3, 11.0)])
        self.assertEqual(len(WeakBoxMergedPoints().combine(boxes, points,
                                                           threshold=0.5)), 2)

    def testOverlapIsMeasuredAgainstTheSmallerBox(self):
        """A small box inside a big one overlaps fully, whatever the IoU."""
        from tt.detectors.fusion import overlapOverSmaller
        share = overlapOverSmaller([1, 1, 2, 2], np.array([[0, 0, 10, 10]]))
        self.assertAlmostEqual(float(share[0]), 1.0)
        share = overlapOverSmaller([0, 0, 4, 4], np.array([[3, 0, 7, 4]]))
        self.assertAlmostEqual(float(share[0]), 0.25)

    def testWeakBoxesBlockEachOther(self):
        """Taken strongest first, an accepted weak box blocks a weaker one."""
        from tt.detectors.fusion import WeakBoxMergedPoints
        boxes = asBoxes([self.box(0, 0, 4, 4, 0.3), self.box(2, 0, 6, 4, 0.2)])
        points = asPoints([self.point(1, 1, 9.0), self.point(3, 3, 11.0),
                           self.point(5, 1, 8.0), self.point(5.5, 3, 7.0)])
        result = WeakBoxMergedPoints().combine(boxes, points, threshold=0.5)
        # the 0.3 box is accepted and merges its two points to x=3; the 0.2
        # box overlaps it by half and is refused, so its points both survive
        self.assertEqual(sorted(result.x), [3, 5, 5.5])

    def testConfirmedUsesTwoThresholds(self):
        from tt.detectors.fusion import Confirmed
        kept = Confirmed().combine(self.boxes, self.points, confirmedAt=0.2,
                                   unconfirmedAt=0.5)
        self.assertEqual(list(kept.scores), [0.9, 0.6, 0.3])
        kept = Confirmed().combine(self.boxes, self.points, confirmedAt=0.5,
                                   unconfirmedAt=0.7)
        self.assertEqual(list(kept.scores), [0.9])

    def testUnionAddsOnlyUncoveredPoints(self):
        from tt.detectors.fusion import Union
        result = Union().combine(self.boxes, self.points, threshold=0.5)
        # boxes 0.9 and 0.6 kept; the points in box 0.9 are covered
        self.assertEqual(len(result), 2 + 2)
        self.assertEqual(sorted(result.x[2:]), [21, 40])

    def testBoxMergesPointsItHolds(self):
        from tt.detectors.fusion import BoxMergedPoints
        result = BoxMergedPoints().combine(self.boxes, self.points,
                                           threshold=0.2)
        # the two points in the first box collapse to the higher (x=3)
        self.assertEqual(sorted(result.x), [3, 21, 40])


class TestSaddleUnion(unittest.TestCase):
    """
    Two cones on a 40 m grid, A (16 m) and B (14 m), with bare ground between.
    A's box is drawn wide enough that its corner covers B's top — the case box
    containment gets wrong.
    """

    def setUp(self):
        from affine import Affine
        from tt.detectors.fusion import SaddleSurface
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
        return asPoints([{"centreX": x, "centreY": y, "score": score,
                          "box": [x, y, x, y]}])

    @staticmethod
    def box(c0, c1, r0, r1, score):
        return {"box": [c0, 40 - r1, c1, 40 - r0], "score": score,
                "centreX": (c0 + c1) / 2.0, "centreY": 40 - (r0 + r1) / 2.0}

    def cross(self):
        from tt.detectors.fusion import UnionSaddleCross
        return UnionSaddleCross(self.surface)

    def pool(self):
        from tt.detectors.fusion import UnionSaddlePool
        return UnionSaddlePool(self.surface)

    def testBoxContainmentLosesTheNeighbour(self):
        from tt.detectors.fusion import Union
        wide = asBoxes([self.box(3, 27, 13, 27, 0.9)])
        self.assertEqual(len(Union().combine(wide, self.at(20, 25), 0.5)), 1)

    def testSaddleKeepsTheNeighbour(self):
        wide = asBoxes([self.box(3, 27, 13, 27, 0.9)])
        self.assertEqual(len(self.cross().combine(wide, self.at(20, 25), 0.5)), 2)

    def testSaddleDropsTheTopOfTheBoxesOwnTree(self):
        wide = asBoxes([self.box(3, 27, 13, 27, 0.9)])
        self.assertEqual(len(self.cross().combine(wide, self.at(20, 10), 0.5)), 1)

    def testPoolMergesASplitCrownButCrossDoesNot(self):
        halves = asBoxes([self.box(3, 11, 13, 27, 0.9),
                          self.box(11, 17, 13, 27, 0.6)])
        none = asPoints([])
        self.assertEqual(len(self.pool().combine(halves, none, 0.5)), 1)
        self.assertEqual(len(self.cross().combine(halves, none, 0.5)), 2)

    def testApexFollowsTheMaskOutline(self):
        """Box over both cones; the outline around B puts the apex on B."""
        wide = self.box(3, 32, 13, 27, 0.9)
        self.assertEqual(tuple(self.surface.apexes(asBoxes([wide]))[0]),
                         (20, 10))
        ring = [[19, 40 - 16], [31, 40 - 16], [31, 40 - 24], [19, 40 - 24]]
        self.assertEqual(tuple(self.surface.apexes(
            asBoxes([dict(wide, polygon=ring)]))[0]), (20, 25))

    def testPoolKeepsTwoTreesApart(self):
        boxes = asBoxes([self.box(3, 17, 13, 27, 0.9),
                         self.box(18, 32, 13, 27, 0.8)])
        self.assertEqual(len(self.pool().combine(boxes, asPoints([]), 0.5)), 2)


class TestFusionOutputs(unittest.TestCase):
    """Each strategy's output is kept apart from its score."""

    def testOutputDoesNotReplaceTheDetectionCount(self):
        full = asBoxes([{"centreX": 1.0, "centreY": 2.0, "score": 0.5,
                         "box": [0, 0, 2, 4], "polygon": [[0, 0]]}])
        self.assertEqual(sorted(full.slim()[0]), ["box", "centreX", "centreY",
                                                  "score"])


class TestTripleFusion(unittest.TestCase):
    """RGB and CHM networks agreeing, CC voting, CC adding prominent tops."""

    def testAgreementVoteAndAdditions(self):
        from tt.detectors.fusion import Triple
        triple = givenProminence(Triple)
        rgb = fusionBoxes((0, 0, 0.40), (10, 0, 0.80), (30, 0, 0.25))
        chm = fusionBoxes((0.2, 0, 0.40), (20, 0, 0.60), (30.1, 0, 0.25),
                          (40, 0, 0.05))
        cc = fusionTops((20, 0.5, 1.0),                  # inside the lone CHM box at 20
                        (50, 0, 3.0),                    # free, prominent
                        (60, 0, 0.2),                    # free, not prominent
                        (39.5, 0, 3.0), (40.5, 0, 2.5))  # two in one CHM box
        off = float("inf")
        xs = lambda out: sorted(round(x, 1) for x in out.x)
        # agreement only: the pair at 0 (both 0.40 >= low) and the confident RGB box
        out = triple.combine(rgb, chm, cc, high=0.7, low=0.3, vote=off, prominence=off)
        self.assertEqual(xs(out), [0.0, 10.0])
        # CC's vote rescues the CHM box at 20 (0.60 >= vote) that holds a top
        out = triple.combine(rgb, chm, cc, high=0.7, low=0.3, vote=0.5, prominence=off)
        self.assertEqual(xs(out), [0.0, 10.0, 20.0])
        # prominent free tops are added; of the two in one CHM box, one only
        out = triple.combine(rgb, chm, cc, high=0.7, low=0.3, vote=off, prominence=2.0)
        self.assertEqual(xs(out), [0.0, 10.0, 39.5, 50.0])

    def testOnePerPair(self):
        from tt.detectors.fusion import Triple
        out = givenProminence(Triple).combine(
            fusionBoxes((0, 0, 0.9)), fusionBoxes((0.3, 0.2, 0.95)), fusionTops(),
            high=0.5, low=0.3, vote=float("inf"), prominence=float("inf"))
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out.scores[0], 0.95)

    def testProminence(self):
        from affine import Affine
        from tt.detectors.fusion import SaddleSurface, prominenceAt
        rows, columns = np.mgrid[0:80, 0:80].astype(float)
        chm = np.maximum(5 * np.clip(1 - np.hypot(rows - 40, columns - 20) / 8, 0, 1),
                         np.maximum(6 * np.clip(1 - np.hypot(rows - 40, columns - 50) / 8, 0, 1),
                                    0.0)).astype(np.float32)
        surface = SaddleSurface(chm, Affine(1, 0, 0, 0, -1, 80), 1.0, dropM=0.5)
        self.assertAlmostEqual(prominenceAt(surface, 20.5, 80 - 40.5), 5.0, delta=0.15)
        self.assertAlmostEqual(prominenceAt(surface, 50.5, 80 - 40.5), 6.0, delta=0.15)
        tops = fusionTops((20.5, 39.5, 0), (50.5, 39.5, 0))
        np.testing.assert_allclose(surface.prominences(tops), [5.0, 6.0], atol=0.15)


class TestReducedFusion(unittest.TestCase):
    """The union plus one idea each: off = the union, cleaning keeps one top per CHM box."""

    def testCleaningKeepsTheMostProminentTopPerChmBox(self):
        from tt.detectors.fusion import UnionClean
        strategy = givenProminence(UnionClean)
        chm = fusionBoxes((0, 0, 0.8), (20, 0, 0.15))
        tops = fusionTops((-0.5, 0, 1.0), (0.6, 0, 3.0), (10, 10, 0.5),
                          (19.5, 0, 2.0), (20.5, 0, 1.0))
        xs = lambda out: sorted(round(x, 1) for x in out.x)
        # CHM box at 0 (score 0.8) groups the first two; the one at 20 needs cleanAt <= 0.15
        self.assertEqual(xs(strategy.cleanedTops(tops, chm, 0.5)), [0.6, 10, 19.5, 20.5])
        self.assertEqual(xs(strategy.cleanedTops(tops, chm, 0.1)), [0.6, 10, 19.5])
        self.assertEqual(xs(strategy.cleanedTops(tops, chm, float("inf"))), xs(tops))

    def testOffIsTheUnion(self):
        from tt.detectors.fusion import Union, UnionClean
        rgb = fusionBoxes((0, 0, 0.9), (10, 0, 0.4), (30, 0, 0.7))
        chm = fusionBoxes((20, 0, 0.9))
        cc = fusionTops((0.2, 0, 1.0), (10, 0, 2.0), (19.5, 0, 1.0), (20.5, 0, 3.0),
                        (50, 0, 1.0))
        clean = givenProminence(UnionClean)
        for threshold in (0.3, 0.5, 0.8):
            a = Union().combine(rgb, cc, threshold)
            b = clean.combine(rgb, chm, cc, threshold, float("inf"))
            self.assertEqual(list(zip(a.x, a.scores)), list(zip(b.x, b.scores)))
        cleaned = clean.combine(rgb, chm, cc, 0.5, 0.5)
        self.assertEqual(len(cleaned), len(Union().combine(rgb, cc, 0.5)) - 1)

    def testAgreedBoxesFixedRules(self):
        from tt.detectors.fusion import AgreeClean
        strategy = AgreeClean()
        strategy.memberThresholds = [0.5, 0.5, -np.inf]
        rgb = fusionBoxes((0, 0, 0.30), (10, 0, 0.10), (30, 0, 0.60))
        chm = fusionBoxes((0.1, 0, 0.30), (10.1, 0, 0.40), (50, 0, 0.55), (60, 0, 0.10))
        out = strategy.agreed(rgb, chm)
        # pair at 0: both at 0.30 >= half of 0.5 -> kept; pair at 10: 0.10 < 0.25 and
        # 0.40 < 0.5 -> dropped; lone RGB 30 (0.60) and lone CHM 50 (0.55) kept; lone 60 dropped
        self.assertEqual(sorted(round(x) for x in out.x), [0, 30, 50])


class TestHeightFilter(unittest.TestCase):
    """RGB boxes over low vegetation are removed using the height model."""

    def setUp(self):
        from affine import Affine
        from tt.detectors.fusion import SaddleSurface
        rows, columns = np.mgrid[0:60, 0:60]
        chm = np.zeros((60, 60), np.float32)
        # four trees of 8 m, and four bushes of 0.8 m, on a 1 m grid
        self.trees = [(10, 10), (10, 40), (40, 10), (40, 40)]
        self.bushes = [(25, 10), (25, 40), (10, 25), (40, 25)]
        for (r, c), h in [(t, 8.0) for t in self.trees] + [(b, 0.8) for b in self.bushes]:
            chm = np.maximum(chm, h * np.clip(1 - np.hypot(rows - r, columns - c) / 4.0, 0, 1))
        self.surface = SaddleSurface(chm, Affine(1, 0, 0, 0, -1, 60), 1.0, dropM=0.5)

    @staticmethod
    def boxes(places, score=0.9):
        from tt.data import Detections
        xs = [c + 0.5 for _, c in places]
        ys = [60 - r - 0.5 for r, _ in places]
        return Detections(xs, ys, [score] * len(places),
                          boxes=[[x - 3, y - 3, x + 3, y + 3]
                                 for x, y in zip(xs, ys)], geometry="boxes")

    def strategy(self, cls):
        return cls(heightSurface=self.surface)

    def testLowBoxesRemovedTallKept(self):
        from tt.detectors.fusion import FusionStrategy
        strategy = self.strategy(FusionStrategy)
        boxes = self.boxes(self.trees + self.bushes)
        self.assertEqual(len(strategy.tallEnough(boxes, 0.0)), 8)
        kept = strategy.tallEnough(boxes, 1.0)
        self.assertEqual(sorted(zip(np.round(kept.x), np.round(kept.y))),
                         sorted((c, 60 - r) for r, c in self.trees))

    def testOwnCacheNotAnotherSurfacesApex(self):
        from affine import Affine
        from tt.detectors.fusion import FusionStrategy, SaddleSurface
        bush = self.boxes(self.bushes[:1])
        other = SaddleSurface(self.surface.chm * 0 + 9.0,
                              Affine(1, 0, 0, 0, -1, 60), 1.0)
        other.apexes(bush)           # another surface's cache on the boxes
        self.assertEqual(len(self.strategy(FusionStrategy)
                             .tallEnough(bush, 1.0)), 0)

    def testUnionHighDropsBushBoxesKeepsPoints(self):
        from tt.data import Detections
        from tt.detectors.fusion import UnionHigh
        point = Detections([55.5], [4.5], [10.0])
        out = self.strategy(UnionHigh).combine(
            self.boxes(self.trees + self.bushes), point, threshold=0.5,
            boxMinHeightM=1.0)
        self.assertEqual(len(out), 5)                  # four trees and the point

    def testCrossValidationTunesTheFilterOnBushes(self):
        import geopandas as gpd
        from shapely.geometry import Point, box as shapeBox
        from tt.data import Detections
        from tt.experiments.fusion import FusionCrossValidation
        from tt.scoring import CrownFrame

        class Run(object):
            def __init__(self, detections, threshold):
                self.detections, self.t = detections, threshold
            def has(self, block):
                return True
            def fold(self, block):
                return self.detections
            def validation(self, block, blocks):
                other = "s_b01" if block == "s_b00" else "s_b00"
                return other, self.detections
            def threshold(self, block):
                return self.t

        crowns = gpd.GeoDataFrame({"class_code": ["tree"] * 4},
                                  geometry=[Point(c + 0.5, 60 - r - 0.5).buffer(3)
                                            for r, c in self.trees])
        blocks = {"s_b00": shapeBox(0, 0, 60, 60), "s_b01": shapeBox(0, 0, 60, 60)}
        validation = FusionCrossValidation.pair(
            CrownFrame(crowns), blocks,
            Run(self.boxes(self.trees + self.bushes), 0.5),
            Run(Detections([], []), -np.inf),
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
        from tt.detectors.fusion import HEIGHT_STRATEGIES, STRATEGIES
        self.assertFalse(set(HEIGHT_STRATEGIES) & set(STRATEGIES))


if __name__ == "__main__":
    unittest.main()
