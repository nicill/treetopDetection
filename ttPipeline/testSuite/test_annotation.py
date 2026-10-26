"""The annotation tools."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np


class TestCrownEdges(unittest.TestCase):
    """Thin canopy edges outside a crown go to it; anything thicker does not."""

    def testRimsAddedNeighboursSplitUnannotatedTreeLeft(self):
        from tt.annotation.crownEdges import findEdges
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


class TestAreaExamples(unittest.TestCase):
    """Example images of the parts left out of the scored area."""

    def testWindowsFoundAndDrawn(self):
        import geopandas as gpd
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import Point, box
        from tt.annotation import areaExamples as ae
        d = tempfile.mkdtemp(prefix="ttExamples")
        try:
            crs, x0, y0, side = "EPSG:32654", 401000.0, 4268000.0, 400
            rng = np.random.default_rng(1)
            # canopy everywhere; trees on a 6 m lattice; only the west half is
            # annotated, so the east half is left out of the area
            trees = [(x0 + cx, y0 - cy) for cx in np.arange(3, side, 6.0)
                     for cy in np.arange(3, side, 6.0)]
            chm = np.full((side * 2, side * 2), 6.0, np.float32)
            tf = from_origin(x0, y0, 0.5, 0.5)
            with rasterio.open(os.path.join(d, "chm.tif"), "w", driver="GTiff",
                               height=side * 2, width=side * 2, count=1,
                               dtype="float32", crs=crs, transform=tf) as out:
                out.write(chm, 1)
            for i, name in enumerate(("a", "b")):
                with rasterio.open(os.path.join(d, "m%s.tif" % name), "w",
                                   driver="GTiff", height=800, width=400,
                                   count=3, dtype="uint8", crs=crs,
                                   transform=from_origin(x0 + 200 * i, y0, 0.5,
                                                         0.5)) as out:
                    out.write(rng.integers(60, 200, (3, 800, 400),
                                           dtype=np.uint8))
            kept = [t for t in trees if t[0] < x0 + side / 2]
            crowns = gpd.GeoDataFrame(geometry=[Point(*t).buffer(2.4)
                                                for t in kept], crs=crs)
            crowns.to_file(os.path.join(d, "crowns.shp"))
            gpd.GeoDataFrame(geometry=[box(x0, y0 - side, x0 + side / 2, y0)],
                             crs=crs).to_file(os.path.join(d, "area.shp"))
            out = os.path.join(d, "ex")
            self.assertEqual(ae.main([
                "--mosaics", os.path.join(d, "m*.tif"),
                "--chm", os.path.join(d, "chm.tif"),
                "--polygons", os.path.join(d, "crowns.shp"),
                "--area", os.path.join(d, "area.shp"), "--output", out,
                "--windowM", "60", "--edge", "1", "--unused", "1", "--used", "1",
                "--separationM", "60", "--pixelM", "0.5"]), 0)
            rows = json.load(open(os.path.join(out, "examples.json")))
            kinds = {r["kind"]: r for r in rows}
            self.assertEqual(set(kinds), {"edge", "mostly left out", "used"})
            self.assertAlmostEqual(kinds["edge"]["unused"], 0.5, delta=0.1)
            self.assertGreater(kinds["mostly left out"]["unused"], 0.9)
            self.assertEqual(kinds["used"]["unused"], 0.0)
            self.assertGreater(kinds["used"]["coveredUsed"], 0.3)
            self.assertLess(kinds["mostly left out"]["coveredUnused"], 0.05)
            for r in rows:
                self.assertTrue(os.path.exists(r["file"]))
            self.assertTrue(os.path.exists(os.path.join(out, "examples_all.png")))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestUnannotatedTrees(unittest.TestCase):
    """The mask of unannotated tree canopy finds the trees left out."""

    def testUnannotatedTreesFoundAndShrubsNot(self):
        import geopandas as gpd
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import Point, box
        from tt.annotation import unannotatedTrees as ut
        d = tempfile.mkdtemp(prefix="ttUnann")
        try:
            crs, x0, y0, side, res = "EPSG:32654", 401000.0, 4268000.0, 60, 0.25
            rng = np.random.default_rng(3)
            n = int(side / res)
            yy, xx = np.mgrid[0:n, 0:n]
            X, Y = x0 + (xx + .5) * res, y0 - (yy + .5) * res
            trees = [(x0 + cx, y0 - cy, 2.5, rng.uniform(6, 14))
                     for cx in np.arange(4, side, 7.0)
                     for cy in np.arange(4, side, 7.0)]
            chm = np.zeros((n, n), np.float32)
            for tx, ty, r, h in trees:
                dd = np.hypot(X - tx, Y - ty)
                chm = np.maximum(chm, np.where(dd < r, h * (1 - 0.4 * dd / r),
                                               0)).astype(np.float32)
            chm[(X > x0 + 45) & (Y < y0 - 45) & (chm < 1)] = 2.5    # shrubs
            corner = (X + (y0 - Y) * 0 > x0 + 40) & ((y0 - Y) < 12)  # no image
            chm[corner] = 9.0                                        # canopy there
            with rasterio.open(os.path.join(d, "chm.tif"), "w", driver="GTiff",
                               height=n, width=n, count=1, dtype="float32",
                               crs=crs, transform=from_origin(x0, y0, res, res)) as o:
                o.write(chm, 1)
            with rasterio.open(os.path.join(d, "m.tif"), "w", driver="GTiff",
                               height=n, width=n, count=3, dtype="uint8",
                               crs=crs, transform=from_origin(x0, y0, res, res)) as o:
                image = np.full((3, n, n), 90, np.uint8)
                image[:, corner] = 0                       # the empty corner
                o.write(image)
            annotated = [t for t in trees if t[0] < x0 + 30]
            gpd.GeoDataFrame(geometry=[Point(t[0], t[1]).buffer(t[2] * 0.9)
                                       for t in annotated], crs=crs) \
                .to_file(os.path.join(d, "poly.shp"))
            gpd.GeoDataFrame(geometry=[box(x0, y0 - side, x0 + 30, y0)],
                             crs=crs).to_file(os.path.join(d, "area.shp"))
            out = os.path.join(d, "out")
            self.assertEqual(ut.main([
                "--chm", os.path.join(d, "chm.tif"),
                "--polygons", os.path.join(d, "poly.shp"),
                "--mosaics", os.path.join(d, "m.tif"),
                "--area", os.path.join(d, "area.shp"), "--output", out,
                "--rgbPixelM", "0.25"]), 0)
            summary = json.load(open(os.path.join(out, "summary.json")))
            real = [t for t in trees if t[0] >= x0 + 30 and not
                    (t[0] > x0 + 40 and y0 - t[1] < 12)]
            self.assertEqual(summary["topsRough"], len(real))   # not the corner
            self.assertEqual(summary["topsInArea"], 0)
            frame = gpd.read_file(os.path.join(out, "unannotated.shp"))
            self.assertEqual(len(frame), len(real))
            self.assertTrue((frame["maxH"] >= 4).all())        # no shrubs
            hits = frame.geometry.centroid.apply(
                lambda p: p.x > x0 + 40 and y0 - p.y < 12).sum()
            self.assertEqual(int(hits), 0)       # nothing in the empty corner
            for name in ("overview.png", "rgb_unannotated.tif",
                         "unannotated_tops.shp"):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestCandidateTrees(unittest.TestCase):
    """Candidate crowns for the unannotated trees, and the way back to the
    annotator's frame."""

    def testCandidatesFoundAndRoundTrip(self):
        import geopandas as gpd
        import rasterio
        from shapely.affinity import translate
        from rasterio.transform import from_origin
        from shapely.geometry import Point
        from tt.annotation import candidateTrees as ct
        from tt.annotation import unannotatedTrees as ut
        d = tempfile.mkdtemp(prefix="ttCand")
        try:
            crs, x0, y0, side, res = "EPSG:32654", 401000.0, 4268000.0, 60, 0.25
            rng = np.random.default_rng(5)
            n = int(side / res)
            yy, xx = np.mgrid[0:n, 0:n]
            X, Y = x0 + (xx + .5) * res, y0 - (yy + .5) * res
            trees = [(x0 + cx, y0 - cy, 2.5, rng.uniform(6, 14))
                     for cx in np.arange(4, side, 7.0)
                     for cy in np.arange(4, side, 7.0)]
            chm = np.zeros((n, n), np.float32)
            for tx, ty, r, h in trees:
                dd = np.hypot(X - tx, Y - ty)
                chm = np.maximum(chm, np.where(dd < r, h * (1 - 0.4 * dd / r),
                                               0)).astype(np.float32)
            with rasterio.open(os.path.join(d, "chm.tif"), "w", driver="GTiff",
                               height=n, width=n, count=1, dtype="float32",
                               crs=crs, transform=from_origin(x0, y0, res, res)) as o:
                o.write(chm, 1)
            for i, name in enumerate(("newtrain", "test")):
                with rasterio.open(os.path.join(d, "m_%s.tif" % name), "w",
                                   driver="GTiff", height=n, width=n // 2,
                                   count=3, dtype="uint8", crs=crs,
                                   transform=from_origin(x0 + 30 * i, y0, res, res)) as o:
                    o.write(np.full((3, n, n // 2), 90, np.uint8))
            annotated = [t for t in trees if t[0] < x0 + 30]
            missing = [t for t in trees if t[0] >= x0 + 30]
            gpd.GeoDataFrame(geometry=[Point(t[0], t[1]).buffer(t[2] * 0.9)
                                       for t in annotated], crs=crs) \
                .to_file(os.path.join(d, "poly.shp"))
            like = os.path.join(d, "like.shp")        # the annotator's frame
            gpd.GeoDataFrame(geometry=[Point(0, 0).buffer(1)], crs=crs) \
                .to_crs("EPSG:3095").to_file(like)
            un = os.path.join(d, "un")
            self.assertEqual(ut.main([
                "--chm", os.path.join(d, "chm.tif"), "--polygons",
                os.path.join(d, "poly.shp"), "--mosaics",
                os.path.join(d, "m_*.tif"), "--output", un,
                "--rgbPixelM", "0.5"]), 0)
            out = os.path.join(d, "cand")
            shifts = ["--shift", "newtrain=-1.37,-0.43", "--shift",
                      "test=-1.67,-0.47"]
            self.assertEqual(ct.main([
                "--chm", os.path.join(d, "chm.tif"), "--unannotated",
                os.path.join(un, "unannotated.shp"), "--polygons",
                os.path.join(d, "poly.shp"), "--mosaics",
                os.path.join(d, "m_*.tif"), "--like", like,
                "--output", out, "--overviewPixelM", "0.5"] + shifts), 0)
            found = gpd.read_file(os.path.join(out, "candidates.shp"))
            self.assertAlmostEqual(len(found), len(missing), delta=2)
            hit = sum(found.contains(Point(t[0], t[1])).any() for t in missing)
            self.assertGreaterEqual(hit, len(missing) - 2)
            self.assertFalse(found.intersects(
                gpd.read_file(os.path.join(d, "poly.shp")).union_all()).any())
            self.assertTrue(set(found["conf"]) <= {"A", "B", "C"})
            # the way back: forward conversion of the original-frame file
            back = gpd.read_file(os.path.join(out, "candidates_original_frame.shp"))
            self.assertEqual(set(back["name"]), {"CAND"})
            forward = back.to_crs(crs)
            for geom, ref in zip(forward.geometry, found.geometry):
                shift = (-1.37, -0.43) if ref.centroid.x < x0 + 30 else (-1.67, -0.47)
                moved = translate(geom, *shift)
                self.assertLess(moved.centroid.distance(ref.centroid), 0.02)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestCandidateValidation(unittest.TestCase):
    """The proposal method inside annotated crowns, where the answer is known."""

    def testOneCandidatePerCrown(self):
        import geopandas as gpd
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import Point
        from tt.annotation import candidateTrees as ct
        d = tempfile.mkdtemp(prefix="ttCandVal")
        try:
            crs, x0, y0, side, res = "EPSG:32654", 401000.0, 4268000.0, 60, 0.25
            rng = np.random.default_rng(9)
            n = int(side / res)
            yy, xx = np.mgrid[0:n, 0:n]
            X, Y = x0 + (xx + .5) * res, y0 - (yy + .5) * res
            trees = [(x0 + cx, y0 - cy, 2.5, rng.uniform(6, 14))
                     for cx in np.arange(4, side, 7.0)
                     for cy in np.arange(4, side, 7.0)]
            chm = np.zeros((n, n), np.float32)
            for tx, ty, r, h in trees:
                dd = np.hypot(X - tx, Y - ty)
                chm = np.maximum(chm, np.where(dd < r, h * (1 - 0.4 * dd / r),
                                               0)).astype(np.float32)
            with rasterio.open(os.path.join(d, "chm.tif"), "w", driver="GTiff",
                               height=n, width=n, count=1, dtype="float32",
                               crs=crs, transform=from_origin(x0, y0, res, res)) as o:
                o.write(chm, 1)
            with rasterio.open(os.path.join(d, "m.tif"), "w", driver="GTiff",
                               height=n, width=n, count=3, dtype="uint8",
                               crs=crs, transform=from_origin(x0, y0, res, res)) as o:
                o.write(np.full((3, n, n), 90, np.uint8))
            gpd.GeoDataFrame(geometry=[Point(t[0], t[1]).buffer(t[2] * 0.9)
                                       for t in trees], crs=crs) \
                .to_file(os.path.join(d, "poly.shp"))
            out = os.path.join(d, "val")
            self.assertEqual(ct.main([
                "--chm", os.path.join(d, "chm.tif"), "--polygons",
                os.path.join(d, "poly.shp"), "--mosaics",
                os.path.join(d, "m.tif"), "--output", out, "--validate"]), 0)
            check = json.load(open(os.path.join(out, "validation.json")))
            self.assertEqual(check["crowns"], len(trees))
            self.assertGreater(check["foundOnce"], 0.95)
            self.assertLess(check["split"], 0.03)
            self.assertLess(check["missed"], 0.03)
            self.assertGreater(check["medianIoU"], 0.3)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestEdgePatches(unittest.TestCase):
    """The edge part of the uncovered canopy, written and drawn."""

    def testEdgeAndInteriorSeparated(self):
        import types
        import geopandas as gpd
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import Point, box
        from tt.annotation import edgePatches as ep
        from tt.prepare import quebec as qp
        d = tempfile.mkdtemp(prefix="ttEdge")
        try:
            site = os.path.join(d, "site")
            os.makedirs(site)
            crs, x0, y0, W, H, res = "EPSG:32654", 500000.0, 4400000.0, 40, 40, 0.25
            nx, ny = int(W / res), int(H / res)
            cc, rr = np.meshgrid(np.arange(nx), np.arange(ny))
            X, Y = x0 + (cc + .5) * res, y0 - (rr + .5) * res
            trees = [(x0 + cx, y0 - cy) for cx in np.arange(2, W, 3.2)
                     for cy in np.arange(2, H, 3.2)]
            chm = np.zeros((ny, nx), np.float32)
            for tx, ty in trees:
                chm[np.hypot(X - tx, Y - ty) < 1.4] = 6.0
            for name, data, count in (("chm", chm[None], 1),
                                      ("rgb", np.full((3, ny, nx), 90, np.uint8), 3)):
                with rasterio.open(os.path.join(site, name + ".tif"), "w",
                                   driver="GTiff", height=ny, width=nx,
                                   count=count, dtype=data.dtype, crs=crs,
                                   transform=from_origin(x0, y0, res, res)) as o:
                    o.write(data)
            area = gpd.GeoDataFrame({"name": ["a"]}, geometry=[box(
                x0 + 1, y0 - H + 1, x0 + W - 1, y0 - 1)], crs=crs)
            area.to_file(os.path.join(site, "area.shp"))
            # west column of trees and one interior tree left unannotated
            interiorTree = min(trees, key=lambda t: np.hypot(t[0] - x0 - W / 2,
                                                             t[1] - y0 + H / 2))
            ann = [t for t in trees if t[0] > x0 + 3 and t != interiorTree]
            crowns = gpd.GeoDataFrame({"class_code": ["tree"] * len(ann)},
                                      geometry=[Point(*t).buffer(1.35) for t in ann],
                                      crs=crs)
            crowns.to_file(os.path.join(site, "crowns.shp"))
            _, frame = qp.uncoveredCanopy(
                os.path.join(site, "chm.tif"), crowns, area,
                types.SimpleNamespace(crownMarginM=0.2, canopyM=1.5,
                                      minPatchM2=0.5))
            frame.to_file(os.path.join(site, "uncovered.shp"))
            out = os.path.join(d, "out")
            self.assertEqual(ep.main(["--site", site, "--output", out,
                                      "--like", os.path.join(site, "area.shp"),
                                      "--zooms", "2", "--zoomM", "12"]), 0)
            summary = json.load(open(os.path.join(out, "summary.json")))
            self.assertGreaterEqual(summary["edgePatches"], 10)   # the west column
            edge = gpd.read_file(os.path.join(out, "edge_patches.shp"))
            self.assertFalse(edge.contains(Point(*interiorTree)).any())
            self.assertTrue((edge.centroid.x < x0 + 4).all())
            for name in ("overview.png", "zooms.png",
                         "edge_patches_original.shp"):
                self.assertTrue(os.path.exists(os.path.join(out, name)), name)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestEdgeOnlyArea(unittest.TestCase):
    """The scored area with only the edge part of the uncovered canopy cut."""

    def testInteriorScoredAgainEdgeStillCut(self):
        import types
        import geopandas as gpd
        import rasterio
        from rasterio.transform import from_origin
        from shapely.geometry import Point, box
        from tt.prepare import quebec as qp
        from tt.annotation.edgeOnlyArea import main
        d = tempfile.mkdtemp(prefix="ttEdgeOnly")
        try:
            crs, x0, y0, W, H, res = "EPSG:32654", 500000.0, 4400000.0, 40, 40, 0.25
            nx, ny = int(W / res), int(H / res)
            cc, rr = np.meshgrid(np.arange(nx), np.arange(ny))
            X, Y = x0 + (cc + .5) * res, y0 - (rr + .5) * res
            trees = [(x0 + cx, y0 - cy) for cx in np.arange(2, W, 3.2)
                     for cy in np.arange(2, H, 3.2)]
            chm = np.zeros((ny, nx), np.float32)
            for tx, ty in trees:
                chm[np.hypot(X - tx, Y - ty) < 1.4] = 6.0
            with rasterio.open(os.path.join(d, "chm.tif"), "w", driver="GTiff",
                               height=ny, width=nx, count=1, dtype="float32",
                               crs=crs, transform=from_origin(x0, y0, res, res)) as o:
                o.write(chm, 1)
            area = gpd.GeoDataFrame({"name": ["a"]}, geometry=[box(
                x0 + 1, y0 - H + 1, x0 + W - 1, y0 - 1)], crs=crs)
            inner = min(trees, key=lambda t: np.hypot(t[0] - x0 - W / 2,
                                                      t[1] - y0 + H / 2))
            ann = [t for t in trees if t[0] > x0 + 3 and t != inner]
            crowns = gpd.GeoDataFrame({"class_code": ["tree"] * len(ann)},
                                      geometry=[Point(*t).buffer(1.35) for t in ann],
                                      crs=crs)
            args = types.SimpleNamespace(crownMarginM=0.2, canopyM=1.5,
                                         minPatchM2=0.5)
            _, unc = qp.uncoveredCanopy(os.path.join(d, "chm.tif"), crowns,
                                        area, args)
            tops = qp.crownTops(os.path.join(d, "chm.tif"), crowns)
            scored, ign, region, _ = qp.scoredRegion(crowns, area, unc, tops, 1.0)
            area.to_file(os.path.join(d, "area.shp"))
            qp.writeLayers(d, {"uncovered": unc, "ignored": ign,
                               "scoredArea": region})
            self.assertEqual(main([d]), 0)
            new = gpd.read_file(os.path.join(d, "scoredAreaEdge.shp")).union_all()
            self.assertTrue(new.contains(Point(*inner)))            # scored again
            west = [t for t in trees if t[0] < x0 + 3 and y0 - H + 4 < t[1] < y0 - 4]
            self.assertFalse(any(new.contains(Point(*t)) for t in west))  # still cut
            self.assertGreater(new.area, region.area.sum())
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
