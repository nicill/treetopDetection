"""
PseudoCrowns — crown polygons and bounding boxes from detected treetops.

Ported from the makeBoxesTopsLI notebook, which built boxes for the YURF sites
by cutting a species mask with the Voronoi diagram of its treetops and then
shrinking the resulting boxes using a distance transform.

The one substitution: that notebook had hand-drawn species masks to cut up, and
here there are none. The canopy mask is instead every cell the scene keeps: the
CHM above the scene's minimum height (1 m for the Quebec plantations, the
height below which a crown counts as invisible; 2 m for Terelj). One fixed,
explained threshold.

A local percentile cut can be asked for (percentile > 0): per 40 m window the
lowest share of canopy heights is dropped, as the detector does. It is not the
default because it removes short trees standing among tall ones whatever their
height, so their pseudo-crowns would shrink or vanish.

The method, per treetop:

  1. Cut the canopy mask along the Voronoi edges of the tops. Every top then
     sits in its own component, because a Voronoi cell is by construction the
     region closer to that top than to any other.
  2. Count how many tops share the *uncut* canopy component, which says how
     crowded the tree is and decides how much to trust the Voronoi box:

     isolated (< MANY tops)   take the component's bounding box outright. For
                              a tree standing alone the component is the tree.
     crowded  (< LOTS tops)   shrink to the largest box centred on the top and
                              still inside the Voronoi box. Treetops are
                              annotated at crown centres, so a centred box is
                              the better guess and the Voronoi box alone tends
                              to overreach into the neighbour.
     dense    (>= LOTS tops)  interpolate between that centred box and the
                              distance-transform box — twice the distance from
                              the top to the edge of the canopy, the largest
                              square certainly inside the canopy. The weight
                              (0.65 toward Voronoi) is the notebook's, set by
                              eye on a handful of cases; it has not been fitted
                              here either.

A caution on what these are. The boxes are a construction from points, not
observations of crowns, so they inherit every error in the tops: a missed tree
leaves its neighbours' cells to swallow the gap, and a spurious top carves a
cell out of a real crown. They are worth what the treetops are worth.
"""

from collections import defaultdict
import os

import cv2
import geopandas as gpd
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from shapely.geometry import box as shapelyBox


MANY = 2
LOTS = 6
VORONOI_WEIGHT = 0.65


class PseudoCrowns(object):

    def __init__(self, scene, tops, percentile=0, windowSizeM=40.0,
                 many=MANY, lots=LOTS, voronoiWeight=VORONOI_WEIGHT,
                 verbose=True):
        self.scene = scene
        self.tops = tops
        self.percentile = int(percentile)
        self.windowSizeM = float(windowSizeM)
        self.many = int(many)
        self.lots = int(lots)
        self.voronoiWeight = float(voronoiWeight)
        self.verbose = verbose

        self.canopy = None
        self.labels = None
        self.boxes = None
        self.grouping = None

    def __repr__(self):
        return ("PseudoCrowns(%d tops, %s)"
                % (len(self.tops),
                   "not built" if self.boxes is None else "%d boxes"
                   % len(self.boxes)))

    # ------------------------------------------------------------------ #

    def canopyMask(self):
        """
        Canopy: every cell the scene keeps (above its minimum height), or,
        with percentile > 0, where the CHM also stands above its local
        percentile.

        The threshold is computed per tile and then bilinearly upsampled rather
        than applied tile by tile, because a piecewise-constant threshold
        leaves straight seams across the mask at the tile borders and those
        seams cut crowns in half.
        """

        if self.percentile <= 0:
            canopy = (self.scene.chm > 0).astype(np.uint8) * 255
            if self.verbose:
                print("[crowns] canopy: every cell above the scene's minimum "
                      "height, %.1f%% of the grid"
                      % (100.0 * np.count_nonzero(canopy) / canopy.size))
            return canopy

        tile = self.scene.metresToPixels(self.windowSizeM, 8)
        rows, columns = self.scene.chm.shape
        tileRows = max(1, int(np.ceil(rows / tile)))
        tileColumns = max(1, int(np.ceil(columns / tile)))

        coarse = np.zeros((tileRows, tileColumns), np.float32)
        for r in range(tileRows):
            for c in range(tileColumns):
                patch = self.scene.chm[r * tile:(r + 1) * tile,
                                       c * tile:(c + 1) * tile]
                values = patch[patch > 0]
                coarse[r, c] = (np.percentile(values, self.percentile)
                                if values.size else np.inf)

        finite = coarse[np.isfinite(coarse)]
        coarse[~np.isfinite(coarse)] = finite.max() if finite.size else 0.0
        threshold = cv2.resize(coarse, (columns, rows),
                               interpolation=cv2.INTER_LINEAR)

        canopy = ((self.scene.chm > 0)
                  & (self.scene.chm >= threshold)).astype(np.uint8) * 255
        if self.verbose:
            print("[crowns] canopy at the %dth local percentile: %.1f%% of "
                  "the grid" % (self.percentile,
                                100.0 * np.count_nonzero(canopy) / canopy.size))
        return canopy

    # ------------------------------------------------------------------ #

    def voronoiEdges(self, thickness=1):
        """
        The Voronoi edges of the tops, painted into a mask.

        Corner points are appended so cells at the edge of the scene close
        instead of running to infinity; scipy drops unbounded regions, and
        without the corners the outermost trees lose their cells entirely.
        """

        rows, columns = self.scene.chm.shape
        points = self.tops.pixels
        if len(points) < 4:
            return np.zeros((rows, columns), np.uint8)

        padded = np.vstack([points,
                            [[0, 0], [columns, 0], [0, rows],
                             [columns, rows]]])
        diagram = Voronoi(padded)

        edges = np.zeros((rows, columns), np.uint8)
        for region in diagram.regions:
            if not region or -1 in region:
                continue
            polygon = np.array([[int(diagram.vertices[p][0]),
                                 int(diagram.vertices[p][1])]
                                for p in region], np.int32)
            cv2.polylines(edges, [polygon], True, 255, thickness=thickness)
        return edges

    def groupingCodes(self, canopy):
        """
        0, 1 or 2 per top: how crowded the uncut canopy component it sits in is.

        This is measured before the Voronoi cut, because the question is how
        many trees share a blob of canopy, not how many share a cell.
        """

        count, labels = cv2.connectedComponents(canopy, connectivity=8)
        perComponent = defaultdict(list)
        for index, point in enumerate(self.tops.points):
            perComponent[labels[point[1], point[0]]].append(index)

        grouping = [0] * len(self.tops)
        for members in perComponent.values():
            code = 0 if len(members) < self.many else (
                1 if len(members) < self.lots else 2)
            for index in members:
                grouping[index] = code
        return grouping

    # ------------------------------------------------------------------ #

    def build(self):
        """Compute the boxes and the per-top component labels."""

        self.canopy = self.canopyMask()
        self.grouping = self.groupingCodes(self.canopy)

        cut = self.canopy.copy()
        cut[self.voronoiEdges() != 0] = 0
        # 4-connectivity, or a one-pixel Voronoi edge leaks diagonally and two
        # cells merge back into one component
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            cut, connectivity=4)
        self.labels = labels

        distance = cv2.distanceTransform(self.canopy, cv2.DIST_L2, 3) \
            if 2 in self.grouping else None

        boxes = []
        for index, (x, y, _) in enumerate(self.tops.points):
            label = labels[y, x]
            if label == 0:
                boxes.append(self._fallbackBox(x, y, distance))
                continue
            box = (stats[label, cv2.CC_STAT_LEFT],
                   stats[label, cv2.CC_STAT_TOP],
                   stats[label, cv2.CC_STAT_WIDTH],
                   stats[label, cv2.CC_STAT_HEIGHT])
            if self.grouping[index] > 0:
                box = self._centreOnTop(box, x, y)
                if self.grouping[index] > 1 and distance is not None:
                    box = self._blendWithDistance(box, x, y,
                                                  2 * distance[y, x])
            boxes.append(tuple(int(round(v)) for v in box))

        self.boxes = boxes
        if self.verbose:
            counts = [self.grouping.count(c) for c in (0, 1, 2)]
            print("[crowns] %d boxes: %d isolated, %d crowded, %d dense"
                  % (len(boxes), counts[0], counts[1], counts[2]))
        return boxes

    def _fallbackBox(self, x, y, distance):
        """A top that fell on a Voronoi edge or outside the canopy."""
        half = float(distance[y, x]) if distance is not None else \
            self.scene.metresToPixels(1.0)
        half = max(half, 1.0)
        return (int(x - half), int(y - half), int(2 * half), int(2 * half))

    @staticmethod
    def _centreOnTop(box, x, y):
        """
        The largest box centred on the top that still fits inside `box`.

        Each side is limited by whichever edge the top is nearer to, so the box
        cannot reach further past the top on one side than on the other.
        """
        px, py, w, h = box
        width = 2 * min(abs(px - x), abs(px + w - x))
        height = 2 * min(abs(py - y), abs(py + h - y))
        return (x - width / 2.0, y - height / 2.0, width, height)

    def _blendWithDistance(self, box, x, y, distanceSide):
        """
        Interpolate between the distance box and the centred Voronoi box.

        The distance box is the largest square certainly inside the canopy, so
        it is a floor; the Voronoi box is the ceiling. The weight decides how
        far toward the ceiling to go, and only ever shrinks the Voronoi box.
        """
        px, py, w, h = box
        if distanceSide < w:
            w = distanceSide + self.voronoiWeight * (w - distanceSide)
            px = x - w / 2.0
        if distanceSide < h:
            h = distanceSide + self.voronoiWeight * (h - distanceSide)
            py = y - h / 2.0
        return (px, py, w, h)

    # ------------------------------------------------------------------ #

    def polygons(self):
        """
        The cut canopy component of each top, as a world-coordinate polygon.

        These are the pseudo-crowns: the actual shape the tree occupies in the
        thresholded canopy, not the box around it. A top whose component is
        empty gets None.
        """

        if self.labels is None:
            self.build()

        shapes = []
        for index, (x, y, _) in enumerate(self.tops.points):
            label = self.labels[y, x]
            if label == 0:
                shapes.append(None)
                continue
            component = (self.labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                shapes.append(None)
                continue
            outline = max(contours, key=cv2.contourArea).squeeze()
            if outline.ndim != 2 or len(outline) < 3:
                shapes.append(None)
                continue
            world = [self.scene.toWorld(c, r) for c, r in outline]
            polygon = Polygon(world)
            shapes.append(polygon if polygon.is_valid else polygon.buffer(0))
        return shapes

    def boxPolygons(self):
        """The bounding boxes as world-coordinate rectangles."""

        if self.boxes is None:
            self.build()
        shapes = []
        for px, py, w, h in self.boxes:
            west, north = self.scene.toWorld(px, py)
            east, south = self.scene.toWorld(px + w, py + h)
            shapes.append(shapelyBox(min(west, east), min(north, south),
                                     max(west, east), max(north, south)))
        return shapes

    # ------------------------------------------------------------------ #

    def attributes(self):
        """The per-top fields written beside every geometry."""
        return {"height": self.tops.heights, "grouping": self.grouping,
                "widthM": [w * self.scene.pixelSize for _, _, w, _ in self.boxes],
                "heightM": [h * self.scene.pixelSize
                            for _, _, _, h in self.boxes]}

    def write(self, outputDir, crowns=True, boxes=True, mask=True):
        os.makedirs(outputDir, exist_ok=True)
        if self.boxes is None:
            self.build()
        written = []
        if boxes:
            written += self._writeBoxes(outputDir)
        if crowns:
            written += self._writeCrowns(outputDir)
        if mask:
            written.append(self._writeBoxImage(outputDir))
        if self.verbose:
            for path in written:
                print("[crowns] wrote %s" % path)
        return written

    def _writeBoxes(self, outputDir):
        shapefile = os.path.join(outputDir, "pseudoBoxes.shp")
        gpd.GeoDataFrame(self.attributes(), geometry=self.boxPolygons(),
                         crs=self.scene.crs).to_file(shapefile)
        listing = os.path.join(outputDir, "pseudoBoxes.txt")
        with open(listing, "w") as handle:
            handle.write("# x y width height heightM grouping\n")
            for (px, py, w, h), height, group in zip(
                    self.boxes, self.tops.heights, self.grouping):
                handle.write("%d %d %d %d %.2f %d\n"
                             % (px, py, w, h, height, group))
        return [shapefile, listing]

    def _writeCrowns(self, outputDir):
        shapes = self.polygons()
        keep = [i for i, shape in enumerate(shapes) if shape is not None]
        if not keep:
            return []
        path = os.path.join(outputDir, "pseudoCrowns.shp")
        gpd.GeoDataFrame({key: [value[i] for i in keep]
                          for key, value in self.attributes().items()},
                         geometry=[shapes[i] for i in keep],
                         crs=self.scene.crs).to_file(path)
        return [path]

    def _writeBoxImage(self, outputDir):
        image = np.zeros(self.scene.chm.shape, np.uint8)
        for px, py, w, h in self.boxes:
            cv2.rectangle(image, (px, py), (px + w, py + h), 255, 1)
        path = os.path.join(outputDir, "pseudoBoxes.png")
        cv2.imwrite(path, image)
        return path

    # ------------------------------------------------------------------ #

    def compareWith(self, trueCrowns):
        """
        Intersection over union against real crowns, where they exist.

        Only meaningful on an annotated site, and worth running there before
        trusting pseudo-crowns anywhere else: it says how much of a real crown
        a construction from points actually recovers.
        """

        shapes = self.polygons()
        valid = [(i, s) for i, s in enumerate(shapes) if s is not None]
        if not valid:
            return None

        frame = gpd.GeoDataFrame({"index": [i for i, _ in valid]},
                                 geometry=[s for _, s in valid],
                                 crs=self.scene.crs)
        joined = gpd.sjoin(frame, trueCrowns[["geometry"]], how="inner",
                           predicate="intersects")

        # Best match per pseudo-crown, not every pair. A pseudo-crown touches
        # about two real crowns on average, and scoring all of those counts
        # each near-miss as a failure of a crown that was in fact matched
        # correctly to its neighbour.
        best = {}
        for pseudoIndex, trueIndex in zip(joined["index"],
                                          joined["index_right"]):
            pseudo = shapes[int(pseudoIndex)]
            true = trueCrowns.geometry.iloc[int(trueIndex)]
            union = pseudo.union(true).area
            if union <= 0:
                continue
            iou = pseudo.intersection(true).area / union
            key = int(pseudoIndex)
            if key not in best or iou > best[key]:
                best[key] = iou

        scores = np.array(list(best.values()))
        return {"matched": len(scores), "pseudoCrowns": len(valid),
                "meanIou": float(scores.mean()) if scores.size else 0.0,
                "medianIou": float(np.median(scores)) if scores.size else 0.0,
                "aboveHalf": float((scores > 0.5).mean()) if scores.size else 0.0,
                "aboveQuarter": float((scores > 0.25).mean())
                if scores.size else 0.0,
                "pseudoAreaMedian": float(np.median([s.area for _, s in valid])),
                "trueAreaMedian": float(trueCrowns.geometry.area.median())}
