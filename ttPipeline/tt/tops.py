"""
Tops — a set of detected or reference treetops, and the three ways they leave
the program.

A bare list of tuples was passed between every stage before, with each one
re-deciding whether the tuple was (x, y), (row, col) or (x, y, height), and
each writing its own mask and text file. The convention is fixed here: x is the
column, y is the row, both in working-resolution pixels, and height is metres.
"""

import os

import cv2
import geopandas as gpd
import numpy as np
from shapely.geometry import Point


class Tops(object):

    def __init__(self, points=None, crownIds=None, classes=None):
        self.points = list(points or [])
        self.crownIds = list(crownIds) if crownIds is not None else None
        self.classes = list(classes) if classes is not None else None

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, index):
        return self.points[index]

    def __repr__(self):
        return "Tops(%d)" % len(self.points)

    @property
    def heights(self):
        return np.array([p[2] for p in self.points], dtype=np.float64)

    @property
    def pixels(self):
        return np.array([[p[0], p[1]] for p in self.points], dtype=np.float64)

    def world(self, scene):
        return np.array([scene.toWorld(p[0], p[1]) for p in self.points])

    def geoPoints(self, scene):
        return [Point(*scene.toWorld(p[0], p[1])) for p in self.points]

    # ------------------------------------------------------------------ #

    def writeMask(self, path, shape, radius=5):
        """White background, black filled circle per top."""
        mask = 255 * np.ones(shape, np.uint8)
        for point in self.points:
            cv2.circle(mask, (int(point[0]), int(point[1])), int(radius), 0, -1)
        _ensureDir(path)
        cv2.imwrite(path, mask)
        return path

    def writeList(self, path, scene=None):
        """x y height, plus easting northing and any carried attributes."""
        columns = ["x", "y", "height"]
        if scene is not None:
            columns += ["easting", "northing"]
        if self.crownIds is not None:
            columns.append("crownId")
        if self.classes is not None:
            columns.append("treeClass")

        _ensureDir(path)
        with open(path, "w") as handle:
            handle.write("# " + " ".join(columns) + "\n")
            for index, point in enumerate(self.points):
                fields = ["%d" % point[0], "%d" % point[1], "%.3f" % point[2]]
                if scene is not None:
                    east, north = scene.toWorld(point[0], point[1])
                    fields += ["%.3f" % east, "%.3f" % north]
                if self.crownIds is not None:
                    fields.append("%d" % self.crownIds[index])
                if self.classes is not None:
                    fields.append("%s" % self.classes[index])
                handle.write(" ".join(fields) + "\n")
        return path

    def writeShapefile(self, path, scene, extra=None):
        frame = {"height": self.heights}
        if self.crownIds is not None:
            frame["crownId"] = self.crownIds
        if extra:
            frame.update(extra)
        gpd.GeoDataFrame(frame, geometry=self.geoPoints(scene),
                         crs=scene.crs).to_file(path)
        return path

    @classmethod
    def read(cls, path):
        points = []
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                points.append((int(float(parts[0])), int(float(parts[1])),
                               float(parts[2])))
        return cls(points)


def _ensureDir(path):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
