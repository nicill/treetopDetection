#!/usr/bin/env python3
"""Stand-in for `pdal pipeline` in tests: height = z - 100, max per cell."""
import json
import re
import sys

import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin

stages = json.load(open(sys.argv[2]))
read, write = stages[0], stages[-1]
x0, x1, y0, y1 = map(float, re.findall(r"[-\d.]+", write["bounds"]))
res = write["resolution"]
cloud = laspy.read(read["filename"])
x, y, h = np.asarray(cloud.x), np.asarray(cloud.y), np.asarray(cloud.z) - 100
width, height = int(round((x1 - x0) / res)) + 1, int(round((y1 - y0) / res)) + 1
grid = np.full((height, width), -9999.0, np.float32)
c = np.floor((x - x0) / res).astype(int)
r = np.floor((y0 + height * res - y) / res).astype(int)
ok = (c >= 0) & (c < width) & (r >= 0) & (r < height)
np.maximum.at(grid, (r[ok], c[ok]), h[ok])
with rasterio.open(write["filename"], "w", driver="GTiff", height=height,
                   width=width, count=1, dtype="float32", crs="EPSG:32619",
                   transform=from_origin(x0, y0 + height * res, res, res),
                   nodata=-9999) as d:
    d.write(grid, 1)
