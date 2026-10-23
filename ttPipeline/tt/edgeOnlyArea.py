"""
The scored area with only the edge part of the uncovered canopy cut out: the
interior gaps between crowns are scored again (a detection there is a false
positive), the canopy at the area's border stays don't-care, and the 'other'
and invisible crowns stay cut out as before.

    python -m tt.edgeOnlyArea Data/sergiOut/koiwainoujo220616 \\
        "$DROP"/Quebec2/20230606_cbblackburn5 ...

For each prepared site folder (area.shp, ignored.shp, uncovered.shp with its
edge flag, as qpPrepare and sergiPrepare write them) writes scoredAreaEdge.shp
next to scoredArea.shp, which is left as it is so the earlier runs stay
reproducible, and prints the areas.
"""

import os
import sys

import geopandas as gpd
import numpy as np

from tt.edgePatches import isEdge


def edgeOnlyRegion(siteDir):
    area = gpd.read_file(os.path.join(siteDir, "area.shp"))
    shape = area.geometry.union_all()
    cuts = []
    ignoredPath = os.path.join(siteDir, "ignored.shp")
    if os.path.exists(ignoredPath):
        ignored = gpd.read_file(ignoredPath)
        cuts += list(ignored.geometry[ignored["reason"] != "uncovered"])
    uncoveredPath = os.path.join(siteDir, "uncovered.shp")
    edgeCount = interiorCount = 0
    if os.path.exists(uncoveredPath):
        uncovered = gpd.read_file(uncoveredPath)
        flag = isEdge(uncovered["edge"])
        cuts += list(uncovered.geometry[flag])
        edgeCount, interiorCount = int(flag.sum()), int((~flag).sum())
    if cuts:
        shape = shape.difference(gpd.GeoSeries(cuts, crs=area.crs).union_all())
    region = gpd.GeoDataFrame({"name": ["scored"]}, geometry=[shape],
                              crs=area.crs)
    return region, {"areaM2": float(area.area.sum()),
                    "edgePatches": edgeCount, "interiorPatches": interiorCount,
                    "scoredEdgeOnlyM2": float(shape.area)}


def main(argv=None):
    sites = list(sys.argv[1:] if argv is None else argv)
    if not sites:
        print(__doc__)
        return 1
    for siteDir in sites:
        region, notes = edgeOnlyRegion(siteDir)
        region.to_file(os.path.join(siteDir, "scoredAreaEdge.shp"))
        oldPath = os.path.join(siteDir, "scoredArea.shp")
        old = gpd.read_file(oldPath).area.sum() if os.path.exists(oldPath) \
            else np.nan
        print("[edgeOnly] %s: area %.0f m2, scored before %.0f m2, now %.0f m2 "
              "(%d edge patches cut, %d interior gaps scored again)"
              % (os.path.basename(os.path.normpath(siteDir)), notes["areaM2"],
                 old, notes["scoredEdgeOnlyM2"], notes["edgePatches"],
                 notes["interiorPatches"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
