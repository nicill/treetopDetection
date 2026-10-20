"""
Quebec Plantations: a photogrammetric CHM for sites already prepared by
tt.qpPrepare, so every method can run on photogrammetry instead of LiDAR.

    python -m tt.qpPhotogrammetry --prepared "$QP" --output "$QP_P1" \\
        --pdal ~/anaconda3/envs/pdal/bin/pdal

Per site, the dataset's photogrammetric point cloud
(Photogrammetry_Products/<site>_p1/<site>_p1_pg_highdis.copc.laz at the afca
sites, <site>_p1_pg.copc.laz at the cb sites) goes through
exactly the pipeline qpPrepare runs on the LiDAR: read from the FRDR server
for the annotated area only, in tiles, statistical outlier removal, SMRF
ground, height above the Delaunay ground, highest point per cell, empty cells
from their nearest neighbour. Ground and heights come from the photographs
alone, processed as the LiDAR was, so the two CHMs differ by their data and
not by how they were made. (Ground from the surface model alone,
tt.groundModel, put the floor between the trees 0.5-2 m above 0 here.)

Finished tiles are kept per site (--workDir) until the site is done, so a
stopped run loses only the tiles in progress; the same command reuses them.

The new CHM is on the LiDAR CHM's grid (checked), 0 outside area.shp. Only
the height model changes: crowns, the scored area and the don't-care areas
are copied from the LiDAR site unchanged, so both CHMs are scored on the same
trees. rgb.tif is linked (relative link, so it works in a synced folder on
any computer).

Per site, in <output>/<site>/: chm.tif, the copied layers, rgb.tif (link),
quicklook.png (LiDAR and photogrammetric CHM with the crowns) and check.json:
the CHM against the field heights (as qpPrepare reports for LiDAR, both given),
crown tops against the LiDAR CHM's, the scored crowns whose photogrammetric
top is below the site's visibleM, and the floor: the photogrammetric CHM
where the LiDAR CHM is 0. check.json is written last: a site with one is
prepared and is skipped.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask

from tt import qpPrepare as qp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)

SELECTED = ("20230712_afcahoule_itrf20", "20230712_afcamoisan_itrf20",
            "20230606_cbblackburn5", "20230606_cbblackburn6",
            "20230608_cbbernard4")
COPIED = ("area", "scoredArea", "crowns", "ignored", "uncovered", "trees")
SHAPE_PARTS = (".shp", ".shx", ".dbf", ".prj", ".cpg")
LIDAR_ZERO_M = 0.05          # LiDAR CHM below this: the floor between trees


CLOUD_NAMES = ("%s_p1_pg_highdis.copc.laz", "%s_p1_pg.copc.laz")


def cloudUrl(site, reachable=None):
    """
    The site's photogrammetric point cloud on the server. Its name differs
    between sites (afca: _pg_highdis, cb: _pg), so each candidate is tried
    and the first the server has is used.
    """
    reachable = reachable or urlExists
    folder = "%sPhotogrammetry_Products/%s_p1/" % (qp.BASE, site)
    candidates = [folder + name % site for name in CLOUD_NAMES]
    for url in candidates:
        if reachable(url):
            return url
    raise OSError("no photogrammetric point cloud for %s; tried %s"
                  % (site, ", ".join(c.rsplit("/", 1)[1] for c in candidates)))


def urlExists(url, timeoutS=60):
    """Whether the server has url: its first byte is fetched."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(request, timeout=timeoutS) as response:
            return response.status in (200, 206)
    except urllib.error.HTTPError:
        return False


def chmResolution(path):
    with rasterio.open(path) as src:
        return src.res[0]


def checkSameGrid(lidarPath, photoPath):
    """The photogrammetric CHM must sit on the LiDAR CHM's grid exactly."""
    with rasterio.open(lidarPath) as a, rasterio.open(photoPath) as b:
        same = (a.shape == b.shape and
                a.transform.almost_equals(b.transform, precision=1e-6))
        grids = (a.shape, a.transform, b.shape, b.transform)
    if not same:
        raise ValueError("photogrammetric CHM not on the LiDAR grid: "
                         "LiDAR %s %s, photogrammetry %s %s" % grids)


def copySite(site, output):
    """The LiDAR site's layers copied, rgb.tif linked."""
    os.makedirs(output, exist_ok=True)
    for stem in COPIED:
        for part in SHAPE_PARTS:
            source = os.path.join(site, stem + part)
            if os.path.exists(source):
                shutil.copyfile(source, os.path.join(output, stem + part))
    source, target = os.path.join(site, "rgb.tif"), os.path.join(output,
                                                                 "rgb.tif")
    if os.path.lexists(target):
        return
    try:
        os.symlink(os.path.relpath(source, output), target)
    except OSError:                       # no links here: a copy instead
        shutil.copyfile(source, target)


def crownComparison(siteDir, chmPath, visibleM):
    """Crown tops, photogrammetric against LiDAR, on the scored crowns."""
    crowns = gpd.read_file(os.path.join(siteDir, "crowns.shp"))
    photo = qp.crownTops(chmPath, crowns)
    lidar = qp.crownTops(os.path.join(siteDir, "chm.tif"), crowns)
    difference = photo - lidar
    return {"crowns": int(len(crowns)),
            "topMinusLidarMedianM": float(np.median(difference)),
            "topMinusLidarMeanAbsM": float(np.mean(np.abs(difference))),
            "topCorrelation": float(np.corrcoef(photo, lidar)[0, 1])
            if len(crowns) > 1 else float("nan"),
            "belowVisibleM": int((photo < visibleM).sum()),
            "visibleM": visibleM}


def floorComparison(siteDir, chmPath):
    """The photogrammetric CHM where the LiDAR CHM is 0 inside the area."""
    area = gpd.read_file(os.path.join(siteDir, "area.shp"))
    with rasterio.open(os.path.join(siteDir, "chm.tif")) as a, \
            rasterio.open(chmPath) as b:
        lidar, photo, transform = a.read(1), b.read(1), a.transform
    inside = ~geometry_mask(area.geometry, lidar.shape, transform)
    floor = photo[inside & (lidar < LIDAR_ZERO_M)]
    if floor.size == 0:
        return {"cells": 0}
    return {"cells": int(floor.size),
            "medianM": float(np.median(floor)),
            "p90M": float(np.percentile(floor, 90)),
            "shareAbove0p5M": float(np.mean(floor > 0.5))}


def fieldComparison(siteDir, chmPath):
    """The CHM against the field heights, as qpPrepare checks the LiDAR."""
    path = os.path.join(siteDir, "trees.shp")
    if not os.path.exists(path):
        return {}, {}
    trees = gpd.read_file(path)
    trees["chmHm"] = qp.chmAtTrees(chmPath, trees)
    return qp.heightCheck(trees, "fieldHm"), qp.heightCheck(trees, "noShootHm")


def readScaled(path, maxPixels):
    with rasterio.open(path) as src:
        scale = max(1.0, max(src.width, src.height) / float(maxPixels))
        data = src.read(1, out_shape=(int(src.height / scale),
                                      int(src.width / scale)),
                        resampling=Resampling.average)
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom,
                  src.bounds.top)
    return data, extent


def quicklook(siteDir, chmPath, outputPath, maxPixels=1500):
    crowns = gpd.read_file(os.path.join(siteDir, "crowns.shp"))
    pairs = (("LiDAR", os.path.join(siteDir, "chm.tif")),
             ("photogrammetry", chmPath))
    images = [readScaled(p, maxPixels) for _, p in pairs]
    top = max(float(np.percentile(d[d > 0], 99)) if (d > 0).any() else 1.0
              for d, _ in images)
    figure, axes = plt.subplots(1, 2, figsize=(16, 8))
    for axis, (name, _), (data, extent) in zip(axes, pairs, images):
        shown = axis.imshow(data, extent=extent, vmin=0, vmax=top)
        crowns.boundary.plot(ax=axis, color="white", linewidth=0.3)
        axis.set_title(name)
        axis.set_axis_off()
    figure.colorbar(shown, ax=axes, shrink=0.7, label="height (m)")
    figure.savefig(outputPath, dpi=150, bbox_inches="tight")
    plt.close(figure)


def lidarCheck(siteDir):
    path = os.path.join(siteDir, "check.json")
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)


def prepareSite(site, args):
    siteDir, outDir = (os.path.join(d, site) for d in (args.prepared,
                                                        args.output))
    copySite(siteDir, outDir)
    area = gpd.read_file(os.path.join(siteDir, "area.shp"))
    lidarPath, chmPath = (os.path.join(d, "chm.tif") for d in (siteDir,
                                                               outDir))
    args.chmResolution = chmResolution(lidarPath)
    source = args.cloudSource(site)
    print("[qpP1] %s: photogrammetric point cloud, %.0f m2 at %.2f m"
          % (site, area.area.iloc[0], args.chmResolution), flush=True)
    tileDir = tileFolder(site, args)
    kept = len([f for f in os.listdir(tileDir) if f.startswith("tile")
                and not f.endswith(".partial.tif")])
    if kept:
        print("[qpP1] %s: %d finished tiles from an earlier run reused"
              % (site, kept), flush=True)
    filled = qp.buildChm(source, area, args, tileDir, chmPath)
    checkSameGrid(lidarPath, chmPath)
    check = finishSite(site, siteDir, outDir, chmPath, source, filled)
    shutil.rmtree(tileDir, ignore_errors=True)
    return check


def tileFolder(site, args):
    """
    Where a site's finished tiles are kept until the site is done, so a
    stopped run loses only the tiles in progress. Named by tile size and
    resolution, so tiles made with other settings are never reused.
    """
    root = args.workDir or os.path.join(tempfile.gettempdir(), "qpP1tiles")
    folder = os.path.join(root, "%s_tile%gm_res%gm" % (site, args.tileM,
                                                       args.chmResolution))
    os.makedirs(folder, exist_ok=True)
    return folder


def finishSite(site, siteDir, outDir, chmPath, source, filled):
    lidar = lidarCheck(siteDir)
    heights, noShoot = fieldComparison(siteDir, chmPath)
    quicklook(siteDir, chmPath, os.path.join(outDir, "quicklook.png"))
    check = {"site": site, "heightSource": "photogrammetry",
             "ground": "SMRF on the photogrammetric point cloud, as for LiDAR",
             "pointCloud": source, "chmCellsFilled": filled,
             "crownTops": crownComparison(siteDir, chmPath,
                                          lidar.get("visibleM", 1.0)),
             "floor": floorComparison(siteDir, chmPath),
             "heights": heights, "heightsNoShoot": noShoot,
             "lidarHeights": lidar.get("heights", {}),
             "lidarHeightsNoShoot": lidar.get("heightsNoShoot", {}),
             "lidarChmCellsFilled": lidar.get("chmCellsFilled")}
    with open(os.path.join(outDir, "check.json"), "w") as handle:
        json.dump(check, handle, indent=2)
    reportSite(check)
    return check


def reportSite(check):
    site, tops, floor = check["site"], check["crownTops"], check["floor"]
    print("[qpP1] %s: cells filled %.1f%%; crown tops minus LiDAR's median "
          "%+.2f m, mean |diff| %.2f m, r %.2f; %d of %d scored crowns below "
          "%.1f m" % (site, 100 * check["chmCellsFilled"],
                      tops["topMinusLidarMedianM"],
                      tops["topMinusLidarMeanAbsM"], tops["topCorrelation"],
                      tops["belowVisibleM"], tops["crowns"],
                      tops["visibleM"]), flush=True)
    print("[qpP1] %s: floor (where LiDAR is 0): median %.2f m, 90%% under "
          "%.2f m, %.1f%% above 0.5 m"
          % (site, floor.get("medianM", np.nan), floor.get("p90M", np.nan),
             100 * floor.get("shareAbove0p5M", np.nan)), flush=True)
    for label, mine, theirs in (("total", "heights", "lidarHeights"),
                                ("no shoot", "heightsNoShoot",
                                 "lidarHeightsNoShoot")):
        a, b = check[mine], check[theirs]
        print("[qpP1] %s: CHM - field %-8s photogrammetry %+.2f m (|diff| "
              "%.2f), LiDAR %+.2f m (|diff| %.2f), %d trees"
              % (site, label, a.get("medianChmMinusFieldM", np.nan),
                 a.get("meanAbsDiffM", np.nan),
                 b.get("medianChmMinusFieldM", np.nan),
                 b.get("meanAbsDiffM", np.nan), a.get("trees", 0)),
              flush=True)


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Quebec Plantations: photogrammetric CHM, built from the "
                    "photogrammetric point cloud as qpPrepare builds the "
                    "LiDAR one, for sites prepared by tt.qpPrepare.")
    parser.add_argument("--prepared", required=True,
                        help="The folder of LiDAR sites written by qpPrepare")
    parser.add_argument("--output", required=True,
                        help="Where the photogrammetric sites go (not the "
                             "--prepared folder)")
    parser.add_argument("--pdal", required=True, help="Path to the pdal program")
    parser.add_argument("--sites", nargs="*", default=list(SELECTED),
                        help="Default: the five selected sites")
    parser.add_argument("--tileM", type=float, default=50.0,
                        help="Tile size; smaller if PDAL runs out of memory "
                             "or time on the dense photogrammetric cloud")
    parser.add_argument("--marginM", type=float, default=5.0,
                        help="Points read this far beyond each tile")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retryWaitS", type=float, default=60.0)
    parser.add_argument("--tileTimeoutMin", type=float, default=30.0)
    parser.add_argument("--workDir", default=None,
                        help="Where each site's finished PDAL tiles are kept "
                             "until the site is done, so a restart reuses "
                             "them (default: qpP1tiles in the system's "
                             "temporary folder; not a synced folder)")
    args = parser.parse_args(argv)
    if os.path.abspath(args.output) == os.path.abspath(args.prepared):
        parser.error("--output must differ from --prepared")
    args.cloudSource = cloudUrl
    return args


def main(argv=None):
    args = parseArguments(argv)
    failed = []
    for site in args.sites:
        if os.path.exists(os.path.join(args.output, site, "check.json")):
            print("[qpP1] %s: already prepared, skipped" % site)
            continue
        if not os.path.exists(os.path.join(args.prepared, site, "chm.tif")):
            print("[qpP1] %s: no LiDAR site in %s, skipped" % (site,
                                                               args.prepared))
            failed.append(site)
            continue
        try:
            prepareSite(site, args)
        except (RuntimeError, OSError, ValueError) as error:
            print("[qpP1] %s: FAILED, skipped: %s" % (site, error), flush=True)
            failed.append(site)
    if failed:
        print("[qpP1] %d site(s) not done; run the same command again to "
              "retry them: %s" % (len(failed), " ".join(failed)), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
