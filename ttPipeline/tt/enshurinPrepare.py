"""
Enshurin (Vlad's broadleaf site) as the same per-site products qpPrepare
writes, so the rest of the pipeline treats it as one more site.

    python -m tt.groundModel --dsm enshurin_dsm025.tif --output enshurin_chm025.tif
    python -m tt.enshurinPrepare --data "Data Vlad's Publication" \\
        --chm enshurin_chm025.tif --output Data/enshurinOut

What differs from the other sites, and what this does about it:

  crowns   the annotation is a class mask per part (newtrain, val, test;
           0 background, 1-4 one species each), not polygons. Every
           4-connected region of a class is one crown: at the mask's own
           1.9 cm the crowns never touch (no region over 300 m2), so no
           crown is split or merged here. Regions under --minCrownM2 are
           dropped as fragments of a crown cut by a gap in the painting.
           The species is kept in crownSpecies.shp; for scoring every crown
           is one class, as at Koiwai.
  area     only four species were annotated, about half the canopy. The
           area scored is made of --cellM cells of the mosaic in which at
           least --annotatedShare of the canopy (CHM >= --canopyCellM) is
           annotated, the annotation grown by --coverMarginM first (outlines
           are drawn tighter than the crown's rim in the CHM): there a
           detection in no crown is a real mistake.
           Cells with no canopy are kept (nothing there to miss). Both
           thresholds are fixed before any detection is scored.
           Within that area, the usual don't-care cut-outs apply
           (qpPrepare.scoredRegion): uncovered canopy, invisible crowns.
  CHM      from the full-site surface model (tt.groundModel on the whole
           DEM, so the ground estimate does not degrade at part edges),
           cropped to the area's grid, 0 outside the area.
  RGB      the three part mosaics of the same date, merged at
           --rgbResolution.

Writes, in <output>/<site>/: area.shp, scoredArea.shp, crowns.shp,
crownSpecies.shp, ignored.shp, uncovered.shp, chm.tif, rgb.tif,
check.json and quicklook.png.
"""

import argparse
import glob
import json
import os
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, shapes
from rasterio.merge import merge
from rasterio.warp import reproject
from rasterio.windows import from_bounds
from scipy.ndimage import binary_dilation
from shapely.geometry import box, shape
from shapely.ops import unary_union

from tt import qpPrepare as qp

PARTS = ("newtrain", "val", "test")
TOP_PERCENTILES = (5, 25, 50, 75, 95)


def partFiles(data, part, date):
    """(mosaic, mask, per-class masks) of one part; the mask names vary."""
    mosaic = os.path.join(data, "%s_%s.tif" % (date, part))
    masks = sorted(glob.glob(os.path.join(data, "mask*%s.tif" % part)))
    if not os.path.exists(mosaic) or len(masks) != 1:
        raise FileNotFoundError("part %s: need %s and one mask*%s.tif in %s"
                                % (part, mosaic, part, data))
    perClass = sorted(glob.glob(os.path.join(data, "raw", "*", "per_class",
                                             "mask_*_%s.tif" % part)))
    return mosaic, masks[0], perClass


def speciesNames(mask, perClass):
    """{class value: species} from the per-class masks, when there are any."""
    names = {}
    with rasterio.open(mask) as m:
        shape_ = (max(1, m.height // 20), max(1, m.width // 20))
        values = m.read(1, out_shape=shape_, resampling=Resampling.nearest)
    for path in perClass:
        species = os.path.basename(path).split("_")[1]
        with rasterio.open(path) as p:
            mine = p.read(1, out_shape=shape_,
                          resampling=Resampling.nearest) > 0
        found = values[mine & (values > 0)]
        if found.size:
            names[int(np.bincount(found).argmax())] = species
    return names


def maskCrowns(mask, names, minCrownM2, part):
    """One polygon per 4-connected region of each class."""
    with rasterio.open(mask) as m:
        values, transform, crs = m.read(1), m.transform, m.crs
    rows, dropped = [], 0
    for geometry, value in shapes(values, mask=values > 0, connectivity=4,
                                  transform=transform):
        polygon = shape(geometry)
        if polygon.area < minCrownM2:
            dropped += 1
            continue
        rows.append({"class_code": "tree", "classValue": int(value),
                     "species": names.get(int(value), str(int(value))),
                     "part": part, "geometry": polygon})
    return gpd.GeoDataFrame(rows, crs=crs), dropped


def footprint(mosaic, resolution=0.25):
    """Where the mosaic has image: its non-black pixels, as polygons."""
    with rasterio.open(mosaic) as src:
        scale = resolution / abs(src.transform.a)
        out = (max(1, int(src.height / scale)), max(1, int(src.width / scale)))
        image = src.read(1, out_shape=out, resampling=Resampling.nearest)
        transform = src.transform * src.transform.scale(src.width / out[1],
                                                        src.height / out[0])
    valid = (image > 0).astype(np.uint8)
    return unary_union([shape(g) for g, v in shapes(valid, mask=valid > 0,
                                                    transform=transform)])


def annotatedOnChm(mask, chmTransform, chmShape, bounds):
    """Share of each CHM cell covered by annotation (0-1), on the CHM grid."""
    share = np.zeros(chmShape, np.float32)
    with rasterio.open(mask) as m:
        binary = (m.read(1) > 0).astype(np.float32)
        reproject(binary, share, src_transform=m.transform, src_crs=m.crs,
                  dst_transform=chmTransform, dst_crs=m.crs,
                  resampling=Resampling.average, src_nodata=None,
                  dst_nodata=None, init_dest_nodata=False)
    return share


def grown(annotated, marginM, pixel):
    """
    The annotation grown by marginM: outlines are drawn tighter than the
    crown's rim in the CHM, and that rim must not count as unannotated canopy.
    """
    steps = int(round(marginM / pixel))
    if steps <= 0:
        return annotated
    disk = np.hypot(*np.mgrid[-steps:steps + 1, -steps:steps + 1]) <= steps
    return binary_dilation(annotated, structure=disk)


def readChm(chmPath, bounds):
    with rasterio.open(chmPath) as src:
        window = from_bounds(*bounds, src.transform).round_offsets() \
            .round_lengths()
        chm = src.read(1, window=window, boundless=True, fill_value=0)
        transform = src.window_transform(window)
        nodata = src.nodata
    chm = chm.astype(np.float32)
    if nodata is not None:
        chm[chm == nodata] = 0
    chm[~np.isfinite(chm) | (chm < 0)] = 0
    return chm, transform


def coveredCells(chm, transform, annotated, inMosaic, args):
    """The cellM cells (inside the mosaic) whose canopy is annotated enough."""
    cell = int(round(args.cellM / abs(transform.a)))
    keep = []
    rows, cols = chm.shape
    for r in range(0, rows, cell):
        for c in range(0, cols, cell):
            inside = inMosaic[r:r + cell, c:c + cell]
            if inside.mean() < 0.5:
                continue
            canopy = (chm[r:r + cell, c:c + cell] >= args.canopyCellM) & inside
            covered = annotated[r:r + cell, c:c + cell]
            if canopy.sum() == 0 or (canopy & covered).sum() \
                    >= args.annotatedShare * canopy.sum():
                x0, y0 = transform * (c, r)
                x1, y1 = transform * (min(c + cell, cols), min(r + cell, rows))
                keep.append(box(min(x0, x1), min(y0, y1), max(x0, x1),
                                max(y0, y1)))
    return keep


def rasterMask(geometry, chmShape, transform):
    return ~geometry_mask([geometry], chmShape, transform)


def buildArea(parts, chmPath, args):
    """Union over parts of the well-annotated cells inside each mosaic."""
    cells, crowns, notes = [], [], {}
    # the class values mean the same species in every part; the per-class
    # masks that name them may exist for only some parts
    names = {}
    for mosaic, mask, perClass in parts.values():
        names.update(speciesNames(mask, perClass))
    for part, (mosaic, mask, perClass) in parts.items():
        partCrowns, dropped = maskCrowns(mask, names, args.minCrownM2, part)
        foot = footprint(mosaic)
        chm, transform = readChm(chmPath, foot.bounds)
        annotated = grown(annotatedOnChm(mask, transform, chm.shape,
                                         foot.bounds) >= 0.5,
                          args.coverMarginM, abs(transform.a))
        inMosaic = rasterMask(foot, chm.shape, transform)
        partCells = coveredCells(chm, transform, annotated, inMosaic, args)
        cells += [c.intersection(foot) for c in partCells]
        crowns.append(partCrowns)
        notes[part] = {"crowns": int(len(partCrowns)),
                       "fragmentsDropped": dropped, "species": names,
                       "mosaicM2": float(foot.area),
                       "keptM2": float(sum(c.intersection(foot).area
                                           for c in partCells))}
    crs = crowns[0].crs
    area = unary_union(cells)
    return (gpd.GeoDataFrame({"name": ["area"]}, geometry=[area], crs=crs),
            gpd.GeoDataFrame(np.concatenate([c.values for c in crowns]),
                             columns=crowns[0].columns, crs=crs), notes)


def writeChm(chmPath, area, args, outPath):
    """The full-site CHM on the area's grid, 0 outside the area."""
    west, south, east, north = qp.gridBounds(area, args.chmResolution)
    chm, transform = readChm(chmPath, (west, south, east, north))
    chm[~rasterMask(area.geometry.iloc[0], chm.shape, transform)] = 0
    with rasterio.open(chmPath) as src:
        crs = src.crs
    qp.writeRaster(outPath, chm[None], transform, crs, "float32")


def writeRgb(mosaics, area, resolution, outPath):
    """The part mosaics merged over the area's bounds at resolution."""
    sources = [rasterio.open(m) for m in mosaics]
    try:
        bounds = qp.gridBounds(area, resolution)
        data, transform = merge(sources, bounds=bounds, res=resolution,
                                indexes=[1, 2, 3],
                                resampling=Resampling.average)
        crs = sources[0].crs
    finally:
        for s in sources:
            s.close()
    qp.writeRaster(outPath, data.astype(np.uint8), transform, crs, "uint8")


def finish(site, outDir, crowns, area, chmPath, rgbPath, notes, args):
    inArea = crowns[crowns.intersects(area.geometry.iloc[0])]
    coverage, uncovered = qp.uncoveredCanopy(chmPath, inArea, area, args)
    tops = qp.crownTops(chmPath, inArea)
    scored, ignored, region, counts = qp.scoredRegion(inArea, area, uncovered,
                                                      tops, args.visibleM)
    qp.writeLayers(outDir, {"uncovered": uncovered,
                            "crowns": scored[["class_code", "geometry"]],
                            "crownSpecies": scored[["species", "part",
                                                    "geometry"]],
                            "ignored": ignored, "scoredArea": region})
    qp.quicklook(rgbPath, scored, area, ignored,
                 os.path.join(outDir, "quicklook.png"))
    check = {"site": site, "parts": notes,
             "annotatedCrowns": int(len(crowns)),
             "crownsInArea": int(len(inArea)), "crowns": int(len(scored)),
             "invisibleCrowns": counts["invisible"],
             "areaM2": float(area.area.iloc[0]),
             "scoredAreaM2": float(region.area.iloc[0]),
             "medianCrownM2": float(scored.area.median()) if len(scored) else 0,
             "coverage": coverage,
             "crownTopM": {str(p): float(v) for p, v in zip(
                 TOP_PERCENTILES, np.percentile(tops, TOP_PERCENTILES))}
             if len(tops) else {},
             "settings": {k: getattr(args, k) for k in (
                 "cellM", "annotatedShare", "coverMarginM", "canopyCellM",
                 "minCrownM2",
                 "canopyM", "crownMarginM", "minPatchM2", "visibleM",
                 "chmResolution", "rgbResolution")}}
    with open(os.path.join(outDir, "check.json"), "w") as handle:
        json.dump(check, handle, indent=2)
    return check


def report(check):
    print("[enshurin] %s: %d annotated crowns, %d in the area, %d scored "
          "(%d invisible); area %.0f m2, scored %.0f m2"
          % (check["site"], check["annotatedCrowns"], check["crownsInArea"],
             check["crowns"], check["invisibleCrowns"], check["areaM2"],
             check["scoredAreaM2"]))
    for part, n in check["parts"].items():
        print("[enshurin]   %-9s %4d crowns (%d fragments dropped), species %s, "
              "kept %.0f of %.0f m2 of mosaic"
              % (part, n["crowns"], n["fragmentsDropped"], n["species"],
                 n["keptM2"], n["mosaicM2"]))
    if check["crownTopM"]:
        print("[enshurin] crown tops above ground, percentiles %s: %s m"
              % ("/".join(check["crownTopM"]),
                 "/".join("%.1f" % v for v in check["crownTopM"].values())))


def prepare(args):
    outDir = os.path.join(args.output, args.site)
    os.makedirs(outDir, exist_ok=True)
    parts = {p: partFiles(args.data, p, args.date) for p in args.parts}
    area, crowns, notes = buildArea(parts, args.chm, args)
    area.to_file(os.path.join(outDir, "area.shp"))
    chmPath, rgbPath = (os.path.join(outDir, n) for n in ("chm.tif", "rgb.tif"))
    writeChm(args.chm, area, args, chmPath)
    writeRgb([m for m, _, _ in parts.values()], area, args.rgbResolution,
             rgbPath)
    check = finish(args.site, outDir, crowns, area, chmPath, rgbPath, notes,
                   args)
    report(check)
    return check


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True,
                        help="folder with <date>_<part>.tif and mask*<part>.tif")
    parser.add_argument("--chm", required=True,
                        help="full-site CHM (tt.groundModel on the whole DEM)")
    parser.add_argument("--output", default="Data/enshurinOut")
    parser.add_argument("--site", default="enshurin")
    parser.add_argument("--date", default="08_22")
    parser.add_argument("--parts", nargs="+", default=list(PARTS))
    parser.add_argument("--cellM", type=float, default=10.0)
    parser.add_argument("--annotatedShare", type=float, default=0.9)
    parser.add_argument("--coverMarginM", type=float, default=0.5,
                        help="annotation grown by this when measuring coverage")
    parser.add_argument("--canopyCellM", type=float, default=2.0,
                        help="canopy height for the annotation-coverage cells")
    parser.add_argument("--minCrownM2", type=float, default=1.0)
    parser.add_argument("--chmResolution", type=float, default=0.25)
    parser.add_argument("--rgbResolution", type=float, default=0.05)
    parser.add_argument("--canopyM", type=float, default=1.5)
    parser.add_argument("--crownMarginM", type=float, default=0.2)
    parser.add_argument("--minPatchM2", type=float, default=0.5)
    parser.add_argument("--visibleM", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv=None):
    prepare(parseArguments(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
