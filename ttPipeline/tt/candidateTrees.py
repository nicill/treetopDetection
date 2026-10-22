"""
Candidate crowns for the trees nobody annotated, for the annotator to check
and correct instead of drawing from nothing.

    python -m tt.candidateTrees \\
        --chm Data/vlad/enshurin_chm025.tif \\
        --unannotated Data/enshurinNewOut/unannotated/unannotated.shp \\
        --polygons Data/enshurinNewOut/enshurin_all_shifted.shp \\
        --mosaics "Data/enshurinNewOut/enshurin/shifted/*.vrt" \\
        --shift newtrain=-1.37,-0.43 --shift test=-1.67,-0.47 \\
        --shift val=-1.68,-0.47 \\
        --like Data/vlad/enshurinAnnotations/enshurin_all.shp \\
        --output Data/enshurinNewOut/candidates

Inside the unannotated tree canopy (tt.unannotatedTrees), a crown is proposed
for every top that rises --minProminenceM above where it joins higher ground
on the CHM smoothed over --smoothM, grown over its own patch by a marker
watershed (the CHM turned upside down, flooded from the tops), so the crowns
are contiguous and never overlap. A crown is grown by --growM (the mask stops
--marginM short of the annotated polygons) and never into an annotated crown
or another candidate; tiny and huge ones are dropped or flagged.

Each candidate has: areaM2, topH (highest CHM cell), meanH, prominence,
solidity (area over convex hull: 1 is a clean blob), and conf, how much it
looks like the annotated trees, measured on them (the annotated crowns' own
area and top-height quartiles):
  A  area within the annotated 10-90% range, top above the annotated lower
     quartile, solidity >= 0.85
  B  top above the annotated 10th percentile, area within half the lower and
     twice the upper end of that range
  C  everything else: likely bushes, branches or fragments, or a pair of crowns

Writes, in --output:
  candidates.shp                 in the CHM's frame (the shifted mosaics')
  candidates_original_frame.shp  moved back to the annotation's own frame (the
                                 inverse of the shift of the part whose mosaic
                                 holds it, then the coordinate system of
                                 --like): for the annotator's own project;
                                 fields name = 'CAND', class = 0, conf, ...
  candidates_tops.shp            the tops
  candidates_overview.png        the whole mosaic with the annotated crowns in
                                 yellow and the candidates in cyan (A), blue
                                 (B) and magenta (C)
  candidates.json                the counts

The candidates are a starting point, not an annotation: they are from the CHM
(a height method), so the trees the annotator confirms and adds must be his
own judgement of the image, and the C ones in particular are likely not trees.
"""

import argparse
import glob
import json
import os
import sys

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

from tt.enshurinPrepare import parseShifts
from tt.unannotatedTrees import footprint, mosaicImage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)


def reference(chm, transform, polygons):
    """Area and top-height quantiles of the annotated crowns themselves."""
    ids = rasterize(((g, i + 1) for i, g in enumerate(polygons.geometry)),
                    out_shape=chm.shape, transform=transform, fill=0,
                    dtype="int32")
    index = np.arange(1, len(polygons) + 1)
    present = ndimage.sum(np.ones_like(ids), ids, index) > 0
    tops = np.asarray(ndimage.maximum(chm, ids, index))[present]
    area = polygons.area.values
    return {"area": np.percentile(area, [10, 90]),
            "top": np.percentile(tops, [10, 25])}


def confidence(areaM2, topH, solidity, ref):
    areaLow, areaHigh = ref["area"]
    topP10, topP25 = ref["top"]
    if (areaLow <= areaM2 <= areaHigh and topH >= topP25 and solidity >= 0.85):
        return "A"
    if topH >= topP10 and 0.5 * areaLow <= areaM2 <= 2 * areaHigh:
        return "B"
    return "C"


def grow(chm, transform, mask, args):
    """Marker watershed of the smoothed CHM inside the mask: label raster."""
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed
    res = abs(transform.a)
    smooth = ndimage.gaussian_filter(chm, args.smoothM / res)
    work = np.where(mask, smooth, 0.0).astype(np.float64)
    tops = h_maxima(work, args.minProminenceM).astype(bool) & mask
    markers, count = ndimage.label(tops, structure=np.ones((3, 3)))
    labels = watershed(-smooth, markers, mask=mask)
    return labels, markers, smooth, count


def crowns(labels, markers, chm, smooth, transform, annotated, ref, args, crs):
    res = abs(transform.a)
    taken = annotated
    rows = []
    for value in np.unique(labels[labels > 0]):
        region = labels == value
        region = ndimage.binary_closing(region, structure=np.ones((3, 3)))
        region = ndimage.binary_fill_holes(region)
        polys = [shape(g) for g, v in shapes(region.astype(np.uint8),
                                             mask=region, transform=transform)]
        if not polys:
            continue
        poly = max(polys, key=lambda p: p.area).simplify(res / 2)
        if args.growM > 0:
            poly = poly.buffer(args.growM, join_style=1)
        poly = poly.difference(taken)
        if poly.is_empty:
            continue
        if poly.geom_type != "Polygon":
            poly = max(getattr(poly, "geoms", [poly]), key=lambda p: p.area)
        if poly.area < args.minCrownM2:
            continue
        taken = unary_union([taken, poly])
        peak = float(chm[region].max())
        prominence = float(smooth[region].max() - smooth[region].min())
        solidity = float(poly.area / poly.convex_hull.area)
        rows.append({"areaM2": round(poly.area, 1), "topH": round(peak, 1),
                     "meanH": round(float(chm[region].mean()), 1),
                     "prominence": round(prominence, 2),
                     "solidity": round(solidity, 2),
                     "conf": confidence(poly.area, peak, solidity, ref),
                     "geometry": poly})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs) if rows else \
        gpd.GeoDataFrame({"areaM2": [], "conf": []}, geometry=[], crs=crs)


def inOriginalFrame(frame, mosaics, shifts, like):
    """Back to the annotation's own frame: unshift by part, then its CRS."""
    parts = {}
    for path in mosaics:
        for name in shifts:
            if name in os.path.basename(path):
                parts[name] = footprint([path])
    out = frame.copy()
    geoms = []
    for geom in frame.geometry:
        shift = (0.0, 0.0)
        for name, foot in parts.items():
            if foot.contains(geom.representative_point()):
                shift = shifts[name]
                break
        geoms.append(shapely_translate(geom, -shift[0], -shift[1]))
    out["geometry"] = geoms
    out = out.set_crs(frame.crs, allow_override=True).to_crs(like)
    out["name"] = "CAND"
    out["class"] = 0
    return out


def shapely_translate(geom, dx, dy):
    from shapely.affinity import translate
    return translate(geom, dx, dy)


def overview(path, mosaics, foot, annotated, candidates, pixelM):
    rgb, _ = mosaicImage(mosaics, foot.bounds, pixelM)
    west, south, east, north = foot.bounds
    figure, axis = plt.subplots(figsize=(13, 13 * (north - south) /
                                         max(east - west, 1)))
    axis.imshow(rgb, extent=(west, east, south, north))
    annotated.boundary.plot(ax=axis, color="yellow", linewidth=0.3)
    for conf, colour in (("A", "cyan"), ("B", "dodgerblue"), ("C", "magenta")):
        part = candidates[candidates["conf"] == conf] if len(candidates) else []
        if len(part):
            part.boundary.plot(ax=axis, color=colour, linewidth=0.9)
    axis.set_xlim(west, east)
    axis.set_ylim(south, north)
    axis.set_axis_off()
    axis.set_title("annotated crowns (yellow); candidates: A cyan, B blue, "
                   "C magenta")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chm", required=True)
    parser.add_argument("--unannotated", required=True,
                        help="unannotated.shp from tt.unannotatedTrees")
    parser.add_argument("--polygons", required=True,
                        help="the annotated crowns, in the CHM's frame")
    parser.add_argument("--mosaics", nargs="+", required=True)
    parser.add_argument("--shift", action="append", default=None,
                        help="PART=DX,DY, as given to tt.enshurinPrepare")
    parser.add_argument("--like", default=None,
                        help="the annotator's own file: the candidates are "
                             "also written in its coordinate system")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoothM", type=float, default=0.75)
    parser.add_argument("--minProminenceM", type=float, default=0.5)
    parser.add_argument("--growM", type=float, default=0.8)
    parser.add_argument("--minCrownM2", type=float, default=3.0)
    parser.add_argument("--overviewPixelM", type=float, default=0.25)
    args = parser.parse_args(argv)
    os.makedirs(args.output, exist_ok=True)

    mosaics = [m for p in args.mosaics for m in sorted(glob.glob(p))]
    with rasterio.open(args.chm) as src:
        chm = src.read(1, masked=True).filled(0).astype(np.float32)
        transform, crs = src.transform, src.crs
    annotated = gpd.read_file(args.polygons).to_crs(crs)
    patches = gpd.read_file(args.unannotated).to_crs(crs)
    mask = rasterize([(g, 1) for g in patches.geometry], out_shape=chm.shape,
                     transform=transform, fill=0, dtype="uint8").astype(bool)
    ref = reference(chm, transform, annotated)
    labels, markers, smooth, count = grow(chm, transform, mask, args)
    found = crowns(labels, markers, chm, smooth, transform,
                   unary_union(annotated.geometry), ref, args, crs)
    found.insert(0, "id", np.arange(1, len(found) + 1))
    found.to_file(os.path.join(args.output, "candidates.shp"))
    if len(found):
        gpd.GeoDataFrame(found[["id"]], geometry=found.geometry.centroid) \
            .to_file(os.path.join(args.output, "candidates_tops.shp"))
    if args.like and len(found):
        like = gpd.read_file(args.like).crs
        inOriginalFrame(found, mosaics, parseShifts(args.shift), like) \
            .to_file(os.path.join(args.output,
                                  "candidates_original_frame.shp"))
    overview(os.path.join(args.output, "candidates_overview.png"), mosaics,
             footprint(mosaics), annotated, found, args.overviewPixelM)
    summary = {"tops": int(count), "candidates": int(len(found)),
               "byConfidence": found["conf"].value_counts().to_dict()
               if len(found) else {},
               "areaM2Median": float(found.area.median()) if len(found) else 0,
               "annotatedReference": {"areaM2P10P90": ref["area"].round(1).tolist(),
                                      "topHP10P25": ref["top"].round(1).tolist()}}
    with open(os.path.join(args.output, "candidates.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print("[candidates] %s" % json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
