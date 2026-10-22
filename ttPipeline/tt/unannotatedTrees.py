"""
A mask of the tree canopy that no annotated polygon covers: where the
annotation is missing trees. As a shapefile, an overview image over the RGB,
and a georeferenced RGB with the mask burned in (for QGIS or for someone
without it).

    python -m tt.unannotatedTrees \\
        --chm Data/vlad/enshurin_chm025.tif \\
        --polygons Data/enshurinNewOut/enshurin_all_shifted.shp \\
        --mosaics "Data/enshurinNewOut/enshurin/shifted/*.vrt" \\
        --area Data/enshurinNewOut/enshurin/area.shp \\
        --output Data/enshurinNewOut/unannotated

Unannotated tree canopy, on the CHM's grid, inside the mosaics:
  CHM >= --minHeightM (tree height; lower vegetation, shrubs, is left out),
  inside the part of the mosaics that has image (not the empty corners),
  farther than --marginM from every polygon (outlines are drawn tighter than
  the crown's rim: the rim is not a missing tree),
  then opened with a disc of --openingM (thin slivers between crowns go) and
  patches under --minPatchM2 dropped.
Treetops in it, for a rough count of the trees missing: local maxima of the
CHM smoothed over 1 m, at least --peakWindowM apart. A rough count: a crown
with two tops counts twice, two touching crowns of equal height may count once.

Writes, in --output:
  unannotated.shp       one polygon per patch: areaM2, meanH, maxH, tops, and
                        inArea (share of the patch inside --area, if given)
  unannotated_tops.shp  the tops counted
  overview.png          the whole mosaic: RGB, the mask in red, the polygons in
                        yellow, the area's edge in cyan (if --area)
  rgb_unannotated.tif   the RGB at --rgbPixelM with the mask burned in red and
                        the polygons' outlines in yellow, georeferenced
  summary.json          the totals, also printed
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
from rasterio.merge import merge
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)


def footprint(mosaics):
    """
    Where the mosaics have image (non-black pixels, as the preparation finds
    them), not the rectangles around them: a rotated mosaic has empty corners
    where the CHM may still show canopy that is not in any image.
    """
    from tt.enshurinPrepare import footprint as validData
    return unary_union([validData(m) for m in mosaics])


def disc(radius):
    r = max(1, int(round(radius)))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def readChm(path):
    with rasterio.open(path) as src:
        chm = src.read(1, masked=True).filled(0).astype(np.float32)
        return chm, src.transform, src.crs


def unannotatedMask(chm, transform, polygons, foot, args):
    """Tree canopy farther than the margin from every polygon, cleaned."""
    res = abs(transform.a)
    burn = lambda geoms: rasterize([(g, 1) for g in geoms if not g.is_empty],
                                   out_shape=chm.shape, transform=transform,
                                   fill=0, dtype="uint8").astype(bool)
    covered = burn(polygons.geometry.buffer(args.marginM))
    inMosaic = burn([foot])
    raw = (chm >= args.minHeightM) & inMosaic & ~covered
    opened = ndimage.binary_opening(raw, structure=disc(args.openingM / res))
    labels, count = ndimage.label(opened)
    sizes = ndimage.sum(np.ones_like(labels), labels, np.arange(1, count + 1))
    keep = np.zeros(count + 1, bool)
    keep[1:] = sizes * res * res >= args.minPatchM2
    return keep[labels], inMosaic, covered


def treetops(chm, transform, mask, args):
    res = abs(transform.a)
    smooth = ndimage.gaussian_filter(chm, 1.0 / res)
    size = max(3, int(round(args.peakWindowM / res)) | 1)
    peaks = (smooth == ndimage.maximum_filter(smooth, size=size)) & mask & \
        (smooth >= args.minHeightM)
    # a flat or symmetric top is several equal cells: one top per group
    labels, count = ndimage.label(peaks, structure=np.ones((3, 3)))
    if count == 0:
        return gpd.GeoDataFrame({"heightM": []}, geometry=[])
    index = np.arange(1, count + 1)
    centres = np.array(ndimage.center_of_mass(peaks, labels, index))
    xs, ys = rasterio.transform.xy(transform, centres[:, 0], centres[:, 1])
    heights = ndimage.maximum(chm, labels, index)
    return gpd.GeoDataFrame({"heightM": np.round(heights, 1)},
                            geometry=gpd.points_from_xy(xs, ys))


def patches(mask, chm, transform, crs, tops, area):
    labels, _ = ndimage.label(mask)
    geoms, ids = [], []
    for geom, value in shapes(labels.astype(np.int32), mask=mask,
                              transform=transform):
        geoms.append(shape(geom))
        ids.append(int(value))
    frame = gpd.GeoDataFrame({"id": ids}, geometry=geoms, crs=crs)
    frame = frame.dissolve(by="id", as_index=False)
    index = frame["id"].values
    frame["areaM2"] = frame.area.round(1)
    frame["meanH"] = np.round(ndimage.mean(chm, labels, index), 1)
    frame["maxH"] = np.round(ndimage.maximum(chm, labels, index), 1)
    joined = gpd.sjoin(tops.set_crs(crs, allow_override=True),
                       frame[["id", "geometry"]], predicate="within")
    frame["tops"] = frame["id"].map(joined["id"].value_counts()).fillna(0) \
        .astype(int)
    if area is not None:
        inside = frame.geometry.intersection(area).area
        frame["inArea"] = np.round(inside / frame.area, 2)
    return frame


def mosaicImage(mosaics, bounds, pixelM):
    data = [rasterio.open(m) for m in mosaics]
    try:
        image, transform = merge(data, bounds=bounds, res=pixelM)
    finally:
        for d in data:
            d.close()
    rgb = np.moveaxis(image[:3], 0, -1)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb / max(float(rgb.max()), 1) * 255, 0, 255) \
            .astype(np.uint8)
    return rgb, transform


def burned(rgb, transform, frame, polygons, alpha=0.45):
    """The RGB with the mask in red and the polygons' outlines in yellow."""
    out = rgb.astype(np.float32).copy()
    burn = lambda geoms, touched: rasterize(
        [(g, 1) for g in geoms if not g.is_empty], out_shape=rgb.shape[:2],
        transform=transform, fill=0, all_touched=touched,
        dtype="uint8").astype(bool)
    mask = burn(frame.geometry, False)
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array([255, 0, 0])
    outline = burn(polygons.geometry.boundary, True)
    out[outline] = (255, 230, 0)
    return out.astype(np.uint8)


def writeTiff(path, rgb, transform, crs):
    with rasterio.open(path, "w", driver="GTiff", height=rgb.shape[0],
                       width=rgb.shape[1], count=3, dtype="uint8", crs=crs,
                       transform=transform, compress="deflate", tiled=True,
                       photometric="RGB") as out:
        out.write(np.moveaxis(rgb, -1, 0))


def overview(path, mosaics, foot, frame, polygons, area, args):
    rgb, transform = mosaicImage(mosaics, foot.bounds, args.overviewPixelM)
    west, south, east, north = foot.bounds
    figure, axis = plt.subplots(figsize=(13, 13 * (north - south) /
                                         max(east - west, 1)))
    axis.imshow(rgb, extent=(west, east, south, north))
    if len(frame):
        frame.plot(ax=axis, color="red", alpha=0.45, linewidth=0)
    polygons.boundary.plot(ax=axis, color="yellow", linewidth=0.3)
    if area is not None:
        gpd.GeoSeries([area.boundary]).plot(ax=axis, color="cyan",
                                            linewidth=1.0)
    axis.set_xlim(west, east)
    axis.set_ylim(south, north)
    axis.set_axis_off()
    axis.set_title("unannotated tree canopy (red), annotated crowns (yellow)"
                   + (", scored area (cyan)" if area is not None else ""))
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def summarise(frame, tops, area, res, mask, covered, inMosaic, chm, args):
    canopy = (chm >= args.minHeightM) & inMosaic
    s = {"patches": int(len(frame)),
         "unannotatedTreeCanopyHa": round(float(mask.sum() * res * res / 1e4), 2),
         "treeCanopyHa": round(float(canopy.sum() * res * res / 1e4), 2),
         "annotatedShareOfTreeCanopy": round(float((canopy & covered).sum() /
                                                   max(canopy.sum(), 1)), 3),
         "topsRough": int(len(tops)), "minHeightM": args.minHeightM}
    if area is not None and len(frame):
        inside = frame.geometry.intersection(area).area.sum()
        s["unannotatedInAreaHa"] = round(float(inside / 1e4), 2)
        s["topsInArea"] = int(tops.within(area).sum())
    return s


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--chm", required=True)
    parser.add_argument("--polygons", required=True)
    parser.add_argument("--mosaics", nargs="+", required=True)
    parser.add_argument("--area", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minHeightM", type=float, default=4.0)
    parser.add_argument("--marginM", type=float, default=1.0)
    parser.add_argument("--openingM", type=float, default=1.0)
    parser.add_argument("--minPatchM2", type=float, default=10.0)
    parser.add_argument("--peakWindowM", type=float, default=3.0)
    parser.add_argument("--overviewPixelM", type=float, default=0.25)
    parser.add_argument("--rgbPixelM", type=float, default=0.1)
    args = parser.parse_args(argv)
    os.makedirs(args.output, exist_ok=True)

    mosaics = [m for p in args.mosaics for m in sorted(glob.glob(p))]
    foot = footprint(mosaics)
    polygons = gpd.read_file(args.polygons)
    chm, transform, crs = readChm(args.chm)
    polygons = polygons.to_crs(crs) if polygons.crs != crs else polygons
    area = gpd.read_file(args.area).to_crs(crs).unary_union \
        if args.area else None
    mask, inMosaic, covered = unannotatedMask(chm, transform, polygons, foot,
                                              args)
    tops = treetops(chm, transform, mask, args).set_crs(crs)
    frame = patches(mask, chm, transform, crs, tops, area)
    frame.to_file(os.path.join(args.output, "unannotated.shp"))
    if len(tops):
        tops.to_file(os.path.join(args.output, "unannotated_tops.shp"))
    overview(os.path.join(args.output, "overview.png"), mosaics, foot, frame,
             polygons, area, args)
    rgb, rgbTransform = mosaicImage(mosaics, foot.bounds, args.rgbPixelM)
    writeTiff(os.path.join(args.output, "rgb_unannotated.tif"),
              burned(rgb, rgbTransform, frame, polygons), rgbTransform, crs)
    s = summarise(frame, tops, area, abs(transform.a), mask, covered,
                  inMosaic, chm, args)
    with open(os.path.join(args.output, "summary.json"), "w") as handle:
        json.dump(s, handle, indent=2)
    print("[unannotated] %s" % json.dumps(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
