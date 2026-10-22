"""
Example images of the parts of a site that were left out of the scored area
because they are not annotated enough (Enshurin: cells with under 90% of their
canopy annotated), next to parts that were used.

    python -m tt.areaExamples \\
        --mosaics Data/enshurinNewOut/shifted/*.vrt \\
        --chm Data/vlad/enshurin_chm025.tif \\
        --polygons Data/enshurinNewOut/enshurin_all_shifted.shp \\
        --area Data/enshurinNewOut/enshurin/area.shp \\
        --output Data/enshurinNewOut/examples

--mosaics the (shifted) part mosaics, --chm the full-site CHM (the layers are
all on its grid), --polygons every annotated crown (shifted like the mosaics),
--area the area the preparation kept (area.shp). Left out = inside the
mosaics and outside the area.

Windows of --windowM metres are chosen on a 1 m grid among those lying wholly
on the mosaics with canopy over 60%: --edge of them half used and half left
out (to see where the rule cuts), --unused of them almost entirely left out,
--used of them entirely used, spread over the site (at least --separationM
apart). --at X Y adds a window centred on a point of your choice.

Per window, <output>/example_<n>_<kind>.png: the mosaic alone on the left
(what the annotator saw); on the right the same with every annotated crown
outlined in yellow, the area's edge in cyan, and what was left out shaded.
The title gives, for canopy (CHM >= --canopyM) in that window, the share lying
inside an annotated crown in the used part and in the left-out part: the
quantity the rule is about. A contact sheet of all windows is
<output>/examples_all.png, and examples.json has the numbers.
"""

import argparse
import glob
import json
import os
import sys
import textwrap

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.ops import unary_union

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)

CELL_M = 1.0


def footprint(mosaics):
    """
    Where the mosaics have image: their non-black pixels, as the preparation
    finds them (not the rectangles around them: a rotated mosaic has empty
    corners, where the CHM may still show canopy).
    """
    from tt.enshurinPrepare import footprint as validData
    return unary_union([validData(m) for m in mosaics])


def siteGrid(chmPath, crowns, area, foot, canopyM):
    """1 m grid over the CHM: canopy, covered by a crown, used, in a mosaic."""
    with rasterio.open(chmPath) as src:
        shape = (int(round(src.height * abs(src.res[1]) / CELL_M)),
                 int(round(src.width * abs(src.res[0]) / CELL_M)))
        chm = src.read(1, out_shape=shape, resampling=Resampling.average,
                       masked=True).filled(0)
        transform = from_origin(src.bounds.left, src.bounds.top, CELL_M, CELL_M)
    burn = lambda geoms: rasterize([(g, 1) for g in geoms], out_shape=shape,
                                   transform=transform, fill=0,
                                   dtype="uint8").astype(bool)
    return {"transform": transform, "shape": shape, "canopy": chm >= canopyM,
            "covered": burn(crowns.geometry), "used": burn(area.geometry),
            "inMosaic": burn([foot])}


def windowStats(grid, row, col, size):
    s = (slice(row, row + size), slice(col, col + size))
    canopy, used = grid["canopy"][s], grid["used"][s]
    covered, inMosaic = grid["covered"][s], grid["inMosaic"][s]
    out = inMosaic & ~used

    def share(mask):
        n = (canopy & mask).sum()
        return float((canopy & mask & covered).sum() / n) if n else float("nan")

    return {"canopy": float(canopy.mean()), "inMosaic": float(inMosaic.mean()),
            "unused": float(out.mean()), "coveredUsed": share(used),
            "coveredUnused": share(out)}


def candidates(grid, size):
    stride = max(1, size // 6)
    rows, cols = grid["shape"]
    found = []
    for r in range(0, rows - size + 1, stride):
        for c in range(0, cols - size + 1, stride):
            s = windowStats(grid, r, c, size)
            if s["inMosaic"] >= 0.99 and s["canopy"] >= 0.6:
                found.append((r, c, s))
    return found


def pick(found, key, count, taken, separation, size):
    """Up to count windows best by key, at least separation from taken ones."""
    chosen = []
    for r, c, s in sorted(found, key=lambda f: key(f[2])):
        if len(chosen) == count:
            break
        if all(np.hypot(r - r2, c - c2) * CELL_M >= separation
               for r2, c2 in taken + [(x[0], x[1]) for x in chosen]):
            chosen.append((r, c, s))
    return chosen


def render(mosaics, window, crowns, area, foot, stats, title, path, pixelM):
    """One two-panel figure; window is (west, south, east, north)."""
    data = [rasterio.open(m) for m in mosaics]
    try:
        image, transform = merge(data, bounds=window, res=pixelM,
                                 resampling=Resampling.average)
    finally:
        for d in data:
            d.close()
    rgb = np.moveaxis(image[:3], 0, -1)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb / max(float(rgb.max()), 1) * 255, 0, 255).astype(np.uint8)
    west, south, east, north = window
    extent = (west, east, south, north)
    frame = box(*window)
    left = gpd.GeoSeries([foot.difference(unary_union(area.geometry))
                          .intersection(frame)])
    figure, axes = plt.subplots(1, 2, figsize=(14, 7.4))
    for axis in axes:
        axis.imshow(rgb, extent=extent)
        axis.set_xlim(west, east)
        axis.set_ylim(south, north)
        axis.set_axis_off()
    axes[0].set_title("mosaic")
    axes[1].set_title(textwrap.fill(title, 62), fontsize=9)
    inside = crowns[crowns.intersects(frame)]
    if len(inside):
        inside.boundary.plot(ax=axes[1], color="yellow", linewidth=0.7)
    if not left.iloc[0].is_empty:
        left.plot(ax=axes[1], color="black", alpha=0.5, linewidth=0)
    edge = unary_union(area.geometry).boundary.intersection(frame)
    if not edge.is_empty:
        gpd.GeoSeries([edge]).plot(ax=axes[1], color="cyan", linewidth=1.4)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    return int(len(inside))


def describe(kind, s):
    f = lambda v: "no canopy" if np.isnan(v) else "%.0f%%" % (100 * v)
    return ("%s: %.0f%% of the window left out. Canopy inside an annotated "
            "crown: %s in the used part, %s in the part left out"
            % (kind, 100 * s["unused"], f(s["coveredUsed"]),
               f(s["coveredUnused"])))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mosaics", nargs="+", required=True)
    parser.add_argument("--chm", required=True)
    parser.add_argument("--polygons", required=True)
    parser.add_argument("--area", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--windowM", type=float, default=60.0)
    parser.add_argument("--edge", type=int, default=3)
    parser.add_argument("--unused", type=int, default=2)
    parser.add_argument("--used", type=int, default=1)
    parser.add_argument("--separationM", type=float, default=80.0)
    parser.add_argument("--canopyM", type=float, default=2.0)
    parser.add_argument("--pixelM", type=float, default=0.1)
    parser.add_argument("--at", nargs=2, type=float, action="append",
                        metavar=("X", "Y"), default=[])
    args = parser.parse_args(argv)
    os.makedirs(args.output, exist_ok=True)

    mosaics = [m for pattern in args.mosaics for m in sorted(glob.glob(pattern))]
    crowns = gpd.read_file(args.polygons)
    area = gpd.read_file(args.area)
    foot = footprint(mosaics)
    grid = siteGrid(args.chm, crowns, area, foot, args.canopyM)
    size = int(round(args.windowM / CELL_M))
    found = candidates(grid, size)
    taken, plan = [], []
    for kind, count, key in (
            ("edge", args.edge, lambda s: abs(s["unused"] - 0.5)),
            ("mostly left out", args.unused, lambda s: 1 - s["unused"]),
            ("used", args.used, lambda s: s["unused"])):
        chosen = pick(found, key, count, taken, args.separationM, size)
        taken += [(r, c) for r, c, _ in chosen]
        plan += [(kind, r, c, s) for r, c, s in chosen]
    for x, y in args.at:
        col = int(round((x - args.windowM / 2 - grid["transform"].c) / CELL_M))
        row = int(round((grid["transform"].f - y - args.windowM / 2) / CELL_M))
        plan.append(("chosen", row, col, windowStats(grid, row, col, size)))

    summary, files = [], []
    for n, (kind, r, c, s) in enumerate(plan, 1):
        west = grid["transform"].c + c * CELL_M
        north = grid["transform"].f - r * CELL_M
        window = (west, north - size * CELL_M, west + size * CELL_M, north)
        path = os.path.join(args.output, "example_%d_%s.png"
                            % (n, kind.replace(" ", "_")))
        count = render(mosaics, window, crowns, area, foot, s,
                       describe(kind, s), path, args.pixelM)
        summary.append(dict(s, kind=kind, window=window, crowns=count, file=path))
        files.append(path)
        print("[examples] %s  (%d crowns)  %s" % (path, count, describe(kind, s)))
    if files:
        columns = 2 if len(files) > 1 else 1
        rows = int(np.ceil(len(files) / columns))
        figure, axes = plt.subplots(rows, columns, squeeze=False,
                                    figsize=(14 * columns / 1.5, 7.4 * rows / 1.5))
        for axis in axes.ravel():
            axis.set_axis_off()
        for axis, path in zip(axes.ravel(), files):
            axis.imshow(plt.imread(path))
        figure.tight_layout()
        figure.savefig(os.path.join(args.output, "examples_all.png"), dpi=110)
        plt.close(figure)
    with open(os.path.join(args.output, "examples.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
