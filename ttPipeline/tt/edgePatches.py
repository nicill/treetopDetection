"""
The edge part of a site's unannotated canopy, to show the annotator what would
be left out of the scoring: canopy outside every crown that lies at the border
of the area (the 'edge' patches of uncovered.shp), as opposed to the interior
gaps between crowns, which stay scored.

    python -m tt.edgePatches --site Data/sergiOut/koiwainoujo220616 \\
        --like Data/sergi/Shapefiles/CanopyMerge.shp \\
        --output Data/sergiOut/edgeForSergi

--site is a prepared site (uncovered.shp, crowns.shp, area.shp, rgb.tif, as
qpPrepare, sergiPrepare and enshurinPrepare write it). Writes, in --output:

  edge_patches.shp          the edge patches, in the site's coordinate system,
                            with their area
  edge_patches_original.shp the same in the coordinate system of --like (the
                            annotator's own file), if given
  overview.png              the whole area: the RGB, the crowns in yellow, the
                            area's outline in cyan, the edge patches filled in
                            red (would be left out) and the interior gaps
                            outlined in magenta (stay scored)
  zooms.png                 --zooms windows of --zoomM metres around the
                            largest edge patches, at full resolution
  summary.json              counts and areas, also printed
"""

import argparse
import json
import os
import sys

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after the backend is chosen)
from matplotlib.patches import Patch  # noqa: E402


def isEdge(values):
    """The edge flag as written to a shapefile (bool, 0/1 or text)."""
    return np.array([str(v).strip().lower() in ("1", "true", "t", "yes")
                     for v in values])


def readRgb(path, bounds, maxPixels=None):
    """The RGB over bounds, downsampled so its long side is at most maxPixels."""
    with rasterio.open(path) as src:
        window = from_bounds(*bounds, src.transform)
        height, width = int(round(window.height)), int(round(window.width))
        scale = max(1.0, max(height, width) / float(maxPixels)) \
            if maxPixels else 1.0
        data = src.read([1, 2, 3], window=window, boundless=True, fill_value=0,
                        out_shape=(3, max(1, int(height / scale)),
                                   max(1, int(width / scale))),
                        resampling=Resampling.average)
    rgb = np.moveaxis(data, 0, -1)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb / max(float(rgb.max()), 1) * 255, 0, 255).astype(np.uint8)
    return rgb


def draw(axis, rgb, bounds, crowns, area, edge, interior, lineScale=1.0):
    west, south, east, north = bounds
    axis.imshow(rgb, extent=(west, east, south, north))
    if len(interior):
        interior.boundary.plot(ax=axis, color="magenta", linewidth=0.5 * lineScale)
    if len(edge):
        edge.plot(ax=axis, color="red", alpha=0.55, linewidth=0)
        edge.boundary.plot(ax=axis, color="red", linewidth=0.6 * lineScale)
    crowns.boundary.plot(ax=axis, color="yellow", linewidth=0.3 * lineScale)
    area.boundary.plot(ax=axis, color="cyan", linewidth=1.2 * lineScale)
    axis.set_xlim(west, east)
    axis.set_ylim(south, north)
    axis.set_axis_off()


LEGEND = [Patch(facecolor="red", alpha=0.55, label="edge: would be left out"),
          Patch(facecolor="none", edgecolor="magenta",
                label="interior gaps: stay scored"),
          Patch(facecolor="none", edgecolor="yellow", label="annotated crowns"),
          Patch(facecolor="none", edgecolor="cyan", label="area (ROI)")]


def overview(path, rgbPath, crowns, area, edge, interior):
    bounds = tuple(area.total_bounds)
    rgb = readRgb(rgbPath, bounds, maxPixels=4000)
    west, south, east, north = bounds
    tall = (north - south) / max(east - west, 1)
    figure, axis = plt.subplots(figsize=(8 if tall > 1.5 else 13,
                                         min(8 * tall, 30) if tall > 1.5
                                         else 13 * tall + 1))
    draw(axis, rgb, bounds, crowns, area, edge, interior)
    axis.legend(handles=LEGEND, loc="upper right", fontsize=8)
    axis.set_title("%d edge patches, %.0f m2 (would be left out); %d interior "
                   "gaps, %.0f m2 (stay scored)" % (len(edge), edge.area.sum(),
                                                    len(interior),
                                                    interior.area.sum()),
                   fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def zooms(path, rgbPath, crowns, area, edge, interior, count, sizeM):
    picks = edge.sort_values("areaM2", ascending=False).head(count)
    if not len(picks):
        return 0
    columns = min(3, len(picks))
    rows = int(np.ceil(len(picks) / columns))
    figure, axes = plt.subplots(rows, columns, squeeze=False,
                                figsize=(5.5 * columns, 5.5 * rows + 0.6))
    for axis in axes.ravel():
        axis.set_axis_off()
    for axis, (_, patch) in zip(axes.ravel(), picks.iterrows()):
        c = patch.geometry.centroid
        bounds = (c.x - sizeM / 2, c.y - sizeM / 2, c.x + sizeM / 2,
                  c.y + sizeM / 2)
        rgb = readRgb(rgbPath, bounds, maxPixels=1500)
        draw(axis, rgb, bounds, crowns, area, edge, interior, lineScale=2.0)
        axis.set_title("edge patch of %.0f m2 at %.0f, %.0f"
                       % (patch.geometry.area, c.x, c.y), fontsize=9)
    figure.legend(handles=LEGEND, loc="lower center", ncol=4, fontsize=9)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return len(picks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--like", default=None,
                        help="the annotator's file: the patches are also "
                             "written in its coordinate system")
    parser.add_argument("--zooms", type=int, default=6)
    parser.add_argument("--zoomM", type=float, default=30.0)
    args = parser.parse_args(argv)
    os.makedirs(args.output, exist_ok=True)

    uncovered = gpd.read_file(os.path.join(args.site, "uncovered.shp"))
    crowns = gpd.read_file(os.path.join(args.site, "crowns.shp"))
    area = gpd.read_file(os.path.join(args.site, "area.shp"))
    flag = isEdge(uncovered["edge"])
    edge = uncovered[flag].copy()
    interior = uncovered[~flag].copy()
    edge["areaM2"] = edge.area.round(2)
    edge[["areaM2", "geometry"]].to_file(
        os.path.join(args.output, "edge_patches.shp"))
    if args.like:
        edge[["areaM2", "geometry"]].to_crs(gpd.read_file(args.like).crs) \
            .to_file(os.path.join(args.output, "edge_patches_original.shp"))
    rgbPath = os.path.join(args.site, "rgb.tif")
    overview(os.path.join(args.output, "overview.png"), rgbPath, crowns, area,
             edge, interior)
    shown = zooms(os.path.join(args.output, "zooms.png"), rgbPath, crowns,
                  area, edge, interior, args.zooms, args.zoomM)
    summary = {"edgePatches": int(len(edge)),
               "edgeM2": round(float(edge.area.sum()), 1),
               "edgeMedianM2": round(float(edge.area.median()), 2)
               if len(edge) else 0,
               "interiorPatches": int(len(interior)),
               "interiorM2": round(float(interior.area.sum()), 1),
               "areaM2": round(float(area.area.sum()), 1),
               "edgeShareOfArea": round(float(edge.area.sum() /
                                              area.area.sum()), 4),
               "zoomsShown": shown}
    with open(os.path.join(args.output, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print("[edge] %s" % json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
