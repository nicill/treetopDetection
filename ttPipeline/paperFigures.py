"""
Two figures that explain the problem, the data and the annotations, drawn from the study folders:

  fig_task     one window of one site (default tereljFirst), six panels. Top, the data: the RGB mosaic,
               the photogrammetric (P1) CHM and the LiDAR CHM where there is one. Bottom, the annotation
               and the scoring: the annotated crowns and the scored area; a detector's held-out output
               (default CC, runs/ccP1) marked as hit, repeat (a second detection in a crown already
               found), false positive (in no crown) and missed crown; and the height profile along a
               line through the window, crowns shaded, detections marked.
  fig_forests  one column per kind of forest (default afcamoisan, koiwai, tereljFirst, enshurinV2), same
               ground scale: RGB, P1 CHM with the crowns, and the height profile along a line, on P1 and
               on LiDAR where it exists.

Nothing is chosen by eye. A window is a square of --window metres inside the scored area, on a regular
grid: for fig_forests the one whose number of crowns is the median over the site's windows (a typical
window); for fig_task the one where the rarest of the four outcomes is most frequent (so all four show).
The profile line is the east-west line through the window that crosses the most crowns.

    cd ttPipeline && python paperFigures.py                 # writes DROP/paperFigures/fig_task.*, fig_forests.*
    python paperFigures.py --taskSite koiwai --taskRun mrcnnRgb --window 20
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
import rasterio                                                  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from matplotlib.patches import Patch, Rectangle                  # noqa: E402
from rasterio.enums import Resampling                            # noqa: E402
from rasterio.warp import transform_bounds                       # noqa: E402
from rasterio.windows import from_bounds                         # noqa: E402
from shapely.geometry import LineString, box                     # noqa: E402

from tt.data import TDData                                       # noqa: E402
from tt.experiments import RunSet                                # noqa: E402
from tt.scoring import AnnotatedCrowns, Evaluator                # noqa: E402

DB = os.environ.get("DROP") or os.path.expanduser("~/Yago Lab Dropbox/Meetings/2026TreeDet")
QUEBEC = ["20230712_afcamoisan_itrf20", "20230712_afcahoule_itrf20",
          "20230606_cbblackburn5", "20230608_cbbernard4", "20230606_cbblackburn6"]
SITES = dict([(s.split("_")[1], os.path.join(DB, "study_edge", s)) for s in QUEBEC]
             + [("koiwai", os.path.join(DB, "study_edge", "koiwainoujo220616"))]
             + [("terelj" + a, os.path.join(DB, "study_terelj" + a, "terelj" + a))
                for a in ("First", "Second", "Third")]
             + [("enshurinV2", os.path.join(DB, "study_enshurinVlad2"))])
LABEL = {"afcamoisan": "Quebec plantation (afcamoisan)", "afcahoule": "Quebec plantation (afcahoule)",
         "cbblackburn5": "Quebec plantation (cbblackburn5)", "cbbernard4": "Quebec plantation (cbbernard4)",
         "cbblackburn6": "Quebec plantation (cbblackburn6)", "koiwai": "Koiwai, dense larch stand",
         "tereljFirst": "Terelj, larch forest", "tereljSecond": "Terelj, larch forest (area 2)",
         "tereljThird": "Terelj, larch forest (area 3)", "enshurinV2": "Enshurin, broadleaf forest"}
PIXELS = 700                       # a window is drawn at this many pixels a side
CROWN, MISSED = "#ffffff", "#e41a1c"
KIND = {"hit": ("#2ca02c", "o", "hit"), "repeat": ("#ff7f0e", "o", "repeat (crown already found)"),
        "background": ("#e41a1c", "X", "false positive (no crown)")}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})


# ------------------------------------------------------------------ the site's files

def resolve(path):
    return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)


def chmOf(dataset):
    with open(os.path.join(dataset, "dataset.json")) as handle:
        meta = json.load(handle)
    return resolve(next(c["path"] for c in meta["channels"] if c["kind"] == "chm"))


def rgbOf(folder):
    path = os.path.join(folder, "ds", "rgb", "dataset.json")
    if not os.path.exists(path):
        with open(os.path.join(folder, "runs", "mrcnnRgb", "results.json")) as handle:
            path = os.path.join(json.load(handle)["settings"]["dataset"], "dataset.json")
    with open(path) as handle:
        meta = json.load(handle)
    return resolve(next(c["path"] for c in meta["channels"] if c["kind"] == "rgb"))


class Site(object):
    """A study site's rasters, crowns, scored area and blocks."""

    def __init__(self, name, folder):
        self.name, self.folder = name, folder
        datasets = {}
        for d in sorted(glob.glob(os.path.join(folder, "ds", "*", "dataset.json"))):
            key = os.path.basename(os.path.dirname(d))
            with open(d) as handle:
                kinds = [c["kind"] for c in json.load(handle).get("channels", [])]
            if kinds == ["chm"]:
                datasets.setdefault("lidar" if "lidar" in key.lower() else key, os.path.dirname(d))
        self.dataset = datasets.get("P1") or datasets.get("chm")
        if self.dataset is None:
            raise SystemExit("%s: no P1 CHM dataset under %s/ds" % (name, folder))
        self.data = TDData.fromDataset(self.dataset)
        self.crowns = self.data.annotations.crowns
        area = self.data.annotations.area
        if area is None:
            from shapely.ops import unary_union
            area = unary_union([b.geometry for b in self.data.blocks])
        self.area = area.geometry.unary_union if hasattr(area, "geometry") else area
        self.p1 = chmOf(self.dataset)
        self.lidar = chmOf(datasets["lidar"]) if "lidar" in datasets else None
        self.rgb = rgbOf(folder)
        self.centroids = np.array([(g.centroid.x, g.centroid.y) for g in self.crowns.geometry])

    def describe(self):
        diameter = float(np.median(2 * np.sqrt(self.crowns.geometry.area.values / np.pi)))
        return "%.1f m crowns, %d trees/ha" % (diameter, round(len(self.crowns) / (self.area.area / 1e4)))


# ------------------------------------------------------------------ windows

def windows(site, size):
    """Lower-left corners of the size-metre squares, on a half-size grid, that lie inside the scored area."""
    x0, y0, x1, y1 = site.area.bounds
    inside, partly = [], []
    for x in np.arange(x0, x1 - size / 2, size / 2):
        for y in np.arange(y0, y1 - size / 2, size / 2):
            square = box(x, y, x + size, y + size)
            if site.area.contains(square):
                inside.append((x, y))
            else:
                partly.append((site.area.intersection(square).area, x, y))
    if inside:
        return inside
    best = max(partly)[0]                      # a narrow area: the squares that hold most of it
    return [(x, y) for a, x, y in partly if a >= 0.9 * best]


def inWindow(points, corner, size):
    if len(points) == 0:
        return np.zeros(0, bool)
    x, y = corner
    return (points[:, 0] >= x) & (points[:, 0] < x + size) & (points[:, 1] >= y) & (points[:, 1] < y + size)


def typicalWindow(site, size):
    corners = windows(site, size)
    counts = np.array([inWindow(site.centroids, c, size).sum() for c in corners])
    order = np.argsort(np.abs(counts - np.median(counts)), kind="stable")
    return corners[int(order[0])]


def outcomeWindow(site, size, outcome):
    """The window where the rarest of the four outcomes is most frequent."""
    best, bestScore = None, None
    for corner in windows(site, size):
        counts = [inWindow(outcome["points"][outcome["kinds"] == k], corner, size).sum() for k in KIND]
        counts.append(inWindow(outcome["missed"], corner, size).sum())
        score = (min(counts), sum(counts))
        if bestScore is None or score > bestScore:
            best, bestScore = corner, score
    return best


def profileLine(site, corner, size):
    """The east-west line through the window that passes through the central part of the most crowns
    (within a third of the crown's radius of its centre); ties go to the longer path inside crowns."""
    x, y = corner
    near = site.crowns[site.crowns.intersects(box(x, y, x + size, y + size))]
    centres = np.array([(g.centroid.x, g.centroid.y, np.sqrt(g.area / np.pi)) for g in near.geometry]).reshape(-1, 3)
    best, bestScore = None, None
    for fraction in np.linspace(0.15, 0.85, 29):
        height = y + fraction * size
        line = LineString([(x, height), (x + size, height)])
        inside = (centres[:, 0] >= x) & (centres[:, 0] <= x + size)
        central = int(np.sum(inside & (np.abs(centres[:, 1] - height) <= centres[:, 2] / 3.0)))
        score = (central, float(near.intersection(line).length.sum()) if len(near) else 0.0)
        if bestScore is None or score > bestScore:
            best, bestScore = line, score
    return best


# ------------------------------------------------------------------ reading and drawing

def readWindow(path, corner, size, crs, pixels=PIXELS, bands=None):
    x, y = corner
    with rasterio.open(path) as src:
        bounds = (x, y, x + size, y + size)
        if crs is not None and src.crs is not None and src.crs != crs:
            bounds = transform_bounds(crs, src.crs, *bounds)
        indexes = list(range(1, (bands or src.count) + 1))
        data = src.read(indexes, window=from_bounds(*bounds, transform=src.transform), boundless=True,
                        out_shape=(len(indexes), pixels, pixels), resampling=Resampling.bilinear,
                        fill_value=0).astype("float32")
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = 0
    return data


def rgbImage(site, corner, size):
    with rasterio.open(site.rgb) as src:
        bands = min(3, src.count)
    data = readWindow(site.rgb, corner, size, site.crowns.crs, bands=bands)
    if data.shape[0] < 3:
        data = np.repeat(data[:1], 3, axis=0)
    high = np.percentile(data, 99.5) if data.max() > 255 else 255.0
    return np.clip(np.moveaxis(data, 0, -1) / max(high, 1e-6), 0, 1)


def chmImage(path, site, corner, size):
    data = readWindow(path, corner, size, site.crowns.crs, bands=1)[0]
    return np.where(np.isfinite(data), np.maximum(data, 0), 0)


def frame(axis, corner, size, title=None):
    x, y = corner
    axis.set_xlim(x, x + size)
    axis.set_ylim(y, y + size)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal")
    if title:
        axis.set_title(title, loc="left")


def show(axis, image, corner, size, **keywords):
    x, y = corner
    return axis.imshow(image, extent=(x, x + size, y, y + size), origin="upper", **keywords)


def outline(axis, site, corner, size, colour=CROWN, width=0.7, only=None):
    square = box(corner[0], corner[1], corner[0] + size, corner[1] + size)
    crowns = site.crowns[site.crowns.intersects(square)] if only is None else only
    for geometry in crowns.geometry if hasattr(crowns, "geometry") else crowns:
        for part in getattr(geometry, "geoms", [geometry]):
            if part.geom_type == "Polygon":
                px, py = part.exterior.xy
                axis.plot(px, py, color=colour, lw=width)


def shadeOutside(axis, site, corner, size):
    """Hatch what lies outside the scored area."""
    outside = box(corner[0], corner[1], corner[0] + size, corner[1] + size).difference(site.area)
    for part in getattr(outside, "geoms", [outside]):
        if part.geom_type == "Polygon" and not part.is_empty:
            px, py = part.exterior.xy
            axis.fill(px, py, facecolor="black", alpha=0.45, hatch="////", edgecolor="white", lw=0)


def scaleBar(axis, corner, size, length=5.0):
    x, y = corner[0] + 0.05 * size, corner[1] + 0.06 * size
    axis.plot([x, x + length], [y, y], color="white", lw=2.5, solid_capstyle="butt")
    axis.text(x + length / 2, y + 0.02 * size, "%g m" % length, color="white", ha="center", va="bottom",
              fontsize=7, fontweight="bold")


def heightBar(figure, axis, image, below=False):
    """A height scale beside (or below) the panel, in its own inset so the panel keeps its size."""
    if below:
        inset = axis.inset_axes([0.0, -0.075, 1.0, 0.04])
        bar = figure.colorbar(image, cax=inset, orientation="horizontal")
    else:
        inset = axis.inset_axes([1.03, 0.0, 0.045, 1.0])
        bar = figure.colorbar(image, cax=inset)
        bar.ax.set_title("m", fontsize=7, pad=3)
    bar.ax.tick_params(labelsize=6.5, length=2)


def profile(axis, site, corner, size, line, surfaces, marks=None):
    """Heights along the line, crowns shaded; surfaces: [(label, image, colour, style)]."""
    x = corner[0]
    y = line.coords[0][1]
    for geometry in site.crowns[site.crowns.intersects(line)].geometry:
        cut = geometry.intersection(line)
        for part in getattr(cut, "geoms", [cut]):
            if part.geom_type == "LineString":
                axis.axvspan(part.bounds[0] - x, part.bounds[2] - x, color="0.85", lw=0)
    top = 0.0
    for label, image, colour, style in surfaces:
        row = int(round((corner[1] + size - y) / size * (image.shape[0] - 1)))
        heights = image[min(max(row, 0), image.shape[0] - 1)]
        axis.plot(np.linspace(0, size, len(heights)), heights, color=colour, ls=style, lw=1.1, label=label)
        top = max(top, float(heights.max()))
    for mx, colour, marker in marks or []:
        axis.plot(mx - x, top * 1.08, marker=marker, color=colour, ms=5, ls="", clip_on=False)
    axis.set_xlim(0, size)
    axis.set_ylim(0, top * 1.15 if top > 0 else 1)
    axis.set_xlabel("m along the line")
    axis.set_ylabel("height (m)")
    axis.spines[["top", "right"]].set_visible(False)


# ------------------------------------------------------------------ the scoring of one run

def outcomes(site, run):
    """Every held-out detection of the run with its verdict, and the centroids of the crowns it misses."""
    evaluator = Evaluator(AnnotatedCrowns(site.data.annotations))
    blocks = {b.name: b.geometry for b in site.data.blocks}
    held = RunSet.load(os.path.join(site.folder, "runs", run)).heldOut(blocks)
    points, kinds, missed, missedShapes, boxes = [], [], [], [], []
    for block, geometry in blocks.items():
        matching = evaluator.match(held.subset(held.cache["block"] == block), geometry)
        d = matching.detections
        points += list(zip(d.x, d.y))
        kinds += list(matching.kinds)
        boxes += list(d.boxes) if d.boxes is not None else [None] * len(d.x)
        for i in matching.missed:
            shape = matching.crowns.geometry.iloc[i]
            missed.append((shape.centroid.x, shape.centroid.y))
            missedShapes.append(shape)
    return {"points": np.array(points).reshape(-1, 2), "kinds": np.array(kinds), "boxes": boxes,
            "missed": np.array(missed).reshape(-1, 2), "missedShapes": missedShapes}


# ------------------------------------------------------------------ the figures

def figureTask(site, run, size, output):
    outcome = outcomes(site, run)
    corner = outcomeWindow(site, size, outcome)
    line = profileLine(site, corner, size)
    rgb = rgbImage(site, corner, size)
    p1 = chmImage(site.p1, site, corner, size)
    lidar = chmImage(site.lidar, site, corner, size) if site.lidar else None
    top = max(float(np.percentile(p1, 99.5)), 1.0)

    figure, axes = plt.subplots(2, 3, figsize=(7.4, 5.4))
    (a, b, c), (d, e, f) = axes
    show(a, rgb, corner, size)
    frame(a, corner, size, "(a) RGB orthomosaic")
    scaleBar(a, corner, size)
    image = show(b, p1, corner, size, cmap="viridis", vmin=0, vmax=top)
    frame(b, corner, size, "(b) height, photogrammetry")
    heightBar(figure, b, image)
    if lidar is not None:
        image = show(c, lidar, corner, size, cmap="viridis", vmin=0, vmax=top)
        frame(c, corner, size, "(c) height, LiDAR")
        heightBar(figure, c, image)
    else:
        frame(c, corner, size, "(c) height, LiDAR")
        c.text(0.5, 0.5, "no LiDAR at this site", transform=c.transAxes, ha="center", color="0.4")
        c.set_facecolor("0.93")

    show(d, rgb, corner, size)
    outline(d, site, corner, size, width=0.9)
    shadeOutside(d, site, corner, size)
    d.plot(*line.xy, color="#ffd92f", lw=0.9, ls="--")
    frame(d, corner, size, "(d) annotation: one polygon per crown")

    show(e, rgb, corner, size, alpha=0.75)
    outline(e, site, corner, size, width=0.6)
    square = box(corner[0], corner[1], corner[0] + size, corner[1] + size)
    outline(e, site, corner, size, colour=MISSED, width=1.4,
            only=[s for s in outcome["missedShapes"] if s.intersects(square)])
    shadeOutside(e, site, corner, size)
    marks = []
    keep = inWindow(outcome["points"], corner, size)
    for kind, (colour, marker, _) in KIND.items():
        chosen = keep & (outcome["kinds"] == kind)
        for (px, py), bounds in zip(outcome["points"][chosen], np.array(outcome["boxes"], object)[chosen]):
            if bounds is not None:
                e.add_patch(Rectangle((bounds[0], bounds[1]), bounds[2] - bounds[0], bounds[3] - bounds[1],
                                      fill=False, edgecolor=colour, lw=0.8))
            if abs(py - line.coords[0][1]) <= 0.02 * size:
                marks.append((px, colour, marker))
        e.plot(outcome["points"][chosen, 0], outcome["points"][chosen, 1], marker, ls="", color=colour,
               ms=4.5, mec="white", mew=0.4)
    frame(e, corner, size, "(e) detections of %s, scored" % ("CC" if run.startswith("cc") else run))

    surfaces = [("photogrammetry", p1, "black", "-")] + ([("LiDAR", lidar, "#3a7dc9", "--")] if lidar is not None else [])
    profile(f, site, corner, size, line, surfaces, marks)
    f.set_title("(f) heights along the dashed line", loc="left")
    f.set_box_aspect(1)

    handles = [Line2D([], [], color=CROWN, lw=1.2, label="annotated crown", path_effects=None),
               Patch(facecolor="0.3", hatch="////", edgecolor="white", label="outside the scored area"),
               Line2D([], [], color=MISSED, lw=1.6, label="missed crown")]
    handles += [Line2D([], [], marker=m, ls="", color=c, mec="white", label=text) for c, m, text in KIND.values()]
    handles.append(Patch(facecolor="0.85", label="crown along the line"))
    handles.append(Line2D([], [], color="black", lw=1.1, label="height, photogrammetry"))
    if lidar is not None:
        handles.append(Line2D([], [], color="#3a7dc9", lw=1.1, ls="--", label="height, LiDAR"))
    legend = figure.legend(handles=handles, loc="lower center", ncol=4, frameon=True, facecolor="0.55",
                           edgecolor="none", labelcolor="white", bbox_to_anchor=(0.5, -0.005))
    legend.get_frame().set_alpha(1.0)
    figure.suptitle("%s: %s, a %g m window" % (LABEL.get(site.name, site.name), site.describe(), size),
                    fontsize=8.5, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0.13, 1, 0.97), w_pad=2.2, h_pad=1.8)
    save(figure, output, "fig_task")
    return corner


def figureForests(sites, size, output):
    n = len(sites)
    figure, axes = plt.subplots(3, n, figsize=(1.85 * n, 5.9), gridspec_kw={"height_ratios": [1, 1, 0.6]},
                                squeeze=False)
    anyLidar = False
    for column, site in enumerate(sites):
        corner = typicalWindow(site, size)
        line = profileLine(site, corner, size)
        rgb = rgbImage(site, corner, size)
        p1 = chmImage(site.p1, site, corner, size)
        lidar = chmImage(site.lidar, site, corner, size) if site.lidar else None
        anyLidar = anyLidar or lidar is not None
        a, b, c = axes[:, column]
        show(a, rgb, corner, size)
        frame(a, corner, size)
        a.set_title("%s\n%s" % (LABEL.get(site.name, site.name), site.describe()), fontsize=7.2)
        if column == 0:
            scaleBar(a, corner, size)
        image = show(b, p1, corner, size, cmap="viridis", vmin=0, vmax=max(float(np.percentile(p1, 99.5)), 1.0))
        outline(b, site, corner, size, width=0.6)
        shadeOutside(b, site, corner, size)
        b.plot(*line.xy, color="#ffd92f", lw=0.8, ls="--")
        frame(b, corner, size)
        heightBar(figure, b, image, below=True)
        surfaces = [("photogrammetry", p1, "black", "-")] + ([("LiDAR", lidar, "#3a7dc9", "--")] if lidar is not None else [])
        profile(c, site, corner, size, line, surfaces)
        if column:
            c.set_ylabel("")
    axes[0, 0].set_ylabel("RGB orthomosaic", fontsize=8)
    axes[1, 0].set_ylabel("height model (m), crowns", fontsize=8)
    handles = [Line2D([], [], color=CROWN, lw=1.2, label="annotated crown"),
               Line2D([], [], color="#ffd92f", lw=1.0, ls="--", label="profile line"),
               Patch(facecolor="0.85", label="crown along the line"),
               Line2D([], [], color="black", lw=1.1, label="height, photogrammetry")]
    if anyLidar:
        handles.append(Line2D([], [], color="#3a7dc9", lw=1.1, ls="--", label="height, LiDAR"))
    legend = figure.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=True, facecolor="0.55",
                           edgecolor="none", labelcolor="white", bbox_to_anchor=(0.5, 0.0))
    legend.get_frame().set_alpha(1.0)
    figure.tight_layout(rect=(0, 0.05, 1, 1), h_pad=1.1)
    save(figure, output, "fig_forests")


def save(figure, output, name):
    os.makedirs(output, exist_ok=True)
    for extension in ("pdf", "png"):
        figure.savefig(os.path.join(output, "%s.%s" % (name, extension)), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print("wrote %s" % os.path.join(output, name))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--taskSite", default="tereljFirst")
    parser.add_argument("--taskRun", default="ccP1", help="the run whose detections fig_task scores")
    parser.add_argument("--forests", nargs="+", default=["afcamoisan", "koiwai", "tereljFirst", "enshurinV2"])
    parser.add_argument("--window", type=float, default=30.0, help="window side in metres")
    parser.add_argument("--site", action="append", default=[], help="name=FOLDER, to add or replace a site")
    parser.add_argument("--output", default=os.path.join(DB, "paperFigures"))
    parser.add_argument("--only", choices=("task", "forests"), default=None)
    args = parser.parse_args(argv)
    folders = dict(SITES, **dict(s.split("=", 1) for s in args.site))
    if args.only != "forests":
        figureTask(Site(args.taskSite, folders[args.taskSite]), args.taskRun, args.window, args.output)
    if args.only != "task":
        figureForests([Site(n, folders[n]) for n in args.forests], args.window, args.output)


if __name__ == "__main__":
    main()
