"""
Figures for the results report: charts with matplotlib, map tiles through
TileRenderer. Every function writes one PNG and returns its path.

Example tiles are picked, not sampled: the tile where a method misses most, or
where its false positives concentrate, because a typical tile shows nothing
worth looking at. Captions say so, since a chosen worst case read as a typical
one would mislead.
"""

import os

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .render import MISSED, TileRenderer, pixelRings  # noqa: E402

PALETTE = ["#2b6ca3", "#d9822b", "#3a9a5b", "#b8434e", "#7a5ea8", "#8c8c8c"]
FATE_COLOURS = {"hit": (0, 190, 0), "repeat": (0, 150, 255),
                "canopy": (255, 90, 0), "low": (0, 0, 230)}
# OpenCV colours are BGR
GROUP_COLOURS = {"both": (0, 190, 0), "first": (0, 140, 255),
                 "second": (200, 0, 200), "neither": (0, 255, 255)}


def _save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


# ---------------------------------------------------------------------- #
# charts
# ---------------------------------------------------------------------- #

def cvBars(rows, path):
    """Recall, precision and F1 per method, F1 with its per-fold spread."""
    figure, axis = plt.subplots(figsize=(8.5, 3.6))
    positions = np.arange(len(rows))
    width = 0.26
    for offset, (key, label) in enumerate((("recall", "recall"),
                                           ("precision", "precision"),
                                           ("f1", "F1"))):
        values = [row[key] for row in rows]
        errors = [row["std"] if key == "f1" else 0 for row in rows]
        axis.bar(positions + (offset - 1) * width, values, width,
                 yerr=errors, capsize=3, label=label, color=PALETTE[offset])
    axis.set_xticks(positions)
    axis.set_xticklabels([row["short"] for row in rows], fontsize=8)
    axis.set_ylim(0.7, 1.0)
    axis.set_ylabel("score")
    axis.legend(fontsize=8, ncol=3, loc="upper left")
    axis.grid(axis="y", alpha=0.3)
    return _save(figure, path)


def perFold(table, path):
    """F1 on each held-out block, one line per method."""
    figure, axis = plt.subplots(figsize=(8.5, 3.4))
    for index, (label, values) in enumerate(table["series"].items()):
        axis.plot(table["blocks"], values, marker="o", label=label,
                  color=PALETTE[index % len(PALETTE)])
    axis.set_xlabel("held-out block")
    axis.set_ylabel("F1")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    return _save(figure, path)


def sweepScatter(rows, path, highlight):
    """Every swept setting as a recall/precision point."""
    figure, axis = plt.subplots(figsize=(5.6, 4.2))
    recall = [r["recall"] for r in rows]
    precision = [r["precision"] for r in rows]
    axis.scatter(recall, precision, s=12, alpha=0.55, color=PALETTE[0])
    for label, row, colour in highlight:
        axis.scatter([row["recall"]], [row["precision"]], s=70,
                     color=colour, edgecolor="black", label=label, zorder=3)
    for f1 in (0.80, 0.85, 0.90):
        r = np.linspace(f1 / (2 - f1) + 1e-3, 1.0, 100)
        axis.plot(r, f1 * r / (2 * r - f1), ":", color="grey", linewidth=0.8)
        if 0.7 <= f1 / (2 - f1) <= 0.95:
            axis.text(r[-1], f1 / (2 - f1), " F1 %.2f" % f1, fontsize=7,
                      color="grey", va="center")
    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_xlim(0.8, 1.0)
    axis.set_ylim(0.7, 0.95)
    axis.legend(fontsize=7, loc="lower left")
    axis.grid(alpha=0.3)
    return _save(figure, path)


def missedHistograms(table, name, label, path):
    """Area and height of the crowns a method found versus missed."""
    found = table["found_" + name]
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))
    for axis, key, unit, bins in ((axes[0], "area", "crown area (m²)",
                                   np.arange(0, 42, 2)),
                                  (axes[1], "peak", "crown height (m)",
                                   np.arange(4, 26, 1))):
        axis.hist(table[key][found], bins=bins, density=True, alpha=0.6,
                  label="found", color=PALETTE[0])
        axis.hist(table[key][~found], bins=bins, density=True, alpha=0.6,
                  label="missed", color=PALETTE[3])
        axis.set_xlabel(unit)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("share of crowns")
    figure.suptitle(label, fontsize=9)
    return _save(figure, path)


def errorStack(breakdowns, path):
    """What every detection turned out to be, one bar per method."""
    keys = [("hits", "hit"), ("repeats", "repeat, same crown"),
            ("canopyLevel", "outside, canopy height"),
            ("belowCanopy", "outside, below canopy"),
            ("noNeighbour", "outside, no neighbours")]
    figure, axis = plt.subplots(figsize=(8.5, 0.6 + 0.55 * len(breakdowns)))
    labels = list(breakdowns)
    left = np.zeros(len(labels))
    for index, (key, text) in enumerate(keys):
        shares = np.array([breakdowns[m][key] / float(
            breakdowns[m]["detections"]) for m in labels])
        axis.barh(labels, shares, left=left, label=text,
                  color=PALETTE[index])
        left += shares
    axis.set_xlim(0.7, 1.0)
    axis.set_xlabel("share of detections (axis starts at 0.7)")
    axis.legend(fontsize=7, ncol=3, loc="upper center",
                bbox_to_anchor=(0.5, -0.35))
    return _save(figure, path)


def agreementBars(agreement, first, second, path):
    """How many crowns each group holds, with their median size."""
    labels = list(agreement)
    counts = [agreement[k]["count"] for k in labels]
    areas = [agreement[k]["area"] or 0 for k in labels]
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 2.8))
    axes[0].bar(range(len(labels)), counts, color=PALETTE[:len(labels)])
    axes[0].set_ylabel("crowns")
    axes[1].bar(range(len(labels)), areas, color=PALETTE[:len(labels)])
    axes[1].set_ylabel("median crown area (m²)")
    for axis in axes:
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels([l.replace("only ", "only\n") for l in labels],
                             fontsize=7)
    figure.suptitle("%s versus %s" % (first, second), fontsize=9)
    return _save(figure, path)


def fusionBars(summary, title, path):
    names = sorted(summary, key=lambda n: -summary[n]["pooled"]["f1"])
    figure, axis = plt.subplots(figsize=(8.5, 3.0))
    positions = np.arange(len(names))
    for offset, key in enumerate(("recall", "precision", "f1")):
        axis.bar(positions + (offset - 1) * 0.26,
                 [summary[n]["pooled"][key] for n in names], 0.26,
                 label=key if key != "f1" else "F1", color=PALETTE[offset])
    axis.set_xticks(positions)
    axis.set_xticklabels(names, fontsize=8)
    axis.set_ylim(0.75, 1.0)
    axis.legend(fontsize=8, ncol=3)
    axis.set_title(title, fontsize=9)
    axis.grid(axis="y", alpha=0.3)
    return _save(figure, path)


# ---------------------------------------------------------------------- #
# map tiles
# ---------------------------------------------------------------------- #

def _crop(renderer, tile):
    c0, r0, c1, r1 = tile
    image, scale = renderer.background(c0, r0, c1, r1)
    return image, scale


def _dot(image, scene, x, y, tile, scale, colour, radius):
    c0, r0, _, _ = tile
    column, row = scene.toPixel(x, y)
    cv2.circle(image, (int((column - c0) * scale), int((row - r0) * scale)),
               radius, colour, -1)


def worstTile(renderer, scene, xs, ys, weight=None):
    """The tile holding the most of some points (misses, false positives)."""
    best, bestCount = None, -1
    for tile in renderer.tiles():
        c0, r0, c1, r1 = tile
        count = 0
        for index, (x, y) in enumerate(zip(xs, ys)):
            column, row = scene.toPixel(x, y)
            if c0 <= column < c1 and r0 <= row < r1:
                count += 1 if weight is None else weight[index]
        if count > bestCount:
            best, bestCount = tile, count
    return best


def fateTile(scene, comparison, name, path, tile=None):
    """One method's detections coloured by fate, missed crowns outlined."""
    renderer = TileRenderer(scene, tileM=30.0, minSidePx=900)
    result = comparison.results[name]
    missed = [i for i in range(len(scene.crowns)) if i not in result["found"]]
    if tile is None:
        centroids = scene.crowns.geometry.centroid
        tile = worstTile(renderer, scene, centroids.x.to_numpy()[missed],
                         centroids.y.to_numpy()[missed])
    image, scale = _crop(renderer, tile)
    renderer.drawCrowns(image, missed, tile[0], tile[1], scale, MISSED)

    fates = {i: "canopy" for i in result["split"]["canopy"]}
    fates.update({i: "low" for i in result["split"]["low"]})
    radius = max(4, int(0.45 / scene.pixelSize * scale))
    for index, kind in enumerate(result["kinds"]):
        fate = kind if kind in ("hit", "repeat") else fates.get(index)
        if fate:
            _dot(image, scene, result["xs"][index], result["ys"][index],
                 tile, scale, FATE_COLOURS[fate], radius)
    renderer.caption(image, "green hit | orange repeat | blue outside at "
                     "canopy height | red outside, low | yellow missed crown")
    cv2.imwrite(path, image)
    return path, tile


def agreementTile(scene, comparison, first, second, path, labels=None):
    """Crowns coloured by which of two methods found them."""
    renderer = TileRenderer(scene, tileM=30.0, minSidePx=900)
    a, b = comparison.results[first]["found"], \
        comparison.results[second]["found"]
    disputed = sorted((a ^ b) | (set(range(len(scene.crowns))) - a - b))
    centroids = scene.crowns.geometry.centroid
    tile = worstTile(renderer, scene, centroids.x.to_numpy()[disputed],
                     centroids.y.to_numpy()[disputed])
    image, scale = _crop(renderer, tile)
    everything = set(range(len(scene.crowns)))
    for group, members in (("both", a & b), ("first", a - b),
                           ("second", b - a), ("neither", everything - a - b)):
        renderer.drawCrowns(image, sorted(members), tile[0], tile[1], scale,
                            GROUP_COLOURS[group])
    names = labels or (first, second)
    renderer.caption(image, "green both | orange only %s | magenta only %s | "
                     "yellow neither" % names)
    cv2.imwrite(path, image)
    return path


def shapeTile(scene, comparison, name, polygons, path, tile=None):
    """Real crowns against a method's own outlines or pseudo-crowns."""
    renderer = TileRenderer(scene, tileM=30.0, minSidePx=900)
    if tile is None:
        tile = next(iter(renderer.tiles()))
    image, scale = _crop(renderer, tile)
    renderer.drawCrowns(image, range(len(scene.crowns)), tile[0], tile[1],
                        scale, (255, 255, 0))
    inverse = ~scene.transform
    for shape in polygons:
        if shape is None or shape.is_empty:
            continue
        cv2.polylines(image, pixelRings(shape, inverse, tile[0], tile[1],
                                        scale),
                      True, (255, 0, 255), max(1, int(scale // 3)))
    renderer.caption(image, "cyan real crowns | magenta %s" % name)
    cv2.imwrite(path, image)
    return path


def ensureDirectory(path):
    os.makedirs(path, exist_ok=True)
    return path
