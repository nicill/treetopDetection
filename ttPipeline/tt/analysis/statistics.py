"""
Statistics shared by every analysis: medians and quantiles of crown or top
measures, nearest-neighbour spacing, the split of false positives by height,
and the paired comparison of two runs over blocks.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon


def quantiles(values, levels=(10, 50, 90)):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return [float(q) for q in np.percentile(values, levels)] if len(values) \
        else []


def nearestDistances(xs, ys):
    """Each point's distance to its nearest other point (inf if alone)."""
    points = np.column_stack([xs, ys])
    if len(points) < 2:
        return np.full(len(points), np.inf)
    return cKDTree(points).query(points, k=2)[0][:, 1]


def profile(mask, area, peak, nearest):
    """Count and medians of area, height and nearest-neighbour distance."""
    if not mask.any():
        return {"count": 0, "area": None, "peak": None, "nearest": None}
    return {"count": int(mask.sum()),
            "area": float(np.median(area[mask])),
            "peak": float(np.median(peak[mask])),
            "nearest": float(np.median(nearest[mask])),
            "areaQuartiles": np.percentile(area[mask], [25, 75]).tolist(),
            "peakQuartiles": np.percentile(peak[mask], [25, 75]).tolist()}


def splitBackground(xs, ys, heights, kinds, neighbourhood=12.0,
                    canopyFraction=0.7):
    """
    Split background detections into canopy-level, low and unknown.

    A detection outside every crown but standing at a height consistent with
    the confirmed trees around it is most likely a tree nobody drew. One well
    below the local canopy is scrub, a branch or ground. The reference is the
    median height of hits within `neighbourhood` metres, so a short tree in a
    short stand is not penalised for being short.
    """
    positions = np.column_stack([xs, ys])
    heights = np.asarray(heights, float)
    hits = [i for i, k in enumerate(kinds) if k == "hit"]
    background = [i for i, k in enumerate(kinds) if k == "background"]
    split = {"canopy": [], "low": [], "unknown": []}
    if not hits:
        split["unknown"] = background
        return split
    tree = cKDTree(positions[hits])
    for index in background:
        neighbours = tree.query_ball_point(positions[index], neighbourhood)
        if not neighbours:
            split["unknown"].append(index)
            continue
        reference = np.median(heights[[hits[j] for j in neighbours]])
        split["canopy" if heights[index] >= canopyFraction * reference
              else "low"].append(index)
    return split


# ---------------------------------------------------------------------- #
# two runs over the same blocks
# ---------------------------------------------------------------------- #

def blockKey(name):
    """A block's position, b<row><col>: datasets of one site share the blocks
    but may carry different names (enshurin_b03, rgb_b03)."""
    return name.rsplit("_", 1)[-1]


def counts(fold):
    """(hits, misses, false detections) of a block, from its scores."""
    if all(k in fold for k in ("tp", "fn", "fp")):
        return float(fold["tp"]), float(fold["fn"]), float(fold["fp"])
    crowns, recall, precision = fold["crowns"], fold["recall"], \
        fold["precision"]
    hits = recall * crowns
    detections = hits / precision if precision > 0 else \
        float(fold.get("detections", 0))
    return hits, crowns - hits, max(detections - hits, 0.0)


def pooledF1(table):
    tp, fn, fp = table.sum(axis=0)
    return 2 * tp / max(2 * tp + fn + fp, 1e-9)


def compareFolds(reference, other, bootstrap, rng):
    """
    Two runs' {block: fold}: the pooled difference (every tree counts once)
    with a bootstrap interval over blocks, and the paired Wilcoxon over the
    blocks' F1 (every block counts once), with the blocks the reference wins.
    """
    blocks = sorted(set(reference) & set(other))
    a = np.array([counts(reference[b]) for b in blocks])
    b = np.array([counts(other[b]) for b in blocks])
    difference = pooledF1(a) - pooledF1(b)
    picks = rng.integers(0, len(blocks), size=(bootstrap, len(blocks)))
    resampled = np.array([pooledF1(a[p]) - pooledF1(b[p]) for p in picks])
    perBlock = np.array([reference[k]["f1"] - other[k]["f1"] for k in blocks])
    p = float(wilcoxon(perBlock).pvalue) if np.any(perBlock) else 1.0
    return {"blocks": len(blocks), "pooledReference": round(pooledF1(a), 4),
            "pooledOther": round(pooledF1(b), 4),
            "pooledDifference": round(float(difference), 4),
            "interval95": [round(float(v), 4) for v in
                           np.percentile(resampled, [2.5, 97.5])],
            "meanBlockDifference": round(float(perBlock.mean()), 4),
            "referenceWins": int((perBlock > 0).sum()), "p": p}


def pairedTest(first, second, key):
    """(mean difference second - first, p, blocks) over the shared blocks."""
    shared = sorted(set(first) & set(second))
    difference = np.array([second[b][key] - first[b][key] for b in shared])
    if len(shared) < 2:
        return float("nan"), float("nan"), len(shared)
    p = float(wilcoxon(difference).pvalue) if np.any(difference) else 1.0
    return float(difference.mean()), p, len(shared)


def holm(pvalues):
    """Holm-adjusted p-values, in the given order."""
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted
