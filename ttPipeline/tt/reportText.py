"""
The sections of the results report, in order. Each takes the ResultsReport and
writes what its evidence supports; a section whose evidence is missing says
which run would supply it rather than being left out silently.
"""

import numpy as np

from . import reportFigures as figures

LABELS = {
    "ccLidar": ("Connected components, LiDAR CHM", "CC LiDAR"),
    "ccP1": ("Connected components, P1 CHM", "CC P1"),
    "mrcnnLidar": ("Mask R-CNN, LiDAR CHM", "MRCNN LiDAR"),
    "mrcnnP1": ("Mask R-CNN, P1 CHM", "MRCNN P1"),
    "mrcnnRgb": ("Mask R-CNN, RGB mosaic", "MRCNN RGB"),
    "mrcnnRgbLidar": ("Mask R-CNN, RGB + LiDAR", "MRCNN RGB+LiDAR"),
    "ccLidarOld": ("Connected components, LiDAR, earlier protocol",
                   "CC LiDAR (old)"),
    "ccP1Old": ("Connected components, P1, earlier protocol", "CC P1 (old)"),
}
CURRENT = ("ccLidar", "ccP1", "mrcnnLidar", "mrcnnP1", "mrcnnRgb",
           "mrcnnRgbLidar")


def label(name, short=False):
    return LABELS.get(name, (name, name))[1 if short else 0]


def isPointMethod(name):
    return name.startswith("cc")


def fmt(value, digits=3):
    return ("%%.%df" % digits) % value if value is not None else "-"


def pct(value):
    return "%.1f%%" % (100 * value)


def current(r):
    return [n for n in CURRENT if n in r.e.cv]


def pending(r, what):
    r.doc.paragraph("**Not yet available: %s.** This needs the learned "
                    "models' predictions, which runs made before prediction "
                    "saving was added do not contain. The commands in the "
                    "last section re-run them, and re-running this report "
                    "afterwards fills this part in." % what)


# ---------------------------------------------------------------------- #

def summary(r):
    rows = r.e.cvRows(current(r))
    best = rows[0]
    topRecall = max(row["recall"] for row in rows)
    recallLeaders = [row["label"] for row in rows
                     if topRecall - row["recall"] < 0.0005]
    r.doc.heading("Summary")
    r.doc.paragraph(
        "Four detectors were compared under the same leave-one-block-out "
        "cross-validation on %d annotated crowns: the connected-component "
        "treetop detector on the LiDAR and on the photogrammetric (P1) canopy "
        "height model, and Mask R-CNN on the LiDAR CHM and on the RGB mosaic. "
        "The best F1 was %s, for %s; the highest recall was %s, for %s. The "
        "whole field spans %s F1 points, while a single method's F1 varies "
        "by %s from one block to the next."
        % (len(r.e.scene.crowns), fmt(best["f1"]), best["label"],
           pct(topRecall), " and ".join(recallLeaders),
           fmt(100 * (rows[0]["f1"] - rows[-1]["f1"]), 1),
           "%.2f-%.2f" % (min(x["std"] for x in rows),
                          max(x["std"] for x in rows))))
    crossPairs = [p for p in r.e.paired(current(r))
                  if p[0].startswith("CC") and p[1].startswith("MRCNN")]
    if crossPairs:
        wins = ["%d/%d" % (p[3], p[4]) for p in crossPairs]
        pValues = [p[5] for p in crossPairs]
        r.doc.paragraph(
            "Compared block by block, the connected-component detector beats "
            "Mask R-CNN on most blocks (%s of the paired comparisons), by "
            "%.1f-%.1f F1 points on average, but none of these differences is "
            "significant (p = %.2f-%.2f over eight blocks). **On this site no "
            "method is reliably better at F1;** the point detector is at "
            "least as good as Mask R-CNN, not measurably better."
            % (", ".join(wins),
               100 * min(p[2] for p in crossPairs),
               100 * max(p[2] for p in crossPairs),
               min(pValues), max(pValues)))
    boxLevel = r.e.comparison is not None and any(
        not isPointMethod(n) for n in r.e.comparison.results)
    r.doc.paragraph(
        "What does separate them is cost and the kind of error. The "
        "connected-component detector needs no training data and runs in "
        "seconds; Mask R-CNN needs annotated crowns from the same kind of "
        "stand and about an hour per cross-validation on a GPU.")
    if r.e.comparison is not None:
        r.doc.paragraph(
            "For the connected-component detector the tree-level analysis is "
            "clear: its lost precision is split crowns and canopy-level "
            "detections outside any drawn polygon, which are likely "
            "unannotated trees, far more than detections on the ground; and "
            "the crowns it misses are small crowns in crowded canopy.%s"
            % ("" if boxLevel else
               " The same analysis for Mask R-CNN needs its predictions, "
               "which the runs so far did not save. Its logs give only the "
               "totals: %s." % mrcnnErrorTotals(r)))


def mrcnnErrorTotals(r):
    parts = []
    for name in ("mrcnnLidar", "mrcnnRgb"):
        if name not in r.e.cv:
            continue
        folds = r.e.cv[name]["folds"]
        total = float(sum(f.get("predictions", 0) for f in folds) or 1)
        parts.append("on %s, %s of detections are repeats inside a found "
                     "crown and %s fall outside every crown"
                     % (label(name, True).replace("MRCNN ", ""),
                        pct(sum(f.get("repeats", 0) for f in folds) / total),
                        pct(sum(f.get("falsePositives", 0) for f in folds)
                            / total)))
    return "; ".join(parts) or "none recorded"


def data(r):
    crowns = r.e.scene.crowns
    r.doc.heading("Data and evaluation protocol")
    r.doc.paragraph(
        "The site covers %.1f ha near Ulaanbaatar, with %d hand-drawn crowns "
        "(median area %.1f m²). Three rasters cover it: a LiDAR CHM (2025), a "
        "CHM derived photogrammetrically from the DJI Zenmuse P1 imagery "
        "(2026), and the P1 RGB orthomosaic. Alignment between all layers was "
        "measured by normalised cross-correlation at about 0.1 m, so no "
        "correction was applied."
        % (r.e.scene.boundary.area / 1e4, len(crowns),
           crowns.geometry.area.median()))
    r.doc.paragraph(
        "The area is split into %d spatial blocks of about 63 x 64 m. Each "
        "fold holds one block out, fits on the rest, and scores only the held-"
        "out block, so no method is ever scored on crowns it was fitted or "
        "tuned on. The learned models also set aside one training block as a "
        "validation block, used to pick the score threshold and never for "
        "fitting; the connected-component detector tunes its parameters on "
        "all seven training blocks." % len(r.e.blocks))
    r.doc.paragraph(
        "Scoring is at the crown level. Each detection is reduced to a point; "
        "a point inside several overlapping crowns goes to the smallest; the "
        "highest-scoring point in a crown is a hit and any others are "
        "repeats; a point in no crown is a false positive. Recall is hits "
        "over crowns, precision hits over detections. A crown counts in the "
        "fold whose block holds its centroid, so every tree is scored exactly "
        "once. The same function scores every method.")


def algorithms(r):
    r.doc.heading("How each algorithm works")
    r.doc.heading("Connected-component treetop detection", 2)
    r.doc.paragraph(
        "A 40 m window slides over the CHM. In each window heights are "
        "stretched between the window's own lower percentile and its 99th, "
        "then eroded, which removes the low connecting canopy so that crowns "
        "begin to separate. Each connected blob large enough to hold a tree "
        "is then walked down in height steps: at every step it is thresholded "
        "and relabelled, and any sub-blob appearing for the first time "
        "contributes its highest pixel as a new top. The descent is what does "
        "the work, since crowns are separate near their tops and fused near "
        "the ground, so the level at which two trees split is found rather "
        "than assumed.")
    r.doc.paragraph(
        "Tops from all windows are then merged. Two tops are the same tree if "
        "they are within a search radius and the canopy between them never "
        "dips more than a set depth below the lower of them — the saddle "
        "rule. Measured on this site, tops within one crown sit a median "
        "0.25 m apart with no dip between them, while tops in neighbouring "
        "crowns sit 5.8 m apart with a median dip of 6.6 m, so the depth "
        "separates the two cases far better than distance does. With the "
        "detector settings held fixed, replacing the original distance-only "
        "merge by the saddle rule raised F1 on the LiDAR CHM from 0.794 to "
        "0.868 (tuned and scored on the whole area).")
    r.doc.heading("Mask R-CNN", 2)
    r.doc.paragraph(
        "torchvision's Mask R-CNN with a ResNet-50 FPN v2 backbone, "
        "pretrained on COCO, with its heads replaced for a single tree class. "
        "For a one-channel CHM the first convolution's three RGB filters are "
        "summed; the RGB mosaic uses the pretrained stem as it is. Tiles are "
        "512 px at 0.05 m with 50% overlap, trained for 40 epochs with AdamW "
        "and a one-cycle schedule, augmented only by flips and 90-degree "
        "rotations, the transformations a nadir canopy image is genuinely "
        "invariant to. Predictions from overlapping tiles are merged by non-"
        "maximum suppression in world coordinates, and the score threshold is "
        "chosen per fold on the validation block.")
    r.doc.heading("Pseudo-crowns", 2)
    r.doc.paragraph(
        "The point detector returns tops, not crowns. Pseudo-crowns turn tops "
        "into outlines: the canopy mask, thresholded at the local 20th "
        "percentile, is cut along the Voronoi edges of the tops, and each top "
        "takes its piece. Where the canopy is crowded the bounding box is "
        "shrunk toward the largest square the distance transform guarantees "
        "is inside the canopy.")


def tests(r):
    r.doc.heading("Tests performed")
    lines = []
    for name in r.e.sweeps:
        lines.append("a sweep of %d detector and merge settings for %s, "
                     "scored on the whole area"
                     % (len(r.e.sweeps[name]), label(name)))
    lines.append("leave-one-block-out cross-validation of %s"
                 % ", ".join(label(n) for n in current(r)))
    r.doc.paragraph("The evidence is: %s." % "; ".join(lines))
    r.doc.paragraph(
        "The sweeps tune and score on the same crowns, so they show which "
        "settings to try and how far recall can be pushed, but their F1 "
        "values are optimistic. Every comparison between methods in this "
        "report uses the cross-validated figures.")
    if "ccLidarOld" in r.e.cv:
        old, new = r.e.cv["ccLidarOld"]["pooled"], r.e.cv["ccLidar"]["pooled"]
        r.doc.paragraph(
            "The connected-component cross-validation was corrected during "
            "this work. The first version tuned each fold by running the "
            "detector on the training blocks with the held-out block zeroed "
            "out, which put an artificial cliff at every block edge and "
            "damaged trees along it. The corrected version runs each setting "
            "once over the whole area and scores each fold from those "
            "detections; the held-out crowns are still never consulted during "
            "tuning. On the LiDAR CHM F1 moved from %s to %s. The earlier "
            "figures are shown for reference only."
            % (fmt(old["f1"]), fmt(new["f1"])))


def best(r):
    r.doc.heading("1. The best result for each algorithm")
    rows = r.e.cvRows(current(r))
    r.doc.table(["method", "recall", "precision", "F1", "per-block F1 sd"],
                [[row["label"], pct(row["recall"]), pct(row["precision"]),
                  fmt(row["f1"]), fmt(row["std"])] for row in rows],
                caption="Cross-validated results, held-out blocks only.",
                widthsCm=[6.5, 2.2, 2.2, 2.0, 3.1])
    r.doc.figure(figures.cvBars(rows, r.figurePath("cvBars")),
                 "Cross-validated recall, precision and F1. Error bars on F1 "
                 "are the standard deviation across the eight blocks.")
    sweepPoints(r)


def sweepPoints(r):
    if not r.e.sweeps:
        return
    rows = []
    for name in r.e.sweeps:
        bestRow, highRecall, ceiling = r.e.sweepPoints(name)
        for what, row in (("best F1", bestRow),
                          ("best precision at recall >= 90%", highRecall),
                          ("highest recall", ceiling)):
            if row:
                rows.append([label(name, True), what, settingText(row),
                             pct(row["recall"]), pct(row["precision"]),
                             fmt(row["f1"])])
    costs = []
    for name in r.e.sweeps:
        _, highRecall, ceiling = r.e.sweepPoints(name)
        if highRecall:
            extra = (ceiling["recall"] - highRecall["recall"]) * \
                len(r.e.scene.crowns)
            added = ceiling["detections"] - highRecall["detections"]
            costs.append("on %s, going from %s to the sweep's highest recall "
                         "of %s finds %d more crowns for %d more detections, "
                         "and precision falls from %s to %s"
                         % (label(name, True), pct(highRecall["recall"]),
                            pct(ceiling["recall"]), round(extra), added,
                            pct(highRecall["precision"]),
                            pct(ceiling["precision"])))
    r.doc.paragraph(
        "For a high-recall operating point the sweeps are more informative "
        "than the cross-validation, which optimises F1. Table %d gives three "
        "points per CHM. Recall can be pushed well past 90%%, but the last "
        "crowns are expensive: %s. An earlier, wider sweep on the LiDAR CHM "
        "that included no erosion at all reached 98.2%% recall, at 35.4%% "
        "precision." % (r.doc.nextTable(), "; ".join(costs)))
    r.doc.table(["CHM", "point", "settings", "recall", "precision", "F1"],
                rows, caption="Operating points from the sweeps (tuned and "
                "scored on the whole area, so optimistic).",
                widthsCm=[1.8, 3.4, 5.6, 1.7, 1.9, 1.4])
    for name in r.e.sweeps:
        bestRow, highRecall, _ = r.e.sweepPoints(name)
        highlight = [("best F1", bestRow, "#d9822b")]
        if highRecall:
            highlight.append(("best at recall >= 90%", highRecall, "#b8434e"))
        r.doc.figure(figures.sweepScatter(r.e.sweeps[name],
                                          r.figurePath("sweep_" + name),
                                          highlight),
                     "Every swept setting for %s as a recall/precision point. "
                     "Dotted lines are constant F1." % label(name), 11)


def settingText(row):
    return "pct %s, minTop %s, step %s, erode %s, drop %s" % (
        row.get("lowerPercentile"), row.get("minTopAreaM2"),
        row.get("topStepM"), row.get("erosionIterations"),
        row.get("saddleDropM"))


def paired(r):
    names = current(r)
    r.doc.heading("Are the differences real?", 2)
    r.doc.paragraph(
        "Blocks differ in difficulty more than methods differ from each "
        "other, so pooled F1 is a weak test. Pairing by block compares "
        "methods on the same trees: Table %d gives, for each pair, the mean "
        "difference in F1 across blocks, how many of the eight blocks the "
        "first method won, and a Wilcoxon signed-rank p-value. With eight "
        "blocks the smallest attainable p-value is 0.008, and a real "
        "difference of one F1 point would need many more to show."
        % r.doc.nextTable())
    rows = [[a, b, "%+.3f" % d, "%d/%d" % (w, n), fmt(p, 2)]
            for a, b, d, w, n, p in r.e.paired(names)]
    r.doc.table(["first", "second", "mean F1 difference", "blocks won",
                 "p"], rows, caption="Paired comparison over held-out "
                 "blocks.", widthsCm=[3.6, 3.6, 3.4, 2.6, 2.4])
    table = {"blocks": sorted(r.e.foldF1(names[0])),
             "series": {label(n, True): [r.e.foldF1(n).get(b) for b in
                                         sorted(r.e.foldF1(names[0]))]
                        for n in names}}
    r.doc.figure(figures.perFold(table, r.figurePath("perFold")),
                 "F1 on each held-out block. The lines cross repeatedly: "
                 "which method wins depends on the block.")
    mrcnnSources(r)


def mrcnnSources(r):
    if not {"mrcnnLidar", "mrcnnRgb"} <= set(r.e.cv):
        return
    lidar, rgb = r.e.cv["mrcnnLidar"], r.e.cv["mrcnnRgb"]
    diff = [x for x in r.e.paired(["mrcnnLidar", "mrcnnRgb"])][0]
    r.doc.heading("Mask R-CNN: CHM or RGB mosaic?", 2)
    counts = {}
    for name, data in (("LiDAR", lidar), ("RGB", rgb)):
        folds = data["folds"]
        total = sum(f.get("predictions", 0) for f in folds) or 1
        counts[name] = (sum(f.get("repeats", 0) for f in folds) / total,
                        sum(f.get("falsePositives", 0) for f in folds) / total)
    r.doc.paragraph(
        "On the LiDAR CHM Mask R-CNN reached F1 %s (recall %s, precision %s); "
        "on the RGB mosaic %s (recall %s, precision %s). The CHM version won "
        "%d of %d blocks, by %+.3f on average (p = %.2f). The RGB model finds "
        "slightly more trees but splits more of them: %s of its detections "
        "are repeats inside an already-found crown against %s for the CHM "
        "model, while false positives outside crowns are about the same "
        "(%s and %s). Height separates touching crowns at their saddle, which "
        "colour does not, so this is the error one would expect RGB to make."
        % (fmt(lidar["pooled"]["f1"]), pct(lidar["pooled"]["recall"]),
           pct(lidar["pooled"]["precision"]), fmt(rgb["pooled"]["f1"]),
           pct(rgb["pooled"]["recall"]), pct(rgb["pooled"]["precision"]),
           diff[3], diff[4], diff[2], diff[5], pct(counts["RGB"][0]),
           pct(counts["LiDAR"][0]), pct(counts["LiDAR"][1]),
           pct(counts["RGB"][1])))
    thresholds = sorted({f.get("threshold") for f in lidar["folds"]
                         if f.get("threshold") is not None})
    if thresholds:
        r.doc.paragraph(
            "The score threshold chosen on the validation block ranged from "
            "%.2f to %.2f across folds. A single validation block of 10-25 "
            "tiles is too little to fix the threshold reliably, and some of "
            "the fold-to-fold variation in Mask R-CNN's precision comes from "
            "that rather than from the model."
            % (min(thresholds), max(thresholds)))


# ---------------------------------------------------------------------- #

def missed(r):
    r.doc.heading("2. Where are the missed trees?")
    c = r.e.comparison
    if c is None:
        pending(r, "the tree-by-tree analysis")
        return
    rows = []
    for name in c.results:
        m = c.missedProfile(name)
        rows.append([label(name, True), str(m["missed"]),
                     "%.1f / %.1f" % (m["missedCrowns"]["area"],
                                      m["found"]["area"]),
                     "%.1f / %.1f" % (m["missedCrowns"]["peak"],
                                      m["found"]["peak"]),
                     "%.2f / %.2f" % (m["missedCrowns"]["nearest"],
                                      m["found"]["nearest"])])
    r.doc.paragraph(
        "Table %d compares the crowns each method missed with those it found, "
        "on the held-out predictions stitched across all folds. Missed crowns "
        "are consistently smaller, and slightly lower and more crowded. Size "
        "dominates: the missed crowns' median area is little more than half "
        "that of the found ones, while their median height is only one to "
        "two metres lower." % r.doc.nextTable())
    r.doc.table(["method", "missed", "median area m² (missed / found)",
                 "median height m", "nearest crown m"], rows,
                caption="Missed versus found crowns.",
                widthsCm=[2.8, 1.6, 4.4, 3.6, 3.6])
    table = c.crownTable()
    first = sorted(c.results, key=lambda n: c.missedProfile(n)["missed"])[0]
    r.doc.figure(figures.missedHistograms(table, first, label(first),
                                          r.figurePath("missed_" + first)),
                 "Area and height of found and missed crowns, %s. The missed "
                 "crowns sit at the small end of the size distribution but "
                 "across most of the height range." % label(first))
    path, _ = figures.fateTile(r.e.scene, c, first,
                               r.figurePath("missedTile_" + first))
    r.doc.figure(path, "The tile where %s misses most, chosen as a worst case "
                 "rather than a typical one. Yellow outlines are missed "
                 "crowns: small crowns pressed against larger neighbours."
                 % label(first), 12)
    if not any(not isPointMethod(n) for n in c.results):
        pending(r, "the same breakdown for Mask R-CNN")


def precision(r):
    r.doc.heading("3. Where does the precision go?")
    c = r.e.comparison
    if c is None:
        pending(r, "the error breakdown")
        return
    breakdowns = {label(n, True): c.errorBreakdown(n) for n in c.results}
    rows = []
    for name, b in breakdowns.items():
        total = float(b["detections"])
        rows.append([name, str(b["detections"]), pct(b["hits"] / total),
                     pct(b["repeats"] / total), pct(b["canopyLevel"] / total),
                     pct(b["belowCanopy"] / total)])
    r.doc.paragraph(
        "A detection that is not a hit is one of three things, and they need "
        "different fixes. A repeat is a second detection inside a crown that "
        "already has one: a crown split in two. A detection outside every "
        "crown but at the height of the confirmed trees around it (at least "
        "70% of the median hit height within 12 m) is most likely a tree "
        "nobody drew. One well below the local canopy is scrub, a branch or "
        "the ground: the genuine false positive.")
    r.doc.table(["method", "detections", "hits", "repeats",
                 "outside, canopy height", "outside, low"], rows,
                caption="What every held-out detection turned out to be.",
                widthsCm=[2.8, 2.0, 1.9, 2.0, 3.8, 3.3])
    lowest = max(b["belowCanopy"] / float(b["detections"])
                 for b in breakdowns.values())
    r.doc.paragraph(
        "Genuine false positives are at most %s of detections for any "
        "method. The rest of the lost precision is split crowns and "
        "canopy-level detections, and if those canopy-level detections are "
        "trees missing from the annotation, precision as scored understates "
        "the real figure by several points." % pct(lowest))
    r.doc.figure(figures.errorStack(breakdowns,
                                    r.figurePath("errorStack")),
                 "Detection fates by method.")
    name = next(iter(c.results))
    xs = c.results[name]["xs"]
    ys = c.results[name]["ys"]
    wrong = [i for i, k in enumerate(c.results[name]["kinds"]) if k != "hit"]
    tile = figures.worstTile(figures.TileRenderer(r.e.scene, tileM=30.0),
                             r.e.scene, xs[wrong], ys[wrong])
    path, _ = figures.fateTile(r.e.scene, c, name,
                               r.figurePath("fateTile_" + name), tile=tile)
    r.doc.figure(path, "The tile where %s's non-hits concentrate. Blue "
                 "detections sit on bright crowns with no polygon: most look "
                 "like real, unannotated trees." % label(name), 12)


def sources(r):
    r.doc.heading("4. Does using different data sources help?")
    c = r.e.comparison
    if c is None:
        pending(r, "the comparison of which trees each source finds")
        return
    pairs = [(a, b) for a, b in [("ccLidar", "ccP1"),
                                 ("mrcnnLidar", "mrcnnRgb"),
                                 ("ccLidar", "mrcnnLidar"),
                                 ("ccP1", "mrcnnRgb")]
             if a in c.results and b in c.results]
    for first, second in pairs:
        agreementBlock(r, c, first, second)
    if not any(not isPointMethod(n) for n in c.results):
        pending(r, "the same comparison for Mask R-CNN on the CHM against "
                "the RGB mosaic, and for Mask R-CNN against connected "
                "components")


def agreementBlock(r, c, first, second):
    groups = c.agreement(first, second)
    names = {"only " + first: "only " + label(first, True),
             "only " + second: "only " + label(second, True)}
    groups = {names.get(g, g): p for g, p in groups.items()}
    rows = [[group, str(p["count"]), fmt(p["area"], 1), fmt(p["peak"], 1),
             fmt(p["nearest"], 2)] for group, p in groups.items()]
    both = groups["both"]["count"]
    union = len(r.e.scene.crowns) - groups["neither"]["count"]
    r.doc.heading("%s and %s" % (label(first, True), label(second, True)), 2)
    r.doc.table(["crowns found by", "count", "median area m²",
                 "median height m", "nearest crown m"], rows,
                caption="Which crowns each found.",
                widthsCm=[4.4, 1.8, 3.0, 3.2, 3.2])
    r.doc.paragraph(
        "The two agree on %d crowns. Together they find %d of %d (%s), so "
        "combining them can recover at most %s of recall over the better one "
        "alone. The crowns only one of them finds are small, and the ones "
        "neither finds are smaller still."
        % (both, union, len(r.e.scene.crowns),
           pct(union / float(len(r.e.scene.crowns))),
           pct((union - max(len(c.results[first]["found"]),
                            len(c.results[second]["found"])))
               / float(len(r.e.scene.crowns)))))
    r.doc.figure(figures.agreementBars(groups, label(first, True),
                                       label(second, True),
                                       r.figurePath("agreement_%s_%s"
                                                    % (first, second))),
                 "Crowns by which method found them, and their median size.")
    r.doc.figure(figures.agreementTile(
        r.e.scene, c, first, second,
        r.figurePath("agreementTile_%s_%s" % (first, second)),
        labels=(label(first, True), label(second, True))),
        "The tile where %s and %s disagree most, chosen as a worst case."
        % (label(first, True), label(second, True)), 12)


def shapes(r):
    r.doc.heading("5. What each approach produces")
    c = r.e.comparison
    if c is None:
        pending(r, "the shape comparison")
        return
    rows = []
    shapesBy = {}
    for name in c.results:
        polygons = r.e.pseudoCrowns(name) if isPointMethod(name) else None
        quality = c.boxQuality(name, polygons)
        shapesBy[name] = polygons
        kind = "pseudo-crown" if isPointMethod(name) else "own box / mask"
        rows.append([label(name, True), kind,
                     fmt(np.median(quality["boxIou"])) if
                     quality["boxIou"].size else "-",
                     fmt(np.median(quality["shapeIou"])) if
                     quality["shapeIou"].size else "-",
                     pct(float((quality["shapeIou"] > 0.5).mean())) if
                     quality["shapeIou"].size else "-"])
    r.doc.paragraph(
        "The point detector returns a location per tree; Mask R-CNN returns a "
        "box and a mask. For the point detector the outline comes from the "
        "pseudo-crowns. Table %d scores each hit's shape against its crown: "
        "box IoU against the crown's bounding box, shape IoU against the "
        "crown polygon itself." % r.doc.nextTable())
    r.doc.table(["method", "outline from", "median box IoU",
                 "median shape IoU", "shape IoU > 0.5"], rows,
                caption="Shape agreement of hits with their crowns.",
                widthsCm=[2.8, 3.2, 3.2, 3.4, 3.2])
    name = next(n for n in c.results if isPointMethod(n)) \
        if any(isPointMethod(n) for n in c.results) else None
    if name:
        r.doc.figure(figures.shapeTile(r.e.scene, c, "pseudo-crowns, "
                                       + label(name, True),
                                       [s for s in shapesBy[name] if s],
                                       r.figurePath("shapes_" + name)),
                     "Real crowns (cyan) and pseudo-crowns (magenta) from "
                     "the held-out tops. Pseudo-crowns follow the Voronoi "
                     "cells, so they are straight-edged where two trees "
                     "meet.", 12)
    if not any(not isPointMethod(n) for n in c.results):
        pending(r, "whether Mask R-CNN's boxes and masks match the crowns "
                "better than the pseudo-crowns")


def combinations(r):
    r.doc.heading("Combining the methods")
    r.doc.paragraph(
        "Four combinations were implemented, each with every parameter, the "
        "thresholds included, tuned per fold on that fold's validation block "
        "and applied unchanged to its test block. **Confirmed** keeps a box "
        "at one score threshold if a connected-component top lies inside it "
        "and at another if none does, which is the general form of giving "
        "boxes with a top extra score. **Agreement** keeps only confirmed "
        "boxes, the precision extreme. **Box-merged points** fuses the tops "
        "that fall inside the same box into the highest of them, so the box "
        "does the merging the saddle rule approximates. **Union** keeps every "
        "box and every top no box covers, the recall extreme.")
    if not r.e.fusion:
        pending(r, "the combinations of Mask R-CNN and connected components")
        return
    for (first, second), summaryTable in r.e.fusion.items():
        fusionBlock(r, first, second, summaryTable)
    if not any(not isPointMethod(a) for a, _ in r.e.fusion):
        pending(r, "the combinations with Mask R-CNN boxes")


def fusionBlock(r, first, second, summaryTable):
    both = isPointMethod(first) and isPointMethod(second)
    title = ("%s tops as the box role, %s tops as points"
             if both else "%s boxes with %s tops") % (label(first, True),
                                                     label(second, True))
    r.doc.heading(title, 2)
    if both:
        r.doc.paragraph(
            "With no boxes available yet, the same code was run on the two "
            "CHMs: %s's tops were given square boxes, their size tuned per "
            "fold with the rest, and combined with %s's tops. Scores here are "
            "heights, not confidences, so no thresholds were applied."
            % (label(first, True), label(second, True)))
    baseline = summaryTable.get("boxes", {}).get("folds")
    rows = []
    display = {"boxes": label(first, True) + " alone",
               "points": label(second, True) + " alone"}
    for name, entry in sorted(summaryTable.items(),
                              key=lambda item: -item[1]["pooled"]["f1"]):
        pooled = entry["pooled"]
        row = [display.get(name, name), pct(pooled["recall"]), pct(pooled["precision"]),
               fmt(pooled["f1"])]
        if baseline and name != "boxes":
            difference = np.array([f["f1"] for f in entry["folds"]]) - \
                np.array([f["f1"] for f in baseline])
            row.append("%+.3f, %d/%d" % (difference.mean(),
                                         int((difference > 0).sum()),
                                         len(difference)))
        else:
            row.append("-")
        rows.append(row)
    r.doc.table(["strategy", "recall", "precision", "F1",
                 "vs %s alone (mean, wins)" % label(first, True)], rows,
                caption="Cross-validated combinations.",
                widthsCm=[3.6, 2.2, 2.4, 1.8, 5.8])
    r.doc.figure(figures.fusionBars(summaryTable, title,
                                    r.figurePath("fusion_%s_%s"
                                                 % (first, second))),
                 "Combinations, cross-validated.")
    union = summaryTable.get("union", {}).get("pooled")
    if union:
        r.doc.paragraph(
            "Union reaches recall %s at precision %s. The F1 differences "
            "between strategies are within the block-to-block noise; what a "
            "combination changes reliably is where on the recall/precision "
            "trade-off it sits, not how far from the curve."
            % (pct(union["recall"]), pct(union["precision"])))


def caveats(r):
    r.doc.heading("Limitations")
    r.doc.paragraph(
        "Everything here comes from one stand of %d crowns in 3.2 ha. Per-"
        "block F1 varies by two to three points, so differences of a point "
        "or two between methods cannot be established on this site, and "
        "none of the tuned settings has been tested on another. The second "
        "annotated area is what would show whether any of them generalise."
        % len(r.e.scene.crowns))
    r.doc.paragraph(
        "The Mask R-CNN runs reported here were made before two changes. "
        "They trained on tiles that torchvision silently upscaled from 512 to "
        "800 px, whereas the current code runs at the tile's native size, as "
        "YOLO does; and they did not save predictions, which is why the "
        "tree-level sections cover only the connected-component detector. "
        "Mask R-CNN's test tiles are those centred in the held-out block, so "
        "a tree on the block's edge may be seen only partially; this slightly "
        "disadvantages it against the point detector, which sees the whole "
        "area.")
    r.doc.paragraph(
        "The tree-level tables stitch held-out predictions across folds, "
        "which counts a detection just inside one block that lands in a "
        "crown centred in the next as a hit. Their totals therefore run "
        "about half a point above the per-fold figures in Table 1, which "
        "remain the ones to quote.")


def reproduce(r):
    r.doc.heading("Reproducing and completing this report")
    r.doc.paragraph(
        "The package is tt. Every run needs repeating once with the current "
        "code, which saves the predictions the tree-level sections and the "
        "combinations are built from: the two connected-component runs take "
        "minutes, the two Mask R-CNN runs about an hour each. The last "
        "command then writes this report with every section complete.")
    r.doc.code('''
LIDAR=Data/chm_lidar_l2_2025_clipped_modified_last.tif
P1=Data/CHM_P1_2026_clipped_last.tif
CROWNS=Data/shp/annotation_new_0925.shp
AREA=Data/shp/area_p1.shp
python -m tt.dl concomp --dataset ds/lidar --output runs/ccLidar \\
    --percentiles 20,30 --minTopAreas 0.12,0.06 --topSteps 0.12,0.25 \\
    --erosions 1,2 --saddleDrops 0.2,0.3,0.5
python -m tt.dl concomp --dataset ds/p1 --output runs/ccP1 \\
    --percentiles 10,20 --minTopAreas 0.25,0.12 --topSteps 0.12,0.25 \\
    --erosions 1,2 --saddleDrops 0.2,0.3,0.5
python -m tt.dl maskrcnn --dataset ds/lidar --output runs/mrcnnLidar \\
    --epochs 40 --batchSize 4 --device 0
python -m tt.dl maskrcnn --dataset ds/rgb --output runs/mrcnnRgb \\
    --epochs 40 --batchSize 4 --device 0
python -m tt.report --lidarChm $LIDAR --p1Chm $P1 --crowns $CROWNS \\
    --boundary $AREA --dataset ds/lidar \\
    --cv ccLidar=runs/ccLidar --cv ccP1=runs/ccP1 \\
    --cv mrcnnLidar=runs/mrcnnLidar --cv mrcnnRgb=runs/mrcnnRgb \\
    --sweep ccLidar=sweepLidar.json --sweep ccP1=sweepP1.json \\
    --output report''')


SECTIONS = [summary, data, algorithms, tests, best, paired, missed,
            precision, sources, shapes, combinations, caveats, reproduce]
