"""
RGB Mask R-CNN, CHM Mask R-CNN and CC combined: the three find partly
different trees. On a site's saved predictions, no retraining:

    python -m tt.tripleFusion --site study_procedure/koiwainoujo220616

The rule, its thresholds tuned per fold on the validation block like every
other combination (the "off" values are in the grid):

  1. agreement   an RGB box and a CHM box overlapping by IoU >= MATCH_IOU are
                 one detection; it is kept when either score >= high, or both
                 >= low. A box with no partner is kept when its score >= high.
  2. CC's vote   a box not kept by 1 is kept when it holds a CC top and its
                 score >= vote.
  3. CC's trees  a CC top inside no kept box is added when its prominence (how
                 far the CHM falls from it before reaching higher ground) >=
                 prominence; among such tops inside one CHM box, only the most
                 prominent (CC's over-splitting cleaned by the CHM network).

Writes report/fused/mrcnnRgb_triple/summary.json beside the report's
combinations (the summary reads it as "mrcnnRgb+triple:triple"), with "boxes"
(the RGB network alone) checked against the report's.
"""

import argparse
import os
import sys

import numpy as np
from scipy.ndimage import label

from tt.comparison import MethodRun, sharedBlocks
from tt.dl import dlCommon as dc
from tt.fusion import _grid, containment, printSummary, slim
from tt.heightFilter import siteInputs
from tt.report import saveFusedOutputs

MATCH_IOU = 0.3
PROMINENCE_RADIUS_M = 10.0
GRID = {"high": [0.5, 0.7, 0.9],
        "low": [0.2, 0.3, 0.4, 0.5, np.inf],
        "vote": [0.1, 0.2, 0.3, np.inf],
        "prominence": [0.5, 1.0, 2.0, 4.0, np.inf]}


def iouMatrix(a, b):
    """IoU between two lists of boxes [x0, y0, x1, y1]."""
    if not a or not b:
        return np.zeros((len(a), len(b)))
    A = np.array([p["box"] for p in a], float)[:, None, :]
    B = np.array([p["box"] for p in b], float)[None, :, :]
    w = np.clip(np.minimum(A[..., 2], B[..., 2]) - np.maximum(A[..., 0], B[..., 0]), 0, None)
    h = np.clip(np.minimum(A[..., 3], B[..., 3]) - np.maximum(A[..., 1], B[..., 1]), 0, None)
    inter = w * h
    area = lambda X: (X[..., 2] - X[..., 0]) * (X[..., 3] - X[..., 1])
    return inter / np.maximum(area(A) + area(B) - inter, 1e-12)


def pairBoxes(rgb, chm, minIou=MATCH_IOU):
    """Greedy one-to-one matching by IoU, highest first: [(i, j)] plus the
    unmatched indices of each list."""
    iou = iouMatrix(rgb, chm)
    pairs, usedA, usedB = [], set(), set()
    if iou.size:
        for flat in np.argsort(-iou, axis=None):
            i, j = np.unravel_index(flat, iou.shape)
            if iou[i, j] < minIou:
                break
            if i in usedA or j in usedB:
                continue
            pairs.append((int(i), int(j)))
            usedA.add(i)
            usedB.add(j)
    return (pairs, [i for i in range(len(rgb)) if i not in usedA],
            [j for j in range(len(chm)) if j not in usedB])


def candidates(rgb, chm):
    """The detections the networks propose: one per matched pair (the more
    confident box, both scores kept) and one per unmatched box."""
    pairs, loneA, loneB = pairBoxes(rgb, chm)
    out = []
    for i, j in pairs:
        a, b = rgb[i], chm[j]
        lead = a if a["score"] >= b["score"] else b
        out.append(dict(lead, scores=(a["score"], b["score"])))
    out += [dict(rgb[i], scores=(rgb[i]["score"],)) for i in loneA]
    out += [dict(chm[j], scores=(chm[j]["score"],)) for j in loneB]
    return out


def triple(rgb, chm, cc, high, low, vote, prominence):
    """The three-way rule (module docstring) on one block's predictions."""
    proposed = candidates(rgb, chm)
    keep = []
    for c in proposed:
        s = c["scores"]
        keep.append(max(s) >= high or (len(s) == 2 and min(s) >= low))
    if np.isfinite(vote) and cc:
        inside = containment(cc, proposed)          # cc x proposed
        hasTop = inside.any(axis=0)
        for k, c in enumerate(proposed):
            if not keep[k] and hasTop[k] and max(c["scores"]) >= vote:
                keep[k] = True
    kept = [c for c, k in zip(proposed, keep) if k]
    out = [{key: v for key, v in c.items() if key != "scores"} for c in kept]
    if np.isfinite(prominence) and cc:
        free = ~containment(cc, kept).any(axis=1) if kept else np.ones(len(cc), bool)
        loose = [p for p, f in zip(cc, free) if f and p.get("prominence", 0) >= prominence]
        if loose and chm:
            within = containment(loose, chm)        # loose x chm boxes
            best = {}
            for i, p in enumerate(loose):
                hits = np.nonzero(within[i])[0]
                key = ("box", int(hits[0])) if len(hits) else ("own", i)
                if key not in best or p["prominence"] > loose[best[key]]["prominence"]:
                    best[key] = i
            loose = [loose[i] for i in sorted(best.values())]
        out += loose
    return out


def prominenceAt(surface, x, y, radiusM=PROMINENCE_RADIUS_M):
    """Height of the CHM at (x, y)'s highest nearby pixel above the level at
    which its region joins higher ground within radiusM (binary search); the
    full height if no higher ground is that close."""
    chm = surface.chm
    row, col = surface.pixel(x, y)
    r = int(round(radiusM / surface.pixelSize))
    r0, c0 = max(0, row - 2), max(0, col - 2)
    patch = chm[r0:row + 3, c0:col + 3]
    if patch.size == 0:
        return 0.0
    dr, dc_ = np.unravel_index(np.argmax(patch), patch.shape)
    row, col = r0 + dr, c0 + dc_
    w0, v0 = max(0, row - r), max(0, col - r)
    win = chm[w0:row + r + 1, v0:col + r + 1]
    pr, pc, h = row - w0, col - v0, float(chm[row, col])
    if h <= 0:
        return 0.0
    if not (win > h).any():
        return h
    structure = np.ones((3, 3), bool)

    def joined(level):
        lab, _ = label(win >= level, structure=structure)
        return bool((win[lab == lab[pr, pc]] > h).any())

    if joined(h):
        return 0.0
    lowLevel, highLevel = 0.0, h
    for _ in range(12):
        mid = (lowLevel + highLevel) / 2
        lowLevel, highLevel = (mid, highLevel) if joined(mid) else (lowLevel, mid)
    return h - lowLevel


class TripleFusion(object):
    """The three-way rule under leave-one-block-out, tuned on validation blocks."""

    def __init__(self, crowns, blocks, rgbRun, chmRun, ccRun, surface, verbose=True):
        self.crowns = crowns
        self.allBlocks = dict(blocks)
        self.runs = (rgbRun, chmRun, ccRun)
        self.blocks = sharedBlocks(blocks, self.runs, verbose)
        self.surface = surface
        self.verbose = verbose

    def _split(self, block, which):
        """(rgb, chm, cc) predictions on this fold's validation or test block;
        CC above its own floor, prominence attached to each top."""
        rgbRun, chmRun, ccRun = self.runs
        if which == "validation":
            region, rgb = rgbRun.validation(block)
            _, chm = chmRun.validation(block)
            _, cc = ccRun.validation(block)
        else:
            region = block
            rgb, chm, cc = (r.record(block)["predictions"] for r in self.runs)
        floor = ccRun.threshold(block)
        cc = [dict(p) for p in cc if p["score"] >= floor]
        area = self.allBlocks[region]
        rgb, chm, cc = (dc.predictionsInRegion(x, area) for x in (rgb, chm, cc))
        for p in cc:
            p["prominence"] = prominenceAt(self.surface, p["centreX"], p["centreY"])
        return region, rgb, chm, cc

    def _score(self, predictions, block):
        return dc.evaluateDetections(predictions, self.crowns, self.allBlocks[block])

    def runFold(self, block):
        vBlock, vRgb, vChm, vCc = self._split(block, "validation")
        _, tRgb, tChm, tCc = self._split(block, "test")
        best, bestScore = None, -1.0
        for setting in _grid(GRID):
            score = dc.objectiveOf(self._score(triple(vRgb, vChm, vCc, **setting), vBlock))
            if score > bestScore:
                best, bestScore = setting, score
        results = {}
        rgbThreshold = self.runs[0].threshold(block)
        for name, output, setting in (
                ("triple", triple(tRgb, tChm, tCc, **best), best),
                ("boxes", [p for p in tRgb if p["score"] >= rgbThreshold],
                 {"threshold": rgbThreshold})):
            result = self._score(output, block)
            result.update(block=block, setting={k: (None if not np.isfinite(v) else v)
                                                for k, v in setting.items()},
                          output=[slim(p) for p in output])
            results[name] = result
        return results

    def run(self):
        perName = {"triple": [], "boxes": []}
        self.outputs = {"triple": {}, "boxes": {}}
        for block in sorted(self.blocks):
            for name, result in self.runFold(block).items():
                self.outputs[name][block] = result.pop("output")
                perName[name].append(result)
        summary = {n: {"folds": f, "pooled": dc.averageFolds(f)} for n, f in perName.items()}
        if self.verbose:
            printSummary(summary)
        return summary


def height(site):
    return "Lidar" if os.path.isdir(os.path.join(site, "runs", "ccLidar")) else "P1"


def fuseSite(site, ccName=None, verbose=True):
    h = height(site)
    meta, scene, surface = siteInputs(site)
    blocks = meta["blockGeometries"]
    ccName = ccName or "cc" + h
    runs = [MethodRun(n, os.path.join(site, "runs", n), blocks)
            for n in ("mrcnnRgb", "mrcnn" + h, ccName)]
    validation = TripleFusion(scene.crowns, blocks, *runs, surface=surface,
                              verbose=verbose)
    summary = validation.run()
    folder = os.path.join(site, "report", "fused", "mrcnnRgb_triple")
    saved = os.path.join(site, "report", "fused", "mrcnnRgb_cc" + h, "summary.json")
    if os.path.exists(saved):
        a = dc.loadJson(saved)["boxes"]["pooled"]
        b = summary["boxes"]["pooled"]
        if (a["hits"], a["predictions"]) != (b["hits"], b["predictions"]):
            raise RuntimeError("%s: the RGB network alone scores %d / %d here but "
                               "%d / %d in the report: not the same inputs"
                               % (site, b["hits"], b["predictions"], a["hits"],
                                  a["predictions"]))
    os.makedirs(folder, exist_ok=True)
    dc.saveJson(summary, os.path.join(folder, "summary.json"))
    saveFusedOutputs(folder, validation.outputs)
    return summary


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site", action="append", required=True,
                        help="a site folder of the procedure (repeatable)")
    parser.add_argument("--cc", default=None,
                        help="the CC run in the site's runs/ (default cc<height>, "
                             "tuned on crowns; e.g. ccAutoLidar)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parseArguments(argv)
    for site in args.site:
        print("[triple] %s" % os.path.basename(os.path.normpath(site)))
        fuseSite(site, args.cc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
