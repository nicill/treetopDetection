"""
Sweep — trying many parameter settings against one scene.

Two sweeps were separate scripts with near-identical 100-line mains. The only
real difference was which parameters varied, so that is the only thing the
caller supplies now.

All of these tune and score on the same crowns, so the numbers are optimistic
in the usual way. Use them to decide what to try; use cross-validation for a
figure to quote.
"""

import itertools
import json
import time


from .detector import ConCompDetector
from .evaluation import CrownEvaluator
from .merging import TopMerger


class Sweep(object):

    def __init__(self, scene, base=None, verbose=True):
        self.scene = scene
        self.base = dict(base or {})
        self.evaluator = CrownEvaluator(scene)
        self.verbose = verbose
        self.rows = []

    # ------------------------------------------------------------------ #

    def run(self, grid, mergerGrid=None):
        """
        `grid` maps detector keyword names to lists of values; `mergerGrid`
        does the same for the merger. Their product is evaluated.
        """
        detectorSettings = _expand(grid)
        mergerSettings = _expand(mergerGrid) if mergerGrid else [{}]
        combinations = list(itertools.product(detectorSettings,
                                              mergerSettings))

        if self.verbose:
            print("%d crowns, %d settings" % (len(self.scene.crowns),
                                              len(combinations)))
        started = time.time()

        for index, (detectorKeywords, mergerKeywords) in \
                enumerate(combinations):
            self.rows.append(self.one(detectorKeywords, mergerKeywords))
            if self.verbose and (index + 1) % 10 == 0:
                elapsed = time.time() - started
                print("  %d/%d (%.0fs elapsed, ~%.0fs left)"
                      % (index + 1, len(combinations), elapsed,
                         elapsed / (index + 1)
                         * (len(combinations) - index - 1)))
        return self.rows

    def one(self, detectorKeywords, mergerKeywords=None):
        """Evaluate a single setting."""
        merger = TopMerger(**mergerKeywords) if mergerKeywords else TopMerger()
        detector = ConCompDetector(merger=merger, verbose=False,
                                   **dict(self.base, **detectorKeywords))
        tops = detector.detect(self.scene)
        result = self.evaluator.score(tops)
        result.update(detectorKeywords)
        result.update(mergerKeywords or {})
        result["label"] = _label(detectorKeywords, mergerKeywords)
        return result

    # ------------------------------------------------------------------ #

    def varyingKeys(self):
        """
        Which swept parameters actually took more than one value.

        Printing the constant ones wastes the width that the varying ones need,
        and truncation then hides exactly the column being compared.
        """
        if not self.rows:
            return []
        candidates = [k for k in self.rows[0]
                      if k not in ("label", "detections", "recall",
                                   "precision", "f1", "crowns", "hits",
                                   "repeats", "background", "repeatShare",
                                   "backgroundShare")]
        return [k for k in candidates
                if len({str(r.get(k)) for r in self.rows}) > 1]

    def table(self, sortBy="f1", limit=20):
        rows = sorted(self.rows, key=lambda r: -r[sortBy])[:limit]
        keys = self.varyingKeys()
        width = max(24, min(56, sum(len(_short(k)) + 8 for k in keys)))

        print("%-*s %6s %7s %7s %7s %8s"
              % (width, "setting", "dets", "recall", "prec", "F1", "repeat%"))
        print("-" * (width + 42))
        for row in rows:
            label = " ".join("%s=%s" % (_short(k), row.get(k)) for k in keys)
            print("%-*s %6d %6.1f%% %6.1f%% %6.1f%% %7.1f%%"
                  % (width, label[:width], row["detections"],
                     100 * row["recall"], 100 * row["precision"],
                     100 * row["f1"], 100 * row["repeatShare"]))
        if len(keys) < len(self.varyingKeys()) + 1:
            constant = [k for k in self.rows[0]
                        if k not in keys and k in ("metric", "epsM",
                                                   "heightWeight")]
            if constant:
                print("held constant: %s"
                      % ", ".join("%s=%s" % (_short(k), self.rows[0][k])
                                  for k in constant))
        return rows

    def best(self, minimumRecall=None):
        candidates = self.rows
        if minimumRecall is not None:
            candidates = [r for r in self.rows if r["recall"] >= minimumRecall]
            if not candidates:
                return None
        return max(candidates, key=lambda r: r["f1"])

    def save(self, path):
        with open(path, "w") as handle:
            json.dump(self.rows, handle, indent=1, default=float)
        return path


def _expand(grid):
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values))
            for values in itertools.product(*[grid[k] for k in keys])]


SHORT = {"lowerPercentile": "pct", "minTopAreaM2": "minTop",
         "topStepM": "step", "erosionIterations": "erode",
         "saddleDropM": "drop", "heightWeight": "w", "epsM": "eps",
         "metric": "metric", "windowSizeM": "window",
         "minTreeAreaM2": "minTree", "minTopAreaSlope": "minTopSlope",
         "saddleDropSlope": "dropSlope"}


def _short(key):
    return SHORT.get(key, key)


def _label(detectorKeywords, mergerKeywords):
    parts = []
    for source in (detectorKeywords, mergerKeywords or {}):
        for key, value in sorted(source.items()):
            short = key.replace("minTopAreaSlope", "minTopSlope") \
                       .replace("saddleDropSlope", "dropSlope") \
                       .replace("lowerPercentile", "pct") \
                       .replace("minTopAreaM2", "minTop") \
                       .replace("topStepM", "step") \
                       .replace("erosionIterations", "erode") \
                       .replace("saddleDropM", "drop") \
                       .replace("heightWeight", "w") \
                       .replace("epsM", "eps")
            parts.append("%s=%s" % (short, value))
    return " ".join(parts)
