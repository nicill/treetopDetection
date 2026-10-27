"""
dlCommon.py

Shared machinery for benchmarking treetop / crown detectors against each other:
tiling, spatial blocking for cross-validation, prediction stitching, and one
evaluation function that every method is scored through.

Why the pieces are shaped this way
----------------------------------

*Spatial blocks, not random splits.* Neighbouring tiles of a forest share
individual trees, illumination and stand structure. A random tile split leaks
the test set into training and flatters every learned model. The area is cut
into a grid of blocks; a fold holds out one whole block, and training tiles must
lie a buffer away from it so no crown appears on both sides.

*Stitched evaluation.* Tiles overlap so that trees near an edge are seen whole
somewhere. Scoring tile by tile then counts those trees more than once and
inflates the result — Burmaa's own README measured that at 0.17 to 0.25 of
detection F1. Predictions are merged back into world coordinates, deduplicated
with NMS, and each tree is scored once.

*One metric for everything.* The connected-component detector produces points;
Mask R-CNN produces masks; YOLO produces boxes or polygons. They are compared on
the crown-level rule already used in conCompTreetopDetection: a prediction is
reduced to its centre, a crown counts as found if any centre falls inside it,
the highest-scoring centre in a crown is the hit and any others are repeats, and
a centre in no crown is a false positive. That is the number reported as recall,
precision and F1, so the learned models and the hand-tuned one are directly
comparable. Box AP50 is reported alongside for readers who expect it, but it is
not the headline.

Normalisation is fixed once per dataset and stored, so a model never sees test
tiles scaled differently from the tiles it trained on.
"""

import datetime
import json
import os
import sys

import numpy as np
from shapely import contains_xy, prepare

# Re-exported for the modules not yet moved to the new classes; the code lives
# in tt.data, tt.scoring and tt.experiments.
from ..data.annotations import crownsInRegion  # noqa: F401
from ..data.patches import (  # noqa: F401
    ChannelSpec, assignTilesToBlocks, blockSuffix, buildGrid,
    chooseValidationBlock, cropCrowns, drawTiles, foldSplit, makeBlocks,
    makeTiles, measureRanges, parseSource, readWindow, tileExtent, tileRegion,
    tileToWorld)
from ..data.detections import Detections
from ..detectors.base import nmsKeep
from ..scoring.evaluator import assignToCrowns  # noqa: F401
from ..experiments.runset import poolFolds
from ..scoring.evaluator import THRESHOLD_CANDIDATES, Evaluator
from ..scoring.objective import (Objective, RECALL_WEIGHT,  # noqa: F401
                                f1Of, weightedOf)
from ..scoring.reference import CrownFrame


def nonMaximumSuppression(predictions, iouThreshold=0.4):
    """Plain NMS in world coordinates (tt.detectors.base.nmsKeep)."""
    keep = nmsKeep([p["box"] for p in predictions],
                   [p["score"] for p in predictions], iouThreshold)
    return [predictions[i] for i in keep]


# ---------------------------------------------------------------------- #
# the tuning objective
# ---------------------------------------------------------------------- #

# Every choice made on held-out-free data (a connected-component setting per
# fold, a network's score threshold, a combination's settings, a pseudo-tuned
# setting) maximises one objective, chosen by the environment so a whole run
# uses one: TT_OBJECTIVE=f1 (the default) or TT_OBJECTIVE=weighted, the
# weighted mean RECALL_WEIGHT * recall + (1 - RECALL_WEIGHT) * precision.
# Every result carries both "f1" and "weighted", whichever was tuned on.
OBJECTIVES = ("f1", "weighted")


def tuningObjective():
    return Objective.fromEnvironment().name


def weightedScore(recall, precision):
    return weightedOf(recall, precision)


def objectiveOf(result):
    """The value the tuning maximises for one scored result."""
    return Objective.fromEnvironment().value(result)


def _scores(recall, precision):
    return f1Of(recall, precision), weightedOf(recall, precision)


# ---------------------------------------------------------------------- #
# the one evaluation
# ---------------------------------------------------------------------- #

def evaluateDetections(predictions, crowns, blockGeometry, verbose=False):
    """
    Score prediction dicts against crowns inside one block, through
    tt.scoring.Evaluator, the one scorer.
    """
    result = Evaluator(CrownFrame(crowns)).score(
        Detections.fromPredictions(predictions), blockGeometry)
    if verbose:
        print("    crowns %d, predictions %d -> hits %d, repeats %d, fp %d  "
              "| R %.3f P %.3f F1 %.3f"
              % (result["crowns"], result["predictions"], result["hits"],
                 result["repeats"], result["falsePositives"],
                 result["recall"], result["precision"], result["f1"]))
    return result


def predictionsInRegion(predictions, region):
    """Predictions whose centre lies in `region`, tested in one vectorised call."""
    if not predictions:
        return []
    prepare(region)
    inside = contains_xy(region,
                         np.array([p["centreX"] for p in predictions]),
                         np.array([p["centreY"] for p in predictions]))
    return [p for p, keep in zip(predictions, inside) if keep]


def selectThreshold(predictions, crowns, blockGeometry,
                    candidates=THRESHOLD_CANDIDATES, verbose=True):
    """The operating point on a validation block (Evaluator.chooseThreshold)."""
    return Evaluator(CrownFrame(crowns)).chooseThreshold(
        Detections.fromPredictions(predictions), blockGeometry, candidates,
        verbose)


def thresholdCurve(predictions, crowns, candidates):
    """[(threshold, objective)] (Evaluator.thresholdCurve)."""
    return Evaluator(CrownFrame(crowns)).thresholdCurve(
        Detections.fromPredictions(predictions), None, candidates)


def averageFolds(foldResults):
    """Pool the folds (tt.experiments.poolFolds)."""
    return poolFolds(foldResults)


# ---------------------------------------------------------------------- #

class Tee(object):
    """Duplicate a stream to a file, so a run is both watchable and recorded."""

    def __init__(self, stream, handle):
        self.stream = stream
        self.handle = handle

    def write(self, text):
        self.stream.write(text)
        self.handle.write(text)
        self.handle.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.handle.flush()

    def isatty(self):
        return getattr(self.stream, "isatty", lambda: False)()


def startLogging(path):
    """
    Send everything printed from here on to `path` as well as the terminal.

    Progress bars write carriage returns, so a log file ends up holding every
    intermediate state of the bar. Worth knowing when reading one back; `col -b`
    or `sed 's/.*\r//'` cleans it up.
    """

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    handle = open(path, "a", buffering=1)
    handle.write("\n%s\n=== %s ===\n" % ("=" * 70, _timestamp()))
    sys.stdout = Tee(sys.stdout, handle)
    sys.stderr = Tee(sys.stderr, handle)
    return handle


def _timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def saveJson(obj, path):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    return path


def loadJson(path):
    with open(path) as f:
        return json.load(f)
