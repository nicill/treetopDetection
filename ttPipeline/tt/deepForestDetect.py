"""
The `deepForestDetect` command: DeepForest's prebuilt tree-crown model
(tt.detectors.dl.deepForest), used as downloaded, written as a run like any
other so connected components can be tuned on its boxes with no annotation of
the site at all.

    python -m tt.deepForestDetect --dataset <site>/ds/rgb \\
        --output <out>/<site>/runs/deepforestRgb

Every box above the score floor is kept for every block, with the operating
point fixed before any result (tt.detectors.dl.deepForest.THRESHOLD).
"""

import argparse
import sys

from .data.site import TDData
from .detectors.dl.deepForest import PretrainedDetector
from .experiments.trained import FixedRun


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mosaic", default=None,
                        help="the RGB mosaic, if the dataset's path moved")
    parser.add_argument("--resolution", type=float, default=0.1)
    args = parser.parse_args(argv)

    detector = PretrainedDetector(resolution=args.resolution,
                                  mosaic=args.mosaic)
    runSet = FixedRun(TDData.fromDataset(args.dataset), detector).run()
    runSet.save(args.output)
    pooled = runSet.pooled
    count = len(next(iter(runSet.predictions.values()), []))
    print("[deepforest] %d boxes; at %.2f: R %.3f P %.3f F1 %.3f W %.3f"
          % (count, detector.params["threshold"], pooled["recall"],
             pooled["precision"], pooled["f1"], pooled["weighted"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
