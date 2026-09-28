# Results on the Ulaanbaatar site

`report.odt` / `report.pdf` — the comparison of connected components and
Mask R-CNN on 921 crowns, under leave-one-block-out cross-validation.

`runs/*/results.json` — the cross-validated scores the report is built from:

| run | where it came from |
|---|---|
| ccLidar, ccP1 | connected components, corrected cross-validation, this code |
| mrcnnLidar, mrcnnRgb | Mask R-CNN, parsed from the logged run of 28 Sep 2026 |
| ccLidarOld, ccP1Old | connected components under the earlier protocol, for reference |

`sweep*.json` — every detector and merge setting tried on each CHM.

The Mask R-CNN runs predate prediction saving, so the report's Mask R-CNN
tree-level sections are marked as pending. `../completeReport.sh` re-runs
everything and rewrites the report with them filled in.
