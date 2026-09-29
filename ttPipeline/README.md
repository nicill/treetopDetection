# tt — treetop detection on canopy height models

```bash
pip install -r requirements.txt

tt detect  --chm CHM.tif --crowns crowns.shp --boundary area.shp --output out
tt analyse --chm CHM.tif --crowns crowns.shp --boundary area.shp --output out
tt sweep   --chm CHM.tif --crowns crowns.shp --boundary area.shp
tt merge   --chm CHM.tif --crowns crowns.shp --boundary area.shp
tt align   --chm CHM.tif --layer photo=ortho.tif --layer crowns=crowns.shp
```

Or from Python, which is the same objects:

```python
from tt import Scene, ConCompDetector, CrownEvaluator, TopMerger

scene = Scene("CHM.tif", crownsPath="crowns.shp", boundaryPath="area.shp")
detector = ConCompDetector(lowerPercentile=10, minTopAreaM2=0.12,
                           topStepM=0.12, erosionIterations=1,
                           merger=TopMerger("saddle", epsM=8.0,
                                            saddleDropM=0.5))
tops = detector.detect(scene)
print(CrownEvaluator(scene).score(tops))
```

## The modules

| module | holds |
|---|---|
| `scene.py` | `Scene` — CHM, crowns and boundary on one grid |
| `detector.py` | `ConCompDetector` — detection, and nothing else |
| `merging.py` | `TopMerger` — every decision about whether two tops are one tree |
| `evaluation.py` | `CrownEvaluator`, `Classification` — crown-level scoring |
| `analysis.py` | `MissAnalysis` — what it gets wrong and why |
| `render.py` | `TileRenderer` — the diagnostic images |
| `sweeps.py` | `Sweep` — trying many settings |
| `alignment.py` | `Alignment` — offsets between CHM, photo and crowns |
| `tops.py` | `Tops` — a set of treetops and how they leave the program |
| `comparison.py` | `MethodRun`, `TreeComparison` — which trees each method finds, from saved predictions |
| `fusion.py` | `FusionCrossValidation` — box detector + point detector combinations, tuned on validation blocks |
| `report.py`, `reportText.py`, `reportFigures.py`, `odt.py` | the ODT results report |
| `cli.py` | the subcommands |
| `dl/` | the YOLO / Mask R-CNN benchmark, moved but not yet rewritten |

## Overnight run

`overnightMongolia.sh` trains YOLO, Faster R-CNN, RetinaNet and FCOS on the
mosaic and YOLO on both CHMs, reusing the runs already in `runs/`; after every
stage it writes the report and an image of every held-out block of every
result. Everything goes to one dated folder under the lab's `ongoingExps`;
`STATUS.txt` there says what finished and with what score.

```bash
./overnightMongolia.sh              # about 4-5 hours on an RTX 4090
DRYRUN=1 ./overnightMongolia.sh     # everything but the training, to check
```

It also runs two things on the Sergi site: connected components at settings
fixed on Mongolia against Sergi's own tuning (`python -m tt.transfer`), and
Mask R-CNN on the Sergi height model, under `sergi/` in the same folder.

Block images alone: `python -m tt.blockImages --help`. Colour code: green hit,
orange repeat, blue outside every crown at canopy height, red outside and
low, yellow outline for a missed crown.

## Code quality pass

Measured before changing anything. Excluding a one-off import cost, the
detector spent its time here:

| where | share | status |
|---|---|---|
| `saddleDrop`, ~10^5 calls per scene | 42% | rewritten |
| sub-blob maxima via full-array `np.where` per label | 22% | bounding boxes |
| the descent itself, on full windows | — | blobs cropped first |
| `_stretch` | not measurable | left alone |

Result: **4.0x faster on the detector, with output provably unchanged** by the
cropping — full-window and cropped descents agree on all 231 blobs of the real
scene. `_stretch` was left exactly as it was: it runs 32 times per scene, and
reordering its floating-point arithmetic can flip a pixel's rounding to uint8.

Things found along the way, which mattered more than the speed:

* **`saddleDrop` was not translation-invariant.** It rounded absolute
  coordinates with `np.rint`, which rounds half to even, so a sample 2.5 px
  along came out as 3 from an odd origin and 2 from an even one. Windows
  overlap with different origins, so the same pair of treetops could get a
  different saddle verdict depending on which window evaluated it. It now
  rounds the displacement and adds the origin as an integer; verified
  invariant over 20000 random pairs and offsets. Scores are unchanged to three
  decimals; one detection moves on the regression setting.
* **`tt align` could not run.** It referenced `cu` and `cv2`, neither defined in
  the module. Nothing exercised it, so nothing noticed. There is now a test per
  subcommand, run end to end, so a NameError anywhere fails the suite.
* **Crowns were loaded three different ways**, in `Scene`, in the benchmark
  shim, and in what alignment expected — the drift `Scene` exists to prevent.
  There is one `readCrowns` now and all three call it.
* **All 88 function-level imports are at module top**, grouped stdlib /
  third-party / local. `pyflakes` reports nothing across the package.
* **`Affine * x` is `Affine @ x`** in all 26 places, as affine 3.0 asks.
  rasterio still uses `*` inside its own windowed reads; a filter scoped to
  warnings raised from rasterio's modules silences only that, and a test
  confirms a `*` reintroduced in this package still errors.

The core package is 12 modules and 2516 lines with one function over 60 lines
(the argument parser). `dl/` still has eleven, its `crossValidate` functions
among them: it works and is tested, but has not been restructured.

## What changed, and why

**`Scene` is the piece that was missing.** Ten scripts each loaded a CHM,
loaded crowns, reprojected them, repaired invalid geometry and burned a plot
raster. They drifted: some applied the crown shift, some did not. That decision
is made once now.

**The detector only detects.** It had grown classification, rendering,
registration diagnostics and false-positive analysis, reaching 1207 lines, and
every analysis script had to construct one to reach them.

**Every proximity decision goes through `TopMerger`.** It used to be spread
over three places in two coordinate systems, which is how a metric added to one
came to silently do nothing in another — the saddle test was implemented in the
global pass but not the per-blob pass, where most merging happens, and so
looked useless when it is the best merge available.

**One CLI.** Fifteen scripts with 90 to 300 line mains, mostly duplicated
argument parsing.

22 files and 7953 lines became 11 modules and 1940, with one function over 60
lines (the argument parser, which is a list). The `dl/` benchmark is moved
unchanged and still has its own entry points; it works, and rewriting it is a
separate job.

## Best settings measured

LiDAR CHM at 0.25 m, 921 crowns:

```
--lowerPercentile 10 --minTopArea 0.12 --topStep 0.12 --erosionIterations 1
--metric saddle --saddleDrop 0.5 --eps 8
```

Recall 93.3%, precision 81.0%, F1 86.7%. Of the detections: 81.0% hit, 7.2%
repeat, 8.1% outside a crown at canopy height (probably unannotated trees),
3.8% outside and below canopy. Precision reads 89.1% if the canopy-level ones
are real. Tuned and scored on the same crowns, so treat as directional.
