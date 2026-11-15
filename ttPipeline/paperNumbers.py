"""
The numbers the paper still needs, from the study folders, in one run:

  sites      per site and height model (the P1 CHM; LiDAR where a LiDAR dataset exists): scored area,
             crowns, trees per hectare, median crown diameter (of a circle with the crown's area) and
             height (the CHM's maximum in the crown), the CHM's native resolution and CC's working one
  check      per site and height model, on the CHM at CC's working resolution (the package's own
             measures, tt.prepare.dsmChm): crowns with a peak of their own (any; prominence at least
             0.5 m; at least 1 m), with two or more peaks of 0.5 m or 1 m, with their highest point on
             their outline, and the share of the scored canopy covered by crowns
  alignment  per Terelj area, the offset left between its P1 and its (aligned) LiDAR CHM
  koiwai     the trees CC misses at Koiwai (runs/ccP1, held-out blocks): how many have no peak of their
             own of 0.05, 0.1 or 0.25 m, and their crown areas against those of the trees it finds
  tests      experiment 3, fixed in advance: the four-channel network, the union of the RGB network with
             CC and its union with the CHM network, each against the RGB network; per site the paired
             Wilcoxon over blocks, Holm over the three; across sites the Wilcoxon over the sites'
             pooled differences, Holm over the three

    cd ttPipeline && python paperNumbers.py            # writes DROP/paperNumbers.json
    python paperNumbers.py --parts sites check         # some parts only
"""

import argparse
import glob
import json
import os

import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy import ndimage
from scipy.stats import wilcoxon

from tt.analysis.statistics import blockKey, compareFolds, holm
from tt.data import TDData
from tt.data.alignment import Alignment
from tt.data.raster import HeightModel
from tt.experiments import RunSet
from tt.experiments.fusion import FusionCrossValidation
from tt.prepare.dsmChm import peakCounts, peaksAbove, topOnOutline
from tt.runCompare import loadRun
from tt.scoring import AnnotatedCrowns, Evaluator

DB = os.environ.get("DROP") or os.path.expanduser("~/Yago Lab Dropbox/Meetings/2026TreeDet")
QUEBEC = ["20230712_afcamoisan_itrf20", "20230712_afcahoule_itrf20",
          "20230606_cbblackburn5", "20230608_cbbernard4", "20230606_cbblackburn6"]
SITES = ([(s.split("_")[1], os.path.join(DB, "study_edge", s)) for s in QUEBEC]
         + [("koiwai", os.path.join(DB, "study_edge", "koiwainoujo220616"))]
         + [("terelj" + a, os.path.join(DB, "study_terelj" + a, "terelj" + a))
            for a in ("First", "Second", "Third")]
         + [("enshurinV2", os.path.join(DB, "study_enshurinVlad2"))])
PARTS = ("sites", "check", "alignment", "koiwai", "tests")
BOOTSTRAP = 2000


# ------------------------------------------------------------------ the site's data

def heightDatasets(folder):
    """{source: dataset folder}: the P1 one (ds/P1 or ds/chm), and any LiDAR one."""
    found = {}
    for d in sorted(glob.glob(os.path.join(folder, "ds", "*", "dataset.json"))):
        name = os.path.basename(os.path.dirname(d))
        with open(d) as handle:
            kinds = [c["kind"] for c in json.load(handle).get("channels", [])]
        if kinds != ["chm"]:
            continue
        if "lidar" in name.lower():
            found.setdefault("LiDAR", os.path.dirname(d))
        elif name in ("P1", "chm"):
            found.setdefault("P1", os.path.dirname(d))
    return found


def chmPath(dataset):
    with open(os.path.join(dataset, "dataset.json")) as handle:
        meta = json.load(handle)
    path = next(c["path"] for c in meta["channels"] if c["kind"] == "chm")
    return path if os.path.isabs(path) else os.path.join(os.getcwd(), path)


def ccSettings(folder, source):
    """CC's working resolution and minimum height, from its run's settings."""
    for name in (("ccP1", "cc") if source == "P1" else ("ccLidar", "ccLiDAR")):
        path = os.path.join(folder, "runs", name, "results.json")
        if os.path.exists(path):
            with open(path) as handle:
                s = json.load(handle).get("settings", {})
            heights = [float(v) for v in str(s.get("minHeights", s.get("minHeight", 2.0))).split(",") if v]
            return float(s.get("resolution", 0.25)), min(heights), name
    return 0.25, 2.0, None


def scoredArea(data):
    area = data.annotations.area
    if area is None:
        from shapely.ops import unary_union
        area = unary_union([b.geometry for b in data.blocks])
    return getattr(area, "geometry", area).unary_union if hasattr(area, "geometry") else area


def crownGrid(crowns, surface):
    """Crown ids (1..N) on the surface's grid."""
    crs = getattr(surface, "crs", None) or getattr(getattr(surface, "grid", None), "crs", None)
    if crs is not None and crowns.crs is not None and crowns.crs != crs:
        crowns = crowns.to_crs(crs)
    return rasterize(((g, i + 1) for i, g in enumerate(crowns.geometry)), out_shape=surface.chm.shape,
                     transform=surface.transform, fill=0, dtype="int32")


# ------------------------------------------------------------------ parts

def siteRow(name, folder, source, dataset, withCheck):
    data = TDData.fromDataset(dataset)
    crowns = data.annotations.crowns
    area = scoredArea(data)
    path = chmPath(dataset)
    with rasterio.open(path) as src:
        native = abs(src.transform.a)
    resolution, minHeight, ccRun = ccSettings(folder, source)
    surface = HeightModel(path).surface(resolution=resolution, minHeight=0.0)
    ids = crownGrid(crowns, surface)
    index = np.arange(1, len(crowns) + 1)
    present = ndimage.sum(np.ones_like(ids), ids, index) > 0
    peak = np.asarray(ndimage.maximum(surface.chm, ids, index))
    areas = crowns.geometry.area.values
    row = {"site": name, "source": source, "dataset": dataset, "ccRun": ccRun,
           "chmResolution": round(native, 4), "ccResolution": resolution, "minHeight": minHeight,
           "areaHa": round(area.area / 1e4, 3), "crowns": int(len(crowns)),
           "treesPerHa": round(len(crowns) / (area.area / 1e4), 1),
           "diameterM": round(float(np.median(2 * np.sqrt(areas / np.pi))), 2),
           "heightM": round(float(np.median(peak[present])), 2)}
    if withCheck:
        for h in (0.05, 0.5, 1.0):
            labels, _ = peaksAbove(surface.chm, h)
            count = peakCounts(ids, labels, len(crowns))
            row["ownPeak%gm" % h] = round(float(np.mean(count[present] >= 1)), 4)
            if h >= 0.5:
                row["severalPeaks%gm" % h] = round(float(np.mean(count[present] >= 2)), 4)
        row["topOnOutline"] = round(topOnOutline(surface.chm, ids, index, present), 4)
        canopy = (surface.chm >= max(minHeight, 0.5)) & surface.grid.mask(area)
        row["covered"] = round(float(np.count_nonzero(canopy & (ids > 0)) / max(np.count_nonzero(canopy), 1)), 4)
    return row


def alignmentRow(name, datasets):
    data = TDData.fromDataset(datasets["P1"])
    boundary = data.annotations.areaPath
    result = Alignment(chmPath(datasets["P1"]), resolution=0.10, minHeight=2.0, boundaryPath=boundary,
                       verbose=False).add("LiDAR", chmPath(datasets["LiDAR"])).measure(tileM=20.0)
    summary = next(iter(result.values()))
    return {"site": name, **(summary or {"tiles": 0})}


def koiwaiRow(name, folder, dataset):
    data = TDData.fromDataset(dataset)
    reference = AnnotatedCrowns(data.annotations)
    evaluator = Evaluator(reference)
    blocks = {b.name: b.geometry for b in data.blocks}
    run = RunSet.load(os.path.join(folder, "runs", "ccP1"))
    held = run.heldOut(blocks)
    found, missed = [], []
    for block, geometry in blocks.items():
        dets = held.subset(held.cache["block"] == block)
        matching = evaluator.match(dets, geometry)
        hit = set(matching.hitCrowns)                  # crown index labels, not positions
        for label, geometry in zip(matching.crowns.index, matching.crowns.geometry):
            (found if label in hit else missed).append(geometry)
    resolution, _, _ = ccSettings(folder, "P1")
    surface = HeightModel(chmPath(dataset)).surface(resolution=resolution, minHeight=0.0)
    import geopandas as gpd
    frame = gpd.GeoDataFrame(geometry=found + missed, crs=data.annotations.crowns.crs)
    ids = crownGrid(frame, surface)
    row = {"site": name, "found": len(found), "missed": len(missed),
           "missedAreaMedianM2": round(float(np.median([g.area for g in missed])), 2) if missed else None,
           "foundAreaMedianM2": round(float(np.median([g.area for g in found])), 2) if found else None}
    present = ndimage.sum(np.ones_like(ids), ids, np.arange(1, len(frame) + 1)) > 0
    for h in (0.05, 0.1, 0.25):
        labels, _ = peaksAbove(surface.chm, h)
        count = peakCounts(ids, labels, len(frame))
        own = count >= 1
        row["missedWithoutPeak%gm" % h] = int(np.sum(~own[len(found):] & present[len(found):]))
        row["foundWithoutPeak%gm" % h] = int(np.sum(~own[:len(found)] & present[:len(found)]))
    return row


COMPARISONS = (("4-channel", "mrcnnRgbP1", None), ("RGB+CC union", "mrcnnRgb", "ccP1"),
               ("RGB+CHM network union", "mrcnnRgb", "mrcnnP1"))


def testRows(sites):
    rng = np.random.default_rng(0)
    perSite, differences = [], {c[0]: [] for c in COMPARISONS}
    for name, folder in sites:
        runs = os.path.join(folder, "runs")
        dataset = heightDatasets(folder).get("P1")
        if dataset is None or not os.path.exists(os.path.join(runs, "mrcnnRgb", "results.json")):
            continue
        data = TDData.fromDataset(dataset)
        reference = AnnotatedCrowns(data.annotations)
        blocks = {b.name: b.geometry for b in data.blocks}
        rgb = loadRun(runs, "mrcnnRgb")[1]
        rows = []
        for label, first, second in COMPARISONS:
            if not all(os.path.exists(os.path.join(runs, r, "results.json")) for r in (first, second) if r):
                continue
            if second is None:
                folds = loadRun(runs, first)[1]
            else:
                out = FusionCrossValidation.pair(reference, blocks, RunSet.load(os.path.join(runs, first)),
                                                 RunSet.load(os.path.join(runs, second)),
                                                 strategies=["union"], verbose=False).run()
                folds = {blockKey(f["block"]): f for f in out["union"]["folds"]}
            r = compareFolds(folds, rgb, BOOTSTRAP, rng)
            rows.append({"comparison": label, **r})
            differences[label].append(r["pooledDifference"])
        for r, adjusted in zip(rows, holm([r["p"] for r in rows]) if rows else []):
            r["pHolm"] = float(adjusted)
        perSite.append({"site": name, "comparisons": rows})
        print("  tests %-13s %s" % (name, "  ".join("%s %+.3f%s" % (r["comparison"], r["pooledDifference"],
                                                                 "*" if r["pHolm"] < 0.05 else "") for r in rows)))
    across = []
    for label, values in differences.items():
        if len(values) >= 2:
            p = float(wilcoxon(values).pvalue) if np.any(values) else 1.0
            across.append({"comparison": label, "sites": len(values), "ahead": int(np.sum(np.array(values) > 0)),
                           "meanDifference": round(float(np.mean(values)), 4), "p": p})
    for r, adjusted in zip(across, holm([r["p"] for r in across]) if across else []):
        r["pHolm"] = float(adjusted)
        print("  across sites: %-22s ahead %d/%d  %s" % (r["comparison"], r["ahead"], r["sites"],
                                                       "significant" if r["pHolm"] < 0.05 else "not significant"))
    return {"perSite": perSite, "acrossSites": across}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    parser.add_argument("--site", action="append", default=None, help="name=FOLDER, instead of the default sites")
    parser.add_argument("--output", default=os.path.join(DB, "paperNumbers.json"))
    args = parser.parse_args(argv)
    sites = [tuple(s.split("=", 1)) for s in args.site] if args.site else SITES
    out = {}
    if "sites" in args.parts or "check" in args.parts:
        out["sites"] = []
        for name, folder in sites:
            for source, dataset in sorted(heightDatasets(folder).items()):
                try:
                    row = siteRow(name, folder, source, dataset, "check" in args.parts)
                except Exception as error:                      # one bad site does not stop the rest
                    row = {"site": name, "source": source, "error": repr(error)}
                out["sites"].append(row)
                print("  site %-13s %-5s %s" % (name, source, {k: v for k, v in row.items()
                                                               if k not in ("site", "source", "dataset")}))
    if "alignment" in args.parts:
        out["alignment"] = []
        for name, folder in sites:
            datasets = heightDatasets(folder)
            if name.startswith("terelj") and "LiDAR" in datasets and "P1" in datasets:
                try:
                    row = alignmentRow(name, datasets)
                except Exception as error:
                    row = {"site": name, "error": repr(error)}
                out["alignment"].append(row)
                print("  alignment %-13s %s" % (name, row))
    if "koiwai" in args.parts:
        for name, folder in sites:
            if name == "koiwai" and "P1" in heightDatasets(folder):
                out["koiwai"] = koiwaiRow(name, folder, heightDatasets(folder)["P1"])
                print("  koiwai %s" % out["koiwai"])
    if "tests" in args.parts:
        out["tests"] = testRows(sites)
    with open(args.output, "w") as handle:
        json.dump(out, handle, indent=1, default=float)
    print("written to %s" % args.output)


if __name__ == "__main__":
    main()
