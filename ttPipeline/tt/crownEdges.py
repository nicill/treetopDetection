"""
Canopy left out of the annotated crowns, added back where it is a thin edge
of a crown. Some manual masks stop short of the canopy: the tree's rim shows
in the CHM as a thin band just outside the outline. Scored as it is, a top on
that rim lands in no crown.

    python -m tt.crownEdges --site Data/qpOut/20230712_afcagauthmelpin_itrf20 \\
        --output Data/qpOutEdges/20230712_afcagauthmelpin_itrf20

Rule, from the CHM and the annotation only (no detector is involved):
  1. uncovered canopy: CHM >= --canopyM and inside no annotated crown
  2. its 8-connected pieces
  3. a piece is a thin edge when every pixel of it lies within --edgeM of
     an annotated crown; a piece reaching further out is something else (most
     likely an unannotated tree) and is left as it is
  4. each pixel of an edge goes to the nearest crown, so an edge between two
     crowns is split between them and no crown grows over another

Writes a copy of the site whose crowns.shp holds the extended crowns (the
CHM, RGB and areas are linked, unchanged), plus:
  crownsOriginal.shp   the annotation as it was
  edges.shp            the pieces added, with the crown each went to
  rejected.shp         uncovered pieces left alone (too thick to be an edge)
  edges.json           counts and areas
"""

import argparse
import json
import os
import shutil
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize, shapes
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt, label
from shapely.geometry import shape
from shapely.ops import unary_union

LINKED = ("chm.tif", "rgb.tif", "area.shp", "scoredArea.shp", "ignored.shp",
          "uncovered.shp", "ground.tif")
SHAPE_PARTS = (".shp", ".shx", ".dbf", ".prj", ".cpg")


def readChm(path, bounds, margin):
    """The CHM over bounds plus a margin, at its native resolution."""
    with rasterio.open(path) as src:
        west, south, east, north = bounds
        window = from_bounds(west - margin, south - margin, east + margin,
                             north + margin, src.transform) \
            .round_offsets().round_lengths()
        chm = src.read(1, window=window, boundless=True, fill_value=0)
        transform = src.window_transform(window)
        nodata = src.nodata
    chm = chm.astype(np.float32)
    if nodata is not None:
        chm[chm == nodata] = 0
    chm[~np.isfinite(chm) | (chm < 0)] = 0
    return chm, transform


def crownIds(crowns, shape_, transform):
    """Raster of crown index + 1, 0 outside every crown."""
    return rasterize(((g, i + 1) for i, g in enumerate(crowns.geometry)
                      if g is not None and not g.is_empty),
                     out_shape=shape_, transform=transform, fill=0,
                     dtype="int32", all_touched=False)


def findEdges(chm, ids, pixel, canopyM, edgeM):
    """
    (added: crown id + 1 per added pixel, 0 elsewhere; rejected: mask of the
    uncovered pieces too thick to be an edge)
    """
    uncovered = (chm >= canopyM) & (ids == 0)
    distance, nearest = distance_transform_edt(ids == 0, sampling=pixel,
                                               return_indices=True)
    nearestId = ids[nearest[0], nearest[1]]
    pieces, count = label(uncovered, structure=np.ones((3, 3), bool))
    if count == 0:
        return np.zeros_like(ids), np.zeros(ids.shape, bool)
    reach = np.zeros(count + 1, np.float32)
    np.maximum.at(reach, pieces.ravel(), distance.ravel())
    thin = reach <= edgeM
    thin[0] = False
    isEdge = thin[pieces]
    added = np.where(isEdge, nearestId, 0).astype(np.int32)
    rejected = uncovered & ~isEdge
    return added, rejected


def polygons(mask, transform, values=None):
    """[(geometry, value)] of the mask's regions (values: an id raster)."""
    source = (values if values is not None else mask.astype(np.int32))
    return [(shape(g), int(v)) for g, v in shapes(source, mask=mask,
                                                  transform=transform)]


def extend(crowns, added, transform):
    """The crowns with their added edge pixels, and the edges as polygons."""
    pieces = polygons(added > 0, transform, added)
    grown = list(crowns.geometry)
    byCrown = {}
    for geometry, crown in pieces:
        byCrown.setdefault(crown - 1, []).append(geometry)
    for index, parts in byCrown.items():
        grown[index] = unary_union([grown[index]] + parts).buffer(0)
    edges = gpd.GeoDataFrame(
        {"crown": [c - 1 for _, c in pieces],
         "area_m2": [round(g.area, 3) for g, _ in pieces]},
        geometry=[g for g, _ in pieces], crs=crowns.crs)
    out = crowns.copy()
    out["geometry"] = grown
    return out, edges, byCrown


def linkSite(site, output):
    """The output site: the unchanged layers linked, the crowns kept aside."""
    os.makedirs(output, exist_ok=True)
    for name in LINKED:
        stem, ext = os.path.splitext(name)
        parts = SHAPE_PARTS if ext == ".shp" else (ext,)
        for part in parts:
            source = os.path.abspath(os.path.join(site, stem + part))
            target = os.path.join(output, stem + part)
            if os.path.exists(source) and not os.path.lexists(target):
                os.symlink(source, target)
    for part in SHAPE_PARTS:
        source = os.path.join(site, "crowns" + part)
        if os.path.exists(source):
            shutil.copyfile(source, os.path.join(output, "crownsOriginal" + part))


def run(args):
    crowns = gpd.read_file(os.path.join(args.site, "crowns.shp"))
    with rasterio.open(os.path.join(args.site, "chm.tif")) as src:
        crowns = crowns.to_crs(src.crs) if crowns.crs != src.crs else crowns
    chm, transform = readChm(os.path.join(args.site, "chm.tif"),
                             crowns.total_bounds, 2 * args.edgeM)
    pixel = abs(transform.a)
    ids = crownIds(crowns, chm.shape, transform)
    added, rejected = findEdges(chm, ids, pixel, args.canopyM, args.edgeM)
    extended, edges, byCrown = extend(crowns, added, transform)
    rejectedPieces = gpd.GeoDataFrame(
        {"area_m2": [round(g.area, 3) for g, _ in polygons(rejected, transform)]},
        geometry=[g for g, _ in polygons(rejected, transform)], crs=crowns.crs)
    linkSite(args.site, args.output)
    extended.to_file(os.path.join(args.output, "crowns.shp"))
    if len(edges):
        edges.to_file(os.path.join(args.output, "edges.shp"))
    if len(rejectedPieces):
        rejectedPieces.to_file(os.path.join(args.output, "rejected.shp"))
    report = {"site": os.path.basename(os.path.normpath(args.site)),
              "edgeM": args.edgeM, "canopyM": args.canopyM, "pixelM": pixel,
              "crowns": int(len(crowns)), "crownsExtended": len(byCrown),
              "edgePieces": int(len(edges)),
              "addedM2": float(edges.area.sum()) if len(edges) else 0.0,
              "crownM2": float(crowns.area.sum()),
              "rejectedPieces": int(len(rejectedPieces)),
              "rejectedM2": float(rejectedPieces.area.sum())
              if len(rejectedPieces) else 0.0}
    with open(os.path.join(args.output, "edges.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print("[edges] %s: %d of %d crowns extended by %d thin edges, %.1f m2 "
          "(%.1f%% of the crown area); %d uncovered pieces left alone, %.1f m2"
          % (report["site"], report["crownsExtended"], report["crowns"],
             report["edgePieces"], report["addedM2"],
             100 * report["addedM2"] / max(report["crownM2"], 1e-9),
             report["rejectedPieces"], report["rejectedM2"]))
    return report


def parseArguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--site", required=True,
                        help="a prepared site (chm.tif, crowns.shp, ...)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--edgeM", type=float, default=0.5,
                        help="an uncovered piece is an edge when all of it is "
                             "within this distance of a crown")
    parser.add_argument("--canopyM", type=float, default=0.5,
                        help="CHM height from which a pixel is canopy")
    return parser.parse_args(argv)


def main(argv=None):
    run(parseArguments(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
