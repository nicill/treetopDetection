"""
NeonTreeEvaluation boxes converted to what the pipeline reads: a crown
shapefile and a boundary shapefile per annotated tile.

    python -m tt.neonImport --annotations annotations --rgb evaluation/RGB \\
        --output Data/neon

The annotation folder is searched at any depth, so the folder the zip unpacked
into works as it is. Each annotation XML (Pascal VOC) names its RGB tile and holds one bounding
box per visible tree, in pixel coordinates of that tile. The tile's GeoTIFF
transform turns the boxes into map polygons. The boundary is the tile's
footprint: the evaluation plots are annotated over the whole tile. The large
training tiles may be annotated over only part of it, so check those before
scoring against the footprint.

Output, one folder per tile, named after the tile:
    <output>/<tile>/crowns.shp    one box polygon per tree
    <output>/<tile>/area.shp      the tile footprint
    <output>/manifest.csv         tile, RGB path, crown count, both shapefiles

The CHM is not touched: the NEON CHM is a separate 1 m raster covering the
same ground, and Scene reprojects crowns and boundary onto whatever grid it
is given.
"""

import argparse
import csv
import os
import sys
import xml.etree.ElementTree as ET

import geopandas as gpd
import rasterio
from shapely.geometry import box

BOX_KEYS = ("xmin", "ymin", "xmax", "ymax")
MAC_JUNK = "__MACOSX"


def annotationFiles(annotationDir):
    """Every annotation XML under the folder, at any depth, sorted."""
    return sorted(os.path.join(folder, name)
                  for folder, _, names in os.walk(annotationDir)
                  for name in names
                  if name.endswith(".xml") and not name.startswith("._")
                  and MAC_JUNK not in folder.split(os.sep))


def readBoxes(xmlPath):
    """The RGB file name the annotation belongs to, and its pixel boxes."""
    root = ET.parse(xmlPath).getroot()
    name = root.find("filename").text
    boxes = [tuple(float(o.find("bndbox/" + key).text) for key in BOX_KEYS)
             for o in root.findall("object")]
    return name, boxes


def mapBox(pixelBox, transform):
    """A pixel box as a polygon in map coordinates."""
    xmin, ymin, xmax, ymax = pixelBox
    x0, y0 = transform @ (xmin, ymin)
    x1, y1 = transform @ (xmax, ymax)
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def tileFrames(pixelBoxes, source):
    """Crown and boundary frames for one tile."""
    crowns = gpd.GeoDataFrame(
        {"crownId": range(1, len(pixelBoxes) + 1)},
        geometry=[mapBox(b, source.transform) for b in pixelBoxes],
        crs=source.crs)
    crowns["area"] = crowns.geometry.area
    boundary = gpd.GeoDataFrame({"name": ["area"]},
                                geometry=[box(*source.bounds)], crs=source.crs)
    return crowns, boundary


def convertTile(xmlPath, rgbDir, outputDir):
    """One annotation to its two shapefiles; None when it cannot be used."""
    name, pixelBoxes = readBoxes(xmlPath)
    rgbPath = os.path.join(rgbDir, name)
    if not pixelBoxes or not os.path.exists(rgbPath):
        return None
    with rasterio.open(rgbPath) as source:
        crowns, boundary = tileFrames(pixelBoxes, source)
    tileDir = os.path.join(outputDir, os.path.splitext(name)[0])
    os.makedirs(tileDir, exist_ok=True)
    crownPath = os.path.join(tileDir, "crowns.shp")
    boundaryPath = os.path.join(tileDir, "area.shp")
    crowns.to_file(crownPath)
    boundary.to_file(boundaryPath)
    return {"tile": os.path.splitext(name)[0], "rgb": rgbPath,
            "crowns": len(crowns), "crownPath": crownPath,
            "boundaryPath": boundaryPath}


def writeManifest(rows, outputDir):
    path = os.path.join(outputDir, "manifest.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def convert(annotationDir, rgbDir, outputDir):
    os.makedirs(outputDir, exist_ok=True)
    paths = annotationFiles(annotationDir)
    if not paths:
        raise SystemExit("no annotation XML found under %s" % annotationDir)
    rows = [r for r in (convertTile(p, rgbDir, outputDir) for p in paths) if r]
    print("[neon] %d of %d annotations matched an RGB tile in %s"
          % (len(rows), len(paths), rgbDir))
    if not rows:
        return None
    print("[neon] %d crowns -> %s" % (sum(r["crowns"] for r in rows),
                                      writeManifest(rows, outputDir)))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="NeonTreeEvaluation boxes to crown and boundary "
                    "shapefiles, one folder per tile.")
    parser.add_argument("--annotations", required=True,
                        help="Folder of Pascal VOC XML files")
    parser.add_argument("--rgb", required=True,
                        help="Folder of the RGB tiles the XML files name")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    return 0 if convert(args.annotations, args.rgb, args.output) else 1


if __name__ == "__main__":
    sys.exit(main())
