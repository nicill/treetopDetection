"""
Compatibility shim for the benchmark code.

The benchmark predates the package and still expects the two loader functions
that used to live in a flat chmUtils module. Rather than edit it — it works,
and rewriting it onto Scene is a separate job — those two are provided here,
implemented on top of Scene so there is one loader in the codebase and not two
that can drift.
"""

import os

import geopandas as gpd
import numpy as np
import rasterio

from ..scene import Scene, readCrowns


def loadChm(chmPath, targetResolution=0.25, resampling="average",
            minHeight=0.0, verbose=True):
    """The dict the benchmark expects, built from a Scene."""
    scene = Scene(chmPath, resolution=targetResolution,
                  resampling=resampling, minHeight=minHeight, verbose=verbose)
    return {"chm": scene.chm, "pixelSize": scene.pixelSize,
            "transform": scene.transform, "crs": scene.crs,
            "shape": scene.chm.shape, "scene": scene}


def writeLabelRaster(labels, outPath, transform, crs):
    """Save an integer label raster as a GeoTIFF."""
    directory = os.path.dirname(os.path.abspath(outPath))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with rasterio.open(outPath, "w", driver="GTiff",
                       height=labels.shape[0], width=labels.shape[1],
                       count=1, dtype="int32", crs=crs, transform=transform,
                       nodata=0, compress="deflate") as destination:
        destination.write(labels.astype(np.int32), 1)
    return outPath


def loadCrowns(annotationPath, crs, classField="tree_class", plotPath=None,
               shiftEastM=0.0, shiftSouthM=0.0, verbose=True):
    """Crowns loaded by the package's one loader, clipped to a plot file."""
    boundary = None
    if plotPath:
        plots = gpd.read_file(plotPath)
        if plots.crs is not None and crs is not None and plots.crs != crs:
            plots = plots.to_crs(crs)
        boundary = max(plots.geometry, key=lambda g: g.area)
    crowns, _ = readCrowns(annotationPath, crs, boundary=boundary,
                           shiftEastM=shiftEastM, shiftSouthM=shiftSouthM)
    return crowns
