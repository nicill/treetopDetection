"""
Compatibility shim: the old detector's interface over the new one.

The benchmark constructs a detector, calls loadChm() then detect(), and reads
.tops and .transform. The new ConCompDetector separates the scene from the
detector, so this adapter holds a Scene internally and presents the old shape.
Merge settings arrive as the old flat keywords and are folded into a TopMerger.
"""

import rasterio

from ..detector import ConCompDetector
from ..merging import TopMerger
from ..scene import Scene


class ConCompTreetopDetection(object):

    def __init__(self, chmPath, outputDir=None, targetResolution=0.25,
                 minHeight=2.0, windowSizeM=40.0, windowOverlap=0.2,
                 lowerPercentile=20, upperPercentile=99,
                 erosionKernelSize=3, erosionIterations=1,
                 minTreeAreaM2=0.5, minTopAreaM2=0.25, topStepM=0.25,
                 refineRadiusM=0.35, mergeMetric="saddle", mergeEpsM=8.0,
                 heightWeight=0.7, saddleDropM=0.3, mergeMode="distance",
                 plotRasterPath=None, refine=True, verbose=True, **ignored):
        self.chmPath = chmPath
        self.targetResolution = targetResolution
        self.minHeight = minHeight
        self.plotRasterPath = plotRasterPath
        self.verbose = verbose

        self.detector = ConCompDetector(
            windowSizeM=windowSizeM, windowOverlap=windowOverlap,
            lowerPercentile=lowerPercentile, upperPercentile=upperPercentile,
            erosionKernelSize=erosionKernelSize,
            erosionIterations=erosionIterations,
            minTreeAreaM2=minTreeAreaM2, minTopAreaM2=minTopAreaM2,
            topStepM=topStepM, refine=refine, verbose=verbose,
            merger=TopMerger(metric=mergeMetric,
                             epsM=mergeEpsM if mergeEpsM else 2 * refineRadiusM,
                             heightWeight=heightWeight,
                             saddleDropM=saddleDropM))
        self.scene = None
        self.tops = []

    @property
    def transform(self):
        return self.scene.transform

    @property
    def chm(self):
        return self.scene.chm

    @property
    def pixelSize(self):
        return self.scene.pixelSize

    @property
    def crs(self):
        return self.scene.crs

    def loadChm(self):
        self.scene = Scene(self.chmPath, resolution=self.targetResolution,
                           minHeight=self.minHeight, restrict=False,
                           verbose=self.verbose)
        if self.plotRasterPath:
            self._restrictToRaster(self.plotRasterPath)
        return self.scene.chm

    def _restrictToRaster(self, path):
        with rasterio.open(path) as source:
            labels = source.read(1)
        if labels.shape != self.scene.chm.shape:
            raise ValueError("plot raster is %s but the CHM grid is %s"
                             % (labels.shape, self.scene.chm.shape))
        self.scene.chm[labels == 0] = 0.0
        self.scene.regionMask = labels > 0

    def detect(self):
        if self.scene is None:
            self.loadChm()
        found = self.detector.detect(self.scene)
        self.tops = found.points
        return self.tops
