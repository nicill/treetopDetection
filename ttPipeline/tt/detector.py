"""
Compatibility: the old ConCompDetector interface (Scene in, Tops out) over the
algorithm, which now lives in tt.detectors.concomp. Kept only while the
modules not yet moved to the new classes still import it.
"""

from .detectors.concomp import ConnectedComponents
from .tops import Tops


class ConCompDetector(ConnectedComponents):

    def detect(self, scene):
        return Tops(super().detect(scene))
