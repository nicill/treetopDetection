"""
Kept as they were, not maintained and no longer runnable: the classes they
were written against (Scene, ConCompDetector, MethodRun, the dict-based
fusion helpers' callers) have been removed in the redesign. Kept for
reference and so their results can be traced.

    calibration   CC's settings proposed from the RGB Mask R-CNN's confident
                  zones by rules fixed in advance, checked and adjusted inside
                  the zones (no grid search). An alternative to pseudo-tuning
                  (tt.experiments.pseudo), which instead chooses among CC's
                  grid by scoring against the network's boxes. Its results
                  (calibration/calibration.json) are still read by
                  tt.report.studySummary.
    fusion        the dict-based combination helpers calibration used; the
                  maintained strategies are tt.detectors.fusion.
"""
