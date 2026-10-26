"""
Tools for the annotations: where crowns are missing, and what is scored.

    candidateTrees    candidate crowns for trees nobody annotated, for the
                      annotator to check and correct
    unannotatedTrees  the canopy no annotated crown covers, as a shapefile and
                      images over the RGB
    crownEdges        canopy left just outside a crown's outline, given back
                      to the crown where it is a thin rim
    edgePatches       the edge part of the unannotated canopy (what the
                      edge-only rule leaves out), shown for the annotator
    edgeOnlyArea      the scored area with only those edge patches cut out
    areaExamples      images of parts left out of the scored area as too
                      little annotated, next to parts that were used

Each runs as python -m tt.annotation.<module>.
"""
