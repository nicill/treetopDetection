"""
Site preparation: raw data to the products the rest of the package reads.

    quebec                 QuebecPreparer: Quebec Plantations, from the server
    quebecPhotogrammetry   QuebecPhotogrammetryPreparer: their P1 CHMs
    sergi                  SergiPreparer: Koiwai, from a photogrammetric cloud
    enshurin               EnshurinPreparer: Enshurin's annotated parts
    groundModel            GroundModel: a CHM from a DSM alone
    dsmChm                 a CHM from a DSM on an existing site's grid
    labelImport            crowns from a label raster

Each runs as python -m tt.prepare.<module>.
"""
