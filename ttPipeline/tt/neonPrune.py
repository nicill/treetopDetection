"""
Remove NeonTreeEvaluation files the pipeline will never use.

    python -m tt.neonPrune --annotations annotations --roots evaluation training
    python -m tt.neonPrune --annotations annotations --roots evaluation training --delete

A file is kept when it belongs to an annotated tile: its name is the tile's
name, or starts with it followed by "_" (BART_036_2019.tif, BART_036_2019_CHM.tif).
For the training crops the parent 1 km tile counts too
(2018_NIWO_2_450000_4426000_image_crop.xml keeps 2018_NIWO_2_450000_4426000_*),
since their CHM and LiDAR may be named after the parent tile.

Always removed: everything under a Hyperspectral folder. LiDAR point clouds of
annotated tiles are kept unless --dropLidar.

Without --delete nothing is removed: the run lists what would go and the space
it would free. Annotated tiles with no CHM found are reported either way.
"""

import argparse
import os
import re
import sys

from tt.neonImport import readBoxes

PARENT_TILE = re.compile(r"^\d{4}_[A-Z]{4}_\d+_\d+_\d+")
HYPERSPECTRAL = "hyperspectral"
LIDAR = "lidar"
CHM = "chm"


def annotatedKeys(annotationDir):
    """Names a kept file may start with: tile stems and their parent tiles."""
    keys = set()
    for name in os.listdir(annotationDir):
        if not name.endswith(".xml"):
            continue
        stem = os.path.splitext(readBoxes(os.path.join(annotationDir, name))[0])[0]
        keys.add(stem)
        parent = PARENT_TILE.match(stem)
        if parent:
            keys.add(parent.group(0))
    return keys


def belongs(fileName, keys):
    stem = os.path.splitext(fileName)[0]
    return any(stem == k or stem.startswith(k + "_") for k in keys)


def folderKind(path):
    parts = [p.lower() for p in path.split(os.sep)]
    for kind in (HYPERSPECTRAL, LIDAR, CHM):
        if kind in parts:
            return kind
    return None


def decide(path, keys, dropLidar):
    """True when the file should be removed."""
    kind = folderKind(os.path.dirname(path))
    if kind == HYPERSPECTRAL:
        return True
    if kind == LIDAR and dropLidar:
        return True
    return not belongs(os.path.basename(path), keys)


def survey(roots, keys, dropLidar):
    """Every file under the roots, split into (remove, keep)."""
    remove, keep = [], []
    for root in roots:
        for folder, _, files in os.walk(root):
            for name in files:
                path = os.path.join(folder, name)
                (remove if decide(path, keys, dropLidar) else keep).append(path)
    return remove, keep


def missingChm(keys, keep):
    """Annotated tile stems with no kept file under a CHM folder."""
    chmNames = [os.path.basename(p) for p in keep
                if folderKind(os.path.dirname(p)) == CHM]
    stems = {k for k in keys if not PARENT_TILE.fullmatch(k)}
    return sorted(k for k in stems
                  if not any(belongs(n, {k}) or belongs(n, parents(k))
                             for n in chmNames))


def parents(stem):
    match = PARENT_TILE.match(stem)
    return {match.group(0)} if match else set()


def gigabytes(paths):
    return sum(os.path.getsize(p) for p in paths) / 1e9


def report(remove, keep, keys):
    print("[prune] keep %d files (%.2f GB), remove %d files (%.2f GB)"
          % (len(keep), gigabytes(keep), len(remove), gigabytes(remove)))
    for stem in missingChm(keys, keep):
        print("[prune] WARNING no CHM found for annotated tile %s" % stem)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Remove NEON files the pipeline will not use.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--roots", nargs="+", required=True,
                        help="Unpacked evaluation and/or training folders")
    parser.add_argument("--dropLidar", action="store_true",
                        help="Also remove point clouds of annotated tiles")
    parser.add_argument("--delete", action="store_true",
                        help="Actually remove; without it, only report")
    args = parser.parse_args(argv)
    keys = annotatedKeys(args.annotations)
    remove, keep = survey(args.roots, keys, args.dropLidar)
    report(remove, keep, keys)
    if not args.delete:
        for path in remove[:20]:
            print("  would remove %s" % path)
        print("[prune] dry run; add --delete to remove")
        return 0
    for path in remove:
        os.remove(path)
    print("[prune] removed %d files" % len(remove))
    return 0


if __name__ == "__main__":
    sys.exit(main())
