"""
Entry point for the benchmark, so its scripts can be run as modules now that
they live inside the package and use relative imports.

    python -m tt.dl prepare   --source ...
    python -m tt.dl maskrcnn  --dataset ...
    python -m tt.dl yolo      --dataset ...
    python -m tt.dl concomp   --dataset ...
    python -m tt.dl benchmark --auto ...
    python -m tt.dl export    --dataset ...
"""

import sys


COMMANDS = {
    "prepare": ("dlPrepare", "main"),
    "maskrcnn": ("dlMaskRcnn", "crossValidate"),
    "yolo": ("dlYolo", "crossValidate"),
    "concomp": ("dlConComp", "crossValidate"),
    "benchmark": ("dlBenchmark", "main"),
    "export": ("dlExportToFramework", "main"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        print("commands: %s" % ", ".join(sorted(COMMANDS)))
        return 1

    moduleName, functionName = COMMANDS[argv[0]]
    module = __import__("tt.dl." + moduleName, fromlist=[functionName])
    sys.argv = ["tt.dl " + argv[0]] + argv[1:]

    if functionName == "crossValidate":
        parse = getattr(module, "parseArguments")
        return 0 if getattr(module, functionName)(parse()) else 1
    return getattr(module, functionName)()


if __name__ == "__main__":
    sys.exit(main())
