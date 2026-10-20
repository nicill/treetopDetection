"""
A finished run's own options, to run it again on another dataset.

    python -m tt.replayRun RESULTS_JSON MODULE [SUBCOMMAND]
    python -m tt.replayRun runs/mrcnnLidar/results.json tt.dl maskrcnn

prints, as shell words, the options the run recorded in its results.json
('settings', which is vars(args)), minus dataset, output, log, jobs and
progress, which belong to the new run. Every option is checked against
MODULE's --help: one it does not accept stops the replay rather than being
dropped, so a replayed run never silently differs from the original. Options
stored under another name (--noPretrained stores pretrained=False) are
written back under their flag.
"""

import json
import re
import shlex
import subprocess
import sys

SKIP = {"dataset", "output", "log", "jobs", "progress"}
NEGATED = {"pretrained": "--noPretrained"}     # dest of a store_false flag


def accepts(helpText, flag):
    """Whether --help lists flag itself (--minHeight is not --minHeights)."""
    return re.search(re.escape(flag) + r"(?![A-Za-z0-9_])", helpText) is not None


def words(settings, helpText):
    if not accepts(helpText, "--dataset"):
        raise ValueError("the module's --help could not be read")
    out = []
    for key, value in settings.items():
        if key in SKIP or value is None:
            continue
        if key in NEGATED:
            if value is False:
                out.append(NEGATED[key])
            continue
        if value is False:
            continue
        flag = "--" + key
        if not accepts(helpText, flag):
            raise ValueError("%s is not an option of the module" % flag)
        out += [flag] if value is True else [flag, str(value)]
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2:
        print(__doc__)
        return 1
    with open(argv[0]) as handle:
        settings = json.load(handle)["settings"]
    helpText = subprocess.run([sys.executable, "-m"] + argv[1:] + ["--help"],
                              capture_output=True, text=True).stdout
    try:
        print(" ".join(shlex.quote(w) for w in words(settings, helpText)))
    except ValueError as error:
        print("replay: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
