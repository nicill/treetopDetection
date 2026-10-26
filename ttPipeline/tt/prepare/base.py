"""
SitePreparer — turns a site's raw data into the products every other part of
the package reads: chm.tif, rgb.tif, crowns.shp, area.shp and check.json
(what was found, and the checks made on it), one folder per site.

A preparer says which sites it has and how to prepare one; the loop over
them (skipping sites already prepared, carrying on past a site that fails
and naming it at the end) is the same for all.
"""

import os


class SitePreparer(object):

    tag = "prepare"            # what its messages start with
    skipPrepared = True        # a site with a check.json is not redone

    def __init__(self, args):
        self.args = args

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, self.args.output)

    def sites(self):
        """The sites to prepare."""
        raise NotImplementedError

    def siteDir(self, site):
        return os.path.join(self.args.output, site)

    def isPrepared(self, site):
        return os.path.exists(os.path.join(self.siteDir(site), "check.json"))

    def missingInput(self, site):
        """Why the site cannot be prepared, or None."""
        return None

    def prepareSite(self, site):
        """Prepare one site; returns its check."""
        raise NotImplementedError

    def recheckSite(self, site):
        """The checks again from a prepared site's products."""
        raise NotImplementedError("%s cannot recheck" % type(self).__name__)

    def run(self, recheck=False):
        """Every site; 1 if any failed (run again to retry them), else 0."""
        failed = []
        for site in self.sites():
            prepared = self.isPrepared(site)
            if recheck:
                if prepared:
                    self.recheckSite(site)
                continue
            if prepared and self.skipPrepared and \
                    not getattr(self.args, "redo", False):
                print("[%s] %s: already prepared, skipped" % (self.tag, site))
                continue
            missing = self.missingInput(site)
            if missing:
                print("[%s] %s: %s, skipped" % (self.tag, site, missing))
                failed.append(site)
                continue
            try:
                self.prepareSite(site)
            except (RuntimeError, OSError, ValueError) as error:
                print("[%s] %s: FAILED, skipped: %s" % (self.tag, site, error),
                      flush=True)
                failed.append(site)
        if failed:
            print("[%s] %d site(s) not done; run the same command again to "
                  "retry them: %s" % (self.tag, len(failed), " ".join(failed)),
                  flush=True)
        return 1 if failed else 0
