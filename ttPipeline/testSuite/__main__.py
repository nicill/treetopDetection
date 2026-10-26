"""
python -m testSuite: every test (from the folder holding tt/ and testSuite/).

The real-data regression test runs only when TT_CHM, TT_CROWNS and
TT_BOUNDARY point at the published site; the torch tests only with torch.
"""

import os
import sys
import unittest

here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, here)
suite = unittest.defaultTestLoader.discover(os.path.join(here, "testSuite"),
                                            top_level_dir=here)
result = unittest.TextTestRunner(verbosity=1).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
