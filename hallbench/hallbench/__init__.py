"""Hall Bench Package."""

from __future__ import absolute_import, print_function

import os
import sys

with open(os.path.join(__path__[0], 'VERSION'), 'r') as _f:
    __version__ = _f.read().strip()

if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    raise ImportError(
        "Python Version 2.6 or above is required for Kulger Bench.")
else:  # Python 3
    from . import GPIB
    from . import Pmac
    from . import calibration
    from . import configuration
    from . import devices
    from . import measurement

del os
del sys
