"""Hall Bench Devices Communication."""

import os
import sys

if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    raise ImportError(
        "Python Version 2.6 or above is required for Kulger Bench.")
else:
    from .PmacLib import Pmac
    from .devices import DigitalMultimeter
    from .devices import Multichannel

del os
del sys
