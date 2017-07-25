"""Pmac Communication."""

from __future__ import absolute_import, print_function

import sys
if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    raise ImportError(
        "Python Version 2.6 or above is required for Kulger Bench.")
else:  # Python 3
    from .PmacLib import Pmac
    # Here we can also check for specific Python 3 versions, if needed

del sys
