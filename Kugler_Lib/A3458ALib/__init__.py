"""
Kulger Bench - Voltimeter Agilent 3458A
"""

from __future__ import absolute_import, print_function

__version__ = "0.1"

import sys
if sys.version_info[0] == 2 and sys.version_info[1] < 6:
    raise ImportError("Python Version 2.6 or above is required for Kulger Bench.")
else:  # Python 3
    pass
    # Here we can also check for specific Python 3 versions, if needed

del sys

from .Agilent_3458AVars import *

