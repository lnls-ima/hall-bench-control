"""Hall Bench Control Application."""

import os as _os

with open(_os.path.join(__path__[0], 'VERSION'), 'r') as _f:
    __version__ = _f.read().strip()

__all__ = (
    'calibration',
    'data',
    'devices',
    'resources',
    'gui',
    )
