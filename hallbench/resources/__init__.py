"""Provides methods to access various resource files for the application."""

__credits__     = [
    ('PixelMixer', 'Add Icon (from findicons.com)'),
    ('Freepik', 'Air Conditioner Icon (from flaticon.com)'),
    ('Pavel InFeRnODeMoN', 'Check Icon (from findicons.com)'),
    ('GNOME icon artists', 'Clear Icon (from findicons.com)'),    
    ('Momenticons', 'Copy Icon (from findicons.com)'),
    ('FatCow Web Hosting', 'Curve Icon (from findicons.com)'),
    ('eponas-deeway', 'Database Icon (from findicons.com)'),
    ('Momenticons', 'Delete Icon (from findicons.com)'),
    ('Momenticons', 'Font Icon (from findicons.com)'),
    ('AG Multimedia Studio', 'Green Light Icon (from findicons.com)'),
    ('Freepik', 'Hot Water Icon (from flaticon.com)'),
    ('FatCow Web Hosting', 'Map Icon (from findicons.com)'),
    ('Pavel InFeRnODeMoN', 'Minus Icon (from findicons.com)'),
    ('Momenticons', 'Move Icon (from findicons.com)'),
    ('Payungkead', 'Pin Icon (from flaticon.com)'),
    ('Freepik', 'Power Off Icon (from flaticon.com)'),
    ('Freepik', 'Power On Icon (from flaticon.com)'),
    ('AG Multimedia Studio', 'Red Light Icon (from findicons.com)'), 
    ('Momenticons', 'Refresh Icon (from findicons.com)'),
    ('AMAZIGH Aneglus', 'Save Icon (from findicons.com)'),
    ('Momenticons', 'Search Icon (from findicons.com)'),
    ('Momenticons', 'Send Icon (from findicons.com)'),
    ('Momenticons', 'Settings Icon (from findicons.com)'),
    ('PixelMixer', 'Statistics Icon (from findicons.com)'),
    ('Momenticons', 'Stop Icon (from findicons.com)'),
    ('Momenticons', 'Table Icon (from findicons.com)'),
    ('adamwhitcroft.com', 'Wave Icon (from findicons.com)'),
    ]


import os.path as _path

_BASE_PATH = _path.dirname(__file__)


def find(relpath):
    """Look up the resource file based on the relative path to this module.

    Args:
        relpath (str)

    Returns:
        str
    """
    return _path.join(_BASE_PATH, relpath)
