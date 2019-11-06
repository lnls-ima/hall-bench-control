"""Provides methods to access various resource files for the application."""

__credits__     = [
    ('Freepik', 'Air Conditioner Icon (from flaticon.com)'),
    ('AG Multimedia Studio', 'Green Light Icon (from findicons.com)'),
    ('Freepik', 'Hot Water Icon (from flaticon.com)'),
    ('AG Multimedia Studio', 'Red Light Icon (from findicons.com)'), 
    ('play_circle_outline-24px.svg', 'Move Icon (from https://material.io/icons/)'),
    ('stop-circle-outline.svg', 'Stop Icon (from https://materialdesignicons.com/)'),
    ('save-24px.svg', 'Save Icon (from https://material.io/icons/)'),
    ('insert_chart_outlined-24px.svg', 'Stats Icon (from https://material.io/icons/)'),
    ('refresh-24px.svg', 'Refresh Icon (from https://material.io/icons/)'),
    ('close-24px.svg', 'Close Icon (from https://material.io/icons/)'),
    ('database.svg', 'Database Icon (from https://materialdesignicons.com/)'),
    ('database-plus.svg', 'Database Plus Icon (from https://materialdesignicons.com/)'),
    ('settings-outline.svg', 'Settings Icon (from https://materialdesignicons.com/)'),
    ('map-marker.svg', 'Pin Icon (from https://materialdesignicons.com/)'),
    ('broom.svg', 'Clear Icon (from https://materialdesignicons.com/)'),
    ('current-ac.svg', 'Cycle Icon (from https://materialdesignicons.com/)'),
    ('redo.svg', 'Send Icon (from https://materialdesignicons.com/)'),
    ('minus.svg', 'Minus Icon (from https://materialdesignicons.com/)'),
    ('plus.svg', 'Plus Icon (from https://materialdesignicons.com/)'),
    ('file-document-box-outline.svg', 'File Icon (from https://materialdesignicons.com/)'),
    ('close-24px.svg', 'Close Icon (from https://material.io/icons/)'),
    ('zoom_out_map-24px.svg', 'Zoom Icon (from https://material.io/icons/)'),
    ('search-24px.svg', 'Search Icon(from https://material.io/icons/)'),
    ('power-24px.svg', 'Power On Icon (from https://material.io/icons/)'),
    ('power_off-24px.svg', 'Power Off Icon(from https://material.io/icons/)'),
    ('chart-line.svg', 'Plot Icon (from https://materialdesignicons.com/)'),
    ('map-plus.svg', 'Map Plus Icon (from https://materialdesignicons.com/)'),
    ('table-large.svg', 'Table Icon (from https://materialdesignicons.com/)'),
    ('check.svg', 'Check Icon (from https://materialdesignicons.com/)'),
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
