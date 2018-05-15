# -*- coding: utf-8 -*-

"""Utils."""

import os.path as _path


def getUiFile(modpath, widget):
    """Get the ui file path.

    Args:
        modpath (str)
        widget  (QWidget)
    """
    # generate the uifile path
    basepath = _path.dirname(modpath)
    basename = '%s.ui' % widget.__class__.__name__.lower()
    uifile = _path.join(basepath, _path.join('ui', basename))

    return uifile
