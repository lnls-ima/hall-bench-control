# -*- coding: utf-8 -*-

"""Utils."""

import os.path as _path


_basepath = _path.dirname(_path.abspath(__file__))


def getUiFile(widget):
    """Get the ui file path.

    Args:
        widget  (QWidget)
    """
    basename = '%s.ui' % widget.__class__.__name__.lower()
    uifile = _path.join(_basepath, _path.join('ui', basename))

    return uifile
