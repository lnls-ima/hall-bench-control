# -*- coding: utf-8 -*-

"""Utils."""

import sys as _sys
import numpy as _np
import pandas as _pd
import os.path as _path
import traceback as _traceback
from qtpy.QtCore import Qt as _Qt
from qtpy.QtWidgets import (
    QDialog as _QDialog,
    QComboBox as _QComboBox,
    QListView as _QListView,
    QVBoxLayout as _QVBoxLayout,
    QPushButton as _QPushButton,
    QTableWidget as _QTableWidget,
    QTableWidgetItem as _QTableWidgetItem,
    )
from qtpy.QtGui import QStandardItemModel as _QStandardItemModel
from matplotlib.figure import Figure as _Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as _FigureCanvas,
    NavigationToolbar2QT as _Toolbar
    )


_basepath = _path.dirname(_path.abspath(__file__))


def getUiFile(widget):
    """Get the ui file path.

    Args:
        widget  (QWidget or class)
    """
    if isinstance(widget, type):
        basename = '%s.ui' % widget.__name__.lower()
    else:
        basename = '%s.ui' % widget.__class__.__name__.lower()
    uifile = _path.join(_basepath, _path.join('ui', basename))

    return uifile