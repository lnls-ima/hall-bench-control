# -*- coding: utf-8 -*-

"""Current Position widget for the Hall Bench Control application."""

from PyQt5.QtWidgets import (
    QWidget as _QWidget,
    QApplication as _QApplication,
    )
from PyQt5.QtCore import QTimer as _QTimer
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile


class CurrentPositionWidget(_QWidget):
    """Current Position Widget class for the Hall Bench Control application."""

    _list_of_axis = [1, 2, 3, 5, 6, 7, 8, 9]
    _timer_interval = 250  # [ms]

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        self.timer = _QTimer()
        self.timer.timeout.connect(self.updatePositions)
        self.timer.start(self._timer_interval)

    @property
    def positions(self):
        """Current posiitons."""
        return _QApplication.instance().positions

    def closeEvent(self, event):
        """Stop timer and close widget."""
        try:
            self.timer.stop()
            event.accept()
        except Exception:
            event.accept()

    def updatePositions(self):
        """Update positions."""
        try:
            for axis in self._list_of_axis:
                if axis in self.positions:
                    pos = self.positions[axis]
                    le = getattr(self.ui, 'posax' + str(axis) + '_le')
                    le.setText('{0:0.4f}'.format(pos))
        except Exception:
            pass