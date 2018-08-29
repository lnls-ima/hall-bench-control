# -*- coding: utf-8 -*-

"""Current Position widget for the Hall Bench Control application."""

from PyQt4.QtGui import (
    QWidget as _QWidget,
    QApplication as _QApplication,
    )
from PyQt4.QtCore import QTimer as _QTimer
import PyQt4.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile


class CurrentPositionWidget(_QWidget):
    """Current Position Widget class for the Hall Bench Control application."""

    _timer_interval = 250  # [ms]

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        self.timer = _QTimer(self)
        self.startTimer()

    @property
    def pmac(self):
        """Pmac communication class."""
        return _QApplication.instance().devices.pmac

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.stopTimer()
            event.accept()
        except Exception:
            event.accept()

    def startTimer(self):
        """Start timer for interface updates."""
        self.timer.timeout.connect(self.updatePositions)
        self.timer.start(self._timer_interval)

    def stopTimer(self):
        """Stop timer."""
        self.timer.stop()

    def updatePositions(self):
        """Update axes positions."""
        if not self.pmac.connected:
            return

        try:
            for axis in self.pmac.commands.list_of_axis:
                pos = self.pmac.get_position(axis)
                le = getattr(self.ui, 'posax' + str(axis) + '_le')
                le.setText('{0:0.4f}'.format(pos))
            _QApplication.processEvents()
        except Exception:
            pass
