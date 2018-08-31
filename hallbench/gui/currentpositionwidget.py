# -*- coding: utf-8 -*-

"""Current Position widget for the Hall Bench Control application."""

from PyQt4.QtGui import (
    QWidget as _QWidget,
    QApplication as _QApplication,
    )
from PyQt4.QtCore import (
    QTimer as _QTimer,
    QThread as _QThread,
    QEventLoop as _QEventLoop,
    )
import PyQt4.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile


class CurrentPositionWidget(_QWidget):
    """Current Position Widget class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        self.positions_thread = PositionsThread(self.ui)
        self.positions_thread.start()

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.positions_thread.quit()
            event.accept()
        except Exception:
            event.accept()


class PositionsThread(_QThread):
    """Thread to read position values from pmac."""

    _timer_interval = 250  # [ms]

    def __init__(self, widget_ui):
        """Initialize object."""
        super().__init__()
        self.widget_ui = widget_ui
        self.timer = _QTimer()
        self.timer.moveToThread(self)
        self.timer.timeout.connect(self.updatePositions)

    @property
    def pmac(self):
        """Pmac communication class."""
        return _QApplication.instance().devices.pmac

    def updatePositions(self):
        """Update axes positions."""
        if not self.pmac.connected:
            return

        try:
            for axis in self.pmac.commands.list_of_axis:
                pos = self.pmac.get_position(axis)
                le = getattr(self.widget_ui, 'posax' + str(axis) + '_le')
                le.setText('{0:0.4f}'.format(pos))
        except Exception:
            pass

    def run(self):
        """Target function."""
        self.timer.start(self._timer_interval)
        loop = _QEventLoop()
        loop.exec_()
