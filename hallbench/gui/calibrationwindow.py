# -*- coding: utf-8 -*-

"""Main window for the Hall probe calibration application."""

from PyQt4.QtGui import (
    QMainWindow as _QMainWindow,
    QApplication as _QApplication,
    )
from PyQt4.QtCore import QTimer as _QTimer
import PyQt4.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.connectionwidget import ConnectionWidget \
    as _ConnectionWidget
from hallbench.gui.motorswidget import MotorsWidget as _MotorsWidget
from hallbench.gui.voltageoffsetwidget import VoltageOffsetWidget \
    as _VoltageOffsetWidget
from hallbench.devices.devices import HallBenchDevices as _HallBenchDevices


class CalibrationWindow(_QMainWindow):
    """Main Window class for the Hall probe calibration application."""

    _timer_interval = 250  # [ms]

    def __init__(self, parent=None):
        """Set up the ui and add main tabs."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # clear out the current tabs
        self.ui.main_tab.clear()

        # variables initialization
        self.devices = _HallBenchDevices()

        # add tabs
        self.connection_tab = _ConnectionWidget(self)
        self.ui.main_tab.addTab(self.connection_tab, 'Connection')

        self.motors_tab = _MotorsWidget(self)
        self.ui.main_tab.addTab(self.motors_tab, 'Motors')

        self.voltagetoffset_tab = _VoltageOffsetWidget(self)
        self.ui.main_tab.addTab(self.voltagetoffset_tab, 'Voltage Offset')

        # create timer
        self.timer = _QTimer()

    def closeEvent(self, event):
        """Close main window and dialogs."""
        self.stopTimer()
        event.accept()

    def refreshInterface(self):
        """Read probes positions and update the interface."""
        try:
            self.motors_tab.updatePositions()
            _QApplication.processEvents()
        except Exception:
            pass

    def startTimer(self):
        """Start timer for interface updates."""
        self.timer.timeout.connect(self.refreshInterface)
        self.timer.start(self._timer_interval)

    def stopTimer(self):
        """Stop timer."""
        self.timer.stop()

    def updateMainTabStatus(self, tab, status):
        """Enable or disable main tabs."""
        self.ui.main_tab.setTabEnabled(tab, status)
