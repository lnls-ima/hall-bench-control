# -*- coding: utf-8 -*-

"""Main window for the Hall probe calibration application."""

from PyQt4.QtGui import QMainWindow as _QMainWindow
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

    def closeEvent(self, event):
        """Close main window and dialogs."""
        try:
            for idx in range(self.ui.main_tab.count()):
                widget = self.ui.main_tab.widget(idx)
                widget.close()
            event.accept()
        except Exception:
            event.accept()

    def updateMainTabStatus(self, tab, status):
        """Enable or disable main tabs."""
        pass
