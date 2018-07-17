# -*- coding: utf-8 -*-

"""Main window for the Hall Bench Control application."""

import os as _os
from PyQt5.QtWidgets import (
    QMainWindow as _QMainWindow,
    QApplication as _QApplication,
    )
from PyQt5.QtCore import QTimer as _QTimer
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.connectionwidget import ConnectionWidget \
    as _ConnectionWidget
from hallbench.gui.motorswidget import MotorsWidget as _MotorsWidget
from hallbench.gui.measurementwidget import MeasurementWidget \
    as _MeasurementWidget
from hallbench.devices.devices import HallBenchDevices as _HallBenchDevices
from hallbench.data.database import create_database as _create_database


_database_filename = 'hall_bench_measurements.db'


class HallBenchWindow(_QMainWindow):
    """Main Window class for the Hall Bench Control application."""

    _timer_interval = 250  # [ms]

    def __init__(self, parent=None):
        """Set up the ui and add main tabs."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(__file__, self)
        self.ui = _uic.loadUi(uifile, self)

        # clear the current tabs
        self.ui.main_tab.clear()

        # variables initialization
        _default_directory = _os.path.dirname(_os.path.dirname(
            _os.path.dirname(_os.path.abspath(__file__))))
        self.database = _os.path.join(_default_directory, _database_filename)
        self.devices = _HallBenchDevices()

        # add tabs
        self.connection_tab = _ConnectionWidget(self)
        self.ui.main_tab.addTab(self.connection_tab, 'Connection')

        self.motors_tab = _MotorsWidget(self)
        self.ui.main_tab.addTab(self.motors_tab, 'Motors')

        self.measurement_tab = _MeasurementWidget(self)
        self.ui.main_tab.addTab(self.measurement_tab, 'Measurement')

        self.timer = _QTimer()

        self.ui.database_le.setText(self.database)
        _create_database(self.database)

        self.updateMainTabStatus()

    @property
    def voltage_data(self):
        """Measurement voltage data."""
        return self.measurement_tab.voltage_data

    @property
    def field_data(self):
        """Measurement field data."""
        return self.measurement_tab.field_data

    @property
    def fieldmap_data(self):
        """Measurement field map data."""
        return self.measurement_tab.fieldmap_data

    def closeEvent(self, event):
        """Close main window and dialogs."""
        try:
            self.measurement_tab.closeDialogs()
            self.stopTimer()
            event.accept()
        except Exception:
            event.accept()

    def refreshInterface(self):
        """Read probes positions and update the interface."""
        try:
            self.motors_tab.updatePositions()
            self.measurement_tab.updatePositions()
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

    def updateMainTabStatus(self):
        """Enable or disable main tabs."""
        try:
            _idx = self.ui.main_tab.indexOf(self.ui.motors_tab)
            if _idx != -1:
                self.ui.main_tab.setTabEnabled(
                    _idx, self.devices.pmac.connected)

            _idx = self.ui.main_tab.indexOf(self.ui.measurement_tab)
            if _idx != -1:
                self.ui.main_tab.setTabEnabled(
                    _idx, self.ui.motors_tab.homing)
        except Exception:
            pass
