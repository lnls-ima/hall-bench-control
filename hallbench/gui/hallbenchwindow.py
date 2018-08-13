# -*- coding: utf-8 -*-

"""Main window for the Hall Bench Control application."""

import os as _os
from PyQt4.QtGui import (
    QMainWindow as _QMainWindow,
    QApplication as _QApplication,
    QMessageBox as _QMessageBox,
    )
from PyQt4.QtCore import QTimer as _QTimer
import PyQt4.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.connectionwidget import ConnectionWidget \
    as _ConnectionWidget
from hallbench.gui.motorswidget import MotorsWidget as _MotorsWidget
from hallbench.gui.measurementwidget import MeasurementWidget \
    as _MeasurementWidget
from hallbench.gui.databasewidget import DatabaseWidget \
    as _DatabaseWidget
from hallbench.gui.voltageoffsetwidget import VoltageOffsetWidget \
    as _VoltageOffsetWidget
from hallbench.devices.devices import HallBenchDevices as _HallBenchDevices
from hallbench.data.calibration import ProbeCalibration as _ProbeCalibration
from hallbench.data.configuration import (
    MeasurementConfig as _MeasurementConfig)
from hallbench.data.measurement import (
    VoltageData as _VoltageData,
    FieldData as _FieldData,
    Fieldmap as _Fieldmap,
    )


_database_filename = 'hall_bench_measurements.db'


class HallBenchWindow(_QMainWindow):
    """Main Window class for the Hall Bench Control application."""

    _timer_interval = 250  # [ms]

    def __init__(self, parent=None):
        """Set up the ui and add main tabs."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
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

        self.voltageoffset_tab = _VoltageOffsetWidget(self)
        self.ui.main_tab.addTab(self.voltageoffset_tab, 'Voltage Offset')

        self.database_tab = _DatabaseWidget(self)
        self.ui.main_tab.addTab(self.database_tab, 'Database')

        self.timer = _QTimer()
        self.startTimer()

        self.create_database(self.database)
        self.ui.database_le.setText(self.database)

        self.updateMainTabStatus()
        self.ui.main_tab.currentChanged.connect(self.updateDatabaseTab)

    @property
    def voltage_data(self):
        """Measurement voltage data."""
        return self.measurement_tab.voltage_data

    @property
    def field_data(self):
        """Measurement field data."""
        return self.measurement_tab.field_data

    @property
    def fieldmap(self):
        """Measurement fieldmap."""
        return self.measurement_tab.fieldmap

    def create_database(self, database):
        """Create database and tables.

        Args:
            database (str): full file path to database.
        """
        success = _ProbeCalibration.create_database_table(database)
        if not success:
            message = 'Fail to create database table'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        success = _MeasurementConfig.create_database_table(database)
        if not success:
            message = 'Fail to create database table'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        success = _VoltageData.create_database_table(database)
        if not success:
            message = 'Fail to create database table'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        success = _FieldData.create_database_table(database)
        if not success:
            message = 'Fail to create database table'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        success = _Fieldmap.create_database_table(database)
        if not success:
            message = 'Fail to create database table'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

    def closeEvent(self, event):
        """Close main window and dialogs."""
        try:
            self.measurement_tab.closeDialogs()
            self.database_tab.closeDialogs()
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

    def updateDatabaseTab(self):
        """Update database tab."""
        database_tab_idx = self.ui.main_tab.indexOf(self.database_tab)
        if self.ui.main_tab.currentIndex() == database_tab_idx:
            self.database_tab.updateDatabaseTables()

    def updateMainTabStatus(self):
        """Enable or disable main tabs."""
        try:
            _idx = self.ui.main_tab.indexOf(self.motors_tab)
            if _idx != -1:
                self.ui.main_tab.setTabEnabled(
                    _idx, self.devices.pmac.connected)

            _idx = self.ui.main_tab.indexOf(self.measurement_tab)
            if _idx != -1:
                self.ui.main_tab.setTabEnabled(
                    _idx, self.ui.motors_tab.homing)
        except Exception:
            pass
