# -*- coding: utf-8 -*-

"""Main window for the Hall Bench Control application."""

import os as _os
from PyQt5.QtWidgets import (
    QMainWindow as _QMainWindow,
    QApplication as _QApplication,
    )
from PyQt5.QtCore import QTimer as _QTimer
import PyQt5.uic as _uic

from hallbench.gui.connectionwidget import ConnectionWidget \
    as _ConnectionWidget
from hallbench.gui.measurementwidget import MeasurementWidget \
    as _MeasurementWidget
from hallbench.gui.setdirectorydialog import SetDirectoryDialog \
    as _SetDirectoryDialog
from hallbench.gui.recoverdatadialog import RecoverDataDialog \
    as _RecoverDataDialog
from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.motorswidget import MotorsWidget as _MotorsWidget
from hallbench.devices.devices import HallBenchDevices as _HallBenchDevices


class HallBenchWindow(_QMainWindow):
    """Main Window class for the Hall Bench Control application."""

    _timer_interval = 250  # [ms]

    def __init__(self, parent=None):
        """Set up the ui and add main tabs."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(__file__, self)
        self.ui = _uic.loadUi(uifile, self)

        # clear out the current tabs
        self.ui.main_tab.clear()

        base_directory = _os.path.split(_os.path.dirname(__file__))[0]
        default_directory = _os.path.join(base_directory, 'measurements')
        if not _os.path.isdir(default_directory):
            try:
                _os.mkdir(default_directory)
            except Exception:
                default_directory = None

        # variables initialization
        self.directory = default_directory
        self.devices = _HallBenchDevices()
        self.database = None

        # create dialogs
        self.directory_dialog = _SetDirectoryDialog()
        self.recoverdata_dialog = _RecoverDataDialog()

        # add tabs
        self.connection_tab = _ConnectionWidget(self)
        self.ui.main_tab.addTab(self.connection_tab, 'Connection')

        self.motors_tab = _MotorsWidget(self)
        self.ui.main_tab.addTab(self.motors_tab, 'Motors')

        self.measurement_tab = _MeasurementWidget(self)
        self.ui.main_tab.addTab(self.measurement_tab, 'Measurement')

        # create timer
        self.timer = _QTimer()

        # create connections
        self.ui.setdir_act.triggered.connect(self.showDirectoryDialog)
        self.ui.savevoltage_act.triggered.connect(self.setSaveVoltageFlag)
        self.ui.savefield_act.triggered.connect(self.setSaveFieldFlag)
        self.ui.recoverdata_act.triggered.connect(self.showRecoverDataDialog)

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
        self.directory_dialog.close()
        self.recoverdata_dialog.close()
        self.measurement_tab.closeDialogs()
        self.stopTimer()
        event.accept()

    def refreshInterface(self):
        """Read probes positions and update the interface."""
        try:
            self.motors_tab.updatePositions()
            self.measurement_tab.updatePositions()
            _QApplication.processEvents()
        except Exception:
            pass

    def showDirectoryDialog(self):
        """Show set directory dialog."""
        self.directory_dialog.show(self.directory)
        self.directory_dialog.directoryChanged.connect(self.updateDirectory)

    def updateDirectory(self, directory):
        """Update directory."""
        self.directory = directory

    def showRecoverDataDialog(self):
        """Show recover data dialog."""
        self.recoverdata_dialog.show(directory=self.directory)

    def setSaveFieldFlag(self):
        """Set save configuration flag."""
        if self.ui.savefield_act.isChecked():
            self.save_field = True
        else:
            self.save_field = False

    def setSaveVoltageFlag(self):
        """Set save voltage flag."""
        if self.ui.savevoltage_act.isChecked():
            self.save_voltage = True
        else:
            self.save_voltage = False

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
