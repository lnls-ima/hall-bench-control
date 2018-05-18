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
from hallbench.gui.calibrationwidget import CalibrationWidget \
    as _CalibrationWidget
from hallbench.gui.measurementwidget import MeasurementWidget \
    as _MeasurementWidget
from hallbench.gui.setdirectorydialog import SetDirectoryDialog \
    as _SetDirectoryDialog
from hallbench.gui.recoverdatadialog import RecoverDataDialog \
    as _RecoverDataDialog
from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.motorswidget import MotorsWidget as _MotorsWidget
from hallbench.devices.GPIBLib import Agilent3458A as _Agilent3458A
from hallbench.devices.GPIBLib import Agilent34970A as _Agilent34970A
from hallbench.devices.PmacLib import Pmac as _Pmac


class HallBenchWindow(_QMainWindow):
    """Main Window class for the Hall Bench Control application."""

    _timer_interval = 250  # [ms]

    def __init__(self, parent=None):
        """Setup the ui and add main tabs."""
        super(HallBenchWindow, self).__init__(parent)

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
        self.save_voltage = True
        self.save_field = True
        self.probe_calibration = None
        self.devices = HallBenchDevices()

        # create dialogs
        self.directory_dialog = _SetDirectoryDialog()
        self.recoverdata_dialog = _RecoverDataDialog()

        # add tabs
        self.connection_tab = _ConnectionWidget(self)
        self.ui.main_tab.addTab(self.connection_tab, 'Connection')

        self.motors_tab = _MotorsWidget(self)
        self.ui.main_tab.addTab(self.motors_tab, 'Motors')

        self.calibration_tab = _CalibrationWidget(self)
        self.ui.main_tab.addTab(self.calibration_tab, 'Probe Calibration')

        self.measurement_tab = _MeasurementWidget(self)
        self.ui.main_tab.addTab(self.measurement_tab, 'Measurement')

        # create timer
        self.timer = _QTimer()

        # create connections
        self.ui.setdir_act.triggered.connect(self.showDirectoryDialog)
        self.ui.savevoltage_act.triggered.connect(self.setSaveVoltageFlag)
        self.ui.savefield_act.triggered.connect(self.setSaveFieldFlag)

        self.ui.recoverdata_act.triggered.connect(self.showRecoverDataDialog)

        self.measurement_tab.measure_btn.clicked.connect(
            self.saveConfigurations)

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
        self.calibration_tab.closeDialogs()
        self.measurement_tab.closeDialogs()
        event.accept()

    def refreshInterface(self):
        """Read probes positions and update the interface."""
        try:
            self.motors_tab.updatePositions()
            self.measurement_tab.updatePositions()
            _QApplication.processEvents()
        except Exception:
            pass

    def saveConfigurations(self):
        """Save configuration files."""
        if self.ui.saveconfig_act.isChecked():
            self.ui.connection_tab.saveConfigurationInMeasurementsDir()
            self.ui.measurement_tab.saveConfigurationInMeasurementsDir()

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


class HallBenchDevices(object):
    """Hall Bench Devices."""

    def __init__(self):
        """Initiate variables."""
        self.pmac = None
        self.voltx = None
        self.volty = None
        self.voltz = None
        self.multich = None
        self.colimator = None
        self.loaded = False

    def clearMultimetersData(self):
        """Clear multimeters stored data and update measurement flags."""
        if not self.loaded:
            return
        self.voltx.end_measurement = False
        self.volty.end_measurement = False
        self.voltz.end_measurement = False
        self.voltx.clear()
        self.volty.clear()
        self.voltz.clear()

    def configurePmacTrigger(self, axis, pos, step, npts):
        """Configure Pmac trigger."""
        self.pmac.set_trigger(axis, pos, step, 10, npts, 1)

    def connect(self, configuration):
        """Connect devices.

        Args:
            configuration (ConnectionConfig): connection configuration.
        """
        if not self.loaded:
            return [False]*5

        status = []
        if configuration.control_voltx_enable:
            status.append(self.voltx.connect(configuration.control_voltx_addr))

        if configuration.control_volty_enable:
            status.append(self.volty.connect(configuration.control_volty_addr))

        if configuration.control_voltz_enable:
            status.append(self.voltz.connect(configuration.control_voltz_addr))

        if configuration.control_pmac_enable:
            status.append(self.pmac.connect())

        if configuration.control_multich_enable:
            status.append(
                self.multich.connect(configuration.control_multich_addr))

        return status

    def disconnect(self):
        """Disconnect devices."""
        if not self.loaded:
            return [True]*5

        status = []
        status.append(self.voltx.disconnect())
        status.append(self.volty.disconnect())
        status.append(self.voltz.disconnect())
        status.append(self.pmac.disconnect())
        status.append(self.multich.disconnect())

        return status

    def initialMeasurementConfiguration(self, configuration):
        """Initial measurement configuration.

        Args:
            configuration (MeasurementConfig): measurement configuration.
        """
        if configuration.meas_probeX:
            self.voltx.config(
                configuration.meas_aper_ms, configuration.meas_precision)
        if configuration.meas_probeY:
            self.volty.config(
                configuration.meas_aper_ms, configuration.meas_precision)
        if configuration.meas_probeZ:
            self.voltz.config(
                configuration.meas_aper_ms, configuration.meas_precision)

        self.pmac.set_axis_speed(1, configuration.meas_vel_ax1)
        self.pmac.set_axis_speed(2, configuration.meas_vel_ax2)
        self.pmac.set_axis_speed(3, configuration.meas_vel_ax3)
        self.pmac.set_axis_speed(5, configuration.meas_vel_ax5)

    def load(self):
        """Load devices."""
        try:
            self.pmac = _Pmac('pmac.log')
            self.voltx = _Agilent3458A('voltx.log')
            self.volty = _Agilent3458A('volty.log')
            self.voltz = _Agilent3458A('voltz.log')
            self.multich = _Agilent34970A('multi.log')
            self.loaded = True
        except Exception:
            self.loaded = False

    def stopTrigger(self):
        """Stop Pmac trigger and update measurement flags."""
        self.pmac.stop_trigger()
        self.voltx.end_measurement = True
        self.volty.end_measurement = True
        self.voltz.end_measurement = True
