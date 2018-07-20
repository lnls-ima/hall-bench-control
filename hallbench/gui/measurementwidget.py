# -*- coding: utf-8 -*-

"""Measurement widget for the Hall Bench Control application."""

import os.path as _path
import time as _time
import numpy as _np
import threading as _threading
import warnings as _warnings
from PyQt5.QtWidgets import (
    QWidget as _QWidget,
    QApplication as _QApplication,
    QVBoxLayout as _QVBoxLayout,
    QMessageBox as _QMessageBox,
    )
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.configurationwidgets import ConfigurationWidget \
    as _ConfigurationWidget
from hallbench.gui.currentpositionwidget import CurrentPositionWidget \
    as _CurrentPositionWidget
from hallbench.gui.fieldmapdialog import FieldMapDialog \
    as _FieldMapDialog
from hallbench.data.measurement import VoltageData as _VoltageData
from hallbench.data.measurement import FieldData as _FieldData


class MeasurementWidget(_QWidget):
    """Measurement widget class for the Hall Bench Control application."""

    _update_graph_time_interval = 0.05  # [s]
    _measurement_axes = [1, 2, 3, 5]

    def __init__(self, parent=None):
        """Set up the ui, add widgets and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # add configuration widget
        self.configuration_widget = _ConfigurationWidget(self)
        _layout = _QVBoxLayout()
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.addWidget(self.configuration_widget)
        self.ui.configuration_wg.setLayout(_layout)

        # add position widget
        self.current_position_widget = _CurrentPositionWidget(self)
        _layout = _QVBoxLayout()
        _layout.addWidget(self.current_position_widget)
        self.ui.position_wg.setLayout(_layout)

        # create dialog
        self.fieldmap_dialog = _FieldMapDialog()

        # variables initialization
        self.threadx = None
        self.thready = None
        self.threadz = None
        self.config = None
        self.config_id = None
        self.probe_calibration = None
        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.position_list = []
        self.scan_id_list = []
        self.field_data_list = []
        self.field_data = None
        self.voltage_data = None
        self.stop = False

        self.connectSignalSlots()

    @property
    def devices(self):
        """Hall Bench Devices."""
        return self.window().devices

    @property
    def database(self):
        """Database filename."""
        return self.window().database

    def clear(self):
        """Clear."""
        self.threadx = None
        self.thready = None
        self.threadz = None
        self.config = None
        self.config_id = None
        self.probe_calibration = None
        self.position_list = []
        self.scan_id_list = []
        self.field_data_list = []
        self.field_data = None
        self.voltage_data = None
        self.stop = False
        self.clearGraph()

    def clearGraph(self):
        """Clear plots."""
        self.ui.graph_pw.plotItem.curves.clear()
        self.ui.graph_pw.clear()
        self.graphx = []
        self.graphy = []
        self.graphz = []

    def closeDialogs(self):
        """Close dialogs."""
        try:
            self.fieldmap_dialog.accept()
            self.configuration_widget.closeDialogs()
        except Exception:
            pass

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.measure_btn.clicked.connect(self.configureAndMeasure)
        self.ui.stop_btn.clicked.connect(self.stopMeasurement)
        self.ui.savefieldmap_btn.clicked.connect(self.showFieldMapDialog)

    def configureAndMeasure(self):
        """Configure devices and start measurement."""
        self.clear()

        if (self.devices is None
           or not self.updateProbeCalibration()
           or not self.updateConfiguration()
           or not self.configureDevices()
           or not self.validDatabase()
           or not self.saveConfiguration()):
            return

        try:
            self.ui.stop_btn.setEnabled(True)

            first_axis = self.config.first_axis
            second_axis = self.config.second_axis
            fixed_axes = self.getFixedAxes()

            for axis in fixed_axes:
                if self.stop is True:
                    return
                pos = self.config.get_start(axis)
                self.moveAxis(axis, pos)

            if second_axis != -1:
                start = self.config.get_start(second_axis)
                end = self.config.get_end(second_axis)
                step = self.config.get_step(second_axis)
                npts = _np.ceil(round((end - start) / step, 4) + 1)
                pos_list = _np.linspace(start, end, npts)

                for pos in pos_list:
                    if self.stop is True:
                        return
                    self.moveAxis(second_axis, pos)
                    if not self.scanLine(first_axis):
                        return
                    self.field_data_list.append(self.field_data)

                # move to initial position
                if self.stop is True:
                    return
                second_axis_start = self.config.get_start(second_axis)
                self.moveAxis(second_axis, second_axis_start)

            else:
                if self.stop is True:
                    return
                if not self.scanLine(first_axis):
                    return
                self.field_data_list.append(self.field_data)

            # move to initial position
            if self.stop is True:
                return
            first_axis_start = self.config.get_start(first_axis)
            self.moveAxis(first_axis, first_axis_start)
            self.plotField()
            self.ui.stop_btn.setEnabled(False)
            self.ui.savefieldmap_btn.setEnabled(True)

            message = 'End of measurements.'
            _QMessageBox.information(
                self, 'Measurements', message, _QMessageBox.Ok)

        except Exception as e:
            _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)

    def configureDevices(self):
        """Configure devices."""
        try:
            if self.config.voltx_enable:
                self.devices.voltx.config(
                    self.config.integration_time,
                    self.config.voltage_precision)
            if self.config.volty_enable:
                self.devices.volty.config(
                    self.config.integration_time,
                    self.config.voltage_precision)
            if self.config.voltz_enable:
                self.devices.voltz.config(
                    self.config.integration_time,
                    self.config.voltage_precision)

            self.devices.pmac.set_axis_speed(1, self.config.vel_ax1)
            self.devices.pmac.set_axis_speed(2, self.config.vel_ax2)
            self.devices.pmac.set_axis_speed(3, self.config.vel_ax3)
            self.devices.pmac.set_axis_speed(5, self.config.vel_ax5)
            return True
        except Exception:
            message = 'Failed to configure devices.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return False

    def configureGraph(self, nr_curves, label):
        """Configure graph.

        Args:
            nr_curves (int): number of curves to plot.
        """
        self.graphx = []
        self.graphy = []
        self.graphz = []

        for idx in range(nr_curves):
            self.graphx.append(
                self.ui.graph_pw.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(255, 0, 0),
                    symbol='o',
                    symbolPen=(255, 0, 0),
                    symbolSize=4))

            self.graphy.append(
                self.ui.graph_pw.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(0, 255, 0),
                    symbol='o',
                    symbolPen=(0, 255, 0),
                    symbolSize=4))

            self.graphz.append(
                self.ui.graph_pw.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(0, 0, 255),
                    symbol='o',
                    symbolPen=(0, 0, 255),
                    symbolSize=4))

        self.ui.graph_pw.setLabel('bottom', 'Scan Position [mm]')
        self.ui.graph_pw.setLabel('left', label)
        self.ui.graph_pw.showGrid(x=True, y=True)

    def getFixedAxes(self):
        """Get fixed axes."""
        first_axis = self.config.first_axis
        second_axis = self.config.second_axis
        fixed_axes = [a for a in self._measurement_axes]
        fixed_axes.remove(first_axis)
        if second_axis != -1:
            fixed_axes.remove(second_axis)
        return fixed_axes

    def killReadingThreads(self):
        """Kill threads."""
        try:
            del self.threadx
            del self.thready
            del self.threadz
        except Exception:
            pass

    def moveAxis(self, axis, position):
        """Move bench axis.

        Args:
            axis (int): axis number.
            position (float): target position.
        """
        if self.stop is False:
            self.devices.pmac.move_axis(axis, position)
            while ((self.devices.pmac.axis_status(axis) & 1) == 0 and
                   self.stop is False):
                _QApplication.processEvents()

    def moveAxisAndUpdateGraph(self, axis, position, idx):
        """Move bench axis and update plot with the measure data.

        Args:
            axis (int): axis number.
            position (float): target position.
            idx (int):  index of the plot to update.
        """
        if self.stop is False:
            self.devices.pmac.move_axis(axis, position)
            while ((self.devices.pmac.axis_status(axis) & 1) == 0 and
                   self.stop is False):
                self.plotVoltage(idx)
                _QApplication.processEvents()
                _time.sleep(self._update_graph_time_interval)

    def plotField(self):
        """Plot field data."""
        self.clearGraph()
        nr_curves = len(self.field_data_list)
        self.configureGraph(nr_curves, 'Magnetic Field [T]')

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            for i in len(nr_curves):
                fd = self.field_data_list[i]
                positions = fd.scan_positions
                self.graphx[i].setData(positions, fd.avgx)
                self.graphy[i].setData(positions, fd.avgy)
                self.graphz[i].setData(positions, fd.avgz)

    def plotVoltage(self, idx):
        """Plot voltage values.

        Args:
            idx (int): index of the plot to update.
        """
        voltagex = self.devices.voltx.voltage
        voltagey = self.devices.volty.voltage
        voltagez = self.devices.voltz.voltage

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            self.graphx[idx].setData(
                self.position_list[:len(voltagex)], voltagex)
            self.graphy[idx].setData(
                self.position_list[:len(voltagey)], voltagey)
            self.graphz[idx].setData(
                self.position_list[:len(voltagez)], voltagez)

    def saveConfiguration(self):
        """Save configuration to database table."""
        try:
            self.config_id = self.config.save_to_database(self.database)
            return True
        except Exception as e:
            _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)
            return False

    def saveScan(self):
        """Save field data to database table."""
        if self.field_data is None:
            message = 'Invalid field data.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return False

        try:
            idn = self.field_data.save_to_database(
                self.database, self.config_id)
            self.scan_id_list.append(idn)
            return True

        except Exception as e:
            _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)
            return False

    def scanLine(self, first_axis):
        """Start line scan."""
        self.field_data = _FieldData()

        start = self.config.get_start(first_axis)
        end = self.config.get_end(first_axis)
        step = self.config.get_step(first_axis)
        extra = self.config.get_extra(first_axis)
        vel = self.config.get_velocity(first_axis)

        aper_displacement = self.config.integration_time * vel
        npts = _np.ceil(round((end - start) / step, 4) + 1)
        scan_list = _np.linspace(start, end, npts)
        to_pos_scan_list = scan_list + aper_displacement/2
        to_neg_scan_list = (scan_list - aper_displacement/2)[::-1]

        nr_measurements = self.config.nr_measurements
        self.configureGraph(nr_measurements, 'Voltage [V]')

        voltage_data_list = []
        for idx in range(nr_measurements):
            if self.stop is True:
                return False

            self.voltage_data = _VoltageData()
            self.devices.voltx.end_measurement = False
            self.devices.volty.end_measurement = False
            self.devices.voltz.end_measurement = False
            self.devices.voltx.clear()
            self.devices.volty.clear()
            self.devices.voltz.clear()
            self.ui.nr_measurements_la.setText(str(idx + 1))

            # flag to check if sensor is going or returning
            to_pos = not(bool(idx % 2))

            # go to initial position
            if to_pos:
                self.position_list = to_pos_scan_list
                self.moveAxis(first_axis, start - extra)
            else:
                self.position_list = to_neg_scan_list
                self.moveAxis(first_axis, end + extra)

            for axis in self.voltage_data.axis_list:
                if axis != first_axis:
                    pos = self.devices.pmac.get_position(axis)
                    setattr(self.voltage_data, 'pos' + str(axis), pos)
                else:
                    setattr(self.voltage_data, 'pos' + str(first_axis),
                            self.position_list)

            if self.stop is True:
                return False

            # start threads
            self.startReadingThreads()

            # start firstger measurement
            if self.stop is False:
                if to_pos:
                    self.devices.pmac.set_trigger(
                        first_axis, start, step, 10, npts, 1)
                    self.moveAxisAndUpdateGraph(first_axis, end + extra, idx)
                else:
                    self.devices.pmac.set_trigger(
                        first_axis, end, step*(-1), 10, npts, 1)
                    self.moveAxisAndUpdateGraph(first_axis, start - extra, idx)

            # stop trigger and copy voltage values to voltage data
            self.stopTrigger()
            self.waitReadingThreads()
            self.voltage_data.sensorx = self.devices.voltx.voltage
            self.voltage_data.sensory = self.devices.volty.voltage
            self.voltage_data.sensorz = self.devices.voltz.voltage
            self.killReadingThreads()

            if self.stop is True:
                return False

            if not to_pos:
                self.voltage_data.reverse()
            voltage_data_list.append(self.voltage_data.copy())

        self.field_data.set_field_data(
            voltage_data_list, self.measurement_probe_calibration)
        success = self.saveScan()
        return success

    def showFieldMapDialog(self):
        """Open fieldmap dialog."""
        self.fieldmap_dialog.show(
            self.field_data_list,
            self.measurement_probe_calibration,
            self.database,
            self.scan_id_list)

    def startReadingThreads(self):
        """Start threads to read voltage values."""
        if self.config.voltx_enable:
            self.threadx = _threading.Thread(
                target=self.devices.voltx.read,
                args=(self.config.voltage_precision,))
            self.threadx.start()

        if self.config.volty_enable:
            self.thready = _threading.Thread(
                target=self.devices.volty.read,
                args=(self.config.voltage_precision,))
            self.thready.start()

        if self.config.voltz_enable:
            self.threadz = _threading.Thread(
                target=self.devices.voltz.read,
                args=(self.config.voltage_precision,))
            self.threadz.start()

    def stopMeasurement(self):
        """Stop measurement to True."""
        try:
            self.stop = True
            self.devices.pmac.stop_all_axis()
            self.ui.stop_btn.setEnabled(False)
            message = 'The user stopped the measurements.'
            _QMessageBox.information(
                self, 'Abort', message, _QMessageBox.Ok)

        except Exception as e:
            _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)

    def stopTrigger(self):
        """Stop trigger."""
        self.devices.pmac.stop_trigger()
        self.devices.voltx.end_measurement = True
        self.devices.volty.end_measurement = True
        self.devices.voltz.end_measurement = True

    def updateConfiguration(self):
        """Update measurement configuration."""
        success = self.configuration_widget.updateConfiguration()
        if success:
            self.config = self.configuration_widget.config
            return True
        else:
            self.config = None
            return False

    def updatePositions(self):
        """Update axes positions."""
        self.current_position_widget.updatePositions()

    def updateProbeCalibration(self):
        """Update probe calibration."""
        self.probe_calibration = self.configuration_widget.probe_calibration
        if self.probe_calibration is not None:
            return True
        else:
            message = 'Invalid probe calibration.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return False

    def validDatabase(self):
        """Check if the database filename is valid."""
        if (self.database is not None
           and len(self.database) != 0 and _path.isfile(self.database)):
            return True
        else:
            message = 'Invalid database filename.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return False

    def waitReadingThreads(self):
        """Wait threads."""
        if self.threadx is not None:
            while self.threadx.is_alive() and self.stop is False:
                _QApplication.processEvents()

        if self.thready is not None:
            while self.thready.is_alive() and self.stop is False:
                _QApplication.processEvents()

        if self.threadz is not None:
            while self.threadz.is_alive() and self.stop is False:
                _QApplication.processEvents()
