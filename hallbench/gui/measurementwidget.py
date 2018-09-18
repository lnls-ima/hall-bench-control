# -*- coding: utf-8 -*-

"""Measurement widget for the Hall Bench Control application."""

import sys as _sys
import os as _os
import time as _time
import numpy as _np
import warnings as _warnings
import pyqtgraph as _pyqtgraph
import traceback as _traceback
from PyQt5.QtWidgets import (
    QWidget as _QWidget,
    QFileDialog as _QFileDialog,
    QApplication as _QApplication,
    QVBoxLayout as _QVBoxLayout,
    QMessageBox as _QMessageBox,
    )
from PyQt5.QtCore import (
    QTimer as _QTimer,
    QThread as _QThread,
    QEventLoop as _QEventLoop,
    pyqtSignal as _pyqtSignal,
    )
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.configurationwidget import ConfigurationWidget \
    as _ConfigurationWidget
from hallbench.gui.currentpositionwidget import CurrentPositionWidget \
    as _CurrentPositionWidget
from hallbench.gui.savefieldmapdialog import SaveFieldmapDialog \
    as _SaveFieldmapDialog
from hallbench.gui.viewscandialog import ViewScanDialog as _ViewScanDialog
from hallbench.data.configuration import MeasurementConfig \
    as _MeasurementConfig
from hallbench.data.measurement import VoltageScan as _VoltageScan
from hallbench.data.measurement import FieldScan as _FieldScan


class MeasurementWidget(_QWidget):
    """Measurement widget class for the Hall Bench Control application."""

    change_current_setpoint = _pyqtSignal([bool])

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
        self.save_fieldmap_dialog = _SaveFieldmapDialog()
        self.view_scan_dialog = _ViewScanDialog()

        self.measurement_configured = False
        self.local_measurement_config = None
        self.local_measurement_config_id = None
        self.local_hall_probe = None
        self.threadx = None
        self.thready = None
        self.threadz = None
        self.voltage_scan = None
        self.field_scan = None
        self.voltage_scan_list = []
        self.field_scan_list = []
        self.position_list = []
        self.field_scan_id_list = []
        self.voltage_scan_id_list = []
        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.stop = False

        self.connectSignalSlots()
        self.current_temperature_thread = CurrentTemperatureThread()

        # Add legend to plot
        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.graph_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

    @property
    def database(self):
        """Database filename."""
        return _QApplication.instance().database

    @property
    def devices(self):
        """Hall Bench Devices."""
        return _QApplication.instance().devices

    @property
    def directory(self):
        """Return the default directory."""
        return _QApplication.instance().directory

    @property
    def hall_probe(self):
        """Hall probe calibration data."""
        return _QApplication.instance().hall_probe

    @property
    def measurement_config(self):
        """Measurement configuration."""
        return _QApplication.instance().measurement_config

    @property
    def power_supply_config(self):
        """Power supply configuration."""
        return _QApplication.instance().power_supply_config

    def clear(self):
        """Clear."""
        self.threadx = None
        self.thready = None
        self.threadz = None
        self.measurement_configured = False
        self.local_measurement_config = None
        self.local_measurement_config_id = None
        self.local_hall_probe = None
        self.voltage_scan = None
        self.field_scan = None
        self.voltage_scan_list = []
        self.field_scan_list = []
        self.position_list = []
        self.field_scan_id_list = []
        self.voltage_scan_id_list = []
        self.stop = False
        self.clearGraph()
        self.ui.view_scan_btn.setEnabled(False)
        self.ui.clear_graph_btn.setEnabled(False)
        self.ui.create_fieldmap_btn.setEnabled(False)
        self.ui.save_scan_files_btn.setEnabled(False)

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
            self.current_position_widget.close()
            self.save_fieldmap_dialog.accept()
            self.view_scan_dialog.accept()
            self.configuration_widget.closeDialogs()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.closeDialogs()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.measure_btn.clicked.connect(self.measureButtonClicked)
        self.ui.stop_btn.clicked.connect(self.stopMeasurement)
        self.ui.create_fieldmap_btn.clicked.connect(self.showFieldmapDialog)
        self.ui.save_scan_files_btn.clicked.connect(self.saveScanFiles)
        self.ui.view_scan_btn.clicked.connect(self.showViewScanDialog)
        self.ui.clear_graph_btn.clicked.connect(self.clearGraph)

    def configureGraph(self, nr_curves, label):
        """Configure graph.

        Args:
            nr_curves (int): number of curves to plot.
        """
        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.legend.removeItem('X')
        self.legend.removeItem('Y')
        self.legend.removeItem('Z')

        for idx in range(nr_curves):
            self.graphx.append(
                self.ui.graph_pw.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(255, 0, 0),
                    symbol='o',
                    symbolPen=(255, 0, 0),
                    symbolSize=4,
                    symbolBrush=(255, 0, 0)))

            self.graphy.append(
                self.ui.graph_pw.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(0, 255, 0),
                    symbol='o',
                    symbolPen=(0, 255, 0),
                    symbolSize=4,
                    symbolBrush=(0, 255, 0)))

            self.graphz.append(
                self.ui.graph_pw.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(0, 0, 255),
                    symbol='o',
                    symbolPen=(0, 0, 255),
                    symbolSize=4,
                    symbolBrush=(0, 0, 255)))

        self.ui.graph_pw.setLabel('bottom', 'Scan Position [mm]')
        self.ui.graph_pw.setLabel('left', label)
        self.ui.graph_pw.showGrid(x=True, y=True)
        self.legend.addItem(self.graphx[0], 'X')
        self.legend.addItem(self.graphy[0], 'Y')
        self.legend.addItem(self.graphz[0], 'Z')

    def configureMeasurement(self):
        """Configure measurement."""
        self.clear()

        if (not self.validDatabase()
           or not self.updateHallProbe()
           or not self.updateConfiguration()
           or not self.multimetersConnected()
           or not self.configurePmac()
           or not self.configureMultichannel()):
            self.measurement_configured = False
        else:
            self.measurement_configured = True

    def configureMultichannel(self):
        """Configure multichannel to monitor dcct current and temperatures."""
        if (not self.ui.save_temperature_chb.isChecked() and
           not self.ui.save_current_chb.isChecked()):
            return True

        if not self.devices.multich.connected:
            msg = 'Multichannel not connected.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        try:
            channels = []
            if self.ui.save_temperature_chb.isChecked():
                channels = channels + self.devices.multich.probe_channels
                channels = channels + self.devices.multich.temperature_channels
            if self.ui.save_current_chb.isChecked():
                channels = channels + self.devices.multich.dcct_channels

            if len(channels) == 0:
                return True

            self.devices.multich.configure(channel_list=channels)
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to configure multichannel.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def configurePmac(self):
        """Configure devices."""
        if not self.devices.pmac.connected:
            msg = 'Pmac not connected.'
            _QMessageBox.critical(
                self, 'Failure', msg, _QMessageBox.Ok)
            return False

        try:
            self.devices.pmac.set_axis_speed(
                1, self.local_measurement_config.vel_ax1)
            self.devices.pmac.set_axis_speed(
                2, self.local_measurement_config.vel_ax2)
            self.devices.pmac.set_axis_speed(
                3, self.local_measurement_config.vel_ax3)
            self.devices.pmac.set_axis_speed(
                5, self.local_measurement_config.vel_ax5)
            return True
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to configure devices.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def createVoltageThreads(self):
        """Start threads to read voltage values."""
        if self.local_measurement_config.voltx_enable:
            self.threadx = VoltageThread(
                self.devices.voltx,
                self.local_measurement_config.voltage_precision)
        else:
            self.threadx = None

        if self.local_measurement_config.volty_enable:
            self.thready = VoltageThread(
                self.devices.volty,
                self.local_measurement_config.voltage_precision)
        else:
            self.thready = None

        if self.local_measurement_config.voltz_enable:
            self.threadz = VoltageThread(
                self.devices.voltz,
                self.local_measurement_config.voltage_precision)
        else:
            self.threadz = None

    def endAutomaticMeasurements(self, setpoint_changed):
        """End automatic measurements."""
        if not self.resetMultimeters():
            return

        if not setpoint_changed:
            msg = ('Automatic measurements failed. ' +
                   'Current setpoint not changed.')
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        self.ui.stop_btn.setEnabled(False)
        self.ui.measure_btn.setEnabled(True)
        self.ui.save_scan_files_btn.setEnabled(True)
        self.ui.view_scan_btn.setEnabled(True)
        self.ui.clear_graph_btn.setEnabled(True)

        msg = 'End of automatic measurements.'
        _QMessageBox.information(
            self, 'Measurements', msg, _QMessageBox.Ok)

    def getFixedAxes(self):
        """Get fixed axes."""
        first_axis = self.local_measurement_config.first_axis
        second_axis = self.local_measurement_config.second_axis
        fixed_axes = [a for a in self._measurement_axes]
        fixed_axes.remove(first_axis)
        if second_axis != -1:
            fixed_axes.remove(second_axis)
        return fixed_axes

    def killVoltageThreads(self):
        """Kill threads."""
        try:
            del self.threadx
            del self.thready
            del self.threadz
            self.threadx = None
            self.thready = None
            self.threadz = None
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def measure(self):
        """Perform one measurement."""
        if not self.measurement_configured:
            return False

        try:
            first_axis = self.local_measurement_config.first_axis
            second_axis = self.local_measurement_config.second_axis
            fixed_axes = self.getFixedAxes()

            self.createVoltageThreads()

            for axis in fixed_axes:
                if self.stop is True:
                    return
                pos = self.local_measurement_config.get_start(axis)
                self.moveAxis(axis, pos)

            if second_axis != -1:
                start = self.local_measurement_config.get_start(second_axis)
                end = self.local_measurement_config.get_end(second_axis)
                step = self.local_measurement_config.get_step(second_axis)
                npts = _np.abs(_np.ceil(round((end - start) / step, 4) + 1))
                pos_list = _np.linspace(start, end, npts)

                for pos in pos_list:
                    if self.stop is True:
                        return
                    self.moveAxis(second_axis, pos)
                    if not self.scanLine(first_axis):
                        return
                    self.field_scan_list.append(self.field_scan)

                # move to initial position
                if self.stop is True:
                    return
                second_axis_start = self.local_measurement_config.get_start(
                    second_axis)
                self.moveAxis(second_axis, second_axis_start)

            else:
                if self.stop is True:
                    return
                if not self.scanLine(first_axis):
                    return
                self.field_scan_list.append(self.field_scan)

            self.current_temperature_thread.quit()

            if self.stop is True:
                return

            # move to initial position
            first_axis_start = self.local_measurement_config.get_start(
                first_axis)
            self.moveAxis(first_axis, first_axis_start)

            self.plotField()
            self.killVoltageThreads()
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.killVoltageThreads()
            self.current_temperature_thread.quit()
            msg = 'Measurement failure.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def measureAndEmitSignal(self, current_setpoint):
        """Measure and emit signal to change current setpoint."""
        if not self.configuration_widget.setMainCurrent(current_setpoint):
            return

        if not self.saveConfiguration():
            return

        if not self.measure():
            return
        self.change_current_setpoint.emit(True)

    def measureButtonClicked(self):
        if self.ui.automatic_current_ramp_chb.isChecked():
            self.startAutomaticMeasurements()
        else:
            self.startMeasurement()

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

    def multimetersConnected(self):
        """Check if multimeters are connected."""
        if self.local_measurement_config is None:
            return False

        if (self.local_measurement_config.voltx_enable
           and not self.devices.voltx.connected):
                msg = 'Multimeter X not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
                return False

        if (self.local_measurement_config.volty_enable
           and not self.devices.volty.connected):
                msg = 'Multimeter Y not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
                return False

        if (self.local_measurement_config.voltz_enable
           and not self.devices.voltz.connected):
                msg = 'Multimeter Z not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
                return False

        return True

    def plotField(self):
        """Plot field scan."""
        if self.local_hall_probe is None:
            return

        self.clearGraph()
        nr_curves = len(self.field_scan_list)
        self.configureGraph(nr_curves, 'Magnetic Field [T]')

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            for i in range(nr_curves):
                fd = self.field_scan_list[i]
                positions = fd.scan_pos
                self.graphx[i].setData(positions, fd.avgx)
                self.graphy[i].setData(positions, fd.avgy)
                self.graphz[i].setData(positions, fd.avgz)

    def plotVoltage(self, idx):
        """Plot voltage values.

        Args:
            idx (int): index of the plot to update.
        """
        if self.threadx is not None:
            voltagex = [v for v in self.threadx.voltage]
        else:
            voltagex = []

        if self.thready is not None:
            voltagey = [v for v in self.thready.voltage]
        else:
            voltagey = []

        if self.threadz is not None:
            voltagez = [v for v in self.threadz.voltage]
        else:
            voltagez = []

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            self.graphx[idx].setData(
                self.position_list[:len(voltagex)], voltagex)
            self.graphy[idx].setData(
                self.position_list[:len(voltagey)], voltagey)
            self.graphz[idx].setData(
                self.position_list[:len(voltagez)], voltagez)

    def resetMultimeters(self):
        """Reset connected multimeters."""
        try:
            if self.devices.voltx.connected:
                self.devices.voltx.reset()

            if self.devices.volty.connected:
                self.devices.volty.reset()

            if self.devices.voltz.connected:
                self.devices.voltz.reset()

            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to reset multimeters.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def saveConfiguration(self):
        """Save configuration to database table."""
        try:
            text = self.configuration_widget.ui.idn_cmb.currentText()
            if len(text) != 0:
                selected_idn = int(text)
                selected_config = _MeasurementConfig(
                    database=self.database, idn=selected_idn)
            else:
                selected_config = _MeasurementConfig()

            if self.local_measurement_config == selected_config:
                idn = selected_idn
            else:
                idn = self.local_measurement_config.save_to_database(
                    self.database)
                self.configuration_widget.updateConfigurationIDs()
                idx = self.configuration_widget.ui.idn_cmb.findText(str(idn))
                self.configuration_widget.ui.idn_cmb.setCurrentIndex(idx)
                self.configuration_widget.ui.loaddb_btn.setEnabled(False)

            self.local_measurement_config_id = idn
            return True
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save configuration to database.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def saveFieldScan(self):
        """Save field scan to database table."""
        if self.field_scan is None:
            msg = 'Invalid field scan.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        try:
            mn = self.local_measurement_config.magnet_name
            mc = self.local_measurement_config.main_current
            self.field_scan.magnet_name = mn
            self.field_scan.main_current = mc
            self.field_scan.configuration_id = self.local_measurement_config_id
            idn = self.field_scan.save_to_database(self.database)
            self.field_scan_id_list.append(idn)
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save FieldScan to database'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def saveScanFiles(self):
        """Save scan files."""
        scan_list = self.field_scan_list
        scan_id_list = self.field_scan_id_list

        directory = _QFileDialog.getExistingDirectory(
            self, caption='Save scan files', directory=self.directory)

        if isinstance(directory, tuple):
            directory = directory[0]

        if len(directory) == 0:
            return

        try:
            for i, scan in enumerate(scan_list):
                idn = scan_id_list[i]
                default_filename = scan.default_filename
                if '.txt' in default_filename:
                    default_filename = default_filename.replace(
                        '.txt', '_ID={0:d}.txt'.format(idn))
                elif '.dat' in default_filename:
                    default_filename = default_filename.replace(
                        '.dat', '_ID={0:d}.dat'.format(idn))
                default_filename = _os.path.join(directory, default_filename)
                scan.save_file(default_filename)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save files.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def saveVoltageScan(self):
        """Save voltage scan to database table."""
        if self.voltage_scan is None:
            msg = 'Invalid voltage scan.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        try:
            mn = self.local_measurement_config.magnet_name
            mc = self.local_measurement_config.main_current
            self.voltage_scan.magnet_name = mn
            self.voltage_scan.main_current = mc
            self.voltage_scan.configuration_id = (
                self.local_measurement_config_id)
            idn = self.voltage_scan.save_to_database(self.database)
            self.voltage_scan_id_list.append(idn)
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save VoltageScan to database'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def scanLine(self, first_axis):
        """Start line scan."""
        self.field_scan = _FieldScan()

        start = self.local_measurement_config.get_start(first_axis)
        end = self.local_measurement_config.get_end(first_axis)
        step = self.local_measurement_config.get_step(first_axis)
        extra = self.local_measurement_config.get_extra(first_axis)
        vel = self.local_measurement_config.get_velocity(first_axis)

        aper_displacement = self.local_measurement_config.integration_time*vel
        npts = _np.ceil(round((end - start) / step, 4) + 1)
        scan_list = _np.linspace(start, end, npts)
        to_pos_scan_list = scan_list + aper_displacement/2
        to_neg_scan_list = (scan_list - aper_displacement/2)[::-1]

        nr_measurements = self.local_measurement_config.nr_measurements
        self.clearGraph()
        self.configureGraph(2*nr_measurements, 'Voltage [V]')

        voltage_scan_list = []
        for idx in range(2*nr_measurements):
            if self.stop is True:
                return False

            self.voltage_scan = _VoltageScan()
            self.configuration_widget.nr_measurements_sb.setValue(
                _np.ceil((idx + 1)/2))

            # flag to check if sensor is going or returning
            to_pos = not(bool(idx % 2))

            # go to initial position
            if to_pos:
                self.position_list = to_pos_scan_list
                self.moveAxis(first_axis, start - extra)
            else:
                self.position_list = to_neg_scan_list
                self.moveAxis(first_axis, end + extra)

            for axis in self.voltage_scan.axis_list:
                if axis != first_axis:
                    pos = self.devices.pmac.get_position(axis)
                    setattr(self.voltage_scan, 'pos' + str(axis), pos)
                else:
                    setattr(self.voltage_scan, 'pos' + str(first_axis),
                            self.position_list)

            if self.stop is True:
                return False
            else:
                if to_pos:
                    self.devices.pmac.set_trigger(
                        first_axis, start, step, 10, npts, 1)
                else:
                    self.devices.pmac.set_trigger(
                        first_axis, end, step*(-1), 10, npts, 1)

            if self.local_measurement_config.voltx_enable:
                self.devices.voltx.config(
                    self.local_measurement_config.integration_time,
                    self.local_measurement_config.voltage_precision)
            if self.local_measurement_config.volty_enable:
                self.devices.volty.config(
                    self.local_measurement_config.integration_time,
                    self.local_measurement_config.voltage_precision)
            if self.local_measurement_config.voltz_enable:
                self.devices.voltz.config(
                    self.local_measurement_config.integration_time,
                    self.local_measurement_config.voltage_precision)

            self.startVoltageThreads()

            if self.stop is False:
                if to_pos:
                    self.moveAxisAndUpdateGraph(first_axis, end + extra, idx)
                else:
                    self.moveAxisAndUpdateGraph(first_axis, start - extra, idx)

            self.stopTrigger()
            self.waitVoltageThreads()
            self.voltage_scan.avgx = (
                self.threadx.voltage if self.threadx is not None else [])
            self.voltage_scan.avgy = (
                self.thready.voltage if self.thready is not None else [])
            self.voltage_scan.avgz = (
                self.threadz.voltage if self.threadz is not None else [])
            self.voltage_scan.current = self.current_temperature_thread.current
            self.voltage_scan.temperature = (
                self.current_temperature_thread.temperature)
            self.current_temperature_thread.clear()

            if self.stop is True:
                return False

            if not to_pos:
                self.voltage_scan.reverse()

            if self.ui.save_voltage_scan_chb.isChecked():
                if not self.saveVoltageScan():
                    return False

            voltage_scan_list.append(self.voltage_scan.copy())

        for vd in voltage_scan_list:
            self.voltage_scan_list.append(vd)

        if self.local_hall_probe is not None:
            self.field_scan.set_field_scan(
                voltage_scan_list, self.local_hall_probe)
            success = self.saveFieldScan()
        else:
            success = True

        return success

    def showFieldmapDialog(self):
        """Open fieldmap dialog."""
        self.save_fieldmap_dialog.show(
            self.field_scan_list,
            self.local_hall_probe,
            self.field_scan_id_list)

    def showViewScanDialog(self):
        """Open view data dialog."""
        if self.local_hall_probe is None:
            self.view_scan_dialog.show(
                self.voltage_scan_list,
                self.voltage_scan_id_list,
                'Voltage [V]')
        else:
            self.view_scan_dialog.show(
                self.field_scan_list,
                self.field_scan_id_list,
                'Magnetic Field [T]')

    def startAutomaticMeasurements(self):
        """Configure and emit signal to start automatic ramp measurements."""
        self.configureMeasurement()
        if not self.measurement_configured:
            return

        self.ui.measure_btn.setEnabled(False)
        self.ui.stop_btn.setEnabled(True)
        self.change_current_setpoint.emit(1)

    def startCurrentAndTemperatureThread(self):
        """Start current and temperatures measurements."""
        if (not self.ui.save_temperature_chb.isChecked() and
           not self.ui.save_current_chb.isChecked()):
            return True

        try:
            index = self.ui.timer_interval_unit_cmb.currentIndex()
            if index == 0:
                mf = 1000
            elif index == 1:
                mf = 1000*60
            else:
                mf = 1000*3600
            timer_interval = self.ui.timer_interval_sb.value()*mf

            dcct_head = self.power_supply_config.dcct_head

            self.current_temperature_thread.timer_interval = timer_interval
            self.current_temperature_thread.dcct_head = dcct_head
            self.current_temperature_thread.clear()
            self.current_temperature_thread.start()
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to start multichannel readings.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def startMeasurement(self):
        """Configure devices and start measurement."""
        self.configureMeasurement()
        if not self.measurement_configured:
            return

        if not self.saveConfiguration():
            return

        self.ui.measure_btn.setEnabled(False)
        self.ui.stop_btn.setEnabled(True)

        if not self.measure():
            return

        if not self.resetMultimeters():
            return

        self.ui.stop_btn.setEnabled(False)
        self.ui.measure_btn.setEnabled(True)
        self.ui.save_scan_files_btn.setEnabled(True)
        self.ui.view_scan_btn.setEnabled(True)
        self.ui.clear_graph_btn.setEnabled(True)
        if self.local_hall_probe is not None:
            self.ui.create_fieldmap_btn.setEnabled(True)

        msg = 'End of measurement.'
        _QMessageBox.information(
            self, 'Measurement', msg, _QMessageBox.Ok)

    def startVoltageThreads(self):
        """Start threads to read voltage values."""
        if self.threadx is not None:
            self.threadx.start()

        if self.thready is not None:
            self.thready.start()

        if self.threadz is not None:
            self.threadz.start()

    def stopMeasurement(self):
        """Stop measurement to True."""
        try:
            self.stop = True
            self.ui.measure_btn.setEnabled(True)
            self.devices.pmac.stop_all_axis()
            self.ui.stop_btn.setEnabled(False)
            self.ui.clear_graph_btn.setEnabled(True)
            self.current_temperature_thread.quit()
            msg = 'The user stopped the measurements.'
            _QMessageBox.information(
                self, 'Abort', msg, _QMessageBox.Ok)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to stop measurements.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def stopTrigger(self):
        """Stop trigger."""
        self.devices.pmac.stop_trigger()
        if self.threadx is not None:
            self.threadx.end_measurement = True
        if self.thready is not None:
            self.thready.end_measurement = True
        if self.threadz is not None:
            self.threadz.end_measurement = True

    def updateConfiguration(self):
        """Update measurement configuration."""
        success = self.configuration_widget.updateConfiguration()
        if success:
            config = self.measurement_config.copy()
            if not any([
                    config.voltx_enable,
                    config.volty_enable,
                    config.voltz_enable]):
                msg = 'No multimeter selected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
                return False

            velocity = config.get_velocity(config.first_axis)
            step = config.get_step(config.first_axis)
            max_integration_time = _np.abs(step/velocity)
            integration_time = config.integration_time
            if integration_time > max_integration_time:
                msg = (
                    'The integration time must be ' +
                    'less than {0:.4f} seconds.'.format(
                        max_integration_time))
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
                return False

            self.local_measurement_config = config
            return True
        else:
            self.local_measurement_config = None
            return False

    def updateHallProbe(self):
        """Update hall probe."""
        self.local_hall_probe = self.hall_probe.copy()
        if self.local_hall_probe.valid_data():
            return True
        else:
            self.local_hall_probe = None
            msg = 'Invalid hall probe. Measure only voltage data?'
            reply = _QMessageBox.question(
                self, 'Message', msg, _QMessageBox.Yes, _QMessageBox.No)
            if reply == _QMessageBox.Yes:
                self.ui.save_voltage_scan_chb.setChecked(True)
                return True
            else:
                return False

    def validDatabase(self):
        """Check if the database filename is valid."""
        if (self.database is not None
           and len(self.database) != 0 and _os.path.isfile(self.database)):
            return True
        else:
            msg = 'Invalid database filename.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def waitVoltageThreads(self):
        """Wait threads."""
        if self.threadx is not None:
            while self.threadx.isRunning() and self.stop is False:
                _QApplication.processEvents()

        if self.thready is not None:
            while self.thready.isRunning() and self.stop is False:
                _QApplication.processEvents()

        if self.threadz is not None:
            while self.threadz.isRunning() and self.stop is False:
                _QApplication.processEvents()


class CurrentTemperatureThread(_QThread):
    """Thread to read values from multichannel."""

    def __init__(self):
        """Initialize object."""
        super().__init__()
        self.timer = _QTimer()
        self.timer.moveToThread(self)
        self.timer.timeout.connect(self.readChannels)
        self.timer_interval = None
        self.dcct_head = None
        self.ps_type = None
        self.current = {}
        self.temperature = {}

    @property
    def multich(self):
        """Multichannel."""
        return _QApplication.instance().devices.multich

    @property
    def ps(self):
        """Power supply."""
        return _QApplication.instance().devices.ps

    def clear(self):
        """Clear values."""
        self.current = {}
        self.temperature = {}

    def readChannels(self):
        """Read channels."""
        try:
            ts = _time.time()

            if self.ps_type is not None:
                self.ps.SetSlaveAdd(self.ps_type)
                ps_current = float(self.ps.Read_iLoad1())
                if 'PS' in self.current.keys():
                    self.current['PS'].append([ts, ps_current])
                else:
                    self.current['PS'] = [[ts, ps_current]]

            r = self.multich.get_converted_readings(dcct_head=self.dcct_head)

            channels = self.multich.config_channels
            dcct_channels = self.multich.dcct_channels
            for i, ch in enumerate(channels):
                if ch in dcct_channels:
                    if 'DCCT' in self.current.keys():
                        self.current['DCCT'].append([ts, r[i]])
                    else:
                        self.current['DCCT'] = [[ts, r[i]]]
                else:
                    if ch in self.temperature.keys():
                        self.temperature[ch].append([ts, r[i]])
                    else:
                        self.temperature[ch] = [[ts, r[i]]]
            _QApplication.processEvents()
        except Exception:
            pass

    def run(self):
        """Read current and temperatures from the device."""
        if self.timer_interval is not None:
            self.timer.start(self.timer_interval)
            loop = _QEventLoop()
            loop.exec_()


class VoltageThread(_QThread):
    """Thread to read values from multimeters."""

    def __init__(self, multimeter, precision):
        """Initialize object."""
        super().__init__()
        self.multimeter = multimeter
        self.precision = precision
        self.voltage = _np.array([])
        self.end_measurement = False

    def clear(self):
        """Clear voltage values."""
        self.voltage = _np.array([])
        self.end_measurement = False

    def run(self):
        """Read voltage from the device."""
        self.clear()
        while (self.end_measurement is False):
            if self.multimeter.inst.stb & 128:
                voltage = self.multimeter.read_voltage(self.precision)
                self.voltage = _np.append(self.voltage, voltage)
        else:
            # check memory
            self.multimeter.send_command(self.multimeter.commands.mcount)
            npoints = int(self.multimeter.read_from_device())
            if npoints > 0:
                # ask data from memory
                self.multimeter.send_command(
                    self.multimeter.commands.rmem + str(npoints))

                for idx in range(npoints):
                    voltage = self.multimeter.read_voltage(self.precision)
                    self.voltage = _np.append(self.voltage, voltage)
