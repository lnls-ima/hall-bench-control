# -*- coding: utf-8 -*-

"""Measurement widget for the Hall Bench Control application."""

import sys as _sys
import os as _os
import time as _time
import numpy as _np
import warnings as _warnings
import pyqtgraph as _pyqtgraph
import traceback as _traceback
from qtpy.QtWidgets import (
    QWidget as _QWidget,
    QFileDialog as _QFileDialog,
    QApplication as _QApplication,
    QVBoxLayout as _QVBoxLayout,
    QMessageBox as _QMessageBox,
    )
from qtpy.QtCore import (
    QThread as _QThread,
    Signal as _Signal,
    )
import qtpy.uic as _uic

from hallbench.gui import utils as _utils
from hallbench.gui.currentpositionwidget import CurrentPositionWidget \
    as _CurrentPositionWidget
from hallbench.data.configuration import MeasurementConfig \
    as _MeasurementConfig
from hallbench.data.measurement import VoltageScan as _VoltageScan
from hallbench.data.measurement import FieldScan as _FieldScan


class MeasurementWidget(_QWidget):
    """Measurement widget class for the Hall Bench Control application."""

    change_current_setpoint = _Signal([bool])
    turn_off_power_supply = _Signal([bool])

    _update_graph_time_interval = 0.05  # [s]
    _measurement_axes = [1, 2, 3, 5]
    _voltage_offset_avg_interval = 10  # [mm]

    def __init__(self, parent=None):
        """Set up the ui, add widgets and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _utils.getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # add position widget
        self.current_position_widget = _CurrentPositionWidget(self)
        _layout = _QVBoxLayout()
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.addWidget(self.current_position_widget)
        self.ui.position_wg.setLayout(_layout)

        self.measurement_configured = False
        self.local_measurement_config = None
        self.local_measurement_config_id = None
        self.local_hall_probe = None
        self.local_power_supply_config = None
        self.voltage_scan = None
        self.field_scan = None
        self.position_list = []
        self.field_scan_id_list = []
        self.voltage_scan_id_list = []
        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.stop = False

        # Connect signals and slots
        self.connectSignalSlots()

        # Add legend to plot
        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.graph_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        # Update probe names and configuration ID combo box
        self.updateProbeNames()
        self.updateConfigurationIDs()

        # Create voltage threads
        self.threadx = VoltageThread(self.devices.voltx)
        self.thready = VoltageThread(self.devices.volty)
        self.threadz = VoltageThread(self.devices.voltz)

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
    def positions(self):
        """Positions dict."""
        return _QApplication.instance().positions

    @property
    def power_supply_config(self):
        """Power supply configuration."""
        return _QApplication.instance().power_supply_config

    @property
    def save_fieldmap_dialog(self):
        """Save fieldmap dialog."""
        return _QApplication.instance().save_fieldmap_dialog

    @property
    def view_probe_dialog(self):
        """View probe dialog."""
        return _QApplication.instance().view_probe_dialog

    @property
    def view_scan_dialog(self):
        """View scan dialog."""
        return _QApplication.instance().view_scan_dialog

    def clearHallProbe(self):
        """Clear hall probe calibration data."""
        self.hall_probe.clear()
        self.ui.probe_name_cmb.setCurrentIndex(-1)

    def clearLoadOptions(self):
        """Clear load options."""
        self.ui.filename_le.setText("")
        self.ui.idn_cmb.setCurrentIndex(-1)

    def copyCurrentStartPosition(self):
        """Copy current start position to line edits."""
        for axis in self._measurement_axes:
            start_le = getattr(self.ui, 'start_ax' + str(axis) + '_le')
            if axis in self.positions.keys():
                start_le.setText('{0:0.4f}'.format(self.positions[axis]))
            else:
                start_le.setText('')

    def disableInvalidLineEdit(self):
        """Disable invalid line edit."""
        for axis in self._measurement_axes:
            first_rb = getattr(self.ui, 'first_ax' + str(axis) + '_rb')
            second_rb = getattr(self.ui, 'second_ax' + str(axis) + '_rb')
            step_le = getattr(self.ui, 'step_ax' + str(axis) + '_le')
            end_le = getattr(self.ui, 'end_ax' + str(axis) + '_le')
            extra_le = getattr(self.ui, 'extra_ax' + str(axis) + '_le')
            if first_rb.isChecked() or second_rb.isChecked():
                step_le.setEnabled(True)
                end_le.setEnabled(True)
                if first_rb.isChecked():
                    extra_le.setEnabled(True)
                else:
                    extra_le.setEnabled(False)
                    extra_le.setText('')
            else:
                step_le.setEnabled(False)
                step_le.setText('')
                end_le.setEnabled(False)
                end_le.setText('')
                extra_le.setEnabled(False)
                extra_le.setText('')

    def disableSecondAxisButton(self):
        """Disable invalid second axis radio buttons."""
        for axis in self._measurement_axes:
            first_rb = getattr(self.ui, 'first_ax' + str(axis) + '_rb')
            second_rb = getattr(self.ui, 'second_ax' + str(axis) + '_rb')
            if first_rb.isChecked():
                second_rb.setChecked(False)
                second_rb.setEnabled(False)
            else:
                if axis != 5:
                    second_rb.setEnabled(True)

    def enableLoadDB(self):
        """Enable button to load configuration from database."""
        if self.ui.idn_cmb.currentIndex() != -1:
            self.ui.loaddb_btn.setEnabled(True)
        else:
            self.ui.loaddb_btn.setEnabled(False)

    def fixEndPositionValue(self, axis):
        """Fix end position value."""
        try:
            start_le = getattr(self.ui, 'start_ax' + str(axis) + '_le')
            start = _utils.getValueFromStringExpresssion(start_le.text())
            if start is None:
                return

            step_le = getattr(self.ui, 'step_ax' + str(axis) + '_le')
            step = _utils.getValueFromStringExpresssion(step_le.text())
            if step is None:
                return

            end_le = getattr(self.ui, 'end_ax' + str(axis) + '_le')
            end = _utils.getValueFromStringExpresssion(end_le.text())
            if end is None:
                return

            npts = _np.round(round((end - start) / step, 4) + 1)
            if start <= end:
                corrected_end = start + (npts-1)*step
            else:
                corrected_end = start
            end_le.setText('{0:0.4f}'.format(corrected_end))

        except Exception:
            pass

    def getAxisParam(self, param, axis):
        """Get axis parameter."""
        le = getattr(self.ui, param + '_ax' + str(axis) + '_le')
        return _utils.getValueFromStringExpresssion(le.text())

    def loadConfig(self):
        """Set measurement parameters."""
        try:
            self.ui.magnet_name_le.setText(self.measurement_config.magnet_name)

            current_sp = self.measurement_config.current_setpoint
            if current_sp is None:
                self.ui.current_setpoint_le.setText('')
            else:
                self.ui.current_setpoint_le.setText(str(current_sp))

            idx = self.ui.probe_name_cmb.findText(
                self.measurement_config.probe_name)
            self.ui.probe_name_cmb.setCurrentIndex(idx)

            self.ui.temperature_le.setText(self.measurement_config.temperature)
            self.ui.operator_le.setText(self.measurement_config.operator)
            self.ui.comments_le.setText(self.measurement_config.comments)

            self.ui.voltx_enable_chb.setChecked(
                self.measurement_config.voltx_enable)
            self.ui.volty_enable_chb.setChecked(
                self.measurement_config.volty_enable)
            self.ui.voltz_enable_chb.setChecked(
                self.measurement_config.voltz_enable)

            self.ui.integration_time_le.setText('{0:0.4f}'.format(
                self.measurement_config.integration_time))
            self.ui.voltage_precision_cmb.setCurrentIndex(
                self.measurement_config.voltage_precision)
            self.ui.voltage_range_le.setText(str(
                self.measurement_config.voltage_range))

            self.ui.nr_measurements_sb.setValue(
                self.measurement_config.nr_measurements)

            first_axis = self.measurement_config.first_axis
            first_rb = getattr(self.ui, 'first_ax' + str(first_axis) + '_rb')
            first_rb.setChecked(True)

            self.disableSecondAxisButton()

            second_axis = self.measurement_config.second_axis
            if second_axis != -1:
                second_rb = getattr(
                    self.ui, 'second_ax' + str(second_axis) + '_rb')
                second_rb.setChecked(True)
                self.uncheckRadioButtons(second_axis)
            else:
                self.uncheckRadioButtons(second_axis)

            self.ui.subtract_voltage_offset_chb.setChecked(
                self.measurement_config.subtract_voltage_offset)
            self.ui.save_voltage_chb.setChecked(
                self.measurement_config.save_voltage)
            self.ui.save_current_chb.setChecked(
                self.measurement_config.save_current)
            self.ui.save_temperature_chb.setChecked(
                self.measurement_config.save_temperature)
            self.ui.automatic_ramp_chb.setChecked(
                self.measurement_config.automatic_ramp)

            for axis in self._measurement_axes:
                start_le = getattr(self.ui, 'start_ax' + str(axis) + '_le')
                value = self.measurement_config.get_start(axis)
                start_le.setText('{0:0.4f}'.format(value))

                step_le = getattr(self.ui, 'step_ax' + str(axis) + '_le')
                value = self.measurement_config.get_step(axis)
                step_le.setText('{0:0.4f}'.format(value))

                end_le = getattr(self.ui, 'end_ax' + str(axis) + '_le')
                value = self.measurement_config.get_end(axis)
                end_le.setText('{0:0.4f}'.format(value))

                extra_le = getattr(self.ui, 'extra_ax' + str(axis) + '_le')
                value = self.measurement_config.get_extra(axis)
                extra_le.setText('{0:0.4f}'.format(value))

                vel_le = getattr(self.ui, 'vel_ax' + str(axis) + '_le')
                value = self.measurement_config.get_velocity(axis)
                vel_le.setText('{0:0.4f}'.format(value))

            self.disableInvalidLineEdit()
            self.updateTriggerStep(first_axis)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to load configuration.'
            _QMessageBox.critical(
                self, 'Failure', msg, _QMessageBox.Ok)

    def loadConfigDB(self):
        """Load configuration from database to set measurement parameters."""
        self.ui.filename_le.setText("")

        try:
            idn = int(self.ui.idn_cmb.currentText())
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.critical(
                self, 'Failure', 'Invalid database ID.', _QMessageBox.Ok)
            return

        self.updateConfigurationIDs()
        idx = self.ui.idn_cmb.findText(str(idn))
        if idx == -1:
            self.ui.idn_cmb.setCurrentIndex(-1)
            _QMessageBox.critical(
                self, 'Failure', 'Invalid database ID.', _QMessageBox.Ok)
            return

        try:
            self.measurement_config.clear()
            self.measurement_config.read_from_database(self.database, idn)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to read configuration from database.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        self.loadConfig()
        self.ui.idn_cmb.setCurrentIndex(self.ui.idn_cmb.findText(str(idn)))
        self.ui.loaddb_btn.setEnabled(False)

    def loadConfigFile(self):
        """Load configuration file to set measurement parameters."""
        self.ui.idn_cmb.setCurrentIndex(-1)

        default_filename = self.ui.filename_le.text()
        if len(default_filename) == 0:
            default_filename = self.directory
        elif len(_os.path.split(default_filename)[0]) == 0:
            default_filename = _os.path.join(self.directory, default_filename)

        filename = _QFileDialog.getOpenFileName(
            self, caption='Open measurement configuration file',
            directory=default_filename, filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            self.measurement_config.clear()
            self.measurement_config.read_file(filename)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to read configuration file.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        self.loadConfig()
        self.ui.filename_le.setText(filename)

    def loadHallProbe(self):
        """Load hall probe from database."""
        self.hall_probe.clear()
        probe_name = self.ui.probe_name_cmb.currentText()
        if len(probe_name) == 0:
            return

        try:
            idn = self.hall_probe.get_hall_probe_id(self.database, probe_name)
            if idn is not None:
                self.hall_probe.read_from_database(self.database, idn)
            else:
                self.ui.probe_name_cmb.setCurrentIndex(-1)
                msg = 'Invalid probe name.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to load Hall probe from database.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def saveConfigDB(self):
        """Save configuration to database."""
        self.ui.idn_cmb.setCurrentIndex(-1)
        if self.database is not None and _os.path.isfile(self.database):
            try:
                if self.updateConfiguration():
                    idn = self.measurement_config.save_to_database(
                        self.database)
                    self.ui.idn_cmb.addItem(str(idn))
                    self.ui.idn_cmb.setCurrentIndex(self.ui.idn_cmb.count()-1)
                    self.ui.loaddb_btn.setEnabled(False)
            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                msg = 'Failed to save configuration to database.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
        else:
            msg = 'Invalid database filename.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def saveConfigFile(self):
        """Save measurement parameters to file."""
        default_filename = self.ui.filename_le.text()
        if len(default_filename) == 0:
            default_filename = self.directory
        elif len(_os.path.split(default_filename)[0]) == 0:
            default_filename = _os.path.join(self.directory, default_filename)

        filename = _QFileDialog.getSaveFileName(
            self, caption='Save measurement configuration file',
            directory=default_filename, filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        if self.updateConfiguration():
            try:
                if (not filename.endswith('.txt')
                   and not filename.endswith('.dat')):
                    filename = filename + '.txt'
                self.measurement_config.save_file(filename)

            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                msg = 'Failed to save configuration to file.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def setFloatLineEditText(
            self, line_edit, precision=4, expression=True,
            positive=False, nonzero=False):
        """Set the line edit string format for float value."""
        try:
            if line_edit.isModified():
                self.clearLoadOptions()
                _utils.setFloatLineEditText(
                    line_edit, precision, expression, positive, nonzero)
        except Exception:
            pass

    def showViewProbeDialog(self):
        """Open view probe dialog."""
        self.view_probe_dialog.show(self.hall_probe)

    def uncheckRadioButtons(self, selected_axis):
        """Uncheck radio buttons."""
        axes = [a for a in self._measurement_axes if a != selected_axis]
        for axis in axes:
            second_rb = getattr(self.ui, 'second_ax' + str(axis) + '_rb')
            second_rb.setChecked(False)

    def updateConfiguration(self):
        """Update measurement configuration parameters."""
        try:
            self.measurement_config.clear()

            _s = self.ui.magnet_name_le.text().strip()
            self.measurement_config.magnet_name = _s if len(_s) != 0 else None

            current_setpoint = _utils.getValueFromStringExpresssion(
                self.ui.current_setpoint_le.text())
            self.measurement_config.current_setpoint = current_setpoint

            _s = self.ui.probe_name_cmb.currentText().strip()
            self.measurement_config.probe_name = _s if len(_s) != 0 else None

            _s = self.ui.temperature_le.text().strip()
            self.measurement_config.temperature = _s if len(_s) != 0 else None

            _s = self.ui.operator_le.text().strip()
            self.measurement_config.operator = _s if len(_s) != 0 else None

            _s = self.ui.comments_le.text().strip()
            self.measurement_config.comments = _s if len(_s) != 0 else ''

            _voltx_enable = self.ui.voltx_enable_chb.isChecked()
            self.measurement_config.voltx_enable = _voltx_enable

            _volty_enable = self.ui.volty_enable_chb.isChecked()
            self.measurement_config.volty_enable = _volty_enable

            _voltz_enable = self.ui.voltz_enable_chb.isChecked()
            self.measurement_config.voltz_enable = _voltz_enable

            idx = self.ui.voltage_precision_cmb.currentIndex()
            self.measurement_config.voltage_precision = idx
            nr_meas = self.ui.nr_measurements_sb.value()
            self.measurement_config.nr_measurements = nr_meas

            integration_time = _utils.getValueFromStringExpresssion(
                self.ui.integration_time_le.text())
            self.measurement_config.integration_time = integration_time

            voltage_range = _utils.getValueFromStringExpresssion(
                self.ui.voltage_range_le.text())
            self.measurement_config.voltage_range = voltage_range

            _ch = self.ui.subtract_voltage_offset_chb.isChecked()
            self.measurement_config.subtract_voltage_offset = 1 if _ch else 0

            _ch = self.ui.save_voltage_chb.isChecked()
            self.measurement_config.save_voltage = 1 if _ch else 0

            _ch = self.ui.save_current_chb.isChecked()
            self.measurement_config.save_current = 1 if _ch else 0

            _ch = self.ui.save_temperature_chb.isChecked()
            self.measurement_config.save_temperature = 1 if _ch else 0

            _ch = self.ui.automatic_ramp_chb.isChecked()
            self.measurement_config.automatic_ramp = 1 if _ch else 0

            for axis in self._measurement_axes:
                first_rb = getattr(self.ui, 'first_ax' + str(axis) + '_rb')
                second_rb = getattr(self.ui, 'second_ax' + str(axis) + '_rb')
                if first_rb.isChecked():
                    self.measurement_config.first_axis = axis
                elif second_rb.isChecked():
                    self.measurement_config.second_axis = axis

                start = self.getAxisParam('start', axis)
                self.measurement_config.set_start(axis, start)

                step_le = getattr(self.ui, 'step_ax' + str(axis) + '_le')
                if step_le.isEnabled():
                    step = self.getAxisParam('step', axis)
                    self.measurement_config.set_step(axis, step)
                else:
                    self.measurement_config.set_step(axis, 0.0)

                end_le = getattr(self.ui, 'end_ax' + str(axis) + '_le')
                if end_le.isEnabled():
                    end = self.getAxisParam('end', axis)
                    self.measurement_config.set_end(axis, end)
                else:
                    self.measurement_config.set_end(axis, start)

                extra_le = getattr(self.ui, 'extra_ax' + str(axis) + '_le')
                if extra_le.isEnabled():
                    extra = self.getAxisParam('extra', axis)
                    self.measurement_config.set_extra(axis, extra)
                else:
                    self.measurement_config.set_extra(axis, 0.0)

                vel = self.getAxisParam('vel', axis)
                self.measurement_config.set_velocity(axis, vel)

            if self.measurement_config.second_axis is None:
                self.measurement_config.second_axis = -1

            if self.measurement_config.valid_data():
                first_axis = self.measurement_config.first_axis
                step = self.measurement_config.get_step(first_axis)
                vel = self.measurement_config.get_velocity(first_axis)
                max_int_time = _np.abs(step/vel)*1000 - 5

                if self.measurement_config.integration_time > max_int_time:
                    self.local_measurement_config = None
                    msg = (
                        'The integration time must be ' +
                        'less than {0:.4f} ms.'.format(max_int_time))
                    _QMessageBox.critical(
                        self, 'Failure', msg, _QMessageBox.Ok)
                    return False

                if not any([
                        self.measurement_config.voltx_enable,
                        self.measurement_config.volty_enable,
                        self.measurement_config.voltz_enable]):
                    self.local_measurement_config = None
                    msg = 'No multimeter selected.'
                    _QMessageBox.critical(
                        self, 'Failure', msg, _QMessageBox.Ok)
                    return False

                self.local_measurement_config = self.measurement_config.copy()
                return True

            else:
                self.local_measurement_config = None
                msg = 'Invalid measurement configuration.'
                _QMessageBox.critical(
                    self, 'Failure', msg, _QMessageBox.Ok)
                return False

        except Exception:
            self.local_measurement_config = None
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to update configuration.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def updateConfigurationIDs(self):
        """Update combo box ids."""
        current_text = self.ui.idn_cmb.currentText()
        load_enabled = self.ui.loaddb_btn.isEnabled()
        self.ui.idn_cmb.clear()
        try:
            idns = self.measurement_config.get_table_column(
                self.database, 'id')
            self.ui.idn_cmb.clear()
            self.ui.idn_cmb.addItems([str(idn) for idn in idns])
            if len(current_text) == 0:
                self.ui.idn_cmb.setCurrentIndex(self.ui.idn_cmb.count()-1)
                self.ui.loaddb_btn.setEnabled(True)
            else:
                self.ui.idn_cmb.setCurrentText(current_text)
                self.ui.loaddb_btn.setEnabled(load_enabled)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def updateProbeNames(self):
        """Update combo box with database probe names."""
        current_text = self.ui.probe_name_cmb.currentText()
        self.ui.probe_name_cmb.clear()
        try:
            probe_names = self.hall_probe.get_table_column(
                self.database, 'probe_name')
            self.ui.probe_name_cmb.addItems(probe_names)
            if len(current_text) == 0:
                self.ui.probe_name_cmb.setCurrentIndex(-1)
            else:
                self.ui.probe_name_cmb.setCurrentText(current_text)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def updateTriggerStep(self, axis):
        """Update trigger step."""
        try:
            first_rb = getattr(self.ui, 'first_ax' + str(axis) + '_rb')
            if first_rb.isChecked():
                step = self.getAxisParam('step', axis)
                vel = self.getAxisParam('vel', axis)
                if step is not None and vel is not None:
                    max_int_time = _np.abs(step/vel)*1000 - 5
                    _s = 'Max. Int. Time [ms]:  {0:.4f}'.format(
                        max_int_time)
                    self.ui.max_integration_time_la.setText(_s)
                else:
                    self.ui.max_integration_time_la.setText('')
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def clear(self):
        """Clear."""
        self.threadx.clear()
        self.thready.clear()
        self.threadz.clear()
        self.measurement_configured = False
        self.local_measurement_config = None
        self.local_measurement_config_id = None
        self.local_hall_probe = None
        self.local_power_supply_config = None
        self.voltage_scan = None
        self.field_scan = None
        self.position_list = []
        self.field_scan_id_list = []
        self.voltage_scan_id_list = []
        self.stop = False
        self.clearGraph()
        self.ui.view_scan_btn.setEnabled(False)
        self.ui.clear_graph_btn.setEnabled(False)
        self.ui.create_fieldmap_btn.setEnabled(False)
        self.ui.save_scan_files_btn.setEnabled(False)

    def clearButtonClicked(self):
        """Clear current measurement and plots."""
        self.clearCurrentMeasurement()
        self.clearGraph()
        self.ui.view_scan_btn.setEnabled(False)
        self.ui.clear_graph_btn.setEnabled(False)
        self.ui.create_fieldmap_btn.setEnabled(False)
        self.ui.save_scan_files_btn.setEnabled(False)

    def clearCurrentMeasurement(self):
        """Clear current measurement data."""
        self.voltage_scan = None
        self.field_scan = None
        self.position_list = []
        self.field_scan_id_list = []
        self.voltage_scan_id_list = []

    def clearGraph(self):
        """Clear plots."""
        self.ui.graph_pw.plotItem.curves.clear()
        self.ui.graph_pw.clear()
        self.graphx = []
        self.graphy = []
        self.graphz = []

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.current_position_widget.close()
            self.killVoltageThreads()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.first_ax1_rb.clicked.connect(self.clearLoadOptions)
        self.ui.first_ax2_rb.clicked.connect(self.clearLoadOptions)
        self.ui.first_ax3_rb.clicked.connect(self.clearLoadOptions)
        self.ui.first_ax5_rb.clicked.connect(self.clearLoadOptions)
        self.ui.second_ax1_rb.clicked.connect(self.clearLoadOptions)
        self.ui.second_ax2_rb.clicked.connect(self.clearLoadOptions)
        self.ui.second_ax3_rb.clicked.connect(self.clearLoadOptions)
        self.ui.second_ax5_rb.clicked.connect(self.clearLoadOptions)
        self.ui.magnet_name_le.editingFinished.connect(self.clearLoadOptions)
        self.ui.current_setpoint_le.editingFinished.connect(
            self.clearLoadOptions)
        self.ui.temperature_le.editingFinished.connect(self.clearLoadOptions)
        self.ui.operator_le.editingFinished.connect(self.clearLoadOptions)
        self.ui.comments_le.editingFinished.connect(self.clearLoadOptions)
        self.ui.nr_measurements_sb.valueChanged.connect(self.clearLoadOptions)
        self.ui.probe_name_cmb.currentIndexChanged.connect(
            self.clearLoadOptions)
        self.ui.voltage_precision_cmb.currentIndexChanged.connect(
            self.clearLoadOptions)
        self.voltx_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.volty_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.voltz_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.ui.idn_cmb.currentIndexChanged.connect(self.enableLoadDB)
        self.ui.update_idn_btn.clicked.connect(self.updateConfigurationIDs)

        self.ui.current_start_btn.clicked.connect(
            self.copyCurrentStartPosition)

        self.ui.current_setpoint_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(self.ui.current_setpoint_le))

        self.ui.start_ax1_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(self.ui.start_ax1_le))
        self.ui.start_ax2_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(self.ui.start_ax2_le))
        self.ui.start_ax3_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(self.ui.start_ax3_le))
        self.ui.start_ax5_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(self.ui.start_ax5_le))

        self.ui.end_ax1_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(self.ui.end_ax1_le))
        self.ui.end_ax2_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(self.ui.end_ax2_le))
        self.ui.end_ax3_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(self.ui.end_ax3_le))
        self.ui.end_ax5_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(self.ui.end_ax5_le))

        self.ui.step_ax1_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(
                self.ui.step_ax1_le, positive=True, nonzero=True))
        self.ui.step_ax2_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(
                self.ui.step_ax2_le, positive=True, nonzero=True))
        self.ui.step_ax3_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(
                self.ui.step_ax3_le, positive=True, nonzero=True))
        self.ui.step_ax5_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(
                self.ui.step_ax5_le, positive=True, nonzero=True))

        self.ui.extra_ax1_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(
                self.ui.extra_ax1_le, positive=True))
        self.ui.extra_ax2_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(
                self.ui.extra_ax2_le, positive=True))
        self.ui.extra_ax3_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(
                self.ui.extra_ax3_le, positive=True))
        self.ui.extra_ax5_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(
                self.ui.extra_ax5_le, positive=True))

        self.ui.vel_ax1_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(
                self.ui.vel_ax1_le, positive=True, nonzero=True))
        self.ui.vel_ax2_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(
                self.ui.vel_ax2_le, positive=True, nonzero=True))
        self.ui.vel_ax3_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(
                self.ui.vel_ax3_le, positive=True, nonzero=True))
        self.ui.vel_ax5_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(
                self.ui.vel_ax5_le, positive=True, nonzero=True))

        self.ui.integration_time_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(
                self.ui.integration_time_le, precision=4))

        self.ui.voltage_range_le.editingFinished.connect(
            lambda: self.setFloatLineEditText(
                self.ui.voltage_range_le, precision=1))

        self.ui.step_ax1_le.editingFinished.connect(
            lambda: self.updateTriggerStep(1))
        self.ui.step_ax2_le.editingFinished.connect(
            lambda: self.updateTriggerStep(2))
        self.ui.step_ax3_le.editingFinished.connect(
            lambda: self.updateTriggerStep(3))
        self.ui.step_ax5_le.editingFinished.connect(
            lambda: self.updateTriggerStep(5))
        self.ui.vel_ax1_le.editingFinished.connect(
            lambda: self.updateTriggerStep(1))
        self.ui.vel_ax2_le.editingFinished.connect(
            lambda: self.updateTriggerStep(2))
        self.ui.vel_ax3_le.editingFinished.connect(
            lambda: self.updateTriggerStep(3))
        self.ui.vel_ax5_le.editingFinished.connect(
            lambda: self.updateTriggerStep(5))

        self.ui.first_ax1_rb.clicked.connect(self.disableSecondAxisButton)
        self.ui.first_ax2_rb.clicked.connect(self.disableSecondAxisButton)
        self.ui.first_ax3_rb.clicked.connect(self.disableSecondAxisButton)
        self.ui.first_ax5_rb.clicked.connect(self.disableSecondAxisButton)

        self.ui.first_ax1_rb.toggled.connect(self.disableInvalidLineEdit)
        self.ui.first_ax2_rb.toggled.connect(self.disableInvalidLineEdit)
        self.ui.first_ax3_rb.toggled.connect(self.disableInvalidLineEdit)
        self.ui.first_ax5_rb.toggled.connect(self.disableInvalidLineEdit)

        self.ui.second_ax1_rb.toggled.connect(self.disableInvalidLineEdit)
        self.ui.second_ax2_rb.toggled.connect(self.disableInvalidLineEdit)
        self.ui.second_ax3_rb.toggled.connect(self.disableInvalidLineEdit)
        self.ui.second_ax5_rb.toggled.connect(self.disableInvalidLineEdit)

        self.ui.second_ax1_rb.clicked.connect(
            lambda: self.uncheckRadioButtons(1))
        self.ui.second_ax2_rb.clicked.connect(
            lambda: self.uncheckRadioButtons(2))
        self.ui.second_ax3_rb.clicked.connect(
            lambda: self.uncheckRadioButtons(3))
        self.ui.second_ax5_rb.clicked.connect(
            lambda: self.uncheckRadioButtons(5))

        self.ui.start_ax1_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(1))
        self.ui.start_ax2_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(2))
        self.ui.start_ax3_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(3))
        self.ui.start_ax5_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(5))

        self.ui.step_ax1_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(1))
        self.ui.step_ax2_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(2))
        self.ui.step_ax3_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(3))
        self.ui.step_ax5_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(5))

        self.ui.end_ax1_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(1))
        self.ui.end_ax2_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(2))
        self.ui.end_ax3_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(3))
        self.ui.end_ax5_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(5))

        self.ui.loadfile_btn.clicked.connect(self.loadConfigFile)
        self.ui.savefile_btn.clicked.connect(self.saveConfigFile)
        self.ui.loaddb_btn.clicked.connect(self.loadConfigDB)
        self.ui.savedb_btn.clicked.connect(self.saveConfigDB)
        self.ui.probe_name_cmb.currentIndexChanged.connect(self.loadHallProbe)
        self.ui.update_probe_name_btn.clicked.connect(self.updateProbeNames)
        self.ui.clear_probe_btn.clicked.connect(self.clearHallProbe)
        self.ui.view_probe_btn.clicked.connect(self.showViewProbeDialog)

        self.ui.measure_btn.clicked.connect(self.measureButtonClicked)
        self.ui.stop_btn.clicked.connect(self.stopMeasurement)
        self.ui.create_fieldmap_btn.clicked.connect(self.showFieldmapDialog)
        self.ui.save_scan_files_btn.clicked.connect(self.saveFieldScanFiles)
        self.ui.view_scan_btn.clicked.connect(self.showViewScanDialog)
        self.ui.clear_graph_btn.clicked.connect(self.clearButtonClicked)

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
           or not self.updatePowerSupplyConfig()
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

    def configureVoltageThreads(self):
        """Configure threads to read voltage values."""
        if self.local_measurement_config.voltx_enable:
            self.threadx.configure(
                self.local_measurement_config.voltage_precision,
                self.local_measurement_config.integration_time)

        if self.local_measurement_config.volty_enable:
            self.thready.configure(
                self.local_measurement_config.voltage_precision,
                self.local_measurement_config.integration_time)

        if self.local_measurement_config.voltz_enable:
            self.threadz.configure(
                self.local_measurement_config.voltage_precision,
                self.local_measurement_config.integration_time)

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
        self.turn_off_power_supply.emit(True)

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
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def measure(self):
        """Perform one measurement."""
        if not self.measurement_configured:
            return False

        try:
            self.power_supply_config.update_display = False

            first_axis = self.local_measurement_config.first_axis
            second_axis = self.local_measurement_config.second_axis
            fixed_axes = self.getFixedAxes()

            self.configureVoltageThreads()

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
                    if not self.scanLine(first_axis, second_axis, pos):
                        return

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

            if self.stop is True:
                return

            # move to initial position
            first_axis_start = self.local_measurement_config.get_start(
                first_axis)
            self.moveAxis(first_axis, first_axis_start)
            self.ui.nr_measurements_la.setText('')

            self.plotField()
            self.quitVoltageThreads()
            self.power_supply_config.update_display = True
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.quitVoltageThreads()
            if self.local_measurement_config.automatic_ramp:
                self.turn_off_power_supply.emit(True)
            self.power_supply_config.update_display = True
            self.ui.nr_measurements_la.setText('')
            msg = 'Measurement failure.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def updateCurrentSetpoint(self, current_setpoint):
        """Update current setpoint value."""
        try:
            current_value_str = str(current_setpoint)
            self.measurement_config.current_setpoint = current_setpoint
            self.ui.current_setpoint_le.setText(current_value_str)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to update current setpoint.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def measureAndEmitSignal(self):
        """Measure and emit signal to change current setpoint."""
        self.clearCurrentMeasurement()

        try:
            self.local_measurement_config.current_setpoint = (
                self.measurement_config.current_setpoint)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to update current setpoint.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        if self.local_measurement_config.current_setpoint == 0:
            self.change_current_setpoint.emit(True)
            return

        if not self.saveConfiguration():
            return

        if not self.measure():
            return

        self.change_current_setpoint.emit(True)

    def measureButtonClicked(self):
        """Start measurements."""
        if self.ui.automatic_ramp_chb.isChecked():
            self.startAutomaticMeasurements()
        else:
            self.startMeasurement()

    def measureCurrentAndTemperature(self):
        """Measure current and temperatures."""
        if (not self.ui.save_temperature_chb.isChecked() and
           not self.ui.save_current_chb.isChecked()):
            return True

        try:
            temperature_dict = {}
            dcct_head = self.local_power_supply_config.dcct_head
            ps_type = self.local_power_supply_config.ps_type
            ts = _time.time()

            # Read power supply current
            if self.ui.save_current_chb.isChecked() and ps_type is not None:
                self.devices.ps.SetSlaveAdd(ps_type)
                ps_current = float(self.devices.ps.Read_iLoad1())
                self.voltage_scan.ps_current_avg = ps_current

            # Read dcct current
            if dcct_head is not None:
                dcct_current = self.devices.dcct.read_current(
                    dcct_head=dcct_head)
                self.voltage_scan.dcct_current_avg = dcct_current

            # Read multichannel
            r = self.devices.multich.get_converted_readings(
                dcct_head=dcct_head)
            channels = self.devices.multich.config_channels
            for i, ch in enumerate(channels):
                temperature_dict[ch] = [[ts, r[i]]]
            _QApplication.processEvents()

            if self.ui.save_temperature_chb.isChecked():
                self.voltage_scan.temperature = temperature_dict

            _QApplication.processEvents()
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return False

    def measureVoltageScan(
            self, idx, to_pos, axis, start, end, step, extra, npts):
        """Measure one voltage scan."""
        if self.stop is True:
            return False

        self.voltage_scan.avgx = []
        self.voltage_scan.avgy = []
        self.voltage_scan.avgz = []
        self.voltage_scan.dcct_current_avg = None
        self.voltage_scan.ps_current_avg = None
        self.voltage_scan.temperature = {}

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            self.graphx[idx].setData([], [])
            self.graphy[idx].setData([], [])
            self.graphz[idx].setData([], [])

        # go to initial position
        if to_pos:
            self.moveAxis(axis, start - extra)
        else:
            self.moveAxis(axis, end + extra)
        _QApplication.processEvents()

        if self.stop is True:
            return False

        self.measureCurrentAndTemperature()
        _QApplication.processEvents()

        if self.stop is True:
            return False
        else:
            if to_pos:
                self.devices.pmac.set_trigger(
                    axis, start, step, 10, npts, 1)
            else:
                self.devices.pmac.set_trigger(
                    axis, end, step*(-1), 10, npts, 1)

        if self.local_measurement_config.voltx_enable:
            self.devices.voltx.config(
                self.local_measurement_config.integration_time,
                self.local_measurement_config.voltage_precision,
                self.local_measurement_config.voltage_range)
        if self.local_measurement_config.volty_enable:
            self.devices.volty.config(
                self.local_measurement_config.integration_time,
                self.local_measurement_config.voltage_precision,
                self.local_measurement_config.voltage_range)
        if self.local_measurement_config.voltz_enable:
            self.devices.voltz.config(
                self.local_measurement_config.integration_time,
                self.local_measurement_config.voltage_precision,
                self.local_measurement_config.voltage_range)
        _QApplication.processEvents()

        self.startVoltageThreads()

        if self.stop is False:
            if to_pos:
                self.moveAxisAndUpdateGraph(axis, end + extra, idx)
            else:
                self.moveAxisAndUpdateGraph(axis, start - extra, idx)

        self.stopTrigger()
        self.waitVoltageThreads()

        _QApplication.processEvents()

        self.voltage_scan.avgx = self.threadx.voltage
        self.voltage_scan.avgy = self.thready.voltage
        self.voltage_scan.avgz = self.threadz.voltage

        _QApplication.processEvents()
        self.quitVoltageThreads()

        if self.voltage_scan.npts == 0:
            return True

        if not to_pos:
            self.voltage_scan.reverse()

        if self.local_measurement_config.subtract_voltage_offset:
            scan_pos = self.voltage_scan.scan_pos
            if scan_pos[-1] - scan_pos[0] <= self._voltage_offset_avg_interval:
                self.voltage_scan.offsetx_start = self.voltage_scan.avgx[0]
                self.voltage_scan.offsetx_end = self.voltage_scan.avgx[-1]
                self.voltage_scan.offsety_start = self.voltage_scan.avgy[0]
                self.voltage_scan.offsety_end = self.voltage_scan.avgy[-1]
                self.voltage_scan.offsetz_start = self.voltage_scan.avgz[0]
                self.voltage_scan.offsetz_end = self.voltage_scan.avgz[-1]
            else:
                idx_start = _np.where(_np.cumsum(_np.diff(
                    scan_pos)) >= self._voltage_offset_avg_interval)[0][0] + 1
                idx_end = len(scan_pos) - idx_start
                self.voltage_scan.offsetx_start = _np.mean(
                    self.voltage_scan.avgx[:idx_start])
                self.voltage_scan.offsetx_end = _np.mean(
                    self.voltage_scan.avgx[idx_end:])
                self.voltage_scan.offsety_start = _np.mean(
                    self.voltage_scan.avgy[:idx_start])
                self.voltage_scan.offsety_end = _np.mean(
                    self.voltage_scan.avgy[idx_end:])
                self.voltage_scan.offsetz_start = _np.mean(
                    self.voltage_scan.avgz[:idx_start])
                self.voltage_scan.offsetz_end = _np.mean(
                    self.voltage_scan.avgz[idx_end:])

        if self.stop is True:
            return False
        else:
            return True

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

        field_scan_list = []
        for idn in self.field_scan_id_list:
            fs = _FieldScan(database=self.database, idn=idn)
            field_scan_list.append(fs)

        self.clearGraph()
        nr_curves = len(field_scan_list)
        self.configureGraph(nr_curves, 'Magnetic Field [T]')

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            for i in range(nr_curves):
                fs = field_scan_list[i]
                positions = fs.scan_pos
                self.graphx[i].setData(positions, fs.avgx)
                self.graphy[i].setData(positions, fs.avgy)
                self.graphz[i].setData(positions, fs.avgz)

    def plotVoltage(self, idx):
        """Plot voltage values.

        Args:
            idx (int): index of the plot to update.
        """
        voltagex = [v for v in self.threadx.voltage]
        voltagey = [v for v in self.thready.voltage]
        voltagez = [v for v in self.threadz.voltage]

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
            text = self.ui.idn_cmb.currentText()
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
                self.updateConfigurationIDs()
                idx = self.ui.idn_cmb.findText(str(idn))
                self.ui.idn_cmb.setCurrentIndex(idx)
                self.ui.loaddb_btn.setEnabled(False)

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
            mc = self.local_measurement_config.current_setpoint
            self.field_scan.magnet_name = mn
            self.field_scan.current_setpoint = mc
            self.field_scan.comments = self.local_measurement_config.comments
            self.field_scan.configuration_id = self.local_measurement_config_id
            idn = self.field_scan.save_to_database(self.database)
            self.field_scan_id_list.append(idn)
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save FieldScan to database'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def saveFieldScanFiles(self):
        """Save scan files."""
        field_scan_list = []
        for idn in self.field_scan_id_list:
            fs = _FieldScan(database=self.database, idn=idn)
            field_scan_list.append(fs)

        directory = _QFileDialog.getExistingDirectory(
            self, caption='Save scan files', directory=self.directory)

        if isinstance(directory, tuple):
            directory = directory[0]

        if len(directory) == 0:
            return

        try:
            for i, scan in enumerate(field_scan_list):
                idn = self.field_scan_id_list[i]
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
            mc = self.local_measurement_config.current_setpoint
            self.voltage_scan.magnet_name = mn
            self.voltage_scan.current_setpoint = mc
            self.voltage_scan.comments = self.local_measurement_config.comments
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

    def scanLine(self, first_axis, second_axis=-1, second_axis_pos=None):
        """Start line scan."""
        self.field_scan = _FieldScan()

        start = self.local_measurement_config.get_start(first_axis)
        end = self.local_measurement_config.get_end(first_axis)
        step = self.local_measurement_config.get_step(first_axis)
        extra = self.local_measurement_config.get_extra(first_axis)
        vel = self.local_measurement_config.get_velocity(first_axis)

        if start == end:
            raise Exception('Start and end positions are equal.')
            return False

        npts = _np.ceil(round((end - start) / step, 4) + 1)
        scan_list = _np.linspace(start, end, npts)

        integration_time = self.local_measurement_config.integration_time/1000
        aper_displacement = integration_time*vel
        to_pos_scan_list = scan_list + aper_displacement/2
        to_neg_scan_list = (scan_list - aper_displacement/2)[::-1]

        nr_measurements = self.local_measurement_config.nr_measurements
        self.clearGraph()
        self.configureGraph(2*nr_measurements, 'Voltage [V]')
        _QApplication.processEvents()

        voltage_scan_list = []
        for idx in range(2*nr_measurements):
            if self.stop is True:
                return False

            self.nr_measurements_la.setText(
                '{0:d}'.format(int(_np.ceil((idx + 1)/2))))

            # flag to check if sensor is going or returning
            to_pos = not(bool(idx % 2))

            # go to initial position
            if to_pos:
                self.position_list = to_pos_scan_list
            else:
                self.position_list = to_neg_scan_list

            # save positions in voltage scan
            self.voltage_scan = _VoltageScan()
            self.voltage_scan.nr_voltage_scans = 1
            for axis in self.voltage_scan.axis_list:
                if axis == first_axis:
                    setattr(self.voltage_scan, 'pos' + str(first_axis),
                            self.position_list)
                elif axis == second_axis and second_axis_pos is not None:
                    setattr(self.voltage_scan, 'pos' + str(second_axis),
                            second_axis_pos)
                else:
                    pos = self.devices.pmac.get_position(axis)
                    setattr(self.voltage_scan, 'pos' + str(axis), pos)
            _QApplication.processEvents()

            if not self.measureVoltageScan(
                    idx, to_pos, first_axis, start, end, step, extra, npts):
                return False

            if self.stop is True:
                return

            if self.voltage_scan.npts == 0:
                _warnings.warn(
                    'Invalid number of points in voltage scan.')
                if not self.measureVoltageScan(
                        idx, to_pos, first_axis,
                        start, end, step, extra, npts):
                    return False

                if self.stop is True:
                    return

                if self.voltage_scan.npts == 0:
                    raise Exception(
                        'Invalid number of points in voltage scan.')
                    return False

            if self.ui.save_voltage_chb.isChecked():
                if not self.saveVoltageScan():
                    return False

            voltage_scan_list.append(self.voltage_scan.copy())

        if self.local_hall_probe is not None:
            self.field_scan.set_field_scan(
                voltage_scan_list, self.local_hall_probe)
            success = self.saveFieldScan()
        else:
            success = True
        _QApplication.processEvents()

        return success

    def showFieldmapDialog(self):
        """Open fieldmap dialog."""
        field_scan_list = []
        for idn in self.field_scan_id_list:
            fs = _FieldScan(database=self.database, idn=idn)
            field_scan_list.append(fs)

        self.save_fieldmap_dialog.show(
            field_scan_list,
            self.local_hall_probe,
            self.field_scan_id_list)

    def showViewScanDialog(self):
        """Open view data dialog."""
        try:
            if self.local_hall_probe is None:
                voltage_scan_list = []
                for idn in self.voltage_scan_id_list:
                    vs = _VoltageScan(database=self.database, idn=idn)
                    voltage_scan_list.append(vs)

                self.view_scan_dialog.show(
                    voltage_scan_list,
                    self.voltage_scan_id_list,
                    'Voltage [V]')
            else:
                field_scan_list = []
                for idn in self.field_scan_id_list:
                    fs = _FieldScan(database=self.database, idn=idn)
                    field_scan_list.append(fs)

                self.view_scan_dialog.show(
                    field_scan_list,
                    self.field_scan_id_list,
                    'Magnetic Field [T]')
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def startAutomaticMeasurements(self):
        """Configure and emit signal to start automatic ramp measurements."""
        self.configureMeasurement()
        if not self.measurement_configured:
            return

        self.ui.measure_btn.setEnabled(False)
        self.ui.stop_btn.setEnabled(True)
        self.change_current_setpoint.emit(1)

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

        if not self.resetMultimeters():
            return

        msg = 'End of measurement.'
        _QMessageBox.information(
            self, 'Measurement', msg, _QMessageBox.Ok)

    def startVoltageThreads(self):
        """Start threads to read voltage values."""
        if self.local_measurement_config.voltx_enable:
            self.threadx.start()

        if self.local_measurement_config.volty_enable:
            self.thready.start()

        if self.local_measurement_config.voltz_enable:
            self.threadz.start()

    def stopMeasurement(self):
        """Stop measurement to True."""
        try:
            self.stop = True
            self.ui.measure_btn.setEnabled(True)
            self.devices.pmac.stop_all_axis()
            self.ui.stop_btn.setEnabled(False)
            self.ui.clear_graph_btn.setEnabled(True)
            self.ui.nr_measurements_la.setText('')
            self.power_supply_config.update_display = True
            msg = 'The user stopped the measurements.'
            _QMessageBox.information(
                self, 'Abort', msg, _QMessageBox.Ok)
        except Exception:
            self.power_supply_config.update_display = True
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to stop measurements.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def stopTrigger(self):
        """Stop trigger."""
        self.devices.pmac.stop_trigger()
        self.threadx.end_measurement = True
        self.thready.end_measurement = True
        self.threadz.end_measurement = True

    def quitVoltageThreads(self):
        """Quit voltage threads."""
        self.threadx.quit()
        self.thready.quit()
        self.threadz.quit()

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
                self.ui.save_voltage_chb.setChecked(True)
                return True
            else:
                return False

    def updatePowerSupplyConfig(self):
        """Update local power supply configuration."""
        self.local_power_supply_config = self.power_supply_config
        return True

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
        while self.threadx.isRunning() and self.stop is False:
            _QApplication.processEvents()

        while self.thready.isRunning() and self.stop is False:
            _QApplication.processEvents()

        while self.threadz.isRunning() and self.stop is False:
            _QApplication.processEvents()


class VoltageThread(_QThread):
    """Thread to read values from multimeters."""

    def __init__(self, multimeter):
        """Initialize object."""
        super().__init__()
        self.multimeter = multimeter
        self.precision = None
        self.integration_time = None
        self.voltage = _np.array([])
        self.end_measurement = False

    def clear(self):
        """Clear voltage values."""
        self.voltage = _np.array([])
        self.end_measurement = False

    def configure(self, precision, integration_time):
        """Configure voltage thread."""
        self.precision = precision
        self.integration_time = integration_time

    def run(self):
        """Read voltage from the device."""
        self.clear()
        while (self.end_measurement is False):
            if self.multimeter.inst.stb & 128:
                voltage = self.multimeter.read_voltage(self.precision)
                self.voltage = _np.append(self.voltage, voltage)
        else:
            self.multimeter.send_command(self.multimeter.commands.mcount)
            npoints = int(self.multimeter.read_from_device())
            if npoints > 0:
                # ask data from memory
                self.multimeter.send_command(
                    self.multimeter.commands.rmem + str(npoints))

                for idx in range(npoints):
                    voltage = self.multimeter.read_voltage(self.precision)
                    self.voltage = _np.append(self.voltage, voltage)
