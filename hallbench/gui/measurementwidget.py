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
    QObject as _QObject,
    Signal as _Signal,
    )
import qtpy.uic as _uic

from hallbench.gui import utils as _utils
from hallbench.gui.auxiliarywidgets import CurrentPositionWidget \
    as _CurrentPositionWidget
from hallbench.data.configuration import MeasurementConfig \
    as _MeasurementConfig
from hallbench.data.measurement import VoltageScan as _VoltageScan
from hallbench.data.measurement import FieldScan as _FieldScan
from hallbench.devices import (
    pmac as _pmac,
    voltx as _voltx,
    volty as _volty,
    voltz as _voltz,
    multich as _multich,
    dcct as _dcct,
    ps as _ps
    )


class MeasurementWidget(_QWidget):
    """Measurement widget class for the Hall Bench Control application."""

    change_current_setpoint = _Signal([bool])
    turn_off_power_supply_current = _Signal([bool])

    _measurement_axes = [1, 2, 3, 5]
    _update_graph_time_interval = 0.1  # [s]

    def __init__(self, parent=None):
        """Set up the ui, add widgets and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _utils.get_ui_file(self)
        self.ui = _uic.loadUi(uifile, self)

        # add position widget
        self.current_position_widget = _CurrentPositionWidget(self)
        _layout = _QVBoxLayout()
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.addWidget(self.current_position_widget)
        self.ui.wg_position.setLayout(_layout)

        self.measurement_configured = False
        self.local_measurement_config = None
        self.local_measurement_config_id = None
        self.local_hall_probe = None
        self.local_power_supply_config = None
        self.voltage_scan = None
        self.field_scan = None
        self.position_list = []
        self.measurements_id_list = []
        self.field_scan_id_list = []
        self.voltage_scan_id_list = []
        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.stop = False

        # Connect signals and slots
        self.connect_signal_slots()

        # Add legend to plot
        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.pw_graph.graphicsItem())
        self.legend.setAutoFillBackground(1)

        # Update probe names and configuration ID combo box
        self.update_probe_names()
        self.update_configuration_ids()

        # Create voltage threads
        self.threadx = _QThread()
        self.thready = _QThread()
        self.threadz = _QThread()

        self.workerx = VoltageWorker(_voltx)
        self.workerx.moveToThread(self.threadx)
        self.threadx.started.connect(self.workerx.run)
        self.workerx.finished.connect(self.threadx.quit)

        self.workery = VoltageWorker(_volty)
        self.workery.moveToThread(self.thready)
        self.thready.started.connect(self.workery.run)
        self.workery.finished.connect(self.thready.quit)

        self.workerz = VoltageWorker(_voltz)
        self.workerz.moveToThread(self.threadz)
        self.threadz.started.connect(self.workerz.run)
        self.workerz.finished.connect(self.threadz.quit)

    @property
    def database(self):
        """Database filename."""
        return _QApplication.instance().database

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

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.current_position_widget.close()
            self.kill_voltage_threads()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def change_offset_page(self):
        """Change offset stacked widget page."""
        try:
            if self.ui.rbt_measure_offsets.isChecked():
                self.ui.stw_offsets.setCurrentIndex(2)
            elif self.ui.rbt_configure_offsets.isChecked():
                self.ui.stw_offsets.setCurrentIndex(1)
            else:
                self.ui.stw_offsets.setCurrentIndex(0)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def clear(self):
        """Clear."""
        self.workerx.clear()
        self.workery.clear()
        self.workerz.clear()
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
        self.measurements_id_list = []
        self.stop = False
        self.clear_graph()
        self.ui.tbt_view_scan.setEnabled(False)
        self.ui.tbt_clear_graph.setEnabled(False)
        self.ui.tbt_create_fieldmap.setEnabled(False)
        self.ui.tbt_save_scan_files.setEnabled(False)

    def clear_button_clicked(self):
        """Clear current measurement and plots."""
        self.clear_current_measurement()
        self.clear_graph()
        self.ui.tbt_view_scan.setEnabled(False)
        self.ui.tbt_clear_graph.setEnabled(False)
        self.ui.tbt_create_fieldmap.setEnabled(False)
        self.ui.tbt_save_scan_files.setEnabled(False)

    def clear_current_measurement(self):
        """Clear current measurement data."""
        self.voltage_scan = None
        self.field_scan = None
        self.position_list = []
        self.field_scan_id_list = []
        self.voltage_scan_id_list = []

    def clear_graph(self):
        """Clear plots."""
        self.ui.pw_graph.plotItem.curves.clear()
        self.ui.pw_graph.clear()
        self.graphx = []
        self.graphy = []
        self.graphz = []

    def clear_hall_probe(self):
        """Clear hall probe calibration data."""
        self.hall_probe.clear()
        self.ui.cmb_probe_name.setCurrentIndex(-1)

    def clear_load_options(self):
        """Clear load options."""
        self.ui.cmb_idn.setCurrentIndex(-1)

    def configure_graph(self, nr_curves, label):
        """Configure graph."""
        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.legend.removeItem('X')
        self.legend.removeItem('Y')
        self.legend.removeItem('Z')

        for idx in range(nr_curves):
            self.graphx.append(
                self.ui.pw_graph.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(255, 0, 0),
                    symbol='o',
                    symbolPen=(255, 0, 0),
                    symbolSize=4,
                    symbolBrush=(255, 0, 0)))

            self.graphy.append(
                self.ui.pw_graph.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(0, 255, 0),
                    symbol='o',
                    symbolPen=(0, 255, 0),
                    symbolSize=4,
                    symbolBrush=(0, 255, 0)))

            self.graphz.append(
                self.ui.pw_graph.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(0, 0, 255),
                    symbol='o',
                    symbolPen=(0, 0, 255),
                    symbolSize=4,
                    symbolBrush=(0, 0, 255)))

        self.ui.pw_graph.setLabel('bottom', 'Scan Position [mm]')
        self.ui.pw_graph.setLabel('left', label)
        self.ui.pw_graph.showGrid(x=True, y=True)
        self.legend.addItem(self.graphx[0], 'X')
        self.legend.addItem(self.graphy[0], 'Y')
        self.legend.addItem(self.graphz[0], 'Z')

    def configure_measurement(self):
        """Configure measurement."""
        self.clear()

        if (not self.valid_database()
                or not self.update_hall_probe()
                or not self.update_configuration()
                or not self.update_power_supply_config()
                or not self.configure_multimeters()
                or not self.configure_pmac()
                or not self.configure_multichannel()):
            self.measurement_configured = False
        else:
            self.measurement_configured = True

    def configure_multichannel(self):
        """Configure multichannel to monitor dcct current and temperatures."""
        if (not self.ui.chb_save_temperature.isChecked() and
                not self.ui.chb_save_current.isChecked()):
            return True

        if not _multich.connected:
            msg = 'Multichannel not connected.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        try:
            channels = []
            if self.ui.chb_save_temperature.isChecked():
                channels = channels + _multich.voltage_channels
                channels = channels + _multich.temperature_channels

            if len(channels) == 0:
                return True

            _multich.configure(channel_list=channels)
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to configure multichannel.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def configure_multimeters(self):
        """Configure multimeters."""
        if self.local_measurement_config is None:
            return False

        if (self.local_measurement_config.voltx_enable
                and not _voltx.connected):
            msg = 'Multimeter X not connected.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        if (self.local_measurement_config.volty_enable
                and not _volty.connected):
            msg = 'Multimeter Y not connected.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        if (self.local_measurement_config.voltz_enable
                and not _voltz.connected):
            msg = 'Multimeter Z not connected.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        try:
            if self.local_measurement_config.voltx_enable:
                _voltx.configure(
                    self.local_measurement_config.integration_time,
                    self.local_measurement_config.voltage_range)
            if self.local_measurement_config.volty_enable:
                _volty.configure(
                    self.local_measurement_config.integration_time,
                    self.local_measurement_config.voltage_range)
            if self.local_measurement_config.voltz_enable:
                _voltz.configure(
                    self.local_measurement_config.integration_time,
                    self.local_measurement_config.voltage_range)
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to configure multimeters.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def configure_pmac(self):
        """Configure pmac."""
        if not _pmac.connected:
            msg = 'Pmac not connected.'
            _QMessageBox.critical(
                self, 'Failure', msg, _QMessageBox.Ok)
            return False

        try:
            _pmac.set_axis_speed(
                1, self.local_measurement_config.vel_ax1)
            _pmac.set_axis_speed(
                2, self.local_measurement_config.vel_ax2)
            _pmac.set_axis_speed(
                3, self.local_measurement_config.vel_ax3)
            _pmac.set_axis_speed(
                5, self.local_measurement_config.vel_ax5)
            return True
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to configure pmac.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def connect_signal_slots(self):
        """Create signal/slot connections."""
        self.ui.rbt_first_ax1.clicked.connect(self.clear_load_options)
        self.ui.rbt_first_ax2.clicked.connect(self.clear_load_options)
        self.ui.rbt_first_ax3.clicked.connect(self.clear_load_options)
        self.ui.rbt_first_ax5.clicked.connect(self.clear_load_options)
        self.ui.rbt_second_ax1.clicked.connect(self.clear_load_options)
        self.ui.rbt_second_ax2.clicked.connect(self.clear_load_options)
        self.ui.rbt_second_ax3.clicked.connect(self.clear_load_options)
        self.ui.rbt_second_ax5.clicked.connect(self.clear_load_options)
        self.ui.le_magnet_name.editingFinished.connect(self.clear_load_options)
        self.ui.le_current_setpoint.editingFinished.connect(
            self.clear_load_options)
        self.ui.le_temperature.editingFinished.connect(self.clear_load_options)
        self.ui.le_operator.editingFinished.connect(self.clear_load_options)
        self.ui.le_comments.editingFinished.connect(self.clear_load_options)
        self.ui.sb_nr_measurements.valueChanged.connect(
            self.clear_load_options)
        self.ui.cmb_probe_name.currentIndexChanged.connect(
            self.clear_load_options)
        self.ui.cmb_voltage_precision.currentIndexChanged.connect(
            self.clear_load_options)
        self.chb_voltx_enable.stateChanged.connect(self.clear_load_options)
        self.chb_volty_enable.stateChanged.connect(self.clear_load_options)
        self.chb_voltz_enable.stateChanged.connect(self.clear_load_options)
        self.ui.cmb_idn.currentIndexChanged.connect(self.enable_load_db)
        self.ui.tbt_update_idn.clicked.connect(self.update_configuration_ids)

        self.ui.tbt_current_start.clicked.connect(
            self.copy_current_start_position)

        self.ui.le_current_setpoint.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_current_setpoint))

        self.ui.le_start_ax1.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_start_ax1))
        self.ui.le_start_ax2.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_start_ax2))
        self.ui.le_start_ax3.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_start_ax3))
        self.ui.le_start_ax5.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_start_ax5))

        self.ui.le_end_ax1.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_end_ax1))
        self.ui.le_end_ax2.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_end_ax2))
        self.ui.le_end_ax3.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_end_ax3))
        self.ui.le_end_ax5.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_end_ax5))

        self.ui.le_step_ax1.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_step_ax1, positive=True, nonzero=True))
        self.ui.le_step_ax2.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_step_ax2, positive=True, nonzero=True))
        self.ui.le_step_ax3.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_step_ax3, positive=True, nonzero=True))
        self.ui.le_step_ax5.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_step_ax5, positive=True, nonzero=True))

        self.ui.le_extra_ax1.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_extra_ax1, positive=True))
        self.ui.le_extra_ax2.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_extra_ax2, positive=True))
        self.ui.le_extra_ax3.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_extra_ax3, positive=True))
        self.ui.le_extra_ax5.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_extra_ax5, positive=True))

        self.ui.le_vel_ax1.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_vel_ax1, positive=True, nonzero=True))
        self.ui.le_vel_ax2.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_vel_ax2, positive=True, nonzero=True))
        self.ui.le_vel_ax3.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_vel_ax3, positive=True, nonzero=True))
        self.ui.le_vel_ax5.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_vel_ax5, positive=True, nonzero=True))

        self.ui.le_integration_time.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_integration_time, precision=4))

        self.ui.le_voltage_range.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_voltage_range, precision=1))

        self.ui.le_offsetx.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_offsetx))
        self.ui.le_offsety.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_offsety))
        self.ui.le_offsetz.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_offsetz))
        self.ui.le_offset_range.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_offset_range, precision=2))

        self.ui.le_step_ax1.editingFinished.connect(
            lambda: self.update_trigger_step(1))
        self.ui.le_step_ax2.editingFinished.connect(
            lambda: self.update_trigger_step(2))
        self.ui.le_step_ax3.editingFinished.connect(
            lambda: self.update_trigger_step(3))
        self.ui.le_step_ax5.editingFinished.connect(
            lambda: self.update_trigger_step(5))
        self.ui.le_vel_ax1.editingFinished.connect(
            lambda: self.update_trigger_step(1))
        self.ui.le_vel_ax2.editingFinished.connect(
            lambda: self.update_trigger_step(2))
        self.ui.le_vel_ax3.editingFinished.connect(
            lambda: self.update_trigger_step(3))
        self.ui.le_vel_ax5.editingFinished.connect(
            lambda: self.update_trigger_step(5))

        self.ui.rbt_first_ax1.clicked.connect(self.disable_second_axis_button)
        self.ui.rbt_first_ax2.clicked.connect(self.disable_second_axis_button)
        self.ui.rbt_first_ax3.clicked.connect(self.disable_second_axis_button)
        self.ui.rbt_first_ax5.clicked.connect(self.disable_second_axis_button)

        self.ui.rbt_first_ax1.toggled.connect(self.disable_invalid_line_edit)
        self.ui.rbt_first_ax2.toggled.connect(self.disable_invalid_line_edit)
        self.ui.rbt_first_ax3.toggled.connect(self.disable_invalid_line_edit)
        self.ui.rbt_first_ax5.toggled.connect(self.disable_invalid_line_edit)

        self.ui.rbt_second_ax1.toggled.connect(self.disable_invalid_line_edit)
        self.ui.rbt_second_ax2.toggled.connect(self.disable_invalid_line_edit)
        self.ui.rbt_second_ax3.toggled.connect(self.disable_invalid_line_edit)
        self.ui.rbt_second_ax5.toggled.connect(self.disable_invalid_line_edit)

        self.ui.rbt_second_ax1.clicked.connect(
            lambda: self.uncheck_radio_buttons(1))
        self.ui.rbt_second_ax2.clicked.connect(
            lambda: self.uncheck_radio_buttons(2))
        self.ui.rbt_second_ax3.clicked.connect(
            lambda: self.uncheck_radio_buttons(3))
        self.ui.rbt_second_ax5.clicked.connect(
            lambda: self.uncheck_radio_buttons(5))

        self.ui.le_start_ax1.editingFinished.connect(
            lambda: self.fix_end_position_value(1))
        self.ui.le_start_ax2.editingFinished.connect(
            lambda: self.fix_end_position_value(2))
        self.ui.le_start_ax3.editingFinished.connect(
            lambda: self.fix_end_position_value(3))
        self.ui.le_start_ax5.editingFinished.connect(
            lambda: self.fix_end_position_value(5))

        self.ui.le_step_ax1.editingFinished.connect(
            lambda: self.fix_end_position_value(1))
        self.ui.le_step_ax2.editingFinished.connect(
            lambda: self.fix_end_position_value(2))
        self.ui.le_step_ax3.editingFinished.connect(
            lambda: self.fix_end_position_value(3))
        self.ui.le_step_ax5.editingFinished.connect(
            lambda: self.fix_end_position_value(5))

        self.ui.le_end_ax1.editingFinished.connect(
            lambda: self.fix_end_position_value(1))
        self.ui.le_end_ax2.editingFinished.connect(
            lambda: self.fix_end_position_value(2))
        self.ui.le_end_ax3.editingFinished.connect(
            lambda: self.fix_end_position_value(3))
        self.ui.le_end_ax5.editingFinished.connect(
            lambda: self.fix_end_position_value(5))

        self.ui.pbt_load_db.clicked.connect(self.load_config_db)
        self.ui.tbt_save_db.clicked.connect(self.save_config_db)
        self.ui.cmb_probe_name.currentIndexChanged.connect(
            self.load_hall_probe)
        self.ui.tbt_update_probe_name.clicked.connect(self.update_probe_names)
        self.ui.tbt_clear_probe.clicked.connect(self.clear_hall_probe)
        self.ui.pbt_view_probe.clicked.connect(self.show_view_probe_dialog)

        self.ui.rbt_ignore_offsets.toggled.connect(self.change_offset_page)
        self.ui.rbt_configure_offsets.toggled.connect(self.change_offset_page)
        self.ui.rbt_measure_offsets.toggled.connect(self.change_offset_page)

        self.ui.pbt_measure.clicked.connect(self.measure_button_clicked)
        self.ui.pbt_stop.clicked.connect(self.stop_measurement)
        self.ui.tbt_create_fieldmap.clicked.connect(self.show_fieldmap_dialog)
        self.ui.tbt_save_scan_files.clicked.connect(self.save_field_scan_files)
        self.ui.tbt_view_scan.clicked.connect(self.show_view_scan_dialog)
        self.ui.tbt_clear_graph.clicked.connect(self.clear_button_clicked)

    def copy_current_start_position(self):
        """Copy current start position to line edits."""
        for axis in self._measurement_axes:
            le_start = getattr(self.ui, 'le_start_ax' + str(axis))
            if axis in self.positions.keys():
                le_start.setText('{0:0.4f}'.format(self.positions[axis]))
            else:
                le_start.setText('')

    def disable_invalid_line_edit(self):
        """Disable invalid line edit."""
        for axis in self._measurement_axes:
            rbt_first = getattr(self.ui, 'rbt_first_ax' + str(axis))
            rbt_second = getattr(self.ui, 'rbt_second_ax' + str(axis))
            le_step = getattr(self.ui, 'le_step_ax' + str(axis))
            le_end = getattr(self.ui, 'le_end_ax' + str(axis))
            le_extra = getattr(self.ui, 'le_extra_ax' + str(axis))
            if rbt_first.isChecked() or rbt_second.isChecked():
                le_step.setEnabled(True)
                le_end.setEnabled(True)
                if rbt_first.isChecked():
                    le_extra.setEnabled(True)
                else:
                    le_extra.setEnabled(False)
                    le_extra.setText('')
            else:
                le_step.setEnabled(False)
                le_step.setText('')
                le_end.setEnabled(False)
                le_end.setText('')
                le_extra.setEnabled(False)
                le_extra.setText('')

    def disable_second_axis_button(self):
        """Disable invalid second axis radio buttons."""
        for axis in self._measurement_axes:
            rbt_first = getattr(self.ui, 'rbt_first_ax' + str(axis))
            rbt_second = getattr(self.ui, 'rbt_second_ax' + str(axis))
            if rbt_first.isChecked():
                rbt_second.setChecked(False)
                rbt_second.setEnabled(False)
            else:
                if axis != 5:
                    rbt_second.setEnabled(True)

    def enable_load_db(self):
        """Enable button to load configuration from database."""
        if self.ui.cmb_idn.currentIndex() != -1:
            self.ui.pbt_load_db.setEnabled(True)
        else:
            self.ui.pbt_load_db.setEnabled(False)

    def end_automatic_measurements(self, setpoint_changed):
        """End automatic measurements."""
        if not self.reset_multimeters():
            return

        if not setpoint_changed:
            msg = ('Automatic measurements failed. ' +
                   'Current setpoint not changed.')
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        self.ui.pbt_stop.setEnabled(False)
        self.ui.pbt_measure.setEnabled(True)
        self.ui.tbt_create_fieldmap.setEnabled(True)
        self.ui.tbt_save_scan_files.setEnabled(False)
        self.ui.tbt_view_scan.setEnabled(False)
        self.ui.tbt_clear_graph.setEnabled(True)
        self.turn_off_power_supply_current.emit(True)

        msg = 'End of automatic measurements.'
        _QMessageBox.information(
            self, 'Measurements', msg, _QMessageBox.Ok)

    def fix_end_position_value(self, axis):
        """Fix end position value."""
        try:
            le_start = getattr(self.ui, 'le_start_ax' + str(axis))
            start = _utils.get_value_from_string(le_start.text())
            if start is None:
                return

            le_step = getattr(self.ui, 'le_step_ax' + str(axis))
            step = _utils.get_value_from_string(le_step.text())
            if step is None:
                return

            le_end = getattr(self.ui, 'le_end_ax' + str(axis))
            end = _utils.get_value_from_string(le_end.text())
            if end is None:
                return

            npts = _np.round(round((end - start) / step, 4) + 1)
            if start <= end:
                corrected_end = start + (npts-1)*step
            else:
                corrected_end = start
            le_end.setText('{0:0.4f}'.format(corrected_end))

        except Exception:
            pass

    def get_axis_param(self, param, axis):
        """Get axis parameter."""
        le = getattr(self.ui, 'le_' + param + '_ax' + str(axis))
        return _utils.get_value_from_string(le.text())

    def get_fixed_axes(self):
        """Get fixed axes."""
        first_axis = self.local_measurement_config.first_axis
        second_axis = self.local_measurement_config.second_axis
        fixed_axes = [a for a in self._measurement_axes]
        fixed_axes.remove(first_axis)
        if second_axis != -1:
            fixed_axes.remove(second_axis)
        return fixed_axes

    def kill_voltage_threads(self):
        """Kill threads."""
        try:
            del self.threadx
            del self.thready
            del self.threadz
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def load_config(self):
        """Set measurement parameters."""
        try:
            self.ui.le_magnet_name.setText(self.measurement_config.magnet_name)

            current_sp = self.measurement_config.current_setpoint
            if current_sp is None:
                self.ui.le_current_setpoint.setText('')
            else:
                self.ui.le_current_setpoint.setText(str(current_sp))

            idx = self.ui.cmb_probe_name.findText(
                self.measurement_config.probe_name)
            self.ui.cmb_probe_name.setCurrentIndex(idx)

            self.ui.le_temperature.setText(self.measurement_config.temperature)
            self.ui.le_operator.setText(self.measurement_config.operator)
            self.ui.le_comments.setText(self.measurement_config.comments)

            self.ui.chb_voltx_enable.setChecked(
                self.measurement_config.voltx_enable)
            self.ui.chb_volty_enable.setChecked(
                self.measurement_config.volty_enable)
            self.ui.chb_voltz_enable.setChecked(
                self.measurement_config.voltz_enable)

            self.ui.le_integration_time.setText('{0:0.4f}'.format(
                self.measurement_config.integration_time))

            precision = self.measurement_config.voltage_precision
            if precision == 4:
                self.ui.cmb_voltage_precision.setCurrentIndex(0)
            elif precision == 5:
                self.ui.cmb_voltage_precision.setCurrentIndex(1)

            self.ui.le_voltage_range.setText(str(
                self.measurement_config.voltage_range))

            self.ui.sb_nr_measurements.setValue(
                self.measurement_config.nr_measurements)

            first_axis = self.measurement_config.first_axis
            rbt_first = getattr(self.ui, 'rbt_first_ax' + str(first_axis))
            rbt_first.setChecked(True)

            self.disable_second_axis_button()

            second_axis = self.measurement_config.second_axis
            if second_axis != -1:
                rbt_second = getattr(
                    self.ui, 'rbt_second_ax' + str(second_axis))
                rbt_second.setChecked(True)
                self.uncheck_radio_buttons(second_axis)
            else:
                self.uncheck_radio_buttons(second_axis)

            self.ui.rbt_ignore_offsets.setChecked(
                not self.measurement_config.subtract_voltage_offset)
            self.ui.rbt_measure_offsets.setChecked(
                self.measurement_config.subtract_voltage_offset)

            self.ui.chb_save_voltage.setChecked(
                self.measurement_config.save_voltage)
            self.ui.chb_save_current.setChecked(
                self.measurement_config.save_current)
            self.ui.chb_save_temperature.setChecked(
                self.measurement_config.save_temperature)
            self.ui.chb_automatic_ramp.setChecked(
                self.measurement_config.automatic_ramp)

            for axis in self._measurement_axes:
                le_start = getattr(self.ui, 'le_start_ax' + str(axis))
                value = self.measurement_config.get_start(axis)
                le_start.setText('{0:0.4f}'.format(value))

                le_step = getattr(self.ui, 'le_step_ax' + str(axis))
                value = self.measurement_config.get_step(axis)
                le_step.setText('{0:0.4f}'.format(value))

                le_end = getattr(self.ui, 'le_end_ax' + str(axis))
                value = self.measurement_config.get_end(axis)
                le_end.setText('{0:0.4f}'.format(value))

                le_extra = getattr(self.ui, 'le_extra_ax' + str(axis))
                value = self.measurement_config.get_extra(axis)
                le_extra.setText('{0:0.4f}'.format(value))

                le_vel = getattr(self.ui, 'le_vel_ax' + str(axis))
                value = self.measurement_config.get_velocity(axis)
                le_vel.setText('{0:0.4f}'.format(value))

            self.disable_invalid_line_edit()
            self.update_trigger_step(first_axis)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to load configuration.'
            _QMessageBox.critical(
                self, 'Failure', msg, _QMessageBox.Ok)

    def load_config_db(self):
        """Load configuration from database to set measurement parameters."""
        try:
            idn = int(self.ui.cmb_idn.currentText())
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.critical(
                self, 'Failure', 'Invalid database ID.', _QMessageBox.Ok)
            return

        self.update_configuration_ids()
        idx = self.ui.cmb_idn.findText(str(idn))
        if idx == -1:
            self.ui.cmb_idn.setCurrentIndex(-1)
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

        self.load_config()
        self.ui.cmb_idn.setCurrentIndex(self.ui.cmb_idn.findText(str(idn)))
        self.ui.pbt_load_db.setEnabled(False)

    def load_hall_probe(self):
        """Load hall probe from database."""
        self.hall_probe.clear()
        probe_name = self.ui.cmb_probe_name.currentText()
        if len(probe_name) == 0:
            return

        try:
            idn = self.hall_probe.get_hall_probe_id(self.database, probe_name)
            if idn is not None:
                self.hall_probe.read_from_database(self.database, idn)
            else:
                self.ui.cmb_probe_name.setCurrentIndex(-1)
                msg = 'Invalid probe name.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to load Hall probe from database.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def measure(self):
        """Perform one measurement."""
        if not self.measurement_configured or self.stop is True:
            return False

        try:
            self.power_supply_config.update_display = False

            nr_measurements = self.local_measurement_config.nr_measurements
            first_axis = self.local_measurement_config.first_axis
            second_axis = self.local_measurement_config.second_axis
            fixed_axes = self.get_fixed_axes()

            for nmeas in range(nr_measurements):
                self.la_nr_measurements.setText('{0:d}'.format(nmeas+1))

                for axis in fixed_axes:
                    if self.stop is True:
                        return False
                    pos = self.local_measurement_config.get_start(axis)
                    self.move_axis(axis, pos)

                if second_axis != -1:
                    start = self.local_measurement_config.get_start(
                        second_axis)
                    end = self.local_measurement_config.get_end(
                        second_axis)
                    step = self.local_measurement_config.get_step(
                        second_axis)
                    npts = _np.abs(
                        _np.ceil(round((end - start) / step, 4) + 1))
                    pos_list = _np.linspace(start, end, npts)

                    for pos in pos_list:
                        if self.stop is True:
                            return False
                        self.move_axis(second_axis, pos)
                        if not self.measure_field_scan(
                                first_axis, second_axis, pos):
                            return False

                    # move to initial position
                    if self.stop is True:
                        return False
                    lmc = self.local_measurement_config
                    second_axis_start = lmc.get_start(
                        second_axis)
                    self.move_axis(second_axis, second_axis_start)

                else:
                    if self.stop is True:
                        return False
                    if not self.measure_field_scan(first_axis):
                        return False

                if self.stop is True:
                    return False

                # move to initial position
                first_axis_start = self.local_measurement_config.get_start(
                    first_axis)
                self.move_axis(first_axis, first_axis_start)

                # self.plot_field()
                self.quit_voltage_threads()

            if nr_measurements > 1:
                lid = self.field_scan_id_list
                if len(lid) == nr_measurements:
                    a = _np.transpose([lid])
                else:
                    a = _np.reshape(
                        lid, (nr_measurements, int(len(lid)/nr_measurements)))
                for lt in a.tolist():
                    self.measurements_id_list.append(lt)
            else:
                self.measurements_id_list.append(self.field_scan_id_list)

            self.power_supply_config.update_display = True
            self.ui.la_nr_measurements.setText('')
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.quit_voltage_threads()
            if self.local_measurement_config.automatic_ramp:
                self.turn_off_power_supply_current.emit(True)
            self.power_supply_config.update_display = True
            self.ui.la_nr_measurements.setText('')
            msg = 'Measurement failure.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def measure_and_emit_signal(self):
        """Measure and emit signal to change current setpoint."""
        self.clear_current_measurement()

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

        if not self.save_configuration():
            return

        if not self.measure():
            return

        self.change_current_setpoint.emit(True)

    def measure_button_clicked(self):
        """Start measurements."""
        if self.ui.chb_automatic_ramp.isChecked():
            self.start_automatic_measurements()
        else:
            self.start_measurement()

    def measure_current_and_temperature(self):
        """Measure current and temperatures."""
        if (not self.ui.chb_save_temperature.isChecked() and
                not self.ui.chb_save_current.isChecked()):
            return True

        try:
            temperature_dict = {}
            dcct_head = self.local_power_supply_config.dcct_head
            ps_type = self.local_power_supply_config.ps_type
            ts = _time.time()

            # Read power supply current
            if self.ui.chb_save_current.isChecked() and ps_type is not None:
                _ps.SetSlaveAdd(ps_type)
                ps_current = float(_ps.Read_iLoad1())
                self.voltage_scan.ps_current_avg = ps_current

            # Read dcct current
            if dcct_head is not None:
                dcct_current = _dcct.read_current(
                    dcct_head=dcct_head)
                self.voltage_scan.dcct_current_avg = dcct_current

            # Read multichannel
            r = _multich.get_converted_readings()
            channels = _multich.config_channels
            for i, ch in enumerate(channels):
                temperature_dict[ch] = [[ts, r[i]]]
            _QApplication.processEvents()

            if self.ui.chb_save_temperature.isChecked():
                self.voltage_scan.temperature = temperature_dict

            _QApplication.processEvents()
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return False

    def measure_field_scan(
            self, first_axis, second_axis=-1, second_axis_pos=None):
        """Start line scan."""
        self.field_scan = _FieldScan()

        start = self.local_measurement_config.get_start(first_axis)
        end = self.local_measurement_config.get_end(first_axis)
        step = self.local_measurement_config.get_step(first_axis)
        extra = self.local_measurement_config.get_extra(first_axis)
        vel = self.local_measurement_config.get_velocity(first_axis)

        if start == end:
            raise Exception('Start and end positions are equal.')

        npts = _np.ceil(round((end - start) / step, 4) + 1)
        scan_list = _np.linspace(start, end, npts)

        integration_time = self.local_measurement_config.integration_time/1000
        aper_displacement = integration_time*vel
        to_pos_scan_list = scan_list + aper_displacement/2
        to_neg_scan_list = (scan_list - aper_displacement/2)[::-1]

        self.clear_graph()
        self.configure_graph(2, 'Voltage [V]')
        _QApplication.processEvents()

        voltage_scan_list = []
        for idx in range(2):
            if self.stop is True:
                return False

            # flag to check if sensor is going or returning
            to_pos = not bool(idx % 2)

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
                    pos = _pmac.get_position(axis)
                    setattr(self.voltage_scan, 'pos' + str(axis), pos)

            _QApplication.processEvents()

            for axis in self.voltage_scan.axis_list:
                if axis not in [first_axis, second_axis]:
                    pos = getattr(self.voltage_scan, 'pos' + str(axis))
                    if len(pos) == 0:
                        pos = _pmac.get_position(axis)
                        setattr(self.voltage_scan, 'pos' + str(axis), pos)

            _QApplication.processEvents()

            if not self.measure_voltage_scan(
                    idx, to_pos, first_axis, start, end, step, extra, npts):
                return False

            for axis in self.voltage_scan.axis_list:
                if axis not in [first_axis, second_axis]:
                    pos = getattr(self.voltage_scan, 'pos' + str(axis))
                    if len(pos) == 0:
                        pos = _pmac.get_position(axis)
                        setattr(self.voltage_scan, 'pos' + str(axis), pos)

            _QApplication.processEvents()

            if self.stop is True:
                return False

            if self.voltage_scan.npts == 0:
                _warnings.warn(
                    'Invalid number of points in voltage scan.')
                if not self.measure_voltage_scan(
                        idx, to_pos, first_axis,
                        start, end, step, extra, npts):
                    return False

                if self.stop is True:
                    return False

                if self.voltage_scan.npts == 0:
                    raise Exception(
                        'Invalid number of points in voltage scan.')

            _QApplication.processEvents()

            for axis in self.voltage_scan.axis_list:
                if axis not in [first_axis, second_axis]:
                    pos = getattr(self.voltage_scan, 'pos' + str(axis))
                    if len(pos) == 0:
                        pos = _pmac.get_position(axis)
                        setattr(self.voltage_scan, 'pos' + str(axis), pos)

            _QApplication.processEvents()

            if self.ui.chb_save_voltage.isChecked():
                if not self.save_voltage_scan():
                    return False

            voltage_scan_list.append(self.voltage_scan.copy())

        if self.local_hall_probe is not None:
            self.field_scan.set_field_scan(
                voltage_scan_list, self.local_hall_probe)
            success = self.save_field_scan()
        else:
            success = True
        _QApplication.processEvents()

        return success

    def measure_voltage_scan(
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
            self.move_axis(axis, start - extra)
        else:
            self.move_axis(axis, end + extra)
        _QApplication.processEvents()

        if self.stop is True:
            return False

        self.measure_current_and_temperature()
        _QApplication.processEvents()

        if self.stop is True:
            return False
        else:
            if to_pos:
                _pmac.set_trigger(
                    axis, start, step, 10, npts, 1)
            else:
                _pmac.set_trigger(
                    axis, end, step*(-1), 10, npts, 1)

        formtype = self.local_measurement_config.voltage_precision
        if self.local_measurement_config.voltx_enable:
            _voltx.configure_reading_format(formtype)
        if self.local_measurement_config.volty_enable:
            _volty.configure_reading_format(formtype)
        if self.local_measurement_config.voltz_enable:
            _voltz.configure_reading_format(formtype)
        _QApplication.processEvents()

        self.start_voltage_threads()

        if self.stop is False:
            if to_pos:
                self.move_axis_and_update_graph(axis, end + extra, idx)
            else:
                self.move_axis_and_update_graph(axis, start - extra, idx)

        self.stop_trigger()
        self.wait_voltage_threads()

        _QApplication.processEvents()

        self.voltage_scan.avgx = self.workerx.voltage
        self.voltage_scan.avgy = self.workery.voltage
        self.voltage_scan.avgz = self.workerz.voltage

        _QApplication.processEvents()
        self.quit_voltage_threads()

        if axis == 5:
            npts = len(self.voltage_scan.scan_pos)
            self.voltage_scan.avgx = self.voltage_scan.avgx[:npts]
            self.voltage_scan.avgy = self.voltage_scan.avgy[:npts]
            self.voltage_scan.avgz = self.voltage_scan.avgz[:npts]

        if self.voltage_scan.npts == 0:
            return True

        if not to_pos:
            self.voltage_scan.reverse()

        if self.local_measurement_config.subtract_voltage_offset:
            if self.ui.rbt_measure_offsets.isChecked():
                offset_range = _utils.get_value_from_string(
                    self.ui.le_offset_range.text())
                scan_pos = self.voltage_scan.scan_pos
                if scan_pos[-1] - scan_pos[0] <= offset_range:
                    self.voltage_scan.offsetx_start = self.voltage_scan.avgx[0]
                    self.voltage_scan.offsetx_end = self.voltage_scan.avgx[-1]
                    self.voltage_scan.offsety_start = self.voltage_scan.avgy[0]
                    self.voltage_scan.offsety_end = self.voltage_scan.avgy[-1]
                    self.voltage_scan.offsetz_start = self.voltage_scan.avgz[0]
                    self.voltage_scan.offsetz_end = self.voltage_scan.avgz[-1]
                else:
                    idx_start = _np.where(_np.cumsum(_np.diff(
                        scan_pos)) >= offset_range)[0][0] + 1
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
            elif self.ui.rbt_configure_offsets.isChecked():
                offsetx = _utils.get_value_from_string(
                    self.ui.le_offsetx.text())/1000
                offsety = _utils.get_value_from_string(
                    self.ui.le_offsety.text())/1000
                offsetz = _utils.get_value_from_string(
                    self.ui.le_offsetz.text())/1000
                self.voltage_scan.offsetx_start = offsetx
                self.voltage_scan.offsetx_end = offsetx
                self.voltage_scan.offsety_start = offsety
                self.voltage_scan.offsety_end = offsety
                self.voltage_scan.offsetz_start = offsetz
                self.voltage_scan.offsetz_end = offsetz

        if self.stop is True:
            return False
        else:
            return True

    def move_axis(self, axis, position):
        """Move bench axis."""
        if self.stop is False:
            _pmac.move_axis(axis, position)
            status = _pmac.axis_status(axis)
            while (status is None or (status & 1) == 0) and self.stop is False:
                status = _pmac.axis_status(axis)
                _QApplication.processEvents()

    def move_axis_and_update_graph(self, axis, position, idx):
        """Move bench axis and update plot with the measure data."""
        if self.stop is False:
            _pmac.move_axis(axis, position)
            status = _pmac.axis_status(axis)
            while (status is None or (status & 1) == 0) and self.stop is False:
                self.plot_voltage(idx)
                status = _pmac.axis_status(axis)
                _QApplication.processEvents()
                _time.sleep(self._update_graph_time_interval)

    def plot_field(self):
        """Plot field scan."""
        if self.local_hall_probe is None:
            return

        field_scan_list = []
        for idn in self.field_scan_id_list:
            fs = _FieldScan(database=self.database, idn=idn)
            field_scan_list.append(fs)

        self.clear_graph()
        nr_curves = len(field_scan_list)
        self.configure_graph(nr_curves, 'Magnetic Field [T]')

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            for i in range(nr_curves):
                fs = field_scan_list[i]
                positions = fs.scan_pos
                self.graphx[i].setData(positions, fs.avgx)
                self.graphy[i].setData(positions, fs.avgy)
                self.graphz[i].setData(positions, fs.avgz)

    def plot_voltage(self, idx):
        """Plot voltage values."""
        npts = len(self.position_list)
        voltagex = [v for v in self.workerx.voltage[:npts]]
        voltagey = [v for v in self.workery.voltage[:npts]]
        voltagez = [v for v in self.workerz.voltage[:npts]]

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            self.graphx[idx].setData(
                self.position_list[:len(voltagex)], voltagex)
            self.graphy[idx].setData(
                self.position_list[:len(voltagey)], voltagey)
            self.graphz[idx].setData(
                self.position_list[:len(voltagez)], voltagez)

    def reset_multimeters(self):
        """Reset connected multimeters."""
        try:
            if _voltx.connected:
                _voltx.reset()

            if _volty.connected:
                _volty.reset()

            if _voltz.connected:
                _voltz.reset()

            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to reset multimeters.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def save_config_db(self):
        """Save configuration to database."""
        self.ui.cmb_idn.setCurrentIndex(-1)
        if self.database is not None and _os.path.isfile(self.database):
            try:
                if self.update_configuration():
                    idn = self.measurement_config.save_to_database(
                        self.database)
                    self.ui.cmb_idn.addItem(str(idn))
                    self.ui.cmb_idn.setCurrentIndex(self.ui.cmb_idn.count()-1)
                    self.ui.pbt_load_db.setEnabled(False)
            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                msg = 'Failed to save configuration to database.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
        else:
            msg = 'Invalid database filename.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def save_configuration(self):
        """Save configuration to database table."""
        try:
            text = self.ui.cmb_idn.currentText()
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
                self.update_configuration_ids()
                idx = self.ui.cmb_idn.findText(str(idn))
                self.ui.cmb_idn.setCurrentIndex(idx)
                self.ui.pbt_load_db.setEnabled(False)

            self.local_measurement_config_id = idn
            return True
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save configuration to database.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def save_field_scan(self):
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
            try:
                idn = self.field_scan.save_to_database(self.database)
                self.field_scan_id_list.append(idn)
                return True

            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                msg = 'Failed to save FieldScan to database'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
                return False

    def save_field_scan_files(self):
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

    def save_voltage_scan(self):
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
            self.voltage_scan.comments = (
                self.local_measurement_config.comments)
            self.voltage_scan.configuration_id = (
                self.local_measurement_config_id)

            idn = self.voltage_scan.save_to_database(self.database)
            self.voltage_scan_id_list.append(idn)
            return True

        except Exception:
            try:
                idn = self.voltage_scan.save_to_database(self.database)
                self.voltage_scan_id_list.append(idn)
                return True

            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                msg = 'Failed to save VoltageScan to database'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
                return False

    def set_float_line_edit_text(
            self, line_edit, precision=4, expression=True,
            positive=False, nonzero=False):
        """Set the line edit string format for float value."""
        try:
            if line_edit.isModified():
                self.clear_load_options()
                _utils.set_float_line_edit_text(
                    line_edit, precision, expression, positive, nonzero)
        except Exception:
            pass

    def show_fieldmap_dialog(self):
        """Open fieldmap dialog."""
        if len(self.measurements_id_list) > 1:
            field_scan_id_list = self.measurements_id_list
        else:
            field_scan_id_list = self.field_scan_id_list

        self.save_fieldmap_dialog.show(
            field_scan_id_list, self.local_hall_probe)

    def show_view_probe_dialog(self):
        """Open view probe dialog."""
        self.view_probe_dialog.show(self.hall_probe)

    def show_view_scan_dialog(self):
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

    def start_automatic_measurements(self):
        """Configure and emit signal to start automatic ramp measurements."""
        self.measurements_id_list = []
        self.configure_measurement()
        if not self.measurement_configured:
            return

        self.ui.pbt_measure.setEnabled(False)
        self.ui.pbt_stop.setEnabled(True)
        self.change_current_setpoint.emit(1)

    def start_measurement(self):
        """Configure devices and start measurement."""
        self.measurements_id_list = []
        self.configure_measurement()
        if not self.measurement_configured:
            return

        if not self.save_configuration():
            return

        self.ui.pbt_measure.setEnabled(False)
        self.ui.pbt_stop.setEnabled(True)

        if not self.measure():
            return

        if not self.reset_multimeters():
            return

        self.ui.pbt_stop.setEnabled(False)
        self.ui.pbt_measure.setEnabled(True)
        self.ui.tbt_save_scan_files.setEnabled(True)
        self.ui.tbt_view_scan.setEnabled(True)
        self.ui.tbt_clear_graph.setEnabled(True)
        if self.local_hall_probe is not None:
            self.ui.tbt_create_fieldmap.setEnabled(True)

        msg = 'End of measurement.'
        _QMessageBox.information(
            self, 'Measurement', msg, _QMessageBox.Ok)

    def start_voltage_threads(self):
        """Start threads to read voltage values."""
        if self.local_measurement_config.voltx_enable:
            self.threadx.start()

        if self.local_measurement_config.volty_enable:
            self.thready.start()

        if self.local_measurement_config.voltz_enable:
            self.threadz.start()

    def stop_measurement(self):
        """Stop measurement to True."""
        try:
            self.stop = True
            _pmac.stop_all_axis()
            self.ui.pbt_measure.setEnabled(True)
            self.ui.pbt_stop.setEnabled(False)
            self.ui.tbt_clear_graph.setEnabled(True)
            self.ui.la_nr_measurements.setText('')
            self.power_supply_config.update_display = True
            msg = 'The user stopped the measurements.'
            _QMessageBox.information(
                self, 'Abort', msg, _QMessageBox.Ok)
        except Exception:
            self.stop = True
            _pmac.stop_all_axis()
            self.power_supply_config.update_display = True
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to stop measurements.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def stop_trigger(self):
        """Stop trigger."""
        _pmac.stop_trigger()
        self.workerx.end_measurement = True
        self.workery.end_measurement = True
        self.workerz.end_measurement = True

    def uncheck_radio_buttons(self, selected_axis):
        """Uncheck radio buttons."""
        axes = [a for a in self._measurement_axes if a != selected_axis]
        for axis in axes:
            rbt_second = getattr(self.ui, 'rbt_second_ax' + str(axis))
            rbt_second.setChecked(False)

    def update_configuration(self):
        """Update measurement configuration parameters."""
        try:
            self.measurement_config.clear()

            _s = self.ui.le_magnet_name.text().strip()
            self.measurement_config.magnet_name = _s if len(_s) != 0 else None

            current_setpoint = _utils.get_value_from_string(
                self.ui.le_current_setpoint.text())
            self.measurement_config.current_setpoint = current_setpoint

            _s = self.ui.cmb_probe_name.currentText().strip()
            self.measurement_config.probe_name = _s if len(_s) != 0 else None

            _s = self.ui.le_temperature.text().strip()
            self.measurement_config.temperature = _s if len(_s) != 0 else None

            _s = self.ui.le_operator.text().strip()
            self.measurement_config.operator = _s if len(_s) != 0 else None

            _s = self.ui.le_comments.text().strip()
            self.measurement_config.comments = _s if len(_s) != 0 else ''

            _voltx_enable = self.ui.chb_voltx_enable.isChecked()
            self.measurement_config.voltx_enable = _voltx_enable

            _volty_enable = self.ui.chb_volty_enable.isChecked()
            self.measurement_config.volty_enable = _volty_enable

            _voltz_enable = self.ui.chb_voltz_enable.isChecked()
            self.measurement_config.voltz_enable = _voltz_enable

            precision = self.ui.cmb_voltage_precision.currentText().lower()
            if precision == 'single':
                self.measurement_config.voltage_precision = 4
            elif precision == 'double':
                self.measurement_config.voltage_precision = 5

            nr_meas = self.ui.sb_nr_measurements.value()
            self.measurement_config.nr_measurements = nr_meas

            integration_time = _utils.get_value_from_string(
                self.ui.le_integration_time.text())
            self.measurement_config.integration_time = integration_time

            voltage_range = _utils.get_value_from_string(
                self.ui.le_voltage_range.text())
            self.measurement_config.voltage_range = voltage_range

            _ch = self.ui.rbt_ignore_offsets.isChecked()
            self.measurement_config.subtract_voltage_offset = 0 if _ch else 1

            _ch = self.ui.chb_save_voltage.isChecked()
            self.measurement_config.save_voltage = 1 if _ch else 0

            _ch = self.ui.chb_save_current.isChecked()
            self.measurement_config.save_current = 1 if _ch else 0

            _ch = self.ui.chb_save_temperature.isChecked()
            self.measurement_config.save_temperature = 1 if _ch else 0

            if self.ui.chb_automatic_ramp.isChecked():
                self.measurement_config.automatic_ramp = 1
                self.ui.sb_nr_measurements.setValue(1)
                self.measurement_config.nr_measurements = 1
            else:
                self.measurement_config.automatic_ramp = 0

            for axis in self._measurement_axes:
                rbt_first = getattr(self.ui, 'rbt_first_ax' + str(axis))
                rbt_second = getattr(self.ui, 'rbt_second_ax' + str(axis))
                if rbt_first.isChecked():
                    self.measurement_config.first_axis = axis
                elif rbt_second.isChecked():
                    self.measurement_config.second_axis = axis

                start = self.get_axis_param('start', axis)
                self.measurement_config.set_start(axis, start)

                le_step = getattr(self.ui, 'le_step_ax' + str(axis))
                if le_step.isEnabled():
                    step = self.get_axis_param('step', axis)
                    self.measurement_config.set_step(axis, step)
                else:
                    self.measurement_config.set_step(axis, 0.0)

                le_end = getattr(self.ui, 'le_end_ax' + str(axis))
                if le_end.isEnabled():
                    end = self.get_axis_param('end', axis)
                    self.measurement_config.set_end(axis, end)
                else:
                    self.measurement_config.set_end(axis, start)

                le_extra = getattr(self.ui, 'le_extra_ax' + str(axis))
                if le_extra.isEnabled():
                    extra = self.get_axis_param('extra', axis)
                    self.measurement_config.set_extra(axis, extra)
                else:
                    self.measurement_config.set_extra(axis, 0.0)

                vel = self.get_axis_param('vel', axis)
                self.measurement_config.set_velocity(axis, vel)

            if self.measurement_config.second_axis is None:
                self.measurement_config.second_axis = -1

            if self.measurement_config.valid_data():
                first_axis = self.measurement_config.first_axis
                step = self.measurement_config.get_step(first_axis)
                vel = self.measurement_config.get_velocity(first_axis)
                max_int_time = 100*(_np.abs(step/vel)*1000 - 5)

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

    def update_configuration_ids(self):
        """Update combo box ids."""
        current_text = self.ui.cmb_idn.currentText()
        load_enabled = self.ui.pbt_load_db.isEnabled()
        self.ui.cmb_idn.clear()
        try:
            idns = self.measurement_config.get_table_column(
                self.database, 'id')
            self.ui.cmb_idn.clear()
            self.ui.cmb_idn.addItems([str(idn) for idn in idns])
            if len(current_text) == 0:
                self.ui.cmb_idn.setCurrentIndex(self.ui.cmb_idn.count()-1)
                self.ui.pbt_load_db.setEnabled(True)
            else:
                self.ui.cmb_idn.setCurrentText(current_text)
                self.ui.pbt_load_db.setEnabled(load_enabled)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def update_current_setpoint(self, current_setpoint):
        """Update current setpoint value."""
        try:
            current_value_str = str(current_setpoint)
            self.measurement_config.current_setpoint = current_setpoint
            self.ui.le_current_setpoint.setText(current_value_str)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to update current setpoint.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def update_hall_probe(self):
        """Update hall probe."""
        self.local_hall_probe = self.hall_probe.copy()
        if self.local_hall_probe.valid_data():
            return True
        else:
            self.local_hall_probe = None
            msg = 'Invalid hall probe. Measure only voltage data?'
            reply = _QMessageBox.question(
                self, 'Message', msg, _QMessageBox.No, _QMessageBox.Yes)
            if reply == _QMessageBox.Yes:
                self.ui.chb_save_voltage.setChecked(True)
                return True
            else:
                return False

    def update_power_supply_config(self):
        """Update local power supply configuration."""
        self.local_power_supply_config = self.power_supply_config
        return True

    def update_probe_names(self):
        """Update combo box with database probe names."""
        current_text = self.ui.cmb_probe_name.currentText()
        self.ui.cmb_probe_name.clear()
        try:
            probe_names = self.hall_probe.get_table_column(
                self.database, 'probe_name')
            self.ui.cmb_probe_name.addItems(probe_names)
            if len(current_text) == 0:
                self.ui.cmb_probe_name.setCurrentIndex(-1)
            else:
                self.ui.cmb_probe_name.setCurrentText(current_text)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def update_trigger_step(self, axis):
        """Update trigger step."""
        try:
            rbt_first = getattr(self.ui, 'rbt_first_ax' + str(axis))
            if rbt_first.isChecked():
                step = self.get_axis_param('step', axis)
                vel = self.get_axis_param('vel', axis)
                if step is not None and vel is not None:
                    max_int_time = _np.abs(step/vel)*1000 - 5
                    _s = 'Max. Int. Time [ms]:  {0:.4f}'.format(
                        max_int_time)
                    self.ui.la_max_integration_time.setText(_s)
                else:
                    self.ui.la_max_integration_time.setText('')
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def quit_voltage_threads(self):
        """Quit voltage threads."""
        self.threadx.quit()
        self.thready.quit()
        self.threadz.quit()

    def valid_database(self):
        """Check if the database filename is valid."""
        if (self.database is not None
                and len(self.database) != 0
                and _os.path.isfile(self.database)):
            return True
        else:
            msg = 'Invalid database filename.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def wait_voltage_threads(self):
        """Wait threads."""
        while self.threadx.isRunning() and self.stop is False:
            _QApplication.processEvents()

        while self.thready.isRunning() and self.stop is False:
            _QApplication.processEvents()

        while self.threadz.isRunning() and self.stop is False:
            _QApplication.processEvents()


class VoltageWorker(_QObject):
    """Read values from multimeters."""

    finished = _Signal([bool])

    def __init__(self, multimeter):
        """Initialize object."""
        super().__init__()
        self.multimeter = multimeter
        self.voltage = _np.array([])
        self.end_measurement = False

    def clear(self):
        """Clear voltage values."""
        self.voltage = _np.array([])
        self.end_measurement = False

    def run(self):
        """Read voltage from the device."""
        try:
            self.clear()
            oformtype = self.multimeter.get_output_format()
            mformtype = self.multimeter.get_memory_format()
            while self.end_measurement is False:
                if self.multimeter.status():
                    voltage = self.multimeter.get_readings(oformtype)
                    self.voltage = _np.append(self.voltage, voltage)
            else:
                voltage = self.multimeter.get_readings_from_memory(mformtype)
                self.voltage = _np.append(self.voltage, voltage)
            self.finished.emit(True)

        except Exception:
            self.finished.emit(True)
