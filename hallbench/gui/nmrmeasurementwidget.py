# -*- coding: utf-8 -*-

"""Measurement widget for the Hall Bench Control application."""

import epics as _epics
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
    QObject as _QObject,
    Signal as _Signal,
    )
import qtpy.uic as _uic

from hallbench.gui import utils as _utils
from hallbench.gui.auxiliarywidgets import CurrentPositionWidget \
    as _CurrentPositionWidget
import hallbench.data as _data
from hallbench.devices import (
    pmac as _pmac,
    nmr as _nmr,
    multich as _multich,
    dcct as _dcct,
    ps as _ps
    )


class NMRMeasurementWidget(_QWidget):
    """NMr measurement widget class for the Hall Bench Control application."""

    change_current_setpoint = _Signal([bool])
    turn_off_power_supply_current = _Signal([bool])
    turn_on_current_display = _Signal([bool])
    turn_off_current_display = _Signal([bool])

    _measurement_axes = [1, 2, 3, 5, 8, 9]
    _sleep_interval = 0.01

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

        self.temp_measurement_config = (
            _data.configuration.NMRMeasurementConfig())

        self.measurement_config = None
        self.measurement_configured = False
        self.position_list = []
        self.measurements_id_list = []
        self.field_scan_id_list = []
        self.field_scan = None
        self.field = []
        self.field_std = []
        self.graphnmr = []
        self.stop = False

        # Connect signals and slots
        self.connect_signal_slots()

        # Update probe names and configuration ID combo box
        self.update_configuration_ids()

    @property
    def database_name(self):
        """Database name."""
        return _QApplication.instance().database_name

    @property
    def mongo(self):
        """MongoDB database."""
        return _QApplication.instance().mongo

    @property
    def server(self):
        """Server for MongoDB database."""
        return _QApplication.instance().server

    @property
    def directory(self):
        """Return the default directory."""
        return _QApplication.instance().directory

    @property
    def positions(self):
        """Positions dict."""
        return _QApplication.instance().positions

    @property
    def save_fieldmap_dialog(self):
        """Save fieldmap dialog."""
        return _QApplication.instance().save_fieldmap_dialog

    @property
    def view_scan_dialog(self):
        """View scan dialog."""
        return _QApplication.instance().view_scan_dialog

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.current_position_widget.close()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def clear(self):
        """Clear."""
        self.measurement_configured = False
        self.measurement_config = None
        self.field_scan = None
        self.position_list = []
        self.field = []
        self.field_std = []
        self.field_scan_id_list = []
        self.measurements_id_list = []
        self.stop = False
        self.clear_graph()
        self.ui.tbt_view_field_scan.setEnabled(False)
        self.ui.tbt_clear_graph.setEnabled(False)
        self.ui.tbt_create_fieldmap.setEnabled(False)
        self.ui.tbt_save_scan_files.setEnabled(False)

    def clear_button_clicked(self):
        """Clear current measurement and plots."""
        self.clear_measurement()
        self.clear_graph()
        self.ui.tbt_view_field_scan.setEnabled(False)
        self.ui.tbt_clear_graph.setEnabled(False)
        self.ui.tbt_create_fieldmap.setEnabled(False)
        self.ui.tbt_save_scan_files.setEnabled(False)

    def clear_measurement(self):
        """Clear current measurement data."""
        self.field_scan = None
        self.field = []
        self.field_std = []
        self.position_list = []
        self.field_scan_id_list = []

    def clear_graph(self):
        """Clear plots."""
        self.ui.pw_graph.plotItem.curves.clear()
        self.ui.pw_graph.clear()
        self.graphnmr = []

    def clear_load_options(self):
        """Clear load options."""
        self.ui.cmb_idn.setCurrentIndex(-1)

    def configure_graph(self, nr_curves, label, field_component):
        """Configure graph."""
        self.graphnmr = []

        for idx in range(nr_curves):
            if field_component == 'X':
                self.graphnmr.append(
                    self.ui.pw_graph.plotItem.plot(
                        _np.array([]),
                        _np.array([]),
                        pen=(255, 0, 0),
                        symbol='o',
                        symbolPen=(255, 0, 0),
                        symbolSize=4,
                        symbolBrush=(255, 0, 0)))
            
            elif field_component == 'Y':
                self.graphnmr.append(
                    self.ui.pw_graph.plotItem.plot(
                        _np.array([]),
                        _np.array([]),
                        pen=(0, 255, 0),
                        symbol='o',
                        symbolPen=(0, 255, 0),
                        symbolSize=4,
                        symbolBrush=(0, 255, 0)))
            
            else:
                self.graphnmr.append(
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

    def configure_measurement(self):
        """Configure measurement."""
        self.clear()

        if (not self.update_configuration()
                or not self.configure_nmr()
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
                if self.stop:
                    return False
                else:
                    return True   

            _multich.configure(channel_list=channels)
            if self.stop:
                return False
            else:
                return True   

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to configure multichannel.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def configure_nmr(self):
        """Configure multimeters."""
        if self.measurement_config is None:
            return False

        if not _nmr.connected:
            msg = 'NMR not connected.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        try:
            if not self.stop:
                _nmr.configure(
                    self.measurement_config.nmr_mode,
                    self.measurement_config.nmr_frequency,
                    self.measurement_config.nmr_sense,
                    1, 0, 1,
                    self.measurement_config.nmr_channel, 1)
            
            _time.sleep(1)
            
            if self.stop:
                return False
            else:
                return True   

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to configure NMR.'
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
            _pmac.set_velocity(
                1, self.measurement_config.vel_ax1)
            _pmac.set_velocity(
                2, self.measurement_config.vel_ax2)
            _pmac.set_velocity(
                3, self.measurement_config.vel_ax3)
            _pmac.set_velocity(
                5, self.measurement_config.vel_ax5)
            
            if self.stop:
                return False
            else:
                return True   
            
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to configure pmac.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def connect_signal_slots(self):
        """Create signal/slot connections."""
        self.ui.rbt_ax1.clicked.connect(self.clear_load_options)
        self.ui.rbt_ax2.clicked.connect(self.clear_load_options)
        self.ui.rbt_ax3.clicked.connect(self.clear_load_options)
        self.ui.rbt_ax5.clicked.connect(self.clear_load_options)
        self.ui.rbt_ax8.clicked.connect(self.clear_load_options)
        self.ui.rbt_ax9.clicked.connect(self.clear_load_options)
        self.ui.le_magnet_name.editingFinished.connect(self.clear_load_options)
        self.ui.le_current_setpoint.editingFinished.connect(
            self.clear_load_options)
        self.ui.le_operator.editingFinished.connect(self.clear_load_options)
        self.ui.te_comments.textChanged.connect(self.clear_load_options)
        self.ui.sb_nr_measurements.valueChanged.connect(
            self.clear_load_options)
        self.ui.cmb_nmr_channel.currentIndexChanged.connect(
            self.clear_load_options)
        self.ui.cmb_nmr_sense.currentIndexChanged.connect(
            self.clear_load_options)
        self.ui.cmb_nmr_mode.currentIndexChanged.connect(
            self.clear_load_options)
        self.ui.cmb_nmr_read_value.currentIndexChanged.connect(
            self.clear_load_options)
        self.ui.cmb_field_component.currentIndexChanged.connect(
            self.clear_load_options)
        self.sb_nmr_frequency.valueChanged.connect(self.clear_load_options)
        self.sb_max_time.valueChanged.connect(self.clear_load_options)
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
        self.ui.le_start_ax8.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_start_ax8))
        self.ui.le_start_ax9.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_start_ax9))

        self.ui.le_end_ax1.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_end_ax1))
        self.ui.le_end_ax2.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_end_ax2))
        self.ui.le_end_ax3.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_end_ax3))
        self.ui.le_end_ax5.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_end_ax5))
        self.ui.le_end_ax8.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_end_ax8))
        self.ui.le_end_ax9.editingFinished.connect(
            lambda: self.set_float_line_edit_text(self.ui.le_end_ax9))

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
        self.ui.le_step_ax8.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_step_ax8, positive=True, nonzero=True))
        self.ui.le_step_ax9.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_step_ax9, positive=True, nonzero=True))

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
        self.ui.le_vel_ax8.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_vel_ax8, positive=True, nonzero=True))
        self.ui.le_vel_ax9.editingFinished.connect(
            lambda: self.set_float_line_edit_text(
                self.ui.le_vel_ax9, positive=True, nonzero=True))

        self.ui.rbt_ax1.toggled.connect(self.disable_invalid_line_edit)
        self.ui.rbt_ax2.toggled.connect(self.disable_invalid_line_edit)
        self.ui.rbt_ax3.toggled.connect(self.disable_invalid_line_edit)
        self.ui.rbt_ax5.toggled.connect(self.disable_invalid_line_edit)
        self.ui.rbt_ax8.toggled.connect(self.disable_invalid_line_edit)
        self.ui.rbt_ax8.toggled.connect(self.disable_invalid_line_edit)

        self.ui.le_start_ax1.editingFinished.connect(
            lambda: self.fix_end_position_value(1))
        self.ui.le_start_ax2.editingFinished.connect(
            lambda: self.fix_end_position_value(2))
        self.ui.le_start_ax3.editingFinished.connect(
            lambda: self.fix_end_position_value(3))
        self.ui.le_start_ax5.editingFinished.connect(
            lambda: self.fix_end_position_value(5))
        self.ui.le_start_ax8.editingFinished.connect(
            lambda: self.fix_end_position_value(8))
        self.ui.le_start_ax9.editingFinished.connect(
            lambda: self.fix_end_position_value(9))

        self.ui.le_step_ax1.editingFinished.connect(
            lambda: self.fix_end_position_value(1))
        self.ui.le_step_ax2.editingFinished.connect(
            lambda: self.fix_end_position_value(2))
        self.ui.le_step_ax3.editingFinished.connect(
            lambda: self.fix_end_position_value(3))
        self.ui.le_step_ax5.editingFinished.connect(
            lambda: self.fix_end_position_value(5))
        self.ui.le_step_ax8.editingFinished.connect(
            lambda: self.fix_end_position_value(8))
        self.ui.le_step_ax9.editingFinished.connect(
            lambda: self.fix_end_position_value(9))
        
        self.ui.le_end_ax1.editingFinished.connect(
            lambda: self.fix_end_position_value(1))
        self.ui.le_end_ax2.editingFinished.connect(
            lambda: self.fix_end_position_value(2))
        self.ui.le_end_ax3.editingFinished.connect(
            lambda: self.fix_end_position_value(3))
        self.ui.le_end_ax5.editingFinished.connect(
            lambda: self.fix_end_position_value(5))
        self.ui.le_end_ax8.editingFinished.connect(
            lambda: self.fix_end_position_value(8))
        self.ui.le_end_ax9.editingFinished.connect(
            lambda: self.fix_end_position_value(9))

        self.ui.pbt_load_db.clicked.connect(self.load_config_db)
        self.ui.tbt_save_db.clicked.connect(self.save_temp_configuration)

        self.ui.pbt_measure.clicked.connect(self.measure_button_clicked)
        self.ui.pbt_stop.clicked.connect(self.stop_measurement)
        self.ui.tbt_create_fieldmap.clicked.connect(self.show_fieldmap_dialog)
        self.ui.tbt_save_scan_files.clicked.connect(self.save_field_scan_files)
        self.ui.tbt_view_field_scan.clicked.connect(
            self.show_view_field_scan_dialog)
        self.ui.tbt_clear_graph.clicked.connect(self.clear_button_clicked)

    def copy_current_start_position(self):
        """Copy current start position to line edits."""
        try:
            for axis in self._measurement_axes:
                le_start = getattr(self.ui, 'le_start_ax' + str(axis))
                if axis in self.positions.keys():
                    le_start.setText('{0:0.4f}'.format(self.positions[axis]))
                else:
                    le_start.setText('')
                
                vel = _pmac.get_velocity(axis)
                le_vel = getattr(self.ui, 'le_vel_ax' + str(axis))
                le_vel.setText('{0:0.4f}'.format(vel))
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def disable_invalid_line_edit(self):
        """Disable invalid line edit."""
        for axis in self._measurement_axes:
            rbt = getattr(self.ui, 'rbt_ax' + str(axis))
            le_step = getattr(self.ui, 'le_step_ax' + str(axis))
            le_end = getattr(self.ui, 'le_end_ax' + str(axis))
            if rbt.isChecked():
                le_step.setEnabled(True)
                le_end.setEnabled(True)
            else:
                le_step.setEnabled(False)
                le_step.setText('')
                le_end.setEnabled(False)
                le_end.setText('')

    def enable_load_db(self):
        """Enable button to load configuration from database."""
        if self.ui.cmb_idn.currentIndex() != -1:
            self.ui.pbt_load_db.setEnabled(True)
        else:
            self.ui.pbt_load_db.setEnabled(False)

    def end_automatic_measurements(self, setpoint_changed):
        """End automatic measurements."""
        self.ui.pbt_stop.setEnabled(False)
        self.ui.pbt_measure.setEnabled(True)
        self.ui.tbt_create_fieldmap.setEnabled(True)
        self.ui.tbt_save_scan_files.setEnabled(False)
        self.ui.tbt_view_field_scan.setEnabled(False)
        self.ui.tbt_clear_graph.setEnabled(True)       

        if not setpoint_changed:
            msg = ('Automatic measurements failed. ' +
                   'Current setpoint not changed.')
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return
        
        msg = 'End of automatic measurements.'
        _QMessageBox.information(
            self, 'Measurements', msg, _QMessageBox.Ok)
        
        self.stop = True

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
        axis = self.measurement_config.axis
        fixed_axes = [a for a in self._measurement_axes]
        fixed_axes.remove(axis)
        return fixed_axes

    def load_config(self):
        """Set measurement parameters."""
        try:
            self.ui.le_magnet_name.setText(
                self.temp_measurement_config.magnet_name)
            
            self.ui.le_operator.setText(self.temp_measurement_config.operator)

            current_sp = self.temp_measurement_config.current_setpoint
            if current_sp is None:
                self.ui.le_current_setpoint.setText('')
            else:
                self.ui.le_current_setpoint.setText(str(current_sp))

            idx = self.ui.cmb_nmr_channel.findText(
                self.temp_measurement_config.nmr_channel)
            self.ui.cmb_nmr_channel.setCurrentIndex(idx)

            idx = self.ui.cmb_nmr_sense.findText(
                self.temp_measurement_config.nmr_sense)
            self.ui.cmb_nmr_sense.setCurrentIndex(idx)

            idx = self.ui.cmb_nmr_mode.findText(
                self.temp_measurement_config.nmr_mode)
            self.ui.cmb_nmr_mode.setCurrentIndex(idx)

            idx = self.ui.cmb_nmr_read_value.findText(
                self.temp_measurement_config.nmr_read_value)
            self.ui.cmb_nmr_read_value.setCurrentIndex(idx)

            idx = self.ui.cmb_field_component.findText(
                self.temp_measurement_config.field_component)
            self.ui.cmb_field_component.setCurrentIndex(idx)

            self.ui.sb_nmr_frequency.setValue(
                self.temp_measurement_config.nmr_frequency)

            self.ui.sb_reading_time.setValue(
                self.temp_measurement_config.reading_time)

            self.ui.sb_max_time.setValue(
                self.temp_measurement_config.max_time)

            self.ui.te_comments.setText(self.temp_measurement_config.comments)

            self.ui.sb_nr_measurements.setValue(
                self.temp_measurement_config.nr_measurements)

            axis = self.temp_measurement_config.axis
            rbt = getattr(self.ui, 'rbt_ax' + str(axis))
            rbt.setChecked(True)

            self.ui.chb_save_current.setChecked(
                self.temp_measurement_config.save_current)
            self.ui.chb_save_temperature.setChecked(
                self.temp_measurement_config.save_temperature)
            self.ui.chb_automatic_ramp.setChecked(
                self.temp_measurement_config.automatic_ramp)

            for axis in self._measurement_axes:
                le_start = getattr(self.ui, 'le_start_ax' + str(axis))
                value = self.temp_measurement_config.get_start(axis)
                le_start.setText('{0:0.4f}'.format(value))

                le_step = getattr(self.ui, 'le_step_ax' + str(axis))
                value = self.temp_measurement_config.get_step(axis)
                le_step.setText('{0:0.4f}'.format(value))

                le_end = getattr(self.ui, 'le_end_ax' + str(axis))
                value = self.temp_measurement_config.get_end(axis)
                le_end.setText('{0:0.4f}'.format(value))

                le_vel = getattr(self.ui, 'le_vel_ax' + str(axis))
                value = self.temp_measurement_config.get_velocity(axis)
                le_vel.setText('{0:0.4f}'.format(value))

            self.disable_invalid_line_edit()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to load configuration.'
            _QMessageBox.critical(
                self, 'Failure', msg, _QMessageBox.Ok)

    def load_config_db(self):
        """Load configuration from database to set measurement parameters."""
        try:
            idn_text = self.ui.cmb_idn.currentText()
            if len(idn_text) == 0:
                _QMessageBox.critical(
                    self, 'Failure', 'Invalid database ID.', _QMessageBox.Ok)
                return
            
            idn = int(idn_text)
        
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
            self.temp_measurement_config.clear()
            self.temp_measurement_config.db_update_database(
                self.database_name, mongo=self.mongo, server=self.server)
            self.temp_measurement_config.db_read(idn)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to read configuration from database.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        self.load_config()
        self.ui.cmb_idn.setCurrentIndex(self.ui.cmb_idn.findText(str(idn)))
        self.ui.pbt_load_db.setEnabled(False)

    def measure(self):
        """Perform one measurement."""
        if not self.measurement_configured or self.stop is True:
            return False

        try:
            self.turn_off_current_display.emit(True)

            nr_measurements = self.measurement_config.nr_measurements
            fixed_axes = self.get_fixed_axes()

            for nmeas in range(nr_measurements):
                self.la_nr_measurements.setText('{0:d}'.format(nmeas+1))

                for ax in fixed_axes:
                    if self.stop is True:
                        return False
                    pos = self.measurement_config.get_start(ax)
                    self.move_axis(ax, pos)

                if self.stop is True:
                    return False
                
                if not self.measure_field_scan(self.measurement_config.axis):
                    return False

                if self.stop is True:
                    return False

                # move to initial position
                start = self.measurement_config.get_start(
                    self.measurement_config.axis)
                self.move_axis(self.measurement_config.axis, start)

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

            self.turn_on_current_display.emit(True)
            self.ui.la_nr_measurements.setText('')

            if self.stop is True:
                return False

            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            if self.measurement_config.automatic_ramp:
                self.turn_off_power_supply_current.emit(True)
            self.turn_on_current_display.emit(True)
            self.ui.la_nr_measurements.setText('')
            msg = 'Measurement failure.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def measure_and_emit_signal(self):
        """Measure and emit signal to change current setpoint."""
        self.clear_measurement()

        try:
            self.measurement_config.current_setpoint = (
                self.temp_measurement_config.current_setpoint)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to update current setpoint.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        if not self.save_configuration():
            return

        if not self.measure():
            return

        self.change_current_setpoint.emit(1)

    def measure_button_clicked(self):
        """Start measurements."""
        try:
            if self.ui.chb_automatic_ramp.isChecked():
                self.start_automatic_measurements()
            else:
                self.start_measurement()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def measure_current_and_temperature(self):
        """Measure current and temperatures."""
        if (not self.ui.chb_save_temperature.isChecked() and
                not self.ui.chb_save_current.isChecked()):
            return True

        try:
            temperature_dict = {}
            ts = _time.time()

            # Read power supply current
            if self.ui.chb_save_current.isChecked():
                ps_current = float(_ps.read_iload1())
                self.field_scan.ps_current_avg = ps_current

            # Read dcct current
            dcct_current = _dcct.read_current()
            self.field_scan.dcct_current_avg = dcct_current

            # Read multichannel
            if self.ui.chb_save_temperature.isChecked():
                r = _multich.get_converted_readings()
                channels = _multich.config_channels
                for i, ch in enumerate(channels):
                    temperature_dict[ch] = [[ts, r[i]]]
                _QApplication.processEvents()

                self.field_scan.temperature = temperature_dict

            _QApplication.processEvents()
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return False

    def measure_field_scan(self, axis):
        """Start line scan."""
        self.field_scan = _data.measurement.FieldScan()

        start = self.measurement_config.get_start(axis)
        end = self.measurement_config.get_end(axis)
        step = self.measurement_config.get_step(axis)
        vel = self.measurement_config.get_velocity(axis)

        if start == end:
            msg = 'Start and end positions are equal.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        npts = _np.ceil(round((end - start) / step, 4) + 1)
        self.position_list = _np.linspace(start, end, npts)

        self.field = []
        self.field_std = []
        self.clear_graph()
        self.configure_graph(
            1, 'Field [T]', self.measurement_config.field_component)
        _QApplication.processEvents()

        if self.stop is True:
            return False
        
        for ax in self.field_scan.axis_list:
            if ax == axis:
                setattr(self.field_scan, 'pos' + str(ax),
                        self.position_list)
            else:
                pos = _pmac.get_position(ax)
                setattr(self.field_scan, 'pos' + str(ax), pos)

        _QApplication.processEvents()

        self.field_scan.dcct_current_avg = None
        self.field_scan.ps_current_avg = None
        self.field_scan.temperature = {}

        idx = 0
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            self.graphnmr[idx].setData([], [])

        # go to initial position
        self.move_axis(axis, start)
        
        _time.sleep(1)
        _QApplication.processEvents()
        
        if self.stop is True:
            return False

        self.measure_current_and_temperature()
        _QApplication.processEvents()
        
        _time.sleep(1)
        _QApplication.processEvents()

        if self.stop is True:
            return False

        for pos in self.position_list:
            if not self.move_axis(axis, pos):
                return False
            
            value_mean, value_std = self.measure_point()
            self.field.append(value_mean)
            self.field_std.append(value_std)
            self.plot_field(idx)
            _QApplication.processEvents()

        if self.stop is True:
            return False
        
        zero_vec = _np.zeros(len(self.position_list))
        if self.measurement_config.field_component == 'X':
            self.field_scan.bx = self.field
            self.field_scan.by = zero_vec
            self.field_scan.bz = zero_vec
            self.field_scan.std_bx = self.field_std
            self.field_scan.std_by = zero_vec
            self.field_scan.std_bz = zero_vec
        elif self.measurement_config.field_component == 'Y':
            self.field_scan.bx = zero_vec
            self.field_scan.by = self.field
            self.field_scan.bz = zero_vec
            self.field_scan.std_bx = zero_vec
            self.field_scan.std_by = self.field_std
            self.field_scan.std_bz = zero_vec            
        else:
            self.field_scan.bx = zero_vec
            self.field_scan.by = zero_vec
            self.field_scan.bz = self.field
            self.field_scan.std_bx = zero_vec
            self.field_scan.std_by = zero_vec
            self.field_scan.std_bz = self.field_std
    
        if self.field_scan.npts == 0:
            _warnings.warn(
                'Invalid number of points in field scan.')
            return False

        _QApplication.processEvents()

        success = self.save_field_scan()

        return success

    def move_axis(self, axis, position):
        """Move bench axis."""
        if self.stop is False:
            _pmac.set_position(axis, position)
            status = _pmac.axis_status(axis)
            while (status is None or (status & 1) == 0) and self.stop is False:
                status = _pmac.axis_status(axis)
                _QApplication.processEvents()
            return True
        else:
            return False

    def measure_point(self):
        """Measure data.""" 
        if self.stop is False:           
            values = []
                       
            reading_time = self.measurement_config.reading_time
            max_time = self.measurement_config.max_time
            
            if self.measurement_config.nmr_read_value == 'Locked':
                init_char = 'L'
            elif self.measurement_config.nmr_read_value == 'Signal':
                init_char = 'S'
            else:
                init_char = None

            t0 = _time.time()
            tr0 = None

            t = t0
            while (t - t0) <= max_time:
                if self.stop:
                    break
                
                b = _nmr.read_b_value().strip().replace('\r\n', '')
                if b.endswith('T'):
                    b = b.replace('T', '')
                    if init_char is None or b.startswith(init_char):
                        values.append(float(b[1:]))
                        if tr0 is None:
                            tr0 = _time.time()

                if tr0 is not None and (t - tr0) > reading_time:
                    break
        
                _time.sleep(self._sleep_interval)
                _QApplication.processEvents()
                t = _time.time()
        
            if len(values) == 0:
                value_mean = _np.nan
                value_std = 0
            else:
                value_mean = _np.mean(values)
                value_std = _np.std(values)
        
            return value_mean, value_std
        
        else:
            return _np.nan, 0
        
    def plot_field(self, idx):
        """Plot values."""
        npts = len(self.position_list)
        field = [b for b in self.field[:npts]]

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            self.graphnmr[idx].setData(
                self.position_list[:len(field)], field)

    def save_temp_configuration(self):
        """Save configuration to database."""
        self.ui.cmb_idn.setCurrentIndex(-1)
        if self.database_name is not None:
            try:
                if self.update_configuration():
                    self.temp_measurement_config.db_update_database(
                        self.database_name, mongo=self.mongo,
                        server=self.server)
                    idn = self.temp_measurement_config.db_save()
                    self.ui.cmb_idn.addItem(str(idn))
                    self.ui.cmb_idn.setCurrentIndex(self.ui.cmb_idn.count()-1)
                    self.ui.pbt_load_db.setEnabled(False)
            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                msg = 'Failed to save configuration to database.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
        else:
            msg = 'Invalid database name.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def save_configuration(self):
        """Save configuration to database table."""
        try:
            selected_config = _data.configuration.NMRMeasurementConfig(
                self.database_name, mongo=self.mongo,
                server=self.server)
            
            text = self.ui.cmb_idn.currentText()            
            if len(text) != 0:
                selected_idn = int(text)
                selected_config.db_read(selected_idn)

            if self.measurement_config == selected_config:
                self.measurement_config.idn = selected_idn
            else:
                self.measurement_config.db_update_database(
                    self.database_name,
                    mongo=self.mongo, server=self.server)
                idn = self.measurement_config.db_save()
                self.update_configuration_ids()
                idx = self.ui.cmb_idn.findText(str(idn))
                self.ui.cmb_idn.setCurrentIndex(idx)
                self.ui.pbt_load_db.setEnabled(False)
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
            config = self.measurement_config
            self.field_scan.magnet_name = config.magnet_name
            self.field_scan.current_setpoint = config.current_setpoint
            self.field_scan.comments = config.comments
            self.field_scan.configuration_id = config.idn

            self.field_scan.db_update_database(
                self.database_name, mongo=self.mongo, server=self.server)
            idn = self.field_scan.db_save()
            self.field_scan_id_list.append(idn)
            return True

        except Exception:
            try:
                self.field_scan.db_update_database(
                    self.database_name, mongo=self.mongo, server=self.server)
                idn = self.field_scan.db_save()
                self.field_scan_id_list.append(idn)
                return True

            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                msg = 'Failed to save FieldScan to database'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
                return False

    def save_field_scan_files(self):
        """Save scan files."""
        try:
            field_scan_list = []
            for idn in self.field_scan_id_list:
                fs = _data.measurement.FieldScan(
                    database_name=self.database_name,
                    mongo=self.mongo, server=self.server)
                fs.db_read(idn)
                field_scan_list.append(fs)
    
            directory = _QFileDialog.getExistingDirectory(
                self, caption='Save scan files', directory=self.directory)
    
            if isinstance(directory, tuple):
                directory = directory[0]
    
            if len(directory) == 0:
                return


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
        try:
            if len(self.measurements_id_list) > 1:
                field_scan_id_list = self.measurements_id_list
            else:
                field_scan_id_list = self.field_scan_id_list
    
            self.save_fieldmap_dialog.show(field_scan_id_list)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def show_view_field_scan_dialog(self):
        """Open view data dialog."""
        try:
            field_scan_list = []
            fs = _data.measurement.FieldScan(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
            for idn in self.field_scan_id_list:
                fs.db_read(idn)
                field_scan_list.append(fs.copy())

            self.view_scan_dialog.show(
                field_scan_list, 'field')
        
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

        self.ui.pbt_stop.setEnabled(False)
        self.ui.pbt_measure.setEnabled(True)
        self.ui.tbt_save_scan_files.setEnabled(True)
        self.ui.tbt_view_field_scan.setEnabled(True)
        self.ui.tbt_clear_graph.setEnabled(True)
        self.ui.tbt_create_fieldmap.setEnabled(True)

        msg = 'End of measurement.'
        _QMessageBox.information(
            self, 'Measurement', msg, _QMessageBox.Ok)

    def stop_measurement(self):
        """Stop measurement to True."""
        try:
            self.stop = True
            _pmac.stop_all_axis()
            self.ui.pbt_measure.setEnabled(True)
            self.ui.pbt_stop.setEnabled(False)
            self.ui.tbt_clear_graph.setEnabled(True)
            self.ui.la_nr_measurements.setText('')
            self.turn_on_current_display.emit(True)
            msg = 'The user stopped the measurements.'
            _QMessageBox.information(
                self, 'Abort', msg, _QMessageBox.Ok)
        except Exception:
            self.stop = True
            _pmac.stop_all_axis()
            self.turn_on_current_display.emit(True)
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to stop measurements.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def update_configuration(self):
        """Update measurement configuration parameters."""
        try:
            self.temp_measurement_config.clear()

            _s = self.ui.le_magnet_name.text().strip()
            self.temp_measurement_config.magnet_name = _s

            current_setpoint = _utils.get_value_from_string(
                self.ui.le_current_setpoint.text())
            self.temp_measurement_config.current_setpoint = current_setpoint

            _s = self.ui.le_operator.text().strip()
            self.temp_measurement_config.operator = _s

            _s = self.ui.te_comments.toPlainText().strip()
            _s += '(NMR measurements)'
            self.temp_measurement_config.comments = _s

            self.temp_measurement_config.nmr_channel = (
                self.ui.cmb_nmr_channel.currentText())

            self.temp_measurement_config.nmr_sense = (
                self.ui.cmb_nmr_sense.currentText())

            self.temp_measurement_config.nmr_mode = (
                self.ui.cmb_nmr_mode.currentText())

            self.temp_measurement_config.nmr_read_value = (
                self.ui.cmb_nmr_read_value.currentText())

            self.temp_measurement_config.field_component = (
                self.ui.cmb_field_component.currentText())

            self.temp_measurement_config.nmr_frequency = (
                self.ui.sb_nmr_frequency.value())

            self.temp_measurement_config.reading_time = (
                self.ui.sb_reading_time.value())

            self.temp_measurement_config.max_time = (
                self.ui.sb_max_time.value())
                        
            nr_meas = self.ui.sb_nr_measurements.value()
            self.temp_measurement_config.nr_measurements = nr_meas

            _ch = self.ui.chb_save_current.isChecked()
            self.temp_measurement_config.save_current = int(_ch)

            _ch = self.ui.chb_save_temperature.isChecked()
            self.temp_measurement_config.save_temperature = int(_ch)

            if self.ui.chb_automatic_ramp.isChecked():
                self.temp_measurement_config.automatic_ramp = 1
                self.ui.sb_nr_measurements.setValue(1)
                self.temp_measurement_config.nr_measurements = 1
            else:
                self.temp_measurement_config.automatic_ramp = 0

            for axis in self._measurement_axes:
                rbt = getattr(self.ui, 'rbt_ax' + str(axis))
                if rbt.isChecked():
                    self.temp_measurement_config.axis = axis

                start = self.get_axis_param('start', axis)
                self.temp_measurement_config.set_start(axis, start)

                le_step = getattr(self.ui, 'le_step_ax' + str(axis))
                if le_step.isEnabled():
                    step = self.get_axis_param('step', axis)
                    self.temp_measurement_config.set_step(axis, step)
                else:
                    self.temp_measurement_config.set_step(axis, 0.0)

                le_end = getattr(self.ui, 'le_end_ax' + str(axis))
                if le_end.isEnabled():
                    end = self.get_axis_param('end', axis)
                    self.temp_measurement_config.set_end(axis, end)
                else:
                    self.temp_measurement_config.set_end(axis, start)

                vel = self.get_axis_param('vel', axis)
                self.temp_measurement_config.set_velocity(axis, vel)

            if self.temp_measurement_config.valid_data():
                self.measurement_config = self.temp_measurement_config.copy()
                
                if self.stop:
                    return False
                else:
                    return True   

            else:
                self.measurement_config = None
                msg = 'Invalid measurement configuration.'
                _QMessageBox.critical(
                    self, 'Failure', msg, _QMessageBox.Ok)
                return False

        except Exception:
            self.measurement_config = None
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
            self.temp_measurement_config.db_update_database(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
            idns = self.temp_measurement_config.db_get_id_list()
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
            self.temp_measurement_config.current_setpoint = current_setpoint
            self.ui.le_current_setpoint.setText(current_value_str)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to update current setpoint.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

