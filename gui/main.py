# -*- coding: utf-8 -*-
"""Hall Bench User Interface."""

# Libraries
import os
import sys
import time
import threading
import numpy as np
from PyQt4 import QtCore
from PyQt4 import QtGui

from interface import Ui_HallBench
from hall_bench.data_handle import calibration
from hall_bench.data_handle import configuration
from hall_bench.data_handle import measurement
from hall_bench.devices_communication import Pmac
from hall_bench.devices_communication import DigitalMultimeter
from hall_bench.devices_communication import Multichannel
import magnets_info


class HallBenchGUI(QtGui.QWidget):
    """Hall Bench Graphical User Interface."""

    def __init__(self, parent=None):
        """Initialize the graphical interface."""
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_HallBench()
        self.ui.setupUi(self)
        self._initialize_variables()
        self._set_interface_initial_state()
        self._connect_signals()
        self._start_timer()

    def _initialize_variables(self):
        self.led_on = ":/images/images/led_green.png"
        self.led_off = ":/images/images/led_red.png"

        self.cconfig = None
        self.mconfig = None
        self.calibration_data = None
        self.devices = None

        self.nr_measurements = 2
        self.dirpath = None
        self.stop = False

        self.current_voltage_data = None
        self.current_postion_list = []
        self.current_voltage_list = []
        self.current_measurement = None

        self.graph_curve_x = np.array([])
        self.graph_curve_y = np.array([])
        self.graph_curve_z = np.array([])

        self.ti = None
        self.tj = None
        self.tk = None

        self.measurement_to_save = None
        self.calibration_data_to_save = None
        self.directory_to_save = None
        self.voltage_data_files = []

    def _set_interface_initial_state(self):
        self.ui.tab_main.setTabEnabled(1, False)
        self.ui.tab_main.setTabEnabled(2, False)
        self.ui.tab_main.setTabEnabled(3, False)

        for idx in range(0, self.ui.tb_move_axis.count()):
            self.ui.tb_move_axis.setItemEnabled(idx, False)

        self.ui.sb_nr_measurements.setValue(self.nr_measurements)
        self.ui.ta_additional_parameter.horizontalHeader().setVisible(True)
        self.main_tab_changed()

    def _connect_signals(self):
        self.ui.tab_main.currentChanged.connect(self.main_tab_changed)
        self._connect_signals_connection_tab()
        self._connect_signals_motors_tab()
        self._connect_signals_calibration_tab()
        self._connect_signals_measurement_tab()
        self._connect_signals_save_tab()

    def main_tab_changed(self):
        """Update interface when the main tab index change."""
        # update motors tab_main
        if self.devices is not None and self.devices.pmac_connected:
            self.ui.fm_homming.setEnabled(True)
            self.ui.fm_limits.setEnabled(True)
            self.ui.la_move_axis.setEnabled(True)
            self._release_access_to_movement()
        else:
            self.ui.fm_homming.setEnabled(False)
            self.ui.fm_limits.setEnabled(False)
            self.ui.la_move_axis.setEnabled(False)

        # update save tab
        if self.current_measurement is None:
            self.ui.rb_recover_measurement.setChecked(True)
            self.ui.rb_current_measurement.setEnabled(False)
        else:
            self.ui.rb_current_measurement.setEnabled(True)
        self.select_measurement_to_save()
        self.disable_invalid_magnet_axes()

    def _connect_signals_connection_tab(self):
        self.ui.pb_load_connection_config.clicked.connect(
            self.load_connection_configuration_file)

        self.ui.pb_save_connection_config.clicked.connect(
            self.save_connection_configuration_file)

        self.ui.pb_connect_devices.clicked.connect(self.connect_devices)
        self.ui.pb_disconnect_devices.clicked.connect(self.disconnect_devices)

    def _connect_signals_motors_tab(self):
        self.ui.pb_activate_bench.clicked.connect(self.activate_bench)
        self.ui.pb_start_homming.clicked.connect(self.start_homming)
        self.ui.pb_stop_all_motors.clicked.connect(self.stop_all_axis)
        self.ui.pb_kill_all_motors.clicked.connect(self.kill_all_axis)

        for axis in [1, 2, 3, 5, 6, 7, 8]:
            pb_move = getattr(self.ui, 'pb_move_to_target_axis' + str(axis))
            pb_move.clicked.connect(self.move_to_target, axis)
            pb_stop = getattr(self.ui, 'pb_stop_motor_axis' + str(axis))
            pb_stop.clicked.connect(self.stop_axis, axis)

        self.ui.tb_move_axis.currentChanged.connect(self.update_axis_speed)

        self.ui.le_setvel1.editingFinished.connect(
            lambda: self._check_value(self.ui.le_setvel1, 0.1, 150))
        self.ui.le_setvel2.editingFinished.connect(
            lambda: self._check_value(self.ui.le_setvel2, 0.1, 10))
        self.ui.le_setvel3.editingFinished.connect(
            lambda: self._check_value(self.ui.le_setvel3, 0.1, 10))
        self.ui.le_setvel5.editingFinished.connect(
            lambda: self._check_value(self.ui.le_setvel5, 0.1, 20))
        self.ui.le_setvel6.editingFinished.connect(
            lambda: self._check_value(self.ui.le_setvel6, 0.1, 5))
        self.ui.le_setvel7.editingFinished.connect(
            lambda: self._check_value(self.ui.le_setvel7, 0.1, 5))
        self.ui.le_setvel8.editingFinished.connect(
            lambda: self._check_value(self.ui.le_setvel8, 0.1, 10))
        self.ui.le_setvel9.editingFinished.connect(
            lambda: self._check_value(self.ui.le_setvel9, 0.1, 10))

        self.ui.le_target1.editingFinished.connect(
            lambda: self._check_value(self.ui.le_target1, -3500, 3500))
        self.ui.le_target2.editingFinished.connect(
            lambda: self._check_value(self.ui.le_target2, -150, 150))
        self.ui.le_target3.editingFinished.connect(
            lambda: self._check_value(self.ui.le_target3, -150, 150))
        self.ui.le_target5.editingFinished.connect(
            lambda: self._check_value(self.ui.le_target5, -180, 180))
        self.ui.le_target6.editingFinished.connect(
            lambda: self._check_value(self.ui.le_target6, -12, 12))
        self.ui.le_target7.editingFinished.connect(
            lambda: self._check_value(self.ui.le_target7, -12, 12))
        self.ui.le_target8.editingFinished.connect(
            lambda: self._check_value(self.ui.le_target8, -10, 10))
        self.ui.le_target9.editingFinished.connect(
            lambda: self._check_value(self.ui.le_target9, -10, 10))

    def _connect_signals_calibration_tab(self):
        self.ui.pb_load_calibration.clicked.connect(self.load_calibration_data)

    def _connect_signals_measurement_tab(self):
        # load and save measurements parameters
        self.ui.pb_load_measurement_config.clicked.connect(
            self.load_measurement_configuration_file)
        self.ui.pb_save_measurement_config.clicked.connect(
            self.save_measurement_configuration_file)

        # check input values for measurement
        self.ui.le_start1.editingFinished.connect(
            lambda: self._check_value(self.ui.le_start1, -3500, 3500))
        self.ui.le_start2.editingFinished.connect(
            lambda: self._check_value(self.ui.le_start2, -150, 150))
        self.ui.le_start3.editingFinished.connect(
            lambda: self._check_value(self.ui.le_start3, -150, 150))
        self.ui.le_start5.editingFinished.connect(
            lambda: self._check_value(self.ui.le_start5, 0, 180))

        self.ui.le_end1.editingFinished.connect(
            lambda: self._check_value(self.ui.le_end1, -3500, 3500))
        self.ui.le_end2.editingFinished.connect(
            lambda: self._check_value(self.ui.le_end2, -150, 150))
        self.ui.le_end3.editingFinished.connect(
            lambda: self._check_value(self.ui.le_end3, -150, 150))
        self.ui.le_end5.editingFinished.connect(
            lambda: self._check_value(self.ui.le_end5, 0, 180))

        self.ui.le_step1.editingFinished.connect(
            lambda: self._check_value(self.ui.le_step1, -10, 10))
        self.ui.le_step2.editingFinished.connect(
            lambda: self._check_value(self.ui.le_step2, -10, 10))
        self.ui.le_step3.editingFinished.connect(
            lambda: self._check_value(self.ui.le_step3, -10, 10))
        self.ui.le_step5.editingFinished.connect(
            lambda: self._check_value(self.ui.le_step5, -10, 10))

        self.ui.le_extra1.editingFinished.connect(
            lambda: self._check_value(self.ui.le_extra1, 0, 100))
        self.ui.le_extra2.editingFinished.connect(
            lambda: self._check_value(self.ui.le_extra2, 0, 100))
        self.ui.le_extra3.editingFinished.connect(
            lambda: self._check_value(self.ui.le_extra3, 0, 100))
        self.ui.le_extra5.editingFinished.connect(
            lambda: self._check_value(self.ui.le_extra5, 0, 100))

        self.ui.le_vel1.editingFinished.connect(
            lambda: self._check_value(self.ui.le_vel1, 0.1, 150))
        self.ui.le_vel2.editingFinished.connect(
            lambda: self._check_value(self.ui.le_vel2, 0.1, 5))
        self.ui.le_vel3.editingFinished.connect(
            lambda: self._check_value(self.ui.le_vel3, 0.1, 5))
        self.ui.le_vel5.editingFinished.connect(
            lambda: self._check_value(self.ui.le_vel5, 0.1, 10))

        # Configure and start measurements
        self.ui.pb_configure_and_measure.clicked.connect(
            self.configure_and_measure)
        self.ui.pb_stop_measurement.clicked.connect(self.stop_measurements)

    def _connect_signals_save_tab(self):
        self.ui.rb_current_measurement.toggled.connect(
            self.select_measurement_to_save)
        self.ui.rb_recover_measurement.toggled.connect(
            self.select_measurement_to_save)

        self.ui.ps_load_voltage_data.clicked.connect(
            self.load_voltage_data_files)

        self.ui.pb_load_recover_calibration_file.clicked.connect(
            self.load_recover_calibration_data)

        self.ui.rb_current_calibration.toggled.connect(
            self.select_calibration_data)
        self.ui.rb_load_calibration.toggled.connect(
            self.select_calibration_data)

        self.ui.pb_recover_data.clicked.connect(
            self.recover_from_voltage_data_files)

        self.ui.cb_magnet_axisx.currentIndexChanged.connect(
            self.disable_invalid_magnet_axes)

        names = magnets_info.get_magnets_name()
        for name in names:
            self.ui.cb_predefined.addItem(name)

        self.ui.cb_predefined.currentIndexChanged.connect(
            self.load_magnet_info)

        self.ui.cb_main.stateChanged.connect(
            lambda: self.enabled_coil(
                self.ui.cb_main, self.ui.fm_main,
                self.ui.le_main_current, self.ui.le_main_turns))

        self.ui.cb_trim.stateChanged.connect(
            lambda: self.enabled_coil(
                self.ui.cb_trim, self.ui.fm_trim,
                self.ui.le_trim_current, self.ui.le_trim_turns))

        self.ui.cb_ch.stateChanged.connect(
            lambda: self.enabled_coil(
                self.ui.cb_ch, self.ui.fm_ch,
                self.ui.le_ch_current, self.ui.le_ch_turns))

        self.ui.cb_cv.stateChanged.connect(
            lambda: self.enabled_coil(
                self.ui.cb_cv, self.ui.fm_cv,
                self.ui.le_cv_current, self.ui.le_cv_turns))

        self.ui.cb_qs.stateChanged.connect(
            lambda: self.enabled_coil(
                self.ui.cb_qs, self.ui.fm_qs,
                self.ui.le_qs_current, self.ui.le_qs_turns))

        self.ui.pb_add_row.clicked.connect(
            lambda: self.ui.ta_additional_parameter.setRowCount(
                self.ui.ta_additional_parameter.rowCount() + 1))

        self.ui.pb_remove_row.clicked.connect(
            lambda: self.ui.ta_additional_parameter.setRowCount(
                self.ui.ta_additional_parameter.rowCount() - 1))

        self.ui.pb_change_directory.clicked.connect(self.change_directory)
        self.ui.pb_save_measurement.clicked.connect(self.save_measurement)

    def _check_value(self, obj, limit_min, limit_max):
        try:
            val = float(obj.text())
            if val >= limit_min and val <= limit_max:
                obj.setText('{0:0.4f}'.format(val))
            else:
                obj.setText('')
        except Exception:
            obj.setText('')

    def _start_timer(self):
        """Start timer for interface updates."""
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._refresh_interface)
        self.timer.start(250)

    def _refresh_interface(self):
        """Read probes positions and update the interface."""
        try:
            if self.devices is not None:
                if self.devices.pmac_connected:
                    for axis in self.devices.pmac.commands.list_of_axis:
                        pos = self.devices.pmac.get_position(axis)
                        le_mo = getattr(self.ui, 'le_mo_pos' + str(axis))
                        le_mo.setText('{0:0.4f}'.format(pos))
                        le_me = getattr(self.ui, 'le_me_pos' + str(axis))
                        le_me.setText('{0:0.4f}'.format(pos))
                    QtGui.QApplication.processEvents()
        except Exception:
            pass

    def load_connection_configuration_file(self):
        """Load configuration file to set connection parameters."""
        filename = QtGui.QFileDialog.getOpenFileName(
            self, 'Open connection configuration file')

        if len(filename) != 0:
            try:
                self.cconfig = configuration.ConnectionConfig(filename)
            except Exception as e:
                QtGui.QMessageBox.critical(
                    self, 'Failure', str(e), QtGui.QMessageBox.Ignore)
                return

            self.ui.le_connection_config_filename.setText(filename)

            self.ui.cb_pmac_enable.setChecked(self.cconfig.control_pmac_enable)

            self.ui.cb_dmm_x_enable.setChecked(
                self.cconfig.control_voltx_enable)
            self.ui.sb_dmm_x_address.setValue(
                self.cconfig.control_voltx_addr)

            self.ui.cb_dmm_y_enable.setChecked(
                self.cconfig.control_volty_enable)
            self.ui.sb_dmm_y_address.setValue(
                self.cconfig.control_volty_addr)

            self.ui.cb_dmm_z_enable.setChecked(
                self.cconfig.control_voltz_enable)
            self.ui.sb_dmm_z_address.setValue(
                self.cconfig.control_voltz_addr)

            self.ui.cb_multich_enable.setChecked(
                self.cconfig.control_multich_enable)

            self.ui.sb_multich_address.setValue(
                self.cconfig.control_multich_addr)

            self.ui.cb_colimator_enable.setChecked(
                self.cconfig.control_colimator_enable)

            self.ui.cb_colimator_port.setCurrentIndex(
                self.cconfig.control_colimator_addr)

    def save_connection_configuration_file(self):
        """Save connection parameters to file."""
        filename = QtGui.QFileDialog.getSaveFileName(
            self, 'Save connection configuration file')

        if len(filename) != 0:
            if self._update_connection_configuration():
                try:
                    self.cconfig.save_file(filename)
                except Exception as e:
                    QtGui.QMessageBox.critical(
                        self, 'Failure', str(e), QtGui.QMessageBox.Ignore)

    def _update_connection_configuration(self):
        if self.cconfig is None:
            self.cconfig = configuration.ConnectionConfig()

        self.cconfig.control_pmac_enable = self.ui.cb_pmac_enable.isChecked()

        self.cconfig.control_voltx_enable = self.ui.cb_dmm_x_enable.isChecked()
        self.cconfig.control_volty_enable = self.ui.cb_dmm_y_enable.isChecked()
        self.cconfig.control_voltz_enable = self.ui.cb_dmm_z_enable.isChecked()

        multich_enable = self.ui.cb_multich_enable.isChecked()
        colimator_enable = self.ui.cb_colimator_enable.isChecked()
        self.cconfig.control_multich_enable = multich_enable
        self.cconfig.control_colimator_enable = colimator_enable

        self.cconfig.control_voltx_addr = self.ui.sb_dmm_x_address.value()
        self.cconfig.control_volty_addr = self.ui.sb_dmm_y_address.value()
        self.cconfig.control_voltz_addr = self.ui.sb_dmm_z_address.value()

        multich_addr = self.ui.sb_multich_address.value()
        colimator_addr = self.ui.cb_colimator_port.currentIndex()
        self.cconfig.control_multich_addr = multich_addr
        self.cconfig.control_colimator_addr = colimator_addr

        if self.cconfig.valid_configuration():
            return True
        else:
            message = 'Invalid connection configuration'
            QtGui.QMessageBox.critical(
                self, 'Failure', message, QtGui.QMessageBox.Ignore)
            return False

    def load_measurement_configuration_file(self):
        """Load configuration data to set measurement parameters."""
        filename = QtGui.QFileDialog.getOpenFileName(
            self, 'Open measurement configuration file')

        if len(filename) != 0:
            try:
                self.mconfig = configuration.MeasurementConfig(filename)
            except Exception as e:
                QtGui.QMessageBox.critical(
                    self, 'Failure', str(e), QtGui.QMessageBox.Ignore)
                return

            self.ui.le_measurement_config_filename.setText(filename)

            self.ui.cb_hall_x_enable.setChecked(self.mconfig.meas_probeX)
            self.ui.cb_hall_y_enable.setChecked(self.mconfig.meas_probeY)
            self.ui.cb_hall_z_enable.setChecked(self.mconfig.meas_probeZ)

            self.ui.le_dmm_aper.setText(str(self.mconfig.meas_aper_ms))
            self.ui.cb_dmm_precision.setCurrentIndex(
                self.mconfig.meas_precision)
            trig_axis = self.mconfig.meas_trig_axis

            if trig_axis == 1:
                self.ui.rb_triggering_axis1.setChecked()
            elif trig_axis == 2:
                self.ui.rb_triggering_axis2.setChecked()
            elif trig_axis == 3:
                self.ui.rb_triggering_axis3.setChecked()

            axis_measurement = [1, 2, 3, 5]
            for axis in axis_measurement:
                tmp = getattr(self.ui, 'le_start' + str(axis))
                value = getattr(self.mconfig, 'meas_startpos_ax' + str(axis))
                tmp.setText(str(value))

                tmp = getattr(self.ui, 'le_end' + str(axis))
                value = getattr(self.mconfig, 'meas_endpos_ax' + str(axis))
                tmp.setText(str(value))

                tmp = getattr(self.ui, 'le_step' + str(axis))
                value = getattr(self.mconfig, 'meas_incr_ax' + str(axis))
                tmp.setText(str(value))

                tmp = getattr(self.ui, 'le_vel' + str(axis))
                value = getattr(self.mconfig, 'meas_vel_ax' + str(axis))
                tmp.setText(str(value))

                tmp = getattr(self.ui, 'le_extra' + str(axis))
                tmp.setText(str(0))

    def save_measurement_configuration_file(self):
        """Save measurement parameters to file."""
        filename = QtGui.QFileDialog.getSaveFileName(
            self, 'Save measurement configuration file')

        if len(filename) != 0:
            if self._update_measurement_configuration():
                try:
                    self.mconfig.save_file(filename)
                except Exception as e:
                    QtGui.QMessageBox.critical(
                        self, 'Failure', str(e), QtGui.QMessageBox.Ignore)

    def _update_measurement_configuration(self):
        if self.mconfig is None:
            self.mconfig = configuration.MeasurementConfig()

        self.mconfig.meas_probeX = self.ui.cb_hall_x_enable.isChecked()
        self.mconfig.meas_probeY = self.ui.cb_hall_y_enable.isChecked()
        self.mconfig.meas_probeZ = self.ui.cb_hall_z_enable.isChecked()

        self.mconfig.meas_precision = self.ui.cb_dmm_precision.currentIndex()

        if self.ui.rb_triggering_axis1.isChecked():
            self.mconfig.meas_trig_axis = 1
        elif self.ui.rb_triggering_axis2.isChecked():
            self.mconfig.meas_trig_axis = 2
        elif self.ui.rb_triggering_axis3.isChecked():
            self.mconfig.meas_trig_axis = 3

        self.nr_measurements = self.ui.sb_nr_measurements.value()

        tmp = self.ui.le_dmm_aper.text()
        if bool(tmp and tmp.strip()):
            self.mconfig.meas_aper_ms = float(tmp)

        axis_measurement = [1, 2, 3, 5]
        for axis in axis_measurement:
            tmp = getattr(self.ui, 'le_start' + str(axis)).text()
            if bool(tmp and tmp.strip()):
                setattr(
                    self.mconfig, 'meas_startpos_ax' + str(axis), float(tmp))

            tmp = getattr(self.ui, 'le_end' + str(axis)).text()
            if bool(tmp and tmp.strip()):
                setattr(self.mconfig, 'meas_endpos_ax' + str(axis), float(tmp))

            tmp = getattr(self.ui, 'le_step' + str(axis)).text()
            if bool(tmp and tmp.strip()):
                setattr(self.mconfig, 'meas_incr_ax' + str(axis), float(tmp))

            tmp = getattr(self.ui, 'le_vel' + str(axis)).text()
            if bool(tmp and tmp.strip()):
                setattr(self.mconfig, 'meas_vel_ax' + str(axis), float(tmp))

        if self.mconfig.valid_configuration():
            return True
        else:
            message = 'Invalid measurement configuration'
            QtGui.QMessageBox.critical(
                self, 'Failure', message, QtGui.QMessageBox.Ignore)
            return False

    def connect_devices(self):
        """Connect bench devices."""
        if not self._update_connection_configuration():
            return

        self.devices = HallBenchDevices(self.cconfig)
        self.devices.load()
        self.devices.connect()
        self._update_led_status()

        if self.cconfig.control_pmac_enable:
            if self.devices.pmac_connected:
                self.ui.tab_main.setTabEnabled(1, True)

        self.activate_bench()

    def disconnect_devices(self):
        """Disconnect bench devices."""
        if self.devices is not None:
            self.devices.disconnect()
        self._update_led_status()

    def _update_led_status(self):
        if self.devices is not None:
            if self.devices.voltx_connected:
                self.ui.la_dmm_x_led.setPixmap(QtGui.QPixmap(self.led_on))
            else:
                self.ui.la_dmm_x_led.setPixmap(QtGui.QPixmap(self.led_off))

            if self.devices.volty_connected:
                self.ui.la_dmm_y_led.setPixmap(QtGui.QPixmap(self.led_on))
            else:
                self.ui.la_dmm_y_led.setPixmap(QtGui.QPixmap(self.led_off))

            if self.devices.voltz_connected:
                self.ui.la_dmm_z_led.setPixmap(QtGui.QPixmap(self.led_on))
            else:
                self.ui.la_dmm_z_led.setPixmap(QtGui.QPixmap(self.led_off))

            if self.devices.pmac_connected:
                self.ui.la_pmac_led.setPixmap(QtGui.QPixmap(self.led_on))
            else:
                self.ui.la_pmac_led.setPixmap(QtGui.QPixmap(self.led_off))

            if self.devices.multich_connected:
                self.ui.la_multich_led.setPixmap(QtGui.QPixmap(self.led_on))
            else:
                self.ui.la_multich_led.setPixmap(QtGui.QPixmap(self.led_off))

    def activate_bench(self):
        """Active bench."""
        if self.devices is None:
            return

        if self.cconfig.control_pmac_enable and self.devices.pmac_connected:
            if self.devices.pmac.activate_bench():
                self.ui.fm_homming.setEnabled(True)
                self.ui.fm_limits.setEnabled(True)
                self.ui.la_move_axis.setEnabled(True)
                self._release_access_to_movement()
            else:
                message = 'Failed to active bench.'
                QtGui.QMessageBox.critical(
                    self, 'Failure', message, QtGui.QMessageBox.Ok)

    def update_axis_speed(self):
        """Update axis velocity values."""
        if self.devices is not None:
            list_of_axis = self.devices.pmac.commands.list_of_axis
            for axis in list_of_axis:
                obj = getattr(self.ui, 'le_setvel' + str(axis))
                vel = self.devices.pmac.get_velocity(axis)
                obj.setText('{0:0.4f}'.format(vel))

    def _release_access_to_movement(self):
        if self.devices is not None and self.devices.pmac_connected:
            status = []
            if (self.devices.pmac.axis_status(1) & 1024) != 0:
                self.ui.tb_move_axis.setItemEnabled(0, 1)
                status.append(True)
            else:
                status.append(False)

            if (self.devices.pmac.axis_status(2) & 1024) != 0:
                self.ui.tb_move_axis.setItemEnabled(1, 1)
                status.append(True)
            else:
                status.append(False)

            if (self.devices.pmac.axis_status(3) & 1024) != 0:
                self.ui.tb_move_axis.setItemEnabled(2, 1)
                status.append(True)
            else:
                status.append(False)

            if (self.devices.pmac.axis_status(5) & 1024) != 0:
                self.ui.tb_move_axis.setItemEnabled(3, 1)
                status.append(True)
            else:
                status.append(False)

            if (self.devices.pmac.axis_status(6) & 1024) != 0:
                self.ui.tb_move_axis.setItemEnabled(4, 1)
                status.append(True)
            else:
                status.append(False)

            if (self.devices.pmac.axis_status(7) & 1024) != 0:
                self.ui.tb_move_axis.setItemEnabled(5, 1)
                status.append(True)
            else:
                status.append(False)

            if (self.devices.pmac.axis_status(8) & 1024) != 0:
                self.ui.tb_move_axis.setItemEnabled(6, 1)
                status.append(True)
            else:
                status.append(False)

            if (self.devices.pmac.axis_status(9) & 1024) != 0:
                self.ui.tb_move_axis.setItemEnabled(7, 1)
                status.append(True)
            else:
                status.append(False)

            if all(status):
                self.ui.tab_main.setTabEnabled(3, True)
                self.ui.tab_main.setTabEnabled(2, True)

    def start_homming(self):
        """Start hommming."""
        if self.devices is None:
            return

        axis_homming_mask = 0
        list_of_axis = self.devices.pmac.commands.list_of_axis

        for axis in list_of_axis:
            obj = getattr(self.ui, 'cb_homming' + str(axis))
            val = int(obj.isChecked())
            axis_homming_mask += (val << (axis-1))

        self.devices.pmac.align_bench(axis_homming_mask)
        time.sleep(0.1)

        while int(self.devices.pmac.read_response(
                  self.devices.pmac.commands.prog_running)) == 1:
            time.sleep(0.5)
        else:
            self._release_access_to_movement()
            message = 'Finished homming of the selected axes.'
            QtGui.QMessageBox.information(
                self, 'Hommming', message, QtGui.QMessageBox.Ok)

    def move_to_target(self, axis):
        """Move Hall probe to target position."""
        if self.devices is None:
            return

        list_of_axis = self.devices.pmac.commands.list_of_axis

        if axis in list_of_axis:
            set_vel = float(getattr(self.ui, 'le_setvel'+str(axis)).text())
            target = float(getattr(self.ui, 'le_target'+str(axis)).text())
            vel = self.devices.pmac.get_velocity(axis)

            if vel != set_vel:
                self.devices.pmac.set_axis_speed(axis, set_vel)

            self.devices.pmac.move_axis(axis, target)

    def stop_axis(self, axis):
        """Stop axis."""
        if self.devices is None:
            return
        list_of_axis = self.devices.pmac.commands.list_of_axis

        if axis in list_of_axis:
            self.devices.pmac.stop_axis(axis)

    def stop_all_axis(self):
        """Stop all axis."""
        if self.devices is None:
            return
        self.devices.pmac.stop_all_axis()

    def kill_all_axis(self):
        """Kill all axis."""
        if self.devices is None:
            return
        self.devices.pmac.kill_all_axis()

    def stop_measurements(self):
        """Stop measurements."""
        self.stop = True

    def set_softlimits(self):
        """Set Hall bench limits."""
        if self.devices is None:
            return

        cts_mm_axis = self.devices.pmac.commands.CTS_MM_AXIS

        p1_min = self.ui.sb_axis1_min.value()
        p1_max = self.ui.sb_axis1_max.value()
        cmd = self.devices.pmac.set_par(
            self.devices.pmac.commands.i_softlimit_neg_list[0],
            p1_min*cts_mm_axis[0])
        self.devices.pmac.get_response(cmd)
        cmd = self.devices.pmac.set_par(
            self.devices.pmac.commands.i_softlimit_pos_list[0],
            p1_max*cts_mm_axis)
        self.devices.pmac.get_response(cmd)

        p2_min = self.ui.sb_axis2_min.value()
        p2_max = self.ui.sb_axis2_max.value()
        cmd = self.devices.pmac.set_par(
            self.devices.pmac.commands.i_softlimit_neg_list[1],
            p2_min*cts_mm_axis[1])
        self.devices.pmac.get_response(cmd)
        cmd = self.devices.pmac.set_par(
            self.devices.pmac.commands.i_softlimit_pos_list[1],
            p2_max*cts_mm_axis[1])
        self.devices.pmac.get_response(cmd)

        p3_min = self.ui.sb_axis3_min.value()
        p3_max = self.ui.sb_axis3_max.value()
        cmd = self.devices.pmac.set_par(
            self.devices.pmac.commands.i_softlimit_neg_list[2],
            p3_min*cts_mm_axis)
        self.devices.pmac.get_response(cmd)
        cmd = self.devices.pmac.set_par(
            self.devices.pmac.commands.i_softlimit_pos_list[2],
            p3_max*cts_mm_axis)
        self.devices.pmac.get_response(cmd)

    def reset_softlimits(self):
        """Reset Hall bench limits."""
        if self.devices is None:
            return

        cmd = self.devices.pmac.set_par(
            self.devices.pmac.commands.i_softlimit_neg_list[0], 0)
        self.devices.pmac.get_response(cmd)
        cmd = self.devices.pmac.set_par(
            self.devices.pmac.commands.i_softlimit_pos_list[0], 0)
        self.devices.pmac.get_response(cmd)
        self.ui.sb_axis1_min.setValue(0)
        self.ui.sb_axis1_max.setValue(0)

        cmd = self.devices.pmac.set_par(
            self.devices.pmac.commands.i_softlimit_neg_list[1], 0)
        self.devices.pmac.get_response(cmd)
        cmd = self.devices.pmac.set_par(
            self.devices.pmac.commands.i_softlimit_pos_list[1], 0)
        self.devices.pmac.get_response(cmd)
        self.ui.sb_axis2_min.setValue(0)
        self.ui.sb_axis2_max.setValue(0)

        cmd = self.devices.pmac.set_par(
            self.devices.pmac.commands.i_softlimit_neg_list[2], 0)
        self.devices.pmac.get_response(cmd)
        cmd = self.devices.pmac.set_par(
            self.devices.pmac.commands.i_softlimit_pos_list[2], 0)
        self.devices.pmac.get_response(cmd)
        self.ui.sb_axis3_min.setValue(0)
        self.ui.sb_axis3_max.setValue(0)

    def load_calibration_data(self):
        """Load calibration data."""
        filename = QtGui.QFileDialog.getOpenFileName(
            self, 'Open calibration file')

        if len(filename) != 0:
            try:
                self.calibration_data = calibration.CalibrationData(filename)
            except Exception as e:
                QtGui.QMessageBox.critical(
                    self, 'Failure', str(e), QtGui.QMessageBox.Ignore)
                return

            self.ui.le_calibration_filename.setText(filename)

            for probe in ['i', 'j', 'k']:
                probe_data = getattr(
                    self.calibration_data, 'probe' + probe + '_data')
                table = getattr(self.ui, 'tw_probe' + probe)
                label = getattr(self.ui, 'la_probe' + probe)
                n_rows = len(probe_data)
                n_columns = max([len(line) for line in probe_data])

                if self.calibration_data.data_type == 'polynomial':
                    label.setText(
                        'Polynomial coefficients: B = C0 + C1*V + C2*V² + ...')
                    table.setColumnCount(n_rows)
                    table.setRowCount(n_columns)
                    labels = ['V minimum', 'V maximum']
                    for i in range(n_columns-2):
                        labels.append('C%i' % i)
                    table.verticalHeader().setVisible(True)
                    table.horizontalHeader().setVisible(False)
                    table.setVerticalHeaderLabels(labels)

                    for i in range(n_rows):
                        for j in range(n_columns):
                            table.setItem(j, i, QtGui.QTableWidgetItem(
                                str(probe_data[i][j])))

                elif self.calibration_data.data_type == 'interpolation':
                    table.setColumnCount(n_columns)
                    table.setRowCount(n_rows)
                    table.setHorizontalHeaderLabels(['V', 'B'])
                    table.verticalHeader().setVisible(False)
                    table.horizontalHeader().setVisible(True)

                    for i in range(n_rows):
                        for j in range(n_columns):
                            table.setItem(i, j, QtGui.QTableWidgetItem(
                                str(probe_data[i][j])))
                else:
                    msg = 'Invalid data type found in calibration data file.'
                    QtGui.QMessageBox.critical(
                        self, 'Failure', msg, QtGui.QMessageBox.Ignore)
                    return

                table.horizontalHeader().setStretchLastSection(True)
                table.resizeColumnsToContents()
                table.resizeRowsToContents()

            voltage = np.linspace(-15, 15, 101)

            self.ui.gv_probei.clear()
            self.ui.gv_probei.plotItem.plot(
                voltage,
                self.calibration_data.convert_voltage_probei(voltage),
                pen={'color': 'b', 'width': 3})
            self.ui.gv_probei.setLabel(
                'bottom', "Voltage ["+self.calibration_data.voltage_unit+"]")
            self.ui.gv_probei.setLabel(
                'left', "Field [" + self.calibration_data.field_unit + "]")
            self.ui.gv_probei.showGrid(x=True, y=True)

            self.ui.gv_probej.clear()
            self.ui.gv_probej.plotItem.plot(
                voltage,
                self.calibration_data.convert_voltage_probej(voltage),
                pen={'color': 'b', 'width': 3})
            self.ui.gv_probej.setLabel(
                'bottom', "Voltage ["+self.calibration_data.voltage_unit+"]")
            self.ui.gv_probej.setLabel(
                'left', "Field [" + self.calibration_data.field_unit + "]")
            self.ui.gv_probej.showGrid(x=True, y=True)

            self.ui.gv_probek.clear()
            self.ui.gv_probek.plotItem.plot(
                voltage,
                self.calibration_data.convert_voltage_probek(voltage),
                pen={'color': 'b', 'width': 3})
            self.ui.gv_probek.setLabel(
                'bottom', "Voltage ["+self.calibration_data.voltage_unit+"]")
            self.ui.gv_probek.setLabel(
                'left', "Field [" + self.calibration_data.field_unit + "]")
            self.ui.gv_probek.showGrid(x=True, y=True)

            self.ui.le_data_type.setText(
                self.calibration_data.data_type.capitalize())
            self.ui.le_distance_probei.setText(str(
                self.calibration_data.distance_probei))
            self.ui.le_distance_probek.setText(str(
                self.calibration_data.distance_probek))
            self.ui.le_stem_shape.setText(
                self.calibration_data.stem_shape.capitalize())

            self.ui.rb_current_calibration.setEnabled(True)

    def configure_and_measure(self):
        """Configure and start measurements."""
        if (self.devices is None or
           not self.devices.pmac_connected or
           not self._update_measurement_configuration()):
            return

        self.dirpath = QtGui.QFileDialog.getExistingDirectory(
            self, 'Select directory to save measurement data',
            os.path.expanduser("~"), QtGui.QFileDialog.ShowDirsOnly)
        if len(self.dirpath) == 0:
            return

        if not self._save_configuration_files():
            return

        self.directory_to_save = self.dirpath
        self.stop = False
        self.current_position_list = []
        self.current_voltage_list = []
        self.current_measurement = None
        self._clear_graph()
        self._set_axes_speed()

        if self.ui.rb_triggering_axis1.isChecked():
            scan_axis = 1
            axis_a = 2
            axis_b = 3

        elif self.ui.rb_triggering_axis2.isChecked():
            scan_axis = 2
            axis_a = 1
            axis_b = 3

        elif self.ui.rb_triggering_axis3.isChecked():
            scan_axis = 3
            axis_a = 1
            axis_b = 2
        else:
            return

        poslist_a = self._get_measurement_parameters(axis_a)[-1]
        poslist_b = self._get_measurement_parameters(axis_b)[-1]

        for pos_a in poslist_a:
            if self.stop is True:
                self.stop_all_axis()
                break

            for pos_b in poslist_b:
                if self.stop is True:
                    self.stop_all_axis()
                    break

                self._move_axis(axis_a, pos_a)

                self._move_axis(axis_b, pos_b)

                self._measure_line(scan_axis)

        if self.stop is False:
            self._move_to_start_position()
            self.current_measurement = measurement.FieldData(
                self.current_voltage_list, self.calibration_data)
            self._plot_all()

            message = 'End of measurements.'
            QtGui.QMessageBox.information(
                self, 'Measurements', message, QtGui.QMessageBox.Ok)

            self.ui.rb_current_measurement.setChecked(True)
            self.directory_to_save = self.dirpath
            self.measurement_to_save = self.current_measurement

        else:
            self.devices.pmac.stop_all_axis()
            message = 'The user stopped the measurements.'
            QtGui.QMessageBox.information(
                self, 'Abort', message, QtGui.QMessageBox.Ok)

    def _save_configuration_files(self):
        cc_filename = 'connection_configuration.txt'
        mc_filename = 'measurement_configuration.txt'
        ca_filename = 'calibration_data.txt'
        cc_fullpath = os.path.join(self.dirpath, cc_filename)
        mc_fullpath = os.path.join(self.dirpath, mc_filename)
        ca_fullpath = os.path.join(self.dirpath, ca_filename)

        try:
            if self._check_config_files(cc_fullpath, mc_fullpath, ca_fullpath):
                self.cconfig.save_file(cc_fullpath)
                self.mconfig.save_file(mc_fullpath)
                self.calibration_data.save_file(ca_fullpath)
                return True
            else:
                question = ('Inconsistent configuration files. ' +
                            'Overwrite existing files?')
                reply = QtGui.QMessageBox.question(
                    self, 'Question', question, 'Yes', button1Text='No')
                if reply == 0:
                    self.cconfig.save_file(cc_fullpath)
                    self.mconfig.save_file(mc_fullpath)
                    self.calibration_data.save_file(ca_fullpath)
                    return True
                else:
                    return False
        except Exception as e:
            QtGui.QMessageBox.critical(
                self, 'Failure', str(e), QtGui.QMessageBox.Ignore)
            return False

    def _check_config_files(self, cc_fullpath, mc_fullpath, ca_fullpath):
        if os.path.isfile(cc_fullpath):
            tmp = configuration.ConnectionConfig(cc_fullpath)
            if not self.cconfig == tmp:
                return False
        if os.path.isfile(mc_fullpath):
            tmp = configuration.MeasurementConfig(mc_fullpath)
            if not self.mconfig == tmp:
                return False
        if os.path.isfile(ca_fullpath):
            tmp = calibration.CalibrationData(ca_fullpath)
            if not self.calibration_data == tmp:
                return False
        return True

    def _measure_line(self, scan_axis):
        (startpos, endpos, incr, velocity, npts,
            scan_list) = self._get_measurement_parameters(scan_axis)

        aper_displacement = (self.mconfig.meas_aper_ms * velocity)

        extra_mm = float(getattr(self.ui, 'le_extra' + str(scan_axis)).text())

        for idx in range(self.nr_measurements):
            self.current_voltage_data.clear()
            self.devices.voltx.end_measurement = False
            self.devices.volty.end_measurement = False
            self.devices.voltz.end_measurement = False
            self._clear_measurement()
            self._configure_graph()
            self._update_measurement_number(idx+1)

            if self.stop is True:
                break

            # flag to check if sensor is going or returning
            to_pos = not(bool(idx % 2))

            if to_pos:
                self.current_position_list = scan_list + aper_displacement/2
                self._move_axis(scan_axis, startpos - extra_mm)
            else:
                self.current_position_list = (
                    scan_list - aper_displacement/2)[::-1]
                self._move_axis(scan_axis, endpos + extra_mm)

            for axis in self.current_voltage_data.axis_list:
                if axis != scan_axis:
                    pos = self.devices.pmac.get_position(axis)
                    setattr(self.current_voltage_data, 'pos' + str(axis), pos)
                else:
                    setattr(self.current_voltage_data, 'pos' + str(scan_axis),
                            self.current_position_list)

            if self.stop is True:
                break

            self._configure_trigger(
                scan_axis, startpos, endpos, incr, npts, to_pos)
            self._configure_multimeters()
            self._start_reading_threads()

            if self.stop is False:
                if to_pos:
                    self._move_axis_and_update_graph(
                        scan_axis, endpos + extra_mm, idx)
                else:
                    self._move_axis_and_update_graph(
                        scan_axis, startpos - extra_mm, idx)

            self.devices.voltx.end_measurement = True
            self.devices.volty.end_measurement = True
            self.devices.voltz.end_measurement = True
            self._stop_trigger()
            self._wait_reading_threads()

            self.current_voltage_data.probei = self.devices.voltx.voltage
            self.current_voltage_data.probej = self.devices.volty.voltage
            self.current_voltage_data.probek = self.devices.voltz.voltage

            self._kill_reading_threads()

            if self.stop is True:
                break

            if to_pos is True:
                voltage_data = self.current_voltage_data.copy()
            else:
                self.current_voltage_data.reverse()
                voltage_data = self.current_voltage_data.copy()

            self.current_voltage_list.append(voltage_data)
            self._save_voltage_data(voltage_data)

    def _save_voltage_data(self, voltage_data):
        if voltage_data.scan_axis == 3:
            pos_str = ('Z=' + '{0:0.4f}'.format(voltage_data.pos1[0]) + 'mm_' +
                       'Y=' + '{0:0.4f}'.format(voltage_data.pos2[0]) + 'mm')
        elif voltage_data.scan_axis == 2:
            pos_str = ('Z=' + '{0:0.4f}'.format(voltage_data.pos1[0]) + 'mm_' +
                       'X=' + '{0:0.4f}'.format(voltage_data.pos3[0]) + 'mm')
        elif voltage_data.scan_axis == 1:
            pos_str = ('Y=' + '{0:0.4f}'.format(voltage_data.pos2[0]) + 'mm_' +
                       'X=' + '{0:0.4f}'.format(voltage_data.pos3[0]) + 'mm')
        else:
            pos_str = None

        if pos_str is not None:
            name = 'raw_voltage_data_' + pos_str
        else:
            name = 'raw_voltage_data'

        extension = '.dat'
        filename = name + extension
        uniq = 2
        while os.path.exists(os.path.join(self.dirpath, filename)):
            filename = name + '_' + '%i' % uniq + extension
            uniq += 1

        filename = os.path.join(self.dirpath, filename)

        try:
            voltage_data.save_file(filename)
        except Exception as e:
            QtGui.QMessageBox.critical(
                self, 'Failure', str(e), QtGui.QMessageBox.Ignore)
            return False

    def _set_axes_speed(self):
        self.devices.pmac.set_axis_speed(1, self.mconfig.meas_vel_ax1)
        self.devices.pmac.set_axis_speed(2, self.mconfig.meas_vel_ax2)
        self.devices.pmac.set_axis_speed(3, self.mconfig.meas_vel_ax3)
        self.devices.pmac.set_axis_speed(5, self.mconfig.meas_vel_ax5)

    def _get_measurement_parameters(self, axis):
        startpos = getattr(self.mconfig, 'meas_startpos_ax' + str(axis))
        endpos = getattr(self.mconfig, 'meas_endpos_ax' + str(axis))
        incr = getattr(self.mconfig, 'meas_incr_ax' + str(axis))
        vel = getattr(self.mconfig, 'meas_vel_ax' + str(axis))
        npts = np.ceil(round((endpos - startpos) / incr, 4) + 1)
        corr_endpos = startpos + (npts-1)*incr
        poslist = np.linspace(startpos, corr_endpos, npts)
        return (startpos, corr_endpos, incr, vel, npts, poslist)

    def _clear_measurement(self):
        self.current_position_list = []
        self.devices.voltx.clear()
        self.devices.volty.clear()
        self.devices.voltz.clear()

    def _update_measurement_number(self, number):
        self.ui.la_nr_measurements_status.setText('{0:1d}'.format(number))

    def _move_axis(self, axis, position):
        if self.stop is False:
            self.devices.pmac.move_axis(axis, position)
            while ((self.devices.pmac.axis_status(axis) & 1) == 0 and
                   self.stop is False):
                QtGui.QApplication.processEvents()

    def _move_axis_and_update_graph(self, axis, position, graph_idx):
        if self.stop is False:
            self.devices.pmac.move_axis(axis, position)
            while ((self.devices.pmac.axis_status(axis) & 1) == 0 and
                   self.stop is False):
                self._update_graph(graph_idx)
                QtGui.QApplication.processEvents()
                time.sleep(0.05)

    def _move_to_start_position(self):
        self._move_axis(1, self.mconfig.meas_startpos_ax1)
        self._move_axis(2, self.mconfig.meas_startpos_ax2)
        self._move_axis(3, self.mconfig.meas_startpos_ax3)

    def _configure_graph(self):
        self.graph_curve_x = np.append(
            self.graph_curve_x,
            self.ui.gv_measurement_graph.plotItem.plot(
                np.array([]),
                np.array([]),
                pen=(255, 0, 0),
                symbol='o',
                symbolPen=(255, 0, 0),
                symbolSize=4))

        self.graph_curve_y = np.append(
            self.graph_curve_y,
            self.ui.gv_measurement_graph.plotItem.plot(
                np.array([]),
                np.array([]),
                pen=(0, 255, 0),
                symbol='o',
                symbolPen=(0, 255, 0),
                symbolSize=4))

        self.graph_curve_z = np.append(
            self.graph_curve_z,
            self.ui.gv_measurement_graph.plotItem.plot(
                np.array([]),
                np.array([]),
                pen=(0, 0, 255),
                symbol='o',
                symbolPen=(0, 0, 255),
                symbolSize=4))

    def _clear_graph(self):
        self.ui.gv_measurement_graph.plotItem.curves.clear()
        self.ui.gv_measurement_graph.clear()
        self.graph_curve_x = np.array([])
        self.graph_curve_y = np.array([])
        self.graph_curve_z = np.array([])

    def _update_graph(self, n):
        self.graph_curve_x[n].setData(
            self.current_position_list[:len(self.devices.voltx.voltage)],
            self.devices.voltx.voltage)

        self.graph_curve_y[n].setData(
            self.current_position_list[:len(self.devices.volty.voltage)],
            self.devices.volty.voltage)

        self.graph_curve_z[n].setData(
            self.current_position_list[:len(self.devices.voltz.voltage)],
            self.devices.voltz.voltage)

    def _plot_all(self):
        self._clear_graph()

        n = 0
        field1 = self.current_measurement.field1
        positions = field1.index.values
        for col in field1.columns.values:
            self._configure_graph()
            self.graph_curve_z[n].setData(positions, field1.loc[:, col])
            n += 1

        n = 0
        field2 = self.current_measurement.field2
        positions = field2.index.values
        for col in field2.columns.values:
            self._configure_graph()
            self.graph_curve_y[n].setData(positions, field2.loc[:, col])
            n += 1

        n = 0
        field3 = self.current_measurement.field3
        positions = field3.index.values
        for col in field3.columns.values:
            self._configure_graph()
            self.graph_curve_x[n].setData(positions, field3.loc[:, col])
            n += 1

    def _configure_multimeters(self):
        if self.mconfig.meas_probeX:
            self.devices.voltx.config(
                self.mconfig.meas_aper_ms, self.mconfig.meas_precision)
        if self.mconfig.meas_probeY:
            self.devices.volty.config(
                self.mconfig.meas_aper_ms, self.mconfig.meas_precision)
        if self.mconfig.meas_probeZ:
            self.devices.voltz.config(
                self.mconfig.meas_aper_ms, self.mconfig.meas_precision)

    def _reset_multimeters(self):
        self.devices.voltx.reset()
        self.devices.volty.reset()
        self.devices.voltz.reset()

    def _configure_trigger(self, axis, startpos, endpos, incr, npts, to_pos):
        if to_pos:
            self.devices.pmac.set_trigger(axis, startpos, incr, 10, npts, 1)
        else:
            self.devices.pmac.set_trigger(axis, endpos, incr*(-1), 10, npts, 1)

    def _stop_trigger(self):
        self.devices.pmac.stop_trigger()

    def _start_reading_threads(self):
        if self.mconfig.meas_probeZ:
            self.tk = threading.Thread(
                target=self.devices.voltz.read,
                args=(self.mconfig.meas_precision,))
            self.tk.start()

        if self.mconfig.meas_probeY:
            self.tj = threading.Thread(
                target=self.devices.volty.read,
                args=(self.mconfig.meas_precision,))
            self.tj.start()

        if self.mconfig.meas_probeX:
            self.ti = threading.Thread(
                target=self.devices.voltx.read,
                args=(self.mconfig.meas_precision,))
            self.ti.start()

    def _wait_reading_threads(self):
        if self.tk is not None:
            while self.tk.is_alive() and self.stop is False:
                QtGui.QApplication.processEvents()

        if self.tj is not None:
            while self.tj.is_alive() and self.stop is False:
                QtGui.QApplication.processEvents()

        if self.ti is not None:
            while self.ti.is_alive() and self.stop is False:
                QtGui.QApplication.processEvents()

    def _kill_reading_threads(self):
        try:
            del self.tk
            del self.tj
            del self.ti
        except Exception:
            pass

    def select_measurement_to_save(self):
        """Select measurement to save."""
        self.ui.le_directory.setText(self.directory_to_save)
        self.ui.pb_recover_data.setEnabled(False)

        if self.ui.rb_current_measurement.isChecked():
            self.ui.gb_voltage_data.setEnabled(False)
            self.ui.gb_select_calibration.setEnabled(False)
            self.ui.lw_voltage_data.clear()
            self.ui.le_recover_calibration_filename.setText('')
            self.measurement_to_save = self.current_measurement
        elif self.ui.rb_recover_measurement.isChecked():
            self.ui.gb_voltage_data.setEnabled(True)
            self.ui.gb_select_calibration.setEnabled(True)
            if self.ui.rb_load_calibration.isChecked():
                self.ui.pb_load_recover_calibration_file.setEnabled(True)
            else:
                if self.calibration_data is None:
                    self.ui.rb_current_calibration.setEnabled(False)
                    self.ui.rb_load_calibration.setChecked(True)
                    self.ui.pb_load_recover_calibration_file.setEnabled(True)
                else:
                    self.ui.pb_load_recover_calibration_file.setEnabled(False)
                    self.ui.rb_current_calibration.setEnabled(True)

        if self.measurement_to_save is not None:
            self.ui.pb_save_measurement.setEnabled(True)
        else:
            self.ui.pb_save_measurement.setEnabled(False)

    def change_directory(self):
        """Change directory."""
        directory = QtGui.QFileDialog.getExistingDirectory(
            self, 'Select directory to save measurement',
            os.path.expanduser("~"), QtGui.QFileDialog.ShowDirsOnly)
        if len(directory) != 0:
            self.directory_to_save = directory
        self.ui.le_directory.setText(self.directory_to_save)

    def load_voltage_data_files(self):
        """Load voltage data files."""
        default_dir = os.path.expanduser('~')
        filepaths = QtGui.QFileDialog.getOpenFileNames(directory=default_dir)
        if len(filepaths) == 0:
            return

        if any([x == -1 for x in [f.find('.dat') for f in filepaths]]):
            QtGui.QMessageBox.warning(
                self, 'Warning', 'Cannot open files. Select valid files.',
                QtGui.QMessageBox.Ok)
            return

        filenames = [os.path.split(f)[1] for f in filepaths]

        self.ui.lw_voltage_data.clear()
        self.ui.lw_voltage_data.insertItems(0, filenames)
        self.voltage_data_files = filepaths
        self.measurement_to_save = None
        self.ui.pb_save_measurement.setEnabled(False)
        if self.calibration_data_to_save is not None:
            self.ui.pb_recover_data.setEnabled(True)
        else:
            self.ui.pb_recover_data.setEnabled(False)

    def select_calibration_data(self):
        """Select calibration data."""
        self.ui.le_recover_calibration_filename.setText('')
        self.measurement_to_save = None
        self.ui.pb_save_measurement.setEnabled(False)
        self.ui.pb_recover_data.setEnabled(False)

        if self.ui.rb_load_calibration.isChecked():
            self.ui.pb_load_recover_calibration_file.setEnabled(True)
            self.calibration_data_to_save = None
        else:
            self.ui.pb_load_recover_calibration_file.setEnabled(False)
            self.calibration_data_to_save = self.calibration_data
            if len(self.voltage_data_files) != 0:
                self.ui.pb_recover_data.setEnabled(True)

    def load_recover_calibration_data(self):
        """Load calibration file to use in data recover."""
        filename = QtGui.QFileDialog.getOpenFileName(
            self, 'Open calibration data file')

        if len(filename) != 0:
            try:
                self.calibration_data_to_save = calibration.CalibrationData(
                    filename)
                self.ui.le_recover_calibration_filename.setText(filename)
                if len(self.voltage_data_files) != 0:
                    self.ui.pb_recover_data.setEnabled(True)
                else:
                    self.ui.pb_recover_data.setEnabled(False)
            except Exception as e:
                QtGui.QMessageBox.critical(
                    self, 'Failure', str(e), QtGui.QMessageBox.Ignore)
                return

    def recover_from_voltage_data_files(self):
        """Recover measurement from voltage data files."""
        if (len(self.voltage_data_files) != 0 and
           self.calibration_data_to_save is not None):
            try:
                self.measurement_to_save = measurement.FieldData(
                    self.voltage_data_files, self.calibration_data_to_save)
                self.ui.pb_save_measurement.setEnabled(True)
                message = 'Measurement data successfully recovered.'
                QtGui.QMessageBox.information(
                    self, 'Information', message, QtGui.QMessageBox.Ok)

            except Exception as e:
                QtGui.QMessageBox.critical(
                    self, 'Failure', str(e), QtGui.QMessageBox.Ignore)

    def disable_invalid_magnet_axes(self):
        """Disable invalid magnet axes."""
        for i in range(6):
            self.ui.cb_magnet_axisy.model().item(i).setEnabled(True)

        idx_axisx = self.ui.cb_magnet_axisx.currentIndex()
        idx_axisy = self.ui.cb_magnet_axisy.currentIndex()
        if idx_axisx in [0, 1]:
            if idx_axisy in [0, 1]:
                self.ui.cb_magnet_axisy.setCurrentIndex(-1)
            self.ui.cb_magnet_axisy.model().item(0).setEnabled(False)
            self.ui.cb_magnet_axisy.model().item(1).setEnabled(False)
        elif idx_axisx in [2, 3]:
            if idx_axisy in [2, 3]:
                self.ui.cb_magnet_axisy.setCurrentIndex(-1)
            self.ui.cb_magnet_axisy.model().item(2).setEnabled(False)
            self.ui.cb_magnet_axisy.model().item(3).setEnabled(False)
        elif idx_axisx in [4, 5]:
            if idx_axisy in [4, 5]:
                self.ui.cb_magnet_axisy.setCurrentIndex(-1)
            self.ui.cb_magnet_axisy.model().item(4).setEnabled(False)
            self.ui.cb_magnet_axisy.model().item(5).setEnabled(False)

    def enabled_coil(self, checkbox, frame, line_edit_1, line_edit_2):
        """Enabled or disabled coil frame."""
        if checkbox.isChecked():
            frame.setEnabled(True)
        else:
            frame.setEnabled(False)
            line_edit_1.clear()
            line_edit_2.clear()

    def load_magnet_info(self):
        """Load pre-defined magnet info."""
        if self.ui.cb_predefined.currentIndex() < 1:
            self.ui.la_predefined_description.setText('')
            self.ui.le_gap.setText('')
            self.ui.le_control_gap.setText('')
            self.ui.le_magnet_length.setText('')
            self.ui.cb_main.setChecked(False)
            self.ui.cb_trim.setChecked(False)
            self.ui.cb_ch.setChecked(False)
            self.ui.cb_cv.setChecked(False)
            self.ui.cb_qs.setChecked(False)
            self.ui.ta_additional_parameter.setRowCount(0)
            return

        m = magnets_info.get_magnet_info(self.ui.cb_predefined.currentText())

        if m is not None:
            m.pop('name')
            self.ui.ta_additional_parameter.setRowCount(0)
            self.ui.la_predefined_description.setText(m.pop('description'))
            self.ui.le_gap.setText(str(m.pop('gap[mm]')))
            self.ui.le_control_gap.setText(str(m.pop('control_gap[mm]')))
            self.ui.le_magnet_length.setText(str(m.pop('magnet_length[mm]')))

            if 'nr_turns_main' in m.keys():
                self.ui.le_main_turns.setText(str(m.pop('nr_turns_main')))
                self.ui.cb_main.setChecked(True)
            else:
                self.ui.cb_main.setChecked(False)

            if 'nr_turns_trim' in m.keys():
                self.ui.le_trim_current.setText('0')
                self.ui.le_trim_turns.setText(str(m.pop('nr_turns_trim')))
                self.ui.cb_trim.setChecked(True)
            else:
                self.ui.cb_trim.setChecked(False)

            if 'nr_turns_ch' in m.keys():
                self.ui.le_ch_current.setText('0')
                self.ui.le_ch_turns.setText(str(m.pop('nr_turns_ch')))
                self.ui.cb_ch.setChecked(True)
            else:
                self.ui.cb_ch.setChecked(False)

            if 'nr_turns_cv' in m.keys():
                self.ui.le_cv_current.setText('0')
                self.ui.le_cv_turns.setText(str(m.pop('nr_turns_cv')))
                self.ui.cb_cv.setChecked(True)
            else:
                self.ui.cb_cv.setChecked(False)

            if 'nr_turns_qs' in m.keys():
                self.ui.le_qs_current.setText('0')
                self.ui.le_qs_turns.setText(str(m.pop('nr_turns_qs')))
                self.ui.cb_qs.setChecked(True)
            else:
                self.ui.cb_qs.setChecked(False)

            if len(m) != 0:
                count = 0
                for parameter, value in m.items():
                    self.ui.ta_additional_parameter.setRowCount(count+1)
                    self.ui.ta_additional_parameter.setItem(
                        count, 0, QtGui.QTableWidgetItem(str(parameter)))
                    self.ui.ta_additional_parameter.setItem(
                        count, 1, QtGui.QTableWidgetItem(str(value)))
                    count = count + 1

    def save_measurement(self):
        """Save measurement."""
        if self.measurement_to_save is not None:
            datetime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
            date = datetime.split('_')[0]
            magnet_name = self.ui.le_magnet_name.text()
            magnet_length = self.ui.le_magnet_length.text()
            gap = self.ui.le_gap.text()
            control_gap = self.ui.le_control_gap.text()

            info = []
            info.append(['fieldmap_name', magnet_name])
            info.append(['timestamp', datetime])
            info.append(['nr_magnets', 1])
            info.append(['magnet_name', magnet_name])
            info.append(['gap[mm]', gap])
            info.append(['control_gap[mm]', control_gap])
            info.append(['magnet_length[mm]', magnet_length])

            if len(magnet_name) != 0:
                filename = magnet_name
            else:
                filename = 'hall_probe_measurement'

            if self.ui.cb_main.isChecked():
                current = self.ui.le_main_current.text()
                info.append(['current_main[A]', current])
                info.append(['nr_turns_main', self.ui.le_main_turns.text()])
                if len(current) != 0:
                    filename = filename + '_Imc=' + current + 'A'
            if self.ui.cb_trim.isChecked():
                current = self.ui.le_trim_current.text()
                info.append(['current_trim[A]', current])
                info.append(['nr_turns_trim', self.ui.le_trim_turns.text()])
                if len(current) != 0:
                    filename = filename + '_Itc=' + current + 'A'
            if self.ui.cb_ch.isChecked():
                current = self.ui.le_ch_current.text()
                info.append(['current_ch[A]', current])
                info.append(['nr_turns_ch', self.ui.le_ch_turns.text()])
                if len(current) != 0:
                    filename = filename + '_Ich=' + current + 'A'
            if self.ui.cb_cv.isChecked():
                current = self.ui.le_cv_current.text()
                info.append(['current_cv[A]', current])
                info.append(['nr_turns_cv', self.ui.le_cv_turns.text()])
                if len(current) != 0:
                    filename = filename + '_Icv=' + current + 'A'
            if self.ui.cb_qs.isChecked():
                current = self.ui.le_qs_current.text()
                info.append(['current_qs[A]', current])
                info.append(['nr_turns_qs', self.ui.le_qs_turns.text()])
                if len(current) != 0:
                    filename = filename + '_Iqs=' + current + 'A'

            filename = '{0:1s}_{1:1s}.dat'.format(date, filename)
            info.insert(2, ['filename', filename])

            for i in range(self.ui.ta_additional_parameter.rowCount()):
                parameter = self.ui.ta_additional_parameter.item(i, 0).text()
                value = self.ui.ta_additional_parameter.item(i, 1).text()
                if len(value) != 0:
                    info.append([parameter.replace(" ", ""), value])

            info.append(['center_pos_z[mm]', '0'])
            info.append(['center_pos_x[mm]', '0'])
            info.append(['rotation[deg]', '0'])

            center_pos3 = self.ui.sb_magnet_center_pos3.value()
            center_pos2 = self.ui.sb_magnet_center_pos2.value()
            center_pos1 = self.ui.sb_magnet_center_pos1.value()
            magnet_center = [center_pos3, center_pos2, center_pos1]

            magnet_x_axis = self.ui.cb_magnet_axisx.currentText()
            magnet_y_axis = self.ui.cb_magnet_axisy.currentText()

            if self.directory_to_save is not None:
                filename = os.path.join(self.directory_to_save, filename)

            try:
                self.measurement_to_save.save_file(
                    filename,
                    header_info=info,
                    magnet_center=magnet_center,
                    magnet_x_axis=magnet_x_axis,
                    magnet_y_axis=magnet_y_axis,
                )
                message = 'Measurement data saved in file: \n%s' % filename
                QtGui.QMessageBox.information(
                    self, 'Information', message, QtGui.QMessageBox.Ok)

            except Exception as e:
                QtGui.QMessageBox.critical(
                    self, 'Failure', str(e), QtGui.QMessageBox.Ignore)


class HallBenchDevices(object):
    """Hall Bench Devices."""

    def __init__(self, connection_configururation):
        """Initiate variables."""
        self.pmac = None
        self.voltx = None
        self.volty = None
        self.voltz = None
        self.multich = None
        self.devices_loaded = False
        self.pmac_connected = False
        self.voltx_connected = False
        self.volty_connected = False
        self.voltz_connected = False
        self.multich_connected = False
        self.config = connection_configururation

    def load(self):
        """Load devices."""
        try:
            self.pmac = Pmac()

            self.voltx = DigitalMultimeter(
                'volt_i.log', self.config.control_voltx_addr)

            self.volty = DigitalMultimeter(
                'volt_j.log', self.config.control_volty_addr)

            self.voltz = DigitalMultimeter(
                'volt_k.log', self.config.control_voltz_addr)

            self.multich = Multichannel(
                'multi.log', self.config.control_multich_addr)

            self.devices_loaded = True
        except Exception:
            self.devices_loaded = False

    def connect(self):
        """Connect devices."""
        if self.devices_loaded:
            if self.config.control_voltx_enable:
                self.voltx_connected = self.voltx.connect()

            if self.config.control_volty_enable:
                self.volty_connected = self.volty.connect()

            if self.config.control_voltz_enable:
                self.voltz_connected = self.voltz.connect()

            if self.config.control_pmac_enable:
                self.pmac_connected = self.pmac.connect()

            if self.config.control_multich_enable:
                self.multich_connected = self.multich.connect()

    def disconnect(self):
        """Disconnect devices."""
        if self.voltx is not None:
            voltx_disconnected = self.voltx.disconnect()
            self.voltx_connected = not voltx_disconnected

        if self.config.control_volty_enable:
            volty_disconnected = self.volty.disconnect()
            self.volty_connected = not volty_disconnected

        if self.config.control_voltz_enable:
            voltz_disconnected = self.voltz.disconnect()
            self.voltz_connected = not voltz_disconnected

        if self.config.control_pmac_enable:
            pmac_disconnected = self.pmac.disconnect()
            self.pmac_connected = pmac_disconnected

        if self.config.control_multich_enable:
            multich_disconnected = self.multich.disconnect()
            self.multich_connected = not multich_disconnected


class GUIThread(threading.Thread):
    """GUI Thread."""

    def __init__(self):
        """Start thread."""
        threading.Thread.__init__(self)
        self.start()

    def run(self):
        """Thread target function."""
        self.app = QtGui.QApplication(sys.argv)
        self.myapp = HallBenchGUI()
        self.myapp.show()
        sys.exit(self.app.exec_())
        self.myapp.timer.stop()


gui_thread = GUIThread()
