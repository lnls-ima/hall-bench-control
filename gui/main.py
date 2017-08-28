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
from save_dialog import Ui_SaveMeasurement
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

        self.ui.cb_select_axis.setCurrentIndex(-1)
        for idx in range(1, self.ui.tab.count()):
            self.ui.tab.setTabEnabled(idx, False)
        self.ui.tb_motors_main.setItemEnabled(1, False)

        self._initialize_variables()
        self._connect_signals_slots()
        self._start_timer()

    def _initialize_variables(self):
        """Initialize variables with default values."""
        self.selected_axis = -1

        self.cconfig = None
        self.mconfig = None
        self.devices = None

        self.tx = None
        self.ty = None
        self.tz = None

        self.graph_curve_x = np.array([])
        self.graph_curve_y = np.array([])
        self.graph_curve_z = np.array([])

        self.calibration_data = calibration.CalibrationData()

        self.current_postion_list = []
        self.current_line_scan = None
        self.current_measurement = None

        self.nr_measurements = 1
        self.dirpath = None

        self.led_on = ":/images/images/led_green.png"
        self.led_off = ":/images/images/led_red.png"

        self.save_measurement_dialog = SaveMeasurementDialog()
        self.end_measurements = False
        self.stop = False

    def _connect_signals_slots(self):
        """Make the connections between signals and slots."""
        # load and save device parameters
        self.ui.pb_load_connection_config.clicked.connect(
            self.load_connection_configuration_file)

        self.ui.pb_save_connection_config.clicked.connect(
            self.save_connection_configuration_file)

        # connect devices
        self.ui.pb_connect_devices.clicked.connect(self.connect_devices)
        self.ui.pb_disconnect_devices.clicked.connect(self.disconnect_devices)

        # activate bench
        self.ui.pb_activate_bench.clicked.connect(self.activate_bench)

        # select axis
        self.ui.cb_select_axis.currentIndexChanged.connect(self.axis_selection)

        # start homming of selected axis
        self.ui.pb_start_homming.clicked.connect(self.start_homming)

        # move to target
        self.ui.pb_move_to_target.clicked.connect(self.move_to_target)

        # stop motor
        self.ui.pb_stop_motor.clicked.connect(self.stop_axis)

        # stop all motors
        self.ui.pb_stop_all_motors.clicked.connect(self.stop_all_axis)

        # kill all motors
        self.ui.pb_kill_all_motors.clicked.connect(self.kill_all_axis)

        # load and save measurements parameters
        self.ui.pb_load_measurement_config.clicked.connect(
            self.load_measurement_configuration_file)

        self.ui.pb_save_measurement_config.clicked.connect(
            self.save_measurement_configuration_file)

        # check limits
        self.ui.le_velocity.editingFinished.connect(
            lambda: self._check_value(self.ui.le_velocity, 0, 150))
        self.ui.le_target_position.editingFinished.connect(
            lambda: self._check_value(self.ui.le_target_position, -3600, 3600))

        # check input values for measurement
        self.ui.le_axis1_start.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis1_start, -3500, 3500))
        self.ui.le_axis2_start.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis2_start, -150, 150))
        self.ui.le_axis3_start.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis3_start, -150, 150))
        self.ui.le_axis5_start.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis5_start, 0, 180))

        self.ui.le_axis1_end.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis1_end, -3500, 3500))
        self.ui.le_axis2_end.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis2_end, -150, 150))
        self.ui.le_axis3_end.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis3_end, -150, 150))
        self.ui.le_axis5_end.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis5_end, 0, 180))

        self.ui.le_axis1_step.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis1_step, -10, 10))
        self.ui.le_axis2_step.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis2_step, -10, 10))
        self.ui.le_axis3_step.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis3_step, -10, 10))
        self.ui.le_axis5_step.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis5_step, -10, 10))

        self.ui.le_axis1_extra.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis1_extra, 0, 100))
        self.ui.le_axis2_extra.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis2_extra, 0, 100))
        self.ui.le_axis3_extra.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis3_extra, 0, 100))
        self.ui.le_axis5_extra.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis5_extra, 0, 100))

        self.ui.le_axis1_vel.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis1_vel, 0.1, 150))
        self.ui.le_axis2_vel.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis2_vel, 0.1, 5))
        self.ui.le_axis3_vel.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis3_vel, 0.1, 5))
        self.ui.le_axis5_vel.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis5_vel, 0.1, 10))

        # Configure and start measurements
        self.ui.pb_configure_and_measure.clicked.connect(
            self.configure_and_measure)

        self.ui.pb_stop_measurement.clicked.connect(self.stop_measurements)

        self.ui.pb_save_measurement.clicked.connect(
            self.open_save_measurement_dialog)

    def _check_value(self, obj, limit_min, limit_max):
        try:
            val = float(obj.text())
            if val >= limit_min and val <= limit_max:
                obj.setText('{0:0.4f}'.format(val))
            else:
                obj.setText('')
                self.axis_selection()
        except Exception:
            obj.setText('')
            self.axis_selection()

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
                        pos_axis = getattr(self.ui, 'le_pos' + str(axis))
                        pos_axis.setText('{0:0.4f}'.format(pos))
                    QtGui.QApplication.processEvents()
        except Exception:
            pass

    def load_connection_configuration_file(self):
        """Load configuration file to set connection parameters."""
        filename = QtGui.QFileDialog.getOpenFileName(
            self, 'Open connection configuration file')

        if len(filename) != 0:
            try:
                if self.cconfig is None:
                    self.cconfig = configuration.ConnectionConfig(filename)
                else:
                    self.cconfig.read_file(filename)
            except configuration.ConfigurationFileError as e:
                QtGui.QMessageBox.critical(
                    self, 'Failure', e.message, QtGui.QMessageBox.Ignore)
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
                except configuration.ConfigurationFileError as e:
                    QtGui.QMessageBox.critical(
                        self, 'Failure', e.message, QtGui.QMessageBox.Ignore)

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
                if self.mconfig is None:
                    self.mconfig = configuration.MeasurementConfig(filename)
                else:
                    self.mconfig.read_file(filename)
            except configuration.ConfigurationFileError as e:
                QtGui.QMessageBox.critical(
                    self, 'Failure', e.message, QtGui.QMessageBox.Ignore)
                return

            self.ui.le_measurement_config_filename.setText(filename)

            self.ui.cb_hall_x_enable.setChecked(self.mconfig.meas_probeX)
            self.ui.cb_hall_y_enable.setChecked(self.mconfig.meas_probeY)
            self.ui.cb_hall_z_enable.setChecked(self.mconfig.meas_probeZ)

            self.ui.le_dmm_aper.setText(str(self.mconfig.meas_aper_ms))
            self.ui.cb_dmm_precision.setCurrentIndex(
                self.mconfig.meas_precision)

            axis_measurement = [1, 2, 3, 5]
            for axis in axis_measurement:
                tmp = getattr(self.ui, 'le_axis' + str(axis) + '_start')
                value = getattr(self.mconfig, 'meas_startpos_ax' + str(axis))
                tmp.setText(str(value))

                tmp = getattr(self.ui, 'le_axis' + str(axis) + '_end')
                value = getattr(self.mconfig, 'meas_endpos_ax' + str(axis))
                tmp.setText(str(value))

                tmp = getattr(self.ui, 'le_axis' + str(axis) + '_step')
                value = getattr(self.mconfig, 'meas_incr_ax' + str(axis))
                tmp.setText(str(value))

                tmp = getattr(self.ui, 'le_axis' + str(axis) + '_vel')
                value = getattr(self.mconfig, 'meas_vel_ax' + str(axis))
                tmp.setText(str(value))

    def save_measurement_configuration_file(self):
        """Save measurement parameters to file."""
        filename = QtGui.QFileDialog.getSaveFileName(
            self, 'Save measurement configuration file')

        if len(filename) != 0:
            if self._update_measurement_configuration():
                try:
                    self.mconfig.save_file(filename)
                except configuration.ConfigurationFileError as e:
                    QtGui.QMessageBox.critical(
                        self, 'Failure', e.message, QtGui.QMessageBox.Ignore)

    def _update_measurement_configuration(self):
        if self.mconfig is None:
            self.mconfig = configuration.MeasurementConfig()

        self.mconfig.meas_probeX = self.ui.cb_hall_x_enable.isChecked()
        self.mconfig.meas_probeY = self.ui.cb_hall_y_enable.isChecked()
        self.mconfig.meas_probeZ = self.ui.cb_hall_z_enable.isChecked()

        self.mconfig.meas_precision = self.ui.cb_dmm_precision.currentIndex()
        self.mconfig.meas_trig_axis = 1

        self.nr_measurements = self.ui.sb_nr_measurements.value()

        tmp = self.ui.le_dmm_aper.text()
        if bool(tmp and tmp.strip()):
            self.mconfig.meas_aper_ms = float(tmp)

        axis_measurement = [1, 2, 3, 5]
        for axis in axis_measurement:
            tmp = getattr(self.ui, 'le_axis' + str(axis) + '_start').text()
            if bool(tmp and tmp.strip()):
                setattr(
                    self.mconfig, 'meas_startpos_ax' + str(axis), float(tmp))

            tmp = getattr(self.ui, 'le_axis' + str(axis) + '_end').text()
            if bool(tmp and tmp.strip()):
                setattr(self.mconfig, 'meas_endpos_ax' + str(axis), float(tmp))

            tmp = getattr(self.ui, 'le_axis' + str(axis) + '_step').text()
            if bool(tmp and tmp.strip()):
                setattr(self.mconfig, 'meas_incr_ax' + str(axis), float(tmp))

            tmp = getattr(self.ui, 'le_axis' + str(axis) + '_vel').text()
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
                self.ui.tab.setTabEnabled(1, True)
                self.ui.tab.setTabEnabled(2, True)

                # check if all axis are hommed and release access to movement.
                list_of_axis = self.devices.pmac.commands.list_of_axis
                status = []
                for axis in list_of_axis:
                    status.append(
                        (self.devices.pmac.axis_status(axis) & 1024) != 0)

                if all(status):
                    self.ui.tb_motors_main.setItemEnabled(1, True)
                    self.ui.tb_motors_main.setCurrentIndex(1)

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
            if not self.devices.pmac.activate_bench():
                message = 'Failed to active bench.'
                QtGui.QMessageBox.critical(
                    self, 'Failure', message, QtGui.QMessageBox.Ok)

    def axis_selection(self):
        """Update seleted axis and velocity values."""
        # get axis selected
        tmp = self.ui.cb_select_axis.currentText()
        if tmp == '':
            self.selected_axis = -1
        else:
            self.selected_axis = int(tmp[1])

            # set target to zero
            self.ui.le_target_position.setText('{0:0.4f}'.format(0))

            if self.devices is not None:
                # read selected axis velocity
                vel = self.devices.pmac.get_velocity(self.selected_axis)
                self.ui.le_velocity.setText('{0:0.4f}'.format(vel))

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
            # check if all axis are homed and release access to movement.
            s = []
            for axis in list_of_axis:
                s.append((self.devices.pmac.axis_status(axis) & 1024) != 0)

            if all(s):
                self.ui.tb_motors_main.setItemEnabled(1, True)
                self.ui.tb_motors_main.setCurrentIndex(1)

    def move_to_target(self):
        """Move Hall probe to target position."""
        if self.devices is None:
            return

        # if any available axis is selected:
        if not self.selected_axis == -1:
            set_vel = float(self.ui.le_velocity.text())
            target = float(self.ui.le_target_position.text())
            vel = self.devices.pmac.get_velocity(self.selected_axis)

            if vel != set_vel:
                self.devices.pmac.set_axis_speed(self.selected_axis, set_vel)

            self.devices.pmac.move_axis(self.selected_axis, target)

    def stop_axis(self):
        """Stop axis."""
        if self.devices is None:
            return
        # if any available axis is selected:
        if not self.selected_axis == -1:
            self.devices.pmac.stop_axis(self.selected_axis)

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

        try:
            self.current_measurement = measurement.Measurement(
                self.cconfig, self.mconfig, self.dirpath)
            print(self.current_measurement)
        except measurement.MeasurementDataError:
            question = 'Inconsistent configuration files. Overwrite files?'
            reply = QtGui.QMessageBox.question(
                self, 'Question', question, 'Yes', button1Text='No')
            if reply == 0:
                self.current_measurement = measurement.Measurement(
                    self.cconfig, self.mconfig, self.dirpath,
                    overwrite_config=True)
            else:
                return

        self.stop = False
        self._clear_graph()
        self._set_axes_speed()

        if self.ui.rb_axis1_triggering.isChecked():
            extra_mm = float(self.ui.le_axis1_extra.text())
            scan_axis = 1
            axis_a = 2
            axis_b = 3

        elif self.ui.rb_axis2_triggering.isChecked():
            extra_mm = float(self.ui.le_axis2_extra.text())
            scan_axis = 2
            axis_a = 1
            axis_b = 3

        elif self.ui.rb_axis3_triggering.isChecked():
            extra_mm = float(self.ui.le_axis3_extra.text())
            scan_axis = 3
            axis_a = 1
            axis_b = 2

        poslist_a = self._get_axis_parameters(axis_a)[-1]
        poslist_b = self._get_axis_parameters(axis_b)[-1]

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

                self._measure_line(axis_a, pos_a, axis_b, pos_b,
                                   scan_axis, extra_mm)

                line_scan = measurement.LineScan.copy(self.current_line_scan)
                self.current_measurement.add_line_scan(line_scan)

        if self.stop is False:
            self._move_to_start_position()
            self._plot_all()

            message = 'End of measurements.'
            QtGui.QMessageBox.information(
                self, 'Measurements', message, QtGui.QMessageBox.Ok)

        else:
            self.devices.pmac.stop_all_axis()
            message = 'The user stopped the measurements.'
            QtGui.QMessageBox.information(
                self, 'Abort', message, QtGui.QMessageBox.Ok)

    def _measure_line(self, axis_a, pos_a, axis_b, pos_b, scan_axis, extra_mm):

        (startpos, endpos, incr, velocity, npts,
            scan_list) = self._get_axis_parameters(scan_axis)

        aper_displacement = (self.mconfig.meas_aper_ms * velocity)

        posx, posy, posz = self._get_position_xyz(
            axis_a, pos_a, axis_b, pos_b, scan_axis, scan_list)

        self.current_line_scan = measurement.LineScan(
            posx, posy, posz, self.calibration_data, self.dirpath)

        for idx in range(self.nr_measurements):

            self.end_measurements = False

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

            self.end_measurements = True

            self._stop_trigger()
            self._wait_reading_threads()

            posx, posy, posz = self._get_position_xyz(
                axis_a, pos_a, axis_b, pos_b,
                scan_axis, self.current_position_list)

            scan = measurement.DataSet(unit='V')  # Fix!!
            scan.posx = posx
            scan.posy = posy
            scan.posz = posz
            scan.datax = self.devices.voltx.voltage
            scan.datay = self.devices.volty.voltage
            scan.dataz = self.devices.voltz.voltage

            self._kill_reading_threads()

            if self.stop is True:
                break

            if to_pos is True:
                self.current_line_scan.add_scan(scan)
            else:
                self.current_line_scan.add_scan(scan.reverse())

        self.current_line_scan.analyse_data()

    def open_save_measurement_dialog(self):
        """Open save measurement dialog."""
        self.save_measurement_dialog.measurement = self.current_measurement
        self.save_measurement_dialog.open()

    def _set_axes_speed(self):
        self.devices.pmac.set_axis_speed(1, self.mconfig.meas_vel_ax1)
        self.devices.pmac.set_axis_speed(2, self.mconfig.meas_vel_ax2)
        self.devices.pmac.set_axis_speed(3, self.mconfig.meas_vel_ax3)
        self.devices.pmac.set_axis_speed(5, self.mconfig.meas_vel_ax5)

    def _get_axis_parameters(self, axis):
        startpos = getattr(self.mconfig, 'meas_startpos_ax' + str(axis))
        endpos = getattr(self.mconfig, 'meas_endpos_ax' + str(axis))
        incr = getattr(self.mconfig, 'meas_incr_ax' + str(axis))
        vel = getattr(self.mconfig, 'meas_vel_ax' + str(axis))
        npts = int(round((endpos - startpos) / incr, 4) + 1)
        poslist = np.linspace(startpos, endpos, npts)
        return (startpos, endpos, incr, vel, npts, poslist)

    def _get_position_xyz(self, axis_a, pos_a, axis_b, pos_b, axis_c, pos_c):
        if axis_a == 3:
            posx = pos_a
        elif axis_b == 3:
            posx == pos_b
        elif axis_c == 3:
            posx = pos_c
        else:
            posx = None

        if axis_a == 2:
            posy = pos_a
        elif axis_b == 2:
            posy == pos_b
        elif axis_c == 2:
            posy = pos_c
        else:
            posy = None

        if axis_a == 1:
            posz = pos_a
        elif axis_b == 1:
            posz == pos_b
        elif axis_c == 1:
            posz = pos_c
        else:
            posz = None

        return (posx, posy, posz)

    def _clear_measurement(self):
        self.current_position_list = []
        self.devices.voltx.voltage = np.array([])
        self.devices.volty.voltage = np.array([])
        self.devices.voltz.voltage = np.array([])

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

    def _plot_all(self, data='avg_field'):
        self._clear_graph()

        n = 0
        for _dict in self.current_measurement.data.values():
            for ls in _dict.values():
                self._configure_graph()
                curve = getattr(ls, data)
                position = ls.scan_positions

                self.graph_curve_x[n].setData(position, curve.datax)
                self.graph_curve_y[n].setData(position, curve.datay)
                self.graph_curve_z[n].setData(position, curve.dataz)
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
            self.tz = threading.Thread(
                target=self.devices.voltz.read,
                args=(self.stop, self.end_measurements,
                      self.mconfig.meas_precision,))
            self.tz.start()

        if self.mconfig.meas_probeY:
            self.ty = threading.Thread(
                target=self.devices.volty.read,
                args=(self.stop, self.end_measurements,
                      self.mconfig.meas_precision,))
            self.ty.start()

        if self.mconfig.meas_probeX:
            self.tx = threading.Thread(
                target=self.devices.voltx.read,
                args=(self.stop, self.end_measurements,
                      self.mconfig.meas_precision,))
            self.tx.start()

    def _wait_reading_threads(self):
        if self.tz is not None:
            while self.tz.is_alive() and self.stop is False:
                QtGui.QApplication.processEvents()

        if self.ty is not None:
            while self.ty.is_alive() and self.stop is False:
                QtGui.QApplication.processEvents()

        if self.tx is not None:
            while self.tx.is_alive() and self.stop is False:
                QtGui.QApplication.processEvents()

    def _kill_reading_threads(self):
        try:
            del self.tz
            del self.ty
            del self.tx
        except Exception:
            pass


class SaveMeasurementDialog(QtGui.QDialog):
    """Save measurement dialog."""

    def __init__(self):
        """Initialize widget."""
        QtGui.QWidget.__init__(self)
        self.ui = Ui_SaveMeasurement()
        self.ui.setupUi(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        self.measurement = None

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
            lambda: self.ui.ta_additional.setRowCount(
                self.ui.ta_additional.rowCount() + 1))

        self.ui.pb_remove_row.clicked.connect(
            lambda: self.ui.ta_additional.setRowCount(
                self.ui.ta_additional.rowCount() - 1))

        self.ui.pb_save.clicked.connect(self.save_measurement)

    def open(self):
        """Open dialog."""
        if self.measurement is not None:
            self.exec_()

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
            return

        m = magnets_info.get_magnet_info(self.ui.cb_predefined.currentText())

        if m is not None:
            m.pop('name')
            self.ui.ta_additional.setRowCount(0)
            self.ui.la_predefined_description.setText(m.pop('description'))
            self.ui.le_gap.setText(str(m.pop('gap[mm]')))
            self.ui.le_control_gap.setText(str(m.pop('control_gap[mm]')))
            self.ui.le_magnet_length.setText(str(m.pop('magnet_length[mm]')))

            if 'nr_turns_main' in m.keys():
                self.ui.le_main_turns.setText(str(m.pop('nr_turns_main')))
                self.ui.cb_main.setChecked(True)
            else:
                self.ui.le_main_turns.setText('')
                self.ui.cb_main.setChecked(False)

            if 'nr_turns_trim' in m.keys():
                self.ui.le_trim_current.setText('0')
                self.ui.le_trim_turns.setText(str(m.pop('nr_turns_trim')))
                self.ui.cb_trim.setChecked(True)
            else:
                self.ui.le_trim_current.setText('')
                self.ui.le_trim_turns.setText('')
                self.ui.cb_trim.setChecked(False)

            if 'nr_turns_ch' in m.keys():
                self.ui.le_ch_current.setText('0')
                self.ui.le_ch_turns.setText(str(m.pop('nr_turns_ch')))
                self.ui.cb_ch.setChecked(True)
            else:
                self.ui.le_ch_current.setText('')
                self.ui.le_ch_turns.setText('')
                self.ui.cb_ch.setChecked(False)

            if 'nr_turns_cv' in m.keys():
                self.ui.le_cv_current.setText('0')
                self.ui.le_cv_turns.setText(str(m.pop('nr_turns_cv')))
                self.ui.cb_cv.setChecked(True)
            else:
                self.ui.le_cv_current.setText('')
                self.ui.le_cv_turns.setText('')
                self.ui.cb_cv.setChecked(False)

            if 'nr_turns_qs' in m.keys():
                self.ui.le_qs_current.setText('0')
                self.ui.le_qs_turns.setText(str(m.pop('nr_turns_qs')))
                self.ui.cb_qs.setChecked(True)
            else:
                self.ui.le_qs_current.setText('')
                self.ui.le_qs_turns.setText('')
                self.ui.cb_qs.setChecked(False)

            if len(m) != 0:
                count = 0
                for parameter, value in m.items():
                    self.ui.ta_additional.setRowCount(count+1)
                    self.ui.ta_additional.setItem(
                        count, 0, QtGui.QTableWidgetItem(str(parameter)))
                    self.ui.ta_additional.setItem(
                        count, 1, QtGui.QTableWidgetItem(str(value)))
                    count = count + 1

    def save_measurement(self):
        """Save measurement."""
        if self.measurement is not None:
            info = []

            datetime = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
            date = datetime.split('_')[0]
            magnet_name = self.ui.le_magnet_name.text()
            magnet_length = self.ui.le_magnet_length.text()
            gap = self.ui.le_gap.text()
            control_gap = self.ui.le_control_gap.text()

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

            for i in range(self.ui.ta_additional.rowCount()):
                parameter = self.ui.ta_additional.item(i, 0).text()
                value = self.ui.ta_additional.item(i, 1).text()
                if len(value) != 0:
                    info.append([parameter.replace(" ", ""), value])

            info.append(['center_pos_z[mm]', '0'])
            info.append(['center_pos_x[mm]', '0'])
            info.append(['rotation[deg]', '0'])

            ref_pos_x = self.ui.sb_ref_pos_x.value()
            ref_pos_y = self.ui.sb_ref_pos_y.value()
            ref_pos_z = self.ui.sb_ref_pos_z.value()
            ref_pos = [ref_pos_x, ref_pos_y, ref_pos_z]

            self.measurement.save(
                fieldmap_info=info,
                reference_position=ref_pos)


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
                'volt_x.log', self.config.control_voltx_addr)

            self.volty = DigitalMultimeter(
                'volt_y.log', self.config.control_volty_addr)

            self.voltz = DigitalMultimeter(
                'volt_z.log', self.config.control_voltz_addr)

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


thread = GUIThread()
