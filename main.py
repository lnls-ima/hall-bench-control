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
from HallBench import calibration
from HallBench import configuration
from HallBench import devices
from HallBench import measurement


class HallBenchGUI(QtGui.QWidget):
    """Hall Bench Graphical User Interface."""

    def __init__(self, parent=None):
        """Initialize the graphical interface."""
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_HallBench()
        self.ui.setupUi(self)

        self.ui.cb_selectaxis.setCurrentIndex(-1)
        for idx in range(1, self.ui.tabWidget.count()):
            self.ui.tabWidget.setTabEnabled(idx, False)
        self.ui.tb_Motors_main.setItemEnabled(1, False)

        self._initialize_variables()
        self._connect_widgets()
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

        self.calibration = calibration.CalibrationData()

        self.current_postion_list = []
        self.current_line_scan = None
        self.current_measurement = None

        self.nr_measurements = 1
        self.dirpath = os.path.join(sys.path[0], 'Data')

        self.end_measurements = False
        self.stop = False

    def _connect_widgets(self):
        """Make the connections between signals and slots."""
        # load and save device parameters
        self.ui.pb_loadconfigfile.clicked.connect(
            self.load_control_configuration_file)

        self.ui.pb_saveconfigfile.clicked.connect(
            self.save_control_configuration_file)

        # connect devices
        self.ui.pb_connectdevices.clicked.connect(self.connect_devices)

        # activate bench
        self.ui.pb_activatebench.clicked.connect(self.activate_bench)

        # select axis
        self.ui.cb_selectaxis.currentIndexChanged.connect(self.axis_selection)

        # start homming of selected axis
        self.ui.pb_starthomming.clicked.connect(self.start_homming)

        # move to target
        self.ui.pb_movetotarget.clicked.connect(self.move_to_target)

        # stop motor
        self.ui.pb_stopmotor.clicked.connect(self.stop_axis)

        # stop all motors
        self.ui.pb_stopallmotors.clicked.connect(self.stop_all_axis)

        # kill all motors
        self.ui.pb_killallmotors.clicked.connect(self.kill_all_axis)

        # load and save measurements parameters
        self.ui.pb_loadmeasurementfile.clicked.connect(
            self.load_measurement_configuration_file)

        self.ui.pb_savemeasurementfile.clicked.connect(
            self.save_measurement_configuration_file)

        # check limits
        self.ui.le_velocity.editingFinished.connect(
            lambda: self._check_value(self.ui.le_velocity, 0, 150))
        self.ui.le_targetposition.editingFinished.connect(
            lambda: self._check_value(self.ui.le_targetposition, -3600, 3600))

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

        self.ui.le_axis1_vel.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis1_vel, 0.1, 150))
        self.ui.le_axis2_vel.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis2_vel, 0.1, 5))
        self.ui.le_axis3_vel.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis3_vel, 0.1, 5))
        self.ui.le_axis5_vel.editingFinished.connect(
            lambda: self._check_value(self.ui.le_axis5_vel, 0.1, 10))

        # Configure and start measurements
        self.ui.pb_configure_measurement.clicked.connect(
            self.configure_and_measure)

        self.ui.pb_stop_measurements.clicked.connect(self.stop_measurements)

    def _check_value(self, obj, limit_min, limit_max):
        try:
            val = float(obj.text())
            if val >= limit_min and val <= limit_max:
                obj.setText('{0:0.4f}'.format(val))
            else:
                self.axis_selection()
        except Exception:
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

    def load_control_configuration_file(self):
        """Load configuration data to set devices parameters."""
        filename = QtGui.QFileDialog.getOpenFileName(
            self, 'Open control configuration file')

        if len(filename) != 0:
            try:
                if self.cconfig is None:
                    self.cconfig = configuration.ControlConfiguration(filename)
                else:
                    self.cconfig.read_file(filename)
            except configuration.ConfigurationFileError as e:
                QtGui.QMessageBox.critical(
                    self, 'Failure', e.message, QtGui.QMessageBox.Ignore)
                return

            self.ui.le_filenameconfig.setText(filename)

            self.ui.cb_PMAC_enable.setChecked(self.cconfig.control_pmac_enable)

            self.ui.cb_DMM_X.setChecked(self.cconfig.control_voltx_enable)
            self.ui.sb_DMM_X_address.setValue(self.cconfig.control_voltx_addr)

            self.ui.cb_DMM_Y.setChecked(self.cconfig.control_volty_enable)
            self.ui.sb_DMM_Y_address.setValue(self.cconfig.control_volty_addr)

            self.ui.cb_DMM_Z.setChecked(self.cconfig.control_voltz_enable)
            self.ui.sb_DMM_Z_address.setValue(self.cconfig.control_voltz_addr)

            self.ui.cb_Multichannel_enable.setChecked(
                self.cconfig.control_multich_enable)

            self.ui.sb_Multichannel_address.setValue(
                self.cconfig.control_multich_addr)

            self.ui.cb_Autocolimator_enable.setChecked(
                self.cconfig.control_colimator_enable)

            self.ui.cb_Autocolimator_port.setCurrentIndex(
                self.cconfig.control_colimator_addr)

    def save_control_configuration_file(self):
        """Save devices parameters to file."""
        filename = QtGui.QFileDialog.getSaveFileName(
            self, 'Save control configuration file')

        if len(filename) != 0:
            if self._update_control_configuration():
                try:
                    self.cconfig.save_file(filename)
                except configuration.ConfigurationFileError as e:
                    QtGui.QMessageBox.critical(
                        self, 'Failure', e.message, QtGui.QMessageBox.Ignore)

    def _update_control_configuration(self):
        if self.cconfig is None:
            self.cconfig = configuration.ControlConfiguration()

        self.cconfig.control_pmac_enable = self.ui.cb_PMAC_enable.isChecked()

        self.cconfig.control_voltx_enable = self.ui.cb_DMM_X.isChecked()
        self.cconfig.control_volty_enable = self.ui.cb_DMM_Y.isChecked()
        self.cconfig.control_voltz_enable = self.ui.cb_DMM_Z.isChecked()

        multich_enable = self.ui.cb_Multichannel_enable.isChecked()
        colimator_enable = self.ui.cb_Autocolimator_enable.isChecked()
        self.cconfig.control_multich_enable = multich_enable
        self.cconfig.control_colimator_enable = colimator_enable

        self.cconfig.control_voltx_addr = self.ui.sb_DMM_X_address.value()
        self.cconfig.control_volty_addr = self.ui.sb_DMM_Y_address.value()
        self.cconfig.control_voltz_addr = self.ui.sb_DMM_Z_address.value()

        multich_addr = self.ui.sb_Multichannel_address.value()
        colimator_addr = self.ui.cb_Autocolimator_port.currentIndex()
        self.cconfig.control_multich_addr = multich_addr
        self.cconfig.control_colimator_addr = colimator_addr

        if self.cconfig.valid_configuration():
            return True
        else:
            message = 'Invalid control configuration'
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
                    self.mconfig = configuration.MeasurementConfiguration(
                        filename)
                else:
                    self.mconfig.read_file(filename)
            except configuration.ConfigurationFileError as e:
                QtGui.QMessageBox.critical(
                    self, 'Failure', e.message, QtGui.QMessageBox.Ignore)
                return

            self.ui.le_filenamemeasurement.setText(filename)

            self.ui.cb_Hall_X_enable.setChecked(self.mconfig.meas_probeX)
            self.ui.cb_Hall_Y_enable.setChecked(self.mconfig.meas_probeY)
            self.ui.cb_Hall_Z_enable.setChecked(self.mconfig.meas_probeZ)

            self.ui.le_DMM_aper.setText(str(self.mconfig.meas_aper_ms))
            self.ui.cb_DMM_precision.setCurrentIndex(
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
            self.mconfig = configuration.MeasurementConfiguration()

        self.mconfig.meas_probeX = self.ui.cb_Hall_X_enable.isChecked()
        self.mconfig.meas_probeY = self.ui.cb_Hall_Y_enable.isChecked()
        self.mconfig.meas_probeZ = self.ui.cb_Hall_Z_enable.isChecked()

        self.mconfig.meas_precision = self.ui.cb_DMM_precision.currentIndex()
        self.mconfig.meas_trig_axis = 1

        self.nr_measurements = self.ui.sb_number_of_measurements.value()

        tmp = self.ui.le_DMM_aper.text()
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
        if not self._update_control_configuration():
            return

        self.devices = devices.HallBenchDevices(self.cconfig)
        self.devices.load()
        self.devices.connect()

        not_connected = sorted(
            [k for k, v in self.devices.check_connection().items() if not v])
        if len(not_connected) != 0:
            message = ('The following devices are not connected: \n\n' +
                       '\n'.join(not_connected))
            QtGui.QMessageBox.warning(
                self, 'Warning', message, QtGui.QMessageBox.Ok)
        else:
            message = 'Devices successfully connected.'
            QtGui.QMessageBox.information(
                self, 'Information', message, QtGui.QMessageBox.Ok)

        if self.cconfig.control_pmac_enable:
            if self.devices.pmac_connected:
                self.ui.tabWidget.setTabEnabled(1, True)
                self.ui.tabWidget.setTabEnabled(2, True)

                # check if all axis are hommed and release access to movement.
                list_of_axis = self.devices.pmac.commands.list_of_axis
                status = []
                for axis in list_of_axis:
                    status.append(
                        (self.devices.pmac.axis_status(axis) & 1024) != 0)

                if all(status):
                    self.ui.tb_Motors_main.setItemEnabled(1, True)
                    self.ui.tb_Motors_main.setCurrentIndex(1)

        self.activate_bench()

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
        tmp = self.ui.cb_selectaxis.currentText()
        if tmp == '':
            self.selected_axis = -1
        else:
            self.selected_axis = int(tmp[1])

            # set target to zero
            self.ui.le_targetposition.setText('{0:0.4f}'.format(0))

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
                self.ui.tb_Motors_main.setItemEnabled(1, True)
                self.ui.tb_Motors_main.setCurrentIndex(1)

    def move_to_target(self):
        """Move Hall probe to target position."""
        if self.devices is None:
            return

        # if any available axis is selected:
        if not self.selected_axis == -1:
            set_vel = float(self.ui.le_velocity.text())
            target = float(self.ui.le_targetposition.text())
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

        self.stop = False

        self._clear_graph()
        self._set_axes_speed()

        self.current_measurement = measurement.Measurement(
            self.cconfig, self.mconfig, self.calibration, self.dirpath)

        if self.ui.rb_triggering_axis1.isChecked():
            extra_mm = 1
            scan_axis = 1
            axis_a = 2
            axis_b = 3

        elif self.ui.rb_triggering_axis2.isChecked():
            extra_mm = 0.1
            scan_axis = 2
            axis_a = 1
            axis_b = 3

        elif self.ui.rb_triggering_axis3.isChecked():
            extra_mm = 0.1
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
            posx, posy, posz, self.cconfig, self.mconfig,
            self.calibration, self.dirpath)

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
        self.ui.l_n_meas_status.setText('{0:1d}'.format(number))

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
            self.ui.graphicsView_1.plotItem.plot(
                np.array([]),
                np.array([]),
                pen=(255, 0, 0),
                symbol='o',
                symbolPen=(255, 0, 0),
                symbolSize=4))

        self.graph_curve_y = np.append(
            self.graph_curve_y,
            self.ui.graphicsView_1.plotItem.plot(
                np.array([]),
                np.array([]),
                pen=(0, 255, 0),
                symbol='o',
                symbolPen=(0, 255, 0),
                symbolSize=4))

        self.graph_curve_z = np.append(
            self.graph_curve_z,
            self.ui.graphicsView_1.plotItem.plot(
                np.array([]),
                np.array([]),
                pen=(0, 0, 255),
                symbol='o',
                symbolPen=(0, 0, 255),
                symbolSize=4))

    def _clear_graph(self):
        self.ui.graphicsView_1.plotItem.curves.clear()
        self.ui.graphicsView_1.clear()
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
        for ls in self.current_measurement.data_list():
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
