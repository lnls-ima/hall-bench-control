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

        self.cconf = None
        self.mconf = None
        self.devices = None

        self.tx = None
        self.ty = None
        self.tz = None

        self.graph_curve_x = np.array([])
        self.graph_curve_y = np.array([])
        self.graph_curve_z = np.array([])

        self.calibration = calibration.CalibrationData()

        self.current_measurement = measurement.Measurement()
        self.current_measurement_list = None
        self.current_measurement_dict = None

        self.nr_measurements = 1
        self.savedir = os.path.join(sys.path[1], 'Data')  # Rever!!!

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
                if self.cconf is None:
                    self.cconf = configuration.ControlConfiguration(filename)
                else:
                    self.cconf.read_configuration_from_file(filename)
            except configuration.ConfigurationFileError as e:
                QtGui.QMessageBox.critical(
                    self, 'Failure', e.message, QtGui.QMessageBox.Ignore)
                return

            self.ui.le_filenameconfig.setText(filename)

            self.ui.cb_PMAC_enable.setChecked(self.cconf.control_pmac_enable)

            self.ui.cb_DMM_X.setChecked(self.cconf.control_voltx_enable)
            self.ui.sb_DMM_X_address.setValue(self.cconf.control_voltx_addr)

            self.ui.cb_DMM_Y.setChecked(self.cconf.control_volty_enable)
            self.ui.sb_DMM_Y_address.setValue(self.cconf.control_volty_addr)

            self.ui.cb_DMM_Z.setChecked(self.cconf.control_voltz_enable)
            self.ui.sb_DMM_Z_address.setValue(self.cconf.control_voltz_addr)

            self.ui.cb_Multichannel_enable.setChecked(
                self.cconf.control_multich_enable)

            self.ui.sb_Multichannel_address.setValue(
                self.cconf.control_multich_addr)

            self.ui.cb_Autocolimator_enable.setChecked(
                self.cconf.control_colimator_enable)

            self.ui.cb_Autocolimator_port.setCurrentIndex(
                self.cconf.control_colimator_addr)

    def save_control_configuration_file(self):
        """Save devices parameters to file."""
        filename = QtGui.QFileDialog.getSaveFileName(
            self, 'Save control configuration file')

        if len(filename) != 0:
            if self._update_control_configuration():
                try:
                    self.cconf.save_configuration_to_file(filename)
                except configuration.ConfigurationFileError as e:
                    QtGui.QMessageBox.critical(
                        self, 'Failure', e.message, QtGui.QMessageBox.Ignore)

    def _update_control_configuration(self):
        if self.cconf is None:
            self.cconf = configuration.ControlConfiguration()

        self.cconf.control_pmac_enable = self.ui.cb_PMAC_enable.isChecked()

        self.cconf.control_voltx_enable = self.ui.cb_DMM_X.isChecked()
        self.cconf.control_volty_enable = self.ui.cb_DMM_Y.isChecked()
        self.cconf.control_voltz_enable = self.ui.cb_DMM_Z.isChecked()

        multich_enable = self.ui.cb_Multichannel_enable.isChecked()
        colimator_enable = self.ui.cb_Autocolimator_enable.isChecked()
        self.cconf.control_multich_enable = multich_enable
        self.cconf.control_colimator_enable = colimator_enable

        self.cconf.control_voltx_addr = self.ui.sb_DMM_X_address.value()
        self.cconf.control_volty_addr = self.ui.sb_DMM_Y_address.value()
        self.cconf.control_voltz_addr = self.ui.sb_DMM_Z_address.value()

        multich_addr = self.ui.sb_Multichannel_address.value()
        colimator_addr = self.ui.cb_Autocolimator_port.currentIndex()
        self.cconf.control_multich_addr = multich_addr
        self.cconf.control_colimator_addr = colimator_addr

        if self.cconf.valid_configuration():
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
                if self.mconf is None:
                    self.mconf = configuration.MeasurementConfiguration(
                        filename)
                else:
                    self.mconf.read_configuration_from_file(filename)
            except configuration.ConfigurationFileError as e:
                QtGui.QMessageBox.critical(
                    self, 'Failure', e.message, QtGui.QMessageBox.Ignore)
                return

            self.ui.le_filenamemeasurement.setText(filename)

            self.ui.cb_Hall_X_enable.setChecked(self.mconf.meas_probeX)
            self.ui.cb_Hall_Y_enable.setChecked(self.mconf.meas_probeY)
            self.ui.cb_Hall_Z_enable.setChecked(self.mconf.meas_probeZ)

            self.ui.le_DMM_aper.setText(str(self.mconf.meas_aper_ms))
            self.ui.cb_DMM_precision.setCurrentIndex(self.mconf.meas_precision)

            axis_measurement = [1, 2, 3, 5]
            for axis in axis_measurement:
                tmp = getattr(self.ui, 'le_axis' + str(axis) + '_start')
                value = getattr(self.mconf, 'meas_startpos_ax' + str(axis))
                tmp.setText(str(value))

                tmp = getattr(self.ui, 'le_axis' + str(axis) + '_end')
                value = getattr(self.mconf, 'meas_endpos_ax' + str(axis))
                tmp.setText(str(value))

                tmp = getattr(self.ui, 'le_axis' + str(axis) + '_step')
                value = getattr(self.mconf, 'meas_incr_ax' + str(axis))
                tmp.setText(str(value))

                tmp = getattr(self.ui, 'le_axis' + str(axis) + '_vel')
                value = getattr(self.mconf, 'meas_vel_ax' + str(axis))
                tmp.setText(str(value))

    def save_measurement_configuration_file(self):
        """Save measurement parameters to file."""
        filename = QtGui.QFileDialog.getSaveFileName(
            self, 'Save measurement configuration file')

        if len(filename) != 0:
            if self._update_measurement_configuration():
                try:
                    self.mconf.save_configuration_to_file(filename)
                except configuration.ConfigurationFileError as e:
                    QtGui.QMessageBox.critical(
                        self, 'Failure', e.message, QtGui.QMessageBox.Ignore)

    def _update_measurement_configuration(self):
        if self.mconf is None:
            self.mconf = configuration.MeasurementConfiguration()

        self.mconf.meas_probeX = self.ui.cb_Hall_X_enable.isChecked()
        self.mconf.meas_probeY = self.ui.cb_Hall_Y_enable.isChecked()
        self.mconf.meas_probeZ = self.ui.cb_Hall_Z_enable.isChecked()

        self.mconf.meas_precision = self.ui.cb_DMM_precision.currentIndex()
        self.mconf.meas_trig_axis = 1

        self.nr_measurements = self.ui.sb_number_of_measurements.value()

        tmp = self.ui.le_DMM_aper.text()
        if bool(tmp and tmp.strip()):
            self.mconf.meas_aper_ms = float(tmp)

        axis_measurement = [1, 2, 3, 5]
        for axis in axis_measurement:
            tmp = getattr(self.ui, 'le_axis' + str(axis) + '_start').text()
            if bool(tmp and tmp.strip()):
                setattr(self.mconf, 'meas_startpos_ax' + str(axis), float(tmp))
                print('aqui')

            tmp = getattr(self.ui, 'le_axis' + str(axis) + '_end').text()
            if bool(tmp and tmp.strip()):
                setattr(self.mconf, 'meas_endpos_ax' + str(axis), float(tmp))

            tmp = getattr(self.ui, 'le_axis' + str(axis) + '_step').text()
            if bool(tmp and tmp.strip()):
                setattr(self.mconf, 'meas_incr_ax' + str(axis), float(tmp))

            tmp = getattr(self.ui, 'le_axis' + str(axis) + '_vel').text()
            if bool(tmp and tmp.strip()):
                setattr(self.mconf, 'meas_vel_ax' + str(axis), float(tmp))

        if self.mconf.valid_configuration():
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

        self.devices = devices.HallBenchDevices(self.cconf)
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

        if self.cconf.control_pmac_enable:
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

        if self.cconf.control_pmac_enable and self.devices.pmac_connected:
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
        """Stop measurements and store current measurement values."""
        self.stop = True
        if self.devices is not None:
            self._update_current_measurement()

    def configure_and_measure(self):
        """Configure and start measurements."""
        if (self.devices is None or
           not self.devices.pmac_connected or
           not self._update_measurement_configuration()):
            return

        self.stop = False
        self.current_measurement_dict = {}

        self._clear_graph()
        self._set_axes_speed()

        if self.ui.rb_triggering_axis1.isChecked():  # Z axis scan
            extra_mm = 1
            scan_axis = 1
            axis_a = 2
            axis_b = 3

        elif self.ui.rb_triggering_axis2.isChecked():  # Y axis scan
            extra_mm = 0.1
            scan_axis = 2
            axis_a = 1
            axis_b = 3

        elif self.ui.rb_triggering_axis3.isChecked():  # X axis scan
            extra_mm = 0.1
            scan_axis = 3
            axis_a = 1
            axis_b = 2

        (scan_startpos, scan_endpos, scan_incr, scan_velocity, scan_npts,
            scan_poslist) = self._get_axis_parameters(scan_axis)

        aper_displacement = (self.mconf.meas_aper_ms * scan_velocity)

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

                # place axis A in position
                self._move_axis(axis_a, pos_a)

                # place axis B in position
                self._move_axis(axis_b, pos_b)

                # perform measurement
                self._measure_axis(scan_axis, scan_startpos, scan_endpos,
                                   scan_incr, scan_npts, scan_poslist,
                                   extra_mm, aper_displacement)

                name = (self._get_axis_str(axis_a) + '=' + str(pos_a) + '_' +
                        self._get_axis_str(axis_b) + '=' + str(pos_b))

                measurement_list = measurement.MeasurementList().copy(
                    self.current_measurement_list)
                measurement_list.analyse_data()
                measurement_list.save_data(name, self.savedir)

                self.current_measurement_dict.update({name: measurement_list})

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

    def _measure_axis(self, axis, startpos, endpos, incr, npts,
                      poslist, extra_mm, aper_displacement):
        self.current_measurement_list = measurement.MeasurementList(
            axis, poslist, self.calibration)

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
                self._move_axis(axis, startpos - extra_mm)
                self.current_measurement.position = (
                    poslist + aper_displacement/2)
            else:
                self._move_axis(axis, endpos + extra_mm)
                self.current_measurement.position = (
                    poslist - aper_displacement/2)[::-1]

            if self.stop is True:
                break

            self._configure_trigger(axis, startpos, endpos, incr, npts, to_pos)
            self._configure_voltmeters()
            self._start_reading_threads()

            if self.stop is False:
                if to_pos:
                    self._move_axis_and_update_graph(
                        axis, endpos + extra_mm, idx)
                else:
                    self._move_axis_and_update_graph(
                        axis, startpos - extra_mm, idx)

            self.end_measurements = True

            self._stop_trigger()
            self._wait_reading_threads()
            self._update_current_measurement()
            self._kill_reading_threads()

            if self.stop is True:
                break

            if to_pos is True:
                self.current_measurement_list.add_measurement(
                    measurement.Measurement().copy(
                        self.current_measurement))
            else:
                self.current_measurement_list.add_measurement(
                    measurement.Measurement().reverse(
                        self.current_measurement))

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
        lx = len(self.devices.voltx.voltage)
        ly = len(self.devices.volty.voltage)
        lz = len(self.devices.voltz.voltage)

        self.graph_curve_x[n].setData(
            self.current_measurement.position[:lx], self.devices.voltx.voltage)
        self.graph_curve_y[n].setData(
            self.current_measurement.position[:ly], self.devices.volty.voltage)
        self.graph_curve_z[n].setData(
            self.current_measurement.position[:lz], self.devices.voltz.voltage)

    def _plot_all(self, data='average_field'):
        self._clear_graph()

        n = 0
        for key in self.current_measurement_dict.keys():
            self._configure_graph()
            curve = getattr(self.current_measurement_dict[key], data)

            self.graph_curve_x[n].setData(curve.position, curve.hallx)
            self.graph_curve_y[n].setData(curve.position, curve.hally)
            self.graph_curve_z[n].setData(curve.position, curve.hallz)
            n += 1

    def _set_axes_speed(self):
        self.devices.pmac.set_axis_speed(1, self.mconf.meas_vel_ax1)
        self.devices.pmac.set_axis_speed(2, self.mconf.meas_vel_ax2)
        self.devices.pmac.set_axis_speed(3, self.mconf.meas_vel_ax3)
        self.devices.pmac.set_axis_speed(5, self.mconf.meas_vel_ax5)

    def _clear_measurement(self):
        self.current_measurement.clear()
        self.devices.voltx.voltage = np.array([])
        self.devices.volty.voltage = np.array([])
        self.devices.voltz.voltage = np.array([])

    def _update_current_measurement(self):
        self.current_measurement.hallx = self.devices.voltx.voltage
        self.current_measurement.hally = self.devices.volty.voltage
        self.current_measurement.hallz = self.devices.voltz.voltage

    def _update_measurement_number(self, number):
        self.ui.l_n_meas_status.setText('{0:1d}'.format(number))

    def _get_axis_parameters(self, axis):
        startpos = getattr(self.mconf, 'meas_startpos_ax' + str(axis))
        endpos = getattr(self.mconf, 'meas_endpos_ax' + str(axis))
        incr = getattr(self.mconf, 'meas_incr_ax' + str(axis))
        vel = getattr(self.mconf, 'meas_vel_ax' + str(axis))
        npts = int(round((endpos - startpos) / incr, 4) + 1)
        poslist = np.linspace(startpos, endpos, npts)
        return (startpos, endpos, incr, vel, npts, poslist)

    def _get_axis_str(axis):
        if axis == 1:
            return 'Z'
        elif axis == 2:
            return 'Y'
        elif axis == 3:
            return 'X'
        else:
            return ''

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
        self._move_axis(1, self.mconf.meas_startpos_ax1)
        self._move_axis(2, self.mconf.meas_startpos_ax2)
        self._move_axis(3, self.mconf.meas_startpos_ax3)

    def _configure_voltmeters(self):
        if self.mconf.meas_probeX:
            self.devices.voltx.config(
                self.mconf.meas_aper_ms, self.mconf.meas_precision)
        if self.mconf.meas_probeY:
            self.devices.volty.config(
                self.mconf.meas_aper_ms, self.mconf.meas_precision)
        if self.mconf.meas_probeZ:
            self.devices.voltz.config(
                self.mconf.meas_aper_ms, self.mconf.meas_precision)

    def _reset_voltmeters(self):
        self.devices.voltx.reset()
        self.devices.volty.reset()
        self.devices.voltz.reset()

    def _configure_trigger(self, axis, startpos, endpos, incr, npts, to_pos):
        if to_pos:
            self.devices.pmac.set_trigger(axis, startpos, incr, 10, npts, 1)
        else:
            self.devices.pmac.set_trigger(axis, endpos, incr*-1, 10, npts, 1)

    def _stop_trigger(self):
        self.devices.pmac.stop_trigger()

    def _start_reading_threads(self):
        if self.mconf.meas_probeZ:
            self.tz = threading.Thread(
                target=self.devices.voltz.read,
                args=(self.stop, self.end_measurements,
                      self.mconf.meas_precision,))
            self.tz.start()

        if self.mconf.meas_probeY:
            self.ty = threading.Thread(
                target=self.devices.volty.read,
                args=(self.stop, self.end_measurements,
                      self.mconf.meas_precision,))
            self.ty.start()

        if self.mconf.meas_probeX:
            self.tx = threading.Thread(
                target=self.devices.voltx.read,
                args=(self.stop, self.end_measurements,
                      self.mconf.meas_precision,))
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

    # def export_magnet_format(
    #     self,
    #     magnet_name,
    #     shiftx = 0,
    #     shifty = 0,
    #     shiftz = 0):
    #     local_time = time.localtime()
    #     date = time.strftime('%Y-%m-%d',local_time)
    #     datetime = time.strftime('%Y-%m-%d_%H-%M-%S', local_time)
    #
    #     filename = '{0:1s}_{1:1s}.dat'.format(date, magnet_name)
    #     file = open(os.path.join(self.savedir, filename),'w')
    #
    #     file.write('fieldmap_name:    \t{0:1s}\n'.format(magnet_name))
    #     file.write('timestamp:        \t{0:1s}\n'.format(datetime))
    #     file.write('filename:         \t{0:1s}\n'.format(filename))
    #     file.write('nr_magnets:       \t1\n')
    #     file.write('\n')
    #     file.write('magnet_name:      \t{0:1s}\n'.format(magnet_name))
    #     file.write('gap[mm]:          \t0\n')
    #     file.write('control_gap:      \t--\n')
    #     file.write('magnet_length[mm]:\t0\n')
    #     file.write('current_main[A]:  \t0\n')
    #     file.write('NI_main[A.esp]:   \t0\n')
    #     file.write('center_pos_z[mm]: \t0\n')
    #     file.write('center_pos_x[mm]: \t0\n')
    #     file.write('rotation[deg]:    \t0\n')
    #     file.write('\n')
    #     file.write('X[mm]\tY[mm]\tZ[mm]\tBx\tBy\tBz [T]\n')
    #     file.write('---------------------------------------------------------------------------------------------\n')
    #
    #     for i in range(npts_ax1):
    #         for pos_ax2 in self.list_meas_ax2:
    #             for pos_ax3 in self.list_meas_ax3:
    #                 dictname = 'Y=' + str(pos_ax2) + '_X=' + str(pos_ax3)
    #                 file.write('{0:0.3f}\t{1:0.3f}\t{2:0.3f}\t{3:0.10e}\t{4:0.10e}\t{5:0.10e}\n'.format(
    #                     pos_ax3-shiftx,
    #                     pos_ax2-shifty,
    #                     self.measurements[dictname].average_Bfield.position[i]-shiftz,
    #                     self.measurements[dictname].average_Bfield.hallx[i],
    #                     self.measurements[dictname].average_Bfield.hally[i],
    #                     self.measurements[dictname].average_Bfield.hallz[i]))
    #     file.close()


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
