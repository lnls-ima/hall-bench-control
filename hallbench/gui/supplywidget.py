# -*- coding: utf-8 -*-

"""Power Supply widget for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import time as _time
import traceback as _traceback
import PyQt5.uic as _uic
from PyQt5.QtWidgets import (
    QWidget as _QWidget,
    QMessageBox as _QMessageBox,
    QApplication as _QApplication,
    QTableWidgetItem as _QTableWidgetItem,
    )
from PyQt5.QtCore import (
    QTimer as _QTimer,
    pyqtSignal as _pyqtSignal,
    )

from hallbench.gui import utils as _utils


class SupplyWidget(_QWidget):
    """Power Supply widget class for the Hall Bench Control application."""

    start_measurement = _pyqtSignal([bool])
    current_ramp_end = _pyqtSignal([bool])
    current_setpoint_changed = _pyqtSignal([float])

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)

        # setup the ui
        uifile = _utils.getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialization
        self.current_array_index = 0
        self.config = self.power_supply_config
        self.drs = self.devices.ps
        self.timer = _QTimer()
        self.plot_dialog = _utils.PlotDialog()

        # fill combobox
        self.list_powersupply()

        # create signal/slot connections
        self.ui.pb_ps_button.clicked.connect(self.start_powersupply)
        self.ui.pb_refresh.clicked.connect(self.display_current)
        self.ui.pb_load_ps.clicked.connect(self.load_powersupply)
        self.ui.pb_save_ps.clicked.connect(self.save_powersupply)
        self.ui.pb_send.clicked.connect(self.send_setpoint)
        self.ui.pb_send_curve.clicked.connect(self.send_curve)
        self.ui.pb_config_pid.clicked.connect(self.config_pid)
        self.ui.pb_reset_inter.clicked.connect(self.reset_interlocks)
        self.ui.pb_cycle.clicked.connect(self.cycling_ps)
        self.ui.pb_plot.clicked.connect(self.plot)
        self.ui.pb_config_ps.clicked.connect(self.config_ps)
        self.ui.pb_add_row.clicked.connect(self.add_row)
        self.ui.pb_remove_row.clicked.connect(self.remove_row)
        self.ui.pb_clear_table.clicked.connect(self.clear_table)
        self.ui.cb_ps_name.currentIndexChanged.connect(self.change_ps)
        self.ui.cb_ps_name.editTextChanged.connect(self.change_ps)
        self.ui.sb_current_setpoint.valueChanged.connect(self.check_setpoint)
        self.timer.timeout.connect(self.status_powersupply)

    @property
    def connection_config(self):
        """Return the connection configuration."""
        return _QApplication.instance().connection_config

    @property
    def devices(self):
        """Hall Bench Devices."""
        return _QApplication.instance().devices

    @property
    def database(self):
        """Database filename."""
        return _QApplication.instance().database

    @property
    def power_supply_config(self):
        """Power Supply configurations."""
        return _QApplication.instance().power_supply_config

    def closeDialogs(self):
        """Close dialogs."""
        try:
            self.plot_dialog.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.turn_off()
            self.closeDialogs()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def turn_off(self):
        """Turn off the Power Supply if it's turned on."""
        if self.config.status:
            self.start_powersupply()

    def list_powersupply(self):
        """Updates available power supply supply names."""
        _l = self.config.get_table_column(self.database, 'name')
        for _ in range(self.ui.cb_ps_name.count()):
            self.ui.cb_ps_name.removeItem(0)
        self.ui.cb_ps_name.addItems(_l)

    def set_address(self, address):
        if self.drs.ser.is_open:
            self.drs.SetSlaveAdd(address)
            return True
        else:
            _QMessageBox.warning(self, 'Warning',
                                 'Power Supply serial port is closed.',
                                 _QMessageBox.Ok)
            return False

    def set_op_mode(self, mode=0):
        self.drs.OpMode(mode)
        _time.sleep(0.1)
        if self.drs.Read_ps_OpMode() == mode:
            return True
        return False

    def change_ps(self):
        """Sets the Load Power Supply button disabled if the selected supply is
           already loaded."""
        if self.ui.cb_ps_name.currentText() == self.config.ps_name:
            self.ui.pb_load_ps.setEnabled(False)
        else:
            self.ui.pb_load_ps.setEnabled(True)

    def start_powersupply(self):
        """Starts/Stops the Power Supply."""
        try:
            self.devices.dcct.config()

            if self.config.ps_type is None:
                _QMessageBox.warning(self, 'Warning',
                                     'Please configure the power supply and '
                                     'try again.', _QMessageBox.Ok)
                if self.config.status is False:
                    self.change_ps_button(True)
                else:
                    self.change_ps_button(False)
                return

            self.ui.pb_ps_button.setEnabled(False)
            self.ui.pb_ps_button.setText('Processing...')
            self.ui.tabWidget_2.setEnabled(False)
            self.ui.pb_send.setEnabled(False)
            _QApplication.processEvents()

            _ps_type = self.config.ps_type
            if not self.set_address(_ps_type):
                if self.config.status is False:
                    self.change_ps_button(True)
                else:
                    self.change_ps_button(False)
                return

            # Status ps is OFF
            if self.config.status is False:
                try:
                    self.drs.Read_iLoad1()
                except Exception:
                    _traceback.print_exc(file=_sys.stdout)
                    _QMessageBox.warning(self, 'Warning',
                                         'Could not read the digital current.',
                                         _QMessageBox.Ok)
                    self.change_ps_button(True)
                    return

                _status_interlocks = self.drs.Read_ps_SoftInterlocks()
                if _status_interlocks != 0:
                    self.ui.pb_interlock.setEnabled(True)
                    self.config.status_interlock = True
                    _QMessageBox.warning(self, 'Warning',
                                         'Software Interlock active!',
                                         _QMessageBox.Ok)
                    self.change_ps_button(True)
                    return
                _status_interlocks = self.drs.Read_ps_HardInterlocks()
                if _status_interlocks != 0:
                    self.ui.pb_interlock.setEnabled(True)
                    self.config.status_interlock = True
                    _QMessageBox.warning(self, 'Warning',
                                         'Hardware Interlock active!',
                                         _QMessageBox.Ok)
                    self.change_ps_button(True)
                    return
                self.config.status_interlock = False

                # PS 1000 A needs to turn dc link on
                if _ps_type == 2:
                    self.drs.SetSlaveAdd(_ps_type-1)
                    # Turn ON ps DClink
                    try:
                        self.drs.TurnOn()
                        _time.sleep(1.2)
                        if self.drs.Read_ps_OnOff() != 1:
                            _QMessageBox.warning(self, 'Warning',
                                                 'Power Supply Capacitor '
                                                 'Bank did not initialize.',
                                                 _QMessageBox.Ok)
                            self.change_ps_button(True)
                            return
                    except Exception:
                        _traceback.print_exc(file=_sys.stdout)
                        _QMessageBox.warning(self, 'Warning',
                                             'Power Supply Capacitor '
                                             'Bank did not initialize.',
                                             _QMessageBox.Ok)
                        self.change_ps_button(True)
                        return
                    # Closing DC link Loop
                    try:
                        # Closed Loop
                        self.drs.ClosedLoop()
                        _time.sleep(1)
                        if self.drs.Read_ps_OpenLoop() == 1:
                            _QMessageBox.warning(self, 'Warning',
                                                 'Power Supply circuit '
                                                 'loop is not closed.',
                                                 _QMessageBox.Ok)
                            self.config.status_loop = False
                            self.change_ps_button(True)
                            return
                    except Exception:
                        _traceback.print_exc(file=_sys.stdout)
                        _QMessageBox.warning(self, 'Warning',
                                             'Power Supply circuit '
                                             'loop is not closed.',
                                             _QMessageBox.Ok)
                        self.config.status_loop = False
                        self.change_ps_button(True)
                        return
                    # Set ISlowRef for DC Link (Capacitor Bank)
                    # Operation mode selection for Slowref
                    if not self.set_op_mode(0):
                        _QMessageBox.warning(self, 'Warning',
                                             'Could not set the slowRef '
                                             'operation mode.',
                                             _QMessageBox.Ok)
                        self.change_ps_button(True)
                        return

                    _dclink_value = self.config.dclink
                    # Set 90 V for Capacitor Bank (default value)
                    self.drs.SetISlowRef(_dclink_value)
                    _time.sleep(1)
                    _feedback_DCLink = round(self.drs.Read_vOutMod1()/2 +
                                             self.drs.Read_vOutMod2()/2, 3)
                    _i = 100
                    while _feedback_DCLink < _dclink_value and _i > 0:
                        _feedback_DCLink = round(self.drs.Read_vOutMod1()/2 +
                                                 self.drs.Read_vOutMod2()/2, 3)
                        _QApplication.processEvents()
                        _time.sleep(0.5)
                        _i = _i-1
                    if _i == 0:
                        _QMessageBox.warning(self, 'Warning', 'DC link '
                                             'setpoint is not set.\n'
                                             'Check the configurations.',
                                             _QMessageBox.Ok)
                        self.drs.TurnOff()
                        self.change_ps_button(True)
                        return
                # Turn on Power Supply
                self.drs.SetSlaveAdd(_ps_type)
                if _ps_type < 4:
                    self.pid_setting()
                self.drs.TurnOn()
                _time.sleep(0.1)
                if _ps_type == 2:
                    _time.sleep(0.9)
                if not self.drs.Read_ps_OnOff():
                    # Turn off DC link
                    self.drs.SetSlaveAdd(_ps_type-1)
                    self.drs.TurnOff()
                    self.change_ps_button(True)
                    _QMessageBox.warning(self, 'Warning', 'The Power Supply '
                                         'did not turn off.',
                                         _QMessageBox.Ok)
                    return
                # Closed Loop
                self.drs.ClosedLoop()
                _time.sleep(0.1)
                if _ps_type == 2:
                    _time.sleep(0.9)
                if self.drs.Read_ps_OpenLoop() == 1:
                    # Turn off DC link
                    self.drs.SetSlaveAdd(_ps_type-1)
                    self.drs.TurnOff()
                    self.change_ps_button(True)
                    _QMessageBox.warning(self, 'Warning', 'Power Supply '
                                         'circuit loop is not closed.',
                                         _QMessageBox.Ok)
                    return
                self.change_ps_button(False)
                self.config.status = True
                self.config.status_loop = True
                self.config.main_current = 0
                self.ui.le_status_loop.setText('Closed')
                self.ui.tabWidget_2.setEnabled(True)
                self.ui.pb_send.setEnabled(True)
                self.ui.tabWidget_3.setEnabled(True)
                self.ui.pb_refresh.setEnabled(True)
                self.ui.pb_send.setEnabled(True)
                self.ui.pb_send_curve.setEnabled(True)
                self.timer.start(30000)
                _QMessageBox.information(self, 'Information', 'The Power '
                                         'Supply started successfully.',
                                         _QMessageBox.Ok)
            else:
                self.drs.SetSlaveAdd(_ps_type)
                self.drs.TurnOff()
                _time.sleep(0.1)
                if _ps_type == 2:
                    _time.sleep(0.9)
                _status = self.drs.Read_ps_OnOff()
                if _status:
                    _QMessageBox.warning(self, 'Warning', 'Could not turn the '
                                         'power supply off.\nPlease try '
                                         'again.', _QMessageBox.Ok)
                    self.change_ps_button(False)
                    return
                if _ps_type == 2:
                    # Turn off DC link
                    self.drs.SetSlaveAdd(_ps_type-1)
                    self.drs.TurnOff()
                    _time.sleep(0.1)
                    if _ps_type == 2:
                        _time.sleep(0.9)
                    _status = self.drs.Read_ps_OnOff()
                    if _status:
                        _QMessageBox.warning(self, 'Warning', 'Could not turn '
                                             'the power supply off.\nPlease '
                                             'try again.', _QMessageBox.Ok)
                        self.change_ps_button(False)
                        return
                self.config.status = False
                self.config.status_loop = False
                self.config.main_current = 0
                self.ui.le_status_loop.setText('Open')
                self.ui.pb_send.setEnabled(False)
                self.ui.pb_cycle.setEnabled(False)
                self.ui.pb_send_curve.setEnabled(False)
                self.change_ps_button(True)
                self.timer.stop()
                _QMessageBox.information(self, 'Information',
                                         'Power supply was turned off.',
                                         _QMessageBox.Ok)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.warning(self, 'Warning', 'Failed to change the power '
                                 'supply state.', _QMessageBox.Ok)
            self.change_ps_button(False)
            return

    def change_ps_button(self, is_off=True):
        """Updates ui when turning power supply on/off.

        Args:
            is_off (bool): True when the power supply is turned off;
                False if turned on"""
        self.ui.pb_ps_button.setEnabled(True)
        if is_off:
            self.ui.pb_ps_button.setChecked(False)
            self.ui.pb_ps_button.setText('Turn ON')
        else:
            self.ui.pb_ps_button.setChecked(True)
            self.ui.pb_ps_button.setText('Turn OFF')
        self.ui.tabWidget_2.setEnabled(True)
        self.ui.pb_send.setEnabled(True)
        _QApplication.processEvents()

    def config_ps(self):
        """Sets Power Supply configurations according to current UI inputs."""

        self.config.ps_name = self.ui.cb_ps_name.currentText()
        self.config.ps_type = self.ui.cb_ps_type.currentIndex() + 2
        self.config.dclink = self.ui.sb_dclink.value()
        self.config.ps_setpoint = self.ui.sb_current_setpoint.value()
        self.config.maximum_current = float(self.ui.le_maximum_current.text())
        self.config.minimum_current = float(self.ui.le_minimum_current.text())
        dcct_head_str = self.ui.cb_dcct_select.currentText().replace(' A', '')
        try:
            self.config.dcct_head = int(dcct_head_str)
        except Exception:
            self.config.dcct_head = None
        self.config.Kp = self.ui.sb_kp.value()
        self.config.Ki = self.ui.sb_ki.value()
        self.config.current_array = self.table_to_array()
        self.config.sinusoidal_amplitude = float(
            self.ui.le_sinusoidal_amplitude.text())
        self.config.sinusoidal_offset = float(
            self.ui.le_sinusoidal_offset.text())
        self.config.sinusoidal_frequency = float(
            self.ui.le_sinusoidal_frequency.text())
        self.config.sinusoidal_ncycles = int(
            self.ui.le_sinusoidal_ncycles.text())
        self.config.sinusoidal_phasei = float(
            self.ui.le_sinusoidal_phase.text())
        self.config.sinusoidal_phasef = float(
            self.ui.le_sinusoidal_phasef.text())
        self.config.dsinusoidal_amplitude = float(
            self.ui.le_damp_sin_ampl.text())
        self.config.dsinusoidal_offset = float(
            self.ui.le_damp_sin_offset.text())
        self.config.dsinusoidal_frequency = float(
            self.ui.le_damp_sin_freq.text())
        self.config.dsinusoidal_ncycles = int(
            self.ui.le_damp_sin_ncycles.text())
        self.config.dsinusoidal_phasei = float(
            self.ui.le_damp_sin_phase.text())
        self.config.dsinusoidal_phasef = float(
            self.ui.le_damp_sin_phasef.text())
        self.config.dsinusoidal_damp = float(
            self.ui.le_damp_sin_damping.text())

    def config_widget(self):
        """Sets current configuration variables into the widget."""

        self.ui.cb_ps_name.setCurrentText(self.config.ps_name)
        self.ui.cb_ps_type.setCurrentIndex(self.config.ps_type - 2)
        self.ui.sb_dclink.setValue(self.config.dclink)
        self.ui.sb_current_setpoint.setValue(self.config.ps_setpoint)
        self.ui.le_maximum_current.setText(str(self.config.maximum_current))
        self.ui.le_minimum_current.setText(str(self.config.minimum_current))
        self.ui.cb_dcct_select.setCurrentText(str(self.config.dcct_head) +
                                              ' A')
        self.ui.sb_kp.setValue(self.config.Kp)
        self.ui.sb_ki.setValue(self.config.Ki)
        self.array_to_table(self.config.current_array)
        self.ui.le_sinusoidal_amplitude.setText(str(
            self.config.sinusoidal_amplitude))
        self.ui.le_sinusoidal_offset.setText(str(
            self.config.sinusoidal_offset))
        self.ui.le_sinusoidal_frequency.setText(str(
            self.config.sinusoidal_frequency))
        self.ui.le_sinusoidal_ncycles.setText(str(
            self.config.sinusoidal_ncycles))
        self.ui.le_sinusoidal_phase.setText(str(self.config.sinusoidal_phasei))
        self.ui.le_sinusoidal_phasef.setText(str(
            self.config.sinusoidal_phasef))
        self.ui.le_damp_sin_ampl.setText(str(
            self.config.dsinusoidal_amplitude))
        self.ui.le_damp_sin_offset.setText(str(self.config.dsinusoidal_offset))
        self.ui.le_damp_sin_freq.setText(str(
            self.config.dsinusoidal_frequency))
        self.ui.le_damp_sin_ncycles.setText(str(
            self.config.dsinusoidal_ncycles))
        self.ui.le_damp_sin_phase.setText(str(self.config.dsinusoidal_phasei))
        self.ui.le_damp_sin_phasef.setText(str(self.config.dsinusoidal_phasef))
        self.ui.le_damp_sin_damping.setText(str(self.config.dsinusoidal_damp))

    def config_pid(self):
        """Configures PID settings."""
        _ans = _QMessageBox.question(self, 'PID settings', 'Be aware that '
                                     'this will overwrite the current '
                                     'configurations.\n Are you sure you want '
                                     'to configure the PID parameters?',
                                     _QMessageBox.Yes | _QMessageBox.No)
        try:
            if _ans == _QMessageBox.Yes:
                _ans = self.pid_setting()
                if _ans:
                    _QMessageBox.information(self, 'Information',
                                             'PID configured.',
                                             _QMessageBox.Ok)
                else:
                    _QMessageBox.warning(self, 'Fail', 'Power Supply PID '
                                         'configuration fault.',
                                         _QMessageBox.Ok)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def pid_setting(self):
        """Set power supply PID configurations."""
        self.config.Kp = self.ui.sb_kp.value()
        self.config.Ki = self.ui.sb_ki.value()
        _ps_type = self.config.ps_type
        if not self.set_address(_ps_type):
            return
        _id_mode = 0
        _elp_PI_dawu = 3
        try:
            # Write ID module from controller
            self.drs.Write_dp_ID(_id_mode)
            # Write DP Class for setting PI
            self.drs.Write_dp_Class(_elp_PI_dawu)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return False
        try:
            _list_coeffs = _np.zeros(16)
            _kp = self.config.Kp
            _ki = self.config.Ki
            _list_coeffs[0] = _kp
            _list_coeffs[1] = _ki
            # Write kp and ki
            self.drs.Write_dp_Coeffs(_list_coeffs.tolist())
            # Configure kp and ki
            self.drs.ConfigDPModule()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return False

        return True

    def emergency(self):
        """Stops power supply current."""
        _ps_type = self.config.ps_type
        if not self.set_address(_ps_type):
            return
        if not self.set_op_mode(0):
            self.set_op_mode(0)
        self.drs.SetISlowRef(0)
        _time.sleep(0.1)
        self.drs.TurnOff()
        _time.sleep(0.1)
        if self.config.ps_type == 2:
            _time.sleep(0.9)
        if self.drs.Read_ps_OnOff() == 0:
            self.config.status = False
            self.config.main_current = 0
            self.ui.pb_ps_button.setChecked(False)
            self.ui.pb_ps_button.setText('Turn ON')
            _QApplication.processEvents()

    def display_current(self):
        """Displays current on interface."""
        if not self.isVisible() or not self.config.update_display:
            return
            
        _ps_type = self.config.ps_type
        try:
            if not self.set_address(_ps_type):
                return
            _refresh_current = round(float(self.drs.Read_iLoad1()), 3)
            self.ui.lcd_ps_reading.display(_refresh_current)
            _QApplication.processEvents()
            if all([self.ui.chb_dcct.isChecked(),
                    self.connection_config.dcct_enable]):
                self.ui.lcd_current_dcct.setEnabled(True)
                self.ui.label_161.setEnabled(True)
                self.ui.label_164.setEnabled(True)
                _current = round(self.devices.dcct.read_current(
                    dcct_head=self.config.dcct_head), 3)
                self.ui.lcd_current_dcct.display(_current)
            _QApplication.processEvents()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.warning(self, 'Warning', 'Could not display Current.',
                                 _QMessageBox.Ok)
            return

    def change_setpoint_and_emit_signal(self):
        """Change current setpoint and emit signal."""
        if self.config.current_array is None:
            self.current_array_index = 0
            self.current_ramp_end.emit(True)
            return

        if self.current_array_index >= len(self.config.current_array):
            self.current_array_index = 0
            self.current_ramp_end.emit(True)
            return

        _setpoint = self.config.current_array[self.current_array_index]
        if self.current_setpoint(setpoint=_setpoint):
            self.current_array_index = self.current_array_index + 1
            self.start_measurement.emit(True)
        else:
            self.current_array_index = 0
            self.current_ramp_end.emit(False)

    def current_setpoint(self, setpoint=0):
        """Changes current setpoint in power supply configuration.

        Args:
            setpoint (float): current setpoint."""
        try:
            self.ui.tabWidget_2.setEnabled(False)
            self.ui.pb_send.setEnabled(False)
            _ps_type = self.config.ps_type
            if not self.set_address(_ps_type):
                return

            # verify current limits
            _setpoint = setpoint
            if not self.verify_current_limits(setpoint):
                self.ui.tabWidget_2.setEnabled(True)
                self.ui.pb_send.setEnabled(True)
                return False

            # send setpoint and wait until current is set
            self.drs.SetISlowRef(_setpoint)
            _time.sleep(0.1)
            for _ in range(30):
                _compare = round(float(self.drs.Read_iLoad1()), 3)
                self.display_current()
                if abs(_compare - _setpoint) <= 0.5:
                    self.ui.tabWidget_2.setEnabled(True)
                    self.ui.pb_send.setEnabled(True)
                    self.config.ps_setpoint = _setpoint
                    self.current_setpoint_changed.emit(_setpoint)
                    return True
                _QApplication.processEvents()
                _time.sleep(1)
            self.ui.tabWidget_2.setEnabled(True)
            self.ui.pb_send.setEnabled(True)
            return False
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.ui.tabWidget_2.setEnabled(True)
            self.ui.pb_send.setEnabled(True)
            return False

    def send_setpoint(self):
        """Sends configured current setpoint to the power supply."""
        try:
            _setpoint = self.ui.sb_current_setpoint.value()
            _ans = self.current_setpoint(_setpoint)
            if _ans:
                _QMessageBox.information(self, 'Information',
                                         'Current properly set.',
                                         _QMessageBox.Ok)
            else:
                _QMessageBox.warning(self, 'Warning',
                                     'Current was not properly set.',
                                     _QMessageBox.Ok)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return

    def check_setpoint(self):
        _setpoint = self.ui.sb_current_setpoint.value()
        _maximum_current = self.config.maximum_current
        _minimum_current = self.config.minimum_current
        if _maximum_current is None:
            return
        if _setpoint > self.config.maximum_current:
            self.ui.sb_current_setpoint.setValue(_maximum_current)
            _QMessageBox.warning(self, 'Warning',
                                 'Current value is too high.',
                                 _QMessageBox.Ok)
        elif _setpoint < self.config.minimum_current:
            self.ui.sb_current_setpoint.setValue(_minimum_current)
            _QMessageBox.warning(self, 'Warning',
                                 'Current value is too low.',
                                 _QMessageBox.Ok)

    def verify_current_limits(self, current, chk_offset=False, offset=0):
        """Check the limits of the current values set.

        Args:
            current (float): current value in [A] to be verified.
            check_offset (bool, optional): flag to check current value plus
                offset; default False.
            offset (float, optional): offset value in [A]; default 0.
        Return:
            True if current is within the limits, False otherwise.
        """
        _current = float(current)
        _offset = float(offset)

        try:
            _current_max = float(self.config.maximum_current)
        except Exception:
            _QMessageBox.warning(self, 'Warning', 'Invalid Maximum Current '
                                 'value.', _QMessageBox.Ok)
            return False
        try:
            _current_min = float(self.config.minimum_current)
        except Exception:
            _QMessageBox.warning(self, 'Warning', 'Invalid Minimum Current '
                                 'value.', _QMessageBox.Ok)
            return False

        if not chk_offset:
            if _current > _current_max:
                self.ui.sb_current_setpoint.setValue(float(_current_max))
                _current = _current_max
                _QMessageBox.warning(self, 'Warning', 'Current value is too '
                                     'high', _QMessageBox.Ok)
                return False
            if _current < _current_min:
                self.ui.sb_current_setpoint.setValue(float(_current_min))
                _current = _current_min
                _QMessageBox.warning(self, 'Warning', 'Current value is too '
                                     'low.', _QMessageBox.Ok)
                return False
        else:
            if any((_current + offset) > _current_max,
                   ((-1*_current + offset) < _current_min)):
                _QMessageBox.warning(self, 'Warning', 'Peak-to-peak current '
                                     'values out of bounds.', _QMessageBox.Ok)
                return False

        return True

    def send_curve(self):
        """UI function to set power supply curve generator."""
        self.ui.tabWidget_2.setEnabled(False)
        if self.curve_gen() is True:
            _QMessageBox.information(self, 'Information',
                                     'Curve sent successfully.',
                                     _QMessageBox.Ok)
            self.ui.tabWidget_2.setEnabled(True)
            self.ui.pb_send.setEnabled(True)
            self.ui.pb_cycle.setEnabled(True)
            _QApplication.processEvents()
        else:
            _QMessageBox.warning(self, 'Warning', 'Failed to send curve.',
                                 _QMessageBox.Ok)
            self.ui.tabWidget_2.setEnabled(True)
            self.ui.pb_send.setEnabled(True)
            _QApplication.processEvents()
            return False

    def curve_gen(self):
        """Configures power supply curve generator."""
        self.config_ps()
        _curve_type = int(self.ui.tabWidget_3.currentIndex())

        _ps_type = self.config.ps_type
        if not self.set_address(_ps_type):
            return

        if _curve_type == 1:    # Sinusoidal
            # For Offset
            try:
                _offset = float(self.config.sinusoidal_offset)
                if not self.verify_current_limits(_offset):
                    self.ui.le_sinusoidal_offset.setText('0')
                    self.config.sinusoidal_offset = 0
                    return False
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please verify the '
                                     'Offset parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Amplitude
            try:
                _amp = float(self.config.sinusoidal_amplitude)
                if not self.verify_current_limits(abs(_amp), True, _offset):
                    self.ui.le_sinusoidal_amplitude.setText('0')
                    self.config.sinusoidal_amplitude = 0
                    return False
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please verify the '
                                     'Amplitude parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Frequency
            try:
                _freq = float(self.config.sinusoidal_frequency)
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please verify the '
                                     'Frequency parameter of the curve.',
                                     _QMessageBox.Ok)
            # For N-cycles
            try:
                _n_cycles = int(self.config.sinusoidal_ncycles)
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please verify the '
                                     '#cycles parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Phase shift
            try:
                _phase_shift = float(self.config.sinusoidal_phasei)
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please verify the '
                                     'Phase parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Final phase
            try:
                _final_phase = float(self.config.sinusoidal_phasef)
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please verify the '
                                     'Final phase parameter of the curve.',
                                     _QMessageBox.Ok)

        # Damped Sinusoidal
        if _curve_type == 0:
            # For Offset
            try:
                _offset = float(self.config.dsinusoidal_offset)
                if not self.verify_current_limits(_offset):
                    self.config.dsinusoidal_offset = 0
                    self.ui.le_damp_sin_offset.setText('0')
                    return False
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please verify the '
                                     'Offset parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Amplitude
            try:
                _amp = float(self.config.dsinusoidal_amplitude)
                if not self.verify_current_limits(abs(_amp), True, _offset):
                    self.config.dsinusoidal_amplitude = 0
                    self.ui.le_damp_sin_ampl.setText('0')
                    return False
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please verify the '
                                     'Amplitude parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Frequency
            try:
                _freq = float(self.config.dsinusoidal_frequency)
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please verify the '
                                     'Frequency parameter of the curve.',
                                     _QMessageBox.Ok)
            # For N-cycles
            try:
                _n_cycles = int(self.config.dsinusoidal_ncycles)
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please verify the '
                                     '#cycles parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Phase shift
            try:
                _phase_shift = float(self.config.dsinusoidal_phasei)
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please verify the '
                                     'Phase parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Final phase
            try:
                _final_phase = float(self.config.dsinusoidal_phasef)
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please verify the '
                                     'Final Phase parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Damping
            try:
                _damping = float(self.config.dsinusoidal_damp)
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please verify the '
                                     'Damping parameter of the curve.',
                                     _QMessageBox.Ok)
        _QApplication.processEvents()

        # Generating curves
        # Sinusoidal
        try:
            if _curve_type == 1:
                _sigType = 0
                self.drs.Write_sigGen_Freq(float(_freq))
                self.drs.Write_sigGen_Amplitude(float(_amp))
                self.drs.Write_sigGen_Offset(float(_offset))
                self.drs.ConfigSigGen(_sigType, _n_cycles, _phase_shift,
                                      _final_phase)

            # Damped Sinusoidal
            if _curve_type == 0:
                _sigType = 4
                self.drs.Write_sigGen_Freq(float(_freq))
                self.drs.Write_sigGen_Amplitude(float(_amp))
                self.drs.Write_sigGen_Offset(float(_offset))

                # Sending sigGenDamped
                self.drs.Write_sigGen_Aux(_damping)
                self.drs.ConfigSigGen(_sigType, _n_cycles,
                                      _phase_shift, _final_phase)

            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.warning(self, 'Warning', 'Failed to configure'
                                 'the signal generator.\nPlease verify the '
                                 'parameters of the Power Supply.',
                                 _QMessageBox.Ok)
            return False

    def reset_interlocks(self):
        """Resets power supply hardware/software interlocks"""
        try:
            _ps_type = self.config.ps_type
            if not self.set_address(_ps_type):
                return
            # 1000A power supply, reset capacitor bank interlock
            if _ps_type == 2:
                self.drs.SetSlaveAdd(_ps_type - 1)
                self.drs.ResetInterlocks()

            self.drs.SetSlaveAdd(_ps_type)
            self.drs.ResetInterlocks()
            self.ui.pb_interlock.setEnabled(False)
            self.status_powersupply()
            _QMessageBox.information(self, 'Information',
                                     'Interlocks reseted.',
                                     _QMessageBox.Ok)
            return
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.warning(self, 'Warning',
                                 'Interlocks not reseted.',
                                 _QMessageBox.Ok)
            return

    def cycling_ps(self):
        """Initializes power supply cycling routine."""
        _curve_type = int(self.ui.tabWidget_3.currentIndex())

        _ps_type = self.config.ps_type
        if not self.set_address(_ps_type):
            return

        try:
            if not self.set_op_mode(3):
                _QMessageBox.warning(self, 'Warning',
                                     'Could not set the sigGen '
                                     'operation mode.',
                                     _QMessageBox.Ok)
                return
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.warning(self, 'Warning', 'Power supply is not in '
                                 'signal generator mode.', _QMessageBox.Ok)
            return
        try:
            if _curve_type == 1:
                self.drs.EnableSigGen()
                _freq = self.config.sinusoidal_frequency
                _n_cycles = self.config.sinusoidal_ncycles
                _offset = self.config.sinusoidal_offset
            if _curve_type == 0:
                self.drs.EnableSigGen()
                _freq = self.config.dsinusoidal_frequency
                _n_cycles = self.config.dsinusoidal_ncycles
                _offset = self.config.dsinusoidal_offset
            _deadline = _time.monotonic() + (1/_freq*_n_cycles)
            while _time.monotonic() < _deadline:
                self.ui.tabWidget_2.setEnabled(False)
                self.ui.pb_send.setEnabled(False)
                self.ui.pb_load_ps.setEnabled(False)
                self.ui.pb_refresh.setEnabled(False)
                _QApplication.processEvents()

            self.config.ps_setpoint = _offset
            _QMessageBox.information(self, 'Information',
                                     'Cycling completed successfully.',
                                     _QMessageBox.Ok)
            self.drs.DisableSigGen()
            self.display_current()
            self.ui.tabWidget_2.setEnabled(True)
            self.ui.pb_send.setEnabled(True)
            self.ui.pb_load_ps.setEnabled(True)
            self.ui.pb_refresh.setEnabled(True)
            _QApplication.processEvents()

            if _curve_type == 2:
                pass
            # returns to mode ISlowRef
            if not self.set_op_mode(0):
                _QMessageBox.warning(self, 'Warning',
                                     'Could not set the slowRef '
                                     'operation mode.',
                                     _QMessageBox.Ok)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.warning(self, 'Warning',
                                 'Cycling was not performed.',
                                 _QMessageBox.Ok)
            return

    def add_row(self):
        """Adds row into tw_currents tableWidget."""
        _idx = self.ui.tw_currents.rowCount()
        self.ui.tw_currents.insertRow(_idx)

    def remove_row(self):
        """Removes selected row from tw_currents tableWidget."""
        _tw = self.ui.tw_currents
        _idx = self.ui.tw_currents.currentRow()
        _tw.removeRow(_idx)

    def clear_table(self):
        """Clears tw_currents tableWidget."""
        _tw = self.ui.tw_currents
        _tw.clearContents()
        _ncells = _tw.rowCount()
        try:
            while _ncells >= 0:
                _tw.removeRow(_ncells)
                _ncells = _ncells - 1
                _QApplication.processEvents()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def table_to_array(self):
        """Returns tw_currents tableWidget values in a numpy array."""
        _tw = self.ui.tw_currents
        _ncells = _tw.rowCount()
        _current_array = []
        try:
            if _ncells > 0:
                for i in range(_ncells):
                    _tw.setCurrentCell(i, 0)
                    if _tw.currentItem() is not None:
                        _current_array.append(float(_tw.currentItem().text()))
            return _np.array(_current_array)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.warning(self, 'Warning',
                                 'Could not convert table to array.\n'
                                 'Check if all inputs are numbers.',
                                 _QMessageBox.Ok)
            return _np.array([])

    def array_to_table(self, array):
        """Inserts array values into tw_currents tableWidget."""
        _tw = self.ui.tw_currents
        _ncells = _tw.rowCount()
        _array = array
        if _ncells > 0:
            self.clear_table()
        try:
            for i in range(len(_array)):
                _tw.insertRow(i)
                _item = _QTableWidgetItem()
                _tw.setItem(i, 0, _item)
                _item.setText(str(_array[i]))
                _QApplication.processEvents()
                _time.sleep(0.01)
            return True
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.warning(self, 'Warning',
                                 'Could not insert array values into table.',
                                 _QMessageBox.Ok)
            return False

    def plot(self):
        try:
            _tab_idx = self.ui.tabWidget_3.currentIndex()
            # plot sinusoidal
            if _tab_idx == 1:
                a = float(self.ui.le_sinusoidal_amplitude.text())
                offset = float(self.ui.le_sinusoidal_offset.text())
                f = float(self.ui.le_sinusoidal_frequency.text())
                ncycles = int(self.ui.le_sinusoidal_ncycles.text())
                theta = float(self.ui.le_sinusoidal_phase.text())
                if any(a == 0, f == 0, ncycles == 0):
                    _QMessageBox.waring(self, 'Warning',
                                        'Please check the parameters.',
                                        _QMessageBox.Ok)
                    return
                sen = lambda t: (a*_np.sin(2*_np.pi*f*t + theta/360*2*_np.pi) +
                                 offset)

            # plot damped sinusoidal
            elif _tab_idx == 0:
                a = float(self.ui.le_damp_sin_ampl.text())
                offset = float(self.ui.le_damp_sin_offset.text())
                f = float(self.ui.le_damp_sin_freq.text())
                ncycles = int(self.ui.le_damp_sin_ncycles.text())
                theta = float(self.ui.le_damp_sin_phase.text())
                tau = float(self.ui.le_damp_sin_damping.text())
                if any(a == 0, f == 0, ncycles == 0, tau == 0):
                    _QMessageBox.waring(self, 'Warning',
                                        'Please check the parameters.',
                                        _QMessageBox.Ok)
                    return
                sen = lambda t: (a*_np.sin(2*_np.pi*f*t + theta/360*2*_np.pi) *
                                 _np.exp(-t/tau) + offset)

            x = _np.linspace(0, ncycles/f, ncycles*20)
            y = sen(x)
            fig = self.plot_dialog.figure
            ax = self.plot_dialog.ax
            ax.clear()
            ax.plot(x, y)
            ax.set_xlabel('Time (s)', size=20)
            ax.set_ylabel('Current (I)', size=20)
            ax.grid('on', alpha=0.3)
            fig.tight_layout()
            self.plot_dialog.show()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def status_powersupply(self):
        """Read and display Power Supply status."""
        if self.isActiveWindow():
            try:
                _on = self.drs.Read_ps_OnOff()
                self.config.status = bool(_on)
                _loop = self.drs.Read_ps_OpenLoop()
                self.config.status_loop = bool(_loop)
                if all([_loop, self.ui.le_status_loop.text() == 'Closed']):
                    self.ui.le_status_loop.setText('Open')
                elif all([not _loop,
                          self.ui.le_status_loop.text() == 'Open']):
                    self.ui.le_status_loop.setText('Closed')
                _interlock = (self.drs.Read_ps_HardInterlocks() +
                              self.drs.Read_ps_SoftInterlocks())
                self.config.status_interlock = bool(_interlock)
                if all([self.ui.le_status_con.text() == 'Not Ok',
                        self.drs.ser.is_open]):
                    self.ui.le_status_con.setText('Ok')
                elif all([self.ui.le_status_con.text() == 'Ok',
                          not self.drs.ser.is_open]):
                    self.ui.le_status_con.setText('Not Ok')
                if _interlock and not self.ui.pb_interlock.isEnabled():
                    self.ui.pb_interlock.setEnabled(True)
                    _QMessageBox.warning(self, 'Warning',
                                         'Power Supply interlock active.',
                                         _QMessageBox.Ok)
                elif not _interlock and self.ui.pb_interlock.isEnabled():
                    self.ui.pb_interlock.setEnabled(False)
                _QApplication.processEvents()
                self.display_current()
            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                self.ui.le_status_con.setText('Not Ok')
                _QApplication.processEvents()
                return

    def save_powersupply(self):
        """Save Power Supply settings into database."""
        self.config_ps()
        _name = self.config.ps_name
        try:
            _idn = self.config.get_database_id(self.database, 'name', _name)
            if not len(_idn):
                if self.config.save_to_database(self.database) is not None:
                    self.list_powersupply()
                    self.ui.cb_ps_name.setCurrentText(self.config.ps_name)
                    _QMessageBox.information(self, 'Information', 'Power '
                                             'Supply saved into database.',
                                             _QMessageBox.Ok)
                else:
                    raise Exception('Failed to save Power Supply')
            else:
                _ans = _QMessageBox.question(self, 'Update Power Supply',
                                             'The power supply already exists.'
                                             '\nAre you sure you want to '
                                             'update the database entry?',
                                             _QMessageBox.Yes |
                                             _QMessageBox.No)
                if _ans == _QMessageBox.Yes:
                    if self.config.update_database_table(self.database,
                                                         _idn[0]):
                        _QMessageBox.information(self, 'Information',
                                                 'Power Supply entry updated.',
                                                 _QMessageBox.Ok)
                    else:
                        raise Exception('Failed to save Power Supply')
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.warning(self, 'Warning',
                                 'Failed to save Power Supply entry.',
                                 _QMessageBox.Ok)
            return

    def load_powersupply(self):
        """Load Power Supply settings from database."""
        self.config.ps_name = self.ui.cb_ps_name.currentText()
        try:
            _idn = self.config.get_database_id(self.database, 'name',
                                               self.config.ps_name)
            if len(_idn):
                self.config.read_from_database(self.database, _idn[0])
                self.config_widget()
                self.ui.gb_start_supply.setEnabled(True)
                self.ui.tabWidget_2.setEnabled(True)
                self.ui.pb_send.setEnabled(True)
                self.ui.tabWidget_3.setEnabled(True)
                self.ui.pb_refresh.setEnabled(True)
                self.ui.pb_load_ps.setEnabled(False)
                _QMessageBox.information(self, 'Information',
                                         'Power Supply loaded.',
                                         _QMessageBox.Ok)
            else:
                _QMessageBox.warning(self, 'Warning',
                                     'Could not load the power supply.\n'
                                     'Check if the name really exists.',
                                     _QMessageBox.Ok)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.warning(self, 'Warning',
                                 'Could not load the power supply settings.\n'
                                 'Check if the configuration values are '
                                 'correct.', _QMessageBox.Ok)
