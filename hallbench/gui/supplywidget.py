# -*- coding: utf-8 -*-

"""Power Supply widget for the Hall Bench Control application."""

from PyQt4.QtGui import (
    QWidget as _QWidget,
    QMessageBox as _QMessageBox,
    QApplication as _QApplication,
    )
import PyQt4.uic as _uic

import numpy as _np
import time as _time

from hallbench.gui.utils import getUiFile as _getUiFile


class SupplyWidget(_QWidget):
    """Power Supply widget class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialization
        self.config = self.power_supply_config
        self.drs = self.devices.ps

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
        self.ui.pb_config_ps.clicked.connect(self.config_ps)
        self.ui.cb_ps_name.currentIndexChanged.connect(self.change_ps)
        self.ui.cb_ps_name.editTextChanged.connect(self.change_ps)

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

    def list_powersupply(self):
        """Updates available power supply supply names."""
        _l = self.config.get_table_column(self.database, 'name')
        for i in range(self.ui.cb_ps_name.count()):
            self.ui.cb_ps_name.removeItem(0)
        self.ui.cb_ps_name.addItems(_l)

    def change_ps(self):
        if self.ui.cb_ps_name.currentText() == self.config.ps_name:
            self.ui.pb_load_ps.setEnabled(False)
        else:
            self.ui.pb_load_ps.setEnabled(True)

    def start_powersupply(self):
        """Starts/Stops Power Supply."""
        try:
            self.ui.pb_ps_button.setEnabled(False)
            self.ui.pb_ps_button.setText('Processing...')
            self.ui.tabWidget_2.setEnabled(False)
            _QApplication.processEvents()

            _ps_type = self.config.ps_type
            self.drs.SetSlaveAdd(_ps_type)

            # Status ps is OFF
            if self.config.status is False:
                try:
                    self.drs.Read_iLoad1()
                except Exception:
                    # traceback.print_exc(file=sys.stdout)
                    _QMessageBox.warning(self, 'Warning',
                                         'Could not read the digital current.',
                                         _QMessageBox.Ok)
                    self.change_ps_button(True)
                    return

                _status_interlocks = self.drs.Read_ps_SoftInterlocks()
                if _status_interlocks != 0:
                    self.ui.pb_interlock.setChecked(True)
                    _QMessageBox.warning(self, 'Warning',
                                         'Soft Interlock activated!',
                                         _QMessageBox.Ok)
                    self.change_ps_button(True)
                    return

                _status_interlocks = self.drs.Read_ps_HardInterlocks()
                if _status_interlocks != 0:
                    self.ui.pb_interlock.setChecked(True)
                    _QMessageBox.warning(self, 'Warning',
                                         'Hard Interlock activated!',
                                         _QMessageBox.Ok)
                    self.change_ps_button(True)
                    return

                # PS 1000 A needs to turn dc link on
                if _ps_type == 2:
                    self.drs.SetSlaveAdd(_ps_type-1)
                    # Turn ON ps DClink
                    try:
                        self.drs.TurnOn()
                        _time.sleep(1)
                        if self.drs.Read_ps_OnOff() != 1:
                            _QMessageBox.warning(self, 'Warning',
                                                 'Power Supply Capacitor '
                                                 'Bank did not initialize.',
                                                 _QMessageBox.Ok)
                            self.change_ps_button(True)
                            return
                    except Exception:
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
                            self.change_ps_button(True)
                            return
                    except Exception:
                        _QMessageBox.warning(self, 'Warning',
                                             'Power Supply circuit '
                                             'loop is not closed.',
                                             _QMessageBox.Ok)
                        self.change_ps_button(True)
                        return
                    # Set ISlowRef for DC Link (Capacitor Bank)
                    # Operation mode selection for Slowref
                    self.drs.OpMode(0)

                    _dclink_value = self.config.dclink
                    # Set 30 V for Capacitor Bank (default value)
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
                                             'Check configurations.',
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
                self.config.main_current = 0
                self.ui.le_status_loop.setText('Closed')
                self.ui.tabWidget_2.setEnabled(True)
                self.ui.tabWidget_3.setEnabled(True)
                self.ui.pb_refresh.setEnabled(True)
                self.ui.pb_send.setEnabled(True)
                self.ui.pb_send_curve.setEnabled(True)
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
                                         'power supply off.\nPlease, try '
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
                                             'the power supply off.\nPlease, '
                                             'try again.', _QMessageBox.Ok)
                        self.change_ps_button(False)
                        return
                self.config.status = False
                self.config.main_current = 0
                self.ui.le_status_loop.setText('Open')
                self.ui.pb_send.setEnabled(False)
                self.ui.pb_cycle.setEnabled(False)
                self.ui.pb_send_curve.setEnabled(False)
                self.change_ps_button(True)
                _QMessageBox.information(self, 'Information',
                                         'Power supply was turned off.',
                                         _QMessageBox.Ok)
        except Exception:
            # traceback.print_exc(file=sys.stdout)
            _QMessageBox.warning(self, 'Warning', 'Failed to change the power '
                                 'supply state.', _QMessageBox.Ok)
            self.change_ps_button(False)
            return

    def change_ps_button(self, on=True):
        """Updates ui when turning power supply on/off."""
        self.ui.pb_ps_button.setEnabled(True)
        if on:
            self.ui.pb_ps_button.setChecked(False)
            self.ui.pb_ps_button.setText('Turn ON')
        else:
            self.ui.pb_ps_button.setChecked(True)
            self.ui.pb_ps_button.setText('Turn OFF')
        self.ui.tabWidget_2.setEnabled(True)
        _QApplication.processEvents()

    def config_ps(self):
        """Sets Power Supply configurations acording to current UI inputs."""

        self.config.ps_name = self.ui.cb_ps_name.currentText()
        self.config.ps_type = self.ui.cb_ps_type.currentIndex() + 2
        self.config.dclink = self.ui.sb_dclink.value()
        self.config.ps_setpoint = self.ui.sb_current_setpoint.value()
        self.config.maximum_current = float(self.ui.le_maximum_current.text())
        self.config.minimum_current = float(self.ui.le_minimum_current.text())
        self.config.dcct_head = self.ui.cb_dcct_select.currentIndex()
        self.config.Kp = self.ui.sb_kp.value()
        self.config.Ki = self.ui.sb_ki.value()
        self.config.sinusoidal_amplitude = float(
            self.ui.le_sinusoidal_amplitude.text())
        self.config.sinusoidal_offset = float(
            self.ui.le_sinusoidal_offset.text())
        self.config.sinusoidal_frequency = float(
            self.ui.le_sinusoidal_frequency.text())
        self.config.sinusoidal_ncycles = int(
            self.ui.le_sinusoidal_n_cycles.text())
        self.config.sinusoidal_phasei = float(self.ui.le_initial_phase.text())
        self.config.sinusoidal_phasef = float(self.ui.le_final_phase.text())
        self.config.dsinusoidal_amplitude = float(
            self.ui.le_damp_sin_ampl.text())
        self.config.dsinusoidal_offset = float(
            self.ui.le_damp_sin_offset.text())
        self.config.dsinusoidal_frequency = float(
            self.ui.le_damp_sin_freq.text())
        self.config.dsinusoidal_ncycles = int(
            self.ui.le_damp_sin_ncycles.text())
        self.config.dsinusoidal_phasei = float(
            self.ui.le_damp_sin_phaseShift.text())
        self.config.dsinusoidal_phasef = float(
            self.ui.le_damp_sin_finalPhase.text())
        self.config.dsinusoidal_damp = float(
            self.ui.le_damp_sin_damping.text())

    def config_widget(self):
        """Sets current configuration variables into the widget."""

#         self.ui.cb_ps_name.setCurrentText(self.config.ps_name) #Qt5
        self.ui.cb_ps_name.setCurrentIndex(
            self.ui.cb_ps_name.findText(self.config.ps_name))
        self.ui.cb_ps_type.setCurrentIndex(self.config.ps_type - 2)
        self.ui.sb_dclink.setValue(self.config.dclink)
        self.ui.sb_current_setpoint.setValue(self.config.ps_setpoint)
        self.ui.le_maximum_current.setText(str(self.config.maximum_current))
        self.ui.le_minimum_current.setText(str(self.config.minimum_current))
        self.ui.cb_dcct_select.setCurrentIndex(self.config.dcct_head)
        self.ui.sb_kp.setValue(self.config.Kp)
        self.ui.sb_ki.setValue(self.config.Ki)
        self.ui.le_sinusoidal_amplitude.setText(str(
            self.config.sinusoidal_amplitude))
        self.ui.le_sinusoidal_offset.setText(str(
            self.config.sinusoidal_offset))
        self.ui.le_sinusoidal_frequency.setText(str(
            self.config.sinusoidal_frequency))
        self.ui.le_sinusoidal_n_cycles.setText(str(
            self.config.sinusoidal_ncycles))
        self.ui.le_initial_phase.setText(str(self.config.sinusoidal_phasei))
        self.ui.le_final_phase.setText(str(self.config.sinusoidal_phasef))
        self.ui.le_damp_sin_ampl.setText(str(
            self.config.dsinusoidal_amplitude))
        self.ui.le_damp_sin_offset.setText(str(self.config.dsinusoidal_offset))
        self.ui.le_damp_sin_freq.setText(str(
            self.config.dsinusoidal_frequency))
        self.ui.le_damp_sin_ncycles.setText(str(
            self.config.dsinusoidal_ncycles))
        self.ui.le_damp_sin_phaseShift.setText(str(
            self.config.dsinusoidal_phasei))
        self.ui.le_damp_sin_finalPhase.setText(str(
            self.config.dsinusoidal_phasef))
        self.ui.le_damp_sin_damping.setText(str(self.config.dsinusoidal_damp))

    def config_pid(self):
        """Configures PID settings."""
        _ans = _QMessageBox.question(self, 'PID settings', 'Be aware that '
                                     'this will overwrite the current '
                                     'configurations.\n Are you sure you want '
                                     'to configure the PID parameters?',
                                     _QMessageBox.Yes | _QMessageBox.No)
        if _ans == _QMessageBox.Yes:
            _ans = self.pid_setting()
            if _ans:
                _QMessageBox.information(self, 'Information',
                                         'PID configured.', _QMessageBox.Ok)
            else:
                _QMessageBox.warning(self, 'Fail',
                                     'Power Supply PID configuration fault.',
                                     _QMessageBox.Ok)

    def pid_setting(self):
        """Set power supply PID configurations."""
        self.config.Kp = self.ui.sb_kp.value()
        self.config.Ki = self.self.ui.sb_ki.text()
        _ps_type = self.config.ps_type
        self.drs.SetSlaveAdd(_ps_type)
        _id_mode = 0
        _elp_PI_dawu = 3
        try:
            # Write ID module from controller
            self.drs.Write_dp_ID(_id_mode)
            # Write DP Class for setting PI
            self.drs.Write_dp_Class(_elp_PI_dawu)
        except Exception:
            # traceback.print_exc(file=sys.stdout)
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
            # traceback.print_exc(file=sys.stdout)
            return False

        return True

    def emergency(self):
        """Stops power supply current."""
        self.drs.SetSlaveAdd(self.config.ps_type)
        self.drs.OpMode(0)
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
        _ps_type = self.config.ps_type
        self.drs.SetSlaveAdd(_ps_type)
        try:
            _refresh_current = round(float(self.drs.Read_iLoad1()), 3)
            self.ui.lcd_ps_reading.display(_refresh_current)
            if (self.ui.chb_dcct.isChecked() and
               self.ui.chb_enable_Agilent34970A.isChecked()):
                self.ui.lcd_current_dcct.setEnabled(True)
                self.ui.label_161.setEnabled(True)
                self.ui.label_164.setEnabled(True)
                _current = round(self.dcct_convert(), 3)
                self.ui.lcd_current_dcct.display(_current)
                _QApplication.processEvents()
        except Exception:
            _QMessageBox.warning(self, 'Warning', 'Could not display Current.',
                                 _QMessageBox.Ok)
            return

    def dcct_convert(self):
        _agilent_reading = Lib.comm.agilent34970a.read_temp_volt()
        if isinstance(Lib.measurement_settings, pd.DataFrame):
            Lib.write_value(Lib.measurement_settings, 'temperature', _agilent_reading[0])
            setattr(self, 'temperature_magnet', _agilent_reading[1])
            setattr(self, 'temperature_water', _agilent_reading[2])
        _voltage = _agilent_reading[4]
        if _voltage == '':
            _current = 0
        else:
            #For 40 A dcct head
            if self.ui.dcct_select.currentIndex() == 0:
                _current = (float(_voltage))*4
            #For 160 A dcct head
            if self.ui.dcct_select.currentIndex() == 1:
                _current = (float(_voltage))*16
            #For 320 A dcct head
            if self.ui.dcct_select.currentIndex() == 2:
                _current = (float(_voltage))*32
        return _current

    def current_setpoint(self, setpoint=0):
        """Changes power supply current setpoint settings."""
        self.ui.tabWidget_2.setEnabled(False)
        _ps_type = self.config.ps_type

        self.drs.SetSlaveAdd(_ps_type)

        # verify current limits
        _setpoint = self.verify_current_limits(0, setpoint, 0)
        if _setpoint == 'False':
            self.ui.tabWidget_2.setEnabled(True)
            return False
        self.config.ps_setpoint = _setpoint

        # send setpoint and wait until current is set
        self.drs.SetISlowRef(_setpoint)
        _time.sleep(0.1)
        for i in range(30):
            _compare = round(float(self.drs.Read_iLoad1()), 3)
            self.display_current()
            if abs(_compare - _setpoint) <= 0.5:
                self.ui.tabWidget_2.setEnabled(True)
                return True
            _QApplication.processEvents()
            _time.sleep(1)
        self.ui.tabWidget_2.setEnabled(True)
        return True

    def send_setpoint(self):
        """Sends new current setpoing to the power supply."""
        self.config_ps()

        _setpoint = self._ps_setpoint
        _ans = self.current_setpoint(_setpoint)
        if _ans:
            _QMessageBox.information(self, 'Information',
                                     'Current properly set.',
                                     _QMessageBox.Ok)
        else:
            _QMessageBox.warning(self, 'Warning',
                                 'Current was not properly set.',
                                 _QMessageBox.Ok)

    def verify_current_limits(self, index, current, offset=0):
        """Check the conditions of the Current values sets;
        if there's an error, returns string 'False' in order to
        avoid confusion when returning the 0.0 value."""

        _current = float(current)

        try:
            _current_max = self.config.maximum_current
        except Exception:
            _QMessageBox.warning(self, 'Warning', 'Incorrect value for '
                                 'maximum Supply Current.\nPlease, verify the '
                                 'value.', _QMessageBox.Ok)
            return 'False'
        try:
            _current_min = self.config.minimum_current
        except Exception:
            _QMessageBox.warning(self, 'Warning', 'Incorrect value for '
                                 'minimum Supply Current.\nPlease, verify the '
                                 'value.', _QMessageBox.Ok)
            return 'False'

        if index == 0 or index == 1:
            if _current > _current_max:
                if (index == 0):
                    _QMessageBox.warning(self, 'Warning', 'Value of current '
                                         'higher than the Supply Limit.',
                                         _QMessageBox.Ok)
                self.ui.dsb_current_setpoint.setValue(float(_current_max))
                _current = _current_max
                return 'False'
            if _current < _current_min:
                if index == 0:
                    _QMessageBox.warning(self, 'Warning', 'Current value '
                                         'lower than the Supply Limit.',
                                         _QMessageBox.Ok)
                self.ui.dsb_current_setpoint.setValue(float(_current_min))
                _current = _current_min
                return 'False'
        elif index == 2:
            if ((_current)+offset) > _current_max:
                _QMessageBox.warning(self, 'Warning', 'Check Peak to Peak '
                                     'Current and Offset values.\n Values out '
                                     'of source limit.', _QMessageBox.Ok)
                return 'False'

            if ((-_current)+offset) < _current_min:
                _QMessageBox.warning(self, 'Warning', 'Check Peak to Peak '
                                     'Current and Offset values.\nValues out '
                                     'of source limit.', _QMessageBox.Ok)
                return 'False'

        return float(_current)

    def send_curve(self):
        """Sends curve points to power supply controller."""

        self.ui.tabWidget_2.setEnabled(False)
        if self.curve_gen() is True:
            _QMessageBox.information(self, 'Information',
                                     'Sending Curve Successfully.',
                                     _QMessageBox.Ok)
            self.ui.tabWidget_2.setEnabled(True)
            self.ui.pb_cycle.setEnabled(True)
            _QApplication.processEvents()
        else:
            _QMessageBox.warning(self, 'Warning', 'Fail to send curve.',
                                 _QMessageBox.Ok)
            self.ui.tabWidget_2.setEnabled(True)
            _QApplication.processEvents()
            return False

    def curve_gen(self):
        """Configures power supply curve generator."""
        self.config_ps()
        _curve_type = int(self.ui.tabWidget_3.currentIndex())

        _ps_type = self.config.ps_type
        self.drs.SetSlaveAdd(_ps_type)

        if _curve_type == 1:    # Sinusoidal
            # For Offset
            try:
                _offset = self.config.sinusoidal_offset
                _offset = self.verify_current_limits(0, _offset)
                if _offset == 'False':
                    self.ui.le_sinusoidal_offset.setText('0')
                    self.config.sinusoidal_offset = 0
                    return False
                self.ui.le_sinusoidal_offset.setText(str(_offset))
                self.config.sinusoidal_offset = _offset
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please, verify the '
                                     'Offset parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Amplitude
            try:
                _amp = self.config.sinusoidal_amplitude
                _amp = self.verify_current_limits(2, abs(_amp), _offset)
                if _amp == 'False':
                    self.ui.le_sinusoidal_amplitude.setText('0')
                    self.config.sinusoidal_amplitude = 0
                    return False
                self.ui.le_sinusoidal_amplitude.setText(str(_amp))
                self.config.sinusoidal_amplitude = _amp
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please, verify the '
                                     'Amplitude parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Frequency
            try:
                _freq = self.config.sinusoidal_frequency
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please, verify the '
                                     'Frequency parameter of the curve.',
                                     _QMessageBox.Ok)
            # For N-cycles
            try:
                _n_cycles = self.config.sinusoidal_ncycles
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please, verify the '
                                     '#cycles parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Phase shift
            try:
                _phase_shift = self.config.sinusoidal_phasei
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please, verify the '
                                     'Phase parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Final phase
            try:
                _final_phase = self.config.sinusoidal_phasef
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please, verify the '
                                     'Final phase parameter of the curve.',
                                     _QMessageBox.Ok)

        # Damped Sinusoidal
        if _curve_type == 0:
            # For Offset
            try:
                _offset = self.config.dsinusoidal_offset
                _offset = self.verify_current_limits(0, _offset)
                if _offset == 'False':
                    self.config.dsinusoidal_offset = 0
                    self.ui.le_damp_sin_offset.setText('0')
                    return False
                self.config.dsinusoidal_offset = _offset
                self.ui.le_damp_sin_offset.setText(str(_offset))
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please, verify the '
                                     'Offset parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Amplitude
            try:
                _amp = self.config.dsinusoidal_amplitude
                _amp = self.verify_current_limits(2, abs(_amp), _offset)
                if _amp == 'False':
                    self.config.dsinusoidal_amplitude = 0
                    self.ui.le_damp_sin_ampl.setText('0')
                    return False
                self.config.dsinusoidal_amplitude = _amp
                self.ui.le_damp_sin_ampl.setText(str(_amp))
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please, verify the '
                                     'Amplitude parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Frequency
            try:
                _freq = self.config.dsinusoidal_frequency
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please, verify the '
                                     'Frequency parameter of the curve.',
                                     _QMessageBox.Ok)
            # For N-cycles
            try:
                _n_cycles = self.config.dsinusoidal_ncycles
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please, verify the '
                                     '#cycles parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Phase shift
            try:
                _phase_shift = self.config.dsinusoidal_phasei
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please, verify the '
                                     'Phase parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Final phase
            try:
                _final_phase = self.config.dsinusoidal_phasef
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please, verify the '
                                     'Final Phase parameter of the curve.',
                                     _QMessageBox.Ok)
            # For Damping
            try:
                _damping = self.config.dsinusoidal_damp
            except Exception:
                _QMessageBox.warning(self, 'Warning', 'Please, verify the '
                                     'Damping _time parameter of the curve.',
                                     _QMessageBox.Ok)

        # Generating curves
        try:
            # Sinusoidal
            if _curve_type == 1:
                try:
                    _sigType = 0
                    # send Frequency
                    self.drs.Write_sigGen_Freq(float(_freq))
                    # send Amplitude
                    self.drs.Write_sigGen_Amplitude(float(_amp))
                    # send Offset
                    self.drs.Write_sigGen_Offset(float(_offset))
                    # Sending curves to ps Controller
                    self.drs.ConfigSigGen(_sigType, _n_cycles, _phase_shift,
                                          _final_phase)
                except Exception:
                    # traceback.print_exc(file=sys.stdout)
                    _QMessageBox.warning(self, 'Warning', 'Failed to send '
                                         'configuration to the controller.\n'
                                         'Please, verify the parameters of the'
                                         'Power Supply.', _QMessageBox.Ok)
                    return

            # Damped Sinusoidal
            if _curve_type == 0:
                try:
                    _sigType = 4
                    self.drs.Write_sigGen_Freq(float(_freq))
                    self.drs.Write_sigGen_Amplitude(float(_amp))
                    self.drs.Write_sigGen_Offset(float(_offset))
                except Exception:
                    _QMessageBox.warning(self, 'Warning', 'Failed to send '
                                         'configuration to the controller.\n'
                                         'Please, verify the parameters of '
                                         'the Power Supply.', _QMessageBox.Ok)
                    return

                # Sending sigGenDamped
                try:
                    self.drs.Write_sigGen_Aux(_damping)
                    self.drs.ConfigSigGen(_sigType, _n_cycles,
                                          _phase_shift, _final_phase)
                except Exception:
                    _QMessageBox.warning(self, 'Warning.',
                                         'Damped Sinusoidal fault.',
                                         _QMessageBox.Ok)
                    # traceback.print_exc(file=sys.stdout)
                    return False

            return True
        except Exception:
            return False

    def reset_interlocks(self):
        """Resets power supply hardware/software interlocks"""

        _ps_type = self.config.ps_type

        # 1000A power supply, reset capacitor bank interlock
        if _ps_type == 2:
            self.drs.SetSlaveAdd(_ps_type - 1)
            self.drs.ResetInterlocks()

        self.drs.SetSlaveAdd(_ps_type)

        try:
            self.drs.ResetInterlocks()
            if self.ui.pb_interlock.isChecked():
                self.ui.pb_interlock.setChecked(False)
            _QMessageBox.information(self, 'Information',
                                     'Interlocks reseted.',
                                     _QMessageBox.Ok)
        except Exception:
            _QMessageBox.warning(self, 'Warning',
                                 'Interlocks not reseted.',
                                 _QMessageBox.Ok)
            return

    def cycling_ps(self):
        """Initializes power supply cycling routine."""
        _curve_type = int(self.ui.tabWidget_3.currentIndex())

        _ps_type = self.config.ps_type
        self.drs.SetSlaveAdd(_ps_type)

        try:
            self.drs.OpMode(3)
            if self.drs.Read_ps_OpMode() != 3:
                _QMessageBox.warning(self, 'Warning', 'Power supply is not '
                                     'in signal generator mode.',
                                     _QMessageBox.Ok)
                return False
        except Exception:
            _QMessageBox.warning(self, 'Warning', 'Power supply is not in '
                                 'signal generator mode.', _QMessageBox.Ok)
            return
        try:
            if _curve_type == 1:
                self.drs.EnableSigGen()
                _freq = self.config.sinusoidal_frequency
                _n_cycles = self.config.sinusoidal_ncycles
            if _curve_type == 0:
                self.drs.EnableSigGen()
                _freq = self.config.dsinusoidal_frequency
                _n_cycles = self.config.dsinusoidal_ncycles
            _deadline = _time.monotonic() + (1/_freq*_n_cycles)
            while _time.monotonic() < _deadline:
                self.ui.tabWidget_2.setEnabled(False)
                self.ui.pb_load_ps.setEnabled(False)
                self.ui.pb_refresh.setEnabled(False)
                self.ui.pb_start_meas.setEnabled(False)
                _QApplication.processEvents()

            _QMessageBox.information(self, 'Information',
                                     'Cycling completed successfully.',
                                     _QMessageBox.Ok)
            self.drs.DisableSigGen()
            self.display_current()
            self.ui.tabWidget_2.setEnabled(True)
            self.ui.pb_load_ps.setEnabled(True)
            self.ui.pb_refresh.setEnabled(True)
            _QApplication.processEvents()

            if _curve_type == 2:
                pass
            # returns to mode ISlowRef
            self.drs.OpMode(0)
        except Exception:
            _QMessageBox.warning(self, 'Warning',
                                 'Cycling was not performed.',
                                 _QMessageBox.Ok)
            return

    def save_powersupply(self):
        """Save Power Supply settings into database"""
        self.config_ps()
        _name = self.config.ps_name
        try:
            _idn = self.config.get_database_id(self.database, 'name', _name)
            if not len(_idn):
                if self.config.save_to_database(self.database) is not None:
                    self.list_powersupply()
                # self.ui.cb_ps_name.setCurrentText(self.config.ps_name) #Qt5
                    self.ui.cb_ps_name.setCurrentIndex(
                        self.ui.cb_ps_name.findText(_name))
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
            raise
            _QMessageBox.warning(self, 'Warning',
                                 'Failed to save Power Supply entry.',
                                 _QMessageBox.Ok)
            return

    def load_powersupply(self):
        """Load Power Supply settings from database"""

        self.config.ps_name = self.ui.cb_ps_name.currentText()
        try:
            _idn = self.config.get_database_id(self.database, 'name',
                                               self.config.ps_name)
            if len(_idn):
                self.config.read_from_database(self.database, _idn[0])
                self.config_widget()
                self.ui.gb_start_supply.setEnabled(True)
                self.ui.tabWidget_2.setEnabled(True)
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
            # traceback.print_exc(file=sys.stdout)
            _QMessageBox.warning(self, 'Warning',
                                 'Could not load the power supply settings.\n'
                                 'Check if the configuration values are '
                                 'correct.', _QMessageBox.Ok)
