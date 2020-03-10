# -*- coding: utf-8 -*-

"""Power Supply widget for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import time as _time
import traceback as _traceback
import qtpy.uic as _uic
from qtpy.QtWidgets import (
    QWidget as _QWidget,
    QMessageBox as _QMessageBox,
    QApplication as _QApplication,
    QProgressDialog as _QProgressDialog,
    QTableWidgetItem as _QTableWidgetItem,
    )
from qtpy.QtCore import (
    QTimer as _QTimer,
    Signal as _Signal,
    )

from hallbench.gui import utils as _utils
from hallbench.gui.auxiliarywidgets import PlotDialog as _PlotDialog
from hallbench.data.configuration import (
    PowerSupplyConfig as _PowerSupplyConfig,
    CyclingCurve as _CyclingCurve,
    )
from hallbench.devices import (
    voltx as _voltx,
    volty as _volty,
    voltz as _voltz,
    nmr as _nmr,
    dcct as _dcct,
    ps as _ps
    )


class PowerSupplyWidget(_QWidget):
    """Power Supply widget class for the Hall Bench Control application."""

    start_measurement = _Signal([bool])
    current_ramp_end = _Signal([bool])
    current_setpoint_changed = _Signal([float])

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)

        # setup the ui
        uifile = _utils.get_ui_file(self)
        self.ui = _uic.loadUi(uifile, self)

        # power supply current slope, [F1000A, F225A, F10A] [A/s]
        self.slope = [50, 90, 1000]

        self.current_array_index = 0
        self.config = _PowerSupplyConfig()
        self.cycling_curve = _CyclingCurve()
        self.drs = _ps
        self.timer = _QTimer()
        self.plot_dialog = _PlotDialog()
        self.dclink_voltage_tol = 1
        self.cycling_error = []
        self.cycling_time_interval = []

        # fill combobox
        self.list_power_supply()

        # create signal/slot connections
        self.ui.pb_emergency.clicked.connect(self.emergency)
        self.ui.pb_ps_button.clicked.connect(self.start_power_supply)
        self.ui.pb_refresh.clicked.connect(self.display_current)
        self.ui.pb_load_ps.clicked.connect(self.load_power_supply)
        self.ui.pb_save_ps.clicked.connect(self.save_power_supply)
        self.ui.pb_send.clicked.connect(self.send_setpoint)
        self.ui.pb_send_curve.clicked.connect(self.send_curve)
        self.ui.pb_config_pid.clicked.connect(self.config_pid)
        self.ui.pb_reset_inter.clicked.connect(self.reset_interlocks)
        self.ui.pb_cycle.clicked.connect(self.cycling_ps)
        self.ui.pb_plot.clicked.connect(self.plot)
        self.ui.pb_config_ps.clicked.connect(self.config_ps)
        self.ui.pb_add_row.clicked.connect(lambda: self.add_row(
            self.ui.tbl_currents))
        self.ui.pb_remove_row.clicked.connect(lambda: self.remove_row(
            self.ui.tbl_currents))
        self.ui.pb_clear_table.clicked.connect(lambda: self.clear_table(
            self.ui.tbl_currents))
        self.ui.pb_dtrapezoidal_add_row.clicked.connect(lambda: self.add_row(
            self.ui.tbl_dtrapezoidal))
        self.ui.pb_dtrapezoidal_remove_row.clicked.connect(
            lambda: self.remove_row(self.ui.tbl_dtrapezoidal))
        self.ui.pb_dtrapezoidal_clear_table.clicked.connect(
            lambda: self.clear_table(self.ui.tbl_dtrapezoidal))
        self.ui.cb_ps_name.currentIndexChanged.connect(self.change_ps)
        self.ui.cb_ps_name.editTextChanged.connect(self.change_ps)
        self.ui.sbd_current_setpoint.valueChanged.connect(self.check_setpoint)
        self.timer.timeout.connect(self.status_power_supply)
        self.ui.cb_ps_type.currentIndexChanged.connect(
            self.set_monopolar_bipolar_enabled)
        self.ui.pb_monopolar_bipolar.clicked.connect(
            self.configure_monopolar_bipolar)
        self.ui.pb_plot_error.clicked.connect(self.plot_cycling_error)   
        
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
    def current_max(self):
        """Power supply maximum current."""
        return _QApplication.instance().current_max

    @current_max.setter
    def current_max(self, value):
        _QApplication.instance().current_max = value

    @property
    def current_min(self):
        """Power supply minimum current."""
        return _QApplication.instance().current_min

    @current_min.setter
    def current_min(self, value):
        _QApplication.instance().current_min = value

    def plot_cycling_error(self):
        try:
            self.plot_dialog.close()
            fig = self.plot_dialog.figure
            ax = self.plot_dialog.ax
            ax.clear()
            ax.plot(self.cycling_time_interval, self.cycling_error)
            ax.set_xlabel('Time (s)', size=20)
            ax.set_ylabel('Current Error (I)', size=20)
            ax.grid('on', alpha=0.3)
            fig.tight_layout()
            self.plot_dialog.show()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def set_monopolar_bipolar_enabled(self):
        if self.ui.cb_ps_type.currentIndex() == 0:
            self.ui.pb_monopolar_bipolar.setEnabled(True)
        else:
            self.ui.pb_monopolar_bipolar.setEnabled(False)

    def configure_monopolar_bipolar(self):
        try:
            self.ui.pb_monopolar_bipolar.setEnabled(False)

            if not self.drs.ser.is_open:
                _QMessageBox.warning(
                    self, 'Warning',
                    'Power Supply serial port is closed.',
                    _QMessageBox.Ok)
                self.ui.pb_monopolar_bipolar.setEnabled(True)
                return

            if self.config.ps_type is None:
                _QMessageBox.warning(
                    self, 'Warning',
                    'Please configure the power supply and try again.',
                    _QMessageBox.Ok)
                self.ui.pb_monopolar_bipolar.setEnabled(True)
                return
    
            if self.config.ps_type != 2:
                _QMessageBox.warning(
                    self, 'Warning',
                    'Not implemented for the selected power supply.',
                    _QMessageBox.Ok)
                self.ui.pb_monopolar_bipolar.setEnabled(True)
                return

            if not self.check_interlocks():
                self.ui.pb_monopolar_bipolar.setEnabled(True)
                return
            
            _QApplication.processEvents()
            
            self.drs.SetSlaveAdd(1)
            ps_on = self.drs.read_ps_onoff()
            
            if ps_on:
                _msg = (
                    'It is necessary to turn off the power supply. ' + 
                    'Do you wish to continue?')
            
                reply = _QMessageBox.question(
                    self, 'Message', _msg,
                    _QMessageBox.No, _QMessageBox.Yes)
            
                if reply == _QMessageBox.No:
                    self.ui.pb_monopolar_bipolar.setEnabled(True)
                    return
                
                if not self.turn_off_power_supply():
                    _QApplication.processEvents()
                    self.ui.pb_monopolar_bipolar.setEnabled(True)
                    return

                if not self.turn_off_dclink():
                    _QApplication.processEvents()
                    self.ui.pb_monopolar_bipolar.setEnabled(True)
                    return
 
                self.config.status = False
                self.configure_widget_on_off(ps_on=False)
       
            _QApplication.processEvents()

            kp = self.config.Kp
            ki = self.config.Ki

            self.drs.SetSlaveAdd(1)
            if self.ui.rbt_monopolar.isChecked():
                self.config.minimum_current = 0
                self.ui.le_minimum_current.setText('0')  
    
                self.drs.set_param('Min_Ref', 0, 0)
                self.drs.set_param('Min_Ref_OpenLoop', 0, 0)
                self.drs.set_param('PWM_Min_Duty', 0, 0)
                self.drs.set_param('PWM_Min_Duty_OpenLoop', 0, 0)
                self.drs.set_dsp_coeffs(3, 0, [kp, ki, 0.9, 0])
                _time.sleep(1)
      
                self.ui.la_monopolar_led.setEnabled(True)
                self.ui.la_bipolar_led.setEnabled(False)
                
                self.ui.pb_monopolar_bipolar.setEnabled(True)
                _msg = 'Configured as monopolar power supply.'
                _QMessageBox.information(
                    self, 'Information', _msg, _QMessageBox.Ok)
                return
                        
            else:
                self.drs.set_param('Min_Ref', 0, -400)
                self.drs.set_param('Min_Ref_OpenLoop', 0, -40)
                self.drs.set_param('PWM_Min_Duty', 0, -0.9)
                self.drs.set_param('PWM_Min_Duty_OpenLoop', 0, -0.4)                
                self.drs.set_dsp_coeffs(3, 0, [kp, ki, 0.9, -0.9])
                _time.sleep(1)

                self.ui.la_monopolar_led.setEnabled(False)
                self.ui.la_bipolar_led.setEnabled(True)

                self.ui.pb_monopolar_bipolar.setEnabled(True)
                _msg = 'Configured as bipolar power supply.'
                _QMessageBox.information(
                    self, 'Information', _msg, _QMessageBox.Ok)
                return
                
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.critical(
                self,
                'Failure',
                'Failed to configure power supply.',
                _QMessageBox.Ok)
            self.ui.pb_monopolar_bipolar.setEnabled(True)

    def closeEvent(self, event):
        """Close widget."""
        try:
            if self.config.status:
                self.start_power_supply()
            self.close_dialogs()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def close_dialogs(self):
        """Close dialogs."""
        try:
            self.plot_dialog.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def set_current_to_zero(self):
        """Set Power Supply current to zero."""
        self.current_array_index = 0
        self.current_setpoint(setpoint=0)

    def list_power_supply(self):
        """Updates available power supply supply names."""
        try:
            self.config.db_update_database(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
            _l = self.config.get_power_supply_list()
            for _ in range(self.ui.cb_ps_name.count()):
                self.ui.cb_ps_name.removeItem(0)
            self.ui.cb_ps_name.addItems(_l)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def set_address(self, address):
        if self.drs.ser.is_open:
            if address == 2:
                self.drs.SetSlaveAdd(1)
            else:
                self.drs.SetSlaveAdd(address)
            return True
        else:
            _QMessageBox.warning(self, 'Warning',
                                 'Power Supply serial port is closed.',
                                 _QMessageBox.Ok)
            return False

    def set_op_mode(self, mode=0):
        """Sets power supply operation mode.

        Args:
            mode (int): 0 for SlowRef, 1 for SigGen.
        Returns:
            True in case of success.
            False otherwise."""
        if mode:
            _mode = 'Cycle'
        else:
            _mode = 'SlowRef'
        self.drs.select_op_mode(_mode)
        _time.sleep(0.1)
        if self.drs.read_ps_opmode() == _mode:
            return True
        return False

    def turn_on_current_display(self):
        """Set update display flag to True."""
        self.config.update_display = True

    def turn_off_current_display(self):
        """Set update display flag to False."""
        self.config.update_display = False

    def change_ps(self):
        """Sets the Load Power Supply button disabled if the selected supply is
           already loaded."""
        if self.ui.cb_ps_name.currentText() == self.config.ps_name:
            self.ui.pb_load_ps.setEnabled(False)
        else:
            self.ui.pb_load_ps.setEnabled(True)

    def check_interlocks(self):
        """Check power supply interlocks."""
        try:
            try:
                self.drs.read_iload1()
            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                _msg = 'Could not read the digital current.'
                _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
                return False
    
            _status_interlocks = self.drs.read_ps_softinterlocks()
            if _status_interlocks != 0:
                self.ui.pb_interlock.setEnabled(True)
                self.config.status_interlock = True
                _msg = 'Software Interlock active!'
                _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
                return False
            
            _status_interlocks = self.drs.read_ps_hardinterlocks()
            if _status_interlocks != 0:
                self.ui.pb_interlock.setEnabled(True)
                self.config.status_interlock = True
                _msg = 'Hardware Interlock active!'
                _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
                return False
                
            self.config.status_interlock = False
            return True        

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _msg = 'Failed to check power supply interlocks.'
            _QMessageBox.critical(self, 'Failure', _msg, _QMessageBox.Ok)
            return False
   
    def turn_on_dclink(self, wait_counter=100):
        try:
            _ps_type = self.config.ps_type
            
            if _ps_type is None:
                _msg = 'Please configure the power supply and try again.'
                _QMessageBox.warning(
                    self, 'Warning', _msg, _QMessageBox.Ok)
                return False                
                
            if _ps_type != 2:
                return True
            
            self.drs.SetSlaveAdd(2)
            _dclink_on = self.drs.read_ps_onoff()
            
            # Turn on DC link
            if not _dclink_on:
                self.drs.turn_on()
                _time.sleep(1.2)
                if not self.drs.read_ps_onoff():
                    self.turn_off_dclink()
                    _msg = 'Power Supply Capacitor Bank did not initialize.'
                    _QMessageBox.warning(
                        self, 'Warning', _msg, _QMessageBox.Ok)
                    return False
            
            # Closing DC link Loop
            self.drs.closed_loop()
            _time.sleep(1)
            if self.drs.read_ps_openloop():
                self.turn_off_dclink()
                _msg = 'Capacitor Bank circuit loop is not closed.'
                _QMessageBox.warning(
                    self, 'Warning', _msg, _QMessageBox.Ok)
                return False           
            
            # Operation mode selection for SlowRef
            if not self.set_op_mode(0):
                self.turn_off_dclink()
                _msg = 'Could not set the slowRef operation mode.'
                _QMessageBox.warning(
                    self, 'Warning', _msg, _QMessageBox.Ok)
                return False    
        
            # Set 90 V for Capacitor Bank (default value)
            _dclink_value = self.config.dclink
            if _dclink_value is None:
                self.turn_off_dclink()
                _msg = 'Invalid DC link setpoint value.'
                _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
                return False

            self.drs.set_slowref(_dclink_value)
            _time.sleep(1)
            _feedback_DCLink = self.drs.read_vdclink()

            _prg_dialog = _QProgressDialog(
                'Setting Capacitor Bank Voltage...',
                '', _feedback_DCLink, _dclink_value)
            _prg_dialog.setWindowTitle('Information')
            _prg_dialog.autoClose()
            _prg_dialog.setCancelButton(None)

            tol = self.dclink_voltage_tol            
            if _feedback_DCLink + tol < _dclink_value:
                _prg_dialog.show()

            _i = wait_counter
            while _feedback_DCLink + tol < _dclink_value  and _i > 0:
                _feedback_DCLink = self.drs.read_vdclink()
                _prg_dialog.setValue(_feedback_DCLink)
                _QApplication.processEvents()
                _time.sleep(0.5)
                _i = _i-1

            if _i == 0:
                self.turn_off_dclink()
                _msg = 'DC link setpoint is not set.'
                _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
                return False
            
            return True
        
        except Exception:
            self.turn_off_dclink()
            _traceback.print_exc(file=_sys.stdout)
            _msg = 'Failed to turn on power supply capacitor bank.'
            _QMessageBox.critical(self, 'Failure', _msg, _QMessageBox.Ok)
            return False
      
    def turn_on_power_supply(self):
        try:
            _ps_type = self.config.ps_type
            if _ps_type == 2:
                self.drs.SetSlaveAdd(1)
            else:
                self.drs.SetSlaveAdd(_ps_type)         

            if _ps_type < 4:
                self.pid_setting()

            _ps_on = self.drs.read_ps_onoff()
            
            if not _ps_on:
                self.drs.turn_on()
                _time.sleep(1.2)
                if not self.drs.read_ps_onoff():
                    self.turn_off_power_supply()
                    self.turn_off_dclink()
                    _msg = 'Power Supply did not initialize.'
                    _QMessageBox.warning(
                        self, 'Warning', _msg, _QMessageBox.Ok)
                    return False

            # Closed Loop
            self.drs.closed_loop()
            _time.sleep(1.2)
            if self.drs.read_ps_openloop() == 1:
                self.turn_off_power_supply()
                self.turn_off_dclink()
                _msg = 'Power Supply circuit loop is not closed.'
                _QMessageBox.warning(
                    self, 'Warning', _msg, _QMessageBox.Ok)
                return False
      
            return True
            
        except Exception:
           self.turn_off_power_supply()
           self.turn_off_dclink()
           _traceback.print_exc(file=_sys.stdout)
           _msg = 'Failed to turn on power supply.'
           _QMessageBox.critical(self, 'Failure', _msg, _QMessageBox.Ok)
           return False
   
    def turn_off_dclink(self):
        try:
            _ps_type = self.config.ps_type
            if _ps_type is None:
                raise Exception
            
            if _ps_type == 2:
                self.drs.SetSlaveAdd(2)
                self.drs.turn_off()
                _time.sleep(1.2)

                if self.drs.read_ps_onoff():
                    _msg = 'Failed to turn off power supply capacitor bank.'
                    _QMessageBox.critical(
                        self, 'Failure', _msg, _QMessageBox.Ok)
                    return False
                
                else:
                    return True
    
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _msg = 'Failed to turn off power supply capacitor bank.'
            _QMessageBox.critical(self, 'Failure', _msg, _QMessageBox.Ok)
            return False
   
    def turn_off_power_supply(self):
        try:
            _ps_type = self.config.ps_type
            if _ps_type is None:
                raise Exception
            
            if _ps_type == 2:
                self.drs.SetSlaveAdd(1)
            else:
                self.drs.SetSlaveAdd(_ps_type)   
            
            self.drs.turn_off()
            _time.sleep(1.2)
            
            if self.drs.read_ps_onoff():
                _msg = 'Failed to turn off power supply.'
                _QMessageBox.critical(self, 'Failure', _msg, _QMessageBox.Ok)
                return False
            
            else:
                return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _msg = 'Failed to turn off power supply.'
            _QMessageBox.critical(self, 'Failure', _msg, _QMessageBox.Ok)
            return False

    def configure_widget_on_off(self, ps_on):
        try:
            if ps_on:
                self.change_ps_button(False)
                self.config.status = True
                self.config.status_loop = True
                self.config.main_current = 0
                self.ui.le_status_loop.setText('Closed')
                self.ui.tabWidget_2.setEnabled(True)
                self.ui.pb_send.setEnabled(True)
                self.ui.twg_cycling_curves.setEnabled(True)
                self.ui.pb_refresh.setEnabled(True)
                self.ui.pb_send.setEnabled(True)
                self.ui.pb_send_curve.setEnabled(True)
                self.timer.start(30000)
            
            else:
                self.config.status = False
                self.config.status_loop = False
                self.config.main_current = 0
                self.ui.le_status_loop.setText('Open')
                self.ui.pb_send.setEnabled(False)
                self.ui.pb_cycle.setEnabled(False)
                self.ui.pb_send_curve.setEnabled(False)
                self.change_ps_button(True)
                self.timer.stop()            
        
        except Exception:
           _traceback.print_exc(file=_sys.stdout)

    def start_power_supply(self):
        """Starts/Stops the Power Supply."""
        try:
            if not self.drs.ser.is_open:
                _QMessageBox.warning(
                    self, 'Warning',
                    'Power Supply serial port is closed.',
                    _QMessageBox.Ok)
                if self.config.status is False:
                    self.change_ps_button(True)
                else:
                    self.change_ps_button(False)
                return          

            if self.config.ps_type is None:
                _msg = 'Please configure the power supply and try again.'
                _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
                if self.config.status is False:
                    self.change_ps_button(True)
                else:
                    self.change_ps_button(False)
                return

            _dcct.dcct_head = self.config.dcct_head
            _dcct.config()

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

            if self.config.status is False:
                if not self.check_interlocks():
                    self.change_ps_button(True)
                    return
                
                if _ps_type  == 2:
                    if not self.turn_on_dclink():
                        self.change_ps_button(True)
                        return  
         
                if self.turn_on_power_supply():
                    self.configure_widget_on_off(ps_on=True)
                    self.status_power_supply()
                    _msg = 'The Power Supply started successfully.'
                    _QMessageBox.information(
                        self, 'Information', _msg, _QMessageBox.Ok)
                
                else:
                    self.configure_widget_on_off(ps_on=False)
            
            else:
                if self.turn_off_power_supply() and self.turn_off_dclink():
                    self.configure_widget_on_off(ps_on=False)
                    _msg = 'Power supply was turned off.'
                    _QMessageBox.information(
                        self, 'Information', _msg, _QMessageBox.Ok)
                else:
                    self.configure_widget_on_off(ps_on=True)
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _msg = 'Failed to change the power supply state.'
            _QMessageBox.critical(self, 'Failure', _msg, _QMessageBox.Ok)
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

        try:
            self.current_array_index = 0
            self.config.ps_name = self.ui.cb_ps_name.currentText()
            self.config.ps_type = self.ui.cb_ps_type.currentIndex() + 2
                     
            self.config.dclink = self.ui.sb_dclink.value()
            self.config.dcct = self.ui.chb_dcct.isChecked()
            self.config.ps_setpoint = self.ui.sbd_current_setpoint.value()
            self.config.maximum_current = float(
                self.ui.le_maximum_current.text())
            self.config.minimum_current = float(
                self.ui.le_minimum_current.text())
            
            dcct_head_str = self.ui.cb_dcct_select.currentText().replace(
                ' A', '') 
            try:
                self.config.dcct_head = int(dcct_head_str)
            except Exception:
                self.config.dcct_head = None
             
            self.config.Kp = self.ui.sbd_kp.value()
            self.config.Ki = self.ui.sbd_ki.value()
            self.config.current_array = self.table_to_array(
                self.ui.tbl_currents)
            
            self.config.sinusoidal_amplitude = float(
                self.ui.le_sinusoidal_amplitude.text())
            self.config.sinusoidal_offset = float(
                self.ui.le_sinusoidal_offset.text())
            self.config.sinusoidal_frequency = float(
                self.ui.le_sinusoidal_frequency.text())
            self.config.sinusoidal_ncycles = int(
                self.ui.le_sinusoidal_ncycles.text())
            self.config.sinusoidal_aux_param_0 = float(
                self.ui.le_sinusoidal_aux_param_0.text())
            self.config.sinusoidal_aux_param_1 = float(
                self.ui.le_sinusoidal_aux_param_1.text())
            
            self.config.dsinusoidal_amplitude = float(
                self.ui.le_dsinusoidal_amplitude.text())
            self.config.dsinusoidal_offset = float(
                self.ui.le_dsinusoidal_offset.text())
            self.config.dsinusoidal_frequency = float(
                self.ui.le_dsinusoidal_frequency.text())
            self.config.dsinusoidal_ncycles = int(
                self.ui.le_dsinusoidal_ncycles.text())
            self.config.dsinusoidal_aux_param_0 = float(
                self.ui.le_dsinusoidal_aux_param_0.text())
            self.config.dsinusoidal_aux_param_1 = float(
                self.ui.le_dsinusoidal_aux_param_1.text())
            self.config.dsinusoidal_aux_param_2 = float(
                self.ui.le_dsinusoidal_aux_param_2.text())
            
            self.config.dsinusoidal2_amplitude = float(
                self.ui.le_dsinusoidal2_amplitude.text())
            self.config.dsinusoidal2_offset = float(
                self.ui.le_dsinusoidal2_offset.text())
            self.config.dsinusoidal2_frequency = float(
                self.ui.le_dsinusoidal2_frequency.text())
            self.config.dsinusoidal2_ncycles = int(
                self.ui.le_dsinusoidal2_ncycles.text())
            self.config.dsinusoidal2_aux_param_0 = float(
                self.ui.le_dsinusoidal2_aux_param_0.text())
            self.config.dsinusoidal2_aux_param_1 = float(
                self.ui.le_dsinusoidal2_aux_param_1.text())
            self.config.dsinusoidal2_aux_param_2 = float(
                self.ui.le_dsinusoidal2_aux_param_2.text())

            self.config.trapezoidal_amplitude = float(
                self.ui.le_trapezoidal_amplitude.text())
            self.config.trapezoidal_offset = float(
                self.ui.le_trapezoidal_offset.text())
            self.config.trapezoidal_ncycles = int(
                self.ui.le_trapezoidal_ncycles.text())
            self.config.trapezoidal_aux_param_0 = float(
                self.ui.le_trapezoidal_aux_param_0.text())
            self.config.trapezoidal_aux_param_1 = float(
                self.ui.le_trapezoidal_aux_param_1.text())
            self.config.trapezoidal_aux_param_2 = float(
                self.ui.le_trapezoidal_aux_param_2.text())
            
            self.config.dtrapezoidal_array = self.table_to_array(
                self.ui.tbl_dtrapezoidal)
            self.config.dtrapezoidal_offset = float(
                self.ui.le_dtrapezoidal_offset.text())

            if self.config.ps_type == 3:
                # Monopolar power supply
                self.ui.twg_cycling_curves.setTabEnabled(0, False)
                self.ui.twg_cycling_curves.setTabEnabled(1, False)
            else:
                self.ui.twg_cycling_curves.setTabEnabled(0, True)
                self.ui.twg_cycling_curves.setTabEnabled(1, True)

            self.current_max = self.config.maximum_current
            self.current_min = self.config.minimum_current
            self.drs.ps_type = self.config.ps_type
            _dcct.dcct_head = self.config.dcct_head
            
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def config_widget(self):
        """Sets current configuration variables into the widget."""

        try:
            self.ui.cb_ps_name.setCurrentText(self.config.ps_name)
            self.ui.cb_ps_type.setCurrentIndex(self.config.ps_type - 2)
            self.ui.sb_dclink.setValue(self.config.dclink)
            self.ui.sbd_current_setpoint.setValue(self.config.ps_setpoint)
            self.ui.le_maximum_current.setText(
                str(self.config.maximum_current))
            self.ui.le_minimum_current.setText(
                str(self.config.minimum_current))
            self.ui.chb_dcct.setChecked(self.config.dcct)
            self.ui.cb_dcct_select.setCurrentText(str(self.config.dcct_head) +
                                                  ' A')
            self.ui.sbd_kp.setValue(self.config.Kp)
            self.ui.sbd_ki.setValue(self.config.Ki)
            self.array_to_table(
                self.config.current_array, self.ui.tbl_currents)

            self.ui.le_sinusoidal_amplitude.setText(str(
                self.config.sinusoidal_amplitude))
            self.ui.le_sinusoidal_offset.setText(str(
                self.config.sinusoidal_offset))
            self.ui.le_sinusoidal_frequency.setText(str(
                self.config.sinusoidal_frequency))
            self.ui.le_sinusoidal_ncycles.setText(str(
                self.config.sinusoidal_ncycles))
            self.ui.le_sinusoidal_aux_param_0.setText(
                str(self.config.sinusoidal_aux_param_0))
            self.ui.le_sinusoidal_aux_param_1.setText(str(
                self.config.sinusoidal_aux_param_1))
            
            self.ui.le_dsinusoidal_amplitude.setText(str(
                self.config.dsinusoidal_amplitude))
            self.ui.le_dsinusoidal_offset.setText(
                str(self.config.dsinusoidal_offset))
            self.ui.le_dsinusoidal_frequency.setText(str(
                self.config.dsinusoidal_frequency))
            self.ui.le_dsinusoidal_ncycles.setText(str(
                self.config.dsinusoidal_ncycles))
            self.ui.le_dsinusoidal_aux_param_0.setText(
                str(self.config.dsinusoidal_aux_param_0))
            self.ui.le_dsinusoidal_aux_param_1.setText(
                str(self.config.dsinusoidal_aux_param_1))
            self.ui.le_dsinusoidal_aux_param_2.setText(
                str(self.config.dsinusoidal_aux_param_2))
            
            self.ui.le_dsinusoidal2_amplitude.setText(str(
                self.config.dsinusoidal2_amplitude))
            self.ui.le_dsinusoidal2_offset.setText(str(
                self.config.dsinusoidal2_offset))
            self.ui.le_dsinusoidal2_frequency.setText(str(
                self.config.dsinusoidal2_frequency))
            self.ui.le_dsinusoidal2_ncycles.setText(str(
                self.config.dsinusoidal2_ncycles))
            self.ui.le_dsinusoidal2_aux_param_0.setText(str(
                self.config.dsinusoidal2_aux_param_0))
            self.ui.le_dsinusoidal2_aux_param_1.setText(str(
                self.config.dsinusoidal2_aux_param_1))
            self.ui.le_dsinusoidal2_aux_param_2.setText(str(
                self.config.dsinusoidal2_aux_param_2))

            self.ui.le_trapezoidal_amplitude.setText(str(
                self.config.trapezoidal_amplitude))
            self.ui.le_trapezoidal_offset.setText(str(
                self.config.trapezoidal_offset))
            self.ui.le_trapezoidal_ncycles.setText(str(
                self.config.trapezoidal_ncycles))
            self.ui.le_trapezoidal_aux_param_0.setText(str(
                self.config.trapezoidal_aux_param_0))
            self.ui.le_trapezoidal_aux_param_1.setText(str(
                self.config.trapezoidal_aux_param_1))
            self.ui.le_trapezoidal_aux_param_2.setText(str(
                self.config.trapezoidal_aux_param_2))

            self.array_to_table(
                self.config.dtrapezoidal_array, self.ui.tbl_dtrapezoidal)
            self.ui.le_dtrapezoidal_offset.setText(str(
                self.config.dtrapezoidal_offset))

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

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
        try:
            _kp = self.ui.sbd_kp.value()
            _ki = self.ui.sbd_ki.value()
            self.config.Kp = _kp
            self.config.Ki = _ki
            _ps_type = self.config.ps_type
            if _ps_type is None:
                _QMessageBox.warning(self, 'Warning', 'Please configure the '
                                     'Power supply first.', _QMessageBox.Ok)
                return False
            if not self.set_address(_ps_type):
                return

            if _ps_type == 3:
                _umin = 0
            else:
                _umin = -0.90

            if _ps_type in [2, 3, 4]:
                _dsp_id = 0
            elif _ps_type == 5:
                _dsp_id = 1
            elif _ps_type == 6:
                _dsp_id = 2
            elif _ps_type == 7:
                _dsp_id = 3

            self.drs.set_dsp_coeffs(3, _dsp_id, [_kp, _ki, 0.90, _umin])

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
        self.drs.set_slowref(0)
        _time.sleep(0.1)
        self.drs.turn_off()
        _time.sleep(0.1)
        if self.config.ps_type == 2:
            _time.sleep(0.9)
        if self.drs.read_ps_onoff() == 0:
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
            _refresh_current = round(float(self.drs.read_iload1()), 3)
            self.ui.lcd_ps_reading.display(_refresh_current)
            _QApplication.processEvents()
            if all([self.ui.chb_dcct.isChecked()]):
                self.ui.lcd_current_dcct.setEnabled(True)
                self.ui.label_161.setEnabled(True)
                self.ui.label_164.setEnabled(True)
                _current = round(_dcct.read_current(), 3)
                self.ui.lcd_current_dcct.display(_current)
            _QApplication.processEvents()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.warning(self, 'Warning', 'Could not display Current.',
                                 _QMessageBox.Ok)
            return

    def change_setpoint_and_emit_signal(self, status):
        """Change current setpoint and emit signal."""
        if not status:
            self.current_array_index = 0
            return
        
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

            if not self.drs.read_ps_onoff():
                return False

            # verify current limits
            _setpoint = setpoint
            if not self.verify_current_limits(setpoint):
                self.ui.tabWidget_2.setEnabled(True)
                self.ui.pb_send.setEnabled(True)
                return False

            # send setpoint and wait until current is set
            self.drs.set_slowref(_setpoint)
            _time.sleep(0.1)
            for _ in range(30):
                _compare = round(float(self.drs.read_iload1()), 3)
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
            _setpoint = self.ui.sbd_current_setpoint.value()
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
        _setpoint = self.ui.sbd_current_setpoint.value()
        _maximum_current = self.config.maximum_current
        _minimum_current = self.config.minimum_current
        if _maximum_current is None:
            return
        if _setpoint > self.config.maximum_current:
            self.ui.sbd_current_setpoint.setValue(_maximum_current)
            _QMessageBox.warning(self, 'Warning',
                                 'Current value is too high.',
                                 _QMessageBox.Ok)
        elif _setpoint < self.config.minimum_current:
            self.ui.sbd_current_setpoint.setValue(_minimum_current)
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
                self.ui.sbd_current_setpoint.setValue(float(_current_max))
                _current = _current_max
                _QMessageBox.warning(self, 'Warning', 'Current value is too '
                                     'high', _QMessageBox.Ok)
                return False
            if _current < _current_min:
                self.ui.sbd_current_setpoint.setValue(float(_current_min))
                _current = _current_min
                _QMessageBox.warning(self, 'Warning', 'Current value is too '
                                     'low.', _QMessageBox.Ok)
                return False
        else:
               
            if (_current + offset) > _current_max:
                _QMessageBox.warning(self, 'Warning', 'Current '
                                     'values out of bounds.', _QMessageBox.Ok)
                return False

        return True

    def send_curve(self):
        """UI function to set power supply curve generator."""
        self.ui.tabWidget_2.setEnabled(False)
        self.ui.pb_plot_error.setEnabled(False)
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

    def update_cycling_curve(self):
        """Configures power supply curve generator."""
        self.config_ps()

        _tab_name = self.ui.twg_cycling_curves.currentWidget().objectName()
        _curve_name = _tab_name.replace('tab_', '')

        try:
            _offset = float(
                getattr(self.config, _curve_name + '_offset'))
            
            if not self.verify_current_limits(_offset):
                _le_offset = getattr(
                    self.ui, 'le_' + _curve_name + '_offset')
                _le_offset.setText('0')
                setattr(self.config, _curve_name + '_offset', 0)
                return False

            if _curve_name == 'dtrapezoidal':
                self.cycling_curve.clear()
                self.cycling_curve.power_supply = self.config.ps_name
                self.cycling_curve.curve_name = _curve_name
                self.cycling_curve.offset = _offset
                self.cycling_curve.dtrapezoidal_array = self.table_to_array(
                    self.ui.tbl_dtrapezoidal)
                return True
                
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _msg = 'Please verify the Offset parameter of the curve.'
            _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
            return False

        try:
            _amplitude = float(
                getattr(self.config, _curve_name + '_amplitude'))
            
            if not self.verify_current_limits(abs(_amplitude), True, _offset):
                _le_amplitude = getattr(
                    self.ui, 'le_' + _curve_name + '_amplitude')
                _le_amplitude.setText('0')
                setattr(self.config, _curve_name + '_amplitude', 0)
                return False
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _msg =  'Please verify the Amplitude parameter of the curve.'
            _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
            return False

        try:
            _ncycles = int(
                getattr(self.config, _curve_name + '_ncycles'))
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _msg = 'Please verify the #cycles parameter of the curve.'
            _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
            return False

        if _curve_name in ['sinusoidal', 'dsinusoidal', 'dsinusoidal2']:
            try:
                _frequency = float(
                    getattr(self.config, _curve_name + '_frequency'))
            
            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                _msg = 'Please verify the Frequency parameter of the curve.'
                _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
                return False
        else:
            _frequency = 0

        try:
            _aux_param_0 = float(
                getattr(self.config, _curve_name + '_aux_param_0'))
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _msg = 'Please verify the Phase parameter of the curve.'
            _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
            return False
        
        try:
            _aux_param_1 = float(
                getattr(self.config, _curve_name + '_aux_param_1'))
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _msg = 'Please verify the Final phase parameter of the curve.'
            _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
            return False

        if _curve_name in ['dsinusoidal', 'dsinusoidal2', 'trapezoidal']:
            try:
                _aux_param_2 = float(
                    getattr(self.config, _curve_name + '_aux_param_2'))
            
            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                _msg = 'Please verify the Damping parameter of the curve.'
                _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
                return False
        
        else:
            _aux_param_2 = 0
        
        _aux_param_3 = 0
        
        _QApplication.processEvents()

        # Generating curves
        try:
            if _curve_name == 'sinusoidal':
                _sigtype = 0
            elif _curve_name == 'dsinusoidal':
                _sigtype = 1
            elif _curve_name == 'trapezoidal':
                _sigtype = 2
            elif _curve_name == 'dsinusoidal2':
                _sigtype = 3               
            else:
                return False
                
            self.cycling_curve.clear()
            self.cycling_curve.power_supply = self.config.ps_name
            self.cycling_curve.curve_name = _curve_name
            self.cycling_curve.siggen_type = _sigtype
            self.cycling_curve.num_cycles = _ncycles
            self.cycling_curve.freq = _frequency
            self.cycling_curve.amplitude = _amplitude
            self.cycling_curve.offset = _offset
            self.cycling_curve.aux_param_0 = _aux_param_0
            self.cycling_curve.aux_param_1 = _aux_param_1
            self.cycling_curve.aux_param_2 = _aux_param_2
            self.cycling_curve.aux_param_3 = _aux_param_3
            
            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return False

    def curve_gen(self):
        """Configures power supply curve generator."""
        
        if not self.update_cycling_curve():
            _msg = 'Failed to update cycling curve.'
            _QMessageBox.warning(self, 'Warning', _msg, _QMessageBox.Ok)
            return False
                           
        _ps_type = self.config.ps_type
        if not self.set_address(_ps_type):
            return False

        try:
            # Sending curves to PS Controller
            self.drs.cfg_siggen(
                self.cycling_curve.siggen_type,
                self.cycling_curve.num_cycles,
                self.cycling_curve.freq,
                self.cycling_curve.amplitude,
                self.cycling_curve.offset,
                self.cycling_curve.aux_param_0,
                self.cycling_curve.aux_param_1,
                self.cycling_curve.aux_param_2,
                self.cycling_curve.aux_param_3)           
            return True
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.warning(self, 'Warning', 'Failed to send '
                                 'configuration to the controller.\n'
                                 'Please, verify the parameters of the'
                                 ' Power Supply.',
                                 _QMessageBox.Ok)
            return False
        
    def reset_interlocks(self):
        """Resets power supply hardware/software interlocks"""
        try:
            _ps_type = self.config.ps_type
            if not self.set_address(_ps_type):
                return

            _interlock = 0
            # 1000A power supply, reset capacitor bank interlock
            if _ps_type == 2:
                self.drs.SetSlaveAdd(_ps_type)
                self.drs.reset_interlocks()

            if _ps_type == 2:
                self.drs.SetSlaveAdd(1)
            else:
                self.drs.SetSlaveAdd(_ps_type)
            
            self.drs.reset_interlocks()
            self.ui.pb_interlock.setEnabled(False)
            self.status_power_supply()
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
        _tab_name = self.ui.twg_cycling_curves.currentWidget().objectName()
        _curve_name = _tab_name.replace('tab_', '')

        _ps_type = self.config.ps_type
        if not self.set_address(_ps_type):
            return

        try:
            if _curve_name != 'dtrapezoidal':
                if not self.set_op_mode(1):
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
            self.ui.tabWidget_2.setEnabled(False)
            self.ui.pb_send.setEnabled(False)
            self.ui.pb_load_ps.setEnabled(False)
            self.ui.pb_refresh.setEnabled(False)
            
            if _curve_name == 'sinusoidal':
                _freq = self.config.sinusoidal_frequency
                _n_cycles = self.config.sinusoidal_ncycles
                _offset = self.config.sinusoidal_offset
                _duration = _n_cycles/_freq
            
            elif _curve_name == 'dsinusoidal':
                _freq = self.config.dsinusoidal_frequency
                _n_cycles = self.config.dsinusoidal_ncycles
                _offset = self.config.dsinusoidal_offset
                _duration = _n_cycles/_freq
            
            elif _curve_name == 'dsinusoidal2':
                _freq = self.config.dsinusoidal2_frequency
                _n_cycles = self.config.dsinusoidal2_ncycles
                _offset = self.config.dsinusoidal2_offset
                _duration = _n_cycles/_freq
            
            elif _curve_name == 'trapezoidal':
                _n_cycles = self.config.trapezoidal_ncycles
                _rise_time = self.config.trapezoidal_aux_param_0
                _cte_time = self.config.trapezoidal_aux_param_1
                _fall_time = self.config.trapezoidal_aux_param_2
                _offset = self.config.trapezoidal_offset
                _duration = _n_cycles*(
                    _rise_time + _fall_time + _cte_time)
            
            if _curve_name == 'dtrapezoidal':
                _offset = self.config.dtrapezoidal_offset
                if not self.dtrapezoidal_cycle():
                    raise Exception('Failure during trapezoidal cycle.')
            else:   
                self.drs.enable_siggen()
                _t0 = _time.monotonic()
                _deadline = _t0 + _duration
                _prg_dialog = _QProgressDialog('Cycling Power Supply...',
                                               'Abort', _t0,
                                               _deadline + 0.3, self)
                _prg_dialog.setWindowTitle('SigGen')
                _prg_dialog.show()
                _abort_flag = False
                _t = _time.monotonic()
                
                self.cycling_error = []
                self.cycling_time_interval = []
                self.cycling_curve.cycling_error_time = []
                self.cycling_curve.cycling_error_current = []
                
                while _t < _deadline:
                    if _prg_dialog.wasCanceled():
                        _ans = _QMessageBox.question(self, 'Abort Cycling',
                                                     'Do you really want to '
                                                     'abort the cycle '
                                                     'process?',
                                                     (_QMessageBox.No |
                                                      _QMessageBox.Yes))
                        if _ans == _QMessageBox.Yes:
                            _abort_flag = True
                            break
                    _prg_dialog.setValue(_t)
                    _QApplication.processEvents()
                    _time.sleep(0.02)
                    _t = _time.monotonic()
                    
                    if _ps_type == 2:
                        try:
                            self.cycling_error.append(
                                self.drs.read_bsmp_variable(28,'float'))
                            self.cycling_time_interval.append(_t - _t0)
                        except Exception:
                            pass
                    
                _prg_dialog.destroy()
                _QApplication.processEvents()
                self.drs.disable_siggen()

                self.cycling_curve.cycling_error_time = (
                    self.cycling_time_interval)
                self.cycling_curve.cycling_error_current = self.cycling_error
                
                self.cycling_curve.db_update_database(
                    database_name=self.database_name,
                    mongo=self.mongo, server=self.server)
                self.cycling_curve.db_save()

                # returns to mode ISlowRef
                if not self.set_op_mode(0):
                    _QMessageBox.warning(self, 'Warning',
                                         'Could not set the slowRef '
                                         'operation mode.',
                                         _QMessageBox.Ok)
            self.config.ps_setpoint = _offset
            self.drs.set_slowref(_offset)

            if _abort_flag:
                _QMessageBox.warning(self, 'Warning', 'The cycle process was '
                                     'aborted.', _QMessageBox.Ok)
            else:
                _QMessageBox.information(self, 'Information', 'Cycle process '
                                         'completed successfully.',
                                         _QMessageBox.Ok)
            self.display_current()
            self.ui.tabWidget_2.setEnabled(True)
            self.ui.pb_send.setEnabled(True)
            self.ui.pb_load_ps.setEnabled(True)
            self.ui.pb_refresh.setEnabled(True)
            self.ui.pb_plot_error.setEnabled(True)
            _QApplication.processEvents()
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.ui.tabWidget_2.setEnabled(True)
            self.ui.pb_send.setEnabled(True)
            self.ui.pb_load_ps.setEnabled(True)
            self.ui.pb_refresh.setEnabled(True)
            self.ui.pb_plot_error.setEnabled(False)
            _QMessageBox.warning(self, 'Warning',
                                 'Cycling was not performed.',
                                 _QMessageBox.Ok)
            return

    def dtrapezoidal_cycle(self):
        try:
            _ps_type = self.config.ps_type
            if not self.set_address(_ps_type):
                return False

            if _ps_type in [2, 3]:
                _slope = self.slope[_ps_type - 2]
            else:
                _slope = self.slope[2]

            _offset = self.config.dtrapezoidal_offset
            _array = self.config.dtrapezoidal_array

            for i in range(len(_array)):
                if not self.verify_current_limits(_array[i, 0] + _offset):
                    return False

            self.cycling_error = []
            self.cycling_time_interval = []
            self.cycling_curve.cycling_error_time = []
            self.cycling_curve.cycling_error_current = []

            self.drs.set_slowref(_offset)
            for i in range(len(_array)):
                _i0 = _offset
                _i = _array[i, 0] + _offset
                _t = _array[i, 1]
                _t_border = abs(_i - _i0) / _slope
                self.drs.set_slowref(_i)
                _deadline = _time.monotonic() + _t_border + _t
                while _time.monotonic() < _deadline:
                    _QApplication.processEvents()
                    _time.sleep(0.01)
                self.drs.set_slowref(_offset)
                _deadline = _time.monotonic() + _t_border + _t
                while _time.monotonic() < _deadline:
                    _QApplication.processEvents()
                    _time.sleep(0.01)

            self.cycling_curve.clear()
            self.cycling_curve.power_supply = self.config.ps_name
            self.cycling_curve.curve_name = 'dtrapezoidal'
            self.cycling_curve.offset = _offset
            self.cycling_curve.dtrapezoidal_array = _array
            
            self.cycling_curve.db_update_database(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
            self.cycling_curve.db_save()
    
            return True
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return False

    def add_row(self, tw):
        """Adds row into tbl_currents tableWidget."""
        _tw = tw
        _idx = _tw.rowCount()
        _tw.insertRow(_idx)

    def remove_row(self, tw):
        """Removes selected row from tbl_currents tableWidget."""
        _tw = tw
        _idx = _tw.currentRow()
        _tw.removeRow(_idx)

    def clear_table(self, tw):
        """Clears tbl_currents tableWidget."""
        _tw = tw
        _tw.clearContents()
        _ncells = _tw.rowCount()
        try:
            while _ncells >= 0:
                _tw.removeRow(_ncells)
                _ncells = _ncells - 1
                _QApplication.processEvents()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def table_to_array(self, tw):
        """Returns tbl_currents tableWidget values in a numpy array."""
        _tw = tw
        _ncells = _tw.rowCount()
        _current_array = []
        _time_flag = False
        if _tw == self.ui.tbl_dtrapezoidal:
            _time_flag = True
            _time_array = []
        try:
            if _ncells > 0:
                for i in range(_ncells):
                    _tw.setCurrentCell(i, 0)
                    if _tw.currentItem() is not None:
                        _current_array.append(float(_tw.currentItem().text()))
                    if _time_flag:
                        _tw.setCurrentCell(i, 1)
                        if _tw.currentItem() is not None:
                            _time_array.append(float(
                                _tw.currentItem().text()))
            if not _time_flag:
                return _np.array(_current_array)
            else:
                return _np.array([_current_array, _time_array]).transpose()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.warning(self, 'Warning',
                                 'Could not convert table to array.\n'
                                 'Check if all inputs are numbers.',
                                 _QMessageBox.Ok)
            return _np.array([])

    def array_to_table(self, array, tw):
        """Inserts array values into tbl_currents tableWidget."""
        _tw = tw
        _ncells = _tw.rowCount()
        _array = array
        _time_flag = False
        if _tw == self.ui.tbl_dtrapezoidal:
            _time_flag = True
        if _ncells > 0:
            self.clear_table(_tw)
        try:
            for i in range(len(_array)):
                _tw.insertRow(i)
                _item = _QTableWidgetItem()
                if not _time_flag:
                    _tw.setItem(i, 0, _item)
                    _item.setText(str(_array[i]))
                else:
                    _item2 = _QTableWidgetItem()
                    _tw.setItem(i, 0, _item)
                    _item.setText(str(_array[i, 0]))
                    _time.sleep(0.1)
                    _tw.setItem(i, 1, _item2)
                    _item2.setText(str(_array[i, 1]))
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
            self.update_cycling_curve()
                      
            fig = self.plot_dialog.figure
            ax = self.plot_dialog.ax
            ax.clear()
            
            if self.cycling_curve.curve_name == 'dtrapezoidal':
                _ps_type = self.config.ps_type
                if _ps_type is None:
                    _QMessageBox.warning(self, 'Warning',
                                         'Please configure the power supply.',
                                         _QMessageBox.Ok)
                    return
                if _ps_type in [2, 3]:
                    _slope = self.slope[_ps_type - 2]
                else:
                    _slope = self.slope[2]
                _offset = float(self.ui.le_dtrapezoidal_offset.text())
                _array = self.table_to_array(self.ui.tbl_dtrapezoidal)
                _t0 = 0
                for i in range(len(_array)):
                    _i0 = _offset
                    _i = _array[i, 0] + _offset
                    _t = _array[i, 1]
                    _t_border = abs(_i - _i0) / _slope
                    ax.plot([_t0, _t0+_t_border], [_offset, _i], 'b-')
                    ax.plot([_t0+_t_border, _t0+_t_border+_t], [_i, _i], 'b-')
                    ax.plot([_t0+_t_border+_t, _t0+2*_t_border+_t],
                            [_i, _offset], 'b-')
                    ax.plot([_t0+2*_t_border+_t, _t0+2*(_t_border+_t)],
                            [_offset, _offset], 'b-')
                    _t0 = _t0+2*(_t_border+_t)
            
            else:
                duration = self.cycling_curve.get_curve_duration()
                npts = self.cycling_curve.num_cycles*100
                x = _np.linspace(0, duration, npts)
                y = self.cycling_curve.get_curve(x)
                ax.plot(x, y)
            
            ax.set_xlabel('Time (s)', size=20)
            ax.set_ylabel('Current (I)', size=20)
            ax.grid('on', alpha=0.3)
            fig.tight_layout()
            self.plot_dialog.show()
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def status_power_supply(self):
        """Read and display Power Supply status."""
        if self.isActiveWindow():
            try:
                _on = self.drs.read_ps_onoff()
                self.config.status = bool(_on)
                _loop = self.drs.read_ps_openloop()
                self.config.status_loop = bool(_loop)
                if all([_loop, self.ui.le_status_loop.text() == 'Closed']):
                    self.ui.le_status_loop.setText('Open')
                elif all([not _loop,
                          self.ui.le_status_loop.text() == 'Open']):
                    self.ui.le_status_loop.setText('Closed')
                _interlock = (self.drs.read_ps_softinterlocks() +
                              self.drs.read_ps_hardinterlocks())
                self.config.status_interlock = bool(_interlock)
                if all([self.ui.le_status_con.text() == 'Not Ok',
                        self.drs.ser.is_open]):
                    self.ui.le_status_con.setText('Ok')
                elif all([self.ui.le_status_con.text() == 'Ok',
                          not self.drs.ser.is_open]):
                    self.ui.le_status_con.setText('Not Ok')
                if _interlock and not self.ui.pb_interlock.isEnabled():
                    self.ui.pb_interlock.setEnabled(True)
                elif not _interlock and self.ui.pb_interlock.isEnabled():
                    self.ui.pb_interlock.setEnabled(False)
                _QApplication.processEvents()
                self.display_current()
            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                self.ui.le_status_con.setText('Not Ok')
                _QApplication.processEvents()
                return

    def save_power_supply(self):
        """Save Power Supply settings into database."""
        self.config_ps()
        _name = self.config.ps_name
        try:
            self.config.db_update_database(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
            _idn = self.config.get_power_supply_id(_name)
            if _idn is None:
                if self.config.db_save() is not None:
                    self.list_power_supply()
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
                    if self.config.db_update(_idn):
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

    def load_power_supply(self):
        """Load Power Supply settings from database."""
        self.config.ps_name = self.ui.cb_ps_name.currentText()
        try:
            _idn = self.config.get_power_supply_id(self.config.ps_name)
            if _idn is not None:
                self.config.db_read(_idn)
                self.config_widget()
                self.ui.gb_start_supply.setEnabled(True)
                self.ui.tabWidget_2.setEnabled(True)
                self.ui.pb_send.setEnabled(True)
                self.ui.twg_cycling_curves.setEnabled(True)
                self.ui.pb_refresh.setEnabled(True)
                self.ui.pb_load_ps.setEnabled(False)
  
                self.current_max = self.config.maximum_current
                self.current_min = self.config.minimum_current
                self.drs.ps_type = self.config.ps_type
                _dcct.dcct_head = self.config.dcct_head

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
