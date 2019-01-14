# -*- coding: utf-8 -*-

"""Main window for the Hall Bench Control application."""

import sys as _sys
import struct as _struct
import traceback as _traceback
import serial.tools.list_ports as _list_ports
from qtpy.QtWidgets import (
    QApplication as _QApplication,
    QDesktopWidget as _QDesktopWidget,
    QMainWindow as _QMainWindow,
    QMessageBox as _QMessageBox,
    QVBoxLayout as _QVBoxLayout,
    )
from qtpy.QtCore import Qt as _Qt
import qtpy.uic as _uic

from hallbench.calibration import utils as _utils


class CalibrationWindow(_QMainWindow):
    """Main Window class for the Hall Bench Calibration application."""

    def __init__(self, parent=None, width=1200, height=700):
        """Set up the ui and add main tabs."""
        super().__init__(parent)

        # setup the ui
        uifile = _utils.getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)
        self.resize(width, height)

        self.clear()
        self.updateSerialPorts()
        self.ui.nmr_gb.setEnabled(False)
        self.ui.mult_gb.setEnabled(False)
               
        self.ui.connect_btn.clicked.connect(self.connectDevices)
        self.ui.disconnect_btn.clicked.connect(self.disconnectDevices)
        self.ui.nmr_config_btn.clicked.connect(self.configureNMR)
        self.ui.nmr_read_btn.clicked.connect(self.readNMRField)
        self.ui.mult_config_btn.clicked.connect(self.configureMult)
        self.ui.mult_reset_btn.clicked.connect(self.resetMult)
        self.ui.mult_read_btn.clicked.connect(self.readMultVoltage)
        self.ui.mult_range_auto_chb.stateChanged.connect(
            self.setEnableMultRange)
        self.ui.mult_aper_auto_chb.stateChanged.connect(
            self.setEnableMultAper)

    @property
    def mch(self):
        """Multichannel."""
        return _QApplication.instance().mch

    @property
    def mult(self):
        """Multimeter."""
        return _QApplication.instance().mult

    @property
    def nmr(self):
        """NMR."""
        return _QApplication.instance().nmr

    def centralizeWindow(self):
        """Centralize window."""
        window_center = _QDesktopWidget().availableGeometry().center()
        self.move(
            window_center.x() - self.geometry().width()/2,
            window_center.y() - self.geometry().height()/2)

    def clear(self):
        """Clear"""
        self.clearNMRVariables()
        self.clearMultichannelVariables()
        self.clearMultimeterVariables()

    def clearNMRVariables(self):
        """Clear NMR variables."""
        self.nmr_enable = None
        self.nmr_port = None
        self.nmr_baudrate = None
        self.nmr_frequency = None
        self.nmr_mode = None
        self.nmr_field_sense = None
        self.nmr_disp_unit = None
        self.nmr_disp_vel = None
        self.nmr_search_time = None
        self.nmr_channel = None
        self.nmr_nr_channels = None

    def clearMultichannelVariables(self):
        """Clear Multichannel variables."""
        self.mch_enable = None
        self.mch_port = None
        self.mch_baudrate = None 

    def clearMultimeterVariables(self):
        """Clear Multimeter variables."""
        self.mult_enable = None
        self.mult_address = None
        self.mult_func = None
        self.mult_tarm = None
        self.mult_trig = None
        self.mult_nrdgs = None
        self.mult_nrdsg_event = None
        self.mult_arange = None
        self.mult_fixedz = None
        self.mult_range = None
        self.mult_range_auto = None
        self.mult_math = None
        self.mult_azero = None
        self.mult_tbuff = None
        self.mult_delay = None
        self.mult_aper = None
        self.mult_aper_auto = None
        self.mult_disp = None
        self.mult_mem = None
        self.mult_format = None
        self.mult_reset_state = True

    def closeEvent(self, event):
        """Close main window and dialogs."""
        try:
            self.disconnectDevices()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()
        
    def connectDevices(self):
        """Connect bench devices."""       
        try:
            self.nmr_enable = self.ui.nmr_enable_chb.isChecked()
            self.nmr_port = self.ui.nmr_port_cmb.currentText()
            self.nmr_baudrate = int(self.ui.nmr_baudrate_cmb.currentText())

            self.mch_enable = self.ui.mch_enable_chb.isChecked()
            self.mch_port = self.ui.mch_port_cmb.currentText()
            self.mch_baudrate = int(self.ui.mch_baudrate_cmb.currentText())
            
            self.mult_enable = self.ui.mult_enable_chb.isChecked() 
            self.mult_address = self.ui.mult_address_sb.value()           
            
            success = True
            if self.nmr_enable:
                self.nmr.connect(self.nmr_port, self.nmr_baudrate)
                if self.nmr.connected:
                    self.ui.nmr_gb.setEnabled(True)
                    self.ui.nmr_led_la.setEnabled(True)
                else:
                    self.ui.nmr_gb.setEnabled(False)
                    self.ui.nmr_led_la.setEnabled(False)
                    success = False
                       
            if self.mch_enable:
                self.mch.connect(self.mch_port, self.mch_baudrate)
                if self.mch.connected:
                    self.ui.mch_led_la.setEnabled(True)
                else:
                    self.ui.mch_led_la.setEnabled(False)
                    success = False                
            
            if self.mult_enable:
                self.mult.connect(self.mult_address)
                if self.mult.connected:
                    self.ui.mult_gb.setEnabled(True)
                    self.ui.mult_led_la.setEnabled(True)
                else:
                    self.ui.mult_gb.setEnabled(False)
                    self.ui.mult_led_la.setEnabled(False)
                    success = False

            if not success:
                msg = 'Failed to connect devices.'
                _QMessageBox.critical(
                    self, 'Failure', msg, _QMessageBox.Ok)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to connect devices.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def configureNMR(self):
        """Configure NMR."""
        try:           
            self.nmr_frequency = self.ui.nmr_frequency_sb.value()
            self.nmr_mode = self.ui.nmr_mode_cmb.currentIndex()
            self.nmr_field_sense = self.ui.nmr_field_sense_cmb.currentIndex()
            self.nmr_disp_unit = self.ui.nmr_disp_unit_cmb.currentIndex()
            self.nmr_disp_vel = self.ui.nmr_disp_vel_cmb.currentIndex()
            self.nmr_search_time = self.ui.nmr_search_time_sb.value()
            self.nmr_channel = self.ui.nmr_channel_cmb.currentText()
            self.nmr_nr_channels = self.ui.nmr_nr_channels_sb.value()            

            configured = self.nmr.configure(
                self.nmr_frequency,
                self.nmr_mode,
                self.nmr_field_sense,
                self.nmr_disp_unit,
                self.nmr_disp_vel,
                self.nmr_search_time,
                self.nmr_channel,
                self.nmr_nr_channels)
            
            if not configured:
                msg = 'Failed to configure NMR.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)                

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to configure NMR.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            
    def configureMult(self):
        """Configure Multimeter."""
        try:
            self.mult_func = self.ui.mult_func_cmb.currentText()
            self.mult_tarm = self.ui.mult_tarm_cmb.currentText()
            self.mult_trig = self.ui.mult_trig_cmb.currentText()
            self.mult_nrdgs = self.ui.mult_nrdgs_sb.value()
            self.mult_nrdsg_event = self.ui.mult_nrdgs_event_cmb.currentText() 
            self.mult_arange = self.ui.mult_arange_cmb.currentText()
            self.mult_fixedz = self.ui.mult_fixedz_cmb.currentText() 
            self.mult_range = self.ui.mult_range_sb.value()
            self.mult_range_auto = self.ui.mult_range_auto_chb.isChecked()
            self.mult_math = self.ui.mult_math_cmb.currentText()
            self.mult_azero = self.ui.mult_azero_cmb.currentText()
            self.mult_tbuff = self.ui.mult_tbuff_cmb.currentText() 
            self.mult_delay = self.ui.mult_delay_sb.value()/1000
            self.mult_aper = self.ui.mult_aper_sb.value()/1000
            self.mult_aper_auto = self.ui.mult_aper_auto_chb.isChecked()
            self.mult_disp = self.ui.mult_disp_cmb.currentText()
            self.mult_mem = self.ui.mult_mem_cmb.currentText()
            self.mult_format = self.ui.mult_format_cmb.currentText()
            
            self.mult.send_command('RESET')
            self.mult.send_command('FUNC ' + self.mult_func)
            self.mult.send_command('TARM ' + self.mult_tarm)
            self.mult.send_command('TRIG ' + self.mult_trig)
            
            self.mult.send_command(
                'NRDGS ' + str(self.mult_nrdgs) + ',' + self.mult_nrdsg_event)
            
            self.mult.send_command('ARANGE ' + self.mult_arange)
            self.mult.send_command('FIXEDZ ' + self.mult_fixedz)
            
            if self.mult_range_auto:
                self.mult.send_command('RANGE AUTO')
            else:            
                self.mult.send_command('RANGE ' + str(self.mult_range))
            
            self.mult.send_command('MATH ' + self.mult_math)
            self.mult.send_command('AZERO ' + self.mult_azero)
            self.mult.send_command('TBUFF ' + self.mult_tbuff)
            self.mult.send_command('DELAY ' + str(self.mult_delay))
            
            if self.mult_aper_auto:
                self.mult.send_command('APER AUTO')
            else:
                self.mult.send_command('APER ' + str(self.mult_aper))
            
            self.mult.send_command('DISP ' + self.mult_disp)
            self.mult.send_command('SCRATCH')
            self.mult.send_command('END ALWAYS')
            self.mult.send_command('MEM ' + self.mult_mem)
            self.mult.send_command('OFORMAT ' + self.mult_format)
            self.mult.send_command('MFORMAT ' + self.mult_format)
            self.mult_reset_state = False

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to configure multimeter.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)            
           
    def disconnectDevices(self):
        """Disconnect bench devices."""
        try:
            self.nmr.disconnect()
            if self.nmr.connected:
                self.ui.nmr_gb.setEnabled(True)
                self.ui.nmr_led_la.setEnabled(True)
            else:
                self.ui.nmr_gb.setEnabled(False)
                self.ui.nmr_led_la.setEnabled(False)

            self.mch.disconnect()
            if self.mch.connected:
                self.ui.mch_led_la.setEnabled(True)
            else:
                self.ui.mch_led_la.setEnabled(False)

            self.mult.disconnect()
            if self.mult.connected:
                self.ui.mult_gb.setEnabled(True)
                self.ui.mult_led_la.setEnabled(True)
            else:
                self.ui.mult_gb.setEnabled(False)
                self.ui.mult_led_la.setEnabled(False)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to disconnect devices.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def readMultVoltage(self):
        """Read multimeter voltage value."""        
        try:
            fmt = '{0:.10e}'
            r = self.mult.read_raw_from_device()
            
            if self.mult_reset_state:
                voltage_str = fmt.format(float(r[:-2]))       
                self.ui.mult_voltage_le.setText(voltage_str)                
            else:
                if self.mult_format == 'SREAL':
                    voltage = [_struct.unpack(
                        '>f', r[i:i+4])[0] for i in range(0, len(r), 4)]
                elif self.mult_format == 'DREAL':
                    voltage = [_struct.unpack(
                            '>d', r[i:i+8])[0] for i in range(0, len(r), 8)]
                else:
                    voltage = []
                if len(voltage) >= 1:
                    voltage_str = fmt.format(voltage[0])
                else:
                    voltage_str = ''
                self.ui.mult_voltage_le.setText(voltage_str)
        
        except Exception:
            self.ui.mult_voltage_le.setText('')
            _traceback.print_exc(file=_sys.stdout)

    def readNMRField(self):
        """Read NMR magnetic field value."""
        try:
            fmt = '{0:.10e}'
            b = self.nmr.read_b_value().strip()
            
            if b.endswith('F'):
                self.ui.nmr_field_le.setText('')
                self.ui.nmr_field_state_le.setText('')
                return
            
            b = b.replace('T', '')
            if b.startswith('L'):
                state = 'Locked'
            elif b.startswith('N'):
                state = 'Not locked'
            elif b.startswith('S'):
                state = 'Signal'
            elif b.startswith('W'):
                state = 'Wrong'
            else:
                state = ''
            
            field = float(b[1:])
            self.ui.nmr_field_le.setText(fmt.format(field))
            self.ui.nmr_field_state_le.setText(state)
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.ui.nmr_field_le.setText('')
            self.ui.nmr_field_state_le.setText('')
    
    def resetMult(self):
        """Reset Multimeter."""
        try:
            self.mult.reset()
            self.clearMultimeterVariables()
            self.mult_reset_state = True
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def setEnableMultAper(self):
        """Enable or disable multimeter aperture."""
        if self.ui.mult_aper_auto_chb.isChecked():
            self.mult_aper_sb.setEnabled(False)
        else:
            self.mult_aper_sb.setEnabled(True)

    def setEnableMultRange(self):
        """Enable or disable multimeter range."""
        if self.ui.mult_range_auto_chb.isChecked():
            self.mult_range_sb.setEnabled(False)
        else:
            self.mult_range_sb.setEnabled(True)

    def updateSerialPorts(self):
        """Update avaliable serial ports."""
        try:
            _l = [p[0] for p in _list_ports.comports()]
    
            if len(_l) == 0:
                return
    
            _ports = []
            _s = ''
            _k = str
            if 'COM' in _l[0]:
                _s = 'COM'
                _k = int
    
            for key in _l:
                _ports.append(key.strip(_s))
            _ports.sort(key=_k)
            _ports = [_s + key for key in _ports]
    
            self.ui.nmr_port_cmb.clear()
            self.ui.nmr_port_cmb.addItems(_ports)

            self.ui.mch_port_cmb.clear()
            self.ui.mch_port_cmb.addItems(_ports)
        
            self.ui.nmr_port_cmb.setCurrentIndex(len(_ports)-2)
            self.ui.mch_port_cmb.setCurrentIndex(len(_ports)-1)
            
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

