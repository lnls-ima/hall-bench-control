# -*- coding: utf-8 -*-

"""Main window for the Hall Bench Control application."""

import sys as _sys
import traceback as _traceback
import serial.tools.list_ports as _list_ports
from qtpy.QtWidgets import (
    QApplication as _QApplication,
    QDesktopWidget as _QDesktopWidget,
    QMainWindow as _QMainWindow,
    QMessageBox as _QMessageBox,
    )
import qtpy.uic as _uic

from hallbench.calibration.utils import getUiFile as _getUiFile


class CalibrationWindow(_QMainWindow):
    """Main Window class for the Hall Bench Calibration application."""

    def __init__(self, parent=None, width=1200, height=700):
        """Set up the ui and add main tabs."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)
        self.resize(width, height)

        self.nmr_enable = False
        self.nmr_port = None
        self.nmr_baudrate = None
        self.mult_enable = False
        self.mult_address = None

        self.updateSerialPorts()
        self.ui.connect_btn.clicked.connect(self.connectDevices)
        self.ui.disconnect_btn.clicked.connect(self.disconnectDevices)

    @property
    def nmr(self):
        """NMR."""
        return _QApplication.instance().nmr

    @property
    def mult(self):
        """Multimeter."""
        return _QApplication.instance().mult

    def centralizeWindow(self):
        """Centralize window."""
        window_center = _QDesktopWidget().availableGeometry().center()
        self.move(
            window_center.x() - self.geometry().width()/2,
            window_center.y() - self.geometry().height()/2)

    def closeEvent(self, event):
        """Close main window and dialogs."""
        try:
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
            self.mult_enable = self.ui.mult_enable_chb.isChecked() 
            self.mult_address = self.ui.mult_address_sb.value()           
            
            success = True
            if self.nmr_enable:
                self.nmr.connect(self.nmr_port, self.nmr_baudrate)
                if self.nmr.connected:
                    self.nmr_led_la.setEnabled(True)
                else:
                    self.nmr_led_la.setEnabled(False)
                    success = False
            
            if self.mult_enable:
                self.mult.connect(self.mult_address)
                if self.mult.connected:
                    self.mult_led_la.setEnabled(True)
                else:
                    self.mult_led_la.setEnabled(False)
                    success = False

            if not success:
                msg = 'Failed to connect devices.'
                _QMessageBox.critical(
                    self, 'Failure', msg, _QMessageBox.Ok)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to connect devices.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def disconnectDevices(self):
        """Disconnect bench devices."""
        try:
            self.nmr.disconnect()
            if self.nmr.connected:
                self.nmr_led_la.setEnabled(True)
            else:
                self.nmr_led_la.setEnabled(False)

            self.mult.disconnect()
            if self.mult.connected:
                self.mult_led_la.setEnabled(True)
            else:
                self.mult_led_la.setEnabled(False)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to disconnect devices.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def updateSerialPorts(self):
        """Update avaliable serial ports."""
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
