# -*- coding: utf-8 -*-

"""Display widget for the Hall Bench Control application."""

import sys as _sys
import traceback as _traceback
import numpy as _np
import serial.tools.list_ports as _list_ports
from qtpy.QtWidgets import (
    QWidget as _QWidget,
    QMessageBox as _QMessageBox,
    QApplication as _QApplication,
    )
from qtpy.QtCore import (
    Qt as _Qt,
    QThread as _QThread,
    QObject as _QObject,
    QTimer as _QTimer,
    Signal as _Signal,
    )
import qtpy.uic as _uic

from hallbench.gui.utils import get_ui_file as _get_ui_file
from hallbench.devices import (
    display as _display,
    )


class DisplayWidget(_QWidget):
    """Display widget class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)

        # setup the ui
        uifile = _get_ui_file(self)
        self.ui = _uic.loadUi(uifile, self)

        self.lcd_posx.setNumDigits(7)
        self.lcd_posx.setDigitCount(7)
        self.lcd_posx.display(0)
        self.lcd_posy.display(0)
        self.lcd_posz.display(0)
        self.lcd_posx.setEnabled(False)
        self.lcd_posy.setEnabled(False)
        self.lcd_posz.setEnabled(False)

        # Create reading thread
        self.wthread = _QThread()
        self.worker = ReadValueWorker()
        self.worker.moveToThread(self.wthread)
        self.wthread.started.connect(self.worker.run)
        self.worker.finished.connect(self.wthread.quit)
        self.worker.finished.connect(self.get_reading)

        # create timer to monitor values
        self.timer = _QTimer(self)
        self.update_monitor_interval()
        self.timer.timeout.connect(self.read_value)

        self.connect_signal_slots()
        self.update_serial_ports()

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.timer.stop()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def connect_display(self):
        try:
            port = self.ui.cmb_display_port.currentText()
            baudrate = int(self.ui.cmb_display_baudrate.currentText())

            _display.connect(
                port, baudrate, bytesize=7, stopbits=2, parity='E')
            self.ui.la_display_led.setEnabled(_display.connected)

            if not _display.connected:
                msg = 'Failed to connect devices.'
                _QMessageBox.critical(
                    self, 'Failure', msg, _QMessageBox.Ok)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            msg = 'Failed to connect devices.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def disconnect_display(self, msgbox=True):
        """Disconnect bench devices."""
        try:
            _display.disconnect()
            self.ui.la_display_led.setEnabled(_display.connected)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            if msgbox:
                msg = 'Failed to disconnect devices.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def connect_signal_slots(self):
        """Create signal/slot connections."""
        self.ui.pbt_connect.clicked.connect(self.connect_display)
        self.ui.pbt_disconnect.clicked.connect(self.disconnect_display)
        self.pbt_monitor.toggled.connect(self.monitor_value)
        self.sbd_frequency.valueChanged.connect(
            self.update_monitor_interval)

    def update_monitor_interval(self):
        """Update monitor interval value."""
        self.timer.setInterval(1000/self.sbd_frequency.value())

    def update_serial_ports(self):
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

        self.ui.cmb_display_port.clear()
        self.ui.cmb_display_port.addItems(_ports)

        idx = self.ui.cmb_display_port.findText('COM6')
        self.ui.cmb_display_port.setCurrentIndex(idx)

    def get_reading(self):
        """Get reading from worker thread."""
        if not self.ui.pbt_monitor.isChecked():
            return

        try:
            if self.worker.posx is None:
                self.lcd_posx.setEnabled(False)
            else:
                self.lcd_posx.setEnabled(True)
                self.lcd_posx.display(self.worker.posx)

            if self.worker.posy is None:
                self.lcd_posy.setEnabled(False)
            else:
                self.lcd_posy.setEnabled(True)
                self.lcd_posy.display(self.worker.posy)

            if self.worker.posz is None:
                self.lcd_posz.setEnabled(False)
            else:
                self.lcd_posz.setEnabled(True)
                self.lcd_posz.display(self.worker.posz)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def monitor_value(self, checked):
        """Monitor values."""
        if checked:
            self.timer.start()
        else:
            self.timer.stop()
            self.lcd_posx.display(0)
            self.lcd_posy.display(0)
            self.lcd_posz.display(0)
            self.lcd_posx.setEnabled(False)
            self.lcd_posy.setEnabled(False)
            self.lcd_posz.setEnabled(False)

    def read_value(self):
        """Read value."""
        if not _display.connected:
            return

        try:
            self.wthread.start()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)


class ReadValueWorker(_QObject):
    """Read values worker."""

    finished = _Signal([bool])

    def __init__(self):
        """Initialize object."""
        self.posx = None
        self.posy = None
        self.posz = None
        super().__init__()

    def run(self):
        """Read values from devices."""
        try:
            self.posx = None
            self.posy = None
            self.posz = None

            px, py, pz = _display.read_display()

            if _np.isnan(px):
                self.posx = None
            else:
                self.posx = px

            if _np.isnan(py):
                self.posy = None
            else:
                self.posy = py

            if _np.isnan(pz):
                self.posz = None
            else:
                self.posz = pz

            self.finished.emit(True)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.finished.emit(True)
