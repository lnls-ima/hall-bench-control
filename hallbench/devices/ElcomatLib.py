# -*- coding: utf-8 -*-
"""Elcomat communication.

Created on 07/12/2012
@author: James Citadini
"""

import re as _re
import time as _time
import numpy as _np
import serial as _serial
import ctypes as _ctypes
import logging as _logging
import threading as _threading


class ElcomatCommands(object):
    """Elcomat Commands."""

    def __init__(self):
        """Load commands."""
        self.read = 'r'
    

class Elcomat(object):
    """Class for communication with Elcomat device."""

    _bytesize = _serial.EIGHTBITS
    _stopbits = _serial.STOPBITS_ONE
    _parity = _serial.PARITY_NONE
    _timeout = 0.3  # [s]
    _delay = 0.05

    def __init__(self, logfile=None):
        """Initiaze all variables and prepare log file.

        Args:
            logfile (str): log file path.
        """
        self.logger = None
        self.logfile = logfile
        self.log_events()

        self.commands = ElcomatCommands()
        self.rlock = _threading.RLock()
        self.ser = None

    @property
    def connected(self):
        """Return True if the port is open, False otherwise."""
        if self.ser is None:
            return False
        else:
            return self.ser.isOpen()

    def log_events(self):
        """Prepare log file to save info, warning and error status."""
        if self.logfile is not None:
            formatter = _logging.Formatter(
                fmt='%(asctime)s\t%(levelname)s\t%(message)s',
                datefmt='%m/%d/%Y %H:%M:%S')
            fileHandler = _logging.FileHandler(self.logfile, mode='w')
            fileHandler.setFormatter(formatter)
            logname = self.logfile.replace('.log', '')
            self.logger = _logging.getLogger(logname)
            self.logger.addHandler(fileHandler)
            self.logger.setLevel(_logging.ERROR)

    def connect(self, port, baudrate):
        """Connect to a serial port.

        Args:
            port (str): device port,
            baudrate (int): baud rate.

        Return:
            True if successful.
        """
        try:
            self.ser = _serial.Serial(port=port, baudrate=baudrate)
            self.ser.bytesize = self._bytesize
            self.ser.stopbits = self._stopbits
            self.ser.parity = self._parity
            self.ser.timeout = self._timeout
            if not self.ser.isOpen():
                self.ser.open()
            self.send_command(self.commands.remote)
            return True
        except Exception:
            if self.logfile is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def disconnect(self):
        """Disconnect the GPIB device."""
        try:
            if self.ser is not None:
                self.ser.close()
            self._connected = False
            return True
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None
   
    def configure(self):
        pass

    def flush(self):
        """Clear input and output."""
        self.ser.flushInput()
        self.ser.flushOutput()        

    def send_command(self, command):
        """Write string message to the device and check size of the answer.

        Args:
            command (str): command to be executed by the device.

        Return:
            True if successful, False otherwise.
        """
        try:
            command = command + '\r\n'
            if self.ser.write(command.encode('utf-8')) == len(command):
                return True
            else:
                return False
        except Exception:
            if self.logfile is not None:
                self.logger.error('exception', exc_info=True)
            return None
 
    def read_from_device(self, n=100):
        """Read a string from the device.

        Stop reading when termination is detected.
        Tries to read from device, if timeout occurs, returns empty string.

        Return:
            the string read from the device.
        """
        try:
            _reading = self.ser.read(n).decode('utf-8')
            return _reading
        except Exception:
            return ''
        
    def get_measurement_lists(self):
        """Read X-axis and Y-axis values from the device."""
        xvals = []
        yvals = []
        try:
            self.send_command(self.commands.read)
            _readings = self.read_from_device(n=1000)
            _rlist = _readings.split('\r')[:-1]
            if len(_rlist) != 0:          
                for _r in _rlist:
                    values = _r.split(' ')
                    x = float(values[2])
                    y = float(values[3])
                    xvals.append(x)
                    yvals.append(y)
                return xvals, yvals
            
            else:
                return [], []
        
        except Exception:
            return [], []

              