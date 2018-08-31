# -*- coding: utf-8 -*-
"""Elcomat communication.

Created on 07/12/2012
@author: James Citadini
"""

import time as _time
import serial as _serial
import logging as _logging


class ElcomatCommands(object):
    """Elcomat Commands."""

    def __init__(self):
        """Load commands."""
        self.read_relative = 'r'
        self.read_absolute = 'a'


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
            if not self.ser.is_open:
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

    def get_absolute_measurement(self, unit='arcsec'):
        """Read X-axis and Y-axis absolute values from the device."""
        try:
            self.flush()
            _time.sleep(0.01)
            self.send_command(self.commands.read_absolute)
            _time.sleep(0.01)
            _readings = self.read_from_device(n=50)
            _rlist = _readings.split('\r')[:-1]
            if len(_rlist) == 1:
                values = _rlist[0].split(' ')
                if values[0] == '4' and values[1] == '003':
                    x = float(values[2])
                    y = float(values[3])
                    if unit == 'deg':
                        x = x/3600
                        y = y/3600
                    elif unit == 'rad':
                        x = x/206264.8062471
                        y = y/206264.8062471                    
                    return (x, y)
                else:
                    return (None, None)
            else:
                return (None, None)

        except Exception:
            return (None, None)

    def get_relative_measurement(self, unit='arcsec'):
        """Read X-axis and Y-axis relative values from the device."""
        try:
            self.flush()
            _time.sleep(0.01)
            self.send_command(self.commands.read_relative)
            _time.sleep(0.01)
            _readings = self.read_from_device(n=50)
            _rlist = _readings.split('\r')[:-1]
            if len(_rlist) == 1:
                values = _rlist[0].split(' ')
                if values[0] == '2' and values[1] == '103':
                    x = float(values[2])
                    y = float(values[3])
                    if unit == 'deg':
                        x = x/3600
                        y = y/3600
                    elif unit == 'rad':
                        x = x/206264.8062471
                        y = y/206264.8062471 
                    return (x, y)
                else:
                    return (None, None)
            else:
                return (None, None)

        except Exception:
            return (None, None)
