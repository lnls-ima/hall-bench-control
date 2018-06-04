# -*- coding: utf-8 -*-
"""NMR Library."""

import time as _time
import serial as _serial
import logging as _logging
import threading as _threading


class NMRCommands(object):
    """PT 2025 NMR Teslameter Commands."""

    def __init__(self):
        """Load commands."""
        self._operation_mode()
        self._aquisition_mode()
        self._frequency_selection()
        self._field_sense_selection()
        self._channel_selection()
        self._search_mode()
        self._reset_time_base()
        self._display()
        self._status()
        self._read()

    def _operation_mode(self):
        self.remote = 'R'  # Remote mode (disables the front panel)
        self.local = 'L'  # Local mode (enables the front panel)

    def _aquisition_mode(self):
        self.aquisition = 'A'  # Manual(0) or Auto(1)

    def _frequency_selection(self):
        self.frequency = 'C'  # value is expressed in decimal

    def _field_sense_selection(self):
        self.field_sense = 'F'  # Negative(0) or Positive(1) fields

    def _channel_selection(self):
        self.channel = 'P'  # A, B, C, D, E, F, G or H channels

    def _search_mode(self):
        self.search = 'H'  # activates the automatic field-searching algorithm
        self.quit_search = 'Q'  # inactivates the search mode
        self.nr_channels = 'X'  # number of channels to be used in search
        self.search_time = 'O'  # search time (n=1 -> 9s per probe)

    def _reset_time_base(self):
        self.reset_time = 'T'  # Reset NMR time-base

    def _display(self):
        self.display_mode = 'D'  # displayed value given in MHz(0) or Tesla(1)
        self.display_vel = 'V'  # Normal(0) or Fast(1)

    def _status(self):
        self.status = 'S'  # returns the status (1 Byte)

    def _read(self):
        self.read = '\x05'  # Leitura. Formato: vdd.ddddddF/T


class NMR(object):
    """Class for communication with NMR device."""

    _bytesize = _serial.EIGHTBITS
    _stopbits = _serial.STOPBITS_ONE
    _parity = _serial.PARITY_NONE
    _timeout = 0.015  # [s]

    def __init__(self, logfile=None):
        """Initiate all variables and prepare log file.

        Args:
            logfile (str): log file path.
        """
        self.logger = None
        self.logfile = logfile
        self.log_events()

        self.commands = NMRCommands()
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

        Returns:
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
            return True
        except Exception:
            self.logger.error('exception', exc_info=True)
            return None

    def disconnect(self):
        """Disconnect the NMR device.

        Returns:
            True if successful.
        """
        if self.ser is None:
            return True

        try:
            self.send_command(self.commands.quit_search)
            _time.sleep(0.5)
            self.send_command(self.commands.local)
            _time.sleep(0.05)
            self.ser.close()
            return True
        except Exception:
            self.logger.error('exception', exc_info=True)
            return None

    def configure(
            self, frequency, aquisition, field_sense, display_mode,
            display_vel, search_time, channel, nr_channels=1):
        """Configure NMR.

        Args:
            frequency (int or str): initial search frequency,
            aquisition (int or str): aquisition mode [Manual(0) or Auto(1)],
            field_sense (int or str): field sense [Negative(0) or Positive(1)],
            display_mode (int or str): display mode [MHz(0) or Tesla(1)],
            display_vel (int or str): display velocity [Normal(0) or Fast(1)],
            search_time (int or str): search time [n=1 -> 9s per probe],
            channel (str or str): initial search channel,
            nr_channels (int or str): number of channels.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self.rlock.acquire()
            self.send_command(self.commands.remote)
            _time.sleep(0.01)

            self.send_command(self.commands.channel + str(channel))
            _time.sleep(0.01)

            self.send_command(self.commands.nr_channels + str(nr_channels))
            _time.sleep(0.01)

            self.send_command(self.commands.frequency+str(frequency)+'\r\n')
            _time.sleep(0.01)

            self.send_command(self.commands.aquisition + str(aquisition))
            _time.sleep(0.01)

            self.send_command(self.commands.field_sense + str(field_sense))
            _time.sleep(0.01)

            self.send_command(self.commands.display_mode + str(display_mode))
            _time.sleep(0.01)

            self.send_command(self.commands.display_vel + str(display_vel))
            _time.sleep(0.01)

            self.send_command(self.commands.search_time + str(search_time))
            _time.sleep(0.01)

            self.rlock.release()
            return True

        except Exception:
            self.rlock.release()
            self.logger.error('exception', exc_info=True)
            return False

    def send_command(self, command):
        """Write string message to the device and check size of the answer.

        Args:
            command (str): command to be executed by the device.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if self.ser.write(command.encode('utf-8')) == len(command):
                return True
            else:
                return False
        except Exception:
            self.logger.error('exception', exc_info=True)
            return None

    def read_from_device(self):
        """Read a string from the device.

        Tries to read from device, if timeout occurs, returns empty string.

        Returns:
            the string read from the device.
        """
        try:
            self.ser.write(self.commands.read.encode('utf-8'))
            _reading = self.ser.read(20).decode('utf-8')
            return _reading
        except Exception:
            return ''
