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
        self.display_unit = 'D'  # displayed value given in MHz(0) or Tesla(1)
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

    @property
    def locked(self):
        """Return lock status."""
        _ans = self.read_status(1)
        if _ans is not None:
            return int(_ans[-6])
        else:
            return None

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

    def connect(self, port='COM4', baudrate=19200):
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
        """Disconnect the NMR device.

        Return:
            True if successful.
        """
        if self.ser is None:
            return True

        try:
            self.send_command(self.commands.quit_search)
            _time.sleep(0.5)
            self.send_command(self.commands.local)
            _time.sleep(self._delay)
            self.ser.close()
            return True
        except Exception:
            if self.logfile is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def configure(
            self, frequency, aquisition, field_sense, display_unit,
            display_vel, search_time, channel, nr_channels=1):
        """Configure NMR.

        Args:
            frequency (int or str): initial search frequency,
            aquisition (int or str): aquisition mode [Manual(0) or Auto(1)],
            field_sense (int or str): field sense [Negative(0) or Positive(1)],
            display_unit (int or str): display unit [MHz(0) or Tesla(1)],
            display_vel (int or str): display velocity [Normal(0) or Fast(1)],
            search_time (int or str): search time [n=1 -> 9s per probe],
            channel (str or str): initial search channel,
            nr_channels (int or str): number of channels.

        Return:
            True if successful, False otherwise.
        """
        try:
            with self.rlock:
                self.rlock.acquire()
                self.send_command(self.commands.remote)
                _time.sleep(self._delay)

                self.send_command(self.commands.channel + str(channel))
                _time.sleep(self._delay)

                self.send_command(self.commands.nr_channels + str(nr_channels))
                _time.sleep(self._delay)

                self.send_command(self.commands.frequency+str(frequency))
                _time.sleep(self._delay)

                self.send_command(self.commands.aquisition + str(aquisition))
                _time.sleep(self._delay)

                self.send_command(self.commands.field_sense + str(field_sense))
                _time.sleep(self._delay)

                self.send_command(
                    self.commands.display_unit + str(display_unit))
                _time.sleep(self._delay)

                self.send_command(self.commands.display_vel + str(display_vel))
                _time.sleep(self._delay)

                self.send_command(self.commands.search_time + str(search_time))
                _time.sleep(self._delay)

            return True

        except Exception:
            self.rlock.release()
            if self.logfile is not None:
                self.logger.error('exception', exc_info=True)
            return False

    def send_command(self, command):
        """Write string message to the device and check size of the answer.

        Args:
            command (str): command to be executed by the device.

        Return:
            True if successful, False otherwise.
        """
        try:
            self.ser.flushInput()
            self.ser.flushOutput()
            command = command + '\r\n'
            if self.ser.write(command.encode('utf-8')) == len(command):
                return True
            else:
                return False
        except Exception:
            if self.logfile is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def read_b_value(self):
        """Read magnetic field value from the device.

        Return:
            B value (str): string read from the device. First character
        indicates lock state ('L' for locked, 'N' for not locked and 'S' to
        NMR signal seen. The last character indicates the unit ('T' for Tesla
        and 'F' for MHz). Iif timeout occurs, returns empty string.
        """
        try:
            self.send_command(self.commands.read)
            _time.sleep(self._delay)
            _reading = self.ser.read_all().decode('utf-8')
            return _reading
        except Exception:
            if self.logfile is not None:
                self.logger.error('exception', exc_info=True)
            return ''

    def read_status(self, register=1):
        """Read a status register from PT2025 NMR.

        Args:
            register (int): register number (from 1 to 4)

        Return:
            string of register bits
        """
        try:
            if 0 < register < 5:
                self.send_command(self.commands.status + str(register))
                _time.sleep(self._delay)
                _ans = self.ser.read_all().decode().strip('S\r\n')
                if _ans == '':
                    return None
                _ans = bin(int(_ans, 16))[2:]
                return '{0:>8}'.format(_ans).replace(' ', '0')
            else:
                print('Register number out of range.')
                return None
        except Exception:
            raise
            if self.logfile is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def read_dac_value(self):
        """Return current internal 12-bit DAC value (from 0 to 4095)."""
        try:
            self.send_command(self.commands.status + '4')
            _time.sleep(self._delay)
            _ans = self.ser.read_all().decode().strip('S\r\n')
            _ans = int(_ans, 16)
            return _ans
        except Exception:
            if self.logfile is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def scan(self, channel='D', dac=0, speed=3):
        """Scan the selected probe to determine the magnetic field value.

        Times out after 30 seconds if the probe does not lock.

        Args:
            channel (str): selects the probe channel (from 'A' to 'H').
            dac (int): selects intial 12 bits DAC value (from 0 to 4095).
            speed (int): search time (from 1 to 6). 1 is the fastest value
        (9 seconds to scan over the entire probe range). Each unit increase
        in the speed increases the search time by 3 seconds.

        Return:
            (B value, DAC value, dt)
            B value (str): measured magnetic field in Tesla.
            DAC value (int): current DAC value.
            dt (float): measurement duration in seconds or -1 if timed out.
        """
        self.send_command(self.commands.channel + channel)
        _time.sleep(self._delay)

        self.send_command(self.commands.aquisition + '1')
        _time.sleep(self._delay)

        self.send_command(self.commands.display_unit + '1')
        _time.sleep(self._delay)

        self.send_command(self.commands.search_time + str(speed))
        _time.sleep(self._delay)

        self.send_command(self.commands.search + str(dac))
        _t0 = _time.time()
        _time.sleep(self._delay)

        _dt = 0
        while not self.locked:
            _time.sleep(0.1)
            _dt = _time.time() - _t0
            if _dt > 30:
                _dt = -1
                break

        _b = float(self.read_b_value().strip('LNST\r\n'))
        _dac = self.read_dac_value()

        self.send_command(self.commands.quit_search)

        return _b, _dac, _dt
