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
            return self.ser.is_open

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
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

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


class Agilent34970ACommands(object):
    """Commands of Agilent 34970 Data Acquisition/Switch Unit."""

    def __init__(self):
        """Load commands."""
        self._reset()
        self._clean()
        self._lock()
        self._configure()
        self._query()

    def _reset(self):
        """Reset function."""
        self.reset = '*RST'

    def _clean(self):
        """Clean error function."""
        self.clean = '*CLS'

    def _lock(self):
        """Lock function."""
        self.lock = ':SYST:LOC'

    def _configure(self):
        """Configure commands."""
        self.rout_scan = ':ROUT:SCAN'
        self.conf_temp = ':CONF:TEMP FRTD,'
        self.conf_volt = ':CONF:VOLT:DC'

    def _query(self):
        """Query commands."""
        self.qid = '*IDN?'
        self.qread = ':READ?'
        self.qscan = ':ROUT:SCAN?'
        self.qscan_size = ':ROUT:SCAN:SIZE?'


class Agilent34970A():
    """Agilent 34970A multichannel for temperatures readings."""

    _probe_channels = ['101', '102', '103', '105']
    _temperature_channels = ['201']
    _bytesize = _serial.EIGHTBITS
    _stopbits = _serial.STOPBITS_ONE
    _parity = _serial.PARITY_NONE
    _timeout = 0.3  # [s]

    def __init__(self, logfile=None):
        """Initiaze variables and prepare logging file.

        Args:
            logfile (str): log file path.
            address (int): device address.
        """
        self._config_channels = []
        self.commands = Agilent34970ACommands()
        self.logfile = logfile
        self.ser = None
        self.log_events()

    @property
    def connected(self):
        """Return True if the port is open, False otherwise."""
        if self.ser is None:
            return False
        else:
            return self.ser.is_open

    @property
    def probe_channels(self):
        """Probe temperature channels."""
        return self._probe_channels

    @property
    def temperature_channels(self):
        """Bench temperature channels."""
        return self._temperature_channels

    @property
    def config_channels(self):
        """Return current channel configuration list."""
        return self._config_channels

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
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

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

    def configure(self, channel_list='all', wait=0.5):
        """Configure channels."""
        if channel_list == 'all':
            volt_channel_list = self._probe_channels
            temp_channel_list = self._temperature_channels

        else:
            volt_channel_list = []
            temp_channel_list = []
            channel_list = [str(ch) for ch in channel_list]
            for ch in channel_list:
                if ch in self._probe_channels:
                    volt_channel_list.append(ch)
                else:
                    temp_channel_list.append(ch)

        all_channels = sorted(volt_channel_list + temp_channel_list)
        if len(all_channels) == 0:
            return False
        elif all_channels == self._config_channels:
            return True

        try:
            self.send_command(self.commands.clean)
            self.send_command(self.commands.reset)

            _cmd = ''
            if len(volt_channel_list) != 0:
                volt_scanlist = '(@' + ','.join(volt_channel_list) + ')'
                _cmd = _cmd + self.commands.conf_volt + ' ' + volt_scanlist

            if len(temp_channel_list) != 0:
                if len(_cmd) != 0:
                    _cmd = _cmd + '; '
                temp_scanlist = '(@' + ','.join(temp_channel_list) + ')'
                _cmd = _cmd + self.commands.conf_temp + ' ' + temp_scanlist

            self.send_command(_cmd)
            _time.sleep(wait)
            scanlist = '(@' + ','.join(all_channels) + ')'
            self.send_command(self.commands.rout_scan + ' ' + scanlist)
            _time.sleep(wait)
            self._config_channels = all_channels.copy()
            return True

        except Exception:
            return False

    def convert_voltage_to_temperature(self, voltage, channel):
        """Convert probe voltage to temperature value."""
        if channel == '101':
            temperature = (voltage + 50e-3)/20e-3
        elif channel == '102':
            temperature = (voltage + 40e-3)/20e-3
        elif channel == '103':
            temperature = (voltage + 50e-3)/20e-3
        elif channel == '105':
            temperature = (voltage + 30e-3)/20e-3
        else:
            temperature = _np.nan
        return temperature

    def get_readings(self, wait=0.5):
        """Get reading list."""
        try:
            self.send_command(self.commands.qread)
            _time.sleep(wait)
            rstr = self.read_from_device()
            if len(rstr) != 0:
                rlist = [float(r) for r in rstr.split(',')]
                return rlist
            else:
                return []
        except Exception:
            return []

    def get_converted_readings(self, dcct_head=None, wait=0.5):
        """Get reading list and convert voltage values."""
        try:
            self.send_command(self.commands.qread)
            _time.sleep(wait)
            rstr = self.read_from_device()

            if len(rstr) != 0:
                rlist = [float(r) for r in rstr.split(',')]
                conv_rlist = []
                for i in range(len(rlist)):
                    ch = self._config_channels[i]
                    rd = rlist[i]
                    if ch in self._probe_channels:
                        conv_rlist.append(
                            self.convert_voltage_to_temperature(rd, ch))
                    else:
                        conv_rlist.append(rd)
                return conv_rlist
            else:
                return []
        except Exception:
            return []

    def get_scan_channels(self, wait=0.1):
        """Return the scan channel list read from the device."""
        try:
            self.send_command(self.commands.qscan)
            _time.sleep(wait)
            rstr = self.read_from_device().replace('\r', '')
            cstr = rstr.split('(@')[1].replace(')', '').replace('\n', '')
            if len(cstr) == 0:
                return []
            else:
                channel_list = cstr.split(',')
                return channel_list
        except Exception:
            return []