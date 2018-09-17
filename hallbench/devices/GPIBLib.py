# -*- coding: utf-8 -*-
"""GPIB communication.

Created on 10/02/2015
@author: James Citadini
"""

import numpy as _np
import visa as _visa
import time as _time
import logging as _logging
import struct as _struct
import threading as _threading


class GPIB(object):
    """GPIB class for communication with GPIB devices."""

    def __init__(self, logfile=None):
        """Initiaze all variables and prepare log file.

        Args:
            logfile (str): log file path.
        """
        self.inst = None
        self.logger = None
        self.logfile = logfile
        self.rlock = _threading.RLock()
        self._connected = False
        self.log_events()

    @property
    def connected(self):
        """Return True if the device is connected, False otherwise."""
        return self._connected

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

    def connect(self, address):
        """Connect to a GPIB device with the given address.

        Args:
            address (int): device address.

        Return:
            True if successful, False otherwise.
        """
        try:
            # resource manager
            _rm = _visa.ResourceManager()
            # connects to the device
            _cmd = 'GPIB0::'+str(address)+'::INSTR'
            # instrument
            _inst = _rm.open_resource(_cmd.encode('utf-8'))

            # check if connected
            if _inst.__str__() == ('GPIBInstrument at ' + _cmd):
                try:
                    self.inst = _inst
                    # set a default timeout to 1
                    self.inst.timeout = 1000  # ms

                    self._connected = True
                    return True
                except Exception:
                    self.inst.close()
                    if self.logger is not None:
                        self.logger.error('exception', exc_info=True)
                    return False
            else:
                self._connected = False
                return False
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def disconnect(self):
        """Disconnect the GPIB device."""
        try:
            if self.inst is not None:
                self.inst.close()
            self._connected = False
            return True
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def send_command(self, command):
        """Write string message to the device and check size of the answer.

        Args:
            command (str): command to be executed by the device.

        Return:
            True if successful, False otherwise.
        """
        try:
            if self.inst.write(command)[0] == (len(command)):
                return True
            else:
                return False
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def read_from_device(self):
        """Read a string from the device.

        Stop reading when termination is detected.
        Tries to read from device, if timeout occurs, returns empty string.

        Return:
            the string read from the device.
        """
        try:
            _reading = self.inst.read()
            return _reading
        except Exception:
            return ''

    def read_raw_from_device(self):
        """Read a string from the device.

        Stop reading when termination is detected.
        Tries to read from device, if timeout occurs, returns empty string.

        Return:
            the string read from the device.
        """
        try:
            _reading = self.inst.read_raw()
            return _reading
        except Exception:
            return ''


class Agilent3458ACommands(object):
    """Commands of Agilent 3458A Digital Multimeter."""

    def __init__(self):
        """Load commands."""
        self._reset()
        self._function()
        self._tarm()
        self._trig()
        self._external_output()
        self._scratch()
        self._delay()
        self._aper()
        self._timer()
        self._number_of_readings()
        self._number_of_digits()
        self._math()
        self._azero()
        self._display()
        self._end_gpib()
        self._trigger_buffer()
        self._mem()
        self._number_of_power_line_cycles()
        self._memory_count_query()
        self._recall_memory()
        self._range()
        self._arange()
        self._output_format()
        self._memory_format()
        self._input_buffer()
        self._query()

    def _reset(self):
        """Reset function."""
        self.reset = 'RESET'

    def _function(self):
        """
        Select the type of measurement (AC, DC, etc).

        Implemente DCV - DC Voltage
        """
        self.func_volt = 'FUNC DCV'

    def _tarm(self):
        """
        Enable Trigger Arm.

        Types:[AUTO, EXT, SGL, HOLD, SYN]
        AUTO    - Always armed (default)
        EXT     - Arms following a low-going TTL transition on the EXT Trig
        SGL     - Arms once and then HOLD
        HOLD    - Triggering disabled
        SYN     - Arms when multimeter's output buffer is empty.
        """
        self.tarm_auto = 'TARM AUTO'
        self.tarm_ext = 'TARM EXT'
        self.tarm_sgl = 'TARM SGL'
        self.tarm_hold = 'TARM HOLD'
        self.tarm_syn = 'TARM SYN'

    def _trig(self):
        """
        Specify the trigger event.

        Types:  [AUTO, EXT, SGL, HOLD, SYN, LEVEL, LINE]
        AUTO    - Trigger whenever the multimeter is not busy (default)
        EXT     - Triggers following a low-going TTL transition on the EXT Trig
        SGL     - Triggers once and then HOLD
        HOLD    - Triggering disabled
        SYN     - Triggers when multimeter's output buffer is empty.
        LEVEL   - Triggers when the input reaches the voltage specified
                  by the LEVEL and SLOPE commands
        LINE    - Triggers on a zero crossing of the AC line voltage
        """
        self.trig_auto = 'TRIG AUTO'
        self.trig_ext = 'TRIG EXT'
        self.trig_sgl = 'TRIG SGL'
        self.trig_hold = 'TRIG HOLD'
        self.trig_syn = 'TRIG SYN'
        self.trig_level = 'TRIG LEVEL'
        self.trig_line = 'TRIG LINE'

    def _external_output(self):
        """
        Specify the external output.

        Specifie the event and polarity that will generate a signal on the
        rear panel Ext Out.

        Events:
        OFF     - disabled
        ICOMP   - Input complete
                  (1us after A/D converter has integrated each reading)
        RCOMP   - Reading complete (1us after each reading)
        APER    - Aperture Waveform

        Polarity:
        NEG     - generates a low-going TTL Signal
        POS     - generates a high-going TTL Signal
        """
        self.extout_off = 'EXTOUT OFF'
        self.extout_icomp_pos = 'EXTOUT ICOMP,POS'
        self.extout_rcomp_pos = 'EXTOUT RCOMP,POS'
        self.extout_aper_pos = 'EXTOUT APER,POS'

    def _scratch(self):
        """Clear all subprograms and stored states from memory."""
        self.scratch = 'SCRATCH'

    def _delay(self):
        """Specify delay between trigger event and first sample event."""
        self.delay = 'DELAY'
        self.delay_0 = 'DELAY 0'

    def _aper(self):
        """Specify the A/D converter integration time in seconds."""
        self.aper = 'APER '

    def _timer(self):
        """Specify the time interval for TIMER event in the NRDGS command."""
        self.timer = 'TIMER '

    def _number_of_readings(self):
        """
        Specify the number of readings taken per trigger and the event.

        Implemented only 1 reading and AUTO sample event.
        """
        self.nrdgs_auto = 'NRDGS 1,AUTO'
        self.nrdgs_syn = 'NRDGS 1,SYN'
        self.nrdgs_ext = 'NRDGS 1,EXT'
        self.nrdgs = 'NRDGS '

    def _number_of_digits(self):
        """
        Designate the number of digits to be displayed by the multimeter.

        The value can be set from 3 to 8 - only visual, not affect the A/D.
        """
        self.ndig = 'NDIG 6'

    def _math(self):
        """
        Enable or disable real time math operations.

        OFF - Disable all enabled real-time math operations
        """
        self.math_off = 'MATH OFF'

    def _azero(self):
        """
        Enable or disable the autozero function.

        OFF - Zero measured once, then stop.
        ON - Zero measurement is updated after every measurement
        ONCE - Zero measurement once, then stop. (Same as OFF)
        """
        self.azero_off = 'AZERO OFF'
        self.azero_on = 'AZERO ON'
        self.azero_once = 'AZERO ONCE'

    def _display(self):
        """Enable or disable the display."""
        self.disp_off = 'DISP OFF'
        self.disp_on = 'DISP ON'

    def _end_gpib(self):
        """
        Enable or disable the GPIB End Or Identify (EOI) function.

        OFF     - never set True
        ON      - set True the last byte of single or multiple readings
        ALWAYS  - EOI line set True when the last byte of each reading sent
        """
        self.end_gpib_off = 'END OFF'
        self.end_gpib_on = 'END ON'
        self.end_gpib_always = 'END ALWAYS'

    def _trigger_buffer(self):
        """Enable or disable external trigger buffer."""
        self.trig_buffer_on = 'TBUFF ON'
        self.trig_buffer_off = 'TBUFF OFF'

    def _mem(self):
        """
        Enable or disable reading memory and designates storage.

        OFF  - stop storing readings
        LIFO - clears reading memory and stores new, last-in-first-out
        FIFO - clears reading memory and stores new, first-in-first-out
        CONT - keeps memory intact and selects previous mode,
               otherwise selects FIFO.
        """
        self.mem_off = 'MEM OFF'
        self.mem_lifo = 'MEM LIFO'
        self.mem_fifo = 'MEM FIFO'
        self.mem_cont = 'MEM CONT'

    def _number_of_power_line_cycles(self):
        """
        Specify the number of power line cycles.

        Specifies de A/D integration time in terms of power line cycles
        The primary use stabilish normal mode noise rejection (NMR) at
        the converter's frequency (LFREQ)
        Any value > 1  provides at least 60 dB of NMR.
        """
        self.nplc = 'NPLC '

    def _memory_count_query(self):
        """Return the total number of stored readings."""
        self.mcount = 'MCOUNT?'

    def _recall_memory(self):
        """
        Return values stored in memory.

        Read and returns the value of a reading or
        a group of readings stored in memory.

        rmem leaves the stored readings intact.
        Syntax: RMEM,1 + number of reading got memory count query
        """
        self.rmem = 'RMEM 1,'

    def _range(self):
        """
        Select the measurement range or the autorange.

        RANGE [max.input][,%_resolution]
        """
        self.range_auto = 'RANGE AUTO'
        self.range = 'RANGE '

    def _arange(self):
        """Enable or disable the autorange function."""
        self.arange_on = 'ARANGE ON'
        self.arange_off = 'ARANGE OFF'
        self.arange_once = 'ARANGE ONCE'

    def _output_format(self):
        """
        Designate the GPIB output format from the device.

        Implemented:
        ASCII - ASCII-15 bytes per reading
        SREAL - Single Real(IEEE 754), 32 bits - 4 bytes per reading
        DREAL - Double Real(IEEE 754), 64 bits - 8 bytes per reading
        """
        self.oformat_ascii = 'OFORMAT ASCII'
        self.oformat_sreal = 'OFORMAT SREAL'
        self.oformat_dreal = 'OFORMAT DREAL'

    def _memory_format(self):
        """
        Set the memory format.

        Clears reading from memory and designates
        the storage format for new readings.

        Implemented:
        ASCII - ASCII-15 bytes per reading
        SREAL - Single Real(IEEE 754), 32 bits - 4 bytes per reading
        DREAL - Double Real(IEEE 754), 64 bits - 8 bytes per reading
        """
        self.mformat_ascii = 'MFORMAT ASCII'
        self.mformat_sreal = 'MFORMAT SREAL'
        self.mformat_dreal = 'MFORMAT DREAL'

    def _input_buffer(self):
        """
        Enable or disable the multimeter's input buffer.

        When enabled, the input buffer temporarily stores the commands it
        receives over the GPIB bus. This releases the bus immediately after
        a command is received, allowing the controller to perform other tasks
        while the multimeter executes the stored command.
        """
        self.inbuf_on = 'INBUF ON'
        self.inbuf_off = 'INBUF OFF'

    def _query(self):
        self.qbeep = 'BEEP?'
        self.qid = 'ID?'


class Agilent3458A(GPIB):
    """Agilent 3458A digital multimeter."""

    def __init__(self, logfile=None):
        """Initiaze variables and prepare log file.

        Args:
            logfile (str): log file path.
        """
        self.commands = Agilent3458ACommands()
        self.logfile = logfile
        super().__init__(self.logfile)

    def connect(self, address):
        """Connect to a GPIB device with the given address.

        Args:
            address (int): device address.

        Return:
            True if successful, False otherwise.
        """
        if super().connect(address):
            try:
                self.inst.write(self.commands.end_gpib_always)
                self.inst.write(self.commands.qid)
                self.inst.read()
                self._connected = True
                return True
            except Exception:
                self._connected = False
                return False
        else:
            self._connected = False
            return False

    def read_voltage(self, formtype=0):
        """Read voltage from the device.

        Args:
            formtype (int): format type [single=0 or double=1].
        """
        r = self.read_raw_from_device()
        if formtype == 0:
            voltage = [_struct.unpack(
                '>f', r[i:i+4])[0] for i in range(0, len(r), 4)]
        else:
            voltage = [_struct.unpack(
                '>d', r[i:i+8])[0] for i in range(0, len(r), 8)]
        return voltage

    def config(self, aper, formtype):
        """Configure device.

        Args:
            aper (float): A/D converter integration time in seconds?
            formtype (int): format type [single=0 or double=1].
        """
        self.send_command(self.commands.reset)
        self.send_command(self.commands.func_volt)
        self.send_command(self.commands.tarm_auto)
        self.send_command(self.commands.trig_auto)
        self.send_command(self.commands.nrdgs_ext)
        self.send_command(self.commands.arange_off)
        self.send_command(self.commands.range + '10')
        self.send_command(self.commands.math_off)
        self.send_command(self.commands.azero_once)
        self.send_command(self.commands.trig_buffer_off)
        self.send_command(self.commands.delay_0)
        self.send_command(self.commands.aper + '{0:0.3f}'.format(aper))
        self.send_command(self.commands.disp_off)
        self.send_command(self.commands.scratch)
        self.send_command(self.commands.end_gpib_always)
        self.send_command(self.commands.mem_fifo)
        if formtype == 0:
            self.send_command(self.commands.oformat_sreal)
            self.send_command(self.commands.mformat_sreal)
        else:
            self.send_command(self.commands.oformat_dreal)
            self.send_command(self.commands.mformat_dreal)

    def reset(self):
        """Reset device."""
        self.send_command(self.commands.reset)
        self.send_command(self.commands.end_gpib_always)


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


class Agilent34970A(GPIB):
    """Agilent 34970A multichannel for temperatures readings."""

    _probe_channels = ['101', '102', '103']
    _dcct_channels = ['104']
    _temperature_channels = [
        '201', '202', '203', '204', '205', '206', '207', '208', '209']

    def __init__(self, logfile=None):
        """Initiaze variables and prepare logging file.

        Args:
            logfile (str): log file path.
            address (int): device address.
        """
        self._config_channels = []
        self.commands = Agilent34970ACommands()
        self.logfile = logfile
        super().__init__(self.logfile)

    @property
    def probe_channels(self):
        """Probe temperature channels."""
        return self._probe_channels

    @property
    def temperature_channels(self):
        """Bench temperature channels."""
        return self._temperature_channels

    @property
    def dcct_channels(self):
        """DCCT current channels."""
        return self._dcct_channels

    @property
    def config_channels(self):
        """Return current channel configuration list."""
        return self._config_channels

    def connect(self, address):
        """Connect to a GPIB device with the given address.

        Args:
            address (int): device address.

        Return:
            True if successful, False otherwise.
        """
        if super().connect(address):
            try:
                self.inst.write(self.commands.qid)
                self.inst.read()
                self._connected = True
                return True
            except Exception:
                self._connected = False
                return False
        else:
            self._connected = False
            return False

    def configure(self, channel_list='all', wait=0.5):
        """Configure channels."""
        if channel_list == 'all':
            volt_channel_list = self._probe_channels + self._dcct_channels
            temp_channel_list = self._temperature_channels

        else:
            volt_channel_list = []
            temp_channel_list = []
            channel_list = [str(ch) for ch in channel_list]
            for ch in channel_list:
                if ch in self._probe_channels or ch in self._dcct_channels:
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

    def convert_voltage_to_temperature(self, voltage):
        """Convert probe voltage to temperature value."""
        temperature = (voltage + 70e-3)/20e-3
        return temperature

    def convert_voltage_to_current(self, voltage, dcct_head):
        """Convert dcct voltage to current value."""
        if dcct_head in [40, 160, 320, 600, 1125]:
            current = voltage * dcct_head/10
        else:
            current = _np.nan
        return current

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
                            self.convert_voltage_to_temperature(rd))
                    elif ch in self._dcct_channels:
                        conv_rlist.append(
                            self.convert_voltage_to_current(rd, dcct_head))
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
            rstr = self.read_from_device()
            cstr = rstr.split('(@')[1].replace(')', '').replace('\n', '')
            if len(cstr) == 0:
                return []
            else:
                channel_list = cstr.split(',')
                return channel_list
        except Exception:
            return []
