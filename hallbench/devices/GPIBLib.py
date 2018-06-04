# -*- coding: utf-8 -*-
"""GPIB communication.

Created on 10/02/2015
@author: James Citadini
"""

import visa as _visa
import logging as _logging
import numpy as _np
import struct as _struct


class GPIB(object):
    """GPIB class for communication with GPIB devices."""

    def __init__(self, logfile=None):
        """Initiate all variables and prepare log file.

        Args:
            logfile (str): log file path.
        """
        self.logger = None
        self.logfile = logfile
        self.log_events()

        self.inst = None
        self._connected = False

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

        Returns:
            True if successful, False otherwise.
        """
        try:
            # resource manager
            _rm = _visa.ResourceManager()
            # connects to the device
            _cmd = 'GPIB0::'+str(address)+'::INSTR'
            # instrument
            _inst = _rm.open_resource(_cmd)

            # check if connected
            if _inst.__str__() == ('GPIBInstrument at ' + _cmd):
                # copy reference to global variable
                self.inst = _inst
                # set a default timeout to 1
                self.inst.timeout = 1000  # ms
                self._connected = True
                return True
            else:
                self._connected = False
                return False
        except Exception:
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
            self.logger.error('exception', exc_info=True)
            return None

    def send_command(self, command):
        """Write string message to the device and check size of the answer.

        Args:
            command (str): command to be executed by the device.

        Returns:
            True if successful, False otherwise.
        """
        try:
            if self.inst.write(command, '')[0] == (len(command)):
                return True
            else:
                return False
        except Exception:
            self.logger.error('exception', exc_info=True)
            return None

    def read_from_device(self):
        """Read a string from the device.

        Stop reading when termination is detected.
        Tries to read from device, if timeout occurs, returns empty string.

        Returns:
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

        Returns:
            the string read from the device.
        """
        try:
            _reading = self.inst.read_raw()
            return _reading
        except Exception:
            return ''


class Agilent34970ACommands(object):
    """Commands of Agilent 34970 Data Acquisition/Switch Unit."""

    def __init__(self):
        """Load commands."""
        self._reset()
        self._clean()
        self._lock()
        self._remote_access()
        self._set_multichannel()

    def _reset(self):
        """Reset function."""
        self.reset = '*RST'

    def _clean(self):
        """Clean error function."""
        self.clean = '*CLS'

    def _lock(self):
        """Lock function."""
        self.lock = ':SYST:LOC'

    def _remote_access(self):
        """Remote access function."""
        self.remote = ':SYST:REM;:SYST:RWL'

    def _set_multichannel(self):
        """List of commands to set the multichannel."""
        self.route = ':ROUT:SCAN (@101:107)'
        self.conf_temp = ':CONF:TEMP RTD,85,(@101:107)'
        self.monitor_on = 'ROUT:MON:STATE ON'
        self.sense_ch1 = ':SENS:TEMP:TRAN:FRTD:RES 100.42,(@101)'
        self.sense_ch2 = ':SENS:TEMP:TRAN:FRTD:RES 100.38,(@102)'
        self.sense_ch3 = ':SENS:TEMP:TRAN:FRTD:RES 100.24,(@103)'
        self.sense_ch4 = ':SENS:TEMP:TRAN:FRTD:RES 100.20,(@104)'
        self.sense_ch5 = ':SENS:TEMP:TRAN:FRTD:RES 100.16,(@105)'
        self.sense_ch6 = ':SENS:TEMP:TRAN:FRTD:RES 102.82,(@106)'
        self.sense_ch7 = ':SENS:TEMP:TRAN:FRTD:RES 100.40,(@107)'


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
               otherwiser selects FIFO.
        """
        self.mem_off = 'MEM OFF'
        self.mem_lifo = 'MEM LIFO'
        self.mem_fifo = 'MEM FIFO'
        self.mem_cont = 'MEM CONT'

    def _number_of_power_line_cycles(self):
        """
        Number of power line cycles.

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
        a command is received, allowing thecontroller to perform other tasks
        while the multimeter executes the storedcommand.
        """
        self.inbuf_on = 'INBUF ON'
        self.inbuf_off = 'INBUF OFF'


class Agilent3458A(GPIB):
    """Agilent 3458A digital multimeter."""

    def __init__(self, logfile=None):
        """Initiate variables and prepare log file.

        Args:
            logfile (str): log file path.
        """
        if not isinstance(logfile, str):
            raise TypeError('logfile must be a string.')

        self.commands = Agilent3458ACommands()
        self.logfile = logfile
        self.end_measurement = False
        self._voltage = _np.array([])
        super().__init__(self.logfile)

    @property
    def voltage(self):
        """Voltage values read from the device."""
        return self._voltage

    def clear(self):
        """Clear voltage data."""
        self._voltage = _np.array([])

    def read_voltage(self, formtype=0):
        """Read voltage from the device.

        Args:
            formtype (int): format type [single=0 or double=1].
        """
        while (self.end_measurement is False):
            if self.inst.stb & 128:
                r = self.read_raw_from_device()
                if formtype == 0:
                    dataset = [_struct.unpack(
                        '>f', r[i:i+4])[0] for i in range(0, len(r), 4)]
                else:
                    dataset = [_struct.unpack(
                        '>d', r[i:i+8])[0] for i in range(0, len(r), 8)]
                self._voltage = _np.append(self._voltage, dataset)
        else:
            # check memory
            self.send_command(self.mcount)
            npoints = int(self.read_from_device())
            if npoints > 0:
                # ask data from memory
                self.send_command(self.rmem + str(npoints))

                for idx in range(npoints):
                    # read data from memory
                    r = self.read_raw_from_device()
                    if formtype == 0:
                        dataset = [_struct.unpack(
                            '>f', r[i:i+4])[0] for i in range(0, len(r), 4)]
                    else:
                        dataset = [_struct.unpack(
                            '>d', r[i:i+8])[0] for i in range(0, len(r), 8)]
                    self._voltage = _np.append(self._voltage, dataset)

    def config(self, aper, formtype):
        """Configure device.

        Args:
            aper (float): A/D converter integration time in seconds?
            formtype (int): format type [single=0 or double=1].
        """
        self.send_command(self.reset)
        self.send_command(self.func_volt)
        self.send_command(self.tarm_auto)
        self.send_command(self.trig_auto)
        self.send_command(self.nrdgs_ext)
        self.send_command(self.arange_off)
        self.send_command(self.range + '15')
        self.send_command(self.math_off)
        self.send_command(self.azero_once)
        self.send_command(self.trig_buffer_off)
        self.send_command(self.delay_0)
        self.send_command(self.aper + '{0:0.3f}'.format(aper))
        self.send_command(self.disp_off)
        self.send_command(self.scratch)
        self.send_command(self.end_gpib_always)
        self.send_command(self.mem_fifo)
        if formtype == 0:
            self.send_command(self.oformat_sreal)
            self.send_command(self.mformat_sreal)
        else:
            self.send_command(self.oformat_dreal)
            self.send_command(self.mformat_dreal)

    def reset(self):
        """Reset device."""
        self.send_command(self.commands.reset)


class Agilent34970A(GPIB):
    """Agilent 34970A multichannel for temperatures readings."""

    def __init__(self, logfile=None):
        """Initiate variables and prepare logging file.

        Args:
            logfile (str): log file path.
            address (int): device address.
        """
        self.commands = Agilent34970ACommands()
        self.logfile = logfile
        super().__init__(self.logfile)
