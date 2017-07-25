# -*- coding: utf-8 -*-
"""Variables and constants to be used in Agilent 3458A."""


class ListOfCommands(object):
    """List of Commands of Agilent 3458A Digital Multimeter."""

    def __init__(self):
        """Initiate all variables."""
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
