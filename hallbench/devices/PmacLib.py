# -*- coding: utf-8 -*-
"""Pmac Lib."""

import ctypes as _ctypes
import logging as _logging


class PmacCommands(object):
    """Commands of Pmac motion controller."""

    def __init__(self):
        """Load commands."""
        self._axis()
        self._constants()
        self._mvariables()
        self._qvariables()
        self._qvariables_trig()
        self._pvariables()
        self._ivariables()
        self._jogging()
        self._motor_reporting()
        self._run_programs_commands()
        self._misc()

    def _axis(self):
        self.list_of_axis = [1, 2, 3, 5, 6, 7, 8, 9]
        self.stop_all_axis = chr(1)
        self.kill_all_axis = chr(11)

    def _constants(self):
        """List of constants to convert counts to mm."""
        self.CTS_MM_AXIS = [20000,
                            100000,
                            100000,
                            0,
                            8192,
                            400,
                            400,
                            400,
                            400]

    def _mvariables(self):
        """M-variables.

        Inputs:
        DI_eStopOK          - State of E-stop relays; 1 = OK, 0 = OFF
        DI_inputPressureOK  - Monitoring input pressure; 0 = fault, 1 = OK
        DI_vacuumOK         - Monitoring vac; 0 = fault, 1 = OK
        prog_running        - Check if there is a programming running

        Outputs:
        None
        """
        self.DI_eStopOK = 'M7000'
        self.DI_inputPressureOK = 'M7004'
        self.DI_vacuumOK = 'M7012'
        self.prog_running = 'M5180'

    def _qvariables(self):
        """Q-variables.

        q_xAxisManualMode - Manual move mode for Z-axis is ON / 0 = Normal mode
        q_motorMask - Bit mask to select the motors
        q_plc5_status - Status of PLC 5
        q_plc10status - Status of PLC 10
        """
        self.q_xAxisManualMode = 'Q90'
        self.q_motorMask = 'Q95'
        self.q_plc5_status = 'q5500'
        self.q_plc10status = 'q6000'

    def _qvariables_trig(self):
        """Trigger Q-varibles.

        q_selectedMotor = 'Q0'    - [1,2,3,5] trigger source, motor number
        q_incremment_mm = 'Q1'    - [mm] trigger pitch
                                    (negative numbers also possible)
        q_loopCount = 'Q2'        - [1] trigger counter
        q_plc0Status = 'Q3'       - Status of plc0
        q_plc0RunControl = 'Q9'   - [1] for starting stopping of plc0
        q_useProgStartPos = 'Q10' - [1/0] to use flexible start position
        q_startPos = 'Q11'        - [mm] position of first pulse if flexible
                                    start position is used
        q_pulseWidth_perc = 'Q12' - [0..100%] pulse width in %
                                    (internally limited to min 10% and max 75%)
        q_maxPulses = 'Q13'       - [0..] max number of pulses
                                    (0 for no limitation)
        q_fallingEdge = 'Q14'     - [0/1] trigger edge
                                    1 = falling edge, 0 = rising edge
        """
        self.q_selectedMotor = 'Q0'
        self.q_incremment_mm = 'Q1'
        self.q_loopCount = 'Q2'
        self.q_plc0Status = 'Q3'
        self.q_plc0RunControl = 'Q9'
        self.q_useProgStartPos = 'Q10'
        self.q_startPos = 'Q11'
        self.q_pulseWidth_perc = 'Q12'
        self.q_maxPulses = 'Q13'
        self.q_fallingEdge = 'Q14'

    def _pvariables(self):
        """P-variables.

        p_axis_mask - Bit mask to select the motors to be homed - b1200r
        p_homing_status - Homing status
        """
        self.p_axis_mask = 'P810'
        self.p_homing_status = 'P813'

    def _ivariables(self):
        """I-Variables.

        i_pos_scale_factor   - Ixx08 Motor xx Position Scale Factor
        i_softlimit_pos_list - List of positive software position limit
                               [motor counts] - Ixx13
        i_softlimit_neg_list - List of negative software position limit
                               [motor counts] - Ixx14
        i_axis_speed         - List of all axis speed - Ixx22 in counts/msec
        """
        self.i_pos_scale_factor = ['I'+str(i)+'08' for i in range(1, 10)]
        self.i_softlimit_pos_list = ['I'+str(i)+'13' for i in range(1, 10)]
        self.i_softlimit_neg_list = ['I'+str(i)+'14' for i in range(1, 10)]
        self.i_axis_speed = ['I'+str(i)+'22' for i in range(1, 10)]

    def _jogging(self):
        """Jogging commands.

        jog_pos          - Jog motor indefinitely in positive direction
        jog_neg          - Jog motor indefinitely in negative direction
        jog_stop         - Stop jog
        jog_abs_position - Jog to absolute position
        jog_rel_position - Jog to relative position
        """
        self.jog_pos = 'j+'
        self.jog_neg = 'j-'
        self.jog_stop = 'j/'
        self.jog_abs_position = 'j='
        self.jog_rel_position = 'j:'

    def _motor_reporting(self):
        """Motor reporting commands.

        current_position - Report position of motor in counts
        current_velocity - Report velocity of motor
        """
        self.current_position = 'p'
        self.current_velocity = 'v'

    def _run_programs_commands(self):
        """Run programs.

        align_axis - Run routine for alignment of actived axis
        """
        self.rp_align_axis = 'b1200r'

    def _misc(self):
        """Miscellaneous.

        enplc5   = 'enaplc5' # Enable (run) PLC5
        enplc10  = 'enaplc10' # Enable (run) PLC10
        displc5  - Disable (stop) PLC5
        displc10 - Disable (stop) PLC10
        enaplc2  = 'enaplc2'  # Enable (run) PLC2
        """
        self.enplc5 = 'enaplc5'
        self.enplc10 = 'enaplc10'
        self.displc5 = 'displc5'
        self.displc10 = 'displc10'
        self.enaplc2 = 'enaplc2'
        self.axis_status = '?'


class Pmac(object):
    """Implementation of the main commands to control the bench."""

    def __init__(self, logfile=None):
        """Initiate all function variables."""
        self.logger = None
        self.logfile = logfile
        self.log_events()

        # load commands
        self.commands = PmacCommands()

        # start goblal variables
        self._pmacdll = None
        self._value = ''
        self._connected = False

    @property
    def connected(self):
        """Return True if the device is connected, False otherwise."""
        return self._connected

    def log_events(self):
        """Prepare logging file to save info, warning and error status."""
        if self.logfile is not None:
            formatter = _logging.Formatter(
                fmt='%(asctime)s\t%(levelname)s\t%(message)s',
                datefmt='%m/%d/%Y %H:%M:%S')
            fileHandler = _logging.FileHandler(self.logfile, mode='w')
            fileHandler.setFormatter(formatter)
            logname = self.logfile.replace('.log', '')
            self.logger = _logging.getLogger(logname)
            self.logger.addHandler(fileHandler)
            self.logger.setLevel(_logging.DEBUG)

    def load_dll(self):
        """Load dll file PComm32W.dll to control the bench.

        Returns:
            True if successful, False otherwise.
        """
        try:
            self._pmacdll = _ctypes.windll.LoadLibrary('PComm32W.dll')
            return True
        except Exception:
            if self.logger is not None:
                self.logger.error('Fail to connect to dll')
            return False

    def connect(self):
        """Connect to Pmac device - OpenPmacDevice(0)."""
        if self._pmacdll is None:
            if not self.load_dll():
                return False

        try:
            status = bool(self._pmacdll.OpenPmacDevice(0))
            self._connected = status
            return status
        except Exception:
            return False

    def disconnect(self):
        """Disconnect Pmac device - ClosePmacDevice(0)."""
        if self._pmacdll is None:
            return True

        try:
            status = bool(self._pmacdll.ClosePmacDevice(0))
            if status is None:
                self._connected = None
            else:
                self._connected = not status
            return status
        except Exception:
            return None

    def lock_pmac(self):
        """Lock Pmac to avoid multiple operations - LockPmac(0)."""
        try:
            return self._pmacdll.LockPmac(0)
        except Exception:
            return None

    def release_pmac(self):
        """Release Pmac - ReleasePmac(0)."""
        try:
            return self._pmacdll.ReleasePmac(0)
        except Exception:
            return None

    def set_par(self, input_par, value):
        """Create string with the desired value."""
        try:
            _parameter = input_par + '=' + str(value)
            return _parameter
        except Exception:
            return input_par

    def get_response(self, str_command):
        """
        Get response of the string command from Pmac device - PmacExA.

        Returns True or False and the resulted value when available
        """
        try:
            MASK_STATUS = 0xF0000000
            COMM_EOT = 0x80000000
            # An acknowledge character (ACK ASCII 9) was received
            # indicating end of transmission from PMAC to Host PC.

            maxchar = 16
            # create empty string with n*maxchar
            response = (' '*maxchar).encode('utf-8')

            # send command and get pmac response
            _retval = self._pmacdll.PmacGetResponseExA(
                0,
                response,
                maxchar,
                str_command.encode('utf-8'))

            # check the status and if it matches with the
            # acknowledge character COMM_EOT
            if _retval & MASK_STATUS == COMM_EOT:
                result = response.decode('utf-8')
                # erase all result after /r
                self._value = result[0:result.find('\r')]

                return True
            else:
                return False
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def read_response(self, str_command):
        """
        Get response of a variable.

        Return the result instead of status.
        """
        try:
            if self.get_response(str_command):
                return self._value
            else:
                return ''
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def activate_bench(self):
        """
        Activate the bench.

        Set the mask value to 503 in q95 and enable plcs 5 and 10.
        """
        try:
            _cmd = self.set_par(self.commands.q_motorMask, 503)
            if self.get_response(_cmd):
                if self.get_response(self.commands.enplc5):
                    if self.get_response(self.commands.enplc10):
                        return True
            return False
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def axis_status(self, axis):
        """Get axis status."""
        try:
            _cmd = '#' + str(axis) + self.commands.axis_status
            if self.get_response(_cmd):
                status = int(self._value, 16)
                return status
            return None
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def align_bench(self, axis_mask):
        """Set the mask of the axis to be aligned and run plc script."""
        try:
            _cmd = self.set_par(self.commands.p_axis_mask, axis_mask)
            if self.get_response(_cmd):
                if self.get_response(self.commands.p_axis_mask):
                    if int(self._value) == axis_mask:
                        if self.get_response(self.commands.rp_align_axis):
                            return True
                        else:
                            if self.logger is not None:
                                self.logger.warning('Fail to set P_axis_mask')
                    else:
                        if self.logger is not None:
                            self.logger.warning('Fail to set P_axis_mask')
            return False
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def get_position(self, axis):
        """Read the current position in counter and convert to mm."""
        try:
            _cmd = '#' + str(axis) + self.commands.current_position
            if self.get_response(_cmd):
                _pos = float(self._value) / self.commands.CTS_MM_AXIS[axis-1]
                return _pos
            else:
                return None
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def get_velocity(self, axis):
        """Read the current velocity in cts/msc."""
        try:
            _cmd = self.commands.i_axis_speed[axis-1]
            if self.get_response(_cmd):
                _vel = float(
                    self._value)/self.commands.CTS_MM_AXIS[axis-1]*1000
                return _vel
            else:
                return None
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def set_axis_speed(self, axis, value):
        """Set the axis speed."""
        try:
            # convert value from mm/sec to cts/msec
            adj_value = value * self.commands.CTS_MM_AXIS[axis-1] / 1000

            # set speed
            _cmd = self.set_par(self.commands.i_axis_speed[axis-1], adj_value)
            if self.get_response(_cmd):
                if self._value != adj_value:
                    return True
            return False
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def move_axis(self, axis, value):
        """Move axis to defined position."""
        try:
            adj_value = value * self.commands.CTS_MM_AXIS[axis-1]
            _cmd = '#' + str(axis) + self.set_par(
                                     self.commands.jog_abs_position,
                                     adj_value)
            if self.get_response(_cmd):
                return True
            return False
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def stop_axis(self, axis):
        """Stop axis."""
        try:
            if self.get_response('#' + str(axis) + self.commands.jog_stop):
                return True
            return False
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def stop_all_axis(self):
        """Stop all axis."""
        try:
            if self.get_response(self.commands.stop_all_axis):
                return True
            return False
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def kill_all_axis(self):
        """Kill all axis."""
        try:
            if self.get_response(self.commands.kill_all_axis):
                return True
            return False
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def set_trigger(
            self,
            axis,
            startPos_mm,
            increments_mm,
            pulseWidth,
            maxPulses,
            edge=1):
        """
        Set the trigger parameters.

        Input Parameters:
        1- Axis
        2- Start Position [mm]
        3- Increments [mm]
        4- Pulse Width
        5- Maximum number of pulses
        6- Edge

        Set trigger parameter following the sequence:
        1- stop triggering
        2- select axis for triggering
        3- set increments and direction
        4- set bit to always use start position
        5- set start position
        6- set pulse width in %
        7- set the maximum number of pulses to trigger
        8- set pulse edge (1 - falling (default), 0 - raising)
        9- enable plc 2
        """
        try:
            cmds = [
                self.set_par(self.commands.q_plc0RunControl, 0),
                self.set_par(self.commands.q_selectedMotor, axis),
                self.set_par(self.commands.q_incremment_mm, increments_mm),
                self.set_par(self.commands.q_useProgStartPos, 1),
                self.set_par(self.commands.q_startPos, startPos_mm),
                self.set_par(self.commands.q_pulseWidth_perc, pulseWidth),
                self.set_par(self.commands.q_maxPulses, maxPulses),
                self.set_par(self.commands.q_fallingEdge, edge),
                self.commands.enaplc2,
            ]
            for cmd in cmds:
                if not self.get_response(cmd):
                    return False
            return True

        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None

    def stop_trigger(self):
        """Stop trigerring."""
        try:
            # stop triggering
            _cmd = self.set_par(self.commands.q_plc0RunControl, 0)
            if self.get_response(_cmd):
                return True
            return False
        except Exception:
            if self.logger is not None:
                self.logger.error('exception', exc_info=True)
            return None
