# -*- coding: utf-8 -*-
"""
Variables and constants to be used in pmac
"""
class ListOfCommands(object):
    def __init__(self):
        """ Initiate all function variables

        # list of available axis
        self.list_of_axis

        # Stop all
        self.stop_all_axis

        # Kill all
        self.kill_all_axis
        
        # Constants
        CTS_MM_AXIS[0:9] - list of constants to convert counts to mm

        # M-variables - inputs
        DI_eStopOK - State of E-stop relays; 1 = OK, 0 = OFF
        DI_inputPressureOK - Monitoring input pressure; 0 = fault, 1 = OK
        DI_vacuumOK - Monitoring vac; 0 = fault, 1 = OK
        prog_running - check if there is a programming running, like homming

        # M-variables - outputs
        None

        # Q-variables
        q_xAxisManualMode - Manual move mode for Z-axis is ON / 0 = Normal mode
        q_motorMask - Bit mask to select the motors
        q_plc5_status - Status of PLC 5
        q_plc10status - Status of PLC 10

        # Q-varibles for triggering
        q_selectedMotor = 'Q0' - [1,2,3,5] trigger source, motor number
        q_incremment_mm = 'Q1' - [mm] trigger pitch (negative numbers also possible)
        q_loopCount = 'Q2' - [1] trigger counter
        q_plc0Status = 'Q3' - Status of plc0
        q_plc0RunControl = 'Q9' - [1] for starting stopping of plc0
        q_useProgStartPos = 'Q10' - [1/0] to use flexible start position
        q_startPos = 'Q11' - [mm]position of first pulse if flexible start position is used
        q_pulseWidth_perc = 'Q12' - [0..100%]pulse width in % (internally limited to min. 10% and max. 75%)
        q_maxPulses = 'Q13' - [0..]max number of pulses (0 for no limitation)
        q_fallingEdge = 'Q14' - [0/1]trigger edge: 1 = falling edge, 0 = rising edge

        # P-variables
        p_axis_mask - Bit mask to select the motors to be homed - b1200r
        p_homming_status - Homming status

        # I-Variables
        i_pos_scale_factor = - Ixx08 Motor xx Position Scale Factor        
        i_softlimit_pos_list - List of positive software position limit [motor counts] - Ixx13
        i_softlimit_neg_list - List of negative software position limit [motor counts] - Ixx14
        i_axis_speed - List of all axis speed - Ixx22 in counts/msec

        # Jogging commands
        jog_pos - Jog motor indefinitely in positive direction
        jog_neg - Jog motor indefinitely in negative direction
        jog_stop - Jog motor indefinitely in negative direction
        jog_abs_position - Jog to absolute position
        jog_rel_position - Jog to relative position

        # Motor Reporting Commands
        current_position - Report position of motor in counts
        current_velocity - Report velocity of motor
        
        # Miscellaneous Commands
        enplc5 = 'enaplc5' # Enable (run) PLC5
        enplc10 = 'enaplc10' # Enable (run) PLC10
        displc5 - Disable (stop) PLC5
        displc10 - Disable (stop) PLC10
        enaplc2 = 'enaplc2'  # Enable (run) PLC2
        
        # run programs
        align_axis - Run routine for alignment of actived axis

        """
        # list of axis to set control
        self.list_of_axis = [1,2,3,5,6,7,8,9]

        # Stop all
        self.stop_all_axis = chr(1)

        # Kill all
        self.kill_all_axis = chr(11)
        
        # Constants
        self.CTS_MM_AXIS = [20000,
                            100000,
                            100000,
                            0,
                            8192,
                            400,
                            400,
                            400,
                            400]
        
        # M-variables - inputs
        self.DI_eStopOK = 'M7000'
        self.DI_inputPressureOK = 'M7004'
        self.DI_vacuumOK = 'M7012'
        self.prog_running = 'M5180'

        # M-variables - outputs
        pass
        
        # Q-variables
        self.q_xAxisManualMode = 'Q90'
        self.q_motorMask = 'Q95' 
        self.q_plc5_status = 'q5500' 
        self.q_plc10status = 'q6000'

        # Q-varibles for trigerring
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

        # P-variables
        self.p_axis_mask = 'P810'
        self.p_homming_status = 'P813'

        # I-Variables
        self.i_pos_scale_factor = ['I'+str(_inum)+'08' for _inum in range(1,10)]
        self.i_softlimit_pos_list = ['I'+str(_inum)+'13' for _inum in range(1,10)]
        self.i_softlimit_neg_list = ['I'+str(_inum)+'14' for _inum in range(1,10)]
        self.i_axis_speed = ['I'+str(_inum)+'22' for _inum in range(1,10)]

        # Jogging commands
        self.jog_pos = 'j+'
        self.jog_neg = 'j-'
        self.jog_stop = 'j/'
        self.jog_abs_position = 'j='
        self.jog_rel_position = 'j:'

        # Motor Reporting Commands
        self.current_position = 'p'
        self.current_velocity = 'v'

        # Miscellaneous Commands
        self.enplc5 = 'enaplc5'
        self.enplc10 = 'enaplc10'
        self.displc5 = 'displc5'
        self.displc10 = 'displc10'
        self.enaplc2 = 'enaplc2'
        self.axis_status = '?'

        # run programs
        self.rp_align_axis = 'b1200r'
        

