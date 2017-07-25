# -*- coding: utf-8 -*-
"""
Pmac lib. Implementation of the main commands to control the bench.
"""
# Libraries
import time
import ctypes
import logging
from .PmacVars import *

class Pmac(object):
    def __init__(self):
        """ Initiate all function variables"""
        # Initialize logging system
        self.log_events()

        # load commands
        self.commands = ListOfCommands()

        # start goblal variables
        self.pmacdll = None
        self.value = ''

        # connect to dll and device
        self.create()

    def log_events(self):
        """ Prepare logging file to save info, warning and error status """
        logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            filename='PmacLib.log',
                            level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
    def create(self):
        """ create an instance, open dll, and connect to device if possible """
        if self.load_dll():
            return True
##            if self.connect():
##                self.activate_bench()
##                return True
##            else:
##                self.logger.error('Fail to connect to device')
        else:
            self.logger.error('Fail to connect to dll')
            
        return False
    
    def set_par(self, input_par, value):
        """ Create string if desired value """
        try:
            _parameter = input_par + '=' + str(value)
            return _parameter
        except:
            return input_par
            
    def load_dll(self):
        """Load dll file PComm32W.dll to control the bench"""
        try:
            self.pmacdll = ctypes.windll.LoadLibrary('PComm32W.dll')
            return True
        except:
            return False

    def connect(self):
        """Connect to Pmac device - OpenPmacDevice(0)"""
        try:
            return bool(self.pmacdll.OpenPmacDevice(0))
        except:
            return False

    def disconnect(self):
        """Disconnect Pmac device - ClosePmacDevice(0)"""
        try:
            return bool(self.pmacdll.ClosePmacDevice(0))
        except:
            return None

    def lock_pmac(self):
        """Lock Pmac to avoid multiple operations - LockPmac(0)"""
        try:
            return self.pmacdll.LockPmac(0)
        except:
            return None
        
    def release_pmac(self):
        """Release Pmac - ReleasePmac(0)"""
        try:
            return self.pmacdll.ReleasePmac(0)
        except:
            return None

    def get_response(self, str_command):
        """ Get response of the string command from Pmac device - PmacExA
            Returns True or False and the resulted value when available
        """
        try:
            MASK_STATUS = 0xF0000000
            COMM_EOT = 0x80000000 # An acknowledge character (ACK ASCII 9) was received indicating end of transmission from PMAC to Host PC.
            
            devnum = 0
            maxchar = 16
            # create empty string with n*maxchar
            response = (' '*maxchar).encode('utf-8')

            # send command and get pmac response
            _retval = self.pmacdll.PmacGetResponseExA(0, response, maxchar, str_command.encode('utf-8'))
            
            # check the status and if it matches with the acknowledge character COMM_EOT
            if _retval & MASK_STATUS == COMM_EOT:
                result = response.decode('utf-8')
                # erase all result after /r
                self.value = result[0:result.find('\r')]

                return True
            else:
                return False
        except:
            self.logger.error('exception',exc_info=True)

    def read_response(self, str_command):
        """ get_response of a variable and return the result instead of status """
        try:
            if self.get_response(str_command):
                return self.value
            else:
                return ''
        except:
            self.logger.error('exception',exc_info=True)
    
    def activate_bench(self):
        """
        Activate the bench. Set the mask value to 503 in q95 and enable plcs 5 and 10
        """
        try:
            _cmd = self.set_par(self.commands.q_motorMask,503)
            if self.get_response(_cmd):
                _cmd = self.commands.enplc5
                if self.get_response(_cmd):
                    _cmd = self.commands.enplc10
                    if self.get_response(_cmd):
                        return True
            return False     
        except:
            self.logger.error('exception',exc_info=True)

    def axis_status(self, axis):
        """
        Get axis status
        """
        try:
            _cmd = '#' + str(axis) + self.commands.axis_status
            if self.get_response(_cmd):
                status = int(self.value,16)
                return status
            return None
        except:
            self.logger.error('exception',exc_info=True)

    def align_bench(self, axis_mask):
        """
        Set the mask of the axis to be aligned and run plc script for alignment
        """
        try:
            _cmd = self.set_par(self.commands.p_axis_mask,axis_mask)
            if self.get_response(_cmd):
                if self.get_response(self.commands.p_axis_mask):
                    if int(self.value) == axis_mask:
                        if self.get_response(self.commands.rp_align_axis):
                            return True
                        else:
                            self.logger.warning('Fail to set P_axis_mask')    
                    else:
                        self.logger.warning('Fail to set P_axis_mask')
            return False     
        except:
            self.logger.error('exception',exc_info=True)

    def get_position(self, axis):
        """ Read the current position in counter and convert to mm """
        try:
            _cmd = '#' + str(axis) + self.commands.current_position
            if self.get_response(_cmd):
                _pos = float(self.value) / self.commands.CTS_MM_AXIS[axis-1]
                return _pos
            else:
                return None
        except:
            self.logger.error('exception',exc_info=True)

    def get_velocity(self, axis):
        """ Read the current velocity in cts/msc """
        try:
            _cmd = self.commands.i_axis_speed[axis-1]
            if self.get_response(_cmd):
                _vel = float(self.value) / self.commands.CTS_MM_AXIS[axis-1] * 1000
                return _vel
            else:
                return None
        except:
            self.logger.error('exception',exc_info=True)
            
    def set_axis_speed(self, axis, value):
        """ Set the axis speed """
        try:
            # convert value from mm/sec to cts/msec
            adj_value = value * self.commands.CTS_MM_AXIS[axis-1] / 1000

            # set speed
            _cmd = self.set_par(self.commands.i_axis_speed[axis-1],adj_value)
            if self.get_response(_cmd):
                if self.value != adj_value:
                    return True
            return False
        except:
            self.logger.error('exception',exc_info=True)
        
    def move_axis(self, axis, value):
        """ Move axis to defined position """
        try:
            adj_value = value * self.commands.CTS_MM_AXIS[axis-1]
            _cmd = '#' + str(axis) + self.set_par(self.commands.jog_abs_position,adj_value)
            if self.get_response(_cmd):
                return True
            return False
        except:
            self.logger.error('exception',exc_info=True)

    def stop_axis(self, axis):
        """ Stop axis """
        try:
            if self.get_response('#' + str(axis) + self.commands.jog_stop):
                return True
            return False
        except:
            self.logger.error('exception',exc_info=True)
            
    def stop_all_axis(self):
        """ Stop all axis """
        try:
            if self.get_response(self.commands.stop_all_axis):
                return True
            return False
        except:
            self.logger.error('exception',exc_info=True)
            
    def kill_all_axis(self):
        """ Kill all axis """
        try:
            if self.get_response(self.commands.kill_all_axis):
                return True
            return False
        except:
            self.logger.error('exception',exc_info=True)
            
    def set_trigger(self, axis, startPos_mm, increments_mm, pulseWidth, maxPulses, edge=1):
        """
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
        8- set pulse edge (1 - falling (default), 0 - raising

        """
        try:
            # stop triggering
            _cmd = self.set_par(self.commands.q_plc0RunControl,0)
            if self.get_response(_cmd):

                # select axis for triggering
                _cmd = self.set_par(self.commands.q_selectedMotor,axis)
                if self.get_response(_cmd):

                    # set increments and direction
                    _cmd = self.set_par(self.commands.q_incremment_mm,increments_mm)
                    if self.get_response(_cmd):
                        
                        # set bit to always use start position
                        _cmd = self.set_par(self.commands.q_useProgStartPos,1)
                        if self.get_response(_cmd):

                            # set start position
                            _cmd = self.set_par(self.commands.q_startPos,startPos_mm)
                            if self.get_response(_cmd):

                                # set pulse width in %
                                _cmd = self.set_par(self.commands.q_pulseWidth_perc,pulseWidth)
                                if self.get_response(_cmd):

                                    # set the maximum number of pulses to trigger
                                    _cmd = self.set_par(self.commands.q_maxPulses,maxPulses)
                                    if self.get_response(_cmd):

                                        # set pulse edge (1 - falling (default), 0 - raising
                                        _cmd = self.set_par(self.commands.q_fallingEdge,edge)
                                        if self.get_response(_cmd):

                                            # enable plc 2
                                            if self.get_response(self.commands.enaplc2):
                                                return True                                    
            return False
        except:
            self.logger.error('exception',exc_info=True)

    def stop_trigger(self):
        """ Stop trigerring """
        try:
            # stop triggering
            _cmd = self.set_par(self.commands.q_plc0RunControl,0)
            if self.get_response(_cmd):
                return True
            return False
        except:
            self.logger.error('exception',exc_info=True)
