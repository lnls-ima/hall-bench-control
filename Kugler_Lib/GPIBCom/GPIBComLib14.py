# -*- coding: utf-8 -*-
"""
Created on 10/02/2015
Version 0.1
@author: James Citadini
"""
# Libraries
import visa
import logging
import struct
import time
import numpy as np

class GPIB(object):
    def __init__(self,logfile=''):
        """
        GPIB class to for communication with GPIB devices
        """
        # Initialize logging system
        self.log_events(logfile)

        # load commands
        self.commands = ''
        
        # start global variables
        self.inst = None

    def log_events(self,logfile):
        """ Prepare logging file to save info, warning and error status """
        if logfile != '':
            logging.basicConfig(format='%(asctime)s\t%(levelname)s\t%(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S',
                                filename = logfile,
                                level=logging.DEBUG)
            self.logger = logging.getLogger(__name__)
            self.logger.warning('Teste')
        
    def connect(self, address):
        """ Connect to a GPIB devide with the given address """
        try:
            # connects to the device
            _cmd = 'GPIB0::'+str(address)
            _inst = visa.instrument(_cmd)

            # check if connected
            if _inst.__module__ == 'pyvisa.highlevel':
                # copy reference to global variable
                self.inst = _inst
                # set a default timeout to 1
                self.inst.timeout = 1
                return True
            else:
                return False
        except:
            self.logger.error('exception',exc_info=True)
       
    def send_command(self,command):
        """ Write a string message to the device and check the size of the answer """
        try:
            if self.inst.write(command) == (len(command) + 2):
                return True
            else:
                return False
        except:
            self.logger.error('exception',exc_info=True)

    def read_from_device(self):
        """
        Read a string from the device. Stop reading when termination is detected
        Tries to read from device, if timeout occurs, returns empty string
        """
        try:
            _reading = self.inst.read()
            return _reading
        except:
            return ''

    def read_raw_from_device(self):
        """
        Read a string from the device. Stop reading when termination is detected
        Tries to read from device, if timeout occurs, returns empty string
        """
        try:
            _reading = self.inst.read_raw()
            return _reading
        except:
            return ''        

    def read_memory_from_device_single_double_real(self,form_type):
        """ Retrieve data from memory """
        try:
            # check the number of points in memory
            self.send_command(self.commands.mcount)
            _num_counts = int(self.read_from_device())

            # if memory not empty
            if _num_counts > 0:

                # Freeze memory to start collecting
                self.send_command(self.commands.rmem + str(_num_counts))
            
            _readings = np.array([])
            for _idx in range(_num_counts):
                if form_type == 's':
                    _readings = np.append(_readings,struct.unpack('>d',self.read_raw_from_device())[0])
                else:
                    _readings = np.append(_readings,struct.unpack('>f',self.read_raw_from_device())[0])

            return _readings
        except:
            self.logger.error('exception',exc_info=True)
            return  np.array([])
