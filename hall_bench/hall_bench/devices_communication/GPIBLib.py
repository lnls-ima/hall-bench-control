# -*- coding: utf-8 -*-
"""
GPIB communication.

Created on 10/02/2015
Modified on 19/07/2017
@author: James Citadini
"""

# Libraries
import visa as _visa
import logging as _logging
from .Agilent_3458AVars import ListOfCommands as _A3458A_Commands
from .Agilent_34970AVars import ListOfCommands as _A34970A_Commands


class GPIB(object):
    """GPIB class for communication with GPIB devices."""

    def __init__(self, logfile=''):
        """Initiate all variables and prepare log file.

        Args:
            logfile (str): log file path.
        """
        self.commands = None
        self.inst = None
        self.logger = None

        self.log_events(logfile)

    def log_events(self, logfile):
        """Prepare log file to save info, warning and error status.

        Args:
            logfile (str): log file path.
        """
        if logfile != '':
            _logging.basicConfig(
                format='%(asctime)s\t%(levelname)s\t%(message)s',
                datefmt='%m/%d/%Y %H:%M:%S',
                filename=logfile,
                level=_logging.DEBUG)
            self.logger = _logging.getLogger(__name__)

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
                return True
            else:
                return False
        except Exception:
            self.logger.error('exception', exc_info=True)

    def disconnect(self):
        """Disconnect the GPIB device."""
        try:
            if self.inst is not None:
                self.inst.close()
            return True
        except Exception:
            self.logger.error('exception', exc_info=True)

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


class GPIB_A34970A(GPIB):
    """GPIB class for communication with Agilent 34970A devices."""

    def __init__(self, logfile=''):
        """Initiate all variables and prepare log file.

        Args:
            logfile (str): log file path.
        """
        super().__init__(logfile)
        self.commands = _A34970A_Commands()


class GPIB_A3458A(GPIB):
    """GPIB class for communication with Agilent 3458A devices."""

    def __init__(self, logfile=''):
        """Initiate all variables and prepare log file.

        Args:
            logfile (str): log file path.
        """
        super().__init__(logfile)
        self.commands = _A3458A_Commands()
