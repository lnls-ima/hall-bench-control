# -*- coding: utf-8 -*-
"""Heidenhain communication.

Created on 28/08/2012
Modified on 22/11/2017
Versão 1.0
@author: James Citadini
"""

import sys as _sys
import time as _time
import numpy as _np
import traceback as _traceback

from . import interfaces as _interfaces


class HeidenhainCommands(object):
    """Heidenhain Display Commands."""

    def __init__(self):
        """Heidenhain commands."""
        # {Tecla 0}
        self.zero0 = '\x1bT0000\r'
        # {Tecla 1}
        self.one1 = '\x1bT0001\r'
        # {Tecla 2}
        self.two2 = '\x1bT0002\r'
        # {Tecla 3}
        self.three3 = '\x1bT0003\r'
        # {Tecla 4}
        self.four4 = '\x1bT0004\r'
        # {Tecla 5}
        self.five5 = '\x1bT0005\r'
        # {Tecla 6}
        self.six6 = '\x1bT0006\r'
        # {Tecla 7}
        self.seven7 = '\x1bT0007\r'
        # {Tecla 8}
        self.eight8 = '\x1bT0008\r'
        # {Tecla 9}
        self.nine9 = '\x1bT0009\r'
        # {Tecla CL}
        self.cl = '\x1bT0100\r'
        # {Tecla -}
        self.minus = '\x1bT0101\r'
        # {Tecla .}
        self.point = '\x1bT0102\r'
        # {Tecla Ent}
        self.ent = '\x1bT0104\r'
        # {Tecla 1/2}
        self.axis12 = '\x1bT0107\r'
        # {Tecla X}
        self.axisx = '\x1bT0109\r'
        # {Tecla Y}
        self.axisy = '\x1bT0110\r'
        # {Tecla Z}
        self.axisz = '\x1bT0111\r'
        # {Tecla Spec Fct}
        self.spec = '\x1bT0129\r'
        # {Tecla R+/-}
        self.rpn = '\x1bT0142\r'

        # {Tecla CE+0}
        self.cezero0 = '\x1bT1000\r'
        # {Tecla CE+1}
        self.ceone1 = '\x1bT1001\r'
        # {Tecla CE+2}
        self.cetwo2 = '\x1bT1002\r'
        # {Tecla CE+3}
        self.cethree3 = '\x1bT1003\r'
        # {Tecla CE+4}
        self.cefour4 = '\x1bT1004\r'
        # {Tecla CE+5}
        self.cefive5 = '\x1bT1005\r'
        # {Tecla CE+6}
        self.cesix6 = '\x1bT1006\r'
        # {Tecla CE+7}
        self.ceseven7 = '\x1bT1007\r'
        # {Tecla CE+8}
        self.ceeight8 = '\x1bT1008\r'
        # {Tecla CE+9}
        self.cenine9 = '\x1bT1009\r'

        # {Output of model designation}
        self.out_model = '\x1bA0000\r'
        # {Output of 14-segment display}
        self.segm = '\x1bA0100\r'
        # {Output of current value}
        self.current_value = '\x1bA0200\r'
        # {Output of error text}
        self.error = '\x1bA0301\r'
        # {Output of software number}
        self.soft = '\x1bA0400\r'
        # {Output of indicators}
        self.ind = '\x1bA0900\r'

        # {Counter RESET}
        self.resetcounter = '\x1bS0000\r'
        # {Lock keyboard}
        self.lock = '\x1bS0001\r'
        # {Unlock keyboard}
        self.unlock = '\x1bS0002\r'


def Heidenhain_factory(baseclass):
    """Create Heidenhain class."""
    class Heidenhain(baseclass):
        """Heidenhain Display."""

        def __init__(self, logfile=None):
            """Initiaze all variables and prepare log file.

            Args:
                logfile (str): log file path.
            """
            self.commands = HeidenhainCommands()
            super().__init__(logfile)

        def write_display_value(self, axis, value, wait=0.2):
            aux = str(abs(value))
            nchar = len(aux)

            self.send_command(self.commands.cl)
            _time.sleep(wait)

            if axis == 0:
                self.send_command(self.commands.axisx)
            elif axis == 1:
                self.send_command(self.commands.axisy)
            elif axis == 2:
                self.send_command(self.commands.axisz)
            _time.sleep(wait)

            for i in range(nchar):
                tmp = aux[i]
                if (tmp == '0'):
                    cmd = self.commands.zero0
                elif (tmp == '1'):
                    cmd = self.commands.one1
                elif (tmp == '2'):
                    cmd = self.commands.two2
                elif (tmp == '3'):
                    cmd = self.commands.three3
                elif (tmp == '4'):
                    cmd = self.commands.four4
                elif (tmp == '5'):
                    cmd = self.commands.five5
                elif (tmp == '6'):
                    cmd = self.commands.six6
                elif (tmp == '7'):
                    cmd = self.commands.seven7
                elif (tmp == '8'):
                    cmd = self.commands.eight8
                elif (tmp == '9'):
                    cmd = self.commands.nine9
                elif (tmp == '.'):
                    cmd = self.commands.point
                self.send_command(cmd)
                _time.sleep(wait)

            if value < 0:
                self.send_command(self.commands.minus)
                _time.sleep(wait)

            self.send_command(self.commands.ent)

        def send_key(self, key, wait=0.2):
            try:
                self.send_command(key)
                _time.sleep(wait)
                reading = self.read_from_device()
                return reading
            except Exception:
                return False

        def read_display(self, model, wait=0.2):
            if model == 'ND-780':
                return self.read_display_ND780(wait=wait)
            elif model == 'ND-760':
                return self.read_display_ND760(wait=wait)
            else:
                return None

        def read_display_ND760(self, wait=0.2):
            try:
                self.send_command(self.commands.current_value)
                _time.sleep(wait)
                reading = self.read_from_device()

                reading = self.clean_string(reading)

                p1 = reading.find('rn')
                reading1 = float(reading[p1-10:p1])/1000
                reading = reading[p1+1:]

                p2 = reading.find('rn')
                reading2 = float(reading[p2-10:p2])/1000
                reading = reading[p2+1:]

                p3 = reading.find('rn')
                reading3 = float(reading[p3-10:p3])/1000

            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                reading1 = _np.nan
                reading2 = _np.nan
                reading3 = _np.nan

            return (reading1, reading2, reading3)

        def read_display_ND780(self, wait=0.2):
            try:
                self.send_command(self.commands.current_value)
                _time.sleep(wait)
                reading = self.read_from_device()

                aux1 = reading[reading.find('X=')+2:reading.find(' R\r\n')]
                aux1 = aux1.replace(' ', '')

                reading = reading[reading.find('R\r\n')+3:]
                aux2 = reading[reading.find('Y=')+2:reading.find(' R\r\n')]
                aux2 = aux2.replace(' ', '')

                reading1 = float(aux1)
                reading2 = float(aux2)

            except Exception:
                reading1 = _np.nan
                reading2 = _np.nan

            return (reading1, reading2)

        def clean_string(self, reading):
            reading = reading.replace('\'', '')
            reading = reading.replace('\\x82', '')
            reading = reading.replace('\\x8d', '')
            reading = reading.replace('\\', '')
            reading = reading.replace('xb', '')
            reading = reading.replace('b', '')
            return reading

        def reset_set_ref(self, wait=0.2):
            self.send_command(self.commands.resetcounter)
            _time.sleep(4)
            self.send_command(self.commands.ent)
            _time.sleep(wait)

    return Heidenhain


HeidenhainSerial = Heidenhain_factory(_interfaces.SerialInterface)
