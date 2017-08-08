# -*- coding: utf-8 -*-
"""Hall bench devices."""

import numpy as _np
import struct as _struct
from . import GPIBLib as _GPIBLib


class DigitalMultimeter(_GPIBLib.GPIB_A3458A):
    """Hall bench digital multimeter."""

    def __init__(self, logfile, address):
        """Initiate variables and prepare log file.

        Args:
            logfile (str): log file path.
            address (int): device address.
        """
        self.logfile = logfile
        self.address = address
        self.voltage = _np.array([])
        super().__init__(self.logfile)

    def connect(self):
        """Connect device."""
        return super(DigitalMultimeter, self).connect(self.address)

    def read(self, stop_flag, end_meas_flag, formtype=0):
        """Read voltage from the device.

        Args:
            stop_flag (bool): stop measurement flag.
            end_meas_flag (bool): end measurement flag.
            formtype (int): format type [single=0 or double=1].
        """
        while (stop_flag is False) and (end_meas_flag is False):
            if self.inst.stb & 128:
                r = self.read_raw_from_device()
                if formtype == 0:
                    dataset = [_struct.unpack(
                        '>f', r[i:i+4])[0] for i in range(0, len(r), 4)]
                else:
                    dataset = [_struct.unpack(
                        '>d', r[i:i+8])[0] for i in range(0, len(r), 8)]
                self.voltage = _np.append(self.voltage, dataset)
        else:
            # check memory
            self.send_command(self.commands.mcount)
            npoints = int(self.read_from_device())
            if npoints > 0:
                # ask data from memory
                self.send_command(self.commands.rmem + str(npoints))

                for idx in range(npoints):
                    # read data from memory
                    r = self.read_raw_from_device()
                    if formtype == 0:
                        dataset = [_struct.unpack(
                            '>f', r[i:i+4])[0] for i in range(0, len(r), 4)]
                    else:
                        dataset = [_struct.unpack(
                            '>d', r[i:i+8])[0] for i in range(0, len(r), 8)]
                    self.voltage = _np.append(self.voltage, dataset)

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
        self.send_command(self.commands.range + '15')
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


class Multichannel(_GPIBLib.GPIB_A34970A):
    """Hall bench multichannel for temperatures readings."""

    def __init__(self, logfile, address):
        """Initiate variables and prepare logging file.

        Args:
            logfile (str): log file path.
            address (int): device address.
        """
        self.logfile = logfile
        self.address = address
        super().__init__(self.logfile)

    def connect(self):
        """Connect device."""
        return super(Multichannel, self).connect(self.address)
