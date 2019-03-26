# -*- coding: utf-8 -*-
"""Hall Bench Devices."""

import os as _os
import numpy as _np
import time as _time

from . import Agilent3458ALib as _Agilent3458ALib
from . import Agilent34401ALib as _Agilent34401ALib
from . import Agilent34970ALib as _Agilent34970ALib
from . import DRSLib as _DRSLib
from . import ElcomatLib as _ElcomatLib
from . import NMRLib as _NMRLib
from . import PmacLib as _PmacLib
from . import UDCLib as _UDCLib


logs_dir = 'logs'

Pmac = _PmacLib.Pmac
NMR = _NMRLib.NMRSerial
Elcomat = _ElcomatLib.ElcomatSerial
PowerSupply = _DRSLib.SerialDRS_FBP


class Multimeter(_Agilent3458ALib.Agilent3458AGPIB):
    """Multimeter class."""

    def configure(self, aper, mrange):
        """Configure multimeter.

        Args:
            aper (float): A/D converter integration time in ms.
            mrange (float): measurement range in volts.
        """
        self.send_command(self.commands.reset)
        self.send_command(self.commands.func_volt)
        self.send_command(self.commands.tarm_auto)
        self.send_command(self.commands.trig_auto)
        self.send_command(self.commands.nrdgs_ext)
        self.send_command(self.commands.arange_off)
        self.send_command(self.commands.fixedz_on)
        self.send_command(self.commands.range + str(mrange))
        self.send_command(self.commands.math_off)
        self.send_command(self.commands.azero_once)
        self.send_command(self.commands.trig_buffer_off)
        self.send_command(self.commands.delay_0)
        self.send_command(
            self.commands.aper + '{0:.10f}'.format(aper/1000))
        self.send_command(self.commands.disp_off)
        self.send_command(self.commands.scratch)
        self.send_command(self.commands.end_gpib_always)

    def configure_reading_format(self, formtype):
        """Configure multimeter reading format.

        Args:
            formtype (int): format type [SREAL=4, DREAL=5].
        """
        self.send_command(self.commands.mem_fifo)
        if formtype == 4:
            self.send_command(self.commands.oformat_sreal)
            self.send_command(self.commands.mformat_sreal)
        elif formtype == 5:
            self.send_command(self.commands.oformat_dreal)
            self.send_command(self.commands.mformat_dreal)


class Multichannel(_Agilent34970ALib.Agilent34970AGPIB):
    """Multichannel class."""

    def __init__(self, logfile=None):
        """Initiaze variables and prepare logging file.

        Args:
            logfile (str): log file path.
        """
        super().__init__(logfile)
        self.temperature_channels = [
            '201', '202', '203', '204', '205', '206', '207', '208', '209']
        self.voltage_channels = ['101', '102', '103', '105']

    def get_converted_readings(self, wait=0.5):
        """Get multichannel reading list and convert voltage values."""
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
                    if ch in self.voltage_channels:
                        # Voltage to temperature (Transducer 135-14)
                        if ch == '101':
                            temp = (rd + 50e-3)/20e-3
                        elif ch == '102':
                            temp = (rd + 40e-3)/20e-3
                        elif ch == '103':
                            temp = (rd + 50e-3)/20e-3
                        elif ch == '105':
                            temp = (rd + 30e-3)/20e-3
                        else:
                            temp = _np.nan
                        conv_rlist.append(temp)
                    else:
                        conv_rlist.append(rd)
                return conv_rlist
            else:
                return []
        except Exception:
            return []


class UDC(_UDCLib.UDCModBus):
    """Honeywell UDC-3500 class."""

    def __init__(self, logfile=None):
        """Honeywell UDC-3500 control class.

        Args:
            logfile (str): log file path.
        """
        super().__init__(logfile)
        self.slave_address = 14
        self.output_register_address = 70
        self.pv1_register_address = 72
        self.pv2_register_address = 74


class DCCT(_Agilent34401ALib.Agilent34401AGPIB):
    """DCCT Multimeter."""

    def read_current(self, dcct_head=None):
        """Read dcct voltage and convert to current."""
        voltage = self.read()
        dcct_heads = [40, 160, 320, 600, 1000, 1125]
        if voltage is not None and dcct_head in dcct_heads:
            current = voltage * dcct_head/10
        else:
            current = _np.nan
        return current


class HallBenchDevices(object):
    """Hall Bench devices class."""

    def __init__(self):
        """Initialize variables and log files."""

        # Log files
        _dir_path = _os.path.dirname(_os.path.dirname(
            _os.path.dirname(_os.path.abspath(__file__))))
        _logs_path = _os.path.join(_dir_path, logs_dir)
        if not _os.path.isdir(_logs_path):
            _os.mkdir(_logs_path)
        log_pmac = _os.path.join(_logs_path, 'pmac.log')
        log_voltx = _os.path.join(_logs_path, 'voltx.log')
        log_volty = _os.path.join(_logs_path, 'volty.log')
        log_voltz = _os.path.join(_logs_path, 'voltz.log')
        log_multich = _os.path.join(_logs_path, 'multich.log')
        log_nmr = _os.path.join(_logs_path, 'nmr.log')
        log_elcomat = _os.path.join(_logs_path, 'elcomat.log')
        log_dcct = _os.path.join(_logs_path, 'dcct.log')
        log_udc = _os.path.join(_logs_path, 'udc.log')

        # Devices
        self.pmac = Pmac(log_pmac)
        self.voltx = Multimeter(log_voltx)
        self.volty = Multimeter(log_volty)
        self.voltz = Multimeter(log_voltz)
        self.multich = Multichannel(log_multich)
        self.nmr = NMR(log_nmr)
        self.elcomat = Elcomat(log_elcomat)
        self.dcct = DCCT(log_dcct)
        self.udc = UDC(log_udc)
        self.ps = PowerSupply()

    def connect(self, config):
        """Connect devices.

        Args:
            config (ConnectionConfig): connection configuration.
        """
        if config.pmac_enable:
            self.pmac.connect()

        if config.voltx_enable:
            self.voltx.connect(config.voltx_address)

        if config.volty_enable:
            self.volty.connect(config.volty_address)

        if config.voltz_enable:
            self.voltz.connect(config.voltz_address)

        if config.multich_enable:
            self.multich.connect(config.multich_address)

        if config.nmr_enable:
            self.nmr.connect(config.nmr_port, config.nmr_baudrate)

        if config.elcomat_enable:
            self.elcomat.connect(config.elcomat_port, config.elcomat_baudrate)

        if config.dcct_enable:
            self.dcct.connect(config.dcct_address)

        if config.udc_enable:
            self.udc.connect(config.udc_port, config.udc_baudrate)

        if config.ps_enable:
            self.ps.Connect(config.ps_port)

    def disconnect(self):
        """Disconnect devices."""
        self.pmac.disconnect()
        self.voltx.disconnect()
        self.volty.disconnect()
        self.voltz.disconnect()
        self.multich.disconnect()
        self.nmr.disconnect()
        self.elcomat.disconnect()
        self.dcct.disconnect()
        self.udc.disconnect()
        self.ps.Disconnect()
