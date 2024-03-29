# -*- coding: utf-8 -*-

"""Main entry poin to the Hall bench control application."""

import numpy as _np
import time as _time

from imautils.devices import Agilent3458ALib as _Agilent3458ALib
from imautils.devices import Agilent34401ALib as _Agilent34401ALib
from imautils.devices import Agilent34970ALib as _Agilent34970ALib
from imautils.devices import pydrs_firmware_updated as _pydrs
from imautils.devices import pydrs as _pydrs_old
from imautils.devices import ElcomatLib as _ElcomatLib
from imautils.devices import NMRLib as _NMRLib
from imautils.devices import UDCLib as _UDCLib
from imautils.devices import FDI2056 as _FDI2056
from imautils.devices import HeidenhainLib as _HeidenhainLib


try:
    from imautils.devices import PmacLib as _PmacLib
    pmac_module = True
except ModuleNotFoundError:
    pmac_module = False


class Multimeter(_Agilent3458ALib.Agilent3458AGPIB):
    """Multimeter class."""

    def configure(self, aper, mrange):
        """Configure multimeter.

        Args:
            aper (float): A/D converter integration time in ms.
            mrange (float): measurement range in volts.
        """
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
        self.send_command(self.commands.mem_fifo)

    def configure_reading_format(self, formtype):
        """Configure multimeter reading format.

        Args:
            formtype (str): format type [SREAL, DREAL].
        """
        self.send_command(self.commands.mem_fifo)
        if formtype == 'SREAL':
            self.send_command(self.commands.oformat_sreal)
            self.send_command(self.commands.mformat_sreal)
        elif formtype == 'DREAL':
            self.send_command(self.commands.oformat_dreal)
            self.send_command(self.commands.mformat_dreal)


class Multichannel(_Agilent34970ALib.Agilent34970AGPIB):
    """Multichannel class."""

    def __init__(self, log=False):
        """Initiaze variables and prepare logging.

        Args:
            log (bool): True to use event logging, False otherwise.
        """
        super().__init__(log=log)
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
#                         # Voltage to temperature (Transducer 135-14)
#                         if ch == '101':
#                             temp = (rd + 50e-3)/20e-3
#                         elif ch == '102':
#                             temp = (rd + 40e-3)/20e-3
#                         elif ch == '103':
#                             temp = (rd + 50e-3)/20e-3
#                         elif ch == '105':
#                             temp = (rd + 30e-3)/20e-3
#                         else:
#                             temp = _np.nan
#                        # Voltage to temperature (Transducer 133-14)
#                        if ch == '101':
#                            temp = (rd + 20e-3)/20e-3
#                        elif ch == '102':
#                            temp = (rd + 25e-3)/20e-3
#                        elif ch == '103':
#                            temp = (rd + 35e-3)/20e-3
#                        elif ch == '105':
#                            temp = (rd + 45e-3)/20e-3
                        # Voltage to temperature(Transducer SN TRIC-30003121)
                        if ch == '101':
                            temp = (rd - 120e-3)/20e-3
                        elif ch == '105':
                            temp = (rd + 25e-3)/20e-3
                        elif ch == '103':
                            temp = (rd + 35e-3)/20e-3
                        elif ch == '102':
                            temp = (rd - 1170e-3)/20.05e-3
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


class DCCT(_Agilent34401ALib.Agilent34401AGPIB):
    """DCCT Multimeter."""

    def __init__(self, log=False):
        super().__init__(log=log)
        self.dcct_head = None

    def read_current(self):
        """Read dcct voltage and convert to current."""
        voltage = self.read()
        dcct_heads = [40, 160, 320, 600, 1000, 1125]
        if voltage is not None and self.dcct_head in dcct_heads:
            current = voltage * self.dcct_head/10
        else:
            current = _np.nan
        return current

    def read_fast(self):
        self.send_command(self.commands.read)
        try:
            val = float(self.read_from_device()[:-1])
        except Exception:
            val = None
        dcct_heads = [40, 160, 320, 600, 1000, 1125]
        if val is not None and self.dcct_head in dcct_heads:
            current = val * self.dcct_head/10
        else:
            current = _np.nan
        return current


class PowerSupply(_pydrs.SerialDRS):
    """Power Supply."""

    def __init__(self):
        self.ps_type = None
        super().__init__()


class TrimPowerSupply(_pydrs_old.SerialDRS):
    """Power Supply."""

    def __init__(self):
        self.ps_type = None
        super().__init__()


Autocollimator = _ElcomatLib.ElcomatSerial
NMR = _NMRLib.NMRSerial
WaterUDC = _UDCLib.UDCModBus
AirUDC = _UDCLib.UDCModBus
Integrator = _FDI2056.EthernetCom
Display = _HeidenhainLib.HeidenhainSerial


if pmac_module:
    Pmac = _PmacLib.Pmac
else:
    class Pmac():

        def __init__(self, log=False):
            super().__init__()
            self.connected = False

        def connect(self):
            return False

        def disconnect(self):
            return True
