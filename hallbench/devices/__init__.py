"""Sub-package for hall bench devices."""

import os as _os

from imadevices import F1000DRSLib as _DRSLib
from imadevices import PmacLib as _PmacLib
from imadevices import ElcomatLib as _ElcomatLib
from imadevices import NMRLib as _NMRLib
from imadevices import UDCLib as _UDCLib

from . import devices as _devices


_logs_dir = 'logs'
_dir_path = _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))))
_logs_path = _os.path.join(_dir_path, _logs_dir)

if not _os.path.isdir(_logs_path):
    _os.mkdir(_logs_path)

_log_pmac = _os.path.join(_logs_path, 'pmac.log')
_log_voltx = _os.path.join(_logs_path, 'voltx.log')
_log_volty = _os.path.join(_logs_path, 'volty.log')
_log_voltz = _os.path.join(_logs_path, 'voltz.log')
_log_multich = _os.path.join(_logs_path, 'multich.log')
_log_nmr = _os.path.join(_logs_path, 'nmr.log')
_log_elcomat = _os.path.join(_logs_path, 'elcomat.log')
_log_dcct = _os.path.join(_logs_path, 'dcct.log')
_log_water_udc = _os.path.join(_logs_path, 'water_udc.log')
_log_air_udc = _os.path.join(_logs_path, 'air_udc.log')


pmac = _PmacLib.Pmac(_log_pmac)
voltx = _devices.Multimeter(_log_voltx)
volty = _devices.Multimeter(_log_volty)
voltz = _devices.Multimeter(_log_voltz)
multich = _devices.Multichannel(_log_multich)
nmr = _NMRLib.NMRSerial(_log_nmr)
elcomat = _ElcomatLib.ElcomatSerial(_log_elcomat)
dcct = _devices.DCCT(_log_dcct)
water_udc = _UDCLib.UDCModBus(_log_water_udc)
air_udc = _UDCLib.UDCModBus(_log_air_udc)
ps = _DRSLib.SerialDRS_FBP()
