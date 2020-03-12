"""Sub-package for hall bench devices."""

import os as _os
import time as _time

from imautils.devices.utils import configure_logging
from . import devices as _devices


_timestamp = _time.strftime('%Y-%m-%d_%H-%M-%S', _time.localtime())

_logs_path = _os.path.join(
    _os.path.dirname(_os.path.dirname(
        _os.path.dirname(
            _os.path.abspath(__file__)))), 'logs')

if not _os.path.isdir(_logs_path):
    _os.mkdir(_logs_path)

logfile = _os.path.join(
    _logs_path, '{0:s}_hall_bench_control.log'.format(_timestamp))
configure_logging(logfile)


pmac = _devices.Pmac(log=True)
voltx = _devices.Multimeter(log=True)
volty = _devices.Multimeter(log=True)
voltz = _devices.Multimeter(log=True)
multich = _devices.Multichannel(log=True)
nmr = _devices.NMR(log=True)
elcomat = _devices.Autocollimator(log=True)
dcct = _devices.DCCT(log=True)
water_udc = _devices.WaterUDC(log=True)
air_udc = _devices.AirUDC(log=True)
ps = _devices.PowerSupply()
integrator = _devices.Integrator()
