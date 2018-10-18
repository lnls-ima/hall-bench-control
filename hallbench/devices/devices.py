# -*- coding: utf-8 -*-
"""Hall Bench Devices."""

import os as _os
from . import GPIBLib as _GPIBLib
from . import PmacLib as _PmacLib
from . import NMRLib as _NMRLib
from . import SerialLib as _SerialLib
from . import SerialDRS as _SerialDRS
from . import UDCLib as _UDCLib

_logs_dir = 'logs'


class HallBenchDevices(object):
    """Hall Bench devices class."""

    def __init__(self):
        """Initialize variables and log files."""
        _dir_path = _os.path.dirname(_os.path.dirname(
            _os.path.dirname(_os.path.abspath(__file__))))
        _logs_path = _os.path.join(_dir_path, _logs_dir)
        if not _os.path.isdir(_logs_path):
            _os.mkdir(_logs_path)
        self.pmac = _PmacLib.Pmac(_os.path.join(_logs_path, 'pmac.log'))
        self.voltx = _GPIBLib.Agilent3458A(
            _os.path.join(_logs_path, 'voltx.log'))
        self.volty = _GPIBLib.Agilent3458A(
            _os.path.join(_logs_path, 'volty.log'))
        self.voltz = _GPIBLib.Agilent3458A(
            _os.path.join(_logs_path, 'voltz.log'))
        self.multich = _GPIBLib.Agilent34970A(
            _os.path.join(_logs_path, 'multich.log'))
        self.nmr = _NMRLib.NMR(_os.path.join(_logs_path, 'nmr.log'))
        self.elcomat = _SerialLib.Elcomat(
            _os.path.join(_logs_path, 'elcomat.log'))
        self.dcct = _GPIBLib.Agilent34401A(
            _os.path.join(_logs_path, 'dcct.log'))
        self.ps = _SerialDRS.SerialDRS_FBP()
        self.udc = _UDCLib.UDC3500()

    def connect(self, config):
        """Connect devices.

        Args:
            config (ConnectionConfig): connection configuration.
        """
        if config.voltx_enable:
            self.voltx.connect(config.voltx_address)

        if config.volty_enable:
            self.volty.connect(config.volty_address)

        if config.voltz_enable:
            self.voltz.connect(config.voltz_address)

        if config.pmac_enable:
            self.pmac.connect()

        if config.multich_enable:
            self.multich.connect(config.multich_address)

        if config.nmr_enable:
            self.nmr.connect(config.nmr_port, config.nmr_baudrate)

        if config.elcomat_enable:
            self.elcomat.connect(config.elcomat_port, config.elcomat_baudrate)

        if config.dcct_enable:
            self.dcct.connect(config.dcct_address)

        if config.ps_enable:
            self.ps.Connect(config.ps_port)
            
        if config.udc_enable:
            self.udc.connect(config.udc_port, config.udc_baudrate)

    def disconnect(self):
        """Disconnect devices."""
        self.voltx.disconnect()
        self.volty.disconnect()
        self.voltz.disconnect()
        self.pmac.disconnect()
        self.multich.disconnect()
        self.nmr.disconnect()
        self.elcomat.disconnect()
        self.dcct.disconnect()
        self.ps.Disconnect()