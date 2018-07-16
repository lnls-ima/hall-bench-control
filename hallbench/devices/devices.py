# -*- coding: utf-8 -*-
"""Hall Bench Devices."""

from . import GPIBLib as _GPIBLib
from . import PmacLib as _PmacLib
from . import NMRLib as _NMRLib


class HallBenchDevices(object):
    """Hall Bench devices class."""

    def __init__(self):
        """Initialize variables and log files."""
        self.pmac = _PmacLib.Pmac('pmac.log')
        self.voltx = _GPIBLib.Agilent3458A('voltx.log')
        self.volty = _GPIBLib.Agilent3458A('volty.log')
        self.voltz = _GPIBLib.Agilent3458A('voltz.log')
        self.multich = _GPIBLib.Agilent34970A('multich.log')
        self.nmr = _NMRLib.NMR('nmr.log')
        self.collimator = None

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

    def disconnect(self):
        """Disconnect devices."""
        self.voltx.disconnect()
        self.volty.disconnect()
        self.voltz.disconnect()
        self.pmac.disconnect()
        self.multich.disconnect()
        self.nmr.disconnect()
