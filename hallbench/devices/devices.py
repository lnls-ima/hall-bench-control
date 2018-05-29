# -*- coding: utf-8 -*-
"""Hall Bench Devices."""

from . import GPIBLib as _GPIBLib
from . import PmacLib as _PmacLib
from . import NMRLib as _NMRLib


class HallBenchDevices(object):
    """Hall Bench devices class."""

    def __init__(self):
        """Initiate variables."""
        self.pmac = None
        self.voltx = None
        self.volty = None
        self.voltz = None
        self.multich = None
        self.nmr = None
        self.colimator = None
        self.loaded = False

    def clearMultimetersData(self):
        """Clear multimeters stored data and update measurement flags."""
        if not self.loaded:
            return
        self.voltx.end_measurement = False
        self.volty.end_measurement = False
        self.voltz.end_measurement = False
        self.voltx.clear()
        self.volty.clear()
        self.voltz.clear()

    def configurePmacTrigger(self, axis, pos, step, npts):
        """Configure Pmac trigger."""
        self.pmac.set_trigger(axis, pos, step, 10, npts, 1)

    def connect(self, configuration):
        """Connect devices.

        Args:
            configuration (ConnectionConfig): connection configuration.
        """
        if not self.loaded:
            if not self.load():
                return

        status = []
        if configuration.control_voltx_enable:
            status.append(self.voltx.connect(configuration.control_voltx_addr))

        if configuration.control_volty_enable:
            status.append(self.volty.connect(configuration.control_volty_addr))

        if configuration.control_voltz_enable:
            status.append(self.voltz.connect(configuration.control_voltz_addr))

        if configuration.control_pmac_enable:
            status.append(self.pmac.connect())

        if configuration.control_multich_enable:
            status.append(
                self.multich.connect(configuration.control_multich_addr))

        return status

    def disconnect(self):
        """Disconnect devices."""
        if not self.loaded:
            return [True]*5

        status = []
        status.append(self.voltx.disconnect())
        status.append(self.volty.disconnect())
        status.append(self.voltz.disconnect())
        status.append(self.pmac.disconnect())
        status.append(self.multich.disconnect())

        return status

    def initialMeasurementConfiguration(self, configuration):
        """Initial measurement configuration.

        Args:
            configuration (MeasurementConfig): measurement configuration.
        """
        if configuration.meas_probeX:
            self.voltx.config(
                configuration.meas_aper_ms, configuration.meas_precision)
        if configuration.meas_probeY:
            self.volty.config(
                configuration.meas_aper_ms, configuration.meas_precision)
        if configuration.meas_probeZ:
            self.voltz.config(
                configuration.meas_aper_ms, configuration.meas_precision)

        self.pmac.set_axis_speed(1, configuration.meas_vel_ax1)
        self.pmac.set_axis_speed(2, configuration.meas_vel_ax2)
        self.pmac.set_axis_speed(3, configuration.meas_vel_ax3)
        self.pmac.set_axis_speed(5, configuration.meas_vel_ax5)

    def load(self):
        """Load devices."""
        try:
            self.pmac = _PmacLib.Pmac('pmac.log')
            self.voltx = _GPIBLib.Agilent3458A('voltx.log')
            self.volty = _GPIBLib.Agilent3458A('volty.log')
            self.voltz = _GPIBLib.Agilent3458A('voltz.log')
            self.multich = _GPIBLib.Agilent34970A('multi.log')
            self.nmr = _NMRLib.NMR('nmr.log')
            self.loaded = True
        except Exception:
            self.loaded = False

    def stopTrigger(self):
        """Stop Pmac trigger and update measurement flags."""
        self.pmac.stop_trigger()
        self.voltx.end_measurement = True
        self.volty.end_measurement = True
        self.voltz.end_measurement = True
