# -*- coding: utf-8 -*-
"""Agilent34970A communication.

Created on 10/02/2015
@author: James Citadini
"""

import time as _time

from . import interfaces as _interfaces


class Agilent34970ACommands(object):
    """Commands of Agilent 34970 Data Acquisition/Switch Unit."""

    def __init__(self):
        """Load commands."""
        self._reset()
        self._clean()
        self._lock()
        self._configure()
        self._query()

    def _reset(self):
        """Reset function."""
        self.reset = '*RST'

    def _clean(self):
        """Clean error function."""
        self.clean = '*CLS'

    def _lock(self):
        """Lock function."""
        self.lock = ':SYST:LOC'

    def _configure(self):
        """Configure commands."""
        self.rout_scan = ':ROUT:SCAN'
        self.conf_temp = ':CONF:TEMP FRTD,'
        self.conf_volt = ':CONF:VOLT:DC'

    def _query(self):
        """Query commands."""
        self.qid = '*IDN?'
        self.qread = ':READ?'
        self.qscan = ':ROUT:SCAN?'
        self.qscan_size = ':ROUT:SCAN:SIZE?'


def Agilent34970A_factory(baseclass):
    """Create Agilent34970A class."""
    class Agilent34970A(baseclass):
        """Agilent 34970A multichannel for temperatures readings."""

        def __init__(self, logfile=None):
            """Initiaze variables and prepare logging file.

            Args:
                logfile (str): log file path.
            """
            self._config_channels = []
            self.temperature_channels = []
            self.voltage_channels = []
            self.commands = Agilent34970ACommands()
            super().__init__(logfile)

        @property
        def config_channels(self):
            """Return current channel configuration list."""
            return self._config_channels

        def configure(self, channel_list='all', wait=0.5):
            """Configure channels."""
            if channel_list == 'all':
                volt_channel_list = self.voltage_channels
                temp_channel_list = self.temperature_channels

            else:
                volt_channel_list = []
                temp_channel_list = []
                channel_list = [str(ch) for ch in channel_list]
                for ch in channel_list:
                    if ch in self.voltage_channels:
                        volt_channel_list.append(ch)
                    else:
                        temp_channel_list.append(ch)

            all_channels = sorted(volt_channel_list + temp_channel_list)
            if len(all_channels) == 0:
                return False
            elif all_channels == self._config_channels:
                return True

            try:
                self.send_command(self.commands.clean)
                self.send_command(self.commands.reset)

                _cmd = ''
                if len(volt_channel_list) != 0:
                    volt_scanlist = '(@' + ','.join(volt_channel_list) + ')'
                    _cmd = _cmd + self.commands.conf_volt + ' ' + volt_scanlist

                if len(temp_channel_list) != 0:
                    if len(_cmd) != 0:
                        _cmd = _cmd + '; '
                    temp_scanlist = '(@' + ','.join(temp_channel_list) + ')'
                    _cmd = _cmd + self.commands.conf_temp + ' ' + temp_scanlist

                self.send_command(_cmd)
                _time.sleep(wait)
                scanlist = '(@' + ','.join(all_channels) + ')'
                self.send_command(self.commands.rout_scan + ' ' + scanlist)
                _time.sleep(wait)
                self._config_channels = all_channels.copy()
                return True

            except Exception:
                return False

        def get_readings(self, wait=0.5):
            """Get reading list."""
            try:
                self.send_command(self.commands.qread)
                _time.sleep(wait)
                rstr = self.read_from_device()
                if len(rstr) != 0:
                    rlist = [float(r) for r in rstr.split(',')]
                    return rlist
                else:
                    return []
            except Exception:
                return []

        def get_scan_channels(self, wait=0.1):
            """Return the scan channel list read from the device."""
            try:
                self.send_command(self.commands.qscan)
                _time.sleep(wait)
                rstr = self.read_from_device()
                cstr = rstr.split('(@')[1].replace(')', '').replace('\n', '')
                if len(cstr) == 0:
                    return []
                else:
                    channel_list = cstr.split(',')
                    return channel_list
            except Exception:
                return []

    return Agilent34970A


Agilent34970AGPIB = Agilent34970A_factory(_interfaces.GPIBInterface)
Agilent34970ASerial = Agilent34970A_factory(_interfaces.SerialInterface)
