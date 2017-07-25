# -*- coding: utf-8 -*-
"""Variables and constants to be used in Agilent 34970A."""


class ListOfCommands(object):
    """List of Commands of Agilent 34970 Data Acquisition/Switch Unit."""

    def __init__(self):
        """Initiate all variables."""
        self._reset()
        self._clean()
        self._lock()
        self._remote_access()
        self._set_multichannel()

    def _reset(self):
        """Reset function."""
        self.reset = '*RST'

    def _clean(self):
        """Clean error function."""
        self.clean = '*CLS'

    def _lock(self):
        """Lock function."""
        self.lock = ':SYST:LOC'

    def _remote_access(self):
        """Remote access function."""
        self.remote = ':SYST:REM;:SYST:RWL'

    def _set_multichannel(self):
        """List of commands to set the multichannel."""
        self.route = ':ROUT:SCAN (@101:107)'
        self.conf_temp = ':CONF:TEMP RTD,85,(@101:107)'
        self.monitor_on = 'ROUT:MON:STATE ON'
        self.sense_ch1 = ':SENS:TEMP:TRAN:FRTD:RES 100.42,(@101)'
        self.sense_ch2 = ':SENS:TEMP:TRAN:FRTD:RES 100.38,(@102)'
        self.sense_ch3 = ':SENS:TEMP:TRAN:FRTD:RES 100.24,(@103)'
        self.sense_ch4 = ':SENS:TEMP:TRAN:FRTD:RES 100.20,(@104)'
        self.sense_ch5 = ':SENS:TEMP:TRAN:FRTD:RES 100.16,(@105)'
        self.sense_ch6 = ':SENS:TEMP:TRAN:FRTD:RES 102.82,(@106)'
        self.sense_ch7 = ':SENS:TEMP:TRAN:FRTD:RES 100.40,(@107)'
