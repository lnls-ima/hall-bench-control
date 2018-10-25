'''
Created on 9 de out de 2018

@author: Vitor Soares
'''

import minimalmodbus as _minimalmodbus
import numpy as _np
import threading as _threading
import time as _time


class UDC3500():
    """Honeywell UDC-3500 control class."""
    
    def __init__(self):
        """Honeywell UDC-3500 control class."""
        self.inst = None
        self.pv1 = _np.array([])
        self.pv2 = _np.array([])
        self.co = _np.array([])
        self.t = _np.array([])
        self.flag_collect = True
        self.interval = 1

    @property
    def connected(self):
        """Return True if the port is open, False otherwise."""
        if self.inst is None:
            return False
        else:
            return self.inst.serial.is_open

    def connect(self, port, baudrate):
        """Connect device."""
        self.inst = _minimalmodbus.Instrument(port, 14)
        self.inst.serial.baudrate = baudrate
        if not self.inst.serial.is_open:
            self.inst.serial.open()

    def disconnect(self):
        """Disconnect the device."""
        try:
            if self.inst is not None:
                self.inst.serial.close()
            return True
        except Exception:
            return None

    def read_pv1(self):
        """Returns process variable."""
        return self.inst.read_float(72)

    def read_pv2(self):
        """Returns process variable 2."""
        return self.inst.read_float(74)

    def read_co(self):
        """Returns controller output."""
        return self.inst.read_float(70)

    def clear(self):
        """Clears arrays."""
        self.pv1 = _np.array([])
        self.pv2 = _np.array([])
        self.co = _np.array([])
        self.t = _np.array([])

    def _collect(self):
        """Collects data."""
        self.clear()
        _t0 = _time.time()
        while self.flag_collect:
            self.pv1 = _np.append(self.pv1, self.read_pv1())
            self.pv2 = _np.append(self.pv2, self.read_pv2())
            self.co = _np.append(self.co, self.read_co())
            self.t = _np.append(self.t, _time.time() - _t0)
            _time.sleep(self.interval)

    def start_collect(self, time_interval):
        """Starts collect data routine."""
        self.interval = time_interval
        thread = _threading.Thread(target=self._collect)
        thread.daemon = True
        thread.start()

    def stop_collect(self):
        """Stops collect data routine."""
        self.flag_collect = False
        time.sleep(2*self.interval)
