# -*- coding: utf-8 -*-

import os as _os
import sys as _sys
import time as _time
import signal as _signal
import ctypes as _ctypes
import numpy as _np
import matplotlib.pyplot as _plt

from hallbench.devices import new_ps as ps


def signal_handler(sig, frame):
    print('Setting current to zero...')
    ps.set_current(0)
    _sys.exit(0)


def cycle_power_supply(ps):
    sig_type = 1
    num_cycles = 15
    freq = 0.1
    amplitude = 50
    offset = 0
    aux0 = 0
    aux1 = 0
    aux2 = 15
    aux3 = 0
    success = ps.cycle(
        sig_type, num_cycles, freq, amplitude,
        offset, aux0, aux1, aux2, aux3)
    return success


# Power Supply F1000 B1
ps_port = 'COM3'
ps_address = 1
kp = 0.0400
ki = 0.1711
slope = 50
dclink = True
dclink_address = 2
dclink_voltage = 90
bipolar = True
current_min = -50
current_max = 50
ps.configure(
    ps_address, kp, ki, slope,
    dclink=dclink, dclink_address=dclink_address,
    dclink_voltage=dclink_voltage, bipolar=bipolar,
    current_min=current_min, current_max=current_max)
ps.connect(ps_port)
_time.sleep(0.1)
ps.configure_pid()
_time.sleep(0.1)

_signal.signal(_signal.SIGINT, signal_handler)
