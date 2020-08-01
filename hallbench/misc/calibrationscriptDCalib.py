# -*- coding: utf-8 -*-

import os as _os
import sys as _sys
import time as _time
import signal as _signal
import ctypes as _ctypes
import numpy as _np
import matplotlib.pyplot as _plt

from hallbench.devices import powersupply as _powersupply
from hallbench.devices import (
    voltx,
    volty,
    voltz,
    nmr,
    multich,
    )


CURRENT_STD_LIMIT = 0.004
VOLTAGE_STD_LIMIT = 0.0005
FIELD_STD_LIMIT = 0.0001
TEMPERATURE_LIMIT = 35
STOP = False


def signal_handler(sig, frame):
    print('Setting current to zero...')
    ps.set_current(0)
    _sys.exit(0)


def msgbox(title, text, style):
    return _ctypes.windll.user32.MessageBoxW(
        0, text, title, style)


def measure_one_point(
        current_setpoint,
        nmr, volt, ps, multich,
        nmr_sense=0, nmr_channel='A', nmr_freq=0,
        reading_time=30, reading_delay=40,
        current_timeout=100, turn_off=True):
    if not ps.connected:
        msg = 'Power supply not connected.'
        raise Exception(msg)

    if not nmr.connected:
        ps.set_current(0)
        msg = 'NMR not connected.'
        raise Exception(msg)

    if not volt.connected:
        ps.set_current(0)
        msg = 'Multimeter not connected.'
        raise Exception(msg)

    if not multich.connected:
        ps.set_current(0)
        msg = 'Multichannel not connected.'
        raise Exception(msg)

    if STOP:
        ps.set_current(0)
        raise Exception('Aborting mearurements.')

    try:
        ts = []
        fs = []
        vs = []
        cs = []

        chs = multich.get_scan_channels()
        rl = multich.get_converted_readings(wait=1)

    except Exception as err:
        ps.set_current(0)
        raise Exception(err)

    if len(rl) != len(chs):
        ps.set_current(0)
        msg = 'Inconsistent temperature measurements. Aborting measurements.'
        raise Exception(msg)

    temp = [r if _np.abs(r) < 1e37 else _np.nan for r in rl]

    if (not any(_np.isnan(temp))
            and any(_np.array(temp) > TEMPERATURE_LIMIT)):
        ps.set_current(0)
        msg = 'Maximum temperature exceeded. Aborting measurements.'
        raise Exception(msg)

    if STOP:
        ps.set_current(0)
        raise Exception('Aborting mearurements.')

    try:
        if not ps.verify_current_limits(current_setpoint):
            ps.set_current(0)
            msg = 'Invalid current value. Aborting measurements.'
            raise Exception(msg)

        ps.set_current(current_setpoint)
        ct0 = _time.monotonic()

        _time.sleep(reading_delay)
        t0 = _time.monotonic()

        t = t0
        while t - t0 < reading_time:
            if t - ct0 >= current_timeout:
                break

            if STOP:
                ps.set_current(0)
                raise Exception('Aborting mearurements.')

            ts.append(t - t0)

            try:
                c = float(ps.get_current())
                cs.append(c)
            except Exception:
                cs.append(_np.nan)

            b = nmr.read_b_value().strip().replace('\r\n', '')
            if b.endswith('T'):
                if b.startswith('L'):
                    try:
                        b = b.replace('T', '')
                        fs.append(float(b[1:]))
                    except Exception:
                        fs.append(_np.nan)

                else:
                    fs.append(_np.nan)
            else:
                fs.append(_np.nan)

            try:
                v = float(volt.read_from_device()[:-2])
                vs.append(v)
            except Exception:
                vs.append(_np.nan)

            t = _time.monotonic()

        print('DAC: ', nmr.read_dac_value())

        if turn_off:
            ps.set_current(0)

        cn = [c for c in cs if not _np.isnan(c)]
        fn = [f for f in fs if not _np.isnan(f)]
        vn = [v for v in vs if not _np.isnan(v)]

        if len(cn) > 0:
            cmean = _np.mean(cn)
            cstd = _np.std(cn)
        else:
            cmean = _np.nan
            cstd = _np.nan

        if len(fn) > 0:
            fmean = _np.mean(fn)
            fstd = _np.std(fn)
        else:
            fmean = _np.nan
            fstd = _np.nan

        if len(vn) > 0:
            vmean = _np.mean(vn)
            vstd = _np.std(vn)
        else:
            vmean = _np.nan
            vstd = _np.nan

        if nmr_sense == 0:
            nmr_sense_str = 'Negative'
        else:
            nmr_sense_str = 'Positive'

        readings = [
            nmr_sense_str, nmr_channel, nmr_freq,
            current_setpoint, cmean, cstd,
            fmean, fstd, vmean, vstd]

        for tr in temp:
            readings.append(tr)

        problem = 0

        if any(_np.isnan([
                cmean, cstd, fmean, fstd, vmean, vstd])):
            print('Problem: NaN found\n')
            problem = 1

        elif cstd >= CURRENT_STD_LIMIT:
            print('Problem: Current STD [A] : {0:f}\n'.format(cstd))
            problem = 1

        elif vstd >= VOLTAGE_STD_LIMIT:
            print('Problem: Voltage STD [V] : {0:f}\n'.format(vstd))
            problem = 1

        elif fstd >= FIELD_STD_LIMIT:
            print('Problem: Field STD [T] : {0:f}\n'.format(fstd))
            problem = 1

        if problem:
            fig, ax1 = _plt.subplots()
            _plt.ion()
            _plt.show()

            color = 'tab:red'
            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Voltage [V]', color=color)
            ax1.plot(ts, vs, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Field [T]', color=color)
            ax2.plot(ts, fs, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig.suptitle(
                'Current [A]: {0:.2f}'.format(current_setpoint),
                fontsize=16)
            fig.tight_layout()
            _plt.draw()
            _plt.pause(0.001)

            return readings, 0

        return readings, 1

    except Exception as err:
        ps.set_current(0)
        raise Exception(err)


def configure_nmr_and_measure_one_point(
        current_setpoint,
        nmr, volt, ps, multich,
        nmr_sense=0, nmr_channel='A', nmr_freq=0,
        nmr_mode=2, nmr_searchtime=1,
        reading_time=30, reading_delay=40,
        current_timeout=100, turn_off=True):
    if not nmr.connected:
        ps.set_current(0)
        msg = 'NMR not connected.'
        raise Exception(msg)

    nmr_configured = nmr.configure(
        nmr_mode, nmr_freq, nmr_sense, 1, 0,
        nmr_searchtime, nmr_channel, 1)

    if not nmr_configured:
        ps.set_current(0)
        msg = 'Failed to configure NMR.'
        raise Exception(msg)

    readings, success = measure_one_point(
        current_setpoint,
        nmr, volt, ps, multich,
        nmr_sense=nmr_sense, nmr_channel=nmr_channel, nmr_freq=nmr_freq,
        reading_time=reading_time, reading_delay=reading_delay,
        current_timeout=current_timeout, turn_off=turn_off)

    if not success:
        ps.set_current(0)

        msg = "Measure for {0:.2f} A failed. Try again?".format(
            current_setpoint)

        reply = msgbox("Question", msg, 3)

        if reply == 6:
            readings, success = measure_one_point(
                current_setpoint,
                nmr, volt, ps, multich,
                nmr_sense=nmr_sense,
                nmr_channel=nmr_channel,
                nmr_freq=nmr_freq,
                reading_time=reading_time,
                reading_delay=reading_delay,
                current_timeout=current_timeout,
                turn_off=turn_off)

        elif reply == 7:
            success = 1

        else:
            success = 0

    return readings, success


def measure_channel(
        nmr, volt, ps, multich, filename, config_list,
        nmr_mode=2, nmr_searchtime=1,
        reading_time=30, reading_delay=40,
        current_timeout=100, turn_off=False, msg=True):
    for config in config_list:
        print("Channel {0:s}: {1:f}".format(config[1], config[3]))

        readings, success = configure_nmr_and_measure_one_point(
            config[3],
            nmr, volt, ps, multich,
            nmr_sense=config[0],
            nmr_channel=config[1],
            nmr_freq=config[2],
            nmr_mode=nmr_mode,
            nmr_searchtime=nmr_searchtime,
            reading_time=reading_time,
            reading_delay=reading_delay,
            current_timeout=current_timeout,
            turn_off=turn_off)

        if not success:
            ps.set_current(0)
            msgbox(
                "Information",
                "{0:s} channel measurements failed.".format(config[1]),
                0)
            return False

        readings = [str(r) for r in readings]

        with open(filename, 'a') as f:
            f.write('\t'.join(readings))
            f.write('\n')

    ps.set_current(0)
    if msg:
        msgbox(
            "Information",
            "Finish {0:s} channel measurements!".format(config_list[0][1]),
            0)
    return True


def measure_channel_B(
        nmr, volt, ps, multich, filename,
        nmr_mode=1, nmr_searchtime=1,
        reading_time=30, reading_delay=20,
        current_timeout=100, turn_off=False):
    config_list = [
        [1, 'B', 3300, -98],
        [1, 'B', 3000, -90],
        [1, 'B', 2700, -82],
        [1, 'B', 2400, -73],
        [1, 'B', 2100, -66],
        [1, 'B', 1800, -58],
        [1, 'B', 1300, -50],
        [1, 'B', 900, -43],
        [0, 'B', 800, 43],
        [0, 'B', 1200, 50],
        [0, 'B', 1700, 58],
        [0, 'B', 2100, 66],
        [0, 'B', 2400, 73],
        [0, 'B', 2700, 82],
        [0, 'B', 3000, 90],
        [0, 'B', 3300, 98],
    ]

    success = measure_channel(
        nmr, volt, ps, multich, filename, config_list,
        nmr_mode=nmr_mode, nmr_searchtime=nmr_searchtime,
        reading_time=reading_time, reading_delay=reading_delay,
        current_timeout=current_timeout, turn_off=turn_off)

    return success


def measure_channel_C(
        nmr, volt, ps, multich, filename,
        nmr_mode=1, nmr_searchtime=1,
        reading_time=30, reading_delay=10,
        current_timeout=60, turn_off=False):
    config_list = [
        [1, 'C', 3200, -198],
        [1, 'C', 3000, -187],
        [1, 'C', 2900, -175],
        [1, 'C', 2700, -165],
        [1, 'C', 2500, -154],
        [1, 'C', 2400, -144],
        [1, 'C', 2200, -135],
        [1, 'C', 2000, -125],
        [1, 'C', 1700, -116],
        [1, 'C', 1500, -107],
        [0, 'C', 1400, 107],
        [0, 'C', 1700, 116],
        [0, 'C', 1900, 125],
        [0, 'C', 2100, 135],
        [0, 'C', 2300, 144],
        [0, 'C', 2500, 154],
        [0, 'C', 2700, 165],
        [0, 'C', 2900, 175],
        [0, 'C', 3000, 187],
        [0, 'C', 3200, 198],
    ]

    success = measure_channel(
        nmr, volt, ps, multich, filename, config_list,
        nmr_mode=nmr_mode, nmr_searchtime=nmr_searchtime,
        reading_time=reading_time, reading_delay=reading_delay,
        current_timeout=current_timeout, turn_off=turn_off)

    return success


def measure_channel_D_part1(
        nmr, volt, ps, multich, filename,
        nmr_mode=1, nmr_searchtime=1,
        reading_time=8, reading_delay=4,
        current_timeout=25, turn_off=False):
    config_list = [
        [1, 'D', 1800, -283],
        [1, 'D', 1600, -266],
        [1, 'D', 1500, -251],
        [1, 'D', 1400, -237],
        [1, 'D', 1300, -223],
        [1, 'D', 1200, -210],
    ]
    success = measure_channel(
        nmr, volt, ps, multich, filename, config_list,
        nmr_mode=nmr_mode, nmr_searchtime=nmr_searchtime,
        reading_time=reading_time, reading_delay=reading_delay,
        current_timeout=current_timeout, turn_off=turn_off, msg=False)

    ps.set_current(0)
    _time.sleep(300)

    config_list = [
        [0, 'D', 1200, 210],
        [0, 'D', 1300, 223],
        [0, 'D', 1400, 237],
        [0, 'D', 1500, 251],
        [0, 'D', 1600, 266],
        [0, 'D', 1800, 283],
    ]
    success = measure_channel(
        nmr, volt, ps, multich, filename, config_list,
        nmr_mode=nmr_mode, nmr_searchtime=nmr_searchtime,
        reading_time=reading_time, reading_delay=reading_delay,
        current_timeout=current_timeout, turn_off=turn_off)

    return success


def measure_channel_D_part2(
        nmr, volt, ps, multich, filename,
        nmr_mode=1, nmr_searchtime=1,
        reading_time=10, reading_delay=20,
        current_timeout=30, turn_off=True):
    config_list = [
        [1, 'D', 2050, -380],
        [1, 'D', 2000, -348],
        [1, 'D', 1900, -323],
        [1, 'D', 1800, -302],
        [0, 'D', 1800, 302],
        [0, 'D', 1900, 323],
        [0, 'D', 2000, 348],
        [0, 'D', 2050, 380],
    ]
    for config in config_list:
        success = measure_channel(
            nmr, volt, ps, multich, filename, [config],
            nmr_mode=nmr_mode, nmr_searchtime=nmr_searchtime,
            reading_time=reading_time, reading_delay=reading_delay,
            current_timeout=current_timeout, turn_off=turn_off, msg=False)
        ps.set_current(0)
        _time.sleep(300)
        if not success:
            return False

    msgbox(
        "Information",
        "Finish {0:s} channel measurements!".format(config_list[0][1]),
        0)

    return True


def create_file(filename, multich, overwrite=False):
    headers = [
        'NMRSense',
        'NMRChannel',
        'NMRFreq',
        'CurrentSP[A]',
        'CurrentAVG[A]',
        'CurrentSTD[A]',
        'NMRFieldAVG[T]',
        'NMRFieldSTD[T]',
        'VoltageAVG[V]',
        'VoltageSTD[V]',
    ]

    chs = multich.get_scan_channels()
    for ch in chs:
        headers.append('CH{0}[degC]'.format(ch))

    if not _os.path.isfile(filename) or overwrite:
        with open(filename, 'w+') as f:
            f.write('\t'.join(headers))
            f.write('\n')


def cycle_power_supply(ps):
    sig_type = 1
    num_cycles = 15
    freq = 0.1
    amplitude = 400
    offset = 0
    aux0 = 0
    aux1 = 0
    aux2 = 15
    aux3 = 0
    success = ps.cycling(
        sig_type, num_cycles, freq, amplitude,
        offset, aux0, aux1, aux2, aux3)
    return success


def get_filename(calibration_name):
    filename = (
        "C:\\Arq\\Work_At_LNLS\\eclipse-workspace\\" +
        "hall-bench-control\\fieldmaps\\2020-07-22_D-Calib\\probes_133-14\\" +
        calibration_name + ".txt")
    return filename


# Power Supply F1000 DCalib
ps_port = 'COM3'
ps_address = 1
kp = 0.0480
ki = 0.6507
slope = 50
dclink = True
dclink_address = 2
dclink_voltage = 90
bipolar = True
current_min = -400
current_max = 400
ps = _powersupply.PowerSupply(
    ps_port, ps_address, kp, ki, slope,
    dclink=dclink, dclink_address=dclink_address,
    dclink_voltage=dclink_voltage, bipolar=bipolar,
    current_min=current_min, current_max=current_max)
ps.connect()
_time.sleep(0.1)
ps.configure_pid()
_time.sleep(0.1)

# NMR
nmr_port = 'COM5'
nmr_baudrate = 19200
nmr.connect(nmr_port, baudrate=nmr_baudrate)

# Multimeter
volty_address = 21
volty.connect(volty_address)
voltz_address = 22
voltz.connect(voltz_address)

# Multichannel
multich_address = 18
multich_chs = [
    '101', '102', '103', '105',
    '205', '206', '207', '209',
]
multich.connect(multich_address)
multich.configure(multich_chs, wait=1)

_signal.signal(_signal.SIGINT, signal_handler)
