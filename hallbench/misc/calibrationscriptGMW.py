# -*- coding: utf-8 -*-

import os as _os
import sys as _sys
import time as _time
import signal as _signal
import ctypes as _ctypes
import numpy as _np
import matplotlib.pyplot as plt

from hallbench.data import calibration as _calibration
from hallbench.devices import powersupply as _powersupply
from hallbench.devices import (
    volty,
    nmr,
    multich,
    )


CURRENT_STD_LIMIT = 0.004
VOLTAGE_STD_LIMIT = 0.0005
FIELD_STD_LIMIT = 0.0001
TEMPERATURE_LIMIT = 30
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
            fig, ax1 = plt.subplots()
            plt.ion()
            plt.show()

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
            plt.draw()
            plt.pause(0.001)

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
        current_timeout=100, turn_off=False):
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
    msgbox(
        "Information",
        "Finish {0:s} channel measurements!".format(config_list[0][1]),
        0)
    return True


def measure_channel_B(
        nmr, volt, ps, multich, filename,
        nmr_mode=1, nmr_searchtime=1,
        reading_time=30, reading_delay=40,
        current_timeout=100, turn_off=False):
    config_list = [
        [0, "B", 2800, -3],
        [0, "B", 800, -1.5],
        [1, "B", 800, 1.5],
        [1, "B", 2800, 3.0],
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
        reading_time=30, reading_delay=40,
        current_timeout=100, turn_off=False):
    config_list = [
        [0, "C", 3700, -7.5],
        [0, "C", 2800, -6],
        [0, "C", 2000, -4.5],
        [0, "C", 800, -3],
        [1, "C", 800, 3],
        [1, "C", 2000, 4.5],
        [1, "C", 2800, 6],
        [1, "C", 3700, 7.5],
    ]

    success = measure_channel(
        nmr, volt, ps, multich, filename, config_list,
        nmr_mode=nmr_mode, nmr_searchtime=nmr_searchtime,
        reading_time=reading_time, reading_delay=reading_delay,
        current_timeout=current_timeout, turn_off=turn_off)

    return success


def measure_channel_D(
        nmr, volt, ps, multich, filename,
        nmr_mode=1, nmr_searchtime=1,
        reading_time=30, reading_delay=40,
        current_timeout=100, turn_off=False):
    config_list = [
        [0, "D", 3500, -14.5],
        [0, "D", 3000, -13],
        [0, "D", 2700, -11.5],
        [0, "D", 2300, -10],
        [0, "D", 2000, -9],
        [0, "D", 1400, -7.5],
        [0, "D", 800, -6],
        [1, "D", 700, 6],
        [1, "D", 1400, 7.5],
        [1, "D", 2000, 9],
        [1, "D", 2300, 10],
        [1, "D", 2700, 11.5],
        [1, "D", 3000, 13],
        [1, "D", 3500, 14.5],
    ]

    success = measure_channel(
        nmr, volt, ps, multich, filename, config_list,
        nmr_mode=nmr_mode, nmr_searchtime=nmr_searchtime,
        reading_time=reading_time, reading_delay=reading_delay,
        current_timeout=current_timeout, turn_off=turn_off)

    return success


def measure_channel_E(
        nmr, volt, ps, multich, filename,
        nmr_mode=1, nmr_searchtime=1,
        reading_time=30, reading_delay=40,
        current_timeout=100, turn_off=False):
    config_list = [
        [0, "E", 3800, -40],
        [0, "E", 3500, -36],
        [0, "E", 3200, -32],
        [0, "E", 3000, -29],
        [0, "E", 2900, -26],
        [0, "E", 2700, -24],
        [0, "E", 2500, -22],
        [0, "E", 2300, -20.5],
        [0, "E", 2100, -19],
        [0, "E", 1900, -17.5],
        [0, "E", 1600, -16],
        [0, "E", 1300, -14.5],
        [0, "E", 1000, -13],
        [0, "E", 700, -11.5],
        [0, "E", 400, -10],
        [1, "E", 400, 10],
        [1, "E", 700, 11.5],
        [1, "E", 1000, 13],
        [1, "E", 1300, 14.5],
        [1, "E", 1600, 16],
        [1, "E", 1900, 17.5],
        [1, "E", 2100, 19],
        [1, "E", 2300, 20.5],
        [1, "E", 2500, 22],
        [1, "E", 2700, 24],
        [1, "E", 2900, 26],
        [1, "E", 3000, 29],
        [1, "E", 3200, 32],
        [1, "E", 3500, 36],
        [1, "E", 3800, 40],
    ]

    success = measure_channel(
        nmr, volt, ps, multich, filename, config_list,
        nmr_mode=nmr_mode, nmr_searchtime=nmr_searchtime,
        reading_time=reading_time, reading_delay=reading_delay,
        current_timeout=current_timeout, turn_off=turn_off)

    return success


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
    num_cycles = 7
    freq = 0.1
    amplitude = 50
    offset = 0
    aux0 = 0
    aux1 = 0
    aux2 = 10
    aux3 = 0
    success = ps.cycling(
        sig_type, num_cycles, freq, amplitude,
        offset, aux0, aux1, aux2, aux3)
    return success


def get_filename(calibration_name):
    filename = (
        "C:\\Arq\\Work_At_LNLS\\eclipse-workspace\\" +
        "hall-bench-control\\fieldmaps\\2020-07-28_GMW_Dipole\\" +
        "probes_133-14\\" + calibration_name + ".txt")
    return filename


def create_calibration_file(
        calibration_name, coeffs, voltage_lim):
    filename = get_filename(calibration_name)
    data = _np.loadtxt(filename, skiprows=1, dtype=str)
    current = data[:, 4].astype(float)
    field = data[:, 6].astype(float)
    voltage = data[:, 8].astype(float)
    temp_probe = data[:, 11].astype(float)
    temp_box = data[:, 13].astype(float)

    for i in range(len(field)):
        field[i] = field[i]*_np.sign(current[i])

    voltage = voltage - VOLTAGE_OFFSET

    voltage, field, temp_probe, temp_box = zip(*sorted(
        zip(voltage, field, temp_probe, temp_box)))

    field = _np.array(field)
    voltage = _np.array(voltage)
    temp_probe = _np.array(temp_probe)
    temp_box = _np.array(temp_box)

    field = field[abs(voltage) < voltage_lim]
    temp_probe = temp_probe[abs(voltage) < voltage_lim]
    temp_box = temp_box[abs(voltage) < voltage_lim]
    voltage = voltage[abs(voltage) < voltage_lim]

    result = _np.polynomial.polynomial.polyfit(
        voltage, field, coeffs, full=True)

    fit = _np.polynomial.polynomial.polyval(voltage, result[0])
    fit_error = (fit - field)*1e4

    hc = _calibration.HallCalibrationCurve()
    hc.function_type = 'polynomial'
    hc.calibration_name = calibration_name
    hc.calibration_magnet = 'GMW'
    hc.voltage_min = _np.min(voltage)
    hc.voltage_max = _np.max(voltage)
    hc.voltage = voltage
    hc.magnetic_field = field
    hc.probe_temperature = temp_probe
    hc.electronic_box_temperature = temp_box
    hc.polynomial_coeffs = result[0]
    hc.save_file(filename.replace('.txt', '_hc.txt'))

    fig, ax1 = plt.subplots()
    plt.ion()
    plt.show()

    color = 'tab:red'
    ax1.set_xlabel('Voltage [V]')
    ax1.set_ylabel('Field [T]', color=color)
    ax1.plot(voltage, field, color=color)
    ax1.plot(voltage, fit, '--k')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Fit Error [G]', color=color)
    ax2.plot(voltage, fit_error, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.grid()
    plt.draw()
    plt.pause(0.001)

    print('\nCoeffs:')
    print(result[0])

    print('\nLinear Coeffs:')
    print(result[0][1])

    return result[1][0][0]


# Power Supply F1000 GMW
ps_port = 'COM3'
ps_address = 1
kp = 0.5590
ki = 3.0718
slope = 50
dclink = True
dclink_address = 2
dclink_voltage = 90
bipolar = True
current_min = -50
current_max = 50
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

# Multichannel
multich_address = 18
multich_chs = [
    '101', '102', '103', '105',
    '204', '205', '207', '208', '209',
]
multich.connect(multich_address)
multich.configure(multich_chs, wait=1)

_signal.signal(_signal.SIGINT, signal_handler)
