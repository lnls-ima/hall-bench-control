# -*- coding: utf-8 -*-

import os as _os
import numpy as _np
import pandas as _pd
import matplotlib.pyplot as _plt

from hallbench.data import calibration as _calibration


def plot_fit_results(calibration_name, voltage, field, poly, residual, error):
    _, ax = _plt.subplots(2)
    ax[0].plot(voltage, field, '-o')
    ax[0].grid()
    ax[0].set_ylabel('Field [T]')
    s = 'Polynomial Fit:\n['
    for p in poly:
        s += '{0:.4g}, '.format(p) if p != 0 else '0, '
    s = s[:-2]
    s += ']'
    ax[0].annotate(
        s,
        (_np.min(voltage)*1.05, _np.max(field)*0.75),
        bbox={'edgecolor': 'black', 'facecolor': 'white'})
    ax[0].set_title(calibration_name)

    result = _np.polynomial.polynomial.polyfit(
        voltage, residual, [2], full=True)
    fit_residual = _np.polynomial.polynomial.polyval(voltage, result[0])
    print('residual: ', result[0]/1e4)

    ax[1].plot(voltage, residual, '-or')
    ax[1].plot(voltage, fit_residual, '--k')
    ax[1].grid()
    ax[1].set_ylabel('Polynomial Fit\nResidue [G]')
    ax[1].set_xlabel('Voltage [V]')
    ax[1].annotate(
        'Fit Error: {0:g}'.format(error),
        (_np.min(voltage)*1.05, _np.max(residual)*0.9),
        bbox={'edgecolor': 'black', 'facecolor': 'white'})


def find_polynomial_fit_senis(
        filename,
        calibration_name,
        coeffs=[1, 2, 3, 5, 7, 9],
        voltage_lim=None,
        voltage_offset=0,
        sign=1):
    data = _np.loadtxt(filename, skiprows=1, dtype=str)
    field = data[:, 0].astype(float)
    voltage = data[:, 1].astype(float)

    if voltage_lim is None:
        voltage_lim = _np.max(_np.abs(voltage))

    voltage = voltage - voltage_offset

    voltage, field = zip(*sorted(
        zip(voltage, field)))

    field = _np.array(field)
    voltage = _np.array(voltage)

    field = field*sign

    field = field[abs(voltage) <= voltage_lim]
    voltage = voltage[abs(voltage) <= voltage_lim]

    result = _np.polynomial.polynomial.polyfit(
        voltage, field, coeffs, full=True)

    poly = result[0]
    error = result[1][0][0]
    fit_field = _np.polynomial.polynomial.polyval(voltage, poly)
    residual = (fit_field - field)*1e4

    plot_fit_results(calibration_name, voltage, field, poly, residual, error)

    print('')
    print(coeffs)
    print(poly)
    print(error*1e6)
    return voltage, field, poly, None, None


def find_polynomial_fit(
        filename,
        calibration_name,
        coeffs=[1, 2, 3, 5, 7, 9],
        voltage_lim=None,
        voltage_offset=0,
        sign=1):
    data = _np.loadtxt(filename, skiprows=1, dtype=str)
    current_sp = data[:, 3].astype(float)
    field = data[:, 6].astype(float)
    voltage = data[:, 8].astype(float)

    if _np.shape(data)[1] >= 14:
        valid_temp = 1
        temp_probe = data[:, 11].astype(float)
        temp_box = data[:, 13].astype(float)
    else:
        valid_temp = 0
        temp_probe = _np.zeros(len(voltage))
        temp_box = _np.zeros(len(voltage))

    if voltage_lim is None:
        voltage_lim = _np.max(_np.abs(voltage))

    for i in range(len(field)):
        field[i] = field[i]*_np.sign(current_sp[i])

    voltage = voltage - voltage_offset

    voltage, field, temp_probe, temp_box = zip(*sorted(
        zip(voltage, field, temp_probe, temp_box)))

    field = _np.array(field)
    voltage = _np.array(voltage)
    temp_probe = _np.array(temp_probe)
    temp_box = _np.array(temp_box)

    field = field[abs(voltage) <= voltage_lim]
    temp_probe = temp_probe[abs(voltage) <= voltage_lim]
    temp_box = temp_box[abs(voltage) <= voltage_lim]
    voltage = voltage[abs(voltage) <= voltage_lim]

    field = field*sign

    result = _np.polynomial.polynomial.polyfit(
        voltage, field, coeffs, full=True)

    poly = result[0]
    error = result[1][0][0]
    fit_field = _np.polynomial.polynomial.polyval(voltage, poly)
    residual = (fit_field - field)*1e4

    plot_fit_results(calibration_name, voltage, field, poly, residual, error)

    print('')
    print(coeffs)
    print(poly)
    print(error*1e6)
    if not valid_temp:
        temp_probe = None
        temp_box = None

    return voltage, field, poly, temp_probe, temp_box


def write_file_and_plot(fd, filename):
    folder = (
        "C:\\Arq\\Work_At_LNLS\\eclipse-workspace\\" +
        "hall-bench-control\\fieldmaps\\2020-08-01_CalibrationData\\RawData")
    fullfilename = _os.path.join(folder, filename)
    d = fd[filename]

    if 'Senis' in filename:
        voltage, field, poly, temp_probe, temp_box = find_polynomial_fit_senis(
            fullfilename,
            d['calibration_name'],
            coeffs=d['coeffs'],
            voltage_lim=None,
            voltage_offset=d['voltage_offset'],
            sign=d['sign'])
    else:
        voltage, field, poly, temp_probe, temp_box = find_polynomial_fit(
            fullfilename,
            d['calibration_name'],
            coeffs=d['coeffs'],
            voltage_lim=None,
            voltage_offset=d['voltage_offset'],
            sign=d['sign'])

    hc = _calibration.HallCalibrationCurve()
    hc.date = d['date']
    hc.hour = '00:00:00'
    hc.function_type = 'polynomial'
    hc.calibration_name = d['calibration_name']
    hc.calibration_magnet = d['calibration_magnet']
    hc.voltage_min = d['voltage_min']
    hc.voltage_max = d['voltage_max']
    hc.voltage_offset = d['voltage_offset']
    hc.voltage = voltage
    hc.magnetic_field = field
    hc.probe_temperature = temp_probe
    hc.electronic_box_temperature = temp_box
    hc.polynomial_coeffs = poly

    save_folder = _os.path.join(folder, 'calibration_files')
    hc.save_file(_os.path.join(save_folder, d['calibration_name'] + '.txt'))

    df = _pd.DataFrame([d['calibration_name']])
    df.to_clipboard(index=False, header=False)
    _plt.show()


# Dict of calibration files
fd = {}

# Senis calibration data for probes 133-14
filename = '133-14X_BySenis.txt'
date = '2020-05-11'
coeffs = [1, 3, 5, 7]
sign = 1
voltage_min = -10
voltage_max = 10
voltage_offset = -1*0.1810/1000
calibration_name = date + '_Probe133-14X_BySenis_2T'
calibration_magnet = 'ElectromagnetBrukerBM6'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '133-14Y_BySenis.txt'
date = '2020-05-11'
coeffs = [1, 3, 5, 7]
sign = 1
voltage_min = -10
voltage_max = 10
voltage_offset = -0.1830/1000
calibration_name = date + '_Probe133-14Y_BySenis_2T'
calibration_magnet = 'ElectromagnetBrukerBE15'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '133-14Z_BySenis.txt'
date = '2020-05-11'
coeffs = [1, 3, 5, 7]
sign = 1
voltage_min = -10
voltage_max = 10
voltage_offset = 0*0.6610/1000
calibration_name = date + '_Probe133-14Z_BySenis_2T'
calibration_magnet = 'ElectromagnetBrukerBE15'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

# Senis calibration data for probes 134-14
filename = '134-14X_BySenis.txt'
date = '2020-05-12'
coeffs = [1, 2, 3, 5, 7]
sign = 1
voltage_min = -10
voltage_max = 10
voltage_offset = -0.280/1000
calibration_name = date + '_Probe134-14X_BySenis_2T'
calibration_magnet = 'ElectromagnetBrukerBM6'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '134-14Y_BySenis.txt'
date = '2020-05-12'
coeffs = [1, 2, 3, 5, 7]
sign = 1
voltage_min = -10
voltage_max = 10
voltage_offset = -1*1.163/1000
calibration_name = date + '_Probe134-14Y_BySenis_2T'
calibration_magnet = 'ElectromagnetBrukerBE15'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '134-14Z_BySenis.txt'
date = '2020-05-12'
coeffs = [1, 2, 3, 5, 7]
sign = 1
voltage_min = -10
voltage_max = 10
voltage_offset = 0.237/1000
calibration_name = date + '_Probe134-14Z_BySenis_2T'
calibration_magnet = 'ElectromagnetBrukerBE15'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

# LNLS calibration data for probe 135-14Y
filename = '135-14Y_M1.txt'
date = '2020-02-21'
coeffs = [1, 3]
sign = 1
voltage_min = -3.5
voltage_max = 3.5
voltage_offset = 1.4740/1000
calibration_name = date + '_Probe135-14Y_M1_0.7T'
calibration_magnet = 'CalibrationDipole'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M2.txt'
date = '2020-03-02'
coeffs = [1, 3]
sign = 1
voltage_min = -3.5
voltage_max = 3.5
voltage_offset = 1.4740/1000
calibration_name = date + '_Probe135-14Y_M2_0.7T'
calibration_magnet = 'CalibrationDipole'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M3.txt'
date = '2020-03-02'
coeffs = [1, 3]
sign = 1
voltage_min = -3.5
voltage_max = 3.5
voltage_offset = 1.4740/1000
calibration_name = date + '_Probe135-14Y_M3_0.7T'
calibration_magnet = 'CalibrationDipole'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M4.txt'
date = '2020-03-20'
coeffs = [1, 2, 3]
sign = 1
voltage_min = -3.5
voltage_max = 3.5
voltage_offset = 1.4740/1000
calibration_name = date + '_Probe135-14Y_M4_0.7T'
calibration_magnet = 'CalibrationDipole'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M5.txt'
date = '2020-03-21'
coeffs = [1, 2, 3]
sign = 1
voltage_min = -3.5
voltage_max = 3.5
voltage_offset = 1.4740/1000
calibration_magnet = 'CalibrationDipole'
calibration_name = date + '_Probe135-14Y_M5_07T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M6.txt'
date = '2020-06-11'
coeffs = [1, 2, 3, 5, 7]
sign = -1
voltage_min = -10
voltage_max = 10
voltage_offset = 1.4740/1000
calibration_magnet = 'GMWDipole'
calibration_name = date + '_Probe135-14Y_M6_2T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M7.txt'
date = '2020-06-11'
coeffs = [1, 2, 3, 4, 5, 7, 9]
sign = -1
voltage_min = -15
voltage_max = 15
voltage_offset = 1.4740/1000
calibration_magnet = 'GMWDipole'
calibration_name = date + '_Probe135-14Y_M7_3T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M8.txt'
date = '2020-06-12'
coeffs = [1, 2, 3, 4, 5, 7, 9]
sign = -1
voltage_min = -15
voltage_max = 15
voltage_offset = 1.4740/1000
calibration_magnet = 'GMWDipole'
calibration_name = date + '_Probe135-14Y_M8_3T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M9.txt'
date = '2020-06-13'
coeffs = [1, 2, 3, 5, 7]
sign = -1
voltage_min = -7
voltage_max = 7
voltage_offset = 1.4740/1000
calibration_magnet = 'GMWDipole'
calibration_name = date + '_Probe135-14Y_M9_1.4T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M10.txt'
date = '2020-07-16'
coeffs = [1, 2, 3, 5, 7]
sign = -1
voltage_min = -11
voltage_max = 11
voltage_offset = 1.4740/1000
calibration_magnet = 'GMWDipole'
calibration_name = date + '_Probe135-14Y_M10_2T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M11.txt'
date = '2020-07-17'
coeffs = [1, 2, 3, 5, 7]
sign = -1
voltage_min = -11
voltage_max = 11
voltage_offset = 1.4740/1000
calibration_magnet = 'GMWDipole'
calibration_name = date + '_Probe135-14Y_M11_2T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M12.txt'
date = '2020-07-21'
coeffs = [1, 2, 3, 5, 7]
sign = -1
voltage_min = -11
voltage_max = 11
voltage_offset = 1.4740/1000
calibration_magnet = 'GMWDipole'
calibration_name = date + '_Probe135-14Y_M12_2T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M13.txt'
date = '2020-07-21'
coeffs = [1, 2, 3, 5, 7]
sign = -1
voltage_min = -10
voltage_max = 10
voltage_offset = 1.4740/1000
calibration_magnet = 'GMWDipole'
calibration_name = date + '_Probe135-14Y_M13_2T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M14.txt'
date = '2020-07-21'
coeffs = [1, 2, 3]
sign = -1
voltage_min = -5
voltage_max = 5
voltage_offset = 1.4740/1000
calibration_magnet = 'GMWDipole'
calibration_name = date + '_Probe135-14Y_M14_1T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M15.txt'
date = '2020-07-22'
coeffs = [1, 2, 3, 5, 7]
sign = -1
voltage_min = -10
voltage_max = 10
voltage_offset = 1.4740/1000
calibration_magnet = 'GMWDipole'
calibration_name = date + '_Probe135-14Y_M15_2T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M16.txt'
date = '2020-07-23'
coeffs = [1, 2, 3]
sign = 1
voltage_min = -3.5
voltage_max = 3.5
voltage_offset = 1.4740/1000
calibration_magnet = 'CalibrationDipole'
calibration_name = date + '_Probe135-14Y_M16_0.7T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M17.txt'
date = '2020-07-23'
coeffs = [1, 2, 3]
sign = 1
voltage_min = -3.5
voltage_max = 3.5
voltage_offset = 1.4740/1000
calibration_magnet = 'CalibrationDipole'
calibration_name = date + '_Probe135-14Y_M17_0.7T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M18.txt'
date = '2020-07-30'
coeffs = [1, 2, 3, 5, 7, 9]
sign = -1
voltage_min = -13
voltage_max = 13
voltage_offset = 1.4740/1000
calibration_magnet = 'GMWDipole'
calibration_name = date + '_Probe135-14Y_M18_2.6T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '135-14Y_M19.txt'
date = '2020-07-30'
coeffs = [1, 2, 3, 5, 7, 9]
sign = -1
voltage_min = -15
voltage_max = 15
voltage_offset = 1.4740/1000
calibration_magnet = 'GMWDipole'
calibration_name = date + '_Probe135-14Y_M19_3T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

# LNLS calibration data for probe 133-14Y
filename = '133-14Y_M1.txt'
date = '2020-07-27'
coeffs = [1, 2, 3]
sign = 1
voltage_min = -3.5
voltage_max = 3.5
voltage_offset = -0.1830/1000
calibration_magnet = 'CalibrationDipole'
calibration_name = date + '_Probe133-14Y_M1_0.7T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '133-14Y_M2.txt'
date = '2020-07-28'
coeffs = [1, 2, 3, 5, 7, 9]
sign = -1
voltage_min = -11
voltage_max = 11
voltage_offset = -0.1830/1000
calibration_magnet = 'GMWDipole'
calibration_name = date + '_Probe133-14Y_M2_2T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

selected_filename = filename

filename = '133-14Y_M3.txt'
date = '2020-07-29'
coeffs = [1, 2, 3, 5, 7, 9]
sign = -1
voltage_min = -15
voltage_max = 15
voltage_offset = -0.1830/1000
calibration_magnet = 'GMWDipole'
calibration_name = date + '_Probe133-14Y_M3_3T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

filename = '133-14Y_M4.txt'
date = '2020-07-30'
coeffs = [1, 2, 3, 5, 7, 9]
sign = -1
voltage_min = -13
voltage_max = 13
voltage_offset = -0.1830/1000
calibration_magnet = 'GMWDipole'
calibration_name = date + '_Probe133-14Y_M4_2.6T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

# LNLS calibration data for probe 133-14Z
filename = '133-14Z.txt'
date = '2020-07-30'
coeffs = [1, 2, 3]
sign = -1
voltage_min = -3.5
voltage_max = 3.5
voltage_offset = 0.6610/1000
calibration_magnet = 'CalibrationDipole'
calibration_name = date + '_Probe133-14Z_0.7T'
fd[filename] = {
    'date': date, 'coeffs': coeffs, 'sign': sign, 'voltage_min': voltage_min,
    'voltage_max': voltage_max, 'voltage_offset': voltage_offset,
    'calibration_name': calibration_name,
    'calibration_magnet': calibration_magnet}

write_file_and_plot(fd, selected_filename)
