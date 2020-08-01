# -*- coding: utf-8 -*-

import os as _os
import numpy as _np
from hallbench.data import calibration as _calibration
import matplotlib.pyplot as _plt


def find_polynomial_fit(
        filename,
        coeffs=[1, 2, 3, 5, 7, 9],
        voltage_lim=None,
        voltage_offset=0,
        sign=1):
    data = _np.loadtxt(filename, skiprows=1, dtype=str)
    current_sp = data[:, 3].astype(float)
    field = data[:, 6].astype(float)
    voltage = data[:, 8].astype(float)

    if voltage_lim is None:
        voltage_lim = _np.max(_np.abs(voltage))

    for i in range(len(field)):
        field[i] = field[i]*_np.sign(current_sp[i])

    voltage = voltage - voltage_offset

    voltage, field = zip(*sorted(
        zip(voltage, field)))

    field = _np.array(field)
    voltage = _np.array(voltage)

    field = field*sign

    field = field[abs(voltage) < voltage_lim]
    voltage = voltage[abs(voltage) < voltage_lim]

    result = _np.polynomial.polynomial.polyfit(
        voltage, field, coeffs, full=True)

    poly = result[0]
    error = result[1][0][0]
    fit_field = _np.polynomial.polynomial.polyval(voltage, poly)
    residual = (fit_field - field)*1e4

    return voltage, field, poly, error, fit_field, residual


folder = (
    "C:\\Arq\\Work_At_LNLS\\eclipse-workspace\\" +
    "hall-bench-control\\fieldmaps\\CalibrationData")
coeffs = [1, 2, 3]
voltage_lim = 4
voltage_offset = 1.4740/1000

filenames = [
    'M1.txt',
    'M2.txt',
    'M3.txt',
    'M4.txt',
    'M5.txt',
    'M6.txt',
    'M7.txt',
    'M8.txt',
    'M9.txt',
    'M10.txt',
    'M11.txt',
    'M12.txt',
    'M13.txt',
    'M14.txt',
    'M15.txt',
    'M16.txt',
    'M17.txt',
    ]

labels = [f.replace('.txt', '') for f in filenames]

data = {}
for label in labels:
    data[label] = {}

fig, ax = _plt.subplots(2)

coeff_idx = 1
coeff = []
fit_error = []
polys = []
for idx, filename in enumerate(filenames):
    fullfilename = _os.path.join(folder, filename)
    label = labels[idx]
    if label in [
            'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15']:
        sign = -1
    else:
        sign = 1
    voltage, field, poly, error, fit_field, residual = find_polynomial_fit(
        fullfilename,
        coeffs=coeffs,
        voltage_lim=voltage_lim,
        voltage_offset=voltage_offset,
        sign=sign)
    coeff.append(poly[coeff_idx])
    fit_error.append(error)
    polys.append(poly)
    ax[0].plot(voltage, field, label=label)
    ax[1].plot(voltage, residual, label=label)

ax[0].grid()
ax[0].legend(loc='best')
ax[0].set_ylabel('Field [T]')

ax[1].grid()
ax[1].set_ylabel('Fit Residual [G]')
ax[1].set_xlabel('Voltage [V]')

fig, ax = _plt.subplots(2)

ax[0].plot(labels, coeff, 'o-')
ax[0].grid()
ax[0].set_ylabel('Coefficient {0:d}'.format(coeff_idx))

ax[1].plot(labels, fit_error, 'go-')
ax[1].grid()
ax[1].set_ylabel('Fit error')

fig, ax = _plt.subplots(3)

la = 'M15'
lb = 'M16'

idxa = labels.index(la)
idxb = labels.index(lb)

voltage = _np.linspace(-voltage_lim, voltage_lim, 101)

fielda = _np.polynomial.polynomial.polyval(voltage, polys[idxa])
fieldb = _np.polynomial.polynomial.polyval(voltage, polys[idxb])
diff = (fieldb - fielda)*1e4
diffp = 100*(fieldb - fielda)/fieldb

ax[0].plot(voltage, fielda, label=la)
ax[0].plot(voltage, fieldb, label=lb)
ax[0].grid()
ax[0].legend(loc='best')
ax[0].set_ylabel('Field [T]')

ax[1].plot(voltage, diff)
ax[1].grid()
ax[1].set_ylabel('Difference [G]')

ax[2].plot(voltage, diffp, 'g')
ax[2].grid()
ax[2].set_ylabel('Difference [%]')
ax[2].set_xlabel('Voltage [V]')

_plt.show()
