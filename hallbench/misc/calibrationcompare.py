# -*- coding: utf-8 -*-

import os as _os
import numpy as _np
import matplotlib.pyplot as _plt


def find_polynomial_fit_senis(
        filename,
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

    field = field[abs(voltage) < voltage_lim]
    voltage = voltage[abs(voltage) < voltage_lim]

    result = _np.polynomial.polynomial.polyfit(
        voltage, field, coeffs, full=True)

    poly = result[0]
    error = result[1][0][0]
    fit_field = _np.polynomial.polynomial.polyval(voltage, poly)
    residual = (fit_field - field)*1e4

    return voltage, field, poly, error, fit_field, residual


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
voltage_offset = 0.68/1000

filenames = [
    'Z_Senis.txt',
    'Z_LNLS.txt',
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
    sign = 1
    if 'Senis' in label:
        voltage, field, poly, error, fitfield, res = find_polynomial_fit_senis(
            fullfilename,
            coeffs=coeffs,
            voltage_lim=voltage_lim,
            voltage_offset=0*voltage_offset,
            sign=sign)
    else:
        voltage, field, poly, error, fitfield, res = find_polynomial_fit(
            fullfilename,
            coeffs=coeffs,
            voltage_lim=voltage_lim,
            voltage_offset=voltage_offset,
            sign=-1)
    coeff.append(poly[coeff_idx])
    fit_error.append(error)
    polys.append(poly)
    print(poly)
    ax[0].plot(voltage, field, '-o', label=label)
    ax[1].plot(voltage, res, '-o', label=label)

ax[0].grid()
ax[0].legend(loc='best')
ax[0].set_ylabel('Field [T]')
s = 'Linear Coefficient [T/V]:\n'
for idx, label in enumerate(labels):
    s += '{0:s}: {1:.4f}\n'.format(label, coeff[idx])
s = s[:-1]
ax[0].annotate(
    s, (0, -0.5),
    bbox={'edgecolor': 'black', 'facecolor': 'white'})

ax[1].grid()
ax[1].set_ylabel('Polynomial Fit\nResidue [G]')
ax[1].set_xlabel('Probe Voltage [V]')

# fig, ax = _plt.subplots(2)

# ax[0].plot(labels, coeff, 'o-')
# ax[0].grid()
# ax[0].set_ylabel('Coefficient {0:d}'.format(coeff_idx))

# ax[1].plot(labels, fit_error, 'go-')
# ax[1].grid()
# ax[1].set_ylabel('Fit error')

fig, ax = _plt.subplots(3)

la = labels[0]
lb = labels[1]

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
ax[0].set_ylabel('Polynomial Fit\nField [T]')

ax[1].plot(voltage, diff, 'g')
ax[1].grid()
ax[1].set_ylabel('Field Difference [G]')

ax[2].plot(voltage, diffp, 'g')
ax[2].grid()
ax[2].set_ylabel('Field Difference [%]')
ax[2].set_xlabel('Probe Voltage [V]')

_plt.show()
