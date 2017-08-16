# -*- coding: utf-8 -*-
"""Implementation of classes to handle calibration files."""

import numpy as _np
from scipy import interpolate as _interpolate
from . import utils as _utils


class CalibrationData(object):
    """Hall probe calibration data."""

    def __init__(self, filename=None):
        """Initialize variables.

        Args:
            filename (str): calibration file path.
        """
        if filename is not None:
            self.read_file(filename)
        else:
            self._data_type = ''
            self._field_unit = ''
            self._voltage_unit = ''
            self._probex_dx = None
            self._probex_dy = None
            self._probex_dz = None
            self._probez_dx = None
            self._probez_dy = None
            self._probez_dz = None
            self._angle_xy = None
            self._angle_yz = None
            self._angle_xz = None
            self._probex_conv_func = None
            self._probey_conv_func = None
            self._probez_conv_func = None
            self._probex_data = []
            self._probey_data = []
            self._probez_data = []

    def __eq__(self, other):
        """Equality method."""
        if isinstance(other, self.__class__):
            if len(self.__dict__) != len(other.__dict__):
                return False

            for key in self.__dict__:
                if key not in other.__dict__:
                    return False

                self_value = self.__dict__[key]
                other_value = other.__dict__[key]

                if callable(self_value):
                    pass
                elif (isinstance(self_value, _np.ndarray) and
                      isinstance(other_value, _np.ndarray)):
                    if not self_value.tolist() == other_value.tolist():
                        return False
                elif (not isinstance(self_value, _np.ndarray) and
                      not isinstance(other_value, _np.ndarray)):
                    if not self_value == other_value:
                        return False
                else:
                    return False

            return True

        else:
            return False

    def __ne__(self, other):
        """Non-equality method."""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    @property
    def field_unit(self):
        """Magnetic field unit."""
        return self._field_unit

    @property
    def voltage_unit(self):
        """Voltage unit."""
        return self._voltage_unit

    @property
    def probex_dx(self):
        """Probe X - displacement x[mm]."""
        return self._probex_dx

    @property
    def probex_dy(self):
        """Probe X - displacement y[mm]."""
        return self._probex_dy

    @property
    def probex_dz(self):
        """Probe X - displacement z[mm]."""
        return self._probex_dz

    @property
    def probez_dx(self):
        """Probe Z - displacement x[mm]."""
        return self._probez_dx

    @property
    def probez_dy(self):
        """Probe Z - displacement y[mm]."""
        return self._probez_dy

    @property
    def probez_dz(self):
        """Probe Z - displacement z[mm]."""
        return self._probez_dz

    @property
    def angle_xy(self):
        """Angle XY [rad]."""
        return self._angle_xy

    @property
    def angle_yz(self):
        """Angle YZ [rad]."""
        return self._angle_yz

    @property
    def angle_xz(self):
        """Angle XZ [rad]."""
        return self._angle_xz

    def read_file(self, filename):
        """Read calibration parameters from file.

        Args:
            filename (str): calibration file path.
        """
        flines = _utils.read_file(filename)
        self._data_type = _utils.find_value(flines, 'data_type')
        self._field_unit = _utils.find_value(flines, 'field_unit')
        self._voltage_unit = _utils.find_value(flines, 'voltage_unit')
        self._probex_dx = _utils.find_value(flines, 'probex_dx', vtype=float)
        self._probex_dy = _utils.find_value(flines, 'probex_dy', vtype=float)
        self._probex_dz = _utils.find_value(flines, 'probex_dz', vtype=float)
        self._probez_dx = _utils.find_value(flines, 'probez_dx', vtype=float)
        self._probez_dy = _utils.find_value(flines, 'probez_dy', vtype=float)
        self._probez_dz = _utils.find_value(flines, 'probez_dz', vtype=float)
        self._angle_xy = _utils.find_value(flines, 'angle_xy', vtype=float)
        self._angle_yz = _utils.find_value(flines, 'angle_yz', vtype=float)
        self._angle_xz = _utils.find_value(flines, 'angle_xz', vtype=float)
        self._probex_data = []
        self._probey_data = []
        self._probez_data = []

        idx = _utils.find_index(flines, '----------')
        for line in flines[idx+1:]:
            probe = line.split()[0].lower()
            if probe == 'x':
                self._probex_data.append([float(v) for v in line.split()[1:]])
            elif probe == 'y':
                self._probey_data.append([float(v) for v in line.split()[1:]])
            elif probe == 'z':
                self._probez_data.append([float(v) for v in line.split()[1:]])
            else:
                raise ValueError('Invalid probe value.')

        self._set_convertion_functions()

    def _set_convertion_functions(self):
        if len(self._probex_data) != 0 and self._data_type == 'interpolation':
            self._probex_conv_func = lambda v: _interpolation_convertion(
                self._probex_data, v)
        elif len(self._probex_data) != 0 and self._data_type == 'polinomial':
            self._probex_conv_func = lambda v: _polinomial_convertion(
                self._probex_data, v)
        else:
            self._probex_conv_func = None

        if len(self._probey_data) != 0 and self._data_type == 'interpolation':
            self._probey_conv_func = lambda v: _interpolation_convertion(
                self._probey_data, v)
        elif len(self._probey_data) != 0 and self._data_type == 'polinomial':
            self._probey_conv_func = lambda v: _polinomial_convertion(
                self._probey_data, v)
        else:
            self._probey_conv_func = None

        if len(self._probez_data) != 0 and self._data_type == 'interpolation':
            self._probez_conv_func = lambda v: _interpolation_convertion(
                self._probez_data, v)
        elif len(self._probez_data) != 0 and self._data_type == 'polinomial':
            self._probez_conv_func = lambda v: _polinomial_convertion(
                self._probez_data, v)
        else:
            self._probez_conv_func = None

    def save_file(self, filename):
        """Save calibration data to file.

        Args:
            filename (str): calibration file path.

        Raises:
            HallBenchFileError: if the calibration data was not saved.
        """
        timestamp = _utils.get_timestamp()

        f = open(filename, mode='w')
        f.write('data_type:     \t{0:1s}\n'.format(self._data_type))
        f.write('timestamp:     \t{0:1s}\n'.format(timestamp))
        f.write('field_unit:    \t{0:1s}\n'.format(self._field_unit))
        f.write('voltage_unit:  \t{0:1s}\n'.format(self._voltage_unit))
        f.write('probex_dx[mm]: \t{0:1s}\n'.format(str(self._probex_dx)))
        f.write('probex_dy[mm]: \t{0:1s}\n'.format(str(self._probex_dy)))
        f.write('probex_dz[mm]: \t{0:1s}\n'.format(str(self._probex_dz)))
        f.write('probez_dx[mm]: \t{0:1s}\n'.format(str(self._probez_dx)))
        f.write('probez_dy[mm]: \t{0:1s}\n'.format(str(self._probez_dy)))
        f.write('probez_dz[mm]: \t{0:1s}\n'.format(str(self._probez_dz)))
        f.write('angle_xy[rad]: \t{0:1s}\n'.format(str(self._angle_xy)))
        f.write('angle_yz[rad]: \t{0:1s}\n'.format(str(self._angle_yz)))
        f.write('angle_xz[rad]: \t{0:1s}\n'.format(str(self._angle_xz)))
        f.write('\n')
        f.write('---------------------------------------------------' +
                '---------------------------------------------------\n')

        for d in self._probex_data:
            f.write('x\t')
            for value in d:
                f.write('{0:+0.10e}\t'.format(value))
            f.write('\n')

        for d in self._probey_data:
            f.write('y\t')
            for value in d:
                f.write('{0:+0.10e}\t'.format(value))
            f.write('\n')

        for d in self._probez_data:
            f.write('z\t')
            for value in d:
                f.write('{0:+0.10e}\t'.format(value))
            f.write('\n')

        f.close()

    def clear(self):
        """Clear calibration data."""
        self._data_type = ''
        self._field_unit = ''
        self._voltage_unit = ''
        self._probex_dx = None
        self._probex_dy = None
        self._probex_dz = None
        self._probez_dx = None
        self._probez_dy = None
        self._probez_dz = None
        self._angle_xy = None
        self._angle_yz = None
        self._angle_xz = None
        self._probex_conv_func = None
        self._probey_conv_func = None
        self._probez_conv_func = None
        self._probex_data = []
        self._probey_data = []
        self._probez_data = []

    def convert_probe_x(self, voltage_array):
        """Convert voltage values to magnetic field values for probe x.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        if self._probex_conv_func is not None:
            return self._probex_conv_func(voltage_array)
        else:
            return _np.ones(len(voltage_array))*_np.nan

    def convert_probe_y(self, voltage_array):
        """Convert voltage values to magnetic field values for probe y.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        if self._probey_conv_func is not None:
            return self._probey_conv_func(voltage_array)
        else:
            return _np.ones(len(voltage_array))*_np.nan

    def convert_probe_z(self, voltage_array):
        """Convert voltage values to magnetic field values for probe z.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        if self._probez_conv_func is not None:
            return self._probez_conv_func(voltage_array)
        else:
            return _np.ones(len(voltage_array))*_np.nan


def _polinomial_convertion(data, voltage_array):
    field_array = _np.ones(len(voltage_array))*_np.nan
    for i in range(len(voltage_array)):
        voltage = voltage_array[i]
        for d in data:
            vmin = d[0]
            vmax = d[1]
            coeffs = d[2:]
            if voltage > vmin and voltage <= vmax:
                field_array[i] = sum(
                    coeffs[j]*(voltage**j) for j in range(len(coeffs)))
    return field_array


def _interpolation_convertion(data, voltage_array):
    d = _np.array(data)
    interp_func = _interpolate.splrep(d[:, 0], d[:, 1], k=1)
    field_array = _interpolate.splev(voltage_array, interp_func)
    return field_array


def _old_hall_probe_calibration_curve(voltage_array):
    field_array = _np.zeros(len(voltage_array))

    for i in range(len(voltage_array)):
        voltage = voltage_array[i]

        if (voltage > -10) and (voltage < 10):
            field = float(voltage)*0.2
        elif voltage <= -10:
            field = (
                -1.8216 +
                (-0.70592*voltage) +
                (-0.047964*voltage**2) +
                (-0.0015304*voltage**3))*(-1)
        else:
            field = (
                2.3614 +
                (-0.82643*voltage) +
                (0.056814*voltage**2) +
                (-0.0017429*voltage**3))*(-1)

        field_array[i] = field

    return field_array
