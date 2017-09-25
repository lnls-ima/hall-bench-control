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
            self._field_unit = ''
            self._voltage_unit = ''
            self._data_type = None
            self._probex_data = []
            self._probey_data = []
            self._probez_data = []
            self._probex_function = None
            self._probey_function = None
            self._probez_function = None
            self.dyx = None
            self.dyz = None
            self.width_axis = None

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

    @field_unit.setter
    def field_unit(self, value):
        if isinstance(value, str):
            self._field_unit = value
        else:
            raise TypeError('field_unit must be a string.')

    @property
    def voltage_unit(self):
        """Voltage unit."""
        return self._voltage_unit

    @voltage_unit.setter
    def voltage_unit(self, value):
        if isinstance(value, str):
            self._voltage_unit = value
        else:
            raise TypeError('voltage_unit must be a string.')

    @property
    def data_type(self):
        """Data type."""
        return self._data_type

    @data_type.setter
    def data_type(self, value):
        if value in ('interpolation', 'polynomial'):
            self._data_type = value
        else:
            raise ValueError('Invalid value for data_type.')

    @property
    def probex_data(self):
        """Probe X calibration data."""
        return self._probex_data

    @probex_data.setter
    def probex_data(self, value):
        if isinstance(value, _np.ndarray):
            value = value.tolist()

        if isinstance(value, (list, tuple)):
            if len(value) == 1 and not isinstance(value[0], (list, tuple)):
                self._probex_data = [value]
                self._set_conversion_function_x()
            elif all(isinstance(item, list) for item in value):
                self._probex_data = value
                self._set_conversion_function_x()
            else:
                raise ValueError('Invalid value for probex_data.')
        else:
            raise TypeError('probex_data must be a list.')

    @property
    def probey_data(self):
        """Probe Y calibration data."""
        return self._probey_data

    @probey_data.setter
    def probey_data(self, value):
        if isinstance(value, _np.ndarray):
            value = value.tolist()

        if isinstance(value, (list, tuple)):
            if len(value) == 1 and not isinstance(value[0], (list, tuple)):
                self._probey_data = [value]
                self._set_conversion_function_y()
            elif all(isinstance(item, list) for item in value):
                self._probey_data = value
                self._set_conversion_function_y()
            else:
                raise ValueError('Invalid value for probey_data.')
        else:
            raise TypeError('probey_data must be a list.')

    @property
    def probez_data(self):
        """Probe Z calibration data."""
        return self._probez_data

    @probez_data.setter
    def probez_data(self, value):
        if isinstance(value, _np.ndarray):
            value = value.tolist()

        if isinstance(value, (list, tuple)):
            if len(value) == 1 and not isinstance(value[0], (list, tuple)):
                self._probez_data = [value]
                self._set_conversion_function_z()
            elif all(isinstance(item, list) for item in value):
                self._probez_data = value
                self._set_conversion_function_z()
            else:
                raise ValueError('Invalid value for probez_data.')
        else:
            raise TypeError('probez_data must be a list.')

    def _set_conversion_function_x(self):
        if len(self.probex_data) != 0 and self.data_type == 'interpolation':
            self._probex_function = lambda v: _interpolation_conversion(
                self.probex_data, v)
        elif len(self.probex_data) != 0 and self.data_type == 'polynomial':
            self._probex_function = lambda v: _polynomial_conversion(
                self.probex_data, v)
        else:
            self._probex_function = None

    def _set_conversion_function_y(self):
        if len(self.probey_data) != 0 and self.data_type == 'interpolation':
            self._probey_function = lambda v: _interpolation_conversion(
                self.probey_data, v)
        elif len(self.probey_data) != 0 and self.data_type == 'polynomial':
            self._probey_function = lambda v: _polynomial_conversion(
                self.probey_data, v)
        else:
            self._probey_function = None

    def _set_conversion_function_z(self):
        if len(self.probez_data) != 0 and self.data_type == 'interpolation':
            self._probez_function = lambda v: _interpolation_conversion(
                self.probez_data, v)
        elif len(self.probez_data) != 0 and self.data_type == 'polynomial':
            self._probez_function = lambda v: _polynomial_conversion(
                self.probez_data, v)
        else:
            self._probez_function = None

    def read_file(self, filename):
        """Read calibration parameters from file.

        Args:
            filename (str): calibration file path.
        """
        flines = _utils.read_file(filename)
        self.data_type = _utils.find_value(flines, 'data_type')
        self.field_unit = _utils.find_value(flines, 'field_unit')
        self.voltage_unit = _utils.find_value(flines, 'voltage_unit')
        self.dyx = _utils.find_value(flines, 'dyx', vtype=float)
        self.dyz = _utils.find_value(flines, 'dyz', vtype=float)
        self.width_axis = _utils.find_value(flines, 'width_axis')

        probex_data = []
        probey_data = []
        probez_data = []

        idx = _utils.find_index(flines, '----------')
        for line in flines[idx+1:]:
            probe = line.split()[0].lower()
            if probe == 'x':
                probex_data.append([float(v) for v in line.split()[1:]])
            elif probe == 'y':
                probey_data.append([float(v) for v in line.split()[1:]])
            elif probe == 'z':
                probez_data.append([float(v) for v in line.split()[1:]])
            else:
                raise ValueError('Invalid probe value.')

        self.probex_data = probex_data
        self.probey_data = probey_data
        self.probez_data = probez_data

    def save_file(self, filename):
        """Save calibration data to file.

        Args:
            filename (str): calibration file path.
        """
        if self.data_type is None:
            raise ValueError('Invalid calibration data.')

        timestamp = _utils.get_timestamp()

        f = open(filename, mode='w')
        f.write('data_type:     \t{0:1s}\n'.format(self.data_type))
        f.write('timestamp:     \t{0:1s}\n'.format(timestamp))
        f.write('field_unit:    \t{0:1s}\n'.format(self.field_unit))
        f.write('voltage_unit:  \t{0:1s}\n'.format(self.voltage_unit))
        f.write('dyx[mm]:       \t{0:1s}\n'.format(str(self.dyx)))
        f.write('dyz[mm]:       \t{0:1s}\n'.format(str(self.dyz)))
        f.write('width_axis:    \t{0:1s}\n'.format(self.width_axis))
        f.write('\n')

        if self.data_type == 'interpolation':
            f.write('probe'.ljust(8)+'\t')
            f.write('voltage[{0:1s}]'.format(self.voltage_unit).ljust(15)+'\t')
            f.write('field[{0:1s}]'.format(self.field_unit).ljust(15)+'\n')
        elif self.data_type == 'polynomial':
            f.write('probe'.ljust(8)+'\t')
            f.write('v_min[{0:1s}]'.format(self.voltage_unit).ljust(15)+'\t')
            f.write('v_max[{0:1s}]'.format(self.voltage_unit).ljust(15)+'\t')
            f.write('polynomial_coefficients\n')

        f.write('---------------------------------------------------' +
                '---------------------------------------------------\n')

        for d in self.probex_data:
            f.write('x'.ljust(8)+'\t')
            for value in d:
                f.write('{0:+14.7e}\t'.format(value))
            f.write('\n')

        for d in self.probey_data:
            f.write('y'.ljust(8)+'\t')
            for value in d:
                f.write('{0:+14.7e}\t'.format(value))
            f.write('\n')

        for d in self.probez_data:
            f.write('z'.ljust(8)+'\t')
            for value in d:
                f.write('{0:+14.7e}\t'.format(value))
            f.write('\n')

        f.close()

    def clear(self):
        """Clear calibration data."""
        self._field_unit = ''
        self._voltage_unit = ''
        self._data_type = None
        self._probex_data = []
        self._probey_data = []
        self._probez_data = []
        self._probex_function = None
        self._probey_function = None
        self._probez_function = None
        self.dyx = None
        self.dyz = None
        self.width_axis = None

    def convert_voltage_probex(self, voltage_array):
        """Convert voltage values to magnetic field values for probe x.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        if self._probex_function is not None:
            return self._probex_function(voltage_array)
        else:
            return _np.ones(len(voltage_array))*_np.nan

    def convert_voltage_probey(self, voltage_array):
        """Convert voltage values to magnetic field values for probe y.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        if self._probey_function is not None:
            return self._probey_function(voltage_array)
        else:
            return _np.ones(len(voltage_array))*_np.nan

    def convert_voltage_probez(self, voltage_array):
        """Convert voltage values to magnetic field values for probe z.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        if self._probez_function is not None:
            return self._probez_function(voltage_array)
        else:
            return _np.ones(len(voltage_array))*_np.nan


def _polynomial_conversion(data, voltage_array):
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


def _interpolation_conversion(data, voltage_array):
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
