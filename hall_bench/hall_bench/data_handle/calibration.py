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
            self._probei_data = []
            self._probej_data = []
            self._probek_data = []
            self._probei_function = None
            self._probej_function = None
            self._probek_function = None
            self._stem_shape = None
            self._distance_probei = None
            self._distance_probek = None

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
        if isinstance(value, str):
            if value in ('interpolation', 'polynomial'):
                self._data_type = value
            else:
                raise ValueError('Invalid value for data_type.')
        else:
            raise TypeError('data_type must be a string.')

    @property
    def distance_probei(self):
        """Distance between Probe I and Probe J (Reference Probe)."""
        return self._distance_probei

    @distance_probei.setter
    def distance_probei(self, value):
        if value >= 0:
            self._distance_probei = value
        else:
            raise ValueError('The probe distance must be a positive number.')

    @property
    def distance_probek(self):
        """Distance between Probe K and Probe J (Reference Probe)."""
        return self._distance_probek

    @distance_probek.setter
    def distance_probek(self, value):
        if value >= 0:
            self._distance_probek = value
        else:
            raise ValueError('The probe distance must be a positive number.')

    @property
    def stem_shape(self):
        """Stem Shape."""
        return self._stem_shape

    @stem_shape.setter
    def stem_shape(self, value):
        if isinstance(value, str):
            if value.capitalize() in ('L-shape', 'Straight'):
                self._stem_shape = value
            else:
                raise ValueError('Invalid value for stem_shape.')
        else:
            raise TypeError('stem_shape must be a string.')

    @property
    def probei_data(self):
        """Probe I calibration data."""
        return self._probei_data

    @probei_data.setter
    def probei_data(self, value):
        if isinstance(value, _np.ndarray):
            value = value.tolist()

        if isinstance(value, (list, tuple)):
            if len(value) == 1 and not isinstance(value[0], (list, tuple)):
                self._probei_data = [value]
                self._set_conversion_function_i()
            elif all(isinstance(item, list) for item in value):
                self._probei_data = value
                self._set_conversion_function_i()
            else:
                raise ValueError('Invalid value for probei_data.')
        else:
            raise TypeError('probei_data must be a list.')

    @property
    def probej_data(self):
        """Probe J calibration data."""
        return self._probej_data

    @probej_data.setter
    def probej_data(self, value):
        if isinstance(value, _np.ndarray):
            value = value.tolist()

        if isinstance(value, (list, tuple)):
            if len(value) == 1 and not isinstance(value[0], (list, tuple)):
                self._probej_data = [value]
                self._set_conversion_function_j()
            elif all(isinstance(item, list) for item in value):
                self._probej_data = value
                self._set_conversion_function_j()
            else:
                raise ValueError('Invalid value for probej_data.')
        else:
            raise TypeError('probej_data must be a list.')

    @property
    def probek_data(self):
        """Probe K calibration data."""
        return self._probek_data

    @probek_data.setter
    def probek_data(self, value):
        if isinstance(value, _np.ndarray):
            value = value.tolist()

        if isinstance(value, (list, tuple)):
            if len(value) == 1 and not isinstance(value[0], (list, tuple)):
                self._probek_data = [value]
                self._set_conversion_function_k()
            elif all(isinstance(item, list) for item in value):
                self._probek_data = value
                self._set_conversion_function_k()
            else:
                raise ValueError('Invalid value for probek_data.')
        else:
            raise TypeError('probek_data must be a list.')

    def _set_conversion_function_i(self):
        if len(self.probei_data) != 0 and self.data_type == 'interpolation':
            self._probei_function = lambda v: _interpolation_conversion(
                self.probei_data, v)
        elif len(self.probei_data) != 0 and self.data_type == 'polynomial':
            self._probei_function = lambda v: _polynomial_conversion(
                self.probei_data, v)
        else:
            self._probei_function = None

    def _set_conversion_function_j(self):
        if len(self.probej_data) != 0 and self.data_type == 'interpolation':
            self._probej_function = lambda v: _interpolation_conversion(
                self.probej_data, v)
        elif len(self.probej_data) != 0 and self.data_type == 'polynomial':
            self._probej_function = lambda v: _polynomial_conversion(
                self.probej_data, v)
        else:
            self._probej_function = None

    def _set_conversion_function_k(self):
        if len(self.probek_data) != 0 and self.data_type == 'interpolation':
            self._probek_function = lambda v: _interpolation_conversion(
                self.probek_data, v)
        elif len(self.probek_data) != 0 and self.data_type == 'polynomial':
            self._probek_function = lambda v: _polynomial_conversion(
                self.probek_data, v)
        else:
            self._probek_function = None

    def read_file(self, filename):
        """Read calibration parameters from file.

        Args:
            filename (str): calibration file path.
        """
        flines = _utils.read_file(filename)
        self.data_type = _utils.find_value(flines, 'data_type')
        self.field_unit = _utils.find_value(flines, 'field_unit')
        self.voltage_unit = _utils.find_value(flines, 'voltage_unit')
        self.distance_probei = _utils.find_value(
            flines, 'distance_probei', vtype=float)
        self.distance_probek = _utils.find_value(
            flines, 'distance_probek', vtype=float)
        self.stem_shape = _utils.find_value(flines, 'stem_shape')

        probei_data = []
        probej_data = []
        probek_data = []

        idx = _utils.find_index(flines, '----------')
        for line in flines[idx+1:]:
            probe = line.split()[0].lower()
            if probe == 'i':
                probei_data.append([float(v) for v in line.split()[1:]])
            elif probe == 'j':
                probej_data.append([float(v) for v in line.split()[1:]])
            elif probe == 'k':
                probek_data.append([float(v) for v in line.split()[1:]])
            else:
                raise ValueError('Invalid probe value.')

        self.probei_data = probei_data
        self.probej_data = probej_data
        self.probek_data = probek_data

    def save_file(self, filename):
        """Save calibration data to file.

        Args:
            filename (str): calibration file path.
        """
        if self.data_type is None:
            raise ValueError('Invalid calibration data.')

        timestamp = _utils.get_timestamp()

        f = open(filename, mode='w')
        f.write('data_type:                     \t{0:1s}\n'.format(
            self.data_type))
        f.write('timestamp:                     \t{0:1s}\n'.format(
            timestamp))
        f.write('field_unit:                    \t{0:1s}\n'.format(
            self.field_unit))
        f.write('voltage_unit:                  \t{0:1s}\n'.format(
            self.voltage_unit))
        f.write('distance_probei[mm]:           \t{0:1s}\n'.format(
            str(self.distance_probei)))
        f.write('distance_probek[mm]:           \t{0:1s}\n'.format(
            str(self.distance_probek)))
        f.write('stem_shape:                    \t{0:1s}\n'.format(
            self.stem_shape))
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

        for d in self.probei_data:
            f.write('i'.ljust(8)+'\t')
            for value in d:
                f.write('{0:+14.7e}\t'.format(value))
            f.write('\n')

        for d in self.probej_data:
            f.write('j'.ljust(8)+'\t')
            for value in d:
                f.write('{0:+14.7e}\t'.format(value))
            f.write('\n')

        for d in self.probek_data:
            f.write('k'.ljust(8)+'\t')
            for value in d:
                f.write('{0:+14.7e}\t'.format(value))
            f.write('\n')

        f.close()

    def clear(self):
        """Clear calibration data."""
        self._field_unit = ''
        self._voltage_unit = ''
        self._data_type = None
        self._probei_data = []
        self._probej_data = []
        self._probek_data = []
        self._probei_function = None
        self._probej_function = None
        self._probek_function = None
        self._stem_shape = None
        self._distance_probei = None
        self._distance_probek = None

    def convert_voltage_probei(self, voltage_array):
        """Convert voltage values to magnetic field values for Probe I.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        if self._probei_function is not None:
            return self._probei_function(voltage_array)
        else:
            return _np.ones(len(voltage_array))*_np.nan

    def convert_voltage_probej(self, voltage_array):
        """Convert voltage values to magnetic field values for Probe J.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        if self._probej_function is not None:
            return self._probej_function(voltage_array)
        else:
            return _np.ones(len(voltage_array))*_np.nan

    def convert_voltage_probek(self, voltage_array):
        """Convert voltage values to magnetic field values for Probe K.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        if self._probek_function is not None:
            return self._probek_function(voltage_array)
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
