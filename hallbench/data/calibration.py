# -*- coding: utf-8 -*-

"""Implementation of classes to handle calibration files."""

import numpy as _np
from scipy import interpolate as _interpolate
from . import utils as _utils


class CalibrationCurve(object):
    """Voltage to magnetic field conversion data."""

    def __init__(self, filename=None):
        """Initialize variables.

        Args:
            filename (str): calibration curve file path.
        """
        self._function_type = None
        self._function = None
        self._filename = None
        self._data = []
        if filename is not None:
            self.read_file(filename)

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
    def function_type(self):
        """Function type."""
        return self._function_type

    @function_type.setter
    def function_type(self, value):
        if isinstance(value, str):
            if value in ('interpolation', 'polynomial'):
                self._function_type = value
            else:
                raise ValueError('Invalid value for function_type.')
        else:
            raise TypeError('function_type must be a string.')

    @property
    def data(self):
        """Calibration curve data."""
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, _np.ndarray):
            value = value.tolist()

        if isinstance(value, (list, tuple)):
            if len(value) == 1 and not isinstance(value[0], (list, tuple)):
                self._data = [value]
                self._set_conversion_function()
            elif all(isinstance(item, list) for item in value):
                self._data = value
                self._set_conversion_function()
            else:
                raise ValueError('Invalid value for calibration data.')
        else:
            raise TypeError('calibration data must be a list.')

    @property
    def filename(self):
        """Name of the calibration file."""
        return self._filename

    def _set_conversion_function(self):
        if (len(self.data) != 0 and
           self.function_type == 'interpolation'):
            self._function = lambda v: _interpolation_conversion(
                self.data, v)
        elif len(self.data) != 0 and self.function_type == 'polynomial':
            self._function = lambda v: _polynomial_conversion(
                self.data, v)
        else:
            self._function = None

    def clear(self):
        """Clear calibration curve data."""
        self._function_type = None
        self._function = None
        self._filename = None
        self._data = []

    def convert_voltage(self, voltage_array):
        """Convert voltage values to magnetic field values.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        if self._function is not None:
            return self._function(_np.array(voltage_array))
        else:
            return _np.ones(len(voltage_array))*_np.nan

    def read_file(self, filename):
        """Read sensor calibration parameters from file.

        Args:
            filename (str): calibration file path.
        """
        self._filename = filename

        flines = _utils.read_file(filename)
        self.function_type = _utils.find_value(flines, 'function_type')

        _data = []
        idx = _utils.find_index(flines, '----------')
        for line in flines[idx+1:]:
            _data.append([float(v) for v in line.split()])
        self._data = _data

    def save_file(self, filename):
        """Save sensor calibration data to file.

        Args:
            filename (str): calibration file path.
        """
        if self.function_type is None:
            raise ValueError('Invalid calibration data.')

        self._filename = filename
        timestamp = _utils.get_timestamp()

        f = open(filename, mode='w')
        f.write('timestamp:                     \t{0:1s}\n'.format(
            timestamp))
        f.write('function_type:                 \t{0:1s}\n'.format(
            self.function_type))
        f.write('\n')

        if self.function_type == 'interpolation':
            f.write('voltage[V]'.ljust(15)+'\t')
            f.write('field[T]'.ljust(15)+'\n')
        elif self.function_type == 'polynomial':
            f.write('v_min[V]'.ljust(15)+'\t')
            f.write('v_max[V]'.ljust(15)+'\t')
            f.write('polynomial_coefficients\n')

        f.write('---------------------------------------------------' +
                '---------------------------------------------------\n')

        for d in self.data:
            for value in d:
                f.write('{0:+14.7e}\t'.format(value))
            f.write('\n')
        f.close()


class ProbeCalibration(object):
    """Hall probe calibration data."""

    def __init__(self, filename=None):
        """Initialize variables.

        Args:
            filename (str): calibration file path.
        """
        self._sensorx = CalibrationCurve()
        self._sensory = CalibrationCurve()
        self._sensorz = CalibrationCurve()
        self._function_type = None
        self._probe_axis = None
        self._distance_xy = None
        self._distance_zy = None
        self._angle_xy = None
        self._angle_yz = None
        self._angle_xz = None
        self._filename = None
        if filename is not None:
            self.read_file(filename)

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
    def sensorx(self):
        """Sensor X calibration data."""
        return self._sensorx

    @sensorx.setter
    def sensorx(self, value):
        if isinstance(value, CalibrationCurve):
            self._sensorx = value
        else:
            raise TypeError('sensorx must be a CalibrationCurve object.')

    @property
    def sensory(self):
        """Sensor Y calibration data."""
        return self._sensory

    @sensory.setter
    def sensory(self, value):
        if isinstance(value, CalibrationCurve):
            self._sensory = value
        else:
            raise TypeError('sensory must be a CalibrationCurve object.')

    @property
    def sensorz(self):
        """Sensor Z calibration data."""
        return self._sensorz

    @sensorz.setter
    def sensorz(self, value):
        if isinstance(value, CalibrationCurve):
            self._sensorz = value
        else:
            raise TypeError('sensorz must be a CalibrationCurve object.')

    @property
    def function_type(self):
        """Function type."""
        return self._function_type

    @function_type.setter
    def function_type(self, value):
        if isinstance(value, str):
            if value in ('interpolation', 'polynomial'):
                self._function_type = value
            else:
                raise ValueError('Invalid value for function_type.')
        else:
            raise TypeError('function_type must be a string.')

    @property
    def distance_xy(self):
        """Distance between Sensor X and Sensor Y (Reference Sensor)."""
        return self._distance_xy

    @distance_xy.setter
    def distance_xy(self, value):
        if value >= 0:
            self._distance_xy = value
        else:
            raise ValueError('The sensor distance must be a positive number.')

    @property
    def distance_zy(self):
        """Distance between Sensor Z and Sensor Y (Reference Sensor)."""
        return self._distance_zy

    @distance_zy.setter
    def distance_zy(self, value):
        if value >= 0:
            self._distance_zy = value
        else:
            raise ValueError('The sensor distance must be a positive number.')

    @property
    def angle_xy(self):
        """Angle between Sensor X and Sensor Y (Reference Sensor)."""
        return self._angle_xy

    @angle_xy.setter
    def angle_xy(self, value):
        if value >= -360.0 and value <= 360.0:
            self._angle_xy = value
        else:
            raise ValueError('Invalid value for angle_xy.')

    @property
    def angle_yz(self):
        """Angle between Sensor Y and Sensor Z (Reference Sensor)."""
        return self._angle_yz

    @angle_yz.setter
    def angle_yz(self, value):
        if value >= -360.0 and value <= 360.0:
            self._angle_yz = value
        else:
            raise ValueError('Invalid value for angle_yz.')

    @property
    def angle_xz(self):
        """Angle between Sensor X and Sensor Z (Reference Sensor)."""
        return self._angle_xz

    @angle_xz.setter
    def angle_xz(self, value):
        if value >= -360.0 and value <= 360.0:
            self._angle_xz = value
        else:
            raise ValueError('Invalid value for angle_xz.')

    @property
    def probe_axis(self):
        """Probe axis."""
        return self._probe_axis

    @probe_axis.setter
    def probe_axis(self, value):
        if value in [1, 3]:
            self._probe_axis = value
        else:
            raise ValueError('Invalid value for probe axis.')

    @property
    def filename(self):
        """Name of the probe calibration file."""
        return self._filename

    def valid_data(self):
        """Check if parameters are valid."""
        return True

    def clear(self):
        """Clear calibration data."""
        self._sensorx = CalibrationCurve()
        self._sensory = CalibrationCurve()
        self._sensorz = CalibrationCurve()
        self._function_type = None
        self._probe_axis = None
        self._distance_xy = None
        self._distance_zy = None
        self._angle_xy = None
        self._angle_yz = None
        self._angle_xz = None
        self._filename = None

    def corrected_position(self, axis, position_array, sensor):
        """Return the corrected position list for the given sensor."""
        if axis == self.probe_axis:
            if sensor == 'x':
                corr_position_array = position_array - self._distance_xy
            elif sensor == 'y':
                corr_position_array = position_array
            elif sensor == 'z':
                corr_position_array = position_array + self._distance_zy
            else:
                raise ValueError('Invalid value for sensor.')
        else:
            corr_position_array = position_array
        return corr_position_array

    def field_in_bench_coordinate_system(self, fieldx, fieldy, fieldz):
        """Return the field components transform to the bench coordinate system.

        Returns:
            [field3 (+X Axis), field2 (+Y Axis), field1 (+Z Axis)]
        """
        if self._probe_axis == 1:
            field3 = fieldx
            field2 = fieldy
            field1 = fieldz
        elif self._probe_axis == 3:
            field3 = fieldz
            field2 = fieldy
            if fieldx is not None:
                field1 = -fieldx
            else:
                field1 = None
        else:
            field3, field2, field1 = None, None, None

        return field3, field2, field1

    def read_file(self, filename):
        """Read calibration parameters from file.

        Args:
            filename (str): calibration file path.
        """
        self._filename = filename

        flines = _utils.read_file(filename)
        self.function_type = _utils.find_value(flines, 'function_type')
        self.probe_axis = _utils.find_value(
            flines, 'probe_axis', vtype=int)

        sensorx_data = []
        sensory_data = []
        sensorz_data = []

        idx = _utils.find_index(flines, '----------')
        for line in flines[idx+1:]:
            sensor = line.split()[0].lower()
            if sensor == 'x':
                sensorx_data.append([float(v) for v in line.split()[1:]])
            elif sensor == 'y':
                sensory_data.append([float(v) for v in line.split()[1:]])
            elif sensor == 'z':
                sensorz_data.append([float(v) for v in line.split()[1:]])
            else:
                raise ValueError('Invalid sensor value.')

        if (len(sensorx_data) == 0 and
           len(sensory_data) == 0 and len(sensorz_data) == 0):
            raise ValueError('Invalid calibration data.')

        if len(sensorx_data) != 0:
            self.sensorx.function_type = self.function_type
            self.sensorx.data = sensorx_data
            self.distance_xy = _utils.find_value(
                flines, 'distance_xy', vtype=float)

        if len(sensory_data) != 0:
            self.sensory.function_type = self.function_type
            self.sensory.data = sensory_data

        if len(sensorz_data) != 0:
            self.sensorz.function_type = self.function_type
            self.sensorz.data = sensorz_data
            self.distance_zy = _utils.find_value(
                flines, 'distance_zy', vtype=float)

    def read_data_from_sensor_files(
            self, filenamex=None, filenamey=None, filenamez=None):
        """Read calibration data from sensor files."""
        self._sensorx = CalibrationCurve()
        self._sensory = CalibrationCurve()
        self._sensorz = CalibrationCurve()
        self._function_type = None

        if filenamex is not None:
            self.sensorx = CalibrationCurve(filenamex)
            self.function_type = self.sensorx.function_type

        if filenamey is not None:
            self.sensory = CalibrationCurve(filenamey)
            if self.function_type is None:
                self.function_type = self.sensory.function_type
            elif self.sensory.function_type != self.function_type:
                raise ValueError('Inconsistent calibration function types.')

        if filenamez is not None:
            self.sensorz = CalibrationCurve(filenamez)
            if self.function_type is None:
                self.function_type = self.sensorz.function_type
            elif self.sensorz.function_type != self.function_type:
                raise ValueError('Inconsistent calibration function types.')

    def save_file(self, filename):
        """Save calibration data to file.

        Args:
            filename (str): calibration file path.
        """
        if self.function_type is None:
            raise ValueError('Invalid calibration data.')

        self._filename = filename

        timestamp = _utils.get_timestamp()

        f = open(filename, mode='w')
        f.write('timestamp:                      \t{0:1s}\n'.format(
            timestamp))
        f.write('function_type:                  \t{0:1s}\n'.format(
            self.function_type))
        f.write('distance_xy[mm]:                \t{0:1s}\n'.format(
            str(self.distance_xy)))
        f.write('distance_zy[mm]:                \t{0:1s}\n'.format(
            str(self.distance_zy)))
        f.write('angle_xy[deg]:                  \t{0:1s}\n'.format(
            str(self.angle_xy)))
        f.write('angle_yz[deg]:                  \t{0:1s}\n'.format(
            str(self.angle_yz)))
        f.write('angle_xz[deg]:                  \t{0:1s}\n'.format(
            str(self.angle_xz)))
        f.write('probe_axis:                     \t{0:1d}\n'.format(
            self.probe_axis))
        f.write('\n')

        if self.function_type == 'interpolation':
            f.write('sensor'.ljust(8)+'\t')
            f.write('voltage[V]'.ljust(15)+'\t')
            f.write('field[T]'.ljust(15)+'\n')
        elif self.function_type == 'polynomial':
            f.write('sensor'.ljust(8)+'\t')
            f.write('v_min[V]'.ljust(15)+'\t')
            f.write('v_max[V]'.ljust(15)+'\t')
            f.write('polynomial_coefficients\n')

        f.write('---------------------------------------------------' +
                '---------------------------------------------------\n')

        for d in self.sensorx.data:
            f.write('x'.ljust(8)+'\t')
            for value in d:
                f.write('{0:+14.7e}\t'.format(value))
            f.write('\n')

        for d in self.sensory.data:
            f.write('y'.ljust(8)+'\t')
            for value in d:
                f.write('{0:+14.7e}\t'.format(value))
            f.write('\n')

        for d in self.sensorz.data:
            f.write('z'.ljust(8)+'\t')
            for value in d:
                f.write('{0:+14.7e}\t'.format(value))
            f.write('\n')

        f.close()


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


def _updated_hall_sensor_calibration_curve(voltage_array):
    field_array = _np.zeros(len(voltage_array))

    for i in range(len(voltage_array)):
        voltage = voltage_array[i]

        field = (
            (-0.19699*voltage) +
            (1.2825e-005*voltage**2) +
            (1.7478e-005*voltage**3) +
            (-2.4556e-008*voltage**4) +
            (-2.6877e-008*voltage**5) +
            (2.9855e-011*voltage**6) +
            (-1.2946e-009*voltage**7))

        field_array[i] = field

    return field_array


def _old_hall_sensor_calibration_curve(voltage_array):
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
