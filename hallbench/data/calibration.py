# -*- coding: utf-8 -*-

"""Implementation of classes to handle calibration files."""

import json as _json
import numpy as _np
import collections as _collections
from scipy import interpolate as _interpolate

from . import utils as _utils
from . import database as _database


class CalibrationError(Exception):
    """Calibration exception."""

    def __init__(self, message, *args):
        """Initialization method."""
        self.message = message


class CalibrationCurve(object):
    """Voltage to magnetic field conversion data."""

    def __init__(self, filename=None):
        """Initialize variables.

        Args:
            filename (str): calibration curve file path.
        """
        self._function_type = None
        self._function = None
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

        timestamp = _utils.get_timestamp()

        with open(filename, mode='w') as f:
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


class ProbeCalibration(object):
    """Hall probe calibration data."""

    _db_table = 'probe_calibrations'
    _db_dict = _collections.OrderedDict([
        ('id', [None, 'INTEGER NOT NULL']),
        ('date', [None, 'TEXT NOT NULL']),
        ('hour', [None, 'TEXT NOT NULL']),
        ('probe_name', ['probe_name', 'TEXT NOT NULL UNIQUE']),
        ('calibration_magnet', ['calibration_magnet', 'TEXT NOT NULL']),
        ('function_type', ['function_type', 'TEXT NOT NULL']),
        ('distance_xy', ['distance_xy', 'REAL NOT NULL']),
        ('distance_zy', ['distance_zy', 'REAL NOT NULL']),
        ('angle_xy', ['angle_xy', 'REAL NOT NULL']),
        ('angle_yz', ['angle_yz', 'REAL NOT NULL']),
        ('angle_xz', ['angle_xz', 'REAL NOT NULL']),
        ('probe_axis', ['probe_axis', 'INTEGER NOT NULL']),
        ('sensorx', ['sensorx', 'TEXT NOT NULL']),
        ('sensory', ['sensory', 'TEXT NOT NULL']),
        ('sensorz', ['sensorz', 'TEXT NOT NULL']),
    ])

    def __init__(self, filename=None, database=None, idn=None):
        """Initialize variables.

        Args:
            filename (str): calibration file path.
        """
        self.probe_name = None
        self.calibration_magnet = None
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

        if filename is not None and idn is not None:
            raise ValueError('Invalid arguments for ProbeCalibration.')

        if idn is not None and database is not None:
            self.read_from_database(database, idn)

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

    @classmethod
    def database_table_name(cls):
        """Return the database table name."""
        return cls._db_table

    @classmethod
    def create_database_table(cls, database):
        """Create database table."""
        variables = []
        for key in cls._db_dict.keys():
            variables.append((key, cls._db_dict[key][1]))
        success = _database.create_table(database, cls._db_table, variables)
        return success

    @property
    def sensorx(self):
        """Sensor X CalibrationCurve object."""
        return self._sensorx

    @sensorx.setter
    def sensorx(self, value):
        if isinstance(value, CalibrationCurve):
            self._sensorx = value
        else:
            raise TypeError('sensorx must be a CalibrationCurve object.')

    @property
    def sensory(self):
        """Sensor Y CalibrationCurve object."""
        return self._sensory

    @sensory.setter
    def sensory(self, value):
        if isinstance(value, CalibrationCurve):
            self._sensory = value
        else:
            raise TypeError('sensory must be a CalibrationCurve object.')

    @property
    def sensorz(self):
        """Sensor Z CalibrationCurve object."""
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

    def clear(self):
        """Clear calibration data."""
        self.probe_name = None
        self.calibration_magnet = None
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
        """Return field components transform to the bench coordinate system.

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
        flines = _utils.read_file(filename)
        self.probe_name = _utils.find_value(flines, 'probe_name')
        self.calibration_magnet = _utils.find_value(
            flines, 'calibration_magnet')
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
            self.angle_xy = _utils.find_value(flines, 'angle_xy', vtype=float)

        if len(sensory_data) != 0:
            self.sensory.function_type = self.function_type
            self.sensory.data = sensory_data

        if len(sensorz_data) != 0:
            self.sensorz.function_type = self.function_type
            self.sensorz.data = sensorz_data
            self.distance_zy = _utils.find_value(
                flines, 'distance_zy', vtype=float)
            self.angle_yz = _utils.find_value(flines, 'angle_yz', vtype=float)

        if len(sensorz_data) != 0 and len(sensorx_data) != 0:
            self.angle_xz = _utils.find_value(flines, 'angle_xz', vtype=float)

    def save_file(self, filename):
        """Save calibration data to file.

        Args:
            filename (str): calibration file path.
        """
        if self.function_type is None:
            raise ValueError('Invalid calibration data.')

        timestamp = _utils.get_timestamp()

        with open(filename, mode='w') as f:
            f.write('timestamp:         \t{0:1s}\n'.format(timestamp))
            f.write('probe_name:        \t{0:1s}\n'.format(self.probe_name))
            f.write('calibration_magnet:\t{0:1s}\n'.format(
                self.calibration_magnet))
            f.write('function_type:     \t{0:1s}\n'.format(self.function_type))
            f.write('distance_xy[mm]:   \t{0:1s}\n'.format(
                str(self.distance_xy)))
            f.write('distance_zy[mm]:   \t{0:1s}\n'.format(
                str(self.distance_zy)))
            f.write('angle_xy[deg]:     \t{0:1s}\n'.format(str(self.angle_xy)))
            f.write('angle_yz[deg]:     \t{0:1s}\n'.format(str(self.angle_yz)))
            f.write('angle_xz[deg]:     \t{0:1s}\n'.format(str(self.angle_xz)))
            f.write('probe_axis:        \t{0:1d}\n'.format(self.probe_axis))
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

    def read_from_database(self, database, idn):
        """Read field data from database entry."""
        db_column_names = _database.get_table_column_names(
            database, self._db_table)
        if len(db_column_names) == 0:
            raise CalibrationError(
                'Failed to read probe calibration from database.')

        db_entry = _database.read_from_database(database, self._db_table, idn)
        if db_entry is None:
            raise ValueError('Invalid database ID.')

        for key in self._db_dict.keys():
            attr_name = self._db_dict[key][0]
            if key not in db_column_names:
                raise CalibrationError(
                    'Failed to read probe calibration from database.')
            else:
                if attr_name is not None:
                    idx = db_column_names.index(key)
                    if attr_name in ['sensorx', 'sensory', 'sensorz']:
                        sensor = CalibrationCurve()
                        sensor.function_type = self.function_type
                        sensor.data = _json.loads(db_entry[idx])
                        setattr(self, attr_name, sensor)
                    else:
                        setattr(self, attr_name, db_entry[idx])

    def save_to_database(self, database):
        """Insert field data into database table."""
        db_column_names = _database.get_table_column_names(
            database, self._db_table)
        if len(db_column_names) == 0:
            raise CalibrationError(
                'Failed to save probe calibration to database.')
            return None

        timestamp = _utils.get_timestamp().split('_')
        date = timestamp[0]
        hour = timestamp[1].replace('-', ':')

        db_values = []
        for key in self._db_dict.keys():
            attr_name = self._db_dict[key][0]
            if key not in db_column_names:
                raise CalibrationError(
                    'Failed to save probe calibration to database.')
                return None
            else:
                if key == "id":
                    db_values.append(None)
                elif attr_name is None:
                    db_values.append(locals()[key])
                elif attr_name in ['sensorx', 'sensory', 'sensorz']:
                    sensor = getattr(self, attr_name)
                    db_values.append(_json.dumps(sensor.data))
                else:
                    db_values.append(getattr(self, attr_name))

        idn = _database.insert_into_database(
            database, self._db_table, db_values)
        return idn


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
