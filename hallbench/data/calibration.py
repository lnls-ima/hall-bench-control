# -*- coding: utf-8 -*-

"""Implementation of classes to handle calibration files."""

import numpy as _np
import collections as _collections
from scipy import interpolate as _interpolate

from . import utils as _utils
from . import database as _database


class CalibrationError(Exception):
    """Calibration exception."""

    def __init__(self, message, *args):
        """Initialize object."""
        self.message = message


class HallSensor(_database.DatabaseObject):
    """Voltage to magnetic field conversion data."""

    _db_table = 'hall_sensors'
    _db_dict = _collections.OrderedDict([
        ('id', [None, 'INTEGER NOT NULL']),
        ('date', [None, 'TEXT NOT NULL']),
        ('hour', [None, 'TEXT NOT NULL']),
        ('sensor_name', ['sensor_name', 'TEXT NOT NULL UNIQUE']),
        ('calibration_magnet', ['calibration_magnet', 'TEXT NOT NULL']),
        ('function_type', ['function_type', 'TEXT NOT NULL']),
        ('data', ['data', 'TEXT NOT NULL']),
    ])
    _db_json_str = ['data']

    def __init__(self, filename=None, database=None, idn=None):
        """Initialize variables.

        Args:
            filename (str): hall sensor file path.
            database (str): database file path.
            idn (int): id in database table.
        """
        self.sensor_name = None
        self.calibration_magnet = None
        self._function_type = None
        self._function = None
        self._data = []

        if filename is not None and idn is not None:
            raise ValueError('Invalid arguments for HallSensor.')

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

    def __str__(self):
        """Printable string representation of the object."""
        fmtstr = '{0:<18s} : {1}\n'
        r = ''
        for key, value in self.__dict__.items():
            if key != '_function':
                if key.startswith('_'):
                    name = key[1:]
                else:
                    name = key
                r += fmtstr.format(name, str(value))
        return r

    @property
    def function_type(self):
        """Return the function type."""
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
        """Hall sensor calibration data."""
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, _np.ndarray):
            value = value.tolist()

        if isinstance(value, (list, tuple)):
            if all(isinstance(item, list) for item in value):
                self._data = value
                self._set_conversion_function()
            elif not any(isinstance(item, list) for item in value):
                self._data = [value]
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
        """Clear hall sensor data."""
        self.sensor_name = None
        self.calibration_magnet = None
        self._function_type = None
        self._function = None
        self._data = []

    def copy(self):
        """Return a copy of the object."""
        _copy = type(self)()
        for key in self.__dict__:
            if isinstance(self.__dict__[key], _np.ndarray):
                _copy.__dict__[key] = _np.copy(self.__dict__[key])
            elif isinstance(self.__dict__[key], list):
                _copy.__dict__[key] = self.__dict__[key].copy()
            else:
                _copy.__dict__[key] = self.__dict__[key]
        return _copy

    def convert_voltage(self, voltage_array):
        """Convert voltage values to magnetic field values.

        Args:
            voltage_array (array): array with voltage values.

        Return:
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
        self.sensor_name = _utils.find_value(flines, 'sensor_name')
        self.calibration_magnet = _utils.find_value(
            flines, 'calibration_magnet')
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
            f.write('timestamp:           \t{0:1s}\n'.format(
                timestamp))
            f.write('sensor_name:         \t{0:1s}\n'.format(
                self.sensor_name))
            f.write('calibration_magnet:  \t{0:1s}\n'.format(
                self.calibration_magnet))
            f.write('function_type:       \t{0:1s}\n'.format(
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

    def valid_data(self):
        """Check if parameters are valid."""
        al = [getattr(self, a) for a in self.__dict__]
        if all([a is not None for a in al]):
            return True
        else:
            return False


class HallProbe(_database.DatabaseObject):
    """Hall probe data."""

    _db_table = 'hall_probes'
    _db_dict = _collections.OrderedDict([
        ('id', [None, 'INTEGER NOT NULL']),
        ('date', [None, 'TEXT NOT NULL']),
        ('hour', [None, 'TEXT NOT NULL']),
        ('probe_name', ['probe_name', 'TEXT NOT NULL UNIQUE']),
        ('sensorx_name', ['sensorx_name', 'TEXT NOT NULL']),
        ('sensory_name', ['sensory_name', 'TEXT NOT NULL']),
        ('sensorz_name', ['sensorz_name', 'TEXT NOT NULL']),
        ('probe_axis', ['probe_axis', 'INTEGER NOT NULL']),
        ('distance_xy', ['distance_xy', 'REAL NOT NULL']),
        ('distance_zy', ['distance_zy', 'REAL NOT NULL']),
        ('angle_xy', ['angle_xy', 'REAL NOT NULL']),
        ('angle_yz', ['angle_yz', 'REAL NOT NULL']),
        ('angle_xz', ['angle_xz', 'REAL NOT NULL']),
    ])

    def __init__(self, filename=None, database=None, idn=None):
        """Initialize variables.

        Args:
            filename (str): calibration file path.
            database (str): database file path.
            idn (int): id in database table.
        """
        self.probe_name = None
        self._sensorx_name = None
        self._sensory_name = None
        self._sensorz_name = None
        self._sensorx = None
        self._sensory = None
        self._sensorz = None
        self._probe_axis = None
        self._distance_xy = None
        self._distance_zy = None
        self._angle_xy = None
        self._angle_yz = None
        self._angle_xz = None

        if filename is not None and idn is not None:
            raise ValueError('Invalid arguments for HallProbe.')

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

    def __str__(self):
        """Printable string representation of the object."""
        fmtstr = '{0:<18s} : {1}\n'
        r = ''
        for key, value in self.__dict__.items():
            if key not in ['_sensorx', '_sensory', '_sensorz']:
                if key.startswith('_'):
                    name = key[1:]
                else:
                    name = key
                r += fmtstr.format(name, str(value))
        return r

    @classmethod
    def get_hall_probe_id(cls, database, probe_name):
        """Search probe name in database and return table ID."""
        idns = cls.get_database_id(database, 'probe_name', probe_name)
        if len(idns) != 0:
            idn = idns[0]
        else:
            idn = None
        return idn

    @classmethod
    def get_probe_name_from_database(cls, database, idn):
        """Return the probe name of the database record."""
        probe_name = cls.get_database_param(database, idn, 'probe_name')
        return probe_name

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
    def sensorx_name(self):
        """Hall Sensor X name."""
        return self._sensorx_name

    @sensorx_name.setter
    def sensorx_name(self, value):
        if isinstance(value, str):
            self._sensorx = None
            self._sensorx_name = value
        else:
            raise ValueError('Invalid value for sensor X name.')

    @property
    def sensory_name(self):
        """Hall Sensor Y name."""
        return self._sensory_name

    @sensory_name.setter
    def sensory_name(self, value):
        if isinstance(value, str):
            self._sensory = None
            self._sensory_name = value
        else:
            raise ValueError('Invalid value for sensor Y name.')

    @property
    def sensorz_name(self):
        """Hall Sensor Z name."""
        return self._sensorz_name

    @sensorz_name.setter
    def sensorz_name(self, value):
        if isinstance(value, str):
            self._sensorz = None
            self._sensorz_name = value
        else:
            raise ValueError('Invalid value for sensor Z name.')

    @property
    def sensorx(self):
        """Hall Sensor X."""
        return self._sensorx

    @sensorx.setter
    def sensorx(self, value):
        if isinstance(value, HallSensor):
            self._sensorx = value
            self._sensorx_name = value.sensor_name
        else:
            raise ValueError('Invalid value for sensor X.')

    @property
    def sensory(self):
        """Hall Sensor Y."""
        return self._sensory

    @sensory.setter
    def sensory(self, value):
        if isinstance(value, HallSensor):
            self._sensory = value
            self._sensory_name = value.sensor_name
        else:
            raise ValueError('Invalid value for sensor Y.')

    @property
    def sensorz(self):
        """Hall Sensor Z."""
        return self._sensorz

    @sensorz.setter
    def sensorz(self, value):
        if isinstance(value, HallSensor):
            self._sensorz = value
            self._sensorz_name = value.sensor_name
        else:
            raise ValueError('Invalid value for sensor Z.')

    def clear(self):
        """Clear calibration data."""
        self.probe_name = None
        self._sensorx_name = None
        self._sensory_name = None
        self._sensorz_name = None
        self._sensorx = None
        self._sensory = None
        self._sensorz = None
        self._probe_axis = None
        self._distance_xy = None
        self._distance_zy = None
        self._angle_xy = None
        self._angle_yz = None
        self._angle_xz = None

    def copy(self):
        """Return a copy of the object."""
        _copy = type(self)()
        for key in self.__dict__:
            if isinstance(self.__dict__[key], _np.ndarray):
                _copy.__dict__[key] = _np.copy(self.__dict__[key])
            elif isinstance(self.__dict__[key], list):
                _copy.__dict__[key] = self.__dict__[key].copy()
            elif isinstance(self.__dict__[key], HallSensor):
                _copy.__dict__[key] = self.__dict__[key].copy()
            else:
                _copy.__dict__[key] = self.__dict__[key]
        return _copy

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

        Return:
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

    def load_sensors_data(self, database):
        """Load Hall sensors data from database."""
        if self.sensorx_name is not None:
            idns = HallSensor.get_database_id(
                database, 'sensor_name', self.sensorx_name)
            if len(idns) != 0:
                idn = idns[0]
                self.sensorx = HallSensor(database=database, idn=idn)
            else:
                return False

        if self.sensory_name is not None:
            idns = HallSensor.get_database_id(
                database, 'sensor_name', self.sensory_name)
            if len(idns) != 0:
                idn = idns[0]
                self.sensory = HallSensor(database=database, idn=idn)
            else:
                return False

        if self.sensory_name is not None:
            idns = HallSensor.get_database_id(
                database, 'sensor_name', self.sensorz_name)
            if len(idns) != 0:
                idn = idns[0]
                self.sensorz = HallSensor(database=database, idn=idn)
            else:
                return False

        return True

    def read_file(self, filename):
        """Read calibration parameters from file.

        Args:
            filename (str): calibration file path.
        """
        flines = _utils.read_file(filename)
        self.probe_name = _utils.find_value(flines, 'probe_name')
        self.sensorx_name = _utils.find_value(flines, 'sensorx_name')
        self.sensory_name = _utils.find_value(flines, 'sensory_name')
        self.sensorz_name = _utils.find_value(flines, 'sensorz_name')
        self.probe_axis = _utils.find_value(flines, 'probe_axis', vtype=int)
        self.distance_xy = _utils.find_value(
            flines, 'distance_xy', vtype=float)
        self.distance_zy = _utils.find_value(
            flines, 'distance_zy', vtype=float)
        self.angle_xy = _utils.find_value(flines, 'angle_xy', vtype=float)
        self.angle_yz = _utils.find_value(flines, 'angle_yz', vtype=float)
        self.angle_xz = _utils.find_value(flines, 'angle_xz', vtype=float)

    def read_from_database(self, database, idn):
        """Read data from database entry."""
        super().read_from_database(database, idn)
        if not self.load_sensors_data(database):
            raise CalibrationError('Fail to load sensors data from database.')

    def save_file(self, filename):
        """Save calibration data to file.

        Args:
            filename (str): calibration file path.
        """
        timestamp = _utils.get_timestamp()

        with open(filename, mode='w') as f:
            f.write('timestamp:      \t{0:1s}\n'.format(timestamp))
            f.write('probe_name:     \t{0:1s}\n'.format(self.probe_name))
            f.write('sensorx_name:   \t{0:1s}\n'.format(self.sensorx_name))
            f.write('sensory_name:   \t{0:1s}\n'.format(self.sensory_name))
            f.write('sensorz_name:   \t{0:1s}\n'.format(self.sensorz_name))
            f.write('probe_axis:     \t{0:1d}\n'.format(self.probe_axis))
            f.write('distance_xy[mm]:\t{0:1s}\n'.format(str(self.distance_xy)))
            f.write('distance_zy[mm]:\t{0:1s}\n'.format(str(self.distance_zy)))
            f.write('angle_xy[deg]:  \t{0:1s}\n'.format(str(self.angle_xy)))
            f.write('angle_yz[deg]:  \t{0:1s}\n'.format(str(self.angle_yz)))
            f.write('angle_xz[deg]:  \t{0:1s}\n'.format(str(self.angle_xz)))

    def valid_data(self):
        """Check if parameters are valid."""
        sensor_names = ['_sensorx_name', '_sensory_name', '_sensorz_name']
        sensors = ['_sensorx', '_sensory', '_sensorz']
        for i in range(len(sensor_names)):
            sn = getattr(self, sensor_names[i])
            s = getattr(self, sensors[i])
            if sn is not None and s is None:
                return False
        valid_none = sensor_names + sensors
        al = [getattr(self, a) for a in self.__dict__ if a not in valid_none]
        if all([a is not None for a in al]):
            return True
        else:
            return False


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


def _updated_hall_sensor_calibration(voltage_array):
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


def _old_hall_sensor_calibration(voltage_array):
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
