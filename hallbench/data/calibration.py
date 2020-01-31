# -*- coding: utf-8 -*-

"""Implementation of classes to handle calibration files."""

import numpy as _np
import json as _json
import collections as _collections
from scipy import interpolate as _interpolate

from imautils.db import database as _database


class CalibrationError(Exception):
    """Calibration exception."""

    def __init__(self, message, *args):
        """Initialize object."""
        self.message = message


class HallCalibrationCurve(_database.DatabaseAndFileDocument):
    """Read, write and stored hall probe calibration curve."""

    label = 'HallCalibrationCurve'
    collection_name = 'hall_calibration_curve'
    db_dict = _collections.OrderedDict([
        ('idn', {'field': 'id', 'dtype': int, 'not_null': True}),
        ('date', {'field': 'date', 'dtype': str, 'not_null': True}),
        ('hour', {'field': 'hour', 'dtype': str, 'not_null': True}),
        ('calibration_name', 
            {'field': 'calibration_name', 'dtype': str, 'not_null': True}),
        ('calibration_magnet', 
            {'field': 'calibration_magnet', 'dtype': str, 'not_null': True}),
        ('function_type', 
            {'field': 'function_type', 'dtype': str, 'not_null': True}),
        ('data', 
            {'field': 'data', 'dtype': _np.ndarray, 'not_null': True}),
    ])


    def __init__(
            self, database_name=None, mongo=False, server=None):
        """Initialize object.

        Args:
            filename (str): connection configuration filepath.
            database_name (str): database file path (sqlite) or name (mongo).
            idn (int): id in database table (sqlite) / collection (mongo).
            mongo (bool): flag indicating mongoDB (True) or sqlite (False).
            server (str): MongoDB server.

        """
        self._function_type = None
        self._function = None
        self._data = []

        super().__init__(
            database_name=database_name, mongo=mongo, server=server)

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

    def get_field(self, voltage):
        """Convert voltage values to magnetic field values.

        Args:
            voltage (array): array with voltage values.

        Return:
            array with magnetic field values.
        """
        if self._function is not None:
            return self._function(_np.array(voltage))
        else:
            return _np.ones(len(_np.array(voltage)))*_np.nan

    def get_calibration_list(self):
        """Get list of calibration names from database."""
        return self.db_get_values(self.db_dict['calibration_name']['field'])

    def update_calibration(self, calibration_name):
        """Update calibration data."""
        if len(calibration_name) == 0:
            return False
        
        docs = self.db_search_field(
            self.db_dict['calibration_name']['field'], calibration_name)
        
        if len(docs) == 0:
            return False
        else:
            idn = docs[-1][self.db_dict['idn']['field']]
            return self.db_read(idn)
        

class HallProbePositions(_database.DatabaseAndFileDocument):
    """Hall probe positions and angles."""

    label = 'HallProbePositions'
    collection_name = 'hall_probe_positions'
    db_dict = _collections.OrderedDict([
        ('idn', {'field': 'id', 'dtype': int, 'not_null': True}),
        ('date', {'field': 'date', 'dtype': str, 'not_null': True}),
        ('hour', {'field': 'hour', 'dtype': str, 'not_null': True}),
        ('probe_name', 
            {'field': 'probe_name', 'dtype': str, 'not_null': True}),
        ('rod_shape', 
            {'field': 'rod_shape', 'dtype': str, 'not_null': True}),
        ('sensorx_position', {
            'field': 'sensorx_position',
            'dtype': _np.ndarray, 'not_null': True}),
        ('sensory_position', {
            'field': 'sensory_position',
            'dtype': _np.ndarray, 'not_null': True}),
        ('sensorz_position', {
            'field': 'sensorz_position',
            'dtype': _np.ndarray, 'not_null': True}),
        ('sensorx_direction', {
            'field': 'sensorx_direction',
            'dtype': _np.ndarray, 'not_null': True}),
        ('sensory_direction', {
            'field': 'sensory_direction',
            'dtype': _np.ndarray, 'not_null': True}),
        ('sensorz_direction', {
            'field': 'sensorz_direction',
            'dtype': _np.ndarray, 'not_null': True}),
    ])

    def __init__(
            self, database_name=None, mongo=False, server=None):
        """Initialize object.

        Args:
            filename (str): connection configuration filepath.
            database_name (str): database file path (sqlite) or name (mongo).
            idn (int): id in database table (sqlite) / collection (mongo).
            mongo (bool): flag indicating mongoDB (True) or sqlite (False).
            server (str): MongoDB server.

        """
        self._rod_shape = None
        self.sensorx_position = _np.array([0, 0, 0])
        self.sensory_position = _np.array([0, 0, 0])
        self.sensorz_position = _np.array([0, 0, 0])
        self.sensorx_direction = _np.array([1, 0, 0])
        self.sensory_direction = _np.array([0, 1, 0])
        self.sensorz_direction = _np.array([0, 0, 1])

        super().__init__(
            database_name=database_name, mongo=mongo, server=server)

    @property
    def rod_shape(self):
        """Return the rod shape."""
        return self._rod_shape

    @rod_shape.setter
    def rod_shape(self, value):
        if isinstance(value, str):
            if value in ('straight', 'L'):
                self._rod_shape = value
            else:
                raise ValueError('Invalid value for rod shape.')
        else:
            raise TypeError('rod_shape must be a string.')

    def clear(self):
        """Clear calibration data."""
        sucess = super().clear()
        self.sensorx_position = _np.array([0, 0, 0])
        self.sensory_position = _np.array([0, 0, 0])
        self.sensorz_position = _np.array([0, 0, 0])
        self.sensorx_direction = _np.array([1, 0, 0])
        self.sensory_direction = _np.array([0, 1, 0])
        self.sensorz_direction = _np.array([0, 0, 1])
        return sucess

    def to_bench_coordinate_system(self, vector):
        """Transform from probe coord. system to bench coord. system."""
        if self._rod_shape == 'L':
            tm = _utils.rotation_matrix([0, 1, 0], -_np.pi/2)
        else:
            tm = _np.eye(3)
        return _np.dot(tm, vector)

    def get_probe_list(self):
        """Get list of probe names from database."""
        return self.db_get_values(self.db_dict['probe_name']['field'])

    def update_probe(self, probe_name):
        """Update probe data."""
        if len(probe_name) == 0:
            return False
        
        docs = self.db_search_field(
            self.db_dict['probe_name']['field'], probe_name)
        
        if len(docs) == 0:
            return False
        else:
            idn = docs[-1][self.db_dict['idn']['field']]
            return self.db_read(idn)


def _interpolation_conversion(data, voltage_array):
    d = _np.array(data)
    interp_func = _interpolate.splrep(d[:, 0], d[:, 1], k=1)
    field_array = _interpolate.splev(voltage_array, interp_func)
    return field_array


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