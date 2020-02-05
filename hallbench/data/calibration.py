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
        ('voltage_min', 
            {'field': 'voltage_min', 'dtype': float}),
        ('voltage_max', 
            {'field': 'voltage_max', 'dtype': float}),
        ('polynomial_coeffs',
            {'field': 'polynomial_coeffs', 'dtype': _np.ndarray}),
        ('voltage', 
            {'field': 'voltage', 'dtype': _np.ndarray}),
        ('magnetic_field', 
            {'field': 'magnetic_field', 'dtype': _np.ndarray}),
        ('probe_temperature', 
            {'field': 'probe_temperature', 'dtype': _np.ndarray}),
        ('electronic_box_temperature', 
            {'field': 'electronic_box_temperature', 'dtype': _np.ndarray}),
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

    def get_field(self, voltage):
        """Convert voltage values to magnetic field values.

        Args:
            voltage (array): array with voltage values.

        Return:
            array with magnetic field values.
        """
        if self.function_type == 'interpolation':
            return _interpolation_conversion(
                self.voltage, self.magnetic_field, voltage)
        else:
            return _polynomial_conversion(
                self.voltage_min, self.voltage_max,
                self.polynomial_coeffs, voltage)

    def get_calibration_list(self):
        """Get list of calibration names from database."""
        return self.db_get_values(self.db_dict['calibration_name']['field'])

    def save_file(self, filename):
        """Save data to file.

        Args:
            filename (str): file fullpath.
        """
        if not self.valid_data():
            message = 'Invalid data.'
            raise ValueError(message)

        columns = []
        if len(self.voltage) != 0 and len(self.magnetic_field) != 0:
            columns = ['voltage', 'magnetic_field']
        if len(self.probe_temperature) != 0:
            columns.append('probe_temperature')
        if len(self.electronic_box_temperature) != 0:
            columns.append('electronic_box_temperature')
        return super().save_file(filename, columns=columns)

    def update_calibration(self, calibration_name):
        """Update calibration data."""
        if calibration_name is None or len(calibration_name) == 0:
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
        ('positionx', {
            'field': 'positionx', 'dtype': _np.ndarray, 'not_null': True}),
        ('positiony', {
            'field': 'positiony', 'dtype': _np.ndarray, 'not_null': True}),
        ('positionz', {
            'field': 'positionz', 'dtype': _np.ndarray, 'not_null': True}),
        ('directionx', {
            'field': 'directionx', 'dtype': _np.ndarray, 'not_null': True}),
        ('directiony', {
            'field': 'directiony', 'dtype': _np.ndarray, 'not_null': True}),
        ('directionz', {
            'field': 'directionz', 'dtype': _np.ndarray, 'not_null': True}),
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
        self.positionx = _np.array([0, 0, 0])
        self.positiony = _np.array([0, 0, 0])
        self.positionz = _np.array([0, 0, 0])
        self.directionx = _np.array([1, 0, 0])
        self.directiony = _np.array([0, 1, 0])
        self.directionz = _np.array([0, 0, 1])

        super().__init__(
            database_name=database_name, mongo=mongo, server=server)

    def clear(self):
        """Clear calibration data."""
        sucess = super().clear()
        self.positionx = _np.array([0, 0, 0])
        self.positiony = _np.array([0, 0, 0])
        self.positionz = _np.array([0, 0, 0])
        self.directionx = _np.array([1, 0, 0])
        self.directiony = _np.array([0, 1, 0])
        self.directionz = _np.array([0, 0, 1])
        return sucess

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


def _interpolation_conversion(voltage, magnetic_field, voltage_array):
    interp_func = _interpolate.splrep(voltage, magnetic_field, k=1)
    field_array = _interpolate.splev(voltage_array, interp_func, ext=1)
    return field_array


def _polynomial_conversion(vmin, vmax, coeffs, voltage_array):
    field_array = _np.zeros(len(voltage_array))
    for i in range(len(voltage_array)):
        voltage = voltage_array[i]
        if vmin is not None and vmax is not None:
            if voltage >= vmin and voltage <= vmax:
                field_array[i] = sum(
                    coeffs[j]*(voltage**j) for j in range(len(coeffs)))
    return field_array