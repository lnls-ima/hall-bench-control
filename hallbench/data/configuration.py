# -*- coding: utf-8 -*-

"""Implementation of classes to handle configuration files."""

import numpy as _np
import json as _json
import collections as _collections

from imautils.db import database as _database


class ConnectionConfig(_database.DatabaseAndFileDocument):
    """Read, write and stored connection configuration data."""

    label = 'Connection'
    collection_name = 'connections'
    db_dict = _collections.OrderedDict([
        ('idn', {'field': 'id', 'dtype': int, 'not_null': True}),
        ('date', {'field': 'date', 'dtype': str, 'not_null': True}),
        ('hour', {'field': 'hour', 'dtype': str, 'not_null': True}),
        ('software_version', 
            {'field': 'software_version', 'dtype': str, 'not_null': False}),
        ('pmac_enable', 
            {'field': 'pmac_enable', 'dtype': int, 'not_null': True}),
        ('voltx_enable', 
            {'field': 'voltx_enable', 'dtype': int, 'not_null': True}),
        ('voltx_address', 
            {'field': 'voltx_address', 'dtype': int, 'not_null': True}),
        ('volty_enable', 
            {'field': 'volty_enable', 'dtype': int, 'not_null': True}),
        ('volty_address', 
            {'field': 'volty_address', 'dtype': int, 'not_null': True}),
        ('voltz_enable', 
            {'field': 'voltz_enable', 'dtype': int, 'not_null': True}),
        ('voltz_address', 
            {'field': 'voltz_address', 'dtype': int, 'not_null': True}),
        ('multich_enable', 
            {'field': 'multich_enable', 'dtype': int, 'not_null': True}),
        ('multich_address', 
            {'field': 'multich_address', 'dtype': int, 'not_null': True}),
        ('nmr_enable', 
            {'field': 'nmr_enable', 'dtype': int, 'not_null': True}),
        ('nmr_port', 
            {'field': 'nmr_port', 'dtype': str, 'not_null': True}),
        ('nmr_baudrate', 
            {'field': 'nmr_baudrate', 'dtype': int, 'not_null': True}),
        ('elcomat_enable', 
            {'field': 'elcomat_enable', 'dtype': int, 'not_null': True}),
        ('elcomat_port', 
            {'field': 'elcomat_port', 'dtype': str, 'not_null': True}),
        ('elcomat_baudrate', 
            {'field': 'elcomat_baudrate', 'dtype': int, 'not_null': True}),
        ('dcct_enable', 
            {'field': 'dcct_enable', 'dtype': int, 'not_null': True}),
        ('dcct_address',
            {'field': 'dcct_address', 'dtype': int, 'not_null': True}),
        ('ps_enable',
            {'field': 'ps_enable', 'dtype': int, 'not_null': True}),
        ('ps_port',
            {'field': 'ps_port', 'dtype': str, 'not_null': True}),
        ('water_udc_enable',
            {'field': 'water_udc_enable', 'dtype': int, 'not_null': True}),
        ('water_udc_port',
            {'field': 'water_udc_port', 'dtype': str, 'not_null': True}),
        ('water_udc_baudrate',
            {'field': 'water_udc_baudrate', 'dtype': int, 'not_null': True}),
        ('water_udc_slave_address',
            {'field': 'water_udc_slave_address',
             'dtype': int, 'not_null': True}),
        ('air_udc_enable',
            {'field': 'air_udc_enable', 'dtype': int, 'not_null': True}),
        ('air_udc_port',
            {'field': 'air_udc_port', 'dtype': str, 'not_null': True}),
        ('air_udc_baudrate',
            {'field': 'air_udc_baudrate', 'dtype': int, 'not_null': True}),
        ('air_udc_slave_address',
            {'field': 'air_udc_slave_address',
             'dtype': int, 'not_null': True}),
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
        super().__init__(
            database_name=database_name, mongo=mongo, server=server)


class MeasurementConfig(_database.DatabaseAndFileDocument):
    """Read, write and stored measurement configuration data."""

    label = 'Configuration'
    collection_name = 'configurations'
    db_dict = _collections.OrderedDict([
        ('idn', {'field': 'id', 'dtype': int, 'not_null': True}),
        ('date', {'field': 'date', 'dtype': str, 'not_null': True}),
        ('hour', {'field': 'hour', 'dtype': str, 'not_null': True}),
        ('magnet_name',
            {'field': 'magnet_name', 'dtype': str, 'not_null': True}),
        ('operator', {'field': 'operator', 'dtype': str, 'not_null': False}),      
        ('current_setpoint',
            {'field': 'current_setpoint', 'dtype': float, 'not_null': False}),
        ('comments', {'field': 'comments', 'dtype': str, 'not_null': False}),
        ('calibrationx',
            {'field': 'calibrationx', 'dtype': str, 'not_null': False}),
        ('calibrationy',
            {'field': 'calibrationy', 'dtype': str, 'not_null': False}),
        ('calibrationz',
            {'field': 'calibrationz', 'dtype': str, 'not_null': False}),
        ('software_version', 
            {'field': 'software_version', 'dtype': str, 'not_null': False}),
        ('voltx_enable',
            {'field': 'voltx_enable', 'dtype': int, 'not_null': True}),
        ('volty_enable',
            {'field': 'volty_enable', 'dtype': int, 'not_null': True}),
        ('voltz_enable',
            {'field': 'voltz_enable', 'dtype': int, 'not_null': True}),
        ('voltage_format',
            {'field': 'voltage_format', 'dtype': str, 'not_null': True}),
        ('voltage_range',
            {'field': 'voltage_range', 'dtype': float, 'not_null': False}),
        ('integration_time',
            {'field': 'integration_time', 'dtype': float, 'not_null': True}),
        ('nr_measurements',
            {'field': 'nr_measurements', 'dtype': int, 'not_null': True}),
        ('first_axis',
            {'field': 'first_axis', 'dtype': int, 'not_null': True}),
        ('second_axis',
            {'field': 'second_axis', 'dtype': int, 'not_null': True}),
        ('start_ax1',
            {'field': 'start_ax1', 'dtype': float, 'not_null': True}),
        ('end_ax1', {'field': 'end_ax1', 'dtype': float, 'not_null': True}),
        ('step_ax1', {'field': 'step_ax1', 'dtype': float, 'not_null': True}),
        ('extra_ax1',
            {'field': 'extra_ax1', 'dtype': float, 'not_null': True}),
        ('vel_ax1', {'field': 'vel_ax1', 'dtype': float, 'not_null': True}),
        ('start_ax2',
            {'field': 'start_ax2', 'dtype': float, 'not_null': True}),
        ('end_ax2', {'field': 'end_ax2', 'dtype': float, 'not_null': True}),
        ('step_ax2', {'field': 'step_ax2', 'dtype': float, 'not_null': True}),
        ('extra_ax2',
            {'field': 'extra_ax2', 'dtype': float, 'not_null': True}),
        ('vel_ax2', {'field': 'vel_ax2', 'dtype': float, 'not_null': True}),
        ('start_ax3',
            {'field': 'start_ax3', 'dtype': float, 'not_null': True}),
        ('end_ax3', {'field': 'end_ax3', 'dtype': float, 'not_null': True}),
        ('step_ax3', {'field': 'step_ax3', 'dtype': float, 'not_null': True}),
        ('extra_ax3',
            {'field': 'extra_ax3', 'dtype': float, 'not_null': True}),
        ('vel_ax3', {'field': 'vel_ax3', 'dtype': float, 'not_null': True}),
        ('start_ax5',
            {'field': 'start_ax5', 'dtype': float, 'not_null': True}),
        ('end_ax5', {'field': 'end_ax5', 'dtype': float, 'not_null': True}),
        ('step_ax5', {'field': 'step_ax5', 'dtype': float, 'not_null': True}),
        ('extra_ax5',
            {'field': 'extra_ax5', 'dtype': float, 'not_null': True}),
        ('vel_ax5', {'field': 'vel_ax5', 'dtype': float, 'not_null': True}),
        ('voltage_offset',
            {'field': 'voltage_offset', 'dtype': str, 'not_null': False}),
        ('offsetx',
            {'field': 'offsetx', 'dtype': float, 'not_null': False}),
        ('offsety',
            {'field': 'offsety', 'dtype': float, 'not_null': False}),
        ('offsetz',
            {'field': 'offsetz', 'dtype': float, 'not_null': False}),
        ('offset_range',
            {'field': 'offset_range', 'dtype': float, 'not_null': False}),
        ('on_the_fly',
            {'field': 'on_the_fly', 'dtype': int, 'not_null': False}),
        ('save_current',
            {'field': 'save_current', 'dtype': float, 'not_null': False}),
        ('save_temperature',
            {'field': 'save_temperature', 'dtype': float, 'not_null': False}),
        ('automatic_ramp',
            {'field': 'automatic_ramp', 'dtype': float, 'not_null': False}),
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
        self.comments = ''
        super().__init__(
            database_name=database_name, mongo=mongo, server=server)

    def get_end(self, axis):
        """Get end position for the given axis."""
        return getattr(self, 'end_ax' + str(axis))
 
    def get_extra(self, axis):
        """Get extra position for the given axis."""
        return getattr(self, 'extra_ax' + str(axis))
 
    def get_start(self, axis):
        """Get start position for the given axis."""
        return getattr(self, 'start_ax' + str(axis))
 
    def get_step(self, axis):
        """Get position step for the given axis."""
        return getattr(self, 'step_ax' + str(axis))
 
    def get_velocity(self, axis):
        """Get velocity for the given axis."""
        return getattr(self, 'vel_ax' + str(axis))
  
    def set_end(self, axis, value):
        """Set end position value for the given axis."""
        attr = 'end_ax' + str(axis)
        setattr(self, attr, value)
 
    def set_extra(self, axis, value):
        """Get extra position for the given axis."""
        attr = 'extra_ax' + str(axis)
        setattr(self, attr, value)
 
    def set_start(self, axis, value):
        """Set start position value for the given axis."""
        attr = 'start_ax' + str(axis)
        setattr(self, attr, value)
 
    def set_step(self, axis, value):
        """Set position step value for the given axis."""
        attr = 'step_ax' + str(axis)
        setattr(self, attr, value)
 
    def set_velocity(self, axis, value):
        """Set velocity value for the given axis."""
        attr = 'vel_ax' + str(axis)
        setattr(self, attr, value)


class PowerSupplyConfig(_database.DatabaseAndFileDocument):
    """Read, write and store Power Supply configuration data."""

    label = 'PowerSupply'
    collection_name = 'power_supply'
    db_dict = _collections.OrderedDict([
        ('idn', {'field': 'id', 'dtype': int, 'not_null': True}),
        ('date', {'field': 'date', 'dtype': str, 'not_null': True}),
        ('hour', {'field': 'hour', 'dtype': str, 'not_null': True}),
        ('ps_name', {'field': 'name', 'dtype': str, 'not_null': True}),
        ('ps_type', {'field': 'type', 'dtype': int, 'not_null': True}),
        ('dclink', {'field': 'dclink', 'dtype': float, 'not_null': False}),
        ('ps_setpoint', 
            {'field': 'setpoint', 'dtype': float, 'not_null': True}),
        ('maximum_current', 
            {'field': 'maximum current', 'dtype': float, 'not_null': True}),
        ('minimum_current', 
            {'field': 'minimum current', 'dtype': float, 'not_null': True}),
        ('dcct', 
            {'field': 'DCCT Enabled', 'dtype': int, 'not_null': True}),
        ('dcct_head', 
            {'field': 'DCCT Head', 'dtype': int, 'not_null': True}),
        ('Kp', {'field': 'Kp', 'dtype': float, 'not_null': True}),
        ('Ki', {'field': 'Ki', 'dtype': float, 'not_null': True}),
        ('current_array', {
            'field': 'current array',
            'dtype': _np.ndarray, 'not_null': False}),
        ('trapezoidal_array', {
            'field': 'trapezoidal array',
            'dtype': _np.ndarray, 'not_null': False}),
        ('trapezoidal_offset', {
            'field': 'trapezoidal offset',
            'dtype': float, 'not_null': True}),
        ('sinusoidal_amplitude', {
            'field': 'sinusoidal amplitude',
            'dtype': float, 'not_null': True}),
        ('sinusoidal_offset', {
            'field': 'sinusoidal offset',
            'dtype': float, 'not_null': True}),
        ('sinusoidal_frequency', {
            'field': 'sinusoidal frequency',
            'dtype': float, 'not_null': True}),
        ('sinusoidal_ncycles', {
            'field': 'sinusoidal cycles',
            'dtype': int, 'not_null': True}),
        ('sinusoidal_phasei', {
            'field': 'sinusoidal initial phase',
            'dtype': float, 'not_null': True}),
        ('sinusoidal_phasef', {
            'field': 'sinusoidal final phase',
            'dtype': float, 'not_null': True}),    
        ('dsinusoidal_amplitude', {
            'field': 'damped sinusoidal amplitude',
            'dtype': float, 'not_null': True}),
        ('dsinusoidal_offset', {
            'field': 'damped sinusoidal offset',
            'dtype': float, 'not_null': True}),
        ('dsinusoidal_frequency', {
            'field': 'damped sinusoidal frequency',
            'dtype': float, 'not_null': True}),
        ('dsinusoidal_ncycles', {
            'field': 'damped sinusoidal cycles',
            'dtype': int, 'not_null': True}),
        ('dsinusoidal_phasei', {
            'field': 'damped sinusoidal initial phase',
            'dtype': float, 'not_null': True}),
        ('dsinusoidal_phasef', {
            'field': 'damped sinusoidal final phase',
            'dtype': float, 'not_null': True}),    
        ('dsinusoidal_damp', {
            'field': 'damped sinusoidal damping',
            'dtype': float, 'not_null': True}), 
        ('dsinusoidal2_amplitude', {
            'field': 'damped sinusoidal2 amplitude',
            'dtype': float, 'not_null': True}),
        ('dsinusoidal2_offset', {
            'field': 'damped sinusoidal2 offset',
            'dtype': float, 'not_null': True}),
        ('dsinusoidal2_frequency', {
            'field': 'damped sinusoidal2 frequency',
            'dtype': float, 'not_null': True}),
        ('dsinusoidal2_ncycles', {
            'field': 'damped sinusoidal2 cycles',
            'dtype': int, 'not_null': True}),
        ('dsinusoidal2_phasei', {
            'field': 'damped sinusoidal2 initial phase',
            'dtype': float, 'not_null': True}),
        ('dsinusoidal2_phasef', {
            'field': 'damped sinusoidal2 final phase',
            'dtype': float, 'not_null': True}),    
        ('dsinusoidal2_damp', {
            'field': 'damped sinusoidal2 damping',
            'dtype': float, 'not_null': True}), 
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
        # DC link voltage (90V is the default)
        self.dclink = 90
        # True for DCCT enabled, False for DCCT disabled
        self.dcct = False
        # Power supply status (False = off, True = on)
        self.status = False
        # Power supply loop status (False = open, True = closed)
        self.status_loop = False
        # Power supply connection status (False = no communication)
        self.status_con = False
        # Power supply interlock status (True = active, False = not active)
        self.status_interlock = False
        # Main current
        self.main_current = 0
        # Flag to enable or disable display update
        self.update_display = True

        super().__init__(
            database_name=database_name, mongo=mongo, server=server)