# -*- coding: utf-8 -*-

"""Implementation of classes to handle configuration files."""

import sys as _sys
import numpy as _np
import json as _json
import traceback as _traceback
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


class IntegratorMeasurementConfig(_database.DatabaseAndFileDocument):
    """Read, write and stored measurement configuration data."""

    label = 'Integrator Configuration'
    collection_name = 'integrator_configurations'
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
        ('software_version', 
            {'field': 'software_version', 'dtype': str, 'not_null': False}),
        ('nr_measurements',
            {'field': 'nr_measurements', 'dtype': int, 'not_null': True}),
        ('integrator_gain',
            {'field': 'integrator_gain', 'dtype': int, 'not_null': True}),
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
        ('offset',
            {'field': 'offset', 'dtype': float, 'not_null': False}),
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

class NMRMeasurementConfig(_database.DatabaseAndFileDocument):
    """Read, write and stored measurement configuration data."""

    label = 'NMR Configuration'
    collection_name = 'nmr_configurations'
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
        ('software_version', 
            {'field': 'software_version', 'dtype': str, 'not_null': False}),
        ('nmr_channel',
            {'field': 'nmr_channel', 'dtype': str, 'not_null': True}),
        ('nmr_sense',
            {'field': 'nmr_sense', 'dtype': str, 'not_null': True}),
        ('nmr_mode',
            {'field': 'nmr_mode', 'dtype': str, 'not_null': True}),
        ('nmr_read_value',
            {'field': 'nmr_read_value', 'dtype': str, 'not_null': True}),
        ('nmr_frequency',
            {'field': 'nmr_frequency', 'dtype': int, 'not_null': True}),
        ('field_component',
            {'field': 'field_component', 'dtype': str, 'not_null': True}),
        ('reading_time',
            {'field': 'reading_time', 'dtype': int, 'not_null': True}),
        ('max_time',
            {'field': 'max_time', 'dtype': int, 'not_null': True}),
        ('nr_measurements',
            {'field': 'nr_measurements', 'dtype': int, 'not_null': True}),
        ('axis',
            {'field': 'axis', 'dtype': int, 'not_null': True}),
        ('start_ax1',
            {'field': 'start_ax1', 'dtype': float, 'not_null': True}),
        ('end_ax1', {'field': 'end_ax1', 'dtype': float, 'not_null': True}),
        ('step_ax1', {'field': 'step_ax1', 'dtype': float, 'not_null': True}),
        ('vel_ax1', {'field': 'vel_ax1', 'dtype': float, 'not_null': True}),
        ('start_ax2',
            {'field': 'start_ax2', 'dtype': float, 'not_null': True}),
        ('end_ax2', {'field': 'end_ax2', 'dtype': float, 'not_null': True}),
        ('step_ax2', {'field': 'step_ax2', 'dtype': float, 'not_null': True}),
        ('vel_ax2', {'field': 'vel_ax2', 'dtype': float, 'not_null': True}),
        ('start_ax3',
            {'field': 'start_ax3', 'dtype': float, 'not_null': True}),
        ('end_ax3', {'field': 'end_ax3', 'dtype': float, 'not_null': True}),
        ('step_ax3', {'field': 'step_ax3', 'dtype': float, 'not_null': True}),
        ('vel_ax3', {'field': 'vel_ax3', 'dtype': float, 'not_null': True}),
        ('start_ax5',
            {'field': 'start_ax5', 'dtype': float, 'not_null': True}),
        ('end_ax5', {'field': 'end_ax5', 'dtype': float, 'not_null': True}),
        ('step_ax5', {'field': 'step_ax5', 'dtype': float, 'not_null': True}),
        ('vel_ax5', {'field': 'vel_ax5', 'dtype': float, 'not_null': True}),
        ('start_ax8',
            {'field': 'start_ax8', 'dtype': float, 'not_null': True}),
        ('end_ax8', {'field': 'end_ax8', 'dtype': float, 'not_null': True}),
        ('step_ax8', {'field': 'step_ax', 'dtype': float, 'not_null': True}),
        ('vel_ax8', {'field': 'vel_ax8', 'dtype': float, 'not_null': True}),
        ('start_ax9',
            {'field': 'start_ax9', 'dtype': float, 'not_null': True}),
        ('end_ax9', {'field': 'end_ax9', 'dtype': float, 'not_null': True}),
        ('step_ax9', {'field': 'step_ax9', 'dtype': float, 'not_null': True}),
        ('vel_ax9', {'field': 'vel_ax9', 'dtype': float, 'not_null': True}),
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
        ('sinusoidal_aux_param_0', {
            'field': 'sinusoidal initial phase',
            'dtype': float, 'not_null': True}),
        ('sinusoidal_aux_param_1', {
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
        ('dsinusoidal_aux_param_0', {
            'field': 'damped sinusoidal initial phase',
            'dtype': float, 'not_null': True}),
        ('dsinusoidal_aux_param_1', {
            'field': 'damped sinusoidal final phase',
            'dtype': float, 'not_null': True}),    
        ('dsinusoidal_aux_param_2', {
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
        ('dsinusoidal2_aux_param_0', {
            'field': 'damped sinusoidal2 initial phase',
            'dtype': float, 'not_null': True}),
        ('dsinusoidal2_aux_param_1', {
            'field': 'damped sinusoidal2 final phase',
            'dtype': float, 'not_null': True}),    
        ('dsinusoidal2_aux_param_2', {
            'field': 'damped sinusoidal2 damping',
            'dtype': float, 'not_null': True}), 
        ('trapezoidal_amplitude', {
            'field': 'trapezoidal amplitude',
            'dtype': float, 'not_null': True}),
        ('trapezoidal_offset', {
            'field': 'trapezoidal offset',
            'dtype': float, 'not_null': True}),
        ('trapezoidal_ncycles', {
            'field': 'trapezoidal cycles',
            'dtype': int, 'not_null': True}),
        ('trapezoidal_aux_param_0', {
            'field': 'trapezoidal rise time',
            'dtype': float, 'not_null': True}),
        ('trapezoidal_aux_param_1', {
            'field': 'trapezoidal cte time',
            'dtype': float, 'not_null': True}), 
        ('trapezoidal_aux_param_2', {
            'field': 'trapezoidal fall time',
            'dtype': float, 'not_null': True}),  
        ('dtrapezoidal_offset', {
            'field': 'dtrapezoidal offset',
            'dtype': float, 'not_null': True}),
        ('dtrapezoidal_array', {
            'field': 'dtrapezoidal array',
            'dtype': _np.ndarray, 'not_null': False}),
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

    def get_power_supply_id(self, ps_name):
        """Get power supply database id number."""
        docs = self.db_search_field(self.db_dict['ps_name']['field'], ps_name)

        if len(docs) == 0:
            return None
        
        return docs[-1][self.db_dict['idn']['field']]
    
    def get_power_supply_list(self):
        """Get list of power supply names from database."""
        return self.db_get_values(self.db_dict['ps_name']['field'])
    
    
class CyclingCurve(_database.DatabaseAndFileDocument):
    """Read, write and store Power Supply configuration data."""

    label = 'CyclingCurve'
    collection_name = 'cycling_curve'
    db_dict = _collections.OrderedDict([
        ('idn', {'field': 'id', 'dtype': int, 'not_null': True}),
        ('date', {'field': 'date', 'dtype': str, 'not_null': True}),
        ('hour', {'field': 'hour', 'dtype': str, 'not_null': True}),
        ('power_supply', {
            'field': 'power_supply', 'dtype': str, 'not_null': True}),
        ('curve_name', {
            'field': 'curve_name', 'dtype': str, 'not_null': True}),
        ('siggen_type', {
            'field': 'siggen_type', 'dtype': int, 'not_null': False}),
        ('num_cycles', {
            'field': 'num_cycles', 'dtype': float, 'not_null': False}),
        ('freq', {
            'field': 'freq', 'dtype': float, 'not_null': False}),
        ('amplitude', {
            'field': 'amplitude', 'dtype': float, 'not_null': False}),
        ('offset', {'field': 'offset', 'dtype': float, 'not_null': False}),
        ('aux_param_0', {
            'field': 'aux_param_0', 'dtype': float, 'not_null': False}),
        ('aux_param_1', {
            'field': 'aux_param_1', 'dtype': float, 'not_null': False}),
        ('aux_param_2', {
            'field': 'aux_param_2', 'dtype': float, 'not_null': False}),
        ('aux_param_3', {
            'field': 'aux_param_3', 'dtype': float, 'not_null': False}),
        ('dtrapezoidal_array', {
            'field': 'dtrapezoidal array',
            'dtype': _np.ndarray, 'not_null': False}),
        ('cycling_error_time', {
            'field': 'cycling_error_time',
            'dtype': _np.ndarray, 'not_null': False}),
        ('cycling_error_current', {
            'field': 'cycling_error_current',
            'dtype': _np.ndarray, 'not_null': False}),
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

    def save_file(self, filename):
        """Save data to file.

        Args:
            filename (str): file fullpath.
        """
        if not self.valid_data():
            message = 'Invalid data.'
            raise ValueError(message)

        columns = ['cycling_error_time', 'cycling_error_current']
        return super().save_file(filename, columns=columns)
    
    def get_sinusoidal_curve(self, time_array):
        try:
            w = 2*_np.pi*self.freq
            total_time = self.num_cycles/self.freq
            
            current_array = self.amplitude*_np.sin(
                w*time_array + self.aux_param_0) + self.offset
            current_array[time_array > total_time] = self.offset
            
            return current_array
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return [_np.nan]*len(time_array)

    def get_dsinusoidal_curve(self, time_array, correct_amplitude=True):
        try:
            w = 2*_np.pi*self.freq
            total_time = self.num_cycles/self.freq
            
            if correct_amplitude:
                t0 = (1/w)*_np.arctan(self.aux_param_2*w)
                amp_corr = _np.exp(t0/self.aux_param_2)/(_np.sin(w*t0))
            else:
                amp_corr = 1
            
            current_array = self.amplitude*amp_corr*_np.sin(
                w*time_array + self.aux_param_0)*_np.exp(
                    -time_array/self.aux_param_2) + self.offset
            current_array[time_array > total_time] = self.offset
            
            return current_array
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return [_np.nan]*len(time_array)

    def get_dsinusoidal2_curve(self, time_array, correct_amplitude=True):
        try:
            w = 2*_np.pi*self.freq
            total_time = self.num_cycles/self.freq
            
            if correct_amplitude:
                t0 = (1/w)*_np.arctan(2*self.aux_param_2*w)
                amp_corr = _np.exp(t0/self.aux_param_2)/(_np.sin(w*t0)**2) 
            else:
                amp_corr = 1
            
            current_array = self.amplitude*amp_corr*(_np.sin(
                w*time_array + self.aux_param_0)**2)*_np.exp(
                    -time_array/self.aux_param_2) + self.offset
            current_array[time_array > total_time] = self.offset
            
            return current_array
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return [_np.nan]*len(time_array)
    
    def get_trapezoidal_curve(self, time_array):
        try:
            t0 = self.aux_param_0
            t1 = self.aux_param_1
            t2 = self.aux_param_2
            total_time = (t0 + t1 + t2)*self.num_cycles 
            ts = _np.array([0, t0, t0+t1, t0+t1+t2], dtype=float)
            cs = _np.array([
                self.offset, self.offset + self.amplitude,
                self.offset + self.amplitude, self.offset], dtype=float)
            
            current_array = _np.interp(time_array, ts, cs, period=t0+t1+t2)
            current_array[time_array>total_time] = self.offset
            
            return current_array
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return [_np.nan]*len(time_array)

    def get_dtrapezoidal_curve(self, time_array, slope=50):
        try:
            t0 = 0
            c0 = self.offset
            ts = []
            cs = []
            for val in self.dtrapezoidal_array:
                c = self.offset + val[0]
                t = val[1]
                t_border = abs(c - c0) / slope
                
                ts = _np.append(ts, [t0, t0 + t_border])
                cs = _np.append(cs, [c0, c])

                ts = _np.append(ts, [t0 + t_border, t0 + t_border + t])
                cs = _np.append(cs, [c, c])

                ts = _np.append(ts, [t0 + t_border +t, t0 + 2*t_border + t])
                cs = _np.append(cs, [c, c0])
                
                ts = _np.append(
                    ts, [t0 + 2*t_border + t, t0 + 2*(t_border + t)])
                cs = _np.append(cs, [c0, c0])
                
                t0 = t0 + 2*(t_border + t)

            current_array = _np.interp(time_array, ts, cs)
            current_array[time_array > t0] = self.offset  

            return current_array

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return [_np.nan]*len(time_array)

    def get_curve_duration(self):
        try:
            if self.curve_name in [
                    'sinusoidal', 'dsinusoidal', 'dsinusoidal2']:
                return self.num_cycles/self.freq
           
            elif self.curve_name == 'trapezoidal':
                return (
                    self.aux_param_0 + self.aux_param_1 +
                        self.aux_param_2)*self.num_cycles 

            elif self.curve_name == 'dtrapezoidal':
                raise NotImplemented
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return 0        

    def get_curve(self, time_array, correct_amplitude=True):
        try:
            if self.curve_name == 'sinusoidal':
                return self.get_sinusoidal_curve(time_array)

            elif self.curve_name == 'dsinusoidal':
                return self.get_dsinusoidal_curve(
                    time_array, correct_amplitude=correct_amplitude) 
            
            elif self.curve_name == 'dsinusoidal2':
                return self.get_dsinusoidal2_curve(
                    time_array, correct_amplitude=correct_amplitude) 
            
            elif self.curve_name == 'trapezoidal':
                return self.get_trapezoidal_curve(time_array) 

            elif self.curve_name == 'dtrapezoidal':
                return [_np.nan]*len(time_array)
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return [_np.nan]*len(time_array)