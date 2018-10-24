# -*- coding: utf-8 -*-

"""Implementation of classes to handle configuration files."""

import numpy as _np
import collections as _collections

from . import utils as _utils
from . import database as _database


_empty_str = '--'


class ConfigurationError(Exception):
    """Configuration exception."""

    def __init__(self, message, *args):
        """Initialize object."""
        self.message = message


class Configuration(_database.DatabaseObject):
    """Base class for configurations."""

    _label = ''
    _db_table = ''
    _db_dict = {}
    _db_json_str = []

    def __init__(self, filename=None, database=None, idn=None):
        """Initialize obejct.

        Args:
            filename (str): connection configuration filepath.
            database (str): database file path.
            idn (int): id in database table.
        """
        if filename is not None and idn is not None:
            raise ValueError('Invalid arguments for Configuration object.')

        if idn is not None and database is not None:
            self.read_from_database(database, idn)
        elif filename is not None:
            self.read_file(filename)

    def __eq__(self, other):
        """Equality method."""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        """Non-equality method."""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __setattr__(self, name, value):
        """Set attribute."""
        tp = self.get_attribute_type(name)
        if value is None or tp is None or isinstance(value, tp):
            super(Configuration, self).__setattr__(name, value)
        elif tp == float and isinstance(value, int):
            super(Configuration, self).__setattr__(name, float(value))
        else:
            raise TypeError('%s must be of type %s.' % (name, tp.__name__))

    def __str__(self):
        """Printable string representation of the object."""
        fmtstr = '{0:<18s} : {1}\n'
        r = ''
        for key, value in self.__dict__.items():
            if key.startswith('_'):
                name = key[1:]
            else:
                name = key
            r += fmtstr.format(name, str(value))
        return r

    @property
    def default_filename(self):
        """Return the default filename."""
        timestamp = _utils.get_timestamp()
        filename = '{0:1s}_{1:1s}.txt'.format(timestamp, self._label)
        return filename

    def clear(self):
        """Clear configuration."""
        for key in self.__dict__:
            self.__dict__[key] = None

    def copy(self):
        """Return a copy of the object."""
        _copy = type(self)()
        for key in self.__dict__:
            if isinstance(self.__dict__[key], list):
                _copy.__dict__[key] = self.__dict__[key].copy()
            else:
                _copy.__dict__[key] = self.__dict__[key]
        return _copy

    def get_attribute_type(self, name):
        """Get attribute type."""
        return None

    def read_file(self, filename):
        """Read configuration from file.

        Args:
            filename (str): configuration filepath.
        """
        data = _utils.read_file(filename)
        for name in self.__dict__:
            tp = self.get_attribute_type(name)
            if tp is not None:
                value_str = _utils.find_value(data, name)
                if value_str == _empty_str:
                    setattr(self, name, None)
                else:
                    value = _utils.find_value(data, name, vtype=tp)
                    setattr(self, name, value)

    def save_file(self, filename):
        """Save configuration to file."""
        pass

    def valid_data(self, valid_none=[]):
        """Check if parameters are valid."""
        al = [getattr(self, a) for a in self.__dict__ if a not in valid_none]
        if all([a is not None for a in al]):
            return True
        else:
            return False


class ConnectionConfig(Configuration):
    """Read, write and stored connection configuration data."""

    _label = 'Connection'
    _db_table = 'connections'
    _db_dict = _collections.OrderedDict([
        ('id', [None, 'INTEGER NOT NULL']),
        ('date', [None, 'TEXT NOT NULL']),
        ('hour', [None, 'TEXT NOT NULL']),
        ('software_version', [None, 'TEXT']),
        ('pmac_enable', ['pmac_enable', 'INTEGER NOT NULL']),
        ('voltx_enable', ['voltx_enable', 'INTEGER NOT NULL']),
        ('voltx_address', ['voltx_address', 'INTEGER NOT NULL']),
        ('volty_enable', ['volty_enable', 'INTEGER NOT NULL']),
        ('volty_address', ['volty_address', 'INTEGER NOT NULL']),
        ('voltz_enable', ['voltz_enable', 'INTEGER NOT NULL']),
        ('voltz_address', ['voltz_address', 'INTEGER NOT NULL']),
        ('multich_enable', ['multich_enable', 'INTEGER NOT NULL']),
        ('multich_address', ['multich_address', 'INTEGER NOT NULL']),
        ('nmr_enable', ['nmr_enable', 'INTEGER NOT NULL']),
        ('nmr_port', ['nmr_port', 'TEXT NOT NULL']),
        ('nmr_baudrate', ['nmr_baudrate', 'INTEGER NOT NULL']),
        ('elcomat_enable', ['elcomat_enable', 'INTEGER NOT NULL']),
        ('elcomat_port', ['elcomat_port', 'TEXT NOT NULL']),
        ('elcomat_baudrate', ['elcomat_baudrate', 'INTEGER NOT NULL']),
        ('dcct_enable', ['dcct_enable', 'INTEGER NOT NULL']),
        ('dcct_address', ['dcct_address', 'INTEGER NOT NULL']),
        ('ps_enable', ['ps_enable', 'INTEGER NOT NULL']),
        ('ps_port', ['ps_port', 'TEXT NOT NULL']),
        ('udc_enable', ['udc_enable', 'INTEGER NOT NULL']),
        ('udc_port', ['udc_port', 'TEXT NOT NULL']),
        ('udc_baudrate', ['udc_baudrate', 'INTEGER NOT NULL']),   
    ])

    def __init__(self, filename=None, database=None, idn=None):
        """Initialize object.

        Args:
            filename (str): connection configuration filepath.
            database (str): database file path.
            idn (int): id in database table.
        """
        self.pmac_enable = None
        self.voltx_enable = None
        self.voltx_address = None
        self.volty_enable = None
        self.volty_address = None
        self.voltz_enable = None
        self.voltz_address = None
        self.multich_enable = None
        self.multich_address = None
        self.nmr_enable = None
        self.nmr_port = None
        self.nmr_baudrate = None
        self.elcomat_enable = None
        self.elcomat_port = None
        self.elcomat_baudrate = None
        self.dcct_enable = None
        self.dcct_address = None
        self.ps_enable = None
        self.ps_port = None
        self.udc_enable = None
        self.udc_port = None
        self.udc_baudrate = None
        super().__init__(filename=filename, database=database, idn=idn)

    def get_attribute_type(self, name):
        """Get attribute type."""
        if 'port' in name:
            return str
        else:
            return int

    def save_file(self, filename):
        """Save connection configuration to file.

        Args:
            filename (str): configuration filepath.

        Raise:
            ConfigurationError: if the configuration was not saved.
        """
        if not self.valid_data():
            message = 'Invalid Configuration.'
            raise ConfigurationError(message)

        try:
            data = [
                '# Configuration File\n\n',
                'pmac_enable      \t{0:d}\n\n'.format(self.pmac_enable),
                'voltx_enable     \t{0:d}\n'.format(self.voltx_enable),
                'voltx_address    \t{0:d}\n'.format(self.voltx_address),
                'volty_enable     \t{0:d}\n'.format(self.volty_enable),
                'volty_address    \t{0:d}\n'.format(self.volty_address),
                'voltz_enable     \t{0:d}\n'.format(self.voltz_enable),
                'voltz_address    \t{0:d}\n\n'.format(self.voltz_address),
                'multich_enable   \t{0:d}\n'.format(self.multich_enable),
                'multich_address  \t{0:d}\n\n'.format(self.multich_address),
                'nmr_enable       \t{0:d}\n'.format(self.nmr_enable),
                'nmr_port         \t{0:s}\n'.format(self.nmr_port),
                'nmr_baudrate     \t{0:d}\n\n'.format(self.nmr_baudrate),
                'elcomat_enable   \t{0:d}\n'.format(self.elcomat_enable),
                'elcomat_port     \t{0:s}\n'.format(self.elcomat_port),
                'elcomat_baudrate \t{0:d}\n\n'.format(self.elcomat_baudrate),
                'dcct_enable      \t{0:d}\n'.format(self.dcct_enable),
                'dcct_address     \t{0:d}\n'.format(self.dcct_address),
                'ps_enable        \t{0:d}\n'.format(self.ps_enable),
                'ps_port          \t{0:s}\n'.format(self.ps_port),
                'udc_enable       \t{0:d}\n'.format(self.udc_enable),
                'udc_port         \t{0:s}\n'.format(self.udc_port),
                'udc_baudrate     \t{0:d}\n\n'.format(self.udc_baudrate),
                ]

            with open(filename, mode='w') as f:
                for item in data:
                    f.write(item)

        except Exception:
            message = 'Failed to save configuration to file: "%s"' % filename
            raise ConfigurationError(message)


class MeasurementConfig(Configuration):
    """Read, write and stored measurement configuration data."""

    _label = 'Configuration'
    _db_table = 'configurations'
    _db_dict = _collections.OrderedDict([
        ('id', [None, 'INTEGER NOT NULL']),
        ('date', [None, 'TEXT NOT NULL']),
        ('hour', [None, 'TEXT NOT NULL']),
        ('magnet_name', ['magnet_name', 'TEXT NOT NULL']),
        ('current_setpoint', ['current_setpoint', 'REAL']),
        ('operator', ['operator', 'TEXT']),
        ('comments', ['comments', 'TEXT']),
        ('temperature', ['temperature', 'TEXT']),
        ('probe_name', ['probe_name', 'TEXT']),
        ('software_version', [None, 'TEXT']),
        ('voltx_enable', ['voltx_enable', 'INTEGER NOT NULL']),
        ('volty_enable', ['volty_enable', 'INTEGER NOT NULL']),
        ('voltz_enable', ['voltz_enable', 'INTEGER NOT NULL']),
        ('voltage_precision', ['voltage_precision', 'INTEGER NOT NULL']),
        ('voltage_range', ['voltage_range', 'REAL']),
        ('integration_time', ['integration_time', 'REAL NOT NULL']),
        ('nr_measurements', ['nr_measurements', 'INTEGER NOT NULL']),
        ('first_axis', ['first_axis', 'INTEGER NOT NULL']),
        ('second_axis', ['second_axis', 'INTEGER NOT NULL']),
        ('start_ax1', ['start_ax1', 'REAL NOT NULL']),
        ('end_ax1', ['end_ax1', 'REAL NOT NULL']),
        ('step_ax1', ['step_ax1', 'REAL NOT NULL']),
        ('extra_ax1', ['extra_ax1', 'REAL NOT NULL']),
        ('vel_ax1', ['vel_ax1', 'REAL NOT NULL']),
        ('start_ax2', ['start_ax2', 'REAL NOT NULL']),
        ('end_ax2', ['end_ax2', 'REAL NOT NULL']),
        ('step_ax2', ['step_ax2', 'REAL NOT NULL']),
        ('extra_ax2', ['extra_ax2', 'REAL NOT NULL']),
        ('vel_ax2', ['vel_ax2', 'REAL NOT NULL']),
        ('start_ax3', ['start_ax3', 'REAL NOT NULL']),
        ('end_ax3', ['end_ax3', 'REAL NOT NULL']),
        ('step_ax3', ['step_ax3', 'REAL NOT NULL']),
        ('extra_ax3', ['extra_ax3', 'REAL NOT NULL']),
        ('vel_ax3', ['vel_ax3', 'REAL NOT NULL']),
        ('start_ax5', ['start_ax5', 'REAL NOT NULL']),
        ('end_ax5', ['end_ax5', 'REAL NOT NULL']),
        ('step_ax5', ['step_ax5', 'REAL NOT NULL']),
        ('extra_ax5', ['extra_ax5', 'REAL NOT NULL']),
        ('vel_ax5', ['vel_ax5', 'REAL NOT NULL']),
        ('subtract_voltage_offset', ['subtract_voltage_offset', 'INTEGER']),
        ('save_voltage', ['save_voltage', 'INTEGER']),
        ('save_current', ['save_current', 'INTEGER']),
        ('save_temperature', ['save_temperature', 'INTEGER']),
        ('automatic_ramp', ['automatic_ramp', 'INTEGER']),
    ])

    def __init__(self, filename=None, database=None, idn=None):
        """Initialize object.

        Args:
            filename (str): measurement configuration filepath.
            database (str): database file path.
            idn (int): id in database table.
        """
        self.magnet_name = None
        self.current_setpoint = None
        self.probe_name = None
        self.temperature = None
        self.operator = None
        self.comments = ''
        self.voltx_enable = None
        self.volty_enable = None
        self.voltz_enable = None
        self.integration_time = None
        self.voltage_precision = None
        self.voltage_range = None
        self.nr_measurements = None
        self.first_axis = None
        self.second_axis = None
        self.start_ax1 = None
        self.end_ax1 = None
        self.step_ax1 = None
        self.extra_ax1 = None
        self.vel_ax1 = None
        self.start_ax2 = None
        self.end_ax2 = None
        self.step_ax2 = None
        self.extra_ax2 = None
        self.vel_ax2 = None
        self.start_ax3 = None
        self.end_ax3 = None
        self.step_ax3 = None
        self.extra_ax3 = None
        self.vel_ax3 = None
        self.start_ax5 = None
        self.end_ax5 = None
        self.step_ax5 = None
        self.extra_ax5 = None
        self.vel_ax5 = None
        self.subtract_voltage_offset = None
        self.save_voltage = None
        self.save_current = None
        self.save_temperature = None
        self.automatic_ramp = None
        super().__init__(filename=filename, database=database, idn=idn)

    @classmethod
    def get_probe_name_from_database(cls, database, idn):
        """Return the probe name of the database record."""
        probe_name = cls.get_database_param(database, idn, 'probe_name')
        return probe_name

    def _set_axis_param(self, param, axis, value):
        axis_param = param + str(axis)
        if value is None:
            setattr(self, axis_param, None)
        else:
            vtype = self.get_attribute_type(axis_param)
            if isinstance(value, vtype):
                setattr(self, axis_param, value)
            elif isinstance(value, int) and vtype == float:
                setattr(self, axis_param, float(value))
            else:
                raise ConfigurationError('Invalid value for "%s"' % axis_param)

    def get_attribute_type(self, name):
        """Get attribute type."""
        if name in ['voltx_enable', 'volty_enable', 'voltz_enable',
                    'nr_measurements', 'voltage_precision',
                    'first_axis', 'second_axis', 'subtract_voltage_offset',
                    'save_voltage', 'save_current', 'save_temperature',
                    'automatic_ramp']:
            return int
        elif name in ['magnet_name', 'probe_name',
                      'temperature', 'operator', 'comments']:
            return str
        else:
            return float

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

    def read_file(self, filename):
        """Read configuration from file.

        Args:
            filename (str): configuration filepath.
        """
        super().read_file(filename)
        if self.comments is None:
            self.comments = ''

    def save_file(self, filename):
        """Save measurement configuration to file.

        Args:
            filename (str): configuration filepath.

        Raise:
            ConfigurationError: if the configuration was not saved.
        """
        if not self.valid_data():
            message = 'Invalid Configuration.'
            raise ConfigurationError(message)

        try:
            if self.current_setpoint is not None:
                current_setpoint = str(self.current_setpoint)
            else:
                current_setpoint = _empty_str

            if self.comments is not None and len(self.comments) != 0:
                comments = self.comments.replace(' ', '_')
            else:
                comments = _empty_str

            if self.probe_name is not None and len(self.probe_name) != 0:
                probe_name = self.probe_name
            else:
                probe_name = _empty_str

            data = [
                '# Measurement Setup\n\n',
                'magnet_name \t{0:s}\n'.format(self.magnet_name),
                'current_setpoint\t{0:s}\n'.format(current_setpoint),
                'temperature \t{0:s}\n'.format(self.temperature),
                'operator    \t{0:s}\n\n'.format(self.operator),
                '# Hall Probe\n',
                'probe_name  \t{0:s}\n\n'.format(probe_name),
                '# Digital Multimeters (X, Y, Z)\n',
                'voltx_enable\t{0:1d}\n'.format(self.voltx_enable),
                'volty_enable\t{0:1d}\n'.format(self.volty_enable),
                'voltz_enable\t{0:1d}\n\n'.format(self.voltz_enable),
                '# Digital Multimeter (aper [ms])\n',
                'integration_time \t{0:4f}\n\n'.format(self.integration_time),
                '# Digital Multimeter (precision [single=0 or double=1])\n',
                'voltage_precision \t{0:1d}\n\n'.format(
                    self.voltage_precision),
                '# Digital Multimeter (range [V])\n',
                'voltage_range \t{0:f}\n\n'.format(self.voltage_range),
                '# Number of measurements\n',
                'nr_measurements \t{0:1d}\n\n'.format(self.nr_measurements),
                '# First Axis (Triggering Axis)\n',
                'first_axis  \t{0:1d}\n\n'.format(self.first_axis),
                '#Second Axis\n',
                'second_axis \t{0:1d}\n\n'.format(self.second_axis),
                '#Flags\n',
                'subtract_voltage_offset \t{0:d}\n'.format(
                    self.subtract_voltage_offset),
                'save_voltage       \t{0:d}\n'.format(self.save_voltage),
                'save_current       \t{0:d}\n'.format(self.save_current),
                'save_temperature   \t{0:d}\n'.format(self.save_temperature),
                'automatic_ramp     \t{0:d}\n\n'.format(self.automatic_ramp),
                '#Comments\n',
                'comments \t{0:s}\n\n'.format(comments),
                '#Axis Parameters (StartPos, EndPos, Step, Extra, Velocity)\n',
                'start_ax1   \t{0:4f}\n'.format(self.start_ax1),
                'end_ax1     \t{0:4f}\n'.format(self.end_ax1),
                'step_ax1    \t{0:2f}\n'.format(self.step_ax1),
                'extra_ax1   \t{0:2f}\n'.format(self.extra_ax1),
                'vel_ax1     \t{0:2f}\n\n'.format(self.vel_ax1),
                'start_ax2   \t{0:4f}\n'.format(self.start_ax2),
                'end_ax2     \t{0:4f}\n'.format(self.end_ax2),
                'step_ax2    \t{0:2f}\n'.format(self.step_ax2),
                'extra_ax2   \t{0:2f}\n'.format(self.extra_ax2),
                'vel_ax2     \t{0:2f}\n\n'.format(self.vel_ax2),
                'start_ax3   \t{0:4f}\n'.format(self.start_ax3),
                'end_ax3     \t{0:4f}\n'.format(self.end_ax3),
                'step_ax3    \t{0:2f}\n'.format(self.step_ax3),
                'extra_ax3   \t{0:2f}\n'.format(self.extra_ax3),
                'vel_ax3     \t{0:2f}\n\n'.format(self.vel_ax3),
                'start_ax5   \t{0:4f}\n'.format(self.start_ax5),
                'end_ax5     \t{0:4f}\n'.format(self.end_ax5),
                'step_ax5    \t{0:2f}\n'.format(self.step_ax5),
                'extra_ax5   \t{0:2f}\n'.format(self.extra_ax5),
                'vel_ax5     \t{0:2f}\n'.format(self.vel_ax5)]

            with open(filename, mode='w') as f:
                for item in data:
                    f.write(item)

        except Exception:
            message = 'Failed to save configuration to file: "%s"' % filename
            raise ConfigurationError(message)

    def set_end(self, axis, value):
        """Set end position value for the given axis."""
        param = 'end_ax'
        self._set_axis_param(param, axis, value)

    def set_extra(self, axis, value):
        """Get extra position for the given axis."""
        param = 'extra_ax'
        self._set_axis_param(param, axis, value)

    def set_start(self, axis, value):
        """Set start position value for the given axis."""
        param = 'start_ax'
        self._set_axis_param(param, axis, value)

    def set_step(self, axis, value):
        """Set position step value for the given axis."""
        param = 'step_ax'
        self._set_axis_param(param, axis, value)

    def set_velocity(self, axis, value):
        """Set velocity value for the given axis."""
        param = 'vel_ax'
        self._set_axis_param(param, axis, value)

    def valid_data(self):
        """Check if parameters are valid."""
        return super().valid_data(valid_none=[
            'current_setpoint', 'comments', 'probe_name'])


class PowerSupplyConfig(Configuration):
    """Read, write and store Power Supply configuration data."""

    _label = 'PowerSupply'
    _db_table = 'power_supply'
    _db_dict = _collections.OrderedDict([
        ('id', [None, 'INTEGER NOT NULL']),
        ('name', ['ps_name', 'TEXT NOT NULL UNIQUE']),
        ('type', ['ps_type', 'INTEGER NOT NULL']),
        ('dclink', ['dclink', 'REAL']),
        ('setpoint', ['ps_setpoint', 'REAL NOT NULL']),
        ('maximum current', ['maximum_current', 'REAL NOT NULL']),
        ('minimum current', ['minimum_current', 'REAL NOT NULL']),
        ('DCCT Head', ['dcct_head', 'INTEGER NOT NULL']),
        ('Kp', ['Kp', 'REAL NOT NULL']),
        ('Ki', ['Ki', 'REAL NOT NULL']),
        ('current array', ['current_array', 'TEXT']),
        ('trapezoidal array', ['trapezoidal_array', 'TEXT']),
        ('sinusoidal amplitude', ['sinusoidal_amplitude', 'REAL NOT NULL']),
        ('sinusoidal offset', ['sinusoidal_offset', 'REAL NOT NULL']),
        ('sinusoidal frequency', ['sinusoidal_frequency', 'REAL NOT NULL']),
        ('sinusoidal n cycles', ['sinusoidal_ncycles', 'INTEGER NOT NULL']),
        ('sinusoidal initial phase', ['sinusoidal_phasei', 'REAL NOT NULL']),
        ('sinusoidal final phase', ['sinusoidal_phasef', 'REAL NOT NULL']),
        ('damped sinusoidal amplitude', ['dsinusoidal_amplitude',
                                         'REAL NOT NULL']),
        ('damped sinusoidal offset', ['dsinusoidal_offset', 'REAL NOT NULL']),
        ('damped sinusoidal frequency', ['dsinusoidal_frequency',
                                         'REAL NOT NULL']),
        ('damped sinusoidal n cycles', ['dsinusoidal_ncycles',
                                        'INTEGER NOT NULL']),
        ('damped sinusoidal initial phase', ['dsinusoidal_phasei',
                                             'REAL NOT NULL']),
        ('damped sinusoidal final phase', ['dsinusoidal_phasef',
                                           'REAL NOT NULL']),
        ('damped sinusoidal damping', ['dsinusoidal_damp', 'REAL NOT NULL']),
        ('damped sinusoidal2 amplitude', ['dsinusoidal2_amplitude',
                                          'REAL NOT NULL']),
        ('damped sinusoidal2 offset', ['dsinusoidal2_offset',
                                       'REAL NOT NULL']),
        ('damped sinusoidal2 frequency', ['dsinusoidal2_frequency',
                                          'REAL NOT NULL']),
        ('damped sinusoidal2 n cycles', ['dsinusoidal2_ncycles',
                                         'INTEGER NOT NULL']),
        ('damped sinusoidal2 initial phase', ['dsinusoidal2_phasei',
                                              'REAL NOT NULL']),
        ('damped sinusoidal2 final phase', ['dsinusoidal2_phasef',
                                            'REAL NOT NULL']),
        ('damped sinusoidal2 damping', ['dsinusoidal2_damp', 'REAL NOT NULL']),
    ])
    _db_json_str = ['current_array']

    def __init__(self, filename=None, database=None, idn=None):
        """Initialize object.

        Args:
            serial_drs(SerialDRS_FBP): power supply serial class instance
            database (str): database file path.
            idn (int): id in database table.
        """
        # Power supply status (False = off, True = on)
        self.status = False
        # Power supply loop status (False = open, True = closed)
        self.status_loop = False
        # Power supply connection status (False = no communication)
        self.status_con = False
        # Power supply interlock status (True = active, False = not active)
        self.status_interlock = False
        # DC link voltage (90V is the default)
        self.dclink = 90
        # True for DCCT enabled, False for DCCT disabled
        self.dcct = False
        # Main current
        self.main_current = 0
        # Flag to enable or disable display update
        self.update_display = True

        # database variables
        self.current_array = None
        self.trapezoidal_array = None
        self.ps_name = None
        self.ps_type = None
        self.ps_setpoint = None
        self.maximum_current = None
        self.minimum_current = None
        self.Kp = None
        self.Ki = None
        self.dcct_head = None
        self.sinusoidal_amplitude = None
        self.sinusoidal_offset = None
        self.sinusoidal_frequency = None
        self.sinusoidal_ncycles = None
        self.sinusoidal_phasei = None
        self.sinusoidal_phasef = None
        self.dsinusoidal_amplitude = None
        self.dsinusoidal_offset = None
        self.dsinusoidal_frequency = None
        self.dsinusoidal_ncycles = None
        self.dsinusoidal_phasei = None
        self.dsinusoidal_phasef = None
        self.dsinusoidal_damp = None
        self.dsinusoidal2_amplitude = None
        self.dsinusoidal2_offset = None
        self.dsinusoidal2_frequency = None
        self.dsinusoidal2_ncycles = None
        self.dsinusoidal2_phasei = None
        self.dsinusoidal2_phasef = None
        self.dsinusoidal2_damp = None
        super().__init__(filename=filename, database=database, idn=idn)

    def get_attribute_type(self, name):
        """Get attribute type."""
        if name in ['ps_type', 'dcct_head', 'status', 'status_loop', 'dcct',
                    'sinusoidal_ncycles', 'dsinusoidal_ncycles']:
            return int
        elif name in ['ps_name']:
            return str
        elif name in ['current_array']:
            return _np.ndarray
        else:
            return float

    def save_file(self, filename):
        """Save measurement configuration to file.

        Args:
            filename (str): configuration filepath.

        Raise:
            ConfigurationError: if the configuration was not saved.
        """
        if not self.valid_data():
            message = 'Invalid Configuration.'
            raise ConfigurationError(message)

        try:
            data = [
                '# Power Supply Settings\n\n',
                'ps_name          \t{0:s}\n'.format(self.ps_name),
                'ps_type          \t{0}\n'.format(self.ps_type),
                'current_setpoint \t{0:2f}\n'.format(self.ps_setpoint),
                'minimum_current  \t{0:2f}\n'.format(self.minimum_current),
                'maximum_current  \t{0:2f}\n'.format(self.maximum_current),
                'Kp               \t{0:2f}\n'.format(self.Kp),
                'Ki               \t{0:2f}\n\n'.format(self.Ki),
                '# Sinusoidal Signal Generator\n',
                'sinusoidal_amplitude  \t{0:2f}\n'.format(
                    self.sinusoidal_amplitude),
                'sinusoidal_offset     \t{0:2f}\n'.format(
                    self.sinusoidal_offset),
                'sinusoidal_frequency  \t{0:2f}\n'.format(
                    self.sinusoidal_frequency),
                'sinusoidal_ncycles    \t{0:d}\n'.format(
                    self.sinusoidal_ncycles),
                'sinusoidal_phasei     \t{0:2f}\n'.format(
                    self.sinusoidal_phasei),
                'sinusoidal_phasef     \t{0:2f}\n\n'.format(
                    self.sinusoidal_phasef),
                '# Damped Sinusoidal Signal Generator\n',
                'dsinusoidal_amplitude \t{0:2f}\n'.format(
                    self.dsinusoidal_amplitude),
                'dsinusoidal_offset    \t{0:2f}\n'.format(
                    self.dsinusoidal_offset),
                'dsinusoidal_frequency \t{0:2f}\n'.format(
                    self.dsinusoidal_frequency),
                'dsinusoidal_ncycles   \t{0:d}\n'.format(
                    self.dsinusoidal_ncycles),
                'dsinusoidal_phasei    \t{0:2f}\n'.format(
                    self.dsinusoidal_phasei),
                'dsinusoidal_phasef    \t{0:2f}\n'.format(
                    self.dsinusoidal_phasef),
                'dsinusoidal_damp      \t{0:2f}\n\n'.format(
                    self.dsinusoidal_damp),
                '# Damped Sinusoidal^2 Signal Generator\n',
                'dsinusoidal2_amplitude \t{0:2f}\n'.format(
                    self.dsinusoidal2_amplitude),
                'dsinusoidal2_offset    \t{0:2f}\n'.format(
                    self.dsinusoidal2_offset),
                'dsinusoidal2_frequency \t{0:2f}\n'.format(
                    self.dsinusoidal2_frequency),
                'dsinusoidal2_ncycles   \t{0:d}\n'.format(
                    self.dsinusoidal2_ncycles),
                'dsinusoidal2_phasei    \t{0:2f}\n'.format(
                    self.dsinusoidal2_phasei),
                'dsinusoidal2_phasef    \t{0:2f}\n'.format(
                    self.dsinusoidal2_phasef),
                'dsinusoidal2_damp      \t{0:2f}\n\n'.format(
                    self.dsinusoidal2_damp),
                '#DCCT Settings\n',
                'dcct                  \t{0}\n'.format(int(self.dcct)),
                'dcct_head             \t{0}\n\n'.format(self.dcct_head),
                '#Automatic current ramp\n'
                'current_array         \t{0}'.format(
                    str(self.current_array)[1:-1])
                ]

            with open(filename, mode='w') as f:
                for item in data:
                    f.write(item)

        except Exception:
            message = 'Failed to save configuration to file: "%s"' % filename
            raise ConfigurationError(message)
