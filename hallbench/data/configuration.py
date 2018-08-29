# -*- coding: utf-8 -*-

"""Implementation of classes to handle configuration files."""

import collections as _collections

from . import utils as _utils
from . import database as _database


class ConfigurationError(Exception):
    """Configuration exception."""

    def __init__(self, message, *args):
        """Initialize object."""
        self.message = message


class Configuration(_database.DatabaseObject):
    """Base class for configurations."""

    _db_table = ''
    _db_dict = {}
    _db_json_str = []

    def __init__(self, filename=None):
        """Initialize obejct.

        Args:
            filename (str): configuration filepath.
        """
        if filename is not None:
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
                value = _utils.find_value(data, name, vtype=tp)
                setattr(self, name, value)

    def save_file(self, filename):
        """Save configuration to file."""
        pass

    def valid_data(self):
        """Check if parameters are valid."""
        al = [getattr(self, a) for a in self.__dict__]
        if all([a is not None for a in al]):
            return True
        else:
            return False


class ConnectionConfig(Configuration):
    """Read, write and stored connection configuration data."""

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
        ('ps_port', ['ps_port', 'TEXT NOT NULL'])
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
        self.ps_port = None

        if filename is not None and idn is not None:
            raise ValueError('Invalid arguments for ConnectionConfig.')

        if idn is not None and database is not None:
            self.read_from_database(database, idn)
        else:
            super().__init__(filename)

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
                'pmac_enable      \t{0:1d}\n\n'.format(self.pmac_enable),
                'voltx_enable     \t{0:1d}\n'.format(self.voltx_enable),
                'voltx_address    \t{0:1d}\n'.format(self.voltx_address),
                'volty_enable     \t{0:1d}\n'.format(self.volty_enable),
                'volty_address    \t{0:1d}\n'.format(self.volty_address),
                'voltz_enable     \t{0:1d}\n\n'.format(self.voltz_enable),
                'voltz_address    \t{0:1d}\n\n'.format(self.voltz_address),
                'multich_enable   \t{0:1d}\n'.format(self.multich_enable),
                'multich_address  \t{0:1d}\n\n'.format(self.multich_address),
                'nmr_enable       \t{0:1d}\n'.format(self.nmr_enable),
                'nmr_port         \t{0:s}\n'.format(self.nmr_port),
                'nmr_baudrate     \t{0:1d}\n'.format(self.nmr_baudrate),
                'elcomat_enable   \t{0:1d}\n\n'.format(self.elcomat_enable),
                'elcomat_port     \t{0:s}\n'.format(self.elcomat_port),
                'elcomat_baudrate \t{0:1d}\n'.format(self.elcomat_baudrate),
                ]

            with open(filename, mode='w') as f:
                for item in data:
                    f.write(item)

        except Exception:
            message = 'Failed to save configuration to file: "%s"' % filename
            raise ConfigurationError(message)


class MeasurementConfig(Configuration):
    """Read, write and stored measurement configuration data."""

    _db_table = 'configurations'
    _db_dict = _collections.OrderedDict([
        ('id', [None, 'INTEGER NOT NULL']),
        ('date', [None, 'TEXT NOT NULL']),
        ('hour', [None, 'TEXT NOT NULL']),
        ('magnet_name', ['magnet_name', 'TEXT NOT NULL']),
        ('main_current', ['main_current', 'TEXT NOT NULL']),
        ('probe_name', ['probe_name', 'TEXT']),
        ('temperature', ['temperature', 'TEXT']),
        ('operator', ['operator', 'TEXT']),
        ('software_version', [None, 'TEXT']),
        ('voltx_enable', ['voltx_enable', 'INTEGER NOT NULL']),
        ('volty_enable', ['volty_enable', 'INTEGER NOT NULL']),
        ('voltz_enable', ['voltz_enable', 'INTEGER NOT NULL']),
        ('voltage_precision', ['voltage_precision', 'INTEGER NOT NULL']),
        ('nr_measurements', ['nr_measurements', 'INTEGER NOT NULL']),
        ('integration_time', ['integration_time', 'REAL NOT NULL']),
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
    ])

    def __init__(self, filename=None, database=None, idn=None):
        """Initialize object.

        Args:
            filename (str): measurement configuration filepath.
            database (str): database file path.
            idn (int): id in database table.
        """
        self.magnet_name = None
        self.main_current = None
        self.probe_name = None
        self.temperature = None
        self.operator = None
        self.voltx_enable = None
        self.volty_enable = None
        self.voltz_enable = None
        self.integration_time = None
        self.voltage_precision = None
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

        if filename is not None and idn is not None:
            raise ValueError('Invalid arguments for MeasurementConfig.')

        if idn is not None and database is not None:
            self.read_from_database(database, idn)
        else:
            super().__init__(filename)

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
                    'first_axis', 'second_axis']:
            return int
        elif name in ['magnet_name', 'main_current', 'probe_name',
                      'temperature', 'operator']:
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
                '# Measurement Setup\n\n',
                'magnet_name \t{0:s}\n'.format(self.magnet_name),
                'main_current\t{0:s}\n'.format(self.main_current),
                'temperature \t{0:s}\n'.format(self.temperature),
                'operator    \t{0:s}\n\n'.format(self.operator),
                '# Hall Probe\n',
                'probe_name  \t{0:s}\n\n'.format(self.probe_name),
                '# Digital Multimeters (X, Y, Z)\n',
                'voltx_enable\t{0:1d}\n'.format(self.voltx_enable),
                'volty_enable\t{0:1d}\n'.format(self.volty_enable),
                'voltz_enable\t{0:1d}\n\n'.format(self.voltz_enable),
                '# Digital Multimeter (aper [s])\n',
                'integration_time \t{0:4f}\n\n'.format(self.integration_time),
                '# Digital Multimeter (precision [single=0 or double=1])\n',
                'voltage_precision\t{0:1d}\n\n'.format(self.voltage_precision),
                '# Number of measurements\n',
                'nr_measurements  \t{0:1d}\n\n'.format(self.nr_measurements),
                '# First Axis (Triggering Axis)\n',
                'first_axis  \t{0:1d}\n\n'.format(self.first_axis),
                '#Second Axis\n',
                'second_axis \t{0:1d}\n\n'.format(self.second_axis),
                ('#Axis Parameters (StartPos, EndPos, Incr, Extra, Velocity)' +
                 ' - Ax1, Ax2, Ax3, Ax5\n'),
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


class PowerSupplyConfig(Configuration):
    """Read, write and store Power Supply configuration data"""

    _db_table = 'power_supply'
    _db_dict = _collections.OrderedDict([
        ('id', [None, 'INTEGER NOT NULL']),
        ('name', ['ps_name', 'TEXT NOT NULL UNIQUE']),
        ('type', ['ps_type', 'INTEGER NOT NULL']),
        ('dclink', ['dclink', 'REAL']),
        ('setpoint', ['ps_setpoint', 'REAL NOT NULL']),
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
        ('maximum current', ['maximum_current', 'REAL NOT NULL']),
        ('minimum current', ['minimum_current', 'REAL NOT NULL']),
        ('Kp', ['Kp', 'REAL NOT NULL']),
        ('Ki', ['Ki', 'REAL NOT NULL']),
        ('DCCT Head', ['dcct_head', 'INTEGER NOT NULL']),
    ])

    def __init__(self):
        """Initialize object.

        Args:
            serial_drs(SerialDRS_FBP): power supply serial class instance
            database (str): database file path.
            idn (int): id in database table.
        """
        #Power supply status (False = off, True = on)
        self.status = False
        #Power supply loop status (False = open, True = closed)
        self.status_loop = False
        #DC link voltage (30V is the default)
        self.dclink = 30
        #True for DCCT enabled, False for DCCT disabled
        self.dcct = False
        #Main current
        self.main_current = 0

        #database variables
        self.ps_name = None
        self.ps_type = None
        self.ps_setpoint = None
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
        self.maximum_current = None
        self.minimum_current = None
        self.Kp = None
        self.Ki = None
        self.dcct_head = None

    def get_attribute_type(self, name):
        """Get attribute type."""
        if name in ['ps_type', 'dcct_head', 'status', 'status_loop', 'dcct',
                    'sinusoidal_ncycles', 'dsinusoidal_ncycles']:
            return int
        elif name in ['ps_name']:
            return str
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
                'amplitude        \t{0:2f}\n'.format(
                    self.sinusoidal_amplitude),
                'offset           \t{0:2f}\n'.format(self.sinusoidal_offset),
                'frequency        \t{0:2f}\n'.format(
                    self.sinusoidal_frequency),
                'n_cycles         \t{0:d}\n'.format(self.sinusoidal_ncycles),
                'initial_phase    \t{0:2f}\n'.format(self.sinusoidal_phasei),
                'final_phase      \t{0:2f}\n\n'.format(self.sinusoidal_phasef),
                '# Damped Sinusoidal Signal Generator\n',
                'amplitude        \t{0:2f}\n'.format(
                    self.dsinusoidal_amplitude),
                'offset           \t{0:2f}\n'.format(self.dsinusoidal_offset),
                'frequency        \t{0:2f}\n'.format(
                    self.dsinusoidal_frequency),
                'n_cycles         \t{0:d}\n'.format(self.dsinusoidal_ncycles),
                'initial_phase    \t{0:2f}\n'.format(self.dsinusoidal_phasei),
                'final_phase      \t{0:2f}\n'.format(self.dsinusoidal_phasef),
                'damping          \t{0:2f}\n\n'.format(self.dsinusoidal_damp),
                '#DCCT Settings\n',
                'dcct_enabled     \t{0}\n'.format(self.dcct),
                'dcct_head        \t{0}'.format(self.dcct_head)
                ]

            with open(filename, mode='w') as f:
                for item in data:
                    f.write(item)

        except Exception:
            message = 'Failed to save configuration to file: "%s"' % filename
            raise ConfigurationError(message)
