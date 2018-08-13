# -*- coding: utf-8 -*-

"""Implementation of classes to handle configuration files."""

import collections as _collections

from . import utils as _utils
from . import database as _database
from hallbench import __version__


class ConfigurationError(Exception):
    """Configuration exception."""

    def __init__(self, message, *args):
        """Initialization method."""
        self.message = message


class Configuration(object):
    """Base class for configurations."""

    def __init__(self, filename=None):
        """Initialization method.

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

    def clear(self):
        """Clear configuration."""
        for key in self.__dict__:
            self.__dict__[key] = None

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

    def __init__(self, filename=None):
        """Initialization method.

        Args:
            filename (str): connection configuration filepath.
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
        self.collimator_enable = None
        self.collimator_port = None
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

        Raises:
            ConfigurationError: if the configuration was not saved.
        """
        if not self.valid_data():
            message = 'Invalid Configuration.'
            raise ConfigurationError(message)

        try:
            data = [
                '# Configuration File\n\n',
                'pmac_enable\t{0:1d}\n\n'.format(self.pmac_enable),
                'voltx_enable\t{0:1d}\n'.format(self.voltx_enable),
                'voltx_address\t{0:1d}\n'.format(self.voltx_address),
                'volty_enable\t{0:1d}\n'.format(self.volty_enable),
                'volty_address\t{0:1d}\n'.format(self.volty_address),
                'voltz_enable\t{0:1d}\n\n'.format(self.voltz_enable),
                'voltz_address\t{0:1d}\n\n'.format(self.voltz_address),
                'multich_enable\t{0:1d}\n'.format(self.multich_enable),
                'multich_address\t{0:1d}\n\n'.format(self.multich_address),
                'nmr_enable\t{0:1d}\n'.format(self.nmr_enable),
                'nmr_port\t{0:s}\n'.format(self.nmr_port),
                'nmr_baudrate\t{0:1d}\n'.format(self.nmr_baudrate),
                'collimator_enable\t{0:1d}\n\n'.format(
                    self.collimator_enable),
                'collimator_port\t{0:s}\n'.format(self.collimator_port),
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
        """Initialization method.

        Args:
            filename (str): measurement configuration filepath.
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
    def create_database_table(cls, database):
        """Create database table."""
        variables = []
        for key in cls._db_dict.keys():
            variables.append((key, cls._db_dict[key][1]))
        success = _database.create_table(database, cls._db_table, variables)
        return success

    @classmethod
    def database_table_name(cls):
        """Return the database table name."""
        return cls._db_table

    @classmethod
    def get_probe_name_from_database(cls, database, idn):
        """Return the probe name of the database record."""
        if len(cls._db_table) == 0:
            return None

        db_column_names = _database.get_table_column_names(
            database, cls._db_table)
        if len(db_column_names) == 0:
            return None

        db_entry = _database.read_from_database(database, cls._db_table, idn)
        if db_entry is None:
            return None

        idx = db_column_names.index('probe_name')
        probe_name = db_entry[idx]
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

    def read_from_database(self, database, idn):
        """Read field data from database entry."""
        db_column_names = _database.get_table_column_names(
            database, self._db_table)
        if len(db_column_names) == 0:
            raise ConfigurationError(
                'Failed to read configuration from database.')

        db_entry = _database.read_from_database(database, self._db_table, idn)

        if db_entry is None:
            raise ValueError('Invalid database ID.')

        for key in self._db_dict.keys():
            attr_name = self._db_dict[key][0]
            if key not in db_column_names:
                raise ConfigurationError(
                    'Failed to read configuration from database.')
            else:
                if attr_name is not None:
                    idx = db_column_names.index(key)
                    setattr(self, attr_name, db_entry[idx])

    def save_file(self, filename):
        """Save measurement configuration to file.

        Args:
            filename (str): configuration filepath.

        Raises:
            ConfigurationError: if the configuration was not saved.
        """
        if not self.valid_data():
            message = 'Invalid Configuration.'
            raise ConfigurationError(message)

        try:
            data = [
                '# Measurement Setup\n\n',
                'magnet_name\t{0:s}\n'.format(self.magnet_name),
                'main_current\t{0:s}\n'.format(self.main_current),
                'temperature\t{0:s}\n'.format(self.temperature),
                'operator\t{0:s}\n\n'.format(self.operator),
                '# Probe Calibration\n',
                'probe_name\t{0:s}\n\n'.format(self.probe_name),
                '# Digital Multimeters (X, Y, Z)\n',
                'voltx_enable\t{0:1d}\n'.format(self.voltx_enable),
                'volty_enable\t{0:1d}\n'.format(self.volty_enable),
                'voltz_enable\t{0:1d}\n\n'.format(self.voltz_enable),
                '# Digital Multimeter (aper [s])\n',
                'integration_time\t{0:4f}\n\n'.format(self.integration_time),
                '# Digital Multimeter (precision [single=0 or double=1])\n',
                'voltage_precision\t{0:1d}\n\n'.format(self.voltage_precision),
                '# Number of measurements\n',
                'nr_measurements\t{0:1d}\n\n'.format(self.nr_measurements),
                '# First Axis (Triggering Axis)\n',
                'first_axis\t{0:1d}\n\n'.format(self.first_axis),
                '#Second Axis\n',
                'second_axis\t{0:1d}\n\n'.format(self.second_axis),
                ('#Axis Parameters (StartPos, EndPos, Incr, Extra, Velocity)' +
                 ' - Ax1, Ax2, Ax3, Ax5\n'),
                'start_ax1  \t{0:4f}\n'.format(self.start_ax1),
                'end_ax1    \t{0:4f}\n'.format(self.end_ax1),
                'step_ax1   \t{0:2f}\n'.format(self.step_ax1),
                'extra_ax1  \t{0:2f}\n'.format(self.extra_ax1),
                'vel_ax1    \t{0:2f}\n\n'.format(self.vel_ax1),
                'start_ax2  \t{0:4f}\n'.format(self.start_ax2),
                'end_ax2    \t{0:4f}\n'.format(self.end_ax2),
                'step_ax2   \t{0:2f}\n'.format(self.step_ax2),
                'extra_ax2  \t{0:2f}\n'.format(self.extra_ax2),
                'vel_ax2    \t{0:2f}\n\n'.format(self.vel_ax2),
                'start_ax3  \t{0:4f}\n'.format(self.start_ax3),
                'end_ax3    \t{0:4f}\n'.format(self.end_ax3),
                'step_ax3   \t{0:2f}\n'.format(self.step_ax3),
                'extra_ax3  \t{0:2f}\n'.format(self.extra_ax3),
                'vel_ax3    \t{0:2f}\n\n'.format(self.vel_ax3),
                'start_ax5  \t{0:4f}\n'.format(self.start_ax5),
                'end_ax5    \t{0:4f}\n'.format(self.end_ax5),
                'step_ax5   \t{0:2f}\n'.format(self.step_ax5),
                'extra_ax5  \t{0:2f}\n'.format(self.extra_ax5),
                'vel_ax5    \t{0:2f}\n'.format(self.vel_ax5)]

            with open(filename, mode='w') as f:
                for item in data:
                    f.write(item)

        except Exception:
            message = 'Failed to save configuration to file: "%s"' % filename
            raise ConfigurationError(message)

    def save_to_database(self, database):
        """Insert field data into database table."""
        db_column_names = _database.get_table_column_names(
            database, self._db_table)
        if len(db_column_names) == 0:
            raise ConfigurationError(
                'Failed to save configuration to database.')
            return None

        timestamp = _utils.get_timestamp().split('_')
        date = timestamp[0]
        hour = timestamp[1].replace('-', ':')
        software_version = __version__

        db_values = []
        for key in self._db_dict.keys():
            attr_name = self._db_dict[key][0]
            if key not in db_column_names:
                raise ConfigurationError(
                    'Failed to save configuration to database.')
                return None
            else:
                if key == "id":
                    db_values.append(None)
                elif attr_name is None:
                    db_values.append(locals()[key])
                else:
                    db_values.append(getattr(self, attr_name))

        idn = _database.insert_into_database(
            database, self._db_table, db_values)
        return idn

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
