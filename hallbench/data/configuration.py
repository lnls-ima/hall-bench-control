# -*- coding: utf-8 -*-

"""Implementation of classes to handle configuration files."""

from . import utils as _utils


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
        self._filename = None
        if filename is not None:
            self.read_file(filename)

    @property
    def filename(self):
        """Name of the configuration file."""
        return self._filename

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

    def get_attribute_type(self, name):
        """Get attribute type."""
        return None

    def valid_configuration(self):
        """Check if parameters are valid."""
        al = [getattr(self, a) for a in self.__dict__ if a != '_filename']
        if all([a is not None for a in al]):
            return True
        else:
            return False

    def clear(self):
        """Clear configuration."""
        for key in self.__dict__:
            self.__dict__[key] = None

    def read_file(self, filename):
        """Read configuration from file.

        Args:
            filename (str): configuration filepath.
        """
        self._filename = filename
        data = _utils.read_file(filename)
        for name in self.__dict__:
            if name != '_filename':
                tp = self.get_attribute_type(name)
                if tp is not None:
                    value = _utils.find_value(data, name, vtype=tp)
                    setattr(self, name, value)

    def save_file(self, filename):
        """Save configuration to file."""
        pass


class ConnectionConfig(Configuration):
    """Read, write and stored connection configuration data."""

    def __init__(self, filename=None):
        """Initialization method.

        Args:
            filename (str): connection configuration filepath.
        """
        self.control_pmac_enable = None
        self.control_voltx_enable = None
        self.control_volty_enable = None
        self.control_voltz_enable = None
        self.control_multich_enable = None
        self.control_colimator_enable = None
        self.control_voltx_addr = None
        self.control_volty_addr = None
        self.control_voltz_addr = None
        self.control_multich_addr = None
        self.control_colimator_addr = None
        super().__init__(filename)

    def get_attribute_type(self, name):
        """Get attribute type."""
        if name == '_filename':
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
        if not self.valid_configuration():
            message = 'Invalid Configuration.'
            raise ConfigurationError(message)

        try:
            self._filename = filename
            data = [
                'Configuration File\n\n',
                '#control_pmac_enable\t{0:1d}\n\n'.format(
                    self.control_pmac_enable),
                '#control_voltx_enable\t{0:1d}\n'.format(
                    self.control_voltx_enable),
                '#control_volty_enable\t{0:1d}\n'.format(
                    self.control_volty_enable),
                '#control_voltz_enable\t{0:1d}\n\n'.format(
                    self.control_voltz_enable),
                '#control_multich_enable\t{0:1d}\n'.format(
                    self.control_multich_enable),
                '#control_colimator_enable\t{0:1d}\n\n'.format(
                    self.control_colimator_enable),
                '#control_voltx_addr\t{0:1d}\n'.format(
                    self.control_voltx_addr),
                '#control_volty_addr\t{0:1d}\n'.format(
                    self.control_volty_addr),
                '#control_voltz_addr\t{0:1d}\n\n'.format(
                    self.control_voltz_addr),
                '#control_multich_addr\t{0:1d}\n\n'.format(
                    self.control_multich_addr),
                '#control_colimator_addr\t{0:1d}\n'.format(
                    self.control_colimator_addr)]

            f = open(filename, mode='w')
            for item in data:
                f.write(item)
            f.close()

        except Exception:
            message = 'Failed to save configuration to file: "%s"' % filename
            raise ConfigurationError(message)


class MeasurementConfig(Configuration):
    """Read, write and stored measurement configuration data."""

    def __init__(self, filename=None):
        """Initialization method.

        Args:
            filename (str): measurement configuration filepath.
        """
        self.meas_probeX = None
        self.meas_probeY = None
        self.meas_probeZ = None
        self.meas_precision = None
        self.meas_first_axis = None
        self.meas_second_axis = None
        self.meas_aper = None
        self.meas_nr = None
        self.meas_startpos_ax1 = None
        self.meas_endpos_ax1 = None
        self.meas_incr_ax1 = None
        self.meas_extra_ax1 = None
        self.meas_vel_ax1 = None
        self.meas_startpos_ax2 = None
        self.meas_endpos_ax2 = None
        self.meas_incr_ax2 = None
        self.meas_extra_ax2 = None
        self.meas_vel_ax2 = None
        self.meas_startpos_ax3 = None
        self.meas_endpos_ax3 = None
        self.meas_incr_ax3 = None
        self.meas_extra_ax3 = None
        self.meas_vel_ax3 = None
        self.meas_startpos_ax5 = None
        self.meas_endpos_ax5 = None
        self.meas_incr_ax5 = None
        self.meas_extra_ax5 = None
        self.meas_vel_ax5 = None
        super().__init__(filename)

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

    def get_start(self, axis):
        """Get start position for the given axis."""
        return getattr(self, 'meas_startpos_ax' + str(axis))

    def set_start(self, axis, value):
        """Set start position value for the given axis."""
        param = 'meas_startpos_ax'
        self._set_axis_param(param, axis, value)

    def get_end(self, axis):
        """Get end position for the given axis."""
        return getattr(self, 'meas_endpos_ax' + str(axis))

    def set_end(self, axis, value):
        """Set end position value for the given axis."""
        param = 'meas_endpos_ax'
        self._set_axis_param(param, axis, value)

    def get_step(self, axis):
        """Get position step for the given axis."""
        return getattr(self, 'meas_incr_ax' + str(axis))

    def set_step(self, axis, value):
        """Set position step value for the given axis."""
        param = 'meas_incr_ax'
        self._set_axis_param(param, axis, value)

    def get_extra(self, axis):
        """Get extra position for the given axis."""
        return getattr(self, 'meas_extra_ax' + str(axis))

    def set_extra(self, axis, value):
        """Get extra position for the given axis."""
        param = 'meas_extra_ax'
        self._set_axis_param(param, axis, value)

    def get_velocity(self, axis):
        """Get velocity for the given axis."""
        return getattr(self, 'meas_vel_ax' + str(axis))

    def set_velocity(self, axis, value):
        """Set velocity value for the given axis."""
        param = 'meas_vel_ax'
        self._set_axis_param(param, axis, value)

    def get_attribute_type(self, name):
        """Get attribute type."""
        if name in ['meas_probeX', 'meas_probeY', 'meas_probeZ', 'meas_nr',
                    'meas_precision', 'meas_first_axis', 'meas_second_axis']:
            return int
        elif name == '_filename':
            return str
        else:
            return float

    def save_file(self, filename):
        """Save measurement configuration to file.

        Args:
            filename (str): configuration filepath.

        Raises:
            ConfigurationError: if the configuration was not saved.
        """
        if not self.valid_configuration():
            message = 'Invalid Configuration.'
            raise ConfigurationError(message)

        try:
            self._filename = filename
            data = [
                'Measurement Setup\n\n',
                'Hall probes (X, Y, Z)\n',
                '#meas_probeX\t{0:1d}\n'.format(self.meas_probeX),
                '#meas_probeY\t{0:1d}\n'.format(self.meas_probeY),
                '#meas_probeZ\t{0:1d}\n\n'.format(self.meas_probeZ),
                'Digital Multimeter (aper [s])\n',
                '#meas_aper\t{0:4f}\n\n'.format(self.meas_aper),
                'Digital Multimeter (precision [single=0 or double=1])\n',
                '#meas_precision\t{0:1d}\n\n'.format(self.meas_precision),
                'Number of measurements\n',
                '#meas_nr\t{0:1d}\n\n'.format(self.meas_nr),
                'First Axis (Triggering Axis)\n',
                '#meas_first_axis\t{0:1d}\n\n'.format(self.meas_first_axis),
                'Second Axis\n',
                '#meas_second_axis\t{0:1d}\n\n'.format(self.meas_second_axis),
                ('Axis Parameters (StartPos, EndPos, Incr, Extra, Velocity)' +
                 ' - Ax1, Ax2, Ax3, Ax5\n'),
                '#meas_startpos_ax1\t{0:4f}\n'.format(self.meas_startpos_ax1),
                '#meas_endpos_ax1\t{0:4f}\n'.format(self.meas_endpos_ax1),
                '#meas_incr_ax1\t{0:2f}\n'.format(self.meas_incr_ax1),
                '#meas_extra_ax1\t{0:2f}\n'.format(self.meas_extra_ax1),
                '#meas_vel_ax1\t{0:2f}\n\n'.format(self.meas_vel_ax1),
                '#meas_startpos_ax2\t{0:4f}\n'.format(self.meas_startpos_ax2),
                '#meas_endpos_ax2\t{0:4f}\n'.format(self.meas_endpos_ax2),
                '#meas_incr_ax2\t{0:2f}\n'.format(self.meas_incr_ax2),
                '#meas_extra_ax2\t{0:2f}\n'.format(self.meas_extra_ax2),
                '#meas_vel_ax2\t{0:2f}\n\n'.format(self.meas_vel_ax2),
                '#meas_startpos_ax3\t{0:4f}\n'.format(self.meas_startpos_ax3),
                '#meas_endpos_ax3\t{0:4f}\n'.format(self.meas_endpos_ax3),
                '#meas_incr_ax3\t{0:2f}\n'.format(self.meas_incr_ax3),
                '#meas_extra_ax3\t{0:2f}\n'.format(self.meas_extra_ax3),
                '#meas_vel_ax3\t{0:2f}\n\n'.format(self.meas_vel_ax3),
                '#meas_startpos_ax5\t{0:4f}\n'.format(self.meas_startpos_ax5),
                '#meas_endpos_ax5\t{0:4f}\n'.format(self.meas_endpos_ax5),
                '#meas_incr_ax5\t{0:2f}\n'.format(self.meas_incr_ax5),
                '#meas_extra_ax5\t{0:2f}\n'.format(self.meas_extra_ax5),
                '#meas_vel_ax5\t{0:2f}\n'.format(self.meas_vel_ax5)]

            f = open(filename, mode='w')
            for item in data:
                f.write(item)
            f.close()

        except Exception:
            message = 'Failed to save configuration to file: "%s"' % filename
            raise ConfigurationError(message)
