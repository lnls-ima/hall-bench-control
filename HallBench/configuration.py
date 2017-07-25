# -*- coding: utf-8 -*-
"""Implementation of classes to handle configuration files."""

import os as _os


class ConfigurationFileError(Exception):
    """Configuration file exception."""

    def __init__(self, message, *args):
        """Initialize variables."""
        self.message = message


class ControlConfiguration(object):
    """Read, write and stored control configuration data."""

    def __init__(self, filename=None):
        """Initialize variables.

        Args:
            filename (str): configuration file path.
        """
        if filename is not None:
            self.read_configuration_from_file(filename)
        else:
            self.filename = None
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

    def read_configuration_from_file(self, filename):
        """Read control parameters from file.

        Args:
            filename (str): configuration file path.
        """
        data = _read_configuration_file(filename)

        self.control_pmac_enable = _find_value(
            data, 'control_pmac_enable')
        self.control_voltx_enable = _find_value(
            data, 'control_voltx_enable')
        self.control_volty_enable = _find_value(
            data, 'control_volty_enable')
        self.control_voltz_enable = _find_value(
            data, 'control_voltz_enable')
        self.control_multich_enable = _find_value(
            data, 'control_multich_enable')
        self.control_colimator_enable = _find_value(
            data, 'control_colimator_enable')
        self.control_voltx_addr = _find_value(
            data, 'control_voltx_addr')
        self.control_volty_addr = _find_value(
            data, 'control_volty_addr')
        self.control_voltz_addr = _find_value(
            data, 'control_voltz_addr')
        self.control_multich_addr = _find_value(
            data, 'control_multich_addr')
        self.control_colimator_addr = _find_value(
            data, 'control_colimator_addr')
        self.filename = filename

    def valid_configuration(self):
        """Check if parameters are valid.

        Returns:
            True if successful, False otherwise.
        """
        atts = [getattr(self, a) for a in dir(self) if a.startswith('control')]
        if all([a is not None for a in atts]):
            return True
        else:
            return False

    def save_configuration_to_file(self, filename):
        """Save control parameters to file.

        Args:
            filename (str): configuration file path.
        """
        if not self.valid_configuration():
            raise ConfigurationFileError('Invalid Configuration')

        try:
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
            raise ConfigurationFileError(message)


class MeasurementConfiguration(object):
    """Read, write and stored measurement configuration data."""

    def __init__(self, filename=None):
        """Initialize variables.

        Args:
            filename (str): configuration file path.
        """
        if filename is not None:
            self.read_configuration_from_file(filename)
        else:
            self.filename = None
            self.meas_probeX = None
            self.meas_probeY = None
            self.meas_probeZ = None
            self.meas_aper_ms = None
            self.meas_precision = None
            self.meas_trig_axis = None

            # Ax1
            self.meas_startpos_ax1 = None
            self.meas_endpos_ax1 = None
            self.meas_incr_ax1 = None
            self.meas_vel_ax1 = None

            # Ax2
            self.meas_startpos_ax2 = None
            self.meas_endpos_ax2 = None
            self.meas_incr_ax2 = None
            self.meas_vel_ax2 = None

            # Ax3
            self.meas_startpos_ax3 = None
            self.meas_endpos_ax3 = None
            self.meas_incr_ax3 = None
            self.meas_vel_ax3 = None

            # Ax5
            self.meas_startpos_ax5 = None
            self.meas_endpos_ax5 = None
            self.meas_incr_ax5 = None
            self.meas_vel_ax5 = None

    def read_configuration_from_file(self, filename):
        """Read measurement parameters from file.

        Args:
            filename (str): configuration file path.
        """
        data = _read_configuration_file(filename)

        self.meas_probeX = _find_value(data, 'meas_probeX')
        self.meas_probeY = _find_value(data, 'meas_probeY')
        self.meas_probeZ = _find_value(data, 'meas_probeZ')

        self.meas_aper_ms = _find_value(data, 'meas_aper_ms', vtype='float')
        self.meas_precision = _find_value(data, 'meas_precision')
        self.meas_trig_axis = _find_value(data, 'meas_trig_axis')

        # Ax1, Ax2, Ax3, Ax5
        axis_measurement = [1, 2, 3, 5]
        for axis in axis_measurement:
            startpos = 'meas_startpos_ax'+str(axis)
            endpos = 'meas_endpos_ax' + str(axis)
            incr = 'meas_incr_ax' + str(axis)
            vel = 'meas_vel_ax' + str(axis)
            setattr(self, startpos, _find_value(data, startpos, vtype='float'))
            setattr(self, endpos, _find_value(data, endpos, vtype='float'))
            setattr(self, incr, _find_value(data, incr, vtype='float'))
            setattr(self, vel, _find_value(data, vel, vtype='float'))

        self.filename = filename

    def valid_configuration(self):
        """Check if parameters are valid.

        Returns:
            True if successful, False otherwise.
        """
        atts = [getattr(self, a) for a in dir(self) if a.startswith('meas')]
        if all([a is not None for a in atts]):
            return True
        else:
            return False

    def save_configuration_to_file(self, filename):
        """Save measurement parameters to file.

        Args:
            filename (str): configuration file path.
        """
        if not self.valid_configuration():
            raise ConfigurationFileError('Invalid Configuration')

        try:
            data = [
                'Measurement Setup\n\n',
                'Hall probes (X, Y, Z)\n',
                '##meas_probeX\t{0:1d}\n'.format(self.meas_probeX),
                '##meas_probeY\t{0:1d}\n'.format(self.meas_probeY),
                '##meas_probeZ\t{0:1d}\n\n'.format(self.meas_probeZ),
                'Digital Multimeter (aper [ms])\n',
                '#meas_aper_ms\t{0:4f}\n\n'.format(self.meas_aper_ms),
                'Digital Multimeter (precision [single=0 or double=1])\n',
                '#meas_precision\t{0:1d}\n\n'.format(self.meas_precision),
                'Triggering Axis\n',
                '#meas_trig_axis\t{0:1d}\n\n'.format(self.meas_trig_axis),
                ('Axis Parameters (StartPos, EndPos, Incr, Velocity) - ' +
                 'Ax1, Ax2, Ax3, Ax5\n'),
                '#meas_startpos_ax1\t{0:4f}\n'.format(self.meas_startpos_ax1),
                '#meas_endpos_ax1\t{0:4f}\n'.format(self.meas_endpos_ax1),
                '#meas_incr_ax1\t{0:2f}\n'.format(self.meas_incr_ax1),
                '#meas_vel_ax1\t{0:2f}\n\n'.format(self.meas_vel_ax1),
                '#meas_startpos_ax2\t{0:4f}\n'.format(self.meas_startpos_ax2),
                '#meas_endpos_ax2\t{0:4f}\n'.format(self.meas_endpos_ax2),
                '#meas_incr_ax2\t{0:2f}\n'.format(self.meas_incr_ax2),
                '#meas_vel_ax2\t{0:2f}\n\n'.format(self.meas_vel_ax2),
                '#meas_startpos_ax3\t{0:4f}\n'.format(self.meas_startpos_ax3),
                '#meas_endpos_ax3\t{0:4f}\n'.format(self.meas_endpos_ax3),
                '#meas_incr_ax3\t{0:2f}\n'.format(self.meas_incr_ax3),
                '#meas_vel_ax3\t{0:2f}\n\n'.format(self.meas_vel_ax3),
                '#meas_startpos_ax5\t{0:4f}\n'.format(self.meas_startpos_ax5),
                '#meas_endpos_ax5\t{0:4f}\n'.format(self.meas_endpos_ax5),
                '#meas_incr_ax5\t{0:2f}\n'.format(self.meas_incr_ax5),
                '#meas_vel_ax5\t{0:2f}\n'.format(self.meas_vel_ax5)]

            f = open(filename, mode='w')
            for item in data:
                f.write(item)
            f.close()

        except Exception:
            message = 'Failed to save configuration to file: "%s"' % filename
            raise ConfigurationFileError(message)


def _read_configuration_file(filename):
    if not _os.path.isfile(filename):
        message = 'File not found: "%s"' % filename
        raise ConfigurationFileError(message)

    if _os.stat(filename).st_size == 0:
        message = 'Empty file: "%s"' % filename
        raise ConfigurationFileError(message)

    try:
        f = open(filename, mode='r')
    except IOError:
        message = 'Failed to open file: "%s"' % filename
        raise ConfigurationFileError(message)

    fdata = f.read()
    f.close()
    data = [item for item in fdata.splitlines() if item.find('#') != -1]

    return data


def _find_value(data, variable, vtype='int'):
    value = next(
        (item.split('\t') for item in data if item.find(variable) != -1), None)
    try:
        value = value[1]
        if vtype == 'int':
            value = int(value)
        elif vtype == 'float':
            value = float(value)
    except Exception:
        message = 'Invalid value for "%s"' % variable
        raise ConfigurationFileError(message)
    return value
