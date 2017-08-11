# -*- coding: utf-8 -*-
"""Implementation of classes to handle configuration files."""

from . import utils as _utils


class ConfigurationError(Exception):
    """Configuration exception."""

    def __init__(self, message, *args):
        """Initialize variables."""
        self.message = message


class DevicesConfig(object):
    """Read, write and stored devices configuration data."""

    def __init__(self, filename=None):
        """Initialize variables.

        Args:
            filename (str): devices configuration file path.
        """
        if filename is not None:
            self.read_file(filename)
        else:
            self._control_pmac_enable = None
            self._control_voltx_enable = None
            self._control_volty_enable = None
            self._control_voltz_enable = None
            self._control_multich_enable = None
            self._control_colimator_enable = None
            self._control_voltx_addr = None
            self._control_volty_addr = None
            self._control_voltz_addr = None
            self._control_multich_addr = None
            self._control_colimator_addr = None

    def __eq__(self, other):
        """Equality method."""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    @property
    def control_pmac_enable(self):
        """Pmac enable."""
        return self._control_pmac_enable

    @control_pmac_enable.setter
    def control_pmac_enable(self, value):
        if value in [0, 1]:
            self._control_pmac_enable = value
        else:
            raise ConfigurationError('Invalid value for control_pmac_enable.')

    @property
    def control_voltx_enable(self):
        """Voltimeter X enable."""
        return self._control_voltx_enable

    @control_voltx_enable.setter
    def control_voltx_enable(self, value):
        if value in [0, 1]:
            self._control_voltx_enable = value
        else:
            raise ConfigurationError('Invalid value for control_voltx_enable.')

    @property
    def control_volty_enable(self):
        """Voltimeter Y enable."""
        return self._control_volty_enable

    @control_volty_enable.setter
    def control_volty_enable(self, value):
        if value in [0, 1]:
            self._control_volty_enable = value
        else:
            raise ConfigurationError('Invalid value for control_volty_enable.')

    @property
    def control_voltz_enable(self):
        """Voltimeter Z enable."""
        return self._control_voltz_enable

    @control_voltz_enable.setter
    def control_voltz_enable(self, value):
        if value in [0, 1]:
            self._control_voltz_enable = value
        else:
            raise ConfigurationError('Invalid value for control_voltz_enable.')

    @property
    def control_multich_enable(self):
        """Multichannel enable."""
        return self._control_multich_enable

    @control_multich_enable.setter
    def control_multich_enable(self, value):
        if value in [0, 1]:
            self._control_multich_enable = value
        else:
            raise ConfigurationError(
                'Invalid value for control_multich_enable.')

    @property
    def control_colimator_enable(self):
        """Auto-colimator enable."""
        return self._control_colimator_enable

    @control_colimator_enable.setter
    def control_colimator_enable(self, value):
        if value in [0, 1]:
            self._control_colimator_enable = value
        else:
            raise ConfigurationError(
                'Invalid value for control_colimator_enable.')

    @property
    def control_voltx_addr(self):
        """Voltimeter X address."""
        return self._control_voltx_addr

    @control_voltx_addr.setter
    def control_voltx_addr(self, value):
        if isinstance(value, int):
            self._control_voltx_addr = value
        else:
            raise ConfigurationError('Invalid value for control_voltx_addr.')

    @property
    def control_volty_addr(self):
        """Voltimeter Y address."""
        return self._control_volty_addr

    @control_volty_addr.setter
    def control_volty_addr(self, value):
        if isinstance(value, int):
            self._control_volty_addr = value
        else:
            raise ConfigurationError('Invalid value for control_volty_addr.')

    @property
    def control_voltz_addr(self):
        """Voltimeter Z address."""
        return self._control_voltz_addr

    @control_voltz_addr.setter
    def control_voltz_addr(self, value):
        if isinstance(value, int):
            self._control_voltz_addr = value
        else:
            raise ConfigurationError('Invalid value for control_voltz_addr.')

    @property
    def control_multich_addr(self):
        """Multichannel address."""
        return self._control_multich_addr

    @control_multich_addr.setter
    def control_multich_addr(self, value):
        if isinstance(value, int):
            self._control_multich_addr = value
        else:
            raise ConfigurationError('Invalid value for control_multich_addr.')

    @property
    def control_colimator_addr(self):
        """Auto-colimator address."""
        return self._control_colimator_addr

    @control_colimator_addr.setter
    def control_colimator_addr(self, value):
        if isinstance(value, int):
            self._control_colimator_addr = value
        else:
            raise ConfigurationError(
                'Invalid value for control_colimator_addr.')

    def read_file(self, filename):
        """Read devices configuration from file.

        Args:
            filename (str): configuration file path.
        """
        data = _utils.read_file(filename)

        self.control_pmac_enable = _utils.find_value(
            data, 'control_pmac_enable', vtype='int')
        self.control_voltx_enable = _utils.find_value(
            data, 'control_voltx_enable', vtype='int')
        self.control_volty_enable = _utils.find_value(
            data, 'control_volty_enable', vtype='int')
        self.control_voltz_enable = _utils.find_value(
            data, 'control_voltz_enable', vtype='int')
        self.control_multich_enable = _utils.find_value(
            data, 'control_multich_enable', vtype='int')
        self.control_colimator_enable = _utils.find_value(
            data, 'control_colimator_enable', vtype='int')
        self.control_voltx_addr = _utils.find_value(
            data, 'control_voltx_addr', vtype='int')
        self.control_volty_addr = _utils.find_value(
            data, 'control_volty_addr', vtype='int')
        self.control_voltz_addr = _utils.find_value(
            data, 'control_voltz_addr', vtype='int')
        self.control_multich_addr = _utils.find_value(
            data, 'control_multich_addr', vtype='int')
        self.control_colimator_addr = _utils.find_value(
            data, 'control_colimator_addr', vtype='int')

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

    def clear(self):
        """Clear devices configuration."""
        self._control_pmac_enable = None
        self._control_voltx_enable = None
        self._control_volty_enable = None
        self._control_voltz_enable = None
        self._control_multich_enable = None
        self._control_colimator_enable = None
        self._control_voltx_addr = None
        self._control_volty_addr = None
        self._control_voltz_addr = None
        self._control_multich_addr = None
        self._control_colimator_addr = None

    def save_file(self, filename):
        """Save devices configuration to file.

        Args:
            filename (str): configuration file path.

        Raises:
            ConfigurationError: if the configuration was not saved.
        """
        if not self.valid_configuration():
            message = 'Invalid Configuration'
            raise ConfigurationError(message)

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
            raise ConfigurationError(message)


class MeasurementConfig(object):
    """Read, write and stored measurement configuration data."""

    def __init__(self, filename=None):
        """Initialize variables.

        Args:
            filename (str): measurement configuration file path.
        """
        if filename is not None:
            self.read_file(filename)
        else:
            self._meas_probeX = None
            self._meas_probeY = None
            self._meas_probeZ = None
            self._meas_precision = None
            self._meas_trig_axis = None
            self.meas_aper_ms = None

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

    def __eq__(self, other):
        """Equality method."""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    @property
    def meas_probeX(self):
        """Measure Hall-probe X voltage."""
        return self._meas_probeX

    @meas_probeX.setter
    def meas_probeX(self, value):
        if value in [0, 1]:
            self._meas_probeX = value
        else:
            raise ConfigurationError('Invalid value for meas_probeX')

    @property
    def meas_probeY(self):
        """Measure Hall-probe Y voltage."""
        return self._meas_probeY

    @meas_probeY.setter
    def meas_probeY(self, value):
        if value in [0, 1]:
            self._meas_probeY = value
        else:
            raise ConfigurationError('Invalid value for meas_probeY')

    @property
    def meas_probeZ(self):
        """Measure Hall-probe Z voltage."""
        return self._meas_probeZ

    @meas_probeZ.setter
    def meas_probeZ(self, value):
        if value in [0, 1]:
            self._meas_probeZ = value
        else:
            raise ConfigurationError('Invalid value for meas_probeZ')

    @property
    def meas_precision(self):
        """Digital multimeter precision [single=0 or double=1]."""
        return self._meas_precision

    @meas_precision.setter
    def meas_precision(self, value):
        if value in [0, 1]:
            self._meas_precision = value
        else:
            raise ConfigurationError('Invalid value for meas_precision')

    @property
    def meas_trig_axis(self):
        """Triggering Axis."""
        return self._meas_trig_axis

    @meas_trig_axis.setter
    def meas_trig_axis(self, value):
        if value in [1, 2, 3, 5]:
            self._meas_trig_axis = value
        else:
            raise ConfigurationError('Invalid value for meas_trig_axis')

    def read_file(self, filename):
        """Read measurement configuration from file.

        Args:
            filename (str): configuration file path.
        """
        data = _utils.read_file(filename)

        self.meas_probeX = _utils.find_value(data, 'meas_probeX', vtype='int')
        self.meas_probeY = _utils.find_value(data, 'meas_probeY', vtype='int')
        self.meas_probeZ = _utils.find_value(data, 'meas_probeZ', vtype='int')

        self.meas_aper_ms = _utils.find_value(
            data, 'meas_aper_ms', vtype='float')
        self.meas_precision = _utils.find_value(
            data, 'meas_precision', vtype='int')
        self.meas_trig_axis = _utils.find_value(
            data, 'meas_trig_axis', vtype='int')

        # Ax1, Ax2, Ax3, Ax5
        axis_measurement = [1, 2, 3, 5]
        for axis in axis_measurement:
            spos = 'meas_startpos_ax'+str(axis)
            epos = 'meas_endpos_ax' + str(axis)
            incr = 'meas_incr_ax' + str(axis)
            vel = 'meas_vel_ax' + str(axis)
            setattr(self, spos, _utils.find_value(data, spos, vtype='float'))
            setattr(self, epos, _utils.find_value(data, epos, vtype='float'))
            setattr(self, incr, _utils.find_value(data, incr, vtype='float'))
            setattr(self, vel, _utils.find_value(data, vel, vtype='float'))

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

    def clear(self):
        """Clear measurement configuration."""
        self._meas_probeX = None
        self._meas_probeY = None
        self._meas_probeZ = None
        self._meas_precision = None
        self._meas_trig_axis = None
        self.meas_aper_ms = None

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

    def save_file(self, filename):
        """Save measurement configuration to file.

        Args:
            filename (str): configuration file path.

        Raises:
            ConfigurationError: if the configuration was not saved.
        """
        if not self.valid_configuration():
            message = 'Invalid Configuration'
            raise ConfigurationError(message)

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
            raise ConfigurationError(message)
