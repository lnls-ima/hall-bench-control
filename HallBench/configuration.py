# -*- coding: utf-8 -*-
"""Implementation of classes to handle configuration files."""

from HallBench import files as _files


class ControlConfiguration(object):
    """Read, write and stored control configuration data."""

    def __init__(self, filename=None):
        """Initialize variables.

        Args:
            filename (str): control configuration file path.
        """
        if filename is not None:
            self.read_file(filename)
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

    def read_file(self, filename):
        """Read control parameters from file.

        Args:
            filename (str): configuration file path.
        """
        data = _files.read_file(filename)

        self.control_pmac_enable = _files.find_value(
            data, 'control_pmac_enable', vtype='int')
        self.control_voltx_enable = _files.find_value(
            data, 'control_voltx_enable', vtype='int')
        self.control_volty_enable = _files.find_value(
            data, 'control_volty_enable', vtype='int')
        self.control_voltz_enable = _files.find_value(
            data, 'control_voltz_enable', vtype='int')
        self.control_multich_enable = _files.find_value(
            data, 'control_multich_enable', vtype='int')
        self.control_colimator_enable = _files.find_value(
            data, 'control_colimator_enable', vtype='int')
        self.control_voltx_addr = _files.find_value(
            data, 'control_voltx_addr', vtype='int')
        self.control_volty_addr = _files.find_value(
            data, 'control_volty_addr', vtype='int')
        self.control_voltz_addr = _files.find_value(
            data, 'control_voltz_addr', vtype='int')
        self.control_multich_addr = _files.find_value(
            data, 'control_multich_addr', vtype='int')
        self.control_colimator_addr = _files.find_value(
            data, 'control_colimator_addr', vtype='int')
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

    def save_file(self, filename):
        """Save control parameters to file.

        Args:
            filename (str): configuration file path.

        Raises:
            HallBenchFileError: if the configuration was not saved.
        """
        if not self.valid_configuration():
            message = 'Invalid Configuration'
            raise _files.HallBenchFileError(message)

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
            raise _files.HallBenchFileError(message)


class MeasurementConfiguration(object):
    """Read, write and stored measurement configuration data."""

    def __init__(self, filename=None):
        """Initialize variables.

        Args:
            filename (str): measurement configuration file path.
        """
        if filename is not None:
            self.read_file(filename)
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

    def read_file(self, filename):
        """Read measurement parameters from file.

        Args:
            filename (str): configuration file path.
        """
        data = _files.read_file(filename)

        self.meas_probeX = _files.find_value(data, 'meas_probeX', vtype='int')
        self.meas_probeY = _files.find_value(data, 'meas_probeY', vtype='int')
        self.meas_probeZ = _files.find_value(data, 'meas_probeZ', vtype='int')

        self.meas_aper_ms = _files.find_value(
            data, 'meas_aper_ms', vtype='float')
        self.meas_precision = _files.find_value(
            data, 'meas_precision', vtype='int')
        self.meas_trig_axis = _files.find_value(
            data, 'meas_trig_axis', vtype='int')

        # Ax1, Ax2, Ax3, Ax5
        axis_measurement = [1, 2, 3, 5]
        for axis in axis_measurement:
            spos = 'meas_startpos_ax'+str(axis)
            epos = 'meas_endpos_ax' + str(axis)
            incr = 'meas_incr_ax' + str(axis)
            vel = 'meas_vel_ax' + str(axis)
            setattr(self, spos, _files.find_value(data, spos, vtype='float'))
            setattr(self, epos, _files.find_value(data, epos, vtype='float'))
            setattr(self, incr, _files.find_value(data, incr, vtype='float'))
            setattr(self, vel, _files.find_value(data, vel, vtype='float'))

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

    def save_file(self, filename):
        """Save measurement parameters to file.

        Args:
            filename (str): configuration file path.

        Raises:
            HallBenchFileError: if the configuration was not saved.
        """
        if not self.valid_configuration():
            message = 'Invalid Configuration'
            raise _files.HallBenchFileError(message)

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
            raise _files.HallBenchFileError(message)
