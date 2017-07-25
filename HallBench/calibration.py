# -*- coding: utf-8 -*-
"""Implementation of classes to handle calibration files."""

import numpy as _np


class CalibrationDataError(Exception):
    """Calibration data exception."""

    def __init__(self, message, *args):
        """Initialize variables."""
        self.message = message


class CalibrationData(object):
    """Hall probe calibration data."""

    def __init__(self, filename=None):
        """Initialize variables.

        Args:
            filename (str): calibration file path.
        """
        if filename is not None:
            self.read_calibration_file(filename)
        else:
            self.filename = None
            self.shift_x_to_y = -10
            self.shift_z_to_y = 10

    def read_calibration_file(self, filename):
        """Read calibration parameters from file.

        Args:
            filename (str): calibration file path.
        """
        pass

    def convert_probe_x(self, voltage_array):
        """Convert voltage values to magnetic field values for probe x.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        return _hall_probe_calibration_curve(voltage_array)

    def convert_probe_y(self, voltage_array):
        """Convert voltage values to magnetic field values for probe y.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        return _hall_probe_calibration_curve(voltage_array)

    def convert_probe_z(self, voltage_array):
        """Convert voltage values to magnetic field values for probe z.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        return _hall_probe_calibration_curve(voltage_array)


def _hall_probe_calibration_curve(voltage_array):
    """Convert voltage values to magnetic field values.

    Args:
        voltage_array (array): array with voltage values.

    Returns:
        array with magnetic field values.
    """
    field_array = _np.zeros(len(voltage_array))

    for i in range(len(voltage_array)):
        voltage = voltage_array[i]

        if (voltage > -10) and (voltage < 10):
            field = float(voltage)*0.2
        elif voltage <= -10:
            field = (
                -1.8216 +
                (-0.70592*voltage) +
                (-0.047964*voltage**2) +
                (-0.0015304*voltage**3))*(-1)
        else:
            field = (
                2.3614 +
                (-0.82643*voltage) +
                (0.056814*voltage**2) +
                (-0.0017429*voltage**3))*(-1)

        field_array[i] = field

    return field_array
