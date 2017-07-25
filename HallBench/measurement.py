# -*- coding: utf-8 -*-
"""Implementation of classes to store and analyse measurement data."""

import os as _os
import math as _math
import numpy as _np
from scipy import interpolate as _interpolate
from scipy.integrate import cumtrapz as _cumtrapz


class Measurement(object):
    """Position and Hall probe values of one measurement."""

    def __init__(self):
        """Initialize variables."""
        self.position = _np.array([])
        self.hallx = _np.array([])
        self.hally = _np.array([])
        self.hallz = _np.array([])

    @staticmethod
    def reverse(m):
        """Return the reverse of a Measurement."""
        reverse_m = Measurement()
        reverse_m.position = m.position[::-1]
        reverse_m.hallx = m.hallx[::-1]
        reverse_m.hally = m.hally[::-1]
        reverse_m.hallz = m.hallz[::-1]
        return reverse_m

    @staticmethod
    def copy(m):
        """Return a copy of a Measurement."""
        mc = Measurement()
        mc.position = _np.copy(m.position)
        mc.hallx = _np.copy(m.hallx)
        mc.hally = _np.copy(m.hally)
        mc.hallz = _np.copy(m.hallz)
        return mc

    def save_to_file(self, filename):
        """Save measurement data to file.

        Args:
            filename (str): measurement file path.
        """
        data = _np.column_stack((
            self.position, self.hallx, self.hally, self.hallz))
        _np.savetxt(filename, data, delimiter='\t', newline='\r\n')

    def read_from_file(self, filename):
        """Read measurement data from file.

        Args:
            filename (str): measurement file path.
        """
        data = _np.loadtxt(filename)
        self.position = data[:, 0]
        self.hallx = data[:, 1]
        self.hally = data[:, 2]
        self.hallz = data[:, 3]

    def clear(self):
        """Clear measurement."""
        self.position = _np.array([])
        self.hallx = _np.array([])
        self.hally = _np.array([])
        self.hallz = _np.array([])


class MeasurementList(object):
    """List of measurements."""

    def __init__(self, axis, position, calibration):
        """Initialize variables.

        Args:
            axis (int): number of the scan axis.
            position (array): array with position values for interpolation.
            calibration (CalibrationData): probe calibration data.
        """
        self.axis = axis
        self.position = position
        self.calibration = calibration

        self._raw = []
        self._interpolated = []
        self._average_voltage = Measurement()
        self._std_voltage = Measurement()
        self._average_field = Measurement()
        self._std_field = Measurement()
        self._first_integral = Measurement()
        self._second_integral = Measurement()

    @staticmethod
    def copy(ml):
        """Return a copy of a MeasurementList."""
        mlc = MeasurementList()
        mlc.axis = ml.axis
        mlc.position = ml.position
        mlc.calibration = ml.calibration
        mlc._raw = [Measurement().copy(m) for m in ml._raw]
        mlc._interpolated = [Measurement().copy(m) for m in ml._interpolated]
        mlc._average_voltage = Measurement().copy(ml._average_voltage)
        mlc._std_voltage = Measurement().copy(ml._std_voltage)
        mlc._average_field = Measurement().copy(ml._average_field)
        mlc._std_field = Measurement().copy(ml._std_field)
        mlc._first_integral = Measurement().copy(ml._first_integral)
        mlc._second_integral = Measurement().copy(ml._second_integral)
        return mlc

    @property
    def nr_measurements(self):
        """Number of measurements."""
        return len(self._raw)

    @property
    def raw(self):
        """List with raw measurement data."""
        return self._raw

    @property
    def interpolated(self):
        """List with the interpolated measurement data."""
        return self._interpolated

    @property
    def average_voltage(self):
        """Average voltage values."""
        return self._average_voltage

    @property
    def std_voltage(self):
        """Standard deviation of voltage values."""
        return self._std_voltage

    @property
    def average_field(self):
        """Average magnetic field values."""
        return self._average_field

    @property
    def std_field(self):
        """Standard deviation of magnetic field values."""
        return self._std_field

    @property
    def first_integral(self):
        """Magnetic field first integral."""
        return self._first_integral

    @property
    def second_integral(self):
        """Magnetic field second integral."""
        return self._second_integral

    def add_measurement(self, measurement):
        """Add a measurement to the list."""
        self._raw.append(measurement)

    def analyse_data(self):
        """Analyse the measurement list data."""
        if self.nr_measurements != 0:
            self._data_interpolation()
            self._calculate_average_std()
            self._convert_voltage_field()
            self._calculate_first_integral()
            self._calculate_second_integral()

    def save_data(self, name, directory):
        """Save the measurement list data."""
        if self.nr_measurements != 0:
            self._save_raw_data(name, directory)
            self._save_interpolated_data(name, directory)
            self._save_avg_std_voltage_data(name, directory)
            self._save_avg_std_field_data(name, directory)
            self._save_field_first_integral(name, directory)
            self._save_field_second_integral(name, directory)

    def _get_shifts(self):
        if self.axis == 1:  # Z axis scan
            shiftx = self.calibration.shift_x_to_y
            shifty = 0
            shiftz = self.calibration.shift_z_to_y
        return (shiftx, shifty, shiftz)

    def _get_number_cuts(self):
        if self.axis == 1:  # Z axis scan
            n_cuts = _math.ceil(_np.array(
                [abs(self._calibration.shift_x_to_y),
                 abs(self._calibration.shift_z_to_y)]).max())
        return n_cuts

    def _data_interpolation(self):
        """Interpolate each measurement."""
        sx, sy, sz = self._get_shifts()

        # correct curves displacement due to trigger and
        # integration time (half integration time)
        self._interpolated = []
        for rm in self._raw:
            m = Measurement()

            m.axis = rm.axis
            m.position = self.position

            fx = _interpolate.splrep(rm.position + sx, rm.hallx, s=0, k=1)
            m.hallx = _interpolate.splev(m.position, fx, der=0)

            fy = _interpolate.splrep(rm.position + sy, rm.hally, s=0, k=1)
            m.hally = _interpolate.splev(m.position, fy, der=0)

            fz = _interpolate.splrep(rm.position + sz, rm.hallz, s=0, k=1)
            m.hallz = _interpolate.splev(m.position, fz, der=0)

            self._interpolated.append(m)

    def _calculate_average_std(self):
        """Calculate the average and std of voltage values."""
        n = self.nr_measurements

        # average calculation
        self._avarage_voltage.position = self.position
        self._average_voltage.hallx = _np.zeros(len(self.position))
        self._average_voltage.hally = _np.zeros(len(self.position))
        self._average_voltage.hallz = _np.zeros(len(self.position))

        if n > 1:
            for i in range(n):
                self._average_voltage.hallx += self._interpolated[i].hallx
                self._average_voltage.hally += self._interpolated[i].hally
                self._average_voltage.hallz += self._interpolated[i].hallz

            self._average_voltage.hallx /= n
            self._average_voltage.hally /= n
            self._average_voltage.hallz /= n
        else:
            self._average_voltage.hallx = self._interpolated[i].hallx
            self._average_voltage.hally = self._interpolated[i].hally
            self._average_voltage.hallz = self._interpolated[i].hallz

        # standard std calculation
        self._std_voltage.position = self.position
        self._std_voltage.hallx = _np.zeros(len(self.position))
        self._std_voltage.hally = _np.zeros(len(self.position))
        self._std_voltage.hallz = _np.zeros(len(self.position))

        if n > 1:
            for i in range(n):
                self._std_voltage.hallx += pow((
                    self._interpolated[i].hallx -
                    self._average_voltage.hallx), 2)

                self._std_voltage.hally += pow((
                    self._interpolated[i].hally -
                    self._average_voltage.hally), 2)

                self._std_voltage.hallz += pow((
                    self._interpolated[i].hallz -
                    self._average_voltage.hallz), 2)

            self._std_voltage.hallx /= n
            self._std_voltage.hally /= n
            self._std_voltage.hallz /= n

        # cut extra points due to shift sensors
        n_cuts = self._get_number_cuts()

        if n_cuts != 0:
            self._average_voltage.position = (
                self._average_voltage.position[n_cuts:-n_cuts])
            self._average_voltage.hallx = (
                self._average_voltage.hallx[n_cuts:-n_cuts])
            self._average_voltage.hally = (
                self._average_voltage.hally[n_cuts:-n_cuts])
            self._average_voltage.hallz = (
                self._average_voltage.hallz[n_cuts:-n_cuts])

            self._std_voltage.position = (
                self._std_voltage.position[n_cuts:-n_cuts])
            self._std_voltage.hallx = (
                self._std_voltage.hallx[n_cuts:-n_cuts])
            self._std_voltage.hally = (
                self._std_voltage.hally[n_cuts:-n_cuts])
            self._std_voltage.hallz = (
                self._std_voltage.hallz[n_cuts:-n_cuts])

    def _convert_voltage_field(self):
        """Calculate the average and std of magnetic field values."""
        self._average_field.position = self.position

        self._average_field.hallx = self.calibration.convert_probe_x(
            self._average_voltage.hallx)

        self._average_field.hally = self.calibration.convert_probe_y(
            self._average_voltage.hally)

        self._average_field.hallz = self.calibration.convert_probe_z(
            self._average_voltage.hallz)

        self._std_field.position = self.position

        self._std_field.hallx = self.calibration.convert_probe_x(
            self._std_voltage.hallx)

        self._std_field.hally = self.calibration.convert_probe_y(
            self._std_voltage.hally)

        self._std_field.hallz = self.calibration.convert_probe_z(
            self._std_voltage.hallz)

    def _calculate_first_integral(self):
        """Calculate the magnetic field first integral."""
        self._first_integral.position = self.position

        self._first_integral.hallx = _cumtrapz(
            x=self._average_field.position,
            y=self._average_field.hallx,
            initial=0)

        self._first_integral.hally = _cumtrapz(
            x=self._average_field.position,
            y=self._average_field.hally,
            initial=0)

        self._first_integral.hallz = _cumtrapz(
            x=self._average_field.position,
            y=self._average_field.hallz,
            initial=0)

    def _calculate_second_integral(self):
        """Calculate the magnetic field second integral."""
        self._second_integral.position = self.position

        self._second_integral.hallx = _cumtrapz(
            x=self._first_integral.position,
            y=self._first_integral.hallx,
            initial=0)

        self._second_integral.hally = _cumtrapz(
            x=self._first_integral.position,
            y=self._first_integral.hally,
            initial=0)

        self._second_integral.hallz = _cumtrapz(
            x=self._first_integral.position,
            y=self._first_integral.hallz,
            initial=0)

    def _save_raw_data(self, name, directory):
        """Save raw data to files.

        Args:
            name (str): name specifying the measurement location.
            directory (str): directory path.
        """
        for i in range(self.nr_measurements):
            filename = 'Raw_Data_' + name + '_' + str(i + 1) + '.dat'
            filename = _os.path.join(directory, filename)
            self._raw[i].save_to_file(filename)

    def _save_interpolated_data(self, name, directory):
        """Save intepolated data to files.

        Args:
            name (str): name specifying the measurement location.
            directory (str): directory path.
        """
        for i in range(self.nr_measurements):
            filename = 'Interpolated_Data_' + name + '_'+str(i + 1)+'.dat'
            filename = _os.path.join(directory, filename)
            self._interpolated[i].save_to_file(filename)

    def _save_avg_std_voltage_data(self, name, directory):
        """Save voltage average and std values to file.

        Args:
            name (str): name specifying the measurement location.
            directory (str): directory path.
        """
        filename = 'Average_Data_' + name + '.dat'
        filename = _os.path.join(directory, filename)
        _save_avg_std(filename, self._average_voltage, self._std_voltage)

    def _save_avg_std_field_data(self, name, directory):
        """Save magnetic field average and std values to file.

        Args:
            name (str): name specifying the measurement location.
            directory (str): directory path.
        """
        filename = 'Average_B_field_Data_' + name + '.dat'
        filename = _os.path.join(directory, filename)
        _save_avg_std(filename, self._average_field, self._std_field)

    def _save_field_first_integral(self, name, directory):
        """Save magnetic field first integral to file.

        Args:
            name (str): name specifying the measurement location.
            directory (str): directory path.
        """
        filename = 'First_integral_B_Data_' + name + '.dat'
        filename = _os.path.join(directory, filename)
        self._first_integral.save_to_file(filename)

    def _save_field_second_integral(self, name, directory):
        """Save magnetic field second integral to file.

        Args:
            name (str): name specifying the measurement location.
            directory (str): directory path.
        """
        filename = 'Second_integral_B_Data_' + name + '.dat'
        filename = _os.path.join(directory, filename)
        self._second_integral.save_to_file(filename)


def _save_avg_std(filename, avg, std):
    data = _np.column_stack((
        avg.position,
        avg.hallx, avg.hally, avg.hallz,
        std.hallx, std.hally, std.hallz))
    _np.savetxt(filename, data, delimiter='\t', newline='\r\n')
