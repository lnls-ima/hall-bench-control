# -*- coding: utf-8 -*-
"""Implementation of classes to handle calibration files."""

import numpy as _np
from hallbench import files as _files


class CalibrationData(object):
    """Hall probe calibration data."""

    def __init__(self, filename=None):
        """Initialize variables.

        Args:
            filename (str): calibration file path.
        """
        if filename is not None:
            self.read_file(filename)
        else:
            self._filename = None
            self._field_unit = 'T'
            self._voltage_unit = 'V'
            self._probex_dx = 0
            self._probex_dy = 0
            self._probex_dz = 0
            self._probez_dx = 0
            self._probez_dy = 0
            self._probez_dz = 0
            self._angle_xy = 0
            self._angle_yz = 0
            self._angle_xz = 0

    @property
    def filename(self):
        """Calibration file path."""
        return self._filename

    @property
    def field_unit(self):
        """Magnetic field unit."""
        return self._field_unit

    @property
    def voltage_unit(self):
        """Voltage unit."""
        return self._voltage_unit

    @property
    def probex_dx(self):
        """Probe X - displacement x[mm]."""
        return self._probex_dx

    @property
    def probex_dy(self):
        """Probe X - displacement y[mm]."""
        return self._probex_dy

    @property
    def probex_dz(self):
        """Probe X - displacement z[mm]."""
        return self._probex_dz

    @property
    def probez_dx(self):
        """Probe Z - displacement x[mm]."""
        return self._probez_dx

    @property
    def probez_dy(self):
        """Probe Z - displacement y[mm]."""
        return self._probez_dy

    @property
    def probez_dz(self):
        """Probe Z - displacement z[mm]."""
        return self._probez_dz

    @property
    def angle_xy(self):
        """Angle XY [rad]."""
        return self._angle_xy

    @property
    def angle_yz(self):
        """Angle YZ [rad]."""
        return self._angle_yz

    @property
    def angle_xz(self):
        """Angle XZ [rad]."""
        return self._angle_xz

    def read_file(self, filename):
        """Read calibration parameters from file.

        Args:
            filename (str): calibration file path.

        Raises:
            HallBenchFileError: if cannot read file data.
        """
        pass

    def save_file(self, filename):
        """Save calibration data to file.

        Args:
            filename (str): calibration file path.

        Raises:
            HallBenchFileError: if the calibration data was not saved.
        """
        f = open(filename, mode='w')
        f.close()

        if self._filename is None:
            self._filename = filename

    def get_conversion_factor(self, voltage_unit):
        """Get voltage consersion factor."""
        if voltage_unit == self._voltage_unit:
            return 1
        else:
            return 0

    def convert_probe_x(self, voltage_array):
        """Convert voltage values to magnetic field values for probe x.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        return _default_hall_probe_calibration_curve(voltage_array)

    def convert_probe_y(self, voltage_array):
        """Convert voltage values to magnetic field values for probe y.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        return _default_hall_probe_calibration_curve(voltage_array)

    def convert_probe_z(self, voltage_array):
        """Convert voltage values to magnetic field values for probe z.

        Args:
            voltage_array (array): array with voltage values.

        Returns:
            array with magnetic field values.
        """
        return _default_hall_probe_calibration_curve(voltage_array)


def _default_hall_probe_calibration_curve(voltage_array):
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

    # def read_file(self)
    #     data = _files.read_file(filename)
    #
    #     self._field_unit = _files.find_value(data, 'field_unit')
    #     self._voltage_unit = _files.find_value(data, 'voltage_unit')
    #     self._probex_dx = _files.find_value(data, 'probex_dx', vtype='float')
    #     self._probex_dy = _files.find_value(data, 'probex_dy', vtype='float')
    #     self._probex_dz = _files.find_value(data, 'probex_dz', vtype='float')
    #     self._probez_dx = _files.find_value(data, 'probez_dx', vtype='float')
    #     self._probez_dy = _files.find_value(data, 'probez_dy', vtype='float')
    #     self._probez_dz = _files.find_value(data, 'probez_dz', vtype='float')
    #     self._angle_xy = _files.find_value(data, 'angle_xy', vtype='float')
    #     self._angle_yz = _files.find_value(data, 'angle_yz', vtype='float')
    #     self._angle_xz = _files.find_value(data, 'angle_xz', vtype='float')
    #
    #     idx_probex = next((i for i in range(len(data))
    #                        if data[i].find("Probe X Data") != -1), None)
    #     if idx_probex is None:
    #         message = 'Probe X data not found in file: "%s"' % filename
    #         raise _files.HallBenchFileError(message)
    #
    #     idx_probey = next((i for i in range(len(data))
    #                        if data[i].find("Probe Y Data") != -1), None)
    #     if idx_probey is None:
    #         message = 'Probe Y data not found in file: "%s"' % filename
    #         raise _files.HallBenchFileError(message)
    #
    #     idx_probez = next((i for i in range(len(data))
    #                        if data[i].find("Probe Z Data") != -1), None)
    #     if idx_probez is None:
    #         message = 'Probe Z data not found in file: "%s"' % filename
    #         raise _files.HallBenchFileError(message)
    #
    #     data_probex = data[idx_probex:idx_probey]
    #     data_probey = data[idx_probey:idx_probez]
    #     data_probez = data[idx_probez:]
    #
    #     self._read_probe_data(data_probex)
    #     self._read_probe_data(data_probey)
    #     self._read_probe_data(data_probez)

# def _is_on_interval(interval, value):
#     lim = [float(v) for v in interval.strip()[1:-1].split(',')]
#     min_lim = lim[0]
#     max_lim = lim[1]
#     min_lim_flag = False
#     max_lim_flag = False
#
#     if '(' in interval:
#         if value > min_lim:
#             min_lim_flag = True
#     elif '[' in interval:
#         if value >= min_lim:
#             min_lim_flag = True
#
#     if ')' in interval:
#         if value < max_lim:
#             max_lim_flag = True
#     elif ']' in interval:
#         if value <= max_lim:
#             max_lim_flag = True
#
#     return all([min_lim_flag, max_lim_flag])
#
#
# def _read_probe_data(data):
#     polylist = []
#     index_list = [i for i in range(len(data))
#                   if data[i].find('voltage_interval') != -1]
#
#     for i in range(len(index_list)):
#         if i == len(index_list)-1:
#             m = data[index_list[i]+2:]
#         else:
#             m = data[index_list[i]+2:index_list[i+1]]
#
#         voltage_interval = data[index_list[i]].split('\t')[1]
#         order = [int(line.split('\t')[0]) for line in m]
#         coeff = [float(line.split('\t')[1]) for line in m]
#
#         polylist.append({
#             'voltage_interval': voltage_interval,
#             'order': order,
#             'coefficients': coeff})
#
#     return polylist
#
#
# def _convert(polylist, volt):
#     field = None
#     for d in polylist:
#         coeff = d['coefficients']
#         order = d['order']
#         if _is_on_interval(d['voltage_interval'], volt):
#             field = sum(coeff[i]*(volt**order[i]) for i in range(len(order)))
#             break
#     return field
