# -*- coding: utf-8 -*-
"""Implementation of classes to store and analyse measurement data."""

import os as _os
import numpy as _np
import pandas as _pd
from scipy import interpolate as _interpolate

from . import utils as _utils
from . import calibration as _calibration


_position_precision = 4


class MeasurementDataError(Exception):
    """Measurement data exception."""

    def __init__(self, message, *args):
        """Initialize variables."""
        self.message = message


class VoltageData(object):
    """Position and voltage values."""

    _axis_list = [1, 2, 3, 5, 6, 7, 8, 9]
    _pos1_unit = 'mm'
    _pos2_unit = 'mm'
    _pos3_unit = 'mm'
    _pos5_unit = 'deg'
    _pos6_unit = 'mm'
    _pos7_unit = 'mm'
    _pos8_unit = 'deg'
    _pos9_unit = 'deg'

    def __init__(self, filename=None):
        """Initialize variables."""
        self._pos1 = _np.array([])
        self._pos2 = _np.array([])
        self._pos3 = _np.array([])
        self._pos5 = _np.array([])
        self._pos6 = _np.array([])
        self._pos7 = _np.array([])
        self._pos8 = _np.array([])
        self._pos9 = _np.array([])
        self._probei = _np.array([])
        self._probej = _np.array([])
        self._probek = _np.array([])
        if filename is not None:
            self.read_file(filename)

    def __str__(self):
        """Printable string representation of VoltageData."""
        fmtstr = '{0:<18s} : {1}\n'
        r = ''
        r += fmtstr.format('scan_axis', str(self.scan_axis))
        r += fmtstr.format('pos1[mm]', str(self.pos1))
        r += fmtstr.format('pos2[mm]', str(self.pos2))
        r += fmtstr.format('pos3[mm]', str(self.pos3))
        r += fmtstr.format('pos5[deg]', str(self.pos5))
        r += fmtstr.format('pos6[mm]', str(self.pos6))
        r += fmtstr.format('pos7[mm]', str(self.pos7))
        r += fmtstr.format('pos8[deg]', str(self.pos8))
        r += fmtstr.format('pos9[deg]', str(self.pos9))
        r += fmtstr.format('probei[V]', str(self.probei))
        r += fmtstr.format('probej[V]', str(self.probej))
        r += fmtstr.format('probek[V]', str(self.probek))
        return r

    @property
    def pos1(self):
        """Position 1 (Axis +Z) [mm]."""
        return self._pos1

    @pos1.setter
    def pos1(self, value):
        self._pos1 = _np.around(_to_array(value), decimals=_position_precision)

    @property
    def pos2(self):
        """Position 2 (Axis +Y) [mm]."""
        return self._pos2

    @pos2.setter
    def pos2(self, value):
        self._pos2 = _np.around(_to_array(value), decimals=_position_precision)

    @property
    def pos3(self):
        """Position 3 (Axis +X) [mm]."""
        return self._pos3

    @pos3.setter
    def pos3(self, value):
        self._pos3 = _np.around(_to_array(value), decimals=_position_precision)

    @property
    def pos5(self):
        """Position 5 (Axis +A) [deg]."""
        return self._pos5

    @pos5.setter
    def pos5(self, value):
        self._pos5 = _np.around(_to_array(value), decimals=_position_precision)

    @property
    def pos6(self):
        """Position 6 (Axis +W) [mm]."""
        return self._pos6

    @pos6.setter
    def pos6(self, value):
        self._pos6 = _np.around(_to_array(value), decimals=_position_precision)

    @property
    def pos7(self):
        """Position 7 (Axis +V) [mm]."""
        return self._pos7

    @pos7.setter
    def pos7(self, value):
        self._pos7 = _np.around(_to_array(value), decimals=_position_precision)

    @property
    def pos8(self):
        """Position 8 (Axis +B) [deg]."""
        return self._pos8

    @pos8.setter
    def pos8(self, value):
        self._pos8 = _np.around(_to_array(value), decimals=_position_precision)

    @property
    def pos9(self):
        """Position 9 (Axis +C) [deg]."""
        return self._pos9

    @pos9.setter
    def pos9(self, value):
        self._pos9 = _np.around(_to_array(value), decimals=_position_precision)

    @property
    def probei(self):
        """Probe I Voltage [V]."""
        return self._probei

    @probei.setter
    def probei(self, value):
        self._probei = _to_array(value)

    @property
    def probej(self):
        """Probe J Voltage [V]."""
        return self._probej

    @probej.setter
    def probej(self, value):
        self._probej = _to_array(value)

    @property
    def probek(self):
        """Probe K Voltage [V]."""
        return self._probek

    @probek.setter
    def probek(self, value):
        self._probek = _to_array(value)

    @property
    def axis_list(self):
        """List of all bench axes."""
        return self._axis_list

    @property
    def scan_axis(self):
        """Scan Axis."""
        pos = []
        for axis in self._axis_list:
            pos.append(getattr(self, 'pos'+str(axis)))
        if _np.count_nonzero([p.size > 1 for p in pos]) != 1:
            return None
        else:
            idx = _np.where([p.size > 1 for p in pos])[0][0]
            return self._axis_list[idx]

    @property
    def scan_pos(self):
        """Scan positions."""
        return getattr(self, 'pos' + str(self.scan_axis))

    @scan_pos.setter
    def scan_pos(self, value):
        setattr(self, 'pos' + str(self.scan_axis), value)

    @property
    def npts(self):
        """Number of data points."""
        if self.scan_axis is None:
            return 0
        else:
            npts = len(getattr(self, 'pos' + str(self.scan_axis)))
            v = [self.probei, self.probej, self.probek]
            if all([vi.size == 0 for vi in v]):
                return 0
            for vi in v:
                if vi.size not in [npts, 0]:
                    return 0
            return npts

    def clear(self):
        """Clear VoltageData."""
        for key in self.__dict__:
            if isinstance(self.__dict__[key], _np.ndarray):
                self.__dict__[key] = _np.array([])

    def copy(self):
        """Return a copy of the object."""
        _copy = VoltageData()
        for key in self.__dict__:
            if isinstance(self.__dict__[key], _np.ndarray):
                _copy.__dict__[key] = _np.copy(self.__dict__[key])
        return _copy

    def reverse(self):
        """Reverse VoltageData."""
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, _np.ndarray) and value.size > 1:
                self.__dict__[key] = value[::-1]

    def read_file(self, filename):
        """Read voltage data from file."""
        flines = _utils.read_file(filename)

        scan_axis = _utils.find_value(flines, 'scan_axis', int)
        for axis in self._axis_list:
            if axis != scan_axis:
                pos_str = 'pos' + str(axis)
                try:
                    pos = _utils.find_value(flines, pos_str, float)
                except ValueError:
                    pos = None
                setattr(self, pos_str, pos)

        idx = next((i for i in range(len(flines))
                    if flines[i].find("----------") != -1), None)
        data = []
        for line in flines[idx+1:]:
            data_line = [float(d) for d in line.split('\t')]
            data.append(data_line)
        data = _np.array(data)

        if data.shape[1] == 4:
            scan_positions = data[:, 0]
            setattr(self, 'pos' + str(scan_axis), scan_positions)
            self.probei = data[:, 1]
            self.probej = data[:, 2]
            self.probek = data[:, 3]
        else:
            message = 'Inconsistent number of columns in file: %s' % filename
            raise MeasurementDataError(message)

    def save_file(self, filename):
        """Save voltage data to file."""
        if self.scan_axis is None or self.npts == 0:
            raise MeasurementDataError('Invalid scan axis.')

        scan_axis = self.scan_axis
        timestamp = _utils.get_timestamp()
        pos1_str = '%f' % self.pos1[0] if self.pos1.size == 1 else '--'
        pos2_str = '%f' % self.pos2[0] if self.pos2.size == 1 else '--'
        pos3_str = '%f' % self.pos3[0] if self.pos3.size == 1 else '--'
        pos5_str = '%f' % self.pos5[0] if self.pos5.size == 1 else '--'
        pos6_str = '%f' % self.pos6[0] if self.pos6.size == 1 else '--'
        pos7_str = '%f' % self.pos7[0] if self.pos7.size == 1 else '--'
        pos8_str = '%f' % self.pos8[0] if self.pos8.size == 1 else '--'
        pos9_str = '%f' % self.pos9[0] if self.pos9.size == 1 else '--'

        scan_axis_unit = getattr(self, '_pos' + str(scan_axis) + '_unit')
        scan_axis_pos = getattr(self, 'pos' + str(scan_axis))

        columns_names = (
            'pos%i[%s]\t' % (scan_axis, scan_axis_unit) +
            'probei[V]\tprobej[V]\tprobek[V]')

        if len(self.probei) != self.npts:
            self.probei = _np.zeros(self.npts)

        if len(self.probej) != self.npts:
            self.probej = _np.zeros(self.npts)

        if len(self.probek) != self.npts:
            self.probek = _np.zeros(self.npts)

        columns = _np.column_stack((scan_axis_pos, self.probei,
                                    self.probej, self.probek))

        f = open(filename, mode='w')
        f.write('timestamp:         \t%s\n' % timestamp)
        f.write('scan_axis:         \t%s\n' % scan_axis)
        f.write('pos1[mm]:          \t%s\n' % pos1_str)
        f.write('pos2[mm]:          \t%s\n' % pos2_str)
        f.write('pos3[mm]:          \t%s\n' % pos3_str)
        f.write('pos5[deg]:         \t%s\n' % pos5_str)
        f.write('pos6[mm]:          \t%s\n' % pos6_str)
        f.write('pos7[mm]:          \t%s\n' % pos7_str)
        f.write('pos8[deg]:         \t%s\n' % pos8_str)
        f.write('pos9[deg]:         \t%s\n' % pos9_str)
        f.write('\n')
        f.write('%s\n' % columns_names)
        f.write('---------------------------------------------------' +
                '---------------------------------------------------\n')
        for i in range(columns.shape[0]):
            line = '{0:+0.4f}'.format(columns[i, 0])
            for j in range(1, columns.shape[1]):
                line = line + '\t' + '{0:+0.10e}'.format(columns[i, j])
            f.write(line + '\n')
        f.close()


class FieldData(object):
    """Position and magnetic field values."""

    def __init__(self, voltage_list, calibration_data):
        """Initialize variables.

        Args:
            voltage_list (list): list of VoltageData objects or
                                 VoltageData filenames.
            calibration_data (CalibrationData): Hall probe calibration data.
        """
        if isinstance(calibration_data, _calibration.CalibrationData):
            self._calibration_data = calibration_data
        else:
            self._calibration_data = None
            raise TypeError(
                'calibration_data must be a CalibrationData object.')

        if all([isinstance(v, VoltageData) for v in voltage_list]):
            self._voltage = _get_average_voltage_list(voltage_list)
        elif all([isinstance(v, str) for v in voltage_list]):
            voltage_list = [VoltageData(filename=fn) for fn in voltage_list]
            self._voltage = _get_average_voltage_list(voltage_list)
        else:
            raise TypeError(
                'voltage_list must be a list of VoltageData object.')

        self._pos1 = _np.array([])
        self._pos2 = _np.array([])
        self._pos3 = _np.array([])
        self._field1 = None
        self._field2 = None
        self._field3 = None
        self._index_axis = None
        self._columns_axis = None
        self._convert_voltage_list_to_field_data()

    def _convert_voltage_list_to_field_data(self):
        # check positions
        _dict = _get_axis_position_dict(self._voltage)
        for i in [5, 6, 7, 8, 9]:
            if len(_dict[i]) > 0 and _dict[i][0] != 0:
                raise NotImplemented

        axes = _get_measurement_axes(self._voltage)
        if len(axes) == 2:
            self._index_axis = axes[0]
            self._columns_axis = axes[1]
        elif len(axes) == 1:
            self._index_axis = axes[0]
        else:
            raise MeasurementDataError('Invalid number of measurement axes.')

        dfi = []
        dfj = []
        dfk = []
        for vd in self._voltage:
            index = getattr(vd, 'pos' + str(self._index_axis))
            if self._columns_axis is not None:
                columns = getattr(vd, 'pos' + str(self._columns_axis))
            else:
                columns = [0]

            # Convert voltage to magnetic field
            bi = self._calibration_data.convert_voltage_probei(vd.probei)
            bj = self._calibration_data.convert_voltage_probej(vd.probej)
            bk = self._calibration_data.convert_voltage_probek(vd.probek)

            index = _pd.Index(index, float)
            columns = _pd.Index(columns, float)
            dfi.append(_pd.DataFrame(bi, index=index, columns=columns))
            dfj.append(_pd.DataFrame(bj, index=index, columns=columns))
            dfk.append(_pd.DataFrame(bk, index=index, columns=columns))

        fieldi = _pd.concat(dfi, axis=1)
        fieldj = _pd.concat(dfj, axis=1)
        fieldk = _pd.concat(dfk, axis=1)

        index = fieldi.index
        columns = fieldi.columns
        if (len(columns) != len(columns.drop_duplicates())):
            msg = 'Duplicate position found in voltage list.'
            raise MeasurementDataError(msg)

        fieldi, fieldj, fieldk = _correct_probe_displacement(
            fieldi, fieldj, fieldk, self._index_axis,
            self._columns_axis, self._calibration_data)

        # update position values
        index = fieldj.index
        columns = fieldj.columns
        pos_sorted = [_dict[1], _dict[2], _dict[3]]
        pos_sorted[self._index_axis - 1] = index.values
        if self._columns_axis is not None:
            pos_sorted[self._columns_axis - 1] = columns.values

        self._pos3 = _np.array(pos_sorted[2])
        self._pos2 = _np.array(pos_sorted[1])
        self._pos1 = _np.array(pos_sorted[0])

        if self._calibration_data.stem_shape.capitalize() == 'L-shape':
            self._field1 = -fieldi
            self._field2 = fieldj
            self._field3 = fieldk
        else:
            self._field1 = fieldk
            self._field2 = fieldj
            self._field3 = fieldi

    @property
    def pos1(self):
        """Position 1 (Axis +Z) [mm]."""
        return self._pos1

    @property
    def pos2(self):
        """Position 2 (Axis +Y) [mm]."""
        return self._pos2

    @property
    def pos3(self):
        """Position 3 (Axis +X) [mm]."""
        return self._pos3

    @property
    def field1(self):
        """Data frame with magnetic field values (Axis #1 Component) [T]."""
        return self._field1

    @property
    def field2(self):
        """Data frame with magnetic field values (Axis #2 Component) [T]."""
        return self._field2

    @property
    def field3(self):
        """Data frame with magnetic field values (Axis #3 Component) [T]."""
        return self._field3

    @property
    def index_axis(self):
        """Index axis."""
        return self._index_axis

    @property
    def columns_axis(self):
        """Column axis."""
        return self._columns_axis

    def get_field_at_point(self, pos):
        """Get the magnetic field value at the given point.

        Args:
            pos (list or array): position list [pos3, pos2, pos1].

        Returns:
            the magnetic field components [b3, b2, b1].
        """
        pos = _np.around(pos, decimals=_position_precision)
        p1 = pos[2]
        p2 = pos[1]
        p3 = pos[0]
        if (p1 not in self._pos1 or p2 not in self._pos2 or
           p3 not in self._pos3):
            return [_np.nan, _np.nan, _np.nan]

        psorted = [p1, p2, p3]
        loc_idx = psorted[self._index_axis-1]
        if self._columns_axis is not None:
            loc_col = psorted[self._columns_axis-1]
        else:
            loc_col = 0
        b3 = self._field3.loc[loc_idx, loc_col]
        b2 = self._field2.loc[loc_idx, loc_col]
        b1 = self._field1.loc[loc_idx, loc_col]
        return [b3, b2, b1]

    def save_file(self, filename, header_info=[], magnet_center=[0, 0, 0],
                  magnet_x_axis='3', magnet_y_axis='2'):
        """Save measurement data.

        Args:
            filename (str): fieldmap file path.
            header_info (list): list of tuples of variables and values to
                                include in the fieldmap header lines.
            magnet_center (list): center position of the magnet.
            magnet_x_axis (str): magnet x-axis direction.
                                 ['3', '-3', '2', '-2', '1' or '-1']
            magnet_y_axis (str): magnet y-axis direction.
                                 ['3', '-3', '2', '-2', '1' or '-1']
        """
        atts = [self._pos1, self._pos2, self._pos3,
                self._field1, self._field2, self._field3]
        if any([att is None for att in atts]):
            message = "Invalid field data."
            raise MeasurementDataError(message)

        if len(header_info) != 0:
            if any(len(line) != 2 for line in header_info):
                raise ValueError('Invalid value for header_info.')

        f = open(filename, 'w')

        for line in header_info:
            variable = (str(line[0]) + ':').ljust(20)
            value = str(line[1])
            f.write('{0:1s}\t{1:1s}\n'.format(variable, value))

        if len(header_info) != 0:
            f.write('\n')

        f.write('X[mm]\tY[mm]\tZ[mm]\tBx[T]\tBy[T]\tBz[T]\n')
        f.write('-----------------------------------------------' +
                '----------------------------------------------\n')

        fieldmap = self._get_transformed_fieldmap(
            magnet_center, magnet_x_axis, magnet_y_axis)
        for i in range(fieldmap.shape[0]):
            f.write('{0:0.4f}\t{1:0.4f}\t{2:0.4f}\t'.format(
                fieldmap[i, 0], fieldmap[i, 1], fieldmap[i, 2]))
            f.write('{0:0.10e}\t{1:0.10e}\t{2:0.10e}\n'.format(
                fieldmap[i, 3], fieldmap[i, 4], fieldmap[i, 5]))
        f.close()

        fn_aux = _os.path.join(
            _os.path.split(filename)[0], 'magnet_coordinate_system.txt')
        f_aux = open(fn_aux, 'w')
        f_aux.write('magnet_center_axis3: {0:0.4f}\n'.format(magnet_center[0]))
        f_aux.write('magnet_center_axis2: {0:0.4f}\n'.format(magnet_center[1]))
        f_aux.write('magnet_center_axis1: {0:0.4f}\n'.format(magnet_center[2]))
        f_aux.write('magnet_x_axis:       {0:1s}\n'.format(
            magnet_x_axis.replace(' ', '')))
        f_aux.write('magnet_y_axis:       {0:1s}\n'.format(
            magnet_y_axis.replace(' ', '')))
        f_aux.close()

        return filename

    def _get_transformed_fieldmap(self, magnet_center=[0, 0, 0],
                                  magnet_x_axis='3', magnet_y_axis='2'):
        tm = _get_transformation_matrix(magnet_x_axis, magnet_y_axis)

        fieldmap = []
        for p1 in self._pos1:
            for p2 in self._pos2:
                for p3 in self._pos3:
                    p = [p3, p2, p1]
                    b = self.get_field_at_point(p)
                    tp = _change_coordinate_system(p, tm, magnet_center)
                    tb = _change_coordinate_system(b, tm)
                    fieldmap.append(_np.append(tp, tb))
        fieldmap = _np.array(fieldmap)

        fieldmap = _np.array(sorted(
            fieldmap, key=lambda x: (x[2], x[1], x[0])))

        return fieldmap

    def _get_fieldmap(self):
        fieldmap = []
        for p1 in self._pos1:
            for p2 in self._pos2:
                for p3 in self._pos3:
                    p = [p3, p2, p1]
                    b = self.get_field_at_point(p)
                    fieldmap.append(_np.append(p, b))
        fieldmap = _np.array(fieldmap)
        return fieldmap


def _to_array(value):
    if value is not None:
        if not isinstance(value, _np.ndarray):
            value = _np.array(value)
        if len(value.shape) == 0:
            value = _np.array([value])
    else:
        value = _np.array([])
    return value


def _get_axis_position_dict(voltage_list):
    if len(voltage_list) == 0:
        raise MeasurementDataError('Empty voltage list.')

    if any([vd.scan_axis is None or vd.npts == 0 for vd in voltage_list]):
        raise MeasurementDataError('Invalid voltage list.')

    if not all([vd.scan_axis == voltage_list[0].scan_axis
               for vd in voltage_list]):
        raise MeasurementDataError('Inconsistent scan axis.')

    _dict = {}
    for axis in voltage_list[0].axis_list:
        pos = set()
        for voltage_data in voltage_list:
            pos.update(getattr(voltage_data, 'pos' + str(axis)))
        _dict[axis] = sorted(list(pos))

    return _dict


def _get_measurement_axes(voltage_list):
    _dict = _get_axis_position_dict(voltage_list)
    scan_axis = voltage_list[0].scan_axis
    axes = [scan_axis]
    for key, value in _dict.items():
        if key != scan_axis and len(value) > 1:
            axes.append(key)
    return axes


def _get_average_voltage_list(voltage_list):
    if len(voltage_list) == 0:
        raise MeasurementDataError('Empty voltage list.')

    if any([vd.scan_axis is None or vd.npts == 0 for vd in voltage_list]):
        raise MeasurementDataError('Invalid voltage list.')

    if not all([vd.scan_axis == voltage_list[0].scan_axis
               for vd in voltage_list]):
        raise MeasurementDataError('Inconsistent scan axis.')

    def _sorted_key(x):
        pos = []
        for axis in x.axis_list:
            if axis != x.scan_axis:
                pos.append(getattr(x, 'pos' + str(axis)))
        return pos

    voltage_list = sorted(voltage_list, key=_sorted_key)

    fixed_axes = [
        a for a in voltage_list[0].axis_list if a != voltage_list[0].scan_axis]

    avg_voltage_list = []
    tmp_voltage_list = []
    prev_pos = None
    for vd in voltage_list:
        pos = []
        for axis in fixed_axes:
            pos.append(getattr(vd, 'pos' + str(axis)))
        if (prev_pos is None or all([
            pos[i] == prev_pos[i] for i in range(len(pos)) if not (
                len(pos[i]) == 0 and len(prev_pos[i]) == 0)])):
            tmp_voltage_list.append(vd)
        else:
            avg, std = _get_avg_std(tmp_voltage_list)
            avg_voltage_list.append(avg)
            tmp_voltage_list = [vd]
        prev_pos = pos

    avg, std = _get_avg_std(tmp_voltage_list)
    avg_voltage_list.append(avg)

    return avg_voltage_list


def _get_avg_std(voltage_list):
    axes = _get_measurement_axes(voltage_list)
    if len(axes) != 1:
        raise MeasurementDataError('Invalid number of measurement axes.')

    if not all([vd.npts == voltage_list[0].npts for vd in voltage_list]):
        raise MeasurementDataError('Inconsistent number of points.')

    scan_axis = axes[0]
    interpolation_pos = _np.mean([vd.scan_pos for vd in voltage_list], axis=0)

    interp_list = []
    for vd in voltage_list:
        interpolated = vd.copy()
        n = len(interpolation_pos)
        interpolated.probei = _np.zeros(n)
        interpolated.probej = _np.zeros(n)
        interpolated.probek = _np.zeros(n)
        setattr(interpolated, 'pos' + str(scan_axis), interpolation_pos)

        if len(vd.probei) == vd.npts:
            fr = _interpolate.splrep(vd.scan_pos, vd.probei, s=0, k=1)
            interpolated.probei = _interpolate.splev(
                interpolation_pos, fr, der=0)

        if len(vd.probej) == vd.npts:
            fs = _interpolate.splrep(vd.scan_pos, vd.probej, s=0, k=1)
            interpolated.probej = _interpolate.splev(
                interpolation_pos, fs, der=0)

        if len(vd.probek) == vd.npts:
            ft = _interpolate.splrep(vd.scan_pos, vd.probek, s=0, k=1)
            interpolated.probek = _interpolate.splev(
                interpolation_pos, ft, der=0)

        interp_list.append(interpolated)

    count = len(interp_list)
    npts = interp_list[0].npts

    avg = interp_list[0].copy()
    avg.probei = _np.zeros(npts)
    avg.probej = _np.zeros(npts)
    avg.probek = _np.zeros(npts)
    for i in range(count):
        avg.probei += interp_list[i].probei
        avg.probej += interp_list[i].probej
        avg.probek += interp_list[i].probek
    avg.probei /= count
    avg.probej /= count
    avg.probek /= count

    std = interp_list[0].copy()
    std.probei = _np.zeros(npts)
    std.probej = _np.zeros(npts)
    std.probek = _np.zeros(npts)
    for i in range(count):
        std.probei += pow((
            interp_list[i].probei - avg.probei), 2)
        std.probej += pow((
            interp_list[i].probej - avg.probej), 2)
        std.probek += pow((
            interp_list[i].probek - avg.probek), 2)
    std.probei /= count
    std.probej /= count
    std.probek /= count

    return avg, std


def _correct_probe_displacement(fieldi, fieldj, fieldk, index_axis,
                                columns_axis, calibration_data):
    if calibration_data.stem_shape.capitalize() == 'L-shape':
        axis_corr = 3
    else:
        axis_corr = 1

    index = fieldi.index
    columns = fieldi.columns

    if axis_corr == index_axis:
        # shift field data
        fieldi.index = index - calibration_data.distance_probei
        fieldk.index = index + calibration_data.distance_probek

        # interpolate field data
        fieldi, fieldj, fieldk = _interpolate_data_frames(
            fieldi, fieldj, fieldk, axis=0)

        # cut field data
        nbeg, nend = _get_number_of_cuts(
            index,
            calibration_data.distance_probei,
            calibration_data.distance_probek)
        fieldi, fieldj, fieldk = _cut_data_frames(
            fieldi, fieldj, fieldk, nbeg, nend)

    elif axis_corr == columns_axis:
        # shift field data
        fieldi.columns = columns - calibration_data.distance_probei
        fieldk.columns = columns + calibration_data.distance_probek

        # interpolate field data
        fieldi, fieldj, fieldk = _interpolate_data_frames(
            fieldi, fieldj, fieldk, axis=1)

        # cut field data
        nbeg, nend = _get_number_of_cuts(
            columns,
            calibration_data.distance_probei,
            calibration_data.distance_probek)
        fieldi, fieldj, fieldk = _cut_data_frames(
            fieldi, fieldj, fieldk,
            nbeg, nend, axis=1)
    else:
        raise NotImplemented

    return fieldi, fieldj, fieldk


def _interpolate_data_frames(dfi, dfj, dfk, axis=0):

    def _interpolate_vec(x, pos):
        f = _interpolate.splrep(x.index, x.values, s=0, k=1)
        return _interpolate.splev(pos, f, der=0)

    if axis == 1:
        pos = dfj.columns
        interp_dfi = dfi.apply(_interpolate_vec, axis=axis, args=[pos])
        interp_dfk = dfk.apply(_interpolate_vec, axis=axis, args=[pos])
        interp_dfi.columns = pos
        interp_dfk.columns = pos
    else:
        pos = dfj.index
        interp_dfi = dfi.apply(_interpolate_vec, args=[pos])
        interp_dfk = dfk.apply(_interpolate_vec, args=[pos])
        interp_dfi.index = pos
        interp_dfk.index = pos

    return interp_dfi, dfj, interp_dfk


def _get_number_of_cuts(vec, dist_beg, dist_end):
    diff_beg = _np.append(0, _np.abs(_np.cumsum(_np.diff(vec))))
    diff_end = _np.append(0, _np.abs(_np.cumsum(_np.diff(vec)[::-1])))
    n_beg = len(diff_beg[diff_beg < dist_beg])
    n_end = len(diff_end[diff_end < dist_end])
    return n_beg, n_end


def _cut_data_frames(dfi, dfj, dfk, nbeg, nend, axis=0):
    if axis == 1:
        if nbeg > 0:
            dfi = dfi.drop(dfi.columns[:nbeg], axis=axis)
            dfj = dfj.drop(dfj.columns[:nbeg], axis=axis)
            dfk = dfk.drop(dfk.columns[:nbeg], axis=axis)
        if nend > 0:
            dfi = dfi.drop(dfi.columns[-nend:], axis=axis)
            dfj = dfj.drop(dfj.columns[-nend:], axis=axis)
            dfk = dfk.drop(dfk.columns[-nend:], axis=axis)
    else:
        if nbeg > 0:
            dfi = dfi.drop(dfi.index[:nbeg])
            dfj = dfj.drop(dfj.index[:nbeg])
            dfk = dfk.drop(dfk.index[:nbeg])
        if nend > 0:
            dfi = dfi.drop(dfi.index[-nend:])
            dfj = dfj.drop(dfj.index[-nend:])
            dfk = dfk.drop(dfk.index[-nend:])
    return dfi, dfj, dfk


def _get_axis_vector(axis_str):
    if '3' in axis_str:
        if axis_str.startswith('-'):
            axis_vec = [-1, 0, 0]
        else:
            axis_vec = [1, 0, 0]
    elif '2' in axis_str:
        if axis_str.startswith('-'):
            axis_vec = [0, -1, 0]
        else:
            axis_vec = [0, 1, 0]
    elif '1' in axis_str:
        if axis_str.startswith('-'):
            axis_vec = [0, 0, -1]
        else:
            axis_vec = [0, 0, 1]
    else:
        axis_vec = None
    return axis_vec


def _get_transformation_matrix(x_str, y_str):
    x = _get_axis_vector(x_str)
    y = _get_axis_vector(y_str)
    if x is None or y is None:
        raise MeasurementDataError('Invalid magnet axes.')
    z = _np.cross(x, y)
    v3 = [1, 0, 0]
    v2 = [0, 1, 0]
    v1 = [0, 0, 1]
    m = _np.outer(v3, x) + _np.outer(v2, y) + _np.outer(v1, z)
    return m


def _change_coordinate_system(vector, transf_matrix, center=[0, 0, 0]):
    vector = _np.array(vector)
    center = _np.array(center)
    transf_vector = _np.dot(transf_matrix, vector - center)
    return transf_vector
