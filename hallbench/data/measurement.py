# -*- coding: utf-8 -*-

"""Implementation of classes to store and analyse measurement data."""

import os.path as _path
import numpy as _np
import pandas as _pd
from scipy import interpolate as _interpolate

from . import utils as _utils
from . import calibration as _calibration

_position_precision = 4
_check_position_precision = 3


class MeasurementDataError(Exception):
    """Measurement data exception."""

    def __init__(self, message, *args):
        """Initialize variables."""
        self.message = message


class Data(object):
    """Position and data values."""

    _axis_list = [1, 2, 3, 5, 6, 7, 8, 9]
    _pos1_unit = 'mm'
    _pos2_unit = 'mm'
    _pos3_unit = 'mm'
    _pos5_unit = 'deg'
    _pos6_unit = 'mm'
    _pos7_unit = 'mm'
    _pos8_unit = 'deg'
    _pos9_unit = 'deg'

    def __init__(self, filename=None, data_unit=''):
        """Initialize variables.

        Args:
            filename (str, optional): file full path.
            data_unit (str, optional): data unit.
        """
        self._pos1 = _np.array([])
        self._pos2 = _np.array([])
        self._pos3 = _np.array([])
        self._pos5 = _np.array([])
        self._pos6 = _np.array([])
        self._pos7 = _np.array([])
        self._pos8 = _np.array([])
        self._pos9 = _np.array([])
        self._sensorx = _np.array([])
        self._sensory = _np.array([])
        self._sensorz = _np.array([])
        self._data_unit = data_unit
        self._filename = None
        if filename is not None:
            self.read_file(filename)

    def __str__(self):
        """Printable string representation of Data."""
        fmtstr = '{0:<18s} : {1}\n'
        r = ''
        r += fmtstr.format('filename', str(self._filename))
        r += fmtstr.format('scan_axis', str(self.scan_axis))
        r += fmtstr.format('pos1[mm]', str(self._pos1))
        r += fmtstr.format('pos2[mm]', str(self._pos2))
        r += fmtstr.format('pos3[mm]', str(self._pos3))
        r += fmtstr.format('pos5[deg]', str(self._pos5))
        r += fmtstr.format('pos6[mm]', str(self._pos6))
        r += fmtstr.format('pos7[mm]', str(self._pos7))
        r += fmtstr.format('pos8[deg]', str(self._pos8))
        r += fmtstr.format('pos9[deg]', str(self._pos9))
        r += fmtstr.format('sensorx[%s]' % self._data_unit, str(self._sensorx))
        r += fmtstr.format('sensory[%s]' % self._data_unit, str(self._sensory))
        r += fmtstr.format('sensorz[%s]' % self._data_unit, str(self._sensorz))
        return r

    @property
    def axis_list(self):
        """List of all bench axes."""
        return self._axis_list

    @property
    def npts(self):
        """Number of data points."""
        if self.scan_axis is None:
            return 0
        else:
            npts = len(getattr(self, '_pos' + str(self.scan_axis)))
            v = [self._sensorx, self._sensory, self._sensorz]
            if all([vi.size == 0 for vi in v]):
                return 0
            for vi in v:
                if vi.size not in [npts, 0]:
                    return 0
            return npts

    @property
    def scan_axis(self):
        """Scan Axis."""
        pos = []
        for axis in self._axis_list:
            pos.append(getattr(self, '_pos' + str(axis)))
        if _np.count_nonzero([p.size > 1 for p in pos]) != 1:
            return None
        else:
            idx = _np.where([p.size > 1 for p in pos])[0][0]
            return self._axis_list[idx]

    @property
    def filename(self):
        """Name of the data file."""
        return self._filename

    def clear(self):
        """Clear Data."""
        self._filename = None
        for key in self.__dict__:
            if isinstance(self.__dict__[key], _np.ndarray):
                self.__dict__[key] = _np.array([])

    def copy(self):
        """Return a copy of the object."""
        _copy = Data()
        for key in self.__dict__:
            if isinstance(self.__dict__[key], _np.ndarray):
                _copy.__dict__[key] = _np.copy(self.__dict__[key])
            elif isinstance(self.__dict__[key], str):
                _copy.__dict__[key] = self.__dict__[key]
        return _copy

    def read_file(self, filename):
        """Read data from file.

        Args:
            filename (str): file full path.
        """
        self._filename = filename
        flines = _utils.read_file(filename)
        scan_axis = _utils.find_value(flines, 'scan_axis', int)

        for axis in self._axis_list:
            if axis != scan_axis:
                pos_str = 'pos' + str(axis)
                try:
                    pos = _utils.find_value(flines, pos_str, float)
                except ValueError:
                    pos = None
                pos = _np.around(_to_array(pos), decimals=_position_precision)
                setattr(self, '_' + pos_str, pos)

        idx = _utils.find_index(flines, '---------------------')

        for line in flines[10:idx-1]:
            line_split = line.split()
            setattr(self, line_split[0][:-1], line_split[1])

        data = []
        for line in flines[idx+1:]:
            data_line = [float(d) for d in line.split('\t')]
            data.append(data_line)
        data = _np.array(data)

        if data.shape[1] == 4:
            scan_positions = _np.around(
                _to_array(data[:, 0]), decimals=_position_precision)
            setattr(self, '_pos' + str(scan_axis), scan_positions)
            self._sensorx = _np.around(
                _to_array(data[:, 1]), decimals=_position_precision)
            self._sensory = _np.around(
                _to_array(data[:, 2]), decimals=_position_precision)
            self._sensorz = _np.around(
                _to_array(data[:, 3]), decimals=_position_precision)
        else:
            message = 'Inconsistent number of columns in file: %s' % filename
            raise MeasurementDataError(message)

    def reverse(self):
        """Reverse Data."""
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, _np.ndarray) and value.size > 1:
                self.__dict__[key] = value[::-1]

    def save_file(self, filename, extras={}):
        """Save data to file.

        Args:
            filename (str): file full path.
            extras (dict, optional): extra parameter names and values.
        """
        if self.scan_axis is None or self.npts == 0:
            raise MeasurementDataError('Invalid scan axis.')

        scan_axis = self.scan_axis
        timestamp = _utils.get_timestamp()

        pos1_str = '%f' % self._pos1[0] if self._pos1.size == 1 else '--'
        pos2_str = '%f' % self._pos2[0] if self._pos2.size == 1 else '--'
        pos3_str = '%f' % self._pos3[0] if self._pos3.size == 1 else '--'
        pos5_str = '%f' % self._pos5[0] if self._pos5.size == 1 else '--'
        pos6_str = '%f' % self._pos6[0] if self._pos6.size == 1 else '--'
        pos7_str = '%f' % self._pos7[0] if self._pos7.size == 1 else '--'
        pos8_str = '%f' % self._pos8[0] if self._pos8.size == 1 else '--'
        pos9_str = '%f' % self._pos9[0] if self._pos9.size == 1 else '--'

        scan_axis_unit = getattr(self, '_pos' + str(scan_axis) + '_unit')
        scan_axis_pos = getattr(self, '_pos' + str(scan_axis))

        columns_names = (
            'pos%i[%s]\t' % (scan_axis, scan_axis_unit) +
            'sensorx[V]\tsensory[V]\tsensorz[V]')

        npts = self.npts
        if len(self._sensorx) != npts:
            self._sensorx = _np.zeros(npts)

        if len(self._sensory) != npts:
            self._sensory = _np.zeros(npts)

        if len(self._sensorz) != npts:
            self._sensorz = _np.zeros(npts)

        columns = _np.column_stack((scan_axis_pos, self._sensorx,
                                    self._sensory, self._sensorz))

        self._filename = filename
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

        for key, value in extras.items():
            f.write('%s\t%s\n' % ((key + ':').ljust(19), str(value)))

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


class VoltageData(Data):
    """Position and voltage values."""

    def __init__(self, filename=None):
        """Initialize variables.

        Args:
            filename (str, optional): file full path.
        """
        super().__init__(filename=filename, data_unit='V')

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
    def sensorx(self):
        """Probe X Voltage [V]."""
        return self._sensorx

    @sensorx.setter
    def sensorx(self, value):
        self._sensorx = _to_array(value)

    @property
    def sensory(self):
        """Probe Y Voltage [V]."""
        return self._sensory

    @sensory.setter
    def sensory(self, value):
        self._sensory = _to_array(value)

    @property
    def sensorz(self):
        """Probe Z Voltage [V]."""
        return self._sensorz

    @sensorz.setter
    def sensorz(self, value):
        self._sensorz = _to_array(value)

    @property
    def scan_pos(self):
        """Scan positions."""
        return getattr(self, 'pos' + str(self.scan_axis))

    @scan_pos.setter
    def scan_pos(self, value):
        setattr(self, 'pos' + str(self.scan_axis), value)

    def copy(self):
        """Return a copy of the object."""
        _copy = VoltageData()
        for key in self.__dict__:
            if isinstance(self.__dict__[key], _np.ndarray):
                _copy.__dict__[key] = _np.copy(self.__dict__[key])
            elif isinstance(self.__dict__[key], str):
                _copy.__dict__[key] = self.__dict__[key]
        return _copy

    def read_file(self, filename):
        """Read voltage data from file.

        Args:
            filename (str): file full path.
        """
        super(VoltageData, self).read_file(filename)

    def save_file(self, filename):
        """Save voltage data to file.

        Args:
            filename (str): file full path.
        """
        super(VoltageData, self).save_file(filename)


class FieldData(Data):
    """Position and magnetic field values."""

    def __init__(self, filename=None):
        """Initialize variables.

        Args:
            filename (str, optional): file full path.
        """
        self._voltage_data_list = None
        self._probe_calibration = None
        super().__init__(filename=filename, data_unit='T')

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
    def pos5(self):
        """Position 5 (Axis +A) [deg]."""
        return self._pos5

    @property
    def pos6(self):
        """Position 6 (Axis +W) [mm]."""
        return self._pos6

    @property
    def pos7(self):
        """Position 7 (Axis +V) [mm]."""
        return self._pos7

    @property
    def pos8(self):
        """Position 8 (Axis +B) [deg]."""
        return self._pos8

    @property
    def pos9(self):
        """Position 9 (Axis +C) [deg]."""
        return self._pos9

    @property
    def sensorx(self):
        """Probe X Field [T]."""
        return self._sensorx

    @property
    def sensory(self):
        """Probe Y Field [T]."""
        return self._sensory

    @property
    def sensorz(self):
        """Probe Z Field [T]."""
        return self._sensorz

    @property
    def scan_pos(self):
        """Scan positions."""
        return getattr(self, 'pos' + str(self.scan_axis))

    @property
    def probe_calibration(self):
        """Calibration data."""
        return self._probe_calibration

    @probe_calibration.setter
    def probe_calibration(self, value):
        if isinstance(value, _calibration.ProbeCalibration):
            self._probe_calibration = value
        elif isinstance(value, str):
            if not _path.isfile(value):
                if self.filename is not None:
                    directory = _path.split(self.filename)[0]
                    value = _path.join(directory, _path.split(value)[1])
            self._probe_calibration = _calibration.ProbeCalibration(value)
        else:
            raise TypeError(
                'probe_calibration must be a ProbeCalibration object.')
        self._set_field_data()

    @property
    def voltage_data_list(self):
        """List of voltage data objects."""
        return self._voltage_data_list

    @voltage_data_list.setter
    def voltage_data_list(self, value):
        if isinstance(value, VoltageData):
            value = [value]

        if len(value) == 0:
            raise MeasurementDataError('Empty voltage data list.')

        if any([vd.scan_axis is None or vd.npts == 0 for vd in value]):
            raise MeasurementDataError('Invalid voltage data list.')

        if not all([vd.scan_axis == value[0].scan_axis for vd in value]):
            raise MeasurementDataError(
                'Inconsistent scan axis found in voltage data list.')

        if not all([vd.npts == value[0].npts for vd in value]):
            raise MeasurementDataError('Inconsistent number of points.')

        fixed_axes = [a for a in value[0].axis_list if a != value[0].scan_axis]
        for axis in fixed_axes:
            pos_set = set()
            for vd in value:
                pos_attr = getattr(vd, 'pos' + str(axis))
                if len(pos_attr) == 1:
                    pos_value = _np.around(
                        pos_attr[0], decimals=_check_position_precision)
                    pos_set.add(pos_value)
                else:
                    raise MeasurementDataError('Invalid position values.')
            if len(pos_set) != 1:
                raise MeasurementDataError('Inconsistent position values.')

        self._voltage_data_list = value
        self._set_field_data()

    def clear(self):
        """Clear FieldData."""
        self._voltage_data_list = None
        self._probe_calibration = None
        super(FieldData, self).clear()

    def copy(self):
        """Return a copy of the object."""
        _copy = FieldData()
        for key in self.__dict__:
            if isinstance(self.__dict__[key], _np.ndarray):
                _copy.__dict__[key] = _np.copy(self.__dict__[key])
            elif isinstance(self.__dict__[key], str):
                _copy.__dict__[key] = self.__dict__[key]
        return _copy

    def _set_field_data(self):
        """Set field data."""
        if self._probe_calibration is None or self._voltage_data_list is None:
            return

        npts = self._voltage_data_list[0].npts
        scan_axis = self._voltage_data_list[0].scan_axis
        interp_pos = _np.mean([
            vd.scan_pos for vd in self._voltage_data_list], axis=0)

        interp_list = []
        for vd in self._voltage_data_list:
            interpolated = vd.copy()
            interpolated.sensorx = _np.zeros(npts)
            interpolated.sensory = _np.zeros(npts)
            interpolated.sensorz = _np.zeros(npts)
            setattr(interpolated, 'pos' + str(scan_axis), interp_pos)

            if len(vd.sensorx) == npts:
                fr = _interpolate.splrep(vd.scan_pos, vd.sensorx, s=0, k=1)
                interpolated.sensorx = _interpolate.splev(
                    interp_pos, fr, der=0)

            if len(vd.sensory) == npts:
                fs = _interpolate.splrep(vd.scan_pos, vd.sensory, s=0, k=1)
                interpolated.sensory = _interpolate.splev(
                    interp_pos, fs, der=0)

            if len(vd.sensorz) == npts:
                ft = _interpolate.splrep(vd.scan_pos, vd.sensorz, s=0, k=1)
                interpolated.sensorz = _interpolate.splev(
                    interp_pos, ft, der=0)

            interp_list.append(interpolated)

        count = len(interp_list)
        avg = interp_list[0].copy()
        avg.sensorx = _np.zeros(npts)
        avg.sensory = _np.zeros(npts)
        avg.sensorz = _np.zeros(npts)
        for i in range(count):
            avg.sensorx += interp_list[i].sensorx
            avg.sensory += interp_list[i].sensory
            avg.sensorz += interp_list[i].sensorz
        avg.sensorx /= count
        avg.sensory /= count
        avg.sensorz /= count

        for axis in self._axis_list:
            pos = getattr(avg, 'pos' + str(axis))
            setattr(self, '_pos' + str(axis), pos)

        # Convert voltage to magnetic field
        bx = self.probe_calibration.sensorx.convert_voltage(avg.sensorx)
        by = self.probe_calibration.sensory.convert_voltage(avg.sensory)
        bz = self.probe_calibration.sensorz.convert_voltage(avg.sensorz)

        self._sensorx = bx
        self._sensory = by
        self._sensorz = bz

    def read_file(self, filename):
        """Read field data from file.

        Args:
            filename (str): file full path.
        """
        super(FieldData, self).read_file(filename)

    def save_file(self, filename):
        """Save field data to file.

        Args:
            filename (str): file full path.
        """
        if self.probe_calibration is None:
            raise MeasurementDataError('Invalid probe calibration.')

        save_probe_calibration_file = False
        dir_field_data = _path.split(filename)[0]

        probe_calibration_filename = self.probe_calibration.filename
        if probe_calibration_filename is None:
            save_probe_calibration_file = True
            timestamp = _utils.get_timestamp()
            probe_calibration_filename = timestamp + '_probe_calibration.txt'
        else:
            dir_probe_calibration = _path.split(probe_calibration_filename)[0]
            if dir_probe_calibration != dir_field_data:
                save_probe_calibration_file = True
            else:
                if not _path.isfile(probe_calibration_filename):
                    save_probe_calibration_file = True

        probe_calibration_filename = _path.split(probe_calibration_filename)[1]
        if save_probe_calibration_file:
            filepath = _path.join(dir_field_data, probe_calibration_filename)
            self.probe_calibration.save_file(filepath)

        super(FieldData, self).save_file(
            filename, extras={'probe_calibration': probe_calibration_filename})


class FieldMapData(object):
    """Map for position and magnetic field values."""

    def __init__(self, filename=None):
        """Initialize variables.

        Args:
            filename (str, optional): file full path.
        """
        self._header_info = []
        self._voltage_data_list = None
        self._field_data_list = None
        self._probe_calibration = None
        self._index_axis = None
        self._columns_axis = None
        self._field1 = None
        self._field2 = None
        self._field3 = None
        self._pos1 = _np.array([])
        self._pos2 = _np.array([])
        self._pos3 = _np.array([])
        self._data_is_set = False
        self._correct_sensor_displacement = True
        self._filename = None
        if filename is not None:
            self.read_file(filename)

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

    @property
    def header_info(self):
        """Field map header lines."""
        return self._header_info

    @header_info.setter
    def header_info(self, value):
        if not isinstance(value, list):
            raise TypeError('header_info must be a list.')
        if len(value) != 0:
            if not all(isinstance(line, (tuple, list)) for line in value):
                raise MeasurementDataError('Invalid value for header_info.')
            elif any(len(line) != 2 for line in value):
                raise MeasurementDataError('Invalid value for header_info.')
        self._header_info = [(l[0], l[1]) for l in value]

    @property
    def probe_calibration(self):
        """Probe calibration data."""
        return self._probe_calibration

    @probe_calibration.setter
    def probe_calibration(self, value):
        if self._data_is_set:
            raise MeasurementDataError(
                "Can't overwrite probe calibration data.")

        if isinstance(value, _calibration.ProbeCalibration):
            self._probe_calibration = value
        elif isinstance(value, str):
            self._probe_calibration = _calibration.ProbeCalibration(value)
        else:
            raise TypeError(
                'probe_calibration must be a ProbeCalibration object.')

        self._set_field_data_list()

    @property
    def voltage_data_list(self):
        """List of VoltageData objects."""
        return self._voltage_data_list

    @voltage_data_list.setter
    def voltage_data_list(self, value):
        if isinstance(value, VoltageData):
            value = [value]

        if (isinstance(value, FieldData)
           or not all(isinstance(vd, VoltageData) for vd in value)):
            raise TypeError(
                'voltage_data_list must be a list of VoltageData objects.')

        if len(value) == 0:
            raise MeasurementDataError('Empty voltage list.')

        if any([vd.scan_axis is None or vd.npts == 0 for vd in value]):
            raise MeasurementDataError('Invalid voltage list.')

        if not all([vd.scan_axis == value[0].scan_axis for vd in value]):
            raise MeasurementDataError('Inconsistent scan axis.')

        def _sorted_key(x):
            pos = []
            for axis in x.axis_list:
                if axis != x.scan_axis:
                    pos.append(getattr(x, 'pos' + str(axis)))
            return pos

        value = sorted(value, key=_sorted_key)
        self._voltage_data_list = value
        self._set_field_data_list()

    @property
    def field_data_list(self):
        """List of FieldData objects."""
        return self._field_data_list

    @field_data_list.setter
    def field_data_list(self, value):
        if isinstance(value, FieldData):
            value = [value]

        if (isinstance(value, VoltageData)
           or not all(isinstance(fd, FieldData) for fd in value)):
            raise TypeError(
                'field_data_list must be a list of FieldData objects.')

        if len(value) == 0:
            raise MeasurementDataError('Empty field data list.')

        if not all([isinstance(fd, FieldData) for fd in value]):
            raise MeasurementDataError('Invalid field data list.')

        if any([fd.scan_axis is None or fd.npts == 0 for fd in value]):
            raise MeasurementDataError('Invalid field data list.')

        if not all([fd.scan_axis == value[0].scan_axis for fd in value]):
            raise MeasurementDataError(
                'Inconsistent scan axis in field data list.')

        if not all([fd.probe_calibration == value[0].probe_calibration
                    for fd in value]):
            raise MeasurementDataError(
                'Inconsistent probe calibration in field data list.')

        self._field_data_list = value
        self._probe_calibration = value[0].probe_calibration
        self._index_axis = value[0].scan_axis
        self._set_columns_axis()
        self._set_field_map_data()

    @property
    def correct_sensor_displacement(self):
        """Sensor displacement correction flag."""
        return self._correct_sensor_displacement

    @correct_sensor_displacement.setter
    def correct_sensor_displacement(self, value):
        if self._data_is_set:
            raise MeasurementDataError("Can't overwrite flag value.")

        if isinstance(value, bool):
            self._correct_sensor_displacement = value
        else:
            raise TypeError('correct_sensor_displacement must be a boolean.')

    @property
    def filename(self):
        """Name of the field map file."""
        return self._filename

    def clear(self):
        """Clear."""
        self._header_info = []
        self._voltage_data_list = None
        self._field_data_list = None
        self._probe_calibration = None
        self._index_axis = None
        self._columns_axis = None
        self._field1 = None
        self._field2 = None
        self._field3 = None
        self._pos1 = _np.array([])
        self._pos2 = _np.array([])
        self._pos3 = _np.array([])
        self._data_is_set = False
        self._correct_sensor_displacement = True
        self._filename = None

    def _get_axis_position_dict(self):
        if self.field_data_list is None:
            return {}

        _dict = {}
        for axis in self.field_data_list[0].axis_list:
            pos = set()
            for field_data in self.field_data_list:
                p = _np.around(getattr(field_data, 'pos' + str(axis)),
                               decimals=_check_position_precision)
                pos.update(p)
            _dict[axis] = sorted(list(pos))
        return _dict

    def _sensor_displacement_correction(self, fieldx, fieldy, fieldz):
        if (self._index_axis is None or self._probe_calibration is None):
            return None, None, None

        index = fieldy.index
        columns = fieldy.columns

        # shift field data
        fieldx.index = self._probe_calibration.corrected_position(
            self._index_axis, index, 'x')
        fieldz.index = self._probe_calibration.corrected_position(
            self._index_axis, index, 'z')
        nbeg, nend = _get_number_of_cuts(
            fieldx.index, fieldy.index, fieldz.index)

        # interpolate field data
        fieldx, fieldy, fieldz = _interpolate_data_frames(
            fieldx, fieldy, fieldz, axis=0)

        # cut field data
        fieldx, fieldy, fieldz = _cut_data_frames(
            fieldx, fieldy, fieldz, nbeg, nend)

        if self._columns_axis is not None:
            # shift field data
            fieldx.columns = self._probe_calibration.corrected_position(
                self._columns_axis, columns, 'x')
            fieldz.columns = self._probe_calibration.corrected_position(
                self._columns_axis, columns, 'z')
            nbeg, nend = _get_number_of_cuts(
                fieldx.columns, fieldy.columns, fieldz.columns)

            # interpolate field data
            fieldx, fieldy, fieldz = _interpolate_data_frames(
                fieldx, fieldy, fieldz, axis=1)

            # cut field data
            fieldx, fieldy, fieldz = _cut_data_frames(
                fieldx, fieldy, fieldz, nbeg, nend, axis=1)

        return fieldx, fieldy, fieldz

    def _set_columns_axis(self):
        _dict = self._get_axis_position_dict()

        columns_axis = []
        for key, value in _dict.items():
            if key != self._index_axis and len(value) > 1:
                columns_axis.append(key)

        if len(columns_axis) > 1:
            raise MeasurementDataError('Invalid number of measurement axes.')

        if len(columns_axis) == 1:
            self._columns_axis = columns_axis[0]

    def _set_field_data_list(self):
        if (self._voltage_data_list is not None
           and self._probe_calibration is not None):
            fixed_axes = [
                a for a in self._voltage_data_list[0].axis_list
                if a != self._voltage_data_list[0].scan_axis]

            field_data_list = []
            tmp_voltage_list = []
            prev_pos = None
            for vd in self._voltage_data_list:
                pos = []
                for axis in fixed_axes:
                    p = _np.around(getattr(vd, 'pos' + str(axis)),
                                   decimals=_check_position_precision)
                    pos.append(p)
                if (prev_pos is None or all([
                    pos[i] == prev_pos[i] for i in range(len(pos)) if not (
                        len(pos[i]) == 0 and len(prev_pos[i]) == 0)])):
                    tmp_voltage_list.append(vd)
                else:
                    field_data = FieldData()
                    field_data.probe_calibration = self.probe_calibration
                    field_data.voltage_data_list = tmp_voltage_list
                    field_data_list.append(field_data)
                    tmp_voltage_list = [vd]
                prev_pos = pos

            field_data = FieldData()
            field_data.probe_calibration = self._probe_calibration
            field_data.voltage_data_list = tmp_voltage_list
            field_data_list.append(field_data)
            self.field_data_list = field_data_list

    def _set_field_map_data(self):
        if self._field_data_list is not None:
            dfx = []
            dfy = []
            dfz = []
            for fd in self._field_data_list:
                index = getattr(fd, 'pos' + str(self._index_axis))
                if self._columns_axis is not None:
                    columns = getattr(fd, 'pos' + str(self._columns_axis))
                else:
                    columns = [0]

                index = _pd.Index(index, float)
                columns = _pd.Index(columns, float)
                dfx.append(_pd.DataFrame(
                    fd.sensorx, index=index, columns=columns))
                dfy.append(_pd.DataFrame(
                    fd.sensory, index=index, columns=columns))
                dfz.append(_pd.DataFrame(
                    fd.sensorz, index=index, columns=columns))

            fieldx = _pd.concat(dfx, axis=1)
            fieldy = _pd.concat(dfy, axis=1)
            fieldz = _pd.concat(dfz, axis=1)

            index = fieldx.index
            columns = fieldx.columns
            if (len(columns) != len(columns.drop_duplicates())):
                msg = 'Duplicate position found in field data list.'
                raise MeasurementDataError(msg)

            if self.correct_sensor_displacement:
                fieldx, fieldy, fieldz = self._sensor_displacement_correction(
                    fieldx, fieldy, fieldz)

            # update position values
            index = fieldx.index
            columns = fieldx.columns
            _dict = self._get_axis_position_dict()
            pos_sorted = [_dict[1], _dict[2], _dict[3]]
            pos_sorted[self._index_axis - 1] = index.values
            if self._columns_axis is not None:
                pos_sorted[self._columns_axis - 1] = columns.values

            self._pos3 = _np.array(pos_sorted[2])  # x-axis
            self._pos2 = _np.array(pos_sorted[1])  # y-axis
            self._pos1 = _np.array(pos_sorted[0])  # z-axis

            field3, field2, field1 = (
                self._probe_calibration.field_in_bench_coordinate_system(
                    fieldx, fieldy, fieldz))

            self._field1 = field1
            self._field2 = field2
            self._field3 = field3
            self._data_is_set = True

    def _get_field_at_point(self, pos):
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

    def get_field_map(self):
        """Return the field map array."""
        field_map = []
        for p1 in self._pos1:
            for p2 in self._pos2:
                for p3 in self._pos3:
                    p = [p3, p2, p1]
                    b = self._get_field_at_point(p)
                    field_map.append(_np.append(p, b))
        field_map = _np.array(field_map)
        return field_map

    def get_transformed_field_map(self, magnet_center=[0, 0, 0],
                                  magnet_x_axis=3, magnet_y_axis=2):
        """Return the transformed field map array."""
        tm = _get_transformation_matrix(magnet_x_axis, magnet_y_axis)

        field_map = []
        for p1 in self._pos1:
            for p2 in self._pos2:
                for p3 in self._pos3:
                    p = [p3, p2, p1]
                    b = self._get_field_at_point(p)
                    tp = _change_coordinate_system(p, tm, magnet_center)
                    tb = _change_coordinate_system(b, tm)
                    field_map.append(_np.append(tp, tb))
        field_map = _np.array(field_map)

        field_map = _np.array(sorted(
            field_map, key=lambda x: (x[2], x[1], x[0])))

        return field_map

    def read_file(self, filename):
        """Read field map file.

        Args:
            filename (str): field map file path.
        """
        self._filename = filename
        flines = _utils.read_file(filename)
        idx = _utils.find_index(flines, '-------------------------')

        for line in flines[:(idx-1)]:
            line_split = line.split()
            name = line_split[0].replace(':', '')
            value = ' '.join(line_split[1:])
            self._header_info.append((name, value))

        data = []
        for line in flines[idx+1:]:
            data_line = [float(d) for d in line.split('\t')]
            data.append(data_line)
        data = _np.array(data)

        pos3 = _np.unique(data[:, 0])
        pos2 = _np.unique(data[:, 1])
        pos1 = _np.unique(data[:, 2])

        npts3 = len(pos3)
        npts2 = len(pos2)
        npts1 = len(pos1)

        measurement_axes = []
        if npts3 > 1:
            measurement_axes.append(3)
        if npts2 > 1:
            measurement_axes.append(2)
        if npts1 > 1:
            measurement_axes.append(1)

        if len(measurement_axes) > 2 or len(measurement_axes) == 0:
            raise MeasurementDataError('Invalid field map file: %s' % filename)

        field3 = data[:, 3]
        field2 = data[:, 4]
        field1 = data[:, 5]

        index = locals()['pos' + str(measurement_axes[0])]
        field3.shape = (-1, len(index))
        field2.shape = (-1, len(index))
        field1.shape = (-1, len(index))

        if len(measurement_axes) == 2:
            columns = locals()['pos' + str(measurement_axes[1])]
        else:
            columns = [0]

        self._pos3 = pos3
        self._pos2 = pos2
        self._pos1 = pos1
        self._field3 = _pd.DataFrame(
            _np.transpose(field3), index=index, columns=columns)
        self._field2 = _pd.DataFrame(
            _np.transpose(field2), index=index, columns=columns)
        self._field1 = _pd.DataFrame(
            _np.transpose(field1), index=index, columns=columns)
        self._index_axis = measurement_axes[0]
        if len(measurement_axes) == 2:
            self._columns_axis = measurement_axes[1]

        self._data_is_set = True

    def save_file(self, filename, magnet_center=[0, 0, 0],
                  magnet_x_axis=3, magnet_y_axis=2):
        """Save field map file.

        Args:
            filename (str): field map file path.
            magnet_center (list): center position of the magnet.
            magnet_x_axis (str): magnet x-axis direction.
                                 [3, -3, 2, -2, 1 or -1]
            magnet_y_axis (str): magnet y-axis direction.
                                 [3, -3, 2, -2, 1 or -1]
        """
        atts = [self._pos1, self._pos2, self._pos3,
                self._field1, self._field2, self._field3]
        if any([att is None for att in atts]):
            message = "Invalid field data."
            raise MeasurementDataError(message)

        self._filename = filename
        f = open(filename, 'w')

        for line in self.header_info:
            variable = (str(line[0]) + ':').ljust(20)
            value = str(line[1])
            f.write('{0:1s}\t{1:1s}\n'.format(variable, value))

        if len(self.header_info) != 0:
            f.write('\n')

        f.write('X[mm]\tY[mm]\tZ[mm]\tBx[T]\tBy[T]\tBz[T]\n')
        f.write('-----------------------------------------------' +
                '----------------------------------------------\n')

        field_map = self.get_transformed_field_map(
            magnet_center, magnet_x_axis, magnet_y_axis)

        for i in range(field_map.shape[0]):
            f.write('{0:0.3f}\t{1:0.3f}\t{2:0.3f}\t'.format(
                field_map[i, 0], field_map[i, 1], field_map[i, 2]))
            f.write('{0:0.10e}\t{1:0.10e}\t{2:0.10e}\n'.format(
                field_map[i, 3], field_map[i, 4], field_map[i, 5]))
        f.close()


def _to_array(value):
    if value is not None:
        if not isinstance(value, _np.ndarray):
            value = _np.array(value)
        if len(value.shape) == 0:
            value = _np.array([value])
    else:
        value = _np.array([])
    return value


def _interpolate_data_frames(dfx, dfy, dfz, axis=0):

    def _interpolate_vec(x, pos):
        f = _interpolate.splrep(x.index, x.values, s=0, k=1)
        return _interpolate.splev(pos, f, der=0)

    if axis == 1:
        pos = dfy.columns
        interp_dfx = dfx.apply(_interpolate_vec, axis=axis, args=[pos])
        interp_dfz = dfz.apply(_interpolate_vec, axis=axis, args=[pos])
        interp_dfx.columns = pos
        interp_dfz.columns = pos
    else:
        pos = dfy.index
        interp_dfx = dfx.apply(_interpolate_vec, args=[pos])
        interp_dfz = dfz.apply(_interpolate_vec, args=[pos])
        interp_dfx.index = pos
        interp_dfz.index = pos

    return interp_dfx, dfy, interp_dfz


def _get_number_of_cuts(px, py, pz):
    nbeg = max(len(_np.where(px < py[0])[0]), len(_np.where(pz < py[0])[0]))
    nend = max(len(_np.where(px > py[-1])[0]), len(_np.where(pz > py[-1])[0]))
    return nbeg, nend


def _cut_data_frames(dfx, dfy, dfz, nbeg, nend, axis=0):
    if axis == 1:
        if nbeg > 0:
            dfx = dfx.drop(dfx.columns[:nbeg], axis=axis)
            dfy = dfy.drop(dfy.columns[:nbeg], axis=axis)
            dfz = dfz.drop(dfz.columns[:nbeg], axis=axis)
        if nend > 0:
            dfx = dfx.drop(dfx.columns[-nend:], axis=axis)
            dfy = dfy.drop(dfy.columns[-nend:], axis=axis)
            dfz = dfz.drop(dfz.columns[-nend:], axis=axis)
    else:
        if nbeg > 0:
            dfx = dfx.drop(dfx.index[:nbeg])
            dfy = dfy.drop(dfy.index[:nbeg])
            dfz = dfz.drop(dfz.index[:nbeg])
        if nend > 0:
            dfx = dfx.drop(dfx.index[-nend:])
            dfy = dfy.drop(dfy.index[-nend:])
            dfz = dfz.drop(dfz.index[-nend:])
    return dfx, dfy, dfz


def _get_transformation_matrix(axis_x, axis_y):
    if abs(axis_x) == (axis_y):
        raise MeasurementDataError('Inconsistent magnet axes.')
    v3 = [1, 0, 0]
    v2 = [0, 1, 0]
    v1 = [0, 0, 1]
    vs = [v1, v2, v3]

    x = vs[abs(axis_x) - 1]
    if abs(axis_x) != axis_x:
        x = [(-1)*i for i in x]

    y = vs[abs(axis_y) - 1]
    if abs(axis_y) != axis_y:
        y = [(-1)*i for i in y]

    z = _np.cross(x, y)
    m = _np.outer(v3, x) + _np.outer(v2, y) + _np.outer(v1, z)
    return m


def _change_coordinate_system(vector, transf_matrix, center=[0, 0, 0]):
    vector = _np.array(vector)
    center = _np.array(center)
    transf_vector = _np.dot(transf_matrix, vector - center)
    return transf_vector
