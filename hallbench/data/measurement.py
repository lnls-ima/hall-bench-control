# -*- coding: utf-8 -*-

"""Implementation of classes to store and analyse measurement data."""

import os as _os
import numpy as _np
import pandas as _pd
import collections as _collections
from scipy import interpolate as _interpolate

from . import utils as _utils
from . import calibration as _calibration
from . import database as _database


_position_precision = 4
_check_position_precision = 3
_empty_str = '--'
_measurements_label = 'Hall'


class MeasurementDataError(Exception):
    """Measurement data exception."""

    def __init__(self, message, *args):
        """Initialize variables."""
        self.message = message


class Scan(_database.DatabaseObject):
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
    _label = ''
    _db_table = ''
    _db_dict = {}
    _db_json_str = []

    def __init__(self, filename=None, database=None, idn=None, data_unit=""):
        """Initialize variables.

        Args:
            filename (str, optional): file full path.
            database (str): database file path.
            idn (int): id in database table.
            data_unit (str, optional): data unit.
        """
        self._magnet_name = None
        self._main_current = None
        self._timestamp = None
        self._configuration_id = None
        self._pos1 = _np.array([])
        self._pos2 = _np.array([])
        self._pos3 = _np.array([])
        self._pos5 = _np.array([])
        self._pos6 = _np.array([])
        self._pos7 = _np.array([])
        self._pos8 = _np.array([])
        self._pos9 = _np.array([])
        self._avgx = _np.array([])
        self._avgy = _np.array([])
        self._avgz = _np.array([])
        self._stdx = _np.array([])
        self._stdy = _np.array([])
        self._stdz = _np.array([])
        self._data_unit = data_unit
        self._current = {}
        self._temperature = {}

        if filename is not None and idn is not None:
            raise ValueError('Invalid arguments for Scan object.')

        if idn is not None and database is not None:
            self.read_from_database(database, idn)

        if filename is not None:
            self.read_file(filename)

    def __str__(self):
        """Printable string representation of Scan."""
        fmtstr = '{0:<18s} : {1}\n'
        r = ''
        r += fmtstr.format('magnet_name', str(self.magnet_name))
        r += fmtstr.format('main_current[A]', str(self.main_current))
        r += fmtstr.format('timestamp', str(self._timestamp))
        r += fmtstr.format('scan_axis', str(self.scan_axis))
        r += fmtstr.format('npts', str(self.npts))
        r += fmtstr.format('pos1[mm]', str(self._pos1))
        r += fmtstr.format('pos2[mm]', str(self._pos2))
        r += fmtstr.format('pos3[mm]', str(self._pos3))
        r += fmtstr.format('pos5[deg]', str(self._pos5))
        r += fmtstr.format('pos6[mm]', str(self._pos6))
        r += fmtstr.format('pos7[mm]', str(self._pos7))
        r += fmtstr.format('pos8[deg]', str(self._pos8))
        r += fmtstr.format('pos9[deg]', str(self._pos9))
        r += fmtstr.format('avgx[%s]' % self._data_unit, str(self._avgx))
        r += fmtstr.format('avgy[%s]' % self._data_unit, str(self._avgy))
        r += fmtstr.format('avgz[%s]' % self._data_unit, str(self._avgz))
        r += fmtstr.format('stdx[%s]' % self._data_unit, str(self._stdx))
        r += fmtstr.format('stdy[%s]' % self._data_unit, str(self._stdy))
        r += fmtstr.format('stdz[%s]' % self._data_unit, str(self._stdz))
        return r

    @classmethod
    def get_configuration_id_from_database(cls, database, idn):
        """Return the configuration ID of the database record."""
        configuration_id = cls.get_database_param(
            database, idn, 'configuration_id')
        return configuration_id

    @property
    def unit(self):
        """Return the data unit."""
        return self._data_unit

    @property
    def magnet_name(self):
        """Return the magnet name."""
        return self._magnet_name

    @magnet_name.setter
    def magnet_name(self, value):
        if value is not None:
            if len(value) == 0 or value == _empty_str:
                value = None
        self._magnet_name = value

    @property
    def main_current(self):
        """Return the main current."""
        return self._main_current

    @main_current.setter
    def main_current(self, value):
        if value is not None:
            if len(value) == 0 or value == _empty_str:
                value = None
        self._main_current = value

    @property
    def configuration_id(self):
        """Return the measurement configuration ID."""
        return self._configuration_id

    @configuration_id.setter
    def configuration_id(self, value):
        if value is None:
            self._configuration_id = value
        elif isinstance(value, int):
            self._configuration_id = value
        else:
            raise MeasurementDataError('Invalid value for configuration_id.')

    @property
    def timestamp(self):
        """Return the timestamp."""
        return self._timestamp

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
    def avgx(self):
        """Probe X Average Values."""
        return self._avgx

    @property
    def avgy(self):
        """Probe Y Average Values."""
        return self._avgy

    @property
    def avgz(self):
        """Probe Z Average Values."""
        return self._avgz

    @property
    def stdx(self):
        """Probe X STD Values."""
        return self._stdx

    @property
    def stdy(self):
        """Probe Y STD Values."""
        return self._stdy

    @property
    def stdz(self):
        """Probe Z STD Values."""
        return self._stdz

    @property
    def current(self):
        """Power supply current."""
        return self._current

    @property
    def temperature(self):
        """Temperature readings."""
        return self._temperature

    @property
    def axis_list(self):
        """List of all bench axes."""
        return self._axis_list

    @property
    def npts(self):
        """Return the number of data points."""
        if self.scan_axis is None:
            return 0
        else:
            npts = len(getattr(self, '_pos' + str(self.scan_axis)))
            v = [self._avgx, self._avgy, self._avgz]
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
    def scan_pos(self):
        """Scan positions."""
        return getattr(self, 'pos' + str(self.scan_axis))

    @scan_pos.setter
    def scan_pos(self, value):
        setattr(self, 'pos' + str(self.scan_axis), value)

    @property
    def default_filename(self):
        """Return the default filename."""
        label = self._label + '_' + _measurements_label
        if self.magnet_name is not None and len(self.magnet_name) != 0:
            name = self.magnet_name + '_' + label
        else:
            name = label

        if self.npts != 0:
            if self.pos1.size == 1:
                name = name + '_pos1={0:.0f}mm'.format(self.pos1[0])
            if self.pos2.size == 1:
                name = name + '_pos2={0:.0f}mm'.format(self.pos2[0])
            if self.pos3.size == 1:
                name = name + '_pos3={0:.0f}mm'.format(self.pos3[0])

        self._timestamp = _utils.get_timestamp()
        filename = '{0:1s}_{1:1s}.dat'.format(self._timestamp, name)
        return filename

    def clear(self):
        """Clear Scan."""
        self._magnet_name = None
        self._main_current = None
        self._timestamp = None
        self._configuration_id = None
        for key in self.__dict__:
            if isinstance(self.__dict__[key], _np.ndarray):
                self.__dict__[key] = _np.array([])

    def copy(self):
        """Return a copy of the object."""
        _copy = type(self)()
        for key in self.__dict__:
            if isinstance(self.__dict__[key], _np.ndarray):
                _copy.__dict__[key] = _np.copy(self.__dict__[key])
            elif isinstance(self.__dict__[key], dict):
                _copy.__dict__[key] = dict(self.__dict__[key])
            else:
                _copy.__dict__[key] = self.__dict__[key]
        return _copy

    def read_file(self, filename):
        """Read data from file.

        Args:
            filename (str): file full path.
        """
        flines = _utils.read_file(filename)
        self._timestamp = _utils.find_value(flines, 'timestamp')
        self.magnet_name = _utils.find_value(flines, 'magnet_name')
        self.main_current = _utils.find_value(flines, 'main_current')

        scan_axis = _utils.find_value(flines, 'scan_axis', int)
        for axis in self._axis_list:
            if axis != scan_axis:
                pos_str = 'pos' + str(axis)
                try:
                    pos = _utils.find_value(flines, pos_str, float)
                except ValueError:
                    pos = None
                pos = _np.around(
                    _utils.to_array(pos), decimals=_position_precision)
                setattr(self, '_' + pos_str, pos)

        idx = _utils.find_index(flines, '---------------------')
        data = []
        for line in flines[idx+1:]:
            data_line = [float(d) for d in line.split('\t')]
            data.append(data_line)
        data = _np.array(data)

        dshape = data.shape[1]
        if dshape in [4, 7]:
            scan_positions = _np.around(
                _utils.to_array(data[:, 0]), decimals=_position_precision)
            setattr(self, '_pos' + str(scan_axis), scan_positions)
            self._avgx = _utils.to_array(data[:, 1])
            self._avgy = _utils.to_array(data[:, 2])
            self._avgz = _utils.to_array(data[:, 3])
            if dshape == 7:
                self._stdx = _utils.to_array(data[:, 4])
                self._stdy = _utils.to_array(data[:, 5])
                self._stdz = _utils.to_array(data[:, 6])
            else:
                self._stdx = _np.zeros(len(scan_positions))
                self._stdy = _np.zeros(len(scan_positions))
                self._stdz = _np.zeros(len(scan_positions))
        else:
            message = 'Inconsistent number of columns in file: %s' % filename
            raise MeasurementDataError(message)

    def reverse(self):
        """Reverse Scan."""
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, _np.ndarray) and value.size > 1:
                self.__dict__[key] = value[::-1]

    def save_file(self, filename, include_std=True):
        """Save data to file.

        Args:
            filename (str): file full path.
            include_std (bool, optional): save std values to file.
        """
        if self.scan_axis is None:
            raise MeasurementDataError('Invalid scan axis.')

        if self.npts == 0:
            raise MeasurementDataError('Invalid scan data.')

        scan_axis = self.scan_axis
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
        npts = self.npts

        columns_names = (
            'pos%i[%s]\t' % (scan_axis, scan_axis_unit) +
            'avgx[{0:s}]\t\tavgy[{0:s}]\t\tavgz[{0:s}]\t\t'.format(
                self._data_unit))

        avgx = self._avgx if len(self._avgx) == npts else _np.zeros(npts)
        avgy = self._avgy if len(self._avgy) == npts else _np.zeros(npts)
        avgz = self._avgz if len(self._avgz) == npts else _np.zeros(npts)

        if include_std:
            columns_names = (
                columns_names +
                'stdx[{0:s}]\t\tstdy[{0:s}]\t\tstdz[{0:s}]'.format(
                    self._data_unit))

            stdx = self._stdx if len(self._stdx) == npts else _np.zeros(npts)
            stdy = self._stdy if len(self._stdy) == npts else _np.zeros(npts)
            stdz = self._stdz if len(self._stdz) == npts else _np.zeros(npts)

            columns = _np.column_stack(
                (scan_axis_pos, avgx, avgy, avgz, stdx, stdy, stdz))
        else:
            columns = _np.column_stack((scan_axis_pos, avgx, avgy, avgz))

        if self._timestamp is None:
            self._timestamp = _utils.get_timestamp()

        magnet_name = (
            self.magnet_name if self.magnet_name is not None
            else _empty_str)
        main_current = (
            self.main_current if self.main_current is not None
            else _empty_str)

        with open(filename, mode='w') as f:
            f.write('timestamp:         \t%s\n' % self._timestamp)
            f.write('magnet_name:       \t%s\n' % magnet_name)
            f.write('main_current[A]:   \t%s\n' % main_current)
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
            f.write('-----------------------------------------------------' +
                    '-----------------------------------------------------\n')
            for i in range(columns.shape[0]):
                line = '{0:+0.4f}'.format(columns[i, 0])
                for j in range(1, columns.shape[1]):
                    line = line + '\t' + '{0:+0.10e}'.format(columns[i, j])
                f.write(line + '\n')


class VoltageScan(Scan):
    """Position and voltage values."""

    _label = 'VoltageScan'
    _db_table = 'voltage_scans'
    _db_dict = _collections.OrderedDict([
        ('id', [None, 'INTEGER NOT NULL']),
        ('date', [None, 'TEXT NOT NULL']),
        ('hour', [None, 'TEXT NOT NULL']),
        ('magnet_name', ['magnet_name', 'TEXT']),
        ('main_current', ['main_current', 'TEXT']),
        ('configuration_id', ['configuration_id', 'INTEGER']),
        ('scan_axis', ['scan_axis', 'INTEGER']),
        ('pos1', ['_pos1', 'TEXT NOT NULL']),
        ('pos2', ['_pos2', 'TEXT NOT NULL']),
        ('pos3', ['_pos3', 'TEXT NOT NULL']),
        ('pos5', ['_pos5', 'TEXT NOT NULL']),
        ('pos6', ['_pos6', 'TEXT NOT NULL']),
        ('pos7', ['_pos7', 'TEXT NOT NULL']),
        ('pos8', ['_pos8', 'TEXT NOT NULL']),
        ('pos9', ['_pos9', 'TEXT NOT NULL']),
        ('voltagex_avg', ['_avgx', 'TEXT NOT NULL']),
        ('voltagey_avg', ['_avgy', 'TEXT NOT NULL']),
        ('voltagez_avg', ['_avgz', 'TEXT NOT NULL']),
        ('voltagex_std', ['_stdx', 'TEXT NOT NULL']),
        ('voltagey_std', ['_stdy', 'TEXT NOT NULL']),
        ('voltagez_std', ['_stdz', 'TEXT NOT NULL']),
        ('current', ['_current', 'TEXT NOT NULL']),
        ('temperature', ['_temperature', 'TEXT NOT NULL']),
    ])
    _db_json_str = [
        '_pos1', '_pos2', '_pos3', '_pos5',
        '_pos6', '_pos7', '_pos8', '_pos9',
        '_avgx', '_avgy', '_avgz',
        '_stdx', '_stdy', '_stdz',
        '_current', '_temperature',
        ]

    def __init__(self, filename=None, database=None, idn=None):
        """Initialize variables.

        Args:
            filename (str, optional): file full path.
            database (str): database file path.
            idn (int): id in database table.
        """
        super().__init__(
            filename=filename, database=database, idn=idn, data_unit='V')

    @Scan.pos1.setter
    def pos1(self, value):
        """Set pos1 value."""
        self._pos1 = _np.around(
            _utils.to_array(value), decimals=_position_precision)

    @Scan.pos2.setter
    def pos2(self, value):
        """Set pos2 value."""
        self._pos2 = _np.around(
            _utils.to_array(value), decimals=_position_precision)

    @Scan.pos3.setter
    def pos3(self, value):
        """Set pos3 value."""
        self._pos3 = _np.around(
            _utils.to_array(value), decimals=_position_precision)

    @Scan.pos5.setter
    def pos5(self, value):
        """Set pos5 value."""
        self._pos5 = _np.around(
            _utils.to_array(value), decimals=_position_precision)

    @Scan.pos6.setter
    def pos6(self, value):
        """Set pos6 value."""
        self._pos6 = _np.around(
            _utils.to_array(value), decimals=_position_precision)

    @Scan.pos7.setter
    def pos7(self, value):
        """Set pos7 value."""
        self._pos7 = _np.around(
            _utils.to_array(value), decimals=_position_precision)

    @Scan.pos8.setter
    def pos8(self, value):
        """Set pos8 value."""
        self._pos8 = _np.around(
            _utils.to_array(value), decimals=_position_precision)

    @Scan.pos9.setter
    def pos9(self, value):
        """Set pos9 value."""
        self._pos9 = _np.around(
            _utils.to_array(value), decimals=_position_precision)

    @Scan.avgx.setter
    def avgx(self, value):
        """Set avgx value."""
        self._avgx = _utils.to_array(value)

    @Scan.avgy.setter
    def avgy(self, value):
        """Set avgy value."""
        self._avgy = _utils.to_array(value)

    @Scan.avgz.setter
    def avgz(self, value):
        """Set avgz value."""
        self._avgz = _utils.to_array(value)

    @Scan.stdx.setter
    def stdx(self, value):
        """Set stdx value."""
        self._stdx = _utils.to_array(value)

    @Scan.stdy.setter
    def stdy(self, value):
        """Set stdy value."""
        self._stdy = _utils.to_array(value)

    @Scan.stdz.setter
    def stdz(self, value):
        """Set stdz value."""
        self._stdz = _utils.to_array(value)

    @Scan.current.setter
    def current(self, value):
        """Set current values."""
        if not isinstance(value, dict):
            raise TypeError('current must be a dict.')
        self._current = value

    @Scan.temperature.setter
    def temperature(self, value):
        """Set temperature values."""
        if not isinstance(value, dict):
            raise TypeError('temperature must be a dict.')
        self._temperature = value


class FieldScan(Scan):
    """Position and magnetic field values."""

    _label = 'FieldScan'
    _db_table = 'field_scans'
    _db_dict = _collections.OrderedDict([
        ('id', [None, 'INTEGER NOT NULL']),
        ('date', [None, 'TEXT NOT NULL']),
        ('hour', [None, 'TEXT NOT NULL']),
        ('magnet_name', ['magnet_name', 'TEXT']),
        ('main_current', ['main_current', 'TEXT']),
        ('configuration_id', ['configuration_id', 'INTEGER']),
        ('scan_axis', ['scan_axis', 'INTEGER']),
        ('pos1', ['_pos1', 'TEXT NOT NULL']),
        ('pos2', ['_pos2', 'TEXT NOT NULL']),
        ('pos3', ['_pos3', 'TEXT NOT NULL']),
        ('pos5', ['_pos5', 'TEXT NOT NULL']),
        ('pos6', ['_pos6', 'TEXT NOT NULL']),
        ('pos7', ['_pos7', 'TEXT NOT NULL']),
        ('pos8', ['_pos8', 'TEXT NOT NULL']),
        ('pos9', ['_pos9', 'TEXT NOT NULL']),
        ('fieldx_avg', ['_avgx', 'TEXT NOT NULL']),
        ('fieldy_avg', ['_avgy', 'TEXT NOT NULL']),
        ('fieldz_avg', ['_avgz', 'TEXT NOT NULL']),
        ('fieldx_std', ['_stdx', 'TEXT NOT NULL']),
        ('fieldy_std', ['_stdy', 'TEXT NOT NULL']),
        ('fieldz_std', ['_stdz', 'TEXT NOT NULL']),
        ('current', ['_current', 'TEXT NOT NULL']),
        ('temperature', ['_temperature', 'TEXT NOT NULL']),
    ])
    _db_json_str = [
        '_pos1', '_pos2', '_pos3', '_pos5',
        '_pos6', '_pos7', '_pos8', '_pos9',
        '_avgx', '_avgy', '_avgz',
        '_stdx', '_stdy', '_stdz',
        '_current', '_temperature',
        ]

    def __init__(self, filename=None, database=None, idn=None):
        """Initialize variables.

        Args:
            filename (str, optional): file full path.
            database (str): database file path.
            idn (int): id in database table.
        """
        super().__init__(
            filename=filename, database=database, idn=idn, data_unit='T')

    def set_field_scan(self, voltage_scan_list, hall_probe):
        """Convert average voltage values to magnetic field."""
        if not isinstance(hall_probe, _calibration.HallProbe):
            raise TypeError(
                'hall_probe must be a HallProbe object.')

        vs = _get_avg_voltage(voltage_scan_list)

        for axis in self._axis_list:
            pos = getattr(vs, 'pos' + str(axis))
            setattr(self, '_pos' + str(axis), pos)

        bx_avg = hall_probe.sensorx.get_field(vs.avgx)
        by_avg = hall_probe.sensory.get_field(vs.avgy)
        bz_avg = hall_probe.sensorz.get_field(vs.avgz)

        bx_std = hall_probe.sensorx.get_field(vs.stdx)
        by_std = hall_probe.sensory.get_field(vs.stdy)
        bz_std = hall_probe.sensorz.get_field(vs.stdz)

        self._avgx = bx_avg
        self._avgy = by_avg
        self._avgz = bz_avg

        self._stdx = bx_std
        self._stdy = by_std
        self._stdz = bz_std

        cur, temp = get_current_and_temperature_values(voltage_scan_list)
        self._temperature = temp
        self._current = cur


class Fieldmap(_database.DatabaseObject):
    """Map for position and magnetic field values."""

    _db_table = 'fieldmaps'
    _db_dict = _collections.OrderedDict([
        ('id', [None, 'INTEGER NOT NULL']),
        ('date', [None, 'TEXT NOT NULL']),
        ('hour', [None, 'TEXT NOT NULL']),
        ('magnet_name', ['magnet_name', 'TEXT']),
        ('main_current', ['current_main', 'TEXT']),
        ('nr_scans', ['nr_scans', 'INTEGER']),
        ('initial_scan', ['initial_scan', 'INTEGER']),
        ('final_scan', ['final_scan', 'INTEGER']),
        ('gap', ['gap', 'TEXT']),
        ('control_gap', ['control_gap', 'TEXT']),
        ('magnet_length', ['magnet_length', 'TEXT']),
        ('comments', ['comments', 'TEXT']),
        ('current_main', ['current_main', 'TEXT']),
        ('nr_turns_main', ['nr_turns_main', 'TEXT']),
        ('current_trim', ['current_trim', 'TEXT']),
        ('nr_turns_trim', ['nr_turns_trim', 'TEXT']),
        ('current_ch', ['current_ch', 'TEXT']),
        ('nr_turns_ch', ['nr_turns_ch', 'TEXT']),
        ('current_cv', ['current_cv', 'TEXT']),
        ('nr_turns_cv', ['nr_turns_cv', 'TEXT']),
        ('current_qs', ['current_qs', 'TEXT']),
        ('nr_turns_qs', ['nr_turns_qs', 'TEXT']),
        ('magnet_center', ['_magnet_center', 'TEXT']),
        ('magnet_x_axis', ['_magnet_x_axis', 'INTEGER']),
        ('magnet_y_axis', ['_magnet_y_axis', 'INTEGER']),
        ('map', ['_map', 'TEXT NOT NULL']),
        ('current', ['_current', 'TEXT NOT NULL']),
        ('temperature', ['_temperature', 'TEXT NOT NULL']),
    ])
    _db_json_str = ['_magnet_center', '_map', '_current', '_temperature']

    def __init__(self, filename=None, database=None, idn=None):
        """Initialize variables.

        Args:
            filename (str, optional): file full path.
            id (int, optional): id in database table.
            database (str, optional): database file path.
        """
        if filename is not None and idn is not None:
            raise ValueError('Invalid arguments for FieldScan.')

        self._timestamp = None
        self.magnet_name = None
        self.gap = None
        self.control_gap = None
        self.magnet_length = None
        self.comments = None
        self.current_main = None
        self.nr_turns_main = None
        self.current_trim = None
        self.nr_turns_trim = None
        self.current_ch = None
        self.nr_turns_ch = None
        self.current_cv = None
        self.nr_turns_cv = None
        self.current_qs = None
        self.nr_turns_qs = None
        self.nr_scans = None
        self.initial_scan = None
        self.final_scan = None
        self._map = _np.array([])
        self._magnet_center = None
        self._magnet_x_axis = None
        self._magnet_y_axis = None
        self._current = {}
        self._temperature = {}

        if idn is not None and database is not None:
            self.read_from_database(database, idn)

        if filename is not None:
            self.read_file(filename)

    @property
    def map(self):
        """Fieldmap."""
        return self._map

    @property
    def magnet_center(self):
        """Magnet center."""
        return self._magnet_center

    @property
    def magnet_x_axis(self):
        """Magnet X axis."""
        return self._magnet_x_axis

    @property
    def magnet_y_axis(self):
        """Magnet Y axis."""
        return self._magnet_y_axis

    @property
    def current(self):
        """Power supply current."""
        return self._current

    @property
    def temperature(self):
        """Temperature readings."""
        return self._temperature

    @property
    def timestamp(self):
        """Return the timestamp."""
        return self._timestamp

    @property
    def default_filename(self):
        """Return the default filename."""
        label = _measurements_label
        if self.magnet_name is not None and len(self.magnet_name) != 0:
            name = self.magnet_name + '_' + label
        else:
            name = label

        if len(self._map) != 0:
            x = _np.unique(self._map[:, 0])
            y = _np.unique(self._map[:, 1])
            z = _np.unique(self._map[:, 2])
            if len(x) > 1:
                name = name + '_X={0:.0f}_{1:.0f}mm'.format(x[0], x[-1])
            if len(y) > 1:
                name = name + '_Y={0:.0f}_{1:.0f}mm'.format(y[0], y[-1])
            if len(z) > 1:
                name = name + '_Z={0:.0f}_{1:.0f}mm'.format(z[0], z[-1])

        for coil in ['main', 'trim', 'ch', 'cv', 'qs']:
            current = getattr(self, 'current_' + coil)
            if current is not None and len(current) != 0:
                if coil == 'trim':
                    ac = 'tc'
                elif coil == 'main':
                    ac = 'mc'
                else:
                    ac = coil
                name = name + '_I' + ac + '=' + current + 'A'

        self._timestamp = _utils.get_timestamp()
        date = self._timestamp.split('_')[0]

        filename = '{0:1s}_{1:1s}.dat'.format(date, name)
        return filename

    def clear(self):
        """Clear."""
        self._timestamp = None
        self.magnet_name = None
        self.gap = None
        self.control_gap = None
        self.magnet_length = None
        self.comments = None
        self.current_main = None
        self.nr_turns_main = None
        self.current_trim = None
        self.nr_turns_trim = None
        self.current_ch = None
        self.nr_turns_ch = None
        self.current_cv = None
        self.nr_turns_cv = None
        self.current_qs = None
        self.nr_turns_qs = None
        self.nr_scans = None
        self.initial_scan = None
        self.final_scan = None
        self._map = _np.array([])
        self._magnet_center = None
        self._magnet_x_axis = None
        self._magnet_y_axis = None
        self._current = {}
        self._temperature = {}

    def get_fieldmap_text(self, filename=None):
        """Get fieldmap text."""
        magnet_name = (
            self.magnet_name if self.magnet_name is not None else _empty_str)
        gap = (
            self.gap if self.gap is not None else _empty_str)
        control_gap = (
            self.control_gap if self.control_gap is not None else _empty_str)
        magnet_length = (
            self.magnet_length if self.magnet_length is not None
            else _empty_str)

        if self._timestamp is None:
            self._timestamp = _utils.get_timestamp()

        header_info = []
        header_info.append(['fieldmap_name', magnet_name])
        header_info.append(['timestamp', self._timestamp])
        if filename is None:
            header_info.append(['filename', ''])
        else:
            header_info.append(['filename', _os.path.split(filename)[1]])
        header_info.append(['nr_magnets', 1])
        header_info.append(['magnet_name', magnet_name])
        header_info.append(['gap[mm]', gap])
        header_info.append(['control_gap[mm]', control_gap])
        header_info.append(['magnet_length[mm]', magnet_length])

        for coil in ['main', 'trim', 'ch', 'cv', 'qs']:
            current = getattr(self, 'current_' + coil)
            turns = getattr(self, 'nr_turns_' + coil)
            if current is not None:
                header_info.append(['current_' + coil + '[A]', current])
                header_info.append(['nr_turns_' + coil, turns])

        header_info.append(['center_pos_z[mm]', '0'])
        header_info.append(['center_pos_x[mm]', '0'])
        header_info.append(['rotation[deg]', '0'])

        text = ''
        for line in header_info:
            variable = (str(line[0]) + ':').ljust(20)
            value = str(line[1])
            text = text + '{0:1s}\t{1:1s}\n'.format(variable, value)

        if len(header_info) != 0:
            text = text + '\n'

        text = text + 'X[mm]\tY[mm]\tZ[mm]\tBx[T]\tBy[T]\tBz[T]\n'
        text = (
            text + '-----------------------------------------------' +
            '----------------------------------------------\n')

        for i in range(self._map.shape[0]):
            text = text + '{0:0.3f}\t{1:0.3f}\t{2:0.3f}\t'.format(
                self._map[i, 0], self._map[i, 1], self._map[i, 2])
            text = text + '{0:0.10e}\t{1:0.10e}\t{2:0.10e}\n'.format(
                self._map[i, 3], self._map[i, 4], self._map[i, 5])

        return text

    def read_file(self, filename):
        """Read fieldmap file.

        Args:
            filename (str): fieldmap file path.
        """
        flines = _utils.read_file(filename)

        _s = _utils.find_value(flines, 'timestamp')
        self._timestamp = _s if _s != _empty_str else None
        _s = _utils.find_value(flines, 'magnet_name')
        self.magnet_name = _s if _s != _empty_str else None
        _s = _utils.find_value(flines, 'gap')
        self.gap = _s if _s != _empty_str else None
        _s = _utils.find_value(flines, 'control_gap')
        self.control_gap = _s if _s != _empty_str else None
        _s = _utils.find_value(flines, 'magnet_length')
        self.magnet_length = _s if _s != _empty_str else None

        self.current_main = _utils.find_value(
            flines, 'current_main', raise_error=False)
        self.nr_turns_main = _utils.find_value(
            flines, 'nr_turns_main', raise_error=False)
        self.current_trim = _utils.find_value(
            flines, 'current_trim', raise_error=False)
        self.nr_turns_trim = _utils.find_value(
            flines, 'nr_turns_trim', raise_error=False)
        self.current_ch = _utils.find_value(
            flines, 'current_ch', raise_error=False)
        self.nr_turns_ch = _utils.find_value(
            flines, 'nr_turns_ch', raise_error=False)
        self.current_cv = _utils.find_value(
            flines, 'current_cv', raise_error=False)
        self.nr_turns_cv = _utils.find_value(
            flines, 'nr_turns_cv', raise_error=False)
        self.current_qs = _utils.find_value(
            flines, 'current_ch', raise_error=False)
        self.nr_turns_qs = _utils.find_value(
            flines, 'nr_turns_ch', raise_error=False)

        data = []
        idx = _utils.find_index(flines, '-------------------------')
        for line in flines[idx+1:]:
            data_line = [float(d) for d in line.split('\t')]
            data.append(data_line)
        data = _np.array(data)

        pos3 = _np.unique(data[:, 0])
        pos2 = _np.unique(data[:, 1])
        pos1 = _np.unique(data[:, 2])
        measurement_axes = []
        if len(pos3) > 1:
            measurement_axes.append(3)
        if len(pos2) > 1:
            measurement_axes.append(2)
        if len(pos1) > 1:
            measurement_axes.append(1)

        if len(measurement_axes) > 2 or len(measurement_axes) == 0:
            raise MeasurementDataError('Invalid fieldmap file: %s' % filename)

        self._map = data

    def save_file(self, filename):
        """Save fieldmap file.

        Args:
            filename (str): fieldmap file path.
        """
        text = self.get_fieldmap_text(filename=filename)
        with open(filename, 'w') as f:
            f.write(text)

    def set_fieldmap_data(
            self, field_scan_list, hall_probe,
            correct_positions, magnet_center,
            magnet_x_axis, magnet_y_axis):
        """Set fieldmap from list of FieldScan objects.

        Args:
            field_scan_list (list): list of FieldScan objects.
            hall_probe (HallProbe): hall probe data.
            correct_positions (bool): correct sensor displacements flag.
            magnet_center (list): magnet center position.
            magnet_x_axis (int): magnet x-axis direction.
                                 [3, -3, 2, -2, 1 or -1]
            magnet_y_axis (int): magnet y-axis direction.
                                 [3, -3, 2, -2, 1 or -1]
        """
        if not isinstance(hall_probe, _calibration.HallProbe):
            raise TypeError(
                'hall_probe must be a HallProbe object.')

        tm = _get_transformation_matrix(magnet_x_axis, magnet_y_axis)
        _map = _get_fieldmap(field_scan_list, hall_probe, correct_positions)
        for i in range(len(_map)):
            p = _map[i, :3]
            b = _map[i, 3:]
            tp = _change_coordinate_system(p, tm, magnet_center)
            tb = _change_coordinate_system(b, tm)
            _map[i] = _np.append(tp, tb)
        _map = _np.array(sorted(_map, key=lambda x: (x[2], x[1], x[0])))

        self._magnet_center = magnet_center
        self._magnet_x_axis = magnet_x_axis
        self._magnet_y_axis = magnet_y_axis
        self._map = _map

        cur, temp = get_current_and_temperature_values(field_scan_list)
        self._current = cur
        self._temperature = temp


def get_current_and_temperature_values(scan_list):
    """Get power supply current and temperature values."""
    current = {}
    temperature = {}
    for scan in scan_list:
        for key, value in scan.current.items():
            if key in current.keys():
                [current[key].append(v) for v in value]
            else:
                current[key] = [v for v in value]
        for key, value in scan.temperature.items():
            if key in temperature.keys():
                [temperature[key].append(v) for v in value]
            else:
                temperature[key] = [v for v in value]

    if len(current) > 0:
        for key, value in current.items():
            current[key] = sorted(value, key=lambda x: x[0])

    if len(temperature) > 0:
        for key, value in temperature.items():
            temperature[key] = sorted(value, key=lambda x: x[0])

    return current, temperature


def get_field_scan_list(voltage_scan_list, hall_probe):
    """Get field_scan_list from voltage_scan_list."""
    field_scan_list = []
    grouped_voltage_scan_list = _group_voltage_scan_list(voltage_scan_list)
    for lt in grouped_voltage_scan_list:
        field_scan = FieldScan()
        field_scan.set_field_scan(lt, hall_probe)
        field_scan.configuration_id = lt[0].configuration_id
        field_scan.magnet_name = lt[0].magnet_name
        field_scan.main_current = lt[0].main_current
        field_scan_list.append(field_scan)
    return field_scan_list


def _change_coordinate_system(vector, transf_matrix, center=[0, 0, 0]):
    vector_array = _np.array(vector)
    center = _np.array(center)
    transf_vector = _np.dot(transf_matrix, vector_array - center)
    return transf_vector


def _cut_data_frame(df, idx_min, idx_max, axis=0):
    idx_min = idx_min + 1
    df = df.drop(df.columns[idx_min:], axis=axis)
    df = df.drop(df.columns[:idx_max], axis=axis)
    return df


def _interpolate_data_frame(df, pos, axis=0):

    def _interpolate_vec(x, pos):
        f = _interpolate.splrep(x.index, x.values, s=0, k=1)
        return _interpolate.splev(pos, f, der=0)

    interp_df = df.apply(
        _interpolate_vec, axis=axis, args=[pos], result_type='broadcast')
    if axis == 0:
        interp_df.index = pos
    else:
        interp_df.columns = pos

    return interp_df


def _get_avg_voltage(voltage_scan_list):
    """Get average voltage and position values."""
    if isinstance(voltage_scan_list, VoltageScan):
        voltage_scan_list = [voltage_scan_list]

    if not _valid_voltage_scan_list(voltage_scan_list):
        raise MeasurementDataError('Invalid voltage scan list.')

    fixed_axes = [a for a in voltage_scan_list[0].axis_list
                  if a != voltage_scan_list[0].scan_axis]
    for axis in fixed_axes:
        pos_set = set()
        for vs in voltage_scan_list:
            pos_attr = getattr(vs, 'pos' + str(axis))
            if len(pos_attr) == 1:
                pos_value = _np.around(
                    pos_attr[0], decimals=_check_position_precision)
                pos_set.add(pos_value)
            else:
                raise MeasurementDataError('Invalid voltage scan list.')
        if len(pos_set) != 1:
            raise MeasurementDataError('Invalid voltage scan list.')

    npts = voltage_scan_list[0].npts
    scan_axis = voltage_scan_list[0].scan_axis
    interp_pos = _np.mean([vs.scan_pos for vs in voltage_scan_list], axis=0)

    voltx_list = []
    volty_list = []
    voltz_list = []
    for vs in voltage_scan_list:
        if len(vs.avgx) == npts:
            fr = _interpolate.splrep(vs.scan_pos, vs.avgx, s=0, k=1)
            voltx = _interpolate.splev(interp_pos, fr, der=0)
        else:
            voltx = _np.zeros(npts)

        if len(vs.avgy) == npts:
            fs = _interpolate.splrep(vs.scan_pos, vs.avgy, s=0, k=1)
            volty = _interpolate.splev(interp_pos, fs, der=0)
        else:
            volty = _np.zeros(npts)

        if len(vs.avgz) == npts:
            ft = _interpolate.splrep(vs.scan_pos, vs.avgz, s=0, k=1)
            voltz = _interpolate.splev(interp_pos, ft, der=0)
        else:
            voltz = _np.zeros(npts)

        voltx_list.append(voltx)
        volty_list.append(volty)
        voltz_list.append(voltz)

    voltage = voltage_scan_list[0].copy()
    setattr(voltage, 'pos' + str(scan_axis), interp_pos)
    voltage.avgx = _np.mean(voltx_list, axis=0)
    voltage.avgy = _np.mean(volty_list, axis=0)
    voltage.avgz = _np.mean(voltz_list, axis=0)
    voltage.stdx = _np.std(voltx_list, axis=0)
    voltage.stdy = _np.std(volty_list, axis=0)
    voltage.stdz = _np.std(voltz_list, axis=0)

    return voltage


def _get_axis_vector(axis):
    if axis == 1:
        return _np.array([0, 0, 1])
    elif axis == 2:
        return _np.array([0, 1, 0])
    elif axis == 3:
        return _np.array([1, 0, 0])
    else:
        return None


def _get_data_frame_limits(values, fieldx, fieldy, fieldz, axis=0):
    if axis == 0:
        lim_min = _np.max([
            fieldx.index.min(), fieldy.index.min(), fieldz.index.min()])

        lim_max = _np.min([
            fieldx.index.max(), fieldy.index.max(), fieldz.index.max()])
    else:
        lim_min = _np.max([
            fieldx.columns.min(), fieldy.columns.min(), fieldz.columns.min()])

        lim_max = _np.min([
            fieldx.columns.max(), fieldy.columns.max(), fieldz.columns.max()])

    try:
        idx_min = _np.where(values >= lim_min)[0][0]
        idx_max = _np.where(values <= lim_max)[0][-1]
        return (idx_min, idx_max)
    except Exception:
        return None


def _get_fieldmap(field_scan_list, hall_probe, correct_positions):
    if isinstance(field_scan_list, FieldScan):
        field_scan_list = [field_scan_list]

    # get fieldmap axes
    axes = _get_fieldmap_axes(field_scan_list)
    if axes is None:
        raise Exception('Invalid field scan list.')
    first_axis, second_axis, third_axis = axes
    first_axis_direction = _get_axis_vector(first_axis)
    second_axis_direction = _get_axis_vector(second_axis)
    third_axis_direction = _get_axis_vector(third_axis)

    # get displacement of each sensor
    px_disp = hall_probe.to_bench_coordinate_system(
        hall_probe.sensorx_position)
    py_disp = hall_probe.to_bench_coordinate_system(
        hall_probe.sensory_position)
    pz_disp = hall_probe.to_bench_coordinate_system(
        hall_probe.sensorz_position)

    # get interpolation direction
    p_disp = [px_disp, py_disp, pz_disp]
    nonzero_norm = _np.nonzero([_utils.vector_norm(pd) for pd in p_disp])[0]
    if len(nonzero_norm) != 0:
        interp_direction = _utils.normalized_vector(p_disp[nonzero_norm[0]])
    else:
        interp_direction = None

    # create data frames with field values
    dfx = []
    dfy = []
    dfz = []
    third_axis_pos = set()
    for fs in field_scan_list:
        index = _pd.Index(getattr(fs, 'pos' + str(first_axis)), float)
        columns = _pd.Index(getattr(fs, 'pos' + str(second_axis)), float)
        third_axis_pos.update(getattr(fs, 'pos' + str(third_axis)))
        dfx.append(_pd.DataFrame(
            fs.avgx, index=index, columns=columns))
        dfy.append(_pd.DataFrame(
            fs.avgy, index=index, columns=columns))
        dfz.append(_pd.DataFrame(
            fs.avgz, index=index, columns=columns))
    fieldx = _pd.concat(dfx, axis=1)
    fieldy = _pd.concat(dfy, axis=1)
    fieldz = _pd.concat(dfz, axis=1)

    if len(third_axis_pos) == 1:
        third_axis_pos = list(third_axis_pos)[0]
    else:
        raise Exception('Invalid field scan list.')

    # correct sensors positions and interpolate magnetic field
    if interp_direction is not None and correct_positions:
        index = fieldx.index
        columns = fieldx.columns

        # correct first axis positions
        fieldx.index = index + _np.dot(first_axis_direction, px_disp)
        fieldy.index = index + _np.dot(first_axis_direction, py_disp)
        fieldz.index = index + _np.dot(first_axis_direction, pz_disp)

        # correct second axis positions
        fieldx.columns = columns + _np.dot(second_axis_direction, px_disp)
        fieldy.columns = columns + _np.dot(second_axis_direction, py_disp)
        fieldz.columns = columns + _np.dot(second_axis_direction, pz_disp)

        # correct third axis positions
        ptx = third_axis_pos + _np.dot(third_axis_direction, px_disp)
        pty = third_axis_pos + _np.dot(third_axis_direction, py_disp)
        ptz = third_axis_pos + _np.dot(third_axis_direction, pz_disp)
        if ptx == pty and ptx == ptz:
            third_axis_pos = ptx
        else:
            raise Exception("Can\'t correct sensors positions.")

        if _utils.parallel_vectors(interp_direction, first_axis_direction):
            axis = 0
            interpolation_grid = index
            if not (
                _np.array_equal(fieldx.columns.values, fieldy.columns.values)
                    and _np.array_equal(
                        fieldx.columns.values, fieldz.columns.values)):
                raise Exception("Can\'t correct sensors positions.")

        elif _utils.parallel_vectors(interp_direction, second_axis_direction):
            axis = 1
            interpolation_grid = columns
            if not (
                _np.array_equal(fieldx.index.values, fieldy.index.values)
                    and _np.array_equal(
                        fieldx.index.values, fieldz.index.values)):
                raise Exception("Can\'t correct sensors positions.")

        else:
            raise Exception("Can\'t correct sensors positions.")

        limits = _get_data_frame_limits(
            interpolation_grid.values, fieldx, fieldy, fieldz, axis=axis)
        if limits is None:
            raise Exception('Insufficient range to correct sensors positions.')

        # interpolate field and cut data frames
        idx_min, idx_max = limits
        fieldx = _interpolate_data_frame(fieldx, interpolation_grid, axis=axis)
        fieldy = _interpolate_data_frame(fieldy, interpolation_grid, axis=axis)
        fieldz = _interpolate_data_frame(fieldz, interpolation_grid, axis=axis)
        fieldx = _cut_data_frame(fieldx, idx_min, idx_max, axis=axis)
        fieldy = _cut_data_frame(fieldy, idx_min, idx_max, axis=axis)
        fieldz = _cut_data_frame(fieldz, idx_min, idx_max, axis=axis)

    # get direction of each sensor
    bx_dir = _np.array(hall_probe.sensorx_direction)
    by_dir = _np.array(hall_probe.sensory_direction)
    bz_dir = _np.array(hall_probe.sensorz_direction)

    # create fieldmap
    index = fieldx.index
    columns = fieldx.columns
    fieldmap = []
    for i in range(fieldx.shape[0]):
        for j in range(fieldx.shape[1]):
            pos = (
                first_axis_direction*index[i] +
                second_axis_direction*columns[j] +
                third_axis_direction*third_axis_pos)
            bx = bx_dir*fieldx.iloc[i, j]
            by = by_dir*fieldy.iloc[i, j]
            bz = bz_dir*fieldz.iloc[i, j]
            b = bx + by + bz
            field = hall_probe.to_bench_coordinate_system(b)
            fieldmap.append(_np.append(pos, field))
    fieldmap = _np.array(fieldmap)
    fieldmap = _np.array(sorted(fieldmap, key=lambda x: (x[2], x[1], x[0])))

    return fieldmap


def _get_fieldmap_axes(field_scan_list):
    if not _valid_field_scan_list(field_scan_list):
        return None

    fieldmap_axes = [1, 2, 3]

    _dict = _get_fieldmap_position_dict(field_scan_list)
    first_axis = field_scan_list[0].scan_axis
    if first_axis not in fieldmap_axes:
        return None

    second_axis_list = []
    for key, value in _dict.items():
        if key != first_axis and len(value) > 1:
            second_axis_list.append(key)

    second_axis = second_axis_list[0] if len(second_axis_list) == 1 else None
    second_axis_pos = []

    if second_axis is not None:
        for fs in field_scan_list:
            pos = getattr(fs, 'pos' + str(second_axis))
            second_axis_pos.append(pos)
    else:
        if len(field_scan_list) > 1:
            return None

    if len(second_axis_pos) != len(_np.unique(second_axis_pos)):
        return None

    fixed_axes = [ax for ax in fieldmap_axes]

    fixed_axes.remove(first_axis)
    if second_axis is None:
        second_axis = fixed_axes[0]
        third_axis = fixed_axes[1]
    else:
        fixed_axes.remove(second_axis)
        third_axis = fixed_axes[0]

    return (first_axis, second_axis, third_axis)


def _get_fieldmap_position_dict(field_scan_list):
    _dict = {}
    for axis in field_scan_list[0].axis_list:
        pos = set()
        for fs in field_scan_list:
            p = _np.around(getattr(fs, 'pos' + str(axis)),
                           decimals=_check_position_precision)
            pos.update(p)
        _dict[axis] = sorted(list(pos))
    return _dict


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


def _group_voltage_scan_list(voltage_scan_list):
    """Group voltage scan list."""
    if isinstance(voltage_scan_list, VoltageScan):
        voltage_scan_list = [voltage_scan_list]

    if not _valid_voltage_scan_list(voltage_scan_list):
        raise MeasurementDataError('Invalid voltage scan list.')

    fixed_axes = [a for a in voltage_scan_list[0].axis_list
                  if a != voltage_scan_list[0].scan_axis]
    search_axis = []
    for axis in fixed_axes:
        pos_set = set()
        for vs in voltage_scan_list:
            pos_attr = getattr(vs, 'pos' + str(axis))
            if len(pos_attr) == 1:
                pos_value = _np.around(
                    pos_attr[0], decimals=_check_position_precision)
                pos_set.add(pos_value)
            else:
                raise MeasurementDataError('Invalid voltage scan list.')
        if len(pos_set) != 1:
            search_axis.append(axis)

    if len(search_axis) > 1:
        raise MeasurementDataError('Invalid voltage scan list.')

    elif len(search_axis) == 1:
        search_axis = search_axis[0]
        _dict = {}
        for vs in voltage_scan_list:
            pos_attr = getattr(vs, 'pos' + str(search_axis))
            pos_value = _np.around(
                pos_attr[0], decimals=_check_position_precision)
            if pos_value in _dict.keys():
                _dict[pos_value].append(vs)
            else:
                _dict[pos_value] = [vs]
        grouped_voltage_scan_list = (_dict.values())

    else:
        grouped_voltage_scan_list = [voltage_scan_list]

    for lt in grouped_voltage_scan_list:
        configuration_id = lt[0].configuration_id
        if not all([vs.configuration_id == configuration_id for vs in lt]):
            raise MeasurementDataError(
                'Inconsistent configuration ID found in voltage scan list.')

        magnet_name = lt[0].magnet_name
        if not all([vs.magnet_name == magnet_name for vs in lt]):
            raise MeasurementDataError(
                'Inconsistent magnet name found in voltage scan list.')

        main_current = lt[0].main_current
        if not all([vs.main_current == main_current for vs in lt]):
            raise MeasurementDataError(
                'Inconsistent main current value found in voltage scan list.')

    return grouped_voltage_scan_list


def _valid_field_scan_list(field_scan_list):
    if (len(field_scan_list) == 0
       or not all([
            isinstance(fs, FieldScan) for fs in field_scan_list])):
        return False

    if any([fs.scan_axis is None or fs.npts == 0 for fs in field_scan_list]):
        return False

    if not all([fs.scan_axis == field_scan_list[0].scan_axis
                for fs in field_scan_list]):
        return False

    return True


def _valid_voltage_scan_list(voltage_scan_list):
    if not all([isinstance(vs, VoltageScan) for vs in voltage_scan_list]):
        return False

    if len(voltage_scan_list) == 0:
        return False

    if any([vs.scan_axis is None or vs.npts == 0 for vs in voltage_scan_list]):
        return False

    if not all([vs.scan_axis == voltage_scan_list[0].scan_axis
                for vs in voltage_scan_list]):
        return False

    if not all([
            vs.npts == voltage_scan_list[0].npts for vs in voltage_scan_list]):
        return False

    return True
