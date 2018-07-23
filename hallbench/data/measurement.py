# -*- coding: utf-8 -*-

"""Implementation of classes to store and analyse measurement data."""

import json as _json
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
    _db_table = ''
    _db_dict = {}
    _db_json_str = []
    _data_label = ''

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

        if filename is not None and idn is not None:
            raise ValueError('Invalid arguments for FieldData.')

        if idn is not None and database is not None:
            self.read_from_database(database, idn)

        if filename is not None:
            self.read_file(filename)

    def __str__(self):
        """Printable string representation of Data."""
        fmtstr = '{0:<18s} : {1}\n'
        r = ''
        r += fmtstr.format('magnet_name', str(self.magnet_name))
        r += fmtstr.format('main_current[A]', str(self.main_current))
        r += fmtstr.format('timestamp', str(self._timestamp))
        r += fmtstr.format('scan_axis', str(self.scan_axis))
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
    def create_database_table(cls, database):
        """Create database table."""
        if len(cls._db_table) == 0:
            return False

        variables = []
        for key in cls._db_dict.keys():
            variables.append((key, cls._db_dict[key][1]))
        success = _database.create_table(database, cls._db_table, variables)
        return success

    @classmethod
    def database_table_name(cls):
        """Return the database table name."""
        return cls._db_table

    @classmethod
    def get_configuration_id_from_database(cls, database, idn):
        """Return the configuration ID of the database record."""
        if len(cls._db_table) == 0:
            return None

        db_column_names = _database.get_table_column_names(
            database, cls._db_table)
        if len(db_column_names) == 0:
            return None

        db_entry = _database.read_from_database(database, cls._db_table, idn)
        if db_entry is None:
            return None

        idx = db_column_names.index('configuration_id')
        configuration_id = db_entry[idx]
        return configuration_id

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
        label = self._data_label + '_' + _measurements_label
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
        """Clear Data."""
        self.magnet_name = None
        self.main_current = None
        self._timestamp = None
        for key in self.__dict__:
            if isinstance(self.__dict__[key], _np.ndarray):
                self.__dict__[key] = _np.array([])

    def copy(self):
        """Return a copy of the object."""
        _copy = type(self)()
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
        flines = _utils.read_file(filename)
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
                pos = _np.around(_to_array(pos), decimals=_position_precision)
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
                _to_array(data[:, 0]), decimals=_position_precision)
            setattr(self, '_pos' + str(scan_axis), scan_positions)
            self._avgx = _to_array(data[:, 1])
            self._avgy = _to_array(data[:, 2])
            self._avgz = _to_array(data[:, 3])
            if dshape == 7:
                self._stdx = _to_array(data[:, 4])
                self._stdy = _to_array(data[:, 5])
                self._stdz = _to_array(data[:, 6])
            else:
                self._stdx = _np.zeros(len(scan_positions))
                self._stdy = _np.zeros(len(scan_positions))
                self._stdz = _np.zeros(len(scan_positions))
        else:
            message = 'Inconsistent number of columns in file: %s' % filename
            raise MeasurementDataError(message)

    def read_from_database(self, database, idn):
        """Read data from database entry."""
        if len(self._db_table) == 0:
            return

        db_column_names = _database.get_table_column_names(
            database, self._db_table)
        if len(db_column_names) == 0:
            raise MeasurementDataError('Failed to read data from database.')

        db_entry = _database.read_from_database(database, self._db_table, idn)
        if db_entry is None:
            raise ValueError('Invalid database ID.')

        for key in self._db_dict.keys():
            attr_name = self._db_dict[key][0]
            if key not in db_column_names:
                raise MeasurementDataError(
                    'Failed to read data from database.')
            else:
                if attr_name is not None and key != 'scan_axis':
                    idx = db_column_names.index(key)
                    if attr_name in self._db_json_str:
                        _l = _json.loads(db_entry[idx])
                        setattr(self, attr_name, _to_array(_l))
                    else:
                        setattr(self, attr_name, db_entry[idx])

    def reverse(self):
        """Reverse Data."""
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
            raise MeasurementDataError('Empty data.')

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

    def save_to_database(self, database, configuration_id):
        """Insert data into database table."""
        if len(self._db_table) == 0:
            return None

        db_column_names = _database.get_table_column_names(
            database, self._db_table)
        if len(db_column_names) == 0:
            raise MeasurementDataError('Failed to save data to database.')
            return None

        if self._timestamp is None:
            self._timestamp = _utils.get_timestamp()

        date = self._timestamp.split('_')[0]
        hour = self._timestamp.split('_')[1].replace('-', ':')

        db_values = []
        for key in self._db_dict.keys():
            attr_name = self._db_dict[key][0]
            if key not in db_column_names:
                raise MeasurementDataError(
                    'Failed to save data to database.')
                return None
            else:
                if key == "id":
                    db_values.append(None)
                elif attr_name is None:
                    db_values.append(locals()[key])
                elif attr_name in self._db_json_str:
                    _l = getattr(self, attr_name).tolist()
                    db_values.append(_json.dumps(_l))
                else:
                    db_values.append(getattr(self, attr_name))

        idn = _database.insert_into_database(
            database, self._db_table, db_values)
        return idn


class VoltageData(Data):
    """Position and voltage values."""

    _db_table = 'raw_data'
    _db_dict = _collections.OrderedDict([
        ('id', [None, 'INTEGER NOT NULL']),
        ('date', [None, 'TEXT NOT NULL']),
        ('hour', [None, 'TEXT NOT NULL']),
        ('magnet_name', ['magnet_name', 'TEXT']),
        ('main_current', ['main_current', 'TEXT']),
        ('configuration_id', [None, 'INTEGER']),
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
    ])
    _db_json_str = [
        '_pos1', '_pos2', '_pos3', '_pos5',
        '_pos6', '_pos7', '_pos8', '_pos9',
        '_avgx', '_avgy', '_avgz',
        '_stdx', '_stdy', '_stdz']
    _data_label = 'RawData'

    def __init__(self, filename=None, database=None, idn=None):
        """Initialize variables.

        Args:
            filename (str, optional): file full path.
            database (str): database file path.
            idn (int): id in database table.
        """
        super().__init__(
            filename=filename, database=database, idn=idn, data_unit='V')

    @Data.pos1.setter
    def pos1(self, value):
        """Set pos1 value."""
        self._pos1 = _np.around(_to_array(value), decimals=_position_precision)

    @Data.pos2.setter
    def pos2(self, value):
        """Set pos2 value."""
        self._pos2 = _np.around(_to_array(value), decimals=_position_precision)

    @Data.pos3.setter
    def pos3(self, value):
        """Set pos3 value."""
        self._pos3 = _np.around(_to_array(value), decimals=_position_precision)

    @Data.pos5.setter
    def pos5(self, value):
        """Set pos5 value."""
        self._pos5 = _np.around(_to_array(value), decimals=_position_precision)

    @Data.pos6.setter
    def pos6(self, value):
        """Set pos6 value."""
        self._pos6 = _np.around(_to_array(value), decimals=_position_precision)

    @Data.pos7.setter
    def pos7(self, value):
        """Set pos7 value."""
        self._pos7 = _np.around(_to_array(value), decimals=_position_precision)

    @Data.pos8.setter
    def pos8(self, value):
        """Set pos8 value."""
        self._pos8 = _np.around(_to_array(value), decimals=_position_precision)

    @Data.pos9.setter
    def pos9(self, value):
        """Set pos9 value."""
        self._pos9 = _np.around(_to_array(value), decimals=_position_precision)

    @Data.avgx.setter
    def avgx(self, value):
        """Set avgx value."""
        self._avgx = _to_array(value)

    @Data.avgy.setter
    def avgy(self, value):
        """Set avgy value."""
        self._avgy = _to_array(value)

    @Data.avgz.setter
    def avgz(self, value):
        """Set avgz value."""
        self._avgz = _to_array(value)

    @Data.stdx.setter
    def stdx(self, value):
        """Set stdx value."""
        self._stdx = _to_array(value)

    @Data.stdy.setter
    def stdy(self, value):
        """Set stdy value."""
        self._stdy = _to_array(value)

    @Data.stdz.setter
    def stdz(self, value):
        """Set stdz value."""
        self._stdz = _to_array(value)


class FieldData(Data):
    """Position and magnetic field values."""

    _db_table = 'scans'
    _db_dict = _collections.OrderedDict([
        ('id', [None, 'INTEGER NOT NULL']),
        ('date', [None, 'TEXT NOT NULL']),
        ('hour', [None, 'TEXT NOT NULL']),
        ('magnet_name', ['magnet_name', 'TEXT']),
        ('main_current', ['main_current', 'TEXT']),
        ('configuration_id', [None, 'INTEGER']),
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
    ])
    _db_json_str = [
        '_pos1', '_pos2', '_pos3', '_pos5',
        '_pos6', '_pos7', '_pos8', '_pos9',
        '_avgx', '_avgy', '_avgz',
        '_stdx', '_stdy', '_stdz']
    _data_label = 'Scan'

    def __init__(self, filename=None, database=None, idn=None):
        """Initialize variables.

        Args:
            filename (str, optional): file full path.
            database (str): database file path.
            idn (int): id in database table.
        """
        super().__init__(
            filename=filename, database=database, idn=idn, data_unit='T')

    def set_field_data(self, voltage_data_list, probe_calibration):
        """Convert average voltage values to magnetic field."""
        if not isinstance(probe_calibration, _calibration.ProbeCalibration):
            raise TypeError(
                'probe_calibration must be a ProbeCalibration object.')

        vd = _get_avg_voltage(voltage_data_list)

        for axis in self._axis_list:
            pos = getattr(vd, 'pos' + str(axis))
            setattr(self, '_pos' + str(axis), pos)

        bx_avg = probe_calibration.sensorx.convert_voltage(vd.avgx)
        by_avg = probe_calibration.sensory.convert_voltage(vd.avgy)
        bz_avg = probe_calibration.sensorz.convert_voltage(vd.avgz)

        bx_std = probe_calibration.sensorx.convert_voltage(vd.stdx)
        by_std = probe_calibration.sensory.convert_voltage(vd.stdy)
        bz_std = probe_calibration.sensorz.convert_voltage(vd.stdz)

        self._avgx = bx_avg
        self._avgy = by_avg
        self._avgz = bz_avg

        self._stdx = bx_std
        self._stdy = by_std
        self._stdz = bz_std


class Fieldmap(object):
    """Map for position and magnetic field values."""

    _db_table = 'fieldmaps'
    _db_dict = _collections.OrderedDict([
        ('id', [None, 'INTEGER NOT NULL']),
        ('date', [None, 'TEXT NOT NULL']),
        ('hour', [None, 'TEXT NOT NULL']),
        ('magnet_name', ['magnet_name', 'TEXT']),
        ('main_current', ['current_main', 'TEXT']),
        ('nr_scans', [None, 'INTEGER']),
        ('initial_scan', [None, 'INTEGER']),
        ('final_scan', [None, 'INTEGER']),
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
    ])
    _db_json_str = ['_magnet_center', '_map']
    _data_label = 'Fieldmap'

    def __init__(self, filename=None, database=None, idn=None):
        """Initialize variables.

        Args:
            filename (str, optional): file full path.
            id (int, optional): id in database table.
            database (str, optional): database file path.
        """
        if filename is not None and idn is not None:
            raise ValueError('Invalid arguments for FieldData.')

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
        self._map = _np.array([])
        self._magnet_center = None
        self._magnet_x_axis = None
        self._magnet_y_axis = None

        if idn is not None and database is not None:
            self.read_from_database(database, idn)

        if filename is not None:
            self.read_file(filename)

    @classmethod
    def database_table_name(cls):
        """Return the database table name."""
        return cls._db_table

    @classmethod
    def create_database_table(cls, database):
        """Create database table."""
        variables = []
        for key in cls._db_dict.keys():
            variables.append((key, cls._db_dict[key][1]))
        success = _database.create_table(database, cls._db_table, variables)
        return success

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
    def default_filename(self):
        """Return the default filename."""
        label = self._data_label + '_' + _measurements_label
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
                ac = coil if coil != 'trim' else 'tc'
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
        self._map = _np.array([])
        self._magnet_center = None
        self._magnet_x_axis = None
        self._magnet_y_axis = None

    def read_file(self, filename):
        """Read fieldmap file.

        Args:
            filename (str): fieldmap file path.
        """
        flines = _utils.read_file(filename)

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

    def read_from_database(self, database, idn):
        """Read fieldmap from database entry."""
        db_column_names = _database.get_table_column_names(
            database, self._db_table)
        if len(db_column_names) == 0:
            raise MeasurementDataError(
                'Failed to read fieldmap from database.')

        db_entry = _database.read_from_database(database, self._db_table, idn)
        if db_entry is None:
            raise ValueError('Invalid database ID.')

        for key in self._db_dict.keys():
            attr_name = self._db_dict[key][0]
            if key not in db_column_names:
                raise MeasurementDataError(
                    'Failed to read fieldmap from database.')
            else:
                if attr_name is not None:
                    idx = db_column_names.index(key)
                    if attr_name in self._db_json_str:
                        _l = _json.loads(db_entry[idx])
                        setattr(self, attr_name, _to_array(_l))
                    else:
                        setattr(self, attr_name, db_entry[idx])

    def save_file(self, filename):
        """Save fieldmap file.

        Args:
            filename (str): fieldmap file path.
        """
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
        header_info.append(['filename', filename])
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

        with open(filename, 'w') as f:
            for line in header_info:
                variable = (str(line[0]) + ':').ljust(20)
                value = str(line[1])
                f.write('{0:1s}\t{1:1s}\n'.format(variable, value))

            if len(header_info) != 0:
                f.write('\n')

            f.write('X[mm]\tY[mm]\tZ[mm]\tBx[T]\tBy[T]\tBz[T]\n')
            f.write('-----------------------------------------------' +
                    '----------------------------------------------\n')

            for i in range(self._map.shape[0]):
                f.write('{0:0.3f}\t{1:0.3f}\t{2:0.3f}\t'.format(
                    self._map[i, 0], self._map[i, 1], self._map[i, 2]))
                f.write('{0:0.10e}\t{1:0.10e}\t{2:0.10e}\n'.format(
                    self._map[i, 3], self._map[i, 4], self._map[i, 5]))

    def save_to_database(self, database, nr_scans, initial_scan, final_scan):
        """Insert field data into database table."""
        db_column_names = _database.get_table_column_names(
            database, self._db_table)

        if len(db_column_names) == 0:
            raise MeasurementDataError('Failed to save fieldmap to database.')
            return None

        if self._timestamp is None:
            self._timestamp = _utils.get_timestamp()

        date = self._timestamp.split('_')[0]
        hour = self._timestamp.split('_')[1].replace('-', ':')

        db_values = []
        for key in self._db_dict.keys():
            attr_name = self._db_dict[key][0]
            if key not in db_column_names:
                raise MeasurementDataError(
                    'Failed to save fieldmap to database.')
                return None

            else:
                if key == "id":
                    db_values.append(None)
                elif attr_name is None:
                    db_values.append(locals()[key])
                elif attr_name in self._db_json_str:
                    _l = getattr(self, attr_name)
                    if not isinstance(_l, list):
                        _l = _l.tolist()
                    db_values.append(_json.dumps(_l))
                else:
                    db_values.append(getattr(self, attr_name))

        idn = _database.insert_into_database(
            database, self._db_table, db_values)
        return idn

    def set_fieldmap_data(
            self, field_data_list, probe_calibration,
            correct_displacements, magnet_center,
            magnet_x_axis, magnet_y_axis):
        """Set fieldmap from list of FieldData objects.

        Args:
            field_data_list (list): list of FieldData objects.
            probe_calibration (ProbeCalibration): probe calibration data.
            correct_displacements (bool): correct sensor displacements flag.
            magnet_center (list): magnet center position.
            magnet_x_axis (int): magnet x-axis direction.
                                 [3, -3, 2, -2, 1 or -1]
            magnet_y_axis (int): magnet y-axis direction.
                                 [3, -3, 2, -2, 1 or -1]
        """
        if not isinstance(probe_calibration, _calibration.ProbeCalibration):
            raise TypeError(
                'probe_calibration must be a ProbeCalibration object.')

        _r = _get_fieldmap_position_and_field_values(
            field_data_list, probe_calibration, correct_displacements)
        pos1, pos2, pos3 = _r[0], _r[1], _r[2]
        field1, field2, field3 = _r[3], _r[4], _r[5]
        index_axis, columns_axis = _r[6], _r[7]

        def _get_field_at_point(pos):
            pos = _np.around(pos, decimals=_position_precision)
            p1, p2, p3 = pos[2], pos[1], pos[0]
            if (p1 not in pos1 or p2 not in pos2 or p3 not in pos3):
                return [_np.nan, _np.nan, _np.nan]

            psorted = [p1, p2, p3]
            loc_idx = psorted[index_axis-1]
            if columns_axis is not None:
                loc_col = psorted[columns_axis-1]
            else:
                loc_col = 0
            b3 = field3.loc[loc_idx, loc_col]
            b2 = field2.loc[loc_idx, loc_col]
            b1 = field1.loc[loc_idx, loc_col]
            return [b3, b2, b1]

        tm = _get_transformation_matrix(magnet_x_axis, magnet_y_axis)

        _map = []
        for p1 in pos1:
            for p2 in pos2:
                for p3 in pos3:
                    p = [p3, p2, p1]
                    b = _get_field_at_point(p)
                    tp = _change_coordinate_system(p, tm, magnet_center)
                    tb = _change_coordinate_system(b, tm)
                    _map.append(_np.append(tp, tb))
        _map = _np.array(_map)
        _map = _np.array(sorted(_map, key=lambda x: (x[2], x[1], x[0])))

        self._magnet_center = magnet_center
        self._magnet_x_axis = magnet_x_axis
        self._magnet_y_axis = magnet_y_axis
        self._map = _map


def _change_coordinate_system(vector, transf_matrix, center=[0, 0, 0]):
    vector = _np.array(vector)
    center = _np.array(center)
    transf_vector = _np.dot(transf_matrix, vector - center)
    return transf_vector


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


def _get_avg_voltage(voltage_data_list):
    """Get average voltage and position values."""
    if not _valid_voltage_data_list(voltage_data_list):
        raise MeasurementDataError('Invalid voltage data list.')

    npts = voltage_data_list[0].npts
    scan_axis = voltage_data_list[0].scan_axis
    interp_pos = _np.mean([vd.scan_pos for vd in voltage_data_list], axis=0)

    voltx_list = []
    volty_list = []
    voltz_list = []
    for vd in voltage_data_list:
        if len(vd.avgx) == npts:
            fr = _interpolate.splrep(vd.scan_pos, vd.avgx, s=0, k=1)
            voltx = _interpolate.splev(interp_pos, fr, der=0)
        else:
            voltx = _np.zeros(npts)

        if len(vd.avgy) == npts:
            fs = _interpolate.splrep(vd.scan_pos, vd.avgy, s=0, k=1)
            volty = _interpolate.splev(interp_pos, fs, der=0)
        else:
            volty = _np.zeros(npts)

        if len(vd.avgz) == npts:
            ft = _interpolate.splrep(vd.scan_pos, vd.avgz, s=0, k=1)
            voltz = _interpolate.splev(interp_pos, ft, der=0)
        else:
            voltz = _np.zeros(npts)

        voltx_list.append(voltx)
        volty_list.append(volty)
        voltz_list.append(voltz)

    voltage = voltage_data_list[0].copy()
    setattr(voltage, 'pos' + str(scan_axis), interp_pos)
    voltage.avgx = _np.mean(voltx_list, axis=0)
    voltage.avgy = _np.mean(volty_list, axis=0)
    voltage.avgz = _np.mean(voltz_list, axis=0)
    voltage.stdx = _np.std(voltx_list, axis=0)
    voltage.stdy = _np.std(volty_list, axis=0)
    voltage.stdz = _np.std(voltz_list, axis=0)

    return voltage


def _get_fieldmap_position_and_field_values(
        field_data_list, probe_calibration, correct_displacements):
    if not _valid_field_data_list(field_data_list):
        raise MeasurementDataError('Invalid field data list.')

    index_axis = field_data_list[0].scan_axis

    _dict = {}
    for axis in field_data_list[0].axis_list:
        pos = set()
        for field_data in field_data_list:
            p = _np.around(getattr(field_data, 'pos' + str(axis)),
                           decimals=_check_position_precision)
            pos.update(p)
        _dict[axis] = sorted(list(pos))

    columns_axis = []
    for key, value in _dict.items():
        if key != index_axis and len(value) > 1:
            columns_axis.append(key)

    if len(columns_axis) > 1:
        raise MeasurementDataError('Invalid number of measurement axes.')

    if len(columns_axis) == 1:
        columns_axis = columns_axis[0]

    if len(columns_axis) == 0:
        columns_axis = None

    dfx = []
    dfy = []
    dfz = []
    for fd in field_data_list:
        index = getattr(fd, 'pos' + str(index_axis))
        if columns_axis is not None:
            columns = getattr(fd, 'pos' + str(columns_axis))
        else:
            columns = [0]

        index = _pd.Index(index, float)
        columns = _pd.Index(columns, float)

        dfx.append(_pd.DataFrame(
            fd.avgx, index=index, columns=columns))
        dfy.append(_pd.DataFrame(
            fd.avgy, index=index, columns=columns))
        dfz.append(_pd.DataFrame(
            fd.avgz, index=index, columns=columns))

    fieldx = _pd.concat(dfx, axis=1)
    fieldy = _pd.concat(dfy, axis=1)
    fieldz = _pd.concat(dfz, axis=1)

    index = fieldx.index
    columns = fieldx.columns
    if (len(columns) != len(columns.drop_duplicates())):
        message = 'Duplicate position found in field data list.'
        raise MeasurementDataError(message)

    if correct_displacements:
        index = fieldy.index
        columns = fieldy.columns

        # shift field data
        fieldx.index = probe_calibration.corrected_position(
            index_axis, index, 'x')
        fieldz.index = probe_calibration.corrected_position(
            index_axis, index, 'z')
        nbeg, nend = _get_number_of_cuts(
            fieldx.index, fieldy.index, fieldz.index)

        # interpolate field data
        fieldx, fieldy, fieldz = _interpolate_data_frames(
            fieldx, fieldy, fieldz, axis=0)

        # cut field data
        fieldx, fieldy, fieldz = _cut_data_frames(
            fieldx, fieldy, fieldz, nbeg, nend)

        if columns_axis is not None:
            # shift field data
            fieldx.columns = probe_calibration.corrected_position(
                columns_axis, columns, 'x')
            fieldz.columns = probe_calibration.corrected_position(
                columns_axis, columns, 'z')
            nbeg, nend = _get_number_of_cuts(
                fieldx.columns, fieldy.columns, fieldz.columns)

            # interpolate field data
            fieldx, fieldy, fieldz = _interpolate_data_frames(
                fieldx, fieldy, fieldz, axis=1)

            # cut field data
            fieldx, fieldy, fieldz = _cut_data_frames(
                fieldx, fieldy, fieldz, nbeg, nend, axis=1)

    # update position values
    index = fieldx.index
    columns = fieldx.columns
    pos_sorted = [_dict[1], _dict[2], _dict[3]]
    pos_sorted[index_axis - 1] = index.values
    if columns_axis is not None:
        pos_sorted[columns_axis - 1] = columns.values

    pos3 = _np.array(pos_sorted[2])  # x-axis
    pos2 = _np.array(pos_sorted[1])  # y-axis
    pos1 = _np.array(pos_sorted[0])  # z-axis

    field3, field2, field1 = (
        probe_calibration.field_in_bench_coordinate_system(
            fieldx, fieldy, fieldz))

    return [pos1, pos2, pos3, field1, field2, field3, index_axis, columns_axis]


def _get_number_of_cuts(px, py, pz):
    nbeg = max(len(_np.where(px < py[0])[0]), len(_np.where(pz < py[0])[0]))
    nend = max(len(_np.where(px > py[-1])[0]), len(_np.where(pz > py[-1])[0]))
    return nbeg, nend


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


def _valid_field_data_list(field_data_list):
    """Check if the field data list is valid.

    Args:
        field_data_list (list): list of FieldData objects.

    Returns:
        True is the list is valid, False otherwise.
    """
    if isinstance(field_data_list, FieldData):
        field_data_list = [field_data_list]

    if not all([isinstance(fd, FieldData) for fd in field_data_list]):
        return False

    if len(field_data_list) == 0:
        return False

    if any([fd.scan_axis is None or fd.npts == 0 for fd in field_data_list]):
        return False

    if not all([fd.scan_axis == field_data_list[0].scan_axis
                for fd in field_data_list]):
        return False

    return True


def _valid_voltage_data_list(voltage_data_list):
    """Check if the voltage data list is valid.

    Args:
        voltage_data_list (list): list of VoltageData objects.

    Returns:
        True is the list is valid, False otherwise.
    """
    if isinstance(voltage_data_list, VoltageData):
        voltage_data_list = [voltage_data_list]

    if not all([isinstance(vd, VoltageData) for vd in voltage_data_list]):
        return False

    if len(voltage_data_list) == 0:
        return False

    if any([vd.scan_axis is None or vd.npts == 0 for vd in voltage_data_list]):
        return False

    if not all([vd.scan_axis == voltage_data_list[0].scan_axis
                for vd in voltage_data_list]):
        return False

    if not all([
            vd.npts == voltage_data_list[0].npts for vd in voltage_data_list]):
        return False

    fixed_axes = [a for a in voltage_data_list[0].axis_list
                  if a != voltage_data_list[0].scan_axis]
    for axis in fixed_axes:
        pos_set = set()
        for vd in voltage_data_list:
            pos_attr = getattr(vd, 'pos' + str(axis))
            if len(pos_attr) == 1:
                pos_value = _np.around(
                    pos_attr[0], decimals=_check_position_precision)
                pos_set.add(pos_value)
            else:
                return False
        if len(pos_set) != 1:
            return False

    return True


def _to_array(value):
    if value is not None:
        if not isinstance(value, _np.ndarray):
            value = _np.array(value)
        if len(value.shape) == 0:
            value = _np.array([value])
    else:
        value = _np.array([])
    return value
