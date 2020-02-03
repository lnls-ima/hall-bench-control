# -*- coding: utf-8 -*-

"""Implementation of classes to store and analyse measurement data."""

import os as _os
import numpy as _np
import pandas as _pd
import collections as _collections
from scipy import interpolate as _interpolate

from imautils.db import database as _database
from imautils.db import utils as _utils


CHECK_POSITION_PRECISION = 2
CHECK_CURRENT_PRECISION = 2


class MeasurementDataError(Exception):
    """Measurement data exception."""

    def __init__(self, message, *args):
        """Initialize variables."""
        self.message = message


class VoltageScan(_database.DatabaseAndFileDocument):
    """Position and voltage values."""

    label = 'VoltageScan'
    collection_name = 'voltage_scan'
    db_dict = _collections.OrderedDict([
        ('idn', {'field': 'id', 'dtype': int, 'not_null': True}),
        ('date', {'field': 'date', 'dtype': str, 'not_null': True}),
        ('hour', {'field': 'hour', 'dtype': str, 'not_null': True}),
        ('magnet_name', {'field': 'magnet_name', 'dtype': str}),
        ('comments', {'field': 'comments', 'dtype': str}),
        ('configuration_id', {'field': 'configuration_id', 'dtype': int}),
        ('current_setpoint', {'field': 'current_setpoint', 'dtype': float}),
        ('dcct_current_avg', {'field': 'dcct_current_avg', 'dtype': float}),
        ('dcct_current_std', {'field': 'dcct_current_std', 'dtype': float}),
        ('ps_current_avg', {'field': 'ps_current_avg', 'dtype': float}),
        ('ps_current_std', {'field': 'ps_current_std', 'dtype': float}),
        ('scan_axis', {'field': 'scan_axis', 'dtype': int}),
        ('pos1', {'field': 'pos1', 'dtype': _np.ndarray, 'not_null': True}),
        ('pos2', {'field': 'pos2', 'dtype': _np.ndarray, 'not_null': True}),
        ('pos3', {'field': 'pos3', 'dtype': _np.ndarray, 'not_null': True}),
        ('pos5', {'field': 'pos5', 'dtype': _np.ndarray, 'not_null': True}),
        ('pos6', {'field': 'pos6', 'dtype': _np.ndarray, 'not_null': True}),
        ('pos7', {'field': 'pos7', 'dtype': _np.ndarray, 'not_null': True}),
        ('pos8', {'field': 'pos8', 'dtype': _np.ndarray, 'not_null': True}),
        ('pos9', {'field': 'pos9', 'dtype': _np.ndarray, 'not_null': True}),
        ('vx', {'field': 'vx', 'dtype': _np.ndarray, 'not_null': True}),
        ('vy', {'field': 'vy', 'dtype': _np.ndarray, 'not_null': True}),
        ('vz', {'field': 'vz', 'dtype': _np.ndarray, 'not_null': True}),
        ('offsetx_start', {'field': 'offsetx_start', 'dtype': float}),
        ('offsetx_end', {'field': 'offsetx_end', 'dtype': float}),
        ('offsety_start', {'field': 'offsety_start', 'dtype': float}),
        ('offsety_end', {'field': 'offsety_end', 'dtype': float}),
        ('offsetz_start', {'field': 'offsetz_start', 'dtype': float}),
        ('offsetz_end', {'field': 'offsetz_end', 'dtype': float}),
        ('temperature', {'field': 'temperature', 'dtype': dict}),
    ])
    _axis_list = [1, 2, 3, 5, 6, 7, 8, 9]
    _pos1_unit = 'mm'
    _pos2_unit = 'mm'
    _pos3_unit = 'mm'
    _pos5_unit = 'deg'
    _pos6_unit = 'mm'
    _pos7_unit = 'mm'
    _pos8_unit = 'deg'
    _pos9_unit = 'deg'

    def __init__(
            self, database_name=None, mongo=False, server=None):
        """Initialize object.

        Args:
            filename (str): connection configuration filepath.
            database_name (str): database file path (sqlite) or name (mongo).
            idn (int): id in database table (sqlite) / collection (mongo).
            mongo (bool): flag indicating mongoDB (True) or sqlite (False).
            server (str): MongoDB server.

        """
        super().__init__(
            database_name=database_name, mongo=mongo, server=server)

    def __setattr__(self, name, value):
        """Set attribute."""
        if name not in ['npts', 'scan_axis']:
            super().__setattr__(name, value)

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
            npts = len(getattr(self, 'pos' + str(self.scan_axis)))
            v = [self.vx, self.vy, self.vz]
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
            pos.append(getattr(self, 'pos' + str(axis)))
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
        if self.magnet_name is not None and len(self.magnet_name) != 0:
            name = self.magnet_name + '_' + self.label
        else:
            name = self.label

        if self.npts != 0:
            if self.pos1.size == 1:
                name = name + '_pos1={0:.0f}mm'.format(self.pos1[0])
            if self.pos2.size == 1:
                name = name + '_pos2={0:.0f}mm'.format(self.pos2[0])
            if self.pos3.size == 1:
                name = name + '_pos3={0:.0f}mm'.format(self.pos3[0])

        self.date, self.hour = _utils.get_date_hour()
        timestamp = _utils.date_hour_to_timestamp(self.date, self.hour)
        filename = '{0:1s}_{1:1s}.dat'.format(timestamp, name)
        return filename

    def reverse(self):
        """Reverse Scan."""
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, _np.ndarray) and value.size > 1:
                self.__dict__[key] = value[::-1]
        return True

    def save_file(self, filename):
        """Save data to file.

        Args:
            filename (str): file fullpath.
        """
        if not self.valid_data():
            message = 'Invalid data.'
            raise ValueError(message)

        npts = self.npts
        scan_pos_name = 'pos' + str(self.scan_axis)

        if len(self.vx) == 0:
            self.vx = _np.zeros(npts)

        if len(self.vy) == 0:
            self.vy = _np.zeros(npts)

        if len(self.vz) == 0:
            self.vz = _np.zeros(npts)

        columns = [scan_pos_name, 'vx', 'vy', 'vz']
        return super().save_file(filename, columns=columns)

    def valid_data(self):
        """Check if parameters are valid."""
        if self.scan_axis is None or self.npts == 0:
            return False
        else:
            return super().valid_data()


class FieldScan(_database.DatabaseAndFileDocument):
    """Position and field values."""

    label = 'FieldScan'
    collection_name = 'field_scan'
    db_dict = _collections.OrderedDict([
        ('idn', {'field': 'id', 'dtype': int, 'not_null': True}),
        ('date', {'field': 'date', 'dtype': str, 'not_null': True}),
        ('hour', {'field': 'hour', 'dtype': str, 'not_null': True}),
        ('magnet_name', {'field': 'magnet_name', 'dtype': str}),
        ('comments', {'field': 'comments', 'dtype': str}),
        ('configuration_id', {'field': 'configuration_id', 'dtype': int}),
        ('nr_voltage_scans', {'field': 'nr_voltage_scans', 'dtype': int}),
        ('voltage_scan_id_list', {
            'field': 'voltage_scan_id_list', 'dtype': _np.ndarray}),
        ('current_setpoint', {'field': 'current_setpoint', 'dtype': float}),
        ('dcct_current_avg', {'field': 'dcct_current_avg', 'dtype': float}),
        ('dcct_current_std', {'field': 'dcct_current_std', 'dtype': float}),
        ('ps_current_avg', {'field': 'ps_current_avg', 'dtype': float}),
        ('ps_current_std', {'field': 'ps_current_std', 'dtype': float}),
        ('scan_axis', {'field': 'scan_axis', 'dtype': int}),
        ('pos1', {'field': 'pos1', 'dtype': _np.ndarray, 'not_null': True}),
        ('pos2', {'field': 'pos2', 'dtype': _np.ndarray, 'not_null': True}),
        ('pos3', {'field': 'pos3', 'dtype': _np.ndarray, 'not_null': True}),
        ('pos5', {'field': 'pos5', 'dtype': _np.ndarray, 'not_null': True}),
        ('pos6', {'field': 'pos6', 'dtype': _np.ndarray, 'not_null': True}),
        ('pos7', {'field': 'pos7', 'dtype': _np.ndarray, 'not_null': True}),
        ('pos8', {'field': 'pos8', 'dtype': _np.ndarray, 'not_null': True}),
        ('pos9', {'field': 'pos9', 'dtype': _np.ndarray, 'not_null': True}),
        ('bx', {'field': 'bx', 'dtype': _np.ndarray, 'not_null': True}),
        ('by', {'field': 'by', 'dtype': _np.ndarray, 'not_null': True}),
        ('bz', {'field': 'bz', 'dtype': _np.ndarray, 'not_null': True}),
        ('std_bx', {'field': 'std_bx', 'dtype': _np.ndarray, 'not_null': True}),
        ('std_by', {'field': 'std_by', 'dtype': _np.ndarray, 'not_null': True}),
        ('std_bz', {'field': 'std_bz', 'dtype': _np.ndarray, 'not_null': True}),
        ('temperature', {'field': 'temperature', 'dtype': dict}),
    ])
    _axis_list = [1, 2, 3, 5, 6, 7, 8, 9]
    _pos1_unit = 'mm'
    _pos2_unit = 'mm'
    _pos3_unit = 'mm'
    _pos5_unit = 'deg'
    _pos6_unit = 'mm'
    _pos7_unit = 'mm'
    _pos8_unit = 'deg'
    _pos9_unit = 'deg'

    def __init__(
            self, database_name=None, mongo=False, server=None):
        """Initialize object.

        Args:
            filename (str): connection configuration filepath.
            database_name (str): database file path (sqlite) or name (mongo).
            idn (int): id in database table (sqlite) / collection (mongo).
            mongo (bool): flag indicating mongoDB (True) or sqlite (False).
            server (str): MongoDB server.

        """
        super().__init__(
            database_name=database_name, mongo=mongo, server=server)

    def __setattr__(self, name, value):
        """Set attribute."""
        if name not in ['npts', 'scan_axis']:
            super().__setattr__(name, value)

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
            npts = len(getattr(self, 'pos' + str(self.scan_axis)))
            v = [self.bx, self.by, self.bz]
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
            pos.append(getattr(self, 'pos' + str(axis)))
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
        if self.magnet_name is not None and len(self.magnet_name) != 0:
            name = self.magnet_name + '_' + self.label
        else:
            name = self.label

        if self.npts != 0:
            if self.pos1.size == 1:
                name = name + '_pos1={0:.0f}mm'.format(self.pos1[0])
            if self.pos2.size == 1:
                name = name + '_pos2={0:.0f}mm'.format(self.pos2[0])
            if self.pos3.size == 1:
                name = name + '_pos3={0:.0f}mm'.format(self.pos3[0])

        self.date, self.hour = _utils.get_date_hour()
        timestamp = _utils.date_hour_to_timestamp(self.date, self.hour)
        filename = '{0:1s}_{1:1s}.dat'.format(timestamp, name)
        return filename

    def reverse(self):
        """Reverse Scan."""
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, _np.ndarray) and value.size > 1:
                self.__dict__[key] = value[::-1]
        return True

    def save_file(self, filename, include_std=True):
        """Save data to file.

        Args:
            filename (str): file full path.
        """
        if not self.valid_data():
            message = 'Invalid data.'
            raise ValueError(message)

        npts = self.npts
        scan_pos_name = 'pos' + str(self.scan_axis)

        if len(self.bx) == 0:
            self.bx = _np.zeros(npts)

        if len(self.by) == 0:
            self.by = _np.zeros(npts)

        if len(self.bz) == 0:
            self.bz = _np.zeros(npts)

        if include_std:
            if len(self.std_bx) == 0:
                self.std_bx = _np.zeros(npts)

            if len(self.std_by) == 0:
                self.std_by = _np.zeros(npts)

            if len(self.std_bz) == 0:
                self.std_bz = _np.zeros(npts)

        columns = [scan_pos_name, 'bx', 'by', 'bz']
        if include_std:
            columns.append('std_bx')
            columns.append('std_by')
            columns.append('std_bz')

        return super().save_file(filename, columns=columns)

    def set_field_scan(
            self, voltage_scan_list, calx, caly, calz):
        """Convert average voltage values to magnetic field."""
        if isinstance(voltage_scan_list, VoltageScan):
            voltage_scan_list = [voltage_scan_list]

        _valid_voltage_scan_list(voltage_scan_list)
        for vs in voltage_scan_list:
            vs = _correct_voltage_offset(vs)

        vs = voltage_scan_list[0]
        npts = vs.npts

        fixed_axes = [a for a in vs.axis_list if a != vs.scan_axis]
        for axis in fixed_axes:
            pos_set = set()
            for v in voltage_scan_list:
                pos_attr = getattr(v, 'pos' + str(axis))
                if len(pos_attr) == 1:
                    pos_value = _np.around(
                        pos_attr[0], decimals=CHECK_POSITION_PRECISION)
                    pos_set.add(pos_value)
                elif len(pos_attr) == 0:
                    pass
                else:
                    msg = 'Invalid positions in voltage scan list.'
                    raise MeasurementDataError(msg)

            if len(pos_set) != 1:
                msg = 'Invalid positions in voltage scan list.'
                raise MeasurementDataError(msg)

        interp_pos = _np.mean(
            [v.scan_pos for v in voltage_scan_list], axis=0)

        vx_list = []
        vy_list = []
        vz_list = []
        for v in voltage_scan_list:
            if len(v.vx) == npts:
                ft = _interpolate.splrep(v.scan_pos, v.vx, s=0, k=1)
                vx = _interpolate.splev(interp_pos, ft, der=0)
            else:
                vx = _np.zeros(npts)

            if len(v.vy) == npts:
                ft = _interpolate.splrep(v.scan_pos, v.vy, s=0, k=1)
                vy = _interpolate.splev(interp_pos, ft, der=0)
            else:
                vy = _np.zeros(npts)

            if len(v.vz) == npts:
                ft = _interpolate.splrep(v.scan_pos, v.vz, s=0, k=1)
                vz = _interpolate.splev(interp_pos, ft, der=0)
            else:
                vz = _np.zeros(npts)

            vx_list.append(vx)
            vy_list.append(vy)
            vz_list.append(vz)

        avg_vx = _np.mean(vx_list, axis=0)
        avg_vy = _np.mean(vy_list, axis=0)
        avg_vz = _np.mean(vz_list, axis=0)
        std_vx = _np.std(vx_list, axis=0)
        std_vy = _np.std(vy_list, axis=0)
        std_vz = _np.std(vz_list, axis=0)

        self.magnet_name = vs.magnet_name
        self.comments = vs.comments
        self.configuration_id = vs.configuration_id
        self.date, self.hour = _utils.get_date_hour()

        setattr(self, 'pos' + str(vs.scan_axis), interp_pos)
        for axis in fixed_axes:
            pos = getattr(vs, 'pos' + str(axis))
            setattr(self, 'pos' + str(axis), pos)
        
        if calx is not None:
            self.bx = calx.get_field(avg_vx)
            self.std_bx = calx.get_field(std_vx)
        else:
            self.bx = _np.zeros(npts)
            self.std_bx = _np.zeros(npts)
            
        if caly is not None:
            self.by = caly.get_field(avg_vy)
            self.std_by = caly.get_field(std_vy)
        else:
            self.by = _np.zeros(npts)
            self.std_by = _np.zeros(npts)
            
        if calz is not None:            
            self.bz = calz.get_field(avg_vz)     
            self.std_bz = calz.get_field(std_vz)
        else:
            self.bz = _np.zeros(npts)
            self.std_bz = _np.zeros(npts)

        self.nr_voltage_scans = len(voltage_scan_list)
        self.voltage_scan_id_list = [v.idn for v in voltage_scan_list]

        self.temperature = get_temperature_values(voltage_scan_list)

        setpoint, dcct_avg, dcct_std, ps_avg, ps_std = get_current_values(
            voltage_scan_list, len(voltage_scan_list))
        self.current_setpoint = setpoint
        self.dcct_current_avg = dcct_avg
        self.dcct_current_std = dcct_std
        self.ps_current_avg = ps_avg
        self.ps_current_std = ps_std

        return True

    def valid_data(self):
        """Check if parameters are valid."""
        if self.scan_axis is None or self.npts == 0:
            return False
        else:
            return super().valid_data()


class Fieldmap(_database.DatabaseAndFileDocument):
    """Map for position and magnetic field values."""

    label = 'Fieldmap'
    collection_name = 'fieldmap'
    db_dict = _collections.OrderedDict([
        ('idn', {'field': 'id', 'dtype': int, 'not_null': True}),
        ('date', {'field': 'date', 'dtype': str, 'not_null': True}),
        ('hour', {'field': 'hour', 'dtype': str, 'not_null': True}),
        ('magnet_name', {'field': 'magnet_name', 'dtype': str}),
        ('comments', {'field': 'comments', 'dtype': str}),
        ('nr_field_scans', {'field': 'nr_field_scans', 'dtype': int}),
        ('field_scan_id_list', {
            'field': 'field_scan_id_list', 'dtype': _np.ndarray}),
        ('probe_positions_id', {'field': 'probe_positions_id', 'dtype': int}),
        ('current_setpoint', {'field': 'current_setpoint', 'dtype': float}),
        ('dcct_current_avg', {'field': 'dcct_current_avg', 'dtype': float}),
        ('dcct_current_std', {'field': 'dcct_current_std', 'dtype': float}),
        ('ps_current_avg', {'field': 'ps_current_avg', 'dtype': float}),
        ('ps_current_std', {'field': 'ps_current_std', 'dtype': float}),
        ('gap', {'field': 'gap', 'dtype': str}),
        ('control_gap', {'field': 'control_gap', 'dtype': str}),
        ('magnet_length', {'field': 'magnet_length', 'dtype': str}),
        ('current_main', {'field': 'current_main', 'dtype': str}),
        ('nr_turns_main', {'field': 'nr_turns_main', 'dtype': str}),
        ('current_trim', {'field': 'current_trim', 'dtype': str}),
        ('nr_turns_trim', {'field': 'nr_turns_trim', 'dtype': str}),
        ('current_ch', {'field': 'current_ch', 'dtype': str}),
        ('nr_turns_ch', {'field': 'nr_turns_ch', 'dtype': str}),
        ('current_cv', {'field': 'current_cv', 'dtype': str}),
        ('nr_turns_cv', {'field': 'nr_turns_cv', 'dtype': str}),
        ('current_qs', {'field': 'current_qs', 'dtype': str}),
        ('nr_turns_qs', {'field': 'nr_turns_qs', 'dtype': str}),
        ('magnet_center', {'field': 'magnet_center', 'dtype': _np.ndarray}),
        ('magnet_x_axis', {'field': 'magnet_x_axis', 'dtype': int}),
        ('magnet_y_axis', {'field': 'magnet_y_axis', 'dtype': int}),
        ('corrected_positions', {
            'field': 'corrected_positions', 'dtype': int}),
        ('map', {'field': 'map', 'dtype': _np.ndarray}),
        ('temperature', {'field': 'temperature', 'dtype': dict}),
    ])

    def __init__(
            self, filename=None, database_name=None, idn=None,
            mongo=False, server=None):
        """Initialize object.

        Args:
            filename (str): connection configuration filepath.
            database_name (str): database file path (sqlite) or name (mongo).
            idn (int): id in database table (sqlite) / collection (mongo).
            mongo (bool): flag indicating mongoDB (True) or sqlite (False).
            server (str): MongoDB server.

        """
        super().__init__(
            database_name=database_name, mongo=mongo, server=server)

    @property
    def default_filename(self):
        """Return the default filename."""
        if self.magnet_name is not None and len(self.magnet_name) != 0:
            name = self.magnet_name + '_' + self.label
        else:
            name = self.label

        if len(self.map) != 0:
            x = _np.unique(self.map[:, 0])
            y = _np.unique(self.map[:, 1])
            z = _np.unique(self.map[:, 2])
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

        self.date, self.hour = _utils.get_date_hour()
        filename = '{0:1s}_{1:1s}.dat'.format(self.date, name)
        return filename

    def get_fieldmap_text(self, filename=None):
        """Get fieldmap text."""
        magnet_name = (
            self.magnet_name if self.magnet_name is not None
            else _utils.EMPTY_STR)
        gap = (
            self.gap if self.gap is not None
            else _utils.EMPTY_STR)
        control_gap = (
            self.control_gap if self.control_gap is not None
            else _utils.EMPTY_STR)
        magnet_length = (
            self.magnet_length if self.magnet_length is not None
            else _utils.EMPTY_STR)

        if self.date is None or self.hour is None:
            self.date, self.hour = _utils.get_date_hour()

        timestamp = _utils.date_hour_to_timestamp(self.date, self.hour)

        header_info = []
        header_info.append(['fieldmap_name', magnet_name])
        header_info.append(['timestamp', timestamp])
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

        for i in range(self.map.shape[0]):
            text = text + '{0:0.3f}\t{1:0.3f}\t{2:0.3f}\t'.format(
                self.map[i, 0], self.map[i, 1], self.map[i, 2])
            text = text + '{0:0.10e}\t{1:0.10e}\t{2:0.10e}\n'.format(
                self.map[i, 3], self.map[i, 4], self.map[i, 5])

        return text

    def read_file(self, filename):
        """Read fieldmap file.

        Args:
            filename (str): fieldmap file path.
        """
        flines = _utils.read_file(filename)

        timestamp = _utils.find_value(flines, 'timestamp')
        self.date, self.hour = _utils.timestamp_to_date_hour(timestamp)

        self.magnet_name = _utils.find_value(flines, 'magnet_name')
        self.gap = _utils.find_value(flines, 'gap')
        self.control_gap = _utils.find_value(flines, 'control_gap')
        self.magnet_length = _utils.find_value(flines, 'magnet_length')

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

        self.map = data

    def save_file(self, filename):
        """Save fieldmap file.

        Args:
            filename (str): fieldmap file path.
        """
        text = self.get_fieldmap_text(filename=filename)
        with open(filename, 'w') as f:
            f.write(text)

    def set_fieldmap_data(
            self, field_scan_list, probe_positions,
            correct_positions, magnet_center,
            magnet_x_axis, magnet_y_axis):
        """Set fieldmap from list of FieldScan objects.

        Args:
            field_scan_list (list): list of FieldScan objects.
            probe_positions (HallProbePositions): hall probe positions.
            correct_positions (bool): correct sensor displacements flag.
            magnet_center (list): magnet center position.
            magnet_x_axis (int): magnet x-axis direction.
                                 [3, -3, 2, -2, 1 or -1]
            magnet_y_axis (int): magnet y-axis direction.
                                 [3, -3, 2, -2, 1 or -1]
        """
        tm = _get_transformation_matrix(magnet_x_axis, magnet_y_axis)
        _map = _get_fieldmap(
            field_scan_list, probe_positions, correct_positions)

        for i in range(len(_map)):
            p = _map[i, :3]
            b = _map[i, 3:]
            tp = _change_coordinate_system(p, tm, magnet_center)
            tb = _change_coordinate_system(b, tm)
            _map[i] = _np.append(tp, tb)
        _map = _np.array(sorted(_map, key=lambda x: (x[2], x[1], x[0])))

        self.magnet_center = magnet_center
        self.magnet_x_axis = magnet_x_axis
        self.magnet_y_axis = magnet_y_axis
        self.corrected_positions = correct_positions
        self.probe_positions_id = probe_positions.idn
        self.nr_field_scans = len(field_scan_list)
        self.field_scan_id_list = [fs.idn for fs in field_scan_list]
        self.map = _map

        self.temperature = get_temperature_values(field_scan_list)

        setpoint, dcct_avg, dcct_std, ps_avg, ps_std = get_current_values(
            field_scan_list, field_scan_list[0].nr_voltage_scans)
        self.current_setpoint = setpoint
        self.dcct_current_avg = dcct_avg
        self.dcct_current_std = dcct_std
        self.ps_current_avg = ps_avg
        self.ps_current_std = ps_std


def get_current_values(scan_list, nmeas):
    """Get average and std current values."""
    setpoint_set = set()
    dcct_avgs = []
    dcct_stds = []
    ps_avgs = []
    ps_stds = []
    for scan in scan_list:
        if scan.current_setpoint is not None:
            setpoint_set.add(_np.around(scan.current_setpoint, 5))

        if scan.dcct_current_avg is not None:
            dcct_avgs.append(scan.dcct_current_avg)
            if scan.dcct_current_std is not None:
                dcct_stds.append(scan.dcct_current_std)
            else:
                dcct_stds.append(0)

        if scan.ps_current_avg is not None:
            ps_avgs.append(scan.ps_current_avg)
            if scan.ps_current_std is not None:
                ps_stds.append(scan.ps_current_std)
            else:
                ps_stds.append(0)

    setpoint = None
    if len(setpoint_set) == 1:
        setpoint = list(setpoint_set)[0]

    dcct_avg, dcct_std = _get_average_std(dcct_avgs, dcct_stds, nmeas)
    ps_avg, ps_std = _get_average_std(ps_avgs, ps_stds, nmeas)

    return setpoint, dcct_avg, dcct_std, ps_avg, ps_std


def get_temperature_values(scan_list):
    """Get temperature values."""
    temperature = {}
    for scan in scan_list:
        for key, value in scan.temperature.items():
            if key in temperature.keys():
                [temperature[key].append(v) for v in value]
            else:
                temperature[key] = [v for v in value]

    if len(temperature) > 0:
        for key, value in temperature.items():
            temperature[key] = sorted(value, key=lambda x: x[0])

    return temperature


def get_field_scan_list(voltage_scan_list, calx, caly, calz):
    """Get field_scan_list from voltage_scan_list."""
    field_scan_list = []
    grouped_voltage_scan_list = _group_voltage_scan_list(voltage_scan_list)
    for lt in grouped_voltage_scan_list:
        field_scan = FieldScan()
        field_scan.set_field_scan(lt, calx, caly, calz)
        field_scan.configuration_id = lt[0].configuration_id
        field_scan.magnet_name = lt[0].magnet_name
        field_scan.current_setpoint = lt[0].current_setpoint
        field_scan.comments = lt[0].comments
        field_scan_list.append(field_scan)
    return field_scan_list


def _change_coordinate_system(vector, transf_matrix, center=[0, 0, 0]):
    vector_array = _np.array(vector)
    center = _np.array(center)
    transf_vector = _np.dot(transf_matrix, vector_array - center)
    return transf_vector


def _correct_voltage_offset(voltage_scan):
    """Subtract voltage offset from measurement values."""
    vs = voltage_scan.copy()
    if vs.npts == 0:
        msg = "Can't correct voltage offset: Empty voltage scan."
        raise MeasurementDataError(msg)
        return None

    p = vs.scan_pos
    npts = vs.npts

    vi = vs.offsetx_start
    vf = vs.offsetx_end
    if vi is not None and vf is not None:
        if npts == 1:
            vs.vx = vs.vx - (vi + vf)/2
        else:
            vs.vx = vs.vx - vi - ((vf - vi)/(p[-1] - p[0]))*(p - p[0])
    elif vi is not None:
        vs.vx = vs.vx - vi
    elif vf is not None:
        vs.vx = vs.vx - vf

    vi = vs.offsety_start
    vf = vs.offsety_end
    if vi is not None and vf is not None:
        if npts == 1:
            vs.vy = vs.vy - (vi + vf)/2
        else:
            vs.vy = vs.vy - vi - ((vf - vi)/(p[-1] - p[0]))*(p - p[0])
    elif vi is not None:
        vs.vy = vs.vy - vi
    elif vf is not None:
        vs.vy = vs.vy - vf

    vi = vs.offsetz_start
    vf = vs.offsetz_end
    if vi is not None and vf is not None:
        if npts == 1:
            vs.vz = vs.vz - (vi + vf)/2
        else:
            vs.vz = vs.vz - vi - ((vf - vi)/(p[-1] - p[0]))*(p - p[0])
    elif vi is not None:
        vs.vz = vs.vz - vi
    elif vf is not None:
        vs.vz = vs.vz - vf

    return vs


def _cut_data_frame(df, idx_min, idx_max, axis=0):
    if axis == 0:
        df = df.drop(df.index[:idx_min], axis=axis)
        df = df.drop(df.index[idx_max-idx_min+1:], axis=axis)
    else:
        df = df.drop(df.columns[:idx_min], axis=axis)
        df = df.drop(df.columns[idx_max-idx_min+1:], axis=axis)
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


def _get_fieldmap(field_scan_list, probe_positions, correct_positions):
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
    px_disp = probe_positions.to_bench_coordinate_system(
        probe_positions.sensorx_position)
    py_disp = probe_positions.to_bench_coordinate_system(
        probe_positions.sensory_position)
    pz_disp = probe_positions.to_bench_coordinate_system(
        probe_positions.sensorz_position)

    # get interpolation direction
    p_disp = [px_disp, py_disp, pz_disp]
    nonzero_norm = _np.nonzero([_vector_norm(pd) for pd in p_disp])[0]
    if len(nonzero_norm) != 0:
        interp_direction = _normalized_vector(p_disp[nonzero_norm[0]])
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
        third_axis_pos.update(
            _np.around(
                getattr(fs, 'pos' + str(third_axis)),
                decimals=CHECK_POSITION_PRECISION))
        dfx.append(_pd.DataFrame(
            fs.bx, index=index, columns=columns))
        dfy.append(_pd.DataFrame(
            fs.by, index=index, columns=columns))
        dfz.append(_pd.DataFrame(
            fs.bz, index=index, columns=columns))
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

        if _parallel_vectors(interp_direction, first_axis_direction):
            axis = 0
            interpolation_grid = index
            if not (_np.allclose(
                fieldx.columns.values, fieldy.columns.values,
                atol=CHECK_POSITION_PRECISION) and _np.allclose(
                    fieldx.columns.values, fieldz.columns.values,
                    atol=CHECK_POSITION_PRECISION)):
                raise Exception("Can\'t correct sensors positions.")

        elif _parallel_vectors(interp_direction, second_axis_direction):
            axis = 1
            interpolation_grid = columns
            if not (_np.allclose(
                fieldx.index.values, fieldy.index.values,
                atol=CHECK_POSITION_PRECISION) and _np.allclose(
                    fieldx.index.values, fieldz.index.values,
                    atol=CHECK_POSITION_PRECISION)):
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
    bx_dir = _np.array(probe_positions.sensorx_direction)
    by_dir = _np.array(probe_positions.sensory_direction)
    bz_dir = _np.array(probe_positions.sensorz_direction)

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
            field = probe_positions.to_bench_coordinate_system(b)
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
            p = _np.around(
                getattr(fs, 'pos' + str(axis)),
                decimals=CHECK_POSITION_PRECISION)
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

    _valid_voltage_scan_list(voltage_scan_list)

    fixed_axes = [a for a in voltage_scan_list[0].axis_list
                  if a != voltage_scan_list[0].scan_axis]
    search_axis = []
    for axis in fixed_axes:
        pos_set = set()
        for vs in voltage_scan_list:
            pos_attr = getattr(vs, 'pos' + str(axis))
            if len(pos_attr) == 1:
                pos_value = _np.around(
                    pos_attr[0], decimals=CHECK_POSITION_PRECISION)
                pos_set.add(pos_value)
            elif len(pos_attr) == 0:
                pass
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
                pos_attr[0], decimals=CHECK_POSITION_PRECISION)
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

        current_setpoint_set = set()
        for l in lt:
            if l.current_setpoint is None:
                current_setpoint_set.add(None)
            else:
                current_setpoint_set.add(
                    _np.around(l.current_setpoint, CHECK_CURRENT_PRECISION))
        if len(current_setpoint_set) > 1:
            raise MeasurementDataError(
                'Inconsistent current setpoint found in voltage scan list.')

    return grouped_voltage_scan_list


def _valid_field_scan_list(field_scan_list):
    if (len(field_scan_list) == 0
       or not all([
            isinstance(fs, FieldScan) for fs in field_scan_list])):
        msg = 'field_scan_list must be a list of FieldScan objects.'
        raise MeasurementDataError(msg)

    if len(field_scan_list) == 0:
        msg = 'Empty field scan list.'
        raise MeasurementDataError(msg)

    if any([fs.scan_axis is None for fs in field_scan_list]):
        msg = 'Invalid scan axis found in field scan list.'
        raise MeasurementDataError(msg)

    if any([fs.npts == 0 for fs in field_scan_list]):
        msg = 'Invalid number of points found in field scan list.'
        raise MeasurementDataError(msg)

    if not all([fs.scan_axis == field_scan_list[0].scan_axis
                for fs in field_scan_list]):
        msg = 'Inconsistent scan axis found in field scan list.'
        raise MeasurementDataError(msg)

    return True


def _valid_voltage_scan_list(voltage_scan_list):
    if not all([isinstance(vs, VoltageScan) for vs in voltage_scan_list]):
        msg = 'voltage_scan_list must be a list of VoltageScan objects.'
        raise MeasurementDataError(msg)

    if len(voltage_scan_list) == 0:
        msg = 'Empty voltage scan list.'
        raise MeasurementDataError(msg)

    if any([vs.scan_axis is None for vs in voltage_scan_list]):
        msg = 'Invalid scan axis found in voltage scan list.'
        raise MeasurementDataError(msg)

    if any([vs.npts == 0 for vs in voltage_scan_list]):
        msg = 'Invalid number of points found in voltage scan list.'
        raise MeasurementDataError(msg)

    if not all([vs.scan_axis == voltage_scan_list[0].scan_axis
                for vs in voltage_scan_list]):
        msg = 'Inconsistent scan axis found in voltage scan list.'
        raise MeasurementDataError(msg)

    if not all([
            vs.npts == voltage_scan_list[0].npts for vs in voltage_scan_list]):
        msg = 'Inconsistent number of points found in voltage scan list.'
        raise MeasurementDataError(msg)

    return True


def _parallel_vectors(vec1, vec2):
    """Return True if the vectors are parallel, False otherwise."""
    tol = 1e-15
    norm = _np.sum((_np.cross(vec1, vec2)**2))
    if norm < tol:
        return True
    else:
        return False


def _vector_norm(vec):
    """Return the norm of the vector."""
    return _np.sqrt((_np.sum(_np.array(vec)**2)))


def _normalized_vector(vec):
    """Return the normalized vector."""
    normalized_vec = vec / _vector_norm(vec)
    return normalized_vec


def _get_average_std(avgs, stds, nmeas):
    """Return the average and STD for a set of averages and STD values."""
    if len(avgs) == 0 and len(stds) == 0:
        return None, None

    if len(avgs) != len(stds):
        raise ValueError('Inconsistent size of input arguments')
        return None, None

    if nmeas == 0:
        raise ValueError('Invalid number of measurements.')
        return None, None

    elif nmeas == 1:
        avg = _np.mean(avgs)
        std = _np.std(avgs, ddof=1)

    else:
        n = len(avgs)*nmeas
        avgs = _np.array(avgs)
        stds = _np.array(stds)

        avg = _np.sum(avgs)*nmeas/n
        std = _np.sqrt((1/(n-1))*(
            _np.sum((nmeas-1)*(stds**2) + nmeas*(avgs**2)) -
            (1/n)*(_np.sum(avgs*nmeas)**2)))

    return avg, std