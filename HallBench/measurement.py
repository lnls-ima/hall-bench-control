# -*- coding: utf-8 -*-
"""Implementation of classes to store and analyse measurement data."""

import os as _os
import time as _time
import numpy as _np
import shutil as _shutil
from scipy import interpolate as _interpolate
from scipy.integrate import cumtrapz as _cumtrapz

from HallBench import files as _files
from HallBench import configuration as _configuration
from HallBench import calibration as _calibration


class MeasurementDataError(Exception):
    """Measurement data exception."""

    def __init__(self, message, *args):
        """Initialize variables."""
        self.message = message


class DataSet(object):
    """Position and data values."""

    def __init__(self, description='', unit=''):
        """Initialize variables.

        Args:
            description (str): data description.
            unit (str): data unit.
        """
        self.description = description
        self.unit = unit
        self.posx = None
        self.posy = None
        self.posz = None
        self.datax = _np.array([])
        self.datay = _np.array([])
        self.dataz = _np.array([])

    @staticmethod
    def reverse(dataset):
        """Return the reverse of a DataSet."""
        reverse_dataset = DataSet()
        reverse_dataset.unit = dataset.unit
        reverse_dataset.description = dataset.description

        if isinstance(dataset.posx, (int, float)):
            reverse_dataset.posx = dataset.posx
        else:
            reverse_dataset.posx = dataset.posx[::-1]

        if isinstance(dataset.posy, (int, float)):
            reverse_dataset.posy = dataset.posy
        else:
            reverse_dataset.posy = dataset.posy[::-1]

        if isinstance(dataset.posz, (int, float)):
            reverse_dataset.posz = dataset.posz
        else:
            reverse_dataset.posz = dataset.posz[::-1]

        reverse_dataset.datax = dataset.datax[::-1]
        reverse_dataset.datay = dataset.datay[::-1]
        reverse_dataset.dataz = dataset.dataz[::-1]
        return reverse_dataset

    @staticmethod
    def copy(dataset):
        """Return a copy of a DataSet."""
        dataset_copy = DataSet()
        dataset_copy.unit = dataset.unit
        dataset_copy.description = dataset.description

        if isinstance(dataset.posx, (int, float)):
            dataset_copy.posx = dataset.posx
        else:
            dataset_copy.posx = _np.copy(dataset.posx)

        if isinstance(dataset.posy, (int, float)):
            dataset_copy.posy = dataset.posy
        else:
            dataset_copy.posy = _np.copy(dataset.posy)

        if isinstance(dataset.posz, (int, float)):
            dataset_copy.posz = dataset.posz
        else:
            dataset_copy.posz = _np.copy(dataset.posz)

        dataset_copy.datax = _np.copy(dataset.datax)
        dataset_copy.datay = _np.copy(dataset.datay)
        dataset_copy.dataz = _np.copy(dataset.dataz)
        return dataset_copy

    def clear(self):
        """Clear DataSet."""
        self.posx = None
        self.posy = None
        self.posz = None
        self.datax = _np.array([])
        self.datay = _np.array([])
        self.dataz = _np.array([])


class LineScan(object):
    """Line scan data."""

    def __init__(self, posx, posy, posz, cconfig, mconfig,
                 calibration, dirpath):
        """Initialize variables.

        Args:
            posx (float or array): x position of the scan line.
            posy (float or array): y position of the scan line.
            posz (float or array): z position of the scan line.
            cconfig (ControlConfiguration or str): control configuration.
            mconfig (MeasurementConfiguration or str): measurement config.
            calibration (CalibrationData or str): probe calibration data.
            dirpath (str): directory path to save files.

        Raises:
            MeasurementDataError: if dirpath is not a valid path.
        """
        if isinstance(posx, (int, float)):
            self._posx = round(posx, 4)
        else:
            self._posx = _np.around(posx, decimals=4)

        if isinstance(posy, (int, float)):
            self._posy = round(posy, 4)
        else:
            self._posy = _np.around(posy, decimals=4)

        if isinstance(posz, (int, float)):
            self._posz = round(posz, 4)
        else:
            self._posz = _np.around(posz, decimals=4)

        self.dirpath = dirpath
        if not _os.path.isdir(self.dirpath):
            try:
                _os.mkdir(self.dirpath)
            except Exception:
                raise MeasurementDataError('Invalid directory path.')

        self._set_configuration(cconfig, mconfig, calibration)

        self._scan_axis = None
        self._timestamp = ''
        self._voltage_raw = []
        self._voltage_interpolated = []
        self._voltage_avg = DataSet()
        self._voltage_std = DataSet()
        self._field_avg = DataSet()
        self._field_std = DataSet()
        self._field_first_integral = DataSet()
        self._field_second_integral = DataSet()
        self._set_scan_axis()

    @staticmethod
    def copy(linescan):
        """Return a copy of a LineScan."""
        lsc = LineScan(
            linescan.posx, linescan.posy, linescan.posz,
            linescan.cconfig, linescan.mconfig, linescan.calibration,
            linescan.dirpath)

        lsc._timestamp = linescan._timestamp

        lsc._voltage_raw = [DataSet().copy(s) for s in linescan._voltage_raw]
        lsc._voltage_interpolated = [
            DataSet().copy(s) for s in linescan._voltage_interpolated]
        lsc._voltage_avg = DataSet().copy(linescan._voltage_avg)
        lsc._voltage_std = DataSet().copy(linescan._voltage_std)

        lsc._field_avg = DataSet().copy(linescan._field_avg)
        lsc._field_std = DataSet().copy(linescan._field_std)
        lsc._field_first_integral = DataSet().copy(
            linescan._field_first_integral)
        lsc._field_second_integral = DataSet().copy(
            linescan._field_second_integral)

        return lsc

    @staticmethod
    def read_from_files(
            dirpath, posx=None, posy=None, posz=None, pos_str=None):
        """Read LineScan data from files."""
        datadir = _os.path.join(dirpath, 'data')
        files = _get_scan_files_list(
            datadir, posx=posx, posy=posy, posz=posz, pos_str=pos_str)

        cconfig = set()
        mconfig = set()
        calibration = set()
        timestamp = set()

        voltage_raw = []
        voltage_interpolated = []
        voltage_avg = None
        voltage_std = None
        field_avg = None
        field_std = None
        field_first_integral = None
        field_second_integral = None

        for filename in files:
            (dataset, ts, cf, mf, caf) = _read_scan_file(filename)

            timestamp.add(ts)
            cconfig.add(cf)
            mconfig.add(mf)
            calibration.add(caf)

            if 'voltage_raw' in filename.lower():
                voltage_raw.append(dataset)
            elif 'voltage_interpolated' in filename.lower():
                voltage_interpolated.append(dataset)
            elif 'voltage_avg' in filename.lower():
                voltage_avg = dataset
            elif 'voltage_std' in filename.lower():
                voltage_std = dataset
            elif 'field_avg' in filename.lower():
                field_avg = dataset
            elif 'field_std' in filename.lower():
                field_std = dataset
            elif 'field_first_integral' in filename.lower():
                field_first_integral = dataset
            elif 'field_second_integral' in filename.lower():
                field_second_integral = dataset

        if len(cconfig) != 1 or len(mconfig) != 1 or len(calibration) != 1:
            message = 'Inconsistent configuration files for LineScan.'
            raise MeasurementDataError(message)

        cconfig_filename = _os.path.join(dirpath, list(cconfig)[0])
        mconfig_filename = _os.path.join(dirpath, list(mconfig)[0])
        calibration_filename = _os.path.join(dirpath, list(calibration)[0])

        linescan = LineScan(
            field_avg.posx, field_avg.posy, field_avg.posz, cconfig_filename,
            mconfig_filename, calibration_filename, dirpath)

        linescan._timestamp = sorted(list(timestamp))[-1]
        linescan._voltage_raw = voltage_raw
        linescan._voltage_interpolated = voltage_interpolated
        linescan._voltage_avg = voltage_avg
        linescan._voltage_std = voltage_std
        linescan._field_avg = field_avg
        linescan._field_std = field_std
        linescan._field_first_integral = field_first_integral
        linescan._field_second_integral = field_second_integral

        return linescan

    @property
    def scan_positions(self):
        """Scan axis position values."""
        if self._scan_axis == 'x':
            return self._posx
        elif self._scan_axis == 'y':
            return self._posy
        elif self._scan_axis == 'z':
            return self._posz
        else:
            return []

    @property
    def scan_axis(self):
        """Scan axis label."""
        return self._scan_axis

    @property
    def posx(self):
        """X position."""
        return self._posx

    @property
    def posy(self):
        """Y position."""
        return self._posy

    @property
    def posz(self):
        """Z position."""
        return self._posz

    @property
    def control_configuration(self):
        """Control configuration."""
        return self._cconfig

    @property
    def measurement_configuration(self):
        """Measurement configuration."""
        return self._mconfig

    @property
    def calibration(self):
        """Calibration data."""
        return self._calibration

    @property
    def nr_scans(self):
        """Number of scans."""
        return len(self._voltage_raw)

    @property
    def timestamp(self):
        """Scan timestamp."""
        return self._timestamp

    @property
    def voltage_raw(self):
        """List with raw scan data."""
        return self._voltage_raw

    @property
    def interpolated(self):
        """List with the interpolated scan data."""
        return self._voltage_interpolated

    @property
    def voltage_avg(self):
        """Average voltage values."""
        return self._voltage_avg

    @property
    def voltage_std(self):
        """Standard deviation of voltage values."""
        return self._voltage_std

    @property
    def field_avg(self):
        """Average magnetic field values."""
        return self._field_avg

    @property
    def field_std(self):
        """Standard deviation of magnetic field values."""
        return self._field_std

    @property
    def field_first_integral(self):
        """Magnetic field first integral."""
        return self._field_first_integral

    @property
    def field_second_integral(self):
        """Magnetic field second integral."""
        return self._field_second_integral

    def add_scan(self, scan):
        """Add a scan to the list."""
        if self._valid_scan(scan):
            scan.description = 'voltage_raw'
            self._voltage_raw.append(scan)
        else:
            raise MeasurementDataError('Invalid scan.')

    def analyse_data(self, save_data=True):
        """Analyse the line scan data."""
        self._timestamp = (
            _time.strftime('%Y-%m-%d_%H-%M-%S', _time.localtime()))

        if save_data:
            self._save_configuration()

        if self.nr_scans != 0:
            self._data_interpolation(save_data)
            self._calculate_voltage_avg_std(save_data)
            self._calculate_field_avg_std(save_data)
            self._calculate_field_first_integral(save_data)
            self._calculate_field_second_integral(save_data)

    def clear(self):
        """Clear LineScan."""
        self._scan_axis = None
        self._timestamp = ''
        self._voltage_raw = []
        self._voltage_interpolated = []
        self._voltage_avg = DataSet()
        self._voltage_std = DataSet()
        self._field_avg = DataSet()
        self._field_std = DataSet()
        self._field_first_integral = DataSet()
        self._field_second_integral = DataSet()

    def _set_scan_axis(self):
        scan_axis = _get_scan_axis(self._posx, self._posy, self._posz)
        if scan_axis is not None:
            self._scan_axis = scan_axis
        else:
            raise MeasurementDataError(
                'Invalid position arguments for LineScan.')

    def _valid_scan(self, scan):
        scan_axis = _get_scan_axis(scan.posx, scan.posy, scan.posz)
        if scan_axis is not None and scan_axis == self._scan_axis:
            if (scan_axis == 'x' and round(scan.posy, 4) == self._posy and
               round(scan.posz, 4) == self._posz):
                return True
            elif (scan_axis == 'y' and round(scan.posx, 4) == self._posx and
                  round(scan.posz, 4) == self._posz):
                return True
            elif (scan_axis == 'z' and round(scan.posx, 4) == self._posx and
                  round(scan.posy, 4) == self._posy):
                return True
            else:
                return False
        else:
            return False

    def _set_configuration(self, cconfig, mconfig, calibration):
        if isinstance(cconfig, str) and _os.path.isfile(cconfig):
            self._cconfig = _configuration.ControlConfiguration(cconfig)
        elif isinstance(cconfig, _configuration.ControlConfiguration):
            self._cconfig = cconfig
        else:
            self._cconfig = None

        if isinstance(mconfig, str) and _os.path.isfile(mconfig):
            self._mconfig = _configuration.MeasurementConfiguration(mconfig)
        elif isinstance(mconfig, _configuration.MeasurementConfiguration):
            self._mconfig = mconfig
        else:
            self._mconfig = None

        if isinstance(calibration, str) and _os.path.isfile(calibration):
            self._calibration = _calibration.CalibrationData(calibration)
        elif isinstance(calibration, _calibration.CalibrationData):
            self._calibration = calibration
        else:
            self._calibration = None

    def _save_configuration(self):
        try:
            if self._cconfig is not None:
                if self._cconfig.filename is not None:
                    _shutil.copy(self._cconfig.filename, self.dirpath)
                else:
                    filename = _os.path.join(
                        self.dirpath, 'control_configuration.txt')
                    self._cconfig.save_file(filename)

            if self._mconfig is not None:
                if self._mconfig.filename is not None:
                    _shutil.copy(self._mconfig.filename, self.dirpath)
                else:
                    filename = _os.path.join(
                        self.dirpath, 'measurement_configuration.txt')
                    self._mconfig.save_file(filename)

            if self._calibration is not None:
                if self._calibration.filename is not None:
                    _shutil.copy(self._calibration.filename, self.dirpath)
                else:
                    filename = _os.path.join(self.dirpath, 'calibration.txt')
                    self._calibration.save_file(filename)
        except Exception:
            pass

    def _data_interpolation(self, save_data=True):
        """Interpolate each scan."""
        # correct curves displacement due to trigger and
        # integration time (half integration time)
        self._voltage_interpolated = []
        scan_pos = self.scan_positions

        idx = 1
        for raw in self._voltage_raw:
            interp = DataSet()
            interp.description = 'voltage_interpolated'
            interp.unit = raw.unit

            interp.posx = self._posx
            interp.posy = self._posy
            interp.posz = self._posz

            rawpos = _get_scan_position(raw.posx, raw.posy, raw.posz)

            fx = _interpolate.splrep(rawpos, raw.datax, s=0, k=1)
            interp.datax = _interpolate.splev(scan_pos, fx, der=0)

            fy = _interpolate.splrep(rawpos, raw.datay, s=0, k=1)
            interp.datay = _interpolate.splev(scan_pos, fy, der=0)

            fz = _interpolate.splrep(rawpos, raw.dataz, s=0, k=1)
            interp.dataz = _interpolate.splev(scan_pos, fz, der=0)

            self._voltage_interpolated.append(interp)

            if save_data:
                self._save_data(raw, idx=idx)
                self._save_data(interp, idx=idx)

            idx += 1

    def _calculate_voltage_avg_std(self, save_data=True):
        """Calculate the average and std of voltage values."""
        n = self.nr_scans

        interpolation_npts = len(self.scan_positions)

        unit = self._voltage_raw[0].unit
        if not all([raw.unit == unit for raw in self._voltage_raw]):
            message = 'Inconsistent unit found in raw voltage list.'
            raise MeasurementDataError(message)

        # average calculation
        self._voltage_avg.description = 'voltage_avg'
        self._voltage_avg.unit = unit
        self._voltage_avg.posx = self._posx
        self._voltage_avg.posy = self._posy
        self._voltage_avg.posz = self._posz
        self._voltage_avg.datax = _np.zeros(interpolation_npts)
        self._voltage_avg.datay = _np.zeros(interpolation_npts)
        self._voltage_avg.dataz = _np.zeros(interpolation_npts)

        if n > 1:
            for i in range(n):
                self._voltage_avg.datax += self._voltage_interpolated[i].datax
                self._voltage_avg.datay += self._voltage_interpolated[i].datay
                self._voltage_avg.dataz += self._voltage_interpolated[i].dataz

            self._voltage_avg.datax /= n
            self._voltage_avg.datay /= n
            self._voltage_avg.dataz /= n

        elif n == 1:
            self._voltage_avg.datax = self._voltage_interpolated[0].datax
            self._voltage_avg.datay = self._voltage_interpolated[0].datay
            self._voltage_avg.dataz = self._voltage_interpolated[0].dataz

        # standard std calculation
        self._voltage_std.description = 'voltage_std'
        self._voltage_std.unit = unit
        self._voltage_std.posx = self._posx
        self._voltage_std.posy = self._posy
        self._voltage_std.posz = self._posz
        self._voltage_std.datax = _np.zeros(interpolation_npts)
        self._voltage_std.datay = _np.zeros(interpolation_npts)
        self._voltage_std.dataz = _np.zeros(interpolation_npts)

        if n > 1:
            for i in range(n):
                self._voltage_std.datax += pow((
                    self._voltage_interpolated[i].datax -
                    self._voltage_avg.datax), 2)

                self._voltage_std.datay += pow((
                    self._voltage_interpolated[i].datay -
                    self._voltage_avg.datay), 2)

                self._voltage_std.dataz += pow((
                    self._voltage_interpolated[i].dataz -
                    self._voltage_avg.dataz), 2)

            self._voltage_std.datax /= n
            self._voltage_std.datay /= n
            self._voltage_std.dataz /= n

        if save_data:
            self._save_data(self._voltage_avg)
            self._save_data(self._voltage_std)

    def _calculate_field_avg_std(self, save_data=True):
        """Calculate the average and std of magnetic field values."""
        if self._calibration is None:
            message = 'Calibration data not found.'
            raise MeasurementDataError(message)

        conversion_factor = self._calibration.get_conversion_factor(
            self.voltage_avg.unit)

        self._field_avg.description = 'field_avg'
        self._field_avg.unit = self._calibration.field_unit
        self._field_avg.posx = self._posx
        self._field_avg.posy = self._posy
        self._field_avg.posz = self._posz

        self._field_avg.datax = self._calibration.convert_probe_x(
            conversion_factor*self._voltage_avg.datax)

        self._field_avg.datay = self._calibration.convert_probe_y(
            conversion_factor*self._voltage_avg.datay)

        self._field_avg.dataz = self._calibration.convert_probe_z(
            conversion_factor*self._voltage_avg.dataz)

        self._field_std.description = 'field_std'
        self._field_std.unit = self._calibration.field_unit
        self._field_std.posx = self._posx
        self._field_std.posy = self._posy
        self._field_std.posz = self._posz

        self._field_std.datax = self._calibration.convert_probe_x(
            conversion_factor*self._voltage_std.datax)

        self._field_std.datay = self._calibration.convert_probe_y(
            conversion_factor*self._voltage_std.datay)

        self._field_std.dataz = self._calibration.convert_probe_z(
            conversion_factor*self._voltage_std.dataz)

        if save_data:
            self._save_data(self._field_avg)
            self._save_data(self._field_std)

    def _calculate_field_first_integral(self, save_data=True):
        """Calculate the magnetic field first integral."""
        self._field_first_integral.description = 'field_first_integral'
        self._field_first_integral.unit = self._field_avg.unit + '.m'
        self._field_first_integral.posx = self._posx
        self._field_first_integral.posy = self._posy
        self._field_first_integral.posz = self._posz

        self._field_first_integral.datax = _cumtrapz(
            x=self.scan_positions/1000,
            y=self._field_avg.datax,
            initial=0)

        self._field_first_integral.datay = _cumtrapz(
            x=self.scan_positions/1000,
            y=self._field_avg.datay,
            initial=0)

        self._field_first_integral.dataz = _cumtrapz(
            x=self.scan_positions/1000,
            y=self._field_avg.dataz,
            initial=0)

        if save_data:
            self._save_data(self._field_first_integral)

    def _calculate_field_second_integral(self, save_data=True):
        """Calculate the magnetic field second integral."""
        self._field_second_integral.description = 'field_second_integral'
        self._field_second_integral.unit = self._field_avg.unit + '.m^2'
        self._field_second_integral.posx = self._posx
        self._field_second_integral.posy = self._posy
        self._field_second_integral.posz = self._posz

        self._field_second_integral.datax = _cumtrapz(
            x=self.scan_positions/1000,
            y=self._field_first_integral.datax,
            initial=0)

        self._field_second_integral.datay = _cumtrapz(
            x=self.scan_positions/1000,
            y=self._field_first_integral.datay,
            initial=0)

        self._field_second_integral.dataz = _cumtrapz(
            x=self.scan_positions/1000,
            y=self._field_first_integral.dataz,
            initial=0)

        if save_data:
            self._save_data(self._field_second_integral)

    def _save_data(self, dataset, idx=None):
        if self._scan_axis == 'x':
            pos_str = ('Z=' + '{0:0.3f}'.format(self._posz) + 'mm_' +
                       'Y=' + '{0:0.3f}'.format(self._posy) + 'mm')
        elif self._scan_axis == 'y':
            pos_str = ('Z=' + '{0:0.3f}'.format(self._posz) + 'mm_' +
                       'X=' + '{0:0.3f}'.format(self._posx) + 'mm')
        elif self._scan_axis == 'z':
            pos_str = ('Y=' + '{0:0.3f}'.format(self._posy) + 'mm_' +
                       'X=' + '{0:0.3f}'.format(self._posx) + 'mm')

        if idx is not None:
            filename = (pos_str + '_' + dataset.description + '_' +
                        str(idx) + '.dat')
        else:
            filename = pos_str + '_' + dataset.description + '.dat'

        datadir = _os.path.join(self.dirpath, 'data')
        if not _os.path.isdir(datadir):
            _os.mkdir(datadir)

        filename = _os.path.join(datadir, filename)

        if self._cconfig is not None and self._cconfig.filename is not None:
            cconfig_filename = _os.path.split(self._cconfig.filename)[1]
        else:
            cconfig_filename = ''

        if self._mconfig is not None and self._mconfig.filename is not None:
            mconfig_filename = _os.path.split(self._mconfig.filename)[1]
        else:
            mconfig_filename = ''

        if (self._calibration is not None and
           self._calibration.filename is not None):
            calibration_filename = _os.path.split(
                self._calibration.filename)[1]
        else:
            calibration_filename = ''

        _save_scan_file(filename, dataset, self._timestamp, cconfig_filename,
                        mconfig_filename, calibration_filename)

    def __str__(self):
        """Printable string representation of LineScan."""
        fmtstr = '{0:<25s} : {1}\n'
        r = ''
        r += fmtstr.format('scan_axis', '"%s"' % self._scan_axis)
        r += fmtstr.format('number_of_scans', self.nr_scans)
        if self._scan_axis == 'x':
            r += fmtstr.format('position y [mm]', self._posy)
            r += fmtstr.format('position z [mm]', self._posz)
        elif self._scan_axis == 'y':
            r += fmtstr.format('position x [mm]', self._posx)
            r += fmtstr.format('position z [mm]', self._posz)
        elif self._scan_axis == 'z':
            r += fmtstr.format('position x [mm]', self._posx)
            r += fmtstr.format('position y [mm]', self._posy)
        if len(self._timestamp) != 0:
            r += fmtstr.format('timestamp', self._timestamp)
        r += fmtstr.format('save directory', self.dirpath)
        if self._cconfig is not None and self._cconfig.filename is not None:
            filename = _os.path.split(self._cconfig.filename)[-1]
            r += fmtstr.format('control configuration', filename)
        if self._mconfig is not None and self._mconfig.filename is not None:
            filename = _os.path.split(self._mconfig.filename)[-1]
            r += fmtstr.format('measurement configuration', filename)
        if (self._calibration is not None and
           self._calibration.filename is not None):
            filename = _os.path.split(self._calibration.filename)[-1]
            r += fmtstr.format('calibration data', filename)
        return r


class Measurement(object):
    """Measurement data."""

    def __init__(self, cconfig, mconfig, calibration, dirpath):
        """Initialize variables.

        Args:
            cconfig (ControlConfiguration or str): control configuration.
            mconfig (MeasurementConfiguration or str): measurement config.
            calibration (CalibrationData or str): probe calibration data.
            dirpath (str): directory path to save files.
        """
        self.dirpath = dirpath
        if not _os.path.isdir(self.dirpath):
            try:
                _os.mkdir(self.dirpath)
            except Exception:
                raise MeasurementDataError('Invalid directory path.')

        self._set_configuration(cconfig, mconfig, calibration)

        self._scan_axis = None
        self._scan_positions = []
        self._data = {}

    @property
    def scan_axis(self):
        """Scan axis label."""
        return self._scan_axis

    @property
    def scan_positions(self):
        """Scan axis positin values."""
        return self._scan_positions

    @property
    def data(self):
        """Measurement data."""
        return self._data

    def add_line_scan(self, ls):
        """Add a line scan to measurement data."""
        if self._scan_axis is None:
            self._scan_axis = ls.scan_axis
            self._scan_positions = ls.scan_positions

        if (ls.scan_axis == self._scan_axis and
           all(ls.scan_positions == self._scan_positions)):
            if self._scan_axis == 'x':
                if ls.posz not in self._data.keys():
                    self._data[ls.posz] = {}
                self._data[ls.posz][ls.posy] = ls
            elif self._scan_axis == 'y':
                if ls.posz not in self._data.keys():
                    self._data[ls.posz] = {}
                self._data[ls.posz][ls.posx] = ls
            elif self._scan_axis == 'z':
                if ls.posy not in self._data.keys():
                    self._data[ls.posy] = {}
                self._data[ls.posy][ls.posx] = ls
            else:
                raise MeasurementDataError('Invalid line scan.')
        else:
            raise MeasurementDataError('Invalid line scan.')

    def clear(self):
        """Clear Measurement."""
        self._scan_axis = None
        self._scan_positions = []
        self._data = {}

    def save_measurement(self, name):
        """Save measurement data."""
        if len(self._data) == 0:
            message = "Empty measurement."
            raise MeasurementDataError(message)

        t = _time.localtime()
        date = _time.strftime('%Y-%m-%d', t)
        datetime = _time.strftime('%Y-%m-%d_%H-%M-%S', t)

        field_data = self._get_field_avg_data()

        filename = '{0:1s}_{1:1s}.dat'.format(date, name)
        f = open(_os.path.join(self.dirpath, filename), 'w')

        f.write('fieldmap_name:     \t{0:1s}\n'.format(name))
        f.write('timestamp:         \t{0:1s}\n'.format(datetime))
        f.write('filename:          \t{0:1s}\n'.format(filename))
        f.write('nr_magnets:        \t1\n')
        f.write('\n')
        f.write('magnet_name:       \t{0:1s}\n'.format(name))
        f.write('gap[mm]:           \t\n')
        f.write('control_gap:       \t--\n')
        f.write('magnet_length[mm]: \t\n')
        f.write('current_main[A]:   \t\n')
        f.write('NI_main[A.esp]:    \t\n')
        f.write('center_pos_z[mm]:  \t0\n')
        f.write('center_pos_x[mm]:  \t0\n')
        f.write('rotation[deg]:     \t0\n')
        f.write('\n')
        f.write('X[mm]\tY[mm]\tZ[mm]\tBx\tBy\tBz [T]\n')
        f.write('-----------------------------------------------' +
                '----------------------------------------------\n')

        for i in range(field_data.shape[0]):
            f.write('{0:0.3f}\t{1:0.3f}\t{2:0.3f}\t'.format(
                field_data[i, 0], field_data[i, 1], field_data[i, 2]))
            f.write('{0:0.10e}\t{1:0.10e}\t{2:0.10e}\n'.format(
                field_data[i, 3], field_data[i, 4], field_data[i, 5]))
        f.close()

    def recover_saved_data(self):
        """Recover measurement data saved in files."""
        datapath = _os.path.join(self.dirpath, 'data')
        files = [f for f in _os.listdir(datapath) if f.endswith('.dat')]

        for filename in files:
            pos_str = '_'.join(filename.split('_')[:2])
            ls = LineScan.read_from_files(self.dirpath, pos_str=pos_str)
            self.add_line_scan(ls)

    def _set_configuration(self, cconfig, mconfig, calibration):
        if isinstance(cconfig, str) and _os.path.isfile(cconfig):
            self._cconfig = _configuration.ControlConfiguration(cconfig)
        elif isinstance(cconfig, _configuration.ControlConfiguration):
            self._cconfig = cconfig
        else:
            self._cconfig = None

        if isinstance(mconfig, str) and _os.path.isfile(mconfig):
            self._mconfig = _configuration.MeasurementConfiguration(mconfig)
        elif isinstance(mconfig, _configuration.MeasurementConfiguration):
            self._mconfig = mconfig
        else:
            self._mconfig = None

        if isinstance(calibration, str) and _os.path.isfile(calibration):
            self._calibration = _calibration.CalibrationData(calibration)
        elif isinstance(calibration, _calibration.CalibrationData):
            self._calibration = calibration
        else:
            self._calibration = None

    def _get_positions(self):
        data_list = []
        for d in self._data.values():
            for v in d.values():
                data_list.append(v)

        if self._scan_axis == 'x':
            posx = self._scan_positions
            posy = sorted(list(set([ls.posy for ls in data_list])))
            posz = sorted(list(set([ls.posz for ls in data_list])))
        elif self._scan_axis == 'y':
            posx = sorted(list(set([ls.posx for ls in data_list])))
            posy = self._scan_positions
            posz = sorted(list(set([ls.posz for ls in data_list])))
        elif self._scan_axis == 'z':
            posx = sorted(list(set([ls.posx for ls in data_list])))
            posy = sorted(list(set([ls.posy for ls in data_list])))
            posz = self._scan_positions

        return posx, posy, posz

    def _get_field_avg_at_point(self, posx, posy, posz):
        try:
            if self._scan_axis == 'x':
                ls = self._data[posz][posy]
                idx = _np.where(ls.posx == posx)[0][0]
            elif self._scan_axis == 'y':
                ls = self._data[posz][posx]
                idx = _np.where(ls.posy == posy)[0][0]
            elif self._scan_axis == 'z':
                ls = self._data[posy][posx]
                idx = _np.where(ls.posz == posz)[0][0]

            bx = ls.field_avg.datax[idx]
            by = ls.field_avg.datay[idx]
            bz = ls.field_avg.dataz[idx]
        except Exception:
            bx, by, bz = _np.nan, _np.nan, _np.nan

        return (bx, by, bz)

    def _get_field_avg_data(self):
        posx_vec, posy_vec, posz_vec = self._get_positions()

        size = len(posx_vec)*len(posy_vec)*len(posz_vec)
        field_data = _np.zeros([size, 6])

        count = 0
        for posz in posz_vec:
            for posy in posy_vec:
                for posx in posx_vec:
                    bx, by, bz = self._get_field_avg_at_point(posx, posy, posz)
                    field_data[count, 0] = posx
                    field_data[count, 1] = posy
                    field_data[count, 2] = posz
                    field_data[count, 3] = bx
                    field_data[count, 4] = by
                    field_data[count, 5] = bz
                    count += 1

        return field_data


def _get_scan_axis(posx, posy, posz):
    if (not isinstance(posx, (int, float)) and
       isinstance(posy, (int, float)) and
       isinstance(posz, (int, float))):
        return 'x'
    elif (not isinstance(posy, (int, float)) and
          isinstance(posx, (int, float)) and
          isinstance(posz, (int, float))):
        return 'y'
    elif (not isinstance(posz, (int, float)) and
          isinstance(posx, (int, float)) and
          isinstance(posy, (int, float))):
        return 'z'
    else:
        return None


def _get_scan_position(posx, posy, posz):
    if (not isinstance(posx, (int, float)) and
       isinstance(posy, (int, float)) and
       isinstance(posz, (int, float))):
        return posx
    elif (not isinstance(posy, (int, float)) and
          isinstance(posx, (int, float)) and
          isinstance(posz, (int, float))):
        return posy
    elif (not isinstance(posz, (int, float)) and
          isinstance(posx, (int, float)) and
          isinstance(posy, (int, float))):
        return posz
    else:
        return None


def _get_scan_files_list(
        datadir, posx=None, posy=None, posz=None, pos_str=None):
    if pos_str is None:
        if posx is None:
            pos_str = ('Z=' + '{0:0.3f}'.format(posz) + 'mm_' +
                       'Y=' + '{0:0.3f}'.format(posy) + 'mm')
        elif posy is None:
            pos_str = ('Z=' + '{0:0.3f}'.format(posz) + 'mm_' +
                       'X=' + '{0:0.3f}'.format(posx) + 'mm')
        elif posz is None:
            pos_str = ('Y=' + '{0:0.3f}'.format(posy) + 'mm_' +
                       'X=' + '{0:0.3f}'.format(posx) + 'mm')
        else:
            message = 'Invalid position arguments for LineScan.'
            raise MeasurementDataError(message)

    tmpfiles = [f for f in _os.listdir(datadir) if f.endswith('.dat')]

    files = [_os.path.join(datadir, f) for f in tmpfiles
             if pos_str in f]

    if len(files) == 0:
        message = 'No files found for the specified position.'
        raise MeasurementDataError(message)

    return files


def _save_scan_file(filename, dataset, timestamp, cconfig_filename,
                    mconfig_filename, calibration_filename):
    scan_axis = _get_scan_axis(dataset.posx, dataset.posy, dataset.posz)

    if scan_axis == 'x':
        pos_str = 'z[mm]'
        pos_values = dataset.posx
    elif scan_axis == 'y':
        pos_str = 'y[mm]'
        pos_values = dataset.posy
    elif scan_axis == 'z':
        pos_str = 'z[mm]'
        pos_values = dataset.posz
    else:
        message = 'Invalid scan axis.'
        raise MeasurementDataError(message)

    columns_names = (
        '%s\t' % pos_str +
        '%s_z[%s]\t' % (dataset.description, dataset.unit) +
        '%s_y[%s]\t' % (dataset.description, dataset.unit) +
        '%s_z[%s]' % (dataset.description, dataset.unit))
    columns = _np.column_stack((
        pos_values, dataset.datax, dataset.datay, dataset.dataz))

    f = open(filename, mode='w')

    f.write('data_type:                \t%s\n' % dataset.description)
    f.write('data_unit:                \t%s\n' % dataset.unit)
    f.write('timestamp:                \t%s\n' % timestamp)
    f.write('control_configuration:    \t%s\n' % cconfig_filename)
    f.write('measurement_configuration:\t%s\n' % mconfig_filename)
    f.write('calibration:              \t%s\n' % calibration_filename)
    f.write('scan_axis:                \t%s\n' % scan_axis)

    if scan_axis == 'x':
        f.write('position_x[mm]:          \t--\n')
        f.write('position_y[mm]:          \t%f\n' % dataset.posy)
        f.write('position_z[mm]:          \t%f\n' % dataset.posz)
    elif scan_axis == 'y':
        f.write('position_x[mm]:          \t%f\n' % dataset.posx)
        f.write('position_y[mm]:          \t--\n')
        f.write('position_z[mm]:          \t%f\n' % dataset.posz)
    elif scan_axis == 'z':
        f.write('position_x[mm]:          \t%f\n' % dataset.posx)
        f.write('position_y[mm]:          \t%f\n' % dataset.posy)
        f.write('position_z[mm]:          \t--\n')

    f.write('\n')
    f.write('%s\n' % columns_names)
    f.write('---------------------------------------------------' +
            '---------------------------------------------------\n')

    for i in range(columns.shape[0]):
        line = '{0:0.3f}'.format(columns[i, 0])
        for j in range(1, columns.shape[1]):
            line = line + '\t' + '{0:0.10e}'.format(columns[i, j])
        f.write(line + '\n')
    f.close()


def _read_scan_file(filename):
    flines = _files.read_file(filename)

    data_type = _files.find_value(flines, 'data_type')
    data_unit = _files.find_value(flines, 'data_unit')
    timestamp = _files.find_value(flines, 'timestamp')
    cconfig_filename = _files.find_value(flines, 'control_configuration')
    mconfig_filename = _files.find_value(flines, 'measurement_configuration')
    calibration_filename = _files.find_value(flines, 'calibration')
    scan_axis = _files.find_value(flines, 'scan_axis')
    posx = _files.find_value(flines, 'position_x')
    posy = _files.find_value(flines, 'position_y')
    posz = _files.find_value(flines, 'position_z')

    idx = next((i for i in range(len(flines))
                if flines[i].find("----------") != -1), None)
    data = []
    for line in flines[idx+1:]:
        data_line = [float(d) for d in line.split('\t')]
        data.append(data_line)
    data = _np.array(data)

    dataset = DataSet(data_type, data_unit)

    if data.shape[1] == 4:
        scan_positions = data[:, 0]
        dataset.datax = data[:, 1]
        dataset.datay = data[:, 2]
        dataset.dataz = data[:, 3]
    else:
        message = 'Inconsistent number of columns in file: %s' % filename
        raise MeasurementDataError(message)

    if scan_axis.lower() == 'x':
        dataset.posx = scan_positions
        dataset.posy = float(posy)
        dataset.posz = float(posz)
    elif scan_axis.lower() == 'y':
        dataset.posx = float(posx)
        dataset.posy = scan_positions
        dataset.posz = float(posz)
    elif scan_axis.lower() == 'z':
        dataset.posx = float(posx)
        dataset.posy = float(posy)
        dataset.posz = scan_positions
    else:
        message = 'Invalid scan axis found in file: %s' % filename
        raise MeasurementDataError(message)

    return (dataset, timestamp, cconfig_filename,
            mconfig_filename, calibration_filename)
