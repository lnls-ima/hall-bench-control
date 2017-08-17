# -*- coding: utf-8 -*-
"""Implementation of classes to store and analyse measurement data."""

import os as _os
import numpy as _np
from scipy import interpolate as _interpolate
from scipy.integrate import cumtrapz as _cumtrapz

from . import utils as _utils
from . import configuration as _configuration
from . import calibration as _calibration


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
        self._posx = _np.array([])
        self._posy = _np.array([])
        self._posz = _np.array([])
        self._datax = _np.array([])
        self._datay = _np.array([])
        self._dataz = _np.array([])

    @property
    def posx(self):
        """X position."""
        return self._posx

    @posx.setter
    def posx(self, value):
        if not isinstance(value, _np.ndarray):
            value = _np.array(value)
        self._posx = value

    @property
    def posy(self):
        """Y position."""
        return self._posy

    @posy.setter
    def posy(self, value):
        if value is not None:
            if not isinstance(value, _np.ndarray):
                value = _np.array(value)
        else:
            value = _np.array([])
        self._posy = value

    @property
    def posz(self):
        """Z position."""
        return self._posz

    @posz.setter
    def posz(self, value):
        if value is not None:
            if not isinstance(value, _np.ndarray):
                value = _np.array(value)
        else:
            value = _np.array([])
        self._posz = value

    @property
    def datax(self):
        """X data."""
        return self._datax

    @datax.setter
    def datax(self, value):
        if value is not None:
            if not isinstance(value, _np.ndarray):
                value = _np.array(value)
        else:
            value = _np.array([])
        self._datax = value

    @property
    def datay(self):
        """Y data."""
        return self._datay

    @datay.setter
    def datay(self, value):
        if value is not None:
            if not isinstance(value, _np.ndarray):
                value = _np.array(value)
        else:
            value = _np.array([])
        self._datay = value

    @property
    def dataz(self):
        """Z data."""
        return self._dataz

    @dataz.setter
    def dataz(self, value):
        if value is not None:
            if not isinstance(value, _np.ndarray):
                value = _np.array(value)
        else:
            value = _np.array([])
        self._dataz = value

    @staticmethod
    def copy(dataset):
        """Return a copy of a DataSet."""
        dataset_copy = DataSet()
        dataset_copy.unit = dataset.unit
        dataset_copy.description = dataset.description
        dataset_copy.posx = _np.copy(dataset.posx)
        dataset_copy.posy = _np.copy(dataset.posy)
        dataset_copy.posz = _np.copy(dataset.posz)
        dataset_copy.datax = _np.copy(dataset.datax)
        dataset_copy.datay = _np.copy(dataset.datay)
        dataset_copy.dataz = _np.copy(dataset.dataz)
        return dataset_copy

    def reverse(self):
        """Reverse DataSet."""
        if self.posx.size > 1:
            self.posx = self.posx[::-1]

        if self.posy.size > 1:
            self.posy = self.posy[::-1]

        if self.posz.size > 1:
            self.posz = self.posz[::-1]

        if self.datax.size > 1:
            self.datax = self.datax[::-1]

        if self.datay.size > 1:
            self.datay = self.datay[::-1]

        if self.dataz.size > 1:
            self.dataz = self.dataz[::-1]

    def clear(self):
        """Clear DataSet."""
        self.posx = _np.array([])
        self.posy = _np.array([])
        self.posz = _np.array([])
        self.datax = _np.array([])
        self.datay = _np.array([])
        self.dataz = _np.array([])


class LineScan(object):
    """Line scan data."""

    def __init__(self, posx, posy, posz, calibration_data, dirpath,
                 overwrite_calibration=False):
        """Initialize variables.

        Args:
            posx (float or array): x position of the scan line.
            posy (float or array): y position of the scan line.
            posz (float or array): z position of the scan line.
            calibration_data (CalibrationData): probe calibration data.
            dirpath (str): directory path to save files.
            overwrite_calibration (bool): if True overwrite calibration file.
        """
        if not isinstance(posx, _np.ndarray):
            posx = _np.array(posx)

        if not isinstance(posy, _np.ndarray):
            posy = _np.array(posy)

        if not isinstance(posz, _np.ndarray):
            posz = _np.array(posz)

        self._scan_axis = _get_scan_axis(posx, posy, posz)
        if self._scan_axis is None:
            raise ValueError('Invalid position arguments for LineScan.')

        self._posx = _np.around(posx, decimals=4)
        self._posy = _np.around(posy, decimals=4)
        self._posz = _np.around(posz, decimals=4)

        if isinstance(dirpath, str):
            self.dirpath = dirpath
            if not _os.path.isdir(self.dirpath):
                try:
                    _os.mkdir(self.dirpath)
                except Exception:
                    raise ValueError('Invalid value for dirpath.')
        else:
            self.dirpath = None
            raise TypeError('dirpath must be a string.')

        if isinstance(calibration_data, _calibration.CalibrationData):
            self._calibration_data = calibration_data
        else:
            self._calibration_data = None
            raise TypeError(
                'calibration_data must be a CalibrationData object.')

        self._save_calibration(overwrite_calibration)
        self._timestamp = ''
        self._voltage_raw = []
        self._voltage_interpolated = []
        self._voltage_avg = None
        self._voltage_std = None
        self._field_avg = None
        self._field_std = None
        self._field_first_integral = None
        self._field_second_integral = None

    def _save_calibration(self, overwrite_calibration):
        calibration_filename = 'calibration_data.txt'
        calibration_fullpath = _os.path.join(
            self.dirpath, calibration_filename)

        if not overwrite_calibration and _os.path.isfile(calibration_fullpath):
            tmp_calib_data = _calibration.CalibrationData(calibration_fullpath)
            if not self._calibration_data == tmp_calib_data:
                raise MeasurementDataError('Inconsistent calibration files.')

        self._calibration_data.save_file(calibration_fullpath)

    @staticmethod
    def copy(linescan):
        """Return a copy of a LineScan."""
        linescan_copy = LineScan(
            linescan.posx, linescan.posy, linescan.posz,
            linescan.calibration_data, linescan.dirpath)

        linescan_copy._timestamp = linescan.timestamp

        linescan_copy._voltage_raw = [
            DataSet.copy(s) for s in linescan.voltage_raw]

        linescan_copy._voltage_interpolated = [
            DataSet.copy(s) for s in linescan.voltage_interpolated]

        if linescan.voltage_avg is not None:
            linescan_copy._voltage_avg = DataSet.copy(linescan.voltage_avg)
        else:
            linescan_copy._voltage_avg = None

        if linescan.voltage_std is not None:
            linescan_copy._voltage_std = DataSet.copy(linescan.voltage_std)
        else:
            linescan_copy._voltage_std = None

        if linescan.field_avg is not None:
            linescan_copy._field_avg = DataSet.copy(linescan.field_avg)
        else:
            linescan_copy._field_avg = None

        if linescan.field_std is not None:
            linescan_copy._field_std = DataSet.copy(linescan.field_std)
        else:
            linescan_copy._field_std = None

        if linescan.field_first_integral is not None:
            linescan_copy._field_first_integral = DataSet.copy(
                linescan.field_first_integral)
        else:
            linescan_copy._field_first_integral = None

        if linescan.field_second_integral is not None:
            linescan_copy._field_second_integral = DataSet.copy(
                linescan.field_second_integral)
        else:
            linescan_copy._field_second_integral = None

        return linescan_copy

    @staticmethod
    def read_from_files(
            dirpath, posx=None, posy=None, posz=None, pos_str=None):
        """Read LineScan data from files."""
        datadir = _os.path.join(dirpath, 'data')
        files = _get_scan_files_list(
            datadir, posx=posx, posy=posy, posz=posz, pos_str=pos_str)

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
            dataset, ts = _read_scan_file(filename)
            timestamp.add(ts)

            if 'voltage_raw' in dataset.description:
                voltage_raw.append(dataset)
            elif 'voltage_interpolated' in dataset.description:
                voltage_interpolated.append(dataset)
            elif 'voltage_avg' in dataset.description:
                voltage_avg = dataset
            elif 'voltage_std' in dataset.description:
                voltage_std = dataset
            elif 'field_avg' in dataset.description:
                field_avg = dataset
            elif 'field_std' in dataset.description:
                field_std = dataset
            elif 'field_first_integral' in dataset.description:
                field_first_integral = dataset
            elif 'field_second_integral' in dataset.description:
                field_second_integral = dataset

        calibration_filename = 'calibration_data.txt'
        calibration_data = _calibration.CalibrationData(
            _os.path.join(dirpath, calibration_filename))

        linescan = LineScan(
            field_avg.posx, field_avg.posy, field_avg.posz,
            calibration_data, dirpath)

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
            return _np.array([])

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
    def calibration_data(self):
        """Calibration data."""
        return self._calibration_data

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
    def voltage_interpolated(self):
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
        self._timestamp = _utils.get_timestamp()

        if self.nr_scans != 0:
            self._data_interpolation(save_data)
            self._calculate_voltage_avg_std(save_data)
            self._calculate_field_avg_std(save_data)
            self._calculate_field_first_integral(save_data)
            self._calculate_field_second_integral(save_data)

    def clear(self):
        """Clear LineScan."""
        self._timestamp = ''
        self._voltage_raw = []
        self._voltage_interpolated = []
        self._voltage_avg = None
        self._voltage_std = None
        self._field_avg = None
        self._field_std = None
        self._field_first_integral = None
        self._field_second_integral = None

    def _valid_scan(self, scan):
        if self._valid_scan_positions(scan):
            if self._valid_scan_data(scan):
                return True
            else:
                return False
        else:
            return False

    def _valid_scan_positions(self, scan):
        scan_axis = _get_scan_axis(scan.posx, scan.posy, scan.posz)
        if scan_axis is not None and scan_axis == self._scan_axis:
            if (scan_axis == 'x' and
               _np.around(scan.posy, decimals=4) == self._posy and
               _np.around(scan.posz, decimals=4) == self._posz):
                return True
            elif (scan_axis == 'y' and
                  _np.around(scan.posx, decimals=4) == self._posx and
                  _np.around(scan.posz, decimals=4) == self._posz):
                return True
            elif (scan_axis == 'z' and
                  _np.around(scan.posx, decimals=4) == self._posx and
                  _np.around(scan.posy, decimals=4) == self._posy):
                return True
            else:
                return False
        else:
            return False

    def _valid_scan_data(self, scan):
        if len(scan.unit) == 0:
            scan.unit = 'V'

        if (scan.datax.size == 0 and
           scan.datay.size == 0 and scan.dataz.size == 0):
            return False

        scan_size = len(_get_scan_positions(scan.posx, scan.posy, scan.posz))
        if scan.datax.size == 0:
            scan.datax = _np.ones(scan_size)*_np.nan
        if scan.datay.size == 0:
            scan.datay = _np.ones(scan_size)*_np.nan
        if scan.dataz.size == 0:
            scan.dataz = _np.ones(scan_size)*_np.nan

        if (scan.datax.size == scan_size and
           scan.datay.size == scan_size and
           scan.dataz.size == scan_size):
            return True
        else:
            return False

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

            rawpos = _get_scan_positions(raw.posx, raw.posy, raw.posz)

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
        self._voltage_avg = DataSet()
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
        self._voltage_std = DataSet()
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
        if self._calibration_data is None:
            raise MeasurementDataError('Calibration data not found.')

        self._field_avg = DataSet()
        self._field_avg.description = 'field_avg'
        self._field_avg.unit = self._calibration_data.field_unit
        self._field_avg.posx = self._posx
        self._field_avg.posy = self._posy
        self._field_avg.posz = self._posz

        self._field_avg.datax = self._calibration_data.convert_probe_x(
            self._voltage_avg.datax)

        self._field_avg.datay = self._calibration_data.convert_probe_y(
            self._voltage_avg.datay)

        self._field_avg.dataz = self._calibration_data.convert_probe_z(
            self._voltage_avg.dataz)

        self._field_std = DataSet()
        self._field_std.description = 'field_std'
        self._field_std.unit = self._calibration_data.field_unit
        self._field_std.posx = self._posx
        self._field_std.posy = self._posy
        self._field_std.posz = self._posz

        self._field_std.datax = self._calibration_data.convert_probe_x(
            self._voltage_std.datax)

        self._field_std.datay = self._calibration_data.convert_probe_y(
            self._voltage_std.datay)

        self._field_std.dataz = self._calibration_data.convert_probe_z(
            self._voltage_std.dataz)

        if save_data:
            self._save_data(self._field_avg)
            self._save_data(self._field_std)

    def _calculate_field_first_integral(self, save_data=True):
        """Calculate the magnetic field first integral."""
        self._field_first_integral = DataSet()
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
        self._field_second_integral = DataSet()
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

        _write_scan_file(filename, dataset, self._timestamp)

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
        return r


class Measurement(object):
    """Measurement data."""

    def __init__(self, devices_config, measurement_config, dirpath,
                 overwrite_config=False):
        """Initialize variables.

        Args:
            devices_config (DevicesConfig): devices configuration.
            measurement_config (MeasurementConfig): measurement configuration.
            dirpath (str): directory path to save files.
            overwrite_config (bool): if True overwrite configuration files.
        """
        if isinstance(dirpath, str):
            self.dirpath = dirpath
            if not _os.path.isdir(self.dirpath):
                try:
                    _os.mkdir(self.dirpath)
                except Exception:
                    raise ValueError('Invalid value for dirpath.')
        else:
            self.dirpath = None
            raise TypeError('dirpath must be a string.')

        if isinstance(devices_config, _configuration.DevicesConfig):
            self._devices_config = devices_config
        else:
            raise TypeError('devices_config must be a DevicesConfig object.')

        if isinstance(measurement_config, _configuration.MeasurementConfig):
            self._measurement_config = measurement_config
        else:
            raise TypeError(
                'measurement_config must be a MeasurementConfig object.')

        self._save_configuration(overwrite_config)
        self._scan_axis = None
        self._data = {}

    def _save_configuration(self, overwrite_config):
        dc_filename = 'devices_configuration.txt'
        mc_filename = 'measurement_configuration.txt'

        dc_fullpath = _os.path.join(self.dirpath, dc_filename)
        mc_fullpath = _os.path.join(self.dirpath, mc_filename)

        if not overwrite_config and _os.path.isfile(dc_fullpath):
            tmp_devices_config = _configuration.DevicesConfig(dc_fullpath)
            if not self._devices_config == tmp_devices_config:
                raise MeasurementDataError('Inconsistent configuration files.')

        if not overwrite_config and _os.path.isfile(mc_fullpath):
            tmp_meas_config = _configuration.MeasurementConfig(mc_fullpath)
            if not self._measurement_config == tmp_meas_config:
                raise MeasurementDataError('Inconsistent configuration files.')

        self._devices_config.save_file(dc_fullpath)
        self._measurement_config.save_file(mc_fullpath)

    @property
    def scan_axis(self):
        """Scan axis label."""
        return self._scan_axis

    @property
    def scan_positions(self):
        """Scan axis positin values."""
        if self._scan_axis == 'x':
            return self.posx
        elif self._scan_axis == 'y':
            return self.posy
        elif self._scan_axis == 'z':
            return self.posz
        else:
            return _np.array([])

    @property
    def posx(self):
        """X position."""
        return self._get_position_list('x')

    @property
    def posy(self):
        """Y position."""
        return self._get_position_list('y')

    @property
    def posz(self):
        """Z position."""
        return self._get_position_list('z')

    @property
    def data(self):
        """Measurement data."""
        return self._data

    @property
    def devices_config(self):
        """Device configuration."""
        return self._devices_config

    @property
    def measurement_config(self):
        """Measurement configuration."""
        return self._measurement_config

    def clear(self):
        """Clear Measurement."""
        self._scan_axis = None
        self._data = {}

    def add_line_scan(self, ls):
        """Add a line scan to measurement data."""
        if ls.nr_scans == 0:
            raise MeasurementDataError('Empty LineScan.')

        if self._scan_axis is None:
            self._scan_axis = ls.scan_axis

        if ls.scan_axis == self._scan_axis:
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
                raise MeasurementDataError('Invalid LineScan.')
        else:
            raise MeasurementDataError('Invalid LineScan.')

    def recover_saved_data(self):
        """Recover measurement data saved in files."""
        datapath = _os.path.join(self.dirpath, 'data')
        files = [f for f in _os.listdir(datapath) if f.endswith('.dat')]

        for filename in files:
            pos_str = '_'.join(filename.split('_')[:2])
            ls = LineScan.read_from_files(self.dirpath, pos_str=pos_str)
            self.add_line_scan(ls)

    def save(self, magnet_name='', magnet_length='', gap='',
             control_gap='', coils=[], reference_position=[0, 0, 0]):
        """Save measurement data.

        Args:
            magnet_name (str): magnet name.
            magnet_length (str or float): magnet length [mm].
            gap (str or float): air gap length [mm].
            control_gap (str or float) : control air gap length [mm].
            coils (list of dicts): list of dictionaries with name, current
                                   and turns of each magnet coil.
            reference_position (list): reference position of the magnet.
        """
        if len(self._data) == 0:
            message = "Empty measurement."
            raise MeasurementDataError(message)

        datetime = _utils.get_timestamp()
        date = datetime.split('_')[0]

        field_data = self._get_field_avg_data()

        if len(magnet_name) == 0:
            fieldmap_name = 'hall_probe_measurement'
        else:
            fieldmap_name = magnet_name
            for coil in coils:
                coil_symbol = coil['name']
                if len(coil_symbol) > 2:
                    coil_symbol = coil_symbol[0] + 'c'
                fieldmap_name = (
                    fieldmap_name + '_' +
                    + 'I' + coil_symbol + '=' + str(coil['current']) + 'A')

        filename = '{0:1s}_{1:1s}.dat'.format(date, fieldmap_name)
        f = open(_os.path.join(self.dirpath, filename), 'w')

        f.write('fieldmap_name:      \t{0:1s}\n'.format(fieldmap_name))
        f.write('timestamp:          \t{0:1s}\n'.format(datetime))
        f.write('filename:           \t{0:1s}\n'.format(filename))
        f.write('nr_magnets:         \t1\n')
        f.write('\n')
        f.write('magnet_name:        \t{0:1s}\n'.format(magnet_name))
        f.write('gap[mm]:            \t{0:1s}\n'.format(str(gap)))
        f.write('control_gap:        \t{0:1s}\n'.format(str(control_gap)))
        f.write('magnet_length[mm]:  \t{0:1s}\n'.format(str(magnet_length)))

        for coil in coils:
            current_label = 'current_{0:1s}[A]:'.format(coil['name']).ljust(20)
            turns_label = 'nr_turns_{0:1s}:'.format(coil['name']).ljust(20)
            f.write(current_label + '\t{1:1s}\n'.format(str(coil['current'])))
            f.write(turns_label + '\t{1:1s}\n'.format(str(coil['turns'])))

        f.write('center_pos_z[mm]:   \t0\n')
        f.write('center_pos_x[mm]:   \t0\n')
        f.write('rotation[deg]:      \t0\n')
        f.write('\n')
        f.write('X[mm]\tY[mm]\tZ[mm]\tBx\tBy\tBz [T]\n')
        f.write('-----------------------------------------------' +
                '----------------------------------------------\n')

        for i in range(field_data.shape[0]):
            x = field_data[i, 0] - reference_position[0]
            y = field_data[i, 1] - reference_position[1]
            z = field_data[i, 2] - reference_position[2]
            f.write('{0:0.3f}\t{0:0.3f}\t{0:0.3f}\t'.format(x, y, z))
            f.write('{0:0.10e}\t{1:0.10e}\t{2:0.10e}\n'.format(
                field_data[i, 3], field_data[i, 4], field_data[i, 5]))
        f.close()

        return filename

    def _get_field_avg_data(self):
        size = len(self.posx)*len(self.posy)*len(self.posz)
        field_data = _np.zeros([size, 6])

        count = 0
        for z in self.posz:
            for y in self.posy:
                for x in self.posx:
                    bx, by, bz = self._get_field_avg_at_point(x, y, z)
                    field_data[count, 0] = x
                    field_data[count, 1] = y
                    field_data[count, 2] = z
                    field_data[count, 3] = bx
                    field_data[count, 4] = by
                    field_data[count, 5] = bz
                    count += 1

        return field_data

    def _get_position_list(self, axis):
        pos = set()
        for data_value in self._data.values():
            for ls in data_value.values():
                ls_pos = getattr(ls, 'pos' + axis.lower())
                if ls_pos.size > 1:
                    pos.update(ls_pos)
                else:
                    pos.add(ls_pos)
        pos = _np.array(sorted(list(pos)))
        return pos

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


def _get_scan_axis(posx, posy, posz):
    if posx is not None:
        if not isinstance(posx, _np.ndarray):
            posx = _np.array(posx)
    else:
        posx = _np.array([])

    if posy is not None:
        if not isinstance(posy, _np.ndarray):
            posy = _np.array(posy)
    else:
        posy = _np.array([])

    if posz is not None:
        if not isinstance(posz, _np.ndarray):
            posz = _np.array(posz)
    else:
        posz = _np.array([])

    if posx.size > 1 and posy.size == 1 and posz.size == 1:
        return 'x'
    elif posy.size > 1 and posx.size == 1 and posz.size == 1:
        return 'y'
    elif posz.size > 1 and posx.size == 1 and posy.size == 1:
        return 'z'
    else:
        return None


def _get_scan_positions(posx, posy, posz):
    if posx is not None:
        if not isinstance(posx, _np.ndarray):
            posx = _np.array(posx)
    else:
        posx = _np.array([])

    if posy is not None:
        if not isinstance(posy, _np.ndarray):
            posy = _np.array(posy)
    else:
        posy = _np.array([])

    if posz is not None:
        if not isinstance(posz, _np.ndarray):
            posz = _np.array(posz)
    else:
        posz = _np.array([])

    if posx.size > 1 and posy.size == 1 and posz.size == 1:
        return posx
    elif posy.size > 1 and posx.size == 1 and posz.size == 1:
        return posy
    elif posz.size > 1 and posx.size == 1 and posy.size == 1:
        return posz
    else:
        return _np.array([])


def _get_scan_files_list(
        datadir, posx=None, posy=None, posz=None, pos_str=None):
    if pos_str is None:
        if posx is None and posy is not None and posz is not None:
            pos_str = ('Z=' + '{0:0.3f}'.format(posz) + 'mm_' +
                       'Y=' + '{0:0.3f}'.format(posy) + 'mm')
        elif posy is None and posx is not None and posz is not None:
            pos_str = ('Z=' + '{0:0.3f}'.format(posz) + 'mm_' +
                       'X=' + '{0:0.3f}'.format(posx) + 'mm')
        elif posz is None and posx is not None and posy is not None:
            pos_str = ('Y=' + '{0:0.3f}'.format(posy) + 'mm_' +
                       'X=' + '{0:0.3f}'.format(posx) + 'mm')
        else:
            message = 'Invalid position arguments for LineScan.'
            raise ValueError(message)

    pos_str_reverse = '_'.join(pos_str.split('_')[::-1])

    tmpfiles = [f for f in _os.listdir(datadir) if f.endswith('.dat')]

    files = [_os.path.join(datadir, f) for f in tmpfiles
             if (pos_str in f or pos_str_reverse in f)]

    if len(files) == 0:
        message = 'No files found for the specified position.'
        raise MeasurementDataError(message)

    return files


def _read_scan_file(filename):
    flines = _utils.read_file(filename)

    data_type = _utils.find_value(flines, 'data_type')
    data_unit = _utils.find_value(flines, 'data_unit')
    timestamp = _utils.find_value(flines, 'timestamp')
    scan_axis = _utils.find_value(flines, 'scan_axis')
    posx = _utils.find_value(flines, 'position_x')
    posy = _utils.find_value(flines, 'position_y')
    posz = _utils.find_value(flines, 'position_z')

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

    return (dataset, timestamp)


def _write_scan_file(filename, dataset, timestamp):
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
        '%s_x[%s]\t' % (dataset.description, dataset.unit) +
        '%s_y[%s]\t' % (dataset.description, dataset.unit) +
        '%s_z[%s]' % (dataset.description, dataset.unit))
    columns = _np.column_stack((
        pos_values, dataset.datax, dataset.datay, dataset.dataz))

    f = open(filename, mode='w')

    f.write('data_type:               \t%s\n' % dataset.description)
    f.write('data_unit:               \t%s\n' % dataset.unit)
    f.write('timestamp:               \t%s\n' % timestamp)
    f.write('scan_axis:               \t%s\n' % scan_axis)

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
