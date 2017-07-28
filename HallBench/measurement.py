# -*- coding: utf-8 -*-
"""Implementation of classes to store and analyse measurement data."""

import os as _os
import time as _time
import numpy as _np
import shutil as _shutil
from scipy import interpolate as _interpolate
from scipy.integrate import cumtrapz as _cumtrapz


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
            unit (str): data unit.
            description (str): data description.
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
            cconfig (ControlConfiguration): control configuration data.
            mconfig (MeasurementConfiguration): measurement configuration data.
            calibration (CalibrationData): probe calibration data.
            dirpath (str): directory path to save files.
        """
        if isinstance(posx, (int, float)):
            self.posx = round(posx, 4)
        else:
            self.posx = _np.around(posx, decimals=4)

        if isinstance(posy, (int, float)):
            self.posy = round(posy, 4)
        else:
            self.posy = _np.around(posy, decimals=4)

        if isinstance(posz, (int, float)):
            self.posz = round(posz, 4)
        else:
            self.posz = _np.around(posz, decimals=4)

        self.cconfig = cconfig
        self.mconfig = mconfig
        self.calibration = calibration
        self.dirpath = dirpath

        self._scan_axis = None
        self._timestamp = ''
        self._raw = []
        self._interpolated = []
        self._avg_voltage = DataSet()
        self._std_voltage = DataSet()
        self._avg_field = DataSet()
        self._std_field = DataSet()
        self._first_integral = DataSet()
        self._second_integral = DataSet()

        self._set_scan_axis()

    @staticmethod
    def copy(linescan):
        """Return a copy of a LineScan."""
        lsc = LineScan()
        lsc.posx = linescan.posx
        lsc.posy = linescan.posy
        lsc.posz = linescan.posz
        lsc.cconfig = linescan.cconfig
        lsc.mconfig = linescan.mconfig
        lsc.calibration = linescan.calibration
        lsc.dirpath = linescan.dirpath
        lsc._scan_axis = linescan._scan_axis
        lsc._timestamp = linescan._timestamp
        lsc._raw = [DataSet().copy(s) for s in linescan._raw]
        lsc._interpolated = [DataSet().copy(s) for s in linescan._interpolated]
        lsc._avg_voltage = DataSet().copy(linescan._avg_voltage)
        lsc._std_voltage = DataSet().copy(linescan._std_voltage)
        lsc._avg_field = DataSet().copy(linescan._avg_field)
        lsc._std_field = DataSet().copy(linescan._std_field)
        lsc._first_integral = DataSet().copy(linescan._first_integral)
        lsc._second_integral = DataSet().copy(linescan._second_integral)
        return lsc

    @property
    def scan_axis(self):
        """Scan axis label."""
        return self._scan_axis

    @property
    def scan_positions(self):
        """Scan axis position values."""
        if self._scan_axis == 'x':
            return self.posx
        elif self._scan_axis == 'y':
            return self.posy
        elif self._scan_axis == 'z':
            return self.posz
        else:
            return []

    @property
    def nr_scans(self):
        """Number of scans."""
        return len(self._raw)

    @property
    def timestamp(self):
        """Scan timestamp."""
        return self._timestamp

    @property
    def raw(self):
        """List with raw scan data."""
        return self._raw

    @property
    def interpolated(self):
        """List with the interpolated scan data."""
        return self._interpolated

    @property
    def avg_voltage(self):
        """Average voltage values."""
        return self._avg_voltage

    @property
    def std_voltage(self):
        """Standard deviation of voltage values."""
        return self._std_voltage

    @property
    def avg_field(self):
        """Average magnetic field values."""
        return self._avg_field

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

    def add_scan(self, scan):
        """Add a scan to the list."""
        if self._valid_scan(scan):
            self._raw.append(scan)
        else:
            raise MeasurementDataError('Invalid scan.')

    def analyse_and_save_data(self):
        """Analyse and save the line scan data."""
        self._timestamp = (
            _time.strftime('%Y-%m-%d_%H-%M-%S', _time.localtime()))

        if self.nr_scans != 0:
            self._data_interpolation()
            self._calculate_voltage_avg_std()
            self._convert_voltage_to_field()
            self._calculate_field_first_integral()
            self._calculate_field_second_integral()

        # copy configuration files
        try:
            _shutil.copy(self.cconfig.filename, self.dirpath)
            _shutil.copy(self.mconfig.filename, self.dirpath)
            _shutil.copy(self.calibration.filename, self.dirpath)
        except Exception:
            pass

    def clear(self):
        """Clear LineScan."""
        self._scan_axis = None
        self._timestamp = ''
        self._raw = []
        self._interpolated = []
        self._avg_voltage = DataSet()
        self._std_voltage = DataSet()
        self._avg_field = DataSet()
        self._std_field = DataSet()
        self._first_integral = DataSet()
        self._second_integral = DataSet()

    def _set_scan_axis(self):
        scan_axis = _get_scan_axis(self.posx, self.posy, self.posz)
        if scan_axis is not None:
            self._scan_axis = scan_axis
        else:
            raise MeasurementDataError(
                'Invalid position arguments for LineScan.')

    def _valid_scan(self, scan):
        scan_axis = _get_scan_axis(scan.posx, scan.posy, scan.posz)
        if scan_axis is not None and scan_axis == self._scan_axis:
            if (scan_axis == 'x' and round(scan.posy, 4) == self.posy and
               round(scan.posz, 4) == self.posz):
                return True
            elif (scan_axis == 'y' and round(scan.posx, 4) == self.posx and
                  round(scan.posz, 4) == self.posz):
                return True
            elif (scan_axis == 'z' and round(scan.posx, 4) == self.posx and
                  round(scan.posy, 4) == self.posy):
                return True
            else:
                return False
        else:
            return False

    def _get_shifts(self):
        if self._scan_axis == 'z':
            shiftx = self.calibration.shift_x_to_y
            shifty = 0
            shiftz = self.calibration.shift_z_to_y
        return (shiftx, shifty, shiftz)

    def _raw_data_unit(self):
        if all([raw.unit == self._raw[0].unit for raw in self._raw]):
            return self._raw[0].unit
        else:
            return ''

    def _data_interpolation(self):
        """Interpolate each scan."""
        sx, sy, sz = self._get_shifts()

        # correct curves displacement due to trigger and
        # integration time (half integration time)
        self._interpolated = []
        scan_pos = self.scan_positions

        idx = 1
        for raw in self._raw:
            interp = DataSet()
            interp.description = 'Interpolated_Voltage'
            interp.unit = raw.unit

            interp.posx = self.posx
            interp.posy = self.posy
            interp.posz = self.posz

            rawpos = _get_scan_position(raw.posx, raw.posy, raw.posz)

            fx = _interpolate.splrep(rawpos + sx, raw.datax, s=0, k=1)
            interp.datax = _interpolate.splev(scan_pos, fx, der=0)

            fy = _interpolate.splrep(rawpos + sy, raw.datay, s=0, k=1)
            interp.datay = _interpolate.splev(scan_pos, fy, der=0)

            fz = _interpolate.splrep(rawpos + sz, raw.dataz, s=0, k=1)
            interp.dataz = _interpolate.splev(scan_pos, fz, der=0)

            self._interpolated.append(interp)

            self._save_data(raw, idx=idx)
            self._save_data(interp, idx=idx)
            idx += 1

    def _calculate_voltage_avg_std(self):
        """Calculate the average and std of voltage values."""
        n = self.nr_scans

        interpolation_npts = len(self.scan_positions)

        # average calculation
        self._avg_voltage.description = 'Avg_Voltage'
        self._avg_voltage.unit = self._raw_data_unit()
        self._avg_voltage.posx = self.posx
        self._avg_voltage.posy = self.posy
        self._avg_voltage.posz = self.posz
        self._avg_voltage.datax = _np.zeros(interpolation_npts)
        self._avg_voltage.datay = _np.zeros(interpolation_npts)
        self._avg_voltage.dataz = _np.zeros(interpolation_npts)

        if n > 1:
            for i in range(n):
                self._avg_voltage.datax += self._interpolated[i].datax
                self._avg_voltage.datay += self._interpolated[i].datay
                self._avg_voltage.dataz += self._interpolated[i].dataz

            self._avg_voltage.datax /= n
            self._avg_voltage.datay /= n
            self._avg_voltage.dataz /= n

        elif n == 1:
            self._avg_voltage.datax = self._interpolated[0].datax
            self._avg_voltage.datay = self._interpolated[0].datay
            self._avg_voltage.dataz = self._interpolated[0].dataz

        # standard std calculation
        self._std_voltage.description = 'Std_Voltage'
        self._std_voltage.unit = self._raw_data_unit()
        self._std_voltage.posx = self.posx
        self._std_voltage.posy = self.posy
        self._std_voltage.posz = self.posz
        self._std_voltage.datax = _np.zeros(interpolation_npts)
        self._std_voltage.datay = _np.zeros(interpolation_npts)
        self._std_voltage.dataz = _np.zeros(interpolation_npts)

        if n > 1:
            for i in range(n):
                self._std_voltage.datax += pow((
                    self._interpolated[i].datax -
                    self._avg_voltage.datax), 2)

                self._std_voltage.datay += pow((
                    self._interpolated[i].datay -
                    self._avg_voltage.datay), 2)

                self._std_voltage.dataz += pow((
                    self._interpolated[i].dataz -
                    self._avg_voltage.dataz), 2)

            self._std_voltage.datax /= n
            self._std_voltage.datay /= n
            self._std_voltage.dataz /= n

        self._save_data(self._avg_voltage, self._std_voltage)

    def _convert_voltage_to_field(self):
        """Calculate the average and std of magnetic field values."""
        self._avg_field.description = 'Avg_Field'
        self._avg_field.unit = 'T'
        self._avg_field.posx = self.posx
        self._avg_field.posy = self.posy
        self._avg_field.posz = self.posz

        self._avg_field.datax = self.calibration.convert_probe_x(
            self._avg_voltage.datax)

        self._avg_field.datay = self.calibration.convert_probe_y(
            self._avg_voltage.datay)

        self._avg_field.dataz = self.calibration.convert_probe_z(
            self._avg_voltage.dataz)

        self._std_field.description = 'Std_Field'
        self._std_field.unit = 'T'
        self._std_field.posx = self.posx
        self._std_field.posy = self.posy
        self._std_field.posz = self.posz

        self._std_field.datax = self.calibration.convert_probe_x(
            self._std_voltage.datax)

        self._std_field.datay = self.calibration.convert_probe_y(
            self._std_voltage.datay)

        self._std_field.dataz = self.calibration.convert_probe_z(
            self._std_voltage.dataz)

        self._save_data(self._avg_field, self._std_field)

    def _calculate_field_first_integral(self):
        """Calculate the magnetic field first integral."""
        self._first_integral.description = 'First_Integral'
        self._first_integral.unit = 'T.mm'
        self._first_integral.posx = self.posx
        self._first_integral.posy = self.posy
        self._first_integral.posz = self.posz

        field_pos = _get_scan_position(self._avg_field.posx,
                                       self._avg_field.posy,
                                       self._avg_field.posz)

        self._first_integral.datax = _cumtrapz(
            x=field_pos, y=self._avg_field.datax, initial=0)

        self._first_integral.datay = _cumtrapz(
            x=field_pos, y=self._avg_field.datay, initial=0)

        self._first_integral.dataz = _cumtrapz(
            x=field_pos, y=self._avg_field.dataz, initial=0)

        self._save_data(self._first_integral)

    def _calculate_field_second_integral(self):
        """Calculate the magnetic field second integral."""
        self._second_integral.description = 'Second_Integral'
        self._second_integral.unit = 'T.mm^2'
        self._second_integral.posx = self.posx
        self._second_integral.posy = self.posy
        self._second_integral.posz = self.posz

        field_pos = _get_scan_position(self._avg_field.posx,
                                       self._avg_field.posy,
                                       self._avg_field.posz)

        self._second_integral.datax = _cumtrapz(
            x=field_pos, y=self._first_integral.datax, initial=0)

        self._second_integral.datay = _cumtrapz(
            x=field_pos, y=self._first_integral.datay, initial=0)

        self._second_integral.dataz = _cumtrapz(
            x=field_pos, y=self._first_integral.dataz, initial=0)

        self._save_data(self._second_integral)

    def _get_filename(self, dataset1, dataset2=None, idx=None):
        if self._scan_axis == 'x':
            linepos = ('Z=' + '{0:0.3f}'.format(self.posz) + 'mm_' +
                       'Y=' + '{0:0.3f}'.format(self.posy) + 'mm')
        elif self._scan_axis == 'y':
            linepos = ('Z=' + '{0:0.3f}'.format(self.posz) + 'mm_' +
                       'X=' + '{0:0.3f}'.format(self.posx) + 'mm')
        elif self._scan_axis == 'z':
            linepos = ('Y=' + '{0:0.3f}'.format(self.posy) + 'mm_' +
                       'X=' + '{0:0.3f}'.format(self.posx) + 'mm')

        if dataset2 is not None:
            description = dataset1.description + '_' + dataset2.description
        else:
            description = dataset1.description

        if idx is not None:
            filename = description + '_' + linepos + '_' + str(idx) + '.dat'
        else:
            filename = description + '_' + linepos + '.dat'

        datadir = _os.path.join(self.dirpath, 'data')
        if not _os.path.isdir(datadir):
            _os.mkdir(datadir)

        filename = _os.path.join(datadir, filename)

        return filename

    def _save_data(self, dataset1, dataset2=None, idx=None):
        filename = self._get_filename(dataset1, dataset2, idx)

        if self._scan_axis == 'x':
            pos_str = 'X [mm]'
            pos_values = dataset1.posx
        elif self._scan_axis == 'y':
            pos_str = 'Y [mm]'
            pos_values = dataset1.posy
        elif self._scan_axis == 'z':
            pos_str = 'Z [mm]'
            pos_values = dataset1.posz

        if dataset2 is not None:
            description = dataset1.description + '_' + dataset2.description
            columns_names = (
                '%s\t' % pos_str +
                '%sX [%s]\t' % (dataset1.description, dataset1.unit) +
                '%sY [%s]\t' % (dataset1.description, dataset1.unit) +
                '%sZ [%s]\t' % (dataset1.description, dataset1.unit) +
                '%sX [%s]\t' % (dataset2.description, dataset2.unit) +
                '%sY [%s]\t' % (dataset2.description, dataset2.unit) +
                '%sZ [%s]' % (dataset2.description, dataset2.unit))
            columns = _np.column_stack((
                pos_values, dataset1.datax, dataset1.datay, dataset1.dataz,
                dataset2.datax, dataset2.datay, dataset2.dataz))
        else:
            description = dataset1.description
            columns_names = (
                '%s\t' % pos_str +
                '%sX [%s]\t' % (dataset1.description, dataset1.unit) +
                '%sY [%s]\t' % (dataset1.description, dataset1.unit) +
                '%sZ [%s]' % (dataset1.description, dataset1.unit))
            columns = _np.column_stack((
                pos_values, dataset1.datax, dataset1.datay, dataset1.dataz))

        f = open(filename, mode='w')

        cconfig_filename = _os.path.split(self.cconfig.filename)[1]
        mconfig_filename = _os.path.split(self.mconfig.filename)[1]
        calibration_filename = _os.path.split(self.calibration.filename)[1]

        f.write('data:                     \t%s\n' % description)
        f.write('timestamp:                \t%s\n' % self._timestamp)
        f.write('control_configuration:    \t%s\n' % cconfig_filename)
        f.write('measurement_configuration:\t%s\n' % mconfig_filename)
        f.write('calibration:              \t%s\n' % calibration_filename)

        if self._scan_axis == 'x':
            f.write('position X [mm]:          \t--\n')
            f.write('position Y [mm]:          \t%f\n' % dataset1.posy)
            f.write('position Z [mm]:          \t%f\n' % dataset1.posz)
        elif self._scan_axis == 'y':
            f.write('position X [mm]:          \t%f\n' % dataset1.posx)
            f.write('position Y [mm]:          \t--\n')
            f.write('position Z [mm]:          \t%f\n' % dataset1.posz)
        elif self._scan_axis == 'z':
            f.write('position X [mm]:          \t%f\n' % dataset1.posx)
            f.write('position Y [mm]:          \t%f\n' % dataset1.posy)
            f.write('position Z [mm]:          \t--\n')

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


class Measurement(object):
    """Measurement data."""

    def __init__(self, cconfig, mconfig, calibration, dirpath):
        """Initialize variables.

        Args:
            cconfig (ControlConfiguration): control configuration data.
            mconfig (MeasurementConfiguration): measurement configuration data.
            calibration (CalibrationData): probe calibration data.
            dirpath (str): directory path to save files.
        """
        self.cconfig = cconfig
        self.mconfig = mconfig
        self.calibration = calibration
        self.dirpath = dirpath
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

    @property
    def data_list(self):
        """Measurement data list."""
        datalist = []
        for d in self._data.values():
            for v in d.values():
                datalist.append(v)
        return datalist

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

    def _get_positions(self):
        data_list = self.data_list

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

    def _get_avg_field_at_point(self, posx, posy, posz):
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

            bx = ls.avg_field.datax[idx]
            by = ls.avg_field.datay[idx]
            bz = ls.avg_field.dataz[idx]
        except Exception:
            bx, by, bz = _np.nan, _np.nan, _np.nan

        return (bx, by, bz)

    def _get_avg_field_data(self):
        posx_vec, posy_vec, posz_vec = self._get_positions()

        size = len(posx_vec)*len(posy_vec)*len(posz_vec)
        field_data = _np.zeros([size, 6])

        count = 0
        for posz in posz_vec:
            for posy in posy_vec:
                for posx in posx_vec:
                    bx, by, bz = self._get_avg_field_at_point(posx, posy, posz)
                    field_data[count, 0] = posx
                    field_data[count, 1] = posy
                    field_data[count, 2] = posz
                    field_data[count, 3] = bx
                    field_data[count, 4] = by
                    field_data[count, 5] = bz
                    count += 1

        return field_data

    def save_measurement(self, name):
        """Save measurement data."""
        t = _time.localtime()
        date = _time.strftime('%Y-%m-%d', t)
        datetime = _time.strftime('%Y-%m-%d_%H-%M-%S', t)

        field_data = self._get_avg_field_data()

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
