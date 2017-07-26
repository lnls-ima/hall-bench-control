# -*- coding: utf-8 -*-
"""Implementation of classes to store and analyse measurement data."""

import os as _os
import time as _time
import math as _math
import numpy as _np
from scipy import interpolate as _interpolate
from scipy.integrate import cumtrapz as _cumtrapz


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
        self.posx = posx
        self.posy = posy
        self.posz = posz
        self.cconfig = cconfig
        self.mconfig = mconfig
        self.calibration = calibration
        self.dirpath = dirpath

        self._axis = None
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
        lsc._axis = linescan._axis
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
    def axis(self):
        """Scan axis."""
        return self._axis

    @property
    def timestamp(self):
        """Scan timestamp."""
        return self._timestamp

    @property
    def nr_scans(self):
        """Number of scans."""
        return len(self._raw)

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
            raise Exception('Invalid scan.')

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

    def _set_scan_axis(self):
        axis = _get_scan_axis(self.posx, self.posy, self.posz)
        if axis is not None:
            self._axis = axis
        else:
            raise Exception('Invalid position arguments for LineScan.')

    def _valid_scan(self, scan):
        axis = _get_scan_axis(scan.posx, scan.posy, scan.posz)
        if axis is not None and axis == self._axis:
            if (axis == 'x' and scan.posy == self.posy and
               scan.posz == self.posz):
                return True
            elif (axis == 'y' and scan.posx == self.posx and
                  scan.posz == self.posz):
                return True
            elif (axis == 'z' and scan.posx == self.posx and
                  scan.posy == self.posy):
                return True
            else:
                return False
        else:
            return False

    def _get_shifts(self):
        if self._axis == 'z':
            shiftx = self.calibration.shift_x_to_y
            shifty = 0
            shiftz = self.calibration.shift_z_to_y
        return (shiftx, shifty, shiftz)

    def _get_number_cuts(self):
        if self._axis == 'z':
            n_cuts = _math.ceil(_np.array(
                [abs(self.calibration.shift_x_to_y),
                 abs(self.calibration.shift_z_to_y)]).max())
        return n_cuts

    def _data_interpolation(self):
        """Interpolate each scan."""
        sx, sy, sz = self._get_shifts()

        # correct curves displacement due to trigger and
        # integration time (half integration time)
        self._interpolated = []
        interp_pos = _get_scan_position(self.posx, self.posy, self.posz)

        idx = 1
        for raw in self._raw:
            interp = DataSet()
            interp.description = 'Interpolated_Voltage'
            interp.unit = 'V'

            interp.posx = self.posx
            interp.posy = self.posy
            interp.posz = self.posz

            rawpos = _get_scan_position(raw.posx, raw.posy, raw.posz)

            fx = _interpolate.splrep(rawpos + sx, raw.datax, s=0, k=1)
            interp.datax = _interpolate.splev(interp_pos, fx, der=0)

            fy = _interpolate.splrep(rawpos + sy, raw.datay, s=0, k=1)
            interp.datay = _interpolate.splev(interp_pos, fy, der=0)

            fz = _interpolate.splrep(rawpos + sz, raw.dataz, s=0, k=1)
            interp.dataz = _interpolate.splev(interp_pos, fz, der=0)

            self._interpolated.append(interp)

            self._save_data(raw, idx=idx)
            self._save_data(interp, idx=idx)
            idx += 1

    def _calculate_voltage_avg_std(self):
        """Calculate the average and std of voltage values."""
        n = self.nr_scans

        interpolation_npts = len(
            _get_scan_position(self.posx, self.posy, self.posz))

        # average calculation
        self._avg_voltage.description = 'Avg_Voltage'
        self._avg_voltage.unit = 'V'
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
        self._std_voltage.unit = 'V'
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

        # cut extra points due to shift sensors
        nc = self._get_number_cuts()

        if nc != 0:
            if self._axis == 'x':
                self._avg_voltage.posx = self._avg_voltage.posx[nc:-nc]
                self._std_voltage.posx = self._std_voltage.posx[nc:-nc]
            elif self._axis == 'y':
                self._avg_voltage.posy = self._avg_voltage.posy[nc:-nc]
                self._std_voltage.posy = self._std_voltage.posy[nc:-nc]
            elif self._axis == 'z':
                self._avg_voltage.posz = self._avg_voltage.posz[nc:-nc]
                self._std_voltage.posz = self._std_voltage.posz[nc:-nc]

            self._avg_voltage.datax = self._avg_voltage.datax[nc:-nc]
            self._avg_voltage.datay = self._avg_voltage.datay[nc:-nc]
            self._avg_voltage.dataz = self._avg_voltage.dataz[nc:-nc]

            self._std_voltage.datax = self._std_voltage.datax[nc:-nc]
            self._std_voltage.datay = self._std_voltage.datay[nc:-nc]
            self._std_voltage.dataz = self._std_voltage.dataz[nc:-nc]

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
        self._first_integral.unit = 'T.m'
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
        self._second_integral.unit = 'T.m^2'
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
        if self._axis == 'x':
            linepos = 'Z=' + str(self.posz) + 'mm_Y=' + str(self.posy) + 'mm'
        elif self._axis == 'y':
            linepos = 'Z=' + str(self.posz) + 'mm_X=' + str(self.posx) + 'mm'
        elif self._axis == 'z':
            linepos = 'Y=' + str(self.posy) + 'mm_X=' + str(self.posx) + 'mm'

        if dataset2 is not None:
            description = dataset1.description + '_' + dataset2.description
        else:
            description = dataset1.description

        if idx is not None:
            filename = description + '_' + linepos + '_' + str(idx) + '.dat'
        else:
            filename = description + '_' + linepos + '.dat'

        filename = _os.path.join(self.dirpath, filename)

        return filename

    def _save_data(self, dataset1, dataset2=None, idx=None):
        filename = self._get_filename(dataset1, dataset2, idx)

        if self._axis == 'x':
            pos_str = 'X [mm]'
            pos_values = dataset1.posx
        elif self._axis == 'y':
            pos_str = 'Y [mm]'
            pos_values = dataset1.posy
        elif self._axis == 'z':
            pos_str = 'Z [mm]'
            pos_values = dataset1.posz

        if dataset2 is not None:
            description = dataset1.description + '_' + dataset2.description
            columns_names = (
                '%s\t' % pos_str +
                '%s X [%s]\t' % (dataset1.description, dataset1.unit) +
                '%s Y [%s]\t' % (dataset1.description, dataset1.unit) +
                '%s Z [%s]\t' % (dataset1.description, dataset1.unit) +
                '%s X [%s]\t' % (dataset2.description, dataset2.unit) +
                '%s Y [%s]\t' % (dataset2.description, dataset2.unit) +
                '%s Z [%s]' % (dataset2.description, dataset2.unit))
            columns = _np.column_stack((
                pos_values, dataset1.datax, dataset1.datay, dataset1.dataz,
                dataset2.datax, dataset2.datay, dataset2.dataz))
        else:
            description = dataset1.description
            columns_names = (
                '%s\t' % pos_str +
                '%s X [%s]\t' % (dataset1.description, dataset1.unit) +
                '%s Y [%s]\t' % (dataset1.description, dataset1.unit) +
                '%s Z [%s]' % (dataset1.description, dataset1.unit))
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

        if self._axis == 'x':
            f.write('position X [mm]:          \t--\n')
            f.write('position Y [mm]:          \t%f\n' % dataset1.posy)
            f.write('position Z [mm]:          \t%f\n' % dataset1.posz)
        elif self._axis == 'y':
            f.write('position X [mm]:          \t%f\n' % dataset1.posx)
            f.write('position Y [mm]:          \t--\n')
            f.write('position Z [mm]:          \t%f\n' % dataset1.posz)
        elif self._axis == 'z':
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
