"""Measurement test."""

import os
import shutil
import unittest
import filecmp
import numpy as np
from hall_bench.data_handle import measurement
from hall_bench.data_handle import configuration
from hall_bench.data_handle import calibration


class TestFunctions(unittest.TestCase):
    """Test functions."""

    def setUp(self):
        """Set up."""
        pass

    def tearDown(self):
        """Tear down."""
        if os.path.isdir('testdir'):
            shutil.rmtree('testdir')

    def test_get_scan_axis(self):
        scan_axis = measurement._get_scan_axis(None, None, None)
        self.assertIsNone(scan_axis)

        scan_axis = measurement._get_scan_axis([], [], [])
        self.assertIsNone(scan_axis)

        scan_axis = measurement._get_scan_axis(1, 2, 3)
        self.assertIsNone(scan_axis)

        scan_axis = measurement._get_scan_axis(1, 2, [3])
        self.assertIsNone(scan_axis)

        scan_axis = measurement._get_scan_axis([1], [2], [3])
        self.assertIsNone(scan_axis)

        scan_axis = measurement._get_scan_axis(1, 2, [3, 4])
        self.assertEqual(scan_axis, 'z')

        scan_axis = measurement._get_scan_axis([1], [2], [3, 4])
        self.assertEqual(scan_axis, 'z')

        scan_axis = measurement._get_scan_axis(np.array([1]), [2], (3, 4))
        self.assertEqual(scan_axis, 'z')

        scan_axis = measurement._get_scan_axis(1, [2, 3], 4)
        self.assertEqual(scan_axis, 'y')

        scan_axis = measurement._get_scan_axis([1, 2], 3, 4)
        self.assertEqual(scan_axis, 'x')

    def test_get_scan_positions(self):
        scan_positions = measurement._get_scan_positions(None, None, None)
        self.assertEqual(scan_positions.size, 0)

        scan_positions = measurement._get_scan_positions([], [], [])
        self.assertEqual(scan_positions.size, 0)

        scan_positions = measurement._get_scan_positions(1, 2, 3)
        self.assertEqual(scan_positions.size, 0)

        scan_positions = measurement._get_scan_positions(1, 2, [3])
        self.assertEqual(scan_positions.size, 0)

        scan_positions = measurement._get_scan_positions([1], [2], [3])
        self.assertEqual(scan_positions.size, 0)

        scan_positions = measurement._get_scan_positions(1, 2, [3, 4])
        np.testing.assert_array_equal(scan_positions, [3, 4])

        scan_positions = measurement._get_scan_positions([1], [2], [3, 4])
        np.testing.assert_array_equal(scan_positions, [3, 4])

        scan_positions = measurement._get_scan_positions(1, [2, 3], 4)
        np.testing.assert_array_equal(scan_positions, [2, 3])

        scan_positions = measurement._get_scan_positions([1, 2], 3, 4)
        np.testing.assert_array_equal(scan_positions, [1, 2])

    def test_read_and_write_scan_file(self):
        filename = 'scan_file.dat'
        dataset, timestamp, cconfig, mconfig, calibration = (
            measurement._read_scan_file(filename))

        field_avg = [
            0.0000000,
            0.4000000,
            0.8000000,
            1.2000000,
            1.6000000,
            1.9644000,
            2.3862752,
            2.8555936,
            3.4560144,
            4.2711968,
            5.3848000,
        ]

        self.assertEqual(timestamp, '2017-08-08_12-26-53')
        self.assertEqual(cconfig, 'control_configuration.txt')
        self.assertEqual(mconfig, 'measurement_configuration.txt')
        self.assertEqual(calibration, 'calibration.txt')
        self.assertEqual(dataset.description, 'field_avg')
        self.assertEqual(dataset.unit, 'T')
        self.assertEqual(dataset.posx, 1)
        self.assertEqual(dataset.posy, 2)
        np.testing.assert_array_almost_equal(
            dataset.posz, np.linspace(0, 20, 11))
        np.testing.assert_array_almost_equal(
            dataset.datax, field_avg)
        np.testing.assert_array_almost_equal(
            dataset.datay, np.ones(11)*np.nan)
        np.testing.assert_array_almost_equal(
            dataset.dataz, np.ones(11)*np.nan)

        new_filename = 'test_write_scan_file.dat'
        measurement._write_scan_file(
            new_filename, dataset, timestamp, cconfig, mconfig, calibration)
        self.assertTrue(filecmp.cmp(new_filename, filename))
        os.remove(new_filename)

    def test_get_scan_files_list(self):
        testdir = 'testdir'
        os.mkdir(testdir)
        os.mknod(os.path.join(testdir, 'Y=1.000mm_Z=5.000mmm_field_avg.dat'))
        os.mknod(os.path.join(testdir, 'Z=1.000mm_Y=5.000mmm_field_avg.dat'))
        os.mknod(os.path.join(testdir, 'Z=1.000mm_Y=5.000mmm_field_std.dat'))
        os.mknod(os.path.join(testdir, 'Z=1.000mm_Y=5.000mmm_field_avg.txt'))

        with self.assertRaises(measurement.MeasurementDataError):
            files = measurement._get_scan_files_list(testdir)

        with self.assertRaises(measurement.MeasurementDataError):
            files = measurement._get_scan_files_list(testdir, posx=1, posy=1)

        files = measurement._get_scan_files_list(testdir, posy=5, posz=1)
        self.assertEqual(len(files), 2)
        self.assertTrue(
            os.path.join(testdir, 'Z=1.000mm_Y=5.000mmm_field_avg.dat')
            in files)
        self.assertTrue(
            os.path.join(testdir, 'Z=1.000mm_Y=5.000mmm_field_std.dat')
            in files)
        self.assertFalse(
            os.path.join(testdir, 'Z=1.000mm_Y=5.000mmm_field_avg.txt')
            in files)

        files = measurement._get_scan_files_list(testdir, posy=1, posz=5)
        self.assertTrue(
            os.path.join(testdir, 'Y=1.000mm_Z=5.000mmm_field_avg.dat')
            in files)


class TestDataSet(unittest.TestCase):
    """Test DataSet."""

    def setUp(self):
        """Set up."""
        pass

    def tearDown(self):
        """Tear down."""
        pass

    def test_initialization(self):
        description = 'description'
        unit = 'unit'
        ds = measurement.DataSet(description, unit)
        self.assertEqual(ds.description, description)
        self.assertEqual(ds.unit, unit)
        self.assertEqual(ds.posx.size, 0)
        self.assertEqual(ds.posy.size, 0)
        self.assertEqual(ds.posz.size, 0)
        self.assertEqual(ds.datax.size, 0)
        self.assertEqual(ds.datay.size, 0)
        self.assertEqual(ds.dataz.size, 0)

    def test_reverse(self):
        ds = measurement.DataSet()
        vec = np.array([1, 2, 3, 4, 5])
        ds.datax = vec
        ds.reverse()
        np.testing.assert_array_equal(ds.datax, vec[::-1])

    def test_copy(self):
        ds = measurement.DataSet()
        vec = [1, 2, 3, 4, 5]
        ds.datax = vec
        ds2 = measurement.DataSet.copy(ds)
        ds.datax = np.array([])
        np.testing.assert_array_equal(ds2.datax, vec)

    def test_clear(self):
        description = 'description'
        unit = 'unit'
        ds = measurement.DataSet(description, unit)
        ds.posx = 1
        ds.posy = 2
        ds.posz = 3
        ds.datax = 4
        ds.datay = 5
        ds.dataz = 6
        ds.clear()
        self.assertEqual(ds.description, description)
        self.assertEqual(ds.unit, unit)
        self.assertEqual(ds.posx.size, 0)
        self.assertEqual(ds.posy.size, 0)
        self.assertEqual(ds.posz.size, 0)
        self.assertEqual(ds.datax.size, 0)
        self.assertEqual(ds.datay.size, 0)
        self.assertEqual(ds.dataz.size, 0)


class TestLineScan(unittest.TestCase):
    """Test LineScan."""

    def setUp(self):
        """Set up."""
        scriptpath = os.path.realpath(__file__)
        parentdir = os.path.split(scriptpath)[0]
        self.dirpath = os.path.join(parentdir, 'measurement_data')
        self.cconfig_file = 'control_configuration_file.txt'
        self.cconfig = configuration.ControlConfiguration(self.cconfig_file)
        self.mconfig_file = 'measurement_configuration_file.txt'
        self.mconfig = configuration.MeasurementConfiguration(
            self.mconfig_file)
        self.calibration = calibration.CalibrationData()

        self.pos = np.linspace(0, 20, 11)
        self.voltage = np.linspace(0, 20, 11)

        self.field_avg = [
            0.0,
            0.4,
            0.8,
            1.2000000,
            1.6,
            1.9644000,
            2.3862752,
            2.8555936,
            3.4560144,
            4.2711968,
            5.3848000,
        ]

        self.field_first_integral = [
            0,
            0.0004,
            0.0016,
            0.0036,
            0.0064,
            0.0099644,
            0.01431508,
            0.01955694,
            0.02586855,
            0.03359576,
            0.04325176,
        ]

        self.field_second_integral = [
            0.00000000e+00,
            4.00000000e-07,
            2.40000000e-06,
            7.60000000e-06,
            1.76000000e-05,
            3.39644000e-05,
            5.82438752e-05,
            9.21158944e-05,
            1.37541390e-04,
            1.97005706e-04,
            2.73853229e-04,
        ]

    def tearDown(self):
        """Tear down."""
        if os.path.isdir(self.dirpath):
            shutil.rmtree(self.dirpath)

    def test_initialization_none_args(self):
        with self.assertRaises(measurement.MeasurementDataError):
            ls = measurement.LineScan(
                None, None, None, None, None, None, None)

    def test_initialization_scan_axis_x(self):
        posx = [3, 4]
        posy = 1
        posz = 2

        ls = measurement.LineScan(
            posx, posy, posz, None, None, None, self.dirpath)

        self.assertIsNone(ls.control_configuration)
        self.assertIsNone(ls.measurement_configuration)
        self.assertIsNone(ls.calibration)
        self.assertEqual(ls.dirpath, self.dirpath)
        self.assertEqual(ls.posy, posy)
        self.assertEqual(ls.posz, posz)
        self.assertEqual(ls.scan_axis, 'x')
        np.testing.assert_array_equal(ls.posx, posx)
        np.testing.assert_array_equal(ls.scan_positions, posx)

    def test_initialization_scan_axis_y(self):
        posx = 1
        posy = [3, 4]
        posz = 2

        ls = measurement.LineScan(
            posx, posy, posz, None, None, None, self.dirpath)

        self.assertIsNone(ls.control_configuration)
        self.assertIsNone(ls.measurement_configuration)
        self.assertIsNone(ls.calibration)
        self.assertEqual(ls.dirpath, self.dirpath)
        self.assertEqual(ls.posx, posx)
        self.assertEqual(ls.posz, posz)
        self.assertEqual(ls.scan_axis, 'y')
        np.testing.assert_array_equal(ls.posy, posy)
        np.testing.assert_array_equal(ls.scan_positions, posy)

    def test_initialization_scan_axis_z(self):
        posx = 1
        posy = 2
        posz = [3, 4]

        ls = measurement.LineScan(
            posx, posy, posz, None, None, None, self.dirpath)

        self.assertIsNone(ls.control_configuration)
        self.assertIsNone(ls.measurement_configuration)
        self.assertIsNone(ls.calibration)
        self.assertEqual(ls.dirpath, self.dirpath)
        self.assertEqual(ls.posx, posx)
        self.assertEqual(ls.posy, posy)
        self.assertEqual(ls.scan_axis, 'z')
        np.testing.assert_array_equal(ls.posz, posz)
        np.testing.assert_array_equal(ls.scan_positions, posz)

    def test_initialization_invalid_configuration_path(self):
        posx = 1
        posy = 2
        posz = [3, 4]
        invalidpath = 'invalidpath'

        ls = measurement.LineScan(
            posx, posy, posz, invalidpath, None, None, self.dirpath)
        self.assertIsNone(ls.control_configuration)

        ls = measurement.LineScan(
            posx, posy, posz, None, invalidpath, None, self.dirpath)
        self.assertIsNone(ls.measurement_configuration)

        ls = measurement.LineScan(
            posx, posy, posz, invalidpath, None, None, self.dirpath)
        self.assertIsNone(ls.calibration)

    def test_initialization_valid_configuration_path(self):
        posx = 1
        posy = 2
        posz = [3, 4]

        ls = measurement.LineScan(
            posx, posy, posz, self.cconfig_file, None, None, self.dirpath)
        self.assertEqual(
            ls.control_configuration.filename, self.cconfig_file)

        ls = measurement.LineScan(
            posx, posy, posz, None, self.mconfig_file, None, self.dirpath)
        self.assertEqual(
            ls.measurement_configuration.filename, self.mconfig_file)

    def test_initialization_configuration_objects(self):
        posx = 1
        posy = 2
        posz = [3, 4]

        ls = measurement.LineScan(
            posx, posy, posz, self.cconfig, None, None, self.dirpath)
        self.assertEqual(
            ls.control_configuration.filename, self.cconfig_file)

        ls = measurement.LineScan(
            posx, posy, posz, None, self.mconfig, None, self.dirpath)
        self.assertEqual(
            ls.measurement_configuration.filename, self.mconfig_file)

    def test_add_invalid_scan(self):
        posx = 1
        posy = 2
        posz = np.linspace(0, 10, 11)
        ls = measurement.LineScan(posx, posy, posz, self.cconfig, self.mconfig,
                                  self.calibration, self.dirpath)

        scan = measurement.DataSet()
        with self.assertRaises(measurement.MeasurementDataError):
            ls.add_scan(scan)

        scan.datax = np.zeros(11)
        scan.datay = np.zeros(11)
        scan.dataz = np.zeros(11)

        scan.posx = 0
        scan.posy = 0
        scan.posz = 0
        with self.assertRaises(measurement.MeasurementDataError):
            ls.add_scan(scan)

        scan.posx = np.linspace(0, 10, 11)
        scan.posy = 0
        scan.posz = 0
        with self.assertRaises(measurement.MeasurementDataError):
            ls.add_scan(scan)

        scan.posx = 0
        scan.posy = np.linspace(0, 10, 11)
        scan.posz = 0
        with self.assertRaises(measurement.MeasurementDataError):
            ls.add_scan(scan)

        scan.posx = 0
        scan.posy = 0
        scan.posz = np.linspace(0, 10, 11)
        with self.assertRaises(measurement.MeasurementDataError):
            ls.add_scan(scan)

        scan.posx = 1
        scan.posy = 2
        scan.posz = np.linspace(0, 10, 11)
        scan.datax = np.array([])
        scan.datay = np.array([])
        scan.dataz = np.array([])
        with self.assertRaises(measurement.MeasurementDataError):
            ls.add_scan(scan)

        scan.posx = 1
        scan.posy = 2
        scan.posz = np.linspace(0, 10, 11)
        scan.datax = np.linspace(0, 10, 5)
        scan.datay = np.array([])
        scan.dataz = np.array([])
        with self.assertRaises(measurement.MeasurementDataError):
            ls.add_scan(scan)

        scan.posx = 1
        scan.posy = 2
        scan.posz = np.linspace(0, 10, 11)
        scan.datax = np.array([])
        scan.datay = np.linspace(0, 10, 5)
        scan.dataz = np.array([])
        with self.assertRaises(measurement.MeasurementDataError):
            ls.add_scan(scan)

        scan.posx = 1
        scan.posy = 2
        scan.posz = np.linspace(0, 10, 11)
        scan.datax = np.array([])
        scan.datay = np.array([])
        scan.dataz = np.linspace(0, 10, 5)
        with self.assertRaises(measurement.MeasurementDataError):
            ls.add_scan(scan)

        scan.posx = 1
        scan.posy = 2
        scan.posz = np.linspace(0, 10, 11)
        scan.datax = np.linspace(0, 10, 5)
        scan.datay = np.linspace(0, 10, 5)
        scan.dataz = np.linspace(0, 10, 5)
        with self.assertRaises(measurement.MeasurementDataError):
            ls.add_scan(scan)

    def test_add_valid_scan(self):
        posx = 1
        posy = 2
        posz = np.linspace(0, 10, 11)
        ls = measurement.LineScan(posx, posy, posz, self.cconfig, self.mconfig,
                                  self.calibration, self.dirpath)

        scan = measurement.DataSet()
        scan.posx = posx
        scan.posy = posy
        scan.posz = np.linspace(0, 10, 11)

        scan.datax = np.linspace(0, 10, 11)
        scan.datay = np.array([])
        scan.dataz = np.array([])
        ls.add_scan(scan)

        scan.datax = np.array([])
        scan.datay = np.linspace(0, 10, 11)
        scan.dataz = np.array([])
        ls.add_scan(scan)

        scan.datax = np.array([])
        scan.datay = np.array([])
        scan.dataz = np.linspace(0, 10, 11)
        ls.add_scan(scan)

        scan.datax = np.linspace(0, 10, 11)
        scan.datay = np.linspace(0, 10, 11)
        scan.dataz = np.linspace(0, 10, 11)
        ls.add_scan(scan)

    def test_analyse_data_one_scan(self):
        posx = 1
        posy = 2
        posz = self.pos
        ls = measurement.LineScan(posx, posy, posz, self.cconfig, self.mconfig,
                                  self.calibration, self.dirpath)

        scan = measurement.DataSet()
        scan.posx = posx
        scan.posy = posy
        scan.posz = self.pos
        scan.datax = self.voltage
        ls.add_scan(scan)
        ls.analyse_data(save_data=False)

        np.testing.assert_array_equal(
            ls.voltage_interpolated[0].datax, self.voltage)
        np.testing.assert_array_equal(
            ls.voltage_avg.datax, self.voltage)
        np.testing.assert_array_equal(
            ls.voltage_std.datax, np.zeros(11))
        np.testing.assert_array_almost_equal(
            ls.field_avg.datax, self.field_avg)
        np.testing.assert_array_equal(
            ls.field_std.datax, np.zeros(11))
        np.testing.assert_array_almost_equal(
            ls.field_first_integral.datax,
            self.field_first_integral, decimal=8)
        np.testing.assert_array_almost_equal(
            ls.field_second_integral.datax,
            self.field_second_integral, decimal=10)

    def test_analyse_data_two_scans(self):
        vec1 = self.pos + 0.5
        vec2 = self.pos - 0.5
        posx = 1
        posy = 2
        posz = self.pos
        ls = measurement.LineScan(posx, posy, posz, self.cconfig, self.mconfig,
                                  self.calibration, self.dirpath)

        scan1 = measurement.DataSet()
        scan1.posx = posx
        scan1.posy = posy
        scan1.posz = vec1
        scan1.datax = vec1
        ls.add_scan(scan1)

        scan2 = measurement.DataSet()
        scan2.posx = posx
        scan2.posy = posy
        scan2.posz = vec2
        scan2.datax = vec2
        ls.add_scan(scan2)

        ls.analyse_data(save_data=False)

        np.testing.assert_array_equal(
            ls.voltage_interpolated[0].datax, self.voltage)
        np.testing.assert_array_equal(
            ls.voltage_interpolated[1].datax, self.voltage)
        np.testing.assert_array_equal(
            ls.voltage_avg.datax, self.voltage)
        np.testing.assert_array_equal(
            ls.voltage_std.datax, np.zeros(11))
        np.testing.assert_array_almost_equal(
            ls.field_avg.datax, self.field_avg)
        np.testing.assert_array_equal(
            ls.field_std.datax, np.zeros(11))
        np.testing.assert_array_almost_equal(
            ls.field_first_integral.datax,
            self.field_first_integral, decimal=8)
        np.testing.assert_array_almost_equal(
            ls.field_second_integral.datax,
            self.field_second_integral, decimal=10)

    def test_save_and_read_measurement_files(self):
        vec1 = self.pos + 0.5
        vec2 = self.pos - 0.5
        posx = 1
        posy = 2
        posz = self.pos
        ls1 = measurement.LineScan(
            posx, posy, posz, self.cconfig,
            self.mconfig, self.calibration, self.dirpath)

        scan1 = measurement.DataSet()
        scan1.posx = posx
        scan1.posy = posy
        scan1.posz = vec1
        scan1.datax = vec1

        scan2 = measurement.DataSet()
        scan2.posx = posx
        scan2.posy = posy
        scan2.posz = vec2
        scan2.datax = vec2

        ls1.add_scan(scan1)
        ls1.add_scan(scan2)
        ls1.analyse_data(save_data=True)

        ls2 = measurement.LineScan.read_from_files(
            self.dirpath, posx=posx, posy=posy)

        self.assertEqual(ls2.nr_scans, 2)
        np.testing.assert_array_equal(
            ls2.voltage_raw[0].datax, vec1)
        np.testing.assert_array_equal(
            ls2.voltage_raw[1].datax, vec2)
        np.testing.assert_array_equal(
            ls2.voltage_interpolated[0].datax, self.voltage)
        np.testing.assert_array_equal(
            ls2.voltage_interpolated[1].datax, self.voltage)
        np.testing.assert_array_equal(
            ls2.voltage_avg.datax, self.voltage)
        np.testing.assert_array_equal(
            ls2.voltage_std.datax, np.zeros(11))
        np.testing.assert_array_almost_equal(
            ls2.field_avg.datax, self.field_avg)
        np.testing.assert_array_equal(
            ls2.field_std.datax, np.zeros(11))
        np.testing.assert_array_almost_equal(
            ls2.field_first_integral.datax,
            self.field_first_integral, decimal=8)
        np.testing.assert_array_almost_equal(
            ls2.field_second_integral.datax,
            self.field_second_integral, decimal=10)

    def test_clear(self):
        vec1 = self.pos + 0.5
        vec2 = self.pos - 0.5
        posx = 1
        posy = 2
        posz = self.pos
        ls = measurement.LineScan(posx, posy, posz, self.cconfig,
                                  self.mconfig, self.calibration, self.dirpath)

        scan1 = measurement.DataSet()
        scan1.posx = posx
        scan1.posy = posy
        scan1.posz = vec1
        scan1.datax = vec1

        scan2 = measurement.DataSet()
        scan2.posx = posx
        scan2.posy = posy
        scan2.posz = vec2
        scan2.datax = vec2

        ls.add_scan(scan1)
        ls.add_scan(scan2)
        ls.clear()
        self.assertEqual(ls.timestamp, '')
        self.assertEqual(len(ls.voltage_raw), 0)
        self.assertEqual(len(ls.voltage_interpolated), 0)
        self.assertIsNone(ls.voltage_avg)
        self.assertIsNone(ls.voltage_std)
        self.assertIsNone(ls.field_avg)
        self.assertIsNone(ls.field_std)
        self.assertIsNone(ls.field_first_integral)
        self.assertIsNone(ls.field_second_integral)

    def test_copy(self):
        vec1 = self.pos + 0.5
        vec2 = self.pos - 0.5
        posx = 1
        posy = 2
        posz = self.pos
        ls1 = measurement.LineScan(
            posx, posy, posz, self.cconfig,
            self.mconfig, self.calibration, self.dirpath)

        scan1 = measurement.DataSet()
        scan1.posx = posx
        scan1.posy = posy
        scan1.posz = vec1
        scan1.datax = vec1

        scan2 = measurement.DataSet()
        scan2.posx = posx
        scan2.posy = posy
        scan2.posz = vec2
        scan2.datax = vec2

        ls1.add_scan(scan1)
        ls1.add_scan(scan2)

        ls2 = measurement.LineScan.copy(ls1)
        ls1.clear()

        self.assertEqual(ls1.nr_scans, 0)
        self.assertEqual(ls2.nr_scans, 2)
        np.testing.assert_array_equal(
            ls2.voltage_raw[0].datax, vec1)
        np.testing.assert_array_equal(
            ls2.voltage_raw[1].datax, vec2)

        ls2.analyse_data(save_data=False)
        ls3 = measurement.LineScan.copy(ls2)
        ls2.clear()
        self.assertEqual(ls2.timestamp, '')
        self.assertEqual(len(ls2.voltage_raw), 0)
        self.assertEqual(len(ls2.voltage_interpolated), 0)
        self.assertIsNone(ls2.voltage_avg)
        self.assertIsNone(ls2.voltage_std)
        self.assertIsNone(ls2.field_avg)
        self.assertIsNone(ls2.field_std)
        self.assertIsNone(ls2.field_first_integral)
        self.assertIsNone(ls2.field_second_integral)

        self.assertEqual(ls3.nr_scans, 2)
        np.testing.assert_array_equal(
            ls3.voltage_raw[0].datax, vec1)
        np.testing.assert_array_equal(
            ls3.voltage_raw[1].datax, vec2)
        np.testing.assert_array_equal(
            ls3.voltage_interpolated[0].datax, self.voltage)
        np.testing.assert_array_equal(
            ls3.voltage_interpolated[1].datax, self.voltage)
        np.testing.assert_array_equal(
            ls3.voltage_avg.datax, self.voltage)
        np.testing.assert_array_equal(
            ls3.voltage_std.datax, np.zeros(11))
        np.testing.assert_array_almost_equal(
            ls3.field_avg.datax, self.field_avg)
        np.testing.assert_array_equal(
            ls3.field_std.datax, np.zeros(11))
        np.testing.assert_array_almost_equal(
            ls3.field_first_integral.datax,
            self.field_first_integral, decimal=8)
        np.testing.assert_array_almost_equal(
            ls3.field_second_integral.datax,
            self.field_second_integral, decimal=10)


class TestMeasurement(unittest.TestCase):
    """Test Measurement."""

    def setUp(self):
        """Set up."""
        scriptpath = os.path.realpath(__file__)
        parentdir = os.path.split(scriptpath)[0]
        self.dirpath = os.path.join(parentdir, 'measurement_data')
        self.calibration = calibration.CalibrationData()

    def tearDown(self):
        """Tear down."""
        if os.path.isdir(self.dirpath):
            shutil.rmtree(self.dirpath)

    def test_initialization_invalid_dirpath(self):
        with self.assertRaises(measurement.MeasurementDataError):
            m = measurement.Measurement('')

    def test_initialization_valid_dirpath(self):
        m = measurement.Measurement(self.dirpath)
        self.assertEqual(m.dirpath, self.dirpath)
        self.assertIsNone(m.scan_axis)
        self.assertEqual(len(m.data), 0)
        self.assertEqual(len(m.posx), 0)
        self.assertEqual(len(m.posy), 0)
        self.assertEqual(len(m.posz), 0)

    def test_add_empty_line_scan(self):
        m = measurement.Measurement(self.dirpath)

        posx = 1
        posy = 2
        posz = np.linspace(0, 20, 11)
        ls = measurement.LineScan(
            posx, posy, posz, None, None, None, self.dirpath)

        with self.assertRaises(measurement.MeasurementDataError):
            m.add_line_scan(ls)

    def test_add_line_scan(self):
        m = measurement.Measurement(self.dirpath)

        posx = 1
        posy = 2
        posz = np.linspace(0, 20, 11)

        scan = measurement.DataSet()
        scan.posx = posx
        scan.posy = posy
        scan.posz = posz
        scan.datax = np.linspace(0, 20, 11)

        ls = measurement.LineScan(
            posx, posy, posz, None, None, None, self.dirpath)
        ls.add_scan(scan)

        m.add_line_scan(ls)
        self.assertEqual(len(m.data), 1)
        self.assertEqual(m.posy[0], posy)
        self.assertEqual(m.posx[0], posx)
        np.testing.assert_array_almost_equal(m.posz, posz)

    def test_add_same_line_scan_twice(self):
        m = measurement.Measurement(self.dirpath)

        posx = 1
        posy = 2
        posz = np.linspace(0, 20, 11)

        scan = measurement.DataSet()
        scan.posx = posx
        scan.posy = posy
        scan.posz = posz
        scan.datax = np.linspace(0, 20, 11)

        ls = measurement.LineScan(
            posx, posy, posz, None, None, None, self.dirpath)
        ls.add_scan(scan)

        m.add_line_scan(ls)
        m.add_line_scan(ls)
        self.assertEqual(len(m.data), 1)

    def test_add_line_scan_different_axis(self):
        m = measurement.Measurement(self.dirpath)

        posx = 1
        posy = 2
        posz = np.linspace(0, 20, 11)

        scan = measurement.DataSet()
        scan.posx = posx
        scan.posy = posy
        scan.posz = posz
        scan.datax = np.linspace(0, 20, 11)

        ls1 = measurement.LineScan(
            posx, posy, posz, None, None, None, self.dirpath)
        ls1.add_scan(scan)
        m.add_line_scan(ls1)

        scan.posy = posz
        scan.posz = posy
        ls2 = measurement.LineScan(
            posx, posz, posy, None, None, None, self.dirpath)
        ls2.add_scan(scan)

        with self.assertRaises(measurement.MeasurementDataError):
            m.add_line_scan(ls2)

    def test_add_line_scan_different_line(self):
        m = measurement.Measurement(self.dirpath)

        posz = np.linspace(0, 20, 11)

        scan = measurement.DataSet()
        scan.posz = posz
        scan.datax = np.linspace(0, 20, 11)

        scan.posx = 1
        scan.posy = 2
        ls1 = measurement.LineScan(
            1, 2, posz, None, None, None, self.dirpath)
        ls1.add_scan(scan)
        m.add_line_scan(ls1)

        scan.posx = 1
        scan.posy = 20
        ls2 = measurement.LineScan(
            1, 20, posz, None, None, None, self.dirpath)
        ls2.add_scan(scan)
        m.add_line_scan(ls2)

        self.assertEqual(len(m.data), 2)

    def test_clear(self):
        m = measurement.Measurement(self.dirpath)

        posx = 1
        posy = 2
        posz = np.linspace(0, 20, 11)

        scan = measurement.DataSet()
        scan.posx = posx
        scan.posy = posy
        scan.posz = posz
        scan.datax = np.linspace(0, 20, 11)

        ls = measurement.LineScan(
            posx, posy, posz, None, None, None, self.dirpath)
        ls.add_scan(scan)

        m.add_line_scan(ls)
        self.assertEqual(m.scan_axis, 'z')
        np.testing.assert_array_almost_equal(m.scan_positions, posz)
        np.testing.assert_array_almost_equal(m.posx[0], posx)
        np.testing.assert_array_almost_equal(m.posy[0], posy)
        np.testing.assert_array_almost_equal(m.posz, posz)
        self.assertEqual(len(m.data), 1)

        m.clear()
        self.assertIsNone(m.scan_axis)
        self.assertEqual(len(m.posx), 0)
        self.assertEqual(len(m.posy), 0)
        self.assertEqual(len(m.posz), 0)
        self.assertEqual(len(m.data), 0)

    def test_recover_saved_data(self):
        posz = np.linspace(0, 20, 11)

        scan = measurement.DataSet()
        scan.posz = posz
        scan.datax = np.linspace(0, 20, 11)

        scan.posx = 1
        scan.posy = 2
        ls1 = measurement.LineScan(
            1, 2, posz, None, None, self.calibration, self.dirpath)
        ls1.add_scan(scan)
        ls1.analyse_data()

        scan.posx = 1
        scan.posy = 20
        ls2 = measurement.LineScan(
            1, 20, posz, None, None, self.calibration, self.dirpath)
        ls2.add_scan(scan)
        ls2.analyse_data()

        x = np.ones(22)
        y = np.array([2, 20]*11)
        z = [
            0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14,
            16, 16, 18, 18, 20, 20,
        ]
        bx = [
            0, 0, 0.4, 0.4, 0.8, 0.8, 1.2, 1.2, 1.6, 1.6, 1.9644, 1.9644,
            2.386275, 2.386275, 2.855594, 2.855594, 3.456014, 3.456014,
            4.271197, 4.271197, 5.3848, 5.3848,
        ]
        by = np.ones(22)*np.nan
        bz = np.ones(22)*np.nan

        m = measurement.Measurement(self.dirpath)
        m.recover_saved_data()
        field_avg_data = m._get_field_avg_data()
        self.assertEqual(m.scan_axis, 'z')
        self.assertFalse(m.check_control_configuration())
        self.assertFalse(m.check_measurement_configuration())
        self.assertTrue(m.check_calibration())
        np.testing.assert_array_almost_equal(m.posx, 1)
        np.testing.assert_array_almost_equal(m.posy, [2, 20])
        np.testing.assert_array_almost_equal(m.posz, np.linspace(0, 20, 11))
        np.testing.assert_array_almost_equal(field_avg_data[:, 0], x)
        np.testing.assert_array_almost_equal(field_avg_data[:, 1], y)
        np.testing.assert_array_almost_equal(field_avg_data[:, 2], z)
        np.testing.assert_array_almost_equal(field_avg_data[:, 3], bx)
        np.testing.assert_array_almost_equal(field_avg_data[:, 4], by)
        np.testing.assert_array_almost_equal(field_avg_data[:, 5], bz)


def get_suite():
    suite_list = []
    suite_list.append(unittest.TestLoader().loadTestsFromTestCase(
        TestFunctions))
    suite_list.append(unittest.TestLoader().loadTestsFromTestCase(
        TestDataSet))
    suite_list.append(unittest.TestLoader().loadTestsFromTestCase(
        TestLineScan))
    suite_list.append(unittest.TestLoader().loadTestsFromTestCase(
        TestMeasurement))
    return unittest.TestSuite(suite_list)
