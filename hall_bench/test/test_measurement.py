"""Measurement test."""

import os
import shutil
import unittest
import numpy as np
import pandas as pd

from hall_bench.data_handle import measurement
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

    def test_to_array(self):
        value = measurement._to_array(None)
        self.assertTrue(isinstance(value, np.ndarray))
        self.assertEqual(value.size, 0)

        value = measurement._to_array(1)
        self.assertTrue(isinstance(value, np.ndarray))
        self.assertEqual(value, np.array(1))

        value = measurement._to_array([1])
        self.assertTrue(isinstance(value, np.ndarray))
        self.assertEqual(value, np.array(1))

        value = measurement._to_array([1, 1])
        self.assertTrue(isinstance(value, np.ndarray))

        value = measurement._to_array(np.array([1, 1]))
        self.assertTrue(isinstance(value, np.ndarray))

    def test_get_axis_position_dict(self):
        with self.assertRaises(measurement.MeasurementDataError):
            _dict = measurement._get_axis_position_dict([])

        vd1 = measurement.VoltageData()
        with self.assertRaises(measurement.MeasurementDataError):
            _dict = measurement._get_axis_position_dict([vd1])

        vd1.pos1 = 1
        vd1.pos2 = 4
        vd1.pos3 = [1, 4]
        with self.assertRaises(measurement.MeasurementDataError):
            _dict = measurement._get_axis_position_dict([vd1])

        vd1.probei = [1, 2]
        _dict = measurement._get_axis_position_dict([vd1])
        self.assertEqual(_dict[1], [1])
        self.assertEqual(_dict[2], [4])
        self.assertEqual(_dict[3], [1, 4])

        vd2 = measurement.VoltageData()
        vd2.pos1 = 1
        vd2.pos2 = 5
        vd2.pos3 = [3, 5]
        vd2.probei = [1, 2]

        vd3 = measurement.VoltageData()
        vd3.pos1 = 1
        vd3.pos2 = 6
        vd3.pos3 = [1, 2]
        vd3.probei = [1, 2]

        _dict = measurement._get_axis_position_dict([vd1, vd2, vd3])
        self.assertEqual(_dict[1], [1])
        self.assertEqual(_dict[2], [4, 5, 6])
        self.assertEqual(_dict[3], [1, 2, 3, 4, 5])

        vd4 = measurement.VoltageData()
        vd4.pos1 = 1
        vd4.pos2 = [1, 2]
        vd4.pos3 = 3
        vd4.probei = [1, 2]
        with self.assertRaises(measurement.MeasurementDataError):
            _dict = measurement._get_axis_position_dict([vd1, vd4])

    def test_get_measurement_axes(self):
        vd1 = measurement.VoltageData()
        vd1.pos1 = 1
        vd1.pos2 = 4
        vd1.pos3 = [1, 4]
        vd1.probei = [1, 2]
        axes = measurement._get_measurement_axes([vd1])
        self.assertEqual(axes, [3])

        vd2 = measurement.VoltageData()
        vd2.pos1 = 1
        vd2.pos2 = 5
        vd2.pos3 = [3, 5]
        vd2.probei = [1, 2]

        vd3 = measurement.VoltageData()
        vd3.pos1 = 1
        vd3.pos2 = 6
        vd3.pos3 = [1, 2]
        vd3.probei = [1, 2]

        axes = measurement._get_measurement_axes([vd1, vd2, vd3])
        self.assertEqual(axes, [3, 2])

        vd4 = measurement.VoltageData()
        vd4.pos1 = 2
        vd4.pos2 = 5
        vd4.pos3 = [1, 2]
        vd4.probei = [1, 2]

        axes = measurement._get_measurement_axes([vd1, vd2, vd3, vd4])
        self.assertEqual(axes, [3, 1, 2])

    def test_get_average_voltage_list(self):
        vd1 = measurement.VoltageData()
        vd1.pos1 = 5
        vd1.pos2 = 4
        vd1.pos3 = [1, 2, 3]
        vd1.probei = [1, 2, 3]

        vd2 = vd1.copy()
        vd2.pos1 = 1
        vd2.pos2 = 7

        vd3 = vd1.copy()
        vd3.pos1 = 2
        vd3.pos2 = 3
        vd3.pos8 = 4

        vd4 = vd1.copy()
        vd4.pos1 = 1
        vd4.pos2 = 2
        vd4.pos7 = 1

        vd5 = vd4.copy()

        voltage_list = measurement._get_average_voltage_list(
            [vd1, vd2, vd3, vd1, vd4, vd2, vd5])

        self.assertEqual(len(voltage_list), 4)

    def test_get_avg_std(self):
        npts = 11
        pos = np.linspace(-10, 10, npts)
        voltage = np.linspace(-15, 15, npts)

        vd1 = measurement.VoltageData()
        with self.assertRaises(measurement.MeasurementDataError):
            avg, std = measurement._get_avg_std([vd1])

        vd1.pos1 = pos
        with self.assertRaises(measurement.MeasurementDataError):
            avg, std = measurement._get_avg_std([vd1])

        vd1.probei = voltage
        avg, std = measurement._get_avg_std([vd1])

        np.testing.assert_array_equal(vd1.pos1, avg.pos1)
        np.testing.assert_array_equal(vd1.pos1, std.pos1)
        np.testing.assert_array_equal(vd1.probei, avg.probei)
        np.testing.assert_array_equal(np.zeros(npts), std.probei)

        vd1 = measurement.VoltageData()
        vd1.pos1 = pos - 1
        vd1.probei = voltage - (voltage[0]/pos[0]) - 1

        vd2 = measurement.VoltageData()
        vd2.pos1 = pos + 1
        vd2.probei = voltage + (voltage[0]/pos[0]) + 1

        avg, std = measurement._get_avg_std([vd1, vd2])
        np.testing.assert_array_equal(avg.pos1, pos)
        np.testing.assert_array_equal(std.pos1, pos)
        np.testing.assert_array_equal(avg.probei, voltage)
        np.testing.assert_array_equal(std.probei, np.ones(npts))

        vd3 = measurement.VoltageData()
        vd3.pos1 = [1, 2]
        vd3.probei = [1, 2]
        with self.assertRaises(measurement.MeasurementDataError):
            avg, std = measurement._get_avg_std([vd1, vd3])

        vd4 = measurement.VoltageData()
        vd4.pos3 = [1, 2]
        vd4.probei = [1, 2]
        with self.assertRaises(measurement.MeasurementDataError):
            avg, std = measurement._get_avg_std([vd1, vd4])

    def test_interpolate_data_frames(self):
        fieldj = pd.DataFrame([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]],
                              index=[0, 1], columns=[1, 2, 3])
        fieldi = pd.DataFrame([[0.15, 0.25, 0.35], [1.15, 1.25, 1.35]],
                              index=[0, 1], columns=[1.5, 2.5, 3.5])
        fieldk = pd.DataFrame([[0.05, 0.15, 0.25], [1.05, 1.15, 1.25]],
                              index=[0, 1], columns=[0.5, 1.5, 2.5])

        fi, fj, fk = measurement._interpolate_data_frames(
            fieldi, fieldj, fieldk, axis=1)

        np.testing.assert_array_almost_equal(fi.values, fj.values)
        np.testing.assert_array_almost_equal(fk.values, fj.values)
        np.testing.assert_array_almost_equal(fi.index.values, fj.index.values)
        np.testing.assert_array_almost_equal(fk.index.values, fj.index.values)
        np.testing.assert_array_almost_equal(
            fi.columns.values, fj.columns.values)
        np.testing.assert_array_almost_equal(
            fk.columns.values, fj.columns.values)

        fieldj = pd.DataFrame([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]],
                              index=[0, 1], columns=[1, 2, 3])
        fieldi = pd.DataFrame([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
                              index=[1, 2], columns=[1, 2, 3])
        fieldk = pd.DataFrame([[2.1, 2.2, 2.3], [3.1, 3.2, 3.3]],
                              index=[2, 3], columns=[1, 2, 3])

        fi, fj, fk = measurement._interpolate_data_frames(
            fieldi, fieldj, fieldk)

        np.testing.assert_array_almost_equal(fi.values, fj.values)
        np.testing.assert_array_almost_equal(fk.values, fj.values)
        np.testing.assert_array_almost_equal(fi.index.values, fj.index.values)
        np.testing.assert_array_almost_equal(fk.index.values, fj.index.values)
        np.testing.assert_array_almost_equal(
            fi.columns.values, fj.columns.values)
        np.testing.assert_array_almost_equal(
            fk.columns.values, fj.columns.values)

    def test_get_number_of_cuts(self):
        vec = [1, 2, 3, 4, 5, 12, 17, 24, 30, 40, 50]
        nbeg, nend = measurement._get_number_of_cuts(vec, 3.5, 8)
        self.assertEqual(nbeg, 4)
        self.assertEqual(nend, 1)

        nbeg, nend = measurement._get_number_of_cuts(vec, 4, 10)
        self.assertEqual(nbeg, 4)
        self.assertEqual(nend, 1)

        nbeg, nend = measurement._get_number_of_cuts(vec, 0, 0)
        self.assertEqual(nbeg, 0)
        self.assertEqual(nend, 0)

    def test_cut_data_frames(self):
        data = np.array([
            [1, 2, 3, 4], [4, 5, 6, 8], [9, 10, 11, 12],
            [13, 14, 15, 16], [17, 18, 19, 20]
        ])
        fieldi = pd.DataFrame(data)
        fieldj = pd.DataFrame()
        fieldk = pd.DataFrame()

        fi, fj, fk = measurement._cut_data_frames(
            fieldi, fieldj, fieldk, 2, 1, axis=0)
        np.testing.assert_array_equal(fi.values, data[2:-1, :])

        fi, fj, fk = measurement._cut_data_frames(
            fieldi, fieldj, fieldk, 2, 1, axis=1)
        np.testing.assert_array_equal(fi.values, data[:, 2:-1])


class TestVoltageData(unittest.TestCase):
    """Test VoltageData."""

    def setUp(self):
        """Set up."""
        self.filename = 'test_voltage_data.txt'

    def tearDown(self):
        """Tear down."""
        pass

    def test_initialization(self):
        vd = measurement.VoltageData()
        self.assertEqual(vd.pos1.size, 0)
        self.assertEqual(vd.pos2.size, 0)
        self.assertEqual(vd.pos3.size, 0)
        self.assertEqual(vd.pos5.size, 0)
        self.assertEqual(vd.pos6.size, 0)
        self.assertEqual(vd.pos7.size, 0)
        self.assertEqual(vd.pos8.size, 0)
        self.assertEqual(vd.pos9.size, 0)
        self.assertEqual(vd.probei.size, 0)
        self.assertEqual(vd.probej.size, 0)
        self.assertEqual(vd.probek.size, 0)

    def test_reverse(self):
        vd = measurement.VoltageData()
        vec = np.array([1, 2, 3, 4, 5])
        vd.probei = vec
        vd.reverse()
        np.testing.assert_array_equal(vd.probei, vec[::-1])

    def test_copy(self):
        vd = measurement.VoltageData()
        vec = [1, 2, 3, 4, 5]
        vd.probei = vec
        vd2 = vd.copy()
        vd.probei = np.array([])
        np.testing.assert_array_equal(vd2.probei, vec)

    def test_clear(self):
        vd = measurement.VoltageData()
        vd.pos1 = 10
        vd.pos2 = 2
        vd.pos3 = 3
        vd.pos5 = 5
        vd.pos6 = 6
        vd.pos7 = 7
        vd.pos8 = 8
        vd.pos9 = 9
        vd.probei = 4
        vd.probej = 5
        vd.probek = 6
        vd.clear()
        self.assertEqual(vd.pos1.size, 0)
        self.assertEqual(vd.pos2.size, 0)
        self.assertEqual(vd.pos3.size, 0)
        self.assertEqual(vd.pos5.size, 0)
        self.assertEqual(vd.pos6.size, 0)
        self.assertEqual(vd.pos7.size, 0)
        self.assertEqual(vd.pos8.size, 0)
        self.assertEqual(vd.pos9.size, 0)
        self.assertEqual(vd.probei.size, 0)
        self.assertEqual(vd.probej.size, 0)
        self.assertEqual(vd.probek.size, 0)

    def test_scan_axis(self):
        vd = measurement.VoltageData()
        self.assertIsNone(vd.scan_axis)

        vd.pos7 = [1, 2]
        self.assertEqual(vd.scan_axis, 7)

        vd.pos2 = [3]
        self.assertEqual(vd.scan_axis, 7)

        vd.pos3 = [4, 5]
        self.assertIsNone(vd.scan_axis)

        vd.pos7 = []
        self.assertEqual(vd.scan_axis, 3)

    def test_npts(self):
        vd = measurement.VoltageData()
        self.assertEqual(vd.npts, 0)

        vd.pos1 = [1, 2]
        self.assertEqual(vd.npts, 0)

        vd.probej = [3]
        self.assertEqual(vd.npts, 0)

        vd.probek = [4, 5]
        self.assertEqual(vd.npts, 0)

        vd.probej = []
        self.assertEqual(vd.npts, 2)

    def test_read_write(self):
        pos_scan = np.linspace(-10, 10, 101)
        voltage = np.linspace(-15, 15, 101)

        vd = measurement.VoltageData()
        with self.assertRaises(measurement.MeasurementDataError):
            vd.save_file(self.filename)

        vd.pos1 = pos_scan
        with self.assertRaises(measurement.MeasurementDataError):
            vd.save_file(self.filename)

        vd.pos2 = 0
        vd.pos3 = 0
        vd.pos5 = 0
        vd.pos6 = 0
        vd.pos7 = 0
        vd.pos8 = 0
        vd.pos9 = 0
        vd.probej = voltage
        vd.save_file(self.filename)

        vdr = measurement.VoltageData()
        vdr.read_file(self.filename)
        self.assertEqual(vd.pos2, vdr.pos2)
        self.assertEqual(vd.pos3, vdr.pos3)
        self.assertEqual(vd.pos5, vdr.pos5)
        self.assertEqual(vd.pos6, vdr.pos6)
        self.assertEqual(vd.pos7, vdr.pos7)
        self.assertEqual(vd.pos8, vdr.pos8)
        self.assertEqual(vd.pos9, vdr.pos9)

        np.testing.assert_array_equal(vd.pos1, vdr.pos1)
        np.testing.assert_array_equal(vd.probei, vdr.probei)
        np.testing.assert_array_almost_equal(vd.probej, vdr.probej)
        np.testing.assert_array_equal(vd.probek, vdr.probek)
        os.remove(self.filename)


class TestFieldData(unittest.TestCase):
    """Test FieldData."""

    def setUp(self):
        """Set up."""
        self.calibration_data = calibration.CalibrationData(
            'calibration_data.txt')

        self.pos = np.linspace(0, 20, 11)
        self.voltage = np.linspace(0, 20, 11)
        self.field = [
            0, 0.4, 0.8, 1.2, 1.6, 2,
            2.386275, 2.855594, 3.456014,
            4.271197, 5.3848,
        ]

        self.vd1 = measurement.VoltageData()
        self.vd1.pos3 = 2
        self.vd1.pos2 = 10
        self.vd1.pos1 = self.pos
        self.vd1.probei = np.zeros(len(self.voltage))
        self.vd1.probej = self.voltage
        self.vd1.probek = np.zeros(len(self.voltage))

        self.vd2 = measurement.VoltageData()
        self.vd2.pos3 = 3
        self.vd2.pos2 = 10
        self.vd2.pos1 = self.pos
        self.vd2.probei = np.zeros(len(self.voltage))
        self.vd2.probej = self.voltage
        self.vd2.probek = np.zeros(len(self.voltage))

    def tearDown(self):
        """Tear down."""
        pass

    def test_initialization(self):
        voltage_list = [self.vd1, self.vd2]
        fd = measurement.FieldData(voltage_list, self.calibration_data)
        np.testing.assert_array_equal(fd.pos1, np.linspace(0, 20, 11))
        np.testing.assert_array_equal(fd.pos2, [10])
        np.testing.assert_array_equal(fd.pos3, [2, 3])
        np.testing.assert_array_equal(fd.field1, np.zeros([11, 2]))
        np.testing.assert_array_almost_equal(
            fd.field2, np.transpose([self.field, self.field]), decimal=4)
        np.testing.assert_array_equal(fd.field3, np.zeros([11, 2]))
        self.assertEqual(fd.index_axis, 1)
        self.assertEqual(fd.columns_axis, 3)

        voltage_list = [self.vd1, self.vd2, self.vd2]
        fd = measurement.FieldData(voltage_list, self.calibration_data)
        np.testing.assert_array_equal(fd.pos1, np.linspace(0, 20, 11))
        np.testing.assert_array_equal(fd.pos2, [10])
        np.testing.assert_array_equal(fd.pos3, [2, 3])
        np.testing.assert_array_equal(fd.field1, np.zeros([11, 2]))
        np.testing.assert_array_almost_equal(
            fd.field2, np.transpose([self.field, self.field]), decimal=4)
        np.testing.assert_array_equal(fd.field3, np.zeros([11, 2]))
        self.assertEqual(fd.index_axis, 1)
        self.assertEqual(fd.columns_axis, 3)

        vd3 = measurement.VoltageData()
        vd3.pos3 = 3
        vd3.pos2 = 11
        vd3.pos1 = self.pos
        vd3.probei = np.zeros(len(self.voltage))
        vd3.probej = self.voltage
        vd3.probek = np.zeros(len(self.voltage))
        with self.assertRaises(measurement.MeasurementDataError):
            voltage_list = [self.vd1, self.vd2, vd3]
            fd = measurement.FieldData(voltage_list, self.calibration_data)

        voltage_list = [self.vd1]
        fd = measurement.FieldData(voltage_list, self.calibration_data)
        self.assertEqual(fd.index_axis, 1)
        self.assertIsNone(fd.columns_axis)
        np.testing.assert_array_equal(fd.pos1, np.linspace(0, 20, 11))
        np.testing.assert_array_equal(fd.pos2, [10])
        np.testing.assert_array_equal(fd.pos3, [2])
        np.testing.assert_array_equal(fd.field1, np.zeros([11, 1]))
        np.testing.assert_array_almost_equal(
            fd.field2, np.transpose([self.field]), decimal=4)
        np.testing.assert_array_equal(fd.field3, np.zeros([11, 1]))

    def test_field_at_point(self):
        voltage_list = [self.vd1, self.vd2]
        fd = measurement.FieldData(voltage_list, self.calibration_data)
        np.testing.assert_array_equal(
            fd.get_field_at_point([2, 10, 0]), [0, 0, 0])
        np.testing.assert_array_equal(
            fd.get_field_at_point([3, 10, 10]), [0, 2, 0])
        np.testing.assert_array_equal(
            fd.get_field_at_point([3, 10, 20]), [0, 5.3848, 0])

    def test_get_fieldmap(self):
        voltage_list = [self.vd1, self.vd2]
        fd = measurement.FieldData(voltage_list, self.calibration_data)
        fieldmap = fd._get_fieldmap()

        x = [2, 3]*11
        y = [10]*22
        z = np.reshape(np.transpose([self.pos, self.pos]), 22)
        bx = [0]*22
        by = np.reshape(np.transpose([self.field, self.field]), 22)
        bz = [0]*22

        np.testing.assert_array_almost_equal(fieldmap[:, 0], x)
        np.testing.assert_array_almost_equal(fieldmap[:, 1], y)
        np.testing.assert_array_almost_equal(fieldmap[:, 2], z)
        np.testing.assert_array_almost_equal(fieldmap[:, 3], bx)
        np.testing.assert_array_almost_equal(fieldmap[:, 4], by)
        np.testing.assert_array_almost_equal(fieldmap[:, 5], bz)

    def test_get_transformed_fieldmap(self):
        voltage_list = [self.vd1, self.vd2]
        fd = measurement.FieldData(voltage_list, self.calibration_data)
        fieldmap = fd._get_transformed_fieldmap(
            magnet_center=[0, 0, 0], magnet_x_axis='3', magnet_y_axis='2')

        x = [2, 3]*11
        y = [10]*22
        z = np.reshape(np.transpose([self.pos, self.pos]), 22)
        bx = [0]*22
        by = np.reshape(np.transpose([self.field, self.field]), 22)
        bz = [0]*22

        np.testing.assert_array_almost_equal(fieldmap[:, 0], x)
        np.testing.assert_array_almost_equal(fieldmap[:, 1], y)
        np.testing.assert_array_almost_equal(fieldmap[:, 2], z)
        np.testing.assert_array_almost_equal(fieldmap[:, 3], bx)
        np.testing.assert_array_almost_equal(fieldmap[:, 4], by)
        np.testing.assert_array_almost_equal(fieldmap[:, 5], bz)

        fieldmap = fd._get_transformed_fieldmap(
            magnet_center=[10, -40, 30])

        x = np.array([2, 3]*11)
        y = np.array([10]*22)
        z = np.reshape(np.transpose([self.pos, self.pos]), 22)
        bx = np.array([0]*22)
        by = np.reshape(np.transpose([self.field, self.field]), 22)
        bz = np.array([0]*22)

        np.testing.assert_array_almost_equal(fieldmap[:, 0], x - 10)
        np.testing.assert_array_almost_equal(fieldmap[:, 1], y + 40)
        np.testing.assert_array_almost_equal(fieldmap[:, 2], z - 30)
        np.testing.assert_array_almost_equal(fieldmap[:, 3], bx)
        np.testing.assert_array_almost_equal(fieldmap[:, 4], by)
        np.testing.assert_array_almost_equal(fieldmap[:, 5], bz)

        fieldmap = fd._get_transformed_fieldmap(
            magnet_x_axis='-1', magnet_y_axis='2')

        x = np.array([2]*11 + [3]*11)
        y = np.array([10]*22)
        z = np.reshape([self.pos, self.pos], 22)[::-1]
        bx = np.array([0]*22)
        by = np.reshape([self.field, self.field], 22)[::-1]
        bz = np.array([0]*22)

        np.testing.assert_array_almost_equal(fieldmap[:, 0], (-1)*z)
        np.testing.assert_array_almost_equal(fieldmap[:, 1], y)
        np.testing.assert_array_almost_equal(fieldmap[:, 2], x)
        np.testing.assert_array_almost_equal(fieldmap[:, 3], bx)
        np.testing.assert_array_almost_equal(fieldmap[:, 4], by)
        np.testing.assert_array_almost_equal(fieldmap[:, 5], bz)

        fieldmap = fd._get_transformed_fieldmap(
            magnet_center=[10, -40, 30], magnet_x_axis='-1', magnet_y_axis='2')

        x = np.array([2]*11 + [3]*11)
        y = np.array([10]*22)
        z = np.reshape([self.pos, self.pos], 22)[::-1]
        bx = np.array([0]*22)
        by = np.reshape([self.field, self.field], 22)[::-1]
        bz = np.array([0]*22)

        np.testing.assert_array_almost_equal(fieldmap[:, 0], (-1)*z + 30)
        np.testing.assert_array_almost_equal(fieldmap[:, 1], y + 40)
        np.testing.assert_array_almost_equal(fieldmap[:, 2], x - 10)
        np.testing.assert_array_almost_equal(fieldmap[:, 3], bx)
        np.testing.assert_array_almost_equal(fieldmap[:, 4], by)
        np.testing.assert_array_almost_equal(fieldmap[:, 5], bz)

    def test_save(self):
        voltage_list = [self.vd1, self.vd2]
        fd = measurement.FieldData(voltage_list, self.calibration_data)
        fd.save_file(
            'test_fieldmap.dat', header_info=[('test_header_info', 10)],
            magnet_center=[0, 0, 0], magnet_x_axis='3', magnet_y_axis='2')
        os.remove('test_fieldmap.dat')
        os.remove('magnet_coordinate_system.txt')


def get_suite():
    suite_list = []
    suite_list.append(unittest.TestLoader().loadTestsFromTestCase(
        TestFunctions))
    suite_list.append(unittest.TestLoader().loadTestsFromTestCase(
        TestVoltageData))
    suite_list.append(unittest.TestLoader().loadTestsFromTestCase(
        TestFieldData))
    return unittest.TestSuite(suite_list)
