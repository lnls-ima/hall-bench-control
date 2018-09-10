"""Measurement test."""

import os
import filecmp
import shutil
import numpy as np
import pandas as pd
from scipy import interpolate
from unittest import TestCase
from hallbench.data import measurement
from hallbench.data import calibration
from hallbench.data import utils


class TestFunctions(TestCase):
    """Test functions."""

    def setUp(self):
        """Set up."""
        vd = measurement.VoltageScan()
        vd.pos1 = [1, 2]
        vd.pos2 = 1
        vd.pos3 = 1
        vd.pos5 = 1
        vd.pos6 = 1
        vd.pos7 = 1
        vd.pos8 = 1
        vd.pos9 = 1
        vd.avgx = [1, 2]
        vd.avgy = [3, 4]
        vd.avgz = [0, 0]
        self.vd = vd

        fd = measurement.FieldScan()
        fd._pos1 = utils.to_array([1, 2])
        fd._pos2 = utils.to_array(1)
        fd._pos3 = utils.to_array(1)
        fd._pos5 = utils.to_array(1)
        fd._pos6 = utils.to_array(1)
        fd._pos7 = utils.to_array(1)
        fd._pos8 = utils.to_array(1)
        fd._pos9 = utils.to_array(1)
        fd._avgx = utils.to_array([1, 2])
        fd._avgy = utils.to_array([3, 4])
        fd._avgz = utils.to_array([0, 0])
        self.fd = fd

        sc = calibration.HallSensor()
        sc.function_type = 'polynomial'
        sc.data = [-1000, 1000, 0, 1]
        pc = calibration.HallProbe()
        pc.sensorx = sc
        pc.sensory = sc
        pc.sensorz = sc
        self.pc = pc

    def tearDown(self):
        """Tear down."""
        pass

    def test_interpolate_data_frames(self):
        fieldy = pd.DataFrame([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]],
                              index=[0, 1], columns=[1, 2, 3])
        fieldx = pd.DataFrame([[0.15, 0.25, 0.35], [1.15, 1.25, 1.35]],
                              index=[0, 1], columns=[1.5, 2.5, 3.5])
        fieldz = pd.DataFrame([[0.05, 0.15, 0.25], [1.05, 1.15, 1.25]],
                              index=[0, 1], columns=[0.5, 1.5, 2.5])

        fx, fy, fz = measurement._interpolate_data_frames(
            fieldx, fieldy, fieldz, axis=1)

        np.testing.assert_array_almost_equal(fx.values, fy.values)
        np.testing.assert_array_almost_equal(fz.values, fy.values)
        np.testing.assert_array_almost_equal(fx.index.values, fy.index.values)
        np.testing.assert_array_almost_equal(fz.index.values, fy.index.values)
        np.testing.assert_array_almost_equal(
            fx.columns.values, fy.columns.values)
        np.testing.assert_array_almost_equal(
            fz.columns.values, fy.columns.values)

        fieldy = pd.DataFrame([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]],
                              index=[0, 1], columns=[1, 2, 3])
        fieldx = pd.DataFrame([[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
                              index=[1, 2], columns=[1, 2, 3])
        fieldz = pd.DataFrame([[2.1, 2.2, 2.3], [3.1, 3.2, 3.3]],
                              index=[2, 3], columns=[1, 2, 3])

        fx, fy, fz = measurement._interpolate_data_frames(
            fieldx, fieldy, fieldz)

        np.testing.assert_array_almost_equal(fx.values, fy.values)
        np.testing.assert_array_almost_equal(fz.values, fy.values)
        np.testing.assert_array_almost_equal(fx.index.values, fy.index.values)
        np.testing.assert_array_almost_equal(fz.index.values, fy.index.values)
        np.testing.assert_array_almost_equal(
            fx.columns.values, fy.columns.values)
        np.testing.assert_array_almost_equal(
            fz.columns.values, fy.columns.values)

    def test_get_number_of_cuts(self):
        px = np.linspace(-10, 0, 11)
        py = np.linspace(-5, 5, 11)
        pz = np.linspace(3, 10, 11)
        nbeg, nend = measurement._get_number_of_cuts(px, py, pz)
        self.assertEqual(nbeg, 5)
        self.assertEqual(nend, 8)

    def test_cut_data_frames(self):
        data = np.array([
            [1, 2, 3, 4], [4, 5, 6, 8], [9, 10, 11, 12],
            [13, 14, 15, 16], [17, 18, 19, 20]
        ])
        fieldx = pd.DataFrame(data)
        fieldy = pd.DataFrame()
        fieldz = pd.DataFrame()

        fx, fy, fz = measurement._cut_data_frames(
            fieldx, fieldy, fieldz, 2, 1, axis=0)
        np.testing.assert_array_equal(fx.values, data[2:-1, :])

        fx, fy, fz = measurement._cut_data_frames(
            fieldx, fieldy, fieldz, 2, 1, axis=1)
        np.testing.assert_array_equal(fx.values, data[:, 2:-1])

    def test_get_transformation_matrix(self):
        m = measurement._get_transformation_matrix(3, 2)
        np.testing.assert_array_equal(m[0], [1, 0, 0])
        np.testing.assert_array_equal(m[1], [0, 1, 0])
        np.testing.assert_array_equal(m[2], [0, 0, 1])

        m = measurement._get_transformation_matrix(-3, 2)
        np.testing.assert_array_equal(m[0], [-1, 0, 0])
        np.testing.assert_array_equal(m[1], [0, 1, 0])
        np.testing.assert_array_equal(m[2], [0, 0, -1])

        m = measurement._get_transformation_matrix(1, 2)
        np.testing.assert_array_equal(m[0], [0, 0, 1])
        np.testing.assert_array_equal(m[1], [0, 1, 0])
        np.testing.assert_array_equal(m[2], [-1, 0, 0])

        m = measurement._get_transformation_matrix(-1, 2)
        np.testing.assert_array_equal(m[0], [0, 0, -1])
        np.testing.assert_array_equal(m[1], [0, 1, 0])
        np.testing.assert_array_equal(m[2], [1, 0, 0])

        m = measurement._get_transformation_matrix(2, 1)
        np.testing.assert_array_equal(m[0], [0, 1, 0])
        np.testing.assert_array_equal(m[1], [0, 0, 1])
        np.testing.assert_array_equal(m[2], [1, 0, 0])

        with self.assertRaises(measurement.MeasurementDataError):
            m = measurement._get_transformation_matrix(2, 2)

    def test_change_coordinate_system(self):
        vec = [1, 2, 3]
        m = measurement._get_transformation_matrix(3, 2)
        tvec = measurement._change_coordinate_system(vec, m)
        np.testing.assert_array_equal(tvec, vec)

        m = measurement._get_transformation_matrix(1, 2)
        tvec = measurement._change_coordinate_system(vec, m)
        np.testing.assert_array_equal(tvec, [3, 2, -1])

        m = measurement._get_transformation_matrix(1, 2)
        center = [-10, -100, -1000]
        tvec = measurement._change_coordinate_system(vec, m, center)
        np.testing.assert_array_equal(tvec, [1003, 102, -11])

    def test_valid_voltage_scan_list(self):
        vd = measurement.VoltageScan()
        valid = measurement._valid_voltage_scan_list([vd])
        self.assertFalse(valid)

        valid = measurement._valid_voltage_scan_list([vd, None])
        self.assertFalse(valid)

        valid = measurement._valid_voltage_scan_list([])
        self.assertFalse(valid)

        valid = measurement._valid_voltage_scan_list([self.vd])
        self.assertTrue(valid)

        valid = measurement._valid_voltage_scan_list([self.vd, self.vd])
        self.assertTrue(valid)

        vdc = self.vd.copy()
        vdc.pos1 = self.vd.pos2
        vdc.pos2 = self.vd.pos1
        self.assertNotEqual(self.vd.scan_axis, vdc.scan_axis)
        valid = measurement._valid_voltage_scan_list([self.vd, vdc])
        self.assertFalse(valid)

        vdc = self.vd.copy()
        vdc.pos1 = np.append(vdc.pos1, 3)
        valid = measurement._valid_voltage_scan_list([self.vd, vdc])
        self.assertFalse(valid)

        vdc = self.vd.copy()
        vdc.pos2 = vdc.pos2 + 1
        valid = measurement._valid_voltage_scan_list([self.vd, vdc])
        self.assertFalse(valid)

    def test_get_avg_voltage(self):
        vdc = self.vd.copy()
        vdc.avgx = vdc.avgx + 2
        vdc.avgy = vdc.avgy + 2
        vdc.avgz = vdc.avgz + 2
        avg = measurement._get_avg_voltage([self.vd, vdc])

        np.testing.assert_array_almost_equal(
            avg.avgx, (self.vd.avgx + vdc.avgx)/2)
        np.testing.assert_array_almost_equal(
            avg.avgy, (self.vd.avgy + vdc.avgy)/2)
        np.testing.assert_array_almost_equal(
            avg.avgz, (self.vd.avgz + vdc.avgz)/2)

        vdc.avgz = []
        avg = measurement._get_avg_voltage([self.vd, vdc])
        np.testing.assert_array_almost_equal(
            avg.avgz, (self.vd.avgz + [0, 0])/2)

        vd = measurement.VoltageScan()
        with self.assertRaises(measurement.MeasurementDataError):
            avg = measurement._get_avg_voltage(vd)

        vdc = self.vd.copy()
        vdc.pos1 = vdc.pos1 + 2
        vdc.avgx = vdc.avgx + 2
        vdc.avgy = vdc.avgy + 2
        vdc.avgz = vdc.avgz + 2
        avg = measurement._get_avg_voltage([self.vd, vdc])

        np.testing.assert_array_almost_equal(
            avg.pos1, self.vd.pos1 + 1)
        np.testing.assert_array_almost_equal(
            avg.avgx, self.vd.avgx + 1)
        np.testing.assert_array_almost_equal(
            avg.avgy, self.vd.avgy + 1)
        np.testing.assert_array_almost_equal(
            avg.avgz, self.vd.avgz + 1)

    def test_get_fieldmap_axes(self):
        fd = measurement.FieldScan()
        axes = measurement._get_fieldmap_axes([fd])
        self.assertEqual(len(axes), 0)

        axes = measurement._get_fieldmap_axes([fd, None])
        self.assertEqual(len(axes), 0)

        axes = measurement._get_fieldmap_axes([])
        self.assertEqual(len(axes), 0)

        axes = measurement._get_fieldmap_axes([self.fd])
        self.assertEqual(len(axes), 1)

        axes = measurement._get_fieldmap_axes([self.fd, self.fd])
        self.assertEqual(len(axes), 0)

        fdc = self.fd.copy()
        fdc._pos1 = self.fd.pos2
        fdc._pos2 = self.fd.pos1
        self.assertNotEqual(self.fd.scan_axis, fdc.scan_axis)
        axes = measurement._get_fieldmap_axes([self.fd, fdc])
        self.assertEqual(len(axes), 0)

        fdc = self.fd.copy()
        fdc._pos2 = fdc.pos2 + 1
        axes = measurement._get_fieldmap_axes([self.fd, fdc])
        self.assertEqual(len(axes), 2)

        fdc = self.fd.copy()
        fdc._pos1 = np.append(fdc.pos1, 3)
        fdc._avgx = np.append(fdc.avgx, 3)
        fdc._avgy = np.append(fdc.avgy, 3)
        fdc._avgz = np.append(fdc.avgz, 3)
        fdc._pos2 = fdc.pos2 + 1
        axes = measurement._get_fieldmap_axes([self.fd, fdc])
        self.assertEqual(len(axes), 2)

    def test_get_fieldmap_position_and_field_values(self):
        voltage = [1, 2, 3, 4]
        field = self.pc.sensorx.get_field(voltage)
        np.testing.assert_array_almost_equal(field, voltage)


# class TestScan(TestCase):
#     """Test Scan."""
#
#     def setUp(self):
#         """Set up."""
#         self.base_directory = os.path.dirname(os.path.abspath(__file__))
#         self.filename = os.path.join(self.base_directory, 'data.txt')
#
#         self.pos1 = [1, 2, 3, 4, 5]
#         self.pos2 = [6]
#         self.pos3 = [7]
#         self.pos5 = [8]
#         self.pos6 = [9]
#         self.pos7 = [10]
#         self.pos8 = [11]
#         self.pos9 = [12]
#         self.sensorx = [1.3, 1.4, 1.5, 1.6, 1.7]
#         self.sensory = [1.8, 1.9, 2.0, 2.1, 2.2]
#         self.sensorz = [2.3, 2.4, 2.5, 2.6, 2.7]
#
#     def tearDown(self):
#         """Tear down."""
#         pass
#
#     def test_initialization_without_filename(self):
#         d = measurement.Scan()
#         np.testing.assert_array_equal(d._pos1, [])
#         np.testing.assert_array_equal(d._pos2, [])
#         np.testing.assert_array_equal(d._pos3, [])
#         np.testing.assert_array_equal(d._pos5, [])
#         np.testing.assert_array_equal(d._pos6, [])
#         np.testing.assert_array_equal(d._pos7, [])
#         np.testing.assert_array_equal(d._pos8, [])
#         np.testing.assert_array_equal(d._pos9, [])
#         np.testing.assert_array_equal(d._sensorx, [])
#         np.testing.assert_array_equal(d._sensory, [])
#         np.testing.assert_array_equal(d._sensorz, [])
#         self.assertEqual(d._data_unit, '')
#         self.assertIsNone(d._filename)
#         self.assertEqual(d.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(d._pos1_unit, 'mm')
#         self.assertEqual(d._pos2_unit, 'mm')
#         self.assertEqual(d._pos3_unit, 'mm')
#         self.assertEqual(d._pos5_unit, 'deg')
#         self.assertEqual(d._pos6_unit, 'mm')
#         self.assertEqual(d._pos7_unit, 'mm')
#         self.assertEqual(d._pos8_unit, 'deg')
#         self.assertEqual(d._pos9_unit, 'deg')
#         self.assertEqual(d.npts, 0)
#         self.assertIsNone(d.scan_axis)
#         self.assertIsNone(d.filename)
#
#     def test_initialization_with_filename(self):
#         d = measurement.Scan(self.filename)
#         np.testing.assert_array_equal(d._pos1, self.pos1)
#         np.testing.assert_array_equal(d._pos2, self.pos2)
#         np.testing.assert_array_equal(d._pos3, self.pos3)
#         np.testing.assert_array_equal(d._pos5, self.pos5)
#         np.testing.assert_array_equal(d._pos6, self.pos6)
#         np.testing.assert_array_equal(d._pos7, self.pos7)
#         np.testing.assert_array_equal(d._pos8, self.pos8)
#         np.testing.assert_array_equal(d._pos9, self.pos9)
#         np.testing.assert_array_equal(d._sensorx, self.sensorx)
#         np.testing.assert_array_equal(d._sensory, self.sensory)
#         np.testing.assert_array_equal(d._sensorz, self.sensorz)
#         self.assertEqual(d._data_unit, '')
#         self.assertEqual(d.npts, 5)
#         self.assertEqual(d.scan_axis, 1)
#         self.assertEqual(d.filename, self.filename)
#         self.assertEqual(d.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(d._pos1_unit, 'mm')
#         self.assertEqual(d._pos2_unit, 'mm')
#         self.assertEqual(d._pos3_unit, 'mm')
#         self.assertEqual(d._pos5_unit, 'deg')
#         self.assertEqual(d._pos6_unit, 'mm')
#         self.assertEqual(d._pos7_unit, 'mm')
#         self.assertEqual(d._pos8_unit, 'deg')
#         self.assertEqual(d._pos9_unit, 'deg')
#
#     def test_read_file(self):
#         d = measurement.Scan()
#         d.read_file(self.filename)
#         np.testing.assert_array_equal(d._pos1, self.pos1)
#         np.testing.assert_array_equal(d._pos2, self.pos2)
#         np.testing.assert_array_equal(d._pos3, self.pos3)
#         np.testing.assert_array_equal(d._pos5, self.pos5)
#         np.testing.assert_array_equal(d._pos6, self.pos6)
#         np.testing.assert_array_equal(d._pos7, self.pos7)
#         np.testing.assert_array_equal(d._pos8, self.pos8)
#         np.testing.assert_array_equal(d._pos9, self.pos9)
#         np.testing.assert_array_equal(d._sensorx, self.sensorx)
#         np.testing.assert_array_equal(d._sensory, self.sensory)
#         np.testing.assert_array_equal(d._sensorz, self.sensorz)
#         self.assertEqual(d._data_unit, '')
#         self.assertEqual(d.npts, 5)
#         self.assertEqual(d.scan_axis, 1)
#         self.assertEqual(d.filename, self.filename)
#         self.assertEqual(d.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(d._pos1_unit, 'mm')
#         self.assertEqual(d._pos2_unit, 'mm')
#         self.assertEqual(d._pos3_unit, 'mm')
#         self.assertEqual(d._pos5_unit, 'deg')
#         self.assertEqual(d._pos6_unit, 'mm')
#         self.assertEqual(d._pos7_unit, 'mm')
#         self.assertEqual(d._pos8_unit, 'deg')
#         self.assertEqual(d._pos9_unit, 'deg')
#
#     def test_clear(self):
#         d = measurement.Scan(self.filename)
#         np.testing.assert_array_equal(d._pos1, self.pos1)
#         np.testing.assert_array_equal(d._pos2, self.pos2)
#         np.testing.assert_array_equal(d._pos3, self.pos3)
#         np.testing.assert_array_equal(d._pos5, self.pos5)
#         np.testing.assert_array_equal(d._pos6, self.pos6)
#         np.testing.assert_array_equal(d._pos7, self.pos7)
#         np.testing.assert_array_equal(d._pos8, self.pos8)
#         np.testing.assert_array_equal(d._pos9, self.pos9)
#         np.testing.assert_array_equal(d._sensorx, self.sensorx)
#         np.testing.assert_array_equal(d._sensory, self.sensory)
#         np.testing.assert_array_equal(d._sensorz, self.sensorz)
#         self.assertEqual(d._data_unit, '')
#         self.assertEqual(d.npts, 5)
#         self.assertEqual(d.scan_axis, 1)
#         self.assertEqual(d.filename, self.filename)
#         self.assertEqual(d.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(d._pos1_unit, 'mm')
#         self.assertEqual(d._pos2_unit, 'mm')
#         self.assertEqual(d._pos3_unit, 'mm')
#         self.assertEqual(d._pos5_unit, 'deg')
#         self.assertEqual(d._pos6_unit, 'mm')
#         self.assertEqual(d._pos7_unit, 'mm')
#         self.assertEqual(d._pos8_unit, 'deg')
#         self.assertEqual(d._pos9_unit, 'deg')
#         d.clear()
#         np.testing.assert_array_equal(d._pos1, [])
#         np.testing.assert_array_equal(d._pos2, [])
#         np.testing.assert_array_equal(d._pos3, [])
#         np.testing.assert_array_equal(d._pos5, [])
#         np.testing.assert_array_equal(d._pos6, [])
#         np.testing.assert_array_equal(d._pos7, [])
#         np.testing.assert_array_equal(d._pos8, [])
#         np.testing.assert_array_equal(d._pos9, [])
#         np.testing.assert_array_equal(d._sensorx, [])
#         np.testing.assert_array_equal(d._sensory, [])
#         np.testing.assert_array_equal(d._sensorz, [])
#         self.assertEqual(d._data_unit, '')
#         self.assertIsNone(d._filename)
#         self.assertEqual(d.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(d._pos1_unit, 'mm')
#         self.assertEqual(d._pos2_unit, 'mm')
#         self.assertEqual(d._pos3_unit, 'mm')
#         self.assertEqual(d._pos5_unit, 'deg')
#         self.assertEqual(d._pos6_unit, 'mm')
#         self.assertEqual(d._pos7_unit, 'mm')
#         self.assertEqual(d._pos8_unit, 'deg')
#         self.assertEqual(d._pos9_unit, 'deg')
#         self.assertEqual(d.npts, 0)
#         self.assertIsNone(d.scan_axis)
#         self.assertIsNone(d.filename)
#
#     def test_copy(self):
#         d = measurement.Scan(self.filename)
#         dc = d.copy()
#         np.testing.assert_array_equal(d._pos1, self.pos1)
#         np.testing.assert_array_equal(d._pos2, self.pos2)
#         np.testing.assert_array_equal(d._pos3, self.pos3)
#         np.testing.assert_array_equal(d._pos5, self.pos5)
#         np.testing.assert_array_equal(d._pos6, self.pos6)
#         np.testing.assert_array_equal(d._pos7, self.pos7)
#         np.testing.assert_array_equal(d._pos8, self.pos8)
#         np.testing.assert_array_equal(d._pos9, self.pos9)
#         np.testing.assert_array_equal(d._sensorx, self.sensorx)
#         np.testing.assert_array_equal(d._sensory, self.sensory)
#         np.testing.assert_array_equal(d._sensorz, self.sensorz)
#         self.assertEqual(dc._data_unit, '')
#         self.assertEqual(dc.npts, 5)
#         self.assertEqual(dc.scan_axis, 1)
#         self.assertEqual(dc.filename, self.filename)
#         self.assertEqual(dc.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(dc._pos1_unit, 'mm')
#         self.assertEqual(dc._pos2_unit, 'mm')
#         self.assertEqual(dc._pos3_unit, 'mm')
#         self.assertEqual(dc._pos5_unit, 'deg')
#         self.assertEqual(dc._pos6_unit, 'mm')
#         self.assertEqual(dc._pos7_unit, 'mm')
#         self.assertEqual(dc._pos8_unit, 'deg')
#         self.assertEqual(dc._pos9_unit, 'deg')
#         self.assertTrue(isinstance(dc, measurement.Scan))
#
#         d._pos2 = 0
#         self.assertEqual(d._pos2, 0)
#         self.assertEqual(dc._pos2, 6)
#
#     def test_reverse(self):
#         d = measurement.Scan(self.filename)
#         d.reverse()
#         np.testing.assert_array_equal(d._pos1, self.pos1[::-1])
#         np.testing.assert_array_equal(d._pos2, self.pos2)
#         np.testing.assert_array_equal(d._pos3, self.pos3)
#         np.testing.assert_array_equal(d._pos5, self.pos5)
#         np.testing.assert_array_equal(d._pos6, self.pos6)
#         np.testing.assert_array_equal(d._pos7, self.pos7)
#         np.testing.assert_array_equal(d._pos8, self.pos8)
#         np.testing.assert_array_equal(d._pos9, self.pos9)
#         np.testing.assert_array_equal(d._sensorx, self.sensorx[::-1])
#         np.testing.assert_array_equal(d._sensory, self.sensory[::-1])
#         np.testing.assert_array_equal(d._sensorz, self.sensorz[::-1])
#         self.assertEqual(d._data_unit, '')
#         self.assertEqual(d.npts, 5)
#         self.assertEqual(d.scan_axis, 1)
#         self.assertEqual(d.filename, self.filename)
#         self.assertEqual(d.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(d._pos1_unit, 'mm')
#         self.assertEqual(d._pos2_unit, 'mm')
#         self.assertEqual(d._pos3_unit, 'mm')
#         self.assertEqual(d._pos5_unit, 'deg')
#         self.assertEqual(d._pos6_unit, 'mm')
#         self.assertEqual(d._pos7_unit, 'mm')
#         self.assertEqual(d._pos8_unit, 'deg')
#         self.assertEqual(d._pos9_unit, 'deg')
#
#     def test_save_file(self):
#         filename = 'data_tmp.txt'
#         filename = os.path.join(self.base_directory, filename)
#         dw = measurement.Scan()
#         dw._pos1 = utils.to_array(1)
#         dw._pos2 = utils.to_array(2)
#         dw._pos3 = utils.to_array(3)
#         dw._pos5 = utils.to_array(4)
#         dw._pos6 = utils.to_array(5)
#         dw._pos7 = utils.to_array(6)
#         dw._pos9 = utils.to_array([8, 9])
#         dw._sensorx = utils.to_array([1, 1])
#         self.assertIsNone(dw.filename)
#         dw.save_file(filename)
#         self.assertEqual(dw.filename, filename)
#
#         dr = measurement.Scan(filename)
#         np.testing.assert_array_equal(dr._pos1, [1])
#         np.testing.assert_array_equal(dr._pos2, [2])
#         np.testing.assert_array_equal(dr._pos3, [3])
#         np.testing.assert_array_equal(dr._pos5, [4])
#         np.testing.assert_array_equal(dr._pos6, [5])
#         np.testing.assert_array_equal(dr._pos7, [6])
#         np.testing.assert_array_equal(dr._pos8, [])
#         np.testing.assert_array_equal(dr._pos9, [8, 9])
#         np.testing.assert_array_equal(dr._sensorx, [1, 1])
#         np.testing.assert_array_equal(dr._sensory, [0, 0])
#         np.testing.assert_array_equal(dr._sensorz, [0, 0])
#         self.assertEqual(dr._data_unit, '')
#         self.assertEqual(dr.npts, 2)
#         self.assertEqual(dr.scan_axis, 9)
#         self.assertEqual(dr.filename, filename)
#         self.assertEqual(dr.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(dr._pos1_unit, 'mm')
#         self.assertEqual(dr._pos2_unit, 'mm')
#         self.assertEqual(dr._pos3_unit, 'mm')
#         self.assertEqual(dr._pos5_unit, 'deg')
#         self.assertEqual(dr._pos6_unit, 'mm')
#         self.assertEqual(dr._pos7_unit, 'mm')
#         self.assertEqual(dr._pos8_unit, 'deg')
#         self.assertEqual(dr._pos9_unit, 'deg')
#         os.remove(filename)
#
#     def test_extras(self):
#         filename = 'data_extras_tmp.txt'
#         filename = os.path.join(self.base_directory, filename)
#         d = measurement.Scan(self.filename)
#         d.save_file(filename, extras={'extra_name': 'extra_value'})
#
#         dr = measurement.Scan(filename)
#         self.assertEqual(dr.extra_name, 'extra_value')
#         os.remove(filename)
#
#     def test_scan_axis(self):
#         d = measurement.Scan()
#         self.assertIsNone(d.scan_axis)
#
#         d._pos7 = utils.to_array([1, 2])
#         self.assertEqual(d.scan_axis, 7)
#
#         d._pos2 = utils.to_array([3])
#         self.assertEqual(d.scan_axis, 7)
#
#         d._pos3 = utils.to_array([4, 5])
#         self.assertIsNone(d.scan_axis)
#
#         d._pos7 = utils.to_array([])
#         self.assertEqual(d.scan_axis, 3)
#
#     def test_npts(self):
#         d = measurement.Scan()
#         self.assertEqual(d.npts, 0)
#
#         d._pos1 = utils.to_array([1, 2])
#         self.assertEqual(d.npts, 0)
#
#         d._sensory = utils.to_array([3])
#         self.assertEqual(d.npts, 0)
#
#         d._sensorz = utils.to_array([4, 5])
#         self.assertEqual(d.npts, 0)
#
#         d._sensory = utils.to_array([])
#         self.assertEqual(d.npts, 2)
#
#
# class TestVoltageScan(TestCase):
#     """Test VoltageScan."""
#
#     def setUp(self):
#         """Set up."""
#         self.base_directory = os.path.dirname(os.path.abspath(__file__))
#         self.filename = os.path.join(
#             self.base_directory, 'voltage_scan.txt')
#
#         self.pos1 = [1, 2, 3, 4, 5]
#         self.pos2 = [6]
#         self.pos3 = [7]
#         self.pos5 = [8]
#         self.pos6 = [9]
#         self.pos7 = [10]
#         self.pos8 = [11]
#         self.pos9 = [12]
#         self.sensorx = [1.3, 1.4, 1.5, 1.6, 1.7]
#         self.sensory = [1.8, 1.9, 2.0, 2.1, 2.2]
#         self.sensorz = [2.3, 2.4, 2.5, 2.6, 2.7]
#
#     def tearDown(self):
#         """Tear down."""
#         pass
#
#     def test_initialization_without_filename(self):
#         d = measurement.VoltageScan()
#         np.testing.assert_array_equal(d.pos1, [])
#         np.testing.assert_array_equal(d.pos2, [])
#         np.testing.assert_array_equal(d.pos3, [])
#         np.testing.assert_array_equal(d.pos5, [])
#         np.testing.assert_array_equal(d.pos6, [])
#         np.testing.assert_array_equal(d.pos7, [])
#         np.testing.assert_array_equal(d.pos8, [])
#         np.testing.assert_array_equal(d.pos9, [])
#         np.testing.assert_array_equal(d.sensorx, [])
#         np.testing.assert_array_equal(d.sensory, [])
#         np.testing.assert_array_equal(d.sensorz, [])
#         self.assertEqual(d._data_unit, 'V')
#         self.assertIsNone(d._filename)
#         self.assertEqual(d.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(d._pos1_unit, 'mm')
#         self.assertEqual(d._pos2_unit, 'mm')
#         self.assertEqual(d._pos3_unit, 'mm')
#         self.assertEqual(d._pos5_unit, 'deg')
#         self.assertEqual(d._pos6_unit, 'mm')
#         self.assertEqual(d._pos7_unit, 'mm')
#         self.assertEqual(d._pos8_unit, 'deg')
#         self.assertEqual(d._pos9_unit, 'deg')
#         self.assertEqual(d.npts, 0)
#         self.assertIsNone(d.scan_axis)
#         self.assertIsNone(d.filename)
#
#     def test_initialization_with_filename(self):
#         d = measurement.VoltageScan(self.filename)
#         np.testing.assert_array_equal(d.pos1, self.pos1)
#         np.testing.assert_array_equal(d.pos2, self.pos2)
#         np.testing.assert_array_equal(d.pos3, self.pos3)
#         np.testing.assert_array_equal(d.pos5, self.pos5)
#         np.testing.assert_array_equal(d.pos6, self.pos6)
#         np.testing.assert_array_equal(d.pos7, self.pos7)
#         np.testing.assert_array_equal(d.pos8, self.pos8)
#         np.testing.assert_array_equal(d.pos9, self.pos9)
#         np.testing.assert_array_equal(d.sensorx, self.sensorx)
#         np.testing.assert_array_equal(d.sensory, self.sensory)
#         np.testing.assert_array_equal(d.sensorz, self.sensorz)
#         self.assertEqual(d._data_unit, 'V')
#         self.assertEqual(d.npts, 5)
#         self.assertEqual(d.scan_axis, 1)
#         self.assertEqual(d.filename, self.filename)
#         self.assertEqual(d.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(d._pos1_unit, 'mm')
#         self.assertEqual(d._pos2_unit, 'mm')
#         self.assertEqual(d._pos3_unit, 'mm')
#         self.assertEqual(d._pos5_unit, 'deg')
#         self.assertEqual(d._pos6_unit, 'mm')
#         self.assertEqual(d._pos7_unit, 'mm')
#         self.assertEqual(d._pos8_unit, 'deg')
#         self.assertEqual(d._pos9_unit, 'deg')
#
#     def test_copy(self):
#         d = measurement.VoltageScan(self.filename)
#         dc = d.copy()
#         np.testing.assert_array_equal(d.pos1, self.pos1)
#         np.testing.assert_array_equal(d.pos2, self.pos2)
#         np.testing.assert_array_equal(d.pos3, self.pos3)
#         np.testing.assert_array_equal(d.pos5, self.pos5)
#         np.testing.assert_array_equal(d.pos6, self.pos6)
#         np.testing.assert_array_equal(d.pos7, self.pos7)
#         np.testing.assert_array_equal(d.pos8, self.pos8)
#         np.testing.assert_array_equal(d.pos9, self.pos9)
#         np.testing.assert_array_equal(d.sensorx, self.sensorx)
#         np.testing.assert_array_equal(d.sensory, self.sensory)
#         np.testing.assert_array_equal(d.sensorz, self.sensorz)
#         self.assertEqual(dc._data_unit, 'V')
#         self.assertEqual(dc.npts, 5)
#         self.assertEqual(dc.scan_axis, 1)
#         self.assertEqual(dc.filename, self.filename)
#         self.assertEqual(dc.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(dc._pos1_unit, 'mm')
#         self.assertEqual(dc._pos2_unit, 'mm')
#         self.assertEqual(dc._pos3_unit, 'mm')
#         self.assertEqual(dc._pos5_unit, 'deg')
#         self.assertEqual(dc._pos6_unit, 'mm')
#         self.assertEqual(dc._pos7_unit, 'mm')
#         self.assertEqual(dc._pos8_unit, 'deg')
#         self.assertEqual(dc._pos9_unit, 'deg')
#         self.assertTrue(isinstance(dc, measurement.VoltageScan))
#
#         d.pos2 = 0
#         self.assertEqual(d.pos2, 0)
#         self.assertEqual(dc.pos2, 6)
#
#     def test_read_file(self):
#         d = measurement.VoltageScan()
#         d.read_file(self.filename)
#         np.testing.assert_array_equal(d.pos1, self.pos1)
#         np.testing.assert_array_equal(d.pos2, self.pos2)
#         np.testing.assert_array_equal(d.pos3, self.pos3)
#         np.testing.assert_array_equal(d.pos5, self.pos5)
#         np.testing.assert_array_equal(d.pos6, self.pos6)
#         np.testing.assert_array_equal(d.pos7, self.pos7)
#         np.testing.assert_array_equal(d.pos8, self.pos8)
#         np.testing.assert_array_equal(d.pos9, self.pos9)
#         np.testing.assert_array_equal(d.sensorx, self.sensorx)
#         np.testing.assert_array_equal(d.sensory, self.sensory)
#         np.testing.assert_array_equal(d.sensorz, self.sensorz)
#         self.assertEqual(d._data_unit, 'V')
#         self.assertEqual(d.npts, 5)
#         self.assertEqual(d.scan_axis, 1)
#         self.assertEqual(d.filename, self.filename)
#         self.assertEqual(d.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(d._pos1_unit, 'mm')
#         self.assertEqual(d._pos2_unit, 'mm')
#         self.assertEqual(d._pos3_unit, 'mm')
#         self.assertEqual(d._pos5_unit, 'deg')
#         self.assertEqual(d._pos6_unit, 'mm')
#         self.assertEqual(d._pos7_unit, 'mm')
#         self.assertEqual(d._pos8_unit, 'deg')
#         self.assertEqual(d._pos9_unit, 'deg')
#
#     def test_save_file(self):
#         filename = 'voltage_scan_tmp.txt'
#         filename = os.path.join(self.base_directory, filename)
#         dw = measurement.VoltageScan()
#         dw.pos1 = 1
#         dw.pos2 = 2
#         dw.pos3 = 3
#         dw.pos5 = 4
#         dw.pos6 = 5
#         dw.pos7 = 6
#         dw.pos9 = [8, 9]
#         dw.sensorx = [1, 1]
#         self.assertIsNone(dw.filename)
#         dw.save_file(filename)
#         self.assertEqual(dw.filename, filename)
#
#         dr = measurement.VoltageScan(filename)
#         np.testing.assert_array_equal(dr.pos1, [1])
#         np.testing.assert_array_equal(dr.pos2, [2])
#         np.testing.assert_array_equal(dr.pos3, [3])
#         np.testing.assert_array_equal(dr.pos5, [4])
#         np.testing.assert_array_equal(dr.pos6, [5])
#         np.testing.assert_array_equal(dr.pos7, [6])
#         np.testing.assert_array_equal(dr.pos8, [])
#         np.testing.assert_array_equal(dr.pos9, [8, 9])
#         np.testing.assert_array_equal(dr.sensorx, [1, 1])
#         np.testing.assert_array_equal(dr.sensory, [0, 0])
#         np.testing.assert_array_equal(dr.sensorz, [0, 0])
#         self.assertEqual(dr._data_unit, 'V')
#         self.assertEqual(dr.npts, 2)
#         self.assertEqual(dr.scan_axis, 9)
#         self.assertEqual(dr.filename, filename)
#         self.assertEqual(dr.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(dr._pos1_unit, 'mm')
#         self.assertEqual(dr._pos2_unit, 'mm')
#         self.assertEqual(dr._pos3_unit, 'mm')
#         self.assertEqual(dr._pos5_unit, 'deg')
#         self.assertEqual(dr._pos6_unit, 'mm')
#         self.assertEqual(dr._pos7_unit, 'mm')
#         self.assertEqual(dr._pos8_unit, 'deg')
#         self.assertEqual(dr._pos9_unit, 'deg')
#         os.remove(filename)
#
#
# class TestFieldScan(TestCase):
#     """Test FieldScan."""
#
#     def setUp(self):
#         """Set up."""
#         self.base_directory = os.path.dirname(os.path.abspath(__file__))
#         self.filename = os.path.join(self.base_directory, 'field_scan.txt')
#
#         self.voltage_scan_filename = os.path.join(
#             self.base_directory, 'voltage_scan.txt')
#         self.voltage_scan = measurement.VoltageScan(self.voltage_scan_filename)
#
#         self.hall_probe_filename = 'hall_probe_polynomial.txt'
#         self.hall_probe = calibration.HallProbe(
#             os.path.join(
#                 self.base_directory, self.hall_probe_filename))
#
#         self.field_sensorx = [0.26, 0.28, 0.3, 0.32, 0.34]
#         self.field_sensory = [0.36, 0.38, 0.4, 0.42, 0.44]
#         self.field_sensorz = [0.46, 0.48, 0.5, 0.52, 0.54]
#
#     def tearDown(self):
#         """Tear down."""
#         pass
#
#     def test_initialization_without_filename(self):
#         d = measurement.FieldScan()
#         np.testing.assert_array_equal(d.pos1, [])
#         np.testing.assert_array_equal(d.pos2, [])
#         np.testing.assert_array_equal(d.pos3, [])
#         np.testing.assert_array_equal(d.pos5, [])
#         np.testing.assert_array_equal(d.pos6, [])
#         np.testing.assert_array_equal(d.pos7, [])
#         np.testing.assert_array_equal(d.pos8, [])
#         np.testing.assert_array_equal(d.pos9, [])
#         np.testing.assert_array_equal(d.sensorx, [])
#         np.testing.assert_array_equal(d.sensory, [])
#         np.testing.assert_array_equal(d.sensorz, [])
#         self.assertEqual(d._data_unit, 'T')
#         self.assertIsNone(d._filename)
#         self.assertEqual(d.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(d._pos1_unit, 'mm')
#         self.assertEqual(d._pos2_unit, 'mm')
#         self.assertEqual(d._pos3_unit, 'mm')
#         self.assertEqual(d._pos5_unit, 'deg')
#         self.assertEqual(d._pos6_unit, 'mm')
#         self.assertEqual(d._pos7_unit, 'mm')
#         self.assertEqual(d._pos8_unit, 'deg')
#         self.assertEqual(d._pos9_unit, 'deg')
#         self.assertEqual(d.npts, 0)
#         self.assertIsNone(d.scan_axis)
#         self.assertIsNone(d.filename)
#         self.assertIsNone(d.hall_probe)
#         self.assertIsNone(d.voltage_scan_list)
#
#     def test_initialization_with_filename(self):
#         d = measurement.FieldScan(self.filename)
#         np.testing.assert_array_equal(d.pos1, [1, 2, 3, 4, 5])
#         np.testing.assert_array_equal(d.pos2, [6])
#         np.testing.assert_array_equal(d.pos3, [7])
#         np.testing.assert_array_equal(d.pos5, [8])
#         np.testing.assert_array_equal(d.pos6, [9])
#         np.testing.assert_array_equal(d.pos7, [10])
#         np.testing.assert_array_equal(d.pos8, [11])
#         np.testing.assert_array_equal(d.pos9, [12])
#         np.testing.assert_array_equal(d.sensorx, self.field_sensorx)
#         np.testing.assert_array_equal(d.sensory, self.field_sensory)
#         np.testing.assert_array_equal(d.sensorz, self.field_sensorz)
#         self.assertEqual(d._data_unit, 'T')
#         self.assertEqual(d.npts, 5)
#         self.assertEqual(d.scan_axis, 1)
#         self.assertEqual(d.filename, self.filename)
#         self.assertEqual(d.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(d._pos1_unit, 'mm')
#         self.assertEqual(d._pos2_unit, 'mm')
#         self.assertEqual(d._pos3_unit, 'mm')
#         self.assertEqual(d._pos5_unit, 'deg')
#         self.assertEqual(d._pos6_unit, 'mm')
#         self.assertEqual(d._pos7_unit, 'mm')
#         self.assertEqual(d._pos8_unit, 'deg')
#         self.assertEqual(d._pos9_unit, 'deg')
#         self.assertEqual(
#             os.path.split(d.hall_probe.filename)[1],
#             self.hall_probe_filename)
#         self.assertIsNone(d.voltage_scan_list)
#
#     def test_clear(self):
#         d = measurement.FieldScan(self.filename)
#         np.testing.assert_array_equal(d.pos1, [1, 2, 3, 4, 5])
#         np.testing.assert_array_equal(d.pos2, [6])
#         np.testing.assert_array_equal(d.pos3, [7])
#         np.testing.assert_array_equal(d.pos5, [8])
#         np.testing.assert_array_equal(d.pos6, [9])
#         np.testing.assert_array_equal(d.pos7, [10])
#         np.testing.assert_array_equal(d.pos8, [11])
#         np.testing.assert_array_equal(d.pos9, [12])
#         np.testing.assert_array_equal(d.sensorx, self.field_sensorx)
#         np.testing.assert_array_equal(d.sensory, self.field_sensory)
#         np.testing.assert_array_equal(d.sensorz, self.field_sensorz)
#         self.assertEqual(d._data_unit, 'T')
#         self.assertEqual(d.npts, 5)
#         self.assertEqual(d.scan_axis, 1)
#         self.assertEqual(d.filename, self.filename)
#         self.assertEqual(d.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(d._pos1_unit, 'mm')
#         self.assertEqual(d._pos2_unit, 'mm')
#         self.assertEqual(d._pos3_unit, 'mm')
#         self.assertEqual(d._pos5_unit, 'deg')
#         self.assertEqual(d._pos6_unit, 'mm')
#         self.assertEqual(d._pos7_unit, 'mm')
#         self.assertEqual(d._pos8_unit, 'deg')
#         self.assertEqual(d._pos9_unit, 'deg')
#         self.assertEqual(
#             os.path.split(d.hall_probe.filename)[1],
#             self.hall_probe_filename)
#         self.assertIsNone(d.voltage_scan_list)
#         d.clear()
#         np.testing.assert_array_equal(d.pos1, [])
#         np.testing.assert_array_equal(d.pos2, [])
#         np.testing.assert_array_equal(d.pos3, [])
#         np.testing.assert_array_equal(d.pos5, [])
#         np.testing.assert_array_equal(d.pos6, [])
#         np.testing.assert_array_equal(d.pos7, [])
#         np.testing.assert_array_equal(d.pos8, [])
#         np.testing.assert_array_equal(d.pos9, [])
#         np.testing.assert_array_equal(d.sensorx, [])
#         np.testing.assert_array_equal(d.sensory, [])
#         np.testing.assert_array_equal(d.sensorz, [])
#         self.assertEqual(d._data_unit, 'T')
#         self.assertIsNone(d._filename)
#         self.assertEqual(d.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(d._pos1_unit, 'mm')
#         self.assertEqual(d._pos2_unit, 'mm')
#         self.assertEqual(d._pos3_unit, 'mm')
#         self.assertEqual(d._pos5_unit, 'deg')
#         self.assertEqual(d._pos6_unit, 'mm')
#         self.assertEqual(d._pos7_unit, 'mm')
#         self.assertEqual(d._pos8_unit, 'deg')
#         self.assertEqual(d._pos9_unit, 'deg')
#         self.assertEqual(d.npts, 0)
#         self.assertIsNone(d.scan_axis)
#         self.assertIsNone(d.filename)
#         self.assertIsNone(d.hall_probe)
#         self.assertIsNone(d.voltage_scan_list)
#
#     def test_copy(self):
#         d = measurement.FieldScan(self.filename)
#         dc = d.copy()
#         np.testing.assert_array_equal(dc.pos1, [1, 2, 3, 4, 5])
#         np.testing.assert_array_equal(dc.pos2, [6])
#         np.testing.assert_array_equal(dc.pos3, [7])
#         np.testing.assert_array_equal(dc.pos5, [8])
#         np.testing.assert_array_equal(dc.pos6, [9])
#         np.testing.assert_array_equal(dc.pos7, [10])
#         np.testing.assert_array_equal(dc.pos8, [11])
#         np.testing.assert_array_equal(dc.pos9, [12])
#         np.testing.assert_array_equal(dc.sensorx, self.field_sensorx)
#         np.testing.assert_array_equal(dc.sensory, self.field_sensory)
#         np.testing.assert_array_equal(dc.sensorz, self.field_sensorz)
#         self.assertEqual(dc._data_unit, 'T')
#         self.assertEqual(dc.npts, 5)
#         self.assertEqual(dc.scan_axis, 1)
#         self.assertEqual(dc.filename, self.filename)
#         self.assertEqual(dc.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(dc._pos1_unit, 'mm')
#         self.assertEqual(dc._pos2_unit, 'mm')
#         self.assertEqual(dc._pos3_unit, 'mm')
#         self.assertEqual(dc._pos5_unit, 'deg')
#         self.assertEqual(dc._pos6_unit, 'mm')
#         self.assertEqual(dc._pos7_unit, 'mm')
#         self.assertEqual(dc._pos8_unit, 'deg')
#         self.assertEqual(dc._pos9_unit, 'deg')
#         self.assertTrue(isinstance(dc, measurement.FieldScan))
#
#         d.clear()
#         np.testing.assert_array_equal(d.pos2, [])
#         self.assertEqual(dc.pos2, 6)
#
#     def test_read_file(self):
#         d = measurement.FieldScan()
#         d.read_file(self.filename)
#         np.testing.assert_array_equal(d.pos1, [1, 2, 3, 4, 5])
#         np.testing.assert_array_equal(d.pos2, [6])
#         np.testing.assert_array_equal(d.pos3, [7])
#         np.testing.assert_array_equal(d.pos5, [8])
#         np.testing.assert_array_equal(d.pos6, [9])
#         np.testing.assert_array_equal(d.pos7, [10])
#         np.testing.assert_array_equal(d.pos8, [11])
#         np.testing.assert_array_equal(d.pos9, [12])
#         np.testing.assert_array_equal(d.sensorx, self.field_sensorx)
#         np.testing.assert_array_equal(d.sensory, self.field_sensory)
#         np.testing.assert_array_equal(d.sensorz, self.field_sensorz)
#         self.assertEqual(d._data_unit, 'T')
#         self.assertEqual(d.npts, 5)
#         self.assertEqual(d.scan_axis, 1)
#         self.assertEqual(d.filename, self.filename)
#         self.assertEqual(d.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(d._pos1_unit, 'mm')
#         self.assertEqual(d._pos2_unit, 'mm')
#         self.assertEqual(d._pos3_unit, 'mm')
#         self.assertEqual(d._pos5_unit, 'deg')
#         self.assertEqual(d._pos6_unit, 'mm')
#         self.assertEqual(d._pos7_unit, 'mm')
#         self.assertEqual(d._pos8_unit, 'deg')
#         self.assertEqual(d._pos9_unit, 'deg')
#
#     def test_save_file(self):
#         filename = 'field_scan_tmp.txt'
#         filename = os.path.join(self.base_directory, filename)
#         dw = measurement.FieldScan()
#         dw.hall_probe = self.hall_probe
#         dw.voltage_scan_list = self.voltage_scan
#         self.assertIsNone(dw.filename)
#         dw.save_file(filename)
#         self.assertEqual(dw.filename, filename)
#
#         dr = measurement.FieldScan(filename)
#         np.testing.assert_array_equal(dr.pos1, [1, 2, 3, 4, 5])
#         np.testing.assert_array_equal(dr.pos2, [6])
#         np.testing.assert_array_equal(dr.pos3, [7])
#         np.testing.assert_array_equal(dr.pos5, [8])
#         np.testing.assert_array_equal(dr.pos6, [9])
#         np.testing.assert_array_equal(dr.pos7, [10])
#         np.testing.assert_array_equal(dr.pos8, [11])
#         np.testing.assert_array_equal(dr.pos9, [12])
#         np.testing.assert_array_equal(dr.sensorx, self.field_sensorx)
#         np.testing.assert_array_equal(dr.sensory, self.field_sensory)
#         np.testing.assert_array_equal(dr.sensorz, self.field_sensorz)
#         self.assertEqual(dr._data_unit, 'T')
#         self.assertEqual(dr.npts, 5)
#         self.assertEqual(dr.scan_axis, 1)
#         self.assertEqual(dr.filename, filename)
#         self.assertEqual(dr.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(dr._pos1_unit, 'mm')
#         self.assertEqual(dr._pos2_unit, 'mm')
#         self.assertEqual(dr._pos3_unit, 'mm')
#         self.assertEqual(dr._pos5_unit, 'deg')
#         self.assertEqual(dr._pos6_unit, 'mm')
#         self.assertEqual(dr._pos7_unit, 'mm')
#         self.assertEqual(dr._pos8_unit, 'deg')
#         self.assertEqual(dr._pos9_unit, 'deg')
#         os.remove(filename)
#
#         dw = measurement.FieldScan()
#         dw.voltage_scan_list = self.voltage_scan
#         with self.assertRaises(measurement.MeasurementDataError):
#             dw.save_file(filename)
#
#         dw = measurement.FieldScan()
#         dw.hall_probe = self.hall_probe
#         with self.assertRaises(measurement.MeasurementDataError):
#             dw.save_file(filename)
#
#         fn_hall_probe = 'hall_probe_tmp.txt'
#         dw = measurement.FieldScan()
#         dw.voltage_scan_list = self.voltage_scan
#         dw.hall_probe = self.hall_probe
#         dw.hall_probe._filename = fn_hall_probe
#         dw.save_file(filename)
#         os.remove(filename)
#         os.remove(os.path.join(self.base_directory, fn_hall_probe))
#
#     def test_hall_probe(self):
#         d = measurement.FieldScan()
#         with self.assertRaises(TypeError):
#             d.hall_probe = None
#
#         with self.assertRaises(FileNotFoundError):
#             d.hall_probe = 'invalid_filename'
#
#         with self.assertRaises(FileNotFoundError):
#             d.hall_probe = self.hall_probe_filename
#
#         hall_probe_filepath = os.path.join(
#             self.base_directory, self.hall_probe_filename)
#         d.hall_probe = hall_probe_filepath
#         self.assertEqual(
#             d.hall_probe, self.hall_probe)
#
#         d.clear()
#         self.assertIsNone(d.hall_probe)
#
#         d.hall_probe = self.hall_probe
#         self.assertEqual(
#             d.hall_probe, self.hall_probe)
#
#     def test_voltage_scan_list(self):
#         d = measurement.FieldScan()
#
#         with self.assertRaises(measurement.MeasurementDataError):
#             d.voltage_scan_list = []
#
#         with self.assertRaises(measurement.MeasurementDataError):
#             d.voltage_scan_list = measurement.VoltageScan()
#
#         d.voltage_scan_list = self.voltage_scan
#         v = d.voltage_scan_list[0]
#         np.testing.assert_array_equal(v.pos1, [1, 2, 3, 4, 5])
#         np.testing.assert_array_equal(v.pos2, [6])
#         np.testing.assert_array_equal(v.pos3, [7])
#         np.testing.assert_array_equal(v.pos5, [8])
#         np.testing.assert_array_equal(v.pos6, [9])
#         np.testing.assert_array_equal(v.pos7, [10])
#         np.testing.assert_array_equal(v.pos8, [11])
#         np.testing.assert_array_equal(v.pos9, [12])
#         np.testing.assert_array_equal(v.sensorx, [1.3, 1.4, 1.5, 1.6, 1.7])
#         np.testing.assert_array_equal(v.sensory, [1.8, 1.9, 2.0, 2.1, 2.2])
#         np.testing.assert_array_equal(v.sensorz, [2.3, 2.4, 2.5, 2.6, 2.7])
#         self.assertEqual(v._data_unit, 'V')
#         self.assertEqual(v.npts, 5)
#         self.assertEqual(v.scan_axis, 1)
#         self.assertEqual(v.filename, self.voltage_scan_filename)
#         self.assertEqual(v.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(v._pos1_unit, 'mm')
#         self.assertEqual(v._pos2_unit, 'mm')
#         self.assertEqual(v._pos3_unit, 'mm')
#         self.assertEqual(v._pos5_unit, 'deg')
#         self.assertEqual(v._pos6_unit, 'mm')
#         self.assertEqual(v._pos7_unit, 'mm')
#         self.assertEqual(v._pos8_unit, 'deg')
#         self.assertEqual(v._pos9_unit, 'deg')
#         d.clear()
#
#         invalid_npts_voltage_scan = self.voltage_scan.copy()
#         invalid_npts_voltage_scan.sensorz = [1, 2]
#         with self.assertRaises(measurement.MeasurementDataError):
#             d.voltage_scan_list = invalid_npts_voltage_scan
#         d.clear()
#
#         v1 = self.voltage_scan.copy()
#         v2 = self.voltage_scan.copy()
#         v2.sensorx = [0, 0, 0, 0, 0]
#         v2.sensory = [0, 0, 0, 0, 0]
#         v2.sensorz = [0, 0, 0, 0, 0]
#         voltage_list = [v1, v2]
#         d.hall_probe = self.hall_probe
#         d.voltage_scan_list = voltage_list
#         np.testing.assert_array_equal(d.pos1, [1, 2, 3, 4, 5])
#         np.testing.assert_array_equal(d.pos2, [6])
#         np.testing.assert_array_equal(d.pos3, [7])
#         np.testing.assert_array_equal(d.pos5, [8])
#         np.testing.assert_array_equal(d.pos6, [9])
#         np.testing.assert_array_equal(d.pos7, [10])
#         np.testing.assert_array_equal(d.pos8, [11])
#         np.testing.assert_array_equal(d.pos9, [12])
#         np.testing.assert_array_almost_equal(
#             d.sensorx, np.array(self.field_sensorx)/2)
#         np.testing.assert_array_almost_equal(
#             d.sensory, np.array(self.field_sensory)/2)
#         np.testing.assert_array_almost_equal(
#             d.sensorz, np.array(self.field_sensorz)/2)
#         self.assertEqual(d._data_unit, 'T')
#         self.assertEqual(d.npts, 5)
#         self.assertEqual(d.scan_axis, 1)
#         self.assertEqual(d.axis_list, [1, 2, 3, 5, 6, 7, 8, 9])
#         self.assertEqual(d._pos1_unit, 'mm')
#         self.assertEqual(d._pos2_unit, 'mm')
#         self.assertEqual(d._pos3_unit, 'mm')
#         self.assertEqual(d._pos5_unit, 'deg')
#         self.assertEqual(d._pos6_unit, 'mm')
#         self.assertEqual(d._pos7_unit, 'mm')
#         self.assertEqual(d._pos8_unit, 'deg')
#         self.assertEqual(d._pos9_unit, 'deg')
#         self.assertIsNone(d.filename)
#         d.clear()
#
#         v1 = self.voltage_scan.copy()
#         v2 = self.voltage_scan.copy()
#         tmp_pos2 = v2.pos2
#         v2.pos2 = v2.pos1
#         v2.pos1 = tmp_pos2
#         inconsistent_voltage_list = [v1, v2]
#         with self.assertRaises(measurement.MeasurementDataError):
#             d.voltage_scan_list = inconsistent_voltage_list
#         d.clear()
#
#         v1 = self.voltage_scan.copy()
#         v2 = self.voltage_scan.copy()
#         v2.pos3 = 6
#         inconsistent_voltage_list = [v1, v2]
#         with self.assertRaises(measurement.MeasurementDataError):
#             d.voltage_scan_list = inconsistent_voltage_list
#         d.clear()
#
#         v1 = self.voltage_scan.copy()
#         v1.sensorx = v1.pos1
#         v1.sensory = v1.pos1
#         v1.sensorz = v1.pos1
#         v2 = self.voltage_scan.copy()
#         v2.pos1 = v1.pos1 + 2
#         v2.sensorx = v2.pos1 + 2
#         v2.sensory = v2.pos1 + 2
#         v2.sensorz = v2.pos1 + 2
#         voltage_list = [v1, v2]
#         p = (v1.pos1 + v2.pos1)/2
#
#         f1 = interpolate.splrep(v1.pos1, v1.sensorx, s=0, k=1)
#         f2 = interpolate.splrep(v2.pos1, v2.sensorx, s=0, k=1)
#         v1.sensorx = interpolate.splev(p, f1, der=0)
#         v2.sensorx = interpolate.splev(p, f2, der=0)
#         vx = (v1.sensorx + v2.sensorx)/2
#         fx = self.hall_probe.sensorx.get_field(vx)
#         d.hall_probe = self.hall_probe
#         d.voltage_scan_list = voltage_list
#         np.testing.assert_array_equal(d.pos1, p)
#         np.testing.assert_array_equal(d.pos2, [6])
#         np.testing.assert_array_equal(d.pos3, [7])
#         np.testing.assert_array_equal(d.pos5, [8])
#         np.testing.assert_array_equal(d.pos6, [9])
#         np.testing.assert_array_equal(d.pos7, [10])
#         np.testing.assert_array_equal(d.pos8, [11])
#         np.testing.assert_array_equal(d.pos9, [12])
#         np.testing.assert_array_almost_equal(
#             d.sensorx, fx)
#         np.testing.assert_array_almost_equal(
#             d.sensory, fx)
#         np.testing.assert_array_almost_equal(
#             d.sensorz, fx)
#
#
# class TestFieldmap(TestCase):
#     """Test Fieldmap."""
#
#     def setUp(self):
#         """Set up."""
#         self.base_directory = os.path.dirname(os.path.abspath(__file__))
#
#         self.voltage_scan_filename = os.path.join(
#             self.base_directory, 'voltage_scan.txt')
#         self.voltage_scan = measurement.VoltageScan(self.voltage_scan_filename)
#
#         self.field_scan_filename = os.path.join(
#             self.base_directory, 'field_scan.txt')
#         self.field_scan = measurement.FieldScan(self.field_scan_filename)
#
#         self.hall_probe_filename = 'hall_probe_polynomial.txt'
#         self.hall_probe = calibration.HallProbe(
#             os.path.join(
#                 self.base_directory, self.hall_probe_filename))
#
#         self.fn_fmd_ndcz = os.path.join(
#             self.base_directory, 'field_map_data_ndcz.txt')
#         self.fn_fmd_ndcy = os.path.join(
#             self.base_directory, 'field_map_data_ndcy.txt')
#         self.fn_fmd_ndcx = os.path.join(
#             self.base_directory, 'field_map_data_ndcx.txt')
#         self.fn_fmd_ndcxz = os.path.join(
#             self.base_directory, 'field_map_data_ndcxz.txt')
#         self.fn_fmd_ndcyz = os.path.join(
#             self.base_directory, 'field_map_data_ndcyz.txt')
#         self.fn_fmd_ndcxy = os.path.join(
#             self.base_directory, 'field_map_data_ndcxy.txt')
#
#     def tearDown(self):
#         """Tear down."""
#         pass
#
#     def test_initialization_without_filename(self):
#         d = measurement.Fieldmap()
#         np.testing.assert_array_equal(d.header_info, [])
#         np.testing.assert_array_equal(d.pos1, [])
#         np.testing.assert_array_equal(d.pos2, [])
#         np.testing.assert_array_equal(d.pos3, [])
#         self.assertIsNone(d.voltage_scan_list)
#         self.assertIsNone(d.field_scan_list)
#         self.assertIsNone(d.hall_probe)
#         self.assertIsNone(d.index_axis)
#         self.assertIsNone(d.columns_axis)
#         self.assertIsNone(d.field1)
#         self.assertIsNone(d.field2)
#         self.assertIsNone(d.field3)
#         self.assertIsNone(d.filename)
#         self.assertFalse(d._data_is_set)
#         self.assertTrue(d.correct_sensor_displacement)
#
#     def test_initialization_with_filename(self):
#         pos_scan = [1, 2, 3, 4, 5]
#         fixed_pos_a = [6]
#         fixed_pos_b = [7]
#         field_sensorx = [0.26, 0.28, 0.3, 0.32, 0.34]
#         field_sensory = [0.36, 0.38, 0.4, 0.42, 0.44]
#         field_sensorz = [0.46, 0.48, 0.5, 0.52, 0.54]
#
#         d = measurement.Fieldmap(self.fn_fmd_ndcz)
#         np.testing.assert_array_equal(d.pos1, pos_scan)
#         np.testing.assert_array_equal(d.pos2, fixed_pos_a)
#         np.testing.assert_array_equal(d.pos3, fixed_pos_b)
#         np.testing.assert_array_almost_equal(
#             d.field1.values, np.transpose([field_sensorz]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             d.field2.values, np.transpose([field_sensory]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             d.field3.values, np.transpose([field_sensorx]), decimal=3)
#
#         d = measurement.Fieldmap(self.fn_fmd_ndcy)
#         np.testing.assert_array_equal(d.pos1, fixed_pos_a)
#         np.testing.assert_array_equal(d.pos2, pos_scan)
#         np.testing.assert_array_equal(d.pos3, fixed_pos_b)
#         np.testing.assert_array_almost_equal(
#             d.field1.values, np.transpose([field_sensorz]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             d.field2.values, np.transpose([field_sensory]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             d.field3.values, np.transpose([field_sensorx]), decimal=3)
#
#         d = measurement.Fieldmap(self.fn_fmd_ndcx)
#         np.testing.assert_array_equal(d.pos1, fixed_pos_b)
#         np.testing.assert_array_equal(d.pos2, fixed_pos_a)
#         np.testing.assert_array_equal(d.pos3, pos_scan)
#         np.testing.assert_array_almost_equal(
#             d.field1.values, np.transpose([field_sensorz]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             d.field2.values, np.transpose([field_sensory]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             d.field3.values, np.transpose([field_sensorx]), decimal=3)
#
#         field_a1 = [0.26, 0.28, 0.3, 0.32, 0.34]
#         field_a2 = [0.026, 0.028, 0.03, 0.032, 0.034]
#         field_b1 = [0.36, 0.38, 0.4, 0.42, 0.44]
#         field_b2 = [0.036, 0.038, 0.04, 0.042, 0.044]
#         field_c1 = [0.46, 0.48, 0.5, 0.52, 0.54]
#         field_c2 = [0.046, 0.048, 0.05, 0.052, 0.054]
#
#         pos_scan = [1, 2, 3, 4, 5]
#         pos_var = [7, 8]
#         pos_fixed = [6]
#         d = measurement.Fieldmap(self.fn_fmd_ndcxz)
#         field3 = np.array([field_a1, field_a2])
#         field2 = np.array([field_b1, field_b2])
#         field1 = np.array([field_c1, field_c2])
#         np.testing.assert_array_equal(d.pos1, pos_scan)
#         np.testing.assert_array_equal(d.pos2, pos_fixed)
#         np.testing.assert_array_equal(d.pos3, pos_var)
#         np.testing.assert_array_almost_equal(
#             d.field1.values, field1, decimal=3)
#         np.testing.assert_array_almost_equal(
#             d.field2.values, field2, decimal=3)
#         np.testing.assert_array_almost_equal(
#             d.field3.values, field3, decimal=3)
#
#         pos_scan = [1, 2, 3, 4, 5]
#         pos_var = [6, 7]
#         pos_fixed = [7]
#         d = measurement.Fieldmap(self.fn_fmd_ndcyz)
#         field3 = np.array([field_a1, field_a2])
#         field2 = np.array([field_b1, field_b2])
#         field1 = np.array([field_c1, field_c2])
#         np.testing.assert_array_equal(d.pos1, pos_scan)
#         np.testing.assert_array_equal(d.pos2, pos_var)
#         np.testing.assert_array_equal(d.pos3, pos_fixed)
#         np.testing.assert_array_almost_equal(
#             d.field1.values, field1, decimal=3)
#         np.testing.assert_array_almost_equal(
#             d.field2.values, field2, decimal=3)
#         np.testing.assert_array_almost_equal(
#             d.field3.values, field3, decimal=3)
#
#         pos_scan = [1, 2, 3, 4, 5]
#         pos_var = [6, 7]
#         pos_fixed = [7]
#         d = measurement.Fieldmap(self.fn_fmd_ndcxy)
#         field3 = np.array([field_a1, field_a2])
#         field2 = np.array([field_b1, field_b2])
#         field1 = np.array([field_c1, field_c2])
#         np.testing.assert_array_equal(d.pos1, pos_fixed)
#         np.testing.assert_array_equal(d.pos2, pos_var)
#         np.testing.assert_array_equal(d.pos3, pos_scan)
#         np.testing.assert_array_almost_equal(
#             d.field1.values, np.transpose(field1), decimal=3)
#         np.testing.assert_array_almost_equal(
#             d.field2.values, np.transpose(field2), decimal=3)
#         np.testing.assert_array_almost_equal(
#             d.field3.values, np.transpose(field3), decimal=3)
#
#     def test_read_and_save_file(self):
#         filename = os.path.join(self.base_directory, 'fmd_tmp.txt')
#
#         d = measurement.Fieldmap()
#         d.read_file(self.fn_fmd_ndcz)
#         d.save_file(filename)
#         self.assertTrue(filecmp.cmp(self.fn_fmd_ndcz, filename))
#         os.remove(filename)
#         d.clear()
#
#         d.read_file(self.fn_fmd_ndcy)
#         d.save_file(filename)
#         self.assertTrue(filecmp.cmp(self.fn_fmd_ndcy, filename))
#         os.remove(filename)
#         d.clear()
#
#         d.read_file(self.fn_fmd_ndcx)
#         d.save_file(filename)
#         self.assertTrue(filecmp.cmp(self.fn_fmd_ndcx, filename))
#         os.remove(filename)
#         d.clear()
#
#         d.read_file(self.fn_fmd_ndcxy)
#         d.save_file(filename)
#         self.assertTrue(filecmp.cmp(self.fn_fmd_ndcxy, filename))
#         os.remove(filename)
#         d.clear()
#
#         d.read_file(self.fn_fmd_ndcyz)
#         d.save_file(filename)
#         self.assertTrue(filecmp.cmp(self.fn_fmd_ndcyz, filename))
#         os.remove(filename)
#         d.clear()
#
#         d.read_file(self.fn_fmd_ndcxz)
#         d.save_file(filename)
#         self.assertTrue(filecmp.cmp(self.fn_fmd_ndcxz, filename))
#         os.remove(filename)
#         d.clear()
#
#     def test_clear(self):
#         pos_scan = [1, 2, 3, 4, 5]
#         fixed_pos_a = [6]
#         fixed_pos_b = [7]
#         field_sensorx = [0.26, 0.28, 0.3, 0.32, 0.34]
#         field_sensory = [0.36, 0.38, 0.4, 0.42, 0.44]
#         field_sensorz = [0.46, 0.48, 0.5, 0.52, 0.54]
#
#         d = measurement.Fieldmap(self.fn_fmd_ndcz)
#         np.testing.assert_array_equal(d.pos1, pos_scan)
#         np.testing.assert_array_equal(d.pos2, fixed_pos_a)
#         np.testing.assert_array_equal(d.pos3, fixed_pos_b)
#         np.testing.assert_array_almost_equal(
#             d.field1.values, np.transpose([field_sensorz]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             d.field2.values, np.transpose([field_sensory]), decimal=3)
#         np.testing.assert_array_almost_equal(
#             d.field3.values, np.transpose([field_sensorx]), decimal=3)
#         d.clear()
#         np.testing.assert_array_equal(d.header_info, [])
#         np.testing.assert_array_equal(d.pos1, [])
#         np.testing.assert_array_equal(d.pos2, [])
#         np.testing.assert_array_equal(d.pos3, [])
#         self.assertIsNone(d.voltage_scan_list)
#         self.assertIsNone(d.field_scan_list)
#         self.assertIsNone(d.hall_probe)
#         self.assertIsNone(d.index_axis)
#         self.assertIsNone(d.columns_axis)
#         self.assertIsNone(d.field1)
#         self.assertIsNone(d.field2)
#         self.assertIsNone(d.field3)
#         self.assertIsNone(d.filename)
#         self.assertFalse(d._data_is_set)
#         self.assertTrue(d.correct_sensor_displacement)
#
#     def test_get_axis_position_dict(self):
#         fmd = measurement.Fieldmap()
#         fmd.hall_probe = self.hall_probe
#
#         d = fmd._get_axis_position_dict()
#         self.assertEqual(len(d), 0)
#
#         vd1 = self.voltage_scan.copy()
#         fmd.voltage_scan_list = vd1
#         d = fmd._get_axis_position_dict()
#         np.testing.assert_array_equal(d[1], [1, 2, 3, 4, 5])
#         np.testing.assert_array_equal(d[2], [6])
#         np.testing.assert_array_equal(d[3], [7])
#         np.testing.assert_array_equal(d[5], [8])
#         np.testing.assert_array_equal(d[6], [9])
#         np.testing.assert_array_equal(d[7], [10])
#         np.testing.assert_array_equal(d[8], [11])
#         np.testing.assert_array_equal(d[9], [12])
#
#         vd2 = self.voltage_scan.copy()
#         vd2.pos1 = [5, 6]
#         vd2.pos2 = [6]
#         vd2.pos3 = [8]
#         vd2.sensorx = vd2.sensorx[:2]
#         vd2.sensory = vd2.sensory[:2]
#         vd2.sensorz = vd2.sensorz[:2]
#
#         vd3 = self.voltage_scan.copy()
#         vd3.pos1 = [5, 7]
#         vd3.pos2 = [6]
#         vd3.pos3 = [6]
#         vd3.sensorx = vd3.sensorx[:2]
#         vd3.sensory = vd3.sensory[:2]
#         vd3.sensorz = vd3.sensorz[:2]
#
#         fmd = measurement.Fieldmap()
#         fmd.hall_probe = self.hall_probe
#         fmd.correct_sensor_displacement = False
#         fmd.voltage_scan_list = [vd1, vd2, vd3]
#         d = fmd._get_axis_position_dict()
#         np.testing.assert_array_equal(d[1], [1, 2, 3, 4, 5, 6, 7])
#         np.testing.assert_array_equal(d[2], [6])
#         np.testing.assert_array_equal(d[3], [6, 7, 8])
#         np.testing.assert_array_equal(d[5], [8])
#         np.testing.assert_array_equal(d[6], [9])
#         np.testing.assert_array_equal(d[7], [10])
#         np.testing.assert_array_equal(d[8], [11])
#         np.testing.assert_array_equal(d[9], [12])
#
#     def test_header_info(self):
#         fmd = measurement.Fieldmap()
#         fmd.header_info = []
#         self.assertEqual(fmd.header_info, [])
#
#         with self.assertRaises(measurement.MeasurementDataError):
#             fmd.header_info = [1]
#
#         with self.assertRaises(measurement.MeasurementDataError):
#             fmd.header_info = [[1, 2], [3]]
#
#         fmd.header_info = [[1, 2], (3, 4)]
#         self.assertEqual(fmd.header_info[0], (1, 2))
#         self.assertEqual(fmd.header_info[1], (3, 4))
#
#     def test_hall_probe(self):
#         fmd = measurement.Fieldmap()
#         with self.assertRaises(TypeError):
#             fmd.hall_probe = None
#
#         with self.assertRaises(FileNotFoundError):
#             fmd.hall_probe = 'invalid_filename'
#
#         with self.assertRaises(FileNotFoundError):
#             fmd.hall_probe = self.hall_probe_filename
#
#         hall_probe_filepath = os.path.join(
#             self.base_directory, self.hall_probe_filename)
#         fmd.hall_probe = hall_probe_filepath
#         self.assertEqual(
#             fmd.hall_probe, self.hall_probe)
#
#         fmd.clear()
#         self.assertIsNone(fmd.hall_probe)
#
#         fmd.hall_probe = self.hall_probe
#         self.assertEqual(
#             fmd.hall_probe, self.hall_probe)
#
#     def test_voltage_scan_list(self):
#         d = measurement.Fieldmap()
#         d.correct_sensor_displacement = False
#
#         with self.assertRaises(measurement.MeasurementDataError):
#             d.voltage_scan_list = []
#
#         with self.assertRaises(measurement.MeasurementDataError):
#             d.voltage_scan_list = measurement.VoltageScan()
#
#         d.voltage_scan_list = self.voltage_scan
#         v = d.voltage_scan_list[0]
#         np.testing.assert_array_equal(v.pos1, [1, 2, 3, 4, 5])
#         np.testing.assert_array_equal(v.pos2, [6])
#         np.testing.assert_array_equal(v.pos3, [7])
#         np.testing.assert_array_equal(v.pos5, [8])
#         np.testing.assert_array_equal(v.pos6, [9])
#         np.testing.assert_array_equal(v.pos7, [10])
#         np.testing.assert_array_equal(v.pos8, [11])
#         np.testing.assert_array_equal(v.pos9, [12])
#         np.testing.assert_array_equal(v.sensorx, [1.3, 1.4, 1.5, 1.6, 1.7])
#         np.testing.assert_array_equal(v.sensory, [1.8, 1.9, 2.0, 2.1, 2.2])
#         np.testing.assert_array_equal(v.sensorz, [2.3, 2.4, 2.5, 2.6, 2.7])
#
#         d.clear()
#         d.correct_sensor_displacement = False
#         invalid_npts_voltage_scan = self.voltage_scan.copy()
#         invalid_npts_voltage_scan.sensorz = [1, 2]
#         with self.assertRaises(measurement.MeasurementDataError):
#             d.voltage_scan_list = invalid_npts_voltage_scan
#
#         d.clear()
#         d.correct_sensor_displacement = False
#         d.hall_probe = self.hall_probe
#         field_sensorx = [0.26, 0.28, 0.3, 0.32, 0.34]
#         field_sensory = [0.36, 0.38, 0.4, 0.42, 0.44]
#         field_sensorz = [0.46, 0.48, 0.5, 0.52, 0.54]
#         v1 = self.voltage_scan.copy()
#         v2 = self.voltage_scan.copy()
#         v2.sensorx = [0, 0, 0, 0, 0]
#         v2.sensory = [0, 0, 0, 0, 0]
#         v2.sensorz = [0, 0, 0, 0, 0]
#         voltage_list = [v1, v2]
#         d.voltage_scan_list = voltage_list
#         np.testing.assert_array_equal(d.pos1, [1, 2, 3, 4, 5])
#         np.testing.assert_array_equal(d.pos2, [6])
#         np.testing.assert_array_equal(d.pos3, [7])
#         np.testing.assert_array_almost_equal(
#             d.field3.values, (np.array(field_sensorx)/2).reshape(5, -1))
#         np.testing.assert_array_almost_equal(
#             d.field2.values, (np.array(field_sensory)/2).reshape(5, -1))
#         np.testing.assert_array_almost_equal(
#             d.field1.values, (np.array(field_sensorz)/2).reshape(5, -1))
#         self.assertIsNone(d.filename)
#         self.assertEqual(len(d.field_scan_list), 1)
#
#         d.clear()
#         d.correct_sensor_displacement = False
#         d.hall_probe = self.hall_probe
#         v1 = self.voltage_scan.copy()
#         v2 = self.voltage_scan.copy()
#         tmp_pos2 = v2.pos2
#         v2.pos2 = v2.pos1
#         v2.pos1 = tmp_pos2
#         inconsistent_voltage_list = [v1, v2]
#         with self.assertRaises(measurement.MeasurementDataError):
#             d.voltage_scan_list = inconsistent_voltage_list
#
#         d.clear()
#         d.correct_sensor_displacement = False
#         d.hall_probe = self.hall_probe
#         v1 = self.voltage_scan.copy()
#         v1.sensorx = v1.pos1
#         v1.sensory = v1.pos1
#         v1.sensorz = v1.pos1
#         v2 = self.voltage_scan.copy()
#         v2.pos1 = v1.pos1 + 2
#         v2.sensorx = v2.pos1 + 2
#         v2.sensory = v2.pos1 + 2
#         v2.sensorz = v2.pos1 + 2
#         voltage_list = [v1, v2]
#         p = (v1.pos1 + v2.pos1)/2
#
#         f1 = interpolate.splrep(v1.pos1, v1.sensorx, s=0, k=1)
#         f2 = interpolate.splrep(v2.pos1, v2.sensorx, s=0, k=1)
#         v1.sensorx = interpolate.splev(p, f1, der=0)
#         v2.sensorx = interpolate.splev(p, f2, der=0)
#         vx = (v1.sensorx + v2.sensorx)/2
#         fx = self.hall_probe.sensorx.get_field(vx).reshape(5, -1)
#         d.hall_probe = self.hall_probe
#         d.voltage_scan_list = voltage_list
#         np.testing.assert_array_equal(d.pos1, p)
#         np.testing.assert_array_equal(d.pos2, [6])
#         np.testing.assert_array_equal(d.pos3, [7])
#         np.testing.assert_array_almost_equal(
#             d.field3.values, fx)
#         np.testing.assert_array_almost_equal(
#             d.field2.values, fx)
#         np.testing.assert_array_almost_equal(
#             d.field1.values, fx)
#         self.assertEqual(len(d.field_scan_list), 1)
#
#         d.clear()
#         d.correct_sensor_displacement = False
#         d.hall_probe = self.hall_probe
#         vd = self.voltage_scan.copy()
#         d.voltage_scan_list = [vd, vd, vd, vd]
#         self.assertEqual(len(d.field_scan_list), 1)
#
#         d.clear()
#         d.correct_sensor_displacement = False
#         d.hall_probe = self.hall_probe
#         fd = self.field_scan.copy()
#         with self.assertRaises(TypeError):
#             d.voltage_scan_list = [fd]
#
#     def test_field_scan_list(self):
#         d = measurement.Fieldmap()
#         d.correct_sensor_displacement = False
#
#         with self.assertRaises(measurement.MeasurementDataError):
#             d.field_scan_list = []
#
#         with self.assertRaises(measurement.MeasurementDataError):
#             d.field_scan_list = measurement.FieldScan()
#
#         field_sensorx = [0.26, 0.28, 0.3, 0.32, 0.34]
#         field_sensory = [0.36, 0.38, 0.4, 0.42, 0.44]
#         field_sensorz = [0.46, 0.48, 0.5, 0.52, 0.54]
#
#         d.field_scan_list = self.field_scan
#         fd = d.field_scan_list[0]
#         np.testing.assert_array_equal(fd.pos1, [1, 2, 3, 4, 5])
#         np.testing.assert_array_equal(fd.pos2, [6])
#         np.testing.assert_array_equal(fd.pos3, [7])
#         np.testing.assert_array_equal(fd.pos5, [8])
#         np.testing.assert_array_equal(fd.pos6, [9])
#         np.testing.assert_array_equal(fd.pos7, [10])
#         np.testing.assert_array_equal(fd.pos8, [11])
#         np.testing.assert_array_equal(fd.pos9, [12])
#         np.testing.assert_array_equal(fd.sensorx, field_sensorx)
#         np.testing.assert_array_equal(fd.sensory, field_sensory)
#         np.testing.assert_array_equal(fd.sensorz, field_sensorz)
#
#         d.clear()
#         d.correct_sensor_displacement = False
#         d.hall_probe = self.hall_probe
#
#         fda = self.field_scan.copy()
#         fdb = self.field_scan.copy()
#         with self.assertRaises(measurement.MeasurementDataError):
#             d.field_scan_list = [fda, fdb]
#
#         d.clear()
#         d.correct_sensor_displacement = False
#         d.hall_probe = self.hall_probe
#         va = self.voltage_scan.copy()
#         vb = self.voltage_scan.copy()
#         tmp_pos2 = vb.pos2
#         vb.pos2 = vb.pos1
#         vb.pos1 = tmp_pos2
#         fda = self.field_scan.copy()
#         fda.voltage_scan_list = va
#         fdb = self.field_scan.copy()
#         fdb.voltage_scan_list = vb
#         inconsistent_field_list = [fda, fdb]
#         with self.assertRaises(measurement.MeasurementDataError):
#             d.field_scan_list = inconsistent_field_list
#
#         d.clear()
#         d.correct_sensor_displacement = False
#         d.hall_probe = self.hall_probe
#         vd = self.voltage_scan.copy()
#         with self.assertRaises(TypeError):
#             d.field_scan_list = [vd]
#
#     def test_axes(self):
#         fmd = measurement.Fieldmap()
#         self.assertIsNone(fmd.index_axis)
#         self.assertIsNone(fmd.columns_axis)
#
#         vd1 = self.voltage_scan
#         fmd.hall_probe = self.hall_probe
#         fmd.voltage_scan_list = vd1
#         self.assertEqual(fmd.index_axis, 1)
#         self.assertIsNone(fmd.columns_axis)
#
#         vd2 = vd1.copy()
#         vd2.pos2 = vd2.pos2 + 1
#         fmd.voltage_scan_list = [vd1, vd2]
#         self.assertEqual(fmd.index_axis, 1)
#         self.assertEqual(fmd.columns_axis, 2)
#
#         vd3 = vd1.copy()
#         vd3.pos3 = vd3.pos3 + 1
#         with self.assertRaises(measurement.MeasurementDataError):
#             fmd.voltage_scan_list = [vd1, vd2, vd3]
#
#     def test_sensor_displacement_correction(self):
#         pass  # AQUI!!!!!
#
#     def test_get_field_at_point(self):
#         vda = self.voltage_scan.copy()
#         vdb = self.voltage_scan.copy()
#         vdb.pos2 = 8
#
#         fmd = measurement.Fieldmap()
#         fmd.hall_probe = self.hall_probe
#         fmd.correct_sensor_displacement = False
#         fmd.voltage_scan_list = [vda, vdb]
#
#         np.testing.assert_array_almost_equal(
#             fmd._get_field_at_point([7, 6, 1]), [0.26,  0.36,  0.46])
#         np.testing.assert_array_almost_equal(
#             fmd._get_field_at_point([7, 8, 3]), [0.3, 0.4, 0.5])
#         np.testing.assert_array_almost_equal(
#             fmd._get_field_at_point([0, 8, 3]), [np.nan, np.nan, np.nan])
#
#     def test_get_field_map(self):
#         vda = self.voltage_scan.copy()
#         vdb = self.voltage_scan.copy()
#         vdb.pos2 = vdb.pos2 + 1
#
#         fmd = measurement.Fieldmap()
#         fmd.hall_probe = self.hall_probe
#         fmd.correct_sensor_displacement = False
#         fmd.voltage_scan_list = [vda, vdb]
#         fieldmap = fmd.get_field_map()
#
#         npts = 2*len(vda.pos1)
#         x = [vda.pos3[0]]*npts
#         y = [vda.pos2[0], vdb.pos2[0]]*len(vda.pos1)
#         z = np.transpose([vda.pos1, vda.pos1]).reshape(npts)
#         fieldx = [0.26, 0.28, 0.3, 0.32, 0.34]
#         fieldy = [0.36, 0.38, 0.4, 0.42, 0.44]
#         fieldz = [0.46, 0.48, 0.5, 0.52, 0.54]
#         bx = np.transpose([fieldx, fieldx]).reshape(npts)
#         by = np.transpose([fieldy, fieldy]).reshape(npts)
#         bz = np.transpose([fieldz, fieldz]).reshape(npts)
#
#         np.testing.assert_array_almost_equal(fieldmap[:, 0], x)
#         np.testing.assert_array_almost_equal(fieldmap[:, 1], y)
#         np.testing.assert_array_almost_equal(fieldmap[:, 2], z)
#         np.testing.assert_array_almost_equal(fieldmap[:, 3], bx)
#         np.testing.assert_array_almost_equal(fieldmap[:, 4], by)
#         np.testing.assert_array_almost_equal(fieldmap[:, 5], bz)
#
#     def test_get_transformed_fieldmap(self):
#         vda = self.voltage_scan.copy()
#         vdb = self.voltage_scan.copy()
#         vdb.pos2 = vdb.pos2 + 1
#
#         fmd = measurement.Fieldmap()
#         fmd.hall_probe = self.hall_probe
#         fmd.correct_sensor_displacement = False
#         fmd.voltage_scan_list = [vda, vdb]
#
#         fieldmap = fmd.get_transformed_field_map(
#             magnet_center=[0, 0, 0], magnet_x_axis=3, magnet_y_axis=2)
#
#         npts = 2*len(vda.pos1)
#         hn = len(vda.pos1)
#         vx = np.array([[vda.pos3[0]]*hn, [vdb.pos3[0]]*hn])
#         vy = np.array([[vda.pos2[0]]*hn, [vdb.pos2[0]]*hn])
#         vz = np.array([vda.pos1, vda.pos1])
#         fieldx = np.array(
#             [[0.26, 0.28, 0.3, 0.32, 0.34], [0.26, 0.28, 0.3, 0.32, 0.34]])
#         fieldy = np.array(
#             [[0.36, 0.38, 0.4, 0.42, 0.44], [0.36, 0.38, 0.4, 0.42, 0.44]])
#         fieldz = np.array(
#             [[0.46, 0.48, 0.5, 0.52, 0.54], [0.46, 0.48, 0.5, 0.52, 0.54]])
#
#         x = vx.reshape(npts)
#         y = np.transpose(vy).reshape(npts)
#         z = np.transpose(vz).reshape(npts)
#         bx = np.transpose(fieldx).reshape(npts)
#         by = np.transpose(fieldy).reshape(npts)
#         bz = np.transpose(fieldz).reshape(npts)
#
#         np.testing.assert_array_almost_equal(fieldmap[:, 0], x)
#         np.testing.assert_array_almost_equal(fieldmap[:, 1], y)
#         np.testing.assert_array_almost_equal(fieldmap[:, 2], z)
#         np.testing.assert_array_almost_equal(fieldmap[:, 3], bx)
#         np.testing.assert_array_almost_equal(fieldmap[:, 4], by)
#         np.testing.assert_array_almost_equal(fieldmap[:, 5], bz)
#
#         fieldmap = fmd.get_transformed_field_map(
#             magnet_center=[10, -40, 30])
#
#         np.testing.assert_array_almost_equal(fieldmap[:, 0], x - 10)
#         np.testing.assert_array_almost_equal(fieldmap[:, 1], y + 40)
#         np.testing.assert_array_almost_equal(fieldmap[:, 2], z - 30)
#         np.testing.assert_array_almost_equal(fieldmap[:, 3], bx)
#         np.testing.assert_array_almost_equal(fieldmap[:, 4], by)
#         np.testing.assert_array_almost_equal(fieldmap[:, 5], bz)
#
#         fieldmap = fmd.get_transformed_field_map(
#             magnet_x_axis=-1, magnet_y_axis=2)
#
#         x = np.transpose(vx).reshape(npts)
#         y = vy.reshape(npts)
#         z = vz.reshape(npts)
#         bx = fieldx.reshape(npts)
#         by = fieldy.reshape(npts)
#         bz = fieldz.reshape(npts)
#         np.testing.assert_array_almost_equal(fieldmap[:, 0], (-1)*z[::-1])
#         np.testing.assert_array_almost_equal(fieldmap[:, 1], y)
#         np.testing.assert_array_almost_equal(fieldmap[:, 2], x)
#         np.testing.assert_array_almost_equal(fieldmap[:, 3], (-1)*bz[::-1])
#         np.testing.assert_array_almost_equal(fieldmap[:, 4], by[::-1])
#         np.testing.assert_array_almost_equal(fieldmap[:, 5], bx[::-1])
#
#         fieldmap = fmd.get_transformed_field_map(
#             magnet_center=[10, -40, 30], magnet_x_axis=-1, magnet_y_axis=2)
#
#         np.testing.assert_array_almost_equal(fieldmap[:, 0], (-1)*z[::-1] + 30)
#         np.testing.assert_array_almost_equal(fieldmap[:, 1], y + 40)
#         np.testing.assert_array_almost_equal(fieldmap[:, 2], x - 10)
#         np.testing.assert_array_almost_equal(fieldmap[:, 3], (-1)*bz[::-1])
#         np.testing.assert_array_almost_equal(fieldmap[:, 4], by[::-1])
#         np.testing.assert_array_almost_equal(fieldmap[:, 5], bx[::-1])
