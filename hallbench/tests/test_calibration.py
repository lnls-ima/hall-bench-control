"""Calibration test."""

import os
import numpy as np
from unittest import TestCase
from hallbench.data import calibration


def make_calibration_curve_poly_file(filename):
    sc = calibration.CalibrationCurve()
    sc.function_type = 'polynomial'
    sc.data = [
        [-1000, -10, 1.8216, 7.0592e-01, 4.7964e-02, 1.5304e-03],
        [-10, 10, 0, 0.2, 0, 0],
        [10, 1000, -2.3614,	8.2643e-01, -5.6814e-02, 1.7429000e-03],
    ]
    try:
        sc.save_file(filename)
    except Exception:
        pass
    return sc


def make_calibration_curve_interp_file(filename):
    sc = calibration.CalibrationCurve()
    sc.function_type = 'interpolation'
    sc.data = [
        [-20.0, -5.3544],
        [-19.6, -5.1117837],
        [-19.2, -4.8826148],
        [-18.8, -4.6663059],
        [-18.4, -4.4622691],
        [-18.0, -4.2699168],
        [-17.6, -4.0886614],
        [-17.2, -3.9179151],
        [-16.8, -3.7570903],
        [-16.4, -3.6055993],
        [-16.0, -3.4628544],
        [-15.6, -3.328268],
        [-15.2, -3.2012524],
        [-14.8, -3.0812199],
        [-14.4, -2.9675829],
        [-14.0, -2.8597536],
        [-13.6, -2.7571444],
        [-13.2, -2.6591677],
        [-12.8, -2.5652357],
        [-12.4, -2.4747607],
        [-12.0, -2.3871552],
        [-11.6, -2.3018314],
        [-11.2, -2.2182017],
        [-10.8, -2.1356783],
        [-10.4, -2.0536736],
        [-10.0, -1.9716],
        [-9.6, -1.92],
        [-9.2, -1.84],
        [-8.8, -1.76],
        [-8.4, -1.68],
        [-8.0, -1.6],
        [-7.6, -1.52],
        [-7.2, -1.44],
        [-6.8, -1.36],
        [-6.4, -1.28],
        [-6.0, -1.2],
        [-5.6, -1.12],
        [-5.2, -1.04],
        [-4.8, -0.96],
        [-4.4, -0.88],
        [-4.0, -0.8],
        [-3.6, -0.72],
        [-3.2, -0.64],
        [-2.8, -0.56],
        [-2.4, -0.48],
        [-2.0, -0.4],
        [-1.6, -0.32],
        [-1.2, -0.24],
        [-0.8, -0.16],
        [-0.4, -0.08],
        [0.0, 0.0],
        [0.4, 0.08],
        [0.8, 0.16],
        [1.2, 0.24],
        [1.6, 0.32],
        [2.0, 0.4],
        [2.4, 0.48],
        [2.8, 0.56],
        [3.2, 0.64],
        [3.6, 0.72],
        [4.0, 0.8],
        [4.4, 0.88],
        [4.8, 0.96],
        [5.2, 1.04],
        [5.6, 1.12],
        [6.0, 1.2],
        [6.4, 1.28],
        [6.8, 1.36],
        [7.2, 1.44],
        [7.6, 1.52],
        [8.0, 1.6],
        [8.4, 1.68],
        [8.8, 1.76],
        [9.2, 1.84],
        [9.6, 1.92],
        [10.0, 2.0],
        [10.4, 2.0489952],
        [10.8, 2.1328111],
        [11.2, 2.2165169],
        [11.6, 2.3007818],
        [12.0, 2.3862752],
        [12.4, 2.4736663],
        [12.8, 2.5636245],
        [13.2, 2.6568189],
        [13.6, 2.7539188],
        [14.0, 2.8555936],
        [14.4, 2.9625125],
        [14.8, 3.0753447],
        [15.2, 3.1947596],
        [15.6, 3.3214264],
        [16.0, 3.4560144],
        [16.4, 3.5991929],
        [16.8, 3.7516311],
        [17.2, 3.9139983],
        [17.6, 4.0869638],
        [18.0, 4.2711968],
        [18.4, 4.4673667],
        [18.8, 4.6761427],
        [19.2, 4.898194],
        [19.6, 5.1341901],
        [20.0, 5.3848],
    ]
    try:
        sc.save_file(filename)
    except Exception:
        pass
    return sc


def make_probe_calibration_poly_file(filename):
    sc = calibration.CalibrationCurve()
    sc.function_type = 'polynomial'
    sc.data = [
        [-1000, -10, 1.8216, 7.0592e-01, 4.7964e-02, 1.5304e-03],
        [-10, 10, 0, 0.2, 0, 0],
        [10, 1000, -2.3614,	8.2643e-01, -5.6814e-02, 1.7429000e-03],
    ]

    pc = calibration.ProbeCalibration()
    pc.probe_name = 'probe_name'
    pc.calibration_magnet = 'calibration_magnet'
    pc.probe_axis = 1
    pc.distance_xy = 10
    pc.distance_zy = 10
    pc.angle_xy = 0
    pc.angle_yz = 0
    pc.angle_xz = 0
    pc.sensorx = sc
    pc.sensory = sc
    pc.sensorz = sc
    try:
        pc.save_file(filename)
    except Exception as e:
        pass
    return pc


def make_probe_calibration_interp_file(filename):
    sc = calibration.CalibrationCurve()
    sc.function_type = 'interpolation'
    sc.data = [
        [-20.0, -5.3544],
        [-19.6, -5.1117837],
        [-19.2, -4.8826148],
        [-18.8, -4.6663059],
        [-18.4, -4.4622691],
        [-18.0, -4.2699168],
        [-17.6, -4.0886614],
        [-17.2, -3.9179151],
        [-16.8, -3.7570903],
        [-16.4, -3.6055993],
        [-16.0, -3.4628544],
        [-15.6, -3.328268],
        [-15.2, -3.2012524],
        [-14.8, -3.0812199],
        [-14.4, -2.9675829],
        [-14.0, -2.8597536],
        [-13.6, -2.7571444],
        [-13.2, -2.6591677],
        [-12.8, -2.5652357],
        [-12.4, -2.4747607],
        [-12.0, -2.3871552],
        [-11.6, -2.3018314],
        [-11.2, -2.2182017],
        [-10.8, -2.1356783],
        [-10.4, -2.0536736],
        [-10.0, -1.9716],
        [-9.6, -1.92],
        [-9.2, -1.84],
        [-8.8, -1.76],
        [-8.4, -1.68],
        [-8.0, -1.6],
        [-7.6, -1.52],
        [-7.2, -1.44],
        [-6.8, -1.36],
        [-6.4, -1.28],
        [-6.0, -1.2],
        [-5.6, -1.12],
        [-5.2, -1.04],
        [-4.8, -0.96],
        [-4.4, -0.88],
        [-4.0, -0.8],
        [-3.6, -0.72],
        [-3.2, -0.64],
        [-2.8, -0.56],
        [-2.4, -0.48],
        [-2.0, -0.4],
        [-1.6, -0.32],
        [-1.2, -0.24],
        [-0.8, -0.16],
        [-0.4, -0.08],
        [0.0, 0.0],
        [0.4, 0.08],
        [0.8, 0.16],
        [1.2, 0.24],
        [1.6, 0.32],
        [2.0, 0.4],
        [2.4, 0.48],
        [2.8, 0.56],
        [3.2, 0.64],
        [3.6, 0.72],
        [4.0, 0.8],
        [4.4, 0.88],
        [4.8, 0.96],
        [5.2, 1.04],
        [5.6, 1.12],
        [6.0, 1.2],
        [6.4, 1.28],
        [6.8, 1.36],
        [7.2, 1.44],
        [7.6, 1.52],
        [8.0, 1.6],
        [8.4, 1.68],
        [8.8, 1.76],
        [9.2, 1.84],
        [9.6, 1.92],
        [10.0, 2.0],
        [10.4, 2.0489952],
        [10.8, 2.1328111],
        [11.2, 2.2165169],
        [11.6, 2.3007818],
        [12.0, 2.3862752],
        [12.4, 2.4736663],
        [12.8, 2.5636245],
        [13.2, 2.6568189],
        [13.6, 2.7539188],
        [14.0, 2.8555936],
        [14.4, 2.9625125],
        [14.8, 3.0753447],
        [15.2, 3.1947596],
        [15.6, 3.3214264],
        [16.0, 3.4560144],
        [16.4, 3.5991929],
        [16.8, 3.7516311],
        [17.2, 3.9139983],
        [17.6, 4.0869638],
        [18.0, 4.2711968],
        [18.4, 4.4673667],
        [18.8, 4.6761427],
        [19.2, 4.898194],
        [19.6, 5.1341901],
        [20.0, 5.3848],
    ]

    pc = calibration.ProbeCalibration()
    pc.probe_name = 'probe_name'
    pc.calibration_magnet = 'calibration_magnet'
    pc.probe_axis = 1
    pc.distance_xy = 10
    pc.distance_zy = 10
    pc.angle_xy = 0
    pc.angle_yz = 0
    pc.angle_xz = 0
    pc.sensorx = sc
    pc.sensory = sc
    pc.sensorz = sc
    try:
        pc.save_file(filename)
    except Exception:
        pass
    return pc


class TestCalibrationCurve(TestCase):
    """Test calibration curve data."""

    def setUp(self):
        """Set up."""
        self.base_directory = os.path.dirname(os.path.abspath(__file__))

        self.filename_polynomial = os.path.join(
            self.base_directory, 'calibration_curve_polynomial.txt')
        self.polynomial = make_calibration_curve_poly_file(
            self.filename_polynomial)

        self.filename_interpolation = os.path.join(
            self.base_directory, 'calibration_curve_interpolation.txt')
        self.interpolation = make_calibration_curve_interp_file(
            self.filename_interpolation)

    def tearDown(self):
        """Tear down."""
        try:
            os.remove(self.filename_polynomial)
            os.remove(self.filename_interpolation)
        except Exception:
            pass

    def test_initialization_without_filename(self):
        c = calibration.CalibrationCurve()
        self.assertIsNone(c.function_type)
        self.assertEqual(len(c.data), 0)
        self.assertIsNone(c._function)

    def test_initialization_with_filename(self):
        c = calibration.CalibrationCurve(self.filename_polynomial)
        self.assertEqual(c.function_type, 'polynomial')
        self.assertEqual(c.data, self.polynomial.data)

        c = calibration.CalibrationCurve(self.filename_interpolation)
        self.assertEqual(c.function_type, 'interpolation')
        self.assertEqual(c.data, self.interpolation.data)

    def test_read_file(self):
        c = calibration.CalibrationCurve()
        c.read_file(self.filename_polynomial)
        self.assertEqual(c.function_type, 'polynomial')
        self.assertEqual(c.data, self.polynomial.data)

        c = calibration.CalibrationCurve()
        c.read_file(self.filename_interpolation)
        self.assertEqual(c.function_type, 'interpolation')
        self.assertEqual(c.data, self.interpolation.data)

    def test_clear(self):
        c = calibration.CalibrationCurve(self.filename_polynomial)
        self.assertEqual(c.function_type, 'polynomial')
        self.assertEqual(c.data, self.polynomial.data)

        c.clear()
        self.assertIsNone(c.function_type)
        self.assertEqual(len(c.data), 0)
        self.assertIsNone(c._function)

    def test_save_file_interpolation(self):
        filename = 'tf_sensor_calibration_interpolation_tmp.txt'
        filename = os.path.join(self.base_directory, filename)
        cw = calibration.CalibrationCurve()
        cw.function_type = 'interpolation'
        cw.data = self.interpolation.data
        cw.save_file(filename)

        cr = calibration.CalibrationCurve(filename)
        self.assertEqual(cr.function_type, cw.function_type)
        self.assertEqual(cr.data, cw.data)
        os.remove(filename)

    def test_save_file_polynomial(self):
        filename = 'tf_sensor_calibration_polynomial_tmp.txt'
        filename = os.path.join(self.base_directory, filename)
        cw = calibration.CalibrationCurve()
        cw.function_type = 'polynomial'
        cw.data = self.polynomial.data
        cw.save_file(filename)

        cr = calibration.CalibrationCurve(filename)
        self.assertEqual(cr.function_type, cw.function_type)
        self.assertEqual(cr.data, cw.data)
        os.remove(filename)

    def test_conversion_polynomial(self):
        c = calibration.CalibrationCurve()
        c.function_type = 'polynomial'
        c.data = self.polynomial.data

        voltage = np.linspace(-15, 15, 101)
        field = calibration._old_hall_sensor_calibration_curve(voltage)
        field_polynomial = c.convert_voltage(voltage)
        np.testing.assert_array_equal(field, field_polynomial)

    def test_conversion_interpolation(self):
        c = calibration.CalibrationCurve()
        c.function_type = 'interpolation'
        c.data = self.interpolation.data

        voltage = np.linspace(-20, -11, 100)
        field = calibration._old_hall_sensor_calibration_curve(voltage)
        field_interpolation = c.convert_voltage(voltage)
        np.testing.assert_array_almost_equal(
            field, field_interpolation, decimal=2)

    def test_equality(self):
        c1 = calibration.CalibrationCurve()
        c2 = calibration.CalibrationCurve()
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 != c2)

        c1 = calibration.CalibrationCurve()
        c2 = calibration.CalibrationCurve(self.filename_polynomial)
        self.assertFalse(c1 == c2)
        self.assertTrue(c1 != c2)

        c1 = calibration.CalibrationCurve(self.filename_polynomial)
        c2 = calibration.CalibrationCurve(self.filename_polynomial)
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 != c2)


class TestProbeCalibration(TestCase):
    """Test probe calibration data."""

    def setUp(self):
        """Set up."""
        self.base_directory = os.path.dirname(os.path.abspath(__file__))

        self.filename_polynomial = os.path.join(
            self.base_directory, 'probe_calibration_polynomial.txt')
        self.polynomial = make_probe_calibration_poly_file(
            self.filename_polynomial)

        self.filename_interpolation = os.path.join(
            self.base_directory, 'probe_calibration_interpolation.txt')
        self.interpolation = make_probe_calibration_interp_file(
            self.filename_interpolation)

        self.database = os.path.join(
            self.base_directory, 'database.db')

    def tearDown(self):
        """Tear down."""
        try:
            os.remove(self.filename_polynomial)
            os.remove(self.filename_interpolation)
        except Exception:
            pass

    def test_initialization_without_filename(self):
        c = calibration.ProbeCalibration()
        self.assertIsNone(c.probe_name)
        self.assertIsNone(c.calibration_magnet)
        self.assertIsNone(c.function_type)
        self.assertIsNone(c.distance_xy)
        self.assertIsNone(c.distance_zy)
        self.assertIsNone(c.angle_xy)
        self.assertIsNone(c.angle_yz)
        self.assertIsNone(c.angle_xz)
        self.assertIsNone(c.probe_axis)
        self.assertEqual(c.sensorx.data, [])
        self.assertEqual(c.sensory.data, [])
        self.assertEqual(c.sensorz.data, [])
        self.assertIsNone(c.sensorx.function_type)
        self.assertIsNone(c.sensory.function_type)
        self.assertIsNone(c.sensorz.function_type)

    def test_initialization_with_filename(self):
        c = calibration.ProbeCalibration(self.filename_polynomial)
        self.assertEqual(c.probe_name, self.polynomial.probe_name)
        self.assertEqual(
            c.calibration_magnet, self.polynomial.calibration_magnet)
        self.assertEqual(c.function_type, self.polynomial.function_type)
        self.assertEqual(c.distance_xy, self.polynomial.distance_xy)
        self.assertEqual(c.distance_zy, self.polynomial.distance_zy)
        self.assertEqual(c.angle_xy, self.polynomial.angle_xy)
        self.assertEqual(c.angle_yz, self.polynomial.angle_yz)
        self.assertEqual(c.angle_xz, self.polynomial.angle_xz)
        self.assertEqual(c.probe_axis, self.polynomial.probe_axis)
        self.assertEqual(c.sensorx, self.polynomial.sensorx)
        self.assertEqual(c.sensory, self.polynomial.sensory)
        self.assertEqual(c.sensorz, self.polynomial.sensorz)

        c = calibration.ProbeCalibration(self.filename_interpolation)
        self.assertEqual(c.probe_name, self.polynomial.probe_name)
        self.assertEqual(
            c.calibration_magnet, self.polynomial.calibration_magnet)
        self.assertEqual(c.function_type, self.interpolation.function_type)
        self.assertEqual(c.distance_xy, self.interpolation.distance_xy)
        self.assertEqual(c.distance_zy, self.interpolation.distance_zy)
        self.assertEqual(c.angle_xy, self.interpolation.angle_xy)
        self.assertEqual(c.angle_yz, self.interpolation.angle_yz)
        self.assertEqual(c.angle_xz, self.interpolation.angle_xz)
        self.assertEqual(c.probe_axis, self.interpolation.probe_axis)
        self.assertEqual(c.sensorx, self.interpolation.sensorx)
        self.assertEqual(c.sensory, self.interpolation.sensory)
        self.assertEqual(c.sensorz, self.interpolation.sensorz)

    def test_read_file(self):
        c = calibration.ProbeCalibration()
        c.read_file(self.filename_polynomial)
        self.assertEqual(c.probe_name, self.polynomial.probe_name)
        self.assertEqual(
            c.calibration_magnet, self.polynomial.calibration_magnet)
        self.assertEqual(c.function_type, self.polynomial.function_type)
        self.assertEqual(c.distance_xy, self.polynomial.distance_xy)
        self.assertEqual(c.distance_zy, self.polynomial.distance_zy)
        self.assertEqual(c.angle_xy, self.polynomial.angle_xy)
        self.assertEqual(c.angle_yz, self.polynomial.angle_yz)
        self.assertEqual(c.angle_xz, self.polynomial.angle_xz)
        self.assertEqual(c.probe_axis, self.polynomial.probe_axis)
        self.assertEqual(c.sensorx, self.polynomial.sensorx)
        self.assertEqual(c.sensory, self.polynomial.sensory)
        self.assertEqual(c.sensorz, self.polynomial.sensorz)

        c = calibration.ProbeCalibration()
        c.read_file(self.filename_interpolation)
        self.assertEqual(c.probe_name, self.polynomial.probe_name)
        self.assertEqual(
            c.calibration_magnet, self.polynomial.calibration_magnet)
        self.assertEqual(c.function_type, self.interpolation.function_type)
        self.assertEqual(c.distance_xy, self.interpolation.distance_xy)
        self.assertEqual(c.distance_zy, self.interpolation.distance_zy)
        self.assertEqual(c.angle_xy, self.interpolation.angle_xy)
        self.assertEqual(c.angle_yz, self.interpolation.angle_yz)
        self.assertEqual(c.angle_xz, self.interpolation.angle_xz)
        self.assertEqual(c.probe_axis, self.interpolation.probe_axis)
        self.assertEqual(c.sensorx, self.interpolation.sensorx)
        self.assertEqual(c.sensory, self.interpolation.sensory)
        self.assertEqual(c.sensorz, self.interpolation.sensorz)

    def test_clear(self):
        c = calibration.ProbeCalibration(self.filename_polynomial)
        self.assertEqual(c.probe_name, self.polynomial.probe_name)
        self.assertEqual(
            c.calibration_magnet, self.polynomial.calibration_magnet)
        self.assertEqual(c.function_type, self.polynomial.function_type)
        self.assertEqual(c.distance_xy, self.polynomial.distance_xy)
        self.assertEqual(c.distance_zy, self.polynomial.distance_zy)
        self.assertEqual(c.angle_xy, self.polynomial.angle_xy)
        self.assertEqual(c.angle_yz, self.polynomial.angle_yz)
        self.assertEqual(c.angle_xz, self.polynomial.angle_xz)
        self.assertEqual(c.probe_axis, self.polynomial.probe_axis)
        self.assertEqual(c.sensorx, self.polynomial.sensorx)
        self.assertEqual(c.sensory, self.polynomial.sensory)
        self.assertEqual(c.sensorz, self.polynomial.sensorz)

        c.clear()
        self.assertIsNone(c.probe_name)
        self.assertIsNone(c.calibration_magnet)
        self.assertIsNone(c.function_type)
        self.assertIsNone(c.distance_xy)
        self.assertIsNone(c.distance_zy)
        self.assertIsNone(c.angle_xy)
        self.assertIsNone(c.angle_yz)
        self.assertIsNone(c.angle_xz)
        self.assertIsNone(c.probe_axis)
        self.assertEqual(c.sensorx.data, [])
        self.assertEqual(c.sensory.data, [])
        self.assertEqual(c.sensorz.data, [])
        self.assertIsNone(c.sensorx.function_type)
        self.assertIsNone(c.sensory.function_type)
        self.assertIsNone(c.sensorz.function_type)

    def test_save_file_interpolation(self):
        filename = os.path.join(
            self.base_directory, 'probe_calibration_interpolation_tmp.txt')
        data = self.interpolation.sensorx.data

        cw = calibration.ProbeCalibration()
        cw.probe_name = 'a'
        cw.calibration_magnet = 'b'
        cw.function_type = 'interpolation'
        cw.distance_xy = 20
        cw.distance_zy = 30
        cw.angle_xy = 10
        cw.angle_yz = 10
        cw.angle_xz = 10
        cw.probe_axis = 3
        cw.sensorx.function_type = 'interpolation'
        cw.sensorx.data = data
        cw.sensory.function_type = 'interpolation'
        cw.sensory.data = data
        cw.sensorz.function_type = 'interpolation'
        cw.sensorz.data = data
        cw.save_file(filename)

        cr = calibration.ProbeCalibration(filename)
        self.assertEqual(cr.probe_name, cw.probe_name)
        self.assertEqual(cr.calibration_magnet, cw.calibration_magnet)
        self.assertEqual(cr.angle_xy, cw.angle_xy)
        self.assertEqual(cr.angle_yz, cw.angle_yz)
        self.assertEqual(cr.angle_xz, cw.angle_xz)
        self.assertEqual(cr.function_type, cw.function_type)
        self.assertEqual(cr.distance_xy, cw.distance_xy)
        self.assertEqual(cr.distance_zy, cw.distance_zy)
        self.assertEqual(cr.probe_axis, cw.probe_axis)
        self.assertEqual(cr.sensorx.data, cw.sensorx.data)
        self.assertEqual(cr.sensory.data, cw.sensory.data)
        self.assertEqual(cr.sensorz.data, cw.sensorz.data)
        os.remove(filename)

    def test_save_file_polynomial(self):
        filename = os.path.join(
            self.base_directory, 'probe_calibration_polynomial_tmp.txt')
        data = self.polynomial.sensorx.data

        cw = calibration.ProbeCalibration()
        cw.probe_name = 'a'
        cw.calibration_magnet = 'b'
        cw.function_type = 'polynomial'
        cw.distance_xy = 20
        cw.distance_zy = 30
        cw.probe_axis = 3
        cw.angle_xy = 10
        cw.angle_yz = 10
        cw.angle_xz = 10
        cw.sensorx.function_type = 'polynomial'
        cw.sensorx.data = data
        cw.sensory.function_type = 'polynomial'
        cw.sensory.data = data
        cw.sensorz.function_type = 'polynomial'
        cw.sensorz.data = data
        cw.save_file(filename)

        cr = calibration.ProbeCalibration(filename)
        self.assertEqual(cr.probe_name, cw.probe_name)
        self.assertEqual(cr.calibration_magnet, cw.calibration_magnet)
        self.assertEqual(cr.angle_xy, cw.angle_xy)
        self.assertEqual(cr.angle_yz, cw.angle_yz)
        self.assertEqual(cr.angle_xz, cw.angle_xz)
        self.assertEqual(cr.function_type, cw.function_type)
        self.assertEqual(cr.distance_xy, cw.distance_xy)
        self.assertEqual(cr.distance_zy, cw.distance_zy)
        self.assertEqual(cr.probe_axis, cw.probe_axis)
        self.assertEqual(cr.sensorx.data, cw.sensorx.data)
        self.assertEqual(cr.sensory.data, cw.sensory.data)
        self.assertEqual(cr.sensorz.data, cw.sensorz.data)
        os.remove(filename)

    def test_equality(self):
        c1 = calibration.ProbeCalibration()
        c2 = calibration.ProbeCalibration()
        self.assertEqual(c1, c2)

        c1 = calibration.ProbeCalibration()
        c2 = calibration.ProbeCalibration(self.filename_polynomial)
        self.assertNotEqual(c1, c2)

        c1 = calibration.ProbeCalibration(self.filename_polynomial)
        c2 = calibration.ProbeCalibration(self.filename_polynomial)
        self.assertEqual(c1, c2)

    def test_corrected_position(self):
        c = calibration.ProbeCalibration(self.filename_polynomial)

        axis = 2
        pos = np.linspace(-5, 5, 11)
        sensor = 'x'
        corr_pos = c.corrected_position(axis, pos, sensor)
        np.testing.assert_array_almost_equal(corr_pos, pos)

        axis = 3
        pos = np.linspace(-5, 5, 11)
        sensor = 'x'
        corr_pos = c.corrected_position(axis, pos, sensor)
        np.testing.assert_array_almost_equal(corr_pos, pos)

        axis = 1
        pos = np.linspace(-5, 5, 11)
        sensor = 'x'
        corr_pos = c.corrected_position(axis, pos, sensor)
        np.testing.assert_array_almost_equal(corr_pos, pos-c.distance_xy)

        axis = 1
        pos = np.linspace(-5, 5, 11)
        sensor = 'y'
        corr_pos = c.corrected_position(axis, pos, sensor)
        np.testing.assert_array_almost_equal(corr_pos, pos)

        axis = 1
        pos = np.linspace(-5, 5, 11)
        sensor = 'z'
        corr_pos = c.corrected_position(axis, pos, sensor)
        np.testing.assert_array_almost_equal(corr_pos, pos+c.distance_zy)

    def test_field_in_bench_coordinate_system(self):
        c = calibration.ProbeCalibration(self.filename_polynomial)
        fieldx = np.array([1, 2, 3])
        fieldy = np.array([4, 5, 6])
        fieldz = np.array([7, 8, 9])

        field3, field2, field1 = c.field_in_bench_coordinate_system(
            fieldx, fieldy, fieldz)
        np.testing.assert_array_equal(field3, fieldx)
        np.testing.assert_array_equal(field2, fieldy)
        np.testing.assert_array_equal(field1, fieldz)

        c.probe_axis = 3
        field3, field2, field1 = c.field_in_bench_coordinate_system(
            fieldx, fieldy, fieldz)
        np.testing.assert_array_equal(field3, fieldz)
        np.testing.assert_array_equal(field2, fieldy)
        np.testing.assert_array_equal(field1, -fieldx)

        field3, field2, field1 = c.field_in_bench_coordinate_system(
            None, None, None)

    def test_database_table_name(self):
        c = calibration.ProbeCalibration()
        tn = calibration.ProbeCalibration.database_table_name()
        self.assertEqual(tn, c._db_table)

    def test_database_functions(self):
        success = calibration.ProbeCalibration.create_database_table(
            self.database)
        self.assertTrue(success)

        cw = calibration.ProbeCalibration(self.filename_polynomial)
        idn = cw.save_to_database(self.database)
        self.assertIsNotNone(idn)

        cr = calibration.ProbeCalibration()
        cr.read_from_database(self.database, idn)
        self.assertEqual(cr, cw)

        cr = calibration.ProbeCalibration(database=self.database, idn=idn)
        self.assertEqual(cr, cw)

        os.remove(self.database)
