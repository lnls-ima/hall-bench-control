"""Calibration test."""

import os
import numpy as np
from unittest import TestCase
from hallbench.data import calibration


class TestSensorCalibration(TestCase):
    """Test sensor calibration data."""

    def setUp(self):
        """Set up."""
        self.base_directory = os.path.dirname(os.path.abspath(__file__))
        self.filename_polynomial = os.path.join(
            self.base_directory, 'tf_sensor_calibration_polynomial.txt')
        self.filename_interpolation = os.path.join(
            self.base_directory, 'tf_sensor_calibration_interpolation.txt')

        self.sensor_data_polynomial = [
            [-1000, -10, 1.8216, 7.0592e-01, 4.7964e-02, 1.5304e-03],
            [-10, 10, 0, 0.2, 0, 0],
            [10, 1000, -2.3614,	8.2643e-01, -5.6814e-02, 1.7429000e-03],
        ]

        self.sensor_data_interpolation = [
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

    def tearDown(self):
        """Tear down."""
        pass

    def test_initialization_without_filename(self):
        c = calibration.SensorCalibration()
        self.assertIsNone(c.function_type)
        self.assertIsNone(c.voltage_offset)
        self.assertEqual(len(c.data), 0)
        self.assertIsNone(c.filename)
        self.assertIsNone(c._function)

    def test_initialization_with_filename(self):
        c = calibration.SensorCalibration(self.filename_polynomial)
        self.assertEqual(c.function_type, 'polynomial')
        self.assertEqual(c.voltage_offset, 0)
        self.assertEqual(c.data, self.sensor_data_polynomial)
        self.assertEqual(c.filename, self.filename_polynomial)

        c = calibration.SensorCalibration(self.filename_interpolation)
        self.assertEqual(c.function_type, 'interpolation')
        self.assertEqual(c.voltage_offset, 0)
        self.assertEqual(c.data, self.sensor_data_interpolation)
        self.assertEqual(c.filename, self.filename_interpolation)

    def test_read_file(self):
        c = calibration.SensorCalibration()
        c.read_file(self.filename_polynomial)
        self.assertEqual(c.function_type, 'polynomial')
        self.assertEqual(c.voltage_offset, 0)
        self.assertEqual(c.data, self.sensor_data_polynomial)
        self.assertEqual(c.filename, self.filename_polynomial)

        c = calibration.SensorCalibration()
        c.read_file(self.filename_interpolation)
        self.assertEqual(c.function_type, 'interpolation')
        self.assertEqual(c.voltage_offset, 0)
        self.assertEqual(c.data, self.sensor_data_interpolation)
        self.assertEqual(c.filename, self.filename_interpolation)

    def test_clear(self):
        c = calibration.SensorCalibration(self.filename_polynomial)
        self.assertEqual(c.function_type, 'polynomial')
        self.assertEqual(c.voltage_offset, 0)
        self.assertEqual(c.data, self.sensor_data_polynomial)
        self.assertEqual(c.filename, self.filename_polynomial)
        c.clear()
        self.assertIsNone(c.function_type)
        self.assertIsNone(c.voltage_offset)
        self.assertEqual(len(c.data), 0)
        self.assertIsNone(c.filename)
        self.assertIsNone(c._function)

    def test_save_file_interpolation(self):
        filename = 'tf_sensor_calibration_interpolation_tmp.txt'
        filename = os.path.join(self.base_directory, filename)
        cw = calibration.SensorCalibration()
        cw.function_type = 'interpolation'
        cw.voltage_offset = 0
        cw.data = self.sensor_data_interpolation
        cw.save_file(filename)

        cr = calibration.SensorCalibration(filename)
        self.assertEqual(cr.function_type, cw.function_type)
        self.assertEqual(cr.voltage_offset, cw.voltage_offset)
        self.assertEqual(cr.data, cw.data)
        os.remove(filename)

    def test_save_file_polynomial(self):
        filename = 'tf_sensor_calibration_polynomial_tmp.txt'
        filename = os.path.join(self.base_directory, filename)
        cw = calibration.SensorCalibration()
        cw.function_type = 'polynomial'
        cw.voltage_offset = 0
        cw.data = self.sensor_data_polynomial
        cw.save_file(filename)

        cr = calibration.SensorCalibration(filename)
        self.assertEqual(cr.function_type, cw.function_type)
        self.assertEqual(cr.voltage_offset, cw.voltage_offset)
        self.assertEqual(cr.data, cw.data)
        os.remove(filename)

    def test_conversion_polynomial(self):
        c = calibration.SensorCalibration()
        c.function_type = 'polynomial'
        c.voltage_offset = 0
        c.data = self.sensor_data_polynomial

        voltage = np.linspace(-15, 15, 101)
        field = calibration._old_hall_sensor_calibration_curve(voltage)
        field_polynomial = c.convert_voltage(voltage)
        np.testing.assert_array_equal(field, field_polynomial)

    def test_conversion_interpolation(self):
        c = calibration.SensorCalibration()
        c.function_type = 'interpolation'
        c.voltage_offset = 0
        c.data = self.sensor_data_interpolation

        voltage = np.linspace(-20, -11, 100)
        field = calibration._old_hall_sensor_calibration_curve(voltage)
        field_interpolation = c.convert_voltage(voltage)
        np.testing.assert_array_almost_equal(
            field, field_interpolation, decimal=2)

    def test_equality(self):
        c1 = calibration.SensorCalibration()
        c2 = calibration.SensorCalibration()
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 != c2)

        c1 = calibration.SensorCalibration()
        c2 = calibration.SensorCalibration(self.filename_polynomial)
        self.assertFalse(c1 == c2)
        self.assertTrue(c1 != c2)

        c1 = calibration.SensorCalibration(self.filename_polynomial)
        c2 = calibration.SensorCalibration(self.filename_polynomial)
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 != c2)


class TestProbeCalibration(TestCase):
    """Test probe calibration data."""

    def setUp(self):
        """Set up."""
        self.base_directory = os.path.dirname(os.path.abspath(__file__))
        self.filename_polynomial = os.path.join(
            self.base_directory, 'tf_probe_calibration_polynomial.txt')
        self.filename_interpolation = os.path.join(
            self.base_directory, 'tf_probe_calibration_interpolation.txt')
        self.sensor_filename = os.path.join(
            self.base_directory, 'tf_sensor_calibration_polynomial.txt')

        self.sensor_data_polynomial = [
            [-1000, -10, 1.8216, 7.0592e-01, 4.7964e-02, 1.5304e-03],
            [-10, 10, 0, 0.2, 0, 0],
            [10, 1000, -2.3614,	8.2643e-01, -5.6814e-02, 1.7429000e-03],
        ]

        self.sensor_data_interpolation = [
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

    def tearDown(self):
        """Tear down."""
        pass

    def test_initialization_without_filename(self):
        c = calibration.ProbeCalibration()
        self.assertIsNone(c.function_type)
        self.assertIsNone(c.distance_xy)
        self.assertIsNone(c.distance_zy)
        self.assertIsNone(c.probe_axis)
        self.assertEqual(c.sensorx.data, [])
        self.assertEqual(c.sensory.data, [])
        self.assertEqual(c.sensorz.data, [])
        self.assertIsNone(c._sensorx._function)
        self.assertIsNone(c._sensory._function)
        self.assertIsNone(c._sensorz._function)

    def test_initialization_with_filename(self):
        c = calibration.ProbeCalibration(self.filename_polynomial)
        self.assertEqual(c.function_type, 'polynomial')
        self.assertEqual(c.distance_xy, 10)
        self.assertEqual(c.distance_zy, 10)
        self.assertEqual(c.probe_axis, 1)
        self.assertEqual(c.sensorx.data, self.sensor_data_polynomial)
        self.assertEqual(c.sensory.data, self.sensor_data_polynomial)
        self.assertEqual(c.sensorz.data, self.sensor_data_polynomial)

        c = calibration.ProbeCalibration(self.filename_interpolation)
        self.assertEqual(c.function_type, 'interpolation')
        self.assertEqual(c.distance_xy, 10)
        self.assertEqual(c.distance_zy, 10)
        self.assertEqual(c.probe_axis, 1)
        self.assertEqual(c.sensorx.data, self.sensor_data_interpolation)
        self.assertEqual(c.sensory.data, self.sensor_data_interpolation)
        self.assertEqual(c.sensorz.data, self.sensor_data_interpolation)

    def test_read_file(self):
        c = calibration.ProbeCalibration()
        c.read_file(self.filename_polynomial)
        self.assertEqual(c.function_type, 'polynomial')
        self.assertEqual(c.distance_xy, 10)
        self.assertEqual(c.distance_zy, 10)
        self.assertEqual(c.probe_axis, 1)
        self.assertEqual(c.sensorx.data, self.sensor_data_polynomial)
        self.assertEqual(c.sensory.data, self.sensor_data_polynomial)
        self.assertEqual(c.sensorz.data, self.sensor_data_polynomial)

        c = calibration.ProbeCalibration()
        c.read_file(self.filename_interpolation)
        self.assertEqual(c.function_type, 'interpolation')
        self.assertEqual(c.distance_xy, 10)
        self.assertEqual(c.distance_zy, 10)
        self.assertEqual(c.probe_axis, 1)
        self.assertEqual(c.sensorx.data, self.sensor_data_interpolation)
        self.assertEqual(c.sensory.data, self.sensor_data_interpolation)
        self.assertEqual(c.sensorz.data, self.sensor_data_interpolation)

        fn = self.sensor_filename
        c = calibration.ProbeCalibration()
        c.distance_xy = 10
        c.distance_zy = 10
        c.probe_axis = 1
        c.read_data_from_sensor_files(filenamex=fn, filenamey=fn, filenamez=fn)
        self.assertEqual(c.distance_xy, 10)
        self.assertEqual(c.distance_zy, 10)
        self.assertEqual(c.probe_axis, 1)
        self.assertEqual(c.sensorx.data, self.sensor_data_polynomial)
        self.assertEqual(c.sensory.data, self.sensor_data_polynomial)
        self.assertEqual(c.sensorz.data, self.sensor_data_polynomial)

    def test_clear(self):
        c = calibration.ProbeCalibration(self.filename_polynomial)
        self.assertEqual(c.function_type, 'polynomial')
        self.assertEqual(c.distance_xy, 10)
        self.assertEqual(c.distance_zy, 10)
        self.assertEqual(c.probe_axis, 1)
        self.assertEqual(c.sensorx.data, self.sensor_data_polynomial)
        self.assertEqual(c.sensory.data, self.sensor_data_polynomial)
        self.assertEqual(c.sensorz.data, self.sensor_data_polynomial)
        c.clear()
        self.assertIsNone(c.function_type)
        self.assertIsNone(c.distance_xy)
        self.assertIsNone(c.distance_zy)
        self.assertIsNone(c.probe_axis)
        self.assertEqual(c.sensorx.data, [])
        self.assertEqual(c.sensory.data, [])
        self.assertEqual(c.sensorz.data, [])
        self.assertIsNone(c._sensorx._function)
        self.assertIsNone(c._sensory._function)
        self.assertIsNone(c._sensorz._function)

    def test_save_file_interpolation(self):
        filename = 'tf_probe_calibration_interpolation_tmp.txt'
        filename = os.path.join(self.base_directory, filename)
        cw = calibration.ProbeCalibration()
        cw.function_type = 'interpolation'
        cw.distance_xy = 20
        cw.distance_zy = 30
        cw.probe_axis = 3
        cw.sensorx.function_type = 'interpolation'
        cw.sensorx.data = self.sensor_data_interpolation
        cw.sensorx.voltage_offset = 0
        cw.sensory.function_type = 'interpolation'
        cw.sensory.data = self.sensor_data_interpolation
        cw.sensory.voltage_offset = 0
        cw.sensorz.function_type = 'interpolation'
        cw.sensorz.data = self.sensor_data_interpolation
        cw.sensorz.voltage_offset = 0
        cw.save_file(filename)

        cr = calibration.ProbeCalibration(filename)
        self.assertEqual(cr.function_type, cw.function_type)
        self.assertEqual(cr.distance_xy, cw.distance_xy)
        self.assertEqual(cr.distance_zy, cw.distance_zy)
        self.assertEqual(cr.probe_axis, cw.probe_axis)
        self.assertEqual(cr.sensorx.data, cw.sensorx.data)
        self.assertEqual(cr.sensory.data, cw.sensory.data)
        self.assertEqual(cr.sensorz.data, cw.sensorz.data)
        os.remove(filename)

    def test_save_file_polynomial(self):
        filename = 'tf_probe_calibration_polynomial_tmp.txt'
        filename = os.path.join(self.base_directory, filename)
        cw = calibration.ProbeCalibration()
        cw.function_type = 'polynomial'
        cw.distance_xy = 20
        cw.distance_zy = 30
        cw.probe_axis = 3
        cw.sensorx.function_type = 'polynomial'
        cw.sensorx.data = self.sensor_data_polynomial
        cw.sensorx.voltage_offset = 0
        cw.sensory.function_type = 'polynomial'
        cw.sensory.data = self.sensor_data_polynomial
        cw.sensory.voltage_offset = 0
        cw.sensorz.function_type = 'polynomial'
        cw.sensorz.data = self.sensor_data_polynomial
        cw.sensorz.voltage_offset = 0
        cw.save_file(filename)

        cr = calibration.ProbeCalibration(filename)
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
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 != c2)

        c1 = calibration.ProbeCalibration()
        c2 = calibration.ProbeCalibration(self.filename_polynomial)
        self.assertFalse(c1 == c2)
        self.assertTrue(c1 != c2)

        c1 = calibration.ProbeCalibration(self.filename_polynomial)
        c2 = calibration.ProbeCalibration(self.filename_polynomial)
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 != c2)

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
