"""Calibration test."""

import os
import unittest
import numpy as np
from hall_bench.data_handle import calibration


class TestCalibrationData(unittest.TestCase):
    """Test calibration data."""

    def setUp(self):
        """Set up."""
        self.filename = 'calibration_data.txt'

        self.probe_data_polynomial = [
            [-1000, -10, 1.8216, 7.0592e-01, 4.7964e-02, 1.5304e-03],
            [-10, 10, 0, 0.2, 0, 0],
            [10, 1000, -2.3614,	8.2643e-01, -5.6814e-02, 1.7429000e-03],
        ]

        self.probe_data_interpolation = [
            [-20.0, -5.3544],
            [-18.0, -4.2699168],
            [-16.0, -3.4628544],
            [-14.0, -2.8597536],
            [-12.0, -2.3871552],
            [-10.0, -1.9716],
            [-8.0, -1.6],
            [-6.0, -1.2],
            [-4.0, -0.8],
            [-2.0, -0.4],
            [0.0, 0.0],
            [2.0, 0.4],
            [4.0, 0.8],
            [6.0, 1.2],
            [8.0, 1.6],
            [10.0, 2.0],
            [12.0, 2.3862751],
            [14.0, 2.8555936],
            [16.0, 3.4560144],
            [18.0, 4.2711968],
            [20.0, 5.3848]
        ]

        self.probe_data_interpolation = [
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
        c = calibration.CalibrationData()
        self.assertEqual(c.field_unit, '')
        self.assertEqual(c.voltage_unit, '')
        self.assertIsNone(c.data_type)
        self.assertIsNone(c.distance_probei)
        self.assertIsNone(c.distance_probek)
        self.assertIsNone(c.stem_shape)
        self.assertEqual(c.probei_data, [])
        self.assertEqual(c.probej_data, [])
        self.assertEqual(c.probek_data, [])
        self.assertIsNone(c._probei_function)
        self.assertIsNone(c._probej_function)
        self.assertIsNone(c._probek_function)

    def test_initialization_with_filename(self):
        c = calibration.CalibrationData(self.filename)
        self.assertEqual(c.field_unit, 'T')
        self.assertEqual(c.voltage_unit, 'V')
        self.assertEqual(c.data_type, 'polynomial')
        self.assertEqual(c.distance_probei, 0)
        self.assertEqual(c.distance_probek, 0)
        self.assertEqual(c.stem_shape, 'Straight')
        self.assertEqual(c.probei_data, self.probe_data_polynomial)
        self.assertEqual(c.probej_data, self.probe_data_polynomial)
        self.assertEqual(c.probek_data, self.probe_data_polynomial)

    def test_read_file(self):
        c = calibration.CalibrationData()
        c.read_file(self.filename)
        self.assertEqual(c.field_unit, 'T')
        self.assertEqual(c.voltage_unit, 'V')
        self.assertEqual(c.data_type, 'polynomial')
        self.assertEqual(c.distance_probei, 0)
        self.assertEqual(c.distance_probek, 0)
        self.assertEqual(c.stem_shape, 'Straight')
        self.assertEqual(c.probei_data, self.probe_data_polynomial)
        self.assertEqual(c.probej_data, self.probe_data_polynomial)
        self.assertEqual(c.probek_data, self.probe_data_polynomial)

    def test_clear(self):
        c = calibration.CalibrationData(self.filename)
        c.clear()
        self.assertEqual(c.field_unit, '')
        self.assertEqual(c.voltage_unit, '')
        self.assertIsNone(c.data_type)
        self.assertIsNone(c.distance_probei)
        self.assertIsNone(c.distance_probek)
        self.assertIsNone(c.stem_shape)
        self.assertEqual(c.probei_data, [])
        self.assertEqual(c.probej_data, [])
        self.assertEqual(c.probek_data, [])
        self.assertIsNone(c._probei_function)
        self.assertIsNone(c._probej_function)
        self.assertIsNone(c._probek_function)

    def test_save_file_interpolation(self):
        filename = 'calibration_data_tmp_file.txt'
        cw = calibration.CalibrationData()
        cw.field_unit = 'mT'
        cw.voltage_unit = 'mV'
        cw.data_type = 'interpolation'
        cw.distance_probei = 1
        cw.distance_probek = 2
        cw.stem_shape = 'L-shape'
        cw.probei_data = self.probe_data_interpolation
        cw.probej_data = self.probe_data_interpolation
        cw.probek_data = self.probe_data_interpolation
        cw.save_file(filename)

        cr = calibration.CalibrationData(filename)
        self.assertEqual(cr.field_unit, cw.field_unit)
        self.assertEqual(cr.voltage_unit, cw.voltage_unit)
        self.assertEqual(cr.data_type, cw.data_type)
        self.assertEqual(
            cr.distance_probei, cw.distance_probei)
        self.assertEqual(
            cr.distance_probek, cw.distance_probek)
        self.assertEqual(cr.stem_shape, cw.stem_shape)
        self.assertEqual(cr.probei_data, cw.probei_data)
        self.assertEqual(cr.probej_data, cw.probej_data)
        self.assertEqual(cr.probek_data, cw.probek_data)
        os.remove(filename)

    def test_save_file_polynomial(self):
        filename = 'calibration_data_tmp_file.txt'
        cw = calibration.CalibrationData()
        cw.field_unit = 'mT'
        cw.voltage_unit = 'mV'
        cw.data_type = 'polynomial'
        cw.distance_probei = 1
        cw.distance_probek = 2
        cw.stem_shape = 'Straight'
        cw.probei_data = self.probe_data_polynomial
        cw.probej_data = self.probe_data_polynomial
        cw.probek_data = self.probe_data_polynomial
        cw.save_file(filename)

        cr = calibration.CalibrationData(filename)
        self.assertEqual(cr.field_unit, cw.field_unit)
        self.assertEqual(cr.voltage_unit, cw.voltage_unit)
        self.assertEqual(cr.data_type, cw.data_type)
        self.assertEqual(
            cr.distance_probei, cw.distance_probei)
        self.assertEqual(
            cr.distance_probek, cw.distance_probek)
        self.assertEqual(cr.stem_shape, cw.stem_shape)
        self.assertEqual(cr.probei_data, cw.probei_data)
        self.assertEqual(cr.probej_data, cw.probej_data)
        self.assertEqual(cr.probek_data, cw.probek_data)
        os.remove(filename)

    def test_conversion_polynomial(self):
        c = calibration.CalibrationData()
        c.field_unit = 'T'
        c.voltage_unit = 'V'
        c.data_type = 'polynomial'
        c.distance_probei = 0
        c.distance_probek = 0
        c.stem_shape = 'Straight'
        c.probei_data = self.probe_data_polynomial
        c.probej_data = self.probe_data_polynomial
        c.probek_data = self.probe_data_polynomial

        voltage = np.linspace(-15, 15, 101)
        field = calibration._old_hall_probe_calibration_curve(voltage)
        field_polynomial = c.convert_voltage_probei(voltage)
        np.testing.assert_array_equal(field, field_polynomial)

    def test_conversion_interpolation(self):
        c = calibration.CalibrationData()
        c.field_unit = 'T'
        c.voltage_unit = 'V'
        c.data_type = 'interpolation'
        c.distance_probei = 0
        c.distance_probek = 0
        c.stem_shape = 'Straight'
        c.probei_data = self.probe_data_interpolation
        c.probej_data = self.probe_data_interpolation
        c.probek_data = self.probe_data_interpolation

        voltage = np.linspace(-20, -11, 100)
        field = calibration._old_hall_probe_calibration_curve(voltage)
        field_interpolation = c.convert_voltage_probei(voltage)
        np.testing.assert_array_almost_equal(
            field, field_interpolation, decimal=2)

        voltage = np.linspace(-9, 9, 100)
        field = calibration._old_hall_probe_calibration_curve(voltage)
        field_interpolation = c.convert_voltage_probei(voltage)
        np.testing.assert_array_almost_equal(field, field_interpolation)

        voltage = np.linspace(11, 20, 100)
        field = calibration._old_hall_probe_calibration_curve(voltage)
        field_interpolation = c.convert_voltage_probei(voltage)
        np.testing.assert_array_almost_equal(
            field, field_interpolation, decimal=2)

    def test_equality(self):
        c1 = calibration.CalibrationData()
        c2 = calibration.CalibrationData()
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 != c2)

        c1 = calibration.CalibrationData()
        c2 = calibration.CalibrationData(self.filename)
        self.assertFalse(c1 == c2)
        self.assertTrue(c1 != c2)

        c1 = calibration.CalibrationData(self.filename)
        c2 = calibration.CalibrationData(self.filename)
        self.assertTrue(c1 == c2)
        self.assertFalse(c1 != c2)


def get_suite():
    suite_list = []
    suite_list.append(unittest.TestLoader().loadTestsFromTestCase(
        TestCalibrationData))
    return unittest.TestSuite(suite_list)
