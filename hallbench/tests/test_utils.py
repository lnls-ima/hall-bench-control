"""Utils test."""

import os
from unittest import TestCase
from hallbench.data import utils


class TestUtils(TestCase):
    """Test utils."""

    def setUp(self):
        """Set up."""
        self.base_directory = os.path.dirname(os.path.abspath(__file__))
        self.filename = os.path.join(
            self.base_directory, 'tf_connection_configuration.txt')

    def tearDown(self):
        """Tear down."""
        pass

    def test_invalid_filename(self):
        with self.assertRaises(IOError):
            flines = utils.read_file('')

    def test_read_file(self):
        read_flines = utils.read_file(self.filename)
        flines = [
            'Configuration File',
            '#control_pmac_enable\t1',
            '#control_voltx_enable\t1',
            '#control_volty_enable\t1',
            '#control_voltz_enable\t1',
            '#control_multich_enable\t1',
            '#control_collimator_enable\t0',
            '#control_voltx_addr\t20',
            '#control_volty_addr\t21',
            '#control_voltz_addr\t22',
            '#control_multich_addr\t18',
            '#control_collimator_addr\t3',
        ]
        for i in range(len(read_flines)):
            self.assertEqual(read_flines[i], flines[i])

    def test_find_value(self):
        flines = utils.read_file(self.filename)
        variable = 'control_voltx_addr'

        with self.assertRaises(ValueError):
            value = utils.find_value([], variable)

        with self.assertRaises(ValueError):
            value = utils.find_value(flines, 'control_addr')

        value = utils.find_value(flines, variable, vtype=int)
        self.assertTrue(isinstance(value, int))
        self.assertEqual(value, 20)

        value = utils.find_value(flines, variable, vtype=float)
        self.assertTrue(isinstance(value, float))
        self.assertEqual(value, float(20))

        value = utils.find_value(flines, variable)
        self.assertTrue(isinstance(value, str))
        self.assertEqual(value, '20')
