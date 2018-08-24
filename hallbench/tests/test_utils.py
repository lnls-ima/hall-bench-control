"""Utils test."""

import os
import numpy as np
from unittest import TestCase
from hallbench.data import utils
from hallbench.data import configuration


def make_connection_configuration_file(filename):
    cc = configuration.ConnectionConfig()
    cc.pmac_enable = 1
    cc.voltx_enable = 1
    cc.voltx_address = 20
    cc.volty_enable = 1
    cc.volty_address = 21
    cc.voltz_enable = 1
    cc.voltz_address = 22
    cc.multich_enable = 1
    cc.multich_address = 18
    cc.nmr_enable = 0
    cc.nmr_port = 'COM1'
    cc.nmr_baudrate = 19200
    cc.elcomat_enable = 0
    cc.elcomat_port = 'COM1'
    cc.elcomat_baudrate = 19200
    try:
        cc.save_file(filename)
    except Exception:
        pass
    return cc


class TestUtils(TestCase):
    """Test utils."""

    def setUp(self):
        """Set up."""
        self.base_directory = os.path.dirname(os.path.abspath(__file__))
        self.filename = os.path.join(
            self.base_directory, 'connection_configuration.txt')
        self.config = make_connection_configuration_file(self.filename)

    def tearDown(self):
        """Tear down."""
        try:
            os.remove(self.filename)
        except Exception:
            pass

    def test_invalid_filename(self):
        with self.assertRaises(IOError):
            flines = utils.read_file('')

    def test_read_file(self):
        read_flines = utils.read_file(self.filename)
        flines = [
            '# Configuration File',
            'pmac_enable      \t1',
            'voltx_enable     \t1',
            'voltx_address    \t20',
            'volty_enable     \t1',
            'volty_address    \t21',
            'voltz_enable     \t1',
            'voltz_address    \t22',
            'multich_enable   \t1',
            'multich_address  \t18',
            'nmr_enable       \t0',
            'nmr_port         \tCOM1',
            'nmr_baudrate     \t19200',
            'elcomat_enable   \t0',
            'elcomat_port     \tCOM1',
            'elcomat_baudrate \t19200',
        ]
        for i in range(len(read_flines)):
            self.assertEqual(read_flines[i], flines[i])

    def test_to_array(self):
        value = utils.to_array(None)
        self.assertTrue(isinstance(value, np.ndarray))
        self.assertEqual(value.size, 0)

        value = utils.to_array(1)
        self.assertTrue(isinstance(value, np.ndarray))
        self.assertEqual(value, np.array(1))

        value = utils.to_array([1])
        self.assertTrue(isinstance(value, np.ndarray))
        self.assertEqual(value, np.array(1))

        value = utils.to_array([1, 1])
        self.assertTrue(isinstance(value, np.ndarray))

        value = utils.to_array(np.array([1, 1]))
        self.assertTrue(isinstance(value, np.ndarray))

    def test_find_value(self):
        flines = utils.read_file(self.filename)
        variable = 'voltx_address'

        with self.assertRaises(ValueError):
            value = utils.find_value([], variable)

        with self.assertRaises(ValueError):
            value = utils.find_value(flines, 'volt_address')

        value = utils.find_value(flines, variable, vtype=int)
        self.assertTrue(isinstance(value, int))
        self.assertEqual(value, 20)

        value = utils.find_value(flines, variable, vtype=float)
        self.assertTrue(isinstance(value, float))
        self.assertEqual(value, float(20))

        value = utils.find_value(flines, variable)
        self.assertTrue(isinstance(value, str))
        self.assertEqual(value, '20')

    def test_find_index(self):
        flines = utils.read_file(self.filename)
        variable = 'voltx_address'
        idx = utils.find_index(flines, variable)
        self.assertEqual(idx, 3)
