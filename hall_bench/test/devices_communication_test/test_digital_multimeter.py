"""Digital Multimeter test."""

import os
import unittest
from hall_bench.devices_communication import devices


class TestDigitalMultimeter(unittest.TestCase):
    """Test control configuration."""

    def setUp(self):
        """Set up."""
        self.logfile = "volt_x.log"

    def tearDown(self):
        """Tear down."""
        pass

    def test_initialization_with_none_args(self):
        with self.assertRaises(devices.DeviceError):
            dm = devices.DigitalMultimeter(None, None)

    def test_initialization_empty_logfile_str(self):
        dm = devices.DigitalMultimeter('', 0)
        self.assertEqual(dm.logfile, '')
        self.assertEqual(dm.address, 0)

    def test_initialization_valid_logfile_path(self):
        dm = devices.DigitalMultimeter(self.logfile, 0)
        self.assertEqual(dm.logfile, self.logfile)
        self.assertTrue(os.path.isfile(self.logfile))
        self.assertTrue(dm.logger is not None)

    def test_connect_invalid_address(self):
        dm = devices.DigitalMultimeter(self.logfile, 0)
        self.assertFalse(dm.connect())


def get_suite():
    suite_list = []
    suite_list.append(unittest.TestLoader().loadTestsFromTestCase(
        TestDigitalMultimeter))
    return unittest.TestSuite(suite_list)
