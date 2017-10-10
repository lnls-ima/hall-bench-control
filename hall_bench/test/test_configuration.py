"""Configuration test."""

import os
import unittest
from hall_bench.data_handle import configuration


class TestConnectionConfig(unittest.TestCase):
    """Test connection configuration."""

    def setUp(self):
        """Set up."""
        self.filename = 'connection_configuration_file.txt'

    def tearDown(self):
        """Tear down."""
        pass

    def test_initialization_without_filename(self):
        c = configuration.ConnectionConfig()
        self.assertIsNone(c.control_pmac_enable)
        self.assertIsNone(c.control_voltx_enable)
        self.assertIsNone(c.control_volty_enable)
        self.assertIsNone(c.control_voltz_enable)
        self.assertIsNone(c.control_multich_enable)
        self.assertIsNone(c.control_colimator_enable)
        self.assertIsNone(c.control_voltx_addr)
        self.assertIsNone(c.control_volty_addr)
        self.assertIsNone(c.control_voltz_addr)
        self.assertIsNone(c.control_multich_addr)
        self.assertIsNone(c.control_colimator_addr)

    def test_initialization_with_filename(self):
        c = configuration.ConnectionConfig(self.filename)
        self.assertEqual(c.control_pmac_enable, 1)
        self.assertEqual(c.control_voltx_enable, 1)
        self.assertEqual(c.control_volty_enable, 1)
        self.assertEqual(c.control_voltz_enable, 1)
        self.assertEqual(c.control_multich_enable, 1)
        self.assertEqual(c.control_colimator_enable, 0)
        self.assertEqual(c.control_voltx_addr, 20)
        self.assertEqual(c.control_volty_addr, 21)
        self.assertEqual(c.control_voltz_addr, 22)
        self.assertEqual(c.control_multich_addr, 18)
        self.assertEqual(c.control_colimator_addr, 3)

    def test_read_file(self):
        c = configuration.ConnectionConfig()
        c.read_file(self.filename)
        self.assertEqual(c.control_pmac_enable, 1)
        self.assertEqual(c.control_voltx_enable, 1)
        self.assertEqual(c.control_volty_enable, 1)
        self.assertEqual(c.control_voltz_enable, 1)
        self.assertEqual(c.control_multich_enable, 1)
        self.assertEqual(c.control_colimator_enable, 0)
        self.assertEqual(c.control_voltx_addr, 20)
        self.assertEqual(c.control_volty_addr, 21)
        self.assertEqual(c.control_voltz_addr, 22)
        self.assertEqual(c.control_multich_addr, 18)
        self.assertEqual(c.control_colimator_addr, 3)

    def test_valid_configuration(self):
        cnf = configuration.ConnectionConfig()
        self.assertFalse(cnf.valid_configuration())

        cwf = configuration.ConnectionConfig(self.filename)
        self.assertTrue(cwf.valid_configuration())

        cwf._control_pmac_enable = None
        self.assertFalse(cwf.valid_configuration())

    def test_clear(self):
        c = configuration.ConnectionConfig(self.filename)
        self.assertTrue(c.valid_configuration())

        c.clear()
        self.assertIsNone(c.control_pmac_enable)
        self.assertIsNone(c.control_voltx_enable)
        self.assertIsNone(c.control_volty_enable)
        self.assertIsNone(c.control_voltz_enable)
        self.assertIsNone(c.control_multich_enable)
        self.assertIsNone(c.control_colimator_enable)
        self.assertIsNone(c.control_voltx_addr)
        self.assertIsNone(c.control_volty_addr)
        self.assertIsNone(c.control_voltz_addr)
        self.assertIsNone(c.control_multich_addr)
        self.assertIsNone(c.control_colimator_addr)

    def test_save_file(self):
        filename = 'connection_configuration_tmp_file.txt'
        cw = configuration.ConnectionConfig()
        cw.control_pmac_enable = 0
        cw.control_voltx_enable = 0
        cw.control_volty_enable = 0
        cw.control_voltz_enable = 0
        cw.control_multich_enable = 0
        cw.control_colimator_enable = 0
        cw.control_voltx_addr = 1
        cw.control_volty_addr = 2
        cw.control_voltz_addr = 3
        cw.control_multich_addr = 4
        cw.control_colimator_addr = 5
        cw.save_file(filename)

        cr = configuration.ConnectionConfig(filename)
        self.assertEqual(cr.control_pmac_enable, cw.control_pmac_enable)
        self.assertEqual(cr.control_voltx_enable, cw.control_voltx_enable)
        self.assertEqual(cr.control_volty_enable, cw.control_volty_enable)
        self.assertEqual(cr.control_voltz_enable, cw.control_voltz_enable)
        self.assertEqual(cr.control_multich_enable, cw.control_multich_enable)
        self.assertEqual(cr.control_colimator_enable,
                         cw.control_colimator_enable)
        self.assertEqual(cr.control_voltx_addr, cw.control_voltx_addr)
        self.assertEqual(cr.control_volty_addr, cw.control_volty_addr)
        self.assertEqual(cr.control_voltz_addr, cw.control_voltz_addr)
        self.assertEqual(cr.control_multich_addr, cw.control_multich_addr)
        self.assertEqual(cr.control_colimator_addr, cw.control_colimator_addr)
        os.remove(filename)

    def test_save_file_raise_exception(self):
        filename = 'connection_configuration_tmp_file.txt'
        c = configuration.ConnectionConfig()
        with self.assertRaises(configuration.ConfigurationError):
            c.save_file(filename)


class TestMeasurementConfig(unittest.TestCase):
    """Test measurement configuration."""

    def setUp(self):
        """Set up."""
        self.filename = 'measurement_configuration_file.txt'

    def tearDown(self):
        """Tear down."""
        pass

    def test_initialization_without_filename(self):
        m = configuration.MeasurementConfig()
        self.assertIsNone(m.meas_probeX)
        self.assertIsNone(m.meas_probeY)
        self.assertIsNone(m.meas_probeZ)
        self.assertIsNone(m.meas_aper_ms)
        self.assertIsNone(m.meas_precision)
        self.assertIsNone(m.meas_trig_axis)
        self.assertIsNone(m.meas_startpos_ax1)
        self.assertIsNone(m.meas_endpos_ax1)
        self.assertIsNone(m.meas_incr_ax1)
        self.assertIsNone(m.meas_vel_ax1)
        self.assertIsNone(m.meas_startpos_ax2)
        self.assertIsNone(m.meas_endpos_ax2)
        self.assertIsNone(m.meas_incr_ax2)
        self.assertIsNone(m.meas_vel_ax2)
        self.assertIsNone(m.meas_startpos_ax3)
        self.assertIsNone(m.meas_endpos_ax3)
        self.assertIsNone(m.meas_incr_ax3)
        self.assertIsNone(m.meas_vel_ax3)
        self.assertIsNone(m.meas_startpos_ax5)
        self.assertIsNone(m.meas_endpos_ax5)
        self.assertIsNone(m.meas_incr_ax5)
        self.assertIsNone(m.meas_vel_ax5)

    def test_initialization_with_filename(self):
        m = configuration.MeasurementConfig(self.filename)
        self.assertEqual(m.meas_probeX, 1)
        self.assertEqual(m.meas_probeY, 1)
        self.assertEqual(m.meas_probeZ, 1)
        self.assertEqual(m.meas_aper_ms, 0.003000)
        self.assertEqual(m.meas_precision, 0)
        self.assertEqual(m.meas_trig_axis, 1)
        self.assertEqual(m.meas_startpos_ax1, -200.000000)
        self.assertEqual(m.meas_endpos_ax1, 200.000000)
        self.assertEqual(m.meas_incr_ax1, 0.500000)
        self.assertEqual(m.meas_vel_ax1, 50.000000)
        self.assertEqual(m.meas_startpos_ax2, -136.300000)
        self.assertEqual(m.meas_endpos_ax2, -136.300000)
        self.assertEqual(m.meas_incr_ax2, 1.000000)
        self.assertEqual(m.meas_vel_ax2, 5.000000)
        self.assertEqual(m.meas_startpos_ax3, 140.200000)
        self.assertEqual(m.meas_endpos_ax3, 140.200000)
        self.assertEqual(m.meas_incr_ax3, 1.000000)
        self.assertEqual(m.meas_vel_ax3, 5.000000)
        self.assertEqual(m.meas_startpos_ax5, 0.000000)
        self.assertEqual(m.meas_endpos_ax5, 0.000000)
        self.assertEqual(m.meas_incr_ax5, 1.000000)
        self.assertEqual(m.meas_vel_ax5, 10.000000)

    def test_read_file(self):
        m = configuration.MeasurementConfig()
        m.read_file(self.filename)
        self.assertEqual(m.meas_probeX, 1)
        self.assertEqual(m.meas_probeY, 1)
        self.assertEqual(m.meas_probeZ, 1)
        self.assertEqual(m.meas_aper_ms, 0.003000)
        self.assertEqual(m.meas_precision, 0)
        self.assertEqual(m.meas_trig_axis, 1)
        self.assertEqual(m.meas_startpos_ax1, -200.000000)
        self.assertEqual(m.meas_endpos_ax1, 200.000000)
        self.assertEqual(m.meas_incr_ax1, 0.500000)
        self.assertEqual(m.meas_vel_ax1, 50.000000)
        self.assertEqual(m.meas_startpos_ax2, -136.300000)
        self.assertEqual(m.meas_endpos_ax2, -136.300000)
        self.assertEqual(m.meas_incr_ax2, 1.000000)
        self.assertEqual(m.meas_vel_ax2, 5.000000)
        self.assertEqual(m.meas_startpos_ax3, 140.200000)
        self.assertEqual(m.meas_endpos_ax3, 140.200000)
        self.assertEqual(m.meas_incr_ax3, 1.000000)
        self.assertEqual(m.meas_vel_ax3, 5.000000)
        self.assertEqual(m.meas_startpos_ax5, 0.000000)
        self.assertEqual(m.meas_endpos_ax5, 0.000000)
        self.assertEqual(m.meas_incr_ax5, 1.000000)
        self.assertEqual(m.meas_vel_ax5, 10.000000)

    def test_valid_configuration(self):
        mnf = configuration.MeasurementConfig()
        self.assertFalse(mnf.valid_configuration())

        mwf = configuration.MeasurementConfig(self.filename)
        self.assertTrue(mwf.valid_configuration())

        mwf._meas_probeX = None
        self.assertFalse(mwf.valid_configuration())

    def test_clear(self):
        m = configuration.MeasurementConfig(self.filename)
        self.assertTrue(m.valid_configuration())

        m.clear()
        self.assertIsNone(m.meas_probeX)
        self.assertIsNone(m.meas_probeY)
        self.assertIsNone(m.meas_probeZ)
        self.assertIsNone(m.meas_aper_ms)
        self.assertIsNone(m.meas_precision)
        self.assertIsNone(m.meas_trig_axis)
        self.assertIsNone(m.meas_startpos_ax1)
        self.assertIsNone(m.meas_endpos_ax1)
        self.assertIsNone(m.meas_incr_ax1)
        self.assertIsNone(m.meas_vel_ax1)
        self.assertIsNone(m.meas_startpos_ax2)
        self.assertIsNone(m.meas_endpos_ax2)
        self.assertIsNone(m.meas_incr_ax2)
        self.assertIsNone(m.meas_vel_ax2)
        self.assertIsNone(m.meas_startpos_ax3)
        self.assertIsNone(m.meas_endpos_ax3)
        self.assertIsNone(m.meas_incr_ax3)
        self.assertIsNone(m.meas_vel_ax3)
        self.assertIsNone(m.meas_startpos_ax5)
        self.assertIsNone(m.meas_endpos_ax5)
        self.assertIsNone(m.meas_incr_ax5)
        self.assertIsNone(m.meas_vel_ax5)

    def test_save_file(self):
        filename = 'measurement_configuration_tmp_file.txt'
        mw = configuration.MeasurementConfig()
        mw.meas_probeX = 0
        mw.meas_probeY = 0
        mw.meas_probeZ = 0
        mw.meas_aper_ms = 1
        mw.meas_precision = 1
        mw.meas_trig_axis = 5
        mw.meas_startpos_ax1 = 7
        mw.meas_endpos_ax1 = 7
        mw.meas_incr_ax1 = 9
        mw.meas_vel_ax1 = 10
        mw.meas_startpos_ax2 = 11
        mw.meas_endpos_ax2 = 12
        mw.meas_incr_ax2 = 13
        mw.meas_vel_ax2 = 14
        mw.meas_startpos_ax3 = 15
        mw.meas_endpos_ax3 = 16
        mw.meas_incr_ax3 = 17
        mw.meas_vel_ax3 = 18
        mw.meas_startpos_ax5 = 19
        mw.meas_endpos_ax5 = 20
        mw.meas_incr_ax5 = 21
        mw.meas_vel_ax5 = 22
        mw.save_file(filename)

        mr = configuration.MeasurementConfig(filename)
        self.assertEqual(mr.meas_probeX, mw.meas_probeX)
        self.assertEqual(mr.meas_probeY, mw.meas_probeY)
        self.assertEqual(mr.meas_probeZ, mw.meas_probeZ)
        self.assertEqual(mr.meas_aper_ms, mw.meas_aper_ms)
        self.assertEqual(mr.meas_precision, mw.meas_precision)
        self.assertEqual(mr.meas_trig_axis, mw.meas_trig_axis)
        self.assertEqual(mr.meas_startpos_ax1, mw.meas_startpos_ax1)
        self.assertEqual(mr.meas_endpos_ax1, mw.meas_endpos_ax1)
        self.assertEqual(mr.meas_incr_ax1, mw.meas_incr_ax1)
        self.assertEqual(mr.meas_vel_ax1, mw.meas_vel_ax1)
        self.assertEqual(mr.meas_startpos_ax2, mw.meas_startpos_ax2)
        self.assertEqual(mr.meas_endpos_ax2, mw.meas_endpos_ax2)
        self.assertEqual(mr.meas_incr_ax2, mw.meas_incr_ax2)
        self.assertEqual(mr.meas_vel_ax2, mw.meas_vel_ax2)
        self.assertEqual(mr.meas_startpos_ax3, mw.meas_startpos_ax3)
        self.assertEqual(mr.meas_endpos_ax3, mw.meas_endpos_ax3)
        self.assertEqual(mr.meas_incr_ax3, mw.meas_incr_ax3)
        self.assertEqual(mr.meas_vel_ax3, mw.meas_vel_ax3)
        self.assertEqual(mr.meas_startpos_ax5, mw.meas_startpos_ax5)
        self.assertEqual(mr.meas_endpos_ax5, mw.meas_endpos_ax5)
        self.assertEqual(mr.meas_incr_ax5, mw.meas_incr_ax5)
        self.assertEqual(mr.meas_vel_ax5, mw.meas_vel_ax5)
        os.remove(filename)

    def test_save_file_raise_exception(self):
        filename = 'measurement_configuration_tmp_file.txt'
        m = configuration.MeasurementConfig()
        with self.assertRaises(configuration.ConfigurationError):
            m.save_file(filename)


def get_suite():
    suite_list = []
    suite_list.append(unittest.TestLoader().loadTestsFromTestCase(
        TestConnectionConfig))
    suite_list.append(unittest.TestLoader().loadTestsFromTestCase(
        TestMeasurementConfig))
    return unittest.TestSuite(suite_list)
