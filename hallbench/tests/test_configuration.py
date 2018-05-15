"""Configuration test."""

import os
from unittest import TestCase
from hallbench.data import configuration


class TestConnectionConfig(TestCase):
    """Test connection configuration."""

    def setUp(self):
        """Set up."""
        self.base_directory = os.path.dirname(os.path.abspath(__file__))
        self.filename = os.path.join(
            self.base_directory, 'tf_connection_configuration.txt')

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
        self.assertIsNone(c.filename)

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
        self.assertEqual(c.filename, self.filename)

    def test_read_file(self):
        c = configuration.ConnectionConfig()
        self.assertIsNone(c.filename)
        c.read_file(self.filename)
        self.assertEqual(c.filename, self.filename)
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
        self.assertIsNone(c.filename)

    def test_save_file(self):
        filename = 'tf_connection_configuration_tmp.txt'
        filename = os.path.join(self.base_directory, filename)
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
        self.assertIsNone(cw.filename)
        cw.save_file(filename)
        self.assertEqual(cw.filename, filename)

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
        self.assertEqual(cr.filename, cw.filename)
        os.remove(filename)

    def test_save_file_raise_exception(self):
        filename = 'tf_connection_configuration_tmp.txt'
        filename = os.path.join(self.base_directory, filename)
        c = configuration.ConnectionConfig()
        with self.assertRaises(configuration.ConfigurationError):
            c.save_file(filename)


class TestMeasurementConfig(TestCase):
    """Test measurement configuration."""

    def setUp(self):
        """Set up."""
        self.base_directory = os.path.dirname(os.path.abspath(__file__))
        self.filename = os.path.join(
            self.base_directory, 'tf_measurement_configuration.txt')

    def tearDown(self):
        """Tear down."""
        pass

    def test_initialization_without_filename(self):
        m = configuration.MeasurementConfig()
        self.assertIsNone(m.meas_probeX)
        self.assertIsNone(m.meas_probeY)
        self.assertIsNone(m.meas_probeZ)
        self.assertIsNone(m.meas_aper)
        self.assertIsNone(m.meas_precision)
        self.assertIsNone(m.meas_nr)
        self.assertIsNone(m.meas_first_axis)
        self.assertIsNone(m.meas_second_axis)
        self.assertIsNone(m.meas_startpos_ax1)
        self.assertIsNone(m.meas_endpos_ax1)
        self.assertIsNone(m.meas_incr_ax1)
        self.assertIsNone(m.meas_extra_ax1)
        self.assertIsNone(m.meas_vel_ax1)
        self.assertIsNone(m.meas_startpos_ax2)
        self.assertIsNone(m.meas_endpos_ax2)
        self.assertIsNone(m.meas_incr_ax2)
        self.assertIsNone(m.meas_extra_ax2)
        self.assertIsNone(m.meas_vel_ax2)
        self.assertIsNone(m.meas_startpos_ax3)
        self.assertIsNone(m.meas_endpos_ax3)
        self.assertIsNone(m.meas_incr_ax3)
        self.assertIsNone(m.meas_extra_ax3)
        self.assertIsNone(m.meas_vel_ax3)
        self.assertIsNone(m.meas_startpos_ax5)
        self.assertIsNone(m.meas_endpos_ax5)
        self.assertIsNone(m.meas_incr_ax5)
        self.assertIsNone(m.meas_extra_ax5)
        self.assertIsNone(m.meas_vel_ax5)
        self.assertIsNone(m.filename)

    def test_initialization_with_filename(self):
        m = configuration.MeasurementConfig(self.filename)
        self.assertEqual(m.meas_probeX, 1)
        self.assertEqual(m.meas_probeY, 1)
        self.assertEqual(m.meas_probeZ, 1)
        self.assertEqual(m.meas_aper, 0.003000)
        self.assertEqual(m.meas_precision, 0)
        self.assertEqual(m.meas_nr, 2)
        self.assertEqual(m.meas_first_axis, 1)
        self.assertEqual(m.meas_second_axis, -1)
        self.assertEqual(m.meas_startpos_ax1, -200.000000)
        self.assertEqual(m.meas_endpos_ax1, 200.000000)
        self.assertEqual(m.meas_incr_ax1, 0.500000)
        self.assertEqual(m.meas_extra_ax1, 0.000000)
        self.assertEqual(m.meas_vel_ax1, 50.000000)
        self.assertEqual(m.meas_startpos_ax2, -136.300000)
        self.assertEqual(m.meas_endpos_ax2, -136.300000)
        self.assertEqual(m.meas_incr_ax2, 1.000000)
        self.assertEqual(m.meas_extra_ax2, 0.000000)
        self.assertEqual(m.meas_vel_ax2, 5.000000)
        self.assertEqual(m.meas_startpos_ax3, 140.200000)
        self.assertEqual(m.meas_endpos_ax3, 140.200000)
        self.assertEqual(m.meas_incr_ax3, 1.000000)
        self.assertEqual(m.meas_extra_ax3, 0.000000)
        self.assertEqual(m.meas_vel_ax3, 5.000000)
        self.assertEqual(m.meas_startpos_ax5, 0.000000)
        self.assertEqual(m.meas_endpos_ax5, 0.000000)
        self.assertEqual(m.meas_incr_ax5, 1.000000)
        self.assertEqual(m.meas_extra_ax5, 0.000000)
        self.assertEqual(m.meas_vel_ax5, 10.000000)
        self.assertEqual(m.filename, self.filename)

    def test_get_axis_properties(self):
        m = configuration.MeasurementConfig(self.filename)
        self.assertEqual(m.get_start(1), -200.000000)
        self.assertEqual(m.get_start(2), -136.300000)
        self.assertEqual(m.get_start(3), 140.200000)
        self.assertEqual(m.get_start(5), 0.000000)
        self.assertEqual(m.get_end(1), 200.000000)
        self.assertEqual(m.get_end(2), -136.300000)
        self.assertEqual(m.get_end(3), 140.200000)
        self.assertEqual(m.get_end(5), 0.000000)
        self.assertEqual(m.get_step(1), 0.500000)
        self.assertEqual(m.get_step(2), 1.000000)
        self.assertEqual(m.get_step(3), 1.000000)
        self.assertEqual(m.get_step(5), 1.000000)
        self.assertEqual(m.get_extra(1), 0.000000)
        self.assertEqual(m.get_extra(2), 0.000000)
        self.assertEqual(m.get_extra(3), 0.000000)
        self.assertEqual(m.get_extra(5), 0.000000)
        self.assertEqual(m.get_velocity(1), 50.000000)
        self.assertEqual(m.get_velocity(2), 5.000000)
        self.assertEqual(m.get_velocity(3), 5.000000)
        self.assertEqual(m.get_velocity(5), 10.000000)

    def test_set_axis_properties(self):
        m = configuration.MeasurementConfig(self.filename)

        m.set_start(1, 1)
        m.set_start(2, 2)
        m.set_start(3, 3)
        m.set_start(5, 4)
        m.set_end(1, 5)
        m.set_end(2, 6)
        m.set_end(3, 7)
        m.set_end(5, 8)
        m.set_step(1, 9)
        m.set_step(2, 10)
        m.set_step(3, 11)
        m.set_step(5, 12)
        m.set_extra(1, 13)
        m.set_extra(2, 14)
        m.set_extra(3, 15)
        m.set_extra(5, 16)
        m.set_velocity(1, 17)
        m.set_velocity(2, 18)
        m.set_velocity(3, 19)
        m.set_velocity(5, 20)

        self.assertEqual(m.meas_startpos_ax1, 1.000000)
        self.assertEqual(m.meas_startpos_ax2, 2.00000)
        self.assertEqual(m.meas_startpos_ax3, 3.000000)
        self.assertEqual(m.meas_startpos_ax5, 4.000000)
        self.assertEqual(m.meas_endpos_ax1, 5.000000)
        self.assertEqual(m.meas_endpos_ax2, 6.000000)
        self.assertEqual(m.meas_endpos_ax3, 7.000000)
        self.assertEqual(m.meas_endpos_ax5, 8.000000)
        self.assertEqual(m.meas_incr_ax1, 9.000000)
        self.assertEqual(m.meas_incr_ax2, 10.000000)
        self.assertEqual(m.meas_incr_ax3, 11.000000)
        self.assertEqual(m.meas_incr_ax5, 12.000000)
        self.assertEqual(m.meas_extra_ax1, 13.000000)
        self.assertEqual(m.meas_extra_ax2, 14.000000)
        self.assertEqual(m.meas_extra_ax3, 15.000000)
        self.assertEqual(m.meas_extra_ax5, 16.000000)
        self.assertEqual(m.meas_vel_ax1, 17.000000)
        self.assertEqual(m.meas_vel_ax2, 18.000000)
        self.assertEqual(m.meas_vel_ax3, 19.000000)
        self.assertEqual(m.meas_vel_ax5, 20.000000)

    def test_read_file(self):
        m = configuration.MeasurementConfig()
        self.assertIsNone(m.filename)
        m.read_file(self.filename)
        self.assertEqual(m.meas_probeX, 1)
        self.assertEqual(m.meas_probeY, 1)
        self.assertEqual(m.meas_probeZ, 1)
        self.assertEqual(m.meas_aper, 0.003000)
        self.assertEqual(m.meas_precision, 0)
        self.assertEqual(m.meas_nr, 2)
        self.assertEqual(m.meas_first_axis, 1)
        self.assertEqual(m.meas_second_axis, -1)
        self.assertEqual(m.meas_startpos_ax1, -200.000000)
        self.assertEqual(m.meas_endpos_ax1, 200.000000)
        self.assertEqual(m.meas_incr_ax1, 0.500000)
        self.assertEqual(m.meas_extra_ax1, 0.000000)
        self.assertEqual(m.meas_vel_ax1, 50.000000)
        self.assertEqual(m.meas_startpos_ax2, -136.300000)
        self.assertEqual(m.meas_endpos_ax2, -136.300000)
        self.assertEqual(m.meas_incr_ax2, 1.000000)
        self.assertEqual(m.meas_extra_ax2, 0.000000)
        self.assertEqual(m.meas_vel_ax2, 5.000000)
        self.assertEqual(m.meas_startpos_ax3, 140.200000)
        self.assertEqual(m.meas_endpos_ax3, 140.200000)
        self.assertEqual(m.meas_incr_ax3, 1.000000)
        self.assertEqual(m.meas_extra_ax3, 0.000000)
        self.assertEqual(m.meas_vel_ax3, 5.000000)
        self.assertEqual(m.meas_startpos_ax5, 0.000000)
        self.assertEqual(m.meas_endpos_ax5, 0.000000)
        self.assertEqual(m.meas_incr_ax5, 1.000000)
        self.assertEqual(m.meas_extra_ax5, 0.000000)
        self.assertEqual(m.meas_vel_ax5, 10.000000)
        self.assertEqual(m.filename, self.filename)

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
        self.assertIsNone(m.meas_aper)
        self.assertIsNone(m.meas_precision)
        self.assertIsNone(m.meas_nr)
        self.assertIsNone(m.meas_first_axis)
        self.assertIsNone(m.meas_second_axis)
        self.assertIsNone(m.meas_startpos_ax1)
        self.assertIsNone(m.meas_endpos_ax1)
        self.assertIsNone(m.meas_incr_ax1)
        self.assertIsNone(m.meas_extra_ax1)
        self.assertIsNone(m.meas_vel_ax1)
        self.assertIsNone(m.meas_startpos_ax2)
        self.assertIsNone(m.meas_endpos_ax2)
        self.assertIsNone(m.meas_incr_ax2)
        self.assertIsNone(m.meas_extra_ax2)
        self.assertIsNone(m.meas_vel_ax2)
        self.assertIsNone(m.meas_startpos_ax3)
        self.assertIsNone(m.meas_endpos_ax3)
        self.assertIsNone(m.meas_incr_ax3)
        self.assertIsNone(m.meas_extra_ax3)
        self.assertIsNone(m.meas_vel_ax3)
        self.assertIsNone(m.meas_startpos_ax5)
        self.assertIsNone(m.meas_endpos_ax5)
        self.assertIsNone(m.meas_incr_ax5)
        self.assertIsNone(m.meas_extra_ax5)
        self.assertIsNone(m.meas_vel_ax5)
        self.assertIsNone(m.filename)

    def test_save_file(self):
        filename = 'tf_measurement_configuration_tmp.txt'
        filename = os.path.join(self.base_directory, filename)
        mw = configuration.MeasurementConfig()
        mw.meas_probeX = 0
        mw.meas_probeY = 0
        mw.meas_probeZ = 0
        mw.meas_aper = 1
        mw.meas_nr = 2
        mw.meas_precision = 1
        mw.meas_first_axis = 5
        mw.meas_second_axis = -1
        mw.meas_startpos_ax1 = 7
        mw.meas_endpos_ax1 = 7
        mw.meas_incr_ax1 = 9
        mw.meas_extra_ax1 = 0
        mw.meas_vel_ax1 = 10
        mw.meas_startpos_ax2 = 11
        mw.meas_endpos_ax2 = 12
        mw.meas_incr_ax2 = 13
        mw.meas_extra_ax2 = 0
        mw.meas_vel_ax2 = 14
        mw.meas_startpos_ax3 = 15
        mw.meas_endpos_ax3 = 16
        mw.meas_incr_ax3 = 17
        mw.meas_extra_ax3 = 0
        mw.meas_vel_ax3 = 18
        mw.meas_startpos_ax5 = 19
        mw.meas_endpos_ax5 = 20
        mw.meas_incr_ax5 = 21
        mw.meas_extra_ax5 = 0
        mw.meas_vel_ax5 = 22
        self.assertIsNone(mw.filename)
        mw.save_file(filename)
        self.assertEqual(mw.filename, filename)

        mr = configuration.MeasurementConfig(filename)
        self.assertEqual(mr.meas_probeX, mw.meas_probeX)
        self.assertEqual(mr.meas_probeY, mw.meas_probeY)
        self.assertEqual(mr.meas_probeZ, mw.meas_probeZ)
        self.assertEqual(mr.meas_aper, mw.meas_aper)
        self.assertEqual(mr.meas_precision, mw.meas_precision)
        self.assertEqual(mr.meas_first_axis, mw.meas_first_axis)
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
        self.assertEqual(mr.filename, mw.filename)
        os.remove(filename)

    def test_save_file_raise_exception(self):
        filename = 'tf_measurement_configuration_tmp.txt'
        filename = os.path.join(self.base_directory, filename)
        m = configuration.MeasurementConfig()
        with self.assertRaises(configuration.ConfigurationError):
            m.save_file(filename)
