"""Configuration test."""

import os
from unittest import TestCase
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


def make_measurement_configuration_file(filename):
    mc = configuration.MeasurementConfig()
    mc.magnet_name = 'magnet_name'
    mc.main_current = 'main_current'
    mc.probe_name = 'probe_name'
    mc.temperature = 'temperature'
    mc.operator = 'operator'
    mc.voltx_enable = 1
    mc.volty_enable = 1
    mc.voltz_enable = 1
    mc.integration_time = 0.003
    mc.voltage_precision = 0
    mc.nr_measurements = 1
    mc.first_axis = 1
    mc.second_axis = -1
    mc.start_ax1 = -200
    mc.end_ax1 = 200
    mc.step_ax1 = 0.5
    mc.extra_ax1 = 0
    mc.vel_ax1 = 50
    mc.start_ax2 = -136.3
    mc.end_ax2 = -136.3
    mc.step_ax2 = 1
    mc.extra_ax2 = 0
    mc.vel_ax2 = 5
    mc.start_ax3 = 140.2
    mc.end_ax3 = 140.2
    mc.step_ax3 = 1
    mc.extra_ax3 = 0
    mc.vel_ax3 = 5
    mc.start_ax5 = 0
    mc.end_ax5 = 0
    mc.step_ax5 = 1
    mc.extra_ax5 = 0
    mc.vel_ax5 = 10
    try:
        mc.save_file(filename)
    except Exception:
        pass
    return mc


class TestConnectionConfig(TestCase):
    """Test connection configuration."""

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

    def test_initialization_without_filename(self):
        c = configuration.ConnectionConfig()
        self.assertIsNone(c.pmac_enable)
        self.assertIsNone(c.voltx_enable)
        self.assertIsNone(c.volty_enable)
        self.assertIsNone(c.voltz_enable)
        self.assertIsNone(c.multich_enable)
        self.assertIsNone(c.nmr_enable)
        self.assertIsNone(c.elcomat_enable)
        self.assertIsNone(c.voltx_address)
        self.assertIsNone(c.volty_address)
        self.assertIsNone(c.voltz_address)
        self.assertIsNone(c.multich_address)
        self.assertIsNone(c.nmr_baudrate)
        self.assertIsNone(c.nmr_port)
        self.assertIsNone(c.elcomat_port)

    def test_initialization_with_filename(self):
        c = configuration.ConnectionConfig(self.filename)
        self.assertEqual(c.pmac_enable, self.config.pmac_enable)
        self.assertEqual(c.voltx_enable, self.config.voltx_enable)
        self.assertEqual(c.volty_enable, self.config.volty_enable)
        self.assertEqual(c.voltz_enable, self.config.voltz_enable)
        self.assertEqual(c.multich_enable, self.config.multich_enable)
        self.assertEqual(c.nmr_enable, self.config.nmr_enable)
        self.assertEqual(c.elcomat_enable,  self.config.elcomat_enable)
        self.assertEqual(c.voltx_address, self.config.voltx_address)
        self.assertEqual(c.volty_address, self.config.volty_address)
        self.assertEqual(c.voltz_address, self.config.voltz_address)
        self.assertEqual(c.multich_address, self.config.multich_address)
        self.assertEqual(c.nmr_baudrate, self.config.nmr_baudrate)
        self.assertEqual(c.nmr_port, self.config.nmr_port)
        self.assertEqual(c.elcomat_port, self.config.elcomat_port)

    def test_equality(self):
        c1 = configuration.ConnectionConfig()
        c2 = configuration.ConnectionConfig()
        self.assertEqual(c1, c2)

    def test_inequality(self):
        c1 = configuration.ConnectionConfig()
        c2 = configuration.ConnectionConfig()
        c2.pmac_enable = 1
        self.assertNotEqual(c1, c2)

    def test_read_file(self):
        c = configuration.ConnectionConfig()
        c.read_file(self.filename)
        self.assertEqual(c.pmac_enable, self.config.pmac_enable)
        self.assertEqual(c.voltx_enable, self.config.voltx_enable)
        self.assertEqual(c.volty_enable, self.config.volty_enable)
        self.assertEqual(c.voltz_enable, self.config.voltz_enable)
        self.assertEqual(c.multich_enable, self.config.multich_enable)
        self.assertEqual(c.nmr_enable, self.config.nmr_enable)
        self.assertEqual(c.elcomat_enable,  self.config.elcomat_enable)
        self.assertEqual(c.voltx_address, self.config.voltx_address)
        self.assertEqual(c.volty_address, self.config.volty_address)
        self.assertEqual(c.voltz_address, self.config.voltz_address)
        self.assertEqual(c.multich_address, self.config.multich_address)
        self.assertEqual(c.nmr_baudrate, self.config.nmr_baudrate)
        self.assertEqual(c.nmr_port, self.config.nmr_port)
        self.assertEqual(c.elcomat_port, self.config.elcomat_port)

    def test_valid_data(self):
        cnf = configuration.ConnectionConfig()
        self.assertFalse(cnf.valid_data())

        cwf = configuration.ConnectionConfig(self.filename)
        self.assertTrue(cwf.valid_data())

        cwf._pmac_enable = None
        self.assertFalse(cwf.valid_data())

    def test_clear(self):
        c = configuration.ConnectionConfig(self.filename)
        self.assertTrue(c.valid_data())

        c.clear()
        self.assertIsNone(c.pmac_enable)
        self.assertIsNone(c.voltx_enable)
        self.assertIsNone(c.volty_enable)
        self.assertIsNone(c.voltz_enable)
        self.assertIsNone(c.multich_enable)
        self.assertIsNone(c.nmr_enable)
        self.assertIsNone(c.elcomat_enable)
        self.assertIsNone(c.voltx_address)
        self.assertIsNone(c.volty_address)
        self.assertIsNone(c.voltz_address)
        self.assertIsNone(c.multich_address)
        self.assertIsNone(c.nmr_baudrate)
        self.assertIsNone(c.nmr_port)
        self.assertIsNone(c.elcomat_port)

    def test_save_file(self):
        filename = 'connection_configuration_tmp.txt'
        filename = os.path.join(self.base_directory, filename)
        cw = configuration.ConnectionConfig()
        cw.pmac_enable = 0
        cw.voltx_enable = 0
        cw.volty_enable = 0
        cw.voltz_enable = 0
        cw.multich_enable = 0
        cw.nmr_enable = 0
        cw.elcomat_enable = 0
        cw.voltx_address = 1
        cw.volty_address = 2
        cw.voltz_address = 3
        cw.multich_address = 4
        cw.nmr_baudrate = 300
        cw.nmr_port = 'COM2'
        cw.elcomat_port = 'COM3'
        cw.elcomat_baudrate = 300
        cw.save_file(filename)

        cr = configuration.ConnectionConfig(filename)
        self.assertEqual(cr.pmac_enable, cw.pmac_enable)
        self.assertEqual(cr.voltx_enable, cw.voltx_enable)
        self.assertEqual(cr.volty_enable, cw.volty_enable)
        self.assertEqual(cr.voltz_enable, cw.voltz_enable)
        self.assertEqual(cr.multich_enable, cw.multich_enable)
        self.assertEqual(cr.nmr_enable, cw.nmr_enable)
        self.assertEqual(cr.elcomat_enable, cw.elcomat_enable)
        self.assertEqual(cr.voltx_address, cw.voltx_address)
        self.assertEqual(cr.volty_address, cw.volty_address)
        self.assertEqual(cr.voltz_address, cw.voltz_address)
        self.assertEqual(cr.multich_address, cw.multich_address)
        self.assertEqual(cr.nmr_baudrate, cw.nmr_baudrate)
        self.assertEqual(cr.nmr_port, cw.nmr_port)
        self.assertEqual(cr.elcomat_port, cw.elcomat_port)
        os.remove(filename)

    def test_save_file_raise_exception(self):
        filename = 'connection_configuration_tmp.txt'
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
            self.base_directory, 'measurement_configuration.txt')
        self.config = make_measurement_configuration_file(self.filename)
        self.database = os.path.join(
            self.base_directory, 'database.db')

    def tearDown(self):
        """Tear down."""
        try:
            os.remove(self.filename)
        except Exception:
            pass

    def test_initialization_without_filename(self):
        m = configuration.MeasurementConfig()
        self.assertIsNone(m.magnet_name)
        self.assertIsNone(m.main_current)
        self.assertIsNone(m.probe_name)
        self.assertIsNone(m.temperature)
        self.assertIsNone(m.operator)
        self.assertIsNone(m.voltx_enable)
        self.assertIsNone(m.volty_enable)
        self.assertIsNone(m.voltz_enable)
        self.assertIsNone(m.integration_time)
        self.assertIsNone(m.voltage_precision)
        self.assertIsNone(m.nr_measurements)
        self.assertIsNone(m.first_axis)
        self.assertIsNone(m.second_axis)
        self.assertIsNone(m.start_ax1)
        self.assertIsNone(m.end_ax1)
        self.assertIsNone(m.step_ax1)
        self.assertIsNone(m.extra_ax1)
        self.assertIsNone(m.vel_ax1)
        self.assertIsNone(m.start_ax2)
        self.assertIsNone(m.end_ax2)
        self.assertIsNone(m.step_ax2)
        self.assertIsNone(m.extra_ax2)
        self.assertIsNone(m.vel_ax2)
        self.assertIsNone(m.start_ax3)
        self.assertIsNone(m.end_ax3)
        self.assertIsNone(m.step_ax3)
        self.assertIsNone(m.extra_ax3)
        self.assertIsNone(m.vel_ax3)
        self.assertIsNone(m.start_ax5)
        self.assertIsNone(m.end_ax5)
        self.assertIsNone(m.step_ax5)
        self.assertIsNone(m.extra_ax5)
        self.assertIsNone(m.vel_ax5)

    def test_initialization_with_filename(self):
        m = configuration.MeasurementConfig(self.filename)
        self.assertEqual(m.magnet_name, self.config.magnet_name)
        self.assertEqual(m.main_current, self.config.main_current)
        self.assertEqual(m.probe_name, self.config.probe_name)
        self.assertEqual(m.temperature, self.config.temperature)
        self.assertEqual(m.operator, self.config.operator)
        self.assertEqual(m.voltx_enable, self.config.voltx_enable)
        self.assertEqual(m.volty_enable, self.config.volty_enable)
        self.assertEqual(m.voltz_enable, self.config.voltz_enable)
        self.assertEqual(m.integration_time, self.config.integration_time)
        self.assertEqual(m.voltage_precision, self.config.voltage_precision)
        self.assertEqual(m.nr_measurements, self.config.nr_measurements)
        self.assertEqual(m.first_axis, self.config.first_axis)
        self.assertEqual(m.second_axis, self.config.second_axis)
        self.assertEqual(m.start_ax1, self.config.start_ax1)
        self.assertEqual(m.end_ax1, self.config.end_ax1)
        self.assertEqual(m.step_ax1, self.config.step_ax1)
        self.assertEqual(m.extra_ax1, self.config.extra_ax1)
        self.assertEqual(m.vel_ax1, self.config.vel_ax1)
        self.assertEqual(m.start_ax2, self.config.start_ax2)
        self.assertEqual(m.end_ax2, self.config.end_ax2)
        self.assertEqual(m.step_ax2, self.config.step_ax2)
        self.assertEqual(m.extra_ax2, self.config.extra_ax2)
        self.assertEqual(m.vel_ax2, self.config.vel_ax2)
        self.assertEqual(m.start_ax3, self.config.start_ax3)
        self.assertEqual(m.end_ax3, self.config.end_ax3)
        self.assertEqual(m.step_ax3, self.config.step_ax3)
        self.assertEqual(m.extra_ax3, self.config.extra_ax3)
        self.assertEqual(m.vel_ax3, self.config.vel_ax3)
        self.assertEqual(m.start_ax5, self.config.start_ax5)
        self.assertEqual(m.end_ax5, self.config.end_ax5)
        self.assertEqual(m.step_ax5, self.config.step_ax5)
        self.assertEqual(m.extra_ax5, self.config.extra_ax5)
        self.assertEqual(m.vel_ax5, self.config.vel_ax5)

    def test_equality(self):
        m1 = configuration.MeasurementConfig()
        m2 = configuration.MeasurementConfig()
        self.assertEqual(m1, m2)

    def test_inequality(self):
        m1 = configuration.MeasurementConfig()
        m2 = configuration.MeasurementConfig()
        m2.voltx_enable = 1
        self.assertNotEqual(m1, m2)

    def test_get_axis_properties(self):
        m = configuration.MeasurementConfig(self.filename)
        self.assertEqual(m.get_start(1), self.config.start_ax1)
        self.assertEqual(m.get_start(2), self.config.start_ax2)
        self.assertEqual(m.get_start(3), self.config.start_ax3)
        self.assertEqual(m.get_start(5), self.config.start_ax5)
        self.assertEqual(m.get_end(1), self.config.end_ax1)
        self.assertEqual(m.get_end(2), self.config.end_ax2)
        self.assertEqual(m.get_end(3), self.config.end_ax3)
        self.assertEqual(m.get_end(5), self.config.end_ax5)
        self.assertEqual(m.get_step(1), self.config.step_ax1)
        self.assertEqual(m.get_step(2), self.config.step_ax2)
        self.assertEqual(m.get_step(3), self.config.step_ax3)
        self.assertEqual(m.get_step(5), self.config.step_ax5)
        self.assertEqual(m.get_extra(1), self.config.extra_ax1)
        self.assertEqual(m.get_extra(2), self.config.extra_ax2)
        self.assertEqual(m.get_extra(3), self.config.extra_ax3)
        self.assertEqual(m.get_extra(5), self.config.extra_ax5)
        self.assertEqual(m.get_velocity(1), self.config.vel_ax1)
        self.assertEqual(m.get_velocity(2), self.config.vel_ax2)
        self.assertEqual(m.get_velocity(3), self.config.vel_ax3)
        self.assertEqual(m.get_velocity(5), self.config.vel_ax5)

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

        self.assertEqual(m.start_ax1, 1.000000)
        self.assertEqual(m.start_ax2, 2.00000)
        self.assertEqual(m.start_ax3, 3.000000)
        self.assertEqual(m.start_ax5, 4.000000)
        self.assertEqual(m.end_ax1, 5.000000)
        self.assertEqual(m.end_ax2, 6.000000)
        self.assertEqual(m.end_ax3, 7.000000)
        self.assertEqual(m.end_ax5, 8.000000)
        self.assertEqual(m.step_ax1, 9.000000)
        self.assertEqual(m.step_ax2, 10.000000)
        self.assertEqual(m.step_ax3, 11.000000)
        self.assertEqual(m.step_ax5, 12.000000)
        self.assertEqual(m.extra_ax1, 13.000000)
        self.assertEqual(m.extra_ax2, 14.000000)
        self.assertEqual(m.extra_ax3, 15.000000)
        self.assertEqual(m.extra_ax5, 16.000000)
        self.assertEqual(m.vel_ax1, 17.000000)
        self.assertEqual(m.vel_ax2, 18.000000)
        self.assertEqual(m.vel_ax3, 19.000000)
        self.assertEqual(m.vel_ax5, 20.000000)

    def test_read_file(self):
        m = configuration.MeasurementConfig()
        m.read_file(self.filename)
        self.assertEqual(m.magnet_name, self.config.magnet_name)
        self.assertEqual(m.main_current, self.config.main_current)
        self.assertEqual(m.probe_name, self.config.probe_name)
        self.assertEqual(m.temperature, self.config.temperature)
        self.assertEqual(m.operator, self.config.operator)
        self.assertEqual(m.voltx_enable, self.config.voltx_enable)
        self.assertEqual(m.volty_enable, self.config.volty_enable)
        self.assertEqual(m.voltz_enable, self.config.voltz_enable)
        self.assertEqual(m.integration_time, self.config.integration_time)
        self.assertEqual(m.voltage_precision, self.config.voltage_precision)
        self.assertEqual(m.nr_measurements, self.config.nr_measurements)
        self.assertEqual(m.first_axis, self.config.first_axis)
        self.assertEqual(m.second_axis, self.config.second_axis)
        self.assertEqual(m.start_ax1, self.config.start_ax1)
        self.assertEqual(m.end_ax1, self.config.end_ax1)
        self.assertEqual(m.step_ax1, self.config.step_ax1)
        self.assertEqual(m.extra_ax1, self.config.extra_ax1)
        self.assertEqual(m.vel_ax1, self.config.vel_ax1)
        self.assertEqual(m.start_ax2, self.config.start_ax2)
        self.assertEqual(m.end_ax2, self.config.end_ax2)
        self.assertEqual(m.step_ax2, self.config.step_ax2)
        self.assertEqual(m.extra_ax2, self.config.extra_ax2)
        self.assertEqual(m.vel_ax2, self.config.vel_ax2)
        self.assertEqual(m.start_ax3, self.config.start_ax3)
        self.assertEqual(m.end_ax3, self.config.end_ax3)
        self.assertEqual(m.step_ax3, self.config.step_ax3)
        self.assertEqual(m.extra_ax3, self.config.extra_ax3)
        self.assertEqual(m.vel_ax3, self.config.vel_ax3)
        self.assertEqual(m.start_ax5, self.config.start_ax5)
        self.assertEqual(m.end_ax5, self.config.end_ax5)
        self.assertEqual(m.step_ax5, self.config.step_ax5)
        self.assertEqual(m.extra_ax5, self.config.extra_ax5)
        self.assertEqual(m.vel_ax5, self.config.vel_ax5)

    def test_valid_data(self):
        mnf = configuration.MeasurementConfig()
        self.assertFalse(mnf.valid_data())

        mwf = configuration.MeasurementConfig(self.filename)
        self.assertTrue(mwf.valid_data())

        mwf.voltx_enable = None
        self.assertFalse(mwf.valid_data())

    def test_clear(self):
        m = configuration.MeasurementConfig(self.filename)
        self.assertTrue(m.valid_data())

        m.clear()
        self.assertIsNone(m.magnet_name)
        self.assertIsNone(m.main_current)
        self.assertIsNone(m.probe_name)
        self.assertIsNone(m.temperature)
        self.assertIsNone(m.operator)
        self.assertIsNone(m.voltx_enable)
        self.assertIsNone(m.volty_enable)
        self.assertIsNone(m.voltz_enable)
        self.assertIsNone(m.integration_time)
        self.assertIsNone(m.voltage_precision)
        self.assertIsNone(m.nr_measurements)
        self.assertIsNone(m.first_axis)
        self.assertIsNone(m.second_axis)
        self.assertIsNone(m.start_ax1)
        self.assertIsNone(m.end_ax1)
        self.assertIsNone(m.step_ax1)
        self.assertIsNone(m.extra_ax1)
        self.assertIsNone(m.vel_ax1)
        self.assertIsNone(m.start_ax2)
        self.assertIsNone(m.end_ax2)
        self.assertIsNone(m.step_ax2)
        self.assertIsNone(m.extra_ax2)
        self.assertIsNone(m.vel_ax2)
        self.assertIsNone(m.start_ax3)
        self.assertIsNone(m.end_ax3)
        self.assertIsNone(m.step_ax3)
        self.assertIsNone(m.extra_ax3)
        self.assertIsNone(m.vel_ax3)
        self.assertIsNone(m.start_ax5)
        self.assertIsNone(m.end_ax5)
        self.assertIsNone(m.step_ax5)
        self.assertIsNone(m.extra_ax5)
        self.assertIsNone(m.vel_ax5)

    def test_save_file(self):
        filename = 'measurement_configuration_tmp.txt'
        filename = os.path.join(self.base_directory, filename)
        mw = configuration.MeasurementConfig()
        mw.magnet_name = 'a'
        mw.main_current = 'b'
        mw.probe_name = 'c'
        mw.temperature = 'd'
        mw.operator = 'e'
        mw.voltx_enable = 0
        mw.volty_enable = 0
        mw.voltz_enable = 0
        mw.integration_time = 1
        mw.nr_measurements = 2
        mw.voltage_precision = 1
        mw.first_axis = 5
        mw.second_axis = -1
        mw.start_ax1 = 7
        mw.end_ax1 = 7
        mw.step_ax1 = 9
        mw.extra_ax1 = 0
        mw.vel_ax1 = 10
        mw.start_ax2 = 11
        mw.end_ax2 = 12
        mw.step_ax2 = 13
        mw.extra_ax2 = 0
        mw.vel_ax2 = 14
        mw.start_ax3 = 15
        mw.end_ax3 = 16
        mw.step_ax3 = 17
        mw.extra_ax3 = 0
        mw.vel_ax3 = 18
        mw.start_ax5 = 19
        mw.end_ax5 = 20
        mw.step_ax5 = 21
        mw.extra_ax5 = 0
        mw.vel_ax5 = 22
        mw.save_file(filename)

        mr = configuration.MeasurementConfig(filename)
        self.assertEqual(mr.magnet_name, mw.magnet_name)
        self.assertEqual(mr.main_current, mw.main_current)
        self.assertEqual(mr.probe_name, mw.probe_name)
        self.assertEqual(mr.temperature, mw.temperature)
        self.assertEqual(mr.operator, mw.operator)
        self.assertEqual(mr.voltx_enable, mw.voltx_enable)
        self.assertEqual(mr.volty_enable, mw.volty_enable)
        self.assertEqual(mr.voltz_enable, mw.voltz_enable)
        self.assertEqual(mr.integration_time, mw.integration_time)
        self.assertEqual(mr.voltage_precision, mw.voltage_precision)
        self.assertEqual(mr.nr_measurements, mw.nr_measurements)
        self.assertEqual(mr.first_axis, mw.first_axis)
        self.assertEqual(mr.second_axis, mw.second_axis)
        self.assertEqual(mr.start_ax1, mw.start_ax1)
        self.assertEqual(mr.end_ax1, mw.end_ax1)
        self.assertEqual(mr.step_ax1, mw.step_ax1)
        self.assertEqual(mr.extra_ax1, mw.extra_ax1)
        self.assertEqual(mr.vel_ax1, mw.vel_ax1)
        self.assertEqual(mr.start_ax2, mw.start_ax2)
        self.assertEqual(mr.end_ax2, mw.end_ax2)
        self.assertEqual(mr.step_ax2, mw.step_ax2)
        self.assertEqual(mr.extra_ax2, mw.extra_ax2)
        self.assertEqual(mr.vel_ax2, mw.vel_ax2)
        self.assertEqual(mr.start_ax3, mw.start_ax3)
        self.assertEqual(mr.end_ax3, mw.end_ax3)
        self.assertEqual(mr.step_ax3, mw.step_ax3)
        self.assertEqual(mr.extra_ax3, mw.extra_ax3)
        self.assertEqual(mr.vel_ax3, mw.vel_ax3)
        self.assertEqual(mr.start_ax5, mw.start_ax5)
        self.assertEqual(mr.end_ax5, mw.end_ax5)
        self.assertEqual(mr.step_ax5, mw.step_ax5)
        self.assertEqual(mr.extra_ax5, mw.extra_ax5)
        self.assertEqual(mr.vel_ax5, mw.vel_ax5)
        os.remove(filename)

    def test_save_file_raise_exception(self):
        filename = 'measurement_configuration_tmp.txt'
        filename = os.path.join(self.base_directory, filename)
        m = configuration.MeasurementConfig()
        with self.assertRaises(configuration.ConfigurationError):
            m.save_file(filename)

    def test_database_table_name(self):
        c = configuration.MeasurementConfig()
        tn = configuration.MeasurementConfig.database_table_name()
        self.assertEqual(tn, c._db_table)

    def test_database_functions(self):
        success = configuration.MeasurementConfig.create_database_table(
            self.database)
        self.assertTrue(success)

        mw = configuration.MeasurementConfig(self.filename)
        idn = mw.save_to_database(self.database)
        self.assertIsNotNone(idn)

        mr = configuration.MeasurementConfig()
        mr.read_from_database(self.database, idn)
        self.assertEqual(mr, mw)

        mr = configuration.MeasurementConfig(database=self.database, idn=idn)
        self.assertEqual(mr, mw)

        os.remove(self.database)
