# -*- coding: utf-8 -*-

"""Main entry poin to the Hall bench control application."""

import os as _os
import sys as _sys
import numpy as _np
import time as _time
import threading as _threading
from qtpy.QtWidgets import QApplication as _QApplication

from hallbench.gui.hallbenchwindow import HallBenchWindow as _HallBenchWindow
from hallbench.gui.viewprobedialog import ViewProbeDialog \
    as _ViewProbeDialog
from hallbench.gui.viewscandialog import ViewScanDialog \
    as _ViewScanDialog
from hallbench.gui.savefieldmapdialog import SaveFieldmapDialog \
    as _SaveFieldmapDialog
from hallbench.gui.viewfieldmapdialog import ViewFieldmapDialog \
    as _ViewFieldmapDialog

from imadevices import Agilent3458ALib as _Agilent3458ALib
from imadevices import F1000DRSLib as _DRSLib
from imadevices import PmacLib as _PmacLib
from imadevices import Agilent34401ALib as _Agilent34401ALib
from imadevices import Agilent34970ALib as _Agilent34970ALib
from imadevices import ElcomatLib as _ElcomatLib
from imadevices import NMRLib as _NMRLib
from imadevices import UDCLib as _UDCLib

import hallbench.data as _data


# Styles: ["windows", "motif", "cde", "plastique", "windowsxp", or "macintosh"]
_style = 'windows'
_width = 1200
_height = 700
_database_filename = 'hall_bench_measurements.db'
_logs_dir = 'logs'

# Devices
Pmac = _PmacLib.Pmac
NMR = _NMRLib.NMRSerial
Elcomat = _ElcomatLib.ElcomatSerial
PowerSupply = _DRSLib.SerialDRS_FBP
UDC = _UDCLib.UDCModBus


class Multimeter(_Agilent3458ALib.Agilent3458AGPIB):
    """Multimeter class."""

    def configure(self, aper, mrange):
        """Configure multimeter.

        Args:
            aper (float): A/D converter integration time in ms.
            mrange (float): measurement range in volts.
        """
        self.send_command(self.commands.reset)
        self.send_command(self.commands.func_volt)
        self.send_command(self.commands.tarm_auto)
        self.send_command(self.commands.trig_auto)
        self.send_command(self.commands.nrdgs_ext)
        self.send_command(self.commands.arange_off)
        self.send_command(self.commands.fixedz_on)
        self.send_command(self.commands.range + str(mrange))
        self.send_command(self.commands.math_off)
        self.send_command(self.commands.azero_once)
        self.send_command(self.commands.trig_buffer_off)
        self.send_command(self.commands.delay_0)
        self.send_command(
            self.commands.aper + '{0:.10f}'.format(aper/1000))
        self.send_command(self.commands.disp_off)
        self.send_command(self.commands.scratch)
        self.send_command(self.commands.end_gpib_always)
        self.send_command(self.commands.mem_fifo)

    def configure_reading_format(self, formtype):
        """Configure multimeter reading format.

        Args:
            formtype (int): format type [SREAL=4, DREAL=5].
        """
        self.send_command(self.commands.mem_fifo)
        if formtype == 4:
            self.send_command(self.commands.oformat_sreal)
            self.send_command(self.commands.mformat_sreal)
        elif formtype == 5:
            self.send_command(self.commands.oformat_dreal)
            self.send_command(self.commands.mformat_dreal)


class Multichannel(_Agilent34970ALib.Agilent34970AGPIB):
    """Multichannel class."""

    def __init__(self, logfile=None):
        """Initiaze variables and prepare logging file.

        Args:
            logfile (str): log file path.
        """
        super().__init__(logfile)
        self.temperature_channels = [
            '201', '202', '203', '204', '205', '206', '207', '208', '209']
        self.voltage_channels = ['101', '102', '103', '105']

    def get_converted_readings(self, wait=0.5):
        """Get multichannel reading list and convert voltage values."""
        try:
            self.send_command(self.commands.qread)
            _time.sleep(wait)
            rstr = self.read_from_device()

            if len(rstr) != 0:
                rlist = [float(r) for r in rstr.split(',')]
                conv_rlist = []
                for i in range(len(rlist)):
                    ch = self._config_channels[i]
                    rd = rlist[i]
                    if ch in self.voltage_channels:
                        # Voltage to temperature (Transducer 135-14)
                        if ch == '101':
                            temp = (rd + 50e-3)/20e-3
                        elif ch == '102':
                            temp = (rd + 40e-3)/20e-3
                        elif ch == '103':
                            temp = (rd + 50e-3)/20e-3
                        elif ch == '105':
                            temp = (rd + 30e-3)/20e-3
                        else:
                            temp = _np.nan
                        conv_rlist.append(temp)
                    else:
                        conv_rlist.append(rd)
                return conv_rlist
            else:
                return []
        except Exception:
            return []


class DCCT(_Agilent34401ALib.Agilent34401AGPIB):
    """DCCT Multimeter."""

    def read_current(self, dcct_head=None):
        """Read dcct voltage and convert to current."""
        voltage = self.read()
        dcct_heads = [40, 160, 320, 600, 1000, 1125]
        if voltage is not None and dcct_head in dcct_heads:
            current = voltage * dcct_head/10
        else:
            current = _np.nan
        return current


class HallBenchDevices(object):
    """Hall Bench devices class."""

    def __init__(self):
        """Initialize variables and log files."""

        # Log files
        _dir_path = _os.path.dirname(_os.path.dirname(
            _os.path.dirname(_os.path.abspath(__file__))))
        _logs_path = _os.path.join(_dir_path, _logs_dir)
        if not _os.path.isdir(_logs_path):
            _os.mkdir(_logs_path)
        log_pmac = _os.path.join(_logs_path, 'pmac.log')
        log_voltx = _os.path.join(_logs_path, 'voltx.log')
        log_volty = _os.path.join(_logs_path, 'volty.log')
        log_voltz = _os.path.join(_logs_path, 'voltz.log')
        log_multich = _os.path.join(_logs_path, 'multich.log')
        log_nmr = _os.path.join(_logs_path, 'nmr.log')
        log_elcomat = _os.path.join(_logs_path, 'elcomat.log')
        log_dcct = _os.path.join(_logs_path, 'dcct.log')
        log_water_udc = _os.path.join(_logs_path, 'water_udc.log')
        log_air_udc = _os.path.join(_logs_path, 'air_udc.log')

        # Devices
        self.pmac = Pmac(log_pmac)
        self.voltx = Multimeter(log_voltx)
        self.volty = Multimeter(log_volty)
        self.voltz = Multimeter(log_voltz)
        self.multich = Multichannel(log_multich)
        self.nmr = NMR(log_nmr)
        self.elcomat = Elcomat(log_elcomat)
        self.dcct = DCCT(log_dcct)
        self.water_udc = UDC(log_water_udc)
        self.air_udc = UDC(log_air_udc)
        self.ps = PowerSupply()

    def connect(self, config):
        """Connect devices.

        Args:
            config (ConnectionConfig): connection configuration.
        """
        if config.pmac_enable:
            self.pmac.connect()

        if config.voltx_enable:
            self.voltx.connect(config.voltx_address)

        if config.volty_enable:
            self.volty.connect(config.volty_address)

        if config.voltz_enable:
            self.voltz.connect(config.voltz_address)

        if config.multich_enable:
            self.multich.connect(config.multich_address)

        if config.nmr_enable:
            self.nmr.connect(config.nmr_port, config.nmr_baudrate)

        if config.elcomat_enable:
            self.elcomat.connect(config.elcomat_port, config.elcomat_baudrate)

        if config.dcct_enable:
            self.dcct.connect(config.dcct_address)

        if config.water_udc_enable:
            self.water_udc.connect(
                config.water_udc_port,
                config.water_udc_baudrate,
                config.water_udc_slave_address)

        if config.air_udc_enable:
            self.air_udc.connect(
                config.air_udc_port,
                config.air_udc_baudrate,
                config.air_udc_slave_address)

        if config.ps_enable:
            self.ps.Connect(config.ps_port)

    def disconnect(self):
        """Disconnect devices."""
        self.pmac.disconnect()
        self.voltx.disconnect()
        self.volty.disconnect()
        self.voltz.disconnect()
        self.multich.disconnect()
        self.nmr.disconnect()
        self.elcomat.disconnect()
        self.dcct.disconnect()
        self.water_udc.disconnect()
        self.air_udc.disconnect()
        self.ps.Disconnect()


class HallBenchApp(_QApplication):
    """Hall bench application."""

    def __init__(self, args):
        """Start application."""
        super().__init__(args)
        self.setStyle(_style)

        self.directory = _os.path.dirname(_os.path.dirname(
            _os.path.dirname(_os.path.abspath(__file__))))
        self.database = _os.path.join(self.directory, _database_filename)
        self.create_database()

        # configurations
        _ConnectionConfig = _data.configuration.ConnectionConfig
        _PowerSupplyConfig = _data.configuration.PowerSupplyConfig
        _MeasurementConfig = _data.configuration.MeasurementConfig
        _HallProbe = _data.calibration.HallProbe

        self.connection_config = _ConnectionConfig()
        self.measurement_config = _MeasurementConfig()
        self.power_supply_config = _PowerSupplyConfig()
        self.hall_probe = _HallProbe()
        self.devices = HallBenchDevices()

        # positions dict
        self.positions = {}

        # create dialogs
        self.view_probe_dialog = _ViewProbeDialog()
        self.view_scan_dialog = _ViewScanDialog()
        self.save_fieldmap_dialog = _SaveFieldmapDialog()
        self.view_fieldmap_dialog = _ViewFieldmapDialog()

    def create_database(self):
        """Create database and tables."""
        _ConnectionConfig = _data.configuration.ConnectionConfig
        _PowerSupplyConfig = _data.configuration.PowerSupplyConfig
        _HallSensor = _data.calibration.HallSensor
        _HallProbe = _data.calibration.HallProbe
        _MeasurementConfig = _data.configuration.MeasurementConfig
        _VoltageScan = _data.measurement.VoltageScan
        _FieldScan = _data.measurement.FieldScan
        _Fieldmap = _data.measurement.Fieldmap

        status = []
        status.append(_ConnectionConfig.create_database_table(self.database))
        status.append(_PowerSupplyConfig.create_database_table(self.database))
        status.append(_HallSensor.create_database_table(self.database))
        status.append(_HallProbe.create_database_table(self.database))
        status.append(_MeasurementConfig.create_database_table(self.database))
        status.append(_VoltageScan.create_database_table(self.database))
        status.append(_FieldScan.create_database_table(self.database))
        status.append(_Fieldmap.create_database_table(self.database))
        if not all(status):
            raise Exception("Failed to create database.")


class GUIThread(_threading.Thread):
    """GUI Thread."""

    def __init__(self):
        """Start thread."""
        _threading.Thread.__init__(self)
        self.app = None
        self.window = None
        self.start()

    def run(self):
        """Thread target function."""
        self.app = None
        if (not _QApplication.instance()):
            self.app = HallBenchApp([])
            self.window = _HallBenchWindow(width=_width, height=_height)
            self.window.show()
            self.window.centralize_window()
            _sys.exit(self.app.exec_())


def run():
    """Run hallbench application."""
    app = None
    if (not _QApplication.instance()):
        app = HallBenchApp([])
        window = _HallBenchWindow(width=_width, height=_height)
        window.show()
        window.centralize_window()
        _sys.exit(app.exec_())


def run_in_thread():
    """Run hallbench application in a thread."""
    return GUIThread()
