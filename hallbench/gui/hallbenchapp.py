# -*- coding: utf-8 -*-

"""Main entry poin to the Hall bench control application."""

import os as _os
import sys as _sys
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
import hallbench.data as _data
import hallbench.devices as _devices


# Style: ["windows", "motif", "cde", "plastique", "windowsxp", or "macintosh"]
_style = 'windows'
_width = 1200
_height = 700
_database_filename = 'hall_bench_measurements.db'


_ConnectionConfig = _data.configuration.ConnectionConfig
_MeasurementConfig = _data.configuration.MeasurementConfig
_PowerSupplyConfig = _data.configuration.PowerSupplyConfig
_HallSensor = _data.calibration.HallSensor
_HallProbe = _data.calibration.HallProbe
_VoltageScan = _data.measurement.VoltageScan
_FieldScan = _data.measurement.FieldScan
_Fieldmap = _data.measurement.Fieldmap
_HallBenchDevices = _devices.devices.HallBenchDevices


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
        self.connection_config = _ConnectionConfig()
        self.measurement_config = _MeasurementConfig()
        self.power_supply_config = _PowerSupplyConfig()
        self.hall_probe = _HallProbe()
        self.devices = _HallBenchDevices()
        
        # positions dict
        self.positions = {}
        
        # create dialogs
        self.view_probe_dialog = _ViewProbeDialog()
        self.view_scan_dialog = _ViewScanDialog()
        self.save_fieldmap_dialog = _SaveFieldmapDialog()
        self.view_fieldmap_dialog = _ViewFieldmapDialog()
        

    def create_database(self):
        """Create database and tables."""
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
            self.window.centralizeWindow()
            _sys.exit(self.app.exec_())


def run():
    """Run hallbench application."""
    app = None
    if (not _QApplication.instance()):
        app = HallBenchApp([])
        window = _HallBenchWindow(width=_width, height=_height)
        window.show()
        window.centralizeWindow()
        _sys.exit(app.exec_())


def run_in_thread():
    """Run hallbench application in a thread."""
    return GUIThread()
