# -*- coding: utf-8 -*-

"""Main entry poin to the Hall bench control application."""

import os as _os
import sys as _sys
import threading as _threading
from PyQt4.QtGui import QApplication as _QApplication

from hallbench.gui.hallbenchwindow import HallBenchWindow as _HallBenchWindow
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
_VoltageData = _data.measurement.VoltageData
_FieldData = _data.measurement.FieldData
_Fieldmap = _data.measurement.Fieldmap
_HallBenchDevices = _devices.devices.HallBenchDevices


class HallBenchApp(_QApplication):
    """Hall bench application."""

    def __init__(self, args):
        """Start application."""
        super().__init__(args)
        self.setStyle(_style)

        _default_directory = _os.path.dirname(_os.path.dirname(
            _os.path.dirname(_os.path.abspath(__file__))))
        self.database = _os.path.join(_default_directory, _database_filename)
        self.create_database()

        self.connection_config = _ConnectionConfig()
        self.measurement_config = _MeasurementConfig()
        self.power_supply_config = _PowerSupplyConfig()
        self.hall_probe = _HallProbe()
        self.devices = _HallBenchDevices()

    def create_database(self):
        """Create database and tables."""
        status = []
        status.append(_ConnectionConfig.create_database_table(self.database))
        status.append(_HallSensor.create_database_table(self.database))
        status.append(_HallProbe.create_database_table(self.database))
        status.append(_MeasurementConfig.create_database_table(self.database))
        status.append(_VoltageData.create_database_table(self.database))
        status.append(_FieldData.create_database_table(self.database))
        status.append(_Fieldmap.create_database_table(self.database))
        if not all(status):
            raise Exception("Fail to create database.")


class GUIThread(_threading.Thread):
    """GUI Thread."""

    def __init__(self, daemon=True):
        """Start thread."""
        _threading.Thread.__init__(self, daemon=daemon)
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


def run_in_thread(daemon):
    """Run hallbench application in a thread."""
    return GUIThread(daemon)
