# -*- coding: utf-8 -*-

"""Main entry poin to the Hall bench control application."""

import os as _os
import sys as _sys
import threading as _threading
from qtpy.QtWidgets import QApplication as _QApplication

from hallbench.gui import utils as _utils
from hallbench.gui.hallbenchwindow import HallBenchWindow as _HallBenchWindow
from hallbench.gui.viewprobedialog import ViewProbeDialog \
    as _ViewProbeDialog
from hallbench.gui.viewscandialog import ViewScanDialog \
    as _ViewScanDialog
from hallbench.gui.savefieldscandialog import SaveFieldScanDialog \
    as _SaveFieldScanDialog
from hallbench.gui.savefieldmapdialog import SaveFieldmapDialog \
    as _SaveFieldmapDialog
from hallbench.gui.viewfieldmapdialog import ViewFieldmapDialog \
    as _ViewFieldmapDialog
from hallbench.gui.auxiliarywidgets import CyclingTablePlotDialog \
    as _CyclingTablePlotDialog
import hallbench.data as _data


class HallBenchApp(_QApplication):
    """Hall bench application."""

    def __init__(self, args):
        """Start application."""
        super().__init__(args)
        self.setStyle(_utils.WINDOW_STYLE)

        self.directory = _utils.BASEPATH
        self.database_name = _utils.DATABASE_NAME
        self.mongo = _utils.MONGO
        self.server = _utils.SERVER
        self.create_database()

        # positions dict
        self.positions = {}
        self.current_max = 0
        self.current_min = 0

        # create dialogs
        self.view_probe_dialog = _ViewProbeDialog()
        self.view_scan_dialog = _ViewScanDialog()
        self.save_field_scan_dialog = _SaveFieldScanDialog()
        self.save_fieldmap_dialog = _SaveFieldmapDialog()
        self.view_fieldmap_dialog = _ViewFieldmapDialog()
        self.cycling_dialog = _CyclingTablePlotDialog()

    def create_database(self):
        """Create database and tables."""
        _ConnectionConfig = _data.configuration.ConnectionConfig(
            database_name=self.database_name,
            mongo=self.mongo, server=self.server)
        _PowerSupplyConfig = _data.configuration.PowerSupplyConfig(
            database_name=self.database_name,
            mongo=self.mongo, server=self.server)
        _CyclingCurve = _data.configuration.CyclingCurve(
            database_name=self.database_name,
            mongo=self.mongo, server=self.server)
        _HallCalibrationCurve = _data.calibration.HallCalibrationCurve(
            database_name=self.database_name,
            mongo=self.mongo, server=self.server)
        _HallProbePositions = _data.calibration.HallProbePositions(
            database_name=self.database_name,
            mongo=self.mongo, server=self.server)
        _MeasurementConfig = _data.configuration.MeasurementConfig(
            database_name=self.database_name,
            mongo=self.mongo, server=self.server)
        _IntegratorMeasurementConfig = (
            _data.configuration.IntegratorMeasurementConfig(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server))
        _NMRMeasurementConfig = (
            _data.configuration.NMRMeasurementConfig(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server))
        _VoltageScan = _data.measurement.VoltageScan(
            database_name=self.database_name,
            mongo=self.mongo, server=self.server)
        _FieldScan = _data.measurement.FieldScan(
            database_name=self.database_name,
            mongo=self.mongo, server=self.server)
        _Fieldmap = _data.measurement.Fieldmap(
            database_name=self.database_name,
            mongo=self.mongo, server=self.server)
 
        status = []
        status.append(_ConnectionConfig.db_create_collection())
        status.append(_PowerSupplyConfig.db_create_collection())
        status.append(_CyclingCurve.db_create_collection())
        status.append(_HallCalibrationCurve.db_create_collection())
        status.append(_HallProbePositions.db_create_collection())
        status.append(_MeasurementConfig.db_create_collection())
        status.append(_IntegratorMeasurementConfig.db_create_collection())
        status.append(_NMRMeasurementConfig.db_create_collection())
        status.append(_VoltageScan.db_create_collection())
        status.append(_FieldScan.db_create_collection())
        status.append(_Fieldmap.db_create_collection())
        if not all(status):
            raise Exception("Failed to create database.")


class GUIThread(_threading.Thread):
    """GUI Thread."""

    def __init__(self):
        """Start thread."""
        _threading.Thread.__init__(self)
        self.app = None
        self.window = None
        self.daemon = True
        self.start()

    def run(self):
        """Thread target function."""
        self.app = None
        if not _QApplication.instance():
            self.app = HallBenchApp([])
            self.window = _HallBenchWindow(
                width=_utils.WINDOW_WIDTH, height=_utils.WINDOW_HEIGHT)
            self.window.show()
            self.window.centralize_window()
            _sys.exit(self.app.exec_())


def run():
    """Run hallbench application."""
    app = None
    if not _QApplication.instance():
        app = HallBenchApp([])
        window = _HallBenchWindow(
            width=_utils.WINDOW_WIDTH, height=_utils.WINDOW_HEIGHT)
        window.show()
        window.centralize_window()
        _sys.exit(app.exec_())


def run_in_thread():
    """Run hallbench application in a thread."""
    return GUIThread()
