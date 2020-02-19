# -*- coding: utf-8 -*-

"""Save fieldmap dialog for the Hall Bench Control application."""

import os as _os
import sys as _sys
import json as _json
import traceback as _traceback
from qtpy.QtCore import Qt as _Qt
from qtpy.QtWidgets import (
    QApplication as _QApplication,
    QDialog as _QDialog,
    QFileDialog as _QFileDialog,
    QLineEdit as _QLineEdit,
    QMessageBox as _QMessageBox,
    )
import qtpy.uic as _uic

from hallbench.gui.utils import get_ui_file as _get_ui_file
import hallbench.data as _data
from hallbench.gui.databasewidget import _HallCalibrationCurve
from hallbench.data import calibration


class SaveFieldScanDialog(_QDialog):
    """Save fieldmap dialog class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _get_ui_file(self)
        self.ui = _uic.loadUi(uifile, self)

        self.voltage_scan_id_list = None
        self.field_scan_id_list = None

        self.connect_signal_slots()

    @property
    def database_name(self):
        """Database name."""
        return _QApplication.instance().database_name

    @property
    def mongo(self):
        """MongoDB database."""
        return _QApplication.instance().mongo

    @property
    def server(self):
        """Server for MongoDB database."""
        return _QApplication.instance().server

    @property
    def directory(self):
        """Return the default directory."""
        return _QApplication.instance().directory

    def accept(self):
        """Close dialog."""
        self.clear()
        super().accept()

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.clear()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def clear(self):
        """Clear data."""
        self.voltage_scan_id_list = None
        self.field_scan_id_list = None
        self.ui.cmb_calibrationx.setCurrentIndex(-1)
        self.ui.cmb_calibrationy.setCurrentIndex(-1)
        self.ui.cmb_calibrationz.setCurrentIndex(-1)
        self.ui.le_nr_voltage_scans.setText('')
        self.ui.te_voltage_scan_id_list.setPlainText('')
        self.disable_save_to_file()

    def connect_signal_slots(self):
        """Create signal/slot connections."""
        self.ui.pbt_savedb.clicked.connect(self.save_to_db)
        self.ui.pbt_savefile.clicked.connect(self.save_to_file)

        self.ui.cmb_calibrationx.currentIndexChanged.connect(
            self.disable_save_to_file)
        self.ui.cmb_calibrationy.currentIndexChanged.connect(
            self.disable_save_to_file)
        self.ui.cmb_calibrationz.currentIndexChanged.connect(
            self.disable_save_to_file)
        self.ui.rbt_ignore_offsets.clicked.connect(self.disable_save_to_file)
        self.ui.rbt_configure_offsets.clicked.connect(
            self.disable_save_to_file)
        self.ui.rbt_measure_offsets.clicked.connect(self.disable_save_to_file)
        self.ui.le_offsetx.editingFinished.connect(self.disable_save_to_file)
        self.ui.le_offsety.editingFinished.connect(self.disable_save_to_file)
        self.ui.le_offsetz.editingFinished.connect(self.disable_save_to_file)
        self.ui.le_offset_range.editingFinished.connect(
            self.disable_save_to_file)

        self.ui.rbt_ignore_offsets.toggled.connect(self.change_offset_page)
        self.ui.rbt_configure_offsets.toggled.connect(self.change_offset_page)
        self.ui.rbt_measure_offsets.toggled.connect(self.change_offset_page)

    def change_offset_page(self):
        """Change offset stacked widget page."""
        try:
            if self.ui.rbt_measure_offsets.isChecked():
                self.ui.stw_offsets.setCurrentIndex(2)
            elif self.ui.rbt_configure_offsets.isChecked():
                self.ui.stw_offsets.setCurrentIndex(1)
            else:
                self.ui.stw_offsets.setCurrentIndex(0)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def disable_save_to_file(self):
        """Disable save to file button."""
        self.ui.pbt_savefile.setEnabled(False)

    def save_to_db(self):
        """Save field scan to database."""
        if self.database_name is None:
            msg = 'Invalid database filename.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        if (self.voltage_scan_id_list is None or
           len(self.voltage_scan_id_list) == 0):
            msg = 'Invalid list of voltage scan IDs.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        self.blockSignals(True)
        _QApplication.setOverrideCursor(_Qt.WaitCursor)

        try:
            vs = _data.measurement.VoltageScan(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
  
            calx = _data.calibration.HallCalibrationCurve(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)

            caly = _data.calibration.HallCalibrationCurve(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
 
            calz = _data.calibration.HallCalibrationCurve(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
 
            mc = _data.configuration.MeasurementConfig(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
 
            voltage_scan_list = []
            for idn in self.voltage_scan_id_list:
                vs.clear()
                vs.db_read(idn)
                voltage_scan_list.append(vs.copy())                 

            calx.update_calibration(self.ui.cmb_calibrationx.currentText())
            caly.update_calibration(self.ui.cmb_calibrationy.currentText())
            calz.update_calibration(self.ui.cmb_calibrationz.currentText())
  
            if self.ui.rbt_ignore_offsets.isChecked():
                voltage_offset = 'ignore'
            elif self.ui.rbt_configure_offsets.isChecked():
                voltage_offset = 'configure'
            else:
                voltage_offset = 'measure'
            
            offsetx = float(self.ui.le_offsetx.text())/1000
            offsety = float(self.ui.le_offsety.text())/1000
            offsetz = float(self.ui.le_offsetz.text())/1000
            offset_range = float(self.ui.le_offset_range.text())
            
            for vs in voltage_scan_list:
                vs = _data.measurement.configure_voltage_offset(
                    vs, voltage_offset, 
                    offsetx, offsety, offsetz, offset_range)
  
            field_scan_list = _data.measurement.get_field_scan_list(
                voltage_scan_list, calx, caly, calz)
   
            self.field_scan_id_list = []
            for i in range(len(field_scan_list)):
                fs = field_scan_list[i]
                fs.db_update_database(
                    database_name=self.database_name,
                    mongo=self.mongo, server=self.server)
                mc.clear()
                mc.db_read(fs.configuration_id)
                mc.calibrationx = calx.calibration_name
                mc.calibrationy = caly.calibration_name
                mc.calibrationz = calz.calibration_name
                mc.voltage_offset = voltage_offset
                mc.offsetx = offsetx
                mc.offsety = offsety
                mc.offsetz = offsetz
                mc.offset_range = offset_range
                # configuration_id = mc.db_save()
                fs.configuration_id = None #configuration_id
                self.field_scan_id_list.append(fs.db_save())
  
            self.ui.pbt_savefile.setEnabled(True)
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()

            if len(self.field_scan_id_list) == 1:
                msg = 'Field scan saved to database table.\nID: ' + str(
                    self.field_scan_id_list[0])
            else:
                msg = 'Field scans saved to database table.\nIDs: ' + str(
                    self.field_scan_id_list)
            _QMessageBox.information(self, 'Information', msg, _QMessageBox.Ok)

        except Exception:
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save field scan to database.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def save_to_file(self):
        """Save field scan to database."""
        if self.field_scan_id_list is None:
            return

        try:
            fs = _data.measurement.FieldScan(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
            
            if len(self.field_scan_id_list) == 1:
                idn = self.field_scan_id_list[0]
                fs.db_read(idn)
                default_filename = fs.default_filename
                filename = _QFileDialog.getSaveFileName(
                    self, caption='Save field scan file',
                    directory=_os.path.join(self.directory, default_filename),
                    filter="Text files (*.txt *.dat)")

                if isinstance(filename, tuple):
                    filename = filename[0]

                if len(filename) == 0:
                    return

                if (not filename.endswith('.txt') and
                    not filename.endswith('.dat')):
                    filename = filename + '.dat'

                fs.save_file(filename)
                msg = 'Field scan saved to file.'

            else:
                directory = _QFileDialog.getExistingDirectory(
                    self, caption='Save field scan files',
                    directory=self.directory)

                if isinstance(directory, tuple):
                    directory = directory[0]

                if len(directory) == 0:
                    return

                for idn in self.field_scan_id_list:
                    fs.clear()
                    fs.db_read(idn)
                    filename = _os.path.join(directory, fs.default_filename)
                    fs.save_file(filename)

                msg = 'Field scan saved to files.'

            _QMessageBox.information(self, 'Information', msg, _QMessageBox.Ok)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save field scan to file.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def show(self, voltage_scan_id_list):
        """Show dialog."""
        if voltage_scan_id_list is None or len(voltage_scan_id_list) == 0:
            msg = 'Invalid voltage scan ID list.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        try:
            self.voltage_scan_id_list = voltage_scan_id_list
            
            self.ui.le_nr_voltage_scans.setText(
                str(len(self.voltage_scan_id_list)))
            
            text = ','.join([str(idn) for idn in self.voltage_scan_id_list])
            self.ui.te_voltage_scan_id_list.setPlainText(text)
            
            self.field_scan_id_list = []
            self.update_probe_calibrations()
            self.disable_save_to_file()
            super().show()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def update_probe_calibrations(self):
        try:
            calibration = _HallCalibrationCurve(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
            
            calibration_names = calibration.get_calibration_list()
        
            calx_name = self.ui.cmb_calibrationx.currentText()
            caly_name = self.ui.cmb_calibrationy.currentText()
            calz_name = self.ui.cmb_calibrationz.currentText()
            
            self.ui.cmb_calibrationx.clear()
            self.ui.cmb_calibrationy.clear()
            self.ui.cmb_calibrationz.clear()
            
            self.ui.cmb_calibrationx.addItems([cn for cn in calibration_names])
            self.ui.cmb_calibrationy.addItems([cn for cn in calibration_names])
            self.ui.cmb_calibrationz.addItems([cn for cn in calibration_names])
            
            if len(calx_name) == 0:
                self.ui.cmb_calibrationx.setCurrentIndex(-1)
            else:
                self.ui.cmb_calibrationx.setCurrentText(calx_name)

            if len(caly_name) == 0:
                self.ui.cmb_calibrationy.setCurrentIndex(-1)
            else:
                self.ui.cmb_calibrationy.setCurrentText(caly_name)

            if len(calz_name) == 0:
                self.ui.cmb_calibrationz.setCurrentIndex(-1)
            else:
                self.ui.cmb_calibrationz.setCurrentText(calz_name)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)