# -*- coding: utf-8 -*-

"""Recover data dialog for the Hall Bench Control application."""

import os.path as _path
from PyQt5.QtWidgets import (
    QDialog as _QDialog,
    QTableWidgetItem as _QTableWidgetItem,
    QFileDialog as _QFileDialog,
    QMessageBox as _QMessageBox,
    )
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.savefieldmapdialog import SaveFieldMapDialog \
    as _SaveFieldMapDialog
from hallbench.data.measurement import VoltageData as _VoltageData
from hallbench.data.measurement import FieldData as _FieldData
from hallbench.data.measurement import FieldMapData as _FieldMapData
from hallbench.data.calibration import ProbeCalibration


class RecoverDataDialog(_QDialog):
    """Recover data dialog class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Setup the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(__file__, self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialization
        self.directory = ''
        self.probe_calibration = None
        self.fieldmap_data = None
        self.filenames = []

        # create save dialog widget
        self.save_dialog = _SaveFieldMapDialog()

        # create connections
        self.ui.calibration_btn.clicked.connect(self.loadCalibrationFile)
        self.ui.addfile_btn.clicked.connect(self.addFileToList)
        self.ui.removefile_btn.clicked.connect(self.removeFileFromList)
        self.ui.clearfiles_btn.clicked.connect(self.clearFileList)
        self.ui.loadandsave_btn.clicked.connect(self.loadDataAndSaveFieldMap)

    def addFileToList(self):
        """Add file to data file list."""
        filenames = _QFileDialog.getOpenFileNames(
            directory=self.directory, filter="Text files (*.txt)")

        if isinstance(filenames, tuple):
            filenames = filenames[0]

        if len(filenames) == 0:
            return

        for filename in filenames:
            self.filenames.append(filename)

        self.directory = _path.split(self.filenames[0])[0]

        self.ui.filenames_ta.clear()
        self.ui.filenames_ta.setRowCount(len(self.filenames))
        self.ui.filecount_sb.setValue(len(self.filenames))

        for i in range(len(self.filenames)):
            item = _QTableWidgetItem()
            item.setText(self.filenames[i])
            self.ui.filenames_ta.setItem(i, 0, item)

    def clearFileList(self):
        """Clear filename list."""
        self.filenames_ta.clear()
        self.filenames = []
        self.filenames_ta.setRowCount(len(self.filenames))
        self.ui.filecount_sb.setValue(len(self.filenames))

    def removeFileFromList(self):
        """Remove file from filename list."""
        items_to_remove = self.ui.filenames_ta.selectedItems()

        if len(items_to_remove) != 0:
            for idx in items_to_remove:
                self.ui.filenames_ta.removeRow(idx.row())

        self.filenames = []
        for idx in range(self.ui.filenames_ta.rowCount()):
            if self.ui.filenames_ta.item(idx, 0):
                self.filenames.append(
                    self.ui.filenames_ta.item(idx, 0).text())

        self.ui.filecount_sb.setValue(len(self.filenames))

    def loadCalibrationFile(self):
        """Load probe calibration file."""
        default_filename = self.ui.calibration_le.text()
        if default_filename == '':
            default_filename = self.directory

        filename = _QFileDialog.getOpenFileName(
            self, caption='Open probe calibration file',
            directory=default_filename, filter="Text files (*.txt)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            self.probe_calibration = ProbeCalibration(filename)
            self.ui.calibration_le.setText(filename)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)
            return

    def loadDataAndSaveFieldMap(self):
        """Load data from files and open save fieldmap dialog."""
        if len(self.filenames) == 0:
            message = 'Empty filename list.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        if self.probe_calibration is None:
            message = 'Invalid calibration data.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        self.fieldmap_data = _FieldMapData()
        self.fieldmap_data.probe_calibration = self.probe_calibration

        if self.ui.voltage_rb.isChecked():
            voltage_data_list = []
            for filename in self.filenames:
                try:
                    voltage_data_list.append(_VoltageData(filename))
                except Exception:
                    message = 'Invalid voltage data file: %s' % filename
                    _QMessageBox.critical(
                        self, 'Failure', message, _QMessageBox.Ok)
                    return
            self.fieldmap_data.voltage_data_list = voltage_data_list
        else:
            field_data_list = []
            for filename in self.filenames:
                try:
                    field_data_list.append(_FieldData(filename))
                except Exception:
                    message = 'Invalid field data file: %s' % filename
                    _QMessageBox.critical(
                        self, 'Failure', message, _QMessageBox.Ok)
                    return
            self.fieldmap_data.field_data_list = field_data_list

        self.showSaveFieldMapDialog()

    def show(self, directory=''):
        """Update directory variable and show dialog."""
        self.directory = directory
        super(RecoverDataDialog, self).show()

    def showSaveFieldMapDialog(self):
        """Open save fieldmap dialog."""
        if self.fieldmap_data is None:
            message = 'Invalid field map data.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        self.save_dialog.show(self.fieldmap_data, self.directory)
