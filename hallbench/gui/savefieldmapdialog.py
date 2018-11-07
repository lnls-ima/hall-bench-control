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

from hallbench.gui.utils import getUiFile as _getUiFile
import hallbench.data.magnets_info as _magnets_info
from hallbench.data.measurement import Fieldmap as _Fieldmap


class SaveFieldmapDialog(_QDialog):
    """Save fieldmap dialog class for the Hall Bench Control application."""

    _coil_list = ['main', 'trim', 'ch', 'cv', 'qs']

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialisation
        self.fieldmap_id_list = None
        self.fieldmap_list = None
        self.field_scan_list = None
        self.field_scan_id_list = None
        self.local_hall_probe = None

        # add predefined magnet names
        names = _magnets_info.get_magnets_name()
        for name in names:
            self.ui.predefined_cmb.addItem(name)

        self.disableInvalidAxes()

        # create connections
        self.ui.main_chb.stateChanged.connect(lambda: self.setCoilFrameEnabled(
            self.ui.main_chb, self.ui.main_fm))

        self.ui.trim_chb.stateChanged.connect(lambda: self.setCoilFrameEnabled(
            self.ui.trim_chb, self.ui.trim_fm))

        self.ui.ch_chb.stateChanged.connect(lambda: self.setCoilFrameEnabled(
            self.ui.ch_chb, self.ui.ch_fm))

        self.ui.cv_chb.stateChanged.connect(lambda: self.setCoilFrameEnabled(
            self.ui.cv_chb, self.ui.cv_fm))

        self.ui.qs_chb.stateChanged.connect(lambda: self.setCoilFrameEnabled(
            self.ui.qs_chb, self.ui.qs_fm))

        self.ui.centerpos3_sb.valueChanged.connect(self.disableSaveToFile)
        self.ui.centerpos2_sb.valueChanged.connect(self.disableSaveToFile)
        self.ui.centerpos1_sb.valueChanged.connect(self.disableSaveToFile)
        self.ui.magnet_x_axis_cmb.currentIndexChanged.connect(
            self.disableSaveToFile)
        self.ui.magnet_y_axis_cmb.currentIndexChanged.connect(
            self.disableSaveToFile)
        self.ui.predefined_cmb.currentIndexChanged.connect(
            self.disableSaveToFile)
        self.ui.magnet_name_le.editingFinished.connect(self.disableSaveToFile)
        self.ui.gap_le.editingFinished.connect(self.disableSaveToFile)
        self.ui.control_gap_le.editingFinished.connect(self.disableSaveToFile)
        self.ui.magnet_length_le.editingFinished.connect(self.disableSaveToFile)
        self.ui.comments_te.textChanged.connect(self.disableSaveToFile)
        self.ui.correct_displacements_chb.stateChanged.connect(
            self.disableSaveToFile)

        for coil in self._coil_list:
            turns_le = getattr(self.ui, 'nr_turns_' + coil + '_le')
            turns_le.editingFinished.connect(self.disableSaveToFile)
            current_le = getattr(self.ui, 'current_' + coil + '_le')
            current_le.editingFinished.connect(self.disableSaveToFile)

        self.ui.clear_btn.clicked.connect(self.clearInfo)
        self.ui.predefined_cmb.currentIndexChanged.connect(self.loadMagnetInfo)
        self.ui.magnet_x_axis_cmb.currentIndexChanged.connect(
            self.disableInvalidAxes)
        self.ui.savedb_btn.clicked.connect(self.saveToDB)
        self.ui.savefile_btn.clicked.connect(self.saveToFile)

    @property
    def database(self):
        """Database filename."""
        return _QApplication.instance().database

    @property
    def directory(self):
        """Return the default directory."""
        return _QApplication.instance().directory

    def accept(self):
        """Close dialog."""
        self.clear()
        super().accept()

    def clear(self):
        """Clear data."""
        self.fieldmap_id_list = None
        self.fieldmap_list = None
        self.field_scan_list = None
        self.field_scan_id_list = None
        self.local_hall_probe = None
        self.disableSaveToFile()

    def clearInfo(self):
        """Clear inputs."""
        self.clearMagnetInfo()
        self.clearAxesInfo()

    def clearAxesInfo(self):
        """Clear axes inputs."""
        self.ui.centerpos3_sb.setValue(0)
        self.ui.centerpos2_sb.setValue(0)
        self.ui.centerpos1_sb.setValue(0)
        self.ui.magnet_x_axis_cmb.setCurrentIndex(0)
        self.ui.magnet_y_axis_cmb.setCurrentIndex(2)
        self.disableSaveToFile()

    def clearMagnetInfo(self):
        """Clear magnet inputs."""
        self.ui.predefined_cmb.setCurrentIndex(0)
        self.ui.predefined_la.setText('')
        self.ui.magnet_name_le.setText('')
        self.ui.gap_le.setText('')
        self.ui.control_gap_le.setText('')
        self.ui.magnet_length_le.setText('')
        self.ui.comments_te.setPlainText('')
        self.ui.main_chb.setChecked(False)
        self.ui.trim_chb.setChecked(False)
        self.ui.ch_chb.setChecked(False)
        self.ui.cv_chb.setChecked(False)
        self.ui.qs_chb.setChecked(False)
        self.ui.correct_displacements_chb.setChecked(True)
        self.disableSaveToFile()

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.clear()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def disableInvalidAxes(self):
        """Disable invalid magnet axes."""
        for i in range(6):
            self.ui.magnet_y_axis_cmb.model().item(i).setEnabled(True)

        idx_axisx = self.ui.magnet_x_axis_cmb.currentIndex()
        idx_axisy = self.ui.magnet_y_axis_cmb.currentIndex()
        if idx_axisx in [0, 1]:
            if idx_axisy in [0, 1]:
                self.ui.magnet_y_axis_cmb.setCurrentIndex(-1)
            self.ui.magnet_y_axis_cmb.model().item(0).setEnabled(False)
            self.ui.magnet_y_axis_cmb.model().item(1).setEnabled(False)
        elif idx_axisx in [2, 3]:
            if idx_axisy in [2, 3]:
                self.ui.magnet_y_axis_cmb.setCurrentIndex(-1)
            self.ui.magnet_y_axis_cmb.model().item(2).setEnabled(False)
            self.ui.magnet_y_axis_cmb.model().item(3).setEnabled(False)
        elif idx_axisx in [4, 5]:
            if idx_axisy in [4, 5]:
                self.ui.magnet_y_axis_cmb.setCurrentIndex(-1)
            self.ui.magnet_y_axis_cmb.model().item(4).setEnabled(False)
            self.ui.magnet_y_axis_cmb.model().item(5).setEnabled(False)

    def disableSaveToFile(self):
        """Disable save to file button."""
        self.ui.savefile_btn.setEnabled(False)

    def getFieldMap(self, idx):
        """Get fieldmap data."""
        if self.field_scan_list is None or self.local_hall_probe is None:
            return None

        magnet_name = self.ui.magnet_name_le.text()
        gap = self.ui.gap_le.text()
        control_gap = self.ui.control_gap_le.text()
        magnet_len = self.ui.magnet_length_le.text()
        comments = self.ui.comments_te.toPlainText()

        fieldmap = _Fieldmap()
        fieldmap.magnet_name = magnet_name if len(magnet_name) != 0 else None
        fieldmap.gap = gap if len(gap) != 0 else None
        fieldmap.control_gap = control_gap if len(control_gap) != 0 else None
        fieldmap.magnet_length = magnet_len if len(magnet_len) != 0 else None
        fieldmap.comments = comments

        center_pos3 = self.ui.centerpos3_sb.value()
        center_pos2 = self.ui.centerpos2_sb.value()
        center_pos1 = self.ui.centerpos1_sb.value()
        magnet_center = [center_pos3, center_pos2, center_pos1]
        magnet_x_axis = self.magnetXAxis()
        magnet_y_axis = self.magnetYAxis()

        correct_displacements = self.ui.correct_displacements_chb.isChecked()

        for coil in self._coil_list:
            if getattr(self.ui, coil + '_chb').isChecked():
                current = getattr(self.ui, 'current_' + coil + '_le').text()
                current_list = _json.loads(current)
                if not isinstance(current_list, list):  
                    setattr(fieldmap, 'current_' + coil, current)
                else:
                    setattr(fieldmap, 'current_' + coil, str(
                        current_list[idx]))
                turns = getattr(self.ui, 'nr_turns_' + coil + '_le').text()
                setattr(fieldmap, 'nr_turns_' + coil, turns)

        try:
            fieldmap.set_fieldmap_data(
                self.field_scan_list[idx], self.local_hall_probe,
                correct_displacements, magnet_center,
                magnet_x_axis, magnet_y_axis)
            return fieldmap

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return None

    def loadMagnetInfo(self):
        """Load pre-defined magnet info."""
        if self.ui.predefined_cmb.currentIndex() < 1:
            self.clearMagnetInfo()
            return

        m = _magnets_info.get_magnet_info(self.ui.predefined_cmb.currentText())

        if m is None:
            return

        self.ui.predefined_la.setText(m.pop('description'))
        self.ui.magnet_name_le.setText(str(m.pop('magnet_name')))
        self.ui.gap_le.setText(str(m.pop('gap[mm]')))
        self.ui.control_gap_le.setText(str(m.pop('control_gap[mm]')))
        self.ui.magnet_length_le.setText(str(m.pop('magnet_length[mm]')))

        for coil in self._coil_list:
            turns_key = 'nr_turns_' + coil
            chb = getattr(self.ui, coil + '_chb')
            if turns_key in m.keys():
                turns_le = getattr(self.ui, 'nr_turns_' + coil + '_le')
                turns_le.setText(str(m.pop(turns_key)))
                chb.setChecked(True)
                if coil != 'main':
                    current_le = getattr(self.ui, 'current_' + coil + '_le')
                    current_le.setText('0')
            else:
                chb.setChecked(False)

    def magnetXAxis(self):
        """Get magnet x-axis value."""
        axis_str = self.ui.magnet_x_axis_cmb.currentText()
        axis = int(axis_str[8])
        if axis_str.startswith('-'):
            axis = axis*(-1)
        return axis

    def magnetYAxis(self):
        """Get magnet y-axis value."""
        axis_str = self.ui.magnet_y_axis_cmb.currentText()
        if '1' in axis_str:
            axis = 1
        elif '2' in axis_str:
            axis = 2
        elif '3' in axis_str:
            axis = 3
        if axis_str.startswith('-'):
            axis = axis*(-1)
        return axis

    def saveToDB(self):
        """Save fieldmap to database."""
        if self.database is None or len(self.database) == 0:
            msg = 'Invalid database filename.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        if (self.field_scan_id_list is None or
           len(self.field_scan_id_list) == 0):
            msg = 'Invalid list of scan IDs.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        self.blockSignals(True)
        _QApplication.setOverrideCursor(_Qt.WaitCursor)

        try:
            self.fieldmap_list = []
            self.fieldmap_id_list = []
            for idx in range(len(self.field_scan_list)):
                fieldmap = self.getFieldMap(idx)
                if fieldmap is None:
                    self.blockSignals(False)
                    _QApplication.restoreOverrideCursor()
                    msg = 'Failed to create Fieldmap.'
                    _QMessageBox.critical(
                        self, 'Failure', msg, _QMessageBox.Ok)
                    return

                nr_scans = len(self.field_scan_id_list[idx])
                initial_scan = self.field_scan_id_list[idx][0]
                final_scan = self.field_scan_id_list[idx][-1]

                fieldmap.nr_scans = nr_scans
                fieldmap.initial_scan = initial_scan
                fieldmap.final_scan = final_scan
                idn = fieldmap.save_to_database(self.database)
                self.fieldmap_list.append(fieldmap)
                self.fieldmap_id_list.append(idn)
                self.ui.savefile_btn.setEnabled(True)
            
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            
            if len(self.fieldmap_list) == 1:
                msg = 'Fieldmap saved to database table.\nID: ' + str(
                    self.fieldmap_id_list[0])                
            else:               
                msg = 'Fieldmaps saved to database table.\nIDs: ' + str(
                    self.fieldmap_id_list)
            _QMessageBox.information(self, 'Information', msg, _QMessageBox.Ok)

        except Exception:
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save Fieldmaps to database.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def saveToFile(self):
        """Save fieldmap to database."""
        if self.fieldmap_list is None:
            return

        try:
            fns = []
            for idx, fieldmap in enumerate(self.fieldmap_list):
                idn = self.fieldmap_id_list[idx]
                default_filename = fieldmap.default_filename
                if '.txt' in default_filename:
                    default_filename = default_filename.replace(
                        '.txt', '_ID={0:d}.txt'.format(idn))
                elif '.dat' in default_filename:
                    default_filename = default_filename.replace(
                        '.dat', '_ID={0:d}.dat'.format(idn))
                fns.append(default_filename)
    
            if len(self.fieldmap_list) == 1:
                filename = _QFileDialog.getSaveFileName(
                    self, caption='Save fieldmap file',
                    directory=_os.path.join(self.directory, fns[0]),
                    filter="Text files (*.txt *.dat)")
    
                if isinstance(filename, tuple):
                    filename = filename[0]
    
                if len(filename) == 0:
                    return
    
                fns[0] = filename
      
            else:
                directory = _QFileDialog.getExistingDirectory(
                    self, caption='Save fieldmap files',
                    directory=self.directory)
    
                if isinstance(directory, tuple):
                    directory = directory[0]
    
                if len(directory) == 0:
                    return
    
                for i in range(len(fns)):
                    fns[i] = _os.path.join(directory, fns[i])

            for idx, fn in enumerate(fns):
                if not fn.endswith('.txt') and not fn.endswith('.dat'):
                    fn = fn + '.dat'
                self.fieldmap_list[idx].save_file(fn)
            
            if len(self.fieldmap_list) == 1:
                msg = 'Fieldmap data saved to file: \n%s' % fns[0]
            else:
                msg = 'Fieldmaps saved to files.'
            _QMessageBox.information(self, 'Information', msg, _QMessageBox.Ok)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save Fieldmaps to file.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def setCoilFrameEnabled(self, checkbox, frame):
        """Enable or disable coil frame."""
        self.disableSaveToFile()
        if checkbox.isChecked():
            frame.setEnabled(True)
        else:
            frame.setEnabled(False)
            lineedits = frame.findChildren(_QLineEdit)
            for le in lineedits:
                le.clear()

    def show(self, field_scan_list, hall_probe, field_scan_id_list):
        """Update fieldmap variable and show dialog."""
        if field_scan_list is None or len(field_scan_list) == 0:
            msg = 'Invalid field scan list.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        if hall_probe is None:
            msg = 'Invalid hall probe data.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        try:
            if not isinstance(field_scan_list[0], list):
                field_scan_list = [field_scan_list]
                field_scan_id_list = [field_scan_id_list]
    
            magnets_name = []
            for lt in field_scan_list:
                for fs in lt:
                    magnets_name.append(fs.magnet_name) 
    
            mn = field_scan_list[0][0].magnet_name
            if all([magnet_name == mn for magnet_name in magnets_name]):
                if mn is not None:
                    mag = mn.split('-')[0]
                    if mag == 'B120':
                        mag = 'B2'
                        mn = mn.replace('B120', 'B2')
                    if mag == 'B80':
                        mag = 'B1'
                        mn = mn.replace('B80', 'B1')
                    idx = self.ui.predefined_cmb.findText(mag)
                    self.ui.predefined_cmb.setCurrentIndex(idx)
                    self.loadMagnetInfo()
                    self.ui.magnet_name_le.setText(mn)
            
            current_list = []
            for lt in field_scan_list:
                cs = lt[0].current_setpoint
                if all([fs.current_setpoint == cs for fs in lt]):
                    current_list.append(cs)
                else:
                    current_list.append(None)
        
            if all(c is None for c in current_list):
                self.ui.current_main_le.setText('')
            elif len(current_list) == 1:
                self.ui.current_main_le.setText(_json.dumps(current_list[0]))
            else:
                self.ui.current_main_le.setText(_json.dumps(current_list))
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

        self.ui.savefile_btn.setEnabled(False)
        self.current_list = current_list
        self.field_scan_list = field_scan_list
        self.local_hall_probe = hall_probe
        self.field_scan_id_list = field_scan_id_list
        super().show()
