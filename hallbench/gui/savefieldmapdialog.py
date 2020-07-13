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
import hallbench.data.magnets_info as _magnets_info
from hallbench.data.calibration import (
    HallProbePositions as _HallProbePositions)
from hallbench.data.measurement import FieldScan as _FieldScan
from hallbench.data.measurement import Fieldmap as _Fieldmap


class SaveFieldmapDialog(_QDialog):
    """Save fieldmap dialog class for the Hall Bench Control application."""

    _coil_list = ['main', 'trim', 'ch', 'cv', 'qs']

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _get_ui_file(self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialisation
        self.fieldmap_id_list = None
        self.field_scan_id_list = None

        # add predefined magnet names
        names = _magnets_info.get_magnets_name()
        for name in names:
            self.ui.cmb_predefined.addItem(name)

        self.disable_invalid_axes()
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
        self.fieldmap_id_list = None
        self.field_scan_id_list = None
        self.disable_save_to_file()

    def clear_info(self):
        """Clear inputs."""
        self.clear_magnet_info()
        self.clear_axes_info()
        self.ui.cmb_probe_names.setCurrentIndex(-1)

    def clear_axes_info(self):
        """Clear axes inputs."""
        self.ui.sbd_centerpos3.setValue(0)
        self.ui.sbd_centerpos2.setValue(0)
        self.ui.sbd_centerpos1.setValue(0)
        self.ui.cmb_magnet_x_axis.setCurrentIndex(1)
        self.ui.cmb_magnet_y_axis.setCurrentIndex(2)
        self.disable_save_to_file()

    def clear_magnet_info(self):
        """Clear magnet inputs."""
        self.ui.cmb_predefined.setCurrentIndex(0)
        self.ui.la_predefined.setText('')
        self.ui.le_magnet_name.setText('')
        self.ui.le_gap.setText('')
        self.ui.le_control_gap.setText('')
        self.ui.le_magnet_length.setText('')
        self.ui.te_comments.setPlainText('')
        self.ui.chb_main.setChecked(False)
        self.ui.chb_trim.setChecked(False)
        self.ui.chb_ch.setChecked(False)
        self.ui.chb_cv.setChecked(False)
        self.ui.chb_qs.setChecked(False)
        self.ui.chb_correct_positions.setChecked(True)
        self.disable_save_to_file()

    def connect_signal_slots(self):
        """Create signal/slot connections."""
        self.ui.chb_main.stateChanged.connect(
            lambda: self.set_coil_frame_enabled(
                self.ui.chb_main, self.ui.main_fm))

        self.ui.chb_trim.stateChanged.connect(
            lambda: self.set_coil_frame_enabled(
                self.ui.chb_trim, self.ui.trim_fm))

        self.ui.chb_ch.stateChanged.connect(
            lambda: self.set_coil_frame_enabled(
                self.ui.chb_ch, self.ui.ch_fm))

        self.ui.chb_cv.stateChanged.connect(
            lambda: self.set_coil_frame_enabled(
                self.ui.chb_cv, self.ui.cv_fm))

        self.ui.chb_qs.stateChanged.connect(
            lambda: self.set_coil_frame_enabled(
                self.ui.chb_qs, self.ui.qs_fm))

        self.ui.sbd_centerpos3.valueChanged.connect(self.disable_save_to_file)
        self.ui.sbd_centerpos2.valueChanged.connect(self.disable_save_to_file)
        self.ui.sbd_centerpos1.valueChanged.connect(self.disable_save_to_file)
        self.ui.cmb_magnet_x_axis.currentIndexChanged.connect(
            self.disable_save_to_file)
        self.ui.cmb_magnet_y_axis.currentIndexChanged.connect(
            self.disable_save_to_file)
        self.ui.cmb_predefined.currentIndexChanged.connect(
            self.disable_save_to_file)
        self.ui.le_magnet_name.editingFinished.connect(
            self.disable_save_to_file)
        self.ui.le_gap.editingFinished.connect(self.disable_save_to_file)
        self.ui.le_control_gap.editingFinished.connect(
            self.disable_save_to_file)
        self.ui.le_magnet_length.editingFinished.connect(
            self.disable_save_to_file)
        self.ui.te_comments.textChanged.connect(self.disable_save_to_file)
        self.ui.chb_correct_positions.stateChanged.connect(
            self.disable_save_to_file)

        for coil in self._coil_list:
            le_turns = getattr(self.ui, 'le_nr_turns_' + coil)
            le_turns.editingFinished.connect(self.disable_save_to_file)
            le_current = getattr(self.ui, 'le_current_' + coil)
            le_current.editingFinished.connect(self.disable_save_to_file)

        self.ui.tbt_clear.clicked.connect(self.clear_info)
        self.ui.cmb_predefined.currentIndexChanged.connect(
            self.load_magnet_info)
        self.ui.cmb_magnet_x_axis.currentIndexChanged.connect(
            self.disable_invalid_axes)
        self.ui.pbt_savedb.clicked.connect(self.save_to_db)
        self.ui.pbt_savefile.clicked.connect(self.save_to_file)
        self.ui.tbt_update_probe_names.clicked.connect(
            self.update_probe_names)

    def disable_invalid_axes(self):
        """Disable invalid magnet axes."""
        for i in range(6):
            self.ui.cmb_magnet_y_axis.model().item(i).setEnabled(True)

        idx_axisx = self.ui.cmb_magnet_x_axis.currentIndex()
        idx_axisy = self.ui.cmb_magnet_y_axis.currentIndex()
        if idx_axisx in [0, 1]:
            if idx_axisy in [0, 1]:
                self.ui.cmb_magnet_y_axis.setCurrentIndex(-1)
            self.ui.cmb_magnet_y_axis.model().item(0).setEnabled(False)
            self.ui.cmb_magnet_y_axis.model().item(1).setEnabled(False)
        elif idx_axisx in [2, 3]:
            if idx_axisy in [2, 3]:
                self.ui.cmb_magnet_y_axis.setCurrentIndex(-1)
            self.ui.cmb_magnet_y_axis.model().item(2).setEnabled(False)
            self.ui.cmb_magnet_y_axis.model().item(3).setEnabled(False)
        elif idx_axisx in [4, 5]:
            if idx_axisy in [4, 5]:
                self.ui.cmb_magnet_y_axis.setCurrentIndex(-1)
            self.ui.cmb_magnet_y_axis.model().item(4).setEnabled(False)
            self.ui.cmb_magnet_y_axis.model().item(5).setEnabled(False)

    def disable_save_to_file(self):
        """Disable save to file button."""
        self.ui.pbt_savefile.setEnabled(False)

    def get_fieldmap(self, idx):
        """Get fieldmap data."""
        if self.field_scan_id_list is None:
            return None

        magnet_name = self.ui.le_magnet_name.text()
        gap = self.ui.le_gap.text()
        control_gap = self.ui.le_control_gap.text()
        magnet_len = self.ui.le_magnet_length.text()
        #comments = self.ui.te_comments.toPlainText()

        fieldmap = _Fieldmap(
            database_name=self.database_name,
            mongo=self.mongo, server=self.server)
        fieldmap.magnet_name = magnet_name if len(magnet_name) != 0 else None
        fieldmap.gap = gap if len(gap) != 0 else None
        fieldmap.control_gap = control_gap if len(control_gap) != 0 else None
        fieldmap.magnet_length = magnet_len if len(magnet_len) != 0 else None
        #fieldmap.comments = comments

        center_pos3 = self.ui.sbd_centerpos3.value()
        center_pos2 = self.ui.sbd_centerpos2.value()
        center_pos1 = self.ui.sbd_centerpos1.value()
        magnet_center = [center_pos3, center_pos2, center_pos1]
        magnet_x_axis = self.magnet_x_axis()
        magnet_y_axis = self.magnet_y_axis()

        correct_positions = self.ui.chb_correct_positions.isChecked()

        for coil in self._coil_list:
            if getattr(self.ui, 'chb_' + coil).isChecked():
                current = getattr(self.ui, 'le_current_' + coil).text()
                current_list = _json.loads(current)
                if not isinstance(current_list, list):
                    setattr(fieldmap, 'current_' + coil, current)
                else:
                    setattr(fieldmap, 'current_' + coil, str(
                        current_list[idx]))
                turns = getattr(self.ui, 'le_nr_turns_' + coil).text()
                setattr(fieldmap, 'nr_turns_' + coil, turns)

        try:
            probe_name = self.ui.cmb_probe_names.currentText()
            probe_positions = _HallProbePositions(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
            probe_positions.update_probe(probe_name)
            
            field_scan_list = []
            for idn in self.field_scan_id_list[idx]:
                fs = _FieldScan(
                    database_name=self.database_name,
                    mongo=self.mongo, server=self.server)
                fs.db_read(idn)
                field_scan_list.append(fs)

            fieldmap.comments = field_scan_list[0].comments
            fieldmap.set_fieldmap_data(
                field_scan_list, probe_positions,
                correct_positions, magnet_center,
                magnet_x_axis, magnet_y_axis)
            return fieldmap

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return None

    def load_magnet_info(self):
        """Load pre-defined magnet info."""
        if self.ui.cmb_predefined.currentIndex() < 1:
            self.clear_magnet_info()
            return

        m = _magnets_info.get_magnet_info(self.ui.cmb_predefined.currentText())

        if m is None:
            return

        self.ui.la_predefined.setText(m.pop('description'))
        self.ui.le_magnet_name.setText(str(m.pop('magnet_name')))
        self.ui.le_gap.setText(str(m.pop('gap[mm]')))
        self.ui.le_control_gap.setText(str(m.pop('control_gap[mm]')))
        self.ui.le_magnet_length.setText(str(m.pop('magnet_length[mm]')))

        for coil in self._coil_list:
            turns_key = 'nr_turns_' + coil
            chb = getattr(self.ui, 'chb_' + coil)
            if turns_key in m.keys():
                le_turns = getattr(self.ui, 'le_nr_turns_' + coil)
                le_turns.setText(str(m.pop(turns_key)))
                chb.setChecked(True)
                if coil != 'main':
                    le_current = getattr(self.ui, 'le_current_' + coil)
                    le_current.setText('0')
            else:
                chb.setChecked(False)

    def magnet_x_axis(self):
        """Get magnet x-axis value."""
        axis_str = self.ui.cmb_magnet_x_axis.currentText()
        axis = int(axis_str[8])
        if axis_str.startswith('-'):
            axis = axis*(-1)
        return axis

    def magnet_y_axis(self):
        """Get magnet y-axis value."""
        axis_str = self.ui.cmb_magnet_y_axis.currentText()
        if '1' in axis_str:
            axis = 1
        elif '2' in axis_str:
            axis = 2
        elif '3' in axis_str:
            axis = 3
        if axis_str.startswith('-'):
            axis = axis*(-1)
        return axis

    def save_to_db(self):
        """Save fieldmap to database."""
        if self.database_name is None:
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
            self.fieldmap_id_list = []
            for idx in range(len(self.field_scan_id_list)):
                fieldmap = self.get_fieldmap(idx)
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
                idn = fieldmap.db_save()
                self.fieldmap_id_list.append(idn)

            self.ui.pbt_savefile.setEnabled(True)
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()

            if len(self.fieldmap_id_list) == 1:
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

    def save_to_file(self):
        """Save fieldmap to database."""
        if self.fieldmap_id_list is None:
            return

        try:
            if len(self.fieldmap_id_list) == 1:
                idn = self.fieldmap_id_list[0]
                fieldmap = _Fieldmap(
                    database_name=self.database_name,
                    mongo=self.mongo, server=self.server)
                fieldmap.db_read(idn)

                default_filename = fieldmap.default_filename

                filename = _QFileDialog.getSaveFileName(
                    self, caption='Save fieldmap file',
                    directory=_os.path.join(self.directory, default_filename),
                    filter="Text files (*.txt *.dat)")

                if isinstance(filename, tuple):
                    filename = filename[0]

                if len(filename) == 0:
                    return

                if (not filename.endswith('.txt') and
                    not filename.endswith('.dat')):
                    filename = filename + '.dat'

                fieldmap.save_file(filename)
                msg = 'Fieldmap data saved to file.'

            else:
                directory = _QFileDialog.getExistingDirectory(
                    self, caption='Save fieldmap files',
                    directory=self.directory)

                if isinstance(directory, tuple):
                    directory = directory[0]

                if len(directory) == 0:
                    return

                for idn in self.fieldmap_id_list:
                    fieldmap = _Fieldmap(
                        database_name=self.database_name,
                        mongo=self.mongo, server=self.server)
                    fieldmap.db_read(idn)
                    default_filename = fieldmap.default_filename
                    filename = _os.path.join(directory, default_filename)
                    fieldmap.save_file(filename)

                msg = 'Fieldmaps saved to files.'

            _QMessageBox.information(self, 'Information', msg, _QMessageBox.Ok)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save Fieldmaps to file.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def set_coil_frame_enabled(self, checkbox, frame):
        """Enable or disable coil frame."""
        self.disable_save_to_file()
        if checkbox.isChecked():
            frame.setEnabled(True)
        else:
            frame.setEnabled(False)
            lineedits = frame.findChildren(_QLineEdit)
            for le in lineedits:
                le.clear()

    def show(self, field_scan_id_list):
        """Show dialog."""
        if field_scan_id_list is None or len(field_scan_id_list) == 0:
            msg = 'Invalid field scan ID list.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        try:
            fs = _FieldScan(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
            
            if not isinstance(field_scan_id_list[0], list):
                field_scan_id_list = [field_scan_id_list]

            mn = fs.db_get_value('magnet_name', field_scan_id_list[0][0])

            if mn is not None and self.ui.le_magnet_name.text() != mn:
                mag = mn.split('-')[0]
                if mag == 'B120':
                    mag = 'B2'
                    mn = mn.replace('B120', 'B2')
                if mag == 'B80':
                    mag = 'B1'
                    mn = mn.replace('B80', 'B1')
                idx = self.ui.cmb_predefined.findText(mag)
                self.ui.cmb_predefined.setCurrentIndex(idx)
                self.load_magnet_info()
                self.ui.le_magnet_name.setText(mn)

            current_list = []
            for lt in field_scan_id_list:
                ltc = []
                for idn in lt:
                    ltc.append(fs.db_get_value('current_setpoint', idn))
                if all([current == ltc[0] for current in ltc]):
                    current_list.append(ltc[0])
                else:
                    current_list.append(None)

            if all(c is None for c in current_list):
                self.ui.le_current_main.setText('')
            elif len(current_list) == 1:
                self.ui.le_current_main.setText(_json.dumps(current_list[0]))
            else:
                self.ui.le_current_main.setText(_json.dumps(current_list))
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

        self.ui.pbt_savefile.setEnabled(False)
        self.current_list = current_list
        self.field_scan_id_list = field_scan_id_list
        self.update_probe_names()
        super().show()

    def update_probe_names(self):
        try:
            probe_positions = _HallProbePositions(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
            
            probe_names = probe_positions.get_probe_list()
        
            current_text = self.ui.cmb_probe_names.currentText()
            self.ui.cmb_probe_names.clear()
            
            self.ui.cmb_probe_names.addItems([pn for pn in probe_names])
            if len(current_text) == 0:
                self.ui.cmb_probe_names.setCurrentIndex(-1)
            else:
                self.ui.cmb_probe_names.setCurrentText(current_text)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)