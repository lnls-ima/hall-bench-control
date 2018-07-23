# -*- coding: utf-8 -*-

"""Field map dialog for the Hall Bench Control application."""

from PyQt5.QtCore import Qt as _Qt
from PyQt5.QtWidgets import (
    QApplication as _QApplication,
    QDialog as _QDialog,
    QFileDialog as _QFileDialog,
    QLineEdit as _QLineEdit,
    QMessageBox as _QMessageBox,
    )
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
import hallbench.data.magnets_info as _magnets_info
from hallbench.data.measurement import Fieldmap as _Fieldmap


class FieldMapDialog(_QDialog):
    """Field map dialog class for the Hall Bench Control application."""

    _coil_list = ['main', 'trim', 'ch', 'cv', 'qs']

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialization
        self.database = None
        self.field_data_list = None
        self.probe_calibration = None
        self.scan_id_list = None

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

        self.ui.clear_btn.clicked.connect(self.clearInfo)
        self.ui.predefined_cmb.currentIndexChanged.connect(self.loadMagnetInfo)
        self.ui.magnet_x_axis_cmb.currentIndexChanged.connect(
            self.disableInvalidAxes)
        self.ui.savedb_btn.clicked.connect(self.saveToDB)
        self.ui.savefile_btn.clicked.connect(self.saveToFile)

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

    def getFieldMap(self):
        """Get fieldmap data."""
        if self.field_data_list is None or self.probe_calibration is None:
            return None

        self.blockSignals(True)
        _QApplication.setOverrideCursor(_Qt.WaitCursor)

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
        fieldmap.comments = comments if len(comments) != 0 else None

        for coil in self._coil_list:
            if getattr(self.ui, coil + '_chb').isChecked():
                current = getattr(self.ui, 'current_' + coil + '_le').text()
                setattr(fieldmap, 'current_' + coil, current)
                turns = getattr(self.ui, 'nr_turns_' + coil + '_le').text()
                setattr(fieldmap, 'nr_turns_' + coil, turns)

        center_pos3 = self.ui.centerpos3_sb.value()
        center_pos2 = self.ui.centerpos2_sb.value()
        center_pos1 = self.ui.centerpos1_sb.value()
        magnet_center = [center_pos3, center_pos2, center_pos1]

        magnet_x_axis = self.magnetXAxis()
        magnet_y_axis = self.magnetYAxis()

        correct_displacements = self.ui.correct_displacements_chb.isChecked()

        try:
            fieldmap.set_fieldmap_data(
                self.field_data_list, self.probe_calibration,
                correct_displacements, magnet_center,
                magnet_x_axis, magnet_y_axis)

            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            return fieldmap

        except Exception as e:
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)
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
        axis = int(axis_str[8])
        if axis_str.startswith('-'):
            axis = axis*(-1)
        return axis

    def saveToDB(self):
        """Save fieldmap to database."""
        if self.database is None or len(self.database) == 0:
            message = 'Invalid database filename.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        if self.scan_id_list is None or len(self.scan_id_list) == 0:
            message = 'Invalid list of scan IDs.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        fieldmap = self.getFieldMap()
        if fieldmap is None:
            return

        nr_scans = len(self.scan_id_list)
        initial_scan = self.scan_id_list[0]
        final_scan = self.scan_id_list[-1]

        try:
            idn = fieldmap.save_to_database(
                self.database, nr_scans, initial_scan, final_scan)
            message = 'Fieldmap data saved to database table. ID: %i' % idn
            _QMessageBox.information(
                self, 'Information', message, _QMessageBox.Ok)

        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def saveToFile(self):
        """Save fieldmap to database."""
        fieldmap = self.getFieldMap()
        if fieldmap is None:
            return

        filename = _QFileDialog.getSaveFileName(
            self, caption='Save fieldmap file',
            directory=fieldmap.default_filename,
            filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            if not filename.endswith('.txt') and not filename.endswith('.dat'):
                filename = filename + '.dat'
            fieldmap.save_file(filename)
            message = 'Fieldmap data saved to file: \n%s' % filename
            _QMessageBox.information(
                self, 'Information', message, _QMessageBox.Ok)

        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def setCoilFrameEnabled(self, checkbox, frame):
        """Enable or disable coil frame."""
        if checkbox.isChecked():
            frame.setEnabled(True)
        else:
            frame.setEnabled(False)
            lineedits = frame.findChildren(_QLineEdit)
            for le in lineedits:
                le.clear()

    def show(self, field_data_list, probe_calibration, database, scan_id_list):
        """Update fieldmap variable and show dialog."""
        if field_data_list is None or len(field_data_list) == 0:
            message = 'Invalid field data list.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        if probe_calibration is None:
            message = 'Invalid probe calibration data.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        self.field_data_list = field_data_list
        self.probe_calibration = probe_calibration
        self.database = database
        self.scan_id_list = scan_id_list
        super().show()
