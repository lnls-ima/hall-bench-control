# -*- coding: utf-8 -*-

"""Save field map dialog for the Hall Bench Control application."""

import os.path as _path
from PyQt5.QtWidgets import (
    QDialog as _QDialog,
    QTableWidgetItem as _QTableWidgetItem,
    QLineEdit as _QLineEdit,
    QMessageBox as _QMessageBox,
    )
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
import hallbench.data.magnets_info as _magnets_info
from hallbench.data.utils import get_timestamp as _get_timestamp


class SaveFieldMapDialog(_QDialog):
    """Save field map dialog class for the Hall Bench Control application."""

    _coil_list = ['main', 'trim', 'ch', 'cv', 'qs']

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(__file__, self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialization
        self.fieldmap = None
        self.directory = ''

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
        self.ui.addrow_btn.clicked.connect(self.addTableRow)
        self.ui.removerow_btn.clicked.connect(self.removeTableRow)
        self.ui.axisx_cmb.currentIndexChanged.connect(self.disableInvalidAxes)
        self.ui.savefieldmap_btn.clicked.connect(self.saveFieldMap)

    def addTableRow(self):
        """Add row to aditional parameters table."""
        self.ui.additionalparam_ta.setRowCount(
            self.ui.additionalparam_ta.rowCount() + 1)

    def clearInfo(self):
        """Clear inputs."""
        self.clearMagnetInfo()
        self.clearAxesInfo()

    def clearAxesInfo(self):
        """Clear axes inputs."""
        self.ui.centerpos3_sb.setValue(0)
        self.ui.centerpos2_sb.setValue(0)
        self.ui.centerpos1_sb.setValue(0)
        self.ui.axisx_cmb.setCurrentIndex(0)
        self.ui.axisy_cmb.setCurrentIndex(2)

    def clearMagnetInfo(self):
        """Clear magnet inputs."""
        self.ui.predefined_cmb.setCurrentIndex(0)
        self.ui.predefined_la.setText('')
        self.ui.magnetname_le.setText('')
        self.ui.gap_le.setText('')
        self.ui.controlgap_le.setText('')
        self.ui.magnetlength_le.setText('')
        self.ui.main_chb.setChecked(False)
        self.ui.trim_chb.setChecked(False)
        self.ui.ch_chb.setChecked(False)
        self.ui.cv_chb.setChecked(False)
        self.ui.qs_chb.setChecked(False)
        self.ui.additionalparam_ta.setRowCount(0)

    def disableInvalidAxes(self):
        """Disable invalid magnet axes."""
        for i in range(6):
            self.ui.axisy_cmb.model().item(i).setEnabled(True)

        idx_axisx = self.ui.axisx_cmb.currentIndex()
        idx_axisy = self.ui.axisy_cmb.currentIndex()
        if idx_axisx in [0, 1]:
            if idx_axisy in [0, 1]:
                self.ui.axisy_cmb.setCurrentIndex(-1)
            self.ui.axisy_cmb.model().item(0).setEnabled(False)
            self.ui.axisy_cmb.model().item(1).setEnabled(False)
        elif idx_axisx in [2, 3]:
            if idx_axisy in [2, 3]:
                self.ui.axisy_cmb.setCurrentIndex(-1)
            self.ui.axisy_cmb.model().item(2).setEnabled(False)
            self.ui.axisy_cmb.model().item(3).setEnabled(False)
        elif idx_axisx in [4, 5]:
            if idx_axisy in [4, 5]:
                self.ui.axisy_cmb.setCurrentIndex(-1)
            self.ui.axisy_cmb.model().item(4).setEnabled(False)
            self.ui.axisy_cmb.model().item(5).setEnabled(False)

    def loadMagnetInfo(self):
        """Load pre-defined magnet info."""
        if self.ui.predefined_cmb.currentIndex() < 1:
            self.clearMagnetInfo()
            return

        m = _magnets_info.get_magnet_info(self.ui.predefined_cmb.currentText())

        if m is None:
            return

        self.ui.additionalparam_ta.setRowCount(0)
        self.ui.predefined_la.setText(m.pop('description'))
        self.ui.magnetname_le.setText(str(m.pop('magnet_name')))
        self.ui.gap_le.setText(str(m.pop('gap[mm]')))
        self.ui.controlgap_le.setText(str(m.pop('control_gap[mm]')))
        self.ui.magnetlength_le.setText(str(m.pop('magnet_length[mm]')))

        for coil in self._coil_list:
            turns_key = 'nr_turns_' + coil
            chb = getattr(self.ui, coil + '_chb')
            if turns_key in m.keys():
                turns_le = getattr(self.ui, coil + 'turns_le')
                turns_le.setText(str(m.pop(turns_key)))
                chb.setChecked(True)
                if coil != 'main':
                    current_le = getattr(self.ui, coil + 'current_le')
                    current_le.setText('0')
            else:
                chb.setChecked(False)

        if len(m) != 0:
            count = 0
            for parameter, value in m.items():
                self.ui.additionalparam_ta.setRowCount(count+1)
                self.ui.additionalparam_ta.setItem(
                    count, 0, _QTableWidgetItem(str(parameter)))
                self.ui.additionalparam_ta.setItem(
                    count, 1, _QTableWidgetItem(str(value)))
                count = count + 1

    def magnetXAxis(self):
        """Get magnet x-axis value."""
        axis_str = self.ui.axisx_cmb.currentText()
        axis = int(axis_str[8])
        if axis_str.startswith('-'):
            axis = axis*(-1)
        return axis

    def magnetYAxis(self):
        """Get magnet y-axis value."""
        axis_str = self.ui.axisy_cmb.currentText()
        axis = int(axis_str[8])
        if axis_str.startswith('-'):
            axis = axis*(-1)
        return axis

    def removeTableRow(self):
        """Remove last row from aditional parameters table."""
        self.ui.additionalparam_ta.setRowCount(
            self.ui.additionalparam_ta.rowCount() - 1)

    def saveCoordinateSystemFile(self, fieldmap_filename):
        """Save file with magnet coordinate system info."""
        try:
            timestamp = _get_timestamp()
            filename = timestamp + '_magnet_coordinate_system.txt'
            filename = _path.join(self.directory, filename)

            with open(filename, 'w') as f:
                center_pos3 = self.ui.centerpos3_sb.value()
                center_pos2 = self.ui.centerpos2_sb.value()
                center_pos1 = self.ui.centerpos1_sb.value()

                magnet_x_axis = self.magnetXAxis()
                magnet_y_axis = self.magnetYAxis()

                f.write('fielmap_file:        {0:s}\n'.format(
                                                            fieldmap_filename))
                f.write('magnet_center_axis3: {0:0.4f}\n'.format(center_pos3))
                f.write('magnet_center_axis2: {0:0.4f}\n'.format(center_pos2))
                f.write('magnet_center_axis1: {0:0.4f}\n'.format(center_pos1))
                f.write('magnet_x_axis:       {0:1d}\n'.format(magnet_x_axis))
                f.write('magnet_y_axis:       {0:1d}\n'.format(magnet_y_axis))

        except Exception:
            pass

    def saveFieldMap(self):
        """Save field map to file."""
        if self.fieldmap is None:
            return

        datetime = _get_timestamp()
        date = datetime.split('_')[0]
        magnet_name = self.ui.magnetname_le.text()
        gap = self.ui.gap_le.text()
        control_gap = self.ui.controlgap_le.text()
        magnet_length = self.ui.magnetlength_le.text()

        header_info = []
        header_info.append(['fieldmap_name', magnet_name])
        header_info.append(['timestamp', datetime])
        header_info.append(['nr_magnets', 1])
        header_info.append(['magnet_name', magnet_name])
        header_info.append(['gap[mm]', gap])
        header_info.append(['control_gap[mm]', control_gap])
        header_info.append(['magnet_length[mm]', magnet_length])

        if len(magnet_name) != 0:
            filename = magnet_name
        else:
            filename = 'hall_probe_measurement'

        for coil in self._coil_list:
            if getattr(self.ui, coil + '_chb').isChecked():
                current = getattr(self.ui, coil + 'current_le').text()
                header_info.append(['current_' + coil + '[A]', current])
                turns = getattr(self.ui, coil + 'turns_le').text()
                header_info.append(['nr_turns_' + coil, turns])
                if len(current) != 0:
                    ac = coil if coil != 'trim' else 'tc'
                    filename = filename + '_I' + ac + '=' + current + 'A'

        filename = '{0:1s}_{1:1s}.dat'.format(date, filename)
        filename = _path.join(self.directory, filename)

        header_info.insert(2, ['filename', filename])

        for i in range(self.ui.additionalparam_ta.rowCount()):
            parameter = self.ui.additionalparam_ta.item(i, 0).text()
            value = self.ui.additionalparam_ta.item(i, 1).text()
            if len(value) != 0:
                header_info.append([parameter.replace(" ", ""), value])

        header_info.append(['center_pos_z[mm]', '0'])
        header_info.append(['center_pos_x[mm]', '0'])
        header_info.append(['rotation[deg]', '0'])

        self.fieldmap.header_info = header_info

        center_pos3 = self.ui.centerpos3_sb.value()
        center_pos2 = self.ui.centerpos2_sb.value()
        center_pos1 = self.ui.centerpos1_sb.value()
        magnet_center = [center_pos3, center_pos2, center_pos1]

        magnet_x_axis = self.magnetXAxis()
        magnet_y_axis = self.magnetYAxis()

        try:
            self.fieldmap.save_file(
                filename,
                magnet_center=magnet_center,
                magnet_x_axis=magnet_x_axis,
                magnet_y_axis=magnet_y_axis,
            )

            self.saveCoordinateSystemFile(filename)

            message = 'Field map data saved in file: \n%s' % filename
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

    def show(self, fieldmap, directory=''):
        """Update fieldmap variable and show dialog."""
        self.fieldmap = fieldmap
        self.directory = directory
        super(SaveFieldMapDialog, self).show()
