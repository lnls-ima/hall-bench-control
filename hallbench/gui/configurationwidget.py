# -*- coding: utf-8 -*-

"""Configuration widget for the Hall Bench Control application."""

import sys as _sys
import os.path as _path
import numpy as _np
import traceback as _traceback
from PyQt5.QtWidgets import (
    QWidget as _QWidget,
    QFileDialog as _QFileDialog,
    QMessageBox as _QMessageBox,
    QApplication as _QApplication,
    )
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.viewprobedialog import ViewProbeDialog \
    as _ViewProbeDialog


class ConfigurationWidget(_QWidget):
    """Configuration widget class for the Hall Bench Control application."""

    _measurement_axes = [1, 2, 3, 5]

    def __init__(self, parent=None):
        """Set up the ui, add widgets and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        self.updateProbeNames()
        self.updateConfigurationIDs()
        self.ui.loaddb_btn.setEnabled(False)

        self.view_probe_dialog = _ViewProbeDialog()
        self.connectSignalSlots()

    @property
    def database(self):
        """Database filename."""
        return _QApplication.instance().database

    @property
    def measurement_config(self):
        """Measurement configuration."""
        return _QApplication.instance().measurement_config

    @property
    def hall_probe(self):
        """Hall probe calibration data."""
        return _QApplication.instance().hall_probe

    @property
    def positions(self):
        """Positions dict."""
        return _QApplication.instance().positions

    def clearHallProbe(self):
        """Clear hall probe calibration data."""
        self.hall_probe.clear()
        self.ui.probe_name_cmb.setCurrentIndex(-1)

    def clearLoadOptions(self):
        """Clear load options."""
        self.ui.filename_le.setText("")
        self.ui.idn_cmb.setCurrentIndex(-1)

    def closeDialogs(self):
        """Close dialogs."""
        try:
            self.view_probe_dialog.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.closeDialogs()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.first_ax1_rb.clicked.connect(self.clearLoadOptions)
        self.ui.first_ax2_rb.clicked.connect(self.clearLoadOptions)
        self.ui.first_ax3_rb.clicked.connect(self.clearLoadOptions)
        self.ui.first_ax5_rb.clicked.connect(self.clearLoadOptions)
        self.ui.second_ax1_rb.clicked.connect(self.clearLoadOptions)
        self.ui.second_ax2_rb.clicked.connect(self.clearLoadOptions)
        self.ui.second_ax3_rb.clicked.connect(self.clearLoadOptions)
        self.ui.second_ax5_rb.clicked.connect(self.clearLoadOptions)
        self.ui.magnet_name_le.editingFinished.connect(self.clearLoadOptions)
        self.ui.main_current_le.editingFinished.connect(self.clearLoadOptions)
        self.ui.temperature_le.editingFinished.connect(self.clearLoadOptions)
        self.ui.operator_le.editingFinished.connect(self.clearLoadOptions)
        self.ui.nr_measurements_sb.valueChanged.connect(self.clearLoadOptions)
        self.ui.probe_name_cmb.currentIndexChanged.connect(
            self.clearLoadOptions)
        self.ui.voltage_precision_cmb.currentIndexChanged.connect(
            self.clearLoadOptions)
        self.voltx_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.volty_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.voltz_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.ui.idn_cmb.currentIndexChanged.connect(self.enableLoadDB)
        self.ui.update_idn_btn.clicked.connect(self.updateConfigurationIDs)

        self.ui.current_start_btn.clicked.connect(
            self.copyCurrentStartPosition)

        self.ui.start_ax1_le.editingFinished.connect(
            lambda: self.setStrFormatFloatSumSub(self.ui.start_ax1_le))
        self.ui.start_ax2_le.editingFinished.connect(
            lambda: self.setStrFormatFloatSumSub(self.ui.start_ax2_le))
        self.ui.start_ax3_le.editingFinished.connect(
            lambda: self.setStrFormatFloatSumSub(self.ui.start_ax3_le))
        self.ui.start_ax5_le.editingFinished.connect(
            lambda: self.setStrFormatFloatSumSub(self.ui.start_ax5_le))

        self.ui.end_ax1_le.editingFinished.connect(
            lambda: self.setStrFormatFloatSumSub(self.ui.end_ax1_le))
        self.ui.end_ax2_le.editingFinished.connect(
            lambda: self.setStrFormatFloatSumSub(self.ui.end_ax2_le))
        self.ui.end_ax3_le.editingFinished.connect(
            lambda: self.setStrFormatFloatSumSub(self.ui.end_ax3_le))
        self.ui.end_ax5_le.editingFinished.connect(
            lambda: self.setStrFormatFloatSumSub(self.ui.end_ax5_le))

        self.ui.step_ax1_le.editingFinished.connect(
            lambda: self.setStrFormatPositiveNonZeroFloat(self.ui.step_ax1_le))
        self.ui.step_ax2_le.editingFinished.connect(
            lambda: self.setStrFormatPositiveNonZeroFloat(self.ui.step_ax2_le))
        self.ui.step_ax3_le.editingFinished.connect(
            lambda: self.setStrFormatPositiveNonZeroFloat(self.ui.step_ax3_le))
        self.ui.step_ax5_le.editingFinished.connect(
            lambda: self.setStrFormatPositiveNonZeroFloat(self.ui.step_ax5_le))

        self.ui.extra_ax1_le.editingFinished.connect(
            lambda: self.setStrFormatPositiveFloat(self.ui.extra_ax1_le))
        self.ui.extra_ax2_le.editingFinished.connect(
            lambda: self.setStrFormatPositiveFloat(self.ui.extra_ax2_le))
        self.ui.extra_ax3_le.editingFinished.connect(
            lambda: self.setStrFormatPositiveFloat(self.ui.extra_ax3_le))
        self.ui.extra_ax5_le.editingFinished.connect(
            lambda: self.setStrFormatPositiveFloat(self.ui.extra_ax5_le))

        self.ui.vel_ax1_le.editingFinished.connect(
            lambda: self.setStrFormatPositiveFloat(self.ui.vel_ax1_le))
        self.ui.vel_ax2_le.editingFinished.connect(
            lambda: self.setStrFormatPositiveFloat(self.ui.vel_ax2_le))
        self.ui.vel_ax3_le.editingFinished.connect(
            lambda: self.setStrFormatPositiveFloat(self.ui.vel_ax3_le))
        self.ui.vel_ax5_le.editingFinished.connect(
            lambda: self.setStrFormatPositiveFloat(self.ui.vel_ax5_le))

        self.ui.integration_time_le.editingFinished.connect(
            lambda: self.setStrFormatFloat(self.ui.integration_time_le))

        self.ui.first_ax1_rb.clicked.connect(self.disableSecondAxisButton)
        self.ui.first_ax2_rb.clicked.connect(self.disableSecondAxisButton)
        self.ui.first_ax3_rb.clicked.connect(self.disableSecondAxisButton)
        self.ui.first_ax5_rb.clicked.connect(self.disableSecondAxisButton)

        self.ui.first_ax1_rb.toggled.connect(self.disableInvalidLineEdit)
        self.ui.first_ax2_rb.toggled.connect(self.disableInvalidLineEdit)
        self.ui.first_ax3_rb.toggled.connect(self.disableInvalidLineEdit)
        self.ui.first_ax5_rb.toggled.connect(self.disableInvalidLineEdit)

        self.ui.second_ax1_rb.toggled.connect(self.disableInvalidLineEdit)
        self.ui.second_ax2_rb.toggled.connect(self.disableInvalidLineEdit)
        self.ui.second_ax3_rb.toggled.connect(self.disableInvalidLineEdit)
        self.ui.second_ax5_rb.toggled.connect(self.disableInvalidLineEdit)

        self.ui.second_ax1_rb.clicked.connect(
            lambda: self.uncheckRadioButtons(1))
        self.ui.second_ax2_rb.clicked.connect(
            lambda: self.uncheckRadioButtons(2))
        self.ui.second_ax3_rb.clicked.connect(
            lambda: self.uncheckRadioButtons(3))
        self.ui.second_ax5_rb.clicked.connect(
            lambda: self.uncheckRadioButtons(5))

        self.ui.start_ax1_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(1))
        self.ui.start_ax2_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(2))
        self.ui.start_ax3_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(3))
        self.ui.start_ax5_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(5))

        self.ui.step_ax1_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(1))
        self.ui.step_ax2_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(2))
        self.ui.step_ax3_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(3))
        self.ui.step_ax5_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(5))

        self.ui.end_ax1_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(1))
        self.ui.end_ax2_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(2))
        self.ui.end_ax3_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(3))
        self.ui.end_ax5_le.editingFinished.connect(
            lambda: self.fixEndPositionValue(5))

        self.ui.loadfile_btn.clicked.connect(self.loadFile)
        self.ui.savefile_btn.clicked.connect(self.saveFile)
        self.ui.loaddb_btn.clicked.connect(self.loadDB)
        self.ui.savedb_btn.clicked.connect(self.saveDB)
        self.ui.probe_name_cmb.currentIndexChanged.connect(self.loadHallProbe)
        self.ui.update_probe_name_btn.clicked.connect(self.updateProbeNames)
        self.ui.clear_probe_btn.clicked.connect(self.clearHallProbe)
        self.ui.view_probe_btn.clicked.connect(self.showViewProbeDialog)

    def copyCurrentStartPosition(self):
        """Copy current start position to line edits."""
        for axis in self._measurement_axes:
            start_le = getattr(self.ui, 'start_ax' + str(axis) + '_le')
            if axis in self.positions.keys():
                start_le.setText('{0:0.4f}'.format(self.positions[axis]))
            else:
                start_le.setText('')

    def disableInvalidLineEdit(self):
        """Disable invalid line edit."""
        for axis in self._measurement_axes:
            first_rb = getattr(self.ui, 'first_ax' + str(axis) + '_rb')
            second_rb = getattr(self.ui, 'second_ax' + str(axis) + '_rb')
            step_le = getattr(self.ui, 'step_ax' + str(axis) + '_le')
            end_le = getattr(self.ui, 'end_ax' + str(axis) + '_le')
            extra_le = getattr(self.ui, 'extra_ax' + str(axis) + '_le')
            if first_rb.isChecked() or second_rb.isChecked():
                step_le.setEnabled(True)
                end_le.setEnabled(True)
                if first_rb.isChecked():
                    extra_le.setEnabled(True)
                else:
                    extra_le.setEnabled(False)
                    extra_le.setText('')
            else:
                step_le.setEnabled(False)
                step_le.setText('')
                end_le.setEnabled(False)
                end_le.setText('')
                extra_le.setEnabled(False)
                extra_le.setText('')

    def disableSecondAxisButton(self):
        """Disable invalid second axis radio buttons."""
        for axis in self._measurement_axes:
            first_rb = getattr(self.ui, 'first_ax' + str(axis) + '_rb')
            second_rb = getattr(self.ui, 'second_ax' + str(axis) + '_rb')
            if first_rb.isChecked():
                second_rb.setChecked(False)
                second_rb.setEnabled(False)
            else:
                if axis != 5:
                    second_rb.setEnabled(True)

    def enableLoadDB(self):
        """Enable button to load configuration from database."""
        if self.ui.idn_cmb.currentIndex() != -1:
            self.ui.loaddb_btn.setEnabled(True)
        else:
            self.ui.loaddb_btn.setEnabled(False)

    def fixEndPositionValue(self, axis):
        """Fix end position value."""
        start_le = getattr(self.ui, 'start_ax' + str(axis) + '_le')
        start_le_text = start_le.text()
        if not bool(start_le_text and start_le_text.strip()):
            return
        start = float(start_le_text)

        step_le = getattr(self.ui, 'step_ax' + str(axis) + '_le')
        step_le_text = step_le.text()
        if not bool(step_le_text and step_le_text.strip()):
            return
        step = float(step_le_text)

        end_le = getattr(self.ui, 'end_ax' + str(axis) + '_le')
        end_le_text = end_le.text()
        if not bool(end_le_text and end_le_text.strip()):
            return
        end = float(end_le_text)

        if start is not None and step is not None and end is not None:
            npts = _np.round(round((end - start) / step, 4) + 1)
            if start <= end:
                corrected_end = start + (npts-1)*step
            else:
                corrected_end = start
            end_le.setText('{0:0.4f}'.format(corrected_end))

    def getAxisParam(self, param, axis):
        """Get axis parameter."""
        le = getattr(self.ui, param + '_ax' + str(axis) + '_le')
        le_text = le.text()
        if bool(le_text and le_text.strip()):
            return float(le_text)
        else:
            return None

    def load(self):
        """Set measurement parameters."""
        try:
            self.ui.magnet_name_le.setText(self.measurement_config.magnet_name)
            self.ui.main_current_le.setText(
                self.measurement_config.main_current)

            idx = self.ui.probe_name_cmb.findText(
                self.measurement_config.probe_name)
            self.ui.probe_name_cmb.setCurrentIndex(idx)

            self.ui.temperature_le.setText(self.measurement_config.temperature)
            self.ui.operator_le.setText(self.measurement_config.operator)

            self.ui.voltx_enable_chb.setChecked(
                self.measurement_config.voltx_enable)
            self.ui.volty_enable_chb.setChecked(
                self.measurement_config.volty_enable)
            self.ui.voltz_enable_chb.setChecked(
                self.measurement_config.voltz_enable)

            self.ui.integration_time_le.setText('{0:0.4f}'.format(
                self.measurement_config.integration_time))
            self.ui.voltage_precision_cmb.setCurrentIndex(
                self.measurement_config.voltage_precision)

            self.ui.nr_measurements_sb.setValue(
                self.measurement_config.nr_measurements)

            first_axis = self.measurement_config.first_axis
            first_rb = getattr(self.ui, 'first_ax' + str(first_axis) + '_rb')
            first_rb.setChecked(True)

            self.disableSecondAxisButton()

            second_axis = self.measurement_config.second_axis
            if second_axis != -1:
                second_rb = getattr(
                    self.ui, 'second_ax' + str(second_axis) + '_rb')
                second_rb.setChecked(True)
                self.uncheckRadioButtons(second_axis)
            else:
                self.uncheckRadioButtons(second_axis)

            for axis in self._measurement_axes:
                start_le = getattr(self.ui, 'start_ax' + str(axis) + '_le')
                value = self.measurement_config.get_start(axis)
                start_le.setText('{0:0.4f}'.format(value))

                step_le = getattr(self.ui, 'step_ax' + str(axis) + '_le')
                value = self.measurement_config.get_step(axis)
                step_le.setText('{0:0.4f}'.format(value))

                end_le = getattr(self.ui, 'end_ax' + str(axis) + '_le')
                value = self.measurement_config.get_end(axis)
                end_le.setText('{0:0.4f}'.format(value))

                extra_le = getattr(self.ui, 'extra_ax' + str(axis) + '_le')
                value = self.measurement_config.get_extra(axis)
                extra_le.setText('{0:0.4f}'.format(value))

                vel_le = getattr(self.ui, 'vel_ax' + str(axis) + '_le')
                value = self.measurement_config.get_velocity(axis)
                vel_le.setText('{0:0.4f}'.format(value))

            self.disableInvalidLineEdit()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to load configuration.'
            _QMessageBox.critical(
                self, 'Failure', msg, _QMessageBox.Ok)

    def loadDB(self):
        """Load configuration from database to set measurement parameters."""
        self.ui.filename_le.setText("")

        try:
            idn = int(self.ui.idn_cmb.currentText())
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            _QMessageBox.critical(
                self, 'Failure', 'Invalid database ID.', _QMessageBox.Ok)
            return

        self.updateConfigurationIDs()
        idx = self.ui.idn_cmb.findText(str(idn))
        if idx == -1:
            self.ui.idn_cmb.setCurrentIndex(-1)
            _QMessageBox.critical(
                self, 'Failure', 'Invalid database ID.', _QMessageBox.Ok)
            return

        try:
            self.measurement_config.clear()
            self.measurement_config.read_from_database(self.database, idn)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to read configuration from database.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        self.load()
        self.ui.idn_cmb.setCurrentIndex(self.ui.idn_cmb.findText(str(idn)))
        self.ui.loaddb_btn.setEnabled(False)

    def loadFile(self):
        """Load configuration file to set measurement parameters."""
        self.ui.idn_cmb.setCurrentIndex(-1)

        default_filename = self.ui.filename_le.text()
        filename = _QFileDialog.getOpenFileName(
            self, caption='Open measurement configuration file',
            directory=default_filename, filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            self.measurement_config.clear()
            self.measurement_config.read_file(filename)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to read configuration file.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        self.load()
        self.ui.filename_le.setText(filename)

    def loadHallProbe(self):
        """Load hall probe from database."""
        self.hall_probe.clear()
        probe_name = self.ui.probe_name_cmb.currentText()
        if len(probe_name) == 0:
            return

        try:
            idn = self.hall_probe.get_hall_probe_id(self.database, probe_name)
            if idn is not None:
                self.hall_probe.read_from_database(self.database, idn)
            else:
                self.ui.probe_name_cmb.setCurrentIndex(-1)
                msg = 'Invalid probe name.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to load Hall probe from database.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def saveDB(self):
        """Save configuration to database."""
        self.ui.idn_cmb.setCurrentIndex(-1)
        if self.database is not None and _path.isfile(self.database):
            try:
                if self.updateConfiguration():
                    idn = self.measurement_config.save_to_database(
                        self.database)
                    self.ui.idn_cmb.addItem(str(idn))
                    self.ui.idn_cmb.setCurrentIndex(self.ui.idn_cmb.count()-1)
                    self.ui.loaddb_btn.setEnabled(False)
            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                msg = 'Failed to save configuration to database.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
        else:
            msg = 'Invalid database filename.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def saveFile(self):
        """Save measurement parameters to file."""
        default_filename = self.ui.filename_le.text()
        filename = _QFileDialog.getSaveFileName(
            self, caption='Save measurement configuration file',
            directory=default_filename, filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        if self.updateConfiguration():
            try:
                if (not filename.endswith('.txt')
                   and not filename.endswith('.dat')):
                    filename = filename + '.txt'
                self.measurement_config.save_file(filename)

            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                msg = 'Failed to save configuration to file.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def setStrFormatFloat(self, obj):
        """Set the line edit string format for float value."""
        try:
            if obj.isModified():
                self.clearLoadOptions()
                value = float(obj.text())
                obj.setText('{0:0.4f}'.format(value))
        except Exception:
            obj.setText('')

    def setStrFormatFloatSumSub(self, obj):
        """Set the line edit string format for float value."""
        try:
            if obj.isModified():
                self.clearLoadOptions()
                text = obj.text()
                if '-' in text or '+' in text:
                    tl = [ti for ti in text.split('-')]
                    for i in range(1, len(tl)):
                        tl[i] = '-' + tl[i]
                    ntl = []
                    for ti in tl:
                        ntl = ntl + ti.split('+')
                    values = [float(ti) for ti in ntl if len(ti) > 0]
                    value = sum(values)
                else:
                    value = float(text)
                obj.setText('{0:0.4f}'.format(value))
        except Exception:
            obj.setText('')

    def setStrFormatPositiveFloat(self, obj):
        """Set the line edit string format for positive float value."""
        try:
            if obj.isModified():
                self.clearLoadOptions()
                value = float(obj.text())
                if value >= 0:
                    obj.setText('{0:0.4f}'.format(value))
                else:
                    obj.setText('')
        except Exception:
            obj.setText('')

    def setStrFormatPositiveNonZeroFloat(self, obj):
        """Set the line edit string format for positive float value."""
        try:
            if obj.isModified():
                self.clearLoadOptions()
                value = float(obj.text())
                if value > 0:
                    obj.setText('{0:0.4f}'.format(value))
                else:
                    obj.setText('')
        except Exception:
            obj.setText('')

    def setStrFormatNonZeroFloat(self, obj):
        """Set the line edit string format for positive float value."""
        try:
            if obj.isModified():
                self.clearLoadOptions()
                value = float(obj.text())
                if value != 0:
                    obj.setText('{0:0.4f}'.format(value))
                else:
                    obj.setText('')
        except Exception:
            obj.setText('')

    def showViewProbeDialog(self):
        """Open view probe dialog."""
        self.view_probe_dialog.show(self.hall_probe)

    def uncheckRadioButtons(self, selected_axis):
        """Uncheck radio buttons."""
        axes = [a for a in self._measurement_axes if a != selected_axis]
        for axis in axes:
            second_rb = getattr(self.ui, 'second_ax' + str(axis) + '_rb')
            second_rb.setChecked(False)

    def updateConfiguration(self):
        """Update measurement configuration parameters."""
        try:
            self.measurement_config.clear()

            _s = self.ui.magnet_name_le.text()
            self.measurement_config.magnet_name = _s if len(_s) != 0 else None

            _s = self.ui.main_current_le.text()
            self.measurement_config.main_current = _s if len(_s) != 0 else None

            _s = self.ui.probe_name_cmb.currentText()
            self.measurement_config.probe_name = _s if len(_s) != 0 else 'None'

            _s = self.ui.temperature_le.text()
            self.measurement_config.temperature = _s if len(_s) != 0 else None

            _s = self.ui.operator_le.text()
            self.measurement_config.operator = _s if len(_s) != 0 else None

            _voltx_enable = self.ui.voltx_enable_chb.isChecked()
            self.measurement_config.voltx_enable = _voltx_enable

            _volty_enable = self.ui.volty_enable_chb.isChecked()
            self.measurement_config.volty_enable = _volty_enable

            _voltz_enable = self.ui.voltz_enable_chb.isChecked()
            self.measurement_config.voltz_enable = _voltz_enable

            idx = self.ui.voltage_precision_cmb.currentIndex()
            self.measurement_config.voltage_precision = idx
            nr_meas = self.ui.nr_measurements_sb.value()
            self.measurement_config.nr_measurements = nr_meas

            integration_time = self.ui.integration_time_le.text()
            if bool(integration_time and integration_time.strip()):
                self.measurement_config.integration_time = float(
                    integration_time)

            for axis in self._measurement_axes:
                first_rb = getattr(self.ui, 'first_ax' + str(axis) + '_rb')
                second_rb = getattr(self.ui, 'second_ax' + str(axis) + '_rb')
                if first_rb.isChecked():
                    self.measurement_config.first_axis = axis
                elif second_rb.isChecked():
                    self.measurement_config.second_axis = axis

                start = self.getAxisParam('start', axis)
                self.measurement_config.set_start(axis, start)

                step_le = getattr(self.ui, 'step_ax' + str(axis) + '_le')
                if step_le.isEnabled():
                    step = self.getAxisParam('step', axis)
                    self.measurement_config.set_step(axis, step)
                else:
                    self.measurement_config.set_step(axis, 0.0)

                end_le = getattr(self.ui, 'end_ax' + str(axis) + '_le')
                if end_le.isEnabled():
                    end = self.getAxisParam('end', axis)
                    self.measurement_config.set_end(axis, end)
                else:
                    self.measurement_config.set_end(axis, start)

                extra_le = getattr(self.ui, 'extra_ax' + str(axis) + '_le')
                if extra_le.isEnabled():
                    extra = self.getAxisParam('extra', axis)
                    self.measurement_config.set_extra(axis, extra)
                else:
                    self.measurement_config.set_extra(axis, 0.0)

                vel = self.getAxisParam('vel', axis)
                self.measurement_config.set_velocity(axis, vel)

            if self.measurement_config.second_axis is None:
                self.measurement_config.second_axis = -1

            if self.measurement_config.valid_data():
                return True

            else:
                msg = 'Invalid measurement configuration.'
                _QMessageBox.critical(
                    self, 'Failure', msg, _QMessageBox.Ok)
                return False

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to update configuration.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def updateConfigurationIDs(self):
        """Update combo box ids."""
        current_text = self.ui.idn_cmb.currentText()
        load_enabled = self.ui.loaddb_btn.isEnabled()
        self.ui.idn_cmb.clear()
        try:
            idns = self.measurement_config.get_table_column(
                self.database, 'id')
            self.ui.idn_cmb.clear()
            self.ui.idn_cmb.addItems([str(idn) for idn in idns])
            if len(current_text) == 0:
                self.ui.idn_cmb.setCurrentIndex(-1)
            else:
                self.ui.idn_cmb.setCurrentText(current_text)
            self.ui.loaddb_btn.setEnabled(load_enabled)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def updateProbeNames(self):
        """Update combo box with database probe names."""
        current_text = self.ui.probe_name_cmb.currentText()
        self.ui.probe_name_cmb.clear()
        try:
            probe_names = self.hall_probe.get_table_column(
                self.database, 'probe_name')
            self.ui.probe_name_cmb.addItems(probe_names)
            if len(current_text) == 0:
                self.ui.probe_name_cmb.setCurrentIndex(-1)
            else:
                self.ui.probe_name_cmb.setCurrentText(current_text)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass
