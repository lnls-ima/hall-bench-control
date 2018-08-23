# -*- coding: utf-8 -*-

"""Configuration widget for the Hall Bench Control application."""

import numpy as _np
from PyQt4.QtGui import (
    QWidget as _QWidget,
    QDialog as _QDialog,
    QFileDialog as _QFileDialog,
    QVBoxLayout as _QVBoxLayout,
    QMessageBox as _QMessageBox,
    )
import PyQt4.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.hallprobedialog import HallProbeDialog \
    as _HallProbeDialog
from hallbench.data.configuration import MeasurementConfig \
    as _MeasurementConfig


class ConfigurationDialog(_QDialog):
    """Configuration dialog class for the Hall Bench Control application."""

    def __init__(self, parent=None, load_enabled=True):
        """Add configuration widget."""
        super().__init__(parent)
        self.database = None
        self._load_enabled = load_enabled

        self.main_wg = ConfigurationWidget(
            self, load_enabled=self._load_enabled)
        self.main_wg.hall_probe_btn.setText('Show Hall Probe')

        _layout = _QVBoxLayout()
        _layout.addWidget(self.main_wg)
        self.setLayout(_layout)

    def show(self, database):
        """Update database and show dialog."""
        self.database = database

        if self.database is not None:
            self.main_wg.idn_le.setEnabled(True)
            if self._load_enabled:
                self.main_wg.loaddb_btn.setEnabled(True)
            else:
                self.main_wg.loaddb_btn.setEnabled(False)
        else:
            self.main_wg.loaddb_btn.setEnabled(False)
            self.main_wg.idn_le.setEnabled(False)

        super().show()


class ConfigurationWidget(_QWidget):
    """Configuration widget class for the Hall Bench Control application."""

    _measurement_axes = [1, 2, 3, 5]

    def __init__(self, parent=None, load_enabled=True):
        """Set up the ui, add widgets and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # create dialog
        self.hall_probe_dialog = _HallProbeDialog(load_enabled=load_enabled)

        # variables initialization
        self.config = None
        self.hall_probe = None
        self._load_enabled = load_enabled

        # Enable or disable load option
        self.ui.loadfile_btn.setEnabled(self._load_enabled)
        self.ui.savefile_btn.setEnabled(self._load_enabled)
        self.ui.loaddb_btn.setEnabled(self._load_enabled)
        self.ui.filename_le.setEnabled(self._load_enabled)
        self.ui.idn_le.setReadOnly(not self._load_enabled)

        self.setControlsEnabled()
        self.connectSignalSlots()

    @property
    def devices(self):
        """Hall Bench Devices."""
        return self.window().devices

    @property
    def database(self):
        """Database filename."""
        return self.window().database

    def closeDialogs(self):
        """Close dialogs."""
        try:
            self.hall_probe_dialog.accept()
        except Exception:
            pass

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.start_ax1_le.editingFinished.connect(
            lambda: self.setStrFormatFloat(self.ui.start_ax1_le))
        self.ui.start_ax2_le.editingFinished.connect(
            lambda: self.setStrFormatFloat(self.ui.start_ax2_le))
        self.ui.start_ax3_le.editingFinished.connect(
            lambda: self.setStrFormatFloat(self.ui.start_ax3_le))
        self.ui.start_ax5_le.editingFinished.connect(
            lambda: self.setStrFormatFloat(self.ui.start_ax5_le))

        self.ui.end_ax1_le.editingFinished.connect(
            lambda: self.setStrFormatFloat(self.ui.end_ax1_le))
        self.ui.end_ax2_le.editingFinished.connect(
            lambda: self.setStrFormatFloat(self.ui.end_ax2_le))
        self.ui.end_ax3_le.editingFinished.connect(
            lambda: self.setStrFormatFloat(self.ui.end_ax3_le))
        self.ui.end_ax5_le.editingFinished.connect(
            lambda: self.setStrFormatFloat(self.ui.end_ax5_le))

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

        self.ui.idn_le.editingFinished.connect(
            lambda: self.setStrFormatInteger(self.ui.idn_le))

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
        self.ui.hall_probe_btn.clicked.connect(
            self.showHallProbeDialog)

        self.hall_probe_dialog.hallProbeChanged.connect(
            self.setHallProbe)

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

    def disableInvalidVoltimeter(self):
        """Disable invalid voltimeters."""
        if self.hall_probe is None:
            return

        sx = self.hall_probe.sensorx
        sy = self.hall_probe.sensory
        sz = self.hall_probe.sensorz
        
        if sx is None or len(sx.data) == 0:
            self.ui.voltx_enable_chb.setChecked(False)
            self.ui.voltx_enable_chb.setEnabled(False)
        else:
            self.ui.voltx_enable_chb.setEnabled(True)

        if sy is None or len(sy.data) == 0:
            self.ui.volty_enable_chb.setChecked(False)
            self.ui.volty_enable_chb.setEnabled(False)
        else:
            self.ui.volty_enable_chb.setEnabled(True)

        if sz is None or len(sz.data) == 0:
            self.ui.voltz_enable_chb.setChecked(False)
            self.ui.voltz_enable_chb.setEnabled(False)
        else:
            self.ui.voltz_enable_chb.setEnabled(True)

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

    def searchHallProbeDB(self):
        """Load hall probe from database."""
        try:
            self.hall_probe_dialog.database = self.database
            self.hall_probe_dialog.searchDB(self.config.probe_name)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def load(self):
        """Set measurement parameters."""
        try:
            self.ui.magnet_name_le.setText(self.config.magnet_name)
            self.ui.main_current_le.setText(self.config.main_current)
            self.ui.probe_name_le.setText(self.config.probe_name)
            self.ui.temperature_le.setText(self.config.temperature)
            self.ui.operator_le.setText(self.config.operator)

            self.ui.voltx_enable_chb.setChecked(self.config.voltx_enable)
            self.ui.volty_enable_chb.setChecked(self.config.volty_enable)
            self.ui.voltz_enable_chb.setChecked(self.config.voltz_enable)

            self.ui.integration_time_le.setText('{0:0.4f}'.format(
                self.config.integration_time))
            self.ui.voltage_precision_cmb.setCurrentIndex(
                self.config.voltage_precision)

            self.ui.nr_measurements_sb.setValue(self.config.nr_measurements)

            first_axis = self.config.first_axis
            first_rb = getattr(self.ui, 'first_ax' + str(first_axis) + '_rb')
            first_rb.setChecked(True)

            self.disableSecondAxisButton()

            second_axis = self.config.second_axis
            if second_axis != -1:
                second_rb = getattr(
                    self.ui, 'second_ax' + str(second_axis) + '_rb')
                second_rb.setChecked(True)
                self.uncheckRadioButtons(second_axis)
            else:
                self.uncheckRadioButtons(second_axis)

            for axis in self._measurement_axes:
                start_le = getattr(self.ui, 'start_ax' + str(axis) + '_le')
                value = self.config.get_start(axis)
                start_le.setText('{0:0.4f}'.format(value))

                step_le = getattr(self.ui, 'step_ax' + str(axis) + '_le')
                value = self.config.get_step(axis)
                step_le.setText('{0:0.4f}'.format(value))

                end_le = getattr(self.ui, 'end_ax' + str(axis) + '_le')
                value = self.config.get_end(axis)
                end_le.setText('{0:0.4f}'.format(value))

                extra_le = getattr(self.ui, 'extra_ax' + str(axis) + '_le')
                value = self.config.get_extra(axis)
                extra_le.setText('{0:0.4f}'.format(value))

                vel_le = getattr(self.ui, 'vel_ax' + str(axis) + '_le')
                value = self.config.get_velocity(axis)
                vel_le.setText('{0:0.4f}'.format(value))

            self.searchHallProbeDB()
            self.disableInvalidLineEdit()
            self.setControlsEnabled()

        except Exception:
            self.setControlsEnabled()
            message = 'Fail to load configuration.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)

    def loadDB(self):
        """Load configuration from database to set measurement parameters."""
        self.ui.filename_le.setText("")

        try:
            idn = int(self.ui.idn_le.text())
        except Exception:
            _QMessageBox.critical(
                self, 'Failure', 'Invalid database ID.', _QMessageBox.Ok)
            return

        try:
            self.config = _MeasurementConfig(database=self.database, idn=idn)
        except Exception as e:
            _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)
            return

        self.load()

    def loadFile(self):
        """Load configuration file to set measurement parameters."""
        self.ui.idn_le.setText("")

        default_filename = self.ui.filename_le.text()
        filename = _QFileDialog.getOpenFileName(
            self, caption='Open measurement configuration file',
            directory=default_filename, filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            self.config = _MeasurementConfig(filename)
        except Exception as e:
            _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)
            return

        self.ui.filename_le.setText(filename)
        self.load()

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
                self.config.save_file(filename)

            except Exception as e:
                _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)

    def setControlsEnabled(self):
        """Enable or disable controls."""
        read_only = True if not self._load_enabled else False

        self.ui.probe_name_le.setReadOnly(True)

        self.ui.magnet_name_le.setReadOnly(read_only)
        self.ui.main_current_le.setReadOnly(read_only)
        self.ui.temperature_le.setReadOnly(read_only)
        self.ui.operator_le.setReadOnly(read_only)
        self.ui.nr_measurements_sb.setReadOnly(read_only)
        self.ui.integration_time_le.setReadOnly(read_only)

        self.ui.voltx_enable_chb.setEnabled(self._load_enabled)
        self.ui.volty_enable_chb.setEnabled(self._load_enabled)
        self.ui.voltz_enable_chb.setEnabled(self._load_enabled)
        self.ui.voltage_precision_cmb.setEnabled(self._load_enabled)

        for axis in self._measurement_axes:
            first_rb = getattr(self.ui, 'first_ax' + str(axis) + '_rb')
            second_rb = getattr(self.ui, 'second_ax' + str(axis) + '_rb')
            if axis == 5:
                first_rb.setEnabled(False)
                second_rb.setEnabled(False)
            else:
                first_rb.setEnabled(self._load_enabled)
                second_rb.setEnabled(self._load_enabled)

            if first_rb.isChecked():
                second_rb.setChecked(False)
                second_rb.setEnabled(False)

            start_le = getattr(self.ui, 'start_ax' + str(axis) + '_le')
            start_le.setReadOnly(read_only)

            step_le = getattr(self.ui, 'step_ax' + str(axis) + '_le')
            step_le.setReadOnly(read_only)

            end_le = getattr(self.ui, 'end_ax' + str(axis) + '_le')
            end_le.setReadOnly(read_only)

            extra_le = getattr(self.ui, 'extra_ax' + str(axis) + '_le')
            extra_le.setReadOnly(read_only)

            vel_le = getattr(self.ui, 'vel_ax' + str(axis) + '_le')
            vel_le.setReadOnly(read_only)

    def setDatabaseID(self, idn, read_only=False):
        """Set database id text."""
        self.ui.filename_le.setText('')
        self.ui.idn_le.setText(str(idn))
        self.ui.idn_le.setReadOnly(read_only)
        self.ui.idn_le.setEnabled(True)

    def setHallProbe(self, hall_probe):
        """Set hall probe."""
        try:
            self.hall_probe = hall_probe
            if self.hall_probe is not None:
                self.ui.probe_name_le.setText(
                    self.hall_probe.probe_name)
                self.disableInvalidVoltimeter()
        except Exception as e:
            _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)

    def setStrFormatFloat(self, obj):
        """Set the line edit string format for float value."""
        try:
            value = float(obj.text())
            obj.setText('{0:0.4f}'.format(value))
        except Exception:
            obj.setText('')

    def setStrFormatInteger(self, obj):
        """Set the line edit string format for integer value."""
        try:
            value = int(obj.text())
            obj.setText('{0:d}'.format(value))
        except Exception:
            obj.setText('')

    def setStrFormatPositiveFloat(self, obj):
        """Set the line edit string format for positive float value."""
        try:
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
            value = float(obj.text())
            if value != 0:
                obj.setText('{0:0.4f}'.format(value))
            else:
                obj.setText('')
        except Exception:
            obj.setText('')

    def showHallProbeDialog(self):
        """Open hall probe dialog."""
        self.hall_probe_dialog.show(self.database)

    def uncheckRadioButtons(self, selected_axis):
        """Uncheck radio buttons."""
        axes = [a for a in self._measurement_axes if a != selected_axis]
        for axis in axes:
            second_rb = getattr(self.ui, 'second_ax' + str(axis) + '_rb')
            second_rb.setChecked(False)

    def updateConfiguration(self):
        """Update measurement configuration parameters."""
        try:
            self.config = _MeasurementConfig()

            _s = self.ui.magnet_name_le.text()
            self.config.magnet_name = _s if len(_s) != 0 else None

            _s = self.ui.main_current_le.text()
            self.config.main_current = _s if len(_s) != 0 else None

            _s = self.ui.probe_name_le.text()
            self.config.probe_name = _s if len(_s) != 0 else 'None'

            _s = self.ui.temperature_le.text()
            self.config.temperature = _s if len(_s) != 0 else None

            _s = self.ui.operator_le.text()
            self.config.operator = _s if len(_s) != 0 else None

            if self.devices.voltx.connected:
                _voltx_enable = self.ui.voltx_enable_chb.isChecked()
                self.config.voltx_enable = _voltx_enable
            else:
                self.ui.voltx_enable_chb.setChecked(False)
                self.config.voltx_enable = 0

            if self.devices.volty.connected:
                _volty_enable = self.ui.volty_enable_chb.isChecked()
                self.config.volty_enable = _volty_enable
            else:
                self.ui.volty_enable_chb.setChecked(False)
                self.config.volty_enable = 0

            if self.devices.voltz.connected:
                _voltz_enable = self.ui.voltz_enable_chb.isChecked()
                self.config.voltz_enable = _voltz_enable
            else:
                self.ui.voltz_enable_chb.setChecked(False)
                self.config.voltz_enable = 0

            idx = self.ui.voltage_precision_cmb.currentIndex()
            self.config.voltage_precision = idx
            self.config.nr_measurements = self.ui.nr_measurements_sb.value()

            integration_time = self.ui.integration_time_le.text()
            if bool(integration_time and integration_time.strip()):
                self.config.integration_time = float(integration_time)

            for axis in self._measurement_axes:
                first_rb = getattr(self.ui, 'first_ax' + str(axis) + '_rb')
                second_rb = getattr(self.ui, 'second_ax' + str(axis) + '_rb')
                if first_rb.isChecked():
                    self.config.first_axis = axis
                elif second_rb.isChecked():
                    self.config.second_axis = axis

                start = self.getAxisParam('start', axis)
                self.config.set_start(axis, start)

                step_le = getattr(self.ui, 'step_ax' + str(axis) + '_le')
                if step_le.isEnabled():
                    step = self.getAxisParam('step', axis)
                    self.config.set_step(axis, step)
                else:
                    self.config.set_step(axis, 0.0)

                end_le = getattr(self.ui, 'end_ax' + str(axis) + '_le')
                if end_le.isEnabled():
                    end = self.getAxisParam('end', axis)
                    self.config.set_end(axis, end)
                else:
                    self.config.set_end(axis, start)

                extra_le = getattr(self.ui, 'extra_ax' + str(axis) + '_le')
                if extra_le.isEnabled():
                    extra = self.getAxisParam('extra', axis)
                    self.config.set_extra(axis, extra)
                else:
                    self.config.set_extra(axis, 0.0)

                vel = self.getAxisParam('vel', axis)
                self.config.set_velocity(axis, vel)

            if self.config.second_axis is None:
                self.config.second_axis = -1

            if self.config.valid_data():
                return True

            else:
                message = 'Invalid measurement configuration.'
                _QMessageBox.critical(
                    self, 'Failure', message, _QMessageBox.Ok)
                return False

        except Exception as e:
            _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)
            return False
