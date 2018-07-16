# -*- coding: utf-8 -*-

"""Measurement widget for the Hall Bench Control application."""

import os.path as _path
import time as _time
import numpy as _np
import threading as _threading
import warnings as _warnings
from PyQt5.QtWidgets import (
    QWidget as _QWidget,
    QApplication as _QApplication,
    QVBoxLayout as _QVBoxLayout,
    QFileDialog as _QFileDialog,
    QMessageBox as _QMessageBox,
    )
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.positionwidget import PositionWidget as _PositionWidget
from hallbench.gui.savefieldmapdialog import SaveFieldMapDialog \
    as _SaveFieldMapDialog
from hallbench.gui.selectcalibrationdialog import SelectCalibrationDialog \
    as _SelectCalibrationDialog
from hallbench.data.configuration import MeasurementConfig \
    as _MeasurementConfig
from hallbench.data.utils import get_timestamp as _get_timestamp
from hallbench.data.measurement import VoltageData as _VoltageData
from hallbench.data.measurement import FieldData as _FieldData
from hallbench.data.measurement import FieldMapData as _FieldMapData


class MeasurementWidget(_QWidget):
    """Measurement widget class for the Hall Bench Control application."""

    _update_graph_time_interval = 0.05  # [s]
    _measurement_axes = [1, 2, 3, 5]

    def __init__(self, parent=None):
        """Set up the ui, add widgets and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(__file__, self)
        self.ui = _uic.loadUi(uifile, self)

        # add position widget
        self.position_widget = _PositionWidget(self)
        layout = _QVBoxLayout()
        layout.addWidget(self.position_widget)
        self.ui.position_wg.setLayout(layout)

        # create save dialog widget
        self.save_dialog = _SaveFieldMapDialog()
        self.select_calibration_dialog = _SelectCalibrationDialog()

        # variables initialization
        self.probe_calibration = None
        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.position_list = []
        self.voltage_data = None
        self.field_data = None
        self.fieldmap_data = None
        self.config = None
        self.measurement_probe_calibration = None
        self.measurement_directory = None
        self.threadx = None
        self.thready = None
        self.threadz = None
        self.stop = False

        self.connectSignalSlots()

    @property
    def devices(self):
        """Hall Bench Devices."""
        return self.window().devices

    @property
    def database(self):
        """Database filename."""
        return self.window().database

    def clearGraph(self):
        """Clear plots."""
        self.ui.graph_pw.plotItem.curves.clear()
        self.ui.graph_pw.clear()
        self.graphx = []
        self.graphy = []
        self.graphz = []

    def closeDialogs(self):
        """Close dialogs."""
        self.save_dialog.close()
        self.select_calibration_dialog.close()

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

        self.ui.loadconfigfile_btn.clicked.connect(self.loadConfigurationFile)
        self.ui.saveconfigfile_btn.clicked.connect(self.saveConfigurationFile)
        self.ui.loadconfigdb_btn.clicked.connect(self.loadConfigurationDB)
        self.ui.selectcalibration_btn.clicked.connect(
            self.showSelectCalibrationDialog)
        self.ui.measure_btn.clicked.connect(self.configureAndMeasure)
        self.ui.stop_btn.clicked.connect(self.stopMeasurement)
        self.ui.savefieldmap_btn.clicked.connect(self.showSaveFieldMapDialog)

        self.select_calibration_dialog.probeCalibrationChanged.connect(
            self.updateProbeCalibration)

    def configureAndMeasure(self):
        """Configure and start measurement."""
        if (self.devices is None
           or not self.validProbeCalibration()
           or not self.updateConfiguration()
           or not self.configureDevices()):
            return

        self.ui.stop_btn.setEnabled(True)
        self.clearGraph()
        self.stop = False
        self.fieldmap_data = None

        first_axis = self.config.first_axis
        second_axis = self.config.second_axis
        fixed_axes = self.getFixedAxes()

        for axis in fixed_axes:
            if self.stop is True:
                return
            pos = self.config.get_start(axis)
            self.moveAxis(axis, pos)

        field_data_list = []
        if second_axis != -1:
            start = self.config.get_start(second_axis)
            end = self.config.get_end(second_axis)
            step = self.config.get_step(second_axis)
            npts = _np.ceil(round((end - start) / step, 4) + 1)
            pos_list = _np.linspace(start, end, npts)

            for pos in pos_list:
                if self.stop is True:
                    return
                self.moveAxis(second_axis, pos)
                self.scanLine(first_axis)
                field_data_list.append(self.field_data)

            # move to initial position
            if self.stop is True:
                return
            second_axis_start = self.config.get_start(second_axis)
            self.moveAxis(second_axis, second_axis_start)

        else:
            if self.stop is True:
                return
            self.scanLine(first_axis)
            field_data_list.append(self.field_data)

        # move to initial position
        if self.stop is True:
            return
        first_axis_start = self.config.get_start(first_axis)
        self.moveAxis(second_axis, first_axis_start)

        self.updateFieldMapData(field_data_list)
        self.plotFieldMapData()
        self.ui.stop_btn.setEnabled(False)
        self.ui.savefieldmap_btn.setEnabled(True)

        message = 'End of measurements.'
        _QMessageBox.information(
            self, 'Measurements', message, _QMessageBox.Ok)

    def configureDevices(self):
        """Configure devices."""
        try:
            if self.config.voltx_enable:
                self.devices.voltx.config(
                    self.config.integration_time,
                    self.config.voltage_precision)
            if self.config.volty_enable:
                self.devices.volty.config(
                    self.config.integration_time,
                    self.config.voltage_precision)
            if self.config.voltz_enable:
                self.devices.voltz.config(
                    self.config.integration_time,
                    self.config.voltage_precision)

            self.devices.pmac.set_axis_speed(1, self.config.vel_ax1)
            self.devices.pmac.set_axis_speed(2, self.config.vel_ax2)
            self.devices.pmac.set_axis_speed(3, self.config.vel_ax3)
            self.devices.pmac.set_axis_speed(5, self.config.vel_ax5)
            return True
        except Exception:
            message = 'Failed to configure devices.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return False

    def configureGraph(self, nr_curves, label):
        """Configure graph.

        Args:
            nr_curves (int): number of curves to plot.
        """
        self.graphx = []
        self.graphy = []
        self.graphz = []

        for idx in range(nr_curves):
            self.graphx.append(
                self.ui.graph_pw.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(255, 0, 0),
                    symbol='o',
                    symbolPen=(255, 0, 0),
                    symbolSize=4))

            self.graphy.append(
                self.ui.graph_pw.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(0, 255, 0),
                    symbol='o',
                    symbolPen=(0, 255, 0),
                    symbolSize=4))

            self.graphz.append(
                self.ui.graph_pw.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(0, 0, 255),
                    symbol='o',
                    symbolPen=(0, 0, 255),
                    symbolSize=4))

        self.ui.graph_pw.setLabel('bottom', 'Scan Position [mm]')
        self.ui.graph_pw.setLabel('left', label)
        self.ui.graph_pw.showGrid(x=True, y=True)

    def configurePmacTrigger(self, axis, pos, step, npts):
        """Configure Pmac trigger."""
        self.devices.pmac.set_trigger(axis, pos, step, 10, npts, 1)

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
        if self.probe_calibration is None:
            return

        if len(self.probe_calibration.sensorx.data) == 0:
            self.ui.voltx_enable_chb.setChecked(False)
            self.ui.voltx_enable_chb.setEnabled(False)
        else:
            self.ui.voltx_enable_chb.setEnabled(True)

        if len(self.probe_calibration.sensory.data) == 0:
            self.ui.volty_enable_chb.setChecked(False)
            self.ui.volty_enable_chb.setEnabled(False)
        else:
            self.ui.volty_enable_chb.setEnabled(True)

        if len(self.probe_calibration.sensorz.data) == 0:
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
            corrected_end = start + (npts-1)*step
            end_le.setText('{0:0.4f}'.format(corrected_end))

    def getAxisParam(self, param, axis):
        """Get axis parameter."""
        le = getattr(self.ui, param + '_ax' + str(axis) + '_le')
        le_text = le.text()
        if bool(le_text and le_text.strip()):
            return float(le_text)
        else:
            return None

    def getFixedAxes(self):
        """Get fixed axes."""
        first_axis = self.config.first_axis
        second_axis = self.config.second_axis
        fixed_axes = [a for a in self._measurement_axes]
        fixed_axes.remove(first_axis)
        if second_axis != -1:
            fixed_axes.remove(second_axis)
        return fixed_axes

    def killReadingThreads(self):
        """Kill threads."""
        try:
            del self.threadx
            del self.thready
            del self.threadz
        except Exception:
            pass

    def loadConfiguration(self):
        """Set measurement parameters."""
        self.ui.magnet_name_le.setText(self.config.magnet_name)
        self.ui.main_current_le.setText(self.config.main_current)
        self.ui.probe_name_le.setText(self.config.probe_name)

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

        self.disableInvalidLineEdit()

    def loadConfigurationDB(self):
        """Load configuration from database to set measurement parameters."""
        self.ui.filename_le.setText("")

        try:
            idn = int(self.ui.idn_le.text())
        except Exception:
            _QMessageBox.critical(
                self, 'Failure', 'Invalid database ID.', _QMessageBox.Ok)
            return

        try:
            self.config = _MeasurementConfig(self.database, idn)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)
            return

        self.loadConfiguration()

    def loadConfigurationFile(self):
        """Load configuration file to set measurement parameters."""
        self.ui.idn_le.setText("")

        default_filename = self.ui.filename_le.text()
        filename = _QFileDialog.getOpenFileName(
            self, caption='Open measurement configuration file',
            directory=default_filename, filter="Text files (*.txt)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            self.config = _MeasurementConfig(filename)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)
            return

        self.ui.filename_le.setText(filename)
        self.loadConfiguration()

    def moveAxis(self, axis, position):
        """Move bench axis.

        Args:
            axis (int): axis number.
            position (float): target position.
        """
        if self.stop is False:
            self.devices.pmac.move_axis(axis, position)
            while ((self.devices.pmac.axis_status(axis) & 1) == 0 and
                   self.stop is False):
                _QApplication.processEvents()

    def moveAxisAndUpdateGraph(self, axis, position, idx):
        """Move bench axis and update plot with the measure data.

        Args:
            axis (int): axis number.
            position (float): target position.
            idx (int):  index of the plot to update.
        """
        if self.stop is False:
            self.devices.pmac.move_axis(axis, position)
            while ((self.devices.pmac.axis_status(axis) & 1) == 0 and
                   self.stop is False):
                self.updateGraph(idx)
                _QApplication.processEvents()
                _time.sleep(self._update_graph_time_interval)

    def plotFieldMapData(self):
        """Plot field map data."""
        self.clearGraph()

        field1 = self.fieldmap_data.field3
        field2 = self.fieldmap_data.field3
        field3 = self.fieldmap_data.field3

        nr_curves = len(field1.shape[1])
        positions = field1.index.values

        self.configureGraph(nr_curves, 'Magnetic Field [T]')

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            for i in len(nr_curves):
                self.graphx[i].setData(positions, field3.iloc[:, i].values)
                self.graphy[i].setData(positions, field2.iloc[:, i].values)
                self.graphz[i].setData(positions, field1.iloc[:, i].values)

    def pmacPosition(self, axis):
        """Get Pmac position."""
        return self.devices.pmac.get_position(axis)

    def saveConfigurationFile(self):
        """Save measurement parameters to file."""
        default_filename = self.ui.filename_le.text()
        filename = _QFileDialog.getSaveFileName(
            self, caption='Save measurement configuration file',
            directory=default_filename, filter="Text files (*.txt)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        if self.updateConfiguration():
            try:
                if not filename.endswith('.txt'):
                    filename = filename + '.txt'
                self.config.save_file(filename)
            except Exception as e:
                _QMessageBox.critical(
                    self, 'Failure', str(e), _QMessageBox.Ok)

    def saveFieldData(self):
        """Save field data to file."""
        if not self.save_field:
            return

        if self.field_data is None:
            message = 'Invalid field data.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        if self.measurement_directory is None:
            message = 'Invalid directory.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        timestamp = _get_timestamp()
        filename = timestamp + '_field_data'
        first_axis = self.config.first_axis

        axes = [1, 2, 3]
        axes.remove(first_axis)
        for axis in axes:
            pos = getattr(self.field_data, 'pos' + str(axis))[0]
            filename = filename+'_pos'+str(axis)+'={0:0.4f}mm'.format(pos)
        filename = filename + '.txt'

        filepath = _path.join(self.measurement_directory, filename)
        self.field_data.save_file(filepath)

    def saveVoltageData(self, to_pos):
        """Save voltage data to file."""
        if not self.save_voltage:
            return

        if self.voltage_data is None:
            message = 'Invalid voltage data.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        if self.measurement_directory is None:
            message = 'Invalid directory.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        timestamp = _get_timestamp()
        filename = timestamp + '_voltage_data'
        first_axis = self.config.first_axis

        axes = [1, 2, 3]
        axes.remove(first_axis)
        for axis in axes:
            pos = getattr(self.voltage_data, 'pos' + str(axis))[0]
            filename = filename+'_pos'+str(axis)+'={0:0.4f}mm'.format(pos)

        if to_pos:
            filename = filename + '_to_pos.txt'
        else:
            filename = filename + '_to_neg.txt'

        filepath = _path.join(self.measurement_directory, filename)
        self.voltage_data.save_file(filepath)

    def scanLine(self, first_axis):
        """Start line scan."""
        start = self.config.get_start(first_axis)
        end = self.config.get_end(first_axis)
        step = self.config.get_step(first_axis)
        extra = self.config.get_extra(first_axis)
        vel = self.config.get_velocity(first_axis)

        aper_displacement = self.config.integration_time * vel
        npts = _np.ceil(round((end - start) / step, 4) + 1)
        scan_list = _np.linspace(start, end, npts)
        to_pos_scan_list = scan_list + aper_displacement/2
        to_neg_scan_list = (scan_list - aper_displacement/2)[::-1]

        nr_measurements = self.config.nr_measurements
        self.configureGraph(nr_measurements, 'Voltage [V]')

        voltage_data_list = []
        for idx in range(nr_measurements):
            if self.stop is True:
                return

            self.voltage_data = _VoltageData()
            self.devices.voltx.end_measurement = False
            self.devices.volty.end_measurement = False
            self.devices.voltz.end_measurement = False
            self.devices.voltx.clear()
            self.devices.volty.clear()
            self.devices.voltz.clear()
            self.updateMeasurementNumber(idx + 1)

            # flag to check if sensor is going or returning
            to_pos = not(bool(idx % 2))

            # go to initial position
            if to_pos:
                self.position_list = to_pos_scan_list
                self.moveAxis(first_axis, start - extra)
            else:
                self.position_list = to_neg_scan_list
                self.moveAxis(first_axis, end + extra)

            self.savePositionValuesToVoltageData()

            if self.stop is True:
                return

            # start threads
            self.startReadingThreads()

            # start firstger measurement
            if self.stop is False:
                if to_pos:
                    self.configurePmacTrigger(first_axis, start, step, npts)
                    self.moveAxisAndUpdateGraph(first_axis, end + extra, idx)
                else:
                    self.configurePmacTrigger(first_axis, end, step*(-1), npts)
                    self.moveAxisAndUpdateGraph(first_axis, start - extra, idx)

            # stop trigger and copy voltage values to voltage data
            self.stopTrigger()
            self.waitReadingThreads()
            self.voltage_data.sensorx = self.devices.voltx.voltage
            self.voltage_data.sensory = self.devices.volty.voltage
            self.voltage_data.sensorz = self.devices.voltz.voltage
            self.killReadingThreads()

            if self.stop is True:
                return

            if not to_pos:
                self.voltage_data.reverse()
            voltage_data_list.append(self.voltage_data.copy())
            self.saveCurrentVoltageData(to_pos)

        self.updateFieldData(voltage_data_list)

    def savePositionValuesToVoltageData(self):
        """Save positions in voltage data."""
        if self.voltage_data is None:
            self.voltage_data = _VoltageData()

        first_axis = self.config.first_axis

        for axis in self.voltage_data.axis_list:
            if axis != first_axis:
                pos = self.pmacPosition(axis)
                setattr(self.voltage_data, 'pos' + str(axis), pos)
            else:
                setattr(self.voltage_data, 'pos' + str(first_axis),
                        self.position_list)

    def setStrFormatFloat(self, obj):
        """Set the line edit string format for float value."""
        try:
            value = float(obj.text())
            obj.setText('{0:0.4f}'.format(value))
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

    def setStrFormatInteger(self, obj):
        """Set the line edit string format for integer value."""
        try:
            value = int(obj.text())
            obj.setText('{0:d}'.format(value))
        except Exception:
            obj.setText('')

    def showSaveFieldMapDialog(self):
        """Open save fieldmap dialog."""
        if self.fieldmap_data is None:
            message = 'Invalid field map data.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        if self.measurement_directory is None:
            self.save_dialog.show(self.fieldmap_data)
        else:
            self.save_dialog.show(
                self.fieldmap_data, self.measurement_directory)

    def showSelectCalibrationDialog(self):
        """Open select calibration dialog."""
        self.select_calibration_dialog.show(self.database)

    def startReadingThreads(self):
        """Start threads to read voltage values."""
        if self.config.voltx_enable:
            self.threadx = _threading.Thread(
                target=self.devices.voltx.read,
                args=(self.config.voltage_precision,))
            self.threadx.start()

        if self.config.volty_enable:
            self.thready = _threading.Thread(
                target=self.devices.volty.read,
                args=(self.config.voltage_precision,))
            self.thready.start()

        if self.config.voltz_enable:
            self.threadz = _threading.Thread(
                target=self.devices.voltz.read,
                args=(self.config.voltage_precision,))
            self.threadz.start()

    def stopMeasurement(self):
        """Stop measurement to True."""
        self.stop = True
        self.devices.pmac.stop_all_axis()
        self.ui.stop_btn.setEnabled(False)
        message = 'The user stopped the measurements.'
        _QMessageBox.information(
            self, 'Abort', message, _QMessageBox.Ok)

    def stopTrigger(self):
        """Stop trigger."""
        self.devices.pmac.stop_trigger()
        self.devices.voltx.end_measurement = True
        self.devices.volty.end_measurement = True
        self.devices.voltz.end_measurement = True

    def uncheckRadioButtons(self, selected_axis):
        """Uncheck radio buttons."""
        axes = [a for a in self._measurement_axes if a != selected_axis]
        for axis in axes:
            second_rb = getattr(self.ui, 'second_ax' + str(axis) + '_rb')
            second_rb.setChecked(False)

    def updateConfiguration(self):
        """Update measurement configuration parameters."""
        if self.config is None:
            self.config = _MeasurementConfig()

        self.config.clear()

        _s = self.ui.magnet_name_le.text()
        self.config.magnet_name = _s if len(_s) != 0 else None

        _s = self.ui.main_current_le.text()
        self.config.main_current = _s if len(_s) != 0 else None

        _s = self.ui.probe_name_le.text()
        self.config.probe_name = _s if len(_s) != 0 else None

        _voltx_enable = self.ui.voltx_enable_chb.isChecked()
        self.config.voltx_enable = _voltx_enable

        _volty_enable = self.ui.volty_enable_chb.isChecked()
        self.config.volty_enable = _volty_enable

        _voltz_enable = self.ui.voltz_enable_chb.isChecked()
        self.config.voltz_enable = _voltz_enable

        idx = self.ui.voltage_precision_cmb.currentIndex()
        self.config.voltage_precision = idx
        self.config.nr_measurements = self.ui.nr_measurements_sb.value()

        integration_time = self.ui.integration_time_le.text()
        if bool(integration_time and integration_time.strip()):
            self.config.integration_time = float(integration_time)

        self.updateParametersFromTable()

        if self.config.valid_data():
            return True
        else:
            message = 'Invalid measurement configuration.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return False

    def updateFieldData(self, voltage_data_list):
        """Update field data."""
        try:
            self.field_data = _FieldData()
            self.field_data.probe_calibration = (
                self.measurement_probe_calibration)
            self.field_data.voltage_data_list = voltage_data_list
            self.saveFieldData()
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def updateFieldMapData(self, field_data_list):
        """Update field map data."""
        try:
            self.fieldmap_data = _FieldMapData()
            self.fieldmap_data.probe_calibration = (
                self.measurement_probe_calibration)
            self.fieldmap_data.correct_sensor_displacement = (
                self.ui.correctdisp_chb.isChecked())
            self.fieldmap_data.field_data_list = field_data_list
            self.fieldmapDataBackup()
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def updateGraph(self, idx):
        """Update plot.

        Args:
            idx (int): index of the plot to update.
        """
        voltagex = self.devices.voltx.voltage
        voltagey = self.devices.volty.voltage
        voltagez = self.devices.voltz.voltage

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            self.graphx[idx].setData(
                self.position_list[:len(voltagex)], voltagex)
            self.graphy[idx].setData(
                self.position_list[:len(voltagey)], voltagey)
            self.graphz[idx].setData(
                self.position_list[:len(voltagez)], voltagez)

    def updateMeasurementNumber(self, measurement_number):
        """Update measurement number."""
        self.ui.nr_measurements_la.setText(str(measurement_number))

    def updateParametersFromTable(self):
        """Update configuration parameters with the table values."""
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

    def updatePositions(self):
        """Update axes positions."""
        if self.devices.pmac is None:
            return
        self.position_widget.updatePositions()

    def updateProbeCalibration(self, probe_calibration):
        """Update probe calibration."""
        self.probe_calibration = probe_calibration
        if self.probe_calibration is not None:
            self.ui.probe_name_le.setText(self.probe_calibration.probe_name)
            self.disableInvalidVoltimeter()

    def validProbeCalibration(self):
        """Check if the probe calibration is valid."""
        self.measurement_probe_calibration = self.probe_calibration

        if self.measurement_probe_calibration is not None:
            self.disableInvalidVoltimeter()
            return True
        else:
            message = 'Invalid probe calibration data.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return False

    def waitReadingThreads(self):
        """Wait threads."""
        if self.threadx is not None:
            while self.threadx.is_alive() and self.stop is False:
                _QApplication.processEvents()

        if self.thready is not None:
            while self.thready.is_alive() and self.stop is False:
                _QApplication.processEvents()

        if self.threadz is not None:
            while self.threadz.is_alive() and self.stop is False:
                _QApplication.processEvents()
