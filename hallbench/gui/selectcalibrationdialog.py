# -*- coding: utf-8 -*-

"""Select Calibration widget for the Hall Bench Control application."""

import numpy as _np
import warnings as _warnings
from PyQt5.QtWidgets import (
    QDialog as _QDialog,
    QFileDialog as _QFileDialog,
    QMessageBox as _QMessageBox,
    )
from PyQt5.QtCore import pyqtSignal as _pyqtSignal
import PyQt5.uic as _uic
import pyqtgraph as _pyqtgraph

from hallbench.gui.interpolationtabledialog import InterpolationTableDialog \
    as _InterpolationTableDialog
from hallbench.gui.polynomialtabledialog import PolynomialTableDialog \
    as _PolynomialTableDialog
from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.data.calibration import ProbeCalibration as _ProbeCalibration


class SelectCalibrationDialog(_QDialog):
    """Select Calibration dialog class for Hall Bench Control application."""

    probeCalibrationChanged = _pyqtSignal(_ProbeCalibration)

    _axis_str_dict = {
        1: '+ Axis #1 (+Z)', 2: '+ Axis #2 (+Y)', 3: '+ Axis #3 (+X)'}

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(__file__, self)
        self.ui = _uic.loadUi(uifile, self)

        self.interpolation_dialog = _InterpolationTableDialog()
        self.polynomial_dialog = _PolynomialTableDialog()

        self._probe_calibration = None
        self.database = None

        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.configureGraph()
        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.viewdata_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        # create connections
        self.ui.loadfile_btn.clicked.connect(self.loadFile)
        self.ui.loaddb_btn.clicked.connect(self.loadDB)
        self.ui.showtable_btn.clicked.connect(self.showTable)
        self.ui.updategraph_btn.clicked.connect(self.updateGraph)
        self.ui.voltage_sb.valueChanged.connect(self.updateField)

    @property
    def probe_calibration(self):
        """Calibration data object."""
        return self._probe_calibration

    @probe_calibration.setter
    def probe_calibration(self, value):
        self._probe_calibration = value
        self.probeCalibrationChanged.emit(self.probe_calibration)
        self.closeDialogs()

    def closeDialogs(self):
        """Close table dialogs."""
        try:
            self.interpolation_dialog.accept()
            self.polynomial_dialog.accept()
        except Exception:
            pass

    def closeEvent(self, event):
        """Close dialog."""
        self.closeDialogs()
        super().closeEvent(event)

    def configureGraph(self, symbol=False):
        """Configure calibration data plots."""
        self.ui.viewdata_pw.clear()

        self.graphx = []
        self.graphy = []
        self.graphz = []

        penx = (255, 0, 0)
        peny = (0, 255, 0)
        penz = (0, 0, 255)

        if symbol:
            plot_item_x = self.ui.viewdata_pw.plotItem.plot(
                _np.array([]), _np.array([]), pen=penx,
                symbol='o', symbolPen=penx, symbolSize=4, symbolBrush=penx)

            plot_item_y = self.ui.viewdata_pw.plotItem.plot(
                _np.array([]), _np.array([]), pen=peny,
                symbol='o', symbolPen=peny, symbolSize=4, symbolBrush=peny)

            plot_item_z = self.ui.viewdata_pw.plotItem.plot(
                _np.array([]), _np.array([]), pen=penz,
                symbol='o', symbolPen=penz, symbolSize=4, symbolBrush=penz)
        else:
            plot_item_x = self.ui.viewdata_pw.plotItem.plot(
                _np.array([]), _np.array([]), pen=penx)

            plot_item_y = self.ui.viewdata_pw.plotItem.plot(
                _np.array([]), _np.array([]), pen=peny)

            plot_item_z = self.ui.viewdata_pw.plotItem.plot(
                _np.array([]), _np.array([]), pen=penz)

        self.graphx.append(plot_item_x)
        self.graphy.append(plot_item_y)
        self.graphz.append(plot_item_z)

        self.ui.viewdata_pw.setLabel('bottom', 'Voltage [V]')
        self.ui.viewdata_pw.setLabel('left', 'Magnetic Field [T]')
        self.ui.viewdata_pw.showGrid(x=True, y=True)

    def load(self):
        """Load probe calibration parameters."""
        try:
            self.ui.probe_name_le.setText(self._probe_calibration.probe_name)
            self.ui.calibration_magnet_le.setText(
                self._probe_calibration.calibration_magnet)
            self.ui.function_type_le.setText(
                self._probe_calibration.function_type.capitalize())

            probe_axis = self._probe_calibration.probe_axis
            if probe_axis in self._axis_str_dict.keys():
                self.ui.probe_axis_le.setText(self._axis_str_dict[probe_axis])
                self.ui.probe_axis_le.setEnabled(True)
            else:
                self.ui.probe_axis_le.setText('')
                self.ui.probe_axis_le.setEnabled(False)

            distance_xy = self._probe_calibration.distance_xy
            if distance_xy is not None:
                self.ui.distance_xy_le.setText('{0:0.4f}'.format(distance_xy))
                self.ui.distance_xy_le.setEnabled(True)
            else:
                self.ui.distance_xy_le.setText('')
                self.ui.distance_xy_le.setEnabled(False)

            distance_zy = self._probe_calibration.distance_zy
            if distance_zy is not None:
                self.ui.distance_zy_le.setText('{0:0.4f}'.format(distance_zy))
                self.ui.distance_zy_le.setEnabled(True)
            else:
                self.ui.distance_zy_le.setText('')
                self.ui.distance_zy_le.setEnabled(False)

            angle_xy = self._probe_calibration.angle_xy
            if angle_xy is not None:
                self.ui.angle_xy_le.setText('{0:0.4f}'.format(angle_xy))
                self.ui.angle_xy_le.setEnabled(True)
            else:
                self.ui.angle_xy_le.setText('')
                self.ui.angle_xy_le.setEnabled(False)

            angle_yz = self._probe_calibration.angle_yz
            if angle_yz is not None:
                self.ui.angle_yz_le.setText('{0:0.4f}'.format(angle_yz))
                self.ui.angle_yz_le.setEnabled(True)
            else:
                self.ui.angle_yz_le.setText('')
                self.ui.angle_yz_le.setEnabled(False)

            angle_xz = self._probe_calibration.angle_xz
            if angle_xz is not None:
                self.ui.angle_xz_le.setText('{0:0.4f}'.format(angle_xz))
                self.ui.angle_xz_le.setEnabled(True)
            else:
                self.ui.angle_xz_le.setText('')
                self.ui.angle_xz_le.setEnabled(False)

            self.ui.fieldx_le.setText('')
            self.ui.fieldy_le.setText('')
            self.ui.fieldz_le.setText('')

        except Exception:
            message = 'Failed to load calibration data.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

    def loadDB(self):
        """Load probe calibration from database."""
        self.ui.filename_le.setText("")
        self.setEnabled(False)
        self.updateGraph()

        try:
            idn = int(self.ui.idn_le.text())
        except Exception:
            _QMessageBox.critical(
                self, 'Failure', 'Invalid database ID.', _QMessageBox.Ok)
            return

        try:
            self.probe_calibration = _ProbeCalibration(
                database=self.database, idn=idn)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)
            return

        self.setEnabled(True)
        self.updateGraph()
        self.load()

    def loadFile(self):
        """Load probe calibration file."""
        self.setDatabaseID('')
        self.setEnabled(False)
        self.updateGraph()

        default_filename = self.ui.filename_le.text()
        filename = _QFileDialog.getOpenFileName(
            self, caption='Load probe calibration file',
            directory=default_filename, filter="Text files (*.txt)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            self.probe_calibration = _ProbeCalibration(filename)
        except Exception as e:
            _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)
            return

        self.filename_le.setText(filename)
        self.setEnabled(True)
        self.updateGraph()
        self.load()

    def setDatabaseID(self, idn, read_only=False):
        """Set database id text."""
        self.ui.idn_le.setText(str(idn))
        self.ui.idn_le.setReadOnly(read_only)
        self.ui.idn_le.setEnabled(True)

    def setEnabled(self, enabled):
        """Enable or disable controls."""
        self.ui.calibrationdata_gb.setEnabled(enabled)
        self.ui.showtable_btn.setEnabled(enabled)
        self.ui.plotoptions_gb.setEnabled(enabled)
        self.ui.updategraph_btn.setEnabled(enabled)
        self.ui.getfield_gb.setEnabled(enabled)

    def setLoadOptionEnabled(self, enabled):
        """Enable or disable load option."""
        self.ui.loadfile_btn.setEnabled(enabled)
        self.ui.loaddb_btn.setEnabled(enabled)
        self.ui.idn_le.setEnabled(enabled)
        self.ui.filename_le.setEnabled(enabled)

    def show(self, database):
        """Update database and show dialog."""
        self.database = database
        self.updateLoadDB()
        super().show()

    def showTable(self):
        """Show calibration data table."""
        if self._probe_calibration is None:
            return

        if self._probe_calibration.function_type == 'interpolation':
            self.polynomial_dialog.close()
            self.interpolation_dialog.show(self._probe_calibration)
        elif self._probe_calibration.function_type == 'polynomial':
            self.interpolation_dialog.close()
            self.polynomial_dialog.show(self._probe_calibration)
        else:
            return

    def updateLoadDB(self):
        """Update load from database controls."""
        if self.database is None:
            self.ui.loaddb_btn.setEnabled(False)
            self.ui.idn_le.setEnabled(False)
        else:
            self.ui.loaddb_btn.setEnabled(True)
            self.ui.idn_le.setEnabled(True)

    def updateField(self):
        """Convert voltage to magnetic field."""
        self.ui.fieldx_le.setText('')
        self.ui.fieldy_le.setText('')
        self.ui.fieldz_le.setText('')

        if self._probe_calibration is None:
            return

        try:
            volt = [self.ui.voltage_sb.value()]
            fieldx = self._probe_calibration.sensorx.convert_voltage(volt)[0]
            fieldy = self._probe_calibration.sensory.convert_voltage(volt)[0]
            fieldz = self._probe_calibration.sensorz.convert_voltage(volt)[0]

            if not _np.isnan(fieldx):
                self.ui.fieldx_le.setText('{0:0.4f}'.format(fieldx))

            if not _np.isnan(fieldy):
                self.ui.fieldy_le.setText('{0:0.4f}'.format(fieldy))

            if not _np.isnan(fieldz):
                self.ui.fieldz_le.setText('{0:0.4f}'.format(fieldz))
        except Exception as e:
            _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)
            return

    def updateGraph(self):
        """Update calibration data plots."""
        try:
            vmin = self.ui.voltagemin_sb.value()
            vmax = self.ui.voltagemax_sb.value()
            npts = self.ui.voltagenpts_sb.value()
            voltage = _np.linspace(vmin, vmax, npts)

            self.ui.viewdata_pw.clear()
            self.legend.removeItem('X')
            self.legend.removeItem('Y')
            self.legend.removeItem('Z')

            if self._probe_calibration is None or len(voltage) == 0:
                return

            fieldx = self._probe_calibration.sensorx.convert_voltage(voltage)
            fieldy = self._probe_calibration.sensory.convert_voltage(voltage)
            fieldz = self._probe_calibration.sensorz.convert_voltage(voltage)

            symbol = self.ui.addmarkers_chb.isChecked()
            self.configureGraph(symbol=symbol)

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")

                if not _np.all(_np.isnan(fieldx)):
                    self.graphx[0].setData(voltage, fieldx)
                    self.legend.addItem(self.graphx[0], 'X')

                if not _np.all(_np.isnan(fieldy)):
                    self.graphy[0].setData(voltage, fieldy)
                    self.legend.addItem(self.graphy[0], 'Y')

                if not _np.all(_np.isnan(fieldz)):
                    self.graphz[0].setData(voltage, fieldz)
                    self.legend.addItem(self.graphz[0], 'Z')

        except Exception:
            message = 'Failed to update plot.'
            _QMessageBox.critical(self, 'Failure', message, _QMessageBox.Ok)
            return
