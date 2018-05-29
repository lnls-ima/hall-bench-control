# -*- coding: utf-8 -*-

"""Load Calibration widget for the Hall Bench Control application."""

import numpy as _np
import warnings as _warnings
from PyQt5.QtWidgets import (
    QWidget as _QWidget,
    QFileDialog as _QFileDialog,
    QMessageBox as _QMessageBox,
    )
import PyQt5.uic as _uic
import pyqtgraph as _pyqtgraph

from hallbench.gui.interpolationtabledialog import InterpolationTableDialog \
    as _InterpolationTableDialog
from hallbench.gui.polynomialtabledialog import PolynomialTableDialog \
    as _PolynomialTableDialog
from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.data.calibration import ProbeCalibration as _ProbeCalibration


class LoadCalibrationWidget(_QWidget):
    """Load Calibration widget class for the Hall Bench Control application."""

    _axis_str_dict = {
        1: '+ Axis #1 (+Z)', 2: '+ Axis #2 (+Y)', 3: '+ Axis #3 (+X)'}

    def __init__(self, parent=None):
        """Setup the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(__file__, self)
        self.ui = _uic.loadUi(uifile, self)

        self.interpolation_dialog = _InterpolationTableDialog()
        self.polynomial_dialog = _PolynomialTableDialog()

        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.configureGraph()
        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.viewdata_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        # create connections
        self.ui.loadfile_btn.clicked.connect(self.loadFile)
        self.ui.showtable_btn.clicked.connect(self.showTable)
        self.ui.updategraph_btn.clicked.connect(self.updateGraph)
        self.ui.voltage_sb.valueChanged.connect(self.updateField)

    @property
    def probe_calibration(self):
        """Calibration data object."""
        return self.window().probe_calibration

    @probe_calibration.setter
    def probe_calibration(self, value):
        self.window().probe_calibration = value
        self.interpolation_dialog.close()
        self.polynomial_dialog.close()

    def closeDialogs(self):
        """Close dialogs."""
        self.interpolation_dialog.close()
        self.polynomial_dialog.close()

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

    def loadFile(self):
        """Load probe calibration file."""
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
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)
            self.probe_calibration = None
            self.setEnabled(False)
            self.updateGraph()
            return

        self.setEnabled(True)
        self.updateGraph()

        self.ui.functiontype_le.setText(
            self.probe_calibration.function_type.capitalize())

        probe_axis = self.probe_calibration.probe_axis
        if probe_axis in self._axis_str_dict.keys():
            self.ui.probeaxis_le.setText(self._axis_str_dict[probe_axis])
            self.ui.probeaxis_le.setEnabled(True)
        else:
            self.ui.probeaxis_le.setText('')
            self.ui.probeaxis_le.setEnabled(False)

        distance_xy = self.probe_calibration.distance_xy
        if distance_xy is not None:
            self.ui.distancexy_le.setText('{0:0.4f}'.format(distance_xy))
            self.ui.distancexy_le.setEnabled(True)
        else:
            self.ui.distancexy_le.setText('')
            self.ui.distancexy_le.setEnabled(False)

        distance_zy = self.probe_calibration.distance_zy
        if distance_zy is not None:
            self.ui.distancezy_le.setText('{0:0.4f}'.format(distance_zy))
            self.ui.distancezy_le.setEnabled(True)
        else:
            self.ui.distancezy_le.setText('')
            self.ui.distancezy_le.setEnabled(False)

        self.ui.fieldx_le.setText('')
        self.ui.fieldy_le.setText('')
        self.ui.fieldz_le.setText('')

    def setEnabled(self, enabled):
        """Enable or disable controls."""
        self.ui.calibrationdata_gb.setEnabled(enabled)
        self.ui.showtable_btn.setEnabled(enabled)
        self.ui.plotoptions_gb.setEnabled(enabled)
        self.ui.updategraph_btn.setEnabled(enabled)
        self.ui.getfield_gb.setEnabled(enabled)

    def showTable(self):
        """Show calibration data table."""
        if self.probe_calibration is None:
            return

        if self.probe_calibration.function_type == 'interpolation':
            self.polynomial_dialog.close()
            self.interpolation_dialog.show(self.probe_calibration)
        elif self.probe_calibration.function_type == 'polynomial':
            self.interpolation_dialog.close()
            self.polynomial_dialog.show(self.probe_calibration)
        else:
            return

    def updateField(self):
        """Convert voltage to magnetic field."""
        self.ui.fieldx_le.setText('')
        self.ui.fieldy_le.setText('')
        self.ui.fieldz_le.setText('')

        if self.probe_calibration is None:
            return

        voltage = [self.ui.voltage_sb.value()]
        fieldx = self.probe_calibration.sensorx.convert_voltage(voltage)[0]
        fieldy = self.probe_calibration.sensory.convert_voltage(voltage)[0]
        fieldz = self.probe_calibration.sensorz.convert_voltage(voltage)[0]

        if not _np.isnan(fieldx):
            self.ui.fieldx_le.setText('{0:0.4f}'.format(fieldx))

        if not _np.isnan(fieldy):
            self.ui.fieldy_le.setText('{0:0.4f}'.format(fieldy))

        if not _np.isnan(fieldz):
            self.ui.fieldz_le.setText('{0:0.4f}'.format(fieldz))

    def updateGraph(self):
        """Update calibration data plots."""
        vmin = self.ui.voltagemin_sb.value()
        vmax = self.ui.voltagemax_sb.value()
        npts = self.ui.voltagenpts_sb.value()
        voltage = _np.linspace(vmin, vmax, npts)

        self.ui.viewdata_pw.clear()
        self.legend.removeItem('X')
        self.legend.removeItem('Y')
        self.legend.removeItem('Z')

        if self.probe_calibration is None or len(voltage) == 0:
            return

        fieldx = self.probe_calibration.sensorx.convert_voltage(voltage)
        fieldy = self.probe_calibration.sensory.convert_voltage(voltage)
        fieldz = self.probe_calibration.sensorz.convert_voltage(voltage)

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
