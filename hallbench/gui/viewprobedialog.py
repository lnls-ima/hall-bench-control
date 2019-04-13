# -*- coding: utf-8 -*-

"""View probe dialog for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import json as _json
import warnings as _warnings
import traceback as _traceback
import pyqtgraph as _pyqtgraph
from qtpy.QtWidgets import (
    QDialog as _QDialog,
    QApplication as _QApplication,
    )
import qtpy.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.auxiliarywidgets import (
    PolynomialTableDialog as _PolynomialTableDialog,
    InterpolationTableDialog as _InterpolationTableDialog,
    )


class ViewProbeDialog(_QDialog):
    """View probe dialog class for Hall Bench Control application."""

    _axis_str_dict = {
        1: '+ Axis #1 (+Z)', 2: '+ Axis #2 (+Y)', 3: '+ Axis #3 (+X)'}

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        self.local_hall_probe = None

        self.polynomial_dialog = _PolynomialTableDialog()
        self.interpolation_dialog = _InterpolationTableDialog()

        self.graphx = []
        self.graphy = []
        self.graphz = []

        self.configureGraph()
        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.plot_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        self.connectSignalSlots()

    @property
    def database(self):
        """Database filename."""
        return _QApplication.instance().database

    def accept(self):
        """Close dialog."""
        self.clear()
        self.closeDialogs()
        super().accept()

    def clear(self):
        """Clear."""
        self.local_hall_probe = None
        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.ui.plot_pw.clear()
        self.interpolation_dialog.clear()
        self.polynomial_dialog.clear()

    def closeDialogs(self):
        """Close dialogs."""
        try:
            self.interpolation_dialog.accept()
            self.polynomial_dialog.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.clear()
            self.closeDialogs()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def configureGraph(self, symbol=False):
        """Configure data plots."""
        self.ui.plot_pw.clear()

        self.graphx = []
        self.graphy = []
        self.graphz = []

        penx = (255, 0, 0)
        peny = (0, 255, 0)
        penz = (0, 0, 255)

        if symbol:
            plot_item_x = self.ui.plot_pw.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=penx,
                symbol='o',
                symbolPen=penx,
                symbolSize=4,
                symbolBrush=penx)

            plot_item_y = self.ui.plot_pw.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=peny,
                symbol='o',
                symbolPen=peny,
                symbolSize=4,
                symbolBrush=peny)

            plot_item_z = self.ui.plot_pw.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=penz,
                symbol='o',
                symbolPen=penz,
                symbolSize=4,
                symbolBrush=penz)
        else:
            plot_item_x = self.ui.plot_pw.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=penx)

            plot_item_y = self.ui.plot_pw.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=peny)

            plot_item_z = self.ui.plot_pw.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=penz)

        self.graphx.append(plot_item_x)
        self.graphy.append(plot_item_y)
        self.graphz.append(plot_item_z)

        self.ui.plot_pw.setLabel('bottom', 'Voltage [V]')
        self.ui.plot_pw.setLabel('left', 'Magnetic Field [T]')
        self.ui.plot_pw.showGrid(x=True, y=True)

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.showtable_btn.clicked.connect(self.showTable)
        self.ui.updategraph_btn.clicked.connect(self.updateGraph)
        self.ui.voltage_sb.valueChanged.connect(self.updateField)

    def load(self):
        """Load hall probe parameters."""
        try:
            self.ui.probe_name_le.setText(self.local_hall_probe.probe_name)
            self.ui.rod_shape_le.setText(self.local_hall_probe.rod_shape)

            if self.local_hall_probe.sensorx_name is not None:
                sensorx_name = self.local_hall_probe.sensorx_name
                sensorx_position = _json.dumps(
                    self.local_hall_probe.sensorx_position.tolist())
                sensorx_direction = _json.dumps(
                    self.local_hall_probe.sensorx_direction.tolist())
            else:
                sensorx_name = ''
                sensorx_position = ''
                sensorx_direction = ''

            if self.local_hall_probe.sensory_name is not None:
                sensory_name = self.local_hall_probe.sensory_name
                sensory_position = _json.dumps(
                    self.local_hall_probe.sensory_position.tolist())
                sensory_direction = _json.dumps(
                    self.local_hall_probe.sensory_direction.tolist())
            else:
                sensory_name = ''
                sensory_position = ''
                sensory_direction = ''

            if self.local_hall_probe.sensorz_name is not None:
                sensorz_name = self.local_hall_probe.sensorz_name
                sensorz_position = _json.dumps(
                    self.local_hall_probe.sensorz_position.tolist())
                sensorz_direction = _json.dumps(
                    self.local_hall_probe.sensorz_direction.tolist())
            else:
                sensorz_name = ''
                sensorz_position = ''
                sensorz_direction = ''

            self.ui.sensorx_name_le.setText(sensorx_name)
            self.ui.sensorx_position_le.setText(sensorx_position)
            self.ui.sensorx_direction_le.setText(sensorx_direction)

            self.ui.sensory_name_le.setText(sensory_name)
            self.ui.sensory_position_le.setText(sensory_position)
            self.ui.sensory_direction_le.setText(sensory_direction)

            self.ui.sensorz_name_le.setText(sensorz_name)
            self.ui.sensorz_position_le.setText(sensorz_position)
            self.ui.sensorz_direction_le.setText(sensorz_direction)

            self.setDataEnabled(True)
            self.updateGraph()
            self.ui.fieldx_le.setText('')
            self.ui.fieldy_le.setText('')
            self.ui.fieldz_le.setText('')

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.setDataEnabled(False)
            self.updateGraph()
            msg = 'Failed to load hall probe data.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def setDataEnabled(self, enabled):
        """Enable or disable controls."""
        self.ui.probedata_gb.setEnabled(enabled)
        self.ui.showtable_btn.setEnabled(enabled)
        self.ui.plotoptions_gb.setEnabled(enabled)
        self.ui.updategraph_btn.setEnabled(enabled)
        self.ui.getfield_gb.setEnabled(enabled)

    def show(self, hall_probe):
        """Show dialog."""
        self.local_hall_probe = hall_probe
        self.load()
        super().show()

    def showTable(self):
        """Show probe data table."""
        if self.local_hall_probe is None:
            return

        if self.local_hall_probe.sensorx is not None:
            function_type_x = self.local_hall_probe.sensorx.function_type
        else:
            function_type_x = None

        if self.local_hall_probe.sensory is not None:
            function_type_y = self.local_hall_probe.sensory.function_type
        else:
            function_type_y = None

        if self.local_hall_probe.sensorz is not None:
            function_type_z = self.local_hall_probe.sensorz.function_type
        else:
            function_type_z = None

        func_type = [function_type_x, function_type_y, function_type_z]

        if all([f == 'interpolation' for f in func_type if f is not None]):
            self.polynomial_dialog.accept()
            self.interpolation_dialog.show(self.local_hall_probe)
        elif all([f == 'polynomial' for f in func_type if f is not None]):
            self.interpolation_dialog.accept()
            self.polynomial_dialog.show(self.local_hall_probe)
        else:
            return

    def updateField(self):
        """Convert voltage to magnetic field."""
        self.ui.fieldx_le.setText('')
        self.ui.fieldy_le.setText('')
        self.ui.fieldz_le.setText('')

        if self.local_hall_probe is None:
            return

        try:
            volt = [self.ui.voltage_sb.value()]

            if self.local_hall_probe.sensorx is not None:
                fieldx = self.local_hall_probe.sensorx.get_field(volt)[0]
            else:
                fieldx = _np.nan

            if self.local_hall_probe.sensory is not None:
                fieldy = self.local_hall_probe.sensory.get_field(volt)[0]
            else:
                fieldy = _np.nan

            if self.local_hall_probe.sensorz is not None:
                fieldz = self.local_hall_probe.sensorz.get_field(volt)[0]
            else:
                fieldz = _np.nan

            if not _np.isnan(fieldx):
                self.ui.fieldx_le.setText('{0:0.4f}'.format(fieldx))

            if not _np.isnan(fieldy):
                self.ui.fieldy_le.setText('{0:0.4f}'.format(fieldy))

            if not _np.isnan(fieldz):
                self.ui.fieldz_le.setText('{0:0.4f}'.format(fieldz))
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to update magnetic field values.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def updateGraph(self):
        """Update data plots."""
        try:
            vmin = self.ui.voltagemin_sb.value()
            vmax = self.ui.voltagemax_sb.value()
            npts = self.ui.voltagenpts_sb.value()
            voltage = _np.linspace(vmin, vmax, npts)
            empty_data = _np.ones(len(voltage))*_np.nan

            self.ui.plot_pw.clear()
            self.legend.removeItem('X')
            self.legend.removeItem('Y')
            self.legend.removeItem('Z')

            if self.local_hall_probe is None or len(voltage) == 0:
                return

            if self.local_hall_probe.sensorx is not None:
                fieldx = self.local_hall_probe.sensorx.get_field(voltage)
            else:
                fieldx = empty_data

            if self.local_hall_probe.sensory is not None:
                fieldy = self.local_hall_probe.sensory.get_field(voltage)
            else:
                fieldy = empty_data

            if self.local_hall_probe.sensorz is not None:
                fieldz = self.local_hall_probe.sensorz.get_field(voltage)
            else:
                fieldz = empty_data

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
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to update plot.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return
