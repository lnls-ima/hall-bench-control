# -*- coding: utf-8 -*-

"""View probe dialog for the Hall Bench Control application."""

import numpy as _np
import json as _json
import warnings as _warnings
from PyQt5.QtWidgets import (
    QDialog as _QDialog,
    QApplication as _QApplication,
    QTableWidgetItem as _QTableWidgetItem,
    QMessageBox as _QMessageBox,
    )
import PyQt5.uic as _uic
import pyqtgraph as _pyqtgraph

from hallbench.gui.utils import getUiFile as _getUiFile


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

        self.interpolation_dialog = InterpolationTableDialog()
        self.polynomial_dialog = PolynomialTableDialog()

        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.configureGraph()
        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.plot_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        # create connections
        self.ui.showtable_btn.clicked.connect(self.showTable)
        self.ui.updategraph_btn.clicked.connect(self.updateGraph)
        self.ui.voltage_sb.valueChanged.connect(self.updateField)

    @property
    def database(self):
        """Database filename."""
        return _QApplication.instance().database

    def accept(self):
        """Close dialog."""
        self.closeDialogs()
        super().accept()

    def closeDialogs(self):
        """Close dialogs."""
        try:
            self.interpolation_dialog.accept()
            self.polynomial_dialog.accept()
        except Exception:
            pass

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.closeDialogs()
            event.accept()
        except Exception:
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
            self.setDataEnabled(False)
            self.updateGraph()
            message = 'Failed to load hall probe data.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
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
        except Exception as e:
            _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)
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
            message = 'Failed to update plot.'
            _QMessageBox.critical(self, 'Failure', message, _QMessageBox.Ok)
            return


class InterpolationTableDialog(_QDialog):
    """Interpolation table class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        self.local_hall_probe = None
        self.clip = _QApplication.clipboard()

        # create connections
        self.ui.copysensorx_btn.clicked.connect(
            lambda: self.copyToClipboard('x'))
        self.ui.copysensory_btn.clicked.connect(
            lambda: self.copyToClipboard('y'))
        self.ui.copysensorz_btn.clicked.connect(
            lambda: self.copyToClipboard('z'))

        self.ui.sensorxprec_sb.valueChanged.connect(self.updateTablesensorX)
        self.ui.sensoryprec_sb.valueChanged.connect(self.updateTablesensorY)
        self.ui.sensorzprec_sb.valueChanged.connect(self.updateTablesensorZ)

    def _updateTable(self, table, data, precision):
        table.setRowCount(0)
        formatstr = '{0:0.%if}' % precision
        for i in range(len(data)):
            table.setRowCount(i+1)
            row = data[i]
            for j in range(len(row)):
                table.setItem(i, j, _QTableWidgetItem(
                    formatstr.format(row[j])))

    def copyToClipboard(self, sensor):
        """Copy table data to clipboard."""
        table = getattr(self.ui, 'sensor' + sensor + '_ta')
        text = ""
        for r in range(table.rowCount()):
            for c in range(table.columnCount()):
                text += str(table.item(r, c).text()) + "\t"
            text = text[:-1] + "\n"
        self.clip.setText(text)

    def show(self, hall_probe):
        """Update hall probe object and show dialog."""
        self.local_hall_probe = hall_probe
        self.updateTables()
        super(InterpolationTableDialog, self).show()

    def updateTables(self):
        """Update table values."""
        if self.local_hall_probe is None:
            return
        self.updateTablesensorX()
        self.updateTablesensorY()
        self.updateTablesensorZ()

    def updateTablesensorX(self):
        """Update sensor x table values."""
        precision = self.sensorxprec_sb.value()
        table = self.ui.sensorx_ta

        if self.local_hall_probe.sensorx is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensorx.data
        self._updateTable(table, data, precision)

    def updateTablesensorY(self):
        """Update sensor y table values."""
        precision = self.sensoryprec_sb.value()
        table = self.ui.sensory_ta

        if self.local_hall_probe.sensory is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensory.data
        self._updateTable(table, data, precision)

    def updateTablesensorZ(self):
        """Update sensor z table values."""
        precision = self.sensorzprec_sb.value()
        table = self.ui.sensorz_ta

        if self.local_hall_probe.sensorz is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensorz.data
        self._updateTable(table, data, precision)


class PolynomialTableDialog(_QDialog):
    """Polynomial table class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        self.clip = _QApplication.clipboard()
        self.local_hall_probe = None

        # create connections
        self.ui.copysensorx_btn.clicked.connect(
            lambda: self.copyToClipboard('x'))
        self.ui.copysensory_btn.clicked.connect(
            lambda: self.copyToClipboard('y'))
        self.ui.copysensorz_btn.clicked.connect(
            lambda: self.copyToClipboard('z'))

        self.ui.sensorxprec_sb.valueChanged.connect(self.updateTableSensorX)
        self.ui.sensoryprec_sb.valueChanged.connect(self.updateTableSensorY)
        self.ui.sensorzprec_sb.valueChanged.connect(self.updateTableSensorZ)

    def _updateTable(self, table, data, precision):
        table.setRowCount(0)

        if len(data) == 0:
            return

        nc = len(data[0])
        table.setColumnCount(nc)
        labels = ['Initial Voltage [V]', 'Final Voltage [V]']
        for j in range(nc-2):
            labels.append('C' + str(j))
        table.setHorizontalHeaderLabels(labels)

        vformatstr = '{0:0.%if}' % precision
        cformatstr = '{0:0.%ie}' % precision
        for i in range(len(data)):
            table.setRowCount(i+1)
            row = data[i]
            for j in range(len(row)):
                if j < 2:
                    table.setItem(i, j, _QTableWidgetItem(
                        vformatstr.format(row[j])))
                else:
                    table.setItem(i, j, _QTableWidgetItem(
                        cformatstr.format(row[j])))

    def copyToClipboard(self, sensor):
        """Copy table data to clipboard."""
        table = getattr(self.ui, 'sensor' + sensor + '_ta')
        text = ""
        for r in range(table.rowCount()):
            for c in range(table.columnCount()):
                text += str(table.item(r, c).text()) + "\t"
            text = text[:-1] + "\n"
        self.clip.setText(text)

    def show(self, hall_probe=None):
        """Update hall probe object and show dialog."""
        self.local_hall_probe = hall_probe
        self.updateTables()
        super().show()

    def updateTables(self):
        """Update table values."""
        if self.local_hall_probe is None:
            return
        self.updateTableSensorX()
        self.updateTableSensorY()
        self.updateTableSensorZ()

    def updateTableSensorX(self):
        """Update sensor x table values."""
        precision = self.sensorxprec_sb.value()
        table = self.ui.sensorx_ta
        if self.local_hall_probe.sensorx is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensorx.data
        self._updateTable(table, data, precision)

    def updateTableSensorY(self):
        """Update sensor y table values."""
        precision = self.sensoryprec_sb.value()
        table = self.ui.sensory_ta
        if self.local_hall_probe.sensory is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensory.data
        self._updateTable(table, data, precision)

    def updateTableSensorZ(self):
        """Update sensor z table values."""
        precision = self.sensorzprec_sb.value()
        table = self.ui.sensorz_ta
        if self.local_hall_probe.sensorz is None:
            table.setRowCount(0)
            return

        data = self.local_hall_probe.sensorz.data
        self._updateTable(table, data, precision)
