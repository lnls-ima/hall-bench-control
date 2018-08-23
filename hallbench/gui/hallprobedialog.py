# -*- coding: utf-8 -*-

"""Hall probe dialog for the Hall Bench Control application."""

import numpy as _np
import warnings as _warnings
from PyQt4.QtGui import (
    QDialog as _QDialog,
    QFileDialog as _QFileDialog,
    QApplication as _QApplication,
    QTableWidgetItem as _QTableWidgetItem,
    QMessageBox as _QMessageBox,
    )
from PyQt4.QtCore import pyqtSignal as _pyqtSignal
import PyQt4.uic as _uic
import pyqtgraph as _pyqtgraph

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.data.calibration import HallProbe as _HallProbe


class HallProbeDialog(_QDialog):
    """Hall probe dialog class for Hall Bench Control application."""

    hallProbeChanged = _pyqtSignal(_HallProbe)

    _axis_str_dict = {
        1: '+ Axis #1 (+Z)', 2: '+ Axis #2 (+Y)', 3: '+ Axis #3 (+X)'}

    def __init__(self, parent=None, load_enabled=True):
        """Set up the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        self.interpolation_dialog = InterpolationTableDialog()
        self.polynomial_dialog = PolynomialTableDialog()

        self._hall_probe = None
        self._load_enabled = load_enabled
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

        # Enable or disable load option
        self.ui.loadfile_btn.setEnabled(self._load_enabled)
        self.ui.loaddb_btn.setEnabled(self._load_enabled)
        self.ui.filename_le.setEnabled(self._load_enabled)
        self.ui.idn_le.setReadOnly(not self._load_enabled)

    @property
    def hall_probe(self):
        """Hall probe object."""
        return self._hall_probe

    @hall_probe.setter
    def hall_probe(self, value):
        self._hall_probe = value
        self.hallProbeChanged.emit(self.hall_probe)
        self.closeDialogs()

    def closeDialogs(self):
        """Close table dialogs."""
        try:
            self.interpolation_dialog.accept()
            self.polynomial_dialog.accept()
        except Exception:
            pass

    def accept(self):
        """Close dialog."""
        self.closeDialogs()
        super().accept()

    def configureGraph(self, symbol=False):
        """Configure data plots."""
        self.ui.viewdata_pw.clear()

        self.graphx = []
        self.graphy = []
        self.graphz = []

        penx = (255, 0, 0)
        peny = (0, 255, 0)
        penz = (0, 0, 255)

        if symbol:
            plot_item_x = self.ui.viewdata_pw.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=penx,
                symbol='o',
                symbolPen=penx,
                symbolSize=4,
                symbolBrush=penx)

            plot_item_y = self.ui.viewdata_pw.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=peny,
                symbol='o',
                symbolPen=peny,
                symbolSize=4,
                symbolBrush=peny)

            plot_item_z = self.ui.viewdata_pw.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=penz,
                symbol='o',
                symbolPen=penz,
                symbolSize=4,
                symbolBrush=penz)
        else:
            plot_item_x = self.ui.viewdata_pw.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=penx)

            plot_item_y = self.ui.viewdata_pw.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=peny)

            plot_item_z = self.ui.viewdata_pw.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=penz)

        self.graphx.append(plot_item_x)
        self.graphy.append(plot_item_y)
        self.graphz.append(plot_item_z)

        self.ui.viewdata_pw.setLabel('bottom', 'Voltage [V]')
        self.ui.viewdata_pw.setLabel('left', 'Magnetic Field [T]')
        self.ui.viewdata_pw.showGrid(x=True, y=True)

    def load(self):
        """Load hall probe parameters."""
        try:
            self.ui.probe_name_le.setText(self._hall_probe.probe_name)
            self.ui.sensorx_name_le.setText(
                self._hall_probe.sensorx_name)
            self.ui.sensory_name_le.setText(
                self._hall_probe.sensory_name)
            self.ui.sensorz_name_le.setText(
                self._hall_probe.sensorz_name)

            probe_axis = self._hall_probe.probe_axis
            if probe_axis in self._axis_str_dict.keys():
                self.ui.probe_axis_le.setText(self._axis_str_dict[probe_axis])
                self.ui.probe_axis_le.setEnabled(True)
            else:
                self.ui.probe_axis_le.setText('')
                self.ui.probe_axis_le.setEnabled(False)

            distance_xy = self._hall_probe.distance_xy
            if distance_xy is not None:
                self.ui.distance_xy_le.setText('{0:0.4f}'.format(distance_xy))
                self.ui.distance_xy_le.setEnabled(True)
            else:
                self.ui.distance_xy_le.setText('')
                self.ui.distance_xy_le.setEnabled(False)

            distance_zy = self._hall_probe.distance_zy
            if distance_zy is not None:
                self.ui.distance_zy_le.setText('{0:0.4f}'.format(distance_zy))
                self.ui.distance_zy_le.setEnabled(True)
            else:
                self.ui.distance_zy_le.setText('')
                self.ui.distance_zy_le.setEnabled(False)

            angle_xy = self._hall_probe.angle_xy
            if angle_xy is not None:
                self.ui.angle_xy_le.setText('{0:0.4f}'.format(angle_xy))
                self.ui.angle_xy_le.setEnabled(True)
            else:
                self.ui.angle_xy_le.setText('')
                self.ui.angle_xy_le.setEnabled(False)

            angle_yz = self._hall_probe.angle_yz
            if angle_yz is not None:
                self.ui.angle_yz_le.setText('{0:0.4f}'.format(angle_yz))
                self.ui.angle_yz_le.setEnabled(True)
            else:
                self.ui.angle_yz_le.setText('')
                self.ui.angle_yz_le.setEnabled(False)

            angle_xz = self._hall_probe.angle_xz
            if angle_xz is not None:
                self.ui.angle_xz_le.setText('{0:0.4f}'.format(angle_xz))
                self.ui.angle_xz_le.setEnabled(True)
            else:
                self.ui.angle_xz_le.setText('')
                self.ui.angle_xz_le.setEnabled(False)

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

    def loadDB(self):
        """Load hall probe from database."""
        self.ui.filename_le.setText('')
        self.setDataEnabled(False)
        self.updateGraph()

        try:
            idn = int(self.ui.idn_le.text())
        except Exception:
            _QMessageBox.critical(
                self, 'Failure', 'Invalid database ID.', _QMessageBox.Ok)
            return

        try:
            self.hall_probe = _HallProbe(
                database=self.database, idn=idn)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)
            return

        self.load()

    def loadFile(self):
        """Load hall probe file."""
        self.setDatabaseID('')
        self.setDataEnabled(False)
        self.updateGraph()

        default_filename = self.ui.filename_le.text()
        filename = _QFileDialog.getOpenFileName(
            self, caption='Load hall probe file',
            directory=default_filename, filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            self.hall_probe = _HallProbe(filename)
        except Exception as e:
            _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)
            return

        self.filename_le.setText(filename)
        self.load()

    def searchDB(self, probe_name):
        """Search hall probe in database."""
        if (self.hall_probe is not None and
           self.hall_probe.probe_name == probe_name):
            return

        if probe_name is None or len(probe_name) == 0:
            return

        if self.database is None or len(self.database) == 0:
            return

        try:
            idn = _HallProbe.get_hall_probe_id(
                self.database, probe_name)
            if idn is not None:
                self.setDatabaseID(idn)
                self.loadDB()
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def setDatabaseID(self, idn):
        """Set database id text."""
        self.ui.filename_le.setText('')
        self.ui.idn_le.setText(str(idn))
        self.ui.idn_le.setEnabled(True)

    def setDataEnabled(self, enabled):
        """Enable or disable controls."""
        self.ui.probedata_gb.setEnabled(enabled)
        self.ui.showtable_btn.setEnabled(enabled)
        self.ui.plotoptions_gb.setEnabled(enabled)
        self.ui.updategraph_btn.setEnabled(enabled)
        self.ui.getfield_gb.setEnabled(enabled)

    def show(self, database):
        """Update database and show dialog."""
        self.database = database

        if self.database is not None:
            self.ui.idn_le.setEnabled(True)
            if self._load_enabled:
                self.ui.loaddb_btn.setEnabled(True)
            else:
                self.ui.loaddb_btn.setEnabled(False)
        else:
            self.ui.loaddb_btn.setEnabled(False)
            self.ui.idn_le.setEnabled(False)

        super().show()

    def showTable(self):
        """Show probe data table."""
        if self._hall_probe is None:
            return

        if self._hall_probe.sensorx is not None:
            function_type_x = self._hall_probe.sensorx.function_type
        else:
            function_type_x = None

        if self._hall_probe.sensory is not None:
            function_type_y = self._hall_probe.sensory.function_type
        else:
            function_type_y = None

        if self._hall_probe.sensorz is not None:
            function_type_z = self._hall_probe.sensorz.function_type
        else:
            function_type_z = None

        func_type = [function_type_x, function_type_y, function_type_z]

        if all([f == 'interpolation' for f in func_type if f is not None]):
            self.polynomial_dialog.accept()
            self.interpolation_dialog.show(self._hall_probe)
        elif all([f == 'polynomial' for f in func_type if f is not None]):
            self.interpolation_dialog.accept()
            self.polynomial_dialog.show(self._hall_probe)
        else:
            return

    def updateField(self):
        """Convert voltage to magnetic field."""
        self.ui.fieldx_le.setText('')
        self.ui.fieldy_le.setText('')
        self.ui.fieldz_le.setText('')

        if self._hall_probe is None:
            return

        try:
            volt = [self.ui.voltage_sb.value()]
            fieldx = self._hall_probe.sensorx.convert_voltage(volt)[0]
            fieldy = self._hall_probe.sensory.convert_voltage(volt)[0]
            fieldz = self._hall_probe.sensorz.convert_voltage(volt)[0]

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

            self.ui.viewdata_pw.clear()
            self.legend.removeItem('X')
            self.legend.removeItem('Y')
            self.legend.removeItem('Z')

            if self._hall_probe is None or len(voltage) == 0:
                return

            fieldx = self._hall_probe.sensorx.convert_voltage(voltage)
            fieldy = self._hall_probe.sensory.convert_voltage(voltage)
            fieldz = self._hall_probe.sensorz.convert_voltage(voltage)

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

        self.hall_probe = None
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
        self.hall_probe = hall_probe
        self.updateTables()
        super(InterpolationTableDialog, self).show()

    def updateTables(self):
        """Update table values."""
        if self.hall_probe is None:
            return
        self.updateTablesensorX()
        self.updateTablesensorY()
        self.updateTablesensorZ()

    def updateTablesensorX(self):
        """Update sensor x table values."""
        precision = self.sensorxprec_sb.value()
        table = self.ui.sensorx_ta
        data = self.hall_probe.sensorx.data

        formatstr = '{0:0.%if}' % precision
        table.setRowCount(0)
        for i in range(len(data)):
            table.setRowCount(i+1)
            row = data[i]
            for j in range(len(row)):
                table.setItem(i, j, _QTableWidgetItem(
                    formatstr.format(row[j])))

    def updateTablesensorY(self):
        """Update sensor y table values."""
        precision = self.sensoryprec_sb.value()
        table = self.ui.sensory_ta
        data = self.hall_probe.sensory.data

        formatstr = '{0:0.%if}' % precision
        table.setRowCount(0)
        for i in range(len(data)):
            table.setRowCount(i+1)
            row = data[i]
            for j in range(len(row)):
                table.setItem(i, j, _QTableWidgetItem(
                    formatstr.format(row[j])))

    def updateTablesensorZ(self):
        """Update sensor z table values."""
        precision = self.sensorzprec_sb.value()
        table = self.ui.sensorz_ta
        data = self.hall_probe.sensorz.data

        formatstr = '{0:0.%if}' % precision
        table.setRowCount(0)
        for i in range(len(data)):
            table.setRowCount(i+1)
            row = data[i]
            for j in range(len(row)):
                table.setItem(i, j, _QTableWidgetItem(
                    formatstr.format(row[j])))


class PolynomialTableDialog(_QDialog):
    """Polynomial table class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        self.hall_probe = None
        self.clip = _QApplication.clipboard()

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
        self.hall_probe = hall_probe
        self.updateTables()
        super(PolynomialTableDialog, self).show()

    def updateTables(self):
        """Update table values."""
        if self.hall_probe is None:
            return
        self.updateTableSensorX()
        self.updateTableSensorY()
        self.updateTableSensorZ()

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

    def updateTableSensorX(self):
        """Update sensor x table values."""
        precision = self.sensorxprec_sb.value()
        table = self.ui.sensorx_ta
        data = self.hall_probe.sensorx.data
        self._updateTable(table, data, precision)

    def updateTableSensorY(self):
        """Update sensor y table values."""
        precision = self.sensoryprec_sb.value()
        table = self.ui.sensory_ta
        data = self.hall_probe.sensory.data
        self._updateTable(table, data, precision)

    def updateTableSensorZ(self):
        """Update sensor z table values."""
        precision = self.sensorzprec_sb.value()
        table = self.ui.sensorz_ta
        data = self.hall_probe.sensorz.data
        self._updateTable(table, data, precision)
