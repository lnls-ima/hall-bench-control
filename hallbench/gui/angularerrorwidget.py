# -*- coding: utf-8 -*-

"""Angular error widget for the Hall Bench Control application."""

import numpy as _np
import pandas as _pd
import time as _time
import datetime as _datetime
import warnings as _warnings
import pyqtgraph as _pyqtgraph
from PyQt4.QtGui import (
    QWidget as _QWidget,
    QMessageBox as _QMessageBox,
    QVBoxLayout as _QVBoxLayout,
    QTableWidgetItem as _QTableWidgetItem,
    )
from PyQt4.QtCore import QTimer as _QTimer
import PyQt4.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.moveaxiswidget import MoveAxisWidget as _MoveAxisWidget


class AngularErrorWidget(_QWidget):
    """Angular error widget class for the Hall Bench Control application."""

    _data_format = '{0:.4f}'
    _data_labels = ['X-axis [arcsec]', 'Y-axis [arcsec]']
    _yaxis_label = 'Angular error [arcsec]'
    _colors = {
        'X-axis [arcsec]': (230, 25, 75),
        'Y-axis [arcsec]': (60, 180, 75),
    }

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # add move axis widget
        self.move_axis_widget = _MoveAxisWidget(self)
        _layout = _QVBoxLayout()
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.addWidget(self.move_axis_widget)
        self.ui.moveaxis_wg.setLayout(_layout)

        # variables initialization
        self.timestamp = []
        self._legend_items = []
        self._readings = {}
        self._graphs = {}
        for label in self._data_labels:
            self._readings[label] = []
            self._graphs[label] = None

        # create timer to monitor temperature
        self.timer = _QTimer(self)
        self.updateMonitorInterval()
        self.timer.timeout.connect(lambda: self.readValue(monitor=True))

        col_labels = ['Date', 'Time']
        for label in self._data_labels:
            col_labels.append(label)
        self.ui.table_ta.setColumnCount(len(col_labels))
        self.ui.table_ta.setHorizontalHeaderLabels(col_labels)
        self.ui.table_ta.setAlternatingRowColors(True)

        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.plot_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        self.configureGraph()
        self.connectSignalSlots()

    @property
    def devices(self):
        """Hall Bench devices."""
        return self.window().devices

    def updateLegendItems(self):
        """Update legend items."""      
        self.clearLegendItems()
        self._legend_items = []
        for label in self._data_labels:
            legend_label = label.split('[')[0]
            self._legend_items.append(legend_label)
            self.legend.addItem(self._graphs[label], legend_label)

    def clearLegendItems(self):
        """Clear plot legend."""
        for label in self._legend_items:
            self.legend.removeItem(label)

    def clearValues(self):
        """Clear all values."""
        self.timestamp = []
        for label in self._data_labels:
            self._readings[label] = []
        self.updateTableValues()
        self.updatePlot()

    def configureGraph(self):
        """Configure data plots."""
        self.ui.plot_pw.clear()

        for label in self._data_labels:
            pen = self._colors[label]
            graph = self.ui.plot_pw.plotItem.plot(
                _np.array([]), _np.array([]), pen=pen,
                symbol='o', symbolPen=pen, symbolSize=3, symbolBrush=pen)
            self._graphs[label] = graph

        self.ui.plot_pw.setLabel('bottom', 'Time interval [s]')
        self.ui.plot_pw.setLabel('left', self._yaxis_label)
        self.ui.plot_pw.showGrid(x=True, y=True)
        self.updateLegendItems()

    def connectSignalSlots(self):
        """Create signal/slot connections."""       
        self.ui.read_btn.clicked.connect(lambda: self.readValue(monitor=False))
        self.ui.monitor_btn.toggled.connect(self.monitorValue)
        self.ui.monitorstep_sb.valueChanged.connect(self.updateMonitorInterval)
        self.ui.monitorunit_cmb.currentIndexChanged.connect(
            self.updateMonitorInterval)
        self.ui.clear_btn.clicked.connect(self.clearValues)
        self.ui.remove_btn.clicked.connect(self.removeValue)
        self.ui.copy_btn.clicked.connect(self.copyToClipboard)

    def copyToClipboard(self):
        """Copy table data to clipboard."""
        nr = self.ui.table_ta.rowCount()
        nc = self.ui.table_ta.columnCount()

        if nr == 0:
            return

        col_labels = ['Date', 'Time']
        for label in self._data_labels:
            col_labels.append(label)
        tdata = []
        for i in range(nr):
            ldata = []
            for j in range(nc):
                value = self.ui.table_ta.item(i, j).text()
                if j >= 2:
                    value = float(value)
                ldata.append(value)
            tdata.append(ldata)
        _df = _pd.DataFrame(_np.array(tdata), columns=col_labels)
        _df.to_clipboard(excel=True)

    def monitorValue(self, checked):
        """Monitor values."""
        if checked:
            self.ui.read_btn.setEnabled(False)
            self.timer.start()
        else:
            self.timer.stop()
            self.ui.read_btn.setEnabled(True)

    def readValue(self, monitor=False):
        """Read  value."""
        if len(self._data_labels) == 0:
            return

        if not self.devices.elcomat.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Auto-collimator not connected.', _QMessageBox.Ok)
            return

        ts = _time.time()
        if self.ui.meastype_cmb.currentText().lower() == 'relative':
            _rl = self.devices.elcomat.get_relative_measurement()
        else:
            _rl = self.devices.elcomat.get_absolute_measurement()
        if len(_rl) != len(self._data_labels):
            return
        
        readings = [r if r is not None else _np.nan for r in _rl]

        try:
            for i in range(len(self._data_labels)):
                label = self._data_labels[i]
                value = readings[i]
                self._readings[label].append(value)

            self.timestamp.append(ts)
            self.updateTableValues()
            self.updatePlot()

        except Exception:
            pass

    def removeValue(self):
        """Remove value from list."""
        selected = self.ui.table_ta.selectedItems()
        rows = [s.row() for s in selected]
        n = len(self.timestamp)

        self.timestamp = [self.timestamp[i] for i in range(n) if i not in rows]
        for label in self._data_labels:
            readings = self._readings[label]
            self._readings[label] = [
                readings[i] for i in range(n) if i not in rows]

        self.updateTableValues(scrollDown=False)
        self.updatePlot()

    def updateMonitorInterval(self):
        """Update monitor interval value."""
        index = self.ui.monitorunit_cmb.currentIndex()
        if index == 0:
            mf = 1000
        elif index == 1:
            mf = 1000*60
        else:
            mf = 1000*3600
        self.timer.setInterval(self.ui.monitorstep_sb.value()*mf)

    def updatePlot(self):
        """Update plot values."""
        if len(self.timestamp) == 0:
            for label in self._data_labels:
                self._graphs[label].setData(
                    _np.array([]), _np.array([]))
            return

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            timeinterval = _np.array(self.timestamp) - self.timestamp[0]
            for label in self._data_labels:
                readings = _np.array(self._readings[label])
                dt = timeinterval[_np.isfinite(readings)]
                rd = readings[_np.isfinite(readings)]
                self._graphs[label].setData(dt, rd)

    def updatePositions(self):
        """Update axes positions."""
        self.move_axis_widget.updatePositions()

    def updateTableValues(self, scrollDown=True):
        """Update table values."""
        n = len(self.timestamp)
        self.ui.table_ta.clearContents()
        self.ui.table_ta.setRowCount(n)

        for j in range(len(self._data_labels)):
            readings = self._readings[self._data_labels[j]]
            for i in range(n):
                dt = _datetime.datetime.fromtimestamp(self.timestamp[i])
                date = dt.strftime("%d/%m/%Y")
                hour = dt.strftime("%H:%M:%S")
                self.ui.table_ta.setItem(i, 0, _QTableWidgetItem(date))
                self.ui.table_ta.setItem(i, 1, _QTableWidgetItem(hour))
                self.ui.table_ta.setItem(
                    i, j+2, _QTableWidgetItem(self._data_format.format(
                        readings[i])))

        if scrollDown:
            vbar = self.table_ta.verticalScrollBar()
            vbar.setValue(vbar.maximum())
