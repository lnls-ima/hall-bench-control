# -*- coding: utf-8 -*-

"""Table and plot widget for the Hall Bench Control application."""

import numpy as _np
import datetime as _datetime
import warnings as _warnings
import pyqtgraph as _pyqtgraph
from PyQt5.QtWidgets import (
    QFileDialog as _QFileDialog,
    QMessageBox as _QMessageBox,
    QTableWidgetItem as _QTableWidgetItem,
    QWidget as _QWidget,
    )
from PyQt5.QtCore import QTimer as _QTimer
import PyQt5.uic as _uic

import hallbench.gui.utils as _utils


class TablePlotWidget(_QWidget):
    """Table and plot Widget class for the Hall Bench Control application."""

    _data_format = '{0:.4f}'
    _data_labels = []
    _yaxis_label = ''
    _colors = {}

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _utils.getUiFile(TablePlotWidget)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialization
        self.timestamp = []
        self._legend_items = []
        self._readings = {}
        self._graphs = {}
        for label in self._data_labels:
            self._readings[label] = []
            self._graphs[label] = None

        # create timer to monitor values
        self.timer = _QTimer(self)
        self.updateMonitorInterval()
        self.timer.timeout.connect(lambda: self.readValue(monitor=True))

        self.table_analysis_dialog = _utils.TableAnalysisDialog()

        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.plot_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        self.configureGraph()
        self.connectSignalSlots()

    def clearLegendItems(self):
        """Clear plot legend."""
        for label in self._legend_items:
            self.legend.removeItem(label)

    def clearValues(self):
        """Clear all values."""
        if hasattr(self, 'position'):
            self.position = []
        self.timestamp = []
        for label in self._data_labels:
            self._readings[label] = []
        self.updateTableValues()
        self.updatePlot()
        self.updateTableAnalysisDialog()

    def closeDialogs(self):
        """Close dialogs."""
        try:
            self.self.table_analysis_dialog.accept()
        except Exception:
            pass

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.timer.stop()
            self.closeDialogs()
            event.accept()
        except Exception:
            event.accept()

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
        self.ui.save_btn.clicked.connect(self.saveToFile)
        self.ui.analysis_btn.clicked.connect(self.showTableAnalysisDialog)

    def copyToClipboard(self):
        """Copy table data to clipboard."""
        df = _utils.tableToDataFrame(self.ui.table_ta)
        if df is not None:
            df.to_clipboard(excel=True)

    def monitorValue(self, checked):
        """Monitor values."""
        if checked:
            self.ui.read_btn.setEnabled(False)
            self.timer.start()
        else:
            self.timer.stop()
            self.ui.read_btn.setEnabled(True)

    def readValue(self, monitor=False):
        """Read value."""
        pass

    def removeValue(self):
        """Remove value from list."""
        selected = self.ui.table_ta.selectedItems()
        rows = [s.row() for s in selected]
        n = len(self.timestamp)

        if hasattr(self, 'position'):
            self.position = [
                self.position[i] for i in range(n) if i not in rows]

        self.timestamp = [
            self.timestamp[i] for i in range(n) if i not in rows]

        for label in self._data_labels:
            readings = self._readings[label]
            self._readings[label] = [
                readings[i] for i in range(n) if i not in rows]

        self.updateTableValues(scrollDown=False)
        self.updatePlot()
        self.updateTableAnalysisDialog()

    def saveToFile(self):
        """Save table values to file."""
        df = _utils.tableToDataFrame(self.ui.table_ta)
        if df is None:
            _QMessageBox.critical(
                self, 'Failure', 'Empty table.', _QMessageBox.Ok)
            return

        filename = _QFileDialog.getSaveFileName(
            self, caption='Save measurements file.',
            filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            if (not filename.endswith('.txt')
               and not filename.endswith('.dat')):
                filename = filename + '.txt'
            df.to_csv(filename, sep='\t')

        except Exception as e:
            _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)

    def showTableAnalysisDialog(self):
        """Show table analysis dialog."""
        df = _utils.tableToDataFrame(self.ui.table_ta)
        self.table_analysis_dialog.accept()
        self.table_analysis_dialog.show(df)

    def updateLegendItems(self):
        """Update legend items."""
        self.clearLegendItems()
        self._legend_items = []
        for label in self._data_labels:
            legend_label = label.split('[')[0]
            self._legend_items.append(legend_label)
            self.legend.addItem(self._graphs[label], legend_label)

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

    def updateTableAnalysisDialog(self):
        """Update table analysis dialog."""
        self.table_analysis_dialog.updateData(
            _utils.tableToDataFrame(self.ui.table_ta))

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

                if hasattr(self, 'position'):
                    pos = '{0:.4f}'.format(self.position[i])
                    self.ui.table_ta.setItem(
                        i, 2, _QTableWidgetItem(pos))
                    self.ui.table_ta.setItem(
                        i, j+3, _QTableWidgetItem(
                            self._data_format.format(readings[i])))
                else:
                    self.ui.table_ta.setItem(
                        i, j+2, _QTableWidgetItem(
                            self._data_format.format(readings[i])))

        if scrollDown:
            vbar = self.table_ta.verticalScrollBar()
            vbar.setValue(vbar.maximum())
