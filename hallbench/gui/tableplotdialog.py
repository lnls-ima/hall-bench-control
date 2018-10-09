# -*- coding: utf-8 -*-

"""Table and plot dialog for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import datetime as _datetime
import warnings as _warnings
import pyqtgraph as _pyqtgraph
import traceback as _traceback
from PyQt5.QtWidgets import (
    QDialog as _QDialog,
    QFileDialog as _QFileDialog,
    QMessageBox as _QMessageBox,
    QTableWidgetItem as _QTableWidgetItem,
    )
import PyQt5.uic as _uic

import hallbench.gui.utils as _utils


class TablePlotDialog(_QDialog):
    """Table and plot dialog class for the Hall Bench Control application."""

    _data_format = '{0:.4f}'
    _colors = [
        (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
        (145, 30, 180), (255, 225, 25), (70, 240, 240),
        (240, 50, 230), (170, 110, 40), (128, 0, 0),
        (0, 0, 0), (128, 128, 128), (170, 255, 195),
    ]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _utils.getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialization
        self._plot_label = ''
        self._timestamp = []
        self._data_labels = []
        self._legend_items = []
        self._readings = {}
        self._graphs = {}

        self.table_analysis_dialog = _utils.TableAnalysisDialog()

        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.plot_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        self.connectSignalSlots()

    def clearLegendItems(self):
        """Clear plot legend."""
        for label in self._legend_items:
            self.legend.removeItem(label)

    def accept(self):
        """Close dialog."""
        self.clear()
        self.closeDialogs()
        super().accept()

    def clear(self):
        """Clear data."""
        self.clearLegendItems()
        self._plot_label = ''
        self._timestamp = []
        self._data_labels = []
        self._legend_items = []
        self._readings = {}
        self._graphs = {}

    def closeDialogs(self):
        """Close dialogs."""
        try:
            self.table_analysis_dialog.accept()
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

    def configurePlot(self):
        """Configure data plots."""
        self.ui.plot_pw.clear()

        for i, label in enumerate(self._data_labels):
            try:
                pen = self._colors[i]
            except IndexError:
                pen = (0, 0, 0)
            graph = self.ui.plot_pw.plotItem.plot(
                _np.array([]), _np.array([]), pen=pen,
                symbol='o', symbolPen=pen, symbolSize=3, symbolBrush=pen)
            self._graphs[label] = graph

        self.ui.plot_pw.setLabel('bottom', 'Time interval [s]')
        self.ui.plot_pw.setLabel('left', self._plot_label)
        self.ui.plot_pw.showGrid(x=True, y=True)
        self.updateLegendItems()

    def configureTable(self):
        """Configure table."""
        col_labels = ['Date', 'Time']
        for label in self._data_labels:
            col_labels.append(label)
        self.ui.table_ta.setColumnCount(len(col_labels))
        self.ui.table_ta.setHorizontalHeaderLabels(col_labels)
        self.ui.table_ta.setAlternatingRowColors(True)

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.copy_btn.clicked.connect(self.copyToClipboard)
        self.ui.save_btn.clicked.connect(self.saveToFile)
        self.ui.analysis_btn.clicked.connect(self.showTableAnalysisDialog)

    def copyToClipboard(self):
        """Copy table data to clipboard."""
        df = _utils.tableToDataFrame(self.ui.table_ta)
        if df is not None:
            df.to_clipboard(excel=True)

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

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save data to file.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def setPlotLabel(self, plot_label):
        """Set plot label value."""
        self._plot_label = plot_label

    def setTableColumnSize(self, size):
        """Set table horizontal header default section size."""
        self.ui.table_ta.horizontalHeader().setDefaultSectionSize(size)

    def show(self, timestamp, readings):
        """Show dialog."""
        self._timestamp = timestamp
        self._readings = readings
        self._data_labels = list(self._readings.keys())
        self.configurePlot()
        self.configureTable()
        self.updatePlot()
        self.updateTableValues()
        super().show()

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

    def updatePlot(self):
        """Update plot values."""
        if len(self._timestamp) == 0:
            for label in self._data_labels:
                self._graphs[label].setData(
                    _np.array([]), _np.array([]))
            return

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            timeinterval = _np.array(self._timestamp) - self._timestamp[0]
            for label in self._data_labels:
                readings = _np.array(self._readings[label])
                dt = timeinterval[_np.isfinite(readings)]
                rd = readings[_np.isfinite(readings)]
                self._graphs[label].setData(dt, rd)

    def updateTableValues(self, scrollDown=True):
        """Update table values."""
        n = len(self._timestamp)
        self.ui.table_ta.clearContents()
        self.ui.table_ta.setRowCount(n)

        for j in range(len(self._data_labels)):
            readings = self._readings[self._data_labels[j]]
            for i in range(n):
                dt = _datetime.datetime.fromtimestamp(self._timestamp[i])
                date = dt.strftime("%d/%m/%Y")
                hour = dt.strftime("%H:%M:%S")

                self.ui.table_ta.setItem(i, 0, _QTableWidgetItem(date))
                self.ui.table_ta.setItem(i, 1, _QTableWidgetItem(hour))
                self.ui.table_ta.setItem(
                    i, j+2, _QTableWidgetItem(
                        self._data_format.format(readings[i])))

        if scrollDown:
            vbar = self.table_ta.verticalScrollBar()
            vbar.setValue(vbar.maximum())
