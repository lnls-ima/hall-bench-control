# -*- coding: utf-8 -*-

"""View data dialog for the Hall Bench Control application."""

import numpy as _np
import warnings as _warnings
from PyQt5.QtWidgets import (
    QDialog as _QDialog,
    QMessageBox as _QMessageBox,
    )
import PyQt5.uic as _uic
import pyqtgraph as _pyqtgraph

from hallbench.gui.utils import getUiFile as _getUiFile


class ViewDataDialog(_QDialog):
    """View data dialog class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialization
        self.data_label = ''
        self.data_list = []
        self.graphx = []
        self.graphy = []
        self.graphz = []

        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.graph_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)
        self.connectSignalSlots()

    def clearGraph(self):
        """Clear plots."""
        self.ui.graph_pw.plotItem.curves.clear()
        self.ui.graph_pw.clear()
        self.graphx = []
        self.graphy = []
        self.graphz = []

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        pass

    def configureGraph(self, nr_curves, label):
        """Configure graph.

        Args:
            nr_curves (int): number of curves to plot.
        """
        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.legend.removeItem('X')
        self.legend.removeItem('Y')
        self.legend.removeItem('Z')

        for idx in range(nr_curves):
            self.graphx.append(
                self.ui.graph_pw.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(255, 0, 0),
                    symbol='o',
                    symbolPen=(255, 0, 0),
                    symbolSize=4,
                    symbolBrush=(255, 0, 0)))

            self.graphy.append(
                self.ui.graph_pw.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(0, 255, 0),
                    symbol='o',
                    symbolPen=(0, 255, 0),
                    symbolSize=4,
                    symbolBrush=(0, 255, 0)))

            self.graphz.append(
                self.ui.graph_pw.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(0, 0, 255),
                    symbol='o',
                    symbolPen=(0, 0, 255),
                    symbolSize=4,
                    symbolBrush=(0, 0, 255)))

        self.legend.addItem(self.graphx[0], 'X')
        self.legend.addItem(self.graphy[0], 'Y')
        self.legend.addItem(self.graphz[0], 'Z')
        self.ui.graph_pw.setLabel('bottom', 'Scan Position [mm]')
        self.ui.graph_pw.setLabel('left', label)
        self.ui.graph_pw.showGrid(x=True, y=True)

    def plotData(self):
        """Plot field data."""
        self.clearGraph()
        nr_curves = len(self.data_list)
        self.configureGraph(nr_curves, self.data_label)

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            for i in range(nr_curves):
                data = self.data_list[i]
                positions = data.scan_pos
                self.graphx[i].setData(positions[:len(data.avgx)], data.avgx)
                self.graphy[i].setData(positions[:len(data.avgy)], data.avgy)
                self.graphz[i].setData(positions[:len(data.avgz)], data.avgz)

    def show(self, data_list, data_label):
        """Update data and show dialog."""
        if data_list is None or len(data_list) == 0:
            message = 'Invalid data list.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return

        if data_label is None:
            data_label = ''

        self.data_list = data_list
        self.data_label = data_label
        self.plotData()
        super().show()
