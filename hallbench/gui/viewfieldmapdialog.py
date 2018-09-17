# -*- coding: utf-8 -*-

"""View data dialog for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import pandas as _pd
import warnings as _warnings
import traceback as _traceback
from PyQt5.QtWidgets import (
    QDialog as _QDialog,
    QMessageBox as _QMessageBox,
    )
import PyQt5.uic as _uic
import pyqtgraph as _pyqtgraph

from hallbench.gui.tableplotdialog import TablePlotDialog as _TablePlotDialog
from hallbench.gui.utils import getUiFile as _getUiFile


class ViewFieldmapDialog(_QDialog):
    """View fieldmap dialog class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        self.fieldmap = None
        self.pos = None
        self.bx_lines = None
        self.by_lines = None
        self.bz_lines = None
        self.graphx = None
        self.graphy = None
        self.graphz = None
        self.xlabel = ''

        # Create current dialog
        self.current_dialog = _TablePlotDialog()
        self.current_dialog.setWindowTitle('Current Readings')
        self.current_dialog.setPlotLabel('Current [A]')
        self.current_dialog.setTableColumnSize(200)

        # Create temperature dialog
        self.temperature_dialog = _TablePlotDialog()
        self.temperature_dialog.setWindowTitle('Temperature Readings')
        self.temperature_dialog.setPlotLabel('Temperature [deg C]')
        self.temperature_dialog.setTableColumnSize(100)

        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.graph_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        self.connectSignalSlots()

    def accept(self):
        """Close dialog."""
        self.closeDialogs()
        super().accept()

    def clearGraph(self):
        """Clear plots."""
        self.ui.graph_pw.plotItem.curves.clear()
        self.ui.graph_pw.clear()

    def closeDialogs(self):
        """Close dialogs."""
        try:
            self.current_dialog.accept()
            self.temperature_dialog.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.closeDialogs()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def configureGraph(self):
        """Configure graph.

        Args:
            nr_curves (int): number of curves to plot.
        """
        self.legend.removeItem('X')
        self.legend.removeItem('Y')
        self.legend.removeItem('Z')
        self.graphx = self.ui.graph_pw.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=(255, 0, 0),
                symbol='o',
                symbolPen=(255, 0, 0),
                symbolSize=4,
                symbolBrush=(255, 0, 0))

        self.graphy = self.ui.graph_pw.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=(0, 255, 0),
                symbol='o',
                symbolPen=(0, 255, 0),
                symbolSize=4,
                symbolBrush=(0, 255, 0))

        self.graphz = self.ui.graph_pw.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=(0, 0, 255),
                symbol='o',
                symbolPen=(0, 0, 255),
                symbolSize=4,
                symbolBrush=(0, 0, 255))

        self.legend.addItem(self.graphx, 'X')
        self.legend.addItem(self.graphy, 'Y')
        self.legend.addItem(self.graphz, 'Z')
        self.ui.graph_pw.setLabel('bottom', self.xlabel)
        self.ui.graph_pw.setLabel('left', 'Magnetic Field [T]')
        self.ui.graph_pw.showGrid(x=True, y=True)

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.update_plot_btn.clicked.connect(self.updatePlot)
        self.ui.view_current_btn.clicked.connect(self.showCurrentDialog)
        self.ui.view_temperature_btn.clicked.connect(
            self.showTemperatureDialog)

    def show(self, fieldmap):
        """Update fieldmap and show dialog."""
        self.clearGraph()
        self.fieldmap = fieldmap

        try:
            self.updateLineEdits()
            self.updatePlotOptions()
            self.updateText()
            if len(self.fieldmap.current) != 0:
                self.ui.view_current_btn.setEnabled(True)
            if len(self.fieldmap.temperature) != 0:
                self.ui.view_temperature_btn.setEnabled(True)
            super().show()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to show dialog.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def showCurrentDialog(self):
        """Open table plot dialog."""
        if len(self.fieldmap.current) == 0:
            return

        try:
            dfs = []
            for key, value in self.fieldmap.current.items():
                v = _np.array(value)
                dfs.append(
                    _pd.DataFrame(v[:, 1], index=v[:, 0], columns=[key]))
            df = _pd.concat(dfs, axis=1)

            timestamp = df.index.values.tolist()
            readings = {}
            for col in df.columns:
                readings[col] = df[col].values.tolist()
            self.current_dialog.show(timestamp, readings)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to open dialog.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def showTemperatureDialog(self):
        """Show dialog with temperature readings."""
        if len(self.fieldmap.temperature) == 0:
            return

        try:
            dfs = []
            for key, value in self.fieldmap.temperature.items():
                v = _np.array(value)
                dfs.append(
                    _pd.DataFrame(v[:, 1], index=v[:, 0], columns=[key]))
            df = _pd.concat(dfs, axis=1)

            timestamp = df.index.values.tolist()
            readings = {}
            for col in df.columns:
                readings[col] = df[col].values.tolist()
            self.temperature_dialog.show(timestamp, readings)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to open dialog.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def updateLineEdits(self):
        """Update line edit texts."""
        self.ui.nr_scans_le.setText('')
        self.ui.initial_scan_le.setText('')
        self.ui.final_scan_le.setText('')
        self.ui.comments_te.setText('')
        self.ui.magnet_center_pos3_le.setText('')
        self.ui.magnet_center_pos2_le.setText('')
        self.ui.magnet_center_pos1_le.setText('')
        self.ui.magnet_x_axis_le.setText('')
        self.ui.magnet_y_axis_le.setText('')
        if self.fieldmap is None:
            return

        try:
            val = (
                self.fieldmap.nr_scans if self.fieldmap.nr_scans is not None
                else '')
            self.ui.nr_scans_le.setText('{0:d}'.format(val))

            val = (
                self.fieldmap.initial_scan if self.fieldmap.initial_scan
                is not None else '')
            self.ui.initial_scan_le.setText('{0:d}'.format(val))

            val = (
                self.fieldmap.final_scan if self.fieldmap.final_scan
                is not None else '')
            self.ui.final_scan_le.setText('{0:d}'.format(val))

            comments = (
                self.fieldmap.comments if self.fieldmap.comments is not None
                else '')
            self.ui.comments_te.setText(comments)

            mc = self.fieldmap.magnet_center
            if mc is None:
                self.ui.magnet_center_pos3_le.setText('')
                self.ui.magnet_center_pos2_le.setText('')
                self.ui.magnet_center_pos1_le.setText('')
            else:
                self.ui.magnet_center_pos3_le.setText('{0:.4f}'.format(mc[0]))
                self.ui.magnet_center_pos2_le.setText('{0:.4f}'.format(mc[1]))
                self.ui.magnet_center_pos1_le.setText('{0:.4f}'.format(mc[2]))

            magnet_x_axis = self.fieldmap.magnet_x_axis
            if magnet_x_axis is not None:
                if magnet_x_axis > 0:
                    self.ui.magnet_x_axis_le.setText('+ Axis #{0:d}'.format(
                        _np.abs(magnet_x_axis)))
                else:
                    self.ui.magnet_x_axis_le.setText('- Axis #{0:d}'.format(
                        _np.abs(magnet_x_axis)))
            else:
                self.ui.magnet_x_axis_le.setText('')

            magnet_y_axis = self.fieldmap.magnet_y_axis
            if magnet_y_axis is not None:
                if magnet_y_axis > 0:
                    self.ui.magnet_y_axis_le.setText('+ Axis #{0:d}'.format(
                        _np.abs(magnet_y_axis)))
                else:
                    self.ui.magnet_y_axis_le.setText('- Axis #{0:d}'.format(
                        _np.abs(magnet_y_axis)))
            else:
                self.ui.magnet_y_axis_le.setText('')

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return

    def updatePlot(self):
        """Update plot."""
        self.clearGraph()
        self.configureGraph()
        if self.fieldmap is None:
            return

        idx = self.ui.plot_cmb.currentIndex()
        if idx is None:
            return

        if (self.pos is None or self.bx_lines is None
           or self.by_lines is None or self.bz_lines is None):
            return

        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                self.graphx.setData(self.pos, self.bx_lines[idx])
                self.graphy.setData(self.pos, self.by_lines[idx])
                self.graphz.setData(self.pos, self.bz_lines[idx])

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to update plot.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def updatePlotOptions(self):
        """Update plot options."""
        self.ui.plot_la.setText('')
        self.ui.plot_cmb.clear()
        self.ui.plot_cmb.setEnabled(False)
        self.ui.update_plot_btn.setEnabled(False)

        if self.fieldmap is None:
            return

        try:
            _map = self.fieldmap.map
            x = _np.unique(_map[:, 0])
            y = _np.unique(_map[:, 1])
            z = _np.unique(_map[:, 2])
            bx = _map[:, 3]
            by = _map[:, 4]
            bz = _map[:, 5]

            if len(x) != 1 and len(y) != 1 and len(z) != 1:
                return

            if len(x) != 1 and len(y) == 1 and len(z) == 1:
                self.ui.plot_la.setText('Z [mm]:')
                for zi in z:
                    self.ui.plot_cmb.addItem('{0:.4f}'.format(zi))
                self.ui.plot_cmb.setCurrentIndex(-1)

                self.bx_lines = _np.array([bx])
                self.by_lines = _np.array([by])
                self.bz_lines = _np.array([bz])

                self.pos = x
                self.xlabel = 'Position X [mm]'

            elif len(y) != 1 and len(x) == 1 and len(x) == 1:
                self.ui.plot_la.setText('Z [mm]:')
                for zi in z:
                    self.ui.plot_cmb.addItem('{0:.4f}'.format(zi))
                self.ui.plot_cmb.setCurrentIndex(-1)

                self.bx_lines = _np.array([bx])
                self.by_lines = _np.array([by])
                self.bz_lines = _np.array([bz])

                self.pos = y
                self.xlabel = 'Position Y [mm]'

            elif len(z) != 1 and len(x) == 1 and len(y) == 1:
                self.ui.plot_la.setText('X [mm]:')
                for xi in x:
                    self.ui.plot_cmb.addItem('{0:.4f}'.format(xi))
                self.ui.plot_cmb.setCurrentIndex(-1)

                self.bx_lines = _np.array([bx])
                self.by_lines = _np.array([by])
                self.bz_lines = _np.array([bz])

                self.pos = z
                self.xlabel = 'Position Z [mm]'

            elif len(x) != 1 or (len(x) == 1 and len(y) == 1):
                self.ui.plot_la.setText('X [mm]:')
                for xi in x:
                    self.ui.plot_cmb.addItem('{0:.4f}'.format(xi))
                self.ui.plot_cmb.setCurrentIndex(-1)

                bx.shape = (-1, len(x))
                by.shape = (-1, len(x))
                bz.shape = (-1, len(x))
                self.bx_lines = _np.transpose(bx)
                self.by_lines = _np.transpose(by)
                self.bz_lines = _np.transpose(bz)

                if len(y) != 1:
                    self.pos = y
                    self.xlabel = 'Position Y [mm]'
                else:
                    self.pos = z
                    self.xlabel = 'Position Z [mm]'
            else:
                self.ui.plot_la.setText('Y [mm]:')
                for yi in y:
                    self.ui.plot_cmb.addItem('{0:.4f}'.format(yi))
                self.ui.plot_cmb.setCurrentIndex(-1)

                bx.shape = (-1, len(y))
                by.shape = (-1, len(y))
                bz.shape = (-1, len(y))
                self.bx_lines = _np.transpose(bx)
                self.by_lines = _np.transpose(by)
                self.bz_lines = _np.transpose(bz)

                self.pos = z
                self.xlabel = 'Position Z [mm]'

            self.ui.plot_cmb.setEnabled(True)
            self.ui.update_plot_btn.setEnabled(True)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return

    def updateText(self):
        """Update text."""
        if self.fieldmap is not None:
            try:
                text = self.fieldmap.get_fieldmap_text()
                self.ui.text_te.setText(text)
            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                return
        else:
            self.ui.text_te.setText('')
