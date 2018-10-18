# -*- coding: utf-8 -*-

"""Cooling System widget for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import time as _time
import warnings as _warnings
import pyqtgraph as _pyqtgraph
import traceback as _traceback
from PyQt5.QtWidgets import (
    QApplication as _QApplication,
    QMessageBox as _QMessageBox,
    QPushButton as _QPushButton,
    QVBoxLayout as _QVBoxLayout,
    QWidget as _QWidget,
    )
from PyQt5.QtCore import (
    pyqtSignal as _pyqtSignal,
    QDateTime as _QDateTime,
    Qt as _Qt,
    )
import PyQt5.uic as _uic

import hallbench.gui.utils as _utils
from hallbench.gui.tableplotwidget import TablePlotWidget as _TablePlotWidget


class CoolingSystemWidget(_TablePlotWidget):
    """Cooling System Widget class for the Hall Bench Control application."""

    _plot_label = 'Water Temperature [deg C]'
    _data_format = '{0:.4f}'
    _data_labels = ['PV1', 'PV2', 'Output']
    _colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # Change default appearance
        self.ui.widget_wg.hide()
        self.ui.table_ta.horizontalHeader().setDefaultSectionSize(200)
        self.ui.read_btn.setText('Read PVs')
        self.ui.monitor_btn.setText('Monitor PVs')
        self.addRightAxes()

    @property
    def udc(self):
        """Honeywell UDC3500."""
        return _QApplication.instance().devices.udc
    
    def addRightAxes(self):
        """Add axis to graph."""
        p = self.ui.plot_pw.plotItem
        pr1 = _pyqtgraph.ViewBox()
        p.showAxis('right')
        ax_pr1 = p.getAxis('right')
        p.scene().addItem(pr1)
        ax_pr1.linkToView(pr1)
        pr1.setXLink(p)
        self.right_axis = ax_pr1

        def updateViews():
            pr1.setGeometry(p.vb.sceneBoundingRect())
            pr1.linkedViewChanged(p.vb, pr1.XAxis)

        updateViews()
        p.vb.sigResized.connect(updateViews)
        self.right_axis.setLabel('Controller Output', color="#0000FF")
        self.right_axis.setStyle(showValues=True)
        self.right_axis_curve = _pyqtgraph.PlotCurveItem(
            [], [], pen=(0, 0, 255))
        self.right_axis.linkedView().addItem(self.right_axis_curve)  
    
    def readValue(self, monitor=False):
        """Read value."""
        try:
            ts = _time.time()
            pv1 = self.udc.read_pv1()
            pv2 = self.udc.read_pv2()
            output = self.udc.read_co()
            self._readings['PV1'].append(pv1)
            self._readings['PV2'].append(pv2)
            self._readings['Output'].append(output)
            self._timestamp.append(ts)
            self.addLastValueToTable()
            self.updatePlot()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass
        
    def updatePlot(self):
        """Update plot values."""
        if len(self._timestamp) == 0:
            for label in self._data_labels:
                if label == 'Output':
                    self.right_axis_curve.setData(
                        _np.array([]), _np.array([]))
                else:
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
                if label == 'Output':
                    self.right_axis_curve.setData(dt, rd)
                else:
                    self._graphs[label].setData(dt, rd)     
