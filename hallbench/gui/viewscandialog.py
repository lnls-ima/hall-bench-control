# -*- coding: utf-8 -*-

"""View data dialog for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import scipy.optimize as _optimize
import collections as _collections
import warnings as _warnings
import traceback as _traceback
from PyQt5.QtWidgets import (
    QDialog as _QDialog,
    QMessageBox as _QMessageBox,
    QTableWidgetItem as _QTableWidgetItem,
    )
import PyQt5.uic as _uic
import pyqtgraph as _pyqtgraph

from hallbench.gui.utils import getUiFile as _getUiFile


class ViewScanDialog(_QDialog):
    """View data dialog class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        self.data_label = ''
        self.scan_list = []
        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.scan_dict = {}
        self.xmin_line = None
        self.xmax_line = None

        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.graph_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)
        self.connectSignalSlots()

    def calcCurveFit(self):
        """Calculate curve fit."""
        selected_idx = self.ui.selectcurve_cmb.currentIndex()
        if selected_idx == -1:
            return

        sel = self.scan_dict[selected_idx]
        pos = sel['pos']
        data = sel['data']
        xoff = sel['xoff']
        yoff = sel['yoff']
        xmult = sel['xmult']
        ymult = sel['ymult']
        xmin = self.ui.xmin_sb.value()
        xmax = self.ui.xmax_sb.value()

        x = _np.array((pos + xoff)*xmult)
        y = _np.array((data + yoff)*ymult)

        y = y[(x >= xmin) & (x <= xmax)]
        x = x[(x >= xmin) & (x <= xmax)]

        func = self.ui.fitfunction_cmb.currentText()
        if func.lower() == 'linear':
            xfit, yfit, param, label = _linear_fit(x, y)
        elif func.lower() == 'polynomial':
            order = self.ui.polyorder_sb.value()
            xfit, yfit, param, label = _polynomial_fit(x, y, order)
        elif func.lower() == 'gaussian':
            xfit, yfit, param, label = _gaussian_fit(x, y)
        else:
            xfit = []
            yfit = []
            param = {}
            label = ''

        self.ui.fitfunction_la.setText(label)
        self.ui.fit_ta.clearContents()
        self.ui.fit_ta.setRowCount(len(param))
        rcount = 0
        for key, value in param.items():
            self.ui.fit_ta.setItem(
                rcount, 0, _QTableWidgetItem(str(key)))
            self.ui.fit_ta.setItem(
                rcount, 1, _QTableWidgetItem(str(value)))
            rcount = rcount + 1

        self.updatePlot()
        self.ui.graph_pw.plotItem.plot(xfit, yfit, pen=(0, 0, 0))

    def clearFit(self):
        """Clear fit."""
        self.ui.fit_ta.clearContents()
        self.ui.fit_ta.setRowCount(0)
        self.ui.fitfunction_la.setText('')
        self.updatePlot()

    def clearGraph(self):
        """Clear plots."""
        self.ui.graph_pw.plotItem.curves.clear()
        self.ui.graph_pw.clear()
        self.graphx = []
        self.graphy = []
        self.graphz = []

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

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.updateplot_btn.clicked.connect(self.updatePlot)
        self.ui.selectcurve_cmb.currentIndexChanged.connect(
            self.updateControls)
        self.ui.offset_btn.clicked.connect(self.updateOffset)
        self.ui.xoff_sb.valueChanged.connect(self.disableOffsetLed)
        self.ui.yoff_sb.valueChanged.connect(self.disableOffsetLed)
        self.ui.xmult_sb.valueChanged.connect(self.disableOffsetLed)
        self.ui.ymult_sb.valueChanged.connect(self.disableOffsetLed)
        self.ui.xmin_sb.valueChanged.connect(self.updateXMin)
        self.ui.xmax_sb.valueChanged.connect(self.updateXMax)
        self.ui.fitfunction_cmb.currentIndexChanged.connect(
            self.fitFunctionChanged)
        self.ui.polyorder_sb.valueChanged.connect(self.polyOrderChanged)
        self.ui.fit_btn.clicked.connect(self.calcCurveFit)
        self.ui.resetlim_btn.clicked.connect(self.resetXLimits)

    def disableOffsetLed(self):
        """Disable offset led."""
        self.ui.offsetled_la.setEnabled(False)

    def fitFunctionChanged(self):
        """Hide or show polynomial fitting order and update dict value."""
        self.clearFit()
        func = self.ui.fitfunction_cmb.currentText()
        if func.lower() == 'polynomial':
            self.ui.polyorder_la.show()
            self.ui.polyorder_sb.show()
        else:
            self.ui.polyorder_la.hide()
            self.ui.polyorder_sb.hide()

        selected_idx = self.ui.selectcurve_cmb.currentIndex()
        if selected_idx == -1:
            return
        else:
            self.scan_dict[selected_idx]['fit_function'] = func

    def polyOrderChanged(self):
        """Update dict value."""
        order = self.ui.polyorder_sb.value()
        selected_idx = self.ui.selectcurve_cmb.currentIndex()
        if selected_idx == -1:
            return
        else:
            self.scan_dict[selected_idx]['fit_polyorder'] = order

    def resetXLimits(self):
        """Reset X limits.."""
        selected_idx = self.ui.selectcurve_cmb.currentIndex()
        if selected_idx == -1:
            return
        else:
            pos = self.scan_dict[selected_idx]['pos']
            xoff = self.scan_dict[selected_idx]['xoff']
            xmult = self.scan_dict[selected_idx]['xmult']

            if len(pos) > 0:
                xmin = (pos[0] + xoff)*xmult
                xmax = (pos[-1] + xoff)*xmult
            else:
                xmin = 0
                xmax = 0

            self.scan_dict[selected_idx]['xmin'] = xmin
            self.ui.xmin_sb.setValue(xmin)

            self.scan_dict[selected_idx]['xmax'] = xmax
            self.ui.xmax_sb.setValue(xmax)

            self.updatePlot()

    def setCurveLabels(self):
        """Set curve labels."""
        self.ui.selectcurve_cmb.clear()

        p1 = set()
        p2 = set()
        p3 = set()
        for data in self.scan_list:
            p1.update(data.pos1)
            p2.update(data.pos2)
            p3.update(data.pos3)

        curve_labels = []
        for data in self.scan_list:
            pos = ''
            if data.pos1.size == 1 and len(p1) != 1:
                pos = pos + 'pos1={0:.0f}mm'.format(data.pos1[0])
            if data.pos2.size == 1 and len(p2) != 1:
                pos = pos + 'pos2={0:.0f}mm'.format(data.pos2[0])
            if data.pos3.size == 1 and len(p3) != 1:
                pos = pos + 'pos3={0:.0f}mm'.format(data.pos3[0])

            if data.timestamp is not None:
                ts = ' ({0:s})'.format(data.timestamp.split('_')[1])
            else:
                ts = ''

            if len(pos) == 0:
                curve_labels.append('X' + ts)
                curve_labels.append('Y' + ts)
                curve_labels.append('Z' + ts)
            else:
                curve_labels.append('X: ' + pos + ts)
                curve_labels.append('Y: ' + pos + ts)
                curve_labels.append('Z: ' + pos + ts)

        self.ui.selectcurve_cmb.addItems(curve_labels)
        self.ui.selectcurve_cmb.setEnabled(True)

    def show(self, scan_list, data_label=''):
        """Update data and show dialog."""
        if scan_list is None or len(scan_list) == 0:
            msg = 'Invalid data list.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        self.scan_list = [d.copy() for d in scan_list]
        self.data_label = data_label

        try:
            idx = 0
            for i in range(len(self.scan_list)):
                data = self.scan_list[i]

                if data.npts == 0:
                    msg = 'Invalid scan found.'
                    _QMessageBox.critical(
                        self, 'Failure', msg, _QMessageBox.Ok)
                    return

                pos = data.scan_pos
                if len(data.avgx) == 0:
                    data.avgx = _np.zeros(len(pos))
                if len(data.avgy) == 0:
                    data.avgy = _np.zeros(len(pos))
                if len(data.avgz) == 0:
                    data.avgz = _np.zeros(len(pos))

                xmin = pos[0]
                xmax = pos[-1]

                self.scan_dict[idx] = {
                    'component': 'x',
                    'pos': pos,
                    'data': data.avgx,
                    'xoff': 0,
                    'yoff': 0,
                    'xmult': 1,
                    'ymult': 1,
                    'xmin': xmin,
                    'xmax': xmax,
                    'fit_function': 'Linear',
                    'fit_polyorder': None,
                    }

                self.scan_dict[idx+1] = {
                    'component': 'y',
                    'pos': pos,
                    'data': data.avgy,
                    'xoff': 0,
                    'yoff': 0,
                    'xmult': 1,
                    'ymult': 1,
                    'xmin': xmin,
                    'xmax': xmax,
                    'fit_function': 'Linear',
                    'fit_polyorder': None,
                    }

                self.scan_dict[idx+2] = {
                    'component': 'z',
                    'pos': pos,
                    'data': data.avgz,
                    'xoff': 0,
                    'yoff': 0,
                    'xmult': 1,
                    'ymult': 1,
                    'xmin': xmin,
                    'xmax': xmax,
                    'fit_function': 'Linear',
                    'fit_polyorder': None,
                    }

                idx = idx + 3

            self.updateXLimits()
            self.updatePlot()
            self.setCurveLabels()
            self.ui.polyorder_la.hide()
            self.ui.polyorder_sb.hide()
            super().show()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to show dialog.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def updateControls(self):
        """Enable offset and fit group box and update offset values."""
        self.clearFit()
        selected_idx = self.ui.selectcurve_cmb.currentIndex()
        if selected_idx == -1:
            self.ui.offset_gb.setEnabled(False)
            self.ui.offsetled_la.setEnabled(False)
            self.ui.fit_gb.setEnabled(False)
            self.ui.xoff_sb.setValue(0)
            self.ui.yoff_sb.setValue(0)
            self.ui.xmult_sb.setValue(1)
            self.ui.ymult_sb.setValue(1)
            self.ui.fitfunction_cmb.setCurrentIndex(-1)
        else:
            self.ui.offset_gb.setEnabled(True)
            self.ui.offsetled_la.setEnabled(False)
            self.ui.xoff_sb.setValue(self.scan_dict[selected_idx]['xoff'])
            self.ui.yoff_sb.setValue(self.scan_dict[selected_idx]['yoff'])
            self.ui.xmult_sb.setValue(self.scan_dict[selected_idx]['xmult'])
            self.ui.ymult_sb.setValue(self.scan_dict[selected_idx]['ymult'])
            self.ui.fit_gb.setEnabled(True)

            func = self.scan_dict[selected_idx]['fit_function']
            if func is None:
                self.ui.fitfunction_cmb.setCurrentIndex(-1)
            else:
                idx = self.ui.fitfunction_cmb.findText(func)
                self.ui.fitfunction_cmb.setCurrentIndex(idx)

            order = self.scan_dict[selected_idx]['fit_polyorder']
            if order is not None:
                self.ui.polyorder_sb.setValue(order)

            self.updateXLimits()
            self.updatePlot()

    def updateOffset(self):
        """Update curve offset."""
        self.clearFit()
        selected_idx = self.ui.selectcurve_cmb.currentIndex()
        if selected_idx == -1:
            self.ui.offsetled_la.setEnabled(False)
            return

        xoff = self.ui.xoff_sb.value()
        yoff = self.ui.yoff_sb.value()
        xmult = self.ui.xmult_sb.value()
        ymult = self.ui.ymult_sb.value()
        xmin = self.scan_dict[selected_idx]['xmin']
        xmax = self.scan_dict[selected_idx]['xmax']

        self.scan_dict[selected_idx]['xoff'] = xoff
        self.scan_dict[selected_idx]['yoff'] = yoff
        self.scan_dict[selected_idx]['xmult'] = xmult
        self.scan_dict[selected_idx]['ymult'] = ymult
        self.scan_dict[selected_idx]['xmin'] = (xmin + xoff)*xmult
        self.scan_dict[selected_idx]['xmax'] = (xmax + xoff)*xmult

        self.updatePlot()
        self.ui.offsetled_la.setEnabled(True)

    def updatePlot(self):
        """Update plot."""
        self.clearGraph()
        if self.ui.showall_rb.isChecked():
            nr_curves = len(self.scan_list)
            scan_dict = self.scan_dict
            show_xlines = False
        else:
            selected_idx = self.ui.selectcurve_cmb.currentIndex()
            if selected_idx == -1:
                return
            nr_curves = 1
            scan_dict = {selected_idx: self.scan_dict[selected_idx]}
            show_xlines = True

        self.configureGraph(nr_curves, self.data_label)

        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                x_count = 0
                y_count = 0
                z_count = 0
                for value in scan_dict.values():
                    component = value['component']
                    pos = value['pos']
                    data = value['data']
                    xoff = value['xoff']
                    yoff = value['yoff']
                    xmult = value['xmult']
                    ymult = value['ymult']
                    x = (pos + xoff)*xmult
                    y = (data + yoff)*ymult
                    if component == 'x':
                        self.graphx[x_count].setData(x, y)
                        x_count = x_count = 1
                    elif component == 'y':
                        self.graphy[y_count].setData(x, y)
                        y_count = y_count = 1
                    elif component == 'z':
                        self.graphz[z_count].setData(x, y)
                        z_count = z_count = 1

            if show_xlines:
                xmin = self.ui.xmin_sb.value()
                xmax = self.ui.xmax_sb.value()

                self.xmin_line = _pyqtgraph.InfiniteLine(
                    xmin, pen=(0, 0, 0), movable=True)
                self.ui.graph_pw.addItem(self.xmin_line)
                self.xmin_line.sigPositionChangeFinished.connect(
                    self.updateXMinSpinBox)

                self.xmax_line = _pyqtgraph.InfiniteLine(
                    xmax, pen=(0, 0, 0), movable=True)
                self.ui.graph_pw.addItem(self.xmax_line)
                self.xmax_line.sigPositionChangeFinished.connect(
                    self.updateXMaxSpinBox)
            else:
                self.xmin_line = None
                self.xmax_line = None

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to update plot.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def updateXLimits(self):
        """Update xmin and xmax values."""
        selected_idx = self.ui.selectcurve_cmb.currentIndex()
        if selected_idx == -1:
            return
        else:
            self.ui.xmin_sb.setValue(self.scan_dict[selected_idx]['xmin'])
            self.ui.xmax_sb.setValue(self.scan_dict[selected_idx]['xmax'])

    def updateXMax(self):
        """Update xmax value."""
        selected_idx = self.ui.selectcurve_cmb.currentIndex()
        if selected_idx == -1:
            return
        else:
            self.scan_dict[selected_idx]['xmax'] = self.ui.xmax_sb.value()

    def updateXMaxSpinBox(self):
        """Update xmax value."""
        self.ui.xmax_sb.setValue(self.xmax_line.pos()[0])

    def updateXMin(self):
        """Update xmin value."""
        selected_idx = self.ui.selectcurve_cmb.currentIndex()
        if selected_idx == -1:
            return
        else:
            self.scan_dict[selected_idx]['xmin'] = self.ui.xmin_sb.value()

    def updateXMinSpinBox(self):
        """Update xmin value."""
        self.ui.xmin_sb.setValue(self.xmin_line.pos()[0])


def _linear_fit(x, y):
    label = 'y = K0 + K1*x'

    if len(x) != len(y) or len(x) < 2:
        xfit = []
        yfit = []
        param = {}

    else:
        try:
            p = _np.polyfit(x, y, 1)
            xfit = _np.linspace(x[0], x[-1], 100)
            yfit = _np.polyval(p, xfit)

            prev = p[::-1]
            if prev[1] != 0:
                x0 = -prev[0]/prev[1]
            else:
                x0 = _np.nan
            param = _collections.OrderedDict([
                ('K0', prev[0]),
                ('K1', prev[1]),
                ('x (y=0)', x0),
            ])
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            xfit = []
            yfit = []
            param = {}

    return (xfit, yfit, param, label)


def _polynomial_fit(x, y, order):
    label = 'y = K0 + K1*x + K2*x^2 + ...'

    if len(x) != len(y) or len(x) < order + 1:
        xfit = []
        yfit = []
        param = {}

    else:
        try:
            p = _np.polyfit(x, y, order)
            xfit = _np.linspace(x[0], x[-1], 100)
            yfit = _np.polyval(p, xfit)
            prev = p[::-1]
            _dict = {}
            for i in range(len(prev)):
                _dict['K' + str(i)] = prev[i]
            param = _collections.OrderedDict(_dict)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            xfit = []
            yfit = []
            param = {}

    return (xfit, yfit, param, label)


def _gaussian_fit(x, y):
    label = 'y = y0 + A*exp(-(x-x0)^2/(2*Sigma^2))'

    def gaussian(x, y0, a, x0, sigma):
        return y0 + a*_np.exp(-(x-x0)**2/(2*sigma**2))

    try:
        mean = sum(x * y) / sum(y)
        sigma = _np.sqrt(sum(y * (x - mean)**2) / sum(y))
        if mean <= _np.max(y):
            a = _np.max(y)
            y0 = _np.min(y)
        else:
            a = _np.min(y)
            y0 = _np.max(y)

        popt, pcov = _optimize.curve_fit(
            gaussian, x, y, p0=[y0, a, mean, sigma])

        xfit = _np.linspace(x[0], x[-1], 100)
        yfit = gaussian(xfit, popt[0], popt[1], popt[2], popt[3])

        param = _collections.OrderedDict([
            ('y0', popt[0]),
            ('A', popt[1]),
            ('x0', popt[2]),
            ('Sigma', popt[3]),
        ])
    except Exception:
        _traceback.print_exc(file=_sys.stdout)
        xfit = []
        yfit = []
        param = {}

    return (xfit, yfit, param, label)
