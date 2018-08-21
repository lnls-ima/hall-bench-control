# -*- coding: utf-8 -*-

"""View data dialog for the Hall Bench Control application."""

import numpy as _np
import scipy.optimize as _optimize
import collections as _collections
import warnings as _warnings
from PyQt4.QtGui import (
    QDialog as _QDialog,
    QMessageBox as _QMessageBox,
    QTableWidgetItem as _QTableWidgetItem,
    )
import PyQt4.uic as _uic
import pyqtgraph as _pyqtgraph

from hallbench.gui.utils import getUiFile as _getUiFile


class ViewDataDialog(_QDialog):
    """View data dialog class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        self.data_label = ''
        self.data_list = []
        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.data_dict = {}
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
              
        sel = self.data_dict[selected_idx]
        pos = sel[1]
        data = sel[2]
        xoff = sel[3]
        yoff = sel[4]
        xmult = sel[5]
        ymult = sel[6]
        xmin = self.ui.xmin_sb.value()
        xmax = self.ui.xmax_sb.value()
        
        x = _np.array((pos + xoff)*xmult)
        y = _np.array((data + yoff)*ymult)
        
        y = y[(x >= xmin) & (x <=xmax)]       
        x = x[(x >= xmin) & (x <=xmax)]

        func = self.ui.fitfunction_cmb.currentText().lower()
        if func == 'linear':
            xfit, yfit, param, label = _linear_fit(x, y)
        elif func == 'polynomial':
            order = self.ui.poly_sb.value()
            xfit, yfit, param, label = _polynomial_fit(x, y, order)
        elif self.ui.fitfunction_cmb.currentText().lower() == 'gaussian':
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

    def clearGraph(self):
        """Clear plots."""
        self.ui.graph_pw.plotItem.curves.clear()
        self.ui.graph_pw.clear()
        self.graphx = []
        self.graphy = []
        self.graphz = []

    def clearFit(self):
        """Clear fit."""
        self.ui.fit_ta.clearContents()
        self.ui.fit_ta.setRowCount(0)
        self.ui.fitfunction_la.setText('')
        self.updatePlot()

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
            self.clearFit)
        self.ui.fitfunction_cmb.currentIndexChanged.connect(
            self.hideShowPolyOrder)
        self.ui.fit_btn.clicked.connect(self.calcCurveFit)
        self.ui.resetlim_btn.clicked.connect(self.resetXLimits)

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

    def disableOffsetLed(self):
        """Disable offset led."""
        self.ui.offsetled_la.setEnabled(False)

    def hideShowPolyOrder(self):
        """Hide or show polynomial fitting order."""
        if self.ui.fitfunction_cmb.currentText().lower() == 'polynomial':
            self.ui.poly_la.show()
            self.ui.poly_sb.show()
        else:
            self.ui.poly_la.hide()
            self.ui.poly_sb.hide()

    def resetXLimits(self):
        """Reset X limits.."""
        selected_idx = self.ui.selectcurve_cmb.currentIndex()
        if selected_idx == -1:
            return
        else:
            pos = self.data_dict[selected_idx][1]
            xoff = self.data_dict[selected_idx][3]
            yoff = self.data_dict[selected_idx][4]
            xmult = self.data_dict[selected_idx][5]
            ymult = self.data_dict[selected_idx][6]

            if len(pos) > 0:
                xmin = (pos[0] + xoff)*xmult
                xmax = (pos[-1] + xoff)*xmult
            else:
                xmin = 0
                xmax = 0
            
            self.data_dict[selected_idx][7] = xmin
            self.ui.xmin_sb.setValue(xmin)
            
            self.data_dict[selected_idx][8] = xmax
            self.ui.xmax_sb.setValue(xmax)
            
            self.updatePlot()

    def setCurveLabels(self):
        """Set curve labels"""
        self.ui.selectcurve_cmb.clear()

        p1 = set()
        p2 = set()
        p3 = set()
        for data in self.data_list:
            p1.update(data.pos1)
            p2.update(data.pos2)
            p3.update(data.pos3)
                
        curve_labels = []    
        for data in self.data_list:
            pos = ''
            if data.pos1.size == 1 and len(p1) != 1:
                pos = pos + 'pos1={0:.0f}mm'.format(data.pos1[0])
            if data.pos2.size == 1  and len(p2) != 1:
                pos = pos + 'pos2={0:.0f}mm'.format(data.pos2[0])
            if data.pos3.size == 1 and len(p3) != 1:
                pos = pos + 'pos3={0:.0f}mm'.format(data.pos3[0])
            if len(pos) == 0:
                curve_labels.append('X')
                curve_labels.append('Y')
                curve_labels.append('Z')
            else:
                curve_labels.append('X: ' + pos)
                curve_labels.append('Y: ' + pos)
                curve_labels.append('Z: ' + pos)
                
        self.ui.selectcurve_cmb.addItems(curve_labels)
        self.ui.selectcurve_cmb.setEnabled(True)

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
        
        try:
            idx = 0
            for i in range(len(self.data_list)):
                data = self.data_list[i]
                pos = data.scan_pos
                if len(pos) > 0:
                    xmin = pos[0]
                    xmax = pos[-1]
                else:
                    xmin = 0
                    xmax = 0
                self.data_dict[idx] = [
                    'x', pos, data.avgx, 0, 0, 1, 1, xmin, xmax]
                self.data_dict[idx+1] = [
                    'y', pos, data.avgy, 0, 0, 1, 1, xmin, xmax]
                self.data_dict[idx+2] = [
                    'z', pos, data.avgz, 0, 0, 1, 1, xmin, xmax]
                idx = idx + 3
            
            self.updateXLimits() 
            self.updatePlot()
            self.setCurveLabels()
            self.ui.poly_la.hide()
            self.ui.poly_sb.hide()
            super().show()     
        
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)
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
        else:
            self.ui.offset_gb.setEnabled(True)
            self.ui.offsetled_la.setEnabled(False)
            self.ui.xoff_sb.setValue(self.data_dict[selected_idx][3])
            self.ui.yoff_sb.setValue(self.data_dict[selected_idx][4])
            self.ui.xmult_sb.setValue(self.data_dict[selected_idx][5])
            self.ui.ymult_sb.setValue(self.data_dict[selected_idx][6])       
            self.ui.fit_gb.setEnabled(True)
            self.updateXLimits()  
            self.updatePlot()            

    def updateXLimits(self):
        """Update xmin and xmax values."""
        selected_idx = self.ui.selectcurve_cmb.currentIndex()
        if selected_idx == -1:
            return
        else:
            self.ui.xmin_sb.setValue(self.data_dict[selected_idx][7])
            self.ui.xmax_sb.setValue(self.data_dict[selected_idx][8])
    
    def updateXMax(self):
        """Update xmax value."""
        selected_idx = self.ui.selectcurve_cmb.currentIndex()
        if selected_idx == -1:
            return
        else:
            self.data_dict[selected_idx][8] = self.ui.xmax_sb.value()

    def updateXMaxSpinBox(self):
        """Update xmax value."""
        self.ui.xmax_sb.setValue(self.xmax_line.pos()[0])
    
    def updateXMin(self):
        """Update xmin value."""
        selected_idx = self.ui.selectcurve_cmb.currentIndex()
        if selected_idx == -1:
            return
        else:
            self.data_dict[selected_idx][7] = self.ui.xmin_sb.value()
    
    def updateXMinSpinBox(self):
        """Update xmin value."""
        self.ui.xmin_sb.setValue(self.xmin_line.pos()[0])

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
        xmin = self.data_dict[selected_idx][7] 
        xmax = self.data_dict[selected_idx][8]
        
        self.data_dict[selected_idx][3] = xoff
        self.data_dict[selected_idx][4] = yoff
        self.data_dict[selected_idx][5] = xmult
        self.data_dict[selected_idx][6] = ymult
        self.data_dict[selected_idx][7] = (xmin + xoff)*xmult
        self.data_dict[selected_idx][8] = (xmax + xoff)*xmult

        self.updatePlot()
        self.ui.offsetled_la.setEnabled(True)

    def updatePlot(self):
        """Update plot."""
        self.clearGraph()
        if self.ui.showall_rb.isChecked():
            nr_curves = len(self.data_list)
            data_dict = self.data_dict
            show_xlines = False
        else:
            selected_idx = self.ui.selectcurve_cmb.currentIndex()
            if selected_idx == -1:
                return
            nr_curves = 1
            data_dict = {selected_idx : self.data_dict[selected_idx]}
            show_xlines = True 
        
        self.configureGraph(nr_curves, self.data_label)       

        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                x_count = 0
                y_count = 0
                z_count = 0
                for value in data_dict.values():
                    graph = value[0]
                    pos = value[1]
                    data = value[2]
                    xoff = value[3]
                    yoff = value[4]
                    xmult = value[5]
                    ymult = value[6]
                    x = (pos + xoff)*xmult
                    y = (data + yoff)*ymult
                    if graph == 'x':
                        self.graphx[x_count].setData(x, y)
                        x_count = x_count = 1
                    elif graph == 'y':
                        self.graphy[y_count].setData(x, y)
                        y_count = y_count = 1
                    elif graph == 'z':
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
                
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)
            return


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
            xfit = []
            yfit = []
            param = {}

    return (xfit, yfit, param, label)


def _gaussian_fit(x, y):
    label = 'y = A*exp(-(x-x0)^2/(2*Sigma^2))'

    def gauss(x, a, x0, sigma):
        return a*_np.exp(-(x-x0)**2/(2*sigma**2))
    
    try:    
        n = len(x)
        mean = sum(x*y)/n
        sigma = sum(y*(x-mean)**2)/n   

        popt, pcov = _optimize.curve_fit(gauss, x, y, p0=[1, mean, sigma])
        
        xfit = _np.linspace(x[0], x[-1], 100)
        yfit = gauss(xfit, popt[0], popt[1], popt[2])
    
        param = _collections.OrderedDict([
            ('A', popt[0]),
            ('x0', popt[1]),
            ('Sigma', popt[2]),
        ])
    except Exception:
        xfit = []
        yfit = []
        param = {}

    return (xfit, yfit, param, label)

