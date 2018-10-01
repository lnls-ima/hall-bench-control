# -*- coding: utf-8 -*-

"""View data dialog for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import pandas as _pd
import scipy.optimize as _optimize
import scipy.integrate as _integrate
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

from hallbench.gui.tableplotdialog import TablePlotDialog as _TablePlotDialog
from hallbench.gui import utils as _utils
from hallbench.data import measurement as _measurement


class ViewScanDialog(_QDialog):
    """View data dialog class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        uifile = _utils.getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        self.plot_label = ''
        self.scan_list = []
        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.scan_dict = {}
        self.xmin_line = None
        self.xmax_line = None
        self.current = None
        self.temperature = {}

        # Create current dialog
        self.current_dialog = _utils.TableDialog()
        self.current_dialog.setWindowTitle('Current Readings')

        # Create temperature dialog
        self.temperature_dialog = _TablePlotDialog()
        self.temperature_dialog.setWindowTitle('Temperature Readings')
        self.temperature_dialog.setPlotLabel('Temperature [deg C]')
        self.temperature_dialog.setTableColumnSize(100)

        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.graph_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        self.addRightAxes()
        self.connectSignalSlots()

    def accept(self):
        """Close dialog."""
        self.closeDialogs()
        super().accept()

    def addRightAxes(self):
        """Add axis to graph."""
        p = self.ui.graph_pw.plotItem

        pr1 = _pyqtgraph.ViewBox()
        p.showAxis('right')
        ax_pr1 = p.getAxis('right')
        p.scene().addItem(pr1)
        ax_pr1.linkToView(pr1)
        pr1.setXLink(p)
        self.first_right_axis = ax_pr1

        pr2 = _pyqtgraph.ViewBox()
        ax_pr2 = _pyqtgraph.AxisItem('left')
        p.layout.addItem(ax_pr2, 2, 3)
        p.scene().addItem(pr2)
        ax_pr2.linkToView(pr2)
        pr2.setXLink(p)
        self.second_right_axis = ax_pr2

        def updateViews():
            pr1.setGeometry(p.vb.sceneBoundingRect())
            pr2.setGeometry(p.vb.sceneBoundingRect())
            pr1.linkedViewChanged(p.vb, pr1.XAxis)
            pr2.linkedViewChanged(p.vb, pr2.XAxis)

        updateViews()
        p.vb.sigResized.connect(updateViews)

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

    def calcCurveFit(self):
        """Calculate curve fit."""
        x, y = self.getSelectedXY()
        if x is None or y is None:
            return
        
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

    def calcIntegrals(self):
        """Calculate integrals."""
        sel = self.getSelectedCurve()
        if sel is None:
            return
        
        unit = sel['unit']
        
        x, y = self.getSelectedXY()
        if x is None or y is None:
            return
        
        if len(y) > 0:
            first_integral = _integrate.cumtrapz(x=x, y=y, initial=0)
            second_integral = _integrate.cumtrapz(
                x=x, y=first_integral, initial=0)

            self.ui.first_integral_le.setText(
                '{0:.5e}'.format(first_integral[-1]))
            self.ui.second_integral_le.setText(
                '{0:.5e}'.format(second_integral[-1]))

            if len(unit) > 0:
                first_integral_unit = unit + '.mm'
                second_integral_unit = unit + '.mm²'
            else:
                first_integral_unit = ''
                second_integral_unit = ''

            self.ui.first_integral_unit_la.setText(first_integral_unit)
            self.ui.second_integral_unit_la.setText(second_integral_unit)

            self.updatePlot()

            if len(first_integral_unit) > 0:
                label = 'First Integral [%s]' % first_integral_unit
            else:
                label = 'First Integral'
            self.first_right_axis.setLabel(label, color="#FFB266")
            self.first_right_axis.setStyle(showValues=True)
            self.first_right_axis.linkedView().addItem(
                _pyqtgraph.PlotCurveItem(
                    x, first_integral, pen=(255, 180, 100)))

            if len(second_integral_unit) > 0:
                label = 'Second Integral [%s]' % second_integral_unit
            else:
                label = 'Second Integral'
            self.second_right_axis.setLabel(label, color="#FF66B2")
            self.second_right_axis.setStyle(showValues=True)
            self.second_right_axis.linkedView().addItem(
                _pyqtgraph.PlotCurveItem(
                    x, second_integral, pen=(255, 100, 180)))

    def clearAll(self):
        """Clear all."""
        self.plot_label = ''
        self.scan_list = []
        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.scan_dict = {}
        self.xmin_line = None
        self.xmax_line = None
        self.current = None
        self.temperature = {}
        self.clearFit()
        self.clearIntegrals()
        self.clearGraph()
        self.ui.select_scan_cmb.setCurrentIndex(-1)
        self.ui.select_comp_cmb.setCurrentIndex(-1)       

    def clearFit(self):
        """Clear fit."""
        self.ui.fit_ta.clearContents()
        self.ui.fit_ta.setRowCount(0)
        self.ui.fitfunction_la.setText('')

    def clearIntegrals(self):
        """Clear integrals."""
        self.ui.first_integral_le.setText('')
        self.ui.second_integral_le.setText('')
        self.ui.first_integral_unit_la.setText('')
        self.ui.second_integral_unit_la.setText('')                   

    def clearGraph(self):
        """Clear plots."""
        self.ui.graph_pw.plotItem.curves.clear()
        self.ui.graph_pw.clear()
        for item in self.first_right_axis.linkedView().addedItems:
            self.first_right_axis.linkedView().removeItem(item)
        self.first_right_axis.setStyle(showValues=False)
        self.first_right_axis.setLabel('')
        for item in self.second_right_axis.linkedView().addedItems:
            self.second_right_axis.linkedView().removeItem(item)
        self.second_right_axis.setStyle(showValues=False)
        self.second_right_axis.setLabel('')
        self.graphx = []
        self.graphy = []
        self.graphz = []

    def configureGraph(self, nr_curves, plot_label):
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
        
        if nr_curves == 0:
            return

        for idx in range(nr_curves):
            self.graphx.append(
                self.ui.graph_pw.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(255, 0, 0)))

            self.graphy.append(
                self.ui.graph_pw.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(0, 255, 0)))

            self.graphz.append(
                self.ui.graph_pw.plotItem.plot(
                    _np.array([]),
                    _np.array([]),
                    pen=(0, 0, 255)))

        self.legend.addItem(self.graphx[0], 'X')
        self.legend.addItem(self.graphy[0], 'Y')
        self.legend.addItem(self.graphz[0], 'Z')
        self.ui.graph_pw.setLabel('bottom', 'Scan Position [mm]')
        self.ui.graph_pw.setLabel('left', plot_label)
        self.ui.graph_pw.showGrid(x=True, y=True)

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.update_plot_btn.clicked.connect(self.updatePlot)
        self.ui.select_scan_cmb.currentIndexChanged.connect(
            self.updateControls)
        self.ui.select_comp_cmb.currentIndexChanged.connect(
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
        self.ui.view_current_btn.clicked.connect(
            self.showCurrentDialog)
        self.ui.view_temperature_btn.clicked.connect(
            self.showTemperatureDialog)
        self.ui.integrals_btn.clicked.connect(self.calcIntegrals)

    def disableOffsetLed(self):
        """Disable offset led."""
        self.ui.offsetled_la.setEnabled(False)

    def fitFunctionChanged(self):
        """Hide or show polynomial fitting order and update dict value."""
        self.clearFit()
        self.updatePlot()
        func = self.ui.fitfunction_cmb.currentText()
        if func.lower() == 'polynomial':
            self.ui.polyorder_la.show()
            self.ui.polyorder_sb.show()
        else:
            self.ui.polyorder_la.hide()
            self.ui.polyorder_sb.hide()

        selected_idx = self.ui.select_scan_cmb.currentIndex()
        selected_comp = self.ui.select_comp_cmb.currentText().lower()
        if selected_idx == -1 or len(selected_comp) == 0:
            return
        else:
            self.scan_dict[selected_idx][selected_comp]['fit_function'] = func

    def getSelectedCurve(self):
        """Get selected curve."""
        selected_idx = self.ui.select_scan_cmb.currentIndex()
        selected_comp = self.ui.select_comp_cmb.currentText().lower()
        if selected_idx == -1 or len(selected_comp) == 0:
            return None
        else:
            return self.scan_dict[selected_idx][selected_comp]

    def getSelectedXY(self):
        """Get selected X, Y values."""
        sel = self.getSelectedCurve()
        if sel is None:
            return None, None

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
        return x, y

    def polyOrderChanged(self):
        """Update dict value."""
        od = self.ui.polyorder_sb.value()
        selected_idx = self.ui.select_scan_cmb.currentIndex()
        selected_comp = self.ui.select_comp_cmb.currentText().lower()
        if selected_idx == -1 or len(selected_comp) == 0:
            return
        else:
            self.scan_dict[selected_idx][selected_comp]['fit_polyorder'] = od

    def resetXLimits(self):
        """Reset X limits.."""
        selected_idx = self.ui.select_scan_cmb.currentIndex()
        selected_comp = self.ui.select_comp_cmb.currentText().lower()
        if selected_idx == -1 or len(selected_comp) == 0:
            return
        else:
            pos = self.scan_dict[selected_idx][selected_comp]['pos']
            xoff = self.scan_dict[selected_idx][selected_comp]['xoff']
            xmult = self.scan_dict[selected_idx][selected_comp]['xmult']

            if len(pos) > 0:
                xmin = (pos[0] + xoff)*xmult
                xmax = (pos[-1] + xoff)*xmult
            else:
                xmin = 0
                xmax = 0

            self.scan_dict[selected_idx][selected_comp]['xmin'] = xmin
            self.ui.xmin_sb.setValue(xmin)

            self.scan_dict[selected_idx][selected_comp]['xmax'] = xmax
            self.ui.xmax_sb.setValue(xmax)

            self.updatePlot()

    def setCurveLabels(self):
        """Set curve labels."""
        self.ui.select_scan_cmb.clear()

        p1 = set()
        p2 = set()
        p3 = set()
        for scan in self.scan_list:
            p1.update(scan.pos1)
            p2.update(scan.pos2)
            p3.update(scan.pos3)

        curve_labels = []
        for i, scan in enumerate(self.scan_list):
            idn = self.scan_id_list[i]
            pos = ''
            if scan.pos1.size == 1 and len(p1) != 1:
                pos = pos + ' pos1={0:.0f}mm'.format(scan.pos1[0])
            if scan.pos2.size == 1 and len(p2) != 1:
                pos = pos + ' pos2={0:.0f}mm'.format(scan.pos2[0])
            if scan.pos3.size == 1 and len(p3) != 1:
                pos = pos + ' pos3={0:.0f}mm'.format(scan.pos3[0])

            if len(pos) == 0:
                curve_labels.append('ID:{0:d}'.format(idn))
            else:
                curve_labels.append('ID:{0:d}'.format(idn) + pos)

        self.ui.select_scan_cmb.addItems(curve_labels)
        self.ui.select_scan_cmb.setCurrentIndex(-1)
        self.ui.select_comp_cmb.setCurrentIndex(-1)
        self.ui.select_scan_cmb.setEnabled(True)

    def show(self, scan_list, scan_id_list, plot_label=''):
        """Update data and show dialog."""
        try:
            self.clearAll()
            
            if scan_list is None or len(scan_list) == 0:
                msg = 'Invalid data list.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
                return
    
            self.scan_list = [d.copy() for d in scan_list]
            self.scan_id_list = scan_id_list
            self.plot_label = plot_label
            
            idx = 0
            for i in range(len(self.scan_list)):
                self.scan_dict[idx] = {}
                data = self.scan_list[i]
                idn = self.scan_id_list[i]

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

                self.scan_dict[idx]['x'] = {
                    'idn': idn,
                    'pos': pos,
                    'data': data.avgx,
                    'unit': data.unit,
                    'xoff': 0,
                    'yoff': 0,
                    'xmult': 1,
                    'ymult': 1,
                    'xmin': xmin,
                    'xmax': xmax,
                    'fit_function': 'Linear',
                    'fit_polyorder': None,
                    }

                self.scan_dict[idx]['y'] = {
                    'idn': idn,
                    'pos': pos,
                    'data': data.avgy,
                    'unit': data.unit,
                    'xoff': 0,
                    'yoff': 0,
                    'xmult': 1,
                    'ymult': 1,
                    'xmin': xmin,
                    'xmax': xmax,
                    'fit_function': 'Linear',
                    'fit_polyorder': None,
                    }

                self.scan_dict[idx]['z'] = {
                    'idn': idn,
                    'pos': pos,
                    'data': data.avgz,
                    'unit': data.unit,
                    'xoff': 0,
                    'yoff': 0,
                    'xmult': 1,
                    'ymult': 1,
                    'xmin': xmin,
                    'xmax': xmax,
                    'fit_function': 'Linear',
                    'fit_polyorder': None,
                    }

                idx = idx + 1

            columns = [
                'ID',
                'Current Setpoint [A]',
                'DCCT AVG [A]',
                'DCCT STD [A]',
                'PS AVG [A]',
                'PS STD [A]',
                ]
            current = []
            for i, scan in enumerate(self.scan_list):
                current.append([
                    self.scan_id_list[i],
                    scan.current_setpoint,
                    scan.dcct_current_avg,
                    scan.dcct_current_std,
                    scan.ps_current_avg,
                    scan.ps_current_std,
                    ])
            current = _np.array(current)
            self.current = _pd.DataFrame(current, columns=columns)

            if self.current.iloc[:, 1:].isnull().values.all():
                self.ui.view_current_btn.setEnabled(False)
            else:
                self.ui.view_current_btn.setEnabled(True)

            self.temperature = _measurement.get_temperature_values(
                self.scan_list)            
            if len(self.temperature) != 0:
                self.ui.view_temperature_btn.setEnabled(True)

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

    def showCurrentDialog(self):
        """Show dialog with current readings."""
        self.current_dialog.clear()
        if self.current is None:
            return

        try:
             self.current_dialog.show(self.current)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to open dialog.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def showTemperatureDialog(self):
        """Show dialog with temperature readings."""
        if len(self.temperature) == 0:
            return

        try:
            dfs = []
            for key, value in self.temperature.items():
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

    def updateControls(self):
        """Enable offset and fit group box and update offset values."""
        self.clearFit()
        self.clearIntegrals()
        self.updatePlot()
        selected_idx = self.ui.select_scan_cmb.currentIndex()
        selected_comp = self.ui.select_comp_cmb.currentText().lower()
        if selected_idx == -1 or len(selected_comp) == 0:
            self.ui.offset_gb.setEnabled(False)
            self.ui.xlimits_gb.setEnabled(False)
            self.ui.fit_gb.setEnabled(False)
            self.ui.integrals_gb.setEnabled(False)
            self.ui.offsetled_la.setEnabled(False)
            self.ui.xoff_sb.setValue(0)
            self.ui.yoff_sb.setValue(0)
            self.ui.xmult_sb.setValue(1)
            self.ui.ymult_sb.setValue(1)
            self.ui.fitfunction_cmb.setCurrentIndex(-1)
        else:
            self.ui.offset_gb.setEnabled(True)
            self.ui.xlimits_gb.setEnabled(True)
            self.ui.fit_gb.setEnabled(True)
            self.ui.integrals_gb.setEnabled(True)
            self.ui.offsetled_la.setEnabled(False)
            self.ui.xoff_sb.setValue(
                self.scan_dict[selected_idx][selected_comp]['xoff'])
            self.ui.yoff_sb.setValue(
                self.scan_dict[selected_idx][selected_comp]['yoff'])
            self.ui.xmult_sb.setValue(
                self.scan_dict[selected_idx][selected_comp]['xmult'])
            self.ui.ymult_sb.setValue(
                self.scan_dict[selected_idx][selected_comp]['ymult'])

            func = self.scan_dict[selected_idx][selected_comp]['fit_function']
            if func is None:
                self.ui.fitfunction_cmb.setCurrentIndex(-1)
            else:
                idx = self.ui.fitfunction_cmb.findText(func)
                self.ui.fitfunction_cmb.setCurrentIndex(idx)

            od = self.scan_dict[selected_idx][selected_comp]['fit_polyorder']
            if od is not None:
                self.ui.polyorder_sb.setValue(od)

            self.updatePlot()

    def updateOffset(self):
        """Update curve offset."""
        self.clearFit()
        self.updatePlot()
        selected_idx = self.ui.select_scan_cmb.currentIndex()
        selected_comp = self.ui.select_comp_cmb.currentText().lower()
        if selected_idx == -1 or len(selected_comp) == 0:
            self.ui.offsetled_la.setEnabled(False)
            return

        xoff = self.ui.xoff_sb.value()
        yoff = self.ui.yoff_sb.value()
        xmult = self.ui.xmult_sb.value()
        ymult = self.ui.ymult_sb.value()
        xmin = self.scan_dict[selected_idx][selected_comp]['xmin']
        xmax = self.scan_dict[selected_idx][selected_comp]['xmax']

        prev_xoff = self.scan_dict[selected_idx][selected_comp]['xoff']
        prev_xmult = self.scan_dict[selected_idx][selected_comp]['xmult']

        self.scan_dict[selected_idx][selected_comp]['xoff'] = xoff
        self.scan_dict[selected_idx][selected_comp]['yoff'] = yoff
        self.scan_dict[selected_idx][selected_comp]['xmult'] = xmult
        self.scan_dict[selected_idx][selected_comp]['ymult'] = ymult
        try:
            self.scan_dict[selected_idx][selected_comp]['xmin'] = (
                xmin*(xmult/prev_xmult) + (xoff - prev_xoff)*xmult)
            self.scan_dict[selected_idx][selected_comp]['xmax'] = (
                xmax*(xmult/prev_xmult) + (xoff - prev_xoff)*xmult)
        except Exception:
            pass

        self.updatePlot()
        self.ui.offsetled_la.setEnabled(True)

    def updatePlot(self):
        """Update plot."""
        try:
            self.clearGraph()
            self.updateXLimits()
            if self.ui.show_all_rb.isChecked():
                nr_curves = len(self.scan_list)
                scan_dict = self.scan_dict
                show_xlines = False
            elif self.ui.show_selected_scan_rb.isChecked():
                selected_idx = self.ui.select_scan_cmb.currentIndex()
                if selected_idx == -1:
                    return
                nr_curves = 1
                scan_dict = {selected_idx: self.scan_dict[selected_idx]}
                show_xlines = False
            else:
                selected_idx = self.ui.select_scan_cmb.currentIndex()
                selected_comp = self.ui.select_comp_cmb.currentText().lower()
                if selected_idx == -1 or len(selected_comp) == 0:
                    return
                nr_curves = 1
                scan_dict = {selected_idx: {
                    selected_comp: self.scan_dict[selected_idx][selected_comp]
                }}
                show_xlines = True
   
            self.configureGraph(nr_curves, self.plot_label)

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                x_count = 0
                y_count = 0
                z_count = 0
                for d in scan_dict.values():
                    for component, value in d.items():
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
                            x_count = x_count + 1
                        elif component == 'y':
                            self.graphy[y_count].setData(x, y)
                            y_count = y_count + 1
                        elif component == 'z':
                            self.graphz[z_count].setData(x, y)
                            z_count = z_count + 1

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
        selected_idx = self.ui.select_scan_cmb.currentIndex()
        selected_comp = self.ui.select_comp_cmb.currentText().lower()
        if selected_idx == -1 or len(selected_comp) == 0:
            return
        else:
            self.ui.xmin_sb.setValue(
                self.scan_dict[selected_idx][selected_comp]['xmin'])
            self.ui.xmax_sb.setValue(
                self.scan_dict[selected_idx][selected_comp]['xmax'])

    def updateXMax(self):
        """Update xmax value."""
        selected_idx = self.ui.select_scan_cmb.currentIndex()
        selected_comp = self.ui.select_comp_cmb.currentText().lower()
        if selected_idx == -1 or len(selected_comp) == 0:
            return
        else:
            self.scan_dict[selected_idx][selected_comp]['xmax'] = (
                self.ui.xmax_sb.value())

    def updateXMaxSpinBox(self):
        """Update xmax value."""
        self.ui.xmax_sb.setValue(self.xmax_line.pos()[0])

    def updateXMin(self):
        """Update xmin value."""
        selected_idx = self.ui.select_scan_cmb.currentIndex()
        selected_comp = self.ui.select_comp_cmb.currentText().lower()
        if selected_idx == -1 or len(selected_comp) == 0:
            return
        else:
            self.scan_dict[selected_idx][selected_comp]['xmin'] = (
                self.ui.xmin_sb.value())

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
