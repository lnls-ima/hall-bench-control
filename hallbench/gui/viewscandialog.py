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
from qtpy.QtWidgets import (
    QDialog as _QDialog,
    QVBoxLayout as _QVBoxLayout,
    QMessageBox as _QMessageBox,
    QTableWidgetItem as _QTableWidgetItem,
    )
from qtpy.QtCore import Qt as _Qt
import qtpy.uic as _uic
import pyqtgraph as _pyqtgraph

from hallbench.gui.auxiliarywidgets import (
    TableDialog as _TableDialog,
    CheckableComboBox as _CheckableComboBox,
    TemperatureTablePlotDialog as _TemperatureTablePlotDialog,
    )
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
        self.current_dialog = _TableDialog()
        self.current_dialog.setWindowTitle('Current Readings')

        # Create temperature dialog
        self.temperature_dialog = _TemperatureTablePlotDialog()
        self.temperature_dialog.setWindowTitle('Temperature Readings')

        # Create legend
        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.graph_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        # Add combo boxes
        self.select_scan_cmb = _CheckableComboBox()
        _layout = _QVBoxLayout()
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.addWidget(self.select_scan_cmb)
        self.ui.select_scan_wg.setLayout(_layout)

        self.connectSignalSlots()

    def accept(self):
        """Close dialog."""
        self.clear()
        self.closeDialogs()
        super().accept()

    def calcCurveFit(self):
        """Calculate curve fit."""
        try:
            selected_idx, selected_comp = self.getSelectedScanComponent()
            if selected_idx is None or selected_comp is None:
                return
            
            x, y = self.getXY(selected_idx, selected_comp)
    
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
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def calcIntegrals(self):
        """Calculate integrals."""
        try:
            selected_idx, _ = self.getSelectedScanComponent()
            if selected_idx is None:
                return
    
            unit = self.scan_dict[selected_idx]['unit']
            if len(unit) > 0:
                first_integral_unit = ' [' + unit + '.mm' + ']'
                second_integral_unit = ' [' + unit + '.mm²' + ']'
            else:
                first_integral_unit = ''
                second_integral_unit = ''
            self.ui.first_integral_la.setText(
                'First Integral{0:s}:'.format(first_integral_unit))
            self.ui.second_integral_la.setText(
                'Second Integral{0:s}:'.format(second_integral_unit))
    
            x, y = self.getXY(selected_idx, 'x')
            if x is None or y is None or len(y) == 0:
                self.clearIntegrals()
                return
    
            first_integral = _integrate.cumtrapz(x=x, y=y, initial=0)
            second_integral = _integrate.cumtrapz(
                x=x, y=first_integral, initial=0)
    
            self.ui.first_integral_x_le.setText(
                '{0:.6g}'.format(first_integral[-1]))
            self.ui.second_integral_x_le.setText(
                '{0:.6g}'.format(second_integral[-1]))
    
            x, y = self.getXY(selected_idx, 'y')
            first_integral = _integrate.cumtrapz(x=x, y=y, initial=0)
            second_integral = _integrate.cumtrapz(
                x=x, y=first_integral, initial=0)

            self.ui.first_integral_y_le.setText(
                '{0:.6g}'.format(first_integral[-1]))
            self.ui.second_integral_y_le.setText(
                '{0:.6g}'.format(second_integral[-1]))
    
            x, y = self.getXY(selected_idx, 'z')
            first_integral = _integrate.cumtrapz(x=x, y=y, initial=0)
            second_integral = _integrate.cumtrapz(
                x=x, y=first_integral, initial=0)

            self.ui.first_integral_z_le.setText(
                '{0:.6g}'.format(first_integral[-1]))
            self.ui.second_integral_z_le.setText(
                '{0:.6g}'.format(second_integral[-1]))
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def clear(self):
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
        self.select_scan_cmb.clear()

    def clearFit(self):
        """Clear fit."""
        self.ui.fit_ta.clearContents()
        self.ui.fit_ta.setRowCount(0)
        self.ui.fitfunction_la.setText('')

    def clearIntegrals(self):
        """Clear integrals."""
        self.ui.first_integral_x_le.setText('')
        self.ui.first_integral_y_le.setText('')
        self.ui.first_integral_z_le.setText('')
        self.ui.second_integral_x_le.setText('')
        self.ui.second_integral_y_le.setText('')
        self.ui.second_integral_z_le.setText('')
        self.ui.first_integral_la.setText('First Integral:')
        self.ui.second_integral_la.setText('Second Integral:')

    def clearGraph(self):
        """Clear plots."""
        self.ui.graph_pw.plotItem.curves.clear()
        self.ui.graph_pw.clear()
        self.graphx = []
        self.graphy = []
        self.graphz = []

    def closeDialogs(self):
        """Close dialogs."""
        try:
            self.current_dialog.accept()
            self.temperature_dialog.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.clear()
            self.closeDialogs()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

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
        self.select_scan_cmb.activated.connect(self.updateControlsAndPlot)
        self.ui.x_chb.stateChanged.connect(self.updateControlsAndPlot)
        self.ui.y_chb.stateChanged.connect(self.updateControlsAndPlot)
        self.ui.z_chb.stateChanged.connect(self.updateControlsAndPlot)
        self.ui.select_all_btn.clicked.connect(
            lambda: self.setSelectionAll(True))
        self.ui.clear_all_btn.clicked.connect(
            lambda: self.setSelectionAll(False))
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

        selected_idx, selected_comp = self.getSelectedScanComponent()
        if selected_idx is None or selected_comp is None:
            return
        self.scan_dict[selected_idx][selected_comp]['fit_function'] = func

    def getSelectedScanComponent(self):
        """Get selected scan and component."""
        try:
            selected_index, selected_comps = self.getSelectedScansComponents()
            if len(selected_index) == 1:
                selected_idx = selected_index[0]
            else:
                selected_idx = None
    
            if len(selected_comps) == 1:
                selected_comp = selected_comps[0]
            else:
                selected_comp = None
    
            return selected_idx, selected_comp
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return None, None

    def getSelectedScansComponents(self):
        """Get all selected scans and components."""
        try:
            selected_idx = self.select_scan_cmb.checkedIndexes()
    
            selected_comp = []
            if self.ui.x_chb.isChecked():
                selected_comp.append('x')
            if self.ui.y_chb.isChecked():
                selected_comp.append('y')
            if self.ui.z_chb.isChecked():
                selected_comp.append('z')
    
            return selected_idx, selected_comp
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return None, None

    def getXY(self, selected_idx, selected_comp):
        """Get X, Y values."""
        sel = self.scan_dict[selected_idx][selected_comp]

        pos = sel['pos']
        data = sel['data']
        xmin = self.ui.xmin_sb.value()
        xmax = self.ui.xmax_sb.value()

        x = _np.array(pos)
        y = _np.array(data)

        y = y[(x >= xmin) & (x <= xmax)]
        x = x[(x >= xmin) & (x <= xmax)]
        return x, y

    def polyOrderChanged(self):
        """Update dict value."""
        selected_idx, selected_comp = self.getSelectedScanComponent()
        if selected_idx is None or selected_comp is None:
            return

        od = self.ui.polyorder_sb.value()
        self.scan_dict[selected_idx][selected_comp]['fit_polyorder'] = od

    def resetXLimits(self):
        """Reset X limits.."""
        selected_idx, _ = self.getSelectedScanComponent()
        if selected_idx is None:
            return

        pos = self.scan_dict[selected_idx]['x']['pos']

        if len(pos) > 0:
            xmin = pos[0]
            xmax = pos[-1]
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
        self.blockSignals(True)
        self.select_scan_cmb.clear()

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

        for index, element in enumerate(curve_labels):
            self.select_scan_cmb.addItem(element)
            item = self.select_scan_cmb.model().item(index, 0)
            item.setCheckState(_Qt.Checked)

        self.ui.x_chb.setChecked(True)
        self.ui.y_chb.setChecked(True)
        self.ui.z_chb.setChecked(True)

        self.select_scan_cmb.setCurrentIndex(-1)
        self.blockSignals(False)

    def setSelectionAll(self, checked):
        """Set selection all."""
        self.blockSignals(True)
        if checked:
            state = _Qt.Checked
        else:
            state = _Qt.Unchecked

        for idx in range(self.select_scan_cmb.count()):
            item = self.select_scan_cmb.model().item(idx, 0)
            item.setCheckState(state)

        self.ui.x_chb.setChecked(checked)
        self.ui.y_chb.setChecked(checked)
        self.ui.z_chb.setChecked(checked)

        self.blockSignals(False)
        self.updateControlsAndPlot()

    def show(self, scan_list, scan_id_list, plot_label=''):
        """Update data and show dialog."""
        try:
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
                self.scan_dict[idx] = {
                    'xmin': xmin,
                    'xmax': xmax,
                    'unit': data.unit}

                self.scan_dict[idx]['x'] = {
                    'idn': idn,
                    'pos': pos,
                    'data': data.avgx,
                    'fit_function': 'Gaussian',
                    'fit_polyorder': None,
                    }

                self.scan_dict[idx]['y'] = {
                    'idn': idn,
                    'pos': pos,
                    'data': data.avgy,
                    'fit_function': 'Gaussian',
                    'fit_polyorder': None,
                    }

                self.scan_dict[idx]['z'] = {
                    'idn': idn,
                    'pos': pos,
                    'data': data.avgz,
                    'fit_function': 'Gaussian',
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

            self.setCurveLabels()
            self.ui.polyorder_la.hide()
            self.ui.polyorder_sb.hide()
            self.updateControlsAndPlot()
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
        """Enable or disable group boxes."""
        self.clearFit()

        selected_idx, selected_comp = self.getSelectedScanComponent()
        if selected_idx is not None and selected_comp is not None:
            self.ui.xlimits_gb.setEnabled(True)
            self.ui.integrals_gb.setEnabled(True)
            self.ui.fit_gb.setEnabled(True)
            func = self.scan_dict[selected_idx][selected_comp]['fit_function']
            if func is None:
                self.ui.fitfunction_cmb.setCurrentIndex(-1)
            else:
                idx = self.ui.fitfunction_cmb.findText(func)
                self.ui.fitfunction_cmb.setCurrentIndex(idx)

            od = self.scan_dict[selected_idx][selected_comp]['fit_polyorder']
            if od is not None:
                self.ui.polyorder_sb.setValue(od)
        else:
            self.ui.fit_gb.setEnabled(False)
            self.ui.fitfunction_cmb.setCurrentIndex(-1)
            if selected_idx is not None:
                self.ui.xlimits_gb.setEnabled(True)
                self.ui.integrals_gb.setEnabled(True)
                self.ui.xlimits_gb.setEnabled(True)
            else:
                self.clearIntegrals()
                self.ui.xlimits_gb.setEnabled(False)
                self.ui.integrals_gb.setEnabled(False)
                self.ui.xlimits_gb.setEnabled(False)

    def updateControlsAndPlot(self):
        """Update controls and plot."""
        self.updateControls()
        self.updatePlot()

    def updatePlot(self):
        """Update plot."""
        try:
            self.clearGraph()
            self.updateXLimits()

            selected_idx, selected_comp = self.getSelectedScansComponents()

            if len(selected_idx) == 0:
                show_xlines = False
                return

            if len(selected_idx) > 1:
                show_xlines = False
            else:
                show_xlines = True

            nr_curves = len(selected_idx)
            self.configureGraph(nr_curves, self.plot_label)

            scan_dict = {}
            for idx in selected_idx:
                scan_dict[idx] = {}
                for comp in selected_comp:
                    scan_dict[idx][comp] = self.scan_dict[idx][comp]

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                x_count = 0
                y_count = 0
                z_count = 0
                for d in scan_dict.values():
                    for component, value in d.items():
                        pos = value['pos']
                        data = value['data']
                        x = pos
                        y = data
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
            return

    def updateXLimits(self):
        """Update xmin and xmax values."""
        selected_idx, _ = self.getSelectedScanComponent()
        if selected_idx is None:
            return

        self.ui.xmin_sb.setValue(
            self.scan_dict[selected_idx]['xmin'])
        self.ui.xmax_sb.setValue(
            self.scan_dict[selected_idx]['xmax'])

    def updateXMax(self):
        """Update xmax value."""
        selected_idx, _ = self.getSelectedScanComponent()
        if selected_idx is None:
            return

        self.scan_dict[selected_idx]['xmax'] = self.ui.xmax_sb.value()

    def updateXMaxSpinBox(self):
        """Update xmax value."""
        self.ui.xmax_sb.setValue(self.xmax_line.pos()[0])

    def updateXMin(self):
        """Update xmin value."""
        selected_idx, _ = self.getSelectedScanComponent()
        if selected_idx is None:
            return

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
