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
    QApplication as _QApplication,
    QTableWidgetItem as _QTableWidgetItem,
    )
from qtpy.QtGui import QFont as _QFont
from qtpy.QtCore import Qt as _Qt
import qtpy.uic as _uic
import pyqtgraph as _pyqtgraph

from hallbench.gui.auxiliarywidgets import (
    CheckableComboBox as _CheckableComboBox,
    TemperatureTablePlotDialog as _TemperatureTablePlotDialog,
    IntegralsTablePlotDialog as _IntegralsTablePlotDialog,
    CurrentTablePlotDialog as _CurrentTablePlotDialog,
    )
from hallbench.gui import utils as _utils
from hallbench.data import measurement as _measurement


TO_PRINT = True


class ViewScanDialog(_QDialog):
    """View data dialog class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        uifile = _utils.get_ui_file(self)
        self.ui = _uic.loadUi(uifile, self)

        self.scan_type = None
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
        self.current_dialog = _CurrentTablePlotDialog()

        # Create temperature dialog
        self.temperature_dialog = _TemperatureTablePlotDialog()

        # Create integrals dialog
        self.integrals_dialog = _IntegralsTablePlotDialog()

        # Create legend
        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.pw_graph.graphicsItem())
        self.legend.setAutoFillBackground(1)
        self.legend_items = []

        # Add combo boxes
        self.cmb_select_scan = _CheckableComboBox()
        _layout = _QVBoxLayout()
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.addWidget(self.cmb_select_scan)
        self.ui.wg_select_scan.setLayout(_layout)

        self.connect_signal_slots()

    def accept(self):
        """Close dialog."""
        self.clear()
        self.close_dialogs()
        super().accept()

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.clear()
            self.close_dialogs()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def calc_curve_fit(self):
        """Calculate curve fit."""
        try:
            selected_idx, selected_comp = self.get_selected_scan_component()
            if selected_idx is None or selected_comp is None:
                return

            x, y = self.get_x_y(selected_idx, selected_comp)

            func = self.ui.cmb_fitfunction.currentText()
            if func.lower() == 'linear':
                xfit, yfit, param, label = _linear_fit(x, y)
            elif func.lower() == 'polynomial':
                order = self.ui.sb_polyorder.value()
                xfit, yfit, param, label = _polynomial_fit(x, y, order)
            elif func.lower() == 'gaussian':
                xfit, yfit, param, label = _gaussian_fit(x, y)
            else:
                xfit = []
                yfit = []
                param = {}
                label = ''

            self.ui.la_fitfunction.setText(label)
            self.ui.tbl_fit.clearContents()
            self.ui.tbl_fit.setRowCount(len(param))
            rcount = 0
            for key, value in param.items():
                self.ui.tbl_fit.setItem(
                    rcount, 0, _QTableWidgetItem(str(key)))
                self.ui.tbl_fit.setItem(
                    rcount, 1, _QTableWidgetItem(str(value)))
                rcount = rcount + 1

            self.update_plot()
            self.ui.pw_graph.plotItem.plot(xfit, yfit, pen=(0, 0, 0))

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def calc_integrals(self):
        """Calculate integrals."""
        try:
            selected_idx, _ = self.get_selected_scan_component()
            if selected_idx is None:
                return

            x, y = self.get_x_y(selected_idx, 'x')
            if x is None or y is None or len(y) == 0:
                self.clear_integrals()
                return

            if self.scan_type == 'field':
                if self.ui.cmb_first_integral_unit.currentIndex() == 1:
                    fmult = 1e3
                else:
                    fmult = 1e-3

                if self.ui.cmb_second_integral_unit.currentIndex() == 1:
                    smult = 1e-2
                else:
                    smult = 1e-6

            else:
                fmult = 1e-3
                smult = 1e-6
                
            first_integral = _integrate.cumtrapz(x=x, y=y, initial=0)
            second_integral = _integrate.cumtrapz(
                x=x, y=first_integral, initial=0)

            fint = first_integral[-1]*fmult
            sint = second_integral[-1]*smult

            self.ui.le_first_integral_x.setText('{0:.6g}'.format(fint))
            self.ui.le_second_integral_x.setText('{0:.6g}'.format(sint))

            x, y = self.get_x_y(selected_idx, 'y')
            first_integral = _integrate.cumtrapz(x=x, y=y, initial=0)
            second_integral = _integrate.cumtrapz(
                x=x, y=first_integral, initial=0)

            fint = first_integral[-1]*fmult
            sint = second_integral[-1]*smult

            self.ui.le_first_integral_y.setText('{0:.6g}'.format(fint))
            self.ui.le_second_integral_y.setText('{0:.6g}'.format(sint))

            x, y = self.get_x_y(selected_idx, 'z')
            first_integral = _integrate.cumtrapz(x=x, y=y, initial=0)
            second_integral = _integrate.cumtrapz(
                x=x, y=first_integral, initial=0)

            fint = first_integral[-1]*fmult
            sint = second_integral[-1]*smult

            self.ui.le_first_integral_z.setText('{0:.6g}'.format(fint))
            self.ui.le_second_integral_z.setText('{0:.6g}'.format(sint))

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def clear(self):
        """Clear all."""
        self.scan_type = None
        self.scan_list = []
        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.scan_dict = {}
        self.xmin_line = None
        self.xmax_line = None
        self.current = None
        self.temperature = {}
        self.clear_fit()
        self.clear_integrals()
        self.clear_graph()
        self.cmb_select_scan.clear()

    def clear_fit(self):
        """Clear fit."""
        self.ui.tbl_fit.clearContents()
        self.ui.tbl_fit.setRowCount(0)
        self.ui.la_fitfunction.setText('')

    def clear_integrals(self):
        """Clear integrals."""
        self.ui.le_first_integral_x.setText('')
        self.ui.le_first_integral_y.setText('')
        self.ui.le_first_integral_z.setText('')
        self.ui.le_second_integral_x.setText('')
        self.ui.le_second_integral_y.setText('')
        self.ui.le_second_integral_z.setText('')

    def clear_graph(self):
        """Clear plots."""
        for item in self.legend_items:
            self.legend.removeItem(item)
        self.legend_items = []
        self.ui.pw_graph.plotItem.curves.clear()
        self.ui.pw_graph.clear()
        self.graphx = []
        self.graphy = []
        self.graphz = []

    def close_dialogs(self):
        """Close dialogs."""
        try:
            self.current_dialog.accept()
            self.temperature_dialog.accept()
            self.integrals_dialog.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def configure_graph(self, nr_curves, idn_list=None):
        """Configure graph.

        Args:
            nr_curves (int): number of curves to plot.
        """
        self.clear_graph()

        if nr_curves == 0:
            return
        
        colorx = (255, 0, 0)
        colory = (0, 255, 0)
        colorz = (0, 0, 255)
        
        if TO_PRINT:
            width = 3
            label_style = {'font-size': '20px'}
        else:
            width = 1
            label_style = {}

        colors = _utils.COLOR_LIST

        if idn_list is not None and len(idn_list) > 0:
            if nr_curves > len(colors):
                colors = [(0, 0, 0)]*nr_curves
            
            for idx in range(nr_curves):
                pen = _pyqtgraph.mkPen(color=colors[idx], width=width)
                plot_data_item = self.ui.pw_graph.plotItem.plot(
                    _np.array([]),
                    _np.array([]))
                plot_data_item.setPen(pen)
                self.graphx.append(plot_data_item)
    
                plot_data_item = self.ui.pw_graph.plotItem.plot(
                    _np.array([]),
                    _np.array([]))
                plot_data_item.setPen(pen)
                self.graphy.append(plot_data_item) 

                plot_data_item = self.ui.pw_graph.plotItem.plot(
                    _np.array([]),
                    _np.array([]))
                plot_data_item.setPen(pen)
                self.graphz.append(plot_data_item)
    
                legend_item = 'ID:{0:d}'.format(idn_list[idx])
                self.legend_items.append(legend_item)
                self.legend.addItem(self.graphx[idx], legend_item)
        
        else:
            for idx in range(nr_curves):
                pen = _pyqtgraph.mkPen(color=colorx, width=width)
                plot_data_item = self.ui.pw_graph.plotItem.plot(
                    _np.array([]),
                    _np.array([]))
                plot_data_item.setPen(pen)
                self.graphx.append(plot_data_item)
    
                pen = _pyqtgraph.mkPen(color=colory, width=width)
                plot_data_item = self.ui.pw_graph.plotItem.plot(
                    _np.array([]),
                    _np.array([]))
                plot_data_item.setPen(pen)
                self.graphy.append(plot_data_item)
    
                pen = _pyqtgraph.mkPen(color=colorz, width=width)
                plot_data_item = self.ui.pw_graph.plotItem.plot(
                    _np.array([]),
                    _np.array([]))
                plot_data_item.setPen(pen)
                self.graphz.append(plot_data_item)
    
            self.legend_items = ['X', 'Y', 'Z']
            self.legend.addItem(self.graphx[0], self.legend_items[0])
            self.legend.addItem(self.graphy[0], self.legend_items[1])
            self.legend.addItem(self.graphz[0], self.legend_items[2])
        
        self.ui.pw_graph.setLabel(
            'bottom', text='Scan Position', **label_style)
        
        if self.scan_type == 'field':
            self.ui.pw_graph.setLabel(
                'left', text='Field [T]', **label_style)
        else:
            self.ui.pw_graph.setLabel(
                'left', text='Voltage [V]', **label_style)
        
        self.ui.pw_graph.showGrid(x=True, y=True)

        if TO_PRINT:        
            font = _QFont()
            font.setPixelSize(20)
            self.ui.pw_graph.getAxis('bottom').tickFont = font
            self.ui.pw_graph.getAxis('bottom').setStyle(
                tickTextOffset = 20)
            self.ui.pw_graph.getAxis('left').tickFont = font
            self.ui.pw_graph.getAxis('left').setStyle(
                tickTextOffset = 20)    

    def connect_signal_slots(self):
        """Create signal/slot connections."""
        self.ui.pbt_update_plot.clicked.connect(self.update_plot)
        self.cmb_select_scan.activated.connect(self.update_controls_and_plot)
        self.ui.chb_x.stateChanged.connect(self.update_controls_and_plot)
        self.ui.chb_y.stateChanged.connect(self.update_controls_and_plot)
        self.ui.chb_z.stateChanged.connect(self.update_controls_and_plot)
        self.ui.pbt_select_all.clicked.connect(
            lambda: self.set_selection_all(True))
        self.ui.pbt_clear_all.clicked.connect(
            lambda: self.set_selection_all(False))
        self.ui.sbd_xmin.valueChanged.connect(self.update_xmin)
        self.ui.sbd_xmax.valueChanged.connect(self.update_xmax)
        self.ui.cmb_fitfunction.currentIndexChanged.connect(
            self.fit_function_changed)
        self.ui.cmb_first_integral_unit.currentIndexChanged.connect(
            self.calc_integrals)
        self.ui.cmb_second_integral_unit.currentIndexChanged.connect(
            self.calc_integrals)
        self.ui.sb_polyorder.valueChanged.connect(self.poly_order_changed)
        self.ui.pbt_fit.clicked.connect(self.calc_curve_fit)
        self.ui.pbt_resetlim.clicked.connect(self.reset_xlimits)
        self.ui.pbt_view_current.clicked.connect(
            self.show_current_dialog)
        self.ui.pbt_view_temperature.clicked.connect(
            self.show_temperature_dialog)
        self.ui.pbt_view_integrals.clicked.connect(
            self.show_integrals_dialog)
        self.ui.pbt_calc_integrals.clicked.connect(self.calc_integrals)
        self.ui.tbt_copy_integrals.clicked.connect(self.copy_integrals)

    def copy_integrals(self):
        """Copy integrals data to clipboard."""
        try:
            field_integrals = [
                self.ui.le_first_integral_x.text(),
                self.ui.le_first_integral_y.text(),
                self.ui.le_first_integral_z.text(),
                self.ui.le_second_integral_x.text(),
                self.ui.le_second_integral_y.text(),
                self.ui.le_second_integral_z.text(),
                ]

            text = "\n".join(field_integrals)
            _QApplication.clipboard().setText(text)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def fit_function_changed(self):
        """Hide or show polynomial fitting order and update dict value."""
        self.clear_fit()
        self.update_plot()
        func = self.ui.cmb_fitfunction.currentText()
        if func.lower() == 'polynomial':
            self.ui.la_polyorder.show()
            self.ui.sb_polyorder.show()
        else:
            self.ui.la_polyorder.hide()
            self.ui.sb_polyorder.hide()

        selected_idx, selected_comp = self.get_selected_scan_component()
        if selected_idx is None or selected_comp is None:
            return
        self.scan_dict[selected_idx][selected_comp]['fit_function'] = func

    def get_selected_scan_component(self):
        """Get selected scan and component."""
        try:
            selected_index, selected_comps = (
                self.get_selected_scans_components())
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

    def get_selected_scans_components(self):
        """Get all selected scans and components."""
        try:
            selected_idx = self.cmb_select_scan.checked_indexes()

            selected_comp = []
            if self.ui.chb_x.isChecked():
                selected_comp.append('x')
            if self.ui.chb_y.isChecked():
                selected_comp.append('y')
            if self.ui.chb_z.isChecked():
                selected_comp.append('z')

            return selected_idx, selected_comp

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return None, None

    def get_x_y(self, selected_idx, selected_comp):
        """Get X, Y values."""
        sel = self.scan_dict[selected_idx][selected_comp]

        pos = sel['pos']
        data = sel['data']
        xmin = self.ui.sbd_xmin.value()
        xmax = self.ui.sbd_xmax.value()

        x = _np.array(pos)
        y = _np.array(data)

        y = y[(x >= xmin) & (x <= xmax)]
        x = x[(x >= xmin) & (x <= xmax)]
        return x, y

    def poly_order_changed(self):
        """Update dict value."""
        selected_idx, selected_comp = self.get_selected_scan_component()
        if selected_idx is None or selected_comp is None:
            return

        od = self.ui.sb_polyorder.value()
        self.scan_dict[selected_idx][selected_comp]['fit_polyorder'] = od

    def reset_xlimits(self):
        """Reset X limits.."""
        selected_idx, _ = self.get_selected_scan_component()
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
        self.ui.sbd_xmin.setValue(xmin)

        self.scan_dict[selected_idx]['xmax'] = xmax
        self.ui.sbd_xmax.setValue(xmax)

        self.update_plot()

    def set_curve_labels(self):
        """Set curve labels."""
        self.blockSignals(True)
        self.cmb_select_scan.clear()

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
            self.cmb_select_scan.addItem(element)
            item = self.cmb_select_scan.model().item(index, 0)
            item.setCheckState(_Qt.Checked)

        self.ui.chb_x.setChecked(True)
        self.ui.chb_y.setChecked(True)
        self.ui.chb_z.setChecked(True)

        self.cmb_select_scan.setCurrentIndex(-1)
        self.blockSignals(False)

    def set_selection_all(self, checked):
        """Set selection all."""
        self.blockSignals(True)
        if checked:
            state = _Qt.Checked
        else:
            state = _Qt.Unchecked

        for idx in range(self.cmb_select_scan.count()):
            item = self.cmb_select_scan.model().item(idx, 0)
            item.setCheckState(state)

        self.ui.chb_x.setChecked(checked)
        self.ui.chb_y.setChecked(checked)
        self.ui.chb_z.setChecked(checked)

        self.blockSignals(False)
        self.update_controls_and_plot()

    def show(self, scan_list, scan_type='field'):
        """Update data and show dialog."""
        try:
            if scan_list is None or len(scan_list) == 0:
                msg = 'Invalid data list.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
                return

            self.scan_list = [d.copy() for d in scan_list]
            self.scan_id_list = [d.idn for d in scan_list]
            self.scan_type = scan_type

            self.ui.cmb_first_integral_unit.clear()
            self.ui.cmb_second_integral_unit.clear()
            if self.scan_type == 'field':
                self.ui.cmb_first_integral_unit.addItem('T.m')
                self.ui.cmb_first_integral_unit.addItem('G.cm')
                self.ui.cmb_first_integral_unit.setCurrentIndex(1)
                
                self.ui.cmb_second_integral_unit.addItem('T.m2')
                self.ui.cmb_second_integral_unit.addItem('G.m2')
                self.ui.cmb_second_integral_unit.setCurrentIndex(1)
            else:
                self.ui.cmb_first_integral_unit.addItem('V.m')          
                self.ui.cmb_second_integral_unit.addItem('V.m2')

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
                
                if self.scan_type == 'field':
                    x = data.bx if len(data.bx) != 0 else _np.zeros(len(pos))
                    y = data.by if len(data.by) != 0 else _np.zeros(len(pos))
                    z = data.bz if len(data.bz) != 0 else _np.zeros(len(pos))
                else:
                    x = data.vx if len(data.vx) != 0 else _np.zeros(len(pos))
                    y = data.vy if len(data.vy) != 0 else _np.zeros(len(pos))
                    z = data.vz if len(data.vz) != 0 else _np.zeros(len(pos))                      
                
                xmin = pos[0]
                xmax = pos[-1]
                self.scan_dict[idx] = {
                    'xmin': xmin,
                    'xmax': xmax}

                self.scan_dict[idx]['x'] = {
                    'idn': idn,
                    'pos': pos,
                    'data': x,
                    'fit_function': 'Gaussian',
                    'fit_polyorder': None,
                    }

                self.scan_dict[idx]['y'] = {
                    'idn': idn,
                    'pos': pos,
                    'data': y,
                    'fit_function': 'Gaussian',
                    'fit_polyorder': None,
                    }

                self.scan_dict[idx]['z'] = {
                    'idn': idn,
                    'pos': pos,
                    'data': z,
                    'fit_function': 'Gaussian',
                    'fit_polyorder': None,
                    }

                idx = idx + 1

            columns = [
                'Current Setpoint [A]',
                'DCCT AVG [A]',
                'DCCT STD [A]',
                'PS AVG [A]',
                'PS STD [A]',
                ]
            current = []
            for i, scan in enumerate(self.scan_list):
                current.append([
                    scan.current_setpoint,
                    scan.dcct_current_avg,
                    scan.dcct_current_std,
                    scan.ps_current_avg,
                    scan.ps_current_std,
                    ])
            current = _np.array(current)
            self.current = _pd.DataFrame(
                current, index=self.scan_id_list, columns=columns)

            if self.current.iloc[:, 1:].isnull().values.all():
                self.ui.pbt_view_current.setEnabled(False)
            else:
                self.ui.pbt_view_current.setEnabled(True)

            self.temperature = _measurement.get_temperature_values(
                self.scan_list)
            if len(self.temperature) != 0:
                self.ui.pbt_view_temperature.setEnabled(True)

            self.set_curve_labels()
            self.ui.la_polyorder.hide()
            self.ui.sb_polyorder.hide()
            self.update_controls_and_plot()
            super().show()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to show dialog.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def show_current_dialog(self):
        """Show dialog with current readings."""
        self.current_dialog.clear()
        if self.current is None:
            return

        try:
            timestamp = self.current.index.values.tolist()
            readings = {}
            for col in self.current.columns:
                readings[col] = self.current[col].values.tolist()

            self.current_dialog.show(timestamp, readings)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to open dialog.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def show_integrals_dialog(self):
        """Show dialog with integrals."""
        try:
            if self.scan_type == 'field':
                if self.ui.cmb_first_integral_unit.currentIndex() == 1:
                    fmult = 1e3
                    ylabel = 'Field Integral [G.cm]'
                    
                else:
                    fmult = 1e-3
                    ylabel = 'Field Integral [T.m]'
            
            else:
                fmult = 1e-3
                ylabel = 'Voltage Integral [V.m]'

            idn_list = []                       
            
            values = {}
            values['IntegralX'] = []
            values['IntegralY'] = []
            values['IntegralZ'] = []
            
            for idx in self.scan_dict.keys():
                idn_list.append(self.scan_dict[idx]['x']['idn'])
                
                x = self.scan_dict[idx]['x']['pos']
                y = self.scan_dict[idx]['x']['data']
                first_integral = _integrate.cumtrapz(x=x, y=y, initial=0)
                fint = first_integral[-1]*fmult
                values['IntegralX'].append(fint)         
          
                x = self.scan_dict[idx]['y']['pos']
                y = self.scan_dict[idx]['y']['data']
                first_integral = _integrate.cumtrapz(x=x, y=y, initial=0)
                fint = first_integral[-1]*fmult
                values['IntegralY'].append(fint)  

                x = self.scan_dict[idx]['z']['pos']
                y = self.scan_dict[idx]['z']['data']
                first_integral = _integrate.cumtrapz(x=x, y=y, initial=0)
                fint = first_integral[-1]*fmult
                values['IntegralZ'].append(fint)  
            
            self.integrals_dialog.set_ylabel(ylabel)
            self.integrals_dialog.show(idn_list, values)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to open dialog.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def show_temperature_dialog(self):
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

    def update_controls(self):
        """Enable or disable group boxes."""
        self.clear_fit()

        selected_idx, selected_comp = self.get_selected_scan_component()
        if selected_idx is not None and selected_comp is not None:
            self.ui.gb_xlimits.setEnabled(True)
            self.ui.gb_integrals.setEnabled(True)
            self.ui.gb_fit.setEnabled(True)
            func = self.scan_dict[selected_idx][selected_comp]['fit_function']
            if func is None:
                self.ui.cmb_fitfunction.setCurrentIndex(-1)
            else:
                idx = self.ui.cmb_fitfunction.findText(func)
                self.ui.cmb_fitfunction.setCurrentIndex(idx)

            od = self.scan_dict[selected_idx][selected_comp]['fit_polyorder']
            if od is not None:
                self.ui.sb_polyorder.setValue(od)
            self.calc_integrals()
        else:
            self.ui.gb_fit.setEnabled(False)
            self.ui.cmb_fitfunction.setCurrentIndex(-1)
            if selected_idx is not None:
                self.ui.gb_xlimits.setEnabled(True)
                self.ui.gb_integrals.setEnabled(True)
                self.ui.gb_xlimits.setEnabled(True)
            else:
                self.clear_integrals()
                self.ui.gb_xlimits.setEnabled(False)
                self.ui.gb_integrals.setEnabled(False)
                self.ui.gb_xlimits.setEnabled(False)
            self.calc_integrals()

    def update_controls_and_plot(self):
        """Update controls and plot."""
        self.update_controls()
        self.update_plot()

    def update_plot(self):
        """Update plot."""
        try:
            self.clear_graph()
            self.update_xlimits()

            selected_idx, selected_comp = self.get_selected_scans_components()

            if len(selected_idx) == 0:
                show_xlines = False
                return

            if len(selected_idx) > 1:
                show_xlines = False
            else:
                show_xlines = True
            
            scan_dict = {}
            idn_list = []
            for idx in selected_idx:
                scan_dict[idx] = {}
                for comp in selected_comp:
                    scan_dict[idx][comp] = self.scan_dict[idx][comp]
                    idn_list.append(self.scan_dict[idx][comp]['idn'])

            if len(selected_comp) > 1 or len(selected_idx) == 1:
                self.configure_graph(len(selected_idx))
            else: 
                self.configure_graph(len(selected_idx), idn_list=idn_list)
                
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
                xmin = self.ui.sbd_xmin.value()
                xmax = self.ui.sbd_xmax.value()

                self.xmin_line = _pyqtgraph.InfiniteLine(
                    xmin, pen=(0, 0, 0), movable=True)
                self.ui.pw_graph.addItem(self.xmin_line)
                self.xmin_line.sigPositionChangeFinished.connect(
                    self.update_xmin_spin_box)

                self.xmax_line = _pyqtgraph.InfiniteLine(
                    xmax, pen=(0, 0, 0), movable=True)
                self.ui.pw_graph.addItem(self.xmax_line)
                self.xmax_line.sigPositionChangeFinished.connect(
                    self.update_xmax_spin_box)
            else:
                self.xmin_line = None
                self.xmax_line = None

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return

    def update_xlimits(self):
        """Update xmin and xmax values."""
        selected_idx, _ = self.get_selected_scan_component()
        if selected_idx is None:
            return

        self.ui.sbd_xmin.setValue(
            self.scan_dict[selected_idx]['xmin'])
        self.ui.sbd_xmax.setValue(
            self.scan_dict[selected_idx]['xmax'])

    def update_xmax(self):
        """Update xmax value."""
        selected_idx, _ = self.get_selected_scan_component()
        if selected_idx is None:
            return

        self.scan_dict[selected_idx]['xmax'] = self.ui.sbd_xmax.value()

    def update_xmax_spin_box(self):
        """Update xmax value."""
        self.ui.sbd_xmax.setValue(self.xmax_line.pos()[0])

    def update_xmin(self):
        """Update xmin value."""
        selected_idx, _ = self.get_selected_scan_component()
        if selected_idx is None:
            return

        self.scan_dict[selected_idx]['xmin'] = self.ui.sbd_xmin.value()

    def update_xmin_spin_box(self):
        """Update xmin value."""
        self.ui.sbd_xmin.setValue(self.xmin_line.pos()[0])


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
