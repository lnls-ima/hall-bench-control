# -*- coding: utf-8 -*-

"""View probe dialog for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import warnings as _warnings
import traceback as _traceback
import pyqtgraph as _pyqtgraph
from qtpy.QtWidgets import (
    QDialog as _QDialog,
    QApplication as _QApplication,
    QMessageBox as _QMessageBox,
    QTableWidgetItem as _QTableWidgetItem,
    )
import qtpy.uic as _uic

from hallbench.gui.utils import get_ui_file as _get_ui_file
from hallbench.data.calibration import (
    HallCalibrationCurve as _HallCalibrationCurve)


class ViewProbeDialog(_QDialog):
    """View probe dialog class for Hall Bench Control application."""

    _axis_str_dict = {
        1: '+ Axis #1 (+Z)', 2: '+ Axis #2 (+Y)', 3: '+ Axis #3 (+X)'}

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _get_ui_file(self)
        self.ui = _uic.loadUi(uifile, self)

        self.calibrationx = _HallCalibrationCurve()
        self.calibrationy = _HallCalibrationCurve()
        self.calibrationz = _HallCalibrationCurve()

        self.graphx = []
        self.graphy = []
        self.graphz = []

        self.configure_graph()
        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.pw_plot.graphicsItem())
        self.legend.setAutoFillBackground(1)

        self.clip = _QApplication.clipboard()

        self.connect_signal_slots()

    @property
    def database_name(self):
        """Database name."""
        return _QApplication.instance().database_name

    @property
    def mongo(self):
        """MongoDB database."""
        return _QApplication.instance().mongo

    @property
    def server(self):
        """Server for MongoDB database."""
        return _QApplication.instance().server

    def accept(self):
        """Close dialog."""
        self.clear()
        super().accept()

    def clear(self):
        """Clear."""
        self.graphx = []
        self.graphy = []
        self.graphz = []
        self.ui.pw_plot.clear()
        self.clear_probe('x')
        self.clear_probe('y')
        self.clear_probe('z')

    def clear_probe(self, probe):
        cal = getattr(self, 'calibration{0:s}'.format(probe))
        cal.clear()

        le_name = getattr(
            self.ui, 'le_probe{0:s}_name'.format(probe))
        le_name.setText('')

        le_magnet = getattr(
            self.ui, 'le_probe{0:s}_magnet'.format(probe))
        le_magnet.setText('')
        
        le_function = getattr(
            self.ui, 'le_probe{0:s}_function'.format(probe))
        le_function.setText('')
        
        le_vmin = getattr(
            self.ui, 'le_probe{0:s}_vmin'.format(probe))
        le_vmin.setText('')

        le_vmax = getattr(
            self.ui, 'le_probe{0:s}_vmax'.format(probe))
        le_vmax.setText('')
        
        sb_poly_prec = getattr(
            self.ui, 'sb_probe{0:s}_poly_prec'.format(probe))
        sb_poly_prec.setValue(4)

        sb_data_prec = getattr(
            self.ui, 'sb_probe{0:s}_data_prec'.format(probe))
        sb_data_prec.setValue(4)
        
        tbl_poly = getattr(
            self.ui, 'tbl_probe{0:s}_poly'.format(probe))
        tbl_poly.clearContents()
        tbl_poly.setRowCount(0)

        tbl_data = getattr(
            self.ui, 'tbl_probe{0:s}_data'.format(probe))
        tbl_data.clearContents()
        tbl_data.setRowCount(0)
        
    def configure_graph(self, symbol=False):
        """Configure data plots."""
        self.ui.pw_plot.clear()

        self.graphx = []
        self.graphy = []
        self.graphz = []

        penx = (255, 0, 0)
        peny = (0, 255, 0)
        penz = (0, 0, 255)

        if symbol:
            plot_item_x = self.ui.pw_plot.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=penx,
                symbol='o',
                symbolPen=penx,
                symbolSize=4,
                symbolBrush=penx)

            plot_item_y = self.ui.pw_plot.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=peny,
                symbol='o',
                symbolPen=peny,
                symbolSize=4,
                symbolBrush=peny)

            plot_item_z = self.ui.pw_plot.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=penz,
                symbol='o',
                symbolPen=penz,
                symbolSize=4,
                symbolBrush=penz)
        else:
            plot_item_x = self.ui.pw_plot.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=penx)

            plot_item_y = self.ui.pw_plot.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=peny)

            plot_item_z = self.ui.pw_plot.plotItem.plot(
                _np.array([]),
                _np.array([]),
                pen=penz)

        self.graphx.append(plot_item_x)
        self.graphy.append(plot_item_y)
        self.graphz.append(plot_item_z)

        self.ui.pw_plot.setLabel('bottom', 'Voltage [V]')
        self.ui.pw_plot.setLabel('left', 'Magnetic Field [T]')
        self.ui.pw_plot.showGrid(x=True, y=True)

    def connect_signal_slots(self):
        """Create signal/slot connections."""
        self.ui.pbt_updategraph.clicked.connect(self.update_graph)
        self.ui.sbd_voltage.valueChanged.connect(self.update_field)
        self.ui.sb_field_prec.valueChanged.connect(self.update_field)
        self.ui.tbt_field_copy.clicked.connect(self.copy_field)
        
        probes = ['x', 'y', 'z']
        for probe in probes:
            sb_poly_prec = getattr(
                self.ui, 'sb_probe{0:s}_poly_prec'.format(probe))
            sb_poly_prec.valueChanged.connect(
                lambda: self.update_table_poly(probe))

            sb_data_prec = getattr(
                self.ui, 'sb_probe{0:s}_data_prec'.format(probe))
            sb_data_prec.valueChanged.connect(
                lambda: self.update_table_data(probe))

            tbt_poly_copy = getattr(
                self.ui, 'tbt_probe{0:s}_poly_copy'.format(probe))
            tbt_poly_copy.clicked.connect(
                lambda: self.copy_table_poly(probe))            

            tbt_data_copy = getattr(
                self.ui, 'tbt_probe{0:s}_data_copy'.format(probe))
            tbt_data_copy.clicked.connect(
                lambda: self.copy_table_data(probe))      

    def copy_field(self):
        try:
            voltage = str(self.ui.sbd_voltage.value())
            fieldx = self.ui.le_fieldx.text()
            fieldy = self.ui.le_fieldy.text()
            fieldz = self.ui.le_fieldz.text()

            text = voltage + "\n"
            text += fieldx + "\n"
            text += fieldy + "\n"
            text += fieldz
            self.clip.setText(text)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)  

    def copy_table(self, table):
        try:
            text = ""
            for r in range(table.rowCount()):
                for c in range(table.columnCount()):
                    text += str(table.item(r, c).text()) + "\t"
                text = text[:-1] + "\n"
            self.clip.setText(text)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)        

    def copy_table_poly(self, probe):
        try:
            table = getattr(self.ui, 'tbl_probe{0:s}_poly'.format(probe))
            self.copy_table(table)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            
    def copy_table_data(self, probe):
        try:
            table = getattr(self.ui, 'tbl_probe{0:s}_data'.format(probe))
            self.copy_table(table)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def load_probe(self, probe):
        """Load hall probe parameters."""
        try:
            cal = getattr(self, 'calibration{0:s}'.format(probe))
                      
            le_name = getattr(
                self.ui, 'le_probe{0:s}_name'.format(probe))
            le_name.setText(cal.calibration_name)
    
            le_magnet = getattr(
                self.ui, 'le_probe{0:s}_magnet'.format(probe))
            le_magnet.setText(cal.calibration_magnet)
            
            le_function = getattr(
                self.ui, 'le_probe{0:s}_function'.format(probe))
            le_function.setText(cal.function_type)
            
            sb_poly_prec = getattr(
                self.ui, 'sb_probe{0:s}_poly_prec'.format(probe))
            poly_prec = sb_poly_prec.value()
            poly_format = '{0:0.%ig}' % poly_prec
            
            le_vmin = getattr(
                self.ui, 'le_probe{0:s}_vmin'.format(probe))
            if cal.voltage_min is not None:
                le_vmin.setText(poly_format.format(cal.voltage_min))
            else:
                le_vmin.setText('')
    
            le_vmax = getattr(
                self.ui, 'le_probe{0:s}_vmax'.format(probe))
            if cal.voltage_max is not None:
                le_vmax.setText(poly_format.format(cal.voltage_max))
            else:
                le_vmax.setText('')
            
            self.update_table_poly(probe)
            self.update_table_data(probe)
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def show(self, calibrationx_name, calibrationy_name, calibrationz_name):
        """Show dialog."""
        try:
            self.clear()

            self.calibrationx.db_update_database(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
            
            self.calibrationy.db_update_database(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
            
            self.calibrationz.db_update_database(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
            
            self.calibrationx.update_calibration(calibrationx_name)
            self.calibrationy.update_calibration(calibrationy_name)
            self.calibrationz.update_calibration(calibrationz_name)
            
            self.update_graph()
            self.load_probe('x')
            self.load_probe('y')
            self.load_probe('z')            
            super().show()
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to show hall probe dialog.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def update_field(self):
        """Convert voltage to magnetic field."""
        self.ui.le_fieldx.setText('')
        self.ui.le_fieldy.setText('')
        self.ui.le_fieldz.setText('')

        try:
            volt = [self.ui.sbd_voltage.value()]
            prec = self.ui.sb_field_prec.value()
            self.ui.sbd_voltage.setDecimals(prec)
            format_str = '{0:0.%ig}' % prec

            if self.calibrationx is not None:
                fieldx = self.calibrationx.get_field(volt)[0]
                self.ui.le_fieldx.setText(format_str.format(fieldx))

            if self.calibrationy is not None:
                fieldy = self.calibrationy.get_field(volt)[0]
                self.ui.le_fieldy.setText(format_str.format(fieldy))

            if self.calibrationz is not None:
                fieldz = self.calibrationz.get_field(volt)[0]
                self.ui.le_fieldz.setText(format_str.format(fieldz))
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to update magnetic field values.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def update_graph(self):
        """Update data plots."""
        try:
            vmin = self.ui.sbd_voltage_min.value()
            vmax = self.ui.sbd_voltage_max.value()
            npts = self.ui.sbd_voltage_npts.value()
            voltage = _np.linspace(vmin, vmax, npts)
            empty_data = _np.ones(len(voltage))*_np.nan

            self.ui.pw_plot.clear()
            self.legend.removeItem('X')
            self.legend.removeItem('Y')
            self.legend.removeItem('Z')

            if self.calibrationx is not None:
                fieldx = self.calibrationx.get_field(voltage)
            else:
                fieldx = empty_data

            if self.calibrationy is not None:
                fieldy = self.calibrationy.get_field(voltage)
            else:
                fieldy = empty_data

            if self.calibrationz is not None:
                fieldz = self.calibrationz.get_field(voltage)
            else:
                fieldz = empty_data

            symbol = self.ui.chb_addmarkers.isChecked()
            self.configure_graph(symbol=symbol)

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")

                if not _np.all(_np.isnan(fieldx)):
                    self.graphx[0].setData(voltage, fieldx)
                    self.legend.addItem(self.graphx[0], 'X')

                if not _np.all(_np.isnan(fieldy)):
                    self.graphy[0].setData(voltage, fieldy)
                    self.legend.addItem(self.graphy[0], 'Y')

                if not _np.all(_np.isnan(fieldz)):
                    self.graphz[0].setData(voltage, fieldz)
                    self.legend.addItem(self.graphz[0], 'Z')

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to update plot.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
    
    def update_table_poly(self, probe):
        cal = getattr(self, 'calibration{0:s}'.format(probe))
        
        table = getattr(self.ui, 'tbl_probe{0:s}_poly'.format(probe))
        table.setRowCount(0)
        
        sb_prec = getattr(
            self.ui, 'sb_probe{0:s}_poly_prec'.format(probe))
        format_str = '{0:0.%ig}' % sb_prec.value()

        if len(cal.polynomial_coeffs) == 0:
            return

        nr = len(cal.polynomial_coeffs)
        table.setRowCount(nr)
        
        labels = []
        for j in range(nr):
            labels.append('C' + str(j))
        table.setVerticalHeaderLabels(labels)

        for i in range(nr):
            table.setItem(i, 0, _QTableWidgetItem(
                format_str.format(cal.polynomial_coeffs[i])))

    def update_table_data(self, probe):
        cal = getattr(self, 'calibration{0:s}'.format(probe))
        
        table = getattr(self.ui, 'tbl_probe{0:s}_data'.format(probe))
        table.setRowCount(0)
        
        sb_prec = getattr(
            self.ui, 'sb_probe{0:s}_data_prec'.format(probe))
        format_str = '{0:0.%ig}' % sb_prec.value()

        if len(cal.voltage) == 0 or len(cal.magnetic_field) == 0:
            return

        nc = 2
        labels = ['Voltage [V]', 'Magnetic Field [T]']
        data = [cal.voltage, cal.magnetic_field]
        
        if len(cal.probe_temperature) != 0:
            nc = nc + 1
            labels.append('Probe Temperature [C]')
            data.append(cal.probe_temperature)

        if len(cal.electronic_box_temperature) != 0:
            nc = nc + 1
            labels.append('Electronic Box Temperature [C]')
            data.append(cal.electronic_box_temperature)

        data = _np.array(data)

        nr = len(cal.voltage)
        table.setRowCount(nr)
        table.setColumnCount(nc)
        table.setHorizontalHeaderLabels(labels)
        
        for i in range(nr):
            for j in range(nc):
                table.setItem(i, j, _QTableWidgetItem(
                    format_str.format(data[i, j])))