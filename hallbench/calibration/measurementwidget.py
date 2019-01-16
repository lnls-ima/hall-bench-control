# -*- coding: utf-8 -*-

"""Measurement Widget for the Hall Bench calibration application."""

import sys as _sys
import numpy as _np
import time as _time
import datetime as _datetime
import warnings as _warnings
import pyqtgraph as _pyqtgraph
import traceback as _traceback
from qtpy.QtWidgets import (
    QApplication as _QApplication,
    QFileDialog as _QFileDialog,
    QHeaderView as _QHeaderView,
    QMessageBox as _QMessageBox,
    QTableWidgetItem as _QTableWidgetItem,
    QWidget as _QWidget,
    )
from qtpy.QtCore import QTimer as _QTimer
import qtpy.uic as _uic
import pyqtgraph as _pyqtgraph

from hallbench.calibration import utils as _utils


class MeasurementWidget(_QWidget):
    """Measurement Widget class for the Hall Bench calibration application."""
    
    _temp_format = '{0:.4f}'
    _voltage_format = '{0:.8e}'
    _field_format = '{0:.8e}'
    
    _data_labels = [
        'CH101', 'CH102', 'CH103', 'CH105', 'CH201', 'Voltage', 'Field']
    _colors = [
        (230, 25, 75), (145, 30, 180), (255, 225, 25), (245, 130, 48), 
        (0, 0, 0),  (60, 180, 75), (0, 130, 200)]
    _voltage_color = '#3CB44B'
    _field_color = '#0082C8'
    
    _mch_channels = ['CH101', 'CH102', 'CH103', 'CH105', 'CH201']
    _mch_delay = 0.2

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _utils.getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialisation
        self._timestamp = []
        self._legend_items = []
        self._readings = {}
        self._graphs = {}
        for i, label in enumerate(self._data_labels):
            self._readings[label] = []

        # create timer to monitor values
        self.timer = _QTimer(self)
        self.updateMonitorInterval()
        self.timer.timeout.connect(lambda: self.readValue(monitor=True))

        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.plot_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        self.configurePlot()
        self.configureTable()
        self.connectSignalSlots()

    @property
    def mch(self):
        """Multichannel."""
        return _QApplication.instance().mch

    @property
    def mult(self):
        """Multimeter."""
        return _QApplication.instance().mult

    @property
    def nmr(self):
        """NMR."""
        return _QApplication.instance().nmr

    @property
    def selected_mch_channels(self):
        """Return the selected temperature channels."""
        selected_mch_channels = []
        for channel in self._mch_channels:
            chb = getattr(self.ui, 'mch_' + channel.lower() + '_chb')
            if chb.isChecked():
                selected_mch_channels.append(channel)
        return selected_mch_channels

    def addLastValueToTable(self):
        """Add the last value read to table."""
        if len(self._timestamp) == 0:
            return

        n = self.ui.table_ta.rowCount() + 1
        self.ui.table_ta.setRowCount(n)

        dt = _datetime.datetime.fromtimestamp(self._timestamp[-1])
        date = dt.strftime("%d/%m/%Y")
        hour = dt.strftime("%H:%M:%S")
        self.ui.table_ta.setItem(n-1, 0, _QTableWidgetItem(date))
        self.ui.table_ta.setItem(n-1, 1, _QTableWidgetItem(hour))

        for j in range(len(self._data_labels)):
            label = self._data_labels[j]
            reading = self._readings[label][-1]
            data_format = self.getDataFormat(label)
            self.ui.table_ta.setItem(n-1, j+2, _QTableWidgetItem(
                    data_format.format(reading)))

        vbar = self.table_ta.verticalScrollBar()
        vbar.setValue(vbar.maximum())

    def addRightAxes(self):
        """Add axis to graph."""
        p = self.ui.plot_pw.plotItem

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
        self.first_right_axis.setStyle(showValues=True)
        self.second_right_axis.setStyle(showValues=True)
        self.first_right_axis_vb = pr1
        self.second_right_axis_vb = pr2

    def clearTempChannelText(self):
        """Clear channel text if channel is not selected."""
        for channel in self._mch_channels:
            if channel not in self.selected_mch_channels:
                le = getattr(self.ui, 'mch_' + channel.lower() + '_le')
                le.setText('')

    def clearLegendItems(self):
        """Clear plot legend."""
        for label in self._legend_items:
            self.legend.removeItem(label)

    def clear(self):
        """Clear all values."""
        self._timestamp = []
        for label in self._data_labels:
            self._readings[label] = []
        self.updateTableValues()
        self.updatePlot()

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.timer.stop()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def configurePlot(self):
        """Configure data plots."""
        self.ui.plot_pw.clear()
        self.addRightAxes()

        for i, label in enumerate(self._data_labels):
            color = self._colors[i]
            if 'voltage' in label.lower():
                graph = _pyqtgraph.PlotDataItem(
                    _np.array([]), _np.array([]),
                    pen=color, symbol='o', symbolPen=color,
                    symbolSize=3, symbolBrush=color)
                self.first_right_axis.linkedView().addItem(graph)
            elif 'field' in label.lower():
                graph = _pyqtgraph.PlotDataItem(
                    _np.array([]), _np.array([]),
                    pen=color, symbol='o', symbolPen=color,
                    symbolSize=3, symbolBrush=color)
                self.second_right_axis.linkedView().addItem(graph)            
            else:
                graph = self.ui.plot_pw.plotItem.plot(
                    _np.array([]), _np.array([]),
                    pen=color, symbol='o', symbolPen=color,
                    symbolSize=3, symbolBrush=color)
            self._graphs[label] = graph

        self.ui.plot_pw.showGrid(x=True, y=True)
        self.ui.plot_pw.setLabel('bottom', 'Time interval [s]')
        self.ui.plot_pw.setLabel('left', 'Temperature [deg C]')
        self.first_right_axis.setLabel(
            'Voltage [V]', color=self._voltage_color)
        self.second_right_axis.setLabel(
            'Magnetic Field [T]', color=self._field_color)
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
        for channel in self._mch_channels:
            chb = getattr(self.ui, 'mch_' + channel.lower() + '_chb')
            chb.stateChanged.connect(self.clearTempChannelText)
        
        self.ui.read_btn.clicked.connect(lambda: self.readValue(monitor=False))
        self.ui.monitor_btn.toggled.connect(self.monitorValue)
        self.ui.monitorstep_sb.valueChanged.connect(self.updateMonitorInterval)
        self.ui.monitorunit_cmb.currentIndexChanged.connect(
            self.updateMonitorInterval)
        self.ui.clear_btn.clicked.connect(self.clear)
        self.ui.remove_btn.clicked.connect(self.removeValue)
        self.ui.copy_btn.clicked.connect(self.copyToClipboard)
        self.ui.save_btn.clicked.connect(self.saveToFile)

    def copyToClipboard(self):
        """Copy table data to clipboard."""
        df = _utils.tableToDataFrame(self.ui.table_ta)
        if df is not None:
            df.to_clipboard(excel=True)

    def getDataFormat(self, label):
        """Get data format."""
        if 'voltage' in label.lower():
            data_format = self._voltage_format
        elif 'field' in label.lower():
            data_format = self._field_format
        else:
            data_format = self._temp_format
        return data_format

    def monitorValue(self, checked):
        """Monitor values."""
        if checked:
            self.ui.read_btn.setEnabled(False)
            self.timer.start()
        else:
            self.timer.stop()
            self.ui.read_btn.setEnabled(True)

    def readValue(self, monitor=False):
        """Read value."""
        if len(self.selected_mch_channels) > 0 and not self.mch.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Multichannel not connected.', _QMessageBox.Ok)
            return

        if self.ui.mult_voltage_chb.isChecked() and not self.mult.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Multimeter not connected.', _QMessageBox.Ok)
            return            

        if self.ui.nmr_field_chb.isChecked() and not self.nmr.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'NMR not connected.', _QMessageBox.Ok)
            return  

        ts = _time.time()

        try:
            if len(self.selected_mch_channels) > 0:
                rl = self.mch.get_converted_readings(wait=self._mch_delay)
                if len(rl) != len(self.selected_mch_channels):
                    chs = [
                        c.replace('CH', '') for c in self.selected_mch_channels]
                    self.mch.configure(chs)
                    
                rl = self.mch.get_converted_readings(wait=self._mch_delay)
                if len(rl) != len(self.selected_mch_channels):
                    return        
        
                readings = [r if _np.abs(r) < 1e37 else _np.nan for r in rl]
                for i in range(len(self.selected_mch_channels)):
                    label = self.selected_mch_channels[i]
                    temperature = readings[i]
                    text = self._temp_format.format(temperature)
                    self.updateTempChannelText(label, text)
                    self._readings[label].append(temperature)
       
            for label in self._mch_channels:
                if label not in self.selected_mch_channels:
                    self.updateTempChannelText(label, '')
                    self._readings[label].append(_np.nan)

            if self.ui.mult_voltage_chb.isChecked():
                volt = _utils.readVoltageFromMultimeter(self.mult)
                if volt is not None:
                    self.ui.mult_voltage_le.setText(
                        self._voltage_format.format(volt))
                else:
                    volt = _np.nan
                    self.ui.mult_voltage_le.setText('') 
            else:
                volt = _np.nan
                self.ui.mult_voltage_le.setText('')
            self._readings['Voltage'].append(volt)

            if self.ui.nmr_field_chb.isChecked():
                field, state = _utils.readFieldFromNMR(self.nmr)
                if field is not None:
                    self.ui.nmr_field_le.setText(
                        self._field_format.format(field))
                    if state is not None:
                        self.ui.nmr_field_state_le.setText(state)
                    else:
                        self.ui.nmr_field_state_le.setText('')   
                else:
                    field = _np.nan
                    self.ui.nmr_field_le.setText('')
                    self.ui.nmr_field_state_le.setText('') 
            else:
                field = _np.nan
                self.ui.nmr_field_le.setText('')
                self.ui.nmr_field_state_le.setText('')
            self._readings['Field'].append(field)
                    
            self._timestamp.append(ts)
            self.addLastValueToTable()
            self.updatePlot()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def removeValue(self):
        """Remove value from list."""
        selected = self.ui.table_ta.selectedItems()
        rows = [s.row() for s in selected]
        n = len(self._timestamp)

        self._timestamp = [
            self._timestamp[i] for i in range(n) if i not in rows]

        for label in self._data_labels:
            readings = self._readings[label]
            self._readings[label] = [
                readings[i] for i in range(n) if i not in rows]

        self.updateTableValues()
        self.updatePlot()

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

    def updateLegendItems(self):
        """Update legend items."""
        self.clearLegendItems()
        self._legend_items = []
        for label in self._data_labels:
            legend_label = label.split('[')[0]
            self._legend_items.append(legend_label)
            self.legend.addItem(self._graphs[label], legend_label)

    def updateMonitorInterval(self):
        """Update monitor interval value."""
        index = self.ui.monitorunit_cmb.currentIndex()
        if index == 0:
            mf = 1000
        elif index == 1:
            mf = 1000*60
        else:
            mf = 1000*3600
        self.timer.setInterval(self.ui.monitorstep_sb.value()*mf)

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
            
            if len(self.selected_mch_channels) > 0:
                self.ui.plot_pw.plotItem.autoRange()
                
            if self.ui.mult_voltage_chb.isChecked():
                self.first_right_axis_vb.autoRange()
            
            if self.ui.nmr_field_chb.isChecked():
                self.second_right_axis_vb.autoRange()

    def updateTableValues(self):
        """Update table values."""
        n = len(self._timestamp)
        self.ui.table_ta.clearContents()
        self.ui.table_ta.setRowCount(n)

        for i in range(n):
            dt = _datetime.datetime.fromtimestamp(self._timestamp[i])
            date = dt.strftime("%d/%m/%Y")
            hour = dt.strftime("%H:%M:%S")
            self.ui.table_ta.setItem(i, 0, _QTableWidgetItem(date))
            self.ui.table_ta.setItem(i, 1, _QTableWidgetItem(hour))

            for j in range(len(self._data_labels)):
                label = self._data_labels[j]
                reading = self._readings[self._data_labels[j]][i]
                data_format = self.getDataFormat(label)
                self.ui.table_ta.setItem(i, j+2, _QTableWidgetItem(
                    data_format.format(reading)))

    def updateTempChannelText(self, channel, text):
        """Update channel text."""
        le = getattr(self.ui, 'mch_' + channel.lower() + '_le')
        le.setText(text)