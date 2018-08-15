# -*- coding: utf-8 -*-

"""Temperature widget for the Hall Bench Control application."""

import numpy as _np
import pandas as _pd
import time as _time
import datetime as _datetime
import warnings as _warnings
import collections as _collections
import pyqtgraph as _pyqtgraph
from PyQt4.QtGui import (
    QWidget as _QWidget,
    QMessageBox as _QMessageBox,
    QTableWidgetItem as _QTableWidgetItem,
    )
from PyQt4.QtCore import (
    QTimer as _QTimer,
    QDateTime as _QDateTime,
    )
import PyQt4.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile


class TemperatureWidget(_QWidget):
    """Temperature Widget class for the Hall Bench Control application."""

    _temperature_format = '{0:.2f}'
    _slot = '1'
    _channels = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    _channel_colors = _collections.OrderedDict([
        ('01', (230, 25, 75)),
        ('02', (60, 180, 75)),
        ('03', (0, 130, 200)),
        ('04', (245, 130, 48)),
        ('05', (145, 30, 180)),
        ('06', (255, 225, 25)),
        ('07', (70, 240, 240)),
        ('08', (240, 50, 230)),
        ('09', (170, 110, 40)),
        ('10', (0, 0, 0)),
    ])        

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialization
        self.timestamp = []
        self.channel_readings = _collections.OrderedDict([
            ('01', []),
            ('02', []),
            ('03', []),
            ('04', []),
            ('05', []),
            ('06', []),
            ('07', []),
            ('08', []),
            ('09', []),
            ('10', []),
        ])
        self.channel_graphs = _collections.OrderedDict([
            ('01', None),
            ('02', None),
            ('03', None),
            ('04', None),
            ('05', None),
            ('06', None),
            ('07', None),
            ('08', None),
            ('09', None),
            ('10', None),
        ])

        # create connections
        self.ui.read_btn.clicked.connect(
            lambda: self.readTemperature(msgbox=True))
        self.ui.monitor_btn.toggled.connect(self.monitorTemperature)
        self.ui.monitorstep_sb.valueChanged.connect(self.updateMonitorInterval)
        self.ui.monitorunit_cmb.currentIndexChanged.connect(
            self.updateMonitorInterval)
        self.ui.clear_btn.clicked.connect(self.clearTemperatureValues)
        self.ui.remove_btn.clicked.connect(self.removeTemperatureValue)
        self.ui.copy_btn.clicked.connect(self.copyToClipboard)

        # create timer to monitor temperature
        self.timer = _QTimer(self)
        self.updateMonitorInterval()
        self.timer.timeout.connect(lambda: self.readTemperature(msgbox=False))

        # configure plot and table
        self.ui.temperature_ta.setAlternatingRowColors(True)
        self.configureGraph()

        dt = _QDateTime()
        dt.setTime_t(_time.time())
        self.ui.time_dte.setDateTime(dt)

    @property
    def devices(self):
        """Hall Bench devices."""
        return self.window().devices

    def clearTemperatureValues(self):
        """Clear all temperature values."""
        self.timestamp = []
        for channel in self._channels:
            self.channel_readings[channel] = []
        self.updateTableValues()
        self.updatePlot()

    def configureGraph(self):
        """Configure data plots."""
        self.ui.temperature_pw.clear()

        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.temperature_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        for channel in self._channels:
            pen = self._channel_colors[channel]
            readings = self.channel_readings[channel]
            graph = self.ui.temperature_pw.plotItem.plot(
                _np.array([]), _np.array([]), pen=pen,
                symbol='o', symbolPen=pen, symbolSize=3, symbolBrush=pen)
            self.channel_graphs[channel] = graph

        self.ui.temperature_pw.setLabel('bottom', 'Time interval [s]')
        self.ui.temperature_pw.setLabel('left', 'Temperature [deg C]')
        self.ui.temperature_pw.showGrid(x=True, y=True)

        for channel in self._channels:
            self.legend.addItem(
                self.channel_graphs[channel], 'CH' + channel)

    def copyToClipboard(self):
        """Copy table data to clipboard."""
        nr = self.ui.temperature_ta.rowCount()
        nc = self.ui.temperature_ta.columnCount()
        col_labels = ['Date', 'Time']
        for channel in self._channels:
            col_labels.append('CH ' + channel)      
        tdata = []
        for i in range(nr):
            ldata = []
            for j in range(nc):
                value = self.ui.temperature_ta.item(i, j).text()
                if j >= 2:
                    value = float(value)
                ldata.append(value)
            tdata.append(ldata)
        _df =_pd.DataFrame(_np.array(tdata), columns=col_labels)
        _df.to_clipboard(excel=True)                        
                       
    def monitorTemperature(self, checked):
        """Monitor temperature values."""
        if checked:
            self.ui.read_btn.setEnabled(False)
            self.timer.start()
        else:
            self.timer.stop()
            self.ui.read_btn.setEnabled(True)

    def readTemperature(self, msgbox=True):
        """Read temperature value."""
        if not self.devices.multich.connected:
            if msgbox:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Multichannel not connected.', _QMessageBox.Ok)
            return
 
        ts = _time.time()
        self.timestamp.append(ts)

        dt = _QDateTime()
        dt.setTime_t(ts)
        self.ui.time_dte.setDateTime(dt)
        
        for channel in self._channels:
            chb = getattr(self.ui, 'channel' + channel + '_chb')
            if chb.isChecked():         
                temperature = float(self.devices.multich.read_channel(
                    self._slot + channel))
                le = getattr(self.ui, 'channel' + channel + '_le')
                le.setText(self._temperature_format.format(temperature))
            else:
                temperature = _np.nan
            self.channel_readings[channel].append(temperature)

        self.updateTableValues()
        self.updatePlot()

    def removeTemperatureValue(self):
        """Remove temperature value from list."""
        selected = self.ui.temperature_ta.selectedItems()
        rows = [s.row() for s in selected]
        n = len(self.timestamp)

        self.timestamp = [self.timestamp[i] for i in range(n) if i not in rows]
        for channel in self._channels:
            readings = self.channel_readings[channel]
            self.channel_readings[channel] = [
                readings[i] for i in range(n) if i not in rows]

        self.updateTableValues(scrollDown=False)
        self.updatePlot()

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
        if len(self.timestamp) == 0:
            for channels in self._channels:
                self.channel_graphs[channels].setData(
                    _np.array([]), _np.array([]))
            return

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            timeinterval = _np.array(self.timestamp) - self.timestamp[0]
            for channel in self._channels:
                readings = _np.array(self.channel_readings[channel])
                dt = timeinterval[_np.isfinite(readings)]
                rd = readings[_np.isfinite(readings)]
                self.channel_graphs[channel].setData(dt, rd)

    def updateTableValues(self, scrollDown=True):
        """Update table values."""
        n = len(self.timestamp)
        self.ui.temperature_ta.clearContents()
        self.ui.temperature_ta.setRowCount(n)

        for j in range(len(self._channels)):
            readings = self.channel_readings[self._channels[j]]
            for i in range(n):
                dt = _datetime.datetime.fromtimestamp(self.timestamp[i])
                date = dt.strftime("%d/%m/%Y")
                hour = dt.strftime("%H:%M:%S")
                self.ui.temperature_ta.setItem(i, 0, _QTableWidgetItem(date))
                self.ui.temperature_ta.setItem(i, 1, _QTableWidgetItem(hour))
                self.ui.temperature_ta.setItem(
                    i, j+2, _QTableWidgetItem(self._temperature_format.format(
                        readings[i])))

        if scrollDown:
            vbar = self.temperature_ta.verticalScrollBar()
            vbar.setValue(vbar.maximum())
