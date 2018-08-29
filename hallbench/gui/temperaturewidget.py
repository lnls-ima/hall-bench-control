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
    QApplication as _QApplication,
    )
from PyQt4.QtCore import (
    QTimer as _QTimer,
    QDateTime as _QDateTime,
    )
import PyQt4.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile


class TemperatureWidget(_QWidget):
    """Temperature Widget class for the Hall Bench Control application."""

    _data_format = '{0:.2f}'
    _probe_channels = ['101', '102', '103']
    _channels = [
        '101', '102', '103', '201', '202', '203', '204',
        '205', '206', '207', '208', '209', '210',
    ]
    _yaxis_label = 'Temperature [deg C]'
    _channel_colors = _collections.OrderedDict([
        ('101', (230, 25, 75)),
        ('102', (60, 180, 75)),
        ('103', (0, 130, 200)),
        ('201', (245, 130, 48)),
        ('202', (145, 30, 180)),
        ('203', (255, 225, 25)),
        ('204', (70, 240, 240)),
        ('205', (240, 50, 230)),
        ('206', (170, 110, 40)),
        ('207', (128, 0, 0)),
        ('208', (170, 255, 195)),
        ('209', (128, 128, 128)),
        ('210', (0, 0, 0)),
    ])

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialization
        self._channels_configured = False
        self._selected_channels = []
        self._legend_items = []
        self.timestamp = []
        self.channel_readings = _collections.OrderedDict([
            ('101', []),
            ('102', []),
            ('103', []),
            ('201', []),
            ('202', []),
            ('203', []),
            ('204', []),
            ('205', []),
            ('206', []),
            ('207', []),
            ('208', []),
            ('209', []),
            ('210', []),
        ])
        self.channel_graphs = _collections.OrderedDict([
            ('101', None),
            ('102', None),
            ('103', None),
            ('201', None),
            ('202', None),
            ('203', None),
            ('204', None),
            ('205', None),
            ('206', None),
            ('207', None),
            ('208', None),
            ('209', None),
            ('210', None),
        ])

        # create timer to monitor values
        self.timer = _QTimer(self)
        self.updateMonitorInterval()
        self.timer.timeout.connect(lambda: self.readValue(monitor=True))

        self.ui.configureled_la.setEnabled(False)
        self.ui.table_ta.setAlternatingRowColors(True)

        dt = _QDateTime()
        dt.setTime_t(_time.time())
        self.ui.time_dte.setDateTime(dt)

        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.plot_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        self.configureGraph()
        self.connectSignalSlots()

    @property
    def devices(self):
        """Hall Bench Devices."""
        return _QApplication.instance().devices

    def updateLegendItems(self):
        """Update legend items."""
        self.clearLegendItems()
        self._legend_items = []
        for channel in self._selected_channels:
            label = 'Ch' + channel
            self._legend_items.append(label)
            self.legend.addItem(self.channel_graphs[channel], label)

    def clearLegendItems(self):
        """Clear plot legend."""
        for label in self._legend_items:
            self.legend.removeItem(label)

    def clearValues(self):
        """Clear all values."""
        self.timestamp = []
        for channel in self._channels:
            self.channel_readings[channel] = []
        self.updateTableValues()
        self.updatePlot()

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.timer.stop()
            event.accept()
        except Exception:
            event.accept()

    def configureChannels(self):
        """Configure channels for temperature measurement."""
        self._selected_channels = []
        for channel in self._channels:
            chb = getattr(self.ui, 'channel' + channel + '_chb')
            if chb.isChecked():
                self._selected_channels.append(channel)
            else:
                le = getattr(self.ui, 'channel' + channel + '_le')
                le.setText('')

        self.updateLegendItems()

        if not self.devices.multich.connected:
            self._channels_configured = False
            _QMessageBox.critical(
                self, 'Failure',
                'Multichannel not connected.', _QMessageBox.Ok)
            return

        wait = self.ui.delay_sb.value()
        if self.devices.multich.configure(self._selected_channels, wait=wait):
            self._channels_configured = True
            self.ui.configureled_la.setEnabled(True)
            return
        else:
            self._channels_configured = False
            return

    def configureGraph(self):
        """Configure data plots."""
        self.ui.plot_pw.clear()

        for channel in self._channels:
            pen = self._channel_colors[channel]
            graph = self.ui.plot_pw.plotItem.plot(
                _np.array([]), _np.array([]), pen=pen,
                symbol='o', symbolPen=pen, symbolSize=3, symbolBrush=pen)
            self.channel_graphs[channel] = graph

        self.ui.plot_pw.setLabel('bottom', 'Time interval [s]')
        self.ui.plot_pw.setLabel('left', self._yaxis_label)
        self.ui.plot_pw.showGrid(x=True, y=True)
        self.updateLegendItems()

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.configure_btn.clicked.connect(self.configureChannels)
        self.ui.read_btn.clicked.connect(
            lambda: self.readValue(monitor=False))
        self.ui.monitor_btn.toggled.connect(self.monitorValue)
        self.ui.monitorstep_sb.valueChanged.connect(self.updateMonitorInterval)
        self.ui.monitorunit_cmb.currentIndexChanged.connect(
            self.updateMonitorInterval)
        self.ui.clear_btn.clicked.connect(self.clearValues)
        self.ui.remove_btn.clicked.connect(self.removeValue)
        self.ui.copy_btn.clicked.connect(self.copyToClipboard)
        self.ui.channel101_chb.stateChanged.connect(self.disableLed)
        self.ui.channel102_chb.stateChanged.connect(self.disableLed)
        self.ui.channel103_chb.stateChanged.connect(self.disableLed)
        self.ui.channel201_chb.stateChanged.connect(self.disableLed)
        self.ui.channel202_chb.stateChanged.connect(self.disableLed)
        self.ui.channel203_chb.stateChanged.connect(self.disableLed)
        self.ui.channel204_chb.stateChanged.connect(self.disableLed)
        self.ui.channel205_chb.stateChanged.connect(self.disableLed)
        self.ui.channel206_chb.stateChanged.connect(self.disableLed)
        self.ui.channel207_chb.stateChanged.connect(self.disableLed)
        self.ui.channel208_chb.stateChanged.connect(self.disableLed)
        self.ui.channel209_chb.stateChanged.connect(self.disableLed)
        self.ui.channel210_chb.stateChanged.connect(self.disableLed)

    def copyToClipboard(self):
        """Copy table data to clipboard."""
        nr = self.ui.table_ta.rowCount()
        nc = self.ui.table_ta.columnCount()

        if nr == 0:
            return

        col_labels = ['Date', 'Time']
        for channel in self._channels:
            col_labels.append(channel)
        tdata = []
        for i in range(nr):
            ldata = []
            for j in range(nc):
                value = self.ui.table_ta.item(i, j).text()
                if j >= 2:
                    value = float(value)
                ldata.append(value)
            tdata.append(ldata)
        _df = _pd.DataFrame(_np.array(tdata), columns=col_labels)
        _df.to_clipboard(excel=True)

    def disableLed(self):
        """Disable configuration led."""
        self.ui.configureled_la.setEnabled(False)

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
        if len(self._selected_channels) == 0:
            return

        if not self.devices.multich.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Multichannel not connected.', _QMessageBox.Ok)
            return

        if not self._channels_configured:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Channels not configured.', _QMessageBox.Ok)
            return

        ts = _time.time()
        wait = self.ui.delay_sb.value()
        rl = self.devices.multich.get_converted_readings(wait=wait)
        if len(rl) != len(self._selected_channels):
            return

        readings = [r if r < 1e37 else _np.nan for r in rl]

        try:
            for i in range(len(self._selected_channels)):
                channel = self._selected_channels[i]
                le = getattr(self.ui, 'channel' + channel + '_le')
                temperature = readings[i]
                le.setText(self._data_format.format(temperature))
                self.channel_readings[channel].append(temperature)

            for channel in self._channels:
                if channel not in self._selected_channels:
                    le = getattr(self.ui, 'channel' + channel + '_le')
                    le.setText('')
                    self.channel_readings[channel].append(_np.nan)

            self.timestamp.append(ts)
            dt = _QDateTime()
            dt.setTime_t(ts)
            self.ui.time_dte.setDateTime(dt)

            self.updateTableValues()
            self.updatePlot()

        except Exception:
            pass

    def removeValue(self):
        """Remove value from list."""
        selected = self.ui.table_ta.selectedItems()
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
        self.ui.table_ta.clearContents()
        self.ui.table_ta.setRowCount(n)

        for j in range(len(self._channels)):
            readings = self.channel_readings[self._channels[j]]
            for i in range(n):
                dt = _datetime.datetime.fromtimestamp(self.timestamp[i])
                date = dt.strftime("%d/%m/%Y")
                hour = dt.strftime("%H:%M:%S")
                self.ui.table_ta.setItem(i, 0, _QTableWidgetItem(date))
                self.ui.table_ta.setItem(i, 1, _QTableWidgetItem(hour))
                self.ui.table_ta.setItem(
                    i, j+2, _QTableWidgetItem(self._data_format.format(
                        readings[i])))

        if scrollDown:
            vbar = self.table_ta.verticalScrollBar()
            vbar.setValue(vbar.maximum())
