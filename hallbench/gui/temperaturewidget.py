# -*- coding: utf-8 -*-

"""Temperature widget for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import time as _time
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
    )
import PyQt5.uic as _uic

import hallbench.gui.utils as _utils
from hallbench.gui.tableplotwidget import TablePlotWidget as _TablePlotWidget


class TemperatureWidget(_TablePlotWidget):
    """Temperature Widget class for the Hall Bench Control application."""

    _plot_label = 'Temperature [deg C]'
    _data_format = '{0:.2f}'
    _data_labels = [
        '101', '102', '103', '201',
        '202', '203', '204',
        '205', '206', '207',
        '208', '209', '210',
    ]
    _colors = [
        (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
        (145, 30, 180), (255, 225, 25), (70, 240, 240),
        (240, 50, 230), (170, 110, 40), (128, 0, 0),
        (170, 255, 195), (128, 128, 128), (0, 0, 0),
    ]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # add channels widget
        self.channels_widget = TemperatureChannelsWidget(self)
        self.channels_widget.channelChanged.connect(self.enableConfigureButton)
        _layout = _QVBoxLayout()
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.addWidget(self.channels_widget)
        self.ui.widget_wg.setLayout(_layout)

        # add configuration button
        self._configured = False
        self.configure_btn = _QPushButton('Configure Channels')
        self.configure_btn.setMinimumHeight(45)
        font = self.configure_btn.font()
        font.setBold(True)
        self.configure_btn.setFont(font)
        self.ui.layout_lt.addWidget(self.configure_btn)
        self.configure_btn.clicked.connect(self.configureChannels)

        # Change default appearance
        self.ui.table_ta.horizontalHeader().setDefaultSectionSize(80)
        self.ui.read_btn.setText('Read Temperature')
        self.ui.monitor_btn.setText('Monitor Temperature')

    @property
    def devices(self):
        """Hall Bench Devices."""
        return _QApplication.instance().devices

    def configureChannels(self):
        """Configure channels for temperature measurement."""
        selected_channels = self.channels_widget.selected_channels

        if not self.devices.multich.connected:
            _QMessageBox.critical(
                self, 'Failure',
                'Multichannel not connected.', _QMessageBox.Ok)
            return

        try:
            wait = self.channels_widget.delay
            self._configured = self.devices.multich.configure(
                selected_channels, wait=wait)
            if self._configured:
                self.configure_btn.setEnabled(False)
            else:
                self.configure_btn.setEnabled(True)
                msg = 'Failed to configure Multichannel.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def enableConfigureButton(self):
        """Enable configure button."""
        self.configure_btn.setEnabled(True)

    def readValue(self, monitor=False):
        """Read value."""
        selected_channels = self.channels_widget.selected_channels
        if len(selected_channels) == 0:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'No channel selected.', _QMessageBox.Ok)
            return

        if not self.devices.multich.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Multichannel not connected.', _QMessageBox.Ok)
            return

        if not self._configured:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Channels not configured.', _QMessageBox.Ok)
            return

        ts = _time.time()
        wait = self.channels_widget.delay
        rl = self.devices.multich.get_converted_readings(wait=wait)
        if len(rl) != len(selected_channels):
            return

        readings = [r if _np.abs(r) < 1e37 else _np.nan for r in rl]

        try:
            for i in range(len(selected_channels)):
                label = selected_channels[i]
                temperature = readings[i]
                text = self._data_format.format(temperature)
                self.channels_widget.updateChannelText(label, text)
                self._readings[label].append(temperature)

            for label in self._data_labels:
                if label not in selected_channels:
                    self.channels_widget.updateChannelText(label, '')
                    self._readings[label].append(_np.nan)

            self._timestamp.append(ts)
            self.channels_widget.updateDateTime(ts)
            self.addLastValueToTable()
            self.updatePlot()

        except Exception:
            pass


class TemperatureChannelsWidget(_QWidget):
    """Temperature channels widget class."""

    channelChanged = _pyqtSignal()

    _channels = [
        '101', '102', '103', '201', '202', '203', '204',
        '205', '206', '207', '208', '209', '210',
    ]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _utils.getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        dt = _QDateTime()
        dt.setTime_t(_time.time())
        self.ui.time_dte.setDateTime(dt)
        self.ui.channel101_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel102_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel103_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel201_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel202_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel203_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel204_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel205_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel206_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel207_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel208_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel209_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel210_chb.stateChanged.connect(self.clearChannelText)

    @property
    def selected_channels(self):
        """Return the selected channels."""
        selected_channels = []
        for channel in self._channels:
            chb = getattr(self.ui, 'channel' + channel + '_chb')
            if chb.isChecked():
                selected_channels.append(channel)
        return selected_channels

    @property
    def delay(self):
        """Return the reading delay."""
        return self.ui.delay_sb.value()

    def clearChannelText(self):
        """Clear channel text if channel is not selected."""
        for channel in self._channels:
            if channel not in self.selected_channels:
                le = getattr(self.ui, 'channel' + channel + '_le')
                le.setText('')
        self.channelChanged.emit()

    def updateChannelText(self, channel, text):
        """Update channel text."""
        le = getattr(self.ui, 'channel' + channel + '_le')
        le.setText(text)

    def updateDateTime(self, timestamp):
        """Update DateTime value."""
        dt = _QDateTime()
        dt.setTime_t(timestamp)
        self.ui.time_dte.setDateTime(dt)
