# -*- coding: utf-8 -*-

"""Temperature widget for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import time as _time
import traceback as _traceback
from qtpy.QtWidgets import (
    QApplication as _QApplication,
    QMessageBox as _QMessageBox,
    QPushButton as _QPushButton,
    QVBoxLayout as _QVBoxLayout,
    QWidget as _QWidget,
    )
from qtpy.QtCore import (
    Qt as _Qt,
    QDateTime as _QDateTime,
    QThread as _QThread,
    QObject as _QObject,
    Signal as _Signal,
    )
import qtpy.uic as _uic

import hallbench.gui.utils as _utils
from hallbench.gui.tableplotwidget import TablePlotWidget as _TablePlotWidget


class TemperatureWidget(_TablePlotWidget):
    """Temperature Widget class for the Hall Bench Control application."""

    _plot_label = 'Temperature [deg C]'
    _data_format = '{0:.4f}'
    _data_labels = [
        'Ch101', 'Ch102', 'Ch103', 'Ch105',
        'Ch201', 'Ch202', 'Ch203', 'Ch204',
        'Ch205', 'Ch206', 'Ch207', 'Ch208', 'Ch209',
    ]
    _colors = [
        (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
        (145, 30, 180), (255, 225, 25), (70, 240, 240), (240, 50, 230),
        (170, 110, 40), (128, 0, 0), (0, 0, 0), (128, 128, 128), (0, 255, 0),
    ]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # add channels widget
        self.channels_widget = TemperatureChannelsWidget(self)
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

        # Create reading thread
        self.thread = _QThread()
        self.worker = ReadValueWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.getReading)

    @property
    def devices(self):
        """Hall Bench Devices."""
        return _QApplication.instance().devices

    def checkConnection(self, monitor=False):
        """Check devices connection."""
        if not self.devices.multich.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Multichannel not connected.', _QMessageBox.Ok)
            return False
        return True

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.timer.stop()
            self.thread.quit()
            del self.thread
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def configureChannels(self):
        """Configure channels for temperature measurement."""
        selected_channels = self.channels_widget.selected_channels

        if not self.checkConnection():
            return

        try:
            self.blockSignals(True)
            _QApplication.setOverrideCursor(_Qt.WaitCursor)

            wait = self.channels_widget.delay
            self._configured = self.devices.multich.configure(
                selected_channels, wait=wait)

            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()

            if not self._configured:
                msg = 'Failed to configure Multichannel.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

        except Exception:
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            _traceback.print_exc(file=_sys.stdout)

    def getReading(self):
        """Get reading from worker thread."""
        try:
            r = self.worker.reading
            if len(r) == 0 or all([_np.isnan(ri) for ri in r[1:]]):
                return

            self._timestamp.append(r[0])
            self.channels_widget.updateDateTime(r[0])

            for i, ch in enumerate(self.channels_widget.channels):
                label = self._data_labels[i]
                temperature = r[i+1]
                if _np.isnan(temperature):
                    text = ''
                else:
                    text = self._data_format.format(temperature)
                self.channels_widget.updateChannelText(ch, text)
                self._readings[label].append(temperature)

            self.addLastValueToTable()
            self.updatePlot()
        except Exception:
            pass

    def readValue(self, monitor=False):
        """Read value."""
        if not self.checkConnection(monitor=monitor):
            return

        selected_channels = self.channels_widget.selected_channels
        if len(selected_channels) == 0:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'No channel selected.', _QMessageBox.Ok)
            return

        if not self._configured:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Channels not configured.', _QMessageBox.Ok)
            return

        try:
            self.worker.delay = self.channels_widget.delay
            self.worker.channels = self.channels_widget.channels
            self.worker.selected_channels = selected_channels
            self.thread.start()
        except Exception:
            pass


class TemperatureChannelsWidget(_QWidget):
    """Temperature channels widget class."""

    channelChanged = _Signal()

    _channels = [
        '101', '102', '103', '105', '201', '202', '203', '204',
        '205', '206', '207', '208', '209',
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
        self.ui.channel105_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel201_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel202_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel203_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel204_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel205_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel206_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel207_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel208_chb.stateChanged.connect(self.clearChannelText)
        self.ui.channel209_chb.stateChanged.connect(self.clearChannelText)

    @property
    def channels(self):
        """Return channels."""
        return self._channels

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


class ReadValueWorker(_QObject):
    """Read values worker."""

    finished = _Signal([bool])

    def __init__(self):
        """Initialize object."""
        self.delay = 0
        self.selected_channels = []
        self.reading = []
        super().__init__()

    @property
    def devices(self):
        """Hall Bench Devices."""
        return _QApplication.instance().devices

    def run(self):
        """Read values from devices."""
        try:
            self.reading = []

            ts = _time.time()
            rl = self.devices.multich.get_converted_readings(wait=self.delay)
            if len(rl) == len(self.selected_channels):
                rl = [r if _np.abs(r) < 1e37 else _np.nan for r in rl]
            else:
                rl = [_np.nan]*len(self.selected_channels)

            count = 0
            self.reading.append(ts)
            for ch in self.channels:
                if ch in self.selected_channels:
                    self.reading.append(rl[count])
                    count = count + 1
                else:
                    self.reading.append(_np.nan)
            self.finished.emit(True)

        except Exception:
            self.reading = []
            self.finished.emit(True)
