# -*- coding: utf-8 -*-

"""Temperature widget for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import time as _time
import traceback as _traceback
from qtpy.QtWidgets import (
    QLabel as _QLabel,
    QWidget as _QWidget,
    QCheckBox as _QCheckBox,
    QGroupBox as _QGroupBox,
    QLineEdit as _QLineEdit,
    QMessageBox as _QMessageBox,
    QPushButton as _QPushButton,
    QHBoxLayout as _QHBoxLayout,
    QGridLayout as _QGridLayout,
    QSizePolicy as _QSizePolicy,
    QApplication as _QApplication,
    QDoubleSpinBox as _QDoubleSpinBox,
    )
from qtpy.QtGui import (
    QFont as _QFont,
    )
from qtpy.QtCore import (
    Qt as _Qt,
    QSize as _QSize,
    Signal as _Signal,
    QThread as _QThread,
    QObject as _QObject,
    )
import qtpy.uic as _uic

import hallbench.gui.utils as _utils
from hallbench.gui.auxiliarywidgets import TablePlotWidget as _TablePlotWidget


class TemperatureWidget(_TablePlotWidget):
    """Temperature Widget class for the Hall Bench Control application."""

    _left_axis_1_label = 'Temperature [deg C]'
    _left_axis_1_format = '{0:.4f}'
    _left_axis_1_data_labels = [
        'CH101', 'CH102', 'CH103', 'CH105',
        'CH201', 'CH202', 'CH203', 'CH204',
        'CH205', 'CH206', 'CH207', 'CH208', 'CH209',
    ]
    _left_axis_1_data_colors = [
        (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
        (145, 30, 180), (255, 225, 25), (70, 240, 240), (240, 50, 230),
        (170, 110, 40), (128, 0, 0), (0, 0, 0), (128, 128, 128), (0, 255, 0),
    ]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # add channels widget
        self.channels = [
            ch.replace('CH', '') for ch in self._left_axis_1_data_labels]
        self.channels_widget = TemperatureChannelsWidget(self.channels)
        self.addWidgetsNextToPlot(self.channels_widget)

        # add configuration button
        self.configure_btn = _QPushButton('Configure Channels')
        self.configure_btn.clicked.connect(self.configureChannels)
        self.addWidgetsNextToTable(self.configure_btn)

        # Change default appearance
        self.setTableColumnSize(80)

        # Create reading thread
        self.wthread = _QThread()
        self.worker = ReadValueWorker(self.channels)
        self.worker.moveToThread(self.wthread)
        self.wthread.started.connect(self.worker.run)
        self.worker.finished.connect(self.wthread.quit)
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
            self.wthread.quit()
            super().closeEvent(event)
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
            configured = self.devices.multich.configure(
                selected_channels, wait=wait)

            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()

            if not configured:
                msg = 'Failed to configure Multichannel.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

        except Exception:
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            _traceback.print_exc(file=_sys.stdout)

    def getReading(self):
        """Get reading from worker thread."""
        try:
            ts = self.worker.timestamp
            r = self.worker.reading
            
            if ts is None:
                return
            
            if len(r) == 0 or all([_np.isnan(ri) for ri in r]):
                return

            self._timestamp.append(ts)

            for i, ch in enumerate(self.channels):
                label = self._data_labels[i]
                temperature = r[i]
                if _np.isnan(temperature):
                    text = ''
                else:
                    text = self._left_axis_1_format.format(temperature)
                self.channels_widget.updateChannelText(ch, text)
                self._readings[label].append(temperature)

            self.addLastValueToTable()
            self.updatePlot()
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

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

        try:
            self.worker.delay = self.channels_widget.delay
            self.worker.selected_channels = selected_channels
            self.wthread.start()
        
        except Exception:
            _traceback.print_exc(file=_sys.stdout)


class TemperatureChannelsWidget(_QWidget):
    """Temperature channels widget class."""

    channelChanged = _Signal()

    def __init__(self, channels, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)
        self.resize(275, 525)
        self.setWindowTitle("Temperature Channels")
        
        font = _QFont()
        font.setPointSize(11)
        font.setBold(False)
        self.setFont(font)
        
        font_bold = _QFont()
        font_bold.setPointSize(11)
        font_bold.setBold(True)
        
        main_layout = _QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout = _QGridLayout()
        
        group_box = _QGroupBox("Temperature [°C]")
        size_policy = _QSizePolicy(
            _QSizePolicy.Maximum, _QSizePolicy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)

        group_box.setSizePolicy(size_policy)
        group_box.setFont(font_bold)
        group_box.setLayout(grid_layout)

        size_policy = _QSizePolicy(
            _QSizePolicy.Minimum, _QSizePolicy.Fixed)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        max_size = _QSize(155, 16777215)

        self.channels = channels
        for idx, ch in enumerate(self.channels):
            chb_label = "CH " + ch
            if ch == '101':
                chb_label = chb_label + ' (X)'
            elif ch == '102':
                chb_label = chb_label + ' (Y)'
            elif ch == '103':
                chb_label = chb_label + ' (Z)'
            chb = _QCheckBox(chb_label)
            chb.setFont(font)
            chb.setChecked(False)
            chb.stateChanged.connect(self.clearChannelText)
            setattr(self, 'channel' + ch + '_chb', chb)
            
            le = _QLineEdit()
            le.setSizePolicy(size_policy)
            le.setMaximumSize(max_size)
            le.setFont(font)
            le.setReadOnly(True)
            setattr(self, 'channel' + ch + '_le', le)

            grid_layout.addWidget(chb, idx, 0, 1, 1)
            grid_layout.addWidget(le, idx, 1, 1, 2)

        delay_label = _QLabel("Reading delay [s]:")
        size_policy = _QSizePolicy(
            _QSizePolicy.Preferred, _QSizePolicy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        delay_label.setSizePolicy(size_policy)
        delay_label.setFont(font)
        delay_label.setAlignment(
            _Qt.AlignRight|_Qt.AlignTrailing|_Qt.AlignVCenter)
        grid_layout.addWidget(delay_label, len(self.channels)+1, 0, 1, 2)

        self.delay_sb = _QDoubleSpinBox()
        size_policy = _QSizePolicy(_QSizePolicy.Maximum, _QSizePolicy.Fixed)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        self.delay_sb.setSizePolicy(size_policy)
        self.delay_sb.setFont(font)
        self.delay_sb.setValue(1.0)
        grid_layout.addWidget(self.delay_sb, len(self.channels)+1, 2, 1, 1)
        
        main_layout.addWidget(group_box)
        self.setLayout(main_layout)           

    @property
    def selected_channels(self):
        """Return the selected channels."""
        selected_channels = []
        for channel in self.channels:
            chb = getattr(self, 'channel' + channel + '_chb')
            if chb.isChecked():
                selected_channels.append(channel)
        return selected_channels

    @property
    def delay(self):
        """Return the reading delay."""
        return self.delay_sb.value()

    def clearChannelText(self):
        """Clear channel text if channel is not selected."""
        for channel in self.channels:
            if channel not in self.selected_channels:
                le = getattr(self, 'channel' + channel + '_le')
                le.setText('')
        self.channelChanged.emit()

    def updateChannelText(self, channel, text):
        """Update channel text."""
        le = getattr(self, 'channel' + channel + '_le')
        le.setText(text)


class ReadValueWorker(_QObject):
    """Read values worker."""

    finished = _Signal([bool])

    def __init__(self, channels):
        """Initialize object."""
        self.channels = channels
        self.delay = 0
        self.selected_channels = []
        self.timestamp = None
        self.reading = []
        super().__init__()

    @property
    def devices(self):
        """Hall Bench Devices."""
        return _QApplication.instance().devices

    def run(self):
        """Read values from devices."""
        try:
            self.timestamp = None
            self.reading = []

            ts = _time.time()
            rl = self.devices.multich.get_converted_readings(wait=self.delay)
            if len(rl) == len(self.selected_channels):
                rl = [r if _np.abs(r) < 1e37 else _np.nan for r in rl]
            else:
                rl = [_np.nan]*len(self.selected_channels)

            count = 0
            self.timestamp = ts
            for ch in self.channels:
                if ch in self.selected_channels:
                    self.reading.append(rl[count])
                    count = count + 1
                else:
                    self.reading.append(_np.nan)
            self.finished.emit(True)

        except Exception:
            self.timestamp = None
            self.reading = []
            self.finished.emit(True)
