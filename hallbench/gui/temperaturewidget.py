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
from hallbench.gui.auxiliarywidgets import (
    TablePlotWidget as _TablePlotWidget,
    TemperatureChannelsWidget as _TemperatureChannelsWidget,
    )


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
        self.channels_widget = _TemperatureChannelsWidget(self.channels)
        self.add_widgets_next_to_plot(self.channels_widget)

        # add configuration button
        self.pbt_configure = _QPushButton('Configure Channels')
        self.pbt_configure.clicked.connect(self.configure_channels)
        self.add_widgets_next_to_table(self.pbt_configure)

        # Change default appearance
        self.set_table_column_size(80)

        # Create reading thread
        self.wthread = _QThread()
        self.worker = ReadValueWorker(self.channels)
        self.worker.moveToThread(self.wthread)
        self.wthread.started.connect(self.worker.run)
        self.worker.finished.connect(self.wthread.quit)
        self.worker.finished.connect(self.get_reading)

    @property
    def devices(self):
        """Hall Bench Devices."""
        return _QApplication.instance().devices

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.wthread.quit()
            super().closeEvent(event)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def check_connection(self, monitor=False):
        """Check devices connection."""
        if not self.devices.multich.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Multichannel not connected.', _QMessageBox.Ok)
            return False
        return True

    def configure_channels(self):
        """Configure channels for temperature measurement."""
        selected_channels = self.channels_widget.selected_channels

        if not self.check_connection():
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

    def get_reading(self):
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
                self.channels_widget.update_channel_text(ch, text)
                self._readings[label].append(temperature)

            self.add_last_value_to_table()
            self.update_plot()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def read_value(self, monitor=False):
        """Read value."""
        if not self.check_connection(monitor=monitor):
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
