# -*- coding: utf-8 -*-

"""Voltage offset widget for the Hall Bench Control application."""

import numpy as _np
import time as _time
from PyQt4.QtGui import (
    QApplication as _QApplication,
    QMessageBox as _QMessageBox,
    QPushButton as _QPushButton,
    QVBoxLayout as _QVBoxLayout,
    )

from hallbench.gui.moveaxiswidget import MoveAxisWidget as _MoveAxisWidget
from hallbench.gui.tableplotwidget import TablePlotWidget as _TablePlotWidget


class VoltageOffsetWidget(_TablePlotWidget):
    """Voltage Offset Widget class for the Hall Bench Control application."""

    _data_mult_factor = 1000  # [V] -> [mV]
    _data_format = '{0:.6f}'
    _data_labels = ['ProbeX [mV]', 'ProbeY [mV]', 'ProbeZ [mV]']
    _yaxis_label = 'Voltage [mV]'
    _colors = {
        'ProbeX [mV]': (255, 0, 0),
        'ProbeY [mV]': (0, 255, 0),
        'ProbeZ [mV]': (0, 0, 255),
    }

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # add move axis widget
        self.move_axis_widget = _MoveAxisWidget(self)
        _layout = _QVBoxLayout()
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.addWidget(self.move_axis_widget)
        self.ui.widget_wg.setLayout(_layout)

        self.reset_btn = _QPushButton('Reset Multimeters')
        self.reset_btn.setMinimumHeight(45)
        font = self.reset_btn.font()
        font.setBold(True)
        self.reset_btn.setFont(font)
        self.ui.layout_lt.addWidget(self.reset_btn)
        self.reset_btn.clicked.connect(self.resetMultimeters)

        self.pos = []
        col_labels = ['Date', 'Time', 'Position']
        for label in self._data_labels:
            col_labels.append(label)
        self.ui.table_ta.setColumnCount(len(col_labels))
        self.ui.table_ta.setHorizontalHeaderLabels(col_labels)
        self.ui.table_ta.setAlternatingRowColors(True)
        self.ui.table_ta.horizontalHeader().setDefaultSectionSize(140)

        self.ui.read_btn.setText('Read Voltage')
        self.ui.monitor_btn.setText('Monitor Voltage')

    @property
    def devices(self):
        """Hall Bench Devices."""
        return _QApplication.instance().devices

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.move_axis_widget.close()
            self.timer.stop()
            self.closeDialogs()
            event.accept()
        except Exception:
            event.accept()

    def readValue(self, monitor=False):
        """Read value."""
        if len(self._data_labels) == 0:
            return

        if (not self.devices.voltx.connected or
           not self.devices.volty.connected or
           not self.devices.voltz.connected):
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Multimeters not connected.', _QMessageBox.Ok)
            return

        try:
            ts = _time.time()

            axis = self.move_axis_widget.selectedAxis()
            if axis is None:
                pos = _np.nan
            else:
                pos = self.devices.pmac.get_position(axis)
                if pos is None:
                    pos = _np.nan

            self.devices.voltx.send_command(
                self.devices.voltx.commands.end_gpib_always)
            voltx = float(self.devices.voltx.read_from_device()[:-2])
            voltx = voltx*self._data_mult_factor
            self._readings[self._data_labels[0]].append(voltx)

            self.devices.volty.send_command(
                self.devices.volty.commands.end_gpib_always)
            volty = float(self.devices.volty.read_from_device()[:-2])
            volty = volty*self._data_mult_factor
            self._readings[self._data_labels[1]].append(volty)

            self.devices.voltz.send_command(
                self.devices.voltz.commands.end_gpib_always)
            voltz = float(self.devices.voltz.read_from_device()[:-2])
            voltz = voltz*self._data_mult_factor
            self._readings[self._data_labels[2]].append(voltz)

            self.pos.append(pos)
            self.timestamp.append(ts)
            self.updateTableValues()
            self.updatePlot()

        except Exception:
            pass

    def resetMultimeters(self):
        """Reset multimeters."""
        if (not self.devices.voltx.connected or
           not self.devices.volty.connected or
           not self.devices.voltz.connected):
            _QMessageBox.critical(
                self, 'Failure',
                'Multimeters not connected.', _QMessageBox.Ok)
            return

        self.devices.voltx.reset()
        self.devices.volty.reset()
        self.devices.voltz.reset()
