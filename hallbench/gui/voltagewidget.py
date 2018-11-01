# -*- coding: utf-8 -*-

"""Voltage offset widget for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import time as _time
import traceback as _traceback
from PyQt5.QtWidgets import (
    QApplication as _QApplication,
    QMessageBox as _QMessageBox,
    QPushButton as _QPushButton,
    QVBoxLayout as _QVBoxLayout,
    QHBoxLayout as _QHBoxLayout,
    QCheckBox as _QCheckBox,
    )
from PyQt5.QtCore import Qt as _Qt

from hallbench.gui.moveaxiswidget import MoveAxisWidget as _MoveAxisWidget
from hallbench.gui.tableplotwidget import TablePlotWidget as _TablePlotWidget


class VoltageWidget(_TablePlotWidget):
    """Voltage Offset Widget class for the Hall Bench Control application."""

    _plot_label = 'Voltage [mV]'
    _data_mult_factor = 1000  # [V] -> [mV]
    _data_format = '{0:.6f}'
    _data_labels = ['X [mV]', 'Y [mV]', 'Z [mV]']
    _colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # add move axis widget
        self.move_axis_widget = _MoveAxisWidget(self)
        _layout = _QVBoxLayout()
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.addWidget(self.move_axis_widget)
        self.ui.widget_wg.setLayout(_layout)

        # add check box
        _layout = _QHBoxLayout()
        self.voltx_chb = _QCheckBox(' X ')
        self.volty_chb = _QCheckBox(' Y ')
        self.voltz_chb = _QCheckBox(' Z ')
        _layout.addWidget(self.voltx_chb)
        _layout.addWidget(self.volty_chb)
        _layout.addWidget(self.voltz_chb)
        self.ui.layout_lt.addLayout(_layout)
        self.ui.layout_lt.addSpacing(5)

        # add reset multimeters button
        self.reset_btn = _QPushButton('Reset Multimeters')
        self.reset_btn.setMinimumHeight(45)
        font = self.reset_btn.font()
        font.setBold(True)
        self.reset_btn.setFont(font)
        self.ui.layout_lt.addWidget(self.reset_btn)
        self.reset_btn.clicked.connect(self.resetMultimeters)

        # variables initialisation
        self._position = []
        self.configureTable()

        # Change default appearance
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
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def readValue(self, monitor=False):
        """Read value."""
        if len(self._data_labels) == 0:
            return

        if self.voltx_chb.isChecked() and not self.devices.voltx.connected:
            if not monitor:
                msg = 'Multimeter X not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        if self.volty_chb.isChecked() and not self.devices.volty.connected:
            if not monitor:
                msg = 'Multimeter Y not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        if self.voltz_chb.isChecked() and not self.devices.voltz.connected:
            if not monitor:
                msg = 'Multimeter Z not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
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

            if self.voltx_chb.isChecked():
                voltx = float(self.devices.voltx.read_from_device()[:-2])
                voltx = voltx*self._data_mult_factor
            else:
                voltx = _np.nan
            self._readings[self._data_labels[0]].append(voltx)

            if self.volty_chb.isChecked():
                volty = float(self.devices.volty.read_from_device()[:-2])
                volty = volty*self._data_mult_factor
            else:
                volty = _np.nan
            self._readings[self._data_labels[1]].append(volty)

            if self.voltz_chb.isChecked():
                voltz = float(self.devices.voltz.read_from_device()[:-2])
                voltz = voltz*self._data_mult_factor
            else:
                voltz = _np.nan
            self._readings[self._data_labels[2]].append(voltz)

            self._timestamp.append(ts)
            self._position.append(pos)
            self.addLastValueToTable()
            self.updatePlot()
        except Exception:
            pass

    def resetMultimeters(self):
        """Reset multimeters."""
        if self.voltx_chb.isChecked() and not self.devices.voltx.connected:
            msg = 'Multimeter X not connected.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        if self.volty_chb.isChecked() and not self.devices.volty.connected:
            msg = 'Multimeter Y not connected.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        if self.voltz_chb.isChecked() and not self.devices.voltz.connected:
            msg = 'Multimeter Z not connected.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        try:
            self.blockSignals(True)
            _QApplication.setOverrideCursor(_Qt.WaitCursor)

            if self.voltx_chb.isChecked():
                self.devices.voltx.reset()

            if self.volty_chb.isChecked():
                self.devices.volty.reset()

            if self.voltz_chb.isChecked():
                self.devices.voltz.reset()

            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()

        except Exception:
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            _traceback.print_exc(file=_sys.stdout)
