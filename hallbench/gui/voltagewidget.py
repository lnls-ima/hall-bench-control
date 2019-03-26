# -*- coding: utf-8 -*-

"""Voltage offset widget for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import time as _time
import traceback as _traceback
from qtpy.QtWidgets import (
    QApplication as _QApplication,
    QMessageBox as _QMessageBox,
    QPushButton as _QPushButton,
    QVBoxLayout as _QVBoxLayout,
    QHBoxLayout as _QHBoxLayout,
    QCheckBox as _QCheckBox,
    )
from qtpy.QtCore import (
    Qt as _Qt,
    QThread as _QThread,
    QObject as _QObject,
    Signal as _Signal,
    )

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
        if self.voltx_chb.isChecked() and not self.devices.voltx.connected:
            if not monitor:
                msg = 'Multimeter X not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        if self.volty_chb.isChecked() and not self.devices.volty.connected:
            if not monitor:
                msg = 'Multimeter Y not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        if self.voltz_chb.isChecked() and not self.devices.voltz.connected:
            if not monitor:
                msg = 'Multimeter Z not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        return True

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.move_axis_widget.close()
            self.timer.stop()
            self.thread.quit()
            del self.thread
            self.closeDialogs()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def getReading(self):
        """Get reading from worker thread."""
        try:
            r = self.worker.reading
            if len(r) == 0 or all([_np.isnan(ri) for ri in r[1:]]):
                return

            self._timestamp.append(r[0])
            self._position.append(r[1])
            self._readings[self._data_labels[0]].append(
                r[2]*self._data_mult_factor)
            self._readings[self._data_labels[1]].append(
                r[3]*self._data_mult_factor)
            self._readings[self._data_labels[2]].append(
                r[4]*self._data_mult_factor)
            self.addLastValueToTable()
            self.updatePlot()
        except Exception:
            pass

    def readValue(self, monitor=False):
        """Read value."""
        if len(self._data_labels) == 0:
            return

        if not self.checkConnection(monitor=monitor):
            return

        try:
            self.worker.pmac_axis = self.move_axis_widget.selectedAxis()
            self.worker.voltx_enabled = self.voltx_chb.isChecked()
            self.worker.volty_enabled = self.volty_chb.isChecked()
            self.worker.voltz_enabled = self.voltz_chb.isChecked()
            self.thread.start()
        except Exception:
            pass

    def resetMultimeters(self):
        """Reset multimeters."""
        if not self.checkConnection():
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


class ReadValueWorker(_QObject):
    """Read values worker."""

    finished = _Signal([bool])

    def __init__(self):
        """Initialize object."""
        self.pmac_axis = None
        self.voltx_enabled = False
        self.volty_enabled = False
        self.voltz_enabled = False
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

            if self.pmac_axis is None:
                pos = _np.nan
            else:
                pos = self.devices.pmac.get_position(self.pmac_axis)
                if pos is None:
                    pos = _np.nan

            if self.voltx_enabled:
                voltx = float(self.devices.voltx.read_from_device()[:-2])
                voltx = voltx
            else:
                voltx = _np.nan

            if self.volty_enabled:
                volty = float(self.devices.volty.read_from_device()[:-2])
                volty = volty
            else:
                volty = _np.nan

            if self.voltz_enabled:
                voltz = float(self.devices.voltz.read_from_device()[:-2])
                voltz = voltz
            else:
                voltz = _np.nan

            self.reading.append(ts)
            self.reading.append(pos)
            self.reading.append(voltx)
            self.reading.append(volty)
            self.reading.append(voltz)
            self.finished.emit(True)

        except Exception:
            self.reading = []
            self.finished.emit(True)
