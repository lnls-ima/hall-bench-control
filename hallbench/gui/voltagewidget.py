# -*- coding: utf-8 -*-

"""Voltage widget for the Hall Bench Control application."""

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

from hallbench.gui.auxiliarywidgets import (
    MoveAxisWidget as _MoveAxisWidget,
    TablePlotWidget as _TablePlotWidget,
    )


class VoltageWidget(_TablePlotWidget):
    """Voltage Widget class for the Hall Bench Control application."""

    _left_axis_1_label = 'Voltage [mV]'
    _left_axis_1_format = '{0:.6f}'
    _left_axis_1_data_labels = ['X [mV]', 'Y [mV]', 'Z [mV]']
    _left_axis_1_data_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    _voltage_mfactor = 1000  # [V] -> [mV]

    _right_axis_1_label = 'Axis Position'
    _right_axis_1_format = '{0:.4f}'
    _right_axis_1_data_labels = ['Position']
    _right_axis_1_data_colors = [(0, 0, 0)]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # add move axis widget
        self.move_axis_widget = _MoveAxisWidget(self)
        self.add_widgets_next_to_plot(self.move_axis_widget)

        # add check box add reset multimeters button
        self.chb_voltx = _QCheckBox(' X ')
        self.chb_volty = _QCheckBox(' Y ')
        self.chb_voltz = _QCheckBox(' Z ')
        self.pbt_reset = _QPushButton('Reset Multimeters')
        self.pbt_reset.clicked.connect(self.reset_multimeters)
        self.add_widgets_next_to_table([
            [self.chb_voltx, self.chb_volty, self.chb_voltz],
            [self.pbt_reset]])

        # Change default appearance
        self.set_table_column_size(140)

        # Hide right axis
        self.hide_right_axes()

        # Create reading thread
        self.wthread = _QThread()
        self.worker = ReadValueWorker(self._voltage_mfactor)
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
            self.move_axis_widget.close()
            self.timer.stop()
            self.wthread.quit()
            del self.wthread
            self.close_dialogs()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def check_connection(self, monitor=False):
        """Check devices connection."""
        if self.chb_voltx.isChecked() and not self.devices.voltx.connected:
            if not monitor:
                msg = 'Multimeter X not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        if self.chb_volty.isChecked() and not self.devices.volty.connected:
            if not monitor:
                msg = 'Multimeter Y not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        if self.chb_voltz.isChecked() and not self.devices.voltz.connected:
            if not monitor:
                msg = 'Multimeter Z not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        return True

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
            for i, label in enumerate(self._data_labels):
                self._readings[label].append(r[i])
            self.add_last_value_to_table()
            self.update_plot()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def read_value(self, monitor=False):
        """Read value."""
        if len(self._data_labels) == 0:
            return

        if not self.check_connection(monitor=monitor):
            return

        try:
            self.worker.pmac_axis = self.move_axis_widget.selected_axis()
            self.worker.voltx_enabled = self.chb_voltx.isChecked()
            self.worker.volty_enabled = self.chb_volty.isChecked()
            self.worker.voltz_enabled = self.chb_voltz.isChecked()
            self.wthread.start()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def reset_multimeters(self):
        """Reset multimeters."""
        if not self.check_connection():
            return

        try:
            self.blockSignals(True)
            _QApplication.setOverrideCursor(_Qt.WaitCursor)

            if self.chb_voltx.isChecked():
                self.devices.voltx.reset()

            if self.chb_volty.isChecked():
                self.devices.volty.reset()

            if self.chb_voltz.isChecked():
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

    def __init__(self, voltage_mfactor):
        """Initialize object."""
        self.pmac_axis = None
        self.voltx_enabled = False
        self.volty_enabled = False
        self.voltz_enabled = False
        self.timestamp = None
        self.reading = []
        self.voltage_mfactor = voltage_mfactor
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

            if self.voltx_enabled:
                voltx = float(self.devices.voltx.read_from_device()[:-2])
                voltx = voltx*self.voltage_mfactor
            else:
                voltx = _np.nan

            if self.volty_enabled:
                volty = float(self.devices.volty.read_from_device()[:-2])
                volty = volty*self.voltage_mfactor
            else:
                volty = _np.nan

            if self.voltz_enabled:
                voltz = float(self.devices.voltz.read_from_device()[:-2])
                voltz = voltz*self.voltage_mfactor
            else:
                voltz = _np.nan

            if self.pmac_axis is None:
                pos = _np.nan
            else:
                pos = self.devices.pmac.get_position(self.pmac_axis)
                if pos is None:
                    pos = _np.nan

            self.timestamp = ts
            self.reading.append(voltx)
            self.reading.append(volty)
            self.reading.append(voltz)
            self.reading.append(pos)
            self.finished.emit(True)

        except Exception:
            self.timestamp = None
            self.reading = []
            self.finished.emit(True)
