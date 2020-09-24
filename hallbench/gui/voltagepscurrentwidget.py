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
    QCheckBox as _QCheckBox,
    )
from qtpy.QtCore import (
    Qt as _Qt,
    QThread as _QThread,
    QObject as _QObject,
    Signal as _Signal,
    )

from hallbench.gui.auxiliarywidgets import TablePlotWidget as _TablePlotWidget
from hallbench.devices import (
    dcct as _dcct,
    volty as _volty,
    )


class VoltagePSCurrentWidget(_TablePlotWidget):
    """Power supply current class for the Hall Bench Control application."""

    _monitor_name = 'voltage_current'
    _left_axis_1_label = 'Current [A]'
    _left_axis_1_format = '{0:.4f}'
    _left_axis_1_data_labels = ['DCCT [A]']
    _left_axis_1_data_colors = [(255, 0, 0)]

    _right_axis_1_label = 'Voltage [mV]'
    _right_axis_1_format = '{0:.15f}'
    _right_axis_1_data_labels = ['Y [mV]']
    _right_axis_1_data_colors = [(0, 255, 0)]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # add check box and configure button
        self.pbt_configure = _QPushButton('Configure Devices')
        self.pbt_configure.clicked.connect(self.configure_devices)
        self.add_widgets_next_to_table([self.pbt_configure])

        # Create reading thread
        self.wthread = _QThread()
        self.worker = ReadValueWorker()
        self.worker.moveToThread(self.wthread)
        self.wthread.started.connect(self.worker.run)
        self.worker.finished.connect(self.wthread.quit)
        self.worker.finished.connect(self.get_reading)

    def check_connection(self, monitor=False):
        """Check devices connection."""
        if not _dcct.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure', 'DCCT not connected.', _QMessageBox.Ok)
            return False
        if not _volty.connected:
            if not monitor:
                msg = 'Multimeter Y not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
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

    def configure_devices(self):
        """Configure channels for current measurement."""
        if not self.check_connection():
            return

        try:
            self.blockSignals(True)
            _QApplication.setOverrideCursor(_Qt.WaitCursor)
            
            _volty.reset()
            
            _dcct.config()

            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()

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
            for i, label in enumerate(self._data_labels):
                self._readings[label].append(r[i])
            self.add_last_value_to_table()
            self.add_last_value_to_file()
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
            self.worker.dcct_enabled = True
            self.worker.volty_enabled = True
            self.wthread.start()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)


class ReadValueWorker(_QObject):
    """Read values worker."""

    finished = _Signal([bool])

    def __init__(self):
        """Initialize object."""
        self.dcct_enabled = False
        self.volty_enabled = False
        self.timestamp = None
        self.reading = []
        super().__init__()

    def run(self):
        """Read values from devices."""
        try:
            self.timestamp = None
            self.reading = []

            ts = _time.time()

            if self.dcct_enabled:
                dcct_current = _dcct.read_current()
            else:
                dcct_current = _np.nan

            if self.volty_enabled:
                volty = float(_volty.read_from_device()[:-2])
                volty = volty*1000
            else:
                volty = _np.nan

            self.timestamp = ts
            self.reading.append(dcct_current)
            self.reading.append(volty)
            self.finished.emit(True)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.timestamp = None
            self.reading = []
            self.finished.emit(True)
