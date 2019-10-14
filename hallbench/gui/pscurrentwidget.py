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
    QHBoxLayout as _QHBoxLayout,
    QCheckBox as _QCheckBox,
    )
from qtpy.QtCore import (
    Qt as _Qt,
    QThread as _QThread,
    QObject as _QObject,
    Signal as _Signal,
    )

from hallbench.gui.auxiliarywidgets import TablePlotWidget as _TablePlotWidget


class PSCurrentWidget(_TablePlotWidget):
    """Power supply current class for the Hall Bench Control application."""

    _left_axis_1_label = 'Current [A]'
    _left_axis_1_format = '{0:.4f}'
    _left_axis_1_data_labels = ['DCCT [A]', 'PS [A]']
    _left_axis_1_data_colors = [(255, 0, 0), (0, 255, 0)]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # add check box and configure button
        self.chb_dcct = _QCheckBox(' DCCT ')
        self.chb_ps = _QCheckBox(' Power Supply ')
        self.pbt_configure = _QPushButton('Configure Devices')
        self.pbt_configure.clicked.connect(self.configure_devices)
        self.add_widgets_next_to_table(
            [[self.chb_dcct, self.chb_ps], [self.pbt_configure]])

        # Create reading thread
        self.wthread = _QThread()
        self.worker = ReadValueWorker()
        self.worker.moveToThread(self.wthread)
        self.wthread.started.connect(self.worker.run)
        self.worker.finished.connect(self.wthread.quit)
        self.worker.finished.connect(self.get_reading)

    @property
    def devices(self):
        """Hall Bench Devices."""
        return _QApplication.instance().devices

    @property
    def power_supply_config(self):
        """Power supply configuration."""
        return _QApplication.instance().power_supply_config

    def check_connection(self, monitor=False):
        """Check devices connection."""
        if self.chb_dcct.isChecked() and not self.devices.dcct.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure', 'DCCT not connected.', _QMessageBox.Ok)
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

            ps_type = self.power_supply_config.ps_type
            if self.chb_ps.isChecked():
                if ps_type is not None:
                    self.devices.ps.SetSlaveAdd(ps_type)
                else:
                    self.blockSignals(False)
                    _QApplication.restoreOverrideCursor()
                    _QMessageBox.critical(
                        self, 'Failure',
                        'Invalid power supply configuration.', _QMessageBox.Ok)
                    return

            if self.chb_dcct.isChecked():
                self.devices.dcct.config()

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
            self.worker.dcct_enabled = self.chb_dcct.isChecked()
            self.worker.ps_enabled = self.chb_ps.isChecked()
            self.wthread.start()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)


class ReadValueWorker(_QObject):
    """Read values worker."""

    finished = _Signal([bool])

    def __init__(self):
        """Initialize object."""
        self.dcct_enabled = False
        self.ps_enabled = False
        self.timestamp = None
        self.reading = []
        super().__init__()

    @property
    def devices(self):
        """Hall Bench Devices."""
        return _QApplication.instance().devices

    @property
    def power_supply_config(self):
        """Power supply configuration."""
        return _QApplication.instance().power_supply_config

    def run(self):
        """Read values from devices."""
        try:
            self.timestamp = None
            self.reading = []

            ts = _time.time()
            dcct_head = self.power_supply_config.dcct_head
            ps_type = self.power_supply_config.ps_type

            if self.dcct_enabled:
                dcct_current = self.devices.dcct.read_current(
                    dcct_head=dcct_head)
            else:
                dcct_current = _np.nan

            if self.ps_enabled and ps_type is not None:
                self.devices.ps.SetSlaveAdd(ps_type)
                ps_current = float(self.devices.ps.Read_iLoad1())
            else:
                ps_current = _np.nan

            self.timestamp = ts
            self.reading.append(dcct_current)
            self.reading.append(ps_current)
            self.finished.emit(True)

        except Exception:
            self.timestamp = None
            self.reading = []
            self.finished.emit(True)
