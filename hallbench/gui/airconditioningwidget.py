# -*- coding: utf-8 -*-

"""Cooling System widget for the Hall Bench Control application."""

import sys as _sys
import time as _time
import numpy as _np
import pyqtgraph as _pyqtgraph
import traceback as _traceback
from qtpy.QtWidgets import (
    QApplication as _QApplication,
    QMessageBox as _QMessageBox,
    QHBoxLayout as _QHBoxLayout,
    QCheckBox as _QCheckBox,
    )
from qtpy.QtCore import (
    QThread as _QThread,
    QObject as _QObject,
    Signal as _Signal,
    )

from hallbench.gui.auxiliarywidgets import TablePlotWidget as _TablePlotWidget


class AirConditioningWidget(_TablePlotWidget):
    """Air Conditioning Widget class for the Hall Bench Control application."""

    _left_axis_1_label = 'Temperature [deg C]'
    _left_axis_1_format = '{0:.4f}'
    _left_axis_1_data_labels = ['PV1', 'PV2']
    _left_axis_1_data_colors = [(255, 0, 0), (0, 255, 0)]

    _right_axis_1_label = 'Controller Output [%]'
    _right_axis_1_format = '{0:.4f}'
    _right_axis_1_data_labels = ['Output1', 'Output2']
    _right_axis_1_data_colors = [(0, 0, 255), (0, 255, 255)]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # add check box
        self.chb_pv1 = _QCheckBox(' PV1 ')
        self.chb_pv2 = _QCheckBox(' PV2 ')
        self.chb_output1 = _QCheckBox(' Output1')
        self.chb_output2 = _QCheckBox(' Output2')
        self.add_widgets_next_to_table(
            [self.chb_pv1, self.chb_pv2, self.chb_output1, self.chb_output2])

        # Change default appearance
        self.set_table_column_size(120)

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

    def closeEvent(self, event):
        """Overwrite closeEvent method."""
        try:
            self.wthread.quit()
            super().closeEvent(event)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def check_connection(self, monitor=False):
        """Check devices connection."""
        if not self.devices.air_udc.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure', 'UDC not connected.', _QMessageBox.Ok)
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
            self.worker.pv1_enabled = self.chb_pv1.isChecked()
            self.worker.pv2_enabled = self.chb_pv2.isChecked()
            self.worker.output1_enabled = self.chb_output1.isChecked()
            self.worker.output2_enabled = self.chb_output2.isChecked()
            self.wthread.start()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)


class ReadValueWorker(_QObject):
    """Read values worker."""

    finished = _Signal([bool])

    def __init__(self):
        """Initialize object."""
        self.pv1_enabled = False
        self.pv2_enabled = False
        self.output1_enabled = False
        self.output2_enabled = False
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

            if self.pv1_enabled:
                pv1 = self.devices.air_udc.read_pv1()
            else:
                pv1 = _np.nan

            if self.pv2_enabled:
                pv2 = self.devices.air_udc.read_pv2()
            else:
                pv2 = _np.nan

            if self.output1_enabled:
                output1 = self.devices.air_udc.read_output1()
            else:
                output1 = _np.nan

            if self.output2_enabled:
                output2 = self.devices.air_udc.read_output2()
            else:
                output2 = _np.nan

            self.timestamp = ts
            self.reading.append(pv1)
            self.reading.append(pv2)
            self.reading.append(output1)
            self.reading.append(output2)
            self.finished.emit(True)

        except Exception:
            self.timestamp = None
            self.reading = []
            self.finished.emit(True)
