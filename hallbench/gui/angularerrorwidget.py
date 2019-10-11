# -*- coding: utf-8 -*-

"""Angular error widget for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import time as _time
import traceback as _traceback
from qtpy.QtWidgets import (
    QApplication as _QApplication,
    QComboBox as _QComboBox,
    QLabel as _QLabel,
    QMessageBox as _QMessageBox,
    QHBoxLayout as _QHBoxLayout,
    )
from qtpy.QtCore import (
    QThread as _QThread,
    QObject as _QObject,
    Signal as _Signal,
    )

from hallbench.gui.auxiliarywidgets import (
    MoveAxisWidget as _MoveAxisWidget,
    TablePlotWidget as _TablePlotWidget,
    )


class AngularErrorWidget(_TablePlotWidget):
    """Angular error widget class for the Hall Bench Control application."""

    _left_axis_1_label = 'Angular error [arcsec]'
    _left_axis_1_format = '{0:.4f}'
    _left_axis_1_data_labels = ['X-axis [arcsec]', 'Y-axis [arcsec]']
    _left_axis_1_data_colors = [(255, 0, 0), (0, 255, 0)]

    _right_axis_1_label = ''
    _right_axis_1_format = '{0:.4f}'
    _right_axis_1_data_labels = ['Position']
    _right_axis_1_data_colors = [(0, 0, 255)]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # add move axis widget
        self.move_axis_widget = _MoveAxisWidget(self)
        self.addWidgetsNextToPlot(self.move_axis_widget)

        # add measurement type combo box
        self.la_meastype = _QLabel("Measurement Type:")
        self.cmb_meastype = _QComboBox()
        self.cmb_meastype.addItems(["Absolute", "Relative"])
        self.addWidgetsNextToTable([self.la_meastype, self.cmb_meastype])

        # Change default appearance
        self.setTableColumnSize(150)

        # Hide right axis
        self.hideRightAxes()

        # Create reading thread
        self.wthread = _QThread()
        self.worker = ReadValueWorker()
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
        if not self.devices.elcomat.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Auto-collimator not connected.', _QMessageBox.Ok)
            return False
        return True

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.move_axis_widget.close()
            self.wthread.quit()
            super().closeEvent(event)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

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
            for i, label in enumerate(self._data_labels):
                self._readings[label].append(r[i])
            self.addLastValueToTable()
            self.updatePlot()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def readValue(self, monitor=False):
        """Read value."""
        if len(self._data_labels) == 0:
            return

        if not self.checkConnection(monitor=monitor):
            return

        try:
            self.worker.pmac_axis = self.move_axis_widget.selectedAxis()
            mt = self.cmb_meastype.currentText().lower()
            self.worker.measurement_type = mt
            self.wthread.start()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)


class ReadValueWorker(_QObject):
    """Read values worker."""

    finished = _Signal([bool])

    def __init__(self):
        """Initialize object."""
        self.pmac_axis = None
        self.measurement_type = None
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

            if self.measurement_type == 'relative':
                rl = self.devices.elcomat.get_relative_measurement()
            elif self.measurement_type == 'absolute':
                rl = self.devices.elcomat.get_absolute_measurement()
            else:
                rl = []
            rl = [r if r is not None else _np.nan for r in rl]

            if self.pmac_axis is None:
                pos = _np.nan
            else:
                pos = self.devices.pmac.get_position(self.pmac_axis)
                if pos is None:
                    pos = _np.nan

            self.timestamp = ts
            if len(rl) == 2:
                self.reading.append(rl[0])
                self.reading.append(rl[1])
            else:
                self.reading.append(_np.nan)
                self.reading.append(_np.nan)
            self.reading.append(pos)
            self.finished.emit(True)

        except Exception:
            self.timestamp = None
            self.reading = []
            self.finished.emit(True)
