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
    QVBoxLayout as _QVBoxLayout,
    )
from qtpy.QtCore import (
    QThread as _QThread,
    QObject as _QObject,
    Signal as _Signal,
    )

from hallbench.gui.moveaxiswidget import MoveAxisWidget as _MoveAxisWidget
from hallbench.gui.tableplotwidget import TablePlotWidget as _TablePlotWidget


class AngularErrorWidget(_TablePlotWidget):
    """Angular error widget class for the Hall Bench Control application."""

    _plot_label = 'Angular error [arcsec]'
    _data_format = '{0:.4f}'
    _data_labels = ['X-axis [arcsec]', 'Y-axis [arcsec]']
    _colors = [(255, 0, 0), (0, 255, 0)]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # add move axis widget
        self.move_axis_widget = _MoveAxisWidget(self)
        _layout = _QVBoxLayout()
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.addWidget(self.move_axis_widget)
        self.ui.widget_wg.setLayout(_layout)

        # add measurement type combo box
        _label = _QLabel("Measurement Type:")
        self.meastype_cmb = _QComboBox()
        self.meastype_cmb.addItems(["Absolute", "Relative"])
        self.ui.layout_lt.addWidget(_label)
        self.ui.layout_lt.addWidget(self.meastype_cmb)

        # variables initialisation
        self._position = []
        self.configureTable()

        # Change default appearance
        self.table_ta.horizontalHeader().setDefaultSectionSize(170)
        self.ui.read_btn.setText('Read Angular Error')
        self.ui.monitor_btn.setText('Monitor Angular Error')

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
            self._readings[self._data_labels[0]].append(r[2])
            self._readings[self._data_labels[1]].append(r[3])
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
            mt = self.meastype_cmb.currentText().lower()
            self.worker.measurement_type = mt
            self.thread.start()
        except Exception:
            pass


class ReadValueWorker(_QObject):
    """Read values worker."""

    finished = _Signal([bool])

    def __init__(self):
        """Initialize object."""
        self.pmac_axis = None
        self.measurement_type = None
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

            if self.measurement_type == 'relative':
                rl = self.devices.elcomat.get_relative_measurement()
            elif self.measurement_type == 'absolute':
                rl = self.devices.elcomat.get_absolute_measurement()
            else:
                rl = []
            rl = [r if r is not None else _np.nan for r in rl]

            self.reading.append(ts)
            self.reading.append(pos)
            if len(rl) == 2:
                self.reading.append(rl[0])
                self.reading.append(rl[1])
            else:
                self.reading.append(_np.nan)
                self.reading.append(_np.nan)
            self.finished.emit(True)

        except Exception:
            self.reading = []
            self.finished.emit(True)
