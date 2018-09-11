# -*- coding: utf-8 -*-

"""Angular error widget for the Hall Bench Control application."""

import numpy as _np
import time as _time
from PyQt5.QtWidgets import (
    QApplication as _QApplication,
    QComboBox as _QComboBox,
    QLabel as _QLabel,
    QMessageBox as _QMessageBox,
    QVBoxLayout as _QVBoxLayout,
    )

from hallbench.gui.moveaxiswidget import MoveAxisWidget as _MoveAxisWidget
from hallbench.gui.tableplotwidget import TablePlotWidget as _TablePlotWidget


class AngularErrorWidget(_TablePlotWidget):
    """Angular error widget class for the Hall Bench Control application."""

    _data_format = '{0:.4f}'
    _data_labels = ['X-axis [arcsec]', 'Y-axis [arcsec]']
    _yaxis_label = 'Angular error [arcsec]'
    _colors = {
        'X-axis [arcsec]': (255, 0, 0),
        'Y-axis [arcsec]': (0, 255, 0),
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

        _label = _QLabel("Measurement Type:")
        self.meastype_cmb = _QComboBox()
        self.meastype_cmb.addItems(["Absolute", "Relative"])
        self.ui.layout_lt.addWidget(_label)
        self.ui.layout_lt.addWidget(self.meastype_cmb)

        self.position = []
        col_labels = ['Date', 'Time', 'Position']
        for label in self._data_labels:
            col_labels.append(label)
        self.ui.table_ta.setColumnCount(len(col_labels))
        self.ui.table_ta.setHorizontalHeaderLabels(col_labels)
        self.ui.table_ta.setAlternatingRowColors(True)
        self.table_ta.horizontalHeader().setDefaultSectionSize(170)

        self.ui.read_btn.setText('Read Angular Error')
        self.ui.monitor_btn.setText('Monitor Angular Error')

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

        if not self.devices.elcomat.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Auto-collimator not connected.', _QMessageBox.Ok)
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

            if self.meastype_cmb.currentText().lower() == 'relative':
                _rl = self.devices.elcomat.get_relative_measurement()
            else:
                _rl = self.devices.elcomat.get_absolute_measurement()
            if len(_rl) != len(self._data_labels):
                return

            readings = [r if r is not None else _np.nan for r in _rl]

            for i in range(len(self._data_labels)):
                label = self._data_labels[i]
                value = readings[i]
                self._readings[label].append(value)

            self.position.append(pos)
            self.timestamp.append(ts)
            self.updateTableValues()
            self.updatePlot()

        except Exception:
            pass
