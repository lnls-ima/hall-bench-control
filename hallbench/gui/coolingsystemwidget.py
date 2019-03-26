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

from hallbench.gui.utils import plotItemAddRightAxis as _plotItemAddRightAxis
from hallbench.gui.tableplotwidget import TablePlotWidget as _TablePlotWidget


class CoolingSystemWidget(_TablePlotWidget):
    """Cooling System Widget class for the Hall Bench Control application."""

    _plot_label = 'Water Temperature [deg C]'
    _data_format = '{0:.4f}'
    _data_labels = ['PV1', 'PV2', 'Output']
    _colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # add check box
        _layout = _QHBoxLayout()
        self.pv1_chb = _QCheckBox(' PV1 ')
        self.pv2_chb = _QCheckBox(' PV2 ')
        self.output_chb = _QCheckBox(' Output ')
        _layout.addWidget(self.pv1_chb)
        _layout.addWidget(self.pv2_chb)
        _layout.addWidget(self.output_chb)
        self.ui.layout_lt.addLayout(_layout)

        # Change default appearance
        self.ui.widget_wg.hide()
        self.ui.table_ta.horizontalHeader().setDefaultSectionSize(200)
        self.ui.read_btn.setText('Read PVs')
        self.ui.monitor_btn.setText('Monitor PVs')

        self.right_axis = _plotItemAddRightAxis(self.ui.plot_pw.plotItem)
        self.right_axis.setLabel('Controller Output', color="#0000FF")
        self.right_axis.setStyle(showValues=True)
        self.right_axis_curve = _pyqtgraph.PlotCurveItem(
            [], [], pen=(0, 0, 255))
        self.right_axis.linkedView().addItem(self.right_axis_curve)
        self._graphs[self._data_labels[2]] = self.right_axis_curve

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
        if not self.devices.udc.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure', 'UDC not connected.', _QMessageBox.Ok)
            return False
        return True

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.timer.stop()
            self.thread.quit()
            del self.thread
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
            self._readings[self._data_labels[0]].append(r[1])
            self._readings[self._data_labels[1]].append(r[2])
            self._readings[self._data_labels[2]].append(r[3])
            self.addLastValueToTable()
            self.updatePlot()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def readValue(self, monitor=False):
        """Read value."""
        if len(self._data_labels) == 0:
            return

        if not self.checkConnection(monitor=monitor):
            return

        try:
            self.worker.pv1_enabled = self.pv1_chb.isChecked()
            self.worker.pv2_enabled = self.pv2_chb.isChecked()
            self.worker.output_enabled = self.output_chb.isChecked()
            self.thread.start()
        except Exception:
            pass


class ReadValueWorker(_QObject):
    """Read values worker."""

    finished = _Signal([bool])

    def __init__(self):
        """Initialize object."""
        self.pv1_enabled = False
        self.pv2_enabled = False
        self.output_enabled = False
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
            if self.pv1_enabled:
                pv1 = self.devices.udc.read_pv1()
            else:
                pv1 = _np.nan

            if self.pv2_enabled:
                pv2 = self.devices.udc.read_pv2()
            else:
                pv2 = _np.nan

            if self.output_enabled:
                output = self.devices.udc.read_co()
            else:
                output = _np.nan

            self.reading.append(ts)
            self.reading.append(pv1)
            self.reading.append(pv2)
            self.reading.append(output)
            self.finished.emit(True)

        except Exception:
            self.reading = []
            self.finished.emit(True)
