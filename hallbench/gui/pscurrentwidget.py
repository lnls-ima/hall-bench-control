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

from hallbench.gui.tableplotwidget import TablePlotWidget as _TablePlotWidget


class PSCurrentWidget(_TablePlotWidget):
    """Temperature Widget class for the Hall Bench Control application."""

    _data_format = '{0:.4f}'
    _data_labels = ['DCCT [A]', 'PS [A]']
    _colors = [(230, 25, 75), (60, 180, 75)]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # add check box
        _layout = _QHBoxLayout()
        self.dcct_chb = _QCheckBox(' DCCT ')
        self.ps_chb = _QCheckBox(' Power Supply ')
        _layout.addWidget(self.dcct_chb)
        _layout.addWidget(self.ps_chb)
        self.ui.layout_lt.addLayout(_layout)
        self.ui.layout_lt.addSpacing(5)

        # add configure button
        self._configured = False
        self.configure_btn = _QPushButton('Configure Devices')
        self.configure_btn.setMinimumHeight(45)
        font = self.configure_btn.font()
        font.setBold(True)
        self.configure_btn.setFont(font)
        self.ui.layout_lt.addWidget(self.configure_btn)
        self.configure_btn.clicked.connect(self.configureDevices)

        # Change default appearance
        self.ui.widget_wg.hide()
        self.ui.table_ta.horizontalHeader().setDefaultSectionSize(200)
        self.ui.read_btn.setText('Read Current')
        self.ui.monitor_btn.setText('Monitor Current')

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

    @property
    def power_supply_config(self):
        """Power supply configuration."""
        return _QApplication.instance().power_supply_config

    def checkConnection(self, monitor=False):
        """Check devices connection."""
        if self.dcct_chb.isChecked() and not self.devices.dcct.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure', 'DCCT not connected.', _QMessageBox.Ok)
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

    def configureDevices(self):
        """Configure channels for current measurement."""
        if not self.checkConnection():
            return

        try:
            self.blockSignals(True)
            _QApplication.setOverrideCursor(_Qt.WaitCursor)

            ps_type = self.power_supply_config.ps_type
            if self.ps_chb.isChecked():
                if ps_type is not None:
                    self.devices.ps.SetSlaveAdd(ps_type)
                else:
                    self.blockSignals(False)
                    _QApplication.restoreOverrideCursor()
                    _QMessageBox.critical(
                        self, 'Failure',
                        'Invalid power supply configuration.', _QMessageBox.Ok)
                    return

            if self.dcct_chb.isChecked():
                self.devices.dcct.config()

            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()

        except Exception:
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            _traceback.print_exc(file=_sys.stdout)

    def getReading(self):
        """Get reading from worker thread."""
        try:
            r = self.worker.reading
            if len(r) == 0 or all([_np.isnan(ri) for ri in r[1:]]):
                return

            self._timestamp.append(r[0])
            self._readings[self._data_labels[0]].append(r[1])
            self._readings[self._data_labels[1]].append(r[2])
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
            self.worker.dcct_enabled = self.dcct_chb.isChecked()
            self.worker.ps_enabled = self.ps_chb.isChecked()
            self.thread.start()
        except Exception:
            pass


class ReadValueWorker(_QObject):
    """Read values worker."""

    finished = _Signal([bool])

    def __init__(self):
        """Initialize object."""
        self.dcct_enabled = False
        self.ps_enabled = False
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

            self.reading.append(ts)
            self.reading.append(dcct_current)
            self.reading.append(ps_current)
            self.finished.emit(True)

        except Exception:
            self.reading = []
            self.finished.emit(True)
