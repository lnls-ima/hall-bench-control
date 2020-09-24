# -*- coding: utf-8 -*-

"""Temperature widget for the Hall Bench Control application."""

import os as _os
import sys as _sys
import numpy as _np
import time as _time
import traceback as _traceback
from qtpy.QtWidgets import (
    QLabel as _QLabel,
    QCheckBox as _QCheckBox,
    QLineEdit as _QLineEdit,
    QMessageBox as _QMessageBox,
    QPushButton as _QPushButton,
    QSizePolicy as _QSizePolicy,
    )
from qtpy.QtCore import (
    Signal as _Signal,
    QThread as _QThread,
    QObject as _QObject,
    )

from hallbench.gui import utils as _utils
from hallbench.gui.auxiliarywidgets import (
    TableAnalysisDialog as _TableAnalysisDialog,
    TablePlotWidget as _TablePlotWidget,
    )
from hallbench.devices import (
    dcct as _dcct,
    ps as _ps,
    )


class CurrentWidget(_TablePlotWidget):
    """Power supply current class for the Hall Bench Control application."""

    _monitor_name = 'current'
    _left_axis_1_label = 'Current [A]'
    _left_axis_1_format = '{0:.8f}'
    _left_axis_1_data_labels = ['DCCT [A]', 'PS [A]']
    _left_axis_1_data_colors = [(255, 0, 0), (0, 255, 0)]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        size_policy = _QSizePolicy(
            _QSizePolicy.Maximum, _QSizePolicy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)

        # add check box and configure button
        self.chb_dcct = _QCheckBox(' DCCT ')
        self.chb_ps = _QCheckBox(' Power Supply ')
        label_range = _QLabel('Multimeter Range:')
        label_nplc = _QLabel('Multimeter NPLC:')
        label_autozero = _QLabel('Multimeter Autozero:')
        self.le_range = _QLineEdit()
        self.le_range.setText('10')
        self.le_range.setSizePolicy(size_policy)
        self.le_nplc = _QLineEdit()
        self.le_nplc.setText('10')
        self.le_nplc.setSizePolicy(size_policy)
        self.le_autozero = _QLineEdit()
        self.le_autozero.setText('OFF')
        self.le_autozero.setSizePolicy(size_policy)
        self.btn_config = _QPushButton('Configure')
        self.add_widgets_next_to_table([
            [self.chb_dcct, self.chb_ps], 
            [label_range, self.le_range], 
            [label_nplc, self.le_nplc], 
            [label_autozero, self.le_autozero], 
            [self.btn_config]])

        self.btn_config.clicked.connect(self.configure_devices)

        # Create reading thread
        self.wthread = _QThread()
        self.worker = ReadValueWorker()
        self.worker.moveToThread(self.wthread)
        self.wthread.started.connect(self.worker.run)
        self.worker.finished.connect(self.wthread.quit)
        self.worker.finished.connect(self.get_reading)

    def check_connection(self):
        """Check devices connection."""
        if self.chb_dcct.isChecked() and not _dcct.connected:
            return False
        return True

    def configure_devices(self):
        if not self.check_connection():
            return
        
        try:
            if self.chb_ps.isChecked() and _ps.ps_type is not None:
                _ps.SetSlaveAdd(_ps.ps_type)

            if self.chb_dcct.isChecked():
                range = self.le_range.text()
                nplc = self.le_nplc.text()
                autozero = self.le_autozero.text()
                _dcct.send_command('CONF:VOLT:DC ' + range + ', DEF')
                _dcct.send_command('VOLT:NPLC ' + nplc)
                _dcct.send_command('SENS:ZERO:AUTO ' + autozero)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            
    def closeEvent(self, event):
        """Close widget."""
        try:
            self.wthread.quit()
            super().closeEvent(event)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

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

        if not self.check_connection():
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure', 'DCCT not connected.', _QMessageBox.Ok)
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

    def run(self):
        """Read values from devices."""
        try:
            self.timestamp = None
            self.reading = []

            ts = _time.time()

            if self.dcct_enabled:
                dcct_current = _dcct.read_fast()
            else:
                dcct_current = _np.nan

            if self.ps_enabled:
                ps_type = _ps.ps_type
                if ps_type is not None:
                    if ps_type == 2:
                        _ps.SetSlaveAdd(1)
                    else:
                        _ps.SetSlaveAdd(ps_type)
                    ps_current = float(_ps.read_iload1())
                    if ps_current > 10000 or ps_current < -10000:
                        ps_current = _np.nan
                    if _np.abs(ps_current) < 1e-15:
                        ps_current = _np.nan
                else:
                    ps_current = _np.nan
            else:
                ps_current = _np.nan

            self.timestamp = ts
            self.reading.append(dcct_current)
            self.reading.append(ps_current)
            self.finished.emit(True)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.timestamp = None
            self.reading = []
            self.finished.emit(True)
