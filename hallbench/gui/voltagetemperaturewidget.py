# -*- coding: utf-8 -*-

"""Voltage and Temperature widget for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import time as _time
import traceback as _traceback
import pyqtgraph as _pyqtgraph
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

from hallbench.gui.auxiliarywidgets import TablePlotWidget as _TablePlotWidget


class VoltageTempWidget(_TablePlotWidget):
    """Voltage and Temperature Widget class."""

    _left_axis_1_label = 'Voltage [mV]'       
    _left_axis_1_format = '{0:.6f}'
    _left_axis_1_data_labels = ['X [mV]', 'Y [mV]', 'Z [mV]']
    _left_axis_1_data_colors = [(230, 25, 75), (60, 180, 75), (0, 130, 200)]
    _voltage_mfactor = 1000  # [V] -> [mV]

    _right_axis_1_label = 'Temperature [deg C]'
    _right_axis_1_format = '{0:.4f}'
    _right_axis_1_data_labels = ['CH101', 'CH102', 'CH103', 'CH105']
    _right_axis_1_data_colors = [
        (245, 130, 48), (145, 30, 180), (255, 225, 25), (0, 0, 0)]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # add check box and configure button
        self.voltx_chb = _QCheckBox(' X ')
        self.volty_chb = _QCheckBox(' Y ')
        self.voltz_chb = _QCheckBox(' Z ')
        self.ch101_chb = _QCheckBox('CH 101')
        self.ch102_chb = _QCheckBox('CH 102')
        self.ch103_chb = _QCheckBox('CH 103')
        self.ch105_chb = _QCheckBox('CH 105')
        self.config_btn = _QPushButton('Configure Devices')
        self.config_btn.clicked.connect(self.configureDevices)
        self.addWidgetsNextToTable([
            [self.voltx_chb, self.volty_chb, self.voltz_chb],
            [self.ch101_chb, self.ch102_chb, self.ch103_chb, self.ch105_chb],
            [self.config_btn]])
 
        # Change default appearance
        self.setTableColumnSize(95)
 
        # Create reading thread
        self.wthread = _QThread()
        self.worker = ReadValueWorker(self._voltage_mfactor)
        self.worker.moveToThread(self.wthread)
        self.wthread.started.connect(self.worker.run)
        self.worker.finished.connect(self.wthread.quit)
        self.worker.finished.connect(self.getReading)
          
        self.wait = None
        self.updateWait()
        self.monitorstep_sb.valueChanged.connect(self.updateWait)
        self.monitorunit_cmb.currentIndexChanged.connect(self.updateWait)

    @property
    def devices(self):
        """Hall Bench Devices."""
        return _QApplication.instance().devices

    def checkConnection(self, monitor=False):
        """Check devices connection."""
        if self.voltx_chb.isChecked() and not self.devices.voltx.connected:
            if not monitor:
                msg = 'Multimeter X not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        if self.volty_chb.isChecked() and not self.devices.volty.connected:
            if not monitor:
                msg = 'Multimeter Y not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        if self.voltz_chb.isChecked() and not self.devices.voltz.connected:
            if not monitor:
                msg = 'Multimeter Z not connected.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

        if not self.devices.multich.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Multichannel not connected.', _QMessageBox.Ok)
            return False

        return True

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.timer.stop()
            self.wthread.quit()
            del self.wthread
            self.closeDialogs()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def configureDevices(self):
        """Configure devices."""
        if not self.checkConnection():
            return

        try:
            self.blockSignals(True)
            _QApplication.setOverrideCursor(_Qt.WaitCursor)

            if self.voltx_chb.isChecked():
                self.devices.voltx.reset()

            if self.volty_chb.isChecked():
                self.devices.volty.reset()

            if self.voltz_chb.isChecked():
                self.devices.voltz.reset()

            selected_channels = []
            if self.ch101_chb.isChecked():
                selected_channels.append('101')
            if self.ch102_chb.isChecked():
                selected_channels.append('102')
            if self.ch103_chb.isChecked():
                selected_channels.append('103')
            if self.ch105_chb.isChecked():
                selected_channels.append('105')
                
            self.devices.multich.configure(selected_channels, wait=self.wait)

            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()

        except Exception:
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            _traceback.print_exc(file=_sys.stdout)

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
            self.worker.voltx_enabled = self.voltx_chb.isChecked()
            self.worker.volty_enabled = self.volty_chb.isChecked()
            self.worker.voltz_enabled = self.voltz_chb.isChecked()
            self.worker.ch101_enabled = self.ch101_chb.isChecked()
            self.worker.ch102_enabled = self.ch102_chb.isChecked()
            self.worker.ch103_enabled = self.ch103_chb.isChecked()
            self.worker.ch105_enabled = self.ch105_chb.isChecked()
            self.worker.wait = self.wait
            self.wthread.start()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def updateWait(self):
        """Update wait value."""
        self.wait = self.monitorstep_sb.value()/2            


class ReadValueWorker(_QObject):
    """Read values worker."""

    finished = _Signal([bool])

    def __init__(self, voltage_mfactor):
        """Initialize object."""
        self.pmac_axis = None
        self.voltx_enabled = False
        self.volty_enabled = False
        self.voltz_enabled = False
        self.ch101_enabled = False
        self.ch102_enabled = False
        self.ch103_enabled = False
        self.ch105_enabled = False
        self.wait = None
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

            if any(
                [self.ch101_enabled, self.ch102_enabled,
                 self.ch103_enabled, self.ch105_enabled]):
                rl = self.devices.multich.get_converted_readings(
                    wait=self.wait)
                rl = [r if _np.abs(r) < 1e37 else _np.nan for r in rl]           
                config_channels = self.devices.multich.config_channels
            
                count = 0
                if '101' in config_channels and self.ch101_enabled:
                    ch101 = rl[count]
                    count = count + 1
                else:
                    ch101 = _np.nan
                
                if '102' in config_channels and self.ch102_enabled:
                    ch102 = rl[count]
                    count = count + 1
                else:
                    ch102 = _np.nan
                
                if '103' in config_channels and self.ch103_enabled:
                    ch103 = rl[count]
                    count = count + 1
                else:
                    ch103 = _np.nan
                
                if '105' in config_channels and self.ch105_enabled:
                    ch105 = rl[count]
                    count = count + 1                
                else:
                    ch105 = _np.nan
            else:
                ch101 = _np.nan
                ch102 = _np.nan
                ch103 = _np.nan
                ch105 = _np.nan
            
            self.timestamp = ts
            self.reading.append(voltx)
            self.reading.append(volty)
            self.reading.append(voltz)
            self.reading.append(ch101)
            self.reading.append(ch102)
            self.reading.append(ch103)
            self.reading.append(ch105)
            
            self.finished.emit(True)

        except Exception:
            self.timestamp = None
            self.reading = []
            self.finished.emit(True)