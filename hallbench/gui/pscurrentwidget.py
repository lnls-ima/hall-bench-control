# -*- coding: utf-8 -*-

"""Temperature widget for the Hall Bench Control application."""

import sys as _sys
import numpy as _np
import time as _time
import traceback as _traceback
from PyQt5.QtWidgets import (
    QApplication as _QApplication,
    QMessageBox as _QMessageBox,
    QPushButton as _QPushButton,
    QVBoxLayout as _QVBoxLayout,
    QWidget as _QWidget,
    )
from PyQt5.QtCore import (
    pyqtSignal as _pyqtSignal,
    QDateTime as _QDateTime,
    Qt as _Qt,
    )
import PyQt5.uic as _uic

import hallbench.gui.utils as _utils
from hallbench.gui.tableplotwidget import TablePlotWidget as _TablePlotWidget


class PSCurrentWidget(_TablePlotWidget):
    """Temperature Widget class for the Hall Bench Control application."""

    _plot_label = 'Current [A]'
    _data_format = '{0:.4f}'
    _data_labels = ['DCCT [A]', 'PS [A]']
    _colors = [(230, 25, 75), (60, 180, 75)]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # add channels_widget
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

    @property
    def devices(self):
        """Hall Bench Devices."""
        return _QApplication.instance().devices
    
    @property
    def power_supply_config(self):
        """Power supply configuration."""
        return _QApplication.instance().power_supply_config

    def configureDevices(self):
        """Configure channels for current measurement."""
        if not self.devices.multich.connected:
            _QMessageBox.critical(
                self, 'Failure',
                'Multichannel not connected.', _QMessageBox.Ok)
            return

        try:
            self.blockSignals(True)
            _QApplication.setOverrideCursor(_Qt.WaitCursor)
            
            ps_type = self.power_supply_config.ps_type
            if ps_type is not None:
                self.devices.ps.SetSlaveAdd(ps_type)
            else:
                self.blockSignals(False)
                _QApplication.restoreOverrideCursor()
                _QMessageBox.critical(
                    self, 'Failure',
                    'Invalid power supply configuration.', _QMessageBox.Ok)
                return       
    
            selected_channels = self.devices.multich.dcct_channels
            self._configured = self.devices.multich.configure(selected_channels)
            
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            
            if self._configured:
                self.configure_btn.setEnabled(False)
            else:
                self.configure_btn.setEnabled(True)
                msg = 'Failed to configure devices.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
        except Exception:
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            _traceback.print_exc(file=_sys.stdout)

    def enableConfigureButton(self):
        """Enable configure button."""
        self.configure_btn.setEnabled(True)

    def readValue(self, monitor=False):
        """Read value."""
        if not self.devices.multich.connected:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Multichannel not connected.', _QMessageBox.Ok)
            return

        if not self._configured:
            if not monitor:
                _QMessageBox.critical(
                    self, 'Failure',
                    'Multichannel not configured.', _QMessageBox.Ok)
            return

        try:
            ts = _time.time()
            channels = self.devices.multich.config_channels
            dcct_head = self.power_supply_config.dcct_head
            ps_type = self.power_supply_config.ps_type
                        
            rl = self.devices.multich.get_converted_readings(
                dcct_head=dcct_head)
            if len(rl) != len(channels):
                return
            readings = [r if _np.abs(r) < 1e37 else _np.nan for r in rl]

            label = 'PS'    
            if ps_type is not None:
                self.devices.ps.SetSlaveAdd(ps_type)
                ps_current = float(self.devices.ps.Read_iLoad1())
                self._readings[label].append(ps_current)
            else:
                self._readings[label].append(_np.nan)
            
            label = 'DCCT'
            dcct_current = readings[0]
            self._readings[label].append(dcct_current)

            self._timestamp.append(ts)
            self.addLastValueToTable()
            self.updatePlot()

        except Exception:
            pass