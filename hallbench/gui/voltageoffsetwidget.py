# -*- coding: utf-8 -*-

"""Voltage offset widget for the Hall Bench Control application."""

import numpy as _np
import time as _time
import datetime as _datetime
import warnings as _warnings
import pyqtgraph as _pyqtgraph
from PyQt5.QtWidgets import (
    QWidget as _QWidget,
    QMessageBox as _QMessageBox,
    QTableWidgetItem as _QTableWidgetItem,
    )
from PyQt5.QtCore import QTimer as _QTimer
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile


class VoltageOffsetWidget(_QWidget):
    """Voltage Offset Widget class for the Hall Bench Control application."""

    _position_format = '{0:0.4f}'
    _voltage_format = '{0:0.6f}'
    _voltage_mult_factor = 1000  # [V] -> [mV]

    def __init__(self, parent=None):
        """Set up the ui and signal/slot connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(__file__, self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialization
        self.timestamp = []
        self.voltx_values = []
        self.volty_values = []
        self.voltz_values = []

        # create connections
        self.ui.posax1_le.editingFinished.connect(
            lambda: self.setPositionStrFormat(self.ui.posax1_le))
        self.ui.posax2_le.editingFinished.connect(
            lambda: self.setPositionStrFormat(self.ui.posax2_le))
        self.ui.posax3_le.editingFinished.connect(
            lambda: self.setPositionStrFormat(self.ui.posax3_le))

        self.ui.move_btn.clicked.connect(self.moveToChamberPosition)
        self.ui.readvolt_btn.clicked.connect(self.readVoltage)
        self.ui.monitorvolt_btn.toggled.connect(self.monitorVoltage)
        self.ui.monitorstep_sb.valueChanged.connect(self.updateMonitorInterval)
        self.ui.monitorunit_cmb.currentIndexChanged.connect(
            self.updateMonitorInterval)
        self.ui.clear_btn.clicked.connect(self.clearVoltageValues)
        self.ui.remove_btn.clicked.connect(self.removeVoltageValue)

        # create timer to monitor voltage
        self.timer = _QTimer(self)
        self.updateMonitorInterval()
        self.timer.timeout.connect(self.readVoltage)

        # configure plot and table
        self.ui.voltoffset_ta.setAlternatingRowColors(True)
        self.configureGraph()

    @property
    def devices(self):
        """Hall Bench devices."""
        return self.window().devices

    @property
    def validPosition(self):
        """Return True if the chamber position is valid, False otherwise."""
        posax1 = self.ui.posax1_le.text()
        posax2 = self.ui.posax2_le.text()
        posax3 = self.ui.posax3_le.text()
        if len(posax1) == 0 or len(posax2) == 0 or len(posax3) == 0:
            return False
        else:
            return True

    def clearVoltageValues(self):
        """Clear all voltage values."""
        self.timestamp = []
        self.voltx_values = []
        self.volty_values = []
        self.voltz_values = []
        self.updateTableValues()
        self.updatePlot()

    def configureGraph(self):
        """Configure data plots."""
        self.ui.voltoffset_pw.clear()

        self.legend = _pyqtgraph.LegendItem(offset=(70, 30))
        self.legend.setParentItem(self.ui.voltoffset_pw.graphicsItem())
        self.legend.setAutoFillBackground(1)

        penx = (255, 0, 0)
        peny = (0, 255, 0)
        penz = (0, 0, 255)

        self.graphx = self.ui.voltoffset_pw.plotItem.plot(
            _np.array([]), _np.array([]), pen=penx,
            symbol='o', symbolPen=penx, symbolSize=4, symbolBrush=penx)

        self.graphy = self.ui.voltoffset_pw.plotItem.plot(
            _np.array([]), _np.array([]), pen=peny,
            symbol='o', symbolPen=peny, symbolSize=4, symbolBrush=peny)

        self.graphz = self.ui.voltoffset_pw.plotItem.plot(
            _np.array([]), _np.array([]), pen=penz,
            symbol='o', symbolPen=penz, symbolSize=4, symbolBrush=penz)

        self.ui.voltoffset_pw.setLabel('bottom', 'Time interval [s]')
        self.ui.voltoffset_pw.setLabel('left', 'Voltage [mV]')
        self.ui.voltoffset_pw.showGrid(x=True, y=True)

        self.legend.addItem(self.graphx, 'X')
        self.legend.addItem(self.graphy, 'Y')
        self.legend.addItem(self.graphz, 'Z')

    def monitorVoltage(self, checked):
        """Monitor voltage value."""
        if checked:
            self.ui.readvolt_btn.setEnabled(False)
            self.ui.move_btn.setEnabled(False)
            self.timer.start()
        else:
            self.timer.stop()
            self.ui.move_btn.setEnabled(True)
            self.ui.readvolt_btn.setEnabled(True)

    def moveToChamberPosition(self):
        """Move probe to the zero Gauss chamber position."""
        if not self.validPosition:
            _QMessageBox.critical(
                self, 'Failure', 'Invalid position.', _QMessageBox.Ok)
            return

        try:
            posax1 = float(self.ui.posax1_le.text())
            posax2 = float(self.ui.posax2_le.text())
            posax3 = float(self.ui.posax3_le.text())

            self.devices.pmac.move_axis(1, posax1)
            self.devices.pmac.move_axis(2, posax2)
            self.devices.pmac.move_axis(3, posax3)

        except Exception:
            _QMessageBox.critical(
                self, 'Failure', 'Failed to move probe.', _QMessageBox.Ok)

    def readVoltage(self):
        """Read voltage value."""
        self.timestamp.append(_time.time())

        self.devices.voltx.send_command(
            self.devices.voltx.commands.end_gpib_always)
        voltx = float(self.devices.voltx.read_from_device()[:-2])
        self.voltx_values.append(voltx)

        self.devices.volty.send_command(
            self.devices.volty.commands.end_gpib_always)
        volty = float(self.devices.volty.read_from_device()[:-2])
        self.volty_values.append(volty)

        self.devices.voltz.send_command(
            self.devices.voltz.commands.end_gpib_always)
        voltz = float(self.devices.voltz.read_from_device()[:-2])
        self.voltz_values.append(voltz)

        self.updateTableValues()
        self.updatePlot()

    def removeVoltageValue(self):
        """Remove voltage value from list."""
        selected = self.ui.voltoffset_ta.selectedItems()
        rows = [s.row() for s in selected]
        n = len(self.timestamp)

        self.timestamp = [self.timestamp[i] for i in range(n) if i not in rows]
        self.voltx_values = [
            self.voltx_values[i] for i in range(n) if i not in rows]
        self.volty_values = [
            self.volty_values[i] for i in range(n) if i not in rows]
        self.voltz_values = [
            self.voltz_values[i] for i in range(n) if i not in rows]

        self.updateTableValues(scrollDown=False)
        self.updatePlot()

    def setPositionStrFormat(self, obj):
        """Set position string format."""
        try:
            value = float(obj.text())
            obj.setText(self._position_format.format(value))
        except Exception:
            obj.setText('')

    def updateAvgStdValues(self):
        """Update average and STD voltage values."""
        if len(self.timestamp) == 0:
            self.ui.avgprobex_le.setText('')
            self.ui.stdprobex_le.setText('')

            self.ui.avgprobey_le.setText('')
            self.ui.stdprobey_le.setText('')

            self.ui.avgprobez_le.setText('')
            self.ui.stdprobez_le.setText('')
        else:
            avgprobex = _np.mean(
                _np.array(self.voltx_values))*self._voltage_mult_factor
            self.ui.avgprobex_le.setText(
                self._voltage_format.format(avgprobex))

            stdprobex = _np.std(
                _np.array(self.voltx_values))*self._voltage_mult_factor
            self.ui.stdprobex_le.setText(
                self._voltage_format.format(stdprobex))

            avgprobey = _np.mean(
                _np.array(self.volty_values))*self._voltage_mult_factor
            self.ui.avgprobey_le.setText(
                self._voltage_format.format(avgprobey))

            stdprobey = _np.std(
                _np.array(self.volty_values))*self._voltage_mult_factor
            self.ui.stdprobey_le.setText(
                self._voltage_format.format(stdprobey))

            avgprobez = _np.mean(
                _np.array(self.voltz_values))*self._voltage_mult_factor
            self.ui.avgprobez_le.setText(
                self._voltage_format.format(avgprobez))

            stdprobez = _np.std(
                _np.array(self.voltz_values))*self._voltage_mult_factor
            self.ui.stdprobez_le.setText(
                self._voltage_format.format(stdprobez))

    def updateMonitorInterval(self):
        """Update monitor interval value."""
        index = self.ui.monitorunit_cmb.currentIndex()
        if index == 0:
            mf = 1000
        elif index == 1:
            mf = 1000*60
        else:
            mf = 1000*3600
        self.timer.setInterval(self.ui.monitorstep_sb.value()*mf)

    def updatePlot(self):
        """Update plot values."""
        if len(self.timestamp) == 0:
            self.graphx.setData(_np.array([]), _np.array([]))
            self.graphy.setData(_np.array([]), _np.array([]))
            self.graphz.setData(_np.array([]), _np.array([]))
            return

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            timeinterval = _np.array(self.timestamp) - self.timestamp[0]
            vx = _np.array(self.voltx_values)*self._voltage_mult_factor
            vy = _np.array(self.volty_values)*self._voltage_mult_factor
            vz = _np.array(self.voltz_values)*self._voltage_mult_factor
            self.graphx.setData(timeinterval, vx)
            self.graphy.setData(timeinterval, vy)
            self.graphz.setData(timeinterval, vz)

    def updateTableValues(self, scrollDown=True):
        """Update table values."""
        n = len(self.timestamp)
        self.ui.voltoffset_ta.clearContents()
        self.ui.voltoffset_ta.setRowCount(n)

        for i in range(n):
            dt = _datetime.datetime.fromtimestamp(self.timestamp[i])
            date = dt.strftime("%d/%m/%Y")
            hour = dt.strftime("%H:%M:%S")
            vx = self.voltx_values[i]*self._voltage_mult_factor
            vy = self.volty_values[i]*self._voltage_mult_factor
            vz = self.voltz_values[i]*self._voltage_mult_factor
            self.ui.voltoffset_ta.setItem(i, 0, _QTableWidgetItem(date))
            self.ui.voltoffset_ta.setItem(i, 1, _QTableWidgetItem(hour))
            self.ui.voltoffset_ta.setItem(
                i, 2, _QTableWidgetItem(self._voltage_format.format(vx)))
            self.ui.voltoffset_ta.setItem(
                i, 3, _QTableWidgetItem(self._voltage_format.format(vy)))
            self.ui.voltoffset_ta.setItem(
                i, 4, _QTableWidgetItem(self._voltage_format.format(vz)))

        self.updateAvgStdValues()

        if scrollDown:
            vbar = self.voltoffset_ta.verticalScrollBar()
            vbar.setValue(vbar.maximum())
