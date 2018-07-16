# -*- coding: utf-8 -*-

"""Polynomial table dialog for the Hall Bench Control application."""

from PyQt5.QtWidgets import (
    QDialog as _QDialog,
    QApplication as _QApplication,
    QTableWidgetItem as _QTableWidgetItem,
    )
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile


class PolynomialTableDialog(_QDialog):
    """Polynomial table class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui and create connections."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(__file__, self)
        self.ui = _uic.loadUi(uifile, self)

        self.probe_calibration = None
        self.clip = _QApplication.clipboard()

        # create connections
        self.ui.copysensorx_btn.clicked.connect(
            lambda: self.copyToClipboard('x'))
        self.ui.copysensory_btn.clicked.connect(
            lambda: self.copyToClipboard('y'))
        self.ui.copysensorz_btn.clicked.connect(
            lambda: self.copyToClipboard('z'))

        self.ui.sensorxprec_sb.valueChanged.connect(self.updateTableSensorX)
        self.ui.sensoryprec_sb.valueChanged.connect(self.updateTableSensorY)
        self.ui.sensorzprec_sb.valueChanged.connect(self.updateTableSensorZ)

    def copyToClipboard(self, sensor):
        """Copy table data to clipboard."""
        table = getattr(self.ui, 'sensor' + sensor + '_ta')
        text = ""
        for r in range(table.rowCount()):
            for c in range(table.columnCount()):
                text += str(table.item(r, c).text()) + "\t"
            text = text[:-1] + "\n"
        self.clip.setText(text)

    def show(self, probe_calibration):
        """Update calibration data object and show dialog."""
        self.probe_calibration = probe_calibration
        self.updateTables()
        super(PolynomialTableDialog, self).show()

    def updateTables(self):
        """Update table values."""
        if self.probe_calibration is None:
            return
        self.updateTableSensorX()
        self.updateTableSensorY()
        self.updateTableSensorZ()

    def updateTableSensorX(self):
        """Update sensor x table values."""
        precision = self.sensorxprec_sb.value()
        table = self.ui.sensorx_ta
        data = self.probe_calibration.sensorx.data

        formatstr = '{0:0.%if}' % precision
        table.setRowCount(0)
        for i in range(len(data)):
            table.setRowCount(i+1)
            row = data[i]
            for j in range(len(row)):
                table.setItem(i, j, _QTableWidgetItem(
                    formatstr.format(row[j])))

    def updateTableSensorY(self):
        """Update sensor y table values."""
        precision = self.sensoryprec_sb.value()
        table = self.ui.sensory_ta
        data = self.probe_calibration.sensory.data

        formatstr = '{0:0.%if}' % precision
        table.setRowCount(0)
        for i in range(len(data)):
            table.setRowCount(i+1)
            row = data[i]
            for j in range(len(row)):
                table.setItem(i, j, _QTableWidgetItem(
                    formatstr.format(row[j])))

    def updateTableSensorZ(self):
        """Update sensor z table values."""
        precision = self.sensorzprec_sb.value()
        table = self.ui.sensorz_ta
        data = self.probe_calibration.sensorz.data

        formatstr = '{0:0.%if}' % precision
        table.setRowCount(0)
        for i in range(len(data)):
            table.setRowCount(i+1)
            row = data[i]
            for j in range(len(row)):
                table.setItem(i, j, _QTableWidgetItem(
                    formatstr.format(row[j])))
