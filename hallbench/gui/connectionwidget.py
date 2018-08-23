# -*- coding: utf-8 -*-

"""Connection widget for the Hall Bench Control application."""

import os.path as _path
from PyQt4.QtGui import (
    QWidget as _QWidget,
    QFileDialog as _QFileDialog,
    QMessageBox as _QMessageBox,
    QApplication as _QApplication,
    )
from PyQt4.QtCore import Qt as _Qt
import PyQt4.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.data.configuration import ConnectionConfig as _ConnectionConfig


class ConnectionWidget(_QWidget):
    """Connection widget class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialization
        self.config = _ConnectionConfig()

        # create signal/slot connections
        self.ui.loadfile_btn.clicked.connect(self.loadFile)
        self.ui.savefile_btn.clicked.connect(self.saveFile)
        self.ui.loaddb_btn.clicked.connect(self.loadDB)
        self.ui.savedb_btn.clicked.connect(self.saveDB)
        self.ui.connect_btn.clicked.connect(self.connectDevices)
        self.ui.disconnect_btn.clicked.connect(self.disconnectDevices)

    @property
    def devices(self):
        """Hall Bench Devices."""
        return self.window().devices

    @property
    def database(self):
        """Database filename."""
        return self.window().database

    def connectDevices(self):
        """Connect bench devices."""
        if not self.updateConfiguration():
            return

        self.blockSignals(True)
        _QApplication.setOverrideCursor(_Qt.WaitCursor)

        try:
            self.devices.connect(self.config)
            self.updateLedStatus()
            connected = self.connectionStatus()

            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()

            if not connected:
                message = 'Fail to connect devices.'
                _QMessageBox.critical(
                    self, 'Failure', message, _QMessageBox.Ok)

        except Exception:
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            message = 'Fail to connect devices.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)

        self.window().updateMainTabStatus()

    def connectionStatus(self):
        """Return the connection status."""
        try:
            if self.config.pmac_enable:
                pmac_connected = self.devices.pmac.connected
                if pmac_connected is None or pmac_connected is False:
                    return False

            if self.config.voltx_enable and not self.devices.voltx.connected:
                return False

            if self.config.volty_enable and not self.devices.volty.connected:
                return False

            if self.config.voltz_enable and not self.devices.voltz.connected:
                return False

            if (self.config.multich_enable and
               not self.devices.multich.connected):
                return False

            if self.config.nmr_enable and not self.devices.nmr.connected:
                return False

            return True

        except Exception:
            return False

    def disconnectDevices(self):
        """Disconnect bench devices."""
        try:
            self.devices.disconnect()
            self.updateLedStatus()

        except Exception:
            message = 'Fail to disconnect devices.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)

    def loadDB(self):
        """Load configuration from database to set parameters."""
        self.ui.filename_le.setText("")

        try:
            idn = int(self.ui.idn_le.text())
        except Exception:
            _QMessageBox.critical(
                self, 'Failure', 'Invalid database ID.', _QMessageBox.Ok)
            return

        try:
            self.config = _ConnectionConfig(database=self.database, idn=idn)
        except Exception as e:
            _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)
            return

        self.load()

    def loadFile(self):
        """Load configuration file to set parameters."""
        self.ui.idn_le.setText("")

        default_filename = self.ui.filename_le.text()
        filename = _QFileDialog.getOpenFileName(
            self, caption='Open connection configuration file',
            directory=default_filename, filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            self.config = _ConnectionConfig(filename)
        except Exception as e:
            _QMessageBox.critical(self, 'Failure', str(e), _QMessageBox.Ok)
            return

        self.ui.filename_le.setText(filename)
        self.load()

    def load(self):
        """Load configuration file to set connection parameters."""
        try:
            self.ui.pmac_enable_chb.setChecked(self.config.pmac_enable)

            self.ui.voltx_enable_chb.setChecked(self.config.voltx_enable)
            self.ui.voltx_address_sb.setValue(self.config.voltx_address)

            self.ui.volty_enable_chb.setChecked(self.config.volty_enable)
            self.ui.volty_address_sb.setValue(self.config.volty_address)

            self.ui.voltz_enable_chb.setChecked(self.config.voltz_enable)
            self.ui.voltz_address_sb.setValue(self.config.voltz_address)

            self.ui.multich_enable_chb.setChecked(self.config.multich_enable)
            self.ui.multich_address_sb.setValue(self.config.multich_address)

            self.ui.nmr_enable_chb.setChecked(self.config.nmr_enable)
            self.ui.nmr_port_cmb.setCurrentIndex(
                self.ui.nmr_port_cmb.findText(self.config.nmr_port))
            self.ui.nmr_baudrate_cmb.setCurrentIndex(
                self.ui.nmr_baudrate_cmb.findText(
                    str(self.config.nmr_baudrate)))

            self.ui.collimator_enable_chb.setChecked(
                self.config.collimator_enable)
            self.ui.collimator_port_cmb.setCurrentIndex(
                self.ui.collimator_port_cmb.findText(
                    self.config.collimator_port))

        except Exception:
            message = 'Fail to load configuration.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)

    def saveDB(self):
        """Save connection parameters to database."""
        if self.database is not None and _path.isfile(self.database):
            try:
                if self.updateConfiguration():
                    self.config.save_to_database(self.database)
            except Exception as e:
                _QMessageBox.critical(
                    self, 'Failure', str(e), _QMessageBox.Ok)
        else:
            msg = 'Invalid database filename.'
            _QMessageBox.critical(
                self, 'Failure', msg, _QMessageBox.Ok)

    def saveFile(self):
        """Save connection parameters to file."""
        default_filename = self.ui.filename_le.text()
        filename = _QFileDialog.getSaveFileName(
            self, caption='Save connection configuration file',
            directory=default_filename, filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            if self.updateConfiguration():
                if (not filename.endswith('.txt')
                   and not filename.endswith('.dat')):
                    filename = filename + '.txt'
                self.config.save_file(filename)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def updateConfiguration(self):
        """Update connection configuration parameters."""
        self.config = _ConnectionConfig()

        try:
            self.config.pmac_enable = self.ui.pmac_enable_chb.isChecked()

            self.config.voltx_enable = self.ui.voltx_enable_chb.isChecked()
            self.config.voltx_address = self.ui.voltx_address_sb.value()

            self.config.volty_enable = self.ui.volty_enable_chb.isChecked()
            self.config.volty_address = self.ui.volty_address_sb.value()

            self.config.voltz_enable = self.ui.voltz_enable_chb.isChecked()
            self.config.voltz_address = self.ui.voltz_address_sb.value()

            self.config.multich_enable = self.ui.multich_enable_chb.isChecked()
            self.config.multich_address = self.ui.multich_address_sb.value()

            self.config.nmr_enable = self.ui.nmr_enable_chb.isChecked()
            self.config.nmr_port = self.ui.nmr_port_cmb.currentText()
            self.config.nmr_baudrate = int(
                self.ui.nmr_baudrate_cmb.currentText())

            self.config.collimator_enable = (
                self.ui.collimator_enable_chb.isChecked())
            self.config.collimator_port = (
                self.ui.collimator_port_cmb.currentText())

        except Exception:
            self.config = _ConnectionConfig()

        if self.config.valid_data():
            return True
        else:
            message = 'Invalid connection configuration.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
            return False

    def updateLedStatus(self):
        """Update led status."""
        try:
            pmac_connected = self.devices.pmac.connected
            if pmac_connected is None:
                self.ui.pmac_led_la.setEnabled(False)
            else:
                self.ui.pmac_led_la.setEnabled(pmac_connected)

            self.ui.voltx_led_la.setEnabled(self.devices.voltx.connected)
            self.ui.volty_led_la.setEnabled(self.devices.volty.connected)
            self.ui.voltz_led_la.setEnabled(self.devices.voltz.connected)
            self.ui.multich_led_la.setEnabled(self.devices.multich.connected)
            self.ui.nmr_led_la.setEnabled(self.devices.nmr.connected)

        except Exception:
            message = 'Fail to update led status.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
