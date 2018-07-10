# -*- coding: utf-8 -*-

"""Connection widget for the Hall Bench Control application."""

import os.path as _path
from PyQt5.QtWidgets import (
    QWidget as _QWidget,
    QFileDialog as _QFileDialog,
    QMessageBox as _QMessageBox,
    )
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.data.configuration import ConnectionConfig as _ConnectionConfig
from hallbench.data.utils import get_timestamp as _get_timestamp


class ConnectionWidget(_QWidget):
    """Connection widget class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Setup the ui."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(__file__, self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialization
        self.config = None

        # create signal/slot connections
        self.ui.loadconfig_btn.clicked.connect(self.loadConfiguration)
        self.ui.saveconfig_btn.clicked.connect(self.saveConfiguration)
        self.ui.connect_btn.clicked.connect(self.connectDevices)
        self.ui.disconnect_btn.clicked.connect(self.disconnectDevices)

    @property
    def devices(self):
        """Hall Bench Devices."""
        return self.window().devices

    @property
    def directory(self):
        """Directory to save files."""
        return self.window().directory

    def connectDevices(self):
        """Connect bench devices."""
        if not self.updateConfiguration():
            return

        if self.devices is None:
            return

        try:
            self.devices.connect(self.config)
            self.updateLedStatus()
            connected = self.connectionStatus()

            if not connected:
                message = 'Fail to connect devices.'
                _QMessageBox.critical(
                    self, 'Failure', message, _QMessageBox.Ok)

        except Exception:
            message = 'Fail to connect devices.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)

    def connectionStatus(self):
        """Connection status."""
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
        if self.devices is None:
            return

        try:
            self.devices.disconnect()
            self.updateLedStatus()

        except Exception:
            message = 'Fail to disconnect devices.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)

    def loadConfiguration(self):
        """Load configuration file to set connection parameters."""
        default_filename = self.ui.filename_le.text()
        filename = _QFileDialog.getOpenFileName(
            self, caption='Open connection configuration file',
            directory=default_filename, filter="Text files (*.txt)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            self.config = _ConnectionConfig(filename)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)
            return

        try:
            self.ui.filename_le.setText(filename)

            self.ui.pmac_chb.setChecked(self.config.pmac_enable)

            self.ui.voltx_chb.setChecked(self.config.voltx_enable)
            self.ui.voltxaddress_sb.setValue(self.config.voltx_address)

            self.ui.volty_chb.setChecked(self.config.volty_enable)
            self.ui.voltyaddress_sb.setValue(self.config.volty_address)

            self.ui.voltz_chb.setChecked(self.config.voltz_enable)
            self.ui.voltzaddress_sb.setValue(self.config.voltz_address)

            self.ui.multich_chb.setChecked(self.config.multich_enable)
            self.ui.multichaddress_sb.setValue(self.config.multich_address)

            self.ui.nmr_chb.setChecked(self.config.nmr_enable)
            self.ui.nmrport_cmb.setCurrentIndex(
                self.ui.nmrport_cmb.findText(self.config.nmr_port))
            self.ui.nmrbaudrate_cmb.setCurrentIndex(
                self.ui.nmrbaudrate_cmb.findText(
                    str(self.config.nmr_baudrate)))

            self.ui.collimator_chb.setChecked(self.config.collimator_enable)
            self.ui.collimatorport_cmb.setCurrentIndex(
                self.ui.collimatorport_cmb.findText(
                    self.config.collimator_port))

        except Exception:
            message = 'Fail to load configuration.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)

    def saveConfiguration(self):
        """Save connection parameters to file."""
        default_filename = self.ui.filename_le.text()
        filename = _QFileDialog.getSaveFileName(
            self, caption='Save connection configuration file',
            directory=default_filename, filter="Text files (*.txt)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            if self.updateConfiguration():
                if not filename.endswith('.txt'):
                    filename = filename + '.txt'
                self.config.save_file(filename)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def saveConfigurationInMeasurementsDir(self):
        """Save configuration file in the measurements directory."""
        if self.directory is None:
            return

        if self.config is None:
            return

        try:
            timestamp = _get_timestamp()
            filename = timestamp + '_' + 'connection_configuration.txt'
            self.config.save_file(_path.join(
                self.directory, filename))
        except Exception:
            pass

    def updateConfiguration(self):
        """Update connection configuration parameters."""
        if self.config is None:
            self.config = _ConnectionConfig()

        try:
            self.config.pmac_enable = self.ui.pmac_chb.isChecked()

            self.config.voltx_enable = self.ui.voltx_chb.isChecked()
            self.config.voltx_address = self.ui.voltxaddress_sb.value()

            self.config.volty_enable = self.ui.volty_chb.isChecked()
            self.config.volty_address = self.ui.voltyaddress_sb.value()

            self.config.voltz_enable = self.ui.voltz_chb.isChecked()
            self.config.voltz_address = self.ui.voltzaddress_sb.value()

            self.config.multich_enable = self.ui.multich_chb.isChecked()
            self.config.multich_address = self.ui.multichaddress_sb.value()

            self.config.nmr_enable = self.ui.nmr_chb.isChecked()
            self.config.nmr_port = self.ui.nmrport_cmb.currentText()
            self.config.nmr_baudrate = int(
                self.ui.nmrbaudrate_cmb.currentText())

            self.config.collimator_enable = self.ui.collimator_chb.isChecked()
            self.config.collimator_port = (
                self.ui.collimatorport_cmb.currentText())

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
                self.ui.pmacled_la.setEnabled(False)
            else:
                self.ui.pmacled_la.setEnabled(pmac_connected)

            self.ui.voltxled_la.setEnabled(self.devices.voltx.connected)
            self.ui.voltyled_la.setEnabled(self.devices.volty.connected)
            self.ui.voltzled_la.setEnabled(self.devices.voltz.connected)
            self.ui.multichled_la.setEnabled(self.devices.multich.connected)
            self.ui.nmrled_la.setEnabled(self.devices.nmr.connected)

        except Exception:
            message = 'Fail to update led status.'
            _QMessageBox.critical(
                self, 'Failure', message, _QMessageBox.Ok)
