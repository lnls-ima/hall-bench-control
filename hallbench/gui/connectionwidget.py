# -*- coding: utf-8 -*-

"""Connection widget for the Hall Bench Control application."""

import os.path as _path
from PyQt4 import QtGui as _QtGui
import PyQt4.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.data.configuration import ConnectionConfig as _ConnectionConfig
from hallbench.data.utils import get_timestamp as _get_timestamp


class ConnectionWidget(_QtGui.QWidget):
    """Connection widget class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Setup the ui."""
        super(ConnectionWidget, self).__init__(parent)

        # setup the ui
        uifile = _getUiFile(__file__, self)
        self.ui = _uic.loadUi(uifile, self)

        # variables initialization
        self.configuration = None

        # create connections
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
            message = 'Invalid value for devices.'
            _QtGui.QMessageBox.critical(
                self, 'Failure', message, _QtGui.QMessageBox.Ok)
            return

        connect_status = self.devices.connect(self.configuration)
        self.updateLedStatus(connect_status)

        if not any(connect_status):
            message = 'Fail to connect devices.'
            _QtGui.QMessageBox.critical(
                self, 'Failure', message, _QtGui.QMessageBox.Ok)
            self.window().updateMainTabStatus(1, False)
            self.window().updateMainTabStatus(2, False)
            self.window().updateMainTabStatus(3, False)
            return
        else:
            self.window().updateMainTabStatus(1, True)
            self.window().updateMainTabStatus(2, True)

    def disconnectDevices(self):
        """Disconnect bench devices."""
        if self.devices is None:
            return

        disconnect_status = self.devices.disconnect()
        connect_status = [not s for s in disconnect_status]
        self.updateLedStatus(connect_status)

    def loadConfiguration(self):
        """Load configuration file to set connection parameters."""
        default_filename = self.ui.filename_le.text()
        filename = _QtGui.QFileDialog.getOpenFileName(
            self, caption='Open connection configuration file',
            directory=default_filename, filter="Text files (*.txt)")

        if len(filename) == 0:
            return

        try:
            self.configuration = _ConnectionConfig(filename)
        except Exception as e:
            _QtGui.QMessageBox.critical(
                self, 'Failure', str(e), _QtGui.QMessageBox.Ok)
            return

        self.ui.filename_le.setText(filename)

        self.ui.voltx_chb.setChecked(self.configuration.control_voltx_enable)
        self.ui.volty_chb.setChecked(self.configuration.control_volty_enable)
        self.ui.voltz_chb.setChecked(self.configuration.control_voltz_enable)

        self.ui.voltxaddress_sb.setValue(
            self.configuration.control_voltx_addr)
        self.ui.voltyaddress_sb.setValue(
            self.configuration.control_volty_addr)
        self.ui.voltzaddress_sb.setValue(
            self.configuration.control_voltz_addr)

        self.ui.pmac_chb.setChecked(self.configuration.control_pmac_enable)

        self.ui.multich_chb.setChecked(
            self.configuration.control_multich_enable)
        self.ui.multichaddress_sb.setValue(
            self.configuration.control_multich_addr)

        self.ui.colimator_chb.setChecked(
            self.configuration.control_colimator_enable)
        self.ui.colimatorport_cmb.setCurrentIndex(
            self.configuration.control_colimator_addr)

    def saveConfiguration(self):
        """Save connection parameters to file."""
        default_filename = self.ui.filename_le.text()
        filename = _QtGui.QFileDialog.getSaveFileName(
            self, caption='Save connection configuration file',
            directory=default_filename, filter="Text files (*.txt)")

        if len(filename) == 0:
            return

        if self.updateConfiguration():
            try:
                if not filename.endswith('.txt'):
                    filename = filename + '.txt'
                self.configuration.save_file(filename)
            except Exception as e:
                _QtGui.QMessageBox.critical(
                    self, 'Failure', str(e), _QtGui.QMessageBox.Ok)

    def saveConfigurationInMeasurementsDir(self):
        """Save configuration file in the measurements directory."""
        if self.directory is None:
            return

        if self.configuration is None:
            return

        try:
            timestamp = _get_timestamp()
            filename = timestamp + '_' + 'connection_configuration.txt'
            self.configuration.save_file(_path.join(
                self.directory, filename))
        except Exception:
            pass

    def updateConfiguration(self):
        """Update connection configuration parameters."""
        if self.configuration is None:
            self.configuration = _ConnectionConfig()

        voltx_enable = self.ui.voltx_chb.isChecked()
        volty_enable = self.ui.volty_chb.isChecked()
        voltz_enable = self.ui.voltz_chb.isChecked()
        self.configuration.control_voltx_enable = voltx_enable
        self.configuration.control_volty_enable = volty_enable
        self.configuration.control_voltz_enable = voltz_enable

        voltx_value = self.ui.voltxaddress_sb.value()
        volty_value = self.ui.voltyaddress_sb.value()
        voltz_value = self.ui.voltzaddress_sb.value()
        self.configuration.control_voltx_addr = voltx_value
        self.configuration.control_volty_addr = volty_value
        self.configuration.control_voltz_addr = voltz_value

        self.configuration.control_pmac_enable = self.ui.pmac_chb.isChecked()

        multich_enable = self.ui.multich_chb.isChecked()
        multich_addr = self.ui.multichaddress_sb.value()
        self.configuration.control_multich_enable = multich_enable
        self.configuration.control_multich_addr = multich_addr

        colimator_enable = self.ui.colimator_chb.isChecked()
        colimator_addr = self.ui.colimatorport_cmb.currentIndex()
        self.configuration.control_colimator_enable = colimator_enable
        self.configuration.control_colimator_addr = colimator_addr

        if self.configuration.valid_configuration():
            return True
        else:
            message = 'Invalid connection configuration.'
            _QtGui.QMessageBox.critical(
                self, 'Failure', message, _QtGui.QMessageBox.Ok)
            return False

    def updateLedStatus(self, status):
        """Update led status."""
        self.ui.voltxled_la.setEnabled(status[0])
        self.ui.voltyled_la.setEnabled(status[1])
        self.ui.voltzled_la.setEnabled(status[2])
        self.ui.pmacled_la.setEnabled(status[3])
        self.ui.multichled_la.setEnabled(status[4])

        if len(status) > 5:
            self.ui.colimatorled_la.setEnabled(status[5])
