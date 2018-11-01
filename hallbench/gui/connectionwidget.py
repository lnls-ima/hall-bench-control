# -*- coding: utf-8 -*-

"""Connection widget for the Hall Bench Control application."""

import os as _os
import sys as _sys
import traceback as _traceback
import serial.tools.list_ports as _list_ports
from PyQt5.QtWidgets import (
    QWidget as _QWidget,
    QFileDialog as _QFileDialog,
    QMessageBox as _QMessageBox,
    QApplication as _QApplication,
    )
from PyQt5.QtCore import Qt as _Qt
import PyQt5.uic as _uic

from hallbench.gui.utils import getUiFile as _getUiFile


class ConnectionWidget(_QWidget):
    """Connection widget class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        self.connectSignalSlots()
        self.updateSerialPorts()

        self.updateConnectionIDs()

    @property
    def connection_config(self):
        """Return the connection configuration."""
        return _QApplication.instance().connection_config

    @property
    def devices(self):
        """Hall Bench Devices."""
        return _QApplication.instance().devices

    @property
    def database(self):
        """Database filename."""
        return _QApplication.instance().database

    @property
    def directory(self):
        """Return the default directory."""
        return _QApplication.instance().directory

    def clearLoadOptions(self):
        """Clear load options."""
        self.ui.filename_le.setText("")
        self.ui.idn_cmb.setCurrentIndex(-1)

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.devices.disconnect()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def connectDevices(self):
        """Connect bench devices."""
        if not self.updateConfiguration():
            return

        self.blockSignals(True)
        _QApplication.setOverrideCursor(_Qt.WaitCursor)

        try:
            self.devices.connect(self.connection_config)
            self.updateLedStatus()
            connected = self.connectionStatus()

            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()

            if not connected:
                msg = 'Failed to connect devices.'
                _QMessageBox.critical(
                    self, 'Failure', msg, _QMessageBox.Ok)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            msg = 'Failed to connect devices.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def connectionStatus(self):
        """Return the connection status."""
        try:
            if self.connection_config.pmac_enable:
                pmac_connected = self.devices.pmac.connected
                if pmac_connected is None or pmac_connected is False:
                    return False

            if (self.connection_config.voltx_enable and
               not self.devices.voltx.connected):
                return False

            if (self.connection_config.volty_enable and
               not self.devices.volty.connected):
                return False

            if (self.connection_config.voltz_enable and
               not self.devices.voltz.connected):
                return False

            if (self.connection_config.multich_enable and
               not self.devices.multich.connected):
                return False

            if (self.connection_config.nmr_enable and
               not self.devices.nmr.connected):
                return False

            if (self.connection_config.elcomat_enable and
               not self.devices.elcomat.connected):
                return False

            if (self.connection_config.dcct_enable and
               not self.devices.dcct.connected):
                return False

            if (self.connection_config.ps_enable and
               not self.devices.ps.ser.is_open):
                return False

            if (self.connection_config.udc_enable and
               not self.devices.udc.connected):
                return False

            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return False

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.pmac_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.ui.voltx_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.ui.voltx_address_sb.valueChanged.connect(self.clearLoadOptions)
        self.ui.volty_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.ui.volty_address_sb.valueChanged.connect(self.clearLoadOptions)
        self.ui.voltz_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.ui.voltz_address_sb.valueChanged.connect(self.clearLoadOptions)
        self.ui.multich_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.ui.multich_address_sb.valueChanged.connect(self.clearLoadOptions)
        self.ui.nmr_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.ui.nmr_port_cmb.currentIndexChanged.connect(
            self.clearLoadOptions)
        self.ui.nmr_baudrate_cmb.currentIndexChanged.connect(
            self.clearLoadOptions)
        self.ui.elcomat_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.ui.elcomat_port_cmb.currentIndexChanged.connect(
            self.clearLoadOptions)
        self.ui.elcomat_baudrate_cmb.currentIndexChanged.connect(
            self.clearLoadOptions)
        self.ui.dcct_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.ui.dcct_address_sb.valueChanged.connect(self.clearLoadOptions)
        self.ui.ps_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.ui.ps_port_cmb.currentIndexChanged.connect(self.clearLoadOptions)
        self.ui.udc_enable_chb.stateChanged.connect(self.clearLoadOptions)
        self.ui.udc_port_cmb.currentIndexChanged.connect(
            self.clearLoadOptions)
        self.ui.udc_baudrate_cmb.currentIndexChanged.connect(
            self.clearLoadOptions)

        self.ui.idn_cmb.currentIndexChanged.connect(self.enableLoadDB)
        self.ui.update_idn_btn.clicked.connect(self.updateConnectionIDs)

        self.ui.loadfile_btn.clicked.connect(self.loadFile)
        self.ui.savefile_btn.clicked.connect(self.saveFile)
        self.ui.loaddb_btn.clicked.connect(self.loadDB)
        self.ui.savedb_btn.clicked.connect(self.saveDB)
        self.ui.connect_btn.clicked.connect(self.connectDevices)
        self.ui.disconnect_btn.clicked.connect(self.disconnectDevices)

    def disconnectDevices(self):
        """Disconnect bench devices."""
        try:
            self.devices.disconnect()
            self.updateLedStatus()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to disconnect devices.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def enableLoadDB(self):
        """Enable button to load configuration from database."""
        if self.ui.idn_cmb.currentIndex() != -1:
            self.ui.loaddb_btn.setEnabled(True)
        else:
            self.ui.loaddb_btn.setEnabled(False)

    def loadDB(self):
        """Load configuration from database to set parameters."""
        self.ui.filename_le.setText("")

        try:
            idn = int(self.ui.idn_cmb.currentText())
        except Exception:
            _QMessageBox.critical(
                self, 'Failure', 'Invalid database ID.', _QMessageBox.Ok)
            return

        self.updateConnectionIDs()
        idx = self.ui.idn_cmb.findText(str(idn))
        if idx == -1:
            self.ui.idn_cmb.setCurrentIndex(-1)
            _QMessageBox.critical(
                self, 'Failure', 'Invalid database ID.', _QMessageBox.Ok)
            return

        try:
            self.connection_config.clear()
            self.connection_config.read_from_database(self.database, idn)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to read connection from database.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        self.load()
        self.ui.idn_cmb.setCurrentIndex(self.ui.idn_cmb.findText(str(idn)))
        self.ui.loaddb_btn.setEnabled(False)

    def loadFile(self):
        """Load configuration file to set parameters."""
        self.ui.idn_cmb.setCurrentIndex(-1)

        default_filename = self.ui.filename_le.text()
        if len(default_filename) == 0:
            default_filename = self.directory
        elif len(_os.path.split(default_filename)[0]) == 0:
            default_filename = _os.path.join(self.directory, default_filename)

        filename = _QFileDialog.getOpenFileName(
            self, caption='Open connection configuration file',
            directory=default_filename, filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            self.connection_config.clear()
            self.connection_config.read_file(filename)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to read connection from file.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        self.load()
        self.ui.filename_le.setText(filename)

    def load(self):
        """Load configuration file to set connection parameters."""
        try:
            self.ui.pmac_enable_chb.setChecked(
                self.connection_config.pmac_enable)

            self.ui.voltx_enable_chb.setChecked(
                self.connection_config.voltx_enable)
            self.ui.voltx_address_sb.setValue(
                self.connection_config.voltx_address)

            self.ui.volty_enable_chb.setChecked(
                self.connection_config.volty_enable)
            self.ui.volty_address_sb.setValue(
                self.connection_config.volty_address)

            self.ui.voltz_enable_chb.setChecked(
                self.connection_config.voltz_enable)
            self.ui.voltz_address_sb.setValue(
                self.connection_config.voltz_address)

            self.ui.multich_enable_chb.setChecked(
                self.connection_config.multich_enable)
            self.ui.multich_address_sb.setValue(
                self.connection_config.multich_address)

            self.ui.nmr_enable_chb.setChecked(
                self.connection_config.nmr_enable)
            self.ui.nmr_port_cmb.setCurrentIndex(
                self.ui.nmr_port_cmb.findText(
                    self.connection_config.nmr_port))
            self.ui.nmr_baudrate_cmb.setCurrentIndex(
                self.ui.nmr_baudrate_cmb.findText(
                    str(self.connection_config.nmr_baudrate)))

            self.ui.elcomat_enable_chb.setChecked(
                self.connection_config.elcomat_enable)
            self.ui.elcomat_port_cmb.setCurrentIndex(
                self.ui.elcomat_port_cmb.findText(
                    self.connection_config.elcomat_port))
            self.ui.elcomat_baudrate_cmb.setCurrentIndex(
                self.ui.elcomat_baudrate_cmb.findText(
                    str(self.connection_config.elcomat_baudrate)))

            self.ui.dcct_enable_chb.setChecked(
                self.connection_config.dcct_enable)
            self.ui.dcct_address_sb.setValue(
                self.connection_config.dcct_address)

            self.ui.ps_enable_chb.setChecked(
                self.connection_config.ps_enable)
            self.ui.ps_port_cmb.setCurrentIndex(
                self.ui.ps_port_cmb.findText(
                    self.connection_config.ps_port))

            self.ui.udc_enable_chb.setChecked(
                self.connection_config.udc_enable)
            self.ui.udc_port_cmb.setCurrentIndex(
                self.ui.udc_port_cmb.findText(
                    self.connection_config.udc_port))
            self.ui.udc_baudrate_cmb.setCurrentIndex(
                self.ui.udc_baudrate_cmb.findText(
                    str(self.connection_config.udc_baudrate)))

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to load configuration.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def saveDB(self):
        """Save connection parameters to database."""
        self.ui.idn_cmb.setCurrentIndex(-1)
        if self.database is not None and _os.path.isfile(self.database):
            try:
                if self.updateConfiguration():
                    idn = self.connection_config.save_to_database(
                        self.database)
                    self.ui.idn_cmb.addItem(str(idn))
                    self.ui.idn_cmb.setCurrentIndex(self.ui.idn_cmb.count()-1)
                    self.ui.loaddb_btn.setEnabled(False)
            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                msg = 'Failed to save connection to file.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
        else:
            msg = 'Invalid database filename.'
            _QMessageBox.critical(
                self, 'Failure', msg, _QMessageBox.Ok)

    def saveFile(self):
        """Save connection parameters to file."""
        default_filename = self.ui.filename_le.text()
        if len(default_filename) == 0:
            default_filename = self.directory
        elif len(_os.path.split(default_filename)[0]) == 0:
            default_filename = _os.path.join(self.directory, default_filename)

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
                self.connection_config.save_file(filename)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save connection to file.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def updateConnectionIDs(self):
        """Update connection IDs in combo box."""
        current_text = self.ui.idn_cmb.currentText()
        load_enabled = self.ui.loaddb_btn.isEnabled()
        self.ui.idn_cmb.clear()
        try:
            idns = self.connection_config.get_table_column(
                self.database, 'id')
            self.ui.idn_cmb.addItems([str(idn) for idn in idns])
            if len(current_text) == 0:
                self.ui.idn_cmb.setCurrentIndex(self.ui.idn_cmb.count()-1)
                self.ui.loaddb_btn.setEnabled(True)
            else:
                self.ui.idn_cmb.setCurrentText(current_text)
                self.ui.loaddb_btn.setEnabled(load_enabled)
        except Exception:
            pass

    def updateConfiguration(self):
        """Update connection configuration parameters."""
        self.connection_config.clear()

        try:
            self.connection_config.pmac_enable = (
                self.ui.pmac_enable_chb.isChecked())

            self.connection_config.voltx_enable = (
                self.ui.voltx_enable_chb.isChecked())
            self.connection_config.voltx_address = (
                self.ui.voltx_address_sb.value())

            self.connection_config.volty_enable = (
                self.ui.volty_enable_chb.isChecked())
            self.connection_config.volty_address = (
                self.ui.volty_address_sb.value())

            self.connection_config.voltz_enable = (
                self.ui.voltz_enable_chb.isChecked())
            self.connection_config.voltz_address = (
                self.ui.voltz_address_sb.value())

            self.connection_config.multich_enable = (
                self.ui.multich_enable_chb.isChecked())
            self.connection_config.multich_address = (
                self.ui.multich_address_sb.value())

            self.connection_config.nmr_enable = (
                self.ui.nmr_enable_chb.isChecked())
            self.connection_config.nmr_port = (
                self.ui.nmr_port_cmb.currentText())
            self.connection_config.nmr_baudrate = int(
                self.ui.nmr_baudrate_cmb.currentText())

            self.connection_config.elcomat_enable = (
                self.ui.elcomat_enable_chb.isChecked())
            self.connection_config.elcomat_port = (
                self.ui.elcomat_port_cmb.currentText())
            self.connection_config.elcomat_baudrate = int(
                self.ui.elcomat_baudrate_cmb.currentText())

            self.connection_config.dcct_enable = (
                self.ui.dcct_enable_chb.isChecked())
            self.connection_config.dcct_address = (
                self.ui.dcct_address_sb.value())

            self.connection_config.ps_enable = (
                self.ui.ps_enable_chb.isChecked())
            self.connection_config.ps_port = (
                self.ui.ps_port_cmb.currentText())

            self.connection_config.udc_enable = (
                self.ui.udc_enable_chb.isChecked())
            self.connection_config.udc_port = (
                self.ui.udc_port_cmb.currentText())
            self.connection_config.udc_baudrate = int(
                self.ui.udc_baudrate_cmb.currentText())

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.connection_config.clear()

        if self.connection_config.valid_data():
            return True
        else:
            msg = 'Invalid connection configuration.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
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
            self.ui.elcomat_led_la.setEnabled(self.devices.elcomat.connected)
            self.ui.dcct_led_la.setEnabled(self.devices.dcct.connected)
            self.ui.ps_led_la.setEnabled(self.devices.ps.ser.is_open)
            self.ui.udc_led_la.setEnabled(self.devices.udc.connected)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

    def updateSerialPorts(self):
        """Update avaliable serial ports."""
        _l = [p[0] for p in _list_ports.comports()]

        if len(_l) == 0:
            return

        _ports = []
        _s = ''
        _k = str
        if 'COM' in _l[0]:
            _s = 'COM'
            _k = int

        for key in _l:
            _ports.append(key.strip(_s))
        _ports.sort(key=_k)
        _ports = [_s + key for key in _ports]

        self.ui.ps_port_cmb.clear()
        self.ui.ps_port_cmb.addItems(_ports)

        self.ui.nmr_port_cmb.clear()
        self.ui.nmr_port_cmb.addItems(_ports)

        self.ui.elcomat_port_cmb.clear()
        self.ui.elcomat_port_cmb.addItems(_ports)

        self.ui.udc_port_cmb.clear()
        self.ui.udc_port_cmb.addItems(_ports)
