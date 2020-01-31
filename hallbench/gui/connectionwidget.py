# -*- coding: utf-8 -*-

"""Connection widget for the Hall Bench Control application."""

import os as _os
import sys as _sys
import traceback as _traceback
import serial.tools.list_ports as _list_ports
from qtpy.QtWidgets import (
    QWidget as _QWidget,
    QMessageBox as _QMessageBox,
    QApplication as _QApplication,
    )
from qtpy.QtCore import Qt as _Qt
import qtpy.uic as _uic

from hallbench.gui.utils import get_ui_file as _get_ui_file
import hallbench.data.configuration as _configuration
from hallbench.devices import (
    pmac as _pmac,
    voltx as _voltx,
    volty as _volty,
    voltz as _voltz,
    multich as _multich,
    nmr as _nmr,
    elcomat as _elcomat,
    dcct as _dcct,
    water_udc as _water_udc,
    air_udc as _air_udc,
    ps as _ps
    )


class ConnectionWidget(_QWidget):
    """Connection widget class for the Hall Bench Control application."""

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)

        # setup the ui
        uifile = _get_ui_file(self)
        self.ui = _uic.loadUi(uifile, self)

        self.connection_config = _configuration.ConnectionConfig()

        self.connect_signal_slots()
        self.update_serial_ports()
        self.update_connection_ids()

    @property
    def database_name(self):
        """Database name."""
        return _QApplication.instance().database_name

    @property
    def mongo(self):
        """MongoDB database."""
        return _QApplication.instance().mongo

    @property
    def server(self):
        """Server for MongoDB database."""
        return _QApplication.instance().server

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.disconnect_devices(msgbox=False)
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def clear_load_options(self):
        """Clear load options."""
        self.ui.cmb_idn.setCurrentIndex(-1)

    def connect_devices(self):
        """Connect bench devices."""
        if not self.update_configuration():
            return

        self.blockSignals(True)
        _QApplication.setOverrideCursor(_Qt.WaitCursor)

        try:
            if self.connection_config.pmac_enable:
                _pmac.connect()

            if self.connection_config.voltx_enable:
                _voltx.connect(
                    self.connection_config.voltx_address)

            if self.connection_config.volty_enable:
                _volty.connect(
                    self.connection_config.volty_address)

            if self.connection_config.voltz_enable:
                _voltz.connect(
                    self.connection_config.voltz_address)

            if self.connection_config.multich_enable:
                _multich.connect(
                    self.connection_config.multich_address)

            if self.connection_config.nmr_enable:
                _nmr.connect(
                    self.connection_config.nmr_port,
                    self.connection_config.nmr_baudrate)

            if self.connection_config.elcomat_enable:
                _elcomat.connect(
                    self.connection_config.elcomat_port,
                    self.connection_config.elcomat_baudrate)

            if self.connection_config.dcct_enable:
                _dcct.connect(
                    self.connection_config.dcct_address)

            if self.connection_config.water_udc_enable:
                _water_udc.connect(
                    self.connection_config.water_udc_port,
                    self.connection_config.water_udc_baudrate,
                    self.connection_config.water_udc_slave_address)

            if self.connection_config.air_udc_enable:
                _air_udc.connect(
                    self.connection_config.air_udc_port,
                    self.connection_config.air_udc_baudrate,
                    self.connection_config.air_udc_slave_address)

            if self.connection_config.ps_enable:
                _ps.Connect(self.connection_config.ps_port)

            self.update_led_status()
            connected = self.connection_status()

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

    def connection_status(self):
        """Return the connection status."""
        try:
            if self.connection_config.pmac_enable:
                pmac_connected = _pmac.connected
                if pmac_connected is None or pmac_connected is False:
                    return False

            if self.connection_config.voltx_enable and not _voltx.connected:
                return False

            if self.connection_config.volty_enable and not _volty.connected:
                return False

            if self.connection_config.voltz_enable and not _voltz.connected:
                return False

            if (self.connection_config.multich_enable and
                    not _multich.connected):
                return False

            if self.connection_config.nmr_enable and not _nmr.connected:
                return False

            if (self.connection_config.elcomat_enable and
                    not _elcomat.connected):
                return False

            if self.connection_config.dcct_enable and not _dcct.connected:
                return False

            if (self.connection_config.ps_enable and
                    not _ps.ser.is_open):
                return False

            if (self.connection_config.water_udc_enable and
                    not _water_udc.connected):
                return False

            if (self.connection_config.air_udc_enable and
                    not _air_udc.connected):
                return False

            return True

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            return False

    def connect_signal_slots(self):
        """Create signal/slot connections."""
        chbs = [
            self.ui.chb_voltx_enable,
            self.ui.chb_volty_enable,
            self.ui.chb_voltz_enable,
            self.ui.chb_pmac_enable,
            self.ui.chb_multich_enable,
            self.ui.chb_ps_enable,
            self.ui.chb_nmr_enable,
            self.ui.chb_elcomat_enable,
            self.ui.chb_dcct_enable,
            self.ui.chb_water_udc_enable,
            self.ui.chb_air_udc_enable,
            ]
        for chb in chbs:
            chb.stateChanged.connect(self.clear_load_options)

        sbs = [
            self.ui.sb_voltx_address,
            self.ui.sb_volty_address,
            self.ui.sb_voltz_address,
            self.ui.sb_multich_address,
            self.ui.sb_dcct_address,
            self.ui.sb_water_udc_slave_address,
            self.ui.sb_air_udc_slave_address,
            ]
        for sb in sbs:
            sb.valueChanged.connect(self.clear_load_options)

        cmbs = [
            self.ui.cmb_ps_port,
            self.ui.cmb_nmr_port,
            self.ui.cmb_nmr_baudrate,
            self.ui.cmb_elcomat_port,
            self.ui.cmb_elcomat_baudrate,
            self.ui.cmb_water_udc_port,
            self.ui.cmb_water_udc_baudrate,
            self.ui.cmb_air_udc_port,
            self.ui.cmb_air_udc_baudrate,
            ]
        for cmb in cmbs:
            cmb.currentIndexChanged.connect(self.clear_load_options)

        self.ui.cmb_idn.currentIndexChanged.connect(self.enable_load_db)
        self.ui.tbt_update_idn.clicked.connect(self.update_connection_ids)
        self.ui.pbt_load_db.clicked.connect(self.load_db)
        self.ui.tbt_save_db.clicked.connect(self.save_db)
        self.ui.pbt_connect.clicked.connect(self.connect_devices)
        self.ui.pbt_disconnect.clicked.connect(self.disconnect_devices)

    def disconnect_devices(self, msgbox=True):
        """Disconnect bench devices."""
        try:
            _pmac.disconnect()
            _voltx.disconnect()
            _volty.disconnect()
            _voltz.disconnect()
            _multich.disconnect()
            _nmr.disconnect()
            _elcomat.disconnect()
            _dcct.disconnect()
            _water_udc.disconnect()
            _air_udc.disconnect()
            _ps.Disconnect()
            self.update_led_status()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            if msgbox:
                msg = 'Failed to disconnect devices.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def enable_load_db(self):
        """Enable button to load configuration from database."""
        if self.ui.cmb_idn.currentIndex() != -1:
            self.ui.pbt_load_db.setEnabled(True)
        else:
            self.ui.pbt_load_db.setEnabled(False)

    def load_db(self):
        """Load configuration from database to set parameters."""
        try:
            idn = int(self.ui.cmb_idn.currentText())
        except Exception:
            _QMessageBox.critical(
                self, 'Failure', 'Invalid database ID.', _QMessageBox.Ok)
            return

        try:
            self.update_connection_ids()
            idx = self.ui.cmb_idn.findText(str(idn))
            if idx == -1:
                self.ui.cmb_idn.setCurrentIndex(-1)
                _QMessageBox.critical(
                    self, 'Failure', 'Invalid database ID.', _QMessageBox.Ok)
                return
            
            self.connection_config.clear()
            self.connection_config.db_update_database(
                self.database_name, mongo=self.mongo, server=self.server)
            self.connection_config.db_read(idn)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to read connection from database.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        self.load()
        self.ui.cmb_idn.setCurrentIndex(self.ui.cmb_idn.findText(str(idn)))
        self.ui.pbt_load_db.setEnabled(False)

    def load(self):
        """Load configuration to set connection parameters."""
        try:
            self.ui.chb_pmac_enable.setChecked(
                self.connection_config.pmac_enable)

            self.ui.chb_voltx_enable.setChecked(
                self.connection_config.voltx_enable)
            self.ui.sb_voltx_address.setValue(
                self.connection_config.voltx_address)

            self.ui.chb_volty_enable.setChecked(
                self.connection_config.volty_enable)
            self.ui.sb_volty_address.setValue(
                self.connection_config.volty_address)

            self.ui.chb_voltz_enable.setChecked(
                self.connection_config.voltz_enable)
            self.ui.sb_voltz_address.setValue(
                self.connection_config.voltz_address)

            self.ui.chb_multich_enable.setChecked(
                self.connection_config.multich_enable)
            self.ui.sb_multich_address.setValue(
                self.connection_config.multich_address)

            self.ui.chb_nmr_enable.setChecked(
                self.connection_config.nmr_enable)
            self.ui.cmb_nmr_port.setCurrentIndex(
                self.ui.cmb_nmr_port.findText(
                    self.connection_config.nmr_port))
            self.ui.cmb_nmr_baudrate.setCurrentIndex(
                self.ui.cmb_nmr_baudrate.findText(
                    str(self.connection_config.nmr_baudrate)))

            self.ui.chb_elcomat_enable.setChecked(
                self.connection_config.elcomat_enable)
            self.ui.cmb_elcomat_port.setCurrentIndex(
                self.ui.cmb_elcomat_port.findText(
                    self.connection_config.elcomat_port))
            self.ui.cmb_elcomat_baudrate.setCurrentIndex(
                self.ui.cmb_elcomat_baudrate.findText(
                    str(self.connection_config.elcomat_baudrate)))

            self.ui.chb_dcct_enable.setChecked(
                self.connection_config.dcct_enable)
            self.ui.sb_dcct_address.setValue(
                self.connection_config.dcct_address)

            self.ui.chb_ps_enable.setChecked(
                self.connection_config.ps_enable)
            self.ui.cmb_ps_port.setCurrentIndex(
                self.ui.cmb_ps_port.findText(
                    self.connection_config.ps_port))

            self.ui.chb_water_udc_enable.setChecked(
                self.connection_config.water_udc_enable)
            self.ui.cmb_water_udc_port.setCurrentIndex(
                self.ui.cmb_water_udc_port.findText(
                    self.connection_config.water_udc_port))
            self.ui.cmb_water_udc_baudrate.setCurrentIndex(
                self.ui.cmb_water_udc_baudrate.findText(
                    str(self.connection_config.water_udc_baudrate)))
            self.ui.sb_water_udc_slave_address.setValue(
                self.connection_config.water_udc_slave_address)

            self.ui.chb_air_udc_enable.setChecked(
                self.connection_config.air_udc_enable)
            self.ui.cmb_air_udc_port.setCurrentIndex(
                self.ui.cmb_air_udc_port.findText(
                    self.connection_config.air_udc_port))
            self.ui.cmb_air_udc_baudrate.setCurrentIndex(
                self.ui.cmb_air_udc_baudrate.findText(
                    str(self.connection_config.air_udc_baudrate)))
            self.ui.sb_air_udc_slave_address.setValue(
                self.connection_config.air_udc_slave_address)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to load configuration.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def save_db(self):
        """Save connection parameters to database."""
        self.ui.cmb_idn.setCurrentIndex(-1)
        if self.database_name is not None:
            try:
                if self.update_configuration():
                    self.connection_config.db_update_database(
                        self.database_name, 
                        mongo=self.mongo, server=self.server)
                    idn = self.connection_config.db_save()
                    self.ui.cmb_idn.addItem(str(idn))
                    self.ui.cmb_idn.setCurrentIndex(self.ui.cmb_idn.count()-1)
                    self.ui.pbt_load_db.setEnabled(False)
            except Exception:
                _traceback.print_exc(file=_sys.stdout)
                msg = 'Failed to save connection to database.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
        else:
            msg = 'Invalid database filename.'
            _QMessageBox.critical(
                self, 'Failure', msg, _QMessageBox.Ok)

    def update_connection_ids(self):
        """Update connection IDs in combo box."""
        current_text = self.ui.cmb_idn.currentText()
        load_enabled = self.ui.pbt_load_db.isEnabled()
        self.ui.cmb_idn.clear()
        try:
            self.connection_config.db_update_database(
                self.database_name, 
                mongo=self.mongo, server=self.server)
            idns = self.connection_config.db_get_id_list()
            self.ui.cmb_idn.addItems([str(idn) for idn in idns])
            if len(current_text) == 0:
                self.ui.cmb_idn.setCurrentIndex(self.ui.cmb_idn.count()-1)
                self.ui.pbt_load_db.setEnabled(True)
            else:
                self.ui.cmb_idn.setCurrentText(current_text)
                self.ui.pbt_load_db.setEnabled(load_enabled)
        except Exception:
            pass

    def update_configuration(self):
        """Update connection configuration parameters."""
        self.connection_config.clear()

        try:
            self.connection_config.pmac_enable = (
                self.ui.chb_pmac_enable.isChecked())

            self.connection_config.voltx_enable = (
                self.ui.chb_voltx_enable.isChecked())
            self.connection_config.voltx_address = (
                self.ui.sb_voltx_address.value())

            self.connection_config.volty_enable = (
                self.ui.chb_volty_enable.isChecked())
            self.connection_config.volty_address = (
                self.ui.sb_volty_address.value())

            self.connection_config.voltz_enable = (
                self.ui.chb_voltz_enable.isChecked())
            self.connection_config.voltz_address = (
                self.ui.sb_voltz_address.value())

            self.connection_config.multich_enable = (
                self.ui.chb_multich_enable.isChecked())
            self.connection_config.multich_address = (
                self.ui.sb_multich_address.value())

            self.connection_config.nmr_enable = (
                self.ui.chb_nmr_enable.isChecked())
            self.connection_config.nmr_port = (
                self.ui.cmb_nmr_port.currentText())
            self.connection_config.nmr_baudrate = int(
                self.ui.cmb_nmr_baudrate.currentText())

            self.connection_config.elcomat_enable = (
                self.ui.chb_elcomat_enable.isChecked())
            self.connection_config.elcomat_port = (
                self.ui.cmb_elcomat_port.currentText())
            self.connection_config.elcomat_baudrate = int(
                self.ui.cmb_elcomat_baudrate.currentText())

            self.connection_config.dcct_enable = (
                self.ui.chb_dcct_enable.isChecked())
            self.connection_config.dcct_address = (
                self.ui.sb_dcct_address.value())

            self.connection_config.ps_enable = (
                self.ui.chb_ps_enable.isChecked())
            self.connection_config.ps_port = (
                self.ui.cmb_ps_port.currentText())

            self.connection_config.water_udc_enable = (
                self.ui.chb_water_udc_enable.isChecked())
            self.connection_config.water_udc_port = (
                self.ui.cmb_water_udc_port.currentText())
            self.connection_config.water_udc_baudrate = int(
                self.ui.cmb_water_udc_baudrate.currentText())
            self.connection_config.water_udc_slave_address = (
                self.ui.sb_water_udc_slave_address.value())

            self.connection_config.air_udc_enable = (
                self.ui.chb_air_udc_enable.isChecked())
            self.connection_config.air_udc_port = (
                self.ui.cmb_air_udc_port.currentText())
            self.connection_config.air_udc_baudrate = int(
                self.ui.cmb_air_udc_baudrate.currentText())
            self.connection_config.air_udc_slave_address = (
                self.ui.sb_air_udc_slave_address.value())

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.connection_config.clear()

        if self.connection_config.valid_data():
            return True
        else:
            msg = 'Invalid connection configuration.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return False

    def update_led_status(self):
        """Update led status."""
        try:
            pmac_connected = _pmac.connected
            if pmac_connected is None:
                self.ui.la_pmac_led.setEnabled(False)
            else:
                self.ui.la_pmac_led.setEnabled(pmac_connected)

            self.ui.la_voltx_led.setEnabled(_voltx.connected)
            self.ui.la_volty_led.setEnabled(_volty.connected)
            self.ui.la_voltz_led.setEnabled(_voltz.connected)
            self.ui.la_multich_led.setEnabled(_multich.connected)
            self.ui.la_nmr_led.setEnabled(_nmr.connected)
            self.ui.la_elcomat_led.setEnabled(_elcomat.connected)
            self.ui.la_dcct_led.setEnabled(_dcct.connected)
            self.ui.la_ps_led.setEnabled(_ps.ser.is_open)
            self.ui.la_water_udc_led.setEnabled(
                _water_udc.connected)
            self.ui.la_air_udc_led.setEnabled(
                _air_udc.connected)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def update_serial_ports(self):
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

        self.ui.cmb_ps_port.clear()
        self.ui.cmb_ps_port.addItems(_ports)

        self.ui.cmb_nmr_port.clear()
        self.ui.cmb_nmr_port.addItems(_ports)

        self.ui.cmb_elcomat_port.clear()
        self.ui.cmb_elcomat_port.addItems(_ports)

        self.ui.cmb_water_udc_port.clear()
        self.ui.cmb_water_udc_port.addItems(_ports)

        self.ui.cmb_air_udc_port.clear()
        self.ui.cmb_air_udc_port.addItems(_ports)
