"""Database tables widgets."""

import os as _os
import sys as _sys
import numpy as _np
import sqlite3 as _sqlite3
import traceback as _traceback
import qtpy.uic as _uic
from qtpy.QtCore import Qt as _Qt
from qtpy.QtWidgets import (
    QWidget as _QWidget,
    QApplication as _QApplication,
    QLabel as _QLabel,
    QTableWidget as _QTableWidget,
    QTableWidgetItem as _QTableWidgetItem,
    QMessageBox as _QMessageBox,
    QVBoxLayout as _QVBoxLayout,
    QHBoxLayout as _QHBoxLayout,
    QSpinBox as _QSpinBox,
    QFileDialog as _QFileDialog,
    QInputDialog as _QInputDialog,
    QAbstractItemView as _QAbstractItemView,
    )

from hallbench.gui.utils import getUiFile as _getUiFile
import hallbench.data as _data


_max_number_rows = 100
_max_str_size = 100

_ConnectionConfig = _data.configuration.ConnectionConfig
_MeasurementConfig = _data.configuration.MeasurementConfig
_PowerSupplyConfig = _data.configuration.PowerSupplyConfig
_HallSensor = _data.calibration.HallSensor
_HallProbe = _data.calibration.HallProbe
_VoltageScan = _data.measurement.VoltageScan
_FieldScan = _data.measurement.FieldScan
_Fieldmap = _data.measurement.Fieldmap


class DatabaseWidget(_QWidget):
    """Database widget class for the Hall Bench Control application."""

    _connection_table_name = _ConnectionConfig.database_table_name()
    _power_supply_table_name = _PowerSupplyConfig.database_table_name()
    _hall_sensor_table_name = _HallSensor.database_table_name()
    _hall_probe_table_name = _HallProbe.database_table_name()
    _configuration_table_name = _MeasurementConfig.database_table_name()
    _voltage_scan_table_name = _VoltageScan.database_table_name()
    _field_scan_table_name = _FieldScan.database_table_name()
    _fieldmap_table_name = _Fieldmap.database_table_name()

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        self.short_version_tables = [
            self._configuration_table_name,
            self._voltage_scan_table_name,
            self._field_scan_table_name,
            self._fieldmap_table_name,
            ]

        self.tables = []
        self.ui.database_tab.clear()
        self.connectSignalSlots()

    @property
    def database(self):
        """Database filename."""
        return _QApplication.instance().database

    @property
    def directory(self):
        """Return the default directory."""
        return _QApplication.instance().directory

    @property
    def save_fieldmap_dialog(self):
        """Save fieldmap dialog."""
        return _QApplication.instance().save_fieldmap_dialog

    @property
    def view_probe_dialog(self):
        """View probe dialog."""
        return _QApplication.instance().view_probe_dialog

    @property
    def view_scan_dialog(self):
        """View scan dialog."""
        return _QApplication.instance().view_scan_dialog

    @property
    def view_fieldmap_dialog(self):
        """View fieldmap dialog."""
        return _QApplication.instance().view_fieldmap_dialog

    def clear(self):
        """Clear."""
        ntabs = self.ui.database_tab.count()
        for idx in range(ntabs):
            self.ui.database_tab.removeTab(idx)
            self.tables[idx].deleteLater()
        self.tables = []
        self.ui.database_tab.clear()

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.refresh_btn.clicked.connect(self.updateDatabaseTables)
        self.ui.clear_btn.clicked.connect(self.clear)
        self.ui.database_tab.currentChanged.connect(self.disableInvalidButtons)

        self.ui.save_connection_btn.clicked.connect(
            lambda: self.saveFiles(
                self._connection_table_name, _ConnectionConfig))
        self.ui.read_connection_btn.clicked.connect(
            lambda: self.readFiles(_ConnectionConfig))
        self.ui.delete_connection_btn.clicked.connect(
            lambda: self.deleteDatabaseRecords(self._connection_table_name))

        self.ui.save_power_supply_btn.clicked.connect(
            lambda: self.saveFiles(
                self._power_supply_table_name, _PowerSupplyConfig))
        self.ui.read_power_supply_btn.clicked.connect(
            lambda: self.readFiles(_PowerSupplyConfig))
        self.ui.delete_power_supply_btn.clicked.connect(
            lambda: self.deleteDatabaseRecords(self._power_supply_table_name))

        self.ui.save_hall_sensor_btn.clicked.connect(
            lambda: self.saveFiles(self._hall_sensor_table_name, _HallSensor))
        self.ui.read_hall_sensor_btn.clicked.connect(
            lambda: self.readFiles(_HallSensor))
        self.ui.delete_hall_sensor_btn.clicked.connect(
            lambda: self.deleteDatabaseRecords(self._hall_sensor_table_name))

        self.ui.view_hall_probe_btn.clicked.connect(self.viewHallProbe)
        self.ui.save_hall_probe_btn.clicked.connect(
            lambda: self.saveFiles(self._hall_probe_table_name, _HallProbe))
        self.ui.read_hall_probe_btn.clicked.connect(
            lambda: self.readFiles(_HallProbe))
        self.ui.delete_hall_probe_btn.clicked.connect(
            lambda: self.deleteDatabaseRecords(self._hall_probe_table_name))

        self.ui.save_configuration_btn.clicked.connect(
            lambda: self.saveFiles(
                self._configuration_table_name, _MeasurementConfig))
        self.ui.read_configuration_btn.clicked.connect(
            lambda: self.readFiles(_MeasurementConfig))
        self.ui.delete_configuration_btn.clicked.connect(
            lambda: self.deleteDatabaseRecords(self._configuration_table_name))
        self.ui.remove_unused_configurations_btn.clicked.connect(
            self.removeUnusedConfigurations)

        self.ui.view_voltage_scan_btn.clicked.connect(self.viewVoltageScan)
        self.ui.save_voltage_scan_btn.clicked.connect(
            lambda: self.saveFiles(
                self._voltage_scan_table_name, _VoltageScan))
        self.ui.read_voltage_scan_btn.clicked.connect(
            lambda: self.readFiles(_VoltageScan))
        self.ui.delete_voltage_scan_btn.clicked.connect(
            lambda: self.deleteDatabaseRecords(self._voltage_scan_table_name))
        self.ui.convert_to_field_scan_btn.clicked.connect(
            self.convertToFieldScan)

        self.ui.view_field_scan_btn.clicked.connect(self.viewFieldScan)
        self.ui.save_field_scan_btn.clicked.connect(
            lambda: self.saveFiles(
                self._field_scan_table_name, _FieldScan))
        self.ui.read_field_scan_btn.clicked.connect(
            lambda: self.readFiles(_FieldScan))
        self.ui.delete_field_scan_btn.clicked.connect(
            lambda: self.deleteDatabaseRecords(self._field_scan_table_name))
        self.ui.create_fieldmap_btn.clicked.connect(self.createFieldmap)

        self.ui.view_fieldmap_btn.clicked.connect(self.viewFieldmap)
        self.ui.save_fieldmap_btn.clicked.connect(
            lambda: self.saveFiles(self._fieldmap_table_name, _Fieldmap))
        self.ui.read_fieldmap_btn.clicked.connect(
            lambda: self.readFiles(_Fieldmap))
        self.ui.delete_fieldmap_btn.clicked.connect(
            lambda: self.deleteDatabaseRecords(self._fieldmap_table_name))

    def convertToFieldScan(self):
        """Convert voltage scans to field scans."""
        idns = self.getTableSelectedIDs(self._voltage_scan_table_name)
        if len(idns) == 0:
            return

        configuration_id_list = []
        configuration_list = []
        voltage_scan_list = []
        try:
            for idn in idns:
                vd = _VoltageScan(database=self.database, idn=idn)
                configuration_id = vd.configuration_id
                if configuration_id is None:
                    msg = 'Invalid configuration ID found in scan list.'
                    _QMessageBox.critical(
                        self, 'Failure', msg, _QMessageBox.Ok)
                    return

                config = _MeasurementConfig(
                    database=self.database, idn=configuration_id)
                configuration_id_list.append(configuration_id)
                configuration_list.append(config)
                voltage_scan_list.append(vd)

            probe_names = _HallProbe.get_table_column(
                self.database, 'probe_name')

            if len(probe_names) == 0:
                msg = 'No Hall Probe found.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
                return

            use_offset, ok = _QInputDialog.getItem(
                self, "Voltage Offset", "Subtract Voltage Offset? ",
                ['True', 'False'], 0, editable=False)

            if not ok:
                return

            if use_offset == 'False':
                subtract_voltage_offset = 0
                for vs in voltage_scan_list:
                    vs.offsetx_start = None
                    vs.offsetx_end = None
                    vs.offsety_start = None
                    vs.offsety_end = None
                    vs.offsetz_start = None
                    vs.offsetz_end = None
            else:
                subtract_voltage_offset = 1
                for vs in voltage_scan_list:
                    off_list = [
                        vs.offsetx_start,
                        vs.offsetx_end,
                        vs.offsety_start,
                        vs.offsety_end,
                        vs.offsetz_start,
                        vs.offsetz_end,
                        ]
                    if any([off is None for off in off_list]):
                        msg = 'Invalid voltage offset value found.'
                        _QMessageBox.critical(
                            self, 'Failure', msg, _QMessageBox.Ok)
                        return

            probe_name, ok = _QInputDialog.getItem(
                self, "Hall Probe", "Select the Hall Probe:",
                probe_names, 0, editable=False)

            if not ok:
                return

            idn = _HallProbe.get_hall_probe_id(self.database, probe_name)
            if idn is None:
                msg = 'Hall probe data not found in database.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
                return

            hall_probe = _HallProbe(database=self.database, idn=idn)
            field_scan_list = _data.measurement.get_field_scan_list(
                voltage_scan_list, hall_probe)

            unique_configuration_id_list, index, inv = _np.unique(
                configuration_id_list, return_index=True, return_inverse=True)
            for i in range(len(unique_configuration_id_list)):
                configuration_id = unique_configuration_id_list[i]
                config = configuration_list[index[i]]
                config.probe_name = probe_name
                config.subtract_voltage_offset = subtract_voltage_offset
                new_config_id = config.save_to_database(self.database)
                unique_configuration_id_list[i] = new_config_id

            idns = []
            for i in range(len(field_scan_list)):
                field_scan = field_scan_list[i]
                config_id = unique_configuration_id_list[inv[i]]
                field_scan.configuration_id = int(config_id)
                idn = field_scan.save_to_database(self.database)
                idns.append(idn)

            self.updateDatabaseTables()

            msg = 'Field scans saved in database.\nIDs: ' + str(idns)
            _QMessageBox.information(self, 'Information', msg, _QMessageBox.Ok)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to convert VoltageScan to FieldScan.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def createFieldmap(self):
        """Create fieldmap from field scan records."""
        self.save_fieldmap_dialog.accept()
        idns = self.getTableSelectedIDs(self._field_scan_table_name)
        if len(idns) == 0:
            return

        probe_name_list = []
        try:
            for idn in idns:
                configuration_id = _FieldScan.get_database_param(
                    self.database, idn, 'configuration_id')
                if configuration_id is None:
                    msg = 'Invalid configuration ID found in scan list.'
                    _QMessageBox.critical(
                        self, 'Failure', msg, _QMessageBox.Ok)
                    return

                probe_name = _MeasurementConfig.get_probe_name_from_database(
                    self.database, configuration_id)
                probe_name_list.append(probe_name)

            if not all([pn == probe_name_list[0] for pn in probe_name_list]):
                msg = 'Inconsistent probe name found in scan list'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
                return

            probe_name = probe_name_list[0]
            if probe_name is None or len(probe_name) == 0:
                msg = 'Invalid Hall probe.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
                return

            idn = _HallProbe.get_hall_probe_id(self.database, probe_name)
            if idn is None:
                msg = 'Hall probe data not found in database.'
                _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
                return

            hall_probe = _HallProbe(database=self.database, idn=idn)
            self.save_fieldmap_dialog.show(idns, hall_probe)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to create Fieldmap.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def deleteDatabaseRecords(self, table_name):
        """Delete record from database table."""
        try:
            idns = self.getTableSelectedIDs(table_name)
            if len(idns) == 0:
                return

            con = _sqlite3.connect(self.database)
            cur = con.cursor()

            msg = 'Delete selected database records?'
            reply = _QMessageBox.question(
                self, 'Message', msg, _QMessageBox.Yes, _QMessageBox.No)
            if reply == _QMessageBox.Yes:
                seq = ','.join(['?']*len(idns))
                cmd = 'DELETE FROM {0} WHERE id IN ({1})'.format(
                    table_name, seq)
                cur.execute(cmd, idns)
                con.commit()
                con.close()
                self.updateDatabaseTables()
            else:
                con.close()
                return
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to delete database records.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def disableInvalidButtons(self):
        """Disable invalid buttons."""
        tables = [
            self._connection_table_name,
            self._power_supply_table_name,
            self._hall_sensor_table_name,
            self._hall_probe_table_name,
            self._configuration_table_name,
            self._voltage_scan_table_name,
            self._field_scan_table_name,
            self._fieldmap_table_name,
        ]

        pages = [
            self.ui.connection_pg,
            self.ui.power_supply_pg,
            self.ui.hall_sensor_pg,
            self.ui.hall_probe_pg,
            self.ui.configuration_pg,
            self.ui.voltage_scan_pg,
            self.ui.field_scan_pg,
            self.ui.fieldmap_pg,
        ]

        current_table = self.getCurrentTable()
        if current_table is not None and current_table.table_name in tables:
            current_table_name = current_table.table_name
            self.ui.buttons_tbx.setEnabled(True)
            for i in range(len(tables)):
                _idx = self.ui.buttons_tbx.indexOf(pages[i])
                _enable = current_table_name == tables[i]
                pages[i].setEnabled(_enable)
                self.ui.buttons_tbx.setItemEnabled(_idx, _enable)
                if _enable:
                    self.ui.buttons_tbx.setCurrentWidget(pages[i])
        else:
            self.ui.buttons_tbx.setCurrentWidget(self.ui.empty_pg)
            self.ui.buttons_tbx.setEnabled(False)

    def getCurrentTable(self):
        """Get current table."""
        idx = self.ui.database_tab.currentIndex()
        if len(self.tables) > idx and idx != -1:
            current_table = self.tables[idx]
            return current_table
        else:
            return None

    def getTableSelectedID(self, table_name):
        """Get table selected ID."""
        current_table = self.getCurrentTable()
        if current_table is None:
            return None

        if current_table.table_name != table_name:
            return None

        idns = current_table.getSelectedIDs()

        if len(idns) == 0:
            return None

        if len(idns) > 1:
            msg = 'Select only one entry of the database table.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return None

        idn = idns[0]
        return idn

    def getTableSelectedIDs(self, table_name):
        """Get table selected IDs."""
        current_table = self.getCurrentTable()
        if current_table is None:
            return []

        if current_table.table_name != table_name:
            return []

        return current_table.getSelectedIDs()

    def loadDatabase(self):
        """Load database."""
        try:
            con = _sqlite3.connect(self.database)
            cur = con.cursor()
            res = cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table';")

            for r in res:
                table_name = r[0]
                if (self.ui.short_version_chb.isChecked() and
                   table_name not in self.short_version_tables):
                    continue

                table = DatabaseTable()
                tab = _QWidget()
                vlayout = _QVBoxLayout()
                hlayout = _QHBoxLayout()

                initial_id_la = _QLabel("Initial ID:")
                initial_id_sb = _QSpinBox()
                initial_id_sb.setMinimumWidth(100)
                initial_id_sb.setButtonSymbols(2)
                hlayout.addStretch(0)
                hlayout.addWidget(initial_id_la)
                hlayout.addWidget(initial_id_sb)
                hlayout.addSpacing(30)

                number_rows_la = _QLabel("Number of rows:")
                number_rows_sb = _QSpinBox()
                number_rows_sb.setMinimumWidth(100)
                number_rows_sb.setButtonSymbols(2)
                number_rows_sb.setReadOnly(True)
                hlayout.addWidget(number_rows_la)
                hlayout.addWidget(number_rows_sb)
                hlayout.addSpacing(30)

                max_number_rows_la = _QLabel("Maximum number of rows:")
                max_number_rows_sb = _QSpinBox()
                max_number_rows_sb.setMinimumWidth(100)
                max_number_rows_sb.setButtonSymbols(2)
                max_number_rows_sb.setReadOnly(True)
                hlayout.addWidget(max_number_rows_la)
                hlayout.addWidget(max_number_rows_sb)

                table.loadDatabaseTable(
                    self.database,
                    table_name,
                    initial_id_sb,
                    number_rows_sb,
                    max_number_rows_sb)

                vlayout.addWidget(table)
                vlayout.addLayout(hlayout)
                tab.setLayout(vlayout)

                self.tables.append(table)
                self.ui.database_tab.addTab(tab, table_name)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to load database.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def readFiles(self, object_class):
        """Read file and save in database."""
        fns = _QFileDialog.getOpenFileNames(
            self, caption='Read files', directory=self.directory,
            filter="Text files (*.txt *.dat)")

        if isinstance(fns, tuple):
            fns = fns[0]

        if len(fns) == 0:
            return

        try:
            idns = []
            for filename in fns:
                obj = object_class(filename=filename)
                idn = obj.save_to_database(self.database)
                idns.append(idn)
            msg = 'Added to database table.\nIDs: ' + str(idns)
            self.updateDatabaseTables()
            _QMessageBox.information(self, 'Information', msg, _QMessageBox.Ok)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to read files and save values in database.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def removeUnusedConfigurations(self):
        """Remove unused configurations from database table."""
        try:
            idns = _MeasurementConfig.get_table_column(self.database, 'id')
            if len(idns) == 0:
                return

            msg = 'Remove all unused configurations from database table?'
            reply = _QMessageBox.question(
                self, 'Message', msg, _QMessageBox.Yes, _QMessageBox.No)
            if reply == _QMessageBox.Yes:

                self.blockSignals(True)
                _QApplication.setOverrideCursor(_Qt.WaitCursor)

                vs_idns = _VoltageScan.get_table_column(
                    self.database, 'configuration_id')
                fs_idns = _FieldScan.get_table_column(
                    self.database, 'configuration_id')

                unused_idns = []
                for idn in idns:
                    if (idn not in vs_idns) and (idn not in fs_idns):
                        unused_idns.append(idn)

                con = _sqlite3.connect(self.database)
                cur = con.cursor()

                seq = ','.join(['?']*len(unused_idns))
                cmd = 'DELETE FROM {0} WHERE id IN ({1})'.format(
                    self._configuration_table_name, seq)
                cur.execute(cmd, unused_idns)
                con.commit()
                con.close()

                self.updateDatabaseTables()

                self.blockSignals(False)
                _QApplication.restoreOverrideCursor()
            else:
                return
        except Exception:
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to remove unused configurations.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def saveFiles(self, table_name, object_class):
        """Save database record to file."""
        idns = self.getTableSelectedIDs(table_name)
        nr_idns = len(idns)
        if nr_idns == 0:
            return

        objs = []
        fns = []
        try:
            for i in range(nr_idns):
                idn = idns[i]
                obj = object_class(database=self.database, idn=idn)
                default_filename = obj.default_filename
                if '.txt' in default_filename:
                    default_filename = default_filename.replace(
                        '.txt', '_ID={0:d}.txt'.format(idn))
                elif '.dat' in default_filename:
                    default_filename = default_filename.replace(
                        '.dat', '_ID={0:d}.dat'.format(idn))
                objs.append(obj)
                fns.append(default_filename)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to read database entries.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

        if nr_idns == 1:
            filename = _QFileDialog.getSaveFileName(
                self, caption='Save file',
                directory=_os.path.join(self.directory, fns[0]),
                filter="Text files (*.txt *.dat)")

            if isinstance(filename, tuple):
                filename = filename[0]

            if len(filename) == 0:
                return

            fns[0] = filename
        else:
            directory = _QFileDialog.getExistingDirectory(
                self, caption='Save files', directory=self.directory)

            if isinstance(directory, tuple):
                directory = directory[0]

            if len(directory) == 0:
                return

            for i in range(len(fns)):
                fns[i] = _os.path.join(directory, fns[i])

        try:
            for i in range(nr_idns):
                obj = objs[i]
                idn = idns[i]
                filename = fns[i]
                if (not filename.endswith('.txt') and
                   not filename.endswith('.dat')):
                    filename = filename + '.txt'
                obj.save_file(filename)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to save files.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def scrollDownTables(self):
        """Scroll down all tables."""
        for idx in range(len(self.tables)):
            self.ui.database_tab.setCurrentIndex(idx)
            self.tables[idx].scrollDown()

    def updateDatabaseTables(self):
        """Update database tables."""
        if not self.isVisible():
            return

        try:
            self.blockSignals(True)
            _QApplication.setOverrideCursor(_Qt.WaitCursor)

            idx = self.ui.database_tab.currentIndex()
            self.clear()
            self.loadDatabase()
            self.scrollDownTables()
            self.ui.database_tab.setCurrentIndex(idx)

            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()

        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            _QMessageBox.critical(
                self, 'Failure', 'Failed to update database.', _QMessageBox.Ok)

    def viewHallProbe(self):
        """Open hall probe dialog."""
        idn = self.getTableSelectedID(self._hall_probe_table_name)
        if idn is None:
            return

        try:
            hall_probe = _HallProbe(database=self.database, idn=idn)
            self.view_probe_dialog.show(hall_probe)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to show Hall probe dialog,'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)

    def viewVoltageScan(self):
        """Open view data dialog."""
        idns = self.getTableSelectedIDs(self._voltage_scan_table_name)
        if len(idns) == 0:
            return

        try:
            self.blockSignals(True)
            _QApplication.setOverrideCursor(_Qt.WaitCursor)

            scan_list = []
            for idn in idns:
                scan_list.append(_VoltageScan(database=self.database, idn=idn))
            self.view_scan_dialog.accept()
            self.view_scan_dialog.show(scan_list, idns, 'Voltage [V]')

            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()

        except Exception:
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to show voltage scan dialog.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def viewFieldScan(self):
        """Open view data dialog."""
        idns = self.getTableSelectedIDs(self._field_scan_table_name)
        if len(idns) == 0:
            return

        try:
            self.blockSignals(True)
            _QApplication.setOverrideCursor(_Qt.WaitCursor)

            scan_list = []
            for idn in idns:
                scan_list.append(_FieldScan(database=self.database, idn=idn))
            self.view_scan_dialog.accept()
            self.view_scan_dialog.show(scan_list, idns, 'Magnetic Field [T]')

            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()

        except Exception:
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to show field scan dialog.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return

    def viewFieldmap(self):
        """Open view fieldmap dialog."""
        idn = self.getTableSelectedID(self._fieldmap_table_name)
        if idn is None:
            return

        try:
            self.blockSignals(True)
            _QApplication.setOverrideCursor(_Qt.WaitCursor)

            fieldmap = _Fieldmap(database=self.database, idn=idn)
            self.view_fieldmap_dialog.accept()
            self.view_fieldmap_dialog.show(fieldmap, idn)

            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
        except Exception:
            self.blockSignals(False)
            _QApplication.restoreOverrideCursor()
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to show fieldmap dialog.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return


class DatabaseTable(_QTableWidget):
    """Database table widget."""

    _datatype_dict = {
        'INTEGER': int,
        'REAL': float,
        'TEXT': str,
        }

    _hidden_columns = [
        'voltagex_avg',
        'voltagex_std',
        'voltagey_avg',
        'voltagey_std',
        'voltagez_avg',
        'voltagez_std',
        'fieldx_avg',
        'fieldx_std',
        'fieldy_avg',
        'fieldy_std',
        'fieldz_avg',
        'fieldz_std',
        'map',
        ]

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)

        self.setAlternatingRowColors(True)
        self.verticalHeader().hide()
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setDefaultSectionSize(120)

        self.database = None
        self.table_name = None
        self.column_names = []
        self.data_types = []
        self.data = []
        self.initial_table_id = None
        self.initial_id_sb = None
        self.number_rows_sb = None
        self.max_number_rows_sb = None

    def changeInitialID(self):
        """Change initial ID."""
        initial_id = self.initial_id_sb.value()
        self.filterColumn(initial_id=initial_id)

    def loadDatabaseTable(
            self, database, table_name,
            initial_id_sb, number_rows_sb, max_number_rows_sb):
        """Set database filename and table name."""
        self.database = database
        self.table_name = table_name

        self.initial_id_sb = initial_id_sb
        self.initial_id_sb.editingFinished.connect(self.changeInitialID)

        self.number_rows_sb = number_rows_sb
        self.number_rows_sb.setMaximum(_max_number_rows)

        self.max_number_rows_sb = max_number_rows_sb
        self.max_number_rows_sb.setMaximum(_max_number_rows)

        self.updateTable()

    def updateTable(self):
        """Update table."""
        if self.database is None or self.table_name is None:
            return

        self.blockSignals(True)
        self.setColumnCount(0)
        self.setRowCount(0)

        con = _sqlite3.connect(self.database)
        cur = con.cursor()

        cmd = "PRAGMA TABLE_INFO({0})".format(self.table_name)
        cur.execute(cmd)
        table_info = cur.fetchall()

        self.column_names = []
        self.data_types = []
        for ti in table_info:
            column_name = ti[1]
            column_type = ti[2]
            if column_name not in self._hidden_columns:
                self.column_names.append(column_name)
                self.data_types.append(self._datatype_dict[column_type])

        self.setColumnCount(len(self.column_names))
        self.setHorizontalHeaderLabels(self.column_names)

        self.setRowCount(1)
        for j in range(len(self.column_names)):
            self.setItem(0, j, _QTableWidgetItem(''))

        column_names_str = ''
        for col_name in self.column_names:
            column_names_str = column_names_str + '"{0:s}", '.format(col_name)
        column_names_str = column_names_str[:-2]

        sel = '(SELECT {0:s} FROM {1:s} ORDER BY id DESC LIMIT {2:d})'.format(
            column_names_str, self.table_name, _max_number_rows)
        cmd = 'SELECT * FROM ' + sel + ' ORDER BY id ASC'
        data = cur.execute(cmd).fetchall()

        if len(data) > 0:
            cmd = 'SELECT MIN(id) FROM {0}'.format(self.table_name)
            min_idn = cur.execute(cmd).fetchone()[0]
            self.initial_id_sb.setMinimum(min_idn)

            cmd = 'SELECT MAX(id) FROM {0}'.format(self.table_name)
            max_idn = cur.execute(cmd).fetchone()[0]
            self.initial_id_sb.setMaximum(max_idn)

            self.max_number_rows_sb.setValue(len(data))
            self.data = data[:]
            self.addRowsToTable(data)
        else:
            self.initial_id_sb.setMinimum(0)
            self.initial_id_sb.setMaximum(0)
            self.max_number_rows_sb.setValue(0)

        self.setSelectionBehavior(_QAbstractItemView.SelectRows)
        self.blockSignals(False)
        self.itemChanged.connect(self.filterChanged)
        self.itemSelectionChanged.connect(self.selectLine)

    def addRowsToTable(self, data):
        """Add rows to table."""
        if len(self.column_names) == 0:
            return

        self.setRowCount(1)

        if len(data) > self.max_number_rows_sb.value():
            tabledata = data[-self.max_number_rows_sb.value()::]
        else:
            tabledata = data

        if len(tabledata) == 0:
            return

        self.initial_id_sb.setValue(int(tabledata[0][0]))
        self.setRowCount(len(tabledata) + 1)
        self.number_rows_sb.setValue(len(tabledata))
        self.initial_table_id = tabledata[0][0]

        for j in range(len(self.column_names)):
            for i in range(len(tabledata)):
                item_str = str(tabledata[i][j])
                if len(item_str) > _max_str_size:
                    item_str = item_str[:10] + '...'
                item = _QTableWidgetItem(item_str)
                item.setFlags(_Qt.ItemIsSelectable | _Qt.ItemIsEnabled)
                self.setItem(i + 1, j, item)

    def scrollDown(self):
        """Scroll down."""
        vbar = self.verticalScrollBar()
        vbar.setValue(vbar.maximum())

    def selectLine(self):
        """Select the entire line."""
        if (self.rowCount() == 0
           or self.columnCount() == 0
           or len(self.column_names) == 0 or len(self.data_types) == 0):
            return

        selected = self.selectedItems()
        rows = [s.row() for s in selected]

        if 0 in rows:
            self.setSelectionBehavior(_QAbstractItemView.SelectItems)
        else:
            self.setSelectionBehavior(_QAbstractItemView.SelectRows)

    def filterChanged(self, item):
        """Apply column filter to data."""
        if item.row() == 0:
            self.filterColumn()

    def filterColumn(self, initial_id=None):
        """Apply column filter to data."""
        if (self.rowCount() == 0
           or self.columnCount() == 0
           or len(self.column_names) == 0 or len(self.data_types) == 0):
            return

        try:
            con = _sqlite3.connect(self.database)
            cur = con.cursor()
            column_names_str = ''
            for col_name in self.column_names:
                column_names_str = column_names_str + '"{0:s}", '.format(
                    col_name)
            column_names_str = column_names_str[:-2]
            cmd = 'SELECT {0:s} FROM {1:s}'.format(
                column_names_str, self.table_name)

            and_flag = False
            filters = []
            for idx in range(len(self.column_names)):
                filters.append(self.item(0, idx).text())

            if any(filt != '' for filt in filters):
                cmd = cmd + ' WHERE '

            for idx in range(len(self.column_names)):
                column = self.column_names[idx]
                data_type = self.data_types[idx]
                filt = filters[idx]

                if filt != '':

                    if and_flag:
                        cmd = cmd + ' AND '
                    and_flag = True

                    if data_type == str:
                        cmd = cmd + column + ' LIKE "%' + filt + '%"'
                    else:
                        if '~' in filt:
                            fs = filt.split('~')
                            if len(fs) == 2:
                                cmd = cmd + column + ' >= ' + fs[0]
                                cmd = cmd + ' AND '
                                cmd = cmd + column + ' <= ' + fs[1]
                        elif filt.lower() == 'none' or filt.lower() == 'null':
                            cmd = cmd + column + ' IS NULL'
                        else:
                            try:
                                value = data_type(filt)
                                cmd = cmd + column + ' = ' + str(value)
                            except ValueError:
                                cmd = cmd + column + ' ' + filt

            if initial_id is not None:
                if 'WHERE' in cmd:
                    cmd = (
                        'SELECT * FROM (' + cmd +
                        ' AND id >= {0:d} LIMIT {1:d})'.format(
                            initial_id, _max_number_rows))
                else:
                    cmd = (
                        'SELECT * FROM (' + cmd +
                        ' WHERE id >= {0:d} LIMIT {1:d})'.format(
                            initial_id, _max_number_rows))

            else:
                cmd = (
                    'SELECT * FROM (' + cmd +
                    ' ORDER BY id DESC LIMIT {0:d}) ORDER BY id ASC'.format(
                        _max_number_rows))

            cur.execute(cmd)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            pass

        data = cur.fetchall()
        self.data = data[:]
        self.addRowsToTable(data)

    def getSelectedIDs(self):
        """Get selected IDs."""
        selected = self.selectedItems()
        rows = [s.row() for s in selected if s.row() != 0]
        rows = _np.unique(rows)

        selected_ids = []
        for row in rows:
            if 'id_0' in self.column_names and 'id_f' in self.column_names:
                idx_id_0 = self.column_names.index('id_0')
                idx_id_f = self.column_names.index('id_f')
                id_0 = int(self.item(row, idx_id_0).text())
                id_f = int(self.item(row, idx_id_f).text())
                for idn in range(id_0, id_f + 1):
                    selected_ids.append(idn)
            elif ('id_0' not in self.column_names
                  and 'id_f' not in self.column_names):
                idn = int(self.item(row, 0).text())
                selected_ids.append(idn)

        return selected_ids
