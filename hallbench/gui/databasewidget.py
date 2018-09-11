"""Database tables widgets."""

import sqlite3 as _sqlite3
import numpy as _np
import PyQt5.uic as _uic
from PyQt5.QtCore import Qt as _Qt
from PyQt5.QtWidgets import (
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
    )

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.viewprobedialog import ViewProbeDialog \
    as _ViewProbeDialog
from hallbench.gui.viewscandialog import ViewScanDialog as _ViewScanDialog
from hallbench.gui.fieldmapdialog import FieldmapDialog \
    as _FieldmapDialog
import hallbench.data as _data


_max_number_rows = 1000
_max_str_size = 1000

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

        # create dialogs
        self.view_probe_dialog = _ViewProbeDialog()
        self.viewscan_dialog = _ViewScanDialog()
        self.fieldmap_dialog = _FieldmapDialog()

        self.tables = []
        self.ui.database_tab.clear()
        self.connectSignalSlots()
        self.loadDatabase()

    @property
    def database(self):
        """Database filename."""
        return _QApplication.instance().database

    def clear(self):
        """Clear."""
        ntabs = self.ui.database_tab.count()
        for idx in range(ntabs):
            self.ui.database_tab.removeTab(idx)
            self.tables[idx].deleteLater()
        self.tables = []
        self.ui.database_tab.clear()

    def clearVoltageScanTable(self):
        """Clear voltage scan table."""
        con = _sqlite3.connect(self.database)
        cur = con.cursor()

        cmd = 'SELECT * FROM {0}'.format(self._voltage_scan_table_name)
        if len(cur.execute(cmd).fetchall()) == 0:
            con.close()
            return

        msg = (
            'Are you sure you want to delete all rows in the ' +
            self._voltage_scan_table_name + ' table?')
        reply = _QMessageBox.question(
            self, 'Message', msg, _QMessageBox.Yes, _QMessageBox.No)

        if reply == _QMessageBox.Yes:
            cmd = 'DELETE FROM {0}'.format(self._voltage_scan_table_name)
            cur.execute(cmd)
            con.commit()
            con.close()
            self.updateDatabaseTables()
        else:
            con.close()
            return

    def closeDialogs(self):
        """Close dialogs."""
        try:
            self.view_probe_dialog.accept()
            self.viewscan_dialog.accept()
            self.fieldmap_dialog.accept()
        except Exception:
            pass

    def closeEvent(self, event):
        """Close widget."""
        try:
            self.closeDialogs()
            event.accept()
        except Exception:
            event.accept()

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.refresh_btn.clicked.connect(self.updateDatabaseTables)
        self.ui.database_tab.currentChanged.connect(self.disableInvalidButtons)

        self.ui.save_connection_btn.clicked.connect(
            lambda: self.saveFile(
                self._connection_table_name, _ConnectionConfig))
        self.ui.read_connection_btn.clicked.connect(
            lambda: self.readFile(_ConnectionConfig))
        self.ui.delete_connection_btn.clicked.connect(
            lambda: self.deleteDatabaseRecords(self._connection_table_name))

        self.ui.save_power_supply_btn.clicked.connect(
            lambda: self.saveFile(
                self._power_supply_table_name, _PowerSupplyConfig))
        self.ui.read_power_supply_btn.clicked.connect(
            lambda: self.readFile(_PowerSupplyConfig))
        self.ui.delete_power_supply_btn.clicked.connect(
            lambda: self.deleteDatabaseRecords(self._power_supply_table_name))

        self.ui.save_hall_sensor_btn.clicked.connect(
            lambda: self.saveFile(self._hall_sensor_table_name, _HallSensor))
        self.ui.read_hall_sensor_btn.clicked.connect(
            lambda: self.readFile(_HallSensor))
        self.ui.delete_hall_sensor_btn.clicked.connect(
            lambda: self.deleteDatabaseRecords(self._hall_sensor_table_name))

        self.ui.view_hall_probe_btn.clicked.connect(self.viewHallProbe)
        self.ui.save_hall_probe_btn.clicked.connect(
            lambda: self.saveFile(self._hall_probe_table_name, _HallProbe))
        self.ui.read_hall_probe_btn.clicked.connect(
            lambda: self.readFile(_HallProbe))
        self.ui.delete_hall_probe_btn.clicked.connect(
            lambda: self.deleteDatabaseRecords(self._hall_probe_table_name))

        self.ui.save_configuration_btn.clicked.connect(
            lambda: self.saveFile(
                self._configuration_table_name, _MeasurementConfig))
        self.ui.read_configuration_btn.clicked.connect(
            lambda: self.readFile(_MeasurementConfig))
        self.ui.delete_configuration_btn.clicked.connect(
            lambda: self.deleteDatabaseRecords(self._configuration_table_name))

        self.ui.view_voltage_scan_btn.clicked.connect(self.viewVoltageScan)
        self.ui.save_voltage_scan_btn.clicked.connect(
            lambda: self.saveFile(
                self._voltage_scan_table_name, _VoltageScan))
        self.ui.read_voltage_scan_btn.clicked.connect(
            lambda: self.readFile(_VoltageScan))
        self.ui.delete_voltage_scan_btn.clicked.connect(
            lambda: self.deleteDatabaseRecords(self._voltage_scan_table_name))
        self.ui.clear_voltage_scan_btn.clicked.connect(
            self.clearVoltageScanTable)
        self.ui.convert_to_field_scan_btn.clicked.connect(
            self.convertToFieldScan)

        self.ui.view_field_scan_btn.clicked.connect(self.viewFieldScan)
        self.ui.save_field_scan_btn.clicked.connect(
            lambda: self.saveFile(
                self._field_scan_table_name, _FieldScan))
        self.ui.read_field_scan_btn.clicked.connect(
            lambda: self.readFile(_FieldScan))
        self.ui.delete_field_scan_btn.clicked.connect(
            lambda: self.deleteDatabaseRecords(self._field_scan_table_name))
        self.ui.create_fieldmap_btn.clicked.connect(self.createFieldmap)

        self.ui.save_fieldmap_btn.clicked.connect(
            lambda: self.saveFile(self._fieldmap_table_name, _Fieldmap))
        self.ui.read_fieldmap_btn.clicked.connect(
            lambda: self.readFile(_Fieldmap))
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
            _QMessageBox.information(
                self, 'Information', msg, _QMessageBox.Ok)

        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def createFieldmap(self):
        """Create fieldmap from field scan records."""
        self.fieldmap_dialog.accept()
        idns = self.getTableSelectedIDs(self._field_scan_table_name)
        if len(idns) == 0:
            return

        probe_name_list = []
        field_scan_list = []
        try:
            for idn in idns:
                fd = _FieldScan(database=self.database, idn=idn)
                configuration_id = fd.configuration_id
                if configuration_id is None:
                    msg = 'Invalid configuration ID found in scan list.'
                    _QMessageBox.critical(
                        self, 'Failure', msg, _QMessageBox.Ok)
                    return

                probe_name = _MeasurementConfig.get_probe_name_from_database(
                    self.database, configuration_id)
                probe_name_list.append(probe_name)
                field_scan_list.append(fd)

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
            self.fieldmap_dialog.show(field_scan_list, hall_probe, idns)

        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def deleteDatabaseRecords(self, table_name):
        """Delete record from database table."""
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
            cmd = 'DELETE FROM {0} WHERE id IN ({1})'.format(table_name, seq)
            cur.execute(cmd, idns)
            con.commit()
            con.close()
            self.updateDatabaseTables()
        else:
            con.close()
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
                _enable = current_table_name == tables[i]
                pages[i].setEnabled(_enable)
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
            message = 'Select only one entry of the database table.'
            _QMessageBox.critical(self, 'Failure', message, _QMessageBox.Ok)
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
        con = _sqlite3.connect(self.database)
        cur = con.cursor()
        res = cur.execute("SELECT name FROM sqlite_master WHERE type='table';")

        for r in res:
            table_name = r[0]
            table = DatabaseTable(self.ui.database_tab)
            tab = _QWidget()
            vlayout = _QVBoxLayout()
            hlayout = _QHBoxLayout()

            initial_id_la = _QLabel("Initial ID:")
            initial_id_sb = _QSpinBox()
            initial_id_sb.setMinimumWidth(100)
            hlayout.addStretch(0)
            hlayout.addWidget(initial_id_la)
            hlayout.addWidget(initial_id_sb)
            hlayout.addSpacing(50)

            number_rows_la = _QLabel("Maximum number of rows:")
            number_rows_sb = _QSpinBox()
            number_rows_sb.setMinimumWidth(100)
            hlayout.addWidget(number_rows_la)
            hlayout.addWidget(number_rows_sb)

            table.loadDatabaseTable(
                self.database, table_name, initial_id_sb, number_rows_sb)

            vlayout.addWidget(table)
            vlayout.addLayout(hlayout)
            tab.setLayout(vlayout)

            self.tables.append(table)
            self.ui.database_tab.addTab(tab, table_name)

    def readFile(self, object_class):
        """Save database record to file."""
        filename = _QFileDialog.getOpenFileName(
            self, caption='Read file',
            filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            obj = object_class(filename=filename)
            idn = obj.save_to_database(self.database)
            msg = 'Added to database table.\nID: ' + str(idn)
            self.updateDatabaseTables()
            _QMessageBox.information(
                self, 'Information', msg, _QMessageBox.Ok)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)
            return

    def saveFile(self, table_name, object_class):
        """Save database record to file."""
        idn = self.getTableSelectedID(table_name)
        if idn is None:
            return

        try:
            obj = object_class(database=self.database, idn=idn)
            default_filename = obj.default_filename
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)
            return

        filename = _QFileDialog.getSaveFileName(
            self, caption='Save file',
            directory=default_filename,
            filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            if not filename.endswith('.txt') and not filename.endswith('.dat'):
                filename = filename + '.txt'
            obj.save_file(filename)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def scrollDownTables(self):
        """Scroll down all tables."""
        for idx in range(len(self.tables)):
            self.ui.database_tab.setCurrentIndex(idx)
            self.tables[idx].scrollDown()

    def updateDatabaseTables(self):
        """Update database tables."""
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
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def viewVoltageScan(self):
        """Open view data dialog."""
        idns = self.getTableSelectedIDs(self._voltage_scan_table_name)
        if len(idns) == 0:
            return

        scan_list = []
        for idn in idns:
            scan_list.append(_VoltageScan(database=self.database, idn=idn))
        self.viewscan_dialog.accept()
        self.viewscan_dialog.show(scan_list, 'Voltage [V]')

    def viewFieldScan(self):
        """Open view data dialog."""
        idns = self.getTableSelectedIDs(self._field_scan_table_name)
        if len(idns) == 0:
            return

        scan_list = []
        for idn in idns:
            scan_list.append(_FieldScan(database=self.database, idn=idn))
        self.viewscan_dialog.accept()
        self.viewscan_dialog.show(scan_list, 'Magnetic Field [T]')


class DatabaseTable(_QTableWidget):
    """Database table widget."""

    _datatype_dict = {
        'INTEGER': int,
        'REAL': float,
        'TEXT': str,
        }

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

    def changeInitialID(self):
        """Change initial ID."""
        initial_id = self.initial_id_sb.value()
        if len(self.data) == 0:
            return

        if self.initial_table_id != initial_id:
            ids = _np.array(
                [self.data[i][0] for i in range(len(self.data))])
            idx = _np.min(
                _np.append(_np.where(ids >= initial_id)[0], _np.inf))
            if _np.isinf(idx):
                data = []
            else:
                data = self.data[int(idx)::]
            if len(data) > self.number_rows_sb.value():
                data = data[:self.number_rows_sb.value()]
            self.addRowsToTable(data)

    def loadDatabaseTable(
            self, database, table_name,
            initial_id_sb, number_rows_sb):
        """Set database filename and table name."""
        self.database = database
        self.table_name = table_name

        self.initial_id_sb = initial_id_sb
        self.initial_id_sb.setButtonSymbols(2)
        self.initial_id_sb.editingFinished.connect(self.changeInitialID)

        self.number_rows_sb = number_rows_sb
        self.number_rows_sb.setButtonSymbols(2)
        self.number_rows_sb.setMaximum(_max_number_rows)
        self.number_rows_sb.setReadOnly(True)

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
        cmd = 'SELECT * FROM ' + self.table_name
        cur.execute(cmd)

        self.column_names = [description[0] for description in cur.description]
        self.setColumnCount(len(self.column_names))
        self.setHorizontalHeaderLabels(self.column_names)

        cmd = 'SELECT * FROM ' + self.table_name
        data = cur.execute(cmd).fetchall()

        self.data_types = []
        cur.execute("PRAGMA TABLE_INFO({0})".format(self.table_name))
        table_info = cur.fetchall()
        for i in range(len(table_info)):
            self.data_types.append(self._datatype_dict[table_info[i][2]])

        self.setRowCount(1)
        for j in range(len(self.column_names)):
            self.setItem(0, j, _QTableWidgetItem(''))

        if len(data) > 0:
            self.initial_id_sb.setMinimum(int(data[0][0]))
            self.initial_id_sb.setMaximum(int(data[-1][0]))
            self.number_rows_sb.setValue(len(data))
            self.data = data[:]
            self.addRowsToTable(data)
        else:
            self.initial_id_sb.setMinimum(0)
            self.initial_id_sb.setMaximum(0)
            self.number_rows_sb.setValue(0)

        self.blockSignals(False)
        self.itemChanged.connect(self.filterColumn)
        self.itemSelectionChanged.connect(self.selectLine)

    def addRowsToTable(self, data):
        """Add rows to table."""
        if len(self.column_names) == 0:
            return

        self.setRowCount(1)

        if len(data) > self.number_rows_sb.value():
            tabledata = data[-self.number_rows_sb.value()::]
        else:
            tabledata = data

        if len(tabledata) == 0:
            return

        self.initial_id_sb.setValue(int(tabledata[0][0]))
        self.setRowCount(len(tabledata) + 1)
        self.initial_table_id = tabledata[0][0]

        for j in range(len(self.column_names)):
            for i in range(len(tabledata)):
                item_str = str(tabledata[i][j])
                if len(item_str) > _max_str_size:
                    item_str = item_str[:20] + '...'
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
            return

        self.blockSignals(True)
        for row in rows:
            for col in range(len(self.column_names)):
                item = self.item(row, col)
                if item and not item.isSelected():
                    item.setSelected(True)
        self.blockSignals(False)

    def filterColumn(self, item):
        """Apply column filter to data."""
        if (self.rowCount() == 0
           or self.columnCount() == 0
           or len(self.column_names) == 0 or len(self.data_types) == 0
           or item.row() != 0):
            return

        con = _sqlite3.connect(self.database)
        cur = con.cursor()
        cmd = 'SELECT * FROM ' + self.table_name

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
                    else:
                        try:
                            value = data_type(filt)
                            cmd = cmd + column + ' = ' + str(value)
                        except ValueError:
                            cmd = cmd + column + ' ' + filt

        try:
            cur.execute(cmd)
        except Exception:
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
