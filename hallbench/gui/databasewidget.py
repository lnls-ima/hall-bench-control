"""Database tables widgets."""

import sqlite3 as _sqlite3
import numpy as _np
import PyQt4.uic as _uic
from PyQt4.QtCore import Qt as _Qt
from PyQt4.QtGui import (
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
    )

from hallbench.gui.utils import getUiFile as _getUiFile
from hallbench.gui.calibrationdialog import CalibrationDialog \
    as _CalibrationDialog
from hallbench.gui.configurationwidgets import ConfigurationDialog \
    as _ConfigurationDialog
from hallbench.gui.viewdatadialog import ViewDataDialog as _ViewDataDialog
from hallbench.gui.fieldmapdialog import FieldMapDialog \
    as _FieldMapDialog

from hallbench.data.calibration import ProbeCalibration as _ProbeCalibration
from hallbench.data.configuration import MeasurementConfig \
    as _MeasurementConfig
from hallbench.data.measurement import VoltageData as _VoltageData
from hallbench.data.measurement import FieldData as _FieldData
from hallbench.data.measurement import Fieldmap as _Fieldmap


_max_number_rows = 1000
_max_str_size = 1000


class DatabaseWidget(_QWidget):
    """Database widget class for the Hall Bench Control application."""

    _calibration_table_name = _ProbeCalibration.database_table_name()
    _configuration_table_name = _MeasurementConfig.database_table_name()
    _raw_data_table_name = _VoltageData.database_table_name()
    _scan_table_name = _FieldData.database_table_name()
    _fieldmap_table_name = _Fieldmap.database_table_name()

    def __init__(self, parent=None):
        """Set up the ui."""
        super().__init__(parent)

        # setup the ui
        uifile = _getUiFile(self)
        self.ui = _uic.loadUi(uifile, self)

        # create dialogs
        self.calibration_dialog = _CalibrationDialog(load_enabled=False)
        self.configuration_dialog = _ConfigurationDialog(load_enabled=False)
        self.view_data_dialog = _ViewDataDialog()
        self.fieldmap_dialog = _FieldMapDialog()

        self.tables = []
        self.ui.database_tab.clear()
        self.connectSignalSlots()
        self.loadDatabase()

    @property
    def database(self):
        """Database filename."""
        return self.window().database

    def clear(self):
        """Clear."""
        ntabs = self.ui.database_tab.count()
        for idx in range(ntabs):
            self.ui.database_tab.removeTab(idx)
            self.tables[idx].deleteLater()
        self.tables = []
        self.ui.database_tab.clear()

    def clearRawDataTable(self):
        """Clear raw data table."""
        con = _sqlite3.connect(self.database)
        cur = con.cursor()

        cmd = 'SELECT * FROM {0}'.format(self._raw_data_table_name)
        if len(cur.execute(cmd).fetchall()) == 0:
            con.close()
            return

        msg = (
            'Are you sure you want to delete all rows in the ' +
            self._raw_data_table_name + ' table?')
        reply = _QMessageBox.question(
            self, 'Message', msg, _QMessageBox.Yes, _QMessageBox.No)

        if reply == _QMessageBox.Yes:
            cmd = 'DELETE FROM {0}'.format(self._raw_data_table_name)
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
            self.calibration_dialog.accept()
            self.configuration_dialog.accept()
            self.view_data_dialog.accept()
            self.fieldmap_dialog.accept()
        except Exception:
            pass

    def connectSignalSlots(self):
        """Create signal/slot connections."""
        self.ui.refresh_btn.clicked.connect(self.updateDatabaseTables)
        self.ui.database_tab.currentChanged.connect(self.disableInvalidButtons)

        self.ui.view_calibration_btn.clicked.connect(self.viewCalibration)
        self.ui.save_calibration_btn.clicked.connect(self.saveCalibration)

        self.ui.view_configuration_btn.clicked.connect(self.viewConfiguration)
        self.ui.save_configuration_btn.clicked.connect(self.saveConfiguration)

        self.ui.view_raw_data_btn.clicked.connect(self.viewRawData)
        self.ui.save_raw_data_btn.clicked.connect(self.saveRawData)
        self.ui.clear_raw_data_btn.clicked.connect(self.clearRawDataTable)

        self.ui.view_scan_btn.clicked.connect(self.viewScan)
        self.ui.save_scan_btn.clicked.connect(self.saveScan)
        self.ui.create_fieldmap_btn.clicked.connect(self.createFieldmap)

        self.ui.save_fieldmap_btn.clicked.connect(self.saveFieldmap)

    def createFieldmap(self):
        """Create fieldmap from scan records."""
        self.fieldmap_dialog.accept()
        idns = self.getTableSelectedIDs(self._scan_table_name)
        if len(idns) == 0:
            return

        field_data_list = []
        configuration_ids = []
        try:
            for idn in idns:
                fd = _FieldData(database=self.database, idn=idn)
                cid = _FieldData.get_configuration_id_from_database(
                    self.database, idn)
                field_data_list.append(fd)
                configuration_ids.append(cid)

            if (any([c is None for c in configuration_ids]) or not all(
                    [c == configuration_ids[0] for c in configuration_ids])):
                msg = 'Invalid configuration ID list.'
                _QMessageBox.critical(
                    self, 'Failure', msg, _QMessageBox.Ok)
                return

            probe_name = _MeasurementConfig.get_probe_name_from_database(
                self.database, configuration_ids[0])
            if probe_name is None or len(probe_name) == 0:
                msg = 'Invalid probe calibration.'
                _QMessageBox.critical(
                    self, 'Failure', msg, _QMessageBox.Ok)
                return

            idn = _ProbeCalibration.get_probe_calibration_id(
                self.database, probe_name)
            if idn is None:
                msg = 'Probe calibration data not found in database.'
                _QMessageBox.critical(
                    self, 'Failure', msg, _QMessageBox.Ok)
                return

            probe_calibration = _ProbeCalibration(
                database=self.database, idn=idn)

            self.fieldmap_dialog.show(
                field_data_list, probe_calibration, self.database, idns)

        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def disableInvalidButtons(self):
        """Disable invalid buttons."""
        current_table = self.getCurrentTable()
        if current_table is None:
            self.ui.calibration_pg.setEnabled(False)
            self.ui.configuration_pg.setEnabled(False)
            self.ui.raw_data_pg.setEnabled(False)
            self.ui.scan_pg.setEnabled(False)
            self.ui.fieldmap_pg.setEnabled(False)
            return

        _enable = current_table.table_name == self._calibration_table_name
        self.ui.calibration_pg.setEnabled(_enable)
        if _enable:
            self.ui.buttons_tbx.setCurrentWidget(self.ui.calibration_pg)

        _enable = current_table.table_name == self._configuration_table_name
        self.ui.configuration_pg.setEnabled(_enable)
        if _enable:
            self.ui.buttons_tbx.setCurrentWidget(self.ui.configuration_pg)

        _enable = current_table.table_name == self._raw_data_table_name
        self.ui.raw_data_pg.setEnabled(_enable)
        if _enable:
            self.ui.buttons_tbx.setCurrentWidget(self.ui.raw_data_pg)

        _enable = current_table.table_name == self._scan_table_name
        self.ui.scan_pg.setEnabled(_enable)
        if _enable:
            self.ui.buttons_tbx.setCurrentWidget(self.ui.scan_pg)

        _enable = current_table.table_name == self._fieldmap_table_name
        self.ui.fieldmap_pg.setEnabled(_enable)
        if _enable:
            self.ui.buttons_tbx.setCurrentWidget(self.ui.fieldmap_pg)

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

    def saveCalibration(self):
        """Save probe calibration data to file."""
        idn = self.getTableSelectedID(self._calibration_table_name)
        if idn is None:
            return

        filename = _QFileDialog.getSaveFileName(
            self, caption='Save probe calibration file',
            filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            if not filename.endswith('.txt') and not filename.endswith('.dat'):
                filename = filename + '.txt'
            calibration = _ProbeCalibration(database=self.database, idn=idn)
            calibration.save_file(filename)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def saveConfiguration(self):
        """Save configuration data to file."""
        idn = self.getTableSelectedID(self._configuration_table_name)
        if idn is None:
            return

        filename = _QFileDialog.getSaveFileName(
            self, caption='Save configuration file',
            filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            if not filename.endswith('.txt') and not filename.endswith('.dat'):
                filename = filename + '.txt'
            configuration = _MeasurementConfig(database=self.database, idn=idn)
            configuration.save_file(filename)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def saveRawData(self):
        """Save raw data to file."""
        idn = self.getTableSelectedID(self._raw_data_table_name)
        if idn is None:
            return

        try:
            raw_data = _VoltageData(database=self.database, idn=idn)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

        filename = _QFileDialog.getSaveFileName(
            self, caption='Save configuration file',
            directory=raw_data.default_filename,
            filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            if not filename.endswith('.txt') and not filename.endswith('.dat'):
                filename = filename + '.dat'
            raw_data.save_file(filename)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def saveScan(self):
        """Save scan data to file."""
        idn = self.getTableSelectedID(self._scan_table_name)
        if idn is None:
            return

        try:
            scan = _FieldData(database=self.database, idn=idn)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

        filename = _QFileDialog.getSaveFileName(
            self, caption='Save configuration file',
            directory=scan.default_filename,
            filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            if not filename.endswith('.txt') and not filename.endswith('.dat'):
                filename = filename + '.dat'
            scan.save_file(filename)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

    def saveFieldmap(self):
        """Save fieldmap to file."""
        idn = self.getTableSelectedID(self._fieldmap_table_name)
        if idn is None:
            return

        try:
            fieldmap = _Fieldmap(database=self.database, idn=idn)
        except Exception as e:
            _QMessageBox.critical(
                self, 'Failure', str(e), _QMessageBox.Ok)

        filename = _QFileDialog.getSaveFileName(
            self, caption='Save fieldmap file',
            directory=fieldmap.default_filename,
            filter="Text files (*.txt *.dat)")

        if isinstance(filename, tuple):
            filename = filename[0]

        if len(filename) == 0:
            return

        try:
            if not filename.endswith('.txt') and not filename.endswith('.dat'):
                filename = filename + '.dat'
            fieldmap.save_file(filename)
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

    def viewCalibration(self):
        """Open calibration dialog."""
        idn = self.getTableSelectedID(self._calibration_table_name)
        if idn is None:
            return

        self.calibration_dialog.show(self.database)
        self.calibration_dialog.setDatabaseID(idn)
        self.calibration_dialog.loadDB()

    def viewConfiguration(self):
        """Open configuration dialog."""
        idn = self.getTableSelectedID(self._configuration_table_name)
        if idn is None:
            return

        self.configuration_dialog.show(self.database)
        self.configuration_dialog.main_wg.setDatabaseID(idn)
        self.configuration_dialog.main_wg.loadDB()

    def viewRawData(self):
        """Open view data dialog."""
        idns = self.getTableSelectedIDs(self._raw_data_table_name)
        if len(idns) == 0:
            return

        data_list = []
        for idn in idns:
            data_list.append(_VoltageData(database=self.database, idn=idn))
        data_label = 'Voltage [V]'
        self.view_data_dialog.accept()
        self.view_data_dialog.show(data_list, data_label)

    def viewScan(self):
        """Open view data dialog."""
        idns = self.getTableSelectedIDs(self._scan_table_name)
        if len(idns) == 0:
            return

        data_list = []
        for idn in idns:
            data_list.append(_FieldData(database=self.database, idn=idn))
        data_label = 'Magnetic Field [T]'
        self.view_data_dialog.accept()
        self.view_data_dialog.show(data_list, data_label)


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
