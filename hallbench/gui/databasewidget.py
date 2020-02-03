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

from imautils.gui import databasewidgets as _databasewidgets
from hallbench.gui.utils import get_ui_file as _get_ui_file
import hallbench.data as _data


_ConnectionConfig = _data.configuration.ConnectionConfig
_PowerSupplyConfig = _data.configuration.PowerSupplyConfig
_HallCalibrationCurve = _data.calibration.HallCalibrationCurve
_HallProbePositions = _data.calibration.HallProbePositions
_MeasurementConfig = _data.configuration.MeasurementConfig
_VoltageScan = _data.measurement.VoltageScan
_FieldScan = _data.measurement.FieldScan
_Fieldmap = _data.measurement.Fieldmap


class DatabaseWidget(_QWidget):
    """Database widget class for the Hall Bench Control application."""
 
    _connection_table_name = _ConnectionConfig.collection_name
    _power_supply_table_name = _PowerSupplyConfig.collection_name
    _hall_sensor_table_name = _HallCalibrationCurve.collection_name
    _hall_probe_table_name = _HallProbePositions.collection_name
    _configuration_table_name = _MeasurementConfig.collection_name
    _voltage_scan_table_name = _VoltageScan.collection_name
    _field_scan_table_name = _FieldScan.collection_name
    _fieldmap_table_name = _Fieldmap.collection_name
 
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
 
        # setup the ui
        uifile = _get_ui_file(self)
        self.ui = _uic.loadUi(uifile, self)
 
        self._table_object_dict = {
            self._connection_table_name: _ConnectionConfig,
            self._power_supply_table_name: _PowerSupplyConfig,
            self._hall_sensor_table_name: _HallCalibrationCurve,
            self._hall_probe_table_name: _HallProbePositions,
            self._configuration_table_name: _MeasurementConfig,
            self._voltage_scan_table_name: _VoltageScan,
            self._field_scan_table_name: _FieldScan,
            self._fieldmap_table_name: _Fieldmap,
            }
 
        self._table_page_dict = {
            self._connection_table_name: None,
            self._power_supply_table_name: None,
            self._hall_sensor_table_name: None,
            self._hall_probe_table_name: None,
            self._configuration_table_name: self.ui.pg_configuration,
            self._voltage_scan_table_name: self.ui.pg_voltage_scan,
            self._field_scan_table_name: self.ui.pg_field_scan,
            self._fieldmap_table_name: self.ui.pg_fieldmap,
            }
 
        self.short_version_hidden_tables = [
            self._connection_table_name,
            self._power_supply_table_name,
            self._hall_sensor_table_name,
            self._hall_probe_table_name,
            ]
 
        self.ui.pbt_remove_unused_configurations.setEnabled(True)
        self.ui.pbt_view_voltage_scan.setEnabled(True)
        self.ui.pbt_convert_to_field_scan.setEnabled(True)
        self.ui.pbt_view_field_scan.setEnabled(True)
        self.ui.pbt_create_fieldmap.setEnabled(True)
        self.ui.pbt_view_fieldmap.setEnabled(True)
        
        self.twg_database = _databasewidgets.DatabaseTabWidget(
            database_name=self.database_name,
            mongo=self.mongo, server=self.server)
        self.ui.lyt_database.addWidget(self.twg_database)
        
        self.connect_signal_slots()
 
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
        try:
            self.twg_database.delete_widgets()
            self.twg_database.clear()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
 
    def connect_signal_slots(self):
        """Create signal/slot connections."""
        self.ui.pbt_save.clicked.connect(self.save_files)
        self.ui.pbt_read.clicked.connect(self.read_files)
        self.ui.pbt_delete.clicked.connect(
            self.twg_database.delete_database_documents)
 
        self.ui.tbt_refresh.clicked.connect(self.update_database_tables)
        self.ui.tbt_clear.clicked.connect(self.clear)
        self.ui.twg_database.currentChanged.connect(
            self.disable_invalid_buttons)
 
        self.ui.pbt_remove_unused_configurations.clicked.connect(
            self.remove_unused_configurations)
 
#         self.ui.pbt_view_voltage_scan.clicked.connect(self.view_voltage_scan)
#         self.ui.pbt_convert_to_field_scan.clicked.connect(
#             self.convert_to_field_scan)
#  
#         self.ui.pbt_view_field_scan.clicked.connect(self.view_field_scan)
#         self.ui.pbt_create_fieldmap.clicked.connect(self.create_fieldmap)
 
        self.ui.pbt_view_fieldmap.clicked.connect(self.view_fieldmap)
 
#     def convert_to_field_scan(self):
#         """Convert voltage scans to field scans."""
#         idns = self.get_table_selected_ids(self._voltage_scan_table_name)
#         if len(idns) == 0:
#             return
# 
#         configuration_id_list = []
#         configuration_list = []
#         voltage_scan_list = []
#         try:
#             for idn in idns:
#                 vd = _VoltageScan(database=self.database, idn=idn)
#                 configuration_id = vd.configuration_id
#                 if configuration_id is None:
#                     msg = 'Invalid configuration ID found in scan list.'
#                     _QMessageBox.critical(
#                         self, 'Failure', msg, _QMessageBox.Ok)
#                     return
# 
#                 config = _MeasurementConfig(
#                     database=self.database, idn=configuration_id)
#                 configuration_id_list.append(configuration_id)
#                 configuration_list.append(config)
#                 voltage_scan_list.append(vd)
# 
#             probe_names = _HallProbe.get_table_column(
#                 self.database, 'probe_name')
# 
#             if len(probe_names) == 0:
#                 msg = 'No Hall Probe found.'
#                 _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
#                 return
# 
#             use_offset, ok = _QInputDialog.getItem(
#                 self, "Voltage Offset", "Subtract Voltage Offset? ",
#                 ['True', 'False'], 0, editable=False)
# 
#             if not ok:
#                 return
# 
#             if use_offset == 'False':
#                 subtract_voltage_offset = 0
#                 for vs in voltage_scan_list:
#                     vs.offsetx_start = None
#                     vs.offsetx_end = None
#                     vs.offsety_start = None
#                     vs.offsety_end = None
#                     vs.offsetz_start = None
#                     vs.offsetz_end = None
#             else:
#                 subtract_voltage_offset = 1
#                 for vs in voltage_scan_list:
#                     off_list = [
#                         vs.offsetx_start,
#                         vs.offsetx_end,
#                         vs.offsety_start,
#                         vs.offsety_end,
#                         vs.offsetz_start,
#                         vs.offsetz_end,
#                         ]
#                     if any([off is None for off in off_list]):
#                         msg = 'Invalid voltage offset value found.'
#                         _QMessageBox.critical(
#                             self, 'Failure', msg, _QMessageBox.Ok)
#                         return
# 
#             probe_name, ok = _QInputDialog.getItem(
#                 self, "Hall Probe", "Select the Hall Probe:",
#                 probe_names, 0, editable=False)
# 
#             if not ok:
#                 return
# 
#             idn = _HallProbe.get_hall_probe_id(self.database, probe_name)
#             if idn is None:
#                 msg = 'Hall probe data not found in database.'
#                 _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
#                 return
# 
#             hall_probe = _HallProbe(database=self.database, idn=idn)
#             field_scan_list = _data.measurement.get_field_scan_list(
#                 voltage_scan_list, hall_probe)
# 
#             unique_configuration_id_list, index, inv = _np.unique(
#                 configuration_id_list, return_index=True, return_inverse=True)
#             for i in range(len(unique_configuration_id_list)):
#                 configuration_id = unique_configuration_id_list[i]
#                 config = configuration_list[index[i]]
#                 config.probe_name = probe_name
#                 config.subtract_voltage_offset = subtract_voltage_offset
#                 new_config_id = config.save_to_database(self.database)
#                 unique_configuration_id_list[i] = new_config_id
# 
#             idns = []
#             for i in range(len(field_scan_list)):
#                 field_scan = field_scan_list[i]
#                 config_id = unique_configuration_id_list[inv[i]]
#                 field_scan.configuration_id = int(config_id)
#                 idn = field_scan.save_to_database(self.database)
#                 idns.append(idn)
# 
#             self.update_database_tables()
# 
#             msg = 'Field scans saved in database.\nIDs: ' + str(idns)
#             _QMessageBox.information(self, 'Information', msg, _QMessageBox.Ok)
# 
#         except Exception:
#             _traceback.print_exc(file=_sys.stdout)
#             msg = 'Failed to convert VoltageScan to FieldScan.'
#             _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
# 
#     def create_fieldmap(self):
#         """Create fieldmap from field scan records."""
#         self.save_fieldmap_dialog.accept()
#         idns = self.get_table_selected_ids(self._field_scan_table_name)
#         if len(idns) == 0:
#             return
# 
#         probe_name_list = []
#         try:
#             for idn in idns:
#                 configuration_id = _FieldScan.get_database_param(
#                     self.database, idn, 'configuration_id')
#                 if configuration_id is None:
#                     msg = 'Invalid configuration ID found in scan list.'
#                     _QMessageBox.critical(
#                         self, 'Failure', msg, _QMessageBox.Ok)
#                     return
# 
#                 probe_name = _MeasurementConfig.get_probe_name_from_database(
#                     self.database, configuration_id)
#                 probe_name_list.append(probe_name)
# 
#             if not all([pn == probe_name_list[0] for pn in probe_name_list]):
#                 msg = 'Inconsistent probe name found in scan list'
#                 _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
#                 return
# 
#             probe_name = probe_name_list[0]
#             if probe_name is None or len(probe_name) == 0:
#                 msg = 'Invalid Hall probe.'
#                 _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
#                 return
# 
#             idn = _HallProbe.get_hall_probe_id(self.database, probe_name)
#             if idn is None:
#                 msg = 'Hall probe data not found in database.'
#                 _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
#                 return
# 
#             hall_probe = _HallProbe(database=self.database, idn=idn)
#             self.save_fieldmap_dialog.show(idns, hall_probe)
# 
#         except Exception:
#             _traceback.print_exc(file=_sys.stdout)
#             msg = 'Failed to create Fieldmap.'
#             _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
 
    def disable_invalid_buttons(self):
        """Disable invalid buttons."""
        try:
            current_table_name = self.twg_database.get_current_table_name()
            if current_table_name is not None:
                self.ui.stw_buttons.setEnabled(True)
 
                for table_name, page in self._table_page_dict.items():
                    if page is not None:
                        page.setEnabled(False)
                        _idx = self.ui.stw_buttons.indexOf(page)
                    else:
                        self.ui.stw_buttons.setCurrentIndex(0)
 
                current_page = self._table_page_dict[current_table_name]
                if current_page is not None:
                    current_page.setEnabled(True)
                    _idx = self.ui.stw_buttons.indexOf(current_page)
                    self.ui.stw_buttons.setCurrentWidget(current_page)
            else:
                self.ui.stw_buttons.setCurrentIndex(0)
                self.ui.stw_buttons.setEnabled(False)
 
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
 
    def load_database(self):
        """Load database."""
        try:
            self.twg_database.database_name = self.database_name
            self.twg_database.mongo = self.mongo
            self.twg_database.server = self.server
            if self.ui.chb_short_version.isChecked():
                hidden_tables = self.short_version_hidden_tables
                self.twg_database.hidden_tables = hidden_tables
            else:
                self.twg_database.hidden_tables = []
            self.twg_database.load_database()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
 
    def read_files(self):
        """Read file and save in database."""
        table_name = self.twg_database.get_current_table_name()
        if table_name is None:
            return
 
        object_class = self._table_object_dict[table_name]
 
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
                obj = object_class(
                    database_name=self.database_name,
                    mongo=self.mongo, server=self.server)
                obj.read_file(filename)
                idn = obj.db_save()
                idns.append(idn)
            msg = 'Added to database table.\nIDs: ' + str(idns)
            self.update_database_tables()
            _QMessageBox.information(self, 'Information', msg, _QMessageBox.Ok)
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            msg = 'Failed to read files and save values in database.'
            _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
            return
 
    def remove_unused_configurations(self):
        """Remove unused configurations from database table."""
        try:
            mc =_MeasurementConfig(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
            idns = mc.db_get_id_list()
            if len(idns) == 0:
                return
 
            msg = 'Remove all unused configurations from database table?'
            reply = _QMessageBox.question(
                self, 'Message', msg, _QMessageBox.Yes, _QMessageBox.No)
            if reply == _QMessageBox.Yes:
 
                self.blockSignals(True)
                _QApplication.setOverrideCursor(_Qt.WaitCursor)
 
                vs = _VoltageScan(
                    database_name=self.database_name,
                    mongo=self.mongo, server=self.server)
                vs_idns = vs.db_get_values('configuration_id')
                
                fs = _FieldScan(
                    database_name=self.database_name,
                    mongo=self.mongo, server=self.server)
                fs_idns = fs.db_get_values('configuration_id')
 
                unused_idns = []
                for idn in idns:
                    if (idn not in vs_idns) and (idn not in fs_idns):
                        unused_idns.append(idn)
 
                mc.db_delete(unused_idns)
                self.update_database_tables()
 
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
 
    def save_files(self):
        """Save database record to file."""
        table_name = self.twg_database.get_current_table_name()
        if table_name is None:
            return
 
        object_class = self._table_object_dict[table_name]
 
        idns = self.twg_database.get_table_selected_ids(table_name)
        nr_idns = len(idns)
        if nr_idns == 0:
            return
 
        objs = []
        fns = []
        try:
            for i in range(nr_idns):
                idn = idns[i]
                obj = object_class(
                    database_name=self.database_name,
                    mongo=self.mongo, server=self.server)
                obj.db_read(idn)
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
 
    def update_database_tables(self):
        """Update database tables."""
        if not self.isVisible():
            return

        try:
            self.twg_database.database_name = self.database_name
            self.twg_database.mongo = self.mongo
            self.twg_database.server = self.server
            if self.ui.chb_short_version.isChecked():
                hidden_tables = self.short_version_hidden_tables
                self.twg_database.hidden_tables = hidden_tables
            else:
                self.twg_database.hidden_tables = []
            self.twg_database.update_database_tables()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)

#     def view_voltage_scan(self):
#         """Open view data dialog."""
#         idns = self.get_table_selected_ids(self._voltage_scan_table_name)
#         if len(idns) == 0:
#             return
# 
#         try:
#             self.blockSignals(True)
#             _QApplication.setOverrideCursor(_Qt.WaitCursor)
# 
#             scan_list = []
#             for idn in idns:
#                 scan_list.append(_VoltageScan(database=self.database, idn=idn))
#             self.view_scan_dialog.accept()
#             self.view_scan_dialog.show(scan_list, idns, 'Voltage [V]')
# 
#             self.blockSignals(False)
#             _QApplication.restoreOverrideCursor()
# 
#         except Exception:
#             self.blockSignals(False)
#             _QApplication.restoreOverrideCursor()
#             _traceback.print_exc(file=_sys.stdout)
#             msg = 'Failed to show voltage scan dialog.'
#             _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
#             return
# 
#     def view_field_scan(self):
#         """Open view data dialog."""
#         idns = self.get_table_selected_ids(self._field_scan_table_name)
#         if len(idns) == 0:
#             return
# 
#         try:
#             self.blockSignals(True)
#             _QApplication.setOverrideCursor(_Qt.WaitCursor)
# 
#             scan_list = []
#             for idn in idns:
#                 scan_list.append(_FieldScan(database=self.database, idn=idn))
#             self.view_scan_dialog.accept()
#             self.view_scan_dialog.show(scan_list, idns, 'Magnetic Field [T]')
# 
#             self.blockSignals(False)
#             _QApplication.restoreOverrideCursor()
# 
#         except Exception:
#             self.blockSignals(False)
#             _QApplication.restoreOverrideCursor()
#             _traceback.print_exc(file=_sys.stdout)
#             msg = 'Failed to show field scan dialog.'
#             _QMessageBox.critical(self, 'Failure', msg, _QMessageBox.Ok)
#             return

    def view_fieldmap(self):
        """Open view fieldmap dialog."""
        try:
            idn = self.twg_database.get_table_selected_id(
                self._fieldmap_table_name)
            if idn is None:
                return
            
            self.blockSignals(True)
            _QApplication.setOverrideCursor(_Qt.WaitCursor)
 
            fieldmap = _Fieldmap(
                database_name=self.database_name,
                mongo=self.mongo, server=self.server)
            fieldmap.db_read(idn)
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