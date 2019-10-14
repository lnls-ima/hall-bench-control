# -*- coding: utf-8 -*-

"""Main window for the Hall Bench Control application."""

import sys as _sys
import traceback as _traceback
from qtpy.QtWidgets import (
    QFileDialog as _QFileDialog,
    QMainWindow as _QMainWindow,
    QApplication as _QApplication,
    QDesktopWidget as _QDesktopWidget,
    )
from qtpy.QtCore import (
    QTimer as _QTimer,
    QRunnable as _QRunnable,
    QThreadPool as _QThreadPool,
    )
import qtpy.uic as _uic

from hallbench.gui.utils import get_ui_file as _get_ui_file
from hallbench.gui.auxiliarywidgets import PreferencesDialog \
    as _PreferencesDialog
from hallbench.gui.connectionwidget import ConnectionWidget \
    as _ConnectionWidget
from hallbench.gui.motorswidget import MotorsWidget as _MotorsWidget
from hallbench.gui.supplywidget import SupplyWidget as _SupplyWidget
from hallbench.gui.measurementwidget import MeasurementWidget \
    as _MeasurementWidget
from hallbench.gui.voltagewidget import VoltageWidget \
    as _VoltageWidget
from hallbench.gui.pscurrentwidget import PSCurrentWidget \
    as _PSCurrentWidget
from hallbench.gui.temperaturewidget import TemperatureWidget \
    as _TemperatureWidget
from hallbench.gui.watersystemwidget import WaterSystemWidget \
    as _WaterSystemWidget
from hallbench.gui.airconditioningwidget import AirConditioningWidget \
    as _AirConditioningWidget
from hallbench.gui.angularerrorwidget import AngularErrorWidget \
    as _AngularErrorWidget
from hallbench.gui.voltagetemperaturewidget import VoltageTempWidget \
    as _VoltageTempWidget
from hallbench.gui.databasewidget import DatabaseWidget \
    as _DatabaseWidget


class HallBenchWindow(_QMainWindow):
    """Main Window class for the Hall Bench Control application."""

    _update_positions_timer_interval = 500  # [ms]

    def __init__(self, parent=None, width=1200, height=700):
        """Set up the ui and add main tabs."""
        super().__init__(parent)

        # setup the ui
        uifile = _get_ui_file(self)
        self.ui = _uic.loadUi(uifile, self)
        self.resize(width, height)

        # clear the current tabs
        self.ui.twg_main.clear()

        # define tab names and corresponding widgets
        self.tab_names = [
            'connection',
            'motors',
            'power_supply',
            'measurement',
            'current',
            'voltage',
            'temperature',
            'voltage_temperature',
            'water_system',
            'air_conditioning',
            'angular_error',
            'database',
            ]

        self.tab_widgets = [
            _ConnectionWidget,
            _MotorsWidget,
            _SupplyWidget,
            _MeasurementWidget,
            _PSCurrentWidget,
            _VoltageWidget,
            _TemperatureWidget,
            _VoltageTempWidget,
            _WaterSystemWidget,
            _AirConditioningWidget,
            _AngularErrorWidget,
            _DatabaseWidget,
            ]

        # add preferences dialog
        self.preferences_dialog = _PreferencesDialog(self.tab_names)
        self.preferences_dialog.preferences_changed.connect(self.change_tabs)

        # show database name
        self.ui.le_database.setText(self.database)

        # start positions update
        self.stop_positions_update = False
        self.threadpool = _QThreadPool.globalInstance()
        self.timer = _QTimer()
        self.timer.timeout.connect(self.update_positions)
        self.timer.start(self._update_positions_timer_interval)

        # connect signals and slots
        self.connect_signal_slots()

    @property
    def database(self):
        """Return the database filename."""
        return _QApplication.instance().database

    @database.setter
    def database(self, value):
        _QApplication.instance().database = value

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

    def closeEvent(self, event):
        """Close main window and dialogs."""
        try:
            self.stop_positions_update = True
            self.threadpool.waitForDone()
            self.timer.stop()
            for idx in range(self.ui.twg_main.count()):
                widget = self.ui.twg_main.widget(idx)
                widget.close()
            self.view_probe_dialog.accept()
            self.view_scan_dialog.accept()
            self.save_fieldmap_dialog.accept()
            self.view_fieldmap_dialog.accept()
            self.preferences_dialog.close()
            event.accept()
        except Exception:
            _traceback.print_exc(file=_sys.stdout)
            event.accept()

    def change_database(self):
        """Change database file."""
        fn = _QFileDialog.getOpenFileName(
            self, caption='Database file', directory=self.directory,
            filter="Database File (*.db)")

        if isinstance(fn, tuple):
            fn = fn[0]

        if len(fn) == 0:
            return

        self.database = fn
        self.ui.le_database.setText(self.database)

    def change_tabs(self, tab_status):
        """Hide or show tabs."""
        try:
            if self.ui.twg_main.count() != 0:
                tab_current = self.ui.twg_main.currentWidget()
            else:
                tab_current = None

            self.ui.twg_main.clear()
            for idx, tab_name in enumerate(self.tab_names):
                tab_attr = 'tab_' + tab_name
                tab_name_split = tab_name.split('_')
                tab_label = ' '.join([s.capitalize() for s in tab_name_split])
                status = tab_status[tab_name]
                if status:
                    if hasattr(self, tab_attr):
                        tab = getattr(self, tab_attr)
                    else:
                        tab = self.tab_widgets[idx]()
                        setattr(self, tab_attr, tab)
                    self.ui.twg_main.addTab(tab, tab_label)

            if tab_current is not None:
                idx = self.ui.twg_main.indexOf(tab_current)
                self.ui.twg_main.setCurrentIndex(idx)

        except Exception:
            _traceback.print_exc(file=_sys.stdout)

    def centralize_window(self):
        """Centralize window."""
        window_center = _QDesktopWidget().availableGeometry().center()
        self.move(
            window_center.x() - self.geometry().width()/2,
            window_center.y() - self.geometry().height()/2)

    def connect_signal_slots(self):
        """Create signal/slot connections."""
        if (hasattr(self, 'tab_measurement') and
            hasattr(self, 'tab_power_supply')):
            self.tab_measurement.change_current_setpoint.connect(
                self.tab_power_supply.change_setpoint_and_emit_signal)

            self.tab_measurement.turn_off_power_supply_current.connect(
                self.tab_power_supply.set_current_to_zero)

            self.tab_power_supply.current_setpoint_changed.connect(
                self.tab_measurement.update_current_setpoint)

            self.tab_power_supply.start_measurement.connect(
                self.tab_measurement.measure_and_emit_signal)

            self.tab_power_supply.current_ramp_end.connect(
                self.tab_measurement.end_automatic_measurements)

        self.preferences_dialog.tabs_preferences_changed()
        self.ui.tbt_preferences.clicked.connect(self.preferences_dialog.show)
        self.ui.tbt_database.clicked.connect(self.change_database)

    def update_positions(self):
        """Update pmac positions."""
        if self.stop_positions_update:
            return

        try:
            worker = PositionsWorker()
            self.threadpool.start(worker)
        except Exception:
            pass


class PositionsWorker(_QRunnable):
    """Read position values from pmac."""

    @property
    def pmac(self):
        """Pmac communication class."""
        return _QApplication.instance().devices.pmac

    @property
    def positions(self):
        """Get current posiitons dict."""
        return _QApplication.instance().positions

    @positions.setter
    def positions(self, value):
        _QApplication.instance().positions = value

    def run(self):
        """Update axes positions."""
        if not self.pmac.connected:
            self.positions = {}
            return

        try:
            for axis in self.pmac.commands.list_of_axis:
                pos = self.pmac.get_position(axis)
                self.positions[axis] = pos

        except Exception:
            pass
